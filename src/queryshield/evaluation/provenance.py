"""Source-path classification and same-run lineage assertions.

This module consumes server-observed product records. It never reads expected
answers, retrieval gold, or model-written source labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass
import hashlib

from queryshield.agent.context import NET_FEN_GROSS_QUERY, NET_FEN_PLAN_ID, NET_FEN_REFUND_QUERY
from queryshield.agent.proposals import _sha256_json
from queryshield.catalog import load_default_catalog
from queryshield.policy.params import ordered_param_values


def _mapping_rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _same_context(item: Mapping[str, object], source: Mapping[str, object]) -> bool:
    return all(item.get(key) == source.get(key) for key in ("id", "source_id", "version", "text_sha256"))


def _sequence_number(event: Mapping[str, object]) -> int | None:
    value = event.get("sequence")
    return value if type(value) is int else None


def _state_after(observation: Mapping[str, object]) -> Mapping[str, object]:
    value = observation.get("state_after")
    return value if isinstance(value, Mapping) else {}


def _current_product_run_id(observation: Mapping[str, object]) -> object:
    """Use the product's persisted run ID for HTTP state routes, else the query run."""

    state = _state_after(observation)
    if type(state.get("run_id")) is str and state.get("run_id"):
        return state.get("run_id")
    ownership = observation.get("result_ownership_observation")
    if isinstance(ownership, Mapping) and type(ownership.get("requested_run_id")) is str:
        return ownership.get("requested_run_id")
    return observation.get("profile_run_id")


def _observed_identity(observation: Mapping[str, object], field: str) -> object:
    direct = observation.get(field)
    if direct is not None:
        return direct
    state_value = _state_after(observation).get(field)
    if state_value is not None:
        return state_value
    shared = observation.get("configuration_shared_identity")
    return shared.get(field) if isinstance(shared, Mapping) else None


def _execution_events(observation: Mapping[str, object]) -> list[Mapping[str, object]]:
    events = _mapping_rows(observation.get("execution_events"))
    if events:
        return events
    checkpoint = _state_after(observation).get("checkpoint")
    last_result = checkpoint.get("last_agent_result") if isinstance(checkpoint, Mapping) else None
    return _mapping_rows(last_result.get("events")) if isinstance(last_result, Mapping) else []


def _sql_records(observation: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Collect server SQL observations from normal and persisted-state routes."""

    candidates: list[Mapping[str, object]] = []
    for key in ("sql_records", "action_sql_records", "state_sql_records"):
        candidates.extend(_mapping_rows(observation.get(key)))
    policy = observation.get("sql_policy_result")
    if isinstance(policy, Mapping):
        candidates.extend(_mapping_rows(policy.get("execution_records")))
    unique: dict[tuple[object, ...], Mapping[str, object]] = {}
    for record in candidates:
        # Identity is part of the key so a copy under another tenant or
        # principal is kept and seen by ownership checks, never merged away.
        key = (
            record.get("result_id"),
            record.get("query_sha256"),
            record.get("run_id"),
            record.get("tenant_id"),
            record.get("principal_id"),
            record.get("status"),
        )
        if any(value is not None for value in key):
            unique.setdefault(key, record)
        else:
            unique[(id(record),)] = record
    return list(unique.values())


def _response_results(observation: Mapping[str, object]) -> list[Mapping[str, object]]:
    results: list[Mapping[str, object]] = []
    payload = observation.get("api_payload")
    if isinstance(payload, Mapping) and isinstance(payload.get("result"), Mapping):
        results.append(payload["result"])
    state_result = _state_after(observation).get("result")
    if isinstance(state_result, Mapping):
        results.append(state_result)
    return results


def _fixture_materialized_results(observation: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    results: dict[str, Mapping[str, object]] = {}
    for record in _mapping_rows(observation.get("fixture_materialization_records")):
        evidence = record.get("actual_result_evidence")
        if isinstance(evidence, Mapping) and type(evidence.get("result_id")) is str:
            results[str(evidence["result_id"])] = evidence
    return results


def _fact_matches_result(fact: Mapping[str, object], result: Mapping[str, object]) -> bool:
    bindings = _mapping_rows(result.get("metric_bindings"))
    binding = next(
        (
            item for item in bindings
            if str(item.get("metric_id", "")).removeprefix("metric.")
            == str(fact.get("metric_id", "")).removeprefix("metric.")
        ),
        None,
    )
    rows = _mapping_rows(result.get("rows"))
    position = binding.get("result_position") if isinstance(binding, Mapping) else None
    return bool(
        isinstance(binding, Mapping)
        and binding.get("catalog_source_id") == fact.get("catalog_source_id")
        and binding.get("catalog_version") == fact.get("catalog_version")
        and binding.get("unit") == fact.get("unit")
        and binding.get("time_window") == fact.get("time_window")
        and len(rows) == 1
        and type(position) is str
        and rows[0].get(position) == fact.get("value")
    )


@dataclass(frozen=True)
class NetFenCompositionVerdict:
    """Outcome of verifying one fact against a server-composed net_fen result."""

    accepted: bool
    reason: str
    # (gross, refund) component result IDs, present only when accepted.
    component_result_ids: tuple[str, str] | None = None


# Per-pair checks in order; a rejection names the furthest stage any pair reached.
_PAIR_STAGE_REASONS = (
    "composition_hash_mismatch",
    "composition_params_mismatch",
    "composition_version_mismatch",
    "composition_value_mismatch",
)


def _non_empty_str(value: object) -> bool:
    return type(value) is str and bool(value)


def _single_int_cell(record: Mapping[str, object], column: str) -> int | None:
    rows = _mapping_rows(record.get("rows"))
    value = rows[0].get(column) if len(rows) == 1 else None
    return value if type(value) is int else None


def _pair_failure_stage(
    gross: Mapping[str, object],
    refund: Mapping[str, object],
    evidence: Mapping[str, object],
    *,
    time_window: object,
    net_fen: int,
) -> int | None:
    """Return the first failing stage for one component pair, or None if it composes."""

    # Same commitments as ControlledTools._execute_net_fen_plan: the combined
    # query hash covers both SQL texts, the params hash the window and both
    # component parameter hashes (component query hashes exclude parameters).
    query_sha256 = hashlib.sha256(
        (
            f"{NET_FEN_PLAN_ID}\n"
            f"gross_query_sha256={gross['query_sha256']}\n"
            f"refund_query_sha256={refund['query_sha256']}"
        ).encode("utf-8")
    ).hexdigest()
    if query_sha256 != evidence.get("query_sha256"):
        return 0
    try:
        params_sha256 = _sha256_json(
            {
                "plan_id": NET_FEN_PLAN_ID,
                "time_window": dict(time_window) if isinstance(time_window, Mapping) else time_window,
                "gross_params_sha256": gross["params_sha256"],
                "refund_params_sha256": refund["params_sha256"],
            }
        )
    except ValueError:
        return 1
    if params_sha256 != evidence.get("params_sha256"):
        return 1
    if any(
        record.get(field) != evidence.get(field)
        for record in (gross, refund)
        for field in ("catalog_version", "policy_version")
    ):
        return 2
    gross_fen = _single_int_cell(gross, "gross_fen")
    refund_fen = _single_int_cell(refund, "refund_fen")
    if gross_fen is None or refund_fen is None or gross_fen < 0 or refund_fen < 0 or gross_fen - refund_fen != net_fen:
        return 3
    return None


def verify_net_fen_composition(
    fact: Mapping[str, object],
    evidence: Mapping[str, object] | None,
    sql_records: Sequence[Mapping[str, object]],
    *,
    run_id: object,
    tenant_id: object,
    principal_id: object,
    claimed_component_ids: AbstractSet[str] = frozenset(),
) -> NetFenCompositionVerdict:
    """Verify a fact that cites a server-composed net_fen plan result.

    ``evidence`` is the composed result's ``ResultEvidence.as_dict()`` and
    ``sql_records`` are this run's recorded executions, unfiltered.  The fact
    is accepted only when its content matches the evidence and one unclaimed
    gross/refund component pair of this run re-derives both evidence hashes
    and the net value.  A component pair backs at most one composed result;
    callers pass the component IDs already used by other composed results.
    Every unverifiable step fails closed with a fixed ``composition_*`` reason.

    The fact may omit ``run_id``: product facts carry only tenant/principal
    ownership, while the run is proven strictly through the evidence and every
    component record, which must all match ``run_id`` exactly.
    """

    def reject(reason: str) -> NetFenCompositionVerdict:
        return NetFenCompositionVerdict(False, reason)

    if not isinstance(evidence, Mapping):
        return reject("composition_evidence_not_found")
    if not isinstance(fact, Mapping):
        return reject("composition_fact_mismatch")
    if evidence.get("metric_plan_id") != NET_FEN_PLAN_ID:
        return reject("composition_plan_mismatch")
    result_id = evidence.get("result_id")
    if not _non_empty_str(result_id) or fact.get("result_id") != result_id:
        return reject("composition_result_id_mismatch")
    identity = {"run_id": run_id, "tenant_id": tenant_id, "principal_id": principal_id}
    if (
        not all(_non_empty_str(value) for value in identity.values())
        or any(evidence.get(key) != value for key, value in identity.items())
        or fact.get("tenant_id") != tenant_id
        or fact.get("principal_id") != principal_id
        or fact.get("run_id") not in (None, run_id)
    ):
        return reject("composition_owner_mismatch")

    raw_bindings = evidence.get("metric_bindings")
    bindings = _mapping_rows(raw_bindings)
    # _mapping_rows drops non-mappings, so also require exactly one raw entry.
    binding = (
        bindings[0]
        if len(bindings) == 1 and isinstance(raw_bindings, Sequence) and len(raw_bindings) == 1
        else None
    )
    rows = _mapping_rows(evidence.get("rows"))
    net_fen = rows[0].get("net_fen") if len(rows) == 1 else None
    fact_metric = fact.get("metric_id")
    if (
        type(fact_metric) is not str
        or fact_metric.removeprefix("metric.") != "net_fen"
        or not isinstance(binding, Mapping)
        or str(binding.get("metric_id", "")).removeprefix("metric.") != "net_fen"
        or binding.get("plan_id") != NET_FEN_PLAN_ID
        or binding.get("result_position") != "net_fen"
        or binding.get("catalog_version") != evidence.get("catalog_version")
        or evidence.get("row_count") != 1
        or type(net_fen) is not int
        or type(fact.get("value")) is not int
        or not _fact_matches_result(fact, evidence)
    ):
        return reject("composition_fact_mismatch")

    # A plan-SQL execution under another identity is never ignored.
    components = [
        record for record in sql_records
        if isinstance(record, Mapping)
        and record.get("status") == "succeeded"
        and record.get("sql") in (NET_FEN_GROSS_QUERY, NET_FEN_REFUND_QUERY)
    ]
    if any(record.get(key) != value for record in components for key, value in identity.items()):
        return reject("composition_component_owner_mismatch")
    usable = [
        record for record in components
        if all(_non_empty_str(record.get(key)) for key in ("result_id", "query_sha256", "params_sha256"))
    ]
    gross = [record for record in usable if record.get("sql") == NET_FEN_GROSS_QUERY]
    refund = [record for record in usable if record.get("sql") == NET_FEN_REFUND_QUERY]
    if not gross or not refund:
        return reject("composition_component_count")

    furthest = 0
    valid_but_claimed = False
    for gross_record in gross:
        for refund_record in refund:
            stage = _pair_failure_stage(
                gross_record,
                refund_record,
                evidence,
                time_window=binding.get("time_window"),
                net_fen=net_fen,
            )
            if stage is not None:
                furthest = max(furthest, stage)
                continue
            pair = (str(gross_record["result_id"]), str(refund_record["result_id"]))
            if pair[0] in claimed_component_ids or pair[1] in claimed_component_ids:
                valid_but_claimed = True
                continue
            return NetFenCompositionVerdict(True, "net_fen_plan_components_verified", pair)
    if valid_but_claimed:
        return reject("composition_components_already_used")
    return reject(_PAIR_STAGE_REASONS[furthest])


def claimed_components_excluding(
    claims: Mapping[str, tuple[str, str]],
    result_id: object,
) -> frozenset[str]:
    """Component IDs used by composed results other than ``result_id``."""

    return frozenset(
        component
        for claimed_result_id, pair in claims.items()
        if claimed_result_id != result_id
        for component in pair
    )


def _context_query_result_refs(
    call: Mapping[str, object],
    context_records: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    call_id = call.get("model_call_id")
    refs: list[Mapping[str, object]] = []
    for context in context_records:
        if context.get("model_call_id") != call_id:
            continue
        if (
            context.get("receipt") == "captured_from_actual_model_request_messages"
            and context.get("status") == "succeeded"
        ):
            refs.extend(_mapping_rows(context.get("query_result_refs")))
    # Backward compatibility for already serialized v16-shaped records. New
    # records use model_context_records as the canonical actual-request receipt.
    if not refs and call.get("prompt_source_receipt") == "captured_from_actual_model_request_messages":
        refs.extend(_mapping_rows(call.get("prompt_query_result_refs")))
    return refs


def _tool_event_model_call_ids(observation: Mapping[str, object], result_id: object) -> list[str]:
    """Bind a result to the latest successful model call before its query tool event."""

    run_id = _current_product_run_id(observation)
    events = _execution_events(observation)
    query_events = [
        event for event in events
        if event.get("kind") == "tool_call"
        and event.get("tool_name") == "query_readonly"
        and event.get("status") == "succeeded"
        and event.get("result_id") == result_id
        and event.get("run_id") == run_id
    ]
    call_ids: list[str] = []
    for tool_event in query_events:
        tool_sequence = _sequence_number(tool_event)
        prior_calls = [
            event for event in events
            if event.get("kind") == "model_call"
            and event.get("status") == "succeeded"
            and type(event.get("model_call_id")) is str
            and event.get("run_id") == run_id
            and tool_sequence is not None
            and (_sequence_number(event) or -1) < tool_sequence
        ]
        if prior_calls:
            latest = max(prior_calls, key=lambda event: _sequence_number(event) or -1)
            call_id = latest.get("model_call_id")
            if type(call_id) is str and call_id not in call_ids:
                call_ids.append(call_id)
    return call_ids


def _b0_single_call_for_result(observation: Mapping[str, object], result_id: str) -> list[str]:
    """Bind a B0 server result to B0's only model call, or return no binding.

    B0 makes one model call and at most one tool call, and its ``result_ids``
    come from that tool's server response.  A server-composed plan executes
    server SQL, so the proposal-SQL join below cannot apply to it.
    """

    profile = observation.get("evaluation_profile") or observation.get("profile")
    if profile != "B0":
        return []
    if any(
        event.get("kind") == "tool_call" and event.get("tool_name") == "query_readonly"
        for event in _execution_events(observation)
    ):
        return []
    call_ids = observation.get("model_call_ids")
    result_ids = observation.get("result_ids")
    metrics = observation.get("execution_metrics")
    if (
        not isinstance(call_ids, Sequence)
        or isinstance(call_ids, (str, bytes))
        or len(call_ids) != 1
        or not _non_empty_str(call_ids[0])
        or not isinstance(result_ids, Sequence)
        or isinstance(result_ids, (str, bytes))
        or list(result_ids) != [result_id]
        or not isinstance(metrics, Mapping)
        or metrics.get("tool_calls") != 1
    ):
        return []
    call_id = call_ids[0]
    records = _mapping_rows(observation.get("model_call_records"))
    if len(records) != 1:
        return []
    record = records[0]
    shape = record.get("response_shape")
    proposal = record.get("proposal")
    is_query_call = (
        isinstance(shape, Mapping)
        and shape.get("action_type") == "tool_call"
        and shape.get("action_name") == "query_readonly"
    ) or (isinstance(proposal, Mapping) and proposal.get("name") == "query_readonly")
    if record.get("model_call_id") != call_id or record.get("status") != "succeeded" or not is_query_call:
        return []
    return [str(call_id)]


def _query_model_call_ids(
    observation: Mapping[str, object],
    sql_record: Mapping[str, object],
) -> list[str]:
    call_ids = _tool_event_model_call_ids(observation, sql_record.get("result_id"))

    # B0's intentionally single generation has no Agent tool event. Join its
    # actual proposal to the recorded SQL text and parameters instead.
    proposal_params = sql_record.get("params")
    for record in _mapping_rows(observation.get("model_call_records")):
        proposal = record.get("proposal")
        if not isinstance(proposal, Mapping) or proposal.get("name") != "query_readonly":
            continue
        params = proposal.get("params")
        if proposal.get("sql") != sql_record.get("sql") or not isinstance(params, Mapping):
            continue
        indexed_params = ordered_param_values(params)
        if indexed_params is None or list(indexed_params) != proposal_params:
            continue
        call_id = record.get("model_call_id")
        if type(call_id) is str and call_id not in call_ids:
            call_ids.append(call_id)
    return call_ids


def classify_source_lineage(observation: Mapping[str, object]) -> dict[str, object]:
    """Classify direct, prepared, and current-run retrieval paths and assert links."""

    run_id = _current_product_run_id(observation)
    tenant_id = _observed_identity(observation, "tenant_id")
    principal_id = _observed_identity(observation, "principal_id")
    profile = observation.get("evaluation_profile") or observation.get("profile")
    entrypoint = (
        observation.get("action_input", {}).get("entrypoint")
        if isinstance(observation.get("action_input"), Mapping)
        else None
    )
    errors: list[str] = []
    direct_sources: list[dict[str, object]] = []
    prepared_sources: list[dict[str, object]] = []
    retrieved_sources: list[dict[str, object]] = []

    if type(run_id) is not str or not run_id:
        errors.append("missing_product_run_id")
    if type(tenant_id) is not str or not tenant_id or type(principal_id) is not str or not principal_id:
        errors.append("missing_product_identity")
    calls = {
        str(record.get("model_call_id")): record
        for record in _mapping_rows(observation.get("model_call_records"))
        if type(record.get("model_call_id")) is str
    }
    context_records = _mapping_rows(observation.get("model_context_records"))
    events = _execution_events(observation)
    catalog = load_default_catalog()
    catalog_entries = {entry.id: entry for entry in catalog.entries}
    model_event_by_id = {
        str(event.get("model_call_id")): event
        for event in events
        if event.get("kind") == "model_call" and type(event.get("model_call_id")) is str
    }
    retrieval_evidence = {
        str(record.get("retrieval_id")): record
        for record in _mapping_rows(observation.get("retrieval_records"))
        if type(record.get("retrieval_id")) is str
    }
    returns = {
        str(record.get("retrieval_id")): record
        for record in _mapping_rows(observation.get("retrieval_return_records"))
        if type(record.get("retrieval_id")) is str
    }
    initial_sources = _mapping_rows(observation.get("initial_retrieval_records"))

    # Validate every returned retrieval record against the actual selected IDs,
    # run, identity, snapshot evidence, and the server's ACL decision.
    valid_return_items: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    for retrieval_id, returned in returns.items():
        evidence = retrieval_evidence.get(retrieval_id)
        if returned.get("run_id") != run_id or not isinstance(evidence, Mapping) or evidence.get("run_id") != run_id:
            errors.append("retrieval_evidence_cross_run_or_missing")
            continue
        if returned.get("tenant_id") != tenant_id or returned.get("principal_id") != principal_id:
            errors.append("retrieval_evidence_identity_mismatch")
        if returned.get("snapshot_id") != evidence.get("snapshot_id"):
            errors.append("retrieval_snapshot_mismatch")
        selected = set(evidence.get("selected_ids", ())) if isinstance(evidence.get("selected_ids"), Sequence) else set()
        return_items = _mapping_rows(returned.get("items"))
        if not return_items:
            # Empty retrieval remains a real outcome in the denominator.
            continue
        for item in return_items:
            visibility = item.get("visibility_check")
            if (
                item.get("id") not in selected
                or item.get("visibility_check") is None
                or not isinstance(visibility, Mapping)
                or visibility.get("passed") is not True
            ):
                errors.append("retrieval_item_not_selected_or_not_visible")
            valid_return_items.append((returned, item))

    # The exact content hashes captured from the actual provider messages must
    # match a same-run tool return or a validated prepared context item.
    for context_record in context_records:
        call_id = context_record.get("model_call_id")
        call = calls.get(str(call_id)) if call_id is not None else None
        source_items = _mapping_rows(context_record.get("source_items"))
        if not source_items:
            continue
        if not isinstance(call, Mapping) or context_record.get("receipt") != "captured_from_actual_model_request_messages":
            errors.append("retrieval_context_missing_actual_model_call_receipt")
            continue
        if call.get("status") != "succeeded":
            # Preserve the failed provider call, but do not misclassify it as a
            # completed answer path or turn the provider failure into a source error.
            continue
        for item in source_items:
            matched_returns = [
                (returned, source)
                for returned, source in valid_return_items
                if _same_context(item, source)
                and returned.get("run_id") == run_id
                and returned.get("tenant_id") == tenant_id
                and returned.get("principal_id") == principal_id
            ]
            if matched_returns:
                for returned, source in matched_returns:
                    retrieved_sources.append({
                        "origin": "current_run_retrieval",
                        "run_id": run_id,
                        "retrieval_id": returned.get("retrieval_id"),
                        "model_call_id": call_id,
                        "candidate_id": source.get("id"),
                        "source_id": source.get("source_id"),
                        "version": source.get("version"),
                        "text_sha256": source.get("text_sha256"),
                        "source_kind": source.get("source_kind"),
                    })
                continue
            matched_initial = [
                source for source in initial_sources
                if source.get("source_id") == item.get("source_id")
                and source.get("version") == item.get("version")
                and source.get("text_sha256") == item.get("text_sha256")
                and (
                    source.get("id", source.get("source_id")) == item.get("id")
                )
            ]
            if matched_initial:
                for source in matched_initial:
                    access_check = source.get("snapshot_source_check")
                    if not isinstance(access_check, Mapping) or access_check.get("visible_for_identity") is not True:
                        errors.append("prepared_context_source_not_visible")
                        continue
                    prepared_sources.append({
                        "origin": "prepared_run_context",
                        "run_id": run_id,
                        "model_call_id": call_id,
                        "candidate_id": item.get("id"),
                        "source_id": item.get("source_id"),
                        "version": item.get("version"),
                        "text_sha256": item.get("text_sha256"),
                    })
                continue
            catalog_result_items = _mapping_rows(context_record.get("catalog_search_items"))
            matched_catalog_results = [
                result_item for result_item in catalog_result_items
                if _same_context(item, result_item)
            ]
            catalog_entry = catalog_entries.get(str(item.get("id")))
            catalog_call_event = model_event_by_id.get(str(call_id)) if call_id is not None else None
            catalog_call_sequence = _sequence_number(catalog_call_event) if isinstance(catalog_call_event, Mapping) else None
            preceding_catalog_search = [
                event for event in events
                if event.get("kind") == "tool_call"
                and event.get("tool_name") == "search_catalog"
                and event.get("status") == "succeeded"
                and event.get("run_id") == run_id
                and catalog_call_sequence is not None
                and _sequence_number(event) is not None
                and _sequence_number(event) < catalog_call_sequence
                and isinstance(event.get("source_ids"), Sequence)
                and not isinstance(event.get("source_ids"), (str, bytes))
                and item.get("source_id") in event.get("source_ids", ())
            ]
            if (
                matched_catalog_results
                and catalog_entry is not None
                and catalog_entry.source_id == item.get("source_id")
                and catalog_entry.version == item.get("version")
                and hashlib.sha256(catalog_entry.text.encode("utf-8")).hexdigest() == item.get("text_sha256")
                and (
                    not catalog_entry.requires_approval
                    or _observed_identity(observation, "role") == "approver"
                )
                and preceding_catalog_search
            ):
                direct_sources.append({
                    "origin": "same_run_catalog_search_context",
                    "run_id": run_id,
                    "model_call_id": call_id,
                    "search_sequence": _sequence_number(max(preceding_catalog_search, key=lambda event: _sequence_number(event) or -1)),
                    "candidate_id": catalog_entry.id,
                    "catalog_source_id": catalog_entry.source_id,
                    "source_version": catalog_entry.version,
                    "text_sha256": item.get("text_sha256"),
                    "catalog_version": catalog.catalog_version,
                    "requires_final_model_receipt": False,
                })
                continue
            errors.append("model_context_source_has_no_same_run_return_or_prepared_record")

    # The HTTP state path records SQL under state_sql_records/action_sql_records;
    # normal /queries replays use sql_records. Both are server-side observations.
    sql_records = _sql_records(observation)
    facts = _mapping_rows(observation.get("facts"))
    succeeded_results = {
        str(record.get("result_id")): record
        for record in sql_records
        if record.get("status") == "succeeded" and type(record.get("result_id")) is str
    }
    response_results: dict[str, Mapping[str, object]] = {}
    for result in _response_results(observation):
        result_id = result.get("result_id")
        if type(result_id) is not str:
            continue
        existing = response_results.get(result_id)
        if existing is not None and any(
            existing.get(field) != result.get(field)
            for field in ("run_id", "tenant_id", "principal_id", "catalog_version", "rows", "metric_bindings")
        ):
            errors.append("state_and_response_result_evidence_mismatch")
        response_results[result_id] = result
    prepared_results = _fixture_materialized_results(observation)
    successful_current_sql = [
        record for record in sql_records
        if record.get("status") == "succeeded"
        and record.get("run_id") == run_id
        and record.get("tenant_id") == tenant_id
        and record.get("principal_id") == principal_id
        and record.get("statement_kind") == "SELECT"
    ]
    action_sql_ids = {
        str(record.get("result_id"))
        for record in _mapping_rows(observation.get("action_sql_records"))
        if record.get("status") == "succeeded" and type(record.get("result_id")) is str
    }
    # Server evidence for composed results that never pass the recording
    # executor; verified only through verify_net_fen_composition.
    composite_evidence = {
        str(record["result_id"]): record
        for record in _mapping_rows(observation.get("composite_result_evidence"))
        if _non_empty_str(record.get("result_id"))
    }
    composition_claims: dict[str, tuple[str, str]] = {}
    composition_verdicts: list[dict[str, object]] = []

    for fact in facts:
        result_id = fact.get("result_id")
        if type(result_id) is not str:
            errors.append("fact_missing_result_id")
            continue
        result_id = str(result_id)
        if (
            fact.get("run_id") not in (None, run_id)
            or fact.get("tenant_id") not in (None, tenant_id)
            or fact.get("principal_id") not in (None, principal_id)
        ):
            errors.append("fact_identity_or_run_mismatch")
            continue
        prepared = prepared_results.get(result_id)
        if isinstance(prepared, Mapping):
            if (
                prepared.get("run_id") != run_id
                or prepared.get("tenant_id") != tenant_id
                or prepared.get("principal_id") != principal_id
                or prepared.get("catalog_version") != fact.get("catalog_version")
            ):
                errors.append("prepared_result_run_identity_or_catalog_mismatch")
                continue
            response_result = response_results.get(result_id)
            if isinstance(response_result, Mapping) and any(
                response_result.get(field) != prepared.get(field)
                for field in ("run_id", "tenant_id", "principal_id", "catalog_version", "rows", "metric_bindings")
            ):
                errors.append("prepared_result_differs_from_product_response")
                continue
            if not _fact_matches_result(fact, prepared):
                errors.append("prepared_fact_does_not_match_materialized_result")
                continue
            prepared_sources.append({
                "origin": "prepared_server_result_fixture",
                "run_id": run_id,
                "tenant_id": tenant_id,
                "principal_id": principal_id,
                "result_id": result_id,
                "catalog_source_id": fact.get("catalog_source_id"),
                "catalog_version": fact.get("catalog_version"),
                "metric_id": fact.get("metric_id"),
                "result_position": next(
                    (binding.get("result_position") for binding in _mapping_rows(prepared.get("metric_bindings"))
                     if binding.get("metric_id") == fact.get("metric_id")),
                    None,
                ),
                "unit": fact.get("unit"),
                "time_window": fact.get("time_window"),
            })
            continue

        result = succeeded_results.get(result_id) or response_results.get(result_id)
        if not isinstance(result, Mapping):
            composite = composite_evidence.get(result_id)
            if composite is None:
                errors.append("fact_result_missing_same_run_server_result")
                continue
            verdict = verify_net_fen_composition(
                fact,
                composite,
                sql_records,
                run_id=run_id,
                tenant_id=tenant_id,
                principal_id=principal_id,
                claimed_component_ids=claimed_components_excluding(composition_claims, result_id),
            )
            composition_verdicts.append(
                {"result_id": result_id, "accepted": verdict.accepted, "reason": verdict.reason}
            )
            if not verdict.accepted or verdict.component_result_ids is None:
                errors.append("fact_result_missing_same_run_server_result")
                continue
            composition_claims[result_id] = verdict.component_result_ids
            composite_calls = (
                _tool_event_model_call_ids(observation, result_id)
                or _b0_single_call_for_result(observation, result_id)
            )
            if not composite_calls:
                errors.append("composite_result_has_no_bound_model_call")
                continue
            direct_sources.append({
                "origin": "verified_net_fen_plan_composition",
                "run_id": run_id,
                "tenant_id": tenant_id,
                "principal_id": principal_id,
                "catalog_source_id": fact.get("catalog_source_id"),
                "catalog_version": fact.get("catalog_version"),
                "metric_id": fact.get("metric_id"),
                "result_position": "net_fen",
                "unit": fact.get("unit"),
                "time_window": fact.get("time_window"),
                "result_id": result_id,
                "component_result_ids": list(verdict.component_result_ids),
                "model_call_ids": composite_calls,
                "requires_final_model_receipt": True,
            })
            continue
        if (
            result.get("run_id") != run_id
            or result.get("tenant_id") != tenant_id
            or result.get("principal_id") != principal_id
            or result.get("catalog_version") != fact.get("catalog_version")
        ):
            errors.append("fact_result_run_identity_or_catalog_mismatch")
            continue
        if fact.get("run_id") not in (None, run_id):
            errors.append("fact_run_id_mismatch")
            continue
        if not _fact_matches_result(fact, result):
            errors.append("fact_does_not_match_trusted_metric_binding_and_actual_row")
            continue
        query_calls = _query_model_call_ids(observation, result)
        is_approved_action = entrypoint == "/runs/{run_id}/approval" and result_id in action_sql_ids
        query_events = [
            event for event in events
            if event.get("kind") == "tool_call"
            and event.get("tool_name") == "query_readonly"
            and event.get("status") == "succeeded"
            and event.get("run_id") == run_id
            and event.get("result_id") == result_id
        ]
        composite_server_result = (
            result_id in response_results
            and bool(query_events)
            and bool(successful_current_sql)
        )
        if not query_calls and not is_approved_action and not composite_server_result:
            errors.append("catalog_result_has_no_bound_same_run_execution")
            continue
        bindings = _mapping_rows(result.get("metric_bindings"))
        binding = next(
            (item for item in bindings if str(item.get("metric_id", "")).removeprefix("metric.") == str(fact.get("metric_id", "")).removeprefix("metric.")),
            None,
        )
        position = binding.get("result_position") if isinstance(binding, Mapping) else None
        direct_sources.append({
            "origin": "direct_database_result_with_catalog_fact" if result_id in succeeded_results else "server_metric_plan_result_from_database_tool",
            "run_id": run_id,
            "tenant_id": tenant_id,
            "principal_id": principal_id,
            "catalog_source_id": fact.get("catalog_source_id"),
            "catalog_version": fact.get("catalog_version"),
            "metric_id": fact.get("metric_id"),
            "result_position": position,
            "unit": fact.get("unit"),
            "time_window": fact.get("time_window"),
            "result_id": result_id,
            "model_call_ids": query_calls,
            "requires_final_model_receipt": not is_approved_action,
        })

    # Grouped customer responses are facts only as a result-bound rowset. They
    # have no scalar Fact object, so validate their metric binding and rows.
    for result_id, result in succeeded_results.items():
        rows = _mapping_rows(result.get("rows"))
        if not rows or not any("customer_id" in row for row in rows):
            continue
        if result_id in prepared_results:
            continue
        bindings = _mapping_rows(result.get("metric_bindings"))
        if (
            result.get("run_id") != run_id
            or result.get("tenant_id") != tenant_id
            or result.get("principal_id") != principal_id
            or not bindings
            or any(binding.get("catalog_version") != result.get("catalog_version") for binding in bindings)
            or any(binding.get("result_position") not in row for binding in bindings for row in rows)
        ):
            errors.append("catalog_rowset_missing_same_run_trusted_binding")
            continue
        query_calls = _query_model_call_ids(observation, result)
        if not query_calls:
            errors.append("catalog_rowset_has_no_bound_model_call")
            continue
        direct_sources.append({
            "origin": "direct_database_result_with_catalog_rowset",
            "run_id": run_id,
            "tenant_id": tenant_id,
            "principal_id": principal_id,
            "catalog_source_id": [binding.get("catalog_source_id") for binding in bindings],
            "catalog_version": result.get("catalog_version"),
            "metric_ids": [binding.get("metric_id") for binding in bindings],
            "result_id": result_id,
            "model_call_ids": query_calls,
            "row_count": len(rows),
            "requires_final_model_receipt": True,
        })

    # Approved HTTP actions return a server-rendered database result and may
    # have no model call. Tie that answer to the exact successful action SQL,
    # persisted owner, and result rows instead of treating it as an unobserved
    # model answer.
    represented_result_ids = {str(source.get("result_id")) for source in direct_sources}
    for result_id, result in response_results.items():
        if result_id in represented_result_ids or result_id in prepared_results:
            continue
        sql_result = succeeded_results.get(result_id)
        if not isinstance(sql_result, Mapping):
            continue
        rows = _mapping_rows(result.get("rows"))
        if (
            result.get("run_id") != run_id
            or result.get("tenant_id") != tenant_id
            or result.get("principal_id") != principal_id
            or sql_result.get("run_id") != run_id
            or sql_result.get("tenant_id") != tenant_id
            or sql_result.get("principal_id") != principal_id
            or sql_result.get("rows") != result.get("rows")
            or not rows
        ):
            errors.append("direct_database_response_result_mismatch")
            continue
        is_approved_action = entrypoint == "/runs/{run_id}/approval" and result_id in action_sql_ids
        if not is_approved_action:
            errors.append("direct_database_response_has_no_bound_catalog_fact_or_query_call")
            continue
        direct_sources.append({
            "origin": "direct_database_result_from_approved_action",
            "run_id": run_id,
            "tenant_id": tenant_id,
            "principal_id": principal_id,
            "catalog_version": result.get("catalog_version"),
            "result_id": result_id,
            "model_call_ids": [],
            "entrypoint": entrypoint,
            "row_count": len(rows),
            "requires_final_model_receipt": False,
        })

    # A successful B1 model answer after SQL must have received that same
    # result ID in its actual prompt. B0 intentionally renders verified facts
    # server-side after its one generation. The approved-action HTTP route is
    # also server-rendered and has its own action SQL receipt.
    if observation.get("status") == "succeeded" and profile == "B1":
        final_calls = [
            record for record in calls.values()
            if record.get("status") == "succeeded"
            and (
                record.get("proposal_type") == "final_answer"
                or (isinstance(record.get("response_shape"), Mapping) and record["response_shape"].get("action_type") == "final_answer")
            )
        ]
        for source in direct_sources:
            if source.get("requires_final_model_receipt") is False:
                continue
            result_id = source.get("result_id")
            if not any(
                any(ref.get("result_id") == result_id for ref in _context_query_result_refs(record, context_records))
                for record in final_calls
            ):
                errors.append("bounded_answer_call_did_not_receive_same_run_query_result")
        answer_call_ids = [
            str(record.get("model_call_id"))
            for record in final_calls
            if type(record.get("model_call_id")) is str
        ]
        event_sequences = {
            str(event.get("model_call_id")): _sequence_number(event)
            for event in events
            if event.get("kind") == "model_call" and type(event.get("model_call_id")) is str
        }
        context_call_ids = {
            str(source.get("model_call_id"))
            for source in (*prepared_sources, *retrieved_sources)
            if type(source.get("model_call_id")) is str
        }
        if context_call_ids and not any(
            event_sequences.get(context_call_id) is not None
            and event_sequences.get(answer_call_id) is not None
            and event_sequences[context_call_id] < event_sequences[answer_call_id]
            for context_call_id in context_call_ids
            for answer_call_id in answer_call_ids
        ):
            errors.append("retrieval_context_not_before_same_run_product_answer")
    else:
        answer_call_ids = []

    if observation.get("status") == "succeeded" and not (direct_sources or prepared_sources or retrieved_sources):
        errors.append("successful_answer_has_no_observed_product_source_path")

    return {
        "status": "fail" if errors else "pass",
        "run_id": run_id,
        "profile": profile,
        "source_paths": {
            "direct_database_catalog": direct_sources,
            "prepared_context": prepared_sources,
            "current_run_retrieval_to_model": retrieved_sources,
        },
        "answer_model_call_ids": answer_call_ids,
        "retrieval_not_traversed_reason": observation.get("retrieval_not_traversed_reason"),
        "composition_verdicts": composition_verdicts,
        "errors": errors,
    }


__all__ = [
    "NetFenCompositionVerdict",
    "claimed_components_excluding",
    "classify_source_lineage",
    "verify_net_fen_composition",
]
