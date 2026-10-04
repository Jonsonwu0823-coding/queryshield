from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import runpy
import json

import pytest

from queryshield.evaluation.w05_provenance import classify_w05_source_lineage


def _observation(*, prepared: bool = False, retrieval: bool = False) -> dict[str, object]:
    run_id = "run-lineage-1"
    result_id = "result-lineage-1"
    source_item = {
        "message_index": 2,
        "id": "semantic-metric-gross@2026-09-21#0001",
        "source_id": "semantic-metric-gross",
        "version": "2026-09-21",
        "text_sha256": "a" * 64,
    }
    sql = {
        "run_id": run_id,
        "tenant_id": "A",
        "principal_id": "principal-A",
        "result_id": result_id,
        "catalog_version": "catalog-v2",
        "status": "succeeded",
        "sql": "SELECT SUM(amount_fen) AS gross_fen FROM orders",
        "params": ["paid"],
        "rows": [{"gross_fen": 15000}],
        "metric_bindings": [{
            "metric_id": "gross_fen",
            "result_position": "gross_fen",
            "unit": "CNY_fen",
            "time_window": {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z", "timezone": "UTC"},
            "catalog_source_id": "commerce-v1",
            "catalog_version": "catalog-v2",
        }],
    }
    query_call = {
        "status": "succeeded",
        "model_call_id": "call-query",
        "proposal": {"name": "query_readonly", "sql": sql["sql"], "params": {"0": "paid"}},
        "response_shape": {"action_type": "tool_call", "action_name": "query_readonly"},
        "prompt_source_items": [source_item] if retrieval or prepared else [],
        "prompt_source_receipt": "captured_from_actual_model_request_messages",
        "prompt_query_result_refs": [],
    }
    final_call = {
        "status": "succeeded",
        "model_call_id": "call-final",
        "proposal_type": "final_answer",
        "response_shape": {"action_type": "final_answer"},
        "prompt_source_items": [],
    }
    observation: dict[str, object] = {
        "profile_run_id": run_id,
        "evaluation_profile": "B1",
        "status": "succeeded",
        "tenant_id": "A",
        "principal_id": "principal-A",
        "role": "requester",
        "facts": [{
            "metric_id": "gross_fen",
            "value": 15000,
            "unit": "CNY_fen",
            "time_window": sql["metric_bindings"][0]["time_window"],
            "result_id": result_id,
            "catalog_source_id": "commerce-v1",
            "catalog_version": "catalog-v2",
        }],
        "sql_records": [sql],
        "model_call_records": [query_call, final_call],
        "model_context_records": [],
        "execution_events": [
            {"kind": "model_call", "status": "succeeded", "sequence": 1, "run_id": run_id, "model_call_id": "call-query"},
            {"kind": "tool_call", "status": "succeeded", "sequence": 2, "run_id": run_id, "tool_name": "query_readonly", "result_id": result_id},
            {"kind": "model_call", "status": "succeeded", "sequence": 3, "run_id": run_id, "model_call_id": "call-final"},
        ],
        "retrieval_records": [],
        "retrieval_return_records": [],
        "initial_retrieval_records": [],
    }
    if prepared or retrieval:
        observation["model_context_records"] = [{
            "model_call_id": "call-query",
            "request_id": "request-query",
            "status": "succeeded",
            "source_items": [source_item],
            "receipt": "captured_from_actual_model_request_messages",
            "query_result_refs": [],
        }]
    if prepared:
        observation["initial_retrieval_records"] = [{
            "id": source_item["id"],
            "source_id": source_item["source_id"],
            "version": source_item["version"],
            "text_sha256": source_item["text_sha256"],
                "snapshot_source_check": {"visible_for_identity": True},
            }]
    observation["model_context_records"].append({
        "model_call_id": "call-final",
        "request_id": "request-final",
        "status": "succeeded",
        "source_items": [],
        "receipt": "captured_from_actual_model_request_messages",
        "query_result_refs": [{"result_id": result_id, "run_scoped_tool_receipt": True}],
    })
    if retrieval:
        observation["retrieval_records"] = [{
            "run_id": run_id,
            "retrieval_id": "retrieval-lineage-1",
            "snapshot_id": "snapshot-1",
            "selected_ids": [source_item["id"]],
        }]
        observation["retrieval_return_records"] = [{
            "run_id": run_id,
            "tenant_id": "A",
            "principal_id": "principal-A",
            "retrieval_id": "retrieval-lineage-1",
            "snapshot_id": "snapshot-1",
            "items": [{
                "id": source_item["id"],
                "source_id": source_item["source_id"],
                "version": source_item["version"],
                "text_sha256": source_item["text_sha256"],
                "source_kind": "knowledge_document",
                "visibility_check": {"passed": True},
            }],
        }]
    return observation


def test_w05_direct_database_catalog_path_passes_without_retrieval() -> None:
    result = classify_w05_source_lineage(_observation())

    assert result["status"] == "pass"
    assert result["source_paths"]["direct_database_catalog"][0]["catalog_source_id"] == "commerce-v1"
    assert result["source_paths"]["current_run_retrieval_to_model"] == []


def test_w05_prepared_context_requires_visible_source_and_actual_model_receipt() -> None:
    result = classify_w05_source_lineage(_observation(prepared=True))

    assert result["status"] == "pass"
    assert result["source_paths"]["prepared_context"][0]["model_call_id"] == "call-query"


def test_w05_current_run_retrieval_must_reach_the_model_before_answer() -> None:
    result = classify_w05_source_lineage(_observation(retrieval=True))

    assert result["status"] == "pass"
    source = result["source_paths"]["current_run_retrieval_to_model"][0]
    assert source["retrieval_id"] == "retrieval-lineage-1"
    assert source["model_call_id"] == "call-query"
    assert source["source_kind"] == "knowledge_document"


def test_w05_b1_answer_uses_query_result_ref_from_actual_context_sidecar() -> None:
    observation = _observation()
    result = classify_w05_source_lineage(observation)

    assert result["status"] == "pass"
    observation["model_context_records"][-1]["query_result_refs"] = []
    missing_receipt = classify_w05_source_lineage(observation)

    assert missing_receipt["status"] == "fail"
    assert "bounded_answer_call_did_not_receive_same_run_query_result" in missing_receipt["errors"]


def test_w05_state_path_approved_database_answer_uses_action_result_and_identity() -> None:
    result_id = "approved-result-1"
    product_run_id = "persisted-run-1"
    result = {
        "result_id": result_id,
        "run_id": product_run_id,
        "tenant_id": "A",
        "principal_id": "principal-A",
        "catalog_version": "catalog-v1",
        "rows": [{"name": "甲"}],
        "metric_bindings": [],
    }
    sql = {
        "result_id": result_id,
        "run_id": product_run_id,
        "tenant_id": "A",
        "principal_id": "principal-A",
        "catalog_version": "catalog-v1",
        "status": "succeeded",
        "statement_kind": "SELECT",
        "rows": [{"name": "甲"}],
        "sql": "SELECT name FROM customers WHERE customer_id = %s",
        "params": ["c1"],
    }
    observation = {
        "profile_run_id": "evaluation-run-1",
        "evaluation_profile": "B0",
        "status": "succeeded",
        "action_input": {"entrypoint": "/runs/{run_id}/approval"},
        "configuration_shared_identity": {"tenant_id": "A", "principal_id": "principal-A"},
        "state_after": {
            "run_id": product_run_id,
            "tenant_id": "A",
            "principal_id": "principal-A",
            "result": result,
        },
        "api_payload": {"result": result},
        "action_sql_records": [sql],
        "state_sql_records": [sql],
        "facts": [],
        "model_call_records": [],
        "model_context_records": [],
    }

    lineage = classify_w05_source_lineage(observation)

    assert lineage["status"] == "pass"
    assert lineage["source_paths"]["direct_database_catalog"][0]["origin"] == "direct_database_result_from_approved_action"


def test_w05_prepared_server_result_uses_materialized_result_not_fixture_expected_value() -> None:
    result_id = "prepared-result-1"
    product_run_id = "persisted-run-2"
    binding = {
        "metric_id": "gross_fen",
        "result_position": "gross_fen",
        "unit": "CNY_fen",
        "time_window": {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z", "timezone": "UTC"},
        "catalog_source_id": "commerce-v1",
        "catalog_version": "catalog-v1",
    }
    result = {
        "result_id": result_id,
        "run_id": product_run_id,
        "tenant_id": "A",
        "principal_id": "principal-A",
        "catalog_version": "catalog-v1",
        "rows": [{"gross_fen": 15000}],
        "metric_bindings": [binding],
    }
    fact = {
        "result_id": result_id,
        "run_id": product_run_id,
        "tenant_id": "A",
        "principal_id": "principal-A",
        "metric_id": "gross_fen",
        "catalog_source_id": "commerce-v1",
        "catalog_version": "catalog-v1",
        "unit": "CNY_fen",
        "time_window": binding["time_window"],
        "value": 15000,
    }
    observation = {
        "profile_run_id": "evaluation-run-2",
        "evaluation_profile": "B0",
        "status": "succeeded",
        "configuration_shared_identity": {"tenant_id": "A", "principal_id": "principal-A"},
        "state_after": {"run_id": product_run_id, "tenant_id": "A", "principal_id": "principal-A", "result": result},
        "api_payload": {"result": result},
        "fixture_materialization_records": [{"actual_result_evidence": result}],
        "facts": [fact],
        "sql_records": [],
        "model_call_records": [],
        "model_context_records": [],
    }

    lineage = classify_w05_source_lineage(observation)

    assert lineage["status"] == "pass"
    assert lineage["source_paths"]["direct_database_catalog"] == []
    assert lineage["source_paths"]["prepared_context"][0]["origin"] == "prepared_server_result_fixture"


def test_w05_state_path_composite_result_requires_same_run_query_tool_and_actual_prompt_ref() -> None:
    observation = _observation()
    product_run_id = "persisted-run-3"
    composite_id = "composite-net-1"
    binding = dict(observation["sql_records"][0]["metric_bindings"][0])
    binding["metric_id"] = "net_fen"
    binding["result_position"] = "net_fen"
    composite = {
        "result_id": composite_id,
        "run_id": product_run_id,
        "tenant_id": "A",
        "principal_id": "principal-A",
        "catalog_version": "catalog-v2",
        "rows": [{"net_fen": 12000}],
        "metric_bindings": [binding],
    }
    observation.update({
        "evaluation_profile": "B1",
        "profile_run_id": "evaluation-run-3",
        "state_after": {"run_id": product_run_id, "tenant_id": "A", "principal_id": "principal-A", "result": composite},
        "api_payload": {"result": composite},
        "action_input": {"entrypoint": "/runs/{run_id}/resume"},
        "sql_records": [
            {"result_id": "gross-subquery-1", "run_id": product_run_id, "tenant_id": "A", "principal_id": "principal-A", "status": "succeeded", "statement_kind": "SELECT", "rows": [{"gross_fen": 15000}]},
            {"result_id": "refund-subquery-1", "run_id": product_run_id, "tenant_id": "A", "principal_id": "principal-A", "status": "succeeded", "statement_kind": "SELECT", "rows": [{"refund_fen": 3000}]},
        ],
        "facts": [{
            "result_id": composite_id,
            "run_id": product_run_id,
            "tenant_id": "A",
            "principal_id": "principal-A",
            "metric_id": "net_fen",
            "catalog_source_id": "commerce-v1",
            "catalog_version": "catalog-v2",
            "unit": "CNY_fen",
            "time_window": binding["time_window"],
            "value": 12000,
        }],
        "model_call_records": [
            {"model_call_id": "call-query", "status": "succeeded", "response_shape": {"action_type": "tool_call"}},
            {"model_call_id": "call-final", "status": "succeeded", "proposal_type": "final_answer"},
        ],
        "model_context_records": [{
            "model_call_id": "call-final", "request_id": "request-final", "status": "succeeded",
            "receipt": "captured_from_actual_model_request_messages", "source_items": [],
            "query_result_refs": [{"result_id": composite_id, "run_scoped_tool_receipt": True}],
        }],
        "execution_events": [
            {"kind": "model_call", "status": "succeeded", "sequence": 1, "run_id": product_run_id, "model_call_id": "call-query"},
            {"kind": "tool_call", "tool_name": "query_readonly", "status": "succeeded", "sequence": 2, "run_id": product_run_id, "result_id": composite_id},
            {"kind": "model_call", "status": "succeeded", "sequence": 3, "run_id": product_run_id, "model_call_id": "call-final"},
        ],
    })

    lineage = classify_w05_source_lineage(observation)

    assert lineage["status"] == "pass"
    observation["model_context_records"][0]["query_result_refs"] = []
    rejected = classify_w05_source_lineage(observation)
    assert rejected["status"] == "fail"
    assert "bounded_answer_call_did_not_receive_same_run_query_result" in rejected["errors"]


@pytest.mark.parametrize("mutation", [
    "missing_return",
    "cross_run",
    "cross_principal",
    "not_selected",
    "wrong_version",
    "revoked",
])
def test_w05_retrieval_source_negative_controls_fail(mutation: str) -> None:
    observation = _observation(retrieval=True)
    if mutation == "missing_return":
        observation["retrieval_return_records"] = []
    else:
        mutated = deepcopy(observation["retrieval_return_records"][0])
        if mutation == "cross_run":
            mutated["run_id"] = "another-run"
        elif mutation == "cross_principal":
            mutated["principal_id"] = "principal-B"
        elif mutation == "not_selected":
            mutated["items"][0]["id"] = "not-selected-candidate"
        elif mutation == "wrong_version":
            mutated["items"][0]["version"] = "old-version"
        elif mutation == "revoked":
            mutated["items"][0]["visibility_check"]["passed"] = False
        observation["retrieval_return_records"] = [mutated]

    result = classify_w05_source_lineage(observation)

    assert result["status"] == "fail"
    assert result["errors"]


def test_w05_en03_source_path_check_writes_all_required_fake_controls(tmp_path) -> None:
    check_w05 = Path(__file__).resolve().parents[1] / "scripts" / "check_w05.py"
    namespace = runpy.run_path(str(check_w05))
    b0 = _observation()
    b0.update({"evaluation_profile": "B0", "case_id": "direct", "action_input": {"entrypoint": "/queries"}})
    prepared = _observation(prepared=True)
    prepared.update({"evaluation_profile": "B1", "case_id": "prepared", "action_input": {"entrypoint": "/queries"}})
    retrieved = _observation(retrieval=True)
    retrieved.update({"evaluation_profile": "B1", "case_id": "retrieved", "action_input": {"entrypoint": "/queries"}})
    raw_path = tmp_path / "w05-stateful-fake-raw.json"
    raw_path.write_text(
        json.dumps({"raw_records": {"B0": [b0], "B1": [prepared, retrieved]}}),
        encoding="utf-8",
    )

    result = namespace["_check_stateful_source_paths"](
        {"raw_evidence": str(raw_path)},
        "fake",
        tmp_path,
    )

    assert result["status"] == "pass"
    assert result["positive_counts"] == {
        "direct_database_catalog": 3,
        "prepared_context": 1,
        "current_run_retrieval_to_model": 1,
    }
    assert result["negative_controls"]["control_count"] == 5


def test_w05_en03_failure_writes_case_and_lineage_errors_before_raising(tmp_path) -> None:
    check_w05 = Path(__file__).resolve().parents[1] / "scripts" / "check_w05.py"
    namespace = runpy.run_path(str(check_w05))
    direct = _observation()
    direct.update({"evaluation_profile": "B0", "case_id": "direct", "action_input": {"entrypoint": "/queries"}})
    broken = _observation()
    broken.update({
        "evaluation_profile": "B1",
        "case_id": "broken-run-lineage",
        "profile_run_id": None,
        "action_input": {"entrypoint": "/queries"},
    })
    raw_path = tmp_path / "w05-stateful-fake-raw.json"
    raw_path.write_text(json.dumps({"raw_records": {"B0": [direct], "B1": [broken]}}), encoding="utf-8")

    with pytest.raises(AssertionError, match="B1/broken-run-lineage"):
        namespace["_check_stateful_source_paths"]({"raw_evidence": str(raw_path)}, "fake", tmp_path)

    diagnostic = json.loads((tmp_path / "w05-fake-source-lineage-failure.json").read_text(encoding="utf-8"))
    assert diagnostic["failed_assertion"] == "completed_product_source_lineage"
    assert diagnostic["dataset_split"] == "development_only"
    assert diagnostic["invalid_completed_cases"][0]["case_id"] == "broken-run-lineage"
    assert "missing_product_run_id" in diagnostic["invalid_completed_cases"][0]["errors"]


# --- B1-2: server-composed net_fen results through the shared verifier ---------------

from test_w05_stateful_replay import (  # noqa: E402 - pytest prepend import mode puts tests/ on sys.path
    _AUG_WINDOW,
    _NET_QUESTION,
    _SEPT_WINDOW,
    _net_plan_parts,
    _plan_components,
    _run_ownership_case,
)

_VERIFIED_ORIGIN = "verified_net_fen_plan_composition"


def _composed_origins(lineage: dict[str, object]) -> list[dict[str, object]]:
    return [
        source for source in lineage["source_paths"]["direct_database_catalog"]
        if source["origin"] == _VERIFIED_ORIGIN
    ]


def _b1_composition_observation(windows, *, mutate=None) -> dict[str, object]:
    """A B1-shaped observation over real product plan evidence for each window."""

    context, records, tools, facts = _net_plan_parts(windows)
    evidence = [tools.get_result_evidence(str(fact["result_id"]), context=context).as_dict() for fact in facts]
    events: list[dict[str, object]] = []
    for index, fact in enumerate(facts):
        call_id = f"call-plan-{index}"
        events.append({"kind": "model_call", "model_call_id": call_id, "status": "succeeded",
                       "run_id": context.run_id, "sequence": 2 * index})
        events.append({"kind": "tool_call", "tool_name": "query_readonly", "status": "succeeded",
                       "run_id": context.run_id, "result_id": fact["result_id"], "sequence": 2 * index + 1})
    events.append({"kind": "model_call", "model_call_id": "call-final", "status": "succeeded",
                   "run_id": context.run_id, "sequence": 2 * len(facts)})
    observation: dict[str, object] = {
        "evaluation_profile": "B1",
        "status": "succeeded",
        "profile_run_id": context.run_id,
        "tenant_id": context.tenant_id,
        "principal_id": context.principal_id,
        "role": context.role,
        "facts": facts,
        "sql_records": records,
        "composite_result_evidence": evidence,
        "execution_events": events,
        "model_call_records": [
            {"model_call_id": "call-final", "status": "succeeded", "proposal_type": "final_answer"},
        ],
        "model_context_records": [{
            "model_call_id": "call-final",
            "status": "succeeded",
            "receipt": "captured_from_actual_model_request_messages",
            "source_items": [],
            "query_result_refs": [{"result_id": fact["result_id"]} for fact in facts],
        }],
    }
    if mutate is not None:
        mutate(observation, records)
    return observation


@pytest.mark.parametrize("profile", ["B0", "B1"])
def test_w05_composition_net_plan_lineage_passes_with_verified_origin(profile: str) -> None:
    _case, observation = _run_ownership_case(_NET_QUESTION, profile, recording_adapter=True)
    lineage = classify_w05_source_lineage(observation)
    assert lineage["status"] == "pass"
    assert lineage["errors"] == []
    composed = _composed_origins(lineage)
    assert [source["result_id"] for source in composed] == [observation["facts"][0]["result_id"]]
    assert composed[0]["model_call_ids"]
    assert set(composed[0]["model_call_ids"]) <= set(observation["model_call_ids"])
    assert lineage["composition_verdicts"] == [{
        "result_id": observation["facts"][0]["result_id"],
        "accepted": True,
        "reason": "net_fen_plan_components_verified",
    }]


def test_w05_composition_b0_net_plan_binds_single_model_call() -> None:
    _case, observation = _run_ownership_case(_NET_QUESTION, "B0", recording_adapter=True)
    lineage = classify_w05_source_lineage(observation)
    assert lineage["status"] == "pass" and lineage["errors"] == []
    assert _composed_origins(lineage)[0]["model_call_ids"] == observation["model_call_ids"]
    assert len(observation["model_call_ids"]) == 1


@pytest.mark.parametrize("forgery", ["two_calls", "non_query_proposal", "result_ids_mismatch", "tool_calls_count"])
def test_w05_composition_b0_unbound_call_is_rejected(forgery: str) -> None:
    _case, observation = _run_ownership_case(_NET_QUESTION, "B0", recording_adapter=True)
    record = observation["model_call_records"][0]
    if forgery == "two_calls":
        observation["model_call_ids"] = [*observation["model_call_ids"], "call-extra"]
    elif forgery == "non_query_proposal":
        record["response_shape"] = dict(record["response_shape"], action_name="search_catalog")
        record.pop("proposal", None)
    elif forgery == "result_ids_mismatch":
        observation["result_ids"] = ["result-other"]
    else:
        observation["execution_metrics"] = dict(observation["execution_metrics"], tool_calls=2)
    lineage = classify_w05_source_lineage(observation)
    assert lineage["status"] == "fail"
    assert "composite_result_has_no_bound_model_call" in lineage["errors"]
    assert _composed_origins(lineage) == []


@pytest.mark.parametrize("windows", [[_SEPT_WINDOW, _AUG_WINDOW], [_SEPT_WINDOW, _SEPT_WINDOW]], ids=["different_windows", "same_window"])
def test_w05_composition_two_plans_pass(windows) -> None:
    lineage = classify_w05_source_lineage(_b1_composition_observation(windows))
    assert lineage["status"] == "pass" and lineage["errors"] == []
    composed = _composed_origins(lineage)
    assert len(composed) == 2
    pairs = [tuple(source["component_result_ids"]) for source in composed]
    assert len({component for pair in pairs for component in pair}) == 4


def test_w05_composition_b1_final_answer_must_receive_composed_result() -> None:
    def mutate(observation, _records):
        observation["model_context_records"][0]["query_result_refs"] = []

    lineage = classify_w05_source_lineage(_b1_composition_observation([_SEPT_WINDOW], mutate=mutate))
    assert lineage["errors"] == ["bounded_answer_call_did_not_receive_same_run_query_result"]


def _drop(records, which: str, index: int) -> None:
    records.remove(_plan_components(records, which)[index])


def _forge(forgery: str):
    def mutate(observation, records):
        fact = observation["facts"][0]
        if forgery == "other_window_component":
            # Keep the September refund with the August gross.
            _drop(records, "gross", 0)
            _drop(records, "refund", 1)
            observation["facts"] = observation["facts"][:1]
            observation["composite_result_evidence"] = observation["composite_result_evidence"][:1]
        elif forgery.startswith("fact_"):
            field = forgery.removeprefix("fact_")
            fact[field] = {"metric_id": "gross_fen", "value": 12001, "unit": "CNY_yuan", "time_window": _AUG_WINDOW}[field]
        elif forgery == "arithmetic":
            _plan_components(records, "refund")[0]["rows"] = [{"refund_fen": 2999}]
        elif forgery in {"foreign_component", "foreign_component_same_result_id"}:
            gross = _plan_components(records, "gross")[0]
            result_id = gross["result_id"] if forgery.endswith("same_result_id") else "result-foreign-copy"
            records.append(dict(gross, tenant_id="B", result_id=result_id))
        elif forgery == "shared_pair":
            _drop(records, "gross", 1)
            _drop(records, "refund", 1)
    return mutate


@pytest.mark.parametrize(
    ("forgery", "windows", "expected_reasons"),
    [
        ("other_window_component", [_SEPT_WINDOW, _AUG_WINDOW], ["composition_params_mismatch"]),
        ("fact_metric_id", [_SEPT_WINDOW], ["composition_fact_mismatch"]),
        ("fact_value", [_SEPT_WINDOW], ["composition_fact_mismatch"]),
        ("fact_unit", [_SEPT_WINDOW], ["composition_fact_mismatch"]),
        ("fact_time_window", [_SEPT_WINDOW], ["composition_fact_mismatch"]),
        ("arithmetic", [_SEPT_WINDOW], ["composition_value_mismatch"]),
        ("foreign_component", [_SEPT_WINDOW], ["composition_component_owner_mismatch"]),
        ("foreign_component_same_result_id", [_SEPT_WINDOW], ["composition_component_owner_mismatch"]),
        ("shared_pair", [_SEPT_WINDOW, _SEPT_WINDOW], ["net_fen_plan_components_verified", "composition_components_already_used"]),
    ],
)
def test_w05_composition_forgery_rejected(forgery: str, windows, expected_reasons) -> None:
    lineage = classify_w05_source_lineage(_b1_composition_observation(windows, mutate=_forge(forgery)))
    assert lineage["status"] == "fail"
    assert "fact_result_missing_same_run_server_result" in lineage["errors"]
    assert [item["reason"] for item in lineage["composition_verdicts"]] == expected_reasons
    assert len(_composed_origins(lineage)) == expected_reasons.count("net_fen_plan_components_verified")


def test_w05_composition_missing_evidence_keeps_original_error() -> None:
    def mutate(observation, _records):
        observation.pop("composite_result_evidence")

    lineage = classify_w05_source_lineage(_b1_composition_observation([_SEPT_WINDOW], mutate=mutate))
    assert "fact_result_missing_same_run_server_result" in lineage["errors"]
    assert lineage["composition_verdicts"] == []


def test_w05_composition_none_result_id_rejected() -> None:
    def mutate(observation, records):
        observation["facts"][0]["result_id"] = None
        records.append(dict(_plan_components(records, "gross")[0], sql="SELECT 1", result_id=None))

    lineage = classify_w05_source_lineage(_b1_composition_observation([_SEPT_WINDOW], mutate=mutate))
    assert lineage["status"] == "fail"
    assert "fact_missing_result_id" in lineage["errors"]
    assert _composed_origins(lineage) == []


@pytest.mark.parametrize("row_value", [12000.0, True])
def test_w05_composition_verifier_requires_int_net_fen_in_evidence(row_value) -> None:
    from queryshield.evaluation.w05_provenance import verify_net_fen_composition

    context, records, tools, facts = _net_plan_parts([_SEPT_WINDOW])
    evidence = tools.get_result_evidence(str(facts[0]["result_id"]), context=context).as_dict()
    identity = {"run_id": context.run_id, "tenant_id": context.tenant_id, "principal_id": context.principal_id}
    assert verify_net_fen_composition(facts[0], evidence, records, **identity).accepted is True
    # 12000.0 == 12000 in Python; only the strict type check rejects it.
    forged_fact = dict(facts[0], value=1) if row_value is True else facts[0]
    forged = dict(evidence, rows=[{"net_fen": row_value}])
    verdict = verify_net_fen_composition(forged_fact, forged, records, **identity)
    assert (verdict.accepted, verdict.reason) == (False, "composition_fact_mismatch")


@pytest.mark.parametrize(
    ("gross_fen", "refund_fen"),
    [(-3000, -15000), (11999, -1), (-1, -12001)],
    ids=["both_negative", "negative_refund", "negative_gross"],
)
def test_negative_component_values_are_rejected_even_when_arithmetic_balances(gross_fen, refund_fen) -> None:
    from queryshield.evaluation.w05_provenance import verify_net_fen_composition

    context, records, tools, facts = _net_plan_parts([_SEPT_WINDOW])
    evidence = tools.get_result_evidence(str(facts[0]["result_id"]), context=context).as_dict()
    identity = {"run_id": context.run_id, "tenant_id": context.tenant_id, "principal_id": context.principal_id}
    # Each pair still satisfies gross - refund == the composed net value.
    assert facts[0]["value"] == 12000 == gross_fen - refund_fen
    _plan_components(records, "gross")[0]["rows"] = [{"gross_fen": gross_fen}]
    _plan_components(records, "refund")[0]["rows"] = [{"refund_fen": refund_fen}]

    verdict = verify_net_fen_composition(facts[0], evidence, records, **identity)
    assert (verdict.accepted, verdict.reason) == (False, "composition_value_mismatch")
