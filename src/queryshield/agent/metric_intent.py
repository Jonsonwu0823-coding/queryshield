"""Model-declared metric intent, bounded and built by the server catalog.

The model may only *declare* which catalog metrics a ``query_readonly`` call
computes and for which UTC window.  Every MetricBinding field (unit, source,
catalog version, plan id) comes from the server: the catalog entry or the
verifier registry below.  The declaration never becomes a fact by itself; the
bindings built here still have to pass the SQL projection check in
``tool_execution`` or the server-owned net_fen plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import re
from typing import Final

from queryshield.agent.context import NET_FEN_PLAN_ID, NO_DATA_ACTION, QUERY_DECLARATION_ERROR_CODES
from queryshield.agent.proposals import MetricBinding
from queryshield.catalog.catalog import ClarificationRule, SemanticCatalog


MAX_DECLARED_METRICS: Final = 4
# Catalog metrics are aggregates over an explicit window.  One year (leap years
# included) covers annual and year-to-date questions; anything wider is the
# "all history" request that the catalog time-window rule sends to clarification.
MAX_TIME_WINDOW_DAYS: Final = 366

# Server algorithms that can prove a declared metric.  Keys must be catalog
# metric ids; plan ids come only from here, never from the model.
METRIC_VERIFIERS: Final[Mapping[str, Mapping[str, str | None]]] = {
    "paid_count": {"verification": "sql_projection", "plan_id": None},
    "gross_fen": {"verification": "sql_projection", "plan_id": None},
    "net_fen": {"verification": "server_plan", "plan_id": NET_FEN_PLAN_ID},
}
DECLARATION_ERROR_CODES: Final = frozenset(QUERY_DECLARATION_ERROR_CODES)
_DECLARATION_FIELDS: Final = frozenset({"metrics", "time_window"})
_UTC_INSTANT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_EXAMPLE_WINDOW: Final = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}
_EXAMPLE_PARAMS: Final = {"0": "paid", "1": _EXAMPLE_WINDOW["start"], "2": _EXAMPLE_WINDOW["end"]}
_WINDOW_FILTER = "status = %s AND created_at >= %s AND created_at < %s"
# Every example shown to the model must pass the real server checks; tests run
# each one through call_tool.
DECLARATION_EXAMPLES: Final[Mapping[str, Mapping[str, object]]] = {
    "paid_count_and_gross_fen": {
        "sql": f"SELECT COUNT(*) AS paid_count, COALESCE(SUM(amount_fen), 0) AS gross_fen FROM orders WHERE {_WINDOW_FILTER}",
        "params": dict(_EXAMPLE_PARAMS),
        "metrics": ["paid_count", "gross_fen"],
        "time_window": dict(_EXAMPLE_WINDOW),
    },
    "net_fen": {
        "sql": f"SELECT COALESCE(SUM(amount_fen), 0) AS gross_fen FROM orders WHERE {_WINDOW_FILTER}",
        "params": dict(_EXAMPLE_PARAMS),
        "metrics": ["net_fen"],
        "time_window": dict(_EXAMPLE_WINDOW),
    },
}


class MetricDeclarationError(ValueError):
    """A model metric declaration failed a server rule."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class ResolvedDeclaration:
    """Tool arguments without declaration fields plus the server bindings."""

    arguments: Mapping[str, object]
    bindings: tuple[MetricBinding, ...]
    declared_metric_ids: tuple[str, ...]


def _instant(value: object, *, field: str) -> datetime:
    if type(value) is not str or not _UTC_INSTANT.fullmatch(value):
        raise MetricDeclarationError("invalid_time_window", f"{field} must be YYYY-MM-DDTHH:MM:SSZ in UTC")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise MetricDeclarationError("invalid_time_window", f"{field} is not a valid UTC instant") from exc


def normalize_time_window(value: object, *, field: str = "time_window") -> dict[str, str]:
    """Return the canonical ``{start, end, timezone: UTC}`` half-open window.

    ``{start, end}`` and ``{start, end, timezone: "UTC"}`` normalize to the
    same value, so request, declared and pre-bound windows compare equal.
    """

    if not isinstance(value, Mapping):
        raise MetricDeclarationError("invalid_time_window", f"{field} must be an object")
    keys = set(value)
    if not {"start", "end"} <= keys or keys - {"start", "end", "timezone"}:
        raise MetricDeclarationError("invalid_time_window", f"{field} fields must be start, end and optional timezone")
    if "timezone" in value and value["timezone"] != "UTC":
        raise MetricDeclarationError("invalid_time_window", f"{field}.timezone must be UTC")
    start = _instant(value["start"], field=f"{field}.start")
    end = _instant(value["end"], field=f"{field}.end")
    if start >= end:
        raise MetricDeclarationError("invalid_time_window", f"{field} start must be before end")
    if end - start > timedelta(days=MAX_TIME_WINDOW_DAYS):
        raise MetricDeclarationError(
            "invalid_time_window",
            f"{field} spans more than {MAX_TIME_WINDOW_DAYS} days",
        )
    return {"start": str(value["start"]), "end": str(value["end"]), "timezone": "UTC"}


_WINDOW_FORMAT_HINT = (
    'use {"start":"YYYY-MM-DDTHH:MM:SSZ","end":"YYYY-MM-DDTHH:MM:SSZ"}: UTC, half-open [start,end), '
    f"at most {MAX_TIME_WINDOW_DAYS} days, same values in the SQL params"
)


def _window_guidance(server_window: Mapping[str, str] | None) -> str:
    """Fixed format guidance plus the server's own window; never echoes model text."""

    if server_window is None:
        return _WINDOW_FORMAT_HINT
    window = json.dumps(
        {"start": server_window["start"], "end": server_window["end"]},
        separators=(",", ":"),
    )
    return f"{_WINDOW_FORMAT_HINT}; the request window is {window}, declare exactly it"


def _catalog_metric(catalog: SemanticCatalog, metric_id: str):
    try:
        return catalog.metric(metric_id)
    except KeyError:
        return None


def declarable_metric_ids(catalog: SemanticCatalog) -> tuple[str, ...]:
    """Catalog metrics, in catalog order, that a server verifier can prove."""

    return tuple(
        entry.id.removeprefix("metric.")
        for entry in catalog.entries
        if entry.kind == "metric" and entry.id.removeprefix("metric.") in METRIC_VERIFIERS
    )


def build_metric_binding(
    catalog: SemanticCatalog,
    metric_id: str,
    time_window: Mapping[str, object],
) -> MetricBinding:
    """Build one binding whose fields all come from the catalog or verifier registry."""

    entry = _catalog_metric(catalog, metric_id)
    if entry is None:
        raise MetricDeclarationError("unknown_metric", "a declared metric is not a catalog metric")
    verifier = METRIC_VERIFIERS.get(metric_id)
    if verifier is None:
        raise MetricDeclarationError("metric_not_verifiable", "the server has no trusted verifier for a declared metric")
    return MetricBinding(
        metric_id=metric_id,
        result_position=metric_id,
        unit=str(entry.payload["unit"]),
        time_window=normalize_time_window(time_window),
        catalog_source_id=entry.source_id,
        catalog_version=catalog.catalog_version,
        plan_id=verifier["plan_id"],
    )


def _declarable_guidance(catalog: SemanticCatalog) -> str:
    """Catalog-derived guidance; never echoes the model's metric ids."""

    ids = json.dumps(list(declarable_metric_ids(catalog)))
    return f"declarable metric ids are {ids}; for an amount after refunds declare only net_fen"


def _declared_metric_ids(value: object, catalog: SemanticCatalog) -> tuple[str, ...]:
    if type(value) is not list or not 1 <= len(value) <= MAX_DECLARED_METRICS:
        raise MetricDeclarationError(
            "invalid_metric_declaration",
            f"metrics must list one to {MAX_DECLARED_METRICS} catalog metric ids",
        )
    if any(type(item) is not str for item in value):
        raise MetricDeclarationError("invalid_metric_declaration", "metrics items must be metric id strings")
    if len(set(value)) != len(value):
        raise MetricDeclarationError("invalid_metric_declaration", "metrics must not repeat a metric id")
    for metric_id in value:
        if _catalog_metric(catalog, metric_id) is None:
            raise MetricDeclarationError(
                "unknown_metric",
                f"a declared metric is not a catalog metric; {_declarable_guidance(catalog)}",
            )
        if metric_id not in METRIC_VERIFIERS:
            raise MetricDeclarationError(
                "metric_not_verifiable",
                f"the server has no trusted verifier for a declared metric; {_declarable_guidance(catalog)}",
            )
    if "net_fen" in value and len(value) > 1:
        raise MetricDeclarationError("invalid_metric_declaration", "net_fen uses the server plan and must be declared alone")
    return tuple(value)


def resolve_query_declaration(
    arguments: Mapping[str, object],
    *,
    catalog: SemanticCatalog | None,
    request_time_window: Mapping[str, object] | None = None,
    prebound: Sequence[MetricBinding] = (),
) -> ResolvedDeclaration:
    """Validate a query_readonly declaration and return catalog-built bindings.

    Without ``metrics`` the call keeps the server pre-bound bindings (possibly
    none) and can only return rows.  With both a declaration and pre-bound
    (user-confirmed) bindings, the declaration must keep every confirmed
    metric, must not add another option of a confirmed clarification, and uses
    the confirmed window; other declared metrics are built from the catalog.
    """

    prebound = tuple(prebound)
    if not isinstance(arguments, Mapping):
        return ResolvedDeclaration(arguments, prebound, ())
    stripped = {key: value for key, value in arguments.items() if key not in _DECLARATION_FIELDS}
    request = (
        normalize_time_window(request_time_window, field="request_time_window")
        if request_time_window is not None
        else None
    )
    prebound_windows = [normalize_time_window(binding.time_window) for binding in prebound]
    if request is not None and any(window != request for window in prebound_windows):
        raise MetricDeclarationError("time_window_mismatch", "the pre-bound window differs from the request window")
    if "metrics" not in arguments:
        if "time_window" in arguments:
            raise MetricDeclarationError("invalid_metric_declaration", "time_window is only valid with metrics")
        return ResolvedDeclaration(stripped, prebound, ())
    if not isinstance(catalog, SemanticCatalog):
        raise MetricDeclarationError("invalid_metric_declaration", "metric declarations require the server catalog")

    metric_ids = _declared_metric_ids(arguments["metrics"], catalog)
    reference = request if request is not None else (prebound_windows[0] if prebound_windows else None)
    try:
        window = normalize_time_window(arguments["time_window"]) if "time_window" in arguments else None
    except MetricDeclarationError as exc:
        raise MetricDeclarationError(exc.code, f"{exc.message}; {_window_guidance(reference)}") from exc
    if window is None:
        window = reference
    elif reference is not None and window != reference:
        raise MetricDeclarationError(
            "time_window_mismatch",
            f"the declared window differs from the server window; {_window_guidance(reference)}",
        )
    if window is None:
        raise MetricDeclarationError(
            "invalid_time_window",
            f"declared metrics require a time_window; {_window_guidance(None)}",
        )

    if prebound:
        confirmed = {binding.metric_id.removeprefix("metric."): binding for binding in prebound}
        if not set(confirmed) <= set(metric_ids):
            raise MetricDeclarationError(
                "metric_declaration_mismatch",
                "declared metrics must keep every server-confirmed metric",
            )
        for rule in catalog.clarifications:
            options = {value.value for value in rule.values if value.metric == value.value}
            if options & set(confirmed) and (options & set(metric_ids)) - set(confirmed):
                raise MetricDeclarationError(
                    "metric_declaration_mismatch",
                    "declared metrics replace a server-confirmed clarification choice",
                )
        bindings = tuple(
            confirmed[metric_id] if metric_id in confirmed else build_metric_binding(catalog, metric_id, window)
            for metric_id in metric_ids
        )
        return ResolvedDeclaration(stripped, bindings, metric_ids)
    bindings = tuple(build_metric_binding(catalog, metric_id, window) for metric_id in metric_ids)
    return ResolvedDeclaration(stripped, bindings, metric_ids)


def declaration_examples(request_time_window: Mapping[str, str] | None = None) -> dict[str, dict[str, object]]:
    """The concrete declaration examples shown to the model.

    With a request window, the examples use it (in time_window and the SQL
    params) so the model copies the window it must declare.
    """

    examples = {name: json.loads(json.dumps(example)) for name, example in DECLARATION_EXAMPLES.items()}
    if request_time_window is not None:
        window = {"start": request_time_window["start"], "end": request_time_window["end"]}
        for example in examples.values():
            example["time_window"] = dict(window)
            example["params"] = {"0": "paid", "1": window["start"], "2": window["end"]}
    return examples


def undeclared_metric_hint(
    catalog: SemanticCatalog,
    cited_metric_ids: Sequence[str],
    request_time_window: Mapping[str, str] | None,
) -> dict[str, object]:
    """Repair hint for a fact_ref whose result has no verified binding.

    Only catalog-checked declarable metric ids and the server request window are
    echoed; any other model text is dropped.
    """

    declarable = declarable_metric_ids(catalog)
    metrics = [metric_id for metric_id in dict.fromkeys(cited_metric_ids) if metric_id in declarable]
    return {
        "action": (
            "Send a tool_call named query_readonly again with arguments.metrics and time_window, then cite that "
            "result's verified_metrics in fact_refs. If request_time_window is set, copy it exactly."
        ),
        "declare_metrics": metrics,
        "request_time_window": (
            {"start": request_time_window["start"], "end": request_time_window["end"]}
            if request_time_window is not None
            else None
        ),
    }


def answer_without_query_hint(request_time_window: Mapping[str, str] | None) -> dict[str, object]:
    """Repair hint for a final answer given before any successful query in this run.

    Fixed text plus the server request window; no model text is echoed.
    """

    return {
        "action": (
            "First send a tool_call named query_readonly with arguments.metrics and time_window, then cite only "
            "result_id values returned in this run. A window with no data is still queried and reported as 0."
        ),
        "request_time_window": (
            {"start": request_time_window["start"], "end": request_time_window["end"]}
            if request_time_window is not None
            else None
        ),
    }


def _request_window_echo(request_time_window: Mapping[str, str] | None) -> dict[str, str] | None:
    if request_time_window is None:
        return None
    return {"start": request_time_window["start"], "end": request_time_window["end"]}


def answer_not_grounded_hint(
    request_time_window: Mapping[str, str] | None,
    *,
    retrieval_available: bool = True,
) -> dict[str, object]:
    """Send-back hint for an answer nothing in this run grounds (B3c-2).

    Querying for business values comes first; knowledge and no_data are only
    conditions, so a data question is never steered away from its query.
    R1: each condition carries an action the model can copy as is (every one
    parses).  Fixed text plus the server request window; no model or user text.
    """

    hint: dict[str, object] = {
        "action": (
            "No query result in this run supports this answer. Business values (counts, amounts, totals) always "
            "need a query: first send a tool_call named query_readonly with arguments.metrics and time_window, "
            "then cite its result. A window with no data is still queried and reported as 0."
        ),
    }
    if retrieval_available:
        hint["only_if_definition"] = (
            "Only if the question asks how a metric is defined, not for any value: first send "
            "definition_search_action, then answer in definition_answer_shape (basis knowledge)."
        )
        hint["definition_search_action"] = DEFINITION_SEARCH_ACTION
        hint["definition_answer_shape"] = DEFINITION_ANSWER_SHAPE
    hint["only_if_no_data_needed"] = (
        "Only if the question needs no business data at all (a greeting, what you can do): send exactly "
        "no_data_action."
    )
    hint["no_data_action"] = NO_DATA_ACTION
    hint["request_time_window"] = _request_window_echo(request_time_window)
    return hint


# Copyable actions in the answer send-back hint (B3c-2 R1); NO_DATA_ACTION is
# the context's own example.  The knowledge source_ids come from the server's
# retrieval sources, so [] is fine.
DEFINITION_SEARCH_ACTION = '{"type":"tool_call","name":"search_catalog","arguments":{"query":"<指标名>","top_k":3}}'
DEFINITION_ANSWER_SHAPE = '{"type":"final_answer","answer":"...","source_ids":[],"fact_refs":[],"basis":"knowledge"}'


def knowledge_from_server_search_hint() -> dict[str, object]:
    """Send-back hint after the server searched the catalog for a knowledge answer (B3c-2 R2).

    Knowledge only: no query or no_data menu, no request window.  Fixed text.
    """

    return {
        "action": (
            "The server ran search_catalog with this run's question; its result is the search_catalog tool result "
            "just before this message. Say how the metric is defined using only those catalog items, with no "
            "business values: send a final_answer with basis knowledge and fact_refs []."
        ),
        "answer_shape": DEFINITION_ANSWER_SHAPE,
    }


def answer_basis_conflict_hint(request_time_window: Mapping[str, str] | None) -> dict[str, object]:
    """Send-back hint for a basis the run contradicts (B3c-2); fixed text only."""

    return {
        "action": (
            "Business values: answer with basis query (the default) and cite this run's verified_metrics in "
            "fact_refs. basis no_data cannot follow a query or cite fact_refs; basis knowledge cannot cite fact_refs."
        ),
        "request_time_window": _request_window_echo(request_time_window),
    }


def metric_declaration_contract(
    catalog: SemanticCatalog,
    request_time_window: Mapping[str, str] | None,
) -> dict[str, object]:
    """Model-facing declaration rules generated from the catalog at runtime.

    The paid_count+gross_fen example is placed in the action contract by the
    context builder; the net_fen example stays here.
    """

    metrics = []
    for metric_id in declarable_metric_ids(catalog):
        entry = catalog.metric(metric_id)
        metrics.append({"metric_id": metric_id, "unit": entry.payload.get("unit"), "definition": entry.text})
    return {
        "rule": (
            "An answer that reports a metric value MUST declare it in arguments.metrics + time_window of a tool_call "
            "named query_readonly, projected under the same lowercase alias. net_fen is declared alone: do not JOIN "
            "refunds or subtract refunds yourself; send the orders query of the net_fen example and the server plan "
            "computes refunds. Undeclared rows are never facts; fact_refs cite returned verified_metrics."
        ),
        "metrics": metrics,
        "time_window": {
            "format": "UTC YYYY-MM-DDTHH:MM:SSZ, [start,end), same values in SQL params",
            "max_days": MAX_TIME_WINDOW_DAYS,
            "request_time_window": dict(request_time_window) if request_time_window is not None else None,
            "if_request_time_window": "declare exactly it",
        },
        "example_arguments": [declaration_examples(request_time_window)["net_fen"]],
        # "apply_rule" sorts before "rules" in the canonical JSON the model sees.
        "clarifications": _clarification_contract(catalog),
    }


CLARIFICATION_APPLY_RULE: Final = (
    "ask_user with the rule's clarification_id only if the question uses its ambiguous_phrases and no option "
    "phrase; a named option phrase (支付金额=gross_fen) means query directly. If neither the question nor "
    "request_time_window states a time range, ask_user only for it, without clarification_id; never guess one. "
    "The server checks both ways; an empty window is queried and reported as 0."
)


def _clarification_contract(catalog: SemanticCatalog) -> dict[str, object]:
    """The catalog phrase table as the model sees it (catalog strings only)."""

    if not catalog.has_phrase_table:
        # Pre-v3 catalogs have no phrase table; keep their rule text.
        return {
            "apply_rule": CLARIFICATION_APPLY_RULE,
            "rules": [
                {"id": rule.id, "condition": rule.condition, "question": rule.question, "allowed_values": list(rule.allowed_values)}
                for rule in catalog.clarifications
            ],
        }
    rules = [_clarification_rule(catalog, rule) for rule in catalog.clarifications if len(rule.values) > 1]
    # Single-option rules are never asked: the server applies them and states
    # the premise in the answer.
    premises = {rule.id: rule.values[0].definition for rule in catalog.clarifications if len(rule.values) == 1}
    return {"apply_rule": CLARIFICATION_APPLY_RULE, "fixed_premises": premises, "rules": rules}


def _clarification_rule(catalog: SemanticCatalog, rule: ClarificationRule) -> dict[str, object]:
    """One multi-option rule: its ambiguous phrases and each option's phrases.

    Metric options list the metric's catalog phrases (ids are shown as the
    option itself); unsupported options are marked.  Only multi-option rules
    are listed; ids and phrases come from the catalog.  The server-side
    resolution text is not shown.
    """

    options = []
    for value in rule.values:
        # A metric option shows its metric's phrases; an option that implies a
        # metric (paid <- paid_count) shows its own phrases and names the metric.
        phrases = list(value.phrases) if value.phrases else list(catalog.metric_phrases(value.metric or ""))
        option: dict[str, object] = {
            "value": value.value,
            "phrases": [item for item in phrases if item != value.value and item != value.metric],
        }
        if value.metric is not None and value.metric != value.value:
            option["metric"] = value.metric
        if not value.supported:
            option["supported"] = False
        options.append(option)
    # The rule's question is not repeated here: the server asks the catalog
    # question of the rule named by clarification_id (see the ask_user example).
    return {
        "id": rule.id,
        "ambiguous_phrases": list(rule.ambiguous_phrases),
        "options": options,
    }


__all__ = [
    "CLARIFICATION_APPLY_RULE",
    "DECLARATION_ERROR_CODES",
    "DECLARATION_EXAMPLES",
    "MAX_DECLARED_METRICS",
    "MAX_TIME_WINDOW_DAYS",
    "METRIC_VERIFIERS",
    "MetricDeclarationError",
    "ResolvedDeclaration",
    "build_metric_binding",
    "answer_basis_conflict_hint",
    "answer_not_grounded_hint",
    "declarable_metric_ids",
    "declaration_examples",
    "undeclared_metric_hint",
    "answer_without_query_hint",
    "metric_declaration_contract",
    "normalize_time_window",
    "resolve_query_declaration",
]
