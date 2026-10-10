"""The single product runtime: profile assembly, result shaping and outcome mapping.

HTTP ``/queries`` (sync and async), ``/runs/{run_id}/resume`` and the
evaluation all assemble B0/B1 through the functions in this module.  The
evaluation wrapper may only add its own arguments to ``BoundedAgent.run``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
import os
from threading import Lock
from time import perf_counter
from typing import Any

from queryshield.agent.config import MULTI_AGENT_VERSIONS, NATIVE_VERSIONS, RunConfig
from queryshield.agent.context import build_context
from queryshield.agent.delegation import COORDINATOR
from queryshield.agent.graph import CLARIFICATION_NOT_NEEDED_CODE, AgentRunResult, BoundedAgent, GraphLimits
from queryshield.agent.proposals import (
    AskUserAction,
    DenyAction,
    ExecutionContext,
    FactRef,
    FinalAnswerAction,
    ModelCallStore,
    ProposalParseError,
    ResultEvidence,
    ToolCallAction,
    parse_query_proposal,
)
from queryshield.agent.tenant_scope import has_explicit_foreign_tenant
from queryshield.agent.tool_execution import (
    CLARIFICATION_VALUE_UNSUPPORTED_CODE,
    METRIC_CONTRADICTS_QUESTION_CODE,
    ClarificationRequiredError,
    ClarificationValueUnsupportedError,
    call_tool,
)
from queryshield.catalog.catalog import SemanticCatalog
from queryshield.catalog.phrases import ClarificationReading, read_clarifications, review_ask
from queryshield.facts import FactResolutionError, FactResolver
from queryshield.facts.facts import is_scalar_metric_result
from queryshield.agent.metric_intent import no_data_metric_names
from queryshield.facts.render import render_no_data_answer, render_verified_answer
from queryshield.providers.contracts import (
    ModelAdapter,
    ModelProviderError,
    new_local_call_id,
    new_request_id,
    usage_is_consistent,
)
from queryshield.tools.semantic import ControlledTools, ToolError


B0_PROFILE = "B0-single-pass"
B1_PROFILE = "B1-bounded-agent"
# The evaluation compares B0 and B1 only.
PROFILES = (B0_PROFILE, B1_PROFILE)
# The coordinator of B2 is the B1 agent with a delegate action; only the server picks it.
B2_PROFILE = "B2-multi-agent"
PRODUCT_PROFILES = PROFILES + (B2_PROFILE,)
# The bounded-graph profiles: they retrieve and can wait for the user (B0 does neither).
BOUNDED_PROFILES = (B1_PROFILE, B2_PROFILE)
DEFAULT_PRODUCT_PROFILE = B1_PROFILE
B0_MODEL_CALL_LIMIT = 1
B1_MODEL_CALL_LIMIT = 6
B1_TOOL_CALL_LIMIT = 8
B1_WALL_CLOCK_SECONDS = 60.0
B0_SYSTEM_PROMPT = """You are the W05 single-pass baseline. Use only the supplied question, the server metric_declaration rules, and approved table schema.
Return exactly one strict JSON object, with no markdown, using exactly one of these shapes:
{"type":"tool_call","name":"query_readonly","arguments":{"sql":"<read-only SQL>","params":{},"metrics":["<declared metric id>"],"time_window":{"start":"<UTC>","end":"<UTC>"}}}
{"type":"ask_user","clarification_id":"<catalog clarification rule id>","question":"<one clarification question>"}
{"type":"deny","reason":"<brief reason>"}
Each JSON member must appear once; duplicate member names are invalid. For a customer aggregate, use orders INNER JOIN customers on both tenant_id and customer_id, then return the existing result rows keyed by customer_id and the bound metric column; never request customer names through this path.
The "type" field is the action-kind discriminator, not the tool name. For a query_readonly call, type MUST be exactly "tool_call"; "query_readonly" belongs only in the "name" field. The parser rejects "query_readonly" as a type value.
Declare every metric you will report in arguments.metrics with its time_window; undeclared rows never become facts. For query_readonly, include both arguments.sql and arguments.params; params must be a JSON object of scalar values (use an empty object when there are no parameters). Do not add or omit fields. Prefer parameter placeholders in SQL and provide their values in params. Never put tenant identity or authorization claims in SQL parameters. Do not return a final answer before verified query facts exist. Never claim a query ran or invent a result. The server owns identity, tenant scope, permissions, SQL validation, execution, and factual results."""


# ---------------------------------------------------------------------------
# Shared outcome mapping: runtime status -> run status, public status,
# oracle terminal state and HTTP code.  HTTP entrypoints and the evaluation
# normalizer both read this table; nothing else maps statuses.
#
# Two different "limits" share a name and must not be confused:
# * status ``limit_reached`` is the Agent's own budget (model/tool calls, wall
#   clock, parallel budget) running out before an answer;
# * error code ``result_row_limit`` is the guarded executor refusing a result
#   larger than 100 rows (GuardedQueryError("limit_reached")).  The run fails
#   with that code and HTTP 422: the question as asked cannot be answered
#   within the row bound and the client should narrow it.  It is not a
#   provider/server failure (502) and not an Agent budget (LIMIT_REACHED).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOutcome:
    run_status: str
    public_status: str
    terminal_state: str
    http_status: int


RUN_OUTCOMES: Mapping[str, RunOutcome] = {
    "succeeded": RunOutcome("SUCCEEDED", "succeeded", "SUCCEEDED", 200),
    "waiting_user": RunOutcome("WAITING_USER", "waiting_user", "WAITING_USER", 202),
    # Accepted and persisted; it waits for a same-tenant approver.
    "waiting_approval": RunOutcome("WAITING_APPROVAL", "waiting_approval", "WAITING_APPROVAL", 202),
    "denied": RunOutcome("DENIED", "denied", "DENIED", 403),
    "failed": RunOutcome("FAILED", "failed", "FAILED", 502),
    # Agent budget exhausted before an answer; the oracle calls it UNKNOWN.
    "limit_reached": RunOutcome("LIMIT_REACHED", "unknown", "UNKNOWN", 502),
    # B0 provider timeout: the provider outcome is unknown.
    "timeout": RunOutcome("FAILED", "unknown", "UNKNOWN", 504),
}
_UNKNOWN_OUTCOME = RunOutcome("FAILED", "unknown", "UNKNOWN", 502)
# A failed run's HTTP code is refined by its error code.
FAILED_ERROR_HTTP: Mapping[str, int] = {
    "missing_model_configuration": 503,
    "invalid_model_configuration": 503,
    "invalid_provider_mode": 503,
    "missing_embedding_configuration": 503,
    "invalid_embedding_configuration": 503,
    "database_unavailable": 503,
    "invalid_database_configuration": 503,
    # The product found no active permission source to bind an approval to.
    "approval_permission_unavailable": 503,
    # The product knowledge base could not be loaded (not a permission problem).
    "knowledge_unavailable": 503,
    # A model gateway's quota or rate limit for this service's account: the service is
    # temporarily unavailable, not the end user sending too much, so not 429.
    "model_quota_exhausted": 503,
    "model_rate_limited": 503,
    "upstream_timeout": 504,
    "query_timeout": 504,
    "result_row_limit": 422,
    # The model's answer had nothing to ground it; like other unusable model
    # output (e.g. query_repair_limit) this is an upstream failure, not a bad request.
    "answer_not_grounded": 502,
    # A final answer whose declared basis the run contradicts (no_data after a
    # query, knowledge or no_data citing results): unusable model output.
    "answer_basis_conflict": 502,
    # The model kept asking a clarification the question already settles.
    CLARIFICATION_NOT_NEEDED_CODE: 502,
    # The question (or the user's clarification answer) asks for a scope no
    # catalog metric supports, e.g. cancelled orders: the request as asked
    # cannot be answered, like result_row_limit.
    CLARIFICATION_VALUE_UNSUPPORTED_CODE: 422,
    # The model declared a value other than the one the question names, and
    # kept doing so (B1) or had no repair turn (B0): unusable model output.
    METRIC_CONTRADICTS_QUESTION_CODE: 502,
    # MCP metadata tools (server setting): the server process could not serve,
    # timed out, broke the protocol, or returned something the host rejected.
    "mcp_unavailable": 503,
    "mcp_timeout": 504,
    "mcp_protocol_error": 502,
    "mcp_result_invalid": 502,
}


def outcome_for(status: object, error_code: object = None) -> RunOutcome:
    outcome = RUN_OUTCOMES.get(status) if type(status) is str else None
    if outcome is None:
        return _UNKNOWN_OUTCOME
    if status == "failed" and type(error_code) is str and error_code in FAILED_ERROR_HTTP:
        return RunOutcome(outcome.run_status, outcome.public_status, outcome.terminal_state, FAILED_ERROR_HTTP[error_code])
    return outcome


_RUN_STATUS_TO_RUNTIME = {
    outcome.run_status: status for status, outcome in RUN_OUTCOMES.items() if status != "timeout"
}


def http_status_for_run(run_status: object, error_code: object = None) -> int | None:
    """HTTP code for a persisted run status, from the same table (None if not a runtime outcome)."""

    status = _RUN_STATUS_TO_RUNTIME.get(run_status) if type(run_status) is str else None
    return outcome_for(status, error_code).http_status if status is not None else None


# ---------------------------------------------------------------------------
# Result shaping shared by every profile and entrypoint
# ---------------------------------------------------------------------------


def usage_record(call: object) -> dict[str, object]:
    usage = getattr(call, "usage", None)
    if usage is None or getattr(call, "usage_status", None) != "known":
        return {
            "usage_status": "unknown",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)
    if not usage_is_consistent(prompt, completion, total):
        return {
            "usage_status": "unknown",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    return {
        "usage_status": "known",
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


# The renderer of the B1 graph; with a phrase-table reading it adds the catalog basis of each metric.
render_fact_records = render_verified_answer


def bind_facts_to_context(
    facts: Sequence[Mapping[str, object]],
    context: ExecutionContext,
) -> list[dict[str, object]]:
    """Add server-owned ownership fields after result evidence was authorized."""

    bound: list[dict[str, object]] = []
    for fact in facts:
        item = dict(fact)
        item["tenant_id"] = context.tenant_id
        item["principal_id"] = context.principal_id
        window = item.get("time_window")
        if isinstance(window, Mapping):
            start = window.get("start")
            end = window.get("end")
            if type(start) is str and type(end) is str and len(start) >= 10 and len(end) >= 10:
                item["window_label"] = f"{start[:10]}/{end[:10]}"
        bound.append(item)
    return bound


# ---------------------------------------------------------------------------
# B0: one proposal call routed through the shared tool facade
# ---------------------------------------------------------------------------


def _b0_record(
    status: str,
    error_code: str | None,
    *,
    extra: Mapping[str, object],
) -> dict[str, object]:
    outcome = outcome_for(status, error_code)
    return {
        "profile": B0_PROFILE,
        "status": status,
        "terminal_state": outcome.terminal_state,
        "error_code": error_code,
        "http_status": outcome.http_status,
        **extra,
    }


def _b0_side_effects(*, readonly_queries: int = 0, fact_count: int = 0) -> dict[str, int]:
    return {
        "model_calls": 1,
        "readonly_queries": readonly_queries,
        "fact_count": fact_count,
        "write_statements": 0,
        "cross_tenant_rows": 0,
        "unauthorized_facts": 0,
    }


def _b0_pre_model_denial() -> dict[str, object]:
    """The question names another tenant: refused before any model call."""

    return _b0_record(
        "denied",
        "forbidden",
        extra={
            "model_call_count": 0,
            "tool_call_count": 0,
            "readonly_queries": 0,
            "repair_count": 0,
            "facts": [],
            "invariants": {},
            "rows": [],
            "side_effects": {
                "model_calls": 0,
                "tool_calls": 0,
                "readonly_queries": 0,
                "fact_count": 0,
                "write_statements": 0,
                "cross_tenant_rows": 0,
                "unauthorized_facts": 0,
            },
            "usage": {"usage_status": "not_run", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "model_call_ids": [],
            "pre_model_rejection": True,
            "elapsed_ms": 0,
        },
    )


def _b0_failed_call(
    status: str,
    error_code: str,
    *,
    usage: Mapping[str, object],
    model_call_id: str,
    started: float,
) -> dict[str, object]:
    """The record of a model call whose proposal never reached the tools."""

    return _b0_record(
        status,
        error_code,
        extra={
            "model_call_count": 1,
            "tool_call_count": 0,
            "readonly_queries": 0,
            "repair_count": 0,
            "facts": [],
            "invariants": {},
            "rows": [],
            "side_effects": _b0_side_effects(),
            "usage": usage,
            "model_call_ids": [model_call_id],
            "elapsed_ms": max(0, int((perf_counter() - started) * 1000)),
        },
    )


def _b0_answer_from_evidence(
    evidence: ResultEvidence, context: ExecutionContext, catalog: SemanticCatalog
) -> tuple[list[dict[str, object]], str | None, str | None]:
    """The facts, or the rowset reply, of a result the server bound to a declared metric."""

    result_id = evidence.result_id
    # Facts come only from bindings the server built and verified for this result
    # (from the model's catalog-checked declaration).
    verified_bindings = evidence.metric_bindings
    if is_scalar_metric_result(evidence):
        references = tuple(FactRef(result_id=result_id, metric_id=binding.metric_id) for binding in verified_bindings)
        envelope = FactResolver(catalog=catalog).resolve(references, context=context, evidences={result_id: evidence})
        return bind_facts_to_context(envelope.as_dict()["facts"], context), None, None
    # Several rows, or one row of a grouped query: the rowset rule.
    if evidence.row_count < 1:
        raise FactResolutionError("evidence_validation_failed", "a server-bound metric query returned no aggregate result")
    bound_positions = {
        item.metric_id.removeprefix("metric."): item.result_position for item in evidence.metric_bindings
    }
    if not any("customer_id" in row for row in evidence.rows) or any(
        type(row.get("customer_id")) is not str
        or any(bound_positions.get(binding.metric_id.removeprefix("metric.")) not in row for binding in verified_bindings)
        for row in evidence.rows
    ):
        raise FactResolutionError("evidence_validation_failed", "grouped customer result is missing its bound rowset columns")
    # Server JSON of the rows, but the column names are the model's SQL aliases.
    reply = json.dumps({"rows": [dict(row) for row in evidence.rows]}, ensure_ascii=False, sort_keys=True)
    return [], reply, "unverified"


def _b0_final_answer(action: FinalAnswerAction, catalog: SemanticCatalog) -> tuple[str, str | None, str | None, str | None]:
    """A final answer with no prior tool result: ``(status, error_code, reply, answer_status)``.

    One pass cannot assert a verified fact, and B0 has no query or retrieval
    source to ground plain text either.  basis no_data needs neither: the
    server writes the reply.
    """

    if action.basis != "query" and action.fact_refs:
        return "failed", "answer_basis_conflict", None, None
    if action.basis == "no_data":
        return "succeeded", None, render_no_data_answer(no_data_metric_names(catalog)), "no_data"
    if action.fact_refs:
        return "failed", "result_not_found", action.answer, None
    return "failed", "answer_not_grounded", None, None


# B0 calls these refusals "denied"; every other tool failure is "failed".
_B0_REFUSAL_CODES = frozenset(
    {"forbidden", "unauthorized", "approval_required", "statement_not_allowed", "table_not_allowed", "reserved_parameter"}
)


def _b0_tool_failure(exc: ToolError | FactResolutionError) -> tuple[str, str]:
    """``(status, error_code)`` of a failed query; a declaration the question contradicts is one of these."""

    if isinstance(exc, ToolError):
        return ("denied" if exc.code in _B0_REFUSAL_CODES else "failed"), exc.code
    return "failed", "evidence_validation_failed"


def _b0_ask_outcome(action: AskUserAction, clarifications: ClarificationReading | None) -> tuple[str, str | None]:
    """B0 has no repair turn: an ask the wording already settles fails, any other ask waits for the user."""

    verdict = review_ask(clarifications, action.clarification_id, action.question) if clarifications is not None else None
    if verdict is not None and verdict.decision == "not_needed":
        return "failed", CLARIFICATION_NOT_NEEDED_CODE
    return "waiting_user", None


def _b0_messages(server_context: str, question: str, schema: object) -> tuple[dict[str, str], dict[str, str]]:
    return (
        {"role": "system", "content": B0_SYSTEM_PROMPT + "\n\n" + server_context},
        {"role": "user", "content": "QUESTION\n" + question.strip() + "\n\nAPPROVED_TABLE_SCHEMA\n" + str(schema)},
    )


def run_b0_single_pass(
    model: ModelAdapter,
    tools: ControlledTools,
    context: ExecutionContext,
    question: str,
    *,
    time_window: Mapping[str, object] | None = None,
    run_config: RunConfig | None = None,
) -> dict[str, object]:
    """Make one proposal call, then route it through the exact shared tool facade."""

    if not isinstance(context, ExecutionContext):
        raise TypeError("context must be a server-created ExecutionContext")
    if type(question) is not str or not question.strip():
        raise ValueError("question must be non-empty")
    if not isinstance(tools, ControlledTools):
        raise TypeError("B0 requires the shared ControlledTools boundary")
    if has_explicit_foreign_tenant(question, context.tenant_id):
        return _b0_pre_model_denial()

    # 1. Build the one prompt: the approved table schema and the trusted server context.
    started = perf_counter()
    catalog = tools.catalog
    if run_config is None:
        run_config = RunConfig(profile=B0_PROFILE, catalog_version=catalog.catalog_version)
    if not isinstance(run_config, RunConfig) or run_config.profile != B0_PROFILE:
        raise TypeError("B0 requires the server-owned B0 RunConfig")
    table_names = sorted({entry.table for entry in catalog.entries if entry.kind == "table" and entry.table})
    schema = tools.describe_tables({"tables": table_names}, context=context)
    clarifications = read_clarifications(catalog, question) if catalog.has_phrase_table else None
    trusted_context = build_context(
        context,
        question,
        run_config=run_config,
        metric_catalog=catalog,
        request_time_window=time_window,
        # B0 is a single pass and never executes parallel reads.
        parallel_available=False,
    )
    messages = _b0_messages(trusted_context.messages[0]["content"], question, schema)
    request_id = new_request_id()
    model_call_id = new_local_call_id()

    # 2. Call the model once.
    try:
        call = model.complete(messages, request_id=request_id, model_call_id=model_call_id, run_id=context.run_id)
    except ModelProviderError as exc:
        usage = {"usage_status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        status = "timeout" if exc.code == "upstream_timeout" else "failed"
        return _b0_failed_call(status, exc.code, usage=usage, model_call_id=model_call_id, started=started)

    # 3. Parse its proposal with the same parser as B1.
    usage = usage_record(call)
    try:
        proposal = parse_query_proposal(call.content, context=context, model_call_id=model_call_id)
    except ProposalParseError as exc:
        return _b0_failed_call("failed", exc.code, usage=usage, model_call_id=model_call_id, started=started)

    # 4. Act on the proposal: a query runs through the shared tool facade, the other actions end the run.
    action = proposal.action
    sql_proposal = action.arguments.get("sql") if isinstance(action, ToolCallAction) else None
    model_answer = action.answer if isinstance(action, FinalAnswerAction) else None
    # verified only for a reply the server rendered from facts.
    answer_status: str | None = None
    result: Mapping[str, object] = {}
    status = "succeeded"
    error_code: str | None = None
    tool_call_count = 0
    readonly_queries = 0
    rows: list[Mapping[str, object]] = []
    facts: list[dict[str, object]] = []
    if isinstance(action, ToolCallAction):
        if action.name != "query_readonly":
            # B0 receives the approved schema in its sole prompt and has no
            # retrieval/repair turn. Never let the baseline silently use B1's
            # semantic retrieval strategy through a tool call.
            status = "failed"
            error_code = "baseline_tool_not_allowed"
        else:
            tool_call_count = 1
            readonly_queries = 1
            try:
                result = call_tool(
                    tools,
                    action.name,
                    action.arguments,
                    context=context,
                    request_time_window=time_window,
                    clarifications=clarifications,
                )
                raw_rows = result.get("rows")
                if type(raw_rows) is list:
                    rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)]
                result_id = result.get("result_id")
                evidence = tools.get_result_evidence(result_id, context=context) if type(result_id) is str else None
                if evidence is not None and evidence.metric_bindings:
                    facts, model_answer, answer_status = _b0_answer_from_evidence(evidence, context, catalog)
            except ClarificationRequiredError:
                # The same phrase-table check as B1: the wording leaves the
                # declared metric open, so nothing ran and the run waits.
                status = "waiting_user"
            except ClarificationValueUnsupportedError as exc:
                status = "failed"
                error_code = exc.code
                model_answer = exc.note
            except (ToolError, FactResolutionError) as exc:
                # Includes MetricContradictsQuestionError: B0 has no repair
                # turn, so a declaration the question contradicts fails before SQL.
                status, error_code = _b0_tool_failure(exc)
    elif isinstance(action, AskUserAction):
        status, error_code = _b0_ask_outcome(action, clarifications)
    elif isinstance(action, DenyAction):
        status = "denied"
    elif isinstance(action, FinalAnswerAction):
        status, error_code, model_answer, answer_status = _b0_final_answer(action, catalog)

    # 5. Shape the record the baseline reports.
    succeeded = status == "succeeded"
    return _b0_record(
        status,
        error_code,
        extra={
            "model_call_count": 1,
            "tool_call_count": tool_call_count,
            "readonly_queries": readonly_queries if succeeded else 0,
            "repair_count": 0,
            "facts": facts,
            "answer": render_fact_records(facts, clarifications=clarifications) if facts else model_answer,
            "answer_status": ("verified" if facts else answer_status) if succeeded else None,
            "sql_proposal": sql_proposal,
            "result_ids": [str(result["result_id"])] if type(result.get("result_id")) is str else [],
            "invariants": {},
            "rows": rows,
            "side_effects": _b0_side_effects(readonly_queries=readonly_queries if succeeded else 0, fact_count=len(facts)),
            "usage": usage,
            "model_call_ids": [model_call_id],
            "elapsed_ms": max(0, int((perf_counter() - started) * 1000)),
        },
    )


# ---------------------------------------------------------------------------
# B1: the bounded agent graph
# ---------------------------------------------------------------------------


def build_b1_agent(
    model: ModelAdapter,
    tools: ControlledTools,
    *,
    call_store: Any | None = None,
    run_config: RunConfig | None = None,
    retrieval_available: bool = True,
    on_step: Callable[[Mapping[str, object]], None] | None = None,
) -> BoundedAgent:
    """Assemble B1 with the server-owned budgets; the only B1 assembly."""

    if run_config is None:
        run_config = RunConfig(profile=B1_PROFILE)
    if not isinstance(run_config, RunConfig) or run_config.profile != B1_PROFILE:
        raise ValueError("B1 requires a server-owned evaluation RunConfig")
    return _bounded_agent(model, tools, call_store, run_config, retrieval_available, on_step)


def build_b2_agent(
    model: ModelAdapter,
    tools: ControlledTools,
    *,
    call_store: Any | None = None,
    run_config: RunConfig,
    retrieval_available: bool = True,
    on_step: Callable[[Mapping[str, object]], None] | None = None,
) -> BoundedAgent:
    """Assemble the B2 coordinator: B1's agent and budgets, plus delegate; the only B2 assembly."""

    if run_config.profile != B2_PROFILE:
        raise ValueError("B2 requires its server-owned RunConfig")
    return _bounded_agent(model, tools, call_store, run_config, retrieval_available, on_step, role=COORDINATOR)


def build_bounded_agent(model: ModelAdapter, tools: ControlledTools, *, run_config: RunConfig, **options: Any) -> BoundedAgent:
    """The B1 agent or the B2 coordinator, by the run's profile (a run and its resume use the same one)."""

    builder = build_b2_agent if run_config.profile == B2_PROFILE else build_b1_agent
    return builder(model, tools, run_config=run_config, **options)


def _bounded_agent(model, tools, call_store, run_config, retrieval_available, on_step, *, role=None) -> BoundedAgent:
    """B1's budgets: the whole run's, also for B2 (its sub-agents get shares of them)."""

    return BoundedAgent(
        model,
        tools=tools,
        call_store=call_store or ModelCallStore(),
        limits=GraphLimits(
            max_model_calls=B1_MODEL_CALL_LIMIT,
            max_tool_calls=B1_TOOL_CALL_LIMIT,
            max_wall_clock_seconds=B1_WALL_CLOCK_SECONDS,
        ),
        run_config=run_config,
        retrieval_available=retrieval_available,
        on_step=on_step,
        role=role,
    )


def b1_result_payload(
    result: AgentRunResult,
    context: ExecutionContext,
    question: str,
) -> dict[str, object]:
    """Shape one bounded-graph result (B1, or B2's coordinator); the only such shaping."""

    payload = result.as_dict()
    facts = payload.get("facts")
    if isinstance(facts, Mapping):
        fact_rows = facts.get("facts")
        if isinstance(fact_rows, Sequence) and not isinstance(fact_rows, (str, bytes)):
            payload["facts"] = {
                **dict(facts),
                "facts": bind_facts_to_context(
                    [item for item in fact_rows if isinstance(item, Mapping)],
                    context,
                ),
            }
    payload["pre_model_rejection"] = (
        has_explicit_foreign_tenant(question, context.tenant_id)
        and payload.get("status") == "denied"
        and payload.get("model_call_count") == 0
        and payload.get("tool_call_count") == 0
    )
    # Only B2 is labelled by its configuration: B1's graph also runs under evaluation configurations.
    profile = B2_PROFILE if result.run_config.profile == B2_PROFILE else B1_PROFILE
    return {"profile": profile, **payload}


# ---------------------------------------------------------------------------
# Product entry: server configuration and one profile run
# ---------------------------------------------------------------------------


class RuntimeConfigurationError(RuntimeError):
    """The server is configured in a way it must refuse to run (blocked, 503)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


_PROFILE_ALIASES = {"b0": B0_PROFILE, "b1": B1_PROFILE, "b2": B2_PROFILE} | {name.lower(): name for name in PRODUCT_PROFILES}


def configured_profile() -> str:
    """The server-owned profile; clients cannot choose it."""

    raw = os.getenv("QUERYSHIELD_AGENT_PROFILE", "").strip()
    if not raw:
        return DEFAULT_PRODUCT_PROFILE
    profile = _PROFILE_ALIASES.get(raw.lower())
    if profile is None:
        raise RuntimeConfigurationError("invalid_agent_profile", "QUERYSHIELD_AGENT_PROFILE is not a registered profile")
    return profile


def provider_mode() -> str:
    return os.getenv("QUERYSHIELD_PROVIDER_MODE", "fake").strip().lower() or "fake"


def model_protocol() -> str:
    """How the model returns its decision: a json action (default) or a native function call."""

    protocol = os.getenv("QUERYSHIELD_MODEL_PROTOCOL", "json").strip().lower() or "json"
    if protocol not in {"json", "native"}:
        raise RuntimeConfigurationError("invalid_model_protocol", "QUERYSHIELD_MODEL_PROTOCOL must be json or native")
    return protocol


def with_model_protocol(config: RunConfig) -> RunConfig:
    """``config`` under the configured protocol; callers apply it to B1 only (B0 is the json baseline)."""

    return replace(config, **NATIVE_VERSIONS) if model_protocol() == "native" else config


def model_for_mode(mode: str) -> ModelAdapter:
    """Fake and real providers are never mixed; an unknown mode is blocked."""

    if mode == "fake":
        from queryshield.providers.fake_model import FakeModel

        return FakeModel()
    if mode == "real":
        from queryshield.providers.openai_compatible import OpenAICompatibleModel

        return OpenAICompatibleModel.from_env()
    raise ModelProviderError(
        "invalid_provider_mode",
        {
            "status": "blocked",
            "mode": mode,
            "provider": None,
            "usage": None,
            "usage_status": "unknown",
            "error_code": "invalid_provider_mode",
        },
    )


def fake_database_requested() -> bool:
    return os.getenv("QUERYSHIELD_FAKE_DB", "").strip().lower() in {"1", "true", "yes"}


def check_fake_database_boundary(mode: str) -> None:
    """A fixture database may only serve the Fake provider.

    A real model over fixture rows could otherwise produce a "verified" answer
    that no real database ever returned.
    """

    if fake_database_requested() and mode != "fake":
        raise RuntimeConfigurationError(
            "fake_database_requires_fake_provider",
            "QUERYSHIELD_FAKE_DB is only allowed with QUERYSHIELD_PROVIDER_MODE=fake",
        )


def product_retriever(mode: str) -> Any | None:
    """The product retriever for this mode.

    Returns the hybrid retriever (default), ``CATALOG_SEARCH_ONLY`` when the
    server is set to catalog-only search, or None when retrieval is disabled.
    """

    from queryshield.db.readonly import DatabaseConfigurationError, check_demo_pairing, demo_dataset_enabled
    from queryshield.knowledge.runtime import (
        CATALOG_SEARCH_ONLY,
        retrieval_setting,
        shared_demo_retrieval_runtime,
        shared_retrieval_runtime,
    )
    from queryshield.providers.embedding import EmbeddingConfigurationError

    # The demo setting and the database name must agree (a fixed error code, no URL).
    # With the setting off and a database that is not a _demo one this changes nothing.
    try:
        check_demo_pairing(os.getenv("QUERYSHIELD_DATABASE_URL"))
        demo = demo_dataset_enabled()
    except DatabaseConfigurationError as exc:
        raise RuntimeConfigurationError(getattr(exc, "code", "invalid_database_configuration"), "the demo dataset setting and the database do not match") from exc
    setting = retrieval_setting()
    if setting == "disabled":
        return None
    if setting == "catalog":
        return CATALOG_SEARCH_ONLY
    if mode not in {"fake", "real"}:
        raise RuntimeConfigurationError("invalid_provider_mode", "retrieval mode must be fake or real")
    try:
        return (shared_demo_retrieval_runtime(mode) if demo else shared_retrieval_runtime(mode)).retriever
    except EmbeddingConfigurationError as exc:
        raise RuntimeConfigurationError(exc.code, "the embedding service is not configured") from exc
    except Exception as exc:  # noqa: BLE001 - never fall back to another retriever
        raise RuntimeConfigurationError("retrieval_unavailable", "the product retriever could not be built") from exc


class CountingExecutor:
    """Count the database executions one run actually made (a net plan counts 2)."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.executions = 0
        # B2's sub-agents query from several threads.
        self._lock = Lock()

    def execute(self, sql, *, context, params=(), metric_bindings=()):
        result = self.delegate.execute(sql, context=context, params=params, metric_bindings=metric_bindings)
        with self._lock:
            self.executions += 1
        return result

    def __getattr__(self, name: str) -> object:
        return getattr(self.delegate, name)


class CountingModel:
    """Count the chat calls one execution started, counted before the call so one that raises counts too."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls = 0
        # B2's sub-agents call the model from several threads.
        self._lock = Lock()

    def complete(self, messages, **kwargs):
        with self._lock:
            self.calls += 1
        return self.delegate.complete(messages, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self.delegate, name)


@dataclass
class RuntimeDependencies:
    """Everything one run needs; HTTP gets these through FastAPI dependencies."""

    model: ModelAdapter
    executor: Any
    retriever: Any | None
    call_store: Any
    profile: str = DEFAULT_PRODUCT_PROFILE


def product_run_config(profile: str, *, catalog: Any, retriever: Any | None) -> RunConfig:
    from queryshield.agent.config import DEFAULT_KNOWLEDGE_SNAPSHOT_ID

    snapshot = getattr(retriever, "snapshot", None)
    config = RunConfig(
        profile=profile,
        catalog_version=catalog.catalog_version,
        knowledge_snapshot_id=getattr(snapshot, "snapshot_id", DEFAULT_KNOWLEDGE_SNAPSHOT_ID),
    )
    protocol_config = with_model_protocol(config)  # read even for B0: a bad setting is blocked, not ignored
    if profile == B2_PROFILE:
        if protocol_config != config:
            raise RuntimeConfigurationError("invalid_model_protocol", "the multi-agent profile runs the json protocol only")
        return replace(config, **MULTI_AGENT_VERSIONS)
    return protocol_config if profile == B1_PROFILE else config


@dataclass
class ProfileRun:
    payload: dict[str, object]
    agent: BoundedAgent | None
    tools: ControlledTools
    run_config: RunConfig


def tools_retriever(retriever: Any | None) -> Any | None:
    """The object ControlledTools searches with; catalog-only search uses none."""

    from queryshield.knowledge.runtime import CATALOG_SEARCH_ONLY

    return None if retriever is None or retriever is CATALOG_SEARCH_ONLY else retriever


def retrieval_label(retriever: Any | None) -> str:
    from queryshield.knowledge.runtime import CATALOG_SEARCH_ONLY

    if retriever is None:
        return "disabled"
    return "catalog" if retriever is CATALOG_SEARCH_ONLY else "hybrid"


def product_tools(deps: RuntimeDependencies, *, executor: Any | None = None, metadata: Any | None = None) -> ControlledTools:
    """One run's tool facade; ``metadata`` (an MCP config) moves the two metadata tools to MCP.

    Only the product service passes ``metadata``; without it the facade is local.
    """

    from queryshield.catalog import load_default_catalog

    arguments = {
        "catalog": load_default_catalog(),
        "executor": executor if executor is not None else deps.executor,
        # B0 never retrieves (the comparison rule); B1 and B2 use the server retriever.
        "retriever": tools_retriever(deps.retriever) if deps.profile in BOUNDED_PROFILES else None,
    }
    if metadata is None:
        return ControlledTools(**arguments)
    from queryshield.mcp_metadata.tools import McpMetadataTools

    return McpMetadataTools(**arguments, metadata_config=metadata)


def run_profile(
    deps: RuntimeDependencies,
    context: ExecutionContext,
    question: str,
    *,
    time_window: Mapping[str, object] | None = None,
    tools: ControlledTools | None = None,
    on_step: Callable[[Mapping[str, object]], None] | None = None,
) -> ProfileRun:
    """Run the server-configured profile once; the product's only run entry.

    ``on_step`` sees the agent state after every graph step (B1 and B2; B0 has no steps).
    """

    if deps.profile not in PRODUCT_PROFILES:
        raise RuntimeConfigurationError("invalid_agent_profile", "the configured profile is not registered")
    tools = tools or product_tools(deps)
    run_config = product_run_config(deps.profile, catalog=tools.catalog, retriever=tools.retriever)
    if deps.profile == B0_PROFILE:
        payload = run_b0_single_pass(deps.model, tools, context, question, time_window=time_window, run_config=run_config)
        return ProfileRun(payload, None, tools, run_config)
    agent = build_bounded_agent(
        deps.model,
        tools,
        call_store=deps.call_store,
        run_config=run_config,
        retrieval_available=deps.retriever is not None,
        on_step=on_step,
    )
    result = agent.run(context, question, request_time_window=time_window)
    return ProfileRun(b1_result_payload(result, context, question), agent, tools, run_config)


__all__ = [
    "CountingExecutor",
    "CountingModel",
    "ProfileRun",
    "RuntimeConfigurationError",
    "RuntimeDependencies",
    "check_fake_database_boundary",
    "configured_profile",
    "http_status_for_run",
    "model_for_mode",
    "product_retriever",
    "product_run_config",
    "product_tools",
    "retrieval_label",
    "tools_retriever",
    "model_protocol",
    "provider_mode",
    "run_profile",
    "B0_PROFILE",
    "B0_SYSTEM_PROMPT",
    "B1_PROFILE",
    "B2_PROFILE",
    "BOUNDED_PROFILES",
    "DEFAULT_PRODUCT_PROFILE",
    "FAILED_ERROR_HTTP",
    "PRODUCT_PROFILES",
    "PROFILES",
    "RUN_OUTCOMES",
    "RunOutcome",
    "b1_result_payload",
    "bind_facts_to_context",
    "build_b1_agent",
    "build_b2_agent",
    "build_bounded_agent",
    "outcome_for",
    "render_fact_records",
    "run_b0_single_pass",
    "usage_record",
    "with_model_protocol",
]
