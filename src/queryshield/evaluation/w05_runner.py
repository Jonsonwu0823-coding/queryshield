"""W05's single-pass baseline and bounded-agent entry points over shared tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
from typing import Any

from queryshield.agent.config import RunConfig
from queryshield.agent.proposals import ExecutionContext
from queryshield.agent.runtime import (
    B0_MODEL_CALL_LIMIT,
    B0_PROFILE,
    B0_SYSTEM_PROMPT,
    B1_MODEL_CALL_LIMIT,
    B1_PROFILE,
    B1_TOOL_CALL_LIMIT,
    b1_result_payload,
    bind_facts_to_context as _bind_facts_to_context,
    build_b1_agent,
    outcome_for,
    render_fact_records as _render_fact_records,
    run_b0_single_pass,
)
from queryshield.agent.tenant_scope import has_explicit_foreign_tenant
from queryshield.evaluation.usage import usage_status_record
from queryshield.providers.contracts import ModelAdapter
from queryshield.tools.semantic import ControlledTools


def run_b1_bounded_agent(
    model: ModelAdapter,
    tools: ControlledTools,
    context: ExecutionContext,
    question: str,
    *,
    time_window: Mapping[str, object] | None = None,
    call_store: Any | None = None,
    run_config: RunConfig | None = None,
    initial_retrieval_items: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Evaluation entry for B1: the product assembly and result shaping.

    Only the ``BoundedAgent.run`` call differs from the product runtime: the
    W05 harness may pass its prepared initial retrieval items (removed in B2c).
    """

    if not isinstance(context, ExecutionContext):
        raise TypeError("context must be a server-created ExecutionContext")
    agent = build_b1_agent(model, tools, call_store=call_store, run_config=run_config)
    result = agent.run(
        context,
        question,
        request_time_window=time_window,
        _evaluation_initial_retrieval_items=initial_retrieval_items,
    )
    return b1_result_payload(result, context, question)


def run_w05_comparison_pair(
    model: ModelAdapter,
    tools: ControlledTools,
    context: ExecutionContext,
    question: str,
    *,
    time_window: Mapping[str, object] | None = None,
    call_store: Any | None = None,
    run_config: RunConfig | None = None,
) -> dict[str, object]:
    """Run B0 then B1 against one model/tools instance and one server identity.

    Separate run IDs prevent cross-profile result/call identity collisions. The
    authenticated tenant, principal and role stay identical; the B1 strategy is
    the only workflow difference.
    """

    if not isinstance(context, ExecutionContext):
        raise TypeError("context must be a server-created ExecutionContext")
    if not isinstance(tools, ControlledTools):
        raise TypeError("comparison requires the shared ControlledTools boundary")
    b0_context = replace(context, run_id=f"{context.run_id}-b0")
    b1_context = replace(context, run_id=f"{context.run_id}-b1")
    b0 = run_b0_single_pass(
        model,
        tools,
        b0_context,
        question,
        time_window=time_window,
    )
    b1 = run_b1_bounded_agent(
        model,
        tools,
        b1_context,
        question,
        time_window=time_window,
        call_store=call_store,
        run_config=run_config,
    )
    pre_model_rejection = has_explicit_foreign_tenant(question, context.tenant_id)
    expected_b0_calls = 0 if pre_model_rejection else 1
    if b0.get("model_call_count") != expected_b0_calls:
        raise AssertionError("B0 model generation count does not match its pre-model rejection path")
    if pre_model_rejection and (
        b1.get("model_call_count") != 0 or b1.get("tool_call_count") != 0
    ):
        raise AssertionError("foreign-tenant requests must be rejected before B1 model and tool calls")
    if b0_context.tenant_id != b1_context.tenant_id or b0_context.principal_id != b1_context.principal_id or b0_context.role != b1_context.role:
        raise AssertionError("B0/B1 server identity differs")
    return {
        "comparison_version": "w05-b0-b1-comparison-v1",
        "shared_runtime": {
            "same_model_adapter_object": True,
            "same_controlled_tools_object": True,
            "same_tenant_id": context.tenant_id,
            "same_principal_id": context.principal_id,
            "same_role": context.role,
            "profile_run_ids": {"B0": b0_context.run_id, "B1": b1_context.run_id},
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        },
        "profiles": {"B0": b0, "B1": b1},
    }


def normalize_profile_observation(
    profile: str,
    result: Mapping[str, object],
    *,
    case_id: str,
) -> dict[str, object]:
    """Normalize either runner's output into the W05 per-case oracle shape."""

    if profile not in {B0_PROFILE, B1_PROFILE}:
        raise ValueError("profile must be a registered W05 profile")
    if type(case_id) is not str or not case_id.strip():
        raise ValueError("case_id must be non-empty")
    if profile == B0_PROFILE:
        required = {"status", "terminal_state", "http_status", "facts", "invariants", "side_effects", "usage", "elapsed_ms"}
        if not required <= set(result):
            raise ValueError("B0 result is missing required oracle fields")
        provenance = {
            name: result[name]
            for name in ("error_code", "answer", "run_id", "rows", "model_call_ids", "result_ids", "sql_proposal", "pre_model_rejection")
            if name in result
        }
        return {"case_id": case_id, **{name: result[name] for name in required}, **provenance}

    status = result.get("status")
    outcome = outcome_for(status, result.get("error_code"))
    terminal = outcome.terminal_state
    public_status = outcome.public_status
    http_status = outcome.http_status
    events = result.get("events")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        events = ()
    tool_events = [event for event in events if isinstance(event, Mapping) and event.get("kind") == "tool_call"]
    readonly_attempts = sum(event.get("tool_name") == "query_readonly" for event in tool_events)
    readonly_successes = sum(
        event.get("tool_name") == "query_readonly" and event.get("status") == "succeeded"
        for event in tool_events
    )
    raw_facts = result.get("facts")
    facts_value = raw_facts.get("facts", []) if isinstance(raw_facts, Mapping) else []
    facts = [dict(item) for item in facts_value if isinstance(item, Mapping)] if isinstance(facts_value, Sequence) and not isinstance(facts_value, (str, bytes)) else []
    invariants_value = result.get("invariants")
    invariants = dict(invariants_value) if isinstance(invariants_value, Mapping) else {}
    usage_summary = result.get("usage_summary")
    call_count = result.get("model_call_count")
    if type(call_count) is not int:
        call_count = len(result.get("model_call_ids", ())) if isinstance(result.get("model_call_ids"), Sequence) else 0
    usage = usage_status_record(
        usage_summary if isinstance(usage_summary, Mapping) else None,
        expected_call_count=call_count,
    )
    return {
        "case_id": case_id,
        "status": public_status,
        "runtime_status": status,
        "http_status": http_status,
        "terminal_state": terminal,
        "facts": facts,
        "invariants": invariants,
        "side_effects": {
            "model_calls": result.get("model_call_count", 0),
            "readonly_queries": readonly_successes,
            "readonly_query_attempts": readonly_attempts,
            "fact_count": len(facts),
            "write_statements": 0,
            "cross_tenant_rows": 0,
            "unauthorized_facts": 0,
        },
        "usage": usage,
        "elapsed_ms": result.get("elapsed_ms"),
        "model_call_ids": list(result.get("model_call_ids", ())) if isinstance(result.get("model_call_ids"), Sequence) else [],
        "tool_event_count": len(tool_events),
        "error_code": result.get("error_code"),
        "answer": result.get("answer"),
        "action": result.get("action"),
        "run_id": result.get("run_id"),
        "pre_model_rejection": result.get("pre_model_rejection") is True,
    }


__all__ = [
    "B0_PROFILE",
    "B1_PROFILE",
    "run_b0_single_pass",
    "run_b1_bounded_agent",
    "run_w05_comparison_pair",
    "normalize_profile_observation",
]
