"""A persisted WAITING_USER checkpoint is read back as untrusted input: every field is checked again.

Each row damages one field of a real checkpoint and states the refusal the
restore gives.  The refusals are the stored-state boundary of the agent, so a
cleanup must keep each one with its code and its place in the order.
"""

from __future__ import annotations

import copy
import json

import pytest

import agent_core_scenarios as scenarios
import test_clarification as clarification_cases
from queryshield.agent.graph import AGENT_CHECKPOINT_VERSION, BoundedAgent, RunResumeError, _serialize_agent_checkpoint
from queryshield.agent.parallel import BranchExecution, ParallelPlan, ParallelScheduler
from queryshield.agent.proposals import ExecutionContext


def _set(**fields):
    return lambda checkpoint: {**checkpoint, **fields}


def _drop(*names):
    return lambda checkpoint: {key: value for key, value in checkpoint.items() if key not in names}


def _inside(key, **fields):
    return lambda checkpoint: {**checkpoint, key: {**checkpoint[key], **fields}}


def _inside_first(key, **fields):
    return lambda checkpoint: {**checkpoint, key: [{**checkpoint[key][0], **fields}]}


def _drop_inside(key, *names):
    return lambda checkpoint: {**checkpoint, key: {k: v for k, v in checkpoint[key].items() if k not in names}}


OTHER_PRINCIPAL = {"principal_id": "someone-else"}
SCHEMA = "checkpoint fields do not match the server schema"
VERSION = "checkpoint version or state is invalid"
IDENTITY = "checkpoint identity is invalid"
CONFIG = "checkpoint run configuration is invalid"
BINDINGS = "checkpoint metric bindings are invalid"
PLAN = "checkpoint parallel plan is invalid"
MISMATCH = "the resume context does not match the original run identity"

# (id, which checkpoint, damage, code, message)
REFUSALS = [
    ("not-a-mapping", "base", lambda checkpoint: [], "invalid_checkpoint", VERSION),
    ("version-unknown", "base", _set(checkpoint_version="x"), "invalid_checkpoint", VERSION),
    ("version-missing", "base", _drop("checkpoint_version"), "invalid_checkpoint", VERSION),
    ("version-v2", "base", _set(checkpoint_version="qs-bounded-agent-checkpoint-v2"), "invalid_checkpoint", VERSION),
    ("version-v1", "base", _set(checkpoint_version="qs-bounded-agent-checkpoint-v1"), "invalid_checkpoint", VERSION),
    ("version-unhashable", "base", _set(checkpoint_version=["v4"]), "invalid_checkpoint", VERSION),
    ("extra-field", "base", _set(extra=1), "invalid_checkpoint", SCHEMA),
    ("missing-field", "base", _drop("events"), "invalid_checkpoint", SCHEMA),
    ("missing-v4-only-field", "base", _drop("answer_bounce_count"), "invalid_checkpoint", SCHEMA),
    ("status", "base", _set(status="RUNNING"), "invalid_checkpoint", VERSION),
    ("time-window", "base", _set(request_time_window={"start": 1}), "invalid_checkpoint", "checkpoint request time window is invalid"),
    ("context-not-a-mapping", "base", _set(context="x"), "invalid_checkpoint", IDENTITY),
    ("context-extra-key", "base", _inside("context", extra=1), "invalid_checkpoint", IDENTITY),
    ("context-empty-role", "base", _inside("context", role=""), "invalid_checkpoint", IDENTITY),
    ("context-other-principal", "base", _inside("context", **OTHER_PRINCIPAL), "resume_context_mismatch", MISMATCH),
    ("context-mismatch-comes-before-field-errors", "base", lambda c: {**_inside("context", **OTHER_PRINCIPAL)(c), "question": ""}, "resume_context_mismatch", MISMATCH),
    ("question-blank", "base", _set(question="  "), "invalid_checkpoint", "checkpoint question is invalid"),
    ("question-not-text", "base", _set(question=5), "invalid_checkpoint", "checkpoint question is invalid"),
    ("waiting-question", "base", _set(waiting_question=None), "invalid_checkpoint", "checkpoint waiting prompt is invalid"),
    ("clarification-id-blank", "base", _set(waiting_clarification_id=" "), "invalid_checkpoint", "checkpoint clarification rule is invalid"),
    ("clarification-id-not-text", "base", _set(waiting_clarification_id=3), "invalid_checkpoint", "checkpoint clarification rule is invalid"),
    ("clarification-bounces-negative", "base", _set(clarification_bounce_count=-1), "invalid_checkpoint", "checkpoint clarification bounce count is invalid"),
    ("clarification-bounces-bool", "base", _set(clarification_bounce_count=True), "invalid_checkpoint", "checkpoint clarification bounce count is invalid"),
    ("answer-bounces-negative", "base", _set(answer_bounce_count=-1), "invalid_checkpoint", "checkpoint answer bounce count is invalid"),
    ("answer-bounces-float", "base", _set(answer_bounce_count=1.0), "invalid_checkpoint", "checkpoint answer bounce count is invalid"),
    ("run-config-not-a-mapping", "base", _set(run_config=[]), "invalid_checkpoint", CONFIG),
    ("run-config-wrong-version", "base", _inside("run_config", run_config_version="run-config-v2"), "invalid_checkpoint", CONFIG),
    ("run-config-missing-version", "base", _drop_inside("run_config", "run_config_version"), "invalid_checkpoint", CONFIG),
    ("run-config-missing-a-defaulted-field", "base", _drop_inside("run_config", "model_version"), "invalid_checkpoint", CONFIG),
    ("run-config-unknown-field", "base", _inside("run_config", extra=1), "invalid_checkpoint", CONFIG),
    ("run-config-blank-field", "base", _inside("run_config", profile=" "), "invalid_checkpoint", CONFIG),
    ("run-config-skills-as-text", "base", _inside("run_config", skill_versions="a"), "invalid_checkpoint", CONFIG),
    ("run-config-duplicate-skills", "base", _inside("run_config", skill_versions=["a", "a"]), "invalid_checkpoint", CONFIG),
    ("run-config-other-profile", "base", _inside("run_config", profile="other-profile"), "resume_profile_mismatch", "checkpoint profile differs from the configured agent"),
    ("bindings-not-a-list", "rich", _set(metric_bindings={}), "invalid_checkpoint", BINDINGS),
    ("binding-not-a-mapping", "rich", _set(metric_bindings=["x"]), "invalid_checkpoint", BINDINGS),
    ("binding-invalid", "rich", _inside_first("metric_bindings", unit=""), "invalid_checkpoint", BINDINGS),
    # Every item must be an object: key/value pairs that dict() would accept are refused too.
    ("binding-as-key-value-pairs", "rich", lambda c: {**c, "metric_bindings": [[[k, v] for k, v in c["metric_bindings"][0].items()]]}, "invalid_checkpoint", BINDINGS),
    ("binding-unknown-field", "rich", _inside_first("metric_bindings", extra=1), "invalid_checkpoint", BINDINGS),
    ("plan-not-a-mapping", "rich", _set(parallel_plan=[]), "invalid_checkpoint", PLAN),
    ("plan-missing-key", "rich", _drop_inside("parallel_plan", "metric_ids"), "invalid_checkpoint", PLAN),
    ("plan-invalid-metric", "rich", _inside("parallel_plan", metric_ids=["nope", "gross_fen"]), "invalid_checkpoint", PLAN),
    ("plan-wrong-hash", "rich", _inside("parallel_plan", plan_hash="0" * 64), "invalid_checkpoint", "checkpoint parallel plan hash is invalid"),
    ("active-time-negative", "base", _set(active_elapsed_before=-1), "invalid_checkpoint", "checkpoint active time is invalid"),
    ("active-time-text", "base", _set(active_elapsed_before="1"), "invalid_checkpoint", "checkpoint active time is invalid"),
    ("active-time-bool", "base", _set(active_elapsed_before=True), "invalid_checkpoint", "checkpoint active time is invalid"),
    ("model-calls-negative", "base", _set(model_call_count=-1), "invalid_checkpoint", "checkpoint model_call_count is invalid"),
    ("tool-calls-float", "base", _set(tool_call_count=1.0), "invalid_checkpoint", "checkpoint tool_call_count is invalid"),
    ("repairs-bool", "base", _set(repair_count=True), "invalid_checkpoint", "checkpoint repair_count is invalid"),
    ("call-ids-not-a-list", "base", _set(model_call_ids="a"), "invalid_checkpoint", "checkpoint model call identities are invalid"),
    ("call-id-empty", "base", _set(model_call_ids=[""]), "invalid_checkpoint", "checkpoint model call identities are invalid"),
    ("events-not-a-list", "base", _set(events={}), "invalid_checkpoint", "checkpoint events are invalid"),
    ("event-not-a-mapping", "base", _set(events=["x"]), "invalid_checkpoint", "checkpoint events are invalid"),
    ("retrieval-items-not-a-list", "base", _set(retrieval_items=()), "invalid_checkpoint", "checkpoint retrieval_items are invalid"),
    ("tool-result-not-a-mapping", "base", _set(tool_results=[1]), "invalid_checkpoint", "checkpoint tool_results are invalid"),
    ("clarifications-not-a-list", "base", _set(clarifications="a"), "invalid_checkpoint", "checkpoint clarifications are invalid"),
    ("clarification-blank", "base", _set(clarifications=[" "]), "invalid_checkpoint", "checkpoint clarifications are invalid"),
    ("context-version-empty", "base", _set(context_version=""), "invalid_checkpoint", "checkpoint context version is invalid"),
    ("context-version-not-text", "base", _set(context_version=16), "invalid_checkpoint", "checkpoint context version is invalid"),
]


@pytest.fixture(scope="module")
def checkpoints():
    agent, context, tools, base = scenarios._waiting_agent("2026年9月订单数是多少", [clarification_cases._query(clarification_cases.COUNT_SQL, ("paid_count",))])
    rich_context = clarification_cases._context("run-resume")
    rich_agent = scenarios.agent_for(clarification_cases._Scripted([clarification_cases._ask("请问是哪一个月？")]), tools, parallel_scheduler=scenarios._parallel_scheduler())
    rich_agent.run(
        rich_context,
        "2026年9月支付金额",
        metric_bindings=scenarios.ask_review._bound("gross_fen"),
        parallel_plan=scenarios._parallel_plan(rich_context),
        request_time_window=scenarios.SEPTEMBER,
    )
    return {"base": base, "rich": rich_agent.export_waiting_checkpoint(rich_context.run_id), "context": context, "tools": tools}


def _resume(checkpoints, which, damaged):
    agent = scenarios.agent_for(clarification_cases._Scripted([]), checkpoints["tools"], parallel_scheduler=scenarios._parallel_scheduler())
    return agent.resume_from_checkpoint(checkpoints["context"], "回答", damaged)


def test_the_undamaged_checkpoints_restore(checkpoints) -> None:
    assert checkpoints["base"]["checkpoint_version"] == AGENT_CHECKPOINT_VERSION
    assert checkpoints["base"]["waiting_clarification_id"] == "clarify.order_status_scope"
    assert checkpoints["rich"]["metric_bindings"] and checkpoints["rich"]["parallel_plan"]


@pytest.mark.parametrize(("name", "which", "damage", "code", "message"), REFUSALS, ids=[row[0] for row in REFUSALS])
def test_a_damaged_checkpoint_is_refused_with_its_own_code(checkpoints, name, which, damage, code, message) -> None:
    damaged = damage(copy.deepcopy(checkpoints[which]))
    with pytest.raises(RunResumeError) as caught:
        _resume(checkpoints, which, damaged)
    assert (caught.value.code, str(caught.value)) == (code, f"{code}: {message}")


def test_the_rich_checkpoint_restores_its_bindings_window_and_plan(checkpoints) -> None:
    rich = checkpoints["rich"]
    context = clarification_cases._context("run-resume")
    agent = scenarios.agent_for(clarification_cases._Scripted([clarification_cases._query(), clarification_cases._cite]), checkpoints["tools"], parallel_scheduler=scenarios._parallel_scheduler())
    assert agent.resume_from_checkpoint(context, "2026年9月", rich).status == "succeeded"
    assert rich["request_time_window"] == {**scenarios.SEPTEMBER, "timezone": "UTC"}


def test_a_restored_parallel_plan_is_the_one_the_parallel_action_runs() -> None:
    tools = scenarios.fixture_tools()[0]
    context = clarification_cases._context("run-plan")
    plan = scenarios._parallel_plan(context)
    first = scenarios.agent_for(clarification_cases._Scripted([clarification_cases._ask("请问是哪一个月？")]), tools)
    first.run(context, "2026年9月支付金额和已支付订单数", parallel_plan=plan, request_time_window=scenarios.SEPTEMBER)
    checkpoint = first.export_waiting_checkpoint(context.run_id)
    assert checkpoint["parallel_plan"]["plan_hash"] == plan.plan_hash

    ran: list[str] = []

    def runner(context, metric_id, branch_id) -> BranchExecution:
        ran.append(metric_id)
        return BranchExecution(metric_id, f"result-{metric_id}", ({metric_id: 1},), "2026-09-21T00:00:00Z")

    steps = [scenarios.PARALLEL, scenarios.PLAIN_ANSWER]
    resumed = scenarios.agent_for(clarification_cases._Scripted(steps), tools, parallel_scheduler=ParallelScheduler(runner))
    resumed._compiled_graph = spy = _Spy(resumed._compiled_graph)
    result = resumed.resume_from_checkpoint(context, "2026年9月", checkpoint)
    restored = spy.states[0]["parallel_plan"]
    assert isinstance(restored, ParallelPlan) and restored.plan_hash == plan.plan_hash
    groups = [event for event in result.events if event["kind"] == "parallel_group"]
    assert [group["status"] for group in groups] == ["SUCCEEDED"]
    assert sorted(ran) == ["gross_fen", "paid_count"] and result.status == "succeeded"


def test_waiting_again_keeps_the_catalog_rule_the_question_belongs_to() -> None:
    agent, context, tools, checkpoint = scenarios._waiting_agent(
        "2026年9月订单数是多少", [clarification_cases._query(clarification_cases.COUNT_SQL, ("paid_count",))]
    )
    assert checkpoint["waiting_clarification_id"] == "clarify.order_status_scope"
    second = scenarios.agent_for(clarification_cases._Scripted([]), tools)
    second.continue_waiting_for_clarification(context, "还没想好", checkpoint)
    again = second.export_waiting_checkpoint(context.run_id)
    assert again["waiting_clarification_id"] == "clarify.order_status_scope"
    assert again["waiting_question"] == checkpoint["waiting_question"]


# --- the answer and the context ----------------------------------------------------


ANSWER_MESSAGES = {
    "invalid_resume_input: answer must be a non-empty string",
    "invalid_resume_input: answer exceeds 8000 characters",
}


@pytest.mark.parametrize("answer", ["", "   ", None, 5, "x" * 8001])
def test_an_unusable_answer_is_refused_before_the_checkpoint_is_read(checkpoints, answer) -> None:
    agent = scenarios.agent_for(clarification_cases._Scripted([]), checkpoints["tools"])
    # The checkpoint is garbage on purpose: the answer is checked first.
    for call in (
        lambda: agent.resume_from_checkpoint(checkpoints["context"], answer, []),
        lambda: agent.continue_waiting_for_clarification(checkpoints["context"], answer, []),
        lambda: agent.fail_unsupported_clarification(checkpoints["context"], answer, [], note="n"),
    ):
        with pytest.raises(RunResumeError) as caught:
            call()
        assert caught.value.code == "invalid_resume_input" and str(caught.value) in ANSWER_MESSAGES


def test_an_answer_of_exactly_8000_characters_is_accepted(checkpoints) -> None:
    agent = scenarios.agent_for(clarification_cases._Scripted([]), checkpoints["tools"])
    result = agent.continue_waiting_for_clarification(checkpoints["context"], "x" * 8000, checkpoints["base"])
    assert result.status == "waiting_user"


def test_resume_needs_a_server_context(checkpoints) -> None:
    agent = scenarios.agent_for(clarification_cases._Scripted([]), checkpoints["tools"])
    with pytest.raises(TypeError, match="context must be an ExecutionContext"):
        agent.resume_from_checkpoint({"run_id": "x"}, "回答", checkpoints["base"])
    with pytest.raises(TypeError, match="context must be an ExecutionContext"):
        agent.run({"run_id": "x"}, "问题")


def test_a_checkpoint_without_a_waiting_question_cannot_wait_again(checkpoints) -> None:
    agent = scenarios.agent_for(clarification_cases._Scripted([]), checkpoints["tools"])
    blank = {**copy.deepcopy(checkpoints["base"]), "waiting_question": ""}
    with pytest.raises(RunResumeError) as caught:
        agent.continue_waiting_for_clarification(checkpoints["context"], "回答", blank)
    assert (caught.value.code, str(caught.value)) == ("invalid_checkpoint", "invalid_checkpoint: waiting question is missing")


def test_in_memory_resume_checks_state_then_identity_then_answer() -> None:
    tools = scenarios.fixture_tools()[0]
    context = clarification_cases._context("run-memory")
    agent = scenarios.agent_for(clarification_cases._Scripted([clarification_cases._ask("请问是哪一个月？")]), tools)
    with pytest.raises(RunResumeError) as nothing:
        agent.resume(context, "回答")
    assert nothing.value.code == "invalid_run_state"
    with pytest.raises(RunResumeError) as no_checkpoint:
        agent.export_waiting_checkpoint(context.run_id)
    assert no_checkpoint.value.code == "invalid_run_state"
    assert agent.run(context, "支付金额是多少").status == "waiting_user"
    with pytest.raises(RunResumeError) as second_run:
        agent.run(context, "支付金额是多少")
    assert second_run.value.code == "run_waiting_user"
    stranger = ExecutionContext(run_id=context.run_id, tenant_id="A", principal_id="someone-else", role="requester")
    with pytest.raises(RunResumeError) as mismatch:
        agent.resume(stranger, "")
    assert mismatch.value.code == "resume_context_mismatch"
    with pytest.raises(RunResumeError) as blank:
        agent.resume(context, " ")
    assert blank.value.code == "invalid_resume_input"


# --- writing a checkpoint -----------------------------------------------------------


def _waiting_state():
    agent, context, tools, checkpoint = scenarios._waiting_agent()
    return agent._waiting_checkpoints[context.run_id], checkpoint


def test_only_a_waiting_state_with_an_ask_is_serialized() -> None:
    state, checkpoint = _waiting_state()
    assert _serialize_agent_checkpoint(state) == checkpoint
    for damage, code in (
        ({"status": "running"}, "invalid_run_state"),
        ({"context": "x"}, "invalid_run_state"),
        ({"run_config": None}, "invalid_checkpoint"),
        ({"final_action": None}, "invalid_checkpoint"),
        ({"final_action": {"type": "deny"}}, "invalid_checkpoint"),
        ({"events": ({"kind": "x", "value": float("nan")},)}, "invalid_checkpoint"),
        ({"tool_results": ({"output": object()},)}, "invalid_checkpoint"),
    ):
        with pytest.raises(RunResumeError) as caught:
            _serialize_agent_checkpoint({**state, **damage})
        assert caught.value.code == code, damage


def test_a_checkpoint_round_trips_to_the_same_bytes() -> None:
    agent, context, tools, checkpoint = scenarios._waiting_agent()
    second = scenarios.agent_for(clarification_cases._Scripted([]), tools)
    second.continue_waiting_for_clarification(context, "还没想好", checkpoint)
    again = second.export_waiting_checkpoint(context.run_id)
    expected = {**checkpoint, "clarifications": checkpoint["clarifications"] + ["还没想好"], "active_elapsed_before": again["active_elapsed_before"]}
    assert json.dumps(again, ensure_ascii=False) == json.dumps(expected, ensure_ascii=False)


# --- the state a graph run starts from ----------------------------------------------

STATE_KEYS = [
    "active_elapsed_before", "answer_bounce_count", "clarification_bounce_count", "clarifications", "context",
    "context_version", "elapsed_ms", "events", "facts", "final_action", "metric_bindings", "model_call_count",
    "model_call_ids", "parallel_plan", "proposal", "public_answer", "question", "reason", "repair_count",
    "request_time_window", "retrieval_items", "run_config", "started_at", "status", "tool_call_count",
    "tool_results", "error_code",
]


class _Spy:
    """Wraps the compiled graph and keeps the state each invocation starts from."""

    def __init__(self, graph) -> None:
        self.graph = graph
        self.states: list[dict] = []

    def invoke(self, state):
        self.states.append(dict(state))
        return self.graph.invoke(state)


def test_a_new_run_and_a_restored_run_start_from_the_same_state_keys() -> None:
    tools = scenarios.fixture_tools()[0]
    context = clarification_cases._context("run-keys")
    fresh = scenarios.agent_for(clarification_cases._Scripted([clarification_cases._ask("请问是哪一个月？")]), tools)
    fresh._compiled_graph = spy = _Spy(fresh._compiled_graph)
    fresh.run(context, "支付金额是多少")
    checkpoint = fresh.export_waiting_checkpoint(context.run_id)
    assert sorted(spy.states[0]) == sorted(STATE_KEYS)
    assert spy.states[0]["started_at"] > 0 and spy.states[0]["retrieval_items"] == ()

    resumed = scenarios.agent_for(clarification_cases._Scripted([clarification_cases._ask("请问是哪一个月？")]), tools)
    resumed._compiled_graph = spy = _Spy(resumed._compiled_graph)
    resumed.resume_from_checkpoint(context, "不知道", checkpoint)
    assert sorted(spy.states[0]) == sorted(STATE_KEYS)
    assert spy.states[0]["clarifications"] == ("不知道",)
    assert spy.states[0]["final_action"] is None and spy.states[0]["status"] == "running"


def test_a_prepared_v3_checkpoint_restores_with_an_unspent_answer_bounce() -> None:
    context = clarification_cases._context("run-prepared-keys")
    checkpoint = BoundedAgent.prepared_waiting_user_checkpoint(
        context, "支付金额是多少", run_config=scenarios.DEFAULT_RUN_CONFIG, waiting_question="请问是哪一个月？"
    )
    assert checkpoint["checkpoint_version"] == "qs-bounded-agent-checkpoint-v3" and "answer_bounce_count" not in checkpoint
    agent = scenarios.agent_for(clarification_cases._Scripted([clarification_cases._ask("请问是哪一个月？")]), scenarios.fixture_tools()[0])
    agent._compiled_graph = spy = _Spy(agent._compiled_graph)
    agent.resume_from_checkpoint(context, "不知道", checkpoint)
    assert sorted(spy.states[0]) == sorted(STATE_KEYS)
    assert spy.states[0]["answer_bounce_count"] == 0 and spy.states[0]["clarification_bounce_count"] == 0
