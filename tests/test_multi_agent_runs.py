"""B2 runs through the product service: delegation, the merged answer, failures, concurrency and every exit.

Service-level: the fixture executor, a temporary state store and a model scripted per agent
(tests/multi_agent_support.py).  Sub-agents without a script are answered by the FakeModel.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from queryshield.agent.runtime import http_status_for_run
from queryshield.agent.runtime import RuntimeConfigurationError
from queryshield.providers.contracts import ModelProviderError
from queryshield.providers.fake_model import FakeModel

from multi_agent_support import (
    AUG,
    COMPOSITE,
    JUL,
    SEP,
    RoleModel,
    b2,  # noqa: F401  (fixture)
    cite_all,
    cite_first,
    delegate,
    model_call_events,
    run_b2,
    steps,
    usage_sum,
)
from test_agent_step_writes import _Gate, _wait_for
from test_approval_api_pins import NAME_QUERY, _approve
from test_clarification import REQUESTER, _Recording, service  # noqa: F401  (fixture)
from test_runs_left_executing_at_startup import env as startup_env  # noqa: F401  (fixture)

TWO = delegate((["gross_fen"], SEP), (["net_fen"], SEP))
DESCRIBE = {"type": "tool_call", "name": "describe_tables", "arguments": {"tables": ["orders"]}}
ASK = {"type": "ask_user", "question": "请说明时间范围"}
DENY = {"type": "deny", "reason": "不查"}
FORBIDDEN_TABLE = {"type": "tool_call", "name": "query_readonly", "arguments": {"sql": "SELECT * FROM pg_user", "params": {}}}
NO_DATA = {"type": "final_answer", "answer": "", "source_ids": [], "fact_refs": [], "basis": "no_data"}


def _fake_action(messages) -> dict:
    return json.loads(FakeModel().complete(messages).content)


def _raises(code):
    def step(messages):
        raise ModelProviderError(code, {"status": "failed", "usage_status": "unknown"})

    return step


def _outcome(run) -> tuple:
    return run["status"], run.get("error_code"), http_status_for_run(run["status"], run.get("error_code"))


# --- the merged answer ----------------------------------------------------------------------


def test_a_delegated_question_is_answered_from_every_subtask_and_each_step_is_stored_once(b2) -> None:
    svc, executor = b2
    model = RoleModel([TWO, cite_all])
    run = run_b2(b2, COMPOSITE, model)

    assert run["status"] == "SUCCEEDED" and run["checkpoint"]["answer_status"] == "verified"
    facts = [(f["metric_id"], f["time_window"]["start"]) for f in run["facts"]["facts"]]
    assert facts == [("gross_fen", SEP["start"]), ("net_fen", SEP["start"])]
    stored = steps(svc, run["run_id"])
    # The coordinator's events, the delegation before any sub-agent step, one end event per subtask, then the answer.
    kinds = [(p.get("agent"), p["kind"]) for p in stored]
    assert kinds[:2] == [("coordinator", "model_call"), ("coordinator", "delegation")]
    assert sorted(kinds[2:6]) == sorted([("subtask-1", "model_call"), ("subtask-1", "tool_call"), ("subtask-2", "model_call"), ("subtask-2", "tool_call")])
    assert kinds[6:] == [("coordinator", "subtask"), ("coordinator", "subtask"), ("coordinator", "model_call"), ("coordinator", "answer")]
    assert [p["subtask_index"] for p in stored if p["kind"] == "subtask"] == [1, 2]
    # Each agent's own sequence is complete and written once.
    for agent in ("coordinator", "subtask-1", "subtask-2"):
        numbers = [p["sequence"] for p in stored if p.get("agent") == agent]
        assert len(numbers) == len(set(numbers))
    events = svc.store.events(run["run_id"], after_event_id=0, limit=1000)
    assert events[-1]["type"] == "terminal" and all(e["type"] != "agent_step" for e in events[events.index(events[-1]):][1:])
    # Counts and usage are those of every agent's calls.
    calls = model_call_events(svc, run["run_id"])
    assert run["model_call_count"] == len(calls) == len(model.calls) == 4
    assert run["tool_call_count"] == 2 and run["sql_exec_count"] == 3  # the net plan runs two queries
    assert run["usage"]["status"] == "known"
    assert {k: run["usage"][k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")} == usage_sum(calls)
    assert sorted(run["checkpoint"]["model_call_ids"]) == sorted(p["model_call_id"] for p in calls)


def test_the_delegation_event_records_each_subtask_and_its_share(b2) -> None:
    svc, _ = b2
    run = run_b2(b2, COMPOSITE, RoleModel([TWO, cite_all]))
    delegation = next(p for p in steps(svc, run["run_id"]) if p["kind"] == "delegation")
    assert [(s["subtask_index"], s["metrics"], s["time_window"]["start"]) for s in delegation["subtasks"]] == [
        (1, ["gross_fen"], SEP["start"]),
        (2, ["net_fen"], SEP["start"]),
    ]
    # 6 - 1 used - 1 for the answer, split in two; 8 tool calls split in two.
    assert {(s["max_model_calls"], s["max_tool_calls"]) for s in delegation["subtasks"]} == {(2, 4)}
    assert 0 < delegation["max_wall_clock_seconds"] <= 60


def test_a_sub_agent_sees_only_its_own_subtask_and_stops_at_its_result(b2) -> None:
    model = RoleModel([TWO, cite_all])
    run_b2(b2, COMPOSITE, model)
    subtask_messages = model.messages_of("fake")
    assert len(subtask_messages) == 2  # one model call each: the result ends the sub-agent
    for messages in subtask_messages:
        text = "\n".join(m["content"] for m in messages)
        assert COMPOSITE not in text and "分别" not in text
        assert '"search_catalog"' not in messages[0]["content"] and '"delegate"' not in messages[0]["content"]
        assert '"source_id"' not in text  # no retrieval items
    questions = sorted(messages[1]["content"] for messages in subtask_messages)
    assert "gross_fen" in questions[0] or "gross_fen" in questions[1]
    assert not any("gross_fen" in q and "net_fen" in q for q in questions)


def test_an_answer_that_leaves_out_a_subtask_result_is_refused(b2) -> None:
    run = run_b2(b2, COMPOSITE, RoleModel([TWO, cite_first]))
    assert _outcome(run) == ("FAILED", "evidence_validation_failed", 502)
    assert run["facts"] is None and run["answer"] is None


# --- subtask failures -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "steps_of_two, expected",
    [
        ([_raises("upstream_timeout")], ("FAILED", "upstream_timeout", 504)),
        ([_raises("model_rate_limited")], ("FAILED", "model_rate_limited", 503)),
        ([ASK], ("FAILED", "subtask_incomplete", 502)),
        ([DENY], ("FAILED", "subtask_incomplete", 502)),
        ([NO_DATA], ("FAILED", "subtask_incomplete", 502)),
        ([FORBIDDEN_TABLE], ("DENIED", "table_not_allowed", 403)),
        ([DESCRIBE, DESCRIBE], ("LIMIT_REACHED", "model_call_limit", 502)),
    ],
    ids=["timeout", "rate-limited", "asks", "refuses", "answers-itself", "server-refusal", "budget"],
)
def test_a_failed_subtask_ends_the_run_without_an_answer(b2, steps_of_two, expected) -> None:
    svc, _ = b2
    model = RoleModel([TWO, cite_all], {"net_fen": steps_of_two})
    run = run_b2(b2, COMPOSITE, model)
    assert _outcome(run) == expected
    assert run["answer"] is None and run["facts"] is None
    # No further coordinator call; every call is counted and in the stored events.
    assert len(model.messages_of("coordinator")) == 1
    calls = model_call_events(svc, run["run_id"])
    assert run["model_call_count"] == len(calls) == len(model.calls)
    ends = [(p["subtask_index"], p["status"]) for p in steps(svc, run["run_id"]) if p["kind"] == "subtask"]
    assert ends == [(1, "completed"), (2, expected[0].lower())]


@pytest.mark.parametrize(
    "first, second, expected",
    [
        # A server refusal comes first, then a budget stop, then any other failure.
        ([_raises("upstream_timeout")], [FORBIDDEN_TABLE], ("DENIED", "table_not_allowed")),
        ([DESCRIBE, DESCRIBE], [FORBIDDEN_TABLE], ("DENIED", "table_not_allowed")),
        ([_raises("upstream_timeout")], [DESCRIBE, DESCRIBE], ("LIMIT_REACHED", "model_call_limit")),
        # Within one kind, the lowest subtask number.
        ([_raises("upstream_timeout")], [_raises("model_rate_limited")], ("FAILED", "upstream_timeout")),
        ([ASK], [_raises("model_rate_limited")], ("FAILED", "subtask_incomplete")),
    ],
    ids=["refusal-over-failure", "refusal-over-budget", "budget-over-failure", "lowest-number", "incomplete-first"],
)
def test_with_several_failed_subtasks_the_run_takes_one_by_a_fixed_rule(b2, first, second, expected) -> None:
    run = run_b2(b2, COMPOSITE, RoleModel([TWO, cite_all], {"gross_fen": first, "net_fen": second}))
    assert _outcome(run)[:2] == expected


# --- concurrency and every exit ----------------------------------------------------------------


def test_sub_agents_run_side_by_side_and_merge_in_subtask_order(b2) -> None:
    barrier = threading.Barrier(2, timeout=5)

    def gross(messages):
        barrier.wait()  # only passes when the other sub-agent is running at the same time
        time.sleep(0.3)  # and finishes after it
        return _fake_action(messages)

    def net(messages):
        barrier.wait()
        return _fake_action(messages)

    svc, _ = b2
    model = RoleModel([TWO, cite_all], {"gross_fen": [gross], "net_fen": [net]})
    run = run_b2(b2, COMPOSITE, model)
    assert run["status"] == "SUCCEEDED"
    stored = steps(svc, run["run_id"])
    tool_steps = [p["agent"] for p in stored if p["kind"] == "tool_call"]
    assert tool_steps == ["subtask-2", "subtask-1"]  # net finished first
    answer_messages = model.messages_of("coordinator")[-1]
    from multi_agent_support import results

    assert [o["verified_metrics"][0]["metric_id"] for o in results(answer_messages)] == ["gross_fen", "net_fen"]
    assert [p["subtask_index"] for p in stored if p["kind"] == "subtask"] == [1, 2]


def test_a_cancel_during_delegation_waits_for_the_sub_agents_and_counts_every_call(b2) -> None:
    svc, _ = b2
    gate = _Gate(_fake_action)
    model = RoleModel([TWO, cite_all], {"gross_fen": [gate]})
    deps = svc.default_dependencies()
    deps.model, deps.retriever = model, None
    run_id = svc.start_async(identity=REQUESTER, question=COMPOSITE, time_window=None, deps=deps)["run_id"]
    assert gate.entered.wait(timeout=10)
    svc.cancel(run_id=run_id, identity=REQUESTER)
    gate.release.set()
    run = _wait_for(lambda: (r := svc.store.get_run(run_id)) and r["status"] == "CANCELLED" and r)
    calls = model_call_events(svc, run_id)
    assert run["model_call_count"] == len(calls) == len(model.calls) == 4  # the coordinator answered before the commit
    assert run["usage"]["status"] == "known"
    assert svc.store.events(run_id, after_event_id=0, limit=1000)[-1]["type"] == "terminal"


def test_an_exception_in_one_sub_agent_waits_for_the_others_and_the_terminal_event_is_last(b2) -> None:
    svc, _ = b2

    def broken(messages):
        raise RuntimeError("a bug in one sub-agent")

    def slow(messages):
        time.sleep(0.5)
        return _fake_action(messages)

    model = RoleModel([TWO, cite_all], {"gross_fen": [broken], "net_fen": [slow]})
    run = run_b2(b2, COMPOSITE, model)
    time.sleep(0.8)  # anything a sub-agent still did would be stored by now
    events = svc.store.events(run["run_id"], after_event_id=0, limit=1000)
    assert run["status"] == "FAILED" and events[-1]["type"] == "terminal"
    assert [p["agent"] for p in steps(svc, run["run_id"]) if p["kind"] == "tool_call"] == ["subtask-2"]
    # The broken sub-agent's call started without a stored event: the usage is unknown.
    stored_calls = model_call_events(svc, run["run_id"])
    assert len(model.calls) == 3 and len(stored_calls) == 2
    assert run["model_call_count"] == 2 and run["usage"]["status"] == "unknown"


class _BrokenRefunds(_Recording):
    """The refund query of the net plan raises a programming error (not a database error)."""

    def execute(self, sql, **kwargs):
        if "refunds" in sql:
            raise RuntimeError("a bug in the executor")
        return super().execute(sql, **kwargs)


def test_an_exception_after_every_call_was_stored_keeps_the_usage_of_all_agents(b2, monkeypatch) -> None:
    svc, _ = b2
    broken = _BrokenRefunds()
    monkeypatch.setattr(svc, "_executor_factory", lambda: broken)
    model = RoleModel([TWO, cite_all])
    run = run_b2(b2, COMPOSITE, model)
    calls = model_call_events(svc, run["run_id"])
    assert run["status"] == "FAILED" and len(model.calls) == len(calls) == 3
    assert run["model_call_count"] == 3 and run["usage"]["status"] == "known"
    assert {k: run["usage"][k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")} == usage_sum(calls)


# --- after delegating, configuration, approval --------------------------------------------------


@pytest.mark.parametrize("after", [DESCRIBE, TWO, ASK], ids=["tool", "second-delegate", "ask"])
def test_after_delegating_only_an_answer_or_a_refusal_is_valid(b2, after) -> None:
    model = RoleModel([TWO, after])
    run = run_b2(b2, COMPOSITE, model)
    assert _outcome(run) == ("FAILED", "invalid_action_after_delegation", 502)
    assert len(model.messages_of("coordinator")) == 2


def test_after_delegating_a_refusal_is_valid(b2) -> None:
    run = run_b2(b2, COMPOSITE, RoleModel([TWO, DENY]))
    assert run["status"] == "DENIED"


def test_the_coordinators_own_sensitive_query_waits_for_approval(b2) -> None:
    run = run_b2(b2, "查询客户姓名", RoleModel([NAME_QUERY]))
    assert run["status"] == "WAITING_APPROVAL"
    done = _approve(b2, run)
    assert done["status"] == "SUCCEEDED"


@pytest.mark.parametrize(
    "setting, code",
    [(("QUERYSHIELD_MODEL_PROTOCOL", "native"), "invalid_model_protocol"), (("QUERYSHIELD_METADATA_TOOLS", "mcp"), "invalid_metadata_tools_configuration")],
    ids=["native", "mcp"],
)
def test_an_unsupported_combination_is_refused_before_any_run_exists(b2, monkeypatch, setting, code) -> None:
    from queryshield.approval.service import METADATA_TOOLS_FROM_ENV

    svc, _ = b2
    monkeypatch.setenv(*setting)
    monkeypatch.setattr(svc, "_metadata_tools", METADATA_TOOLS_FROM_ENV)
    with pytest.raises(RuntimeConfigurationError) as caught:
        run_b2(b2, COMPOSITE, RoleModel([TWO, cite_all]))
    assert caught.value.code == code
    assert svc.store._connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


# --- the coordinator's own checks ---------------------------------------------------------------


def test_a_declaration_error_uses_the_one_query_repair(b2) -> None:
    bad = delegate((["gross_fen"], SEP), (["no_such_metric"], SEP))
    model = RoleModel([bad, TWO, cite_all])
    run = run_b2(b2, COMPOSITE, model)
    assert run["status"] == "SUCCEEDED"
    model = RoleModel([bad, bad])
    run = run_b2(b2, COMPOSITE, model)
    assert _outcome(run) == ("FAILED", "query_repair_limit", 502)


def test_a_wording_the_catalog_leaves_open_waits_for_the_user(b2) -> None:
    run = run_b2(b2, "2026年9月的销售额和已支付订单数分别是多少？", RoleModel([delegate((["gross_fen"], SEP), (["paid_count"], SEP))]))
    assert run["status"] == "WAITING_USER"
    assert run["checkpoint"]["agent_checkpoint"]["waiting_clarification_id"] == "clarify.metric_basis"


def test_a_budget_that_cannot_be_split_stops_the_run(b2) -> None:
    three = delegate((["paid_count"], JUL), (["paid_count"], AUG), (["paid_count"], SEP))
    run = run_b2(b2, "2026年7月、2026年8月和2026年9月的已支付订单数分别是多少？", RoleModel([DESCRIBE, DESCRIBE, DESCRIBE, three]))
    assert _outcome(run) == ("LIMIT_REACHED", "delegation_budget_insufficient", 502)


def test_clarify_then_resume_then_delegate(b2) -> None:
    svc, executor = b2
    question = "2026年9月的销售额和已支付订单数分别是多少？"
    run = run_b2(b2, question, FakeModel())
    assert run["status"] == "WAITING_USER"
    from queryshield.agent import ModelCallStore

    done = svc.resume_waiting_user(
        run_id=run["run_id"], answer="按支付金额统计", identity=REQUESTER, model=FakeModel(), call_store=ModelCallStore(), executor=executor
    )
    assert done["status"] == "SUCCEEDED"
    assert sorted(f["metric_id"] for f in done["facts"]["facts"]) == ["gross_fen", "paid_count"]
    assert any(p["kind"] == "delegation" for p in steps(svc, run["run_id"]))
    assert done["run_config"]["profile"] == "B2-multi-agent"


# --- a process exit during delegation -----------------------------------------------------------


def test_a_process_exit_during_delegation_is_ended_at_the_next_startup(startup_env, monkeypatch) -> None:
    from queryshield.approval.service import RunService
    from queryshield.db.state_store import StateStore
    from test_runs_left_executing_at_startup import REQ, _exit_here, _ProcessExit, _start_up

    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "b2")
    store = StateStore(startup_env)
    svc = RunService(store=store, executor_factory=lambda: _Recording(), mode="fake")
    deps = svc.default_dependencies()
    deps.model, deps.retriever = RoleModel([TWO, cite_all], {"gross_fen": [_exit_here]}), None
    with pytest.raises(_ProcessExit):
        svc.run_sync(identity=REQ, question=COMPOSITE, time_window=None, deps=deps)
    (run_id,) = [row[0] for row in store._connection.execute("SELECT run_id FROM runs")]
    assert store.get_run(run_id)["status"] == "RUNNING"
    store.close()

    _start_up()
    reader = StateStore(startup_env)
    try:
        run = reader.get_run(run_id)
        events = reader.events(run_id, after_event_id=0, limit=1000)
        assert (run["status"], run["error_code"]) == ("FAILED", "execution_interrupted")
        assert events[-1]["type"] == "terminal" and run["usage"]["status"] == "unknown"
        # The other sub-agent finished before the exit reached the run; its steps are kept.
        assert any(e["payload"].get("agent") == "subtask-2" and e["payload"].get("kind") == "tool_call" for e in events)
    finally:
        reader.close()
