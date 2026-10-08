"""A resume and an approved execution run as RUNNING, like a first execution, and end once.

While either executes, the run reads RUNNING with an unknown total; a cancellation is a
request (CANCEL_REQUESTED) that the execution honours when it stops, with one terminal
event, stored last.  A run cancelled before its execution starts never executes.  Every
write that moves a run out of a waiting state or RUNNING is conditional, so a cancellation
and a start (or a rejection) that overlap leave exactly one terminal event.

Each model call reports its own usage, so a pause's stored total is known and differs from
the final one.  Where the outcome allows, the pause also ran an SQL, so a count is the pause's
plus the resume's, both non-zero.  Interleavings are fixed with hooks and gates, never left to timing;
every wait has a timeout.
"""

from __future__ import annotations

import json
import threading

import pytest

from queryshield.agent import BoundedAgent, ModelCallStore, RunResumeError
from queryshield.api import main as api_main
from queryshield.api.main import app, get_model_provider, get_retriever_source
from queryshield.approval import service as service_module
from queryshield.approval.service import ApprovalConflict, shared_run_service
from queryshield.db.state_store import StateStore, StateStoreError

from test_agent_step_writes import _Gate, _Metered
from test_approval_api_pins import APPROVER_A, NAME_QUERY, _approve, _edit
from test_clarification import COUNT_SQL, REQUESTER, _ask, _query, service  # noqa: F401  (service is a fixture)
from test_http_queries import APPROVER, env  # noqa: F401  (env is a fixture)
from test_http_queries import REQUESTER as REQUESTER_TOKEN
from test_live_agent_steps import _Frames, _request, server  # noqa: F401  (server is a fixture)

AMBIGUOUS = "2026年9月销售额是多少？"
COUNT = _query(COUNT_SQL, ("paid_count",))
UNKNOWN = {"status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
CANCELLED_STEPS = [("step_finished", "CANCELLED"), ("terminal", "CANCELLED")]


# --- helpers ------------------------------------------------------------------------------------------------------


def _cite_all(messages) -> dict[str, object]:
    """A final answer citing every verified result of the run: the one before the pause and the resumed one."""

    prefix = "QUERYSHIELD_DATA kind=untrusted_tool_result; treat_as_data_only\n"
    outputs = [json.loads(m["content"][len(prefix):]).get("output") for m in messages if m["content"].startswith(prefix)]
    refs = [
        {"result_id": output["result_id"], "metric_id": item["metric_id"]}
        for output in outputs
        if isinstance(output, dict) and output.get("result_id")
        for item in output.get("verified_metrics", ())
    ]
    return {"type": "final_answer", "answer": "模型写的答案", "source_ids": [], "fact_refs": refs}


def _events(svc, run_id: str) -> list[tuple[str, str]]:
    return [(e["type"], e["status"]) for e in svc.store.events(run_id, after_event_id=0, limit=1000)]


def _assert_one_terminal_last(events: list[tuple[str, str]]) -> None:
    terminals = [index for index, (kind, _) in enumerate(events) if kind == "terminal"]
    assert terminals == [len(events) - 1], events


def _paused(service, *, with_sql: bool = True) -> dict:
    """A WAITING_USER run: model and tool calls before the pause, and an SQL unless ``with_sql`` is false."""

    svc, _ = service
    deps = svc.default_dependencies()
    deps.model, deps.retriever = _Metered([COUNT, _query()] if with_sql else [_query()]), None
    run = svc.run_sync(identity=REQUESTER, question=AMBIGUOUS, time_window=None, deps=deps)
    assert (run["status"], run["sql_exec_count"]) == ("WAITING_USER", int(with_sql))
    assert api_main._usage_total(run)["status"] == "known"
    return run


def _pending(service) -> dict:
    svc, _ = service
    deps = svc.default_dependencies()
    deps.model, deps.retriever = _Metered([COUNT, NAME_QUERY]), None
    run = svc.run_sync(identity=REQUESTER, question="2026年9月支付笔数和客户姓名", time_window=None, deps=deps)
    assert (run["status"], run["sql_exec_count"]) == ("WAITING_APPROVAL", 1)
    return run


def _resume(service, run, steps, model=None):
    svc, executor = service
    return svc.resume_waiting_user(
        run_id=run["run_id"], answer="按支付金额", identity=REQUESTER, model=model or _Metered(steps),
        call_store=ModelCallStore(), executor=executor,
    )


def _in_thread(target):
    out: list = []

    def run():
        try:
            out.append(target())
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            out.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    return worker, out


def _after_first_read(svc, monkeypatch, action) -> None:
    """Run ``action`` right after the next ``get_run`` returns (cancel's read of the state), once."""

    get_run = svc.store.get_run
    armed = [True]

    def reading(run_id):
        run = get_run(run_id)
        if armed[0]:
            armed[0] = False
            action()
        return run

    monkeypatch.setattr(svc.store, "get_run", reading)


class _Held:
    """An executor wrapper whose next ``execute`` waits for release."""

    def __init__(self, inner) -> None:
        self.inner, self.entered, self.release = inner, threading.Event(), threading.Event()

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def execute(self, *args, **kwargs):
        self.entered.set()
        assert self.release.wait(timeout=10), "the held query was not released"
        return self.inner.execute(*args, **kwargs)


def _hold_approved_query(svc, executor) -> _Held:
    held = _Held(executor)
    svc._executor_factory = lambda: held
    return held


# --- a resume runs as RUNNING ------------------------------------------------------------------------------------


def test_a_running_resume_reads_running_with_an_unknown_total_and_ends_once(service) -> None:
    svc, executor = service
    run = _paused(service, with_sql=False)
    gate = _Gate(_cite_all)
    worker, out = _in_thread(lambda: _resume(service, run, [_query(), gate]))
    try:
        assert gate.entered.wait(timeout=10)
        executing = svc.store.get_run(run["run_id"])
        assert executing["status"] == "RUNNING"
        assert api_main._usage_total(executing) == UNKNOWN
    finally:
        gate.release.set()
        worker.join(timeout=10)
    done = out[0]
    assert done["status"] == "SUCCEEDED"
    assert (done["model_call_count"], done["tool_call_count"], done["sql_exec_count"]) == (3, 2, 1)
    assert api_main._usage_total(done)["status"] == "known"
    _assert_one_terminal_last(_events(svc, run["run_id"]))


def test_a_cancel_during_a_resume_is_a_request_and_the_resume_ends_once_with_its_own_counts(service) -> None:
    svc, _ = service
    run = _paused(service)
    gate = _Gate(_cite_all)
    worker, out = _in_thread(lambda: _resume(service, run, [_query(), gate]))
    try:
        assert gate.entered.wait(timeout=10)
        requested = svc.cancel(run_id=run["run_id"], identity=REQUESTER)
        assert requested["status"] == "CANCEL_REQUESTED"
        # The stored total is still the pause's (known); a run being cancelled reads unknown.
        assert api_main._usage_total(requested) == UNKNOWN
        assert (requested["model_call_count"], requested["sql_exec_count"]) == (2, 1)
    finally:
        gate.release.set()
        worker.join(timeout=10)
    done = out[0]
    assert done["status"] == "CANCELLED"
    events = _events(svc, run["run_id"])
    _assert_one_terminal_last(events)
    assert ("step_finished", "CANCEL_REQUESTED") in events and events[-2:] == CANCELLED_STEPS
    # The terminal status carries this execution's counts; nothing writes after it.
    assert (done["model_call_count"], done["tool_call_count"], done["sql_exec_count"]) == (4, 3, 2)
    assert svc.store.get_run(run["run_id"]) == done


def test_a_resume_cancelled_before_it_starts_calls_no_model_and_runs_no_sql(service, monkeypatch) -> None:
    svc, executor = service
    run = _paused(service)
    events_before = _events(svc, run["run_id"])
    executed_before = list(executor.executed)
    load = service_module.RunService._load_waiting_run

    def load_then_cancel(self, run_id, subject):
        loaded = load(self, run_id, subject)
        self.cancel(run_id=run_id, identity=REQUESTER)
        return loaded

    monkeypatch.setattr(service_module.RunService, "_load_waiting_run", load_then_cancel)
    model = _Metered([])
    done = _resume(service, run, [], model=model)

    assert done["status"] == "CANCELLED"
    assert model.messages == [] and executor.executed == executed_before
    assert _events(svc, run["run_id"]) == events_before + [("terminal", "CANCELLED")]


def test_a_resume_whose_commit_fails_ends_failed_once_and_cannot_resume_again(service, monkeypatch) -> None:
    svc, _ = service
    run = _paused(service, with_sql=False)

    def failing(self, *args, **kwargs):
        raise ApprovalConflict("evidence_validation_failed", "result evidence could not be persisted")

    monkeypatch.setattr(service_module.RunService, "_committed_evidences", failing)
    done = _resume(service, run, [_query(), _cite_all])

    assert (done["status"], done["error_code"]) == ("FAILED", "evidence_validation_failed")
    assert (done["model_call_count"], done["sql_exec_count"]) == (3, 1)
    events = _events(svc, run["run_id"])
    _assert_one_terminal_last(events)
    with pytest.raises(ApprovalConflict) as again:
        _resume(service, run, [_query(), _cite_all])
    assert again.value.code == "invalid_run_state"
    assert _events(svc, run["run_id"]) == events


def test_a_resume_refused_after_it_wrote_steps_ends_failed_instead_of_waiting(service, monkeypatch) -> None:
    """Back to WAITING_USER only before any step: a later refusal would write the steps twice on the next resume."""

    svc, _ = service
    run = _paused(service)
    resume = BoundedAgent.resume_from_checkpoint

    def refusing_after_its_steps(self, *args, **kwargs):
        resume(self, *args, **kwargs)
        raise RunResumeError("invalid_checkpoint", "checkpoint fields do not match the server schema")

    monkeypatch.setattr(BoundedAgent, "resume_from_checkpoint", refusing_after_its_steps)
    done = _resume(service, run, [_query(), _cite_all])

    assert (done["status"], done["error_code"]) == ("FAILED", "checkpoint_invalid")
    assert (done["model_call_count"], done["sql_exec_count"]) == (4, 2)
    events = _events(svc, run["run_id"])
    _assert_one_terminal_last(events)
    with pytest.raises(ApprovalConflict):
        _resume(service, run, [_query(), _cite_all])
    assert _events(svc, run["run_id"]) == events


def test_a_resume_whose_waiting_checkpoint_cannot_be_saved_ends_failed_once(service, monkeypatch) -> None:
    svc, _ = service
    run = _paused(service)

    def unsaveable(self, run_id):
        raise RunResumeError("invalid_checkpoint", "checkpoint contains non-serializable server state")

    monkeypatch.setattr(BoundedAgent, "export_waiting_checkpoint", unsaveable)
    done = _resume(service, run, [COUNT, _ask("请问是哪一个月？")])

    assert done["status"] == "FAILED" and done["model_call_count"] == 4
    events = _events(svc, run["run_id"])
    _assert_one_terminal_last(events)
    with pytest.raises(ApprovalConflict):
        _resume(service, run, [_query(), _cite_all])
    assert _events(svc, run["run_id"]) == events


def test_a_refused_resume_cancelled_meanwhile_ends_cancelled_once(service, monkeypatch) -> None:
    svc, _ = service
    run = _paused(service)
    _edit(svc, run, agent=lambda checkpoint: checkpoint["context"].update(principal_id="someone-else"))
    resume = BoundedAgent.resume_from_checkpoint

    def cancel_then_refuse(self, *args, **kwargs):
        assert svc.cancel(run_id=run["run_id"], identity=REQUESTER)["status"] == "CANCEL_REQUESTED"
        return resume(self, *args, **kwargs)

    monkeypatch.setattr(BoundedAgent, "resume_from_checkpoint", cancel_then_refuse)
    done = _resume(service, run, [])

    assert done["status"] == "CANCELLED"
    _assert_one_terminal_last(_events(svc, run["run_id"]))


def test_a_cancel_after_the_commits_check_ends_the_resume_cancelled(service, monkeypatch) -> None:
    svc, _ = service
    run = _paused(service, with_sql=False)
    answer_envelope = service_module._answer_envelope

    def cancel_then(*args, **kwargs):
        svc.cancel(run_id=run["run_id"], identity=REQUESTER)
        return answer_envelope(*args, **kwargs)

    monkeypatch.setattr(service_module, "_answer_envelope", cancel_then)
    done = _resume(service, run, [_query(), _cite_all])

    # The commit's final write is conditional: the run is no longer RUNNING, so it ends CANCELLED.
    assert done["status"] == "CANCELLED" and done["sql_exec_count"] == 1
    events = _events(svc, run["run_id"])
    _assert_one_terminal_last(events)
    assert events[-3:] == [("step_finished", "CANCEL_REQUESTED"), *CANCELLED_STEPS]


# --- an approved execution runs as RUNNING -----------------------------------------------------------------------


def test_a_cancel_during_an_approved_query_is_a_request_and_the_run_ends_once(service) -> None:
    svc, executor = service
    run = _pending(service)
    held = _hold_approved_query(svc, executor)
    worker, out = _in_thread(lambda: _approve(service, run))
    try:
        assert held.entered.wait(timeout=10)
        executing = svc.store.get_run(run["run_id"])
        assert executing["status"] == "RUNNING" and api_main._usage_total(executing) == UNKNOWN
        assert svc.cancel(run_id=run["run_id"], identity=REQUESTER)["status"] == "CANCEL_REQUESTED"
    finally:
        held.release.set()
        worker.join(timeout=10)
    done = out[0]
    assert done["status"] == "CANCELLED" and done["sql_exec_count"] == 2
    events = _events(svc, run["run_id"])
    _assert_one_terminal_last(events)
    assert events[-3:] == [("step_finished", "CANCEL_REQUESTED"), *CANCELLED_STEPS]


def test_an_approval_cancelled_before_its_query_runs_executes_nothing(service, monkeypatch) -> None:
    svc, executor = service
    run = _pending(service)
    events_before = _events(svc, run["run_id"])
    executed_before = list(executor.executed)
    check = service_module.RunService._require_current_permission

    def cancel_then_check(self, approval, subject):
        self.cancel(run_id=run["run_id"], identity=REQUESTER)
        return check(self, approval, subject)

    monkeypatch.setattr(service_module.RunService, "_require_current_permission", cancel_then_check)
    done = _approve(service, run)

    assert done["status"] == "CANCELLED" and executor.executed == executed_before
    assert svc.store.get_approval(run["approval_id"])["status"] == "APPROVED"
    assert _events(svc, run["run_id"]) == events_before + [("terminal", "CANCELLED")]
    monkeypatch.setattr(service_module.RunService, "_require_current_permission", check)
    # Approving again replays the cancelled run.
    assert _approve(service, run)["status"] == "CANCELLED" and executor.executed == executed_before


def test_an_approval_whose_consumed_record_no_longer_matches_leaves_the_run_waiting(service, monkeypatch) -> None:
    svc, executor = service
    run = _pending(service)
    events_before = _events(svc, run["run_id"])
    executed_before = list(executor.executed)
    decide = svc.store.decide_approval

    def changed(*args, **kwargs):
        return {**decide(*args, **kwargs), "action_hash": "0" * 64}

    monkeypatch.setattr(svc.store, "decide_approval", changed)
    with pytest.raises(ApprovalConflict):
        _approve(service, run)

    assert svc.store.get_run(run["run_id"])["status"] == "WAITING_APPROVAL"
    assert executor.executed == executed_before and _events(svc, run["run_id"]) == events_before


def test_a_cancel_after_the_approved_querys_check_ends_the_run_cancelled(service, monkeypatch) -> None:
    svc, _ = service
    run = _pending(service)
    refs = service_module._approved_fact_refs

    def cancel_then(*args, **kwargs):
        svc.cancel(run_id=run["run_id"], identity=REQUESTER)
        return refs(*args, **kwargs)

    monkeypatch.setattr(service_module, "_approved_fact_refs", cancel_then)
    done = _approve(service, run)

    # The approved query's final write is conditional: the run ends CANCELLED, with its SQL counted.
    assert done["status"] == "CANCELLED" and done["sql_exec_count"] == 2
    events = _events(svc, run["run_id"])
    _assert_one_terminal_last(events)
    assert events[-3:] == [("step_finished", "CANCEL_REQUESTED"), *CANCELLED_STEPS]


# --- a cancel that overlaps a start, a commit or a rejection -----------------------------------------------------


def test_a_cancel_that_read_waiting_while_a_resume_started_becomes_a_request(service, monkeypatch) -> None:
    svc, _ = service
    run = _paused(service)
    gate = _Gate(_cite_all)
    started: list = []

    def start_resume():
        started.append(_in_thread(lambda: _resume(service, run, [_query(), gate])))
        assert gate.entered.wait(timeout=10)

    _after_first_read(svc, monkeypatch, start_resume)
    try:
        cancelled = svc.cancel(run_id=run["run_id"], identity=REQUESTER)
        assert cancelled["status"] == "CANCEL_REQUESTED"
    finally:
        gate.release.set()
        if started:
            started[0][0].join(timeout=10)
    assert started[0][1][0]["status"] == "CANCELLED"
    events = _events(svc, run["run_id"])
    _assert_one_terminal_last(events)
    assert events[-2:] == CANCELLED_STEPS


def test_a_cancel_that_read_waiting_while_an_approved_query_started_becomes_a_request(service, monkeypatch) -> None:
    svc, executor = service
    run = _pending(service)
    held = _hold_approved_query(svc, executor)
    started: list = []

    def start_approval():
        started.append(_in_thread(lambda: _approve(service, run)))
        assert held.entered.wait(timeout=10)

    _after_first_read(svc, monkeypatch, start_approval)
    try:
        assert svc.cancel(run_id=run["run_id"], identity=REQUESTER)["status"] == "CANCEL_REQUESTED"
    finally:
        held.release.set()
        if started:
            started[0][0].join(timeout=10)
    assert started[0][1][0]["status"] == "CANCELLED"
    _assert_one_terminal_last(_events(svc, run["run_id"]))


def test_a_cancel_that_read_running_after_which_the_run_ended_leaves_the_outcome(service, monkeypatch) -> None:
    svc, _ = service
    run = _paused(service, with_sql=False)
    gate = _Gate(_cite_all)
    worker, out = _in_thread(lambda: _resume(service, run, [_query(), gate]))
    assert gate.entered.wait(timeout=10)

    def finish_resume():
        gate.release.set()
        worker.join(timeout=10)

    _after_first_read(svc, monkeypatch, finish_resume)
    try:
        cancelled = svc.cancel(run_id=run["run_id"], identity=REQUESTER)
    finally:
        gate.release.set()
        worker.join(timeout=10)
    assert out[0]["status"] == "SUCCEEDED" and cancelled["status"] == "SUCCEEDED"
    events = _events(svc, run["run_id"])
    _assert_one_terminal_last(events)
    assert ("step_finished", "CANCEL_REQUESTED") not in events


def test_a_cancel_that_read_running_while_the_resume_asked_again_cancels_the_waiting_run(service, monkeypatch) -> None:
    svc, _ = service
    run = _paused(service)
    gate = _Gate(_ask("请问是哪一个月？"))
    worker, out = _in_thread(lambda: _resume(service, run, [COUNT, gate]))
    assert gate.entered.wait(timeout=10)

    def finish_resume():
        gate.release.set()
        worker.join(timeout=10)

    _after_first_read(svc, monkeypatch, finish_resume)
    try:
        cancelled = svc.cancel(run_id=run["run_id"], identity=REQUESTER)
    finally:
        gate.release.set()
        worker.join(timeout=10)
    assert out[0]["status"] == "WAITING_USER" and cancelled["status"] == "CANCELLED"
    _assert_one_terminal_last(_events(svc, run["run_id"]))
    with pytest.raises(ApprovalConflict) as again:
        _resume(service, run, [_query(), _cite_all])
    assert again.value.code == "invalid_run_state"


def test_a_cancel_that_read_running_while_the_first_execution_paused_for_approval_cancels_it(service, monkeypatch) -> None:
    svc, executor = service
    created: list[str] = []
    create_run = svc.store.create_run

    def recording(**kwargs):
        created.append(kwargs["run_id"])
        return create_run(**kwargs)

    monkeypatch.setattr(svc.store, "create_run", recording)
    gate = _Gate(NAME_QUERY)
    deps = svc.default_dependencies()
    deps.model, deps.retriever = _Metered([COUNT, gate]), None
    worker, out = _in_thread(
        lambda: svc.run_sync(identity=REQUESTER, question="2026年9月支付笔数和客户姓名", time_window=None, deps=deps)
    )
    assert gate.entered.wait(timeout=10)

    def finish_execution():
        gate.release.set()
        worker.join(timeout=10)

    _after_first_read(svc, monkeypatch, finish_execution)
    try:
        cancelled = svc.cancel(run_id=created[0], identity=REQUESTER)
    finally:
        gate.release.set()
        worker.join(timeout=10)
    assert out[0]["status"] == "WAITING_APPROVAL" and cancelled["status"] == "CANCELLED"
    _assert_one_terminal_last(_events(svc, created[0]))
    executed = list(executor.executed)
    with pytest.raises(ApprovalConflict) as approving:
        _approve(service, cancelled)
    assert approving.value.code == "invalid_run_state" and executor.executed == executed


def test_a_cancel_flag_left_on_a_waiting_run_does_not_cancel_the_next_resume(service) -> None:
    """Such a run comes only from data stored before the final writes were conditional.

    The old commit committed its outcome after a cancel had arrived past its check, and left
    the cancel flag on the waiting run.  A resume starts clean.
    """

    svc, _ = service
    run = _paused(service, with_sql=False)
    svc.store.update_run(run["run_id"], cancel_requested=1)

    done = _resume(service, run, [_query(), _cite_all])

    assert done["status"] == "SUCCEEDED"
    _assert_one_terminal_last(_events(svc, run["run_id"]))


def test_a_cancel_flag_left_on_a_waiting_run_does_not_cancel_the_approved_query(service) -> None:
    """As above, for a run waiting for approval: the approved execution starts clean."""

    svc, _ = service
    run = _pending(service)
    svc.store.update_run(run["run_id"], cancel_requested=1)

    done = _approve(service, run)

    assert done["status"] == "SUCCEEDED" and done["sql_exec_count"] == 2
    _assert_one_terminal_last(_events(svc, run["run_id"]))


def test_a_cancel_whose_run_left_and_came_back_to_waiting_decides_again(service, monkeypatch) -> None:
    """The waiting transition fails because the run started, and the run is already waiting again when re-read."""

    svc, _ = service
    run = _paused(service)
    transition = svc.store.transition_run
    moved = [False]

    def started_and_waited_again(run_id, from_status, to_status, **kwargs):
        if moved[0] or to_status != "CANCELLED":
            return transition(run_id, from_status, to_status, **kwargs)
        moved[0] = True
        svc.store.update_run(run_id, status="RUNNING")
        try:
            return transition(run_id, from_status, to_status, **kwargs)
        finally:
            svc.store.update_run(run_id, status="WAITING_USER")

    monkeypatch.setattr(svc.store, "transition_run", started_and_waited_again)
    cancelled = svc.cancel(run_id=run["run_id"], identity=REQUESTER)

    assert moved[0] and cancelled["status"] == "CANCELLED"
    events = _events(svc, run["run_id"])
    _assert_one_terminal_last(events)
    assert events[-1] == ("terminal", "CANCELLED")


def test_a_rejection_overlapped_by_a_cancel_keeps_the_cancel(service, monkeypatch) -> None:
    svc, _ = service
    run = _pending(service)
    decide = svc.store.decide_approval

    def decide_then_cancel(*args, **kwargs):
        decided = decide(*args, **kwargs)
        svc.cancel(run_id=run["run_id"], identity=REQUESTER)
        return decided

    monkeypatch.setattr(svc.store, "decide_approval", decide_then_cancel)
    done = _approve(service, run, decision="reject")

    assert done["status"] == "CANCELLED"
    assert svc.store.get_approval(run["approval_id"])["status"] == "REJECTED"
    events = _events(svc, run["run_id"])
    _assert_one_terminal_last(events)
    assert events[-1] == ("terminal", "CANCELLED")


def test_a_cancel_that_read_waiting_while_the_approval_was_rejected_keeps_the_rejection(service, monkeypatch) -> None:
    svc, _ = service
    run = _pending(service)
    _after_first_read(svc, monkeypatch, lambda: _approve(service, run, decision="reject"))

    cancelled = svc.cancel(run_id=run["run_id"], identity=REQUESTER)

    assert cancelled["status"] == "DENIED"
    events = _events(svc, run["run_id"])
    _assert_one_terminal_last(events)
    assert events[-1] == ("terminal", "DENIED")


# --- the conditional transition -----------------------------------------------------------------------------------


def _store_with_run(tmp_path) -> StateStore:
    store = StateStore(tmp_path / "state.sqlite3")
    store.create_run(
        run_id="run-t", tenant_id="A", principal_id="p", role="requester", question="q", mode="fake",
        checkpoint={}, run_config={}, model_call_count=0,
    )
    return store


def test_a_transition_applies_only_from_the_named_status_and_writes_its_event_only_then(tmp_path) -> None:
    store = _store_with_run(tmp_path)
    try:
        before = store.events("run-t", after_event_id=0, limit=10)
        assert store.get_run("run-t")["status"] == "RUNNING"
        assert not store.transition_run("run-t", "WAITING_USER", "CANCELLED", events=[{"event_type": "terminal", "status": "CANCELLED"}])
        assert (store.get_run("run-t")["status"], store.events("run-t", after_event_id=0, limit=10)) == ("RUNNING", before)

        assert store.transition_run(
            "run-t", "RUNNING", "CANCEL_REQUESTED", events=[{"event_type": "step_finished", "status": "CANCEL_REQUESTED", "payload": {"x": 1}}], cancel_requested=1
        )
        run = store.get_run("run-t")
        assert (run["status"], run["cancel_requested"]) == ("CANCEL_REQUESTED", True)
        [event] = store.events("run-t", after_event_id=0, limit=10)[len(before):]
        assert (event["type"], event["status"], event["payload"]) == ("step_finished", "CANCEL_REQUESTED", {"x": 1})
        assert not store.transition_run("run-t", "RUNNING", "SUCCEEDED")
    finally:
        store.close()


def test_a_transition_whose_write_fails_stores_neither_its_event_nor_its_status(tmp_path) -> None:
    store = _store_with_run(tmp_path)
    try:
        before = store.events("run-t", after_event_id=0, limit=10)
        with pytest.raises(StateStoreError):
            store.transition_run("run-t", "RUNNING", "CANCELLED", events=[{"event_type": "terminal", "status": "CANCELLED"}], no_such_field=1)
        assert (store.get_run("run-t")["status"], store.events("run-t", after_event_id=0, limit=10)) == ("RUNNING", before)
    finally:
        store.close()


# --- over HTTP ----------------------------------------------------------------------------------------------------


@pytest.fixture()
def http_model(server):
    model = _Metered([])
    app.dependency_overrides[get_model_provider] = lambda: model
    app.dependency_overrides[get_retriever_source] = lambda: (lambda: None)
    return model


def _http_paused(port, model, *, with_sql: bool = True) -> str:
    model.steps += [COUNT, _query()] if with_sql else [_query()]
    code, body = _request(port, "POST", "/queries", REQUESTER_TOKEN, {"question": AMBIGUOUS})
    assert (code, body["status"]) == (202, "WAITING_USER")
    return body["run_id"]


def test_over_http_a_running_resume_streams_its_steps_and_a_cancel_answers_202(server, http_model) -> None:
    run_id = _http_paused(server, http_model)
    paused_events = len(shared_run_service().store.events(run_id, after_event_id=0, limit=1000))
    gate = _Gate(_cite_all)
    http_model.steps += [_query(), gate]
    stream = _Frames(server, run_id, REQUESTER_TOKEN)
    worker, out = _in_thread(lambda: _request(server, "POST", f"/runs/{run_id}/resume", REQUESTER_TOKEN, {"answer": "按支付金额"}))
    try:
        assert gate.entered.wait(timeout=10)
        frames = stream.until(lambda frames: len(frames) >= paused_events + 2)
        assert [frame["type"] for frame in frames[paused_events:]] == ["agent_step", "agent_step"]
        code, body = _request(server, "GET", f"/runs/{run_id}", REQUESTER_TOKEN)
        assert (code, body["status"], body["usage_total"]) == (200, "RUNNING", UNKNOWN)
        code, body = _request(server, "POST", f"/runs/{run_id}/cancel", REQUESTER_TOKEN, {})
        assert (code, body["status"], body["usage_total"]) == (202, "CANCEL_REQUESTED", UNKNOWN)
    finally:
        gate.release.set()
        worker.join(timeout=10)
    frames = stream.until(lambda frames: frames and frames[-1]["type"] == "terminal")
    stream.close()
    assert [frame["type"] for frame in frames].count("terminal") == 1
    code, body = out[0]
    assert (code, body["error"]["code"], body["status"]) == (409, "run_cancelled", "CANCELLED")


def test_over_http_the_approver_does_not_see_an_approved_query_while_it_runs(server, http_model, monkeypatch) -> None:
    http_model.steps += [COUNT, NAME_QUERY]
    code, body = _request(server, "POST", "/queries", REQUESTER_TOKEN, {"question": "2026年9月支付笔数和客户姓名"})
    assert body["status"] == "WAITING_APPROVAL"
    run_id, approval_id = body["run_id"], body["approval_id"]
    svc = shared_run_service()
    held = _Held(svc._executor_factory())
    monkeypatch.setattr(svc, "_executor_factory", lambda: held)
    worker, out = _in_thread(
        lambda: _request(server, "POST", f"/runs/{run_id}/approval", APPROVER, {"approval_id": approval_id, "decision": "approve"})
    )
    try:
        assert held.entered.wait(timeout=10)
        assert _request(server, "GET", f"/runs/{run_id}", APPROVER)[0] == 404
        code, body = _request(server, "GET", f"/runs/{run_id}", REQUESTER_TOKEN)
        assert (code, body["status"], body["usage_total"]) == (200, "RUNNING", UNKNOWN)
    finally:
        held.release.set()
        worker.join(timeout=10)
    assert out[0][0] == 200 and out[0][1]["status"] == "SUCCEEDED"


def test_over_http_a_resume_whose_commit_fails_answers_502_and_then_409(server, http_model, monkeypatch) -> None:
    run_id = _http_paused(server, http_model, with_sql=False)

    def failing(self, *args, **kwargs):
        raise ApprovalConflict("evidence_validation_failed", "result evidence could not be persisted")

    monkeypatch.setattr(service_module.RunService, "_committed_evidences", failing)
    http_model.steps += [_query(), _cite_all]
    code, body = _request(server, "POST", f"/runs/{run_id}/resume", REQUESTER_TOKEN, {"answer": "按支付金额"})
    assert (code, body["status"], body["error"]["code"]) == (502, "FAILED", "evidence_validation_failed")
    code, body = _request(server, "POST", f"/runs/{run_id}/resume", REQUESTER_TOKEN, {"answer": "按支付金额"})
    assert (code, body["error"]["code"]) == (409, "invalid_run_state")

