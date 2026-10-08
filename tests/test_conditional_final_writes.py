"""The execution's final write is a conditional transition, like a start, a cancel and a denial.

Committing a result, pausing again, pausing for approval, an approved query's success
and an error exit all move the run out of RUNNING in one transaction with their events;
a run that is no longer RUNNING was asked to cancel, and ends CANCELLED (from
CANCEL_REQUESTED, also conditional).  So a cancel answered 202 always ends the run
CANCELLED, the terminal event is the last one, the cancel request comes before it, and
a ``waiting`` event comes before any later terminal event.  Interleavings are fixed with
hooks, never left to timing.
"""

from __future__ import annotations

import pytest

from queryshield.approval import service as service_module
from queryshield.db.state_store import StateStoreError, json_text

from test_agent_step_writes import _Gate, _Metered
from test_approval_api_pins import APPROVER_A, NAME_QUERY
from test_clarification import REQUESTER, _cite, _query, service  # noqa: F401  (service is a fixture)
from test_resumed_and_approved_execution import (
    AMBIGUOUS,
    CANCELLED_STEPS,
    COUNT,
    _after_first_read,
    _assert_one_terminal_last,
    _cite_all,
    _events,
    _in_thread,
    _paused,
    _pending,
    _resume,
    _store_with_run,
)
from test_results_from_before_a_pause import ASK_AGAIN

PAID = "2026年9月支付金额是多少"
MIXED = "2026年9月支付笔数和客户姓名"
REQUESTED = ("step_finished", "CANCEL_REQUESTED")


def _start(service, question, steps) -> dict:
    svc, _ = service
    deps = svc.default_dependencies()
    deps.model, deps.retriever = _Metered(steps), None
    return svc.run_sync(identity=REQUESTER, question=question, time_window=None, deps=deps)


def _approvals(svc, run_id: str) -> list[str]:
    rows = svc.store._connection.execute("SELECT status FROM approvals WHERE run_id = ?", (run_id,)).fetchall()
    return [row[0] for row in rows]


def _assert_cancelled_once(svc, run_id: str, requested: dict) -> dict:
    """The cancel answered 202; the run ended CANCELLED with one terminal event, after the request."""

    assert requested["status"] == "CANCEL_REQUESTED"
    run = svc.store.get_run(run_id)
    assert run["status"] == "CANCELLED"
    # A cancelled run carries no outcome of the execution it stopped.
    assert (run["answer"], run["result"], run["facts"]) == (None, None, None)
    events = _events(svc, run_id)
    _assert_one_terminal_last(events)
    assert events.count(REQUESTED) == 1 and events.index(REQUESTED) < len(events) - 2
    assert events[-2:] == CANCELLED_STEPS
    assert all(kind != "waiting" for kind, _ in events[events.index(REQUESTED):])
    return run


def _cancel_in_commit(svc, monkeypatch, seen: list) -> None:
    """Cancel once, after the commit's own cancel check and before its final write."""

    append = svc._append_agent_events

    def cancelling(run_id, *args, **kwargs):
        if not seen:
            seen.append(svc.cancel(run_id=run_id, identity=REQUESTER))
        return append(run_id, *args, **kwargs)

    monkeypatch.setattr(svc, "_append_agent_events", cancelling)


def _cancel_on_error_exit(svc, monkeypatch, seen: list, run_id=None) -> None:
    """Cancel once while an error exit computes its code (after it read the run, before its write)."""

    error_code = service_module._error_code

    def cancelling(exc):
        if not seen:
            seen.append(svc.cancel(run_id=run_id or next(iter(svc._active)), identity=REQUESTER))
        return error_code(exc)

    monkeypatch.setattr(service_module, "_error_code", cancelling)


def _fail_commit(svc, monkeypatch) -> None:
    def failing(*args, **kwargs):
        raise RuntimeError("the commit failed")

    monkeypatch.setattr(svc, "_commit", failing)


# --- a cancel between the commit's check and its final write is never lost ------------------------------------


@pytest.mark.parametrize(
    ("question", "steps", "counts"),
    [
        (PAID, [_query(), _cite], (2, 1, 1)),
        (AMBIGUOUS, [_query()], (1, 1, 0)),
        (MIXED, [COUNT, NAME_QUERY], (2, 2, 1)),
    ],
    ids=["succeeded", "waiting_user", "waiting_approval"],
)
def test_a_first_execution_cancelled_before_its_final_write_ends_cancelled(service, monkeypatch, question, steps, counts) -> None:
    svc, _ = service
    seen: list = []
    _cancel_in_commit(svc, monkeypatch, seen)

    done = _start(service, question, steps)

    run = _assert_cancelled_once(svc, done["run_id"], seen[0])
    assert done == run
    assert (run["model_call_count"], run["tool_call_count"], run["sql_exec_count"]) == counts
    assert _approvals(svc, run["run_id"]) == []
    assert run["usage"]["status"] == "known"


def test_a_first_execution_cancelled_while_it_ends_on_an_error_ends_cancelled(service, monkeypatch) -> None:
    svc, _ = service
    seen: list = []
    _fail_commit(svc, monkeypatch)
    _cancel_on_error_exit(svc, monkeypatch, seen)

    done = _start(service, PAID, [_query(), _cite])

    run = _assert_cancelled_once(svc, done["run_id"], seen[0])
    assert (run["model_call_count"], run["tool_call_count"], run["sql_exec_count"], run["error_code"]) == (2, 1, 1, None)


@pytest.mark.parametrize(
    ("steps", "counts"),
    [([_query(), _cite_all], (3, 2, 1)), ([_query(), ASK_AGAIN], (3, 2, 1))],
    ids=["succeeded", "waiting_user"],
)
def test_a_resume_cancelled_before_its_final_write_ends_cancelled(service, monkeypatch, steps, counts) -> None:
    svc, _ = service
    run = _paused(service, with_sql=False)
    seen: list = []
    _cancel_in_commit(svc, monkeypatch, seen)

    done = _resume(service, run, steps)

    cancelled = _assert_cancelled_once(svc, run["run_id"], seen[0])
    assert done == cancelled
    assert (cancelled["model_call_count"], cancelled["tool_call_count"], cancelled["sql_exec_count"]) == counts


def test_a_resume_cancelled_while_it_ends_on_an_error_ends_cancelled(service, monkeypatch) -> None:
    svc, _ = service
    run = _paused(service)
    seen: list = []
    _fail_commit(svc, monkeypatch)
    _cancel_on_error_exit(svc, monkeypatch, seen, run_id=run["run_id"])

    _resume(service, run, [_query(), _cite_all])

    cancelled = _assert_cancelled_once(svc, run["run_id"], seen[0])
    assert (cancelled["model_call_count"], cancelled["tool_call_count"], cancelled["sql_exec_count"]) == (4, 3, 2)


def test_an_approved_query_cancelled_before_its_final_write_ends_cancelled(service, monkeypatch) -> None:
    svc, _ = service
    run = _pending(service)
    seen: list = []
    approved_answer = service_module._approved_answer

    def cancelling(*args, **kwargs):
        if not seen:
            seen.append(svc.cancel(run_id=run["run_id"], identity=REQUESTER))
        return approved_answer(*args, **kwargs)

    monkeypatch.setattr(service_module, "_approved_answer", cancelling)

    done = svc.approve(run_id=run["run_id"], approval_id=run["approval_id"], identity=APPROVER_A, decision="approve")

    cancelled = _assert_cancelled_once(svc, run["run_id"], seen[0])
    assert done == cancelled and cancelled["sql_exec_count"] == 2
    assert cancelled["result"] is None and cancelled["answer"] is None


def test_an_approved_query_cancelled_while_it_ends_on_an_error_ends_cancelled(service, monkeypatch) -> None:
    svc, _ = service
    run = _pending(service)
    seen: list = []

    def failing(*args, **kwargs):
        raise RuntimeError("rendering failed")

    monkeypatch.setattr(service_module, "_approved_answer", failing)
    _cancel_on_error_exit(svc, monkeypatch, seen, run_id=run["run_id"])

    svc.approve(run_id=run["run_id"], approval_id=run["approval_id"], identity=APPROVER_A, decision="approve")

    cancelled = _assert_cancelled_once(svc, run["run_id"], seen[0])
    assert (cancelled["sql_exec_count"], cancelled["error_code"]) == (2, None)


def _comparable(envelope: dict) -> dict:
    # Call ids and the run's own id are random; a waiting checkpoint carries elapsed times.
    return {k: v for k, v in envelope.items() if k not in {"model_call_id", "model_call_ids", "idempotency_key", "agent_checkpoint"}}


@pytest.mark.parametrize(
    ("question", "steps"),
    [(PAID, [_query(), _cite]), (AMBIGUOUS, [COUNT, _query()])],
    ids=["succeeded", "waiting_user"],
)
def test_a_cancel_at_the_final_write_stores_the_envelope_of_a_cancel_at_the_commit_check(service, monkeypatch, question, steps) -> None:
    svc, _ = service
    close_metadata = svc._close_metadata
    early_seen: list = []

    def cancelling_before_the_commit(tools, written):
        # The graph has finished; the commit has not read the run yet.
        if not early_seen:
            early_seen.append(svc.cancel(run_id=next(iter(svc._active)), identity=REQUESTER))
        return close_metadata(tools, written)

    monkeypatch.setattr(svc, "_close_metadata", cancelling_before_the_commit)
    early = svc.store.get_run(_start(service, question, steps)["run_id"])
    monkeypatch.setattr(svc, "_close_metadata", close_metadata)
    seen: list = []
    _cancel_in_commit(svc, monkeypatch, seen)
    late = svc.store.get_run(_start(service, question, steps)["run_id"])

    assert early_seen[0]["status"] == seen[0]["status"] == "CANCEL_REQUESTED"
    assert early["status"] == late["status"] == "CANCELLED"
    assert _comparable(late["checkpoint"]) == _comparable(early["checkpoint"])
    assert late["checkpoint"]["status"] == "CANCELLED"
    assert (late["checkpoint"]["agent_checkpoint"] is None) == (early["checkpoint"]["agent_checkpoint"] is None)
    assert not {"paused_results", "pre_approval_results", "answer_status"} & set(late["checkpoint"])


# --- the final transaction: a cancel before it, or after it ----------------------------------------------------


def _around_final_write(svc, monkeypatch, to_status: str, *, before: bool, seen: list, run_id_of=lambda run_id: run_id) -> None:
    """Cancel once, just before or just after the run's transition RUNNING -> ``to_status``."""

    transition = svc.store.transition_run

    def hooked(run_id, from_status, to, **kwargs):
        mine = not seen and from_status == "RUNNING" and to == to_status
        if mine and before:
            seen.append(svc.cancel(run_id=run_id, identity=REQUESTER))
        moved = transition(run_id, from_status, to, **kwargs)
        if mine and not before:
            seen.append(svc.cancel(run_id=run_id, identity=REQUESTER))
        return moved

    monkeypatch.setattr(svc.store, "transition_run", hooked)


def test_a_cancel_just_before_the_succeeded_transaction_cancels_the_run(service, monkeypatch) -> None:
    svc, _ = service
    seen: list = []
    _around_final_write(svc, monkeypatch, "SUCCEEDED", before=True, seen=seen)

    done = _start(service, PAID, [_query(), _cite])

    _assert_cancelled_once(svc, done["run_id"], seen[0])


def test_a_cancel_just_after_the_succeeded_transaction_leaves_the_outcome_and_writes_nothing(service, monkeypatch) -> None:
    svc, _ = service
    seen: list = []
    _around_final_write(svc, monkeypatch, "SUCCEEDED", before=False, seen=seen)

    done = _start(service, PAID, [_query(), _cite])

    assert seen[0]["status"] == done["status"] == "SUCCEEDED"
    events = _events(svc, done["run_id"])
    _assert_one_terminal_last(events)
    assert REQUESTED not in events and events[-1] == ("terminal", "SUCCEEDED")


def test_a_cancel_just_before_a_resume_pauses_again_cancels_the_run(service, monkeypatch) -> None:
    svc, _ = service
    run = _paused(service, with_sql=False)
    seen: list = []
    _around_final_write(svc, monkeypatch, "WAITING_USER", before=True, seen=seen)

    _resume(service, run, [_query(), ASK_AGAIN])

    _assert_cancelled_once(svc, run["run_id"], seen[0])


def test_a_cancel_just_after_a_resume_paused_again_cancels_the_waiting_run_after_its_waiting_event(service, monkeypatch) -> None:
    svc, _ = service
    run = _paused(service, with_sql=False)
    seen: list = []
    _around_final_write(svc, monkeypatch, "WAITING_USER", before=False, seen=seen)

    _resume(service, run, [_query(), ASK_AGAIN])

    # The cancel found the run waiting again and ended it there.
    assert seen[0]["status"] == "CANCELLED"
    events = _events(svc, run["run_id"])
    _assert_one_terminal_last(events)
    assert events[-2:] == [("waiting", "WAITING_USER"), ("terminal", "CANCELLED")]


def test_a_cancel_just_before_an_approval_pause_creates_no_approval(service, monkeypatch) -> None:
    svc, _ = service
    seen: list = []
    _around_final_write(svc, monkeypatch, "WAITING_APPROVAL", before=True, seen=seen)

    done = _start(service, MIXED, [COUNT, NAME_QUERY])

    _assert_cancelled_once(svc, done["run_id"], seen[0])
    assert _approvals(svc, done["run_id"]) == [] and done["approval_id"] is None


def test_a_commit_whose_run_was_ended_by_another_writer_writes_nothing(service, monkeypatch) -> None:
    svc, _ = service
    transition = svc.store.transition_run
    counted: list = []

    def ended_meanwhile(run_id, from_status, to, **kwargs):
        if from_status == "RUNNING" and to == "SUCCEEDED" and not counted:
            svc.store.update_run(run_id, status="FAILED", error_code="execution_failed")
            counted.append(len(_events(svc, run_id)))
        return transition(run_id, from_status, to, **kwargs)

    monkeypatch.setattr(svc.store, "transition_run", ended_meanwhile)

    done = _start(service, PAID, [_query(), _cite])

    assert (done["status"], done["error_code"]) == ("FAILED", "execution_failed")
    assert len(_events(svc, done["run_id"])) == counted[0]


# --- two cancels; a cancel of CANCEL_REQUESTED -----------------------------------------------------------------


def test_two_cancels_that_both_read_running_answer_202_with_one_request_event(service, monkeypatch) -> None:
    svc, _ = service
    gate = _Gate(_cite)
    deps = svc.default_dependencies()
    deps.model, deps.retriever = _Metered([_query(), gate]), None
    worker, out = _in_thread(lambda: svc.run_sync(identity=REQUESTER, question=PAID, time_window=None, deps=deps))
    try:
        assert gate.entered.wait(timeout=10)
        run_id = next(iter(svc._active))
        inner: list = []
        _after_first_read(svc, monkeypatch, lambda: inner.append(svc.cancel(run_id=run_id, identity=REQUESTER)))
        outer = svc.cancel(run_id=run_id, identity=REQUESTER)
    finally:
        gate.release.set()
        worker.join(timeout=10)
    assert outer["status"] == inner[0]["status"] == "CANCEL_REQUESTED"
    events = _events(svc, run_id)
    assert events.count(REQUESTED) == 1
    assert out[0]["status"] == "CANCELLED"
    _assert_one_terminal_last(events)


def test_a_cancel_of_a_run_already_being_cancelled_answers_202_and_writes_nothing(service) -> None:
    svc, _ = service
    gate = _Gate(_cite)
    deps = svc.default_dependencies()
    deps.model, deps.retriever = _Metered([_query(), gate]), None
    worker, out = _in_thread(lambda: svc.run_sync(identity=REQUESTER, question=PAID, time_window=None, deps=deps))
    try:
        assert gate.entered.wait(timeout=10)
        run_id = next(iter(svc._active))
        first = svc.cancel(run_id=run_id, identity=REQUESTER)
        events = _events(svc, run_id)
        again = svc.cancel(run_id=run_id, identity=REQUESTER)
        assert first["status"] == again["status"] == "CANCEL_REQUESTED"
        assert again == svc.store.get_run(run_id)
        assert _events(svc, run_id) == events
    finally:
        gate.release.set()
        worker.join(timeout=10)
    assert out[0]["status"] == "CANCELLED"


# --- the pause for approval stores the run record as before ----------------------------------------------------


def test_an_approval_pause_stores_the_action_as_the_store_serializes_it(service) -> None:
    svc, _ = service
    run = _pending(service)
    raw = svc.store._connection.execute("SELECT action_json, approval_id FROM runs WHERE run_id = ?", (run["run_id"],)).fetchone()
    approval = svc.store._connection.execute("SELECT action_json FROM approvals WHERE run_id = ?", (run["run_id"],)).fetchone()
    assert raw[0] == approval[0] == json_text(run["action"])
    assert raw[1] == run["approval_id"]
    assert _events(svc, run["run_id"])[-1] == ("waiting", "WAITING_APPROVAL")


# --- the store's conditional transition ------------------------------------------------------------------------


_APPROVAL = {
    "approval_id": "approval-t",
    "tenant_id": "A",
    "requester_principal_id": "p",
    "action_hash": "h" * 64,
    "action": {"kind": "query_readonly", "sql": "SELECT 1"},
    "policy_version": "policy",
    "catalog_version": "catalog",
    "knowledge_snapshot_id": "snapshot",
    "expires_at": service_module.utc_now(),
}


def test_a_transition_writes_every_event_in_order_with_its_result_id(tmp_path) -> None:
    store = _store_with_run(tmp_path)
    try:
        before = len(store.events("run-t", after_event_id=0, limit=10))
        assert store.transition_run(
            "run-t", "RUNNING", "SUCCEEDED",
            events=[
                {"event_type": "step_finished", "status": "SUCCEEDED", "result_id": "result-1", "payload": {"step": "x"}},
                {"event_type": "terminal", "status": "SUCCEEDED", "result_id": "result-1"},
            ],
        )
        written = store.events("run-t", after_event_id=0, limit=10)[before:]
        assert [(e["type"], e["status"], e["result_id"], e["payload"]) for e in written] == [
            ("step_finished", "SUCCEEDED", "result-1", {"step": "x"}),
            ("terminal", "SUCCEEDED", "result-1", {}),
        ]
    finally:
        store.close()


def test_a_transition_to_an_approval_pause_inserts_the_approval_only_from_the_named_status(tmp_path) -> None:
    store = _store_with_run(tmp_path)
    try:
        waiting = [{"event_type": "waiting", "status": "WAITING_APPROVAL", "payload": {"approval_id": "approval-t"}}]
        assert not store.transition_run("run-t", "WAITING_USER", "WAITING_APPROVAL", events=waiting, approval=_APPROVAL)
        assert store.get_approval("approval-t") is None and store.get_run("run-t")["status"] == "RUNNING"

        assert store.transition_run("run-t", "RUNNING", "WAITING_APPROVAL", events=waiting, approval=_APPROVAL)
        run = store.get_run("run-t")
        assert (run["status"], run["approval_id"], run["action"]) == ("WAITING_APPROVAL", "approval-t", _APPROVAL["action"])
        assert store.get_approval("approval-t")["status"] == "PENDING"
        assert store.events("run-t", after_event_id=0, limit=10)[-1]["type"] == "waiting"
    finally:
        store.close()


def test_a_transition_whose_write_fails_stores_no_approval(tmp_path) -> None:
    store = _store_with_run(tmp_path)
    try:
        before = store.events("run-t", after_event_id=0, limit=10)
        with pytest.raises(StateStoreError):
            store.transition_run("run-t", "RUNNING", "WAITING_APPROVAL", approval=_APPROVAL, no_such_field=1)
        assert store.get_approval("approval-t") is None
        assert (store.get_run("run-t")["status"], store.events("run-t", after_event_id=0, limit=10)) == ("RUNNING", before)
    finally:
        store.close()


def test_creating_an_approval_directly_still_pauses_the_run(tmp_path) -> None:
    store = _store_with_run(tmp_path)
    try:
        store.create_approval(run_id="run-t", **_APPROVAL)
        run = store.get_run("run-t")
        assert (run["status"], run["approval_id"]) == ("WAITING_APPROVAL", "approval-t")
        assert store.events("run-t", after_event_id=0, limit=10)[-1]["type"] == "waiting"
    finally:
        store.close()
