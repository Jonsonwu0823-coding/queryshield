"""A process exit leaves runs executing; the next startup ends them.

At startup, after the parallel groups, every run that is RUNNING or CANCEL_REQUESTED and
not executing in this process ends with one conditional transition: RUNNING -> FAILED
``execution_interrupted``, CANCEL_REQUESTED -> CANCELLED (the events of an execution that
ends on a cancel request).  The counts stay as stored, the usage becomes unknown; waiting
and ended runs are not touched.

An exit is simulated in this process: the execution raises a ``BaseException`` where it
would block, so no final write happens and only ``finally`` blocks run, as when the process
dies there.  Runs that do execute in this process (gated) are never ended.  Interleavings
are fixed with hooks; every wait has a timeout.
"""

from __future__ import annotations

import json
import threading

import pytest
from fastapi.testclient import TestClient

from queryshield.agent import ModelCallStore
from queryshield.api import main as api_main
from queryshield.api.main import app, get_model_provider
from queryshield.approval import service as service_module
from queryshield.approval.service import RunService, reset_shared_state_stores, shared_run_service
from queryshield.approval.fixture_executor import FixtureQueryExecutor
from queryshield.db.state_store import StateStore

from test_agent_step_writes import _Gate, _Metered, _wait_for
from test_approval_api_pins import NAME_QUERY
from test_clarification import _Recording, _cite, _query
from test_http_queries import APPROVER, REQUESTER, auth
from test_live_agent_steps import _Frames, server  # noqa: F401  (server is a fixture)
from test_resumed_and_approved_execution import AMBIGUOUS, COUNT, _cite_all

REQ = {"tenant_id": "A", "principal_id": "a-requester", "role": "requester"}
APPROVER_A = {"tenant_id": "A", "principal_id": "a-approver", "role": "approver"}
PAID = "2026年9月支付金额是多少"
MIXED = "2026年9月支付笔数和客户姓名"
UNKNOWN = {"status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
FAILED_TAIL = [("terminal", "FAILED", {"error_code": "execution_interrupted"})]
CANCELLED_TAIL = [("step_finished", "CANCELLED", {"cancelled_before_commit": True}), ("terminal", "CANCELLED", {})]
INTERRUPTED = ("running_first", "running_resume", "running_approved", "cancel_requested", "cancel_requested_approved")
UNTOUCHED = ("succeeded", "cancelled", "waiting_user", "waiting_approval")


class _ProcessExit(BaseException):
    """The process dies here: ``except Exception`` does not catch it; only ``finally`` blocks run."""


def _exit_here(messages):
    raise _ProcessExit()


class _ExitingExecutor(_Recording):
    """An executor whose process dies inside the query, after ``before`` (if any)."""

    def __init__(self, before=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.before = before

    def execute(self, sql, **kwargs):
        if self.before is not None:
            self.before()
        raise _ProcessExit()


class _GatedExecutor(_Recording):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.entered, self.release = threading.Event(), threading.Event()

    def execute(self, sql, **kwargs):
        self.entered.set()
        assert self.release.wait(timeout=10), "the gated query was not released"
        return super().execute(sql, **kwargs)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A file state store the product service uses; no lifespan entered yet."""

    path = tmp_path / "state.sqlite3"
    monkeypatch.setenv("QUERYSHIELD_STATE_STORE_PATH", str(path))
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    monkeypatch.delenv("QUERYSHIELD_AGENT_PROFILE", raising=False)
    monkeypatch.delenv("QUERYSHIELD_RETRIEVAL", raising=False)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", REQUESTER)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_APPROVER", APPROVER)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_B_REQUESTER", "startup-b-requester")
    monkeypatch.setenv("QUERYSHIELD_TOKEN_B_APPROVER", "startup-b-approver")
    reset_shared_state_stores()
    yield path
    app.dependency_overrides.clear()
    reset_shared_state_stores()


class _Old:
    """The service and store of the process that exits (its own store on the same file)."""

    def __init__(self, path) -> None:
        self.store = StateStore(path)
        self.executor: object = _Recording()
        self.svc = RunService(store=self.store, executor_factory=lambda: self.executor, mode="fake")

    def sync(self, question, steps) -> dict:
        deps = self.svc.default_dependencies()
        deps.model, deps.retriever = _Metered(steps), None
        return self.svc.run_sync(identity=REQ, question=question, time_window=None, deps=deps)

    def resume(self, run_id, steps, model=None) -> dict:
        return self.svc.resume_waiting_user(
            run_id=run_id, answer="按支付金额", identity=REQ, model=model or _Metered(steps),
            call_store=ModelCallStore(), executor=self.executor,
        )

    def approve(self, run) -> dict:
        return self.svc.approve(run_id=run["run_id"], approval_id=str(run["approval_id"]), identity=APPROVER_A, decision="approve")

    def dies(self, action) -> None:
        with pytest.raises(_ProcessExit):
            action()

    def running_id(self) -> str:
        return next(iter(self.svc._active))


def _left_by_an_exit(path) -> dict[str, str]:
    """Runs in every status, four of them left executing by an exit, one of them asked to cancel."""

    old = _Old(path)
    ids = {"succeeded": old.sync(PAID, [_query(), _cite])["run_id"]}
    cancelled = old.sync(AMBIGUOUS, [COUNT, _query()])
    old.svc.cancel(run_id=cancelled["run_id"], identity=REQ)
    ids["cancelled"] = cancelled["run_id"]
    ids["waiting_user"] = old.sync(AMBIGUOUS, [COUNT, _query()])["run_id"]
    ids["waiting_approval"] = old.sync(MIXED, [COUNT, NAME_QUERY])["run_id"]

    # Paused with (model, tool, sql) stored; the resume writes more steps and runs an SQL first.
    ids["running_resume"] = old.sync(AMBIGUOUS, [COUNT, _query()])["run_id"]
    old.dies(lambda: old.resume(ids["running_resume"], [COUNT, _exit_here]))

    approved, asked = old.sync(MIXED, [COUNT, NAME_QUERY]), old.sync(MIXED, [COUNT, NAME_QUERY])
    ids["running_approved"], ids["cancel_requested_approved"] = approved["run_id"], asked["run_id"]
    old.executor = _ExitingExecutor()
    old.dies(lambda: old.approve(approved))
    old.executor = _ExitingExecutor(before=lambda: old.svc.cancel(run_id=asked["run_id"], identity=REQ))
    old.dies(lambda: old.approve(asked))
    old.executor = _Recording()

    def first(messages):
        ids["running_first"] = old.running_id()
        raise _ProcessExit()

    old.dies(lambda: old.sync(PAID, [_query(), first]))

    def cancelled_then_exit(messages):
        ids["cancel_requested"] = old.running_id()
        old.svc.cancel(run_id=ids["cancel_requested"], identity=REQ)
        raise _ProcessExit()

    old.dies(lambda: old.sync(PAID, [_query(), cancelled_then_exit]))
    old.store.close()
    return ids


def _rows(store, run_id) -> tuple:
    connection = store._connection
    return (
        dict(connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()),
        [tuple(row) for row in connection.execute("SELECT * FROM events WHERE run_id = ? ORDER BY event_id", (run_id,))],
        [tuple(row) for row in connection.execute("SELECT * FROM approvals WHERE run_id = ? ORDER BY approval_id", (run_id,))],
    )


def _everything(store) -> tuple:
    connection = store._connection
    return tuple(
        [tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1, 2")]
        for table in ("runs", "events", "approvals", "parallel_groups", "parallel_branches")
    )


def _tail(store, run_id, after: int) -> list[tuple]:
    return [(e["type"], e["status"], e["payload"]) for e in store.events(run_id, after_event_id=after, limit=100)]


def _start_up() -> object:
    """A new process's app lifespan over the configured state store; the app's state."""

    reset_shared_state_stores()
    with TestClient(app):
        pass
    return app.state


@pytest.fixture()
def left(env):
    ids = _left_by_an_exit(env)
    reader = StateStore(env)
    yield ids, reader, {name: _rows(reader, run_id) for name, run_id in ids.items()}
    reader.close()


def test_the_test_runs_are_left_as_an_exit_leaves_them(left) -> None:
    """The setup itself: four executing runs; the stored counts differ from the steps written."""

    ids, reader, _ = left
    statuses = {name: reader.get_run(run_id)["status"] for name, run_id in ids.items()}
    assert statuses == {
        "succeeded": "SUCCEEDED", "cancelled": "CANCELLED", "waiting_user": "WAITING_USER",
        "waiting_approval": "WAITING_APPROVAL", "running_resume": "RUNNING", "running_approved": "RUNNING",
        "cancel_requested_approved": "CANCEL_REQUESTED", "running_first": "RUNNING", "cancel_requested": "CANCEL_REQUESTED",
    }
    resumed = reader.get_run(ids["running_resume"])
    model_steps = [e for e in reader.events(ids["running_resume"]) if e["payload"].get("kind") == "model_call"]
    assert (resumed["model_call_count"], resumed["tool_call_count"], resumed["sql_exec_count"]) == (2, 2, 1)
    assert len(model_steps) == 3 and resumed["usage"]["status"] == "known"
    first = reader.get_run(ids["running_first"])
    assert first["model_call_count"] == 0 and any(e["payload"].get("kind") == "model_call" for e in reader.events(ids["running_first"]))


def test_runs_left_executing_end_at_startup_and_the_others_stay_as_they_were(left) -> None:
    ids, reader, before = left
    state = _start_up()

    assert state.parallel_recovery == {"recovered_submitted": [], "failed_uncertain": [], "reconciled_terminal": [], "executor_calls": 0}
    assert {key: set(value) for key, value in state.interrupted_runs.items()} == {
        "failed": {ids["running_resume"], ids["running_approved"], ids["running_first"]},
        "cancelled": {ids["cancel_requested_approved"], ids["cancel_requested"]},
    }
    for name in INTERRUPTED:
        run_id = ids[name]
        old_run, old_events, old_approvals = before[name]
        run, events, approvals = _rows(reader, run_id)
        failed = name.startswith("running")
        assert (run["status"], run["error_code"]) == (("FAILED", "execution_interrupted") if failed else ("CANCELLED", None)), name
        assert json.loads(run["usage_json"]) == UNKNOWN, name
        assert api_main._usage_total(reader.get_run(run_id))["status"] == "unknown", name
        # Only the status, its code, the usage and the time change: counts, envelope, answer, result, facts stay.
        changed = {key for key in run if run[key] != old_run[key]}
        assert changed <= {"status", "error_code", "usage_json", "updated_at"}, (name, changed)
        assert events[: len(old_events)] == old_events, name
        assert _tail(reader, run_id, len(old_events)) == (FAILED_TAIL if failed else CANCELLED_TAIL), name
        assert approvals == old_approvals, name
    approvals = reader._connection.execute("SELECT status FROM approvals WHERE run_id = ?", (ids["running_approved"],))
    assert [row[0] for row in approvals] == ["APPROVED"]
    for name in UNTOUCHED:
        assert _rows(reader, ids[name]) == before[name], name


def test_a_second_startup_writes_nothing(left) -> None:
    ids, reader, _ = left
    _start_up()
    after_first = _everything(reader)
    state = _start_up()
    assert _everything(reader) == after_first
    assert state.interrupted_runs == {"failed": [], "cancelled": []}


def test_after_startup_the_ended_runs_answer_from_their_end_and_nothing_executes(left, monkeypatch) -> None:
    ids, reader, _ = left
    executed: list[str] = []
    monkeypatch.setattr(FixtureQueryExecutor, "execute", lambda self, sql, **kwargs: executed.append(sql))
    reset_shared_state_stores()
    with TestClient(app) as client:
        model_calls: list = []

        class _NoModel:
            mode = "fake"

            def complete(self, *args, **kwargs):
                model_calls.append(1)
                raise AssertionError("no model call after startup")

        app.dependency_overrides[get_model_provider] = lambda: _NoModel()
        expected = {
            "running_first": ("FAILED", 404, "approval_not_found"),
            "running_resume": ("FAILED", 404, "approval_not_found"),
            "running_approved": ("FAILED", 502, "execution_interrupted"),
            "cancel_requested": ("CANCELLED", 404, "approval_not_found"),
            "cancel_requested_approved": ("CANCELLED", 200, None),
        }
        for name, (status, approval_http, approval_code) in expected.items():
            run_id = ids[name]
            events_before = reader.event_bounds(run_id)
            got = client.get(f"/runs/{run_id}", headers=auth(REQUESTER))
            assert (got.status_code, got.json()["status"], got.json()["usage_total"]["status"]) == (200, status, "unknown"), name
            resumed = client.post(f"/runs/{run_id}/resume", json={"answer": "按支付金额"}, headers=auth(REQUESTER))
            assert (resumed.status_code, resumed.json()["error"]["code"]) == (409, "invalid_run_state"), name
            approval_id = reader.get_run(run_id)["approval_id"] or "approval-none"
            approved = client.post(f"/runs/{run_id}/approval", json={"approval_id": approval_id, "decision": "approve"}, headers=auth(APPROVER))
            assert approved.status_code == approval_http, name
            if approval_code is None:
                assert approved.json()["status"] == status, name
            else:
                assert approved.json()["error"]["code"] == approval_code, name
            cancelled = client.post(f"/runs/{run_id}/cancel", json={}, headers=auth(REQUESTER))
            assert (cancelled.status_code, cancelled.json()["status"]) == (200, status), name
            assert reader.event_bounds(run_id) == events_before, name
        assert executed == [] and model_calls == []
        assert [a[0] for a in reader._connection.execute("SELECT status FROM approvals WHERE run_id = ?", (ids["running_approved"],))] == ["APPROVED"]


def test_after_startup_the_event_stream_of_an_ended_run_ends(left, server) -> None:  # noqa: F811
    ids, _, _ = left
    _start_up()
    for name in INTERRUPTED:
        frames = _Frames(server, ids[name], REQUESTER)
        try:
            seen = frames.until(lambda f: bool(f) and f[-1]["type"] == "terminal")
            assert frames.response.read1(4096) == b"", name  # closed after the terminal event
            assert [f["type"] for f in seen].count("terminal") == 1, name
        finally:
            frames.close()


def test_after_startup_the_waiting_runs_still_resume_and_approve(left) -> None:
    ids, reader, _ = left
    reset_shared_state_stores()
    with TestClient(app) as client:
        own = RunService(store=reader, executor_factory=lambda: _Recording(), mode="fake")
        resumed = own.resume_waiting_user(
            run_id=ids["waiting_user"], answer="按支付金额", identity=REQ, model=_Metered([_query(), _cite_all]),
            call_store=ModelCallStore(), executor=_Recording(),
        )
        assert resumed["status"] == "SUCCEEDED"
        pending = reader.get_run(ids["waiting_approval"])
        approved = client.post(
            f"/runs/{ids['waiting_approval']}/approval",
            json={"approval_id": pending["approval_id"], "decision": "approve"},
            headers=auth(APPROVER),
        )
        assert (approved.status_code, approved.json()["status"]) == (200, "SUCCEEDED")


def _parallel_run(store, run_id: str, *, uncertain: bool) -> None:
    """A RUNNING run with a durable parallel group, as the parallel checks persist one."""

    store.create_run(run_id=run_id, tenant_id="A", principal_id="a-requester", role="requester", question="q", mode="fake", model_call_count=3)
    store.update_run(run_id, sql_exec_count=6)
    group_id = f"group-{run_id}"
    store.create_parallel_group(group_id=group_id, run_id=run_id, plan_hash="plan", plan={"metric_ids": ["gross_fen", "paid_count"]}, metric_ids=("gross_fen", "paid_count"))
    for index, branch in enumerate(store.get_parallel_group(run_id)["branches"]):
        if uncertain and index == 1:
            store.update_parallel_branch(group_id, str(branch["branch_id"]), status="RUNNING")
            continue
        store.update_parallel_branch(group_id, str(branch["branch_id"]), status="SUCCEEDED", result={"metric_id": branch["metric_id"], "rows": [{"v": 1}]})


def test_a_run_with_a_parallel_group_is_ended_once_by_the_group_recovery(env) -> None:
    store = StateStore(env)
    _parallel_run(store, "run-parallel-committed", uncertain=False)
    _parallel_run(store, "run-parallel-uncertain", uncertain=True)
    try:
        state = _start_up()
        assert state.parallel_recovery["recovered_submitted"] == ["run-parallel-committed"]
        assert state.parallel_recovery["failed_uncertain"] == ["run-parallel-uncertain"]
        assert state.interrupted_runs == {"failed": [], "cancelled": []}
        for run_id, status, code in (("run-parallel-committed", "SUCCEEDED", None), ("run-parallel-uncertain", "FAILED", "recovery_required")):
            run = store.get_run(run_id)
            terminals = [e for e in store.events(run_id) if e["type"] == "terminal"]
            assert (run["status"], run["error_code"]) == (status, code)
            assert len(terminals) == 1 and terminals[0]["payload"]["recovered_parallel_group"] == f"group-{run_id}"
    finally:
        store.close()


def test_runs_executing_in_this_process_are_not_ended(env) -> None:
    """Gated first execution, resume and approved query of another service on its own store."""

    old = _Old(env)
    to_resume = old.sync(AMBIGUOUS, [COUNT, _query()])["run_id"]
    to_approve = old.sync(MIXED, [COUNT, NAME_QUERY])
    resume_gate, first_gate, query_gate = _Gate(_cite_all), _Gate(_cite), _GatedExecutor()
    threads = []

    def in_thread(target) -> None:
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        threads.append(thread)

    in_thread(lambda: old.resume(to_resume, [COUNT, resume_gate]))
    assert resume_gate.entered.wait(timeout=10)
    old.executor = query_gate
    in_thread(lambda: old.approve(to_approve))
    assert query_gate.entered.wait(timeout=10)
    old.executor = _Recording()
    deps = old.svc.default_dependencies()
    deps.model, deps.retriever = _Metered([_query(), first_gate]), None
    first = old.svc.start_async(identity=REQ, question=PAID, time_window=None, deps=deps)["run_id"]
    assert first_gate.entered.wait(timeout=10)
    executing = (first, to_resume, to_approve["run_id"])
    try:
        before = {run_id: _rows(old.store, run_id) for run_id in executing}
        state = _start_up()
        assert state.interrupted_runs == {"failed": [], "cancelled": []}
        assert {run_id: _rows(old.store, run_id) for run_id in executing} == before
    finally:
        for gate in (resume_gate, first_gate, query_gate):
            gate.release.set()
        for thread in threads:
            thread.join(timeout=10)
    _wait_for(lambda: all(old.store.get_run(run_id)["status"] == "SUCCEEDED" for run_id in executing))
    for run_id in executing:
        assert [e["type"] for e in old.store.events(run_id)].count("terminal") == 1
    old.store.close()


def test_a_refused_resume_or_approval_of_an_executing_run_keeps_it_registered(env) -> None:
    """Both register the run and refuse it (not waiting); its execution is still not ended."""

    from queryshield.approval.service import ApprovalConflict, ApprovalNotFound

    old = _Old(env)
    gate = _Gate(_cite)
    deps = old.svc.default_dependencies()
    deps.model, deps.retriever = _Metered([_query(), gate]), None
    run_id = old.svc.start_async(identity=REQ, question=PAID, time_window=None, deps=deps)["run_id"]
    try:
        assert gate.entered.wait(timeout=10)
        with pytest.raises(ApprovalConflict):
            old.resume(run_id, [_cite])
        with pytest.raises(ApprovalNotFound):
            old.svc.approve(run_id=run_id, approval_id="approval-none", identity=APPROVER_A, decision="approve")
        assert _start_up().interrupted_runs == {"failed": [], "cancelled": []}
        assert old.store.get_run(run_id)["status"] == "RUNNING"
    finally:
        gate.release.set()
    _wait_for(lambda: old.store.get_run(run_id)["status"] == "SUCCEEDED")
    old.store.close()


def test_a_run_being_created_is_not_ended(env, monkeypatch) -> None:
    """Startup between the insert of a new run (RUNNING) and its execution leaves it executing."""

    old = _Old(env)
    seen: list[dict] = []
    create_run = old.store.create_run

    def created_then_startup(**fields):
        run = create_run(**fields)
        seen.append(shared_run_service().end_interrupted_runs())
        return run

    monkeypatch.setattr(old.store, "create_run", created_then_startup)
    run = old.sync(PAID, [_query(), _cite])
    assert seen == [{"failed": [], "cancelled": []}]
    assert run["status"] == "SUCCEEDED"
    assert [e["type"] for e in old.store.events(run["run_id"])].count("terminal") == 1
    old.store.close()


def test_a_run_whose_creation_failed_after_its_insert_is_ended_at_the_next_startup(env, monkeypatch) -> None:
    """The row is stored RUNNING, then its first event fails: nothing executes it, so it is not left registered."""

    from queryshield.db.state_store import StateStoreError

    old = _Old(env)
    append_event = old.store.append_event

    def accepted_fails(run_id, event_type, status, **kwargs):
        if event_type == "accepted":
            raise StateStoreError("disk full")
        return append_event(run_id, event_type, status, **kwargs)

    monkeypatch.setattr(old.store, "append_event", accepted_fails)
    with pytest.raises(StateStoreError):
        old.sync(PAID, [_query(), _cite])
    (run_id,) = old.store.run_ids_with_status("RUNNING")
    assert shared_run_service().end_interrupted_runs() == {"failed": [run_id], "cancelled": []}
    old.store.close()


def _dead_running_run(path) -> str:
    old = _Old(path)
    ids: list[str] = []

    def exit_after_one_step(messages):
        ids.append(old.running_id())
        raise _ProcessExit()

    old.dies(lambda: old.sync(PAID, [_query(), exit_after_one_step]))
    old.store.close()
    return ids[0]


def _between_read_and_write(monkeypatch, service, action) -> None:
    """Run ``action`` once, after startup read a run and before its first transition."""

    transition = service.store.transition_run
    done: list[bool] = []

    def hooked(run_id, from_status, to_status, **kwargs):
        if not done and to_status in {"FAILED", "CANCELLED"}:
            done.append(True)
            action(run_id)
        return transition(run_id, from_status, to_status, **kwargs)

    monkeypatch.setattr(service.store, "transition_run", hooked)


def test_a_cancel_request_between_the_read_and_the_write_ends_the_run_cancelled(env, monkeypatch) -> None:
    run_id = _dead_running_run(env)
    reset_shared_state_stores()
    service = shared_run_service()
    _between_read_and_write(monkeypatch, service, lambda run_id: service.cancel(run_id=run_id, identity=REQ))
    assert service.end_interrupted_runs() == {"failed": [], "cancelled": [run_id]}
    run = service.store.get_run(run_id)
    tail = [(e["type"], e["status"]) for e in service.store.events(run_id)][-3:]
    assert run["status"] == "CANCELLED"
    assert tail == [("step_finished", "CANCEL_REQUESTED"), ("step_finished", "CANCELLED"), ("terminal", "CANCELLED")]


def test_a_run_ended_by_another_writer_between_the_read_and_the_write_is_left_as_it_is(env, monkeypatch) -> None:
    run_id = _dead_running_run(env)
    reset_shared_state_stores()
    service = shared_run_service()
    ended = {"event_type": "terminal", "status": "SUCCEEDED"}
    _between_read_and_write(monkeypatch, service, lambda run_id: StateStore.transition_run(service.store, run_id, "RUNNING", "SUCCEEDED", events=[ended]))
    assert service.end_interrupted_runs() == {"failed": [], "cancelled": []}
    assert service.store.get_run(run_id)["status"] == "SUCCEEDED"
    assert [(e["type"], e["status"]) for e in service.store.events(run_id) if e["type"] == "terminal"] == [("terminal", "SUCCEEDED")]


def test_startup_overlapping_an_execution_that_starts_or_unregisters_neither_blocks_nor_ends_it(env, monkeypatch) -> None:
    """Startup holds the registry while a resume starts; then a finished execution waits before it unregisters."""

    old = _Old(env)
    to_resume = old.sync(AMBIGUOUS, [COUNT, _query()])["run_id"]
    reset_shared_state_stores()
    service = shared_run_service()
    scanning, finish_scan = threading.Event(), threading.Event()
    run_ids_with_status = service.store.run_ids_with_status

    def held_scan(*statuses):
        scanning.set()
        assert finish_scan.wait(timeout=10), "the scan was not released"
        return run_ids_with_status(*statuses)

    monkeypatch.setattr(service.store, "run_ids_with_status", held_scan)
    summaries: list[dict] = []
    startup = threading.Thread(target=lambda: summaries.append(service.end_interrupted_runs()), daemon=True)
    startup.start()
    assert scanning.wait(timeout=10)
    # A resume starts while the scan holds the registry: it waits to register, then runs.
    results: list[dict] = []
    resume = threading.Thread(target=lambda: results.append(old.resume(to_resume, [_query(), _cite_all])), daemon=True)
    resume.start()
    resume.join(timeout=0.2)
    assert old.store.get_run(to_resume)["status"] == "WAITING_USER"
    finish_scan.set()
    startup.join(timeout=10)
    resume.join(timeout=10)
    assert not startup.is_alive() and not resume.is_alive()
    assert summaries == [{"failed": [], "cancelled": []}]
    assert results[0]["status"] == "SUCCEEDED"

    # A first execution has made its final write and waits before it unregisters; startup does not wait for it.
    monkeypatch.setattr(service.store, "run_ids_with_status", run_ids_with_status)
    written, unregister = threading.Event(), threading.Event()
    transition = old.store.transition_run

    def final_write_then_wait(run_id, from_status, to_status, **kwargs):
        moved = transition(run_id, from_status, to_status, **kwargs)
        if to_status == "SUCCEEDED":
            written.set()
            assert unregister.wait(timeout=10), "the execution was not released"
        return moved

    monkeypatch.setattr(old.store, "transition_run", final_write_then_wait)
    finished: list[dict] = []
    first = threading.Thread(target=lambda: finished.append(old.sync(PAID, [_query(), _cite])), daemon=True)
    first.start()
    try:
        assert written.wait(timeout=10)
        overlapping = threading.Thread(target=lambda: summaries.append(service.end_interrupted_runs()), daemon=True)
        overlapping.start()
        overlapping.join(timeout=10)
        assert not overlapping.is_alive() and summaries[-1] == {"failed": [], "cancelled": []}
    finally:
        unregister.set()
        first.join(timeout=10)
    assert finished[0]["status"] == "SUCCEEDED"
    assert [e["type"] for e in old.store.events(finished[0]["run_id"])].count("terminal") == 1
    old.store.close()


def test_runs_are_listed_by_status_in_creation_order(tmp_path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    try:
        for run_id, status in (("run-b", "RUNNING"), ("run-a", "WAITING_USER"), ("run-c", "CANCEL_REQUESTED"), ("run-d", "RUNNING")):
            store.create_run(run_id=run_id, tenant_id="A", principal_id="p", role="requester", question="q", mode="fake")
            store.update_run(run_id, status=status)
        assert store.run_ids_with_status("RUNNING", "CANCEL_REQUESTED") == ("run-b", "run-c", "run-d")
        assert store.run_ids_with_status("WAITING_USER") == ("run-a",)
        assert store.run_ids_with_status("SUCCEEDED") == ()
    finally:
        store.close()


def test_the_interrupted_code_maps_to_the_default_failure_status() -> None:
    from queryshield.agent.runtime import FAILED_ERROR_HTTP, http_status_for_run

    assert service_module.INTERRUPTED_CODE == "execution_interrupted"
    assert "execution_interrupted" not in FAILED_ERROR_HTTP
    assert http_status_for_run("FAILED", "execution_interrupted") == 502
