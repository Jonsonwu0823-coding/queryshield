"""Each agent event is stored once, as its graph node completes, on every path that runs or ends a run.

Service-level: scripted models, the fixture executor and a temporary state store.  The
expected events come from the agent's own result (or checkpoint), not from the store.
"""

from __future__ import annotations

from dataclasses import replace
import json
import threading
import time

import pytest

from queryshield.agent import BoundedAgent, ModelCallStore, ParallelScheduler
from queryshield.agent.parallel import BranchExecution
from queryshield.api import main as api_main
from queryshield.approval import service as service_module
from queryshield.approval.service import _StepWriter
from queryshield.db.state_store import StateStore
from queryshield.providers.contracts import ModelProviderError, ModelUsage

from test_approval_api_pins import ACCEPTED, NAME_QUERY, NAME_SQL, STARTED, _approve, _pending, _seq
from test_clarification import GROSS_SQL, REQUESTER, _cite, _query, _resume, _Scripted, _start, service  # noqa: F401
from test_parallel import _ParallelModel, _context, _plan

PAID = "2026年9月支付金额是多少"
AMBIGUOUS = "2026年9月销售额是多少？"
UNKNOWN_TOTAL = {"status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


def _json_shape(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def _stored_steps(svc, run_id: str) -> list[dict]:
    return [e["payload"] for e in svc.store.events(run_id, after_event_id=0, limit=1000) if e["type"] == "agent_step"]


def _assert_stored_once(stored: list[dict], events) -> None:
    """The stored steps are exactly the agent's events, each once, in order."""

    assert stored == _json_shape([dict(event) for event in events])
    if all("sequence" in event for event in stored):  # the refusal before the graph has none
        assert [event["sequence"] for event in stored] == list(range(1, len(stored) + 1))


def _spy_payloads(monkeypatch) -> list[dict]:
    """The payload of every profile run the service makes."""

    payloads: list[dict] = []
    run_profile = service_module.run_profile

    def spying(*args, **kwargs):
        profile_run = run_profile(*args, **kwargs)
        payloads.append(profile_run.payload)
        return profile_run

    monkeypatch.setattr(service_module, "run_profile", spying)
    return payloads


class _Metered(_Scripted):
    """A scripted model whose calls report usage, each call its own counts."""

    def complete(self, messages, **kwargs):
        result = super().complete(messages, **kwargs)
        n = len(self.messages)
        return replace(result, usage=ModelUsage(prompt_tokens=100 * n + 7, completion_tokens=n + 3, total_tokens=101 * n + 10), usage_status="known")


def _waiting(service):
    run = _start(service, AMBIGUOUS, [_query()])
    assert run["status"] == "WAITING_USER"
    return run


# --- each event once, on every path ---------------------------------------------------------


def test_a_first_execution_stores_each_event_once(service, monkeypatch) -> None:
    svc, _ = service
    payloads = _spy_payloads(monkeypatch)
    run = _start(service, PAID, [_query(), _cite])
    assert run["status"] == "SUCCEEDED"
    _assert_stored_once(_stored_steps(svc, run["run_id"]), payloads[0]["events"])


def test_a_resumed_run_stores_only_the_events_after_the_pause(service) -> None:
    svc, _ = service
    run = _waiting(service)
    paused = _stored_steps(svc, run["run_id"])
    assert [step["kind"] for step in paused] == ["model_call", "tool_call"]

    done = _resume(service, run, "按支付金额", [_query(), _cite])

    assert done["status"] == "SUCCEEDED"
    stored = _stored_steps(svc, run["run_id"])
    _assert_stored_once(stored, done["checkpoint"]["last_agent_result"]["events"])
    assert stored[: len(paused)] == paused and len(stored) > len(paused)
    types = [event["type"] for event in svc.store.events(run["run_id"], after_event_id=0, limit=1000)]
    assert types.index("waiting") == types.index("agent_step") + len(paused)


def test_a_resume_that_keeps_waiting_stores_no_step(service) -> None:
    svc, _ = service
    run = _waiting(service)
    paused = _stored_steps(svc, run["run_id"])

    still = _resume(service, run, "2026年9月", [])

    assert still["status"] == "WAITING_USER"
    assert _stored_steps(svc, run["run_id"]) == paused
    assert [event["type"] for event in svc.store.events(run["run_id"], after_event_id=0, limit=1000)][-2:] == ["waiting", "waiting"]


def test_a_run_cancelled_while_running_stores_the_later_steps_after_the_request(service) -> None:
    svc, _ = service

    def cancelling(messages):
        svc.cancel(run_id=next(iter(svc._active)), identity=REQUESTER)
        return _cite(messages)

    run = _start(service, PAID, [_query(), cancelling])

    assert run["status"] == "CANCELLED"
    assert _seq(svc.store, run) == [ACCEPTED, STARTED] + [
        ("agent_step", "RUNNING", False, "model_call"),
        ("agent_step", "RUNNING", True, "tool_call"),
        ("step_finished", "CANCEL_REQUESTED", False, {"cancel_requested": True}),
        ("agent_step", "RUNNING", False, "model_call"),
        ("agent_step", "RUNNING", False, "answer"),
        ("step_finished", "CANCELLED", False, {"cancelled_before_commit": True}),
        ("terminal", "CANCELLED", False, {}),
    ]


def test_an_approval_stores_no_agent_step(service) -> None:
    svc, _ = service
    run = _pending(service)
    paused = _stored_steps(svc, run["run_id"])
    assert [step["kind"] for step in paused] == ["model_call", "tool_call"]

    done = _approve(service, run)

    assert done["status"] == "SUCCEEDED"
    assert _stored_steps(svc, run["run_id"]) == paused
    assert [event["type"] for event in svc.store.events(run["run_id"], after_event_id=0, limit=1000)][-1] == "terminal"


def test_a_refusal_without_the_graph_stores_its_event_before_the_terminal_status(service, monkeypatch) -> None:
    svc, _ = service
    payloads = _spy_payloads(monkeypatch)

    run = _start(service, "tenant-B 的支付金额", [])

    assert run["status"] == "DENIED"
    _assert_stored_once(_stored_steps(svc, run["run_id"]), payloads[0]["events"])
    assert [(event["type"], event["status"]) for event in svc.store.events(run["run_id"], after_event_id=0, limit=1000)] == [
        ("accepted", "RUNNING"), ("step_started", "RUNNING"), ("agent_step", "RUNNING"), ("terminal", "DENIED"),
    ]


def test_a_parallel_step_is_stored_once(tmp_path) -> None:
    context = _context("run-parallel-writes")
    store = StateStore(tmp_path / "state.sqlite3")
    store.create_run(
        run_id=context.run_id, tenant_id=context.tenant_id, principal_id=context.principal_id,
        role=context.role, question="订单总额和净额", mode="fake",
    )
    writer = _StepWriter(store, context.run_id)

    def runner(context, metric_id, branch_id):
        return BranchExecution(metric_id, f"result-{metric_id}", ({metric_id: 1},), "2026-09-21T00:00:00Z")

    agent = BoundedAgent(
        _ParallelModel(), call_store=ModelCallStore(), parallel_scheduler=ParallelScheduler(runner), on_step=writer
    )
    result = agent.run(context, "订单总额和净额", parallel_plan=_plan(context))

    assert result.status == "succeeded" and result.tool_call_count == 3
    stored = [e["payload"] for e in store.events(context.run_id, after_event_id=0, limit=1000) if e["type"] == "agent_step"]
    _assert_stored_once(stored, result.events)
    assert [step["kind"] for step in stored].count("parallel_group") == 1
    assert writer.state["tool_call_count"] == 3 and writer.model_calls_written == 2
    store.close()


# --- a run that raises: what was written stays, and its counts are those of the last written step ----------


def test_a_run_that_raises_after_two_steps_keeps_them_and_their_counts(service, monkeypatch) -> None:
    svc, _ = service

    def raises(messages):
        raise RuntimeError("connection reset")

    run = _start(service, PAID, [_query(), raises])

    assert (run["status"], run["error_code"]) == ("FAILED", "execution_failed")
    stored = _stored_steps(svc, run["run_id"])
    assert [step["kind"] for step in stored] == ["model_call", "tool_call"]
    assert [step["sequence"] for step in stored] == [1, 2]
    assert (run["model_call_count"], run["tool_call_count"], run["sql_exec_count"]) == (1, 1, 1)
    # The second call started and stored no event: the total is not known.
    assert run["usage"] == UNKNOWN_TOTAL
    assert svc.store.events(run["run_id"], after_event_id=0, limit=1000)[-1]["type"] == "terminal"


def test_a_failed_step_write_ends_the_run_with_the_counts_of_the_last_written_step(service, monkeypatch) -> None:
    svc, _ = service
    append_event = svc.store.append_event
    failures = []

    def failing_second_step(run_id, event_type, status, **kwargs):
        if event_type == "agent_step" and kwargs.get("payload", {}).get("kind") == "tool_call" and not failures:
            failures.append(event_type)
            raise RuntimeError("disk full")
        return append_event(run_id, event_type, status, **kwargs)

    monkeypatch.setattr(svc.store, "append_event", failing_second_step)
    run = _start(service, PAID, [_query(), _cite])

    assert failures and (run["status"], run["error_code"]) == ("FAILED", "execution_failed")
    assert [step["kind"] for step in _stored_steps(svc, run["run_id"])] == ["model_call"]
    # The tool ran, but its step was not stored: the counts are the model step's.
    assert (run["model_call_count"], run["tool_call_count"], run["sql_exec_count"]) == (1, 0, 1)
    assert svc.store.events(run["run_id"], after_event_id=0, limit=1000)[-1]["type"] == "terminal"


def test_a_model_failure_is_a_stored_step(service) -> None:
    svc, _ = service

    def unavailable(messages):
        raise ModelProviderError("upstream_timeout", {"status": "failed", "error_code": "upstream_timeout"})

    run = _start(service, PAID, [unavailable])

    assert run["status"] == "FAILED" and run["model_call_count"] == 1
    assert [(step["kind"], step["status"]) for step in _stored_steps(svc, run["run_id"])] == [("model_call", "failed")]


# --- written while the run executes ------------------------------------------------------------------------


class _Gate:
    """Holds one scripted step until released; every wait has a timeout."""

    def __init__(self, step) -> None:
        self.step = step
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, messages):
        self.entered.set()
        assert self.release.wait(timeout=10), "the gated model step was not released"
        return self.step(messages) if callable(self.step) else self.step


def _wait_for(predicate, *, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("condition not reached in time")


def test_steps_are_stored_while_the_run_is_still_executing(service) -> None:
    svc, _ = service
    gate = _Gate(_cite)
    deps = svc.default_dependencies()
    deps.model, deps.retriever = _Metered([_query(), gate]), None
    try:
        run_id = svc.start_async(identity=REQUESTER, question=PAID, time_window=None, deps=deps)["run_id"]
        assert gate.entered.wait(timeout=10)
        running = svc.store.get_run(run_id)
        assert running["status"] == "RUNNING"
        assert [step["kind"] for step in _stored_steps(svc, run_id)] == ["model_call", "tool_call"]
        # Still executing: the counts are not committed and the total is not shown.
        assert running["model_call_count"] == 0
        assert api_main._usage_total(running) == UNKNOWN_TOTAL
    finally:
        gate.release.set()
    done = _wait_for(lambda: (r := svc.store.get_run(run_id))["status"] == "SUCCEEDED" and r)
    assert [step["kind"] for step in _stored_steps(svc, run_id)] == ["model_call", "tool_call", "model_call", "answer"]
    assert api_main._usage_total(done)["status"] == "known"


def test_a_resume_shows_the_pause_and_its_total_until_it_commits(service) -> None:
    """A resume runs while the stored run still waits: status and total stay the pause's until the commit."""

    svc, executor = service
    run = _start(service, AMBIGUOUS, [_query()])
    paused = {"status": "known", "prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}
    svc.store.update_run(run["run_id"], usage_json=json.dumps(paused))
    gate = _Gate(_query())
    result: list[dict] = []

    def resume():
        result.append(svc.resume_waiting_user(
            run_id=run["run_id"], answer="按支付金额", identity=REQUESTER, model=_Metered([gate, _cite]),
            call_store=ModelCallStore(), executor=executor,
        ))

    worker = threading.Thread(target=resume, daemon=True)
    try:
        worker.start()
        assert gate.entered.wait(timeout=10)
        executing = svc.store.get_run(run["run_id"])
        assert executing["status"] == "WAITING_USER" and api_main._usage_total(executing) == paused
        assert [step["kind"] for step in _stored_steps(svc, run["run_id"])] == ["model_call", "tool_call"]
    finally:
        gate.release.set()
        worker.join(timeout=10)
    assert result and result[0]["status"] == "SUCCEEDED"
    # The commit stores the run's total: the scripted calls before the pause report none.
    assert api_main._usage_total(svc.store.get_run(run["run_id"])) == UNKNOWN_TOTAL


# --- what an SSE frame shows of a step ------------------------------------------------------------------------

ALLOWED_STEP_KEYS = {
    "kind", "status", "error_code", "tool_name", "elapsed_ms", "model", "usage_status",
    "prompt_tokens", "completion_tokens", "total_tokens",
}


def _frames(svc, run_id: str) -> list[dict]:
    frames = []
    for event in svc.store.events(run_id, after_event_id=0, limit=1000):
        data_line = api_main._sse_frame(event).split("\n")[2]
        frames.append({"event": event, "data": json.loads(data_line.removeprefix("data: "))})
    return frames


def test_a_frame_shows_only_the_allowed_fields_of_each_kind_of_step(service, monkeypatch) -> None:
    svc, _ = service
    texts = ["支付金额", "DELETE", "customers", NAME_SQL, GROSS_SQL]

    def unavailable(messages):
        raise ModelProviderError("upstream_timeout", {"status": "failed", "error_code": "upstream_timeout", "model": "m-1"})

    def garbage(messages):
        return {"type": "nonsense", "secret": "支付金额"}

    runs = [
        _start(service, PAID, [_query(), _cite]),  # model_call, tool_call, answer
        _pending(service),  # approval_required
        _start(service, AMBIGUOUS, [_query()]),  # clarification_required
        _start(service, PAID, [{"type": "tool_call", "name": "query_readonly", "arguments": {"sql": "DELETE FROM customers", "params": {}}}]),
        _start(service, PAID, [unavailable]),  # a failed model call
        _start(service, PAID, [garbage]),  # proposal_validation
        _start(service, "tenant-B 的支付金额", []),  # authorization
    ]
    deps = svc.default_dependencies()
    deps.model, deps.retriever = _Metered([_query(), _cite]), None
    runs.append(svc.run_sync(identity=REQUESTER, question=PAID, time_window=None, deps=deps))

    kinds = set()
    for run in runs:
        for frame in _frames(svc, run["run_id"]):
            data, event = frame["data"], frame["event"]
            if event["type"] != "agent_step":
                assert "step" not in data
                continue
            step, payload = data["step"], event["payload"]
            kinds.add(step["kind"])
            assert set(step) <= ALLOWED_STEP_KEYS, set(step) - ALLOWED_STEP_KEYS
            assert all(value is None or type(value) in (str, int) for value in step.values())
            for key in ALLOWED_STEP_KEYS & set(payload):
                assert step[key] == payload[key]
            usage = payload.get("usage") or {}
            assert {key: step[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens") if key in step} == usage
            text = json.dumps(data, ensure_ascii=False)
            assert not any(item in text for item in texts), text
    assert kinds >= {"model_call", "tool_call", "answer", "proposal_validation", "authorization"}
    metered = [f["data"]["step"] for f in _frames(svc, runs[-1]["run_id"]) if f["data"].get("step", {}).get("kind") == "model_call"]
    assert [step["prompt_tokens"] for step in metered] == [107, 207]
    assert [step["total_tokens"] for step in metered] == [111, 212]
    statuses = {(f["data"]["step"]["kind"], f["data"]["step"]["status"]) for run in runs for f in _frames(svc, run["run_id"]) if "step" in f["data"]}
    assert {("tool_call", "approval_required"), ("tool_call", "clarification_required"), ("tool_call", "failed"), ("model_call", "failed")} <= statuses
