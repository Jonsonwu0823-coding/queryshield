"""Agent steps reach the event table as each graph node completes, and a run that raises keeps them.

The gateway-side checks run the product HTTP path with the Real adapters pointed at the fake
upstream (as in ``test_run_usage_total``).  The fake upstream's client keeps the ``X-Run-Id``
and the usage of every chat response it sent, so the expected counts and totals come from
what the upstream saw, not from the run the product stored.
"""

from __future__ import annotations

from dataclasses import replace
import http.client
import json
import threading
import time

import httpx
import pytest
import uvicorn

from queryshield.agent import graph as graph_module
from queryshield.agent import runtime as runtime_module
from queryshield.api.main import app, get_model_provider
from queryshield.approval.service import shared_run_service
from queryshield.providers.fake_model import FakeModel
from test_clarification import _query
from test_http_queries import APPROVER, REQUESTER, auth, env  # noqa: F401  (env is a fixture)
from test_run_usage_total import (  # noqa: F401  (embedder, model and service are fixtures)
    PAID_TOTAL,
    SEARCH,
    TOKENS,
    UNKNOWN,
    _Upstream,
    _ask,
    _embedding_error,
    _known,
    _resume,
    _status_total,
    embedder,
    model,
    service,
)


class _RunIdUpstream(_Upstream):
    """The fake upstream; also keeps the ``X-Run-Id`` of every chat request beside its usage."""

    def __init__(self) -> None:
        super().__init__()
        self.chat: list[tuple[str | None, dict[str, int]]] = []

    def _keep(self, response: httpx.Response) -> None:
        super()._keep(response)
        if response.request.url.path == "/v1/chat/completions":
            self.chat.append((response.request.headers.get("X-Run-Id"), response.json()["usage"]))

    def usage_of(self, run_id: str) -> list[dict[str, int]]:
        return [usage for sent_for, usage in self.chat if sent_for == run_id]


@pytest.fixture()
def upstream() -> _RunIdUpstream:
    return _RunIdUpstream()


def _stored_kinds(run_id: str) -> list[str]:
    return [
        event["payload"]["kind"]
        for event in shared_run_service().store.events(run_id, after_event_id=0, limit=1000)
        if event["type"] == "agent_step"
    ]


# --- a run that raises keeps the calls it made before the error --------------------------


def test_a_first_execution_that_raises_keeps_its_calls_before_the_error(service, upstream, model, embedder) -> None:
    model.search = True
    embedder.error = _embedding_error()

    body = _ask(service, PAID_TOTAL)

    assert (body["status"], body["error"]["code"]) == ("FAILED", "upstream_http_error")
    sent = upstream.usage_of(body["run_id"])
    assert len(sent) == 1, "one chat call reached the upstream before the error"
    assert (body["model_call_count"], body["tool_call_count"]) == (len(sent), 0)
    assert _stored_kinds(body["run_id"]) == ["model_call"]
    assert body["usage_total"] == _known(sent)
    assert _status_total(service, body["run_id"]) == _known(sent)


class _Script:
    """The upstream model with some calls steered: a call's entry replaces its content, or raises after the
    request reached the upstream (``RAISE``); calls past the list keep the upstream's content."""

    RAISE = object()

    def __init__(self, inner, steps=()) -> None:
        self.inner = inner
        self.steps = list(steps)
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def complete(self, messages, **kwargs):
        result = self.inner.complete(messages, **kwargs)
        step = self.steps[self.calls] if self.calls < len(self.steps) else None
        self.calls += 1
        if step is _Script.RAISE:
            raise RuntimeError("the connection dropped after the reply")
        return result if step is None else replace(result, content=step)


DESCRIBE = json.dumps({"type": "tool_call", "name": "describe_tables", "arguments": {"tables": ["orders"]}})
AMBIGUOUS = "2026年9月销售额是多少？"
NOT_RUN = {"status": "not_run", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@pytest.fixture()
def script(upstream, model) -> _Script:
    """Steered calls in place of the metered upstream model (same fixture name, so ``service`` serves this one)."""

    steered = _Script(upstream.model())
    app.dependency_overrides[get_model_provider] = lambda: steered
    return steered


def test_a_resume_that_raises_counts_the_calls_and_tools_before_and_after_the_pause(service, upstream, script, embedder) -> None:
    # The first call declares a metric the question leaves open: one tool call, then the run waits.
    script.steps = [json.dumps(_query())]
    waiting = _ask(service, AMBIGUOUS)
    assert waiting["status"] == "WAITING_USER"
    run_id = waiting["run_id"]
    paused_calls, paused_tools = waiting["model_call_count"], waiting["tool_call_count"]
    assert paused_calls >= 1 and paused_tools >= 1
    script.steps = [None] * script.calls + [DESCRIBE, SEARCH]
    embedder.error = _embedding_error()

    resumed = _resume(service, run_id)

    assert (resumed["status"], resumed["error"]["code"]) == ("FAILED", "upstream_http_error")
    sent = upstream.usage_of(run_id)
    assert len(sent) == paused_calls + 2
    # The search raised inside its tool node: its call is not a stored step, the describe call is.
    assert (resumed["model_call_count"], resumed["tool_call_count"]) == (len(sent), paused_tools + 1)
    assert _stored_kinds(run_id).count("model_call") == len(sent)
    assert resumed["usage_total"] == _known(sent)
    assert _known(sent[:paused_calls])["total_tokens"] and _known(sent[paused_calls:])["total_tokens"]


def test_a_call_that_raises_after_reaching_the_upstream_leaves_the_total_unknown(service, upstream, script) -> None:
    script.steps = [_Script.RAISE]

    body = _ask(service, PAID_TOTAL)

    assert (body["status"], body["error"]["code"]) == ("FAILED", "execution_failed")
    assert len(upstream.usage_of(body["run_id"])) == 1, "the request reached the upstream"
    assert (body["model_call_count"], body["tool_call_count"]) == (0, 0)
    assert _stored_kinds(body["run_id"]) == []
    assert body["usage_total"] == UNKNOWN
    assert shared_run_service().store.get_run(body["run_id"])["usage"]["status"] == "unknown"


def test_a_resume_whose_call_raises_after_reaching_the_upstream_leaves_the_total_unknown(service, upstream, script) -> None:
    waiting = _ask(service, AMBIGUOUS)
    paused_calls = waiting["model_call_count"]
    script.steps = [None] * script.calls + [_Script.RAISE]

    resumed = _resume(service, waiting["run_id"])

    assert resumed["status"] == "FAILED"
    assert len(upstream.usage_of(waiting["run_id"])) == paused_calls + 1
    assert resumed["model_call_count"] == paused_calls
    assert resumed["usage_total"] == UNKNOWN


def test_a_node_that_raises_after_its_call_returned_stores_nothing_of_it(service, upstream, monkeypatch) -> None:
    """The call's event was built, then the node raised: the event is not stored, and the total is unknown."""

    def broken_parser(*args, **kwargs):
        raise RuntimeError("parser bug")

    monkeypatch.setattr(graph_module, "parse_query_proposal", broken_parser)

    body = _ask(service, PAID_TOTAL)

    assert body["status"] == "FAILED"
    assert len(upstream.usage_of(body["run_id"])) == 1
    assert (body["model_call_count"], _stored_kinds(body["run_id"])) == (0, [])
    assert body["usage_total"] == UNKNOWN


@pytest.mark.parametrize("profile", ["b0", "b1"])
def test_a_run_that_raises_after_its_model_call_has_an_unknown_total(service, upstream, script, monkeypatch, profile) -> None:
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", profile)
    script.steps = [_Script.RAISE]

    body = _ask(service, PAID_TOTAL)

    assert body["status"] == "FAILED"
    assert len(upstream.usage_of(body["run_id"])) == 1
    assert body["usage_total"] == UNKNOWN


@pytest.mark.parametrize("profile, module", [("b0", runtime_module), ("b1", graph_module)])
def test_a_run_that_raises_before_any_model_call_has_no_calls(service, upstream, monkeypatch, profile, module) -> None:
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", profile)

    def broken_context(*args, **kwargs):
        raise RuntimeError("context bug")

    monkeypatch.setattr(module, "build_context", broken_context)

    body = _ask(service, PAID_TOTAL)

    assert body["status"] == "FAILED"
    assert upstream.chat == []
    assert (body["model_call_count"], body["tool_call_count"]) == (0, 0)
    assert body["usage_total"] == NOT_RUN


def test_the_sync_list_of_a_run_that_raised_adds_up_to_its_total(service, upstream, model, embedder) -> None:
    model.search = True
    embedder.error = _embedding_error()

    body = _ask(service, PAID_TOTAL)

    assert body["status"] == "FAILED" and len(body["usage"]) == 1
    assert body["usage_total"] == {"status": "known", **{name: sum(entry[name] for entry in body["usage"]) for name in TOKENS}}
    assert body["usage"][0]["provider_request_id"]


# --- SSE delivers the steps while the run executes ----------------------------------------------------------


class _HeldFake(FakeModel):
    """The Fake model; its second call waits until released (with a timeout)."""

    def __init__(self) -> None:
        self.calls = 0
        self.second = threading.Event()
        self.release = threading.Event()

    def complete(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 2:
            self.second.set()
            assert self.release.wait(timeout=15), "the held model call was not released"
        return super().complete(messages, **kwargs)


@pytest.fixture()
def server(env):
    """The app on a real local socket (the TestClient returns a stream only once it ends)."""

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not instance.started:
        assert time.monotonic() < deadline and thread.is_alive(), "the local server did not start"
        time.sleep(0.02)
    port = instance.servers[0].sockets[0].getsockname()[1]
    try:
        yield port
    finally:
        instance.should_exit = True
        thread.join(timeout=10)


def _request(port: int, method: str, path: str, token: str, body: dict | None = None, **headers) -> tuple[int, dict]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        connection.request(method, path, body=payload, headers={**auth(token), "Content-Type": "application/json", **headers})
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"{}")
    finally:
        connection.close()


class _Frames:
    """An open SSE connection read frame by frame; every read is bounded by a deadline."""

    def __init__(self, port: int, run_id: str, token: str) -> None:
        self.connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        self.connection.request("GET", f"/runs/{run_id}/events", headers={**auth(token), "Accept": "text/event-stream"})
        self.response = self.connection.getresponse()
        assert self.response.status == 200
        self.buffer = b""
        self.frames: list[dict] = []

    def until(self, predicate, *, timeout: float = 10.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        while not predicate(self.frames):
            assert time.monotonic() < deadline, f"frames so far: {[f['type'] for f in self.frames]}"
            chunk = self.response.read1(4096)
            assert chunk, "the stream closed early"
            self.buffer += chunk
            while b"\n\n" in self.buffer:
                block, self.buffer = self.buffer.split(b"\n\n", 1)
                for line in block.decode("utf-8").splitlines():
                    if line.startswith("data: "):
                        self.frames.append(json.loads(line[len("data: "):]))
        return self.frames

    def close(self) -> None:
        self.connection.close()


def _step_kinds(frames: list[dict]) -> list[str]:
    return [frame["step"]["kind"] for frame in frames if frame["type"] == "agent_step"]


def test_sse_delivers_steps_before_the_run_ends(server) -> None:
    held = _HeldFake()
    app.dependency_overrides[get_model_provider] = lambda: held
    stream = None
    try:
        code, accepted = _request(server, "POST", "/queries", REQUESTER, {"question": "2026年9月已支付订单总额"}, Prefer="respond-async")
        assert code == 202
        run_id = accepted["run_id"]
        stream = _Frames(server, run_id, REQUESTER)
        frames = stream.until(lambda frames: {"model_call", "tool_call"} <= set(_step_kinds(frames)))
        # The second model call is still held: the run has not committed anything.
        assert held.second.wait(timeout=10) and not held.release.is_set()
        assert _request(server, "GET", f"/runs/{run_id}", REQUESTER)[1]["status"] == "RUNNING"
        assert "terminal" not in [frame["type"] for frame in frames]
        assert _step_kinds(frames) == ["model_call", "tool_call"]
        held.release.set()
        frames = stream.until(lambda frames: frames and frames[-1]["type"] == "terminal")
        kinds = _step_kinds(frames)
        assert kinds[:3] == ["model_call", "tool_call", "model_call"] and kinds[-1] == "answer"
        assert frames[-1]["status"] == "SUCCEEDED"
    finally:
        held.release.set()
        if stream is not None:
            stream.close()


def test_an_approver_sees_step_summaries_of_a_waiting_run(server) -> None:
    code, pending = _request(server, "POST", "/queries", REQUESTER, {"question": "查询客户姓名"})
    assert code == 202 and pending["status"] == "WAITING_APPROVAL"
    stream = _Frames(server, pending["run_id"], APPROVER)
    try:
        frames = stream.until(lambda frames: frames and frames[-1]["type"] == "waiting")
    finally:
        stream.close()
    steps = [frame["step"] for frame in frames if frame["type"] == "agent_step"]
    assert {step["kind"] for step in steps} == {"model_call", "tool_call"}
    assert (steps[-1]["status"], steps[-1]["tool_name"]) == ("approval_required", "query_readonly")
    text = json.dumps(frames, ensure_ascii=False)
    assert "SELECT" not in text and "name" not in text.replace("tool_name", "") and "查询客户姓名" not in text
