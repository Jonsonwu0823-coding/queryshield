"""Every chat and embedding request carries a W3C ``traceparent``: one trace per run, one span per HTTP attempt.

A call inside a run sends the trace-id derived from its run id (the first 32 hex digits of the
SHA-256 of the run id's UTF-8 bytes), so the run's chat calls, query embeddings, sub-agents and
calls after a resume in a new process all share it; a call outside a run gets a random trace of
its own, which its retry keeps.  Every attempt gets a new parent-id, recorded as the attempt's
``span_id``.  The run's public responses carry the same trace-id as ``trace_id``.

The expected trace-id is computed here from the rule, not with the product's function.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from multi_agent_support import COMPOSITE
from test_model_call_retry import ADAPTERS, CALL_ID, RETRIED_FIRSTS, RUN_ID, _send
from test_model_call_retry_runs import _b0_over
from test_run_id_header import REQUESTER, _ask, _Upstream
from queryshield.api.main import app, get_model_provider, get_retriever_source
from queryshield.approval.service import reset_shared_state_stores, shared_state_store
from queryshield.catalog import load_default_catalog
from queryshield.knowledge.index import build_embedding_index
from queryshield.knowledge.retrieval import HybridRetriever
from queryshield.knowledge.runtime import product_knowledge

TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-01$")
PAID_TOTAL = "2026年9月已支付订单总额是多少？"
KNOWLEDGE = "退款后净额是怎么算的？"
# The adapter passes its timeout to the injected client; TestClient ignores it and warns.
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")


def _expected_trace(run_id: str) -> str:
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:32]


def _trace_and_parent(request: httpx.Request) -> tuple[str, str]:
    """The request's trace-id and parent-id; the header must match the W3C version-00 form exactly."""

    assert "tracestate" not in request.headers
    match = TRACEPARENT.fullmatch(request.headers["traceparent"])
    assert match, request.headers["traceparent"]
    return match.group(1), match.group(2)


def _traces(requests: list[httpx.Request]) -> list[str]:
    return [_trace_and_parent(request)[0] for request in requests]


def _parents(requests: list[httpx.Request]) -> list[str]:
    return [_trace_and_parent(request)[1] for request in requests]


@contextmanager
def _serve(tmp_path, monkeypatch, upstream: _Upstream, profile: str | None = None):
    """The product app over the state file in ``tmp_path``, with new adapters and a new retriever each time."""

    monkeypatch.setenv("QUERYSHIELD_STATE_STORE_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    if profile is None:
        monkeypatch.delenv("QUERYSHIELD_AGENT_PROFILE", raising=False)
    else:
        monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", profile)
    monkeypatch.delenv("QUERYSHIELD_MODEL_PROTOCOL", raising=False)
    monkeypatch.delenv("QUERYSHIELD_RETRIEVAL", raising=False)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", REQUESTER)
    reset_shared_state_stores()
    embedder = upstream.embedding()
    build = build_embedding_index(product_knowledge(demo=False).snapshot, embedder, ingest_job_id="traceparent")
    retriever = HybridRetriever(catalog=load_default_catalog(), snapshot=build.snapshot, index=build.index, embedder=embedder)
    model = upstream.model()
    app.dependency_overrides[get_model_provider] = lambda: model
    app.dependency_overrides[get_retriever_source] = lambda: lambda: retriever
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        reset_shared_state_stores()


def _in_run(upstream: _Upstream) -> list[httpx.Request]:
    return [request for request in upstream.requests if "x-run-id" in request.headers]


def _outside_runs(upstream: _Upstream) -> list[httpx.Request]:
    return [request for request in upstream.requests if "x-run-id" not in request.headers]


def _get_run(client: TestClient, run_id: str) -> dict:
    return client.get(f"/runs/{run_id}", headers={"Authorization": f"Bearer {REQUESTER}"}).json()


def _model_call_events(run_id: str) -> list[dict]:
    events = shared_state_store().events(run_id, after_event_id=0, limit=1000)
    return [event["payload"] for event in events if event["type"] == "agent_step" and event["payload"].get("kind") == "model_call"]


# --- one trace per run ---------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", ["json", "native"])
def test_every_request_of_a_run_carries_the_trace_derived_from_its_run_id(tmp_path, monkeypatch, protocol) -> None:
    upstream = _Upstream()
    with _serve(tmp_path, monkeypatch, upstream) as client:
        monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", protocol)
        upstream.requests.clear()
        body = _ask(client, KNOWLEDGE)
        fetched = _get_run(client, body["run_id"])
        events = _model_call_events(body["run_id"])

    assert body["status"] == "SUCCEEDED"
    trace = _expected_trace(body["run_id"])
    chats, embeddings = upstream.sent("chat/completions"), upstream.sent("embeddings")
    assert chats and embeddings, "the run made chat calls and a query embedding"
    assert _traces(upstream.requests) == [trace] * len(upstream.requests)
    assert body["trace_id"] == fetched["trace_id"] == trace
    # A new parent-id for every request.
    assert len(set(_parents(upstream.requests))) == len(upstream.requests)
    # Each recorded attempt names the span its request carried.
    parent_of = {request.headers["x-client-request-id"]: _trace_and_parent(request)[1] for request in chats}
    assert len(events) == len(chats)
    for event in events:
        assert [attempt["span_id"] for attempt in event["attempts"]] == [parent_of[attempt["request_id"]] for attempt in event["attempts"]]


def test_the_b0_call_carries_the_trace_of_its_run(tmp_path, monkeypatch) -> None:
    upstream = _Upstream()
    with _serve(tmp_path, monkeypatch, upstream, profile="b0") as client:
        upstream.requests.clear()
        body = _ask(client, PAID_TOTAL)

    assert body["status"] == "SUCCEEDED"
    (chat,) = upstream.sent("chat/completions")
    assert _traces([chat]) == [_expected_trace(body["run_id"])] == [body["trace_id"]]


def test_every_call_of_a_delegated_run_carries_the_root_run_trace(tmp_path, monkeypatch) -> None:
    upstream = _Upstream()
    with _serve(tmp_path, monkeypatch, upstream, profile="b2") as client:
        upstream.requests.clear()
        body = _ask(client, COMPOSITE)

    assert body["status"] == "SUCCEEDED" and body["profile"] == "B2-multi-agent"
    chats = upstream.sent("chat/completions")
    assert len(chats) == body["model_call_count"] == 5  # search, delegate, two sub-agents, the answer
    assert _traces(upstream.requests) == [_expected_trace(body["run_id"])] * len(upstream.requests) == [body["trace_id"]] * len(upstream.requests)
    assert len(set(_parents(upstream.requests))) == len(upstream.requests)


def test_a_run_resumed_by_a_new_process_keeps_its_trace(tmp_path, monkeypatch) -> None:
    """The second service, runtime and adapters share only the state file with the first."""

    first = _Upstream()
    with _serve(tmp_path, monkeypatch, first) as client:
        first.requests.clear()
        waiting = _ask(client, "2026年9月销售额是多少？")
    assert waiting["status"] == "WAITING_USER" and first.requests

    second = _Upstream()
    with _serve(tmp_path, monkeypatch, second) as client:
        second.requests.clear()
        resumed = client.post(
            f"/runs/{waiting['run_id']}/resume", headers={"Authorization": f"Bearer {REQUESTER}"}, json={"answer": "按支付金额统计"}
        ).json()
        fetched = _get_run(client, waiting["run_id"])

    assert resumed["status"] == "SUCCEEDED" and second.requests
    trace = _expected_trace(waiting["run_id"])
    assert _traces(first.requests) == [trace] * len(first.requests)
    assert _traces(second.requests) == [trace] * len(second.requests)
    assert waiting["trace_id"] == resumed["trace_id"] == fetched["trace_id"] == trace


def test_two_runs_have_two_traces(tmp_path, monkeypatch) -> None:
    upstream = _Upstream()
    with _serve(tmp_path, monkeypatch, upstream) as client:
        upstream.requests.clear()
        first = _ask(client, PAID_TOTAL)
        first_calls = len(upstream.requests)
        second = _ask(client, PAID_TOTAL)

    assert first["trace_id"] != second["trace_id"]
    assert set(_traces(upstream.requests[:first_calls])) == {first["trace_id"]} == {_expected_trace(first["run_id"])}
    assert set(_traces(upstream.requests[first_calls:])) == {second["trace_id"]} == {_expected_trace(second["run_id"])}


def test_each_call_outside_a_run_has_a_trace_of_its_own(tmp_path, monkeypatch) -> None:
    """Building the index (one request per batch) and two probe calls: no retries, so one request per call."""

    upstream = _Upstream()
    with _serve(tmp_path, monkeypatch, upstream) as client:
        upstream.model().complete([{"role": "user", "content": "ping"}])
        upstream.model().complete([{"role": "user", "content": "ping"}])
        body = _ask(client, KNOWLEDGE)

    outside, inside = _outside_runs(upstream), _in_run(upstream)
    assert len(outside) >= 3 and inside
    traces = _traces(outside)
    assert len(set(traces)) == len(outside)
    assert body["trace_id"] not in traces
    assert set(_traces(inside)) == {body["trace_id"]}
    assert len(set(_parents(upstream.requests))) == len(upstream.requests)


# --- retries and attempt records -----------------------------------------------------------------


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("first, _first_record", RETRIED_FIRSTS)
@pytest.mark.parametrize("run_id", [RUN_ID, None], ids=["in_a_run", "outside_a_run"])
def test_a_retry_keeps_the_trace_and_takes_a_new_span(adapter, first, _first_record, run_id) -> None:
    ok = adapter[1]
    result, requests, _waits = _send(adapter, [first(), ok(9)], run_id=run_id)

    one, two = requests
    (trace_one, parent_one), (trace_two, parent_two) = _trace_and_parent(one), _trace_and_parent(two)
    assert trace_one == trace_two
    if run_id is not None:
        assert trace_one == _expected_trace(run_id)
        assert one.headers["idempotency-key"] == two.headers["idempotency-key"] == CALL_ID
    assert parent_one != parent_two
    assert one.headers["x-client-request-id"] != two.headers["x-client-request-id"]
    assert one.content == two.content
    assert [attempt["span_id"] for attempt in result.attempts] == [parent_one, parent_two]


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_two_calls_outside_a_run_have_two_traces(adapter) -> None:
    ok = adapter[1]
    _, (one,), _ = _send(adapter, [ok(5)], run_id=None)
    _, (two,), _ = _send(adapter, [ok(5)], run_id=None)
    assert _traces([one]) != _traces([two])


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_a_failed_attempt_records_the_span_it_sent(adapter) -> None:
    error, (request,), _ = _send(adapter, [httpx.ReadTimeout("slow")])

    assert isinstance(error, adapter[2])
    assert [attempt["span_id"] for attempt in error.record["attempts"]] == [_trace_and_parent(request)[1]]
    assert _traces([request]) == [_expected_trace(RUN_ID)]


def test_the_b0_attempts_record_the_spans_they_sent() -> None:
    record, requests, _waits, _executor = _b0_over([httpx.ConnectError("refused"), "not json"])

    assert [attempt["span_id"] for attempt in record["model_call_attempts"]] == _parents(requests)
    assert set(_traces(requests)) == {_expected_trace("run-b0-failed")}
