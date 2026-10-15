"""QueryShield's own spans, exported over OTLP/HTTP when QUERYSHIELD_OTLP_TRACES_URL is set (off by default).

With export on, a run is one trace: each execution segment of the run is a root span on the
run's trace-id (the first 32 hex digits of the SHA-256 of the run id's UTF-8 bytes, computed
here from the rule), under it the agent steps, the tools, the logical model calls and one CLIENT
span per HTTP attempt.  An attempt span's id is the parent-id its request header carried and its
attempt record names, so the gateway's spans hang under it.  A call outside a run is a trace of
its own.  Spans carry ids, enums and numbers only.  With export off nothing is built.

Spans are read with the SDK's in-memory exporter, installed through the same function the
product's start uses; the OTLP tests use a small receiver started here.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
from pathlib import Path
import re
import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from multi_agent_support import COMPOSITE, SEP as SEP_WINDOW, RoleModel, b2, cite_all, delegate, run_b2  # noqa: F401  (b2 is a fixture)
from test_http_queries import APPROVER, auth
from test_model_call_retry import ADAPTERS, CALL_ID, NOT_RETRIED, RETRIED_FIRSTS, RUN_ID, TWO_FAILURES, _chat, _chat_ok, _send
from test_run_id_header import PLACEHOLDER_KEY, REQUESTER, _Upstream
from test_runs_left_executing_at_startup import PAID, _Old, _ProcessExit
from test_approval_api_pins import NAME_QUERY
from test_clarification import REQUESTER as CLARIFICATION_REQUESTER, SEPTEMBER, _Scripted, _query, _start, service  # noqa: F401  (service is a fixture)
from queryshield import tracing
from queryshield.agent import tool_execution
from queryshield.agent.tool_execution import ApprovalRequiredError, ClarificationRequiredError, ClarificationValueUnsupportedError
from queryshield.tools import ToolError
from queryshield.agent.runtime import B0_PROFILE, RuntimeConfigurationError
from queryshield.api.main import app, get_model_provider, get_retriever_source
from queryshield.approval.service import reset_shared_state_stores, shared_state_store
from queryshield.catalog import load_default_catalog
from queryshield.knowledge.index import build_embedding_index
from queryshield.knowledge.retrieval import HybridRetriever
from queryshield.knowledge.runtime import product_knowledge
from queryshield.providers.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-01$")
PAID_TOTAL = "2026年9月已支付订单总额是多少？"
KNOWLEDGE = "退款后净额是怎么算的？"
AMBIGUOUS = "2026年9月销售额是多少？"
NAMES = "查询客户姓名"
VARIABLE = "QUERYSHIELD_OTLP_TRACES_URL"
# The one list of what a span or a span event may carry.
ALLOWED = {
    "queryshield.run_id", "queryshield.segment", "queryshield.profile", "queryshield.approval_id",
    "queryshield.model_call_id", "queryshield.request_id", "queryshield.agent", "queryshield.subtask_index",
    "queryshield.endpoint", "queryshield.tool_name", "queryshield.skill_id", "queryshield.status",
    "queryshield.error_code", "queryshield.provider_error_code", "queryshield.attempt_count", "queryshield.row_count",
    "queryshield.model_call_count", "queryshield.tool_call_count", "queryshield.sql_exec_count",
    "queryshield.prompt_tokens", "queryshield.completion_tokens", "queryshield.total_tokens",
    "http.request.method", "http.response.status_code",
}
# The adapter passes its timeout to the injected client; TestClient ignores it and warns.
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")


# --- helpers ----------------------------------------------------------------------------------------


def _expected_trace(run_id: str) -> str:
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:32]


def _trace(span) -> str:
    return format(span.context.trace_id, "032x")


def _id(span) -> str:
    return format(span.context.span_id, "016x")


def _parent(span) -> str | None:
    return format(span.parent.span_id, "016x") if span.parent is not None else None


def _header(request: httpx.Request) -> tuple[str, str]:
    match = TRACEPARENT.fullmatch(request.headers["traceparent"])
    assert match, request.headers["traceparent"]
    return match.group(1), match.group(2)


def _named(spans, name: str) -> list:
    return [span for span in spans if span.name == name]


def _of_trace(spans, trace_id: str) -> list:
    return [span for span in spans if _trace(span) == trace_id]


def _root(span, by_id: dict):
    """The span's root; every parent on the way must have been exported."""

    while span.parent is not None:
        assert _parent(span) in by_id, f"{span.name} has a parent that was not exported"
        span = by_id[_parent(span)]
    return span


def _assert_one_tree_per_segment(spans, trace_id: str) -> list:
    """Every span of the run's trace leads to a ``queryshield.run`` root of that trace; the roots."""

    run_spans = _of_trace(spans, trace_id)
    by_id = {_id(span): span for span in run_spans}
    roots = {_id(_root(span, by_id)): _root(span, by_id) for span in run_spans}
    assert roots and all(root.name == "queryshield.run" for root in roots.values())
    return list(roots.values())


def _parent_name(span, spans) -> str | None:
    by_id = {_id(other): other for other in spans}
    return by_id[_parent(span)].name if _parent(span) in by_id else None


@pytest.fixture()
def exported():
    """Export on, into memory, for this test."""

    exporter = InMemorySpanExporter()
    tracing._use(SimpleSpanProcessor(exporter))
    yield exporter
    tracing.stop()


@contextmanager
def _serve(tmp_path, monkeypatch, upstream: _Upstream, *, profile: str | None = None, retriever: bool = True, traces_url: str | None = None):
    """The product app over the state file in ``tmp_path``, with new adapters (and a new retriever) each time."""

    monkeypatch.setenv("QUERYSHIELD_STATE_STORE_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    for name, value in (("QUERYSHIELD_AGENT_PROFILE", profile), (VARIABLE, traces_url)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.delenv("QUERYSHIELD_MODEL_PROTOCOL", raising=False)
    monkeypatch.delenv("QUERYSHIELD_RETRIEVAL", raising=False)
    monkeypatch.delenv("QUERYSHIELD_METADATA_TOOLS", raising=False)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", REQUESTER)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_APPROVER", APPROVER)
    reset_shared_state_stores()
    built = None
    if retriever:
        embedder = upstream.embedding()
        build = build_embedding_index(product_knowledge(demo=False).snapshot, embedder, ingest_job_id="span-export")
        built = HybridRetriever(catalog=load_default_catalog(), snapshot=build.snapshot, index=build.index, embedder=embedder)
    model = upstream.model()
    app.dependency_overrides[get_model_provider] = lambda: model
    app.dependency_overrides[get_retriever_source] = lambda: lambda: built
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        reset_shared_state_stores()


def _ask(client: TestClient, question: str, *, asynchronous: bool = False) -> dict:
    headers = auth(REQUESTER) | ({"Prefer": "respond-async"} if asynchronous else {})
    return client.post("/queries", headers=headers, json={"question": question}).json()


def _wait(client: TestClient, run_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/runs/{run_id}", headers=auth(REQUESTER)).json()
        if body["status"] not in {"RUNNING", "CANCEL_REQUESTED"}:
            return body
        time.sleep(0.02)
    raise AssertionError(f"{run_id} did not end")


def _resume(client: TestClient, run_id: str) -> dict:
    return client.post(f"/runs/{run_id}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额统计"}).json()


def _attempt_records(run_id: str) -> list[dict]:
    events = shared_state_store().events(run_id, after_event_id=0, limit=1000)
    return [
        attempt
        for event in events
        if event["type"] == "agent_step" and event["payload"].get("kind") == "model_call"
        for attempt in event["payload"].get("attempts", ())
    ]


def _assert_attempts_are_the_header_spans(spans, requests: list[httpx.Request]) -> None:
    """One CLIENT span per request, whose id is the request's parent-id, on the request's trace, under a model call."""

    attempts = {_id(span): span for span in _named(spans, "queryshield.http_attempt")}
    for request in requests:
        trace_id, parent_id = _header(request)
        assert parent_id in attempts, "the request's parent-id is not an exported attempt span"
        span = attempts[parent_id]
        assert _trace(span) == trace_id and span.kind is SpanKind.CLIENT
        assert span.attributes["queryshield.request_id"] == request.headers["x-client-request-id"]
        assert _parent_name(span, spans) == "queryshield.model_call"


# --- one trace per run --------------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", ["json", "native"])
def test_a_b1_run_is_one_trace_of_steps_tools_model_calls_and_attempts(tmp_path, monkeypatch, exported, protocol) -> None:
    upstream = _Upstream()
    with _serve(tmp_path, monkeypatch, upstream) as client:
        monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", protocol)
        upstream.requests.clear()
        body = _ask(client, KNOWLEDGE)

    assert body["status"] == "SUCCEEDED"
    spans = exported.get_finished_spans()
    trace_id = _expected_trace(body["run_id"])
    [run] = _assert_one_tree_per_segment(spans, trace_id)
    assert run.kind is SpanKind.INTERNAL and run.parent is None
    assert dict(run.attributes) == {
        "queryshield.run_id": body["run_id"], "queryshield.segment": "execute", "queryshield.profile": "B1-bounded-agent",
        "queryshield.status": "SUCCEEDED", "queryshield.model_call_count": body["model_call_count"],
        "queryshield.tool_call_count": body["tool_call_count"], "queryshield.sql_exec_count": body["sql_exec_count"],
    }
    mine = _of_trace(spans, trace_id)
    steps = [span for span in mine if span.name.startswith("queryshield.step.")]
    assert {"queryshield.step.model_decision", "queryshield.step.execute_tool", "queryshield.step.finish"} <= {s.name for s in steps}
    assert all(_parent_name(step, mine) == "queryshield.run" for step in steps)
    for tool in _named(mine, "queryshield.tool"):
        assert _parent_name(tool, mine) == "queryshield.step.execute_tool"
    calls = _named(mine, "queryshield.model_call")
    assert {call.attributes["queryshield.endpoint"] for call in calls} == {"chat", "embeddings"}
    for call in calls:
        assert call.attributes["queryshield.run_id"] == body["run_id"]
        expected_parent = "queryshield.step.model_decision" if call.attributes["queryshield.endpoint"] == "chat" else "queryshield.tool"
        assert _parent_name(call, mine) == expected_parent
    assert len(_named(mine, "queryshield.model_call")) == len(upstream.requests)  # no retries here
    # The step's own agent events, as span events.
    decisions = _named(mine, "queryshield.step.model_decision")
    assert [event.name for step in decisions for event in step.events if event.name == "model_call"]
    for step in steps:
        assert "queryshield.status" in step.attributes, step.name
    # The fake upstream reports usage, so each model call event carries its token counts.
    for event in (event for step in decisions for event in step.events if event.name == "model_call"):
        assert all(type(event.attributes.get(f"queryshield.{name}")) is int for name in ("prompt_tokens", "completion_tokens", "total_tokens"))
    stored = [e["payload"]["kind"] for e in shared_state_store().events(body["run_id"], after_event_id=0, limit=1000) if e["type"] == "agent_step"]
    assert [event.name for step in sorted(steps, key=lambda s: s.start_time) for event in step.events] == stored


@pytest.mark.parametrize("profile", ["b0", "b1", "b1s", "b2"])
def test_every_span_of_a_run_of_each_profile_is_on_the_run_trace_under_its_segment(tmp_path, monkeypatch, exported, profile) -> None:
    upstream = _Upstream()
    with _serve(tmp_path, monkeypatch, upstream, profile=profile) as client:
        upstream.requests.clear()
        body = _ask(client, COMPOSITE if profile in {"b1s", "b2"} else PAID_TOTAL)

    assert body["status"] == "SUCCEEDED"
    spans = exported.get_finished_spans()
    trace_id = _expected_trace(body["run_id"])
    [run] = _assert_one_tree_per_segment(spans, trace_id)
    mine = _of_trace(spans, trace_id)
    _assert_attempts_are_the_header_spans(mine, upstream.requests)
    assert {_header(request)[0] for request in upstream.requests} == {trace_id}
    names = {span.name for span in mine}
    if profile == "b0":
        # No graph, so no steps: the model call and the tool hang under the segment.
        assert not any(name.startswith("queryshield.step.") for name in names)
        assert {_parent_name(span, mine) for span in _named(mine, "queryshield.model_call")} == {"queryshield.run"}
        assert {_parent_name(span, mine) for span in _named(mine, "queryshield.tool")} == {"queryshield.run"}
    if profile == "b1s":
        [load] = _named(mine, "queryshield.step.load_skill")
        assert [event.attributes["queryshield.skill_id"] for event in load.events] == ["composite-question"]


def test_the_sub_agents_of_a_delegated_run_hang_under_the_step_that_runs_them(tmp_path, monkeypatch, exported) -> None:
    upstream = _Upstream()
    with _serve(tmp_path, monkeypatch, upstream, profile="b2") as client:
        upstream.requests.clear()
        body = _ask(client, COMPOSITE)

    assert body["status"] == "SUCCEEDED" and body["model_call_count"] == 5
    mine = _of_trace(exported.get_finished_spans(), _expected_trace(body["run_id"]))
    subtasks = _named(mine, "queryshield.subtask")
    assert sorted(span.attributes["queryshield.subtask_index"] for span in subtasks) == [1, 2]
    assert {_parent_name(span, mine) for span in subtasks} == {"queryshield.step.run_subtasks"}
    # The sub-agents' own events are on their steps, not again on the step that ran them.
    [run_subtasks] = _named(mine, "queryshield.step.run_subtasks")
    assert [event.name for event in run_subtasks.events] == ["subtask", "subtask"]
    by_id = {_id(span): span for span in mine}
    for subtask in subtasks:
        label = subtask.attributes["queryshield.agent"]
        steps = [span for span in mine if _parent(span) == _id(subtask)]
        assert steps and all(step.name.startswith("queryshield.step.") and step.attributes["queryshield.agent"] == label for step in steps)
        # Each sub-agent's model calls stay in its own subtree.
        calls = [span for span in _named(mine, "queryshield.model_call") if _root_below(span, by_id, subtask)]
        assert calls
    chats = [span for span in _named(mine, "queryshield.model_call") if span.attributes["queryshield.endpoint"] == "chat"]
    assert len(chats) == 5  # search, delegate, two sub-agents, the answer


def _root_below(span, by_id: dict, ancestor) -> bool:
    while span.parent is not None:
        if _parent(span) == _id(ancestor):
            return True
        span = by_id[_parent(span)]
    return False


def test_an_async_run_is_on_its_trace_from_the_worker_thread(tmp_path, monkeypatch, exported) -> None:
    upstream = _Upstream()
    with _serve(tmp_path, monkeypatch, upstream) as client:
        upstream.requests.clear()
        accepted = _ask(client, KNOWLEDGE, asynchronous=True)
        body = _wait(client, accepted["run_id"])

    assert body["status"] == "SUCCEEDED"
    spans = exported.get_finished_spans()
    trace_id = _expected_trace(body["run_id"])
    [run] = _assert_one_tree_per_segment(spans, trace_id)
    assert run.attributes["queryshield.segment"] == "execute"
    _assert_attempts_are_the_header_spans(_of_trace(spans, trace_id), upstream.requests)


# --- attempt spans are the header's spans --------------------------------------------------------------


def test_each_attempt_span_is_the_span_its_request_header_and_its_record_name(tmp_path, monkeypatch, exported) -> None:
    upstream = _Upstream()
    with _serve(tmp_path, monkeypatch, upstream) as client:
        outside = list(upstream.requests)  # building the index: calls outside a run
        upstream.requests.clear()
        body = _ask(client, KNOWLEDGE)

    spans = exported.get_finished_spans()
    _assert_attempts_are_the_header_spans(spans, upstream.requests + outside)
    assert len(_named(spans, "queryshield.http_attempt")) == len(upstream.requests) + len(outside)
    attempt_ids = {_id(span) for span in _named(spans, "queryshield.http_attempt")}
    records = _attempt_records(body["run_id"])
    assert records and all(record["span_id"] in attempt_ids for record in records)
    # Each call outside a run is a root of its own trace, the trace its request carried.
    for request in outside:
        trace_id, parent_id = _header(request)
        [attempt] = [span for span in spans if _id(span) == parent_id]
        call = next(span for span in spans if _id(span) == _parent(attempt))
        assert call.parent is None and _trace(call) == trace_id and "queryshield.run_id" not in call.attributes
    assert len({_header(request)[0] for request in outside}) == len(outside)


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("first, _first_record", RETRIED_FIRSTS)
@pytest.mark.parametrize("run_id", [RUN_ID, None], ids=["in_a_run", "outside_a_run"])
def test_a_retried_call_is_one_model_call_with_two_attempt_spans(exported, adapter, first, _first_record, run_id) -> None:
    _result, requests, _waits = _send(adapter, [first(), adapter[1](9)], run_id=run_id)

    spans = exported.get_finished_spans()
    [call] = _named(spans, "queryshield.model_call")
    attempts = _named(spans, "queryshield.http_attempt")
    assert [_id(span) for span in attempts] == [_header(request)[1] for request in requests]
    assert all(_parent(span) == _id(call) for span in attempts)
    assert {_trace(span) for span in spans} == {_header(requests[0])[0]}
    assert call.attributes["queryshield.attempt_count"] == 2 and call.attributes["queryshield.status"] == "succeeded"
    assert [span.attributes["queryshield.status"] for span in attempts] == ["failed", "succeeded"]
    assert [span.status.status_code for span in attempts] == [trace.StatusCode.ERROR, trace.StatusCode.UNSET]
    assert call.status.status_code is trace.StatusCode.UNSET
    if run_id is not None:
        assert _trace(call) == _expected_trace(run_id)


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize(
    "outcomes",
    [pytest.param(case.values[0], id=case.id) for case in TWO_FAILURES]
    + [pytest.param([case.values[0]], id=f"once_{case.id}") for case in NOT_RETRIED],
)
def test_a_call_that_fails_marks_its_span_and_every_attempt_span_failed(exported, adapter, outcomes) -> None:
    error, requests, _waits = _send(adapter, [*outcomes, adapter[1](9)])

    assert isinstance(error, adapter[2])
    spans = exported.get_finished_spans()
    [call] = _named(spans, "queryshield.model_call")
    attempts = _named(spans, "queryshield.http_attempt")
    assert len(attempts) == len(requests) == len(outcomes)
    assert call.status.status_code is trace.StatusCode.ERROR
    assert call.attributes["queryshield.status"] == "failed" and call.attributes["queryshield.error_code"] == error.code
    assert all(span.status.status_code is trace.StatusCode.ERROR for span in attempts)
    assert all(span.attributes["queryshield.status"] == "failed" for span in attempts)


def test_a_call_in_a_run_from_a_thread_without_the_context_keeps_the_header_trace(exported) -> None:
    """The gateway hangs its spans under the attempt span, so its trace must be the header's."""

    requests: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _chat_ok(5)

    def call() -> None:
        with httpx.Client(transport=httpx.MockTransport(upstream)) as client:
            _chat(client, lambda seconds: None, RUN_ID)

    with tracing.span("unrelated", trace_id="0123456789abcdef0123456789abcdef"):
        worker = threading.Thread(target=call)
        worker.start()
        worker.join()

    [request] = requests
    trace_id, parent_id = _header(request)
    assert trace_id == _expected_trace(RUN_ID)
    spans = exported.get_finished_spans()
    [call_span] = _named(spans, "queryshield.model_call")
    [attempt] = _named(spans, "queryshield.http_attempt")
    assert _trace(call_span) == _trace(attempt) == trace_id
    assert _id(attempt) == parent_id and _parent(attempt) == _id(call_span)


# --- later segments: resume, after a restart, approval, startup -------------------------------------------


def test_a_resumed_run_adds_a_segment_to_its_trace(tmp_path, monkeypatch, exported) -> None:
    upstream = _Upstream()
    with _serve(tmp_path, monkeypatch, upstream) as client:
        waiting = _ask(client, AMBIGUOUS)
        upstream.requests.clear()
        resumed = _resume(client, waiting["run_id"])

    assert waiting["status"] == "WAITING_USER" and resumed["status"] == "SUCCEEDED"
    spans = exported.get_finished_spans()
    trace_id = _expected_trace(waiting["run_id"])
    roots = _assert_one_tree_per_segment(spans, trace_id)
    assert sorted((root.attributes["queryshield.segment"], root.attributes["queryshield.status"]) for root in roots) == [
        ("execute", "WAITING_USER"), ("resume", "SUCCEEDED"),
    ]
    _assert_attempts_are_the_header_spans(_of_trace(spans, trace_id), upstream.requests)


def test_a_run_resumed_or_approved_by_a_new_process_stays_on_its_trace(tmp_path, monkeypatch, exported) -> None:
    first = _Upstream()
    with _serve(tmp_path, monkeypatch, first) as client:
        waiting = _ask(client, AMBIGUOUS)
        pending = _ask(client, NAMES)
    assert waiting["status"] == "WAITING_USER" and pending["status"] == "WAITING_APPROVAL"

    second = _Upstream()
    with _serve(tmp_path, monkeypatch, second) as client:
        second.requests.clear()
        resumed = _resume(client, waiting["run_id"])
        resume_requests = list(second.requests)
        approved = client.post(
            f"/runs/{pending['run_id']}/approval", headers=auth(APPROVER), json={"approval_id": pending["approval_id"], "decision": "approve"}
        ).json()

    assert resumed["status"] == "SUCCEEDED" and approved["status"] == "SUCCEEDED"
    spans = exported.get_finished_spans()
    resumed_roots = _assert_one_tree_per_segment(spans, _expected_trace(waiting["run_id"]))
    assert sorted(root.attributes["queryshield.segment"] for root in resumed_roots) == ["execute", "resume"]
    _assert_attempts_are_the_header_spans(_of_trace(spans, _expected_trace(waiting["run_id"])), resume_requests)
    approval_trace = _expected_trace(pending["run_id"])
    roots = _assert_one_tree_per_segment(spans, approval_trace)
    [segment] = [root for root in roots if root.attributes["queryshield.segment"] == "approved"]
    assert segment.attributes["queryshield.approval_id"] == pending["approval_id"]
    assert segment.attributes["queryshield.status"] == "SUCCEEDED"
    [tool] = [span for span in _of_trace(spans, approval_trace) if _parent(span) == _id(segment)]
    assert tool.name == "queryshield.tool" and tool.attributes["queryshield.tool_name"] == "query_readonly"
    assert tool.attributes["queryshield.row_count"] == 2
    # The first execution paused on the approval: its tool span is not failed, it has the step's status and the code.
    queries = [span for span in _named(_of_trace(spans, approval_trace), "queryshield.tool") if span.attributes["queryshield.tool_name"] == "query_readonly"]
    [paused] = [span for span in queries if span is not tool]
    assert paused.status.status_code is trace.StatusCode.UNSET
    assert paused.attributes["queryshield.status"] == "waiting_approval"
    assert paused.attributes["queryshield.error_code"] == "approval_required"


def test_a_query_that_waits_for_the_users_answer_leaves_its_tool_span_unfailed(service, exported) -> None:
    run = _start(service, AMBIGUOUS, [_query()])

    assert run["status"] == "WAITING_USER"
    spans = _of_trace(exported.get_finished_spans(), _expected_trace(run["run_id"]))
    [paused] = _named(spans, "queryshield.tool")
    assert paused.status.status_code is trace.StatusCode.UNSET
    assert paused.attributes["queryshield.status"] == "waiting_user"
    assert paused.attributes["queryshield.error_code"] == "clarification_required"
    [step] = [span for span in spans if span.name == "queryshield.step.execute_tool"]
    assert step.attributes["queryshield.status"] == "waiting_user" and step.status.status_code is trace.StatusCode.UNSET


PAUSES = (ApprovalRequiredError, ClarificationRequiredError)


@pytest.mark.parametrize(
    ("error", "pauses", "status", "failed"),
    [
        pytest.param(lambda: ApprovalRequiredError({"sql": "x"}), PAUSES, "waiting_approval", False, id="approval_that_pauses"),
        pytest.param(lambda: ClarificationRequiredError(SimpleNamespace(id="clarify.x")), PAUSES, "waiting_user", False, id="clarification_that_pauses"),
        pytest.param(lambda: ApprovalRequiredError({"sql": "x"}), (), "failed", True, id="approval_the_caller_does_not_pause_on"),
        pytest.param(lambda: ClarificationRequiredError(SimpleNamespace(id="clarify.x")), (), "failed", True, id="clarification_the_caller_does_not_pause_on"),
        pytest.param(
            lambda: ClarificationValueUnsupportedError(SimpleNamespace(id="clarify.x"), SimpleNamespace(unsupported_note="n")),
            PAUSES,
            "failed",
            True,
            id="unsupported_scope",
        ),
        pytest.param(lambda: ToolError("table_not_allowed", "t"), PAUSES, "failed", True, id="tool_error"),
    ],
)
def test_a_tool_span_is_failed_unless_its_caller_says_the_error_pauses_the_run(exported, error, pauses, status, failed) -> None:
    raised = error()

    def call():
        raise raised

    with pytest.raises(type(raised)) as caught:
        tool_execution._in_tool_span("query_readonly", call, pauses=pauses)

    assert caught.value is raised
    [span] = exported.get_finished_spans()
    assert span.status.status_code is (trace.StatusCode.ERROR if failed else trace.StatusCode.UNSET)
    assert span.attributes["queryshield.status"] == status and span.attributes["queryshield.error_code"] == raised.code


def _tool_spans(exported, run_id: str) -> list:
    return _named(_of_trace(exported.get_finished_spans(), _expected_trace(run_id)), "queryshield.tool")


def _b0_run(service, question: str, steps) -> dict:
    svc, _ = service
    deps = svc.default_dependencies()
    deps.model, deps.retriever, deps.profile = _Scripted(steps), None, B0_PROFILE
    return svc.run_sync(identity=CLARIFICATION_REQUESTER, question=question, time_window={**SEPTEMBER, "timezone": "UTC"}, deps=deps)


def test_b0_ends_on_an_approval_so_its_tool_span_is_failed(service, exported) -> None:
    run = _b0_run(service, "列出前三个客户的名字", [NAME_QUERY])

    assert (run["status"], run["error_code"]) == ("DENIED", "approval_required")
    [tool] = _tool_spans(exported, run["run_id"])
    assert tool.status.status_code is trace.StatusCode.ERROR
    assert (tool.attributes["queryshield.status"], tool.attributes["queryshield.error_code"]) == ("failed", "approval_required")


def test_b0_waits_on_the_catalogs_question_so_its_tool_span_is_not_failed(service, exported) -> None:
    run = _b0_run(service, AMBIGUOUS, [_query()])

    assert run["status"] == "WAITING_USER"
    [tool] = _tool_spans(exported, run["run_id"])
    assert tool.status.status_code is trace.StatusCode.UNSET
    assert (tool.attributes["queryshield.status"], tool.attributes["queryshield.error_code"]) == ("waiting_user", "clarification_required")


@pytest.mark.parametrize(
    "pause",
    [
        pytest.param(lambda: ApprovalRequiredError({"sql": "x"}), id="approval"),
        pytest.param(lambda: ClarificationRequiredError(SimpleNamespace(id="clarify.metric_basis", question="q")), id="clarification"),
    ],
)
def test_a_sub_agent_does_not_wait_so_its_tool_span_is_failed(b2, exported, monkeypatch, pause) -> None:
    """A sub-agent's approval or question leaves its subtask incomplete and fails the run."""

    def query_raises(tools, name, arguments, **options):
        raise pause()

    monkeypatch.setattr(tool_execution, "_call_tool", query_raises)
    run = run_b2(b2, COMPOSITE, RoleModel([delegate((["gross_fen"], SEP_WINDOW), (["net_fen"], SEP_WINDOW)), cite_all]))

    assert (run["status"], run["error_code"]) == ("FAILED", "subtask_incomplete")
    tools = _tool_spans(exported, run["run_id"])
    assert len(tools) == 2 and all(span.attributes["queryshield.agent"] != "coordinator" for span in _parents(tools, exported))
    assert all(span.status.status_code is trace.StatusCode.ERROR for span in tools)
    assert {span.attributes["queryshield.status"] for span in tools} == {"failed"}


def test_a_coordinator_that_queries_a_value_needing_approval_waits_and_its_tool_span_is_not_failed(b2, exported) -> None:
    run = run_b2(b2, "列出前三个客户的名字", RoleModel([NAME_QUERY]))

    assert run["status"] == "WAITING_APPROVAL"
    [tool] = _tool_spans(exported, run["run_id"])
    assert tool.status.status_code is trace.StatusCode.UNSET
    assert (tool.attributes["queryshield.status"], tool.attributes["queryshield.error_code"]) == ("waiting_approval", "approval_required")
    [step] = _parents([tool], exported)
    assert step.name == "queryshield.step.execute_tool" and step.attributes["queryshield.agent"] == "coordinator"


def _parents(spans, exported) -> list:
    by_id = {_id(span): span for span in exported.get_finished_spans()}
    return [by_id[_parent(span)] for span in spans]


def test_a_run_ended_at_startup_gets_a_short_segment_on_its_trace(tmp_path, monkeypatch, receiver) -> None:
    """The service starts the export before its startup recovery, so the run that recovery ends is exported."""

    upstream = _Upstream()
    with _serve(tmp_path, monkeypatch, upstream):
        pass
    ids: list[str] = []
    old = _Old(tmp_path / "state.sqlite3")

    def exits(messages):
        ids.append(old.running_id())
        raise _ProcessExit()

    old.dies(lambda: old.sync(PAID, [_query(), exits]))
    old.store.close()
    with _serve(tmp_path, monkeypatch, upstream, traces_url=receiver.url):
        pass

    segments = [span for span in _received_spans(receiver) if span.name == "queryshield.run"]
    [ended] = [span for span in segments if _attribute(span, "queryshield.segment") == "interrupted"]
    assert ended.trace_id.hex() == _expected_trace(ids[0])
    assert _attribute(ended, "queryshield.status") == "FAILED"
    assert _attribute(ended, "queryshield.error_code") == "execution_interrupted"


# --- what spans carry --------------------------------------------------------------------------------------


MARK = "QSMARK7f3c"
SECRET_KEY = "span-export-model-key-0123456789"


class _ScriptedUpstream:
    """An OpenAI-compatible chat upstream that answers with the scripted replies, then with the last one."""

    def __init__(self, replies) -> None:
        self.replies = list(replies)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, httpx.Response):
            return reply
        body = {"id": "chat", "model": "scripted", "choices": [{"message": {"content": json.dumps(reply, ensure_ascii=False)}, "finish_reason": "stop"}]}
        return httpx.Response(200, json=body)


def _provides(model):
    return lambda: model


def test_spans_carry_no_question_answer_sql_token_or_key(tmp_path, monkeypatch, exported) -> None:
    # Identifiers are lower case in SQL, so the SQL carries the mark in lower case.
    sql = f"SELECT COUNT(*) AS {MARK.lower()} FROM orders"
    replies = [
        {"type": "tool_call", "name": "query_readonly", "arguments": {"sql": sql, "params": {}}},
        {"type": "final_answer", "answer": f"{MARK} 是答案", "source_ids": []},
    ]
    upstream = _Upstream()
    with _serve(tmp_path, monkeypatch, upstream, retriever=False) as client:
        for script in (replies, [httpx.Response(400, json={"error": {"code": "bad_request", "message": MARK}})]):
            config = OpenAICompatibleConfig(base_url="http://scripted/v1", api_key=SECRET_KEY, model="scripted")
            model = OpenAICompatibleModel(config, client=httpx.Client(transport=httpx.MockTransport(_ScriptedUpstream(script))))
            app.dependency_overrides[get_model_provider] = _provides(model)
            body = _ask(client, f"{MARK} 2026年9月已支付订单总额是多少？")
            assert body["run_id"]

    spans = exported.get_finished_spans()
    assert spans
    text = "\n".join(span.to_json() for span in spans)
    assert [span.attributes["queryshield.tool_name"] for span in _named(spans, "queryshield.tool")] == ["query_readonly"]
    for value in (MARK, MARK.lower(), SECRET_KEY, REQUESTER, PLACEHOLDER_KEY, "SELECT", "tenant_id"):
        assert value not in text, value
    for span in spans:
        assert set(span.attributes) <= ALLOWED, set(span.attributes) - ALLOWED
        assert span.status.description is None
        assert all(set(event.attributes) <= ALLOWED for event in span.events)
        assert dict(span.resource.attributes) == {"service.name": "queryshield"}
    assert any(span.status.status_code is trace.StatusCode.ERROR for span in spans)


def test_an_exception_inside_a_span_only_marks_it_failed(exported) -> None:
    """Product error messages are fixed text, but a message may still hold SQL or text: it is never recorded."""

    with pytest.raises(ValueError):
        with tracing.span("queryshield.tool"):
            raise ValueError(f"{MARK} SELECT 1")

    [span] = exported.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.ERROR and span.status.description is None
    assert span.events == () and MARK not in span.to_json()


# --- off by default ---------------------------------------------------------------------------------------


class _Receiver:
    """A local OTLP/HTTP receiver: records each request; answers with ``status`` after ``delay`` seconds."""

    def __init__(self, status: int = 200, delay: float = 0.0) -> None:
        self.requests: list[tuple[str, str, bytes]] = []
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                receiver.requests.append((self.path, self.headers.get("Content-Type", ""), body))
                time.sleep(delay)
                self.send_response(status)
                self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1/traces"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def receiver():
    receiver = _Receiver()
    yield receiver
    receiver.close()


def _received_spans(receiver: _Receiver) -> list:
    """Every span the receiver got, each request checked to be OTLP/HTTP protobuf from service queryshield."""

    spans = []
    for path, content_type, payload in receiver.requests:
        assert path == "/v1/traces" and content_type == "application/x-protobuf"
        for resource_spans in ExportTraceServiceRequest.FromString(payload).resource_spans:
            resource = {item.key: item.value.string_value for item in resource_spans.resource.attributes}
            assert resource == {"service.name": "queryshield"}
            spans += [span for scope in resource_spans.scope_spans for span in scope.spans]
    return spans


def _attribute(span, name: str) -> str | None:
    return next((item.value.string_value for item in span.attributes if item.key == name), None)


def _export_threads() -> list[str]:
    return [thread.name for thread in threading.enumerate() if thread.name.startswith("OtelBatch")]


@pytest.mark.parametrize("setting", [None, "", "   "], ids=["unset", "empty", "blank"])
def test_without_the_setting_nothing_is_built_and_nothing_is_sent(tmp_path, monkeypatch, receiver, setting) -> None:
    upstream = _Upstream()
    with _serve(tmp_path, monkeypatch, upstream, traces_url=setting) as client:
        upstream.requests.clear()
        body = _ask(client, KNOWLEDGE)
        assert tracing._provider is None and _export_threads() == []
        assert not isinstance(trace.get_tracer_provider(), SdkTracerProvider)

    assert body["status"] == "SUCCEEDED"
    assert tracing._provider is None and _export_threads() == []
    assert receiver.requests == []
    assert all(TRACEPARENT.fullmatch(request.headers["traceparent"]) for request in upstream.requests)


def test_compose_passes_the_setting_through_empty_by_default() -> None:
    text = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert f"      {VARIABLE}: ${{{VARIABLE}:-}}\n" in text


@pytest.mark.parametrize(
    "url", ["ftp://collector:4318/v1/traces", "http:///v1/traces", "http://user@collector:4318/v1/traces", "not a url"]
)
def test_a_bad_address_stops_the_start(tmp_path, monkeypatch, url) -> None:
    with pytest.raises(RuntimeConfigurationError) as raised:
        with _serve(tmp_path, monkeypatch, _Upstream(), retriever=False, traces_url=url):
            pass
    assert raised.value.code == "invalid_otlp_traces_url"
    assert tracing._provider is None and _export_threads() == []


# --- export over OTLP/HTTP ----------------------------------------------------------------------------------


def test_the_spans_are_sent_over_otlp_http_to_the_address(tmp_path, monkeypatch, receiver) -> None:
    upstream = _Upstream()
    with _serve(tmp_path, monkeypatch, upstream, traces_url=receiver.url) as client:
        body = _ask(client, PAID_TOTAL)
        assert body["status"] == "SUCCEEDED"
        assert not isinstance(trace.get_tracer_provider(), SdkTracerProvider), "the global provider stays unset"

    assert receiver.requests, "the remaining spans are sent on shutdown"
    runs = [span for span in _received_spans(receiver) if span.name == "queryshield.run"]
    assert [span.trace_id.hex() for span in runs] == [_expected_trace(body["run_id"])]
    assert tracing._provider is None


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.mark.parametrize("collector", ["refused", "error", "hanging"])
def test_a_collector_that_fails_never_reaches_the_run_and_the_stop_is_bounded(tmp_path, monkeypatch, collector) -> None:
    monkeypatch.setattr(tracing, "SHUTDOWN_SECONDS", 0.5)
    receiver = None if collector == "refused" else _Receiver(status=500 if collector == "error" else 200, delay=8.0 if collector == "hanging" else 0.0)
    url = receiver.url if receiver is not None else f"http://127.0.0.1:{_free_port()}/v1/traces"
    upstream = _Upstream()
    try:
        with _serve(tmp_path, monkeypatch, upstream, traces_url=url) as client:
            body = _ask(client, PAID_TOTAL)
            assert tracing._provider is not None
            stopping = time.monotonic()
        stopped_after = time.monotonic() - stopping
    finally:
        if receiver is not None:
            receiver.close()

    assert body["status"] == "SUCCEEDED" and body["answer_status"] == "verified"
    assert stopped_after < tracing.SHUTDOWN_SECONDS + 1.0, stopped_after
    assert tracing._provider is None


# --- the MCP host adds no trace context ---------------------------------------------------------------------


def test_mcp_requests_carry_no_trace_context_with_export_on(tmp_path, monkeypatch, exported) -> None:
    """K10: the host's MCP SDK injects the current context into params._meta; it sees none of QueryShield's."""

    from mcp.shared import jsonrpc_dispatcher

    original = jsonrpc_dispatcher.inject_trace_context
    sent: list[dict] = []

    def recording(meta) -> None:
        original(meta)
        sent.append(dict(meta))

    monkeypatch.setattr(jsonrpc_dispatcher, "inject_trace_context", recording)
    # As in test_mcp_product_path: the in-process Fake model and the product's own retriever.
    monkeypatch.setenv("QUERYSHIELD_STATE_STORE_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", REQUESTER)
    monkeypatch.setenv("QUERYSHIELD_METADATA_TOOLS", "mcp")
    for name in ("QUERYSHIELD_AGENT_PROFILE", "QUERYSHIELD_RETRIEVAL", "QUERYSHIELD_MODEL_PROTOCOL", VARIABLE):
        monkeypatch.delenv(name, raising=False)
    reset_shared_state_stores()
    try:
        with TestClient(app) as client:
            body = _ask(client, "已支付订单有几笔")
    finally:
        reset_shared_state_stores()

    assert body["status"] == "SUCCEEDED"
    assert sent, "the run sent MCP requests"
    assert all("traceparent" not in meta and "tracestate" not in meta for meta in sent)
    spans = exported.get_finished_spans()
    assert all(span.instrumentation_scope.name == "queryshield" for span in spans)
    tools = [span for span in _named(spans, "queryshield.tool") if span.attributes["queryshield.tool_name"] == "search_catalog"]
    assert tools and _trace(tools[0]) == _expected_trace(body["run_id"])


# --- dependencies -------------------------------------------------------------------------------------------


NEW_PACKAGES = {
    "googleapis-common-protos": "1.75.5",
    "opentelemetry-exporter-http-transport": "0.66b0",
    "opentelemetry-exporter-otlp-common": "0.66b0",
    "opentelemetry-exporter-otlp-proto-common": "1.45.0",
    "opentelemetry-exporter-otlp-proto-http": "1.45.0",
    "opentelemetry-proto": "1.45.0",
    "opentelemetry-sdk": "1.45.0",
    "opentelemetry-semantic-conventions": "0.66b0",
    "protobuf": "7.36.2",
}


def test_the_lock_pins_the_exporter_and_no_instrumentation_package() -> None:
    lines = (PROJECT_ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
    pins = dict(line.split("==", 1) for line in lines if "==" in line and not line.startswith("#"))
    assert {name: pins.get(name) for name in NEW_PACKAGES} == NEW_PACKAGES
    assert pins["opentelemetry-api"] == "1.45.0"
    assert not [name for name in pins if name.startswith("opentelemetry-instrumentation")]
    assert {name: importlib.metadata.version(name) for name in NEW_PACKAGES} == NEW_PACKAGES
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for name in ("opentelemetry-api", "opentelemetry-sdk", "opentelemetry-exporter-otlp-proto-http"):
        assert f'"{name}>=1.45,<2",' in pyproject
