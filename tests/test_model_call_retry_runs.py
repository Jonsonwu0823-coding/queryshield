"""A retried model call is still one logical call of the run: one count, one event, one usage.

The product HTTP path runs with the Real adapters pointed at the fake upstream (as in
tests/test_run_id_header.py).  The chosen chat request is refused before it is sent
(httpx.ConnectError raised from the client's request hook), so the adapter sends it once more.
The run must count the call once, take its usage from the response that came back, record a
``transport_retry`` row under the same model_call_id in the durable call store, and list both
attempts in the call's ``model_call`` event.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from agent_core_scenarios import B0_WINDOW, fixture_tools
from multi_agent_support import COMPOSITE
import test_clarification as clarification_cases
from test_run_id_header import BASE_URL, PLACEHOLDER_KEY, REQUESTER, _ask
from queryshield.agent import DurableModelCallStore
from queryshield.agent.runtime import run_b0_single_pass
from queryshield.api.main import app, get_model_provider, get_retriever_source
from queryshield.approval.service import reset_shared_state_stores, shared_state_store
from queryshield.catalog import load_default_catalog
from queryshield.knowledge.index import build_embedding_index
from queryshield.knowledge.retrieval import HybridRetriever
from queryshield.knowledge.runtime import product_knowledge
from queryshield.providers.embedding import EmbeddingConfig, OpenAICompatibleEmbedding
from queryshield.providers.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleModel
from scripts.fake_upstream import FAKE_UPSTREAM_MODEL
from scripts.fake_upstream import app as fake_upstream_app

QUESTION = "2026年9月已支付订单总额是多少？"
CHAT_PATH = "/v1/chat/completions"
TOKENS = ("prompt_tokens", "completion_tokens", "total_tokens")
# The adapter passes its timeout to the injected client; TestClient ignores it and warns.
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")


class _RefusingUpstream:
    """The fake upstream behind a client that refuses the chosen chat requests before they are sent."""

    def __init__(self, refuse: set[int]) -> None:
        self.refuse = refuse  # 1-based positions among the chat requests
        self.chat_requests: list[httpx.Request] = []
        self.chat_responses: list[dict] = []
        self.waits: list[float] = []
        self.client = TestClient(fake_upstream_app)
        self.client.event_hooks = {"request": [self._on_request], "response": [self._on_response]}

    def _on_request(self, request: httpx.Request) -> None:
        if request.url.path == CHAT_PATH:
            self.chat_requests.append(request)
            if len(self.chat_requests) in self.refuse:
                raise httpx.ConnectError("refused", request=request)

    def _on_response(self, response: httpx.Response) -> None:
        if response.request.url.path == CHAT_PATH:
            response.read()
            self.chat_responses.append(response.json())

    def model(self) -> OpenAICompatibleModel:
        config = OpenAICompatibleConfig(base_url=BASE_URL, api_key=PLACEHOLDER_KEY, model=FAKE_UPSTREAM_MODEL)
        return OpenAICompatibleModel(config, client=self.client, sleep=self.waits.append)

    def embedding(self) -> OpenAICompatibleEmbedding:
        config = EmbeddingConfig(
            base_url=BASE_URL, api_key=PLACEHOLDER_KEY, model=FAKE_UPSTREAM_MODEL, model_revision=FAKE_UPSTREAM_MODEL, dimensions=128
        )
        return OpenAICompatibleEmbedding(config, client=self.client, sleep=self.waits.append)

    def usage_total(self) -> dict[str, int]:
        return {name: sum(response["usage"][name] for response in self.chat_responses) for name in TOKENS}


def _service(tmp_path, monkeypatch, upstream: _RefusingUpstream, profile: str) -> TestClient:
    monkeypatch.setenv("QUERYSHIELD_STATE_STORE_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.setenv("QUERYSHIELD_CALL_STORE_PATH", str(tmp_path / "calls.sqlite3"))
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", profile)
    monkeypatch.delenv("QUERYSHIELD_MODEL_PROTOCOL", raising=False)
    monkeypatch.delenv("QUERYSHIELD_RETRIEVAL", raising=False)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", REQUESTER)
    reset_shared_state_stores()
    embedder = upstream.embedding()
    build = build_embedding_index(product_knowledge(demo=False).snapshot, embedder, ingest_job_id="retry-runs")
    retriever = HybridRetriever(catalog=load_default_catalog(), snapshot=build.snapshot, index=build.index, embedder=embedder)
    model = upstream.model()
    app.dependency_overrides[get_model_provider] = lambda: model
    app.dependency_overrides[get_retriever_source] = lambda: lambda: retriever
    return TestClient(app)


@pytest.fixture()
def cleanup():
    yield
    app.dependency_overrides.clear()
    reset_shared_state_stores()


def _model_call_events(run_id: str) -> list[dict]:
    events = shared_state_store().events(run_id, after_event_id=0, limit=1000)
    return [event["payload"] for event in events if event["type"] == "agent_step" and event["payload"].get("kind") == "model_call"]


def _assert_one_logical_call_retried(tmp_path, upstream: _RefusingUpstream, body: dict, refused: int) -> dict:
    """The run counted each logical call once; the refused one has two attempts everywhere they are recorded."""

    assert body["status"] == "SUCCEEDED"
    assert upstream.waits == [1.0]
    # One request more than there are logical calls, and every logical call has one response.
    assert len(upstream.chat_requests) == body["model_call_count"] + 1 == len(upstream.chat_responses) + 1
    assert body["usage_total"] == {"status": "known", **upstream.usage_total()}

    first, second = upstream.chat_requests[refused - 1], upstream.chat_requests[refused]
    assert first.headers["idempotency-key"] == second.headers["idempotency-key"]
    assert first.headers["x-run-id"] == second.headers["x-run-id"] == body["run_id"]
    assert first.content == second.content
    call_id = first.headers["idempotency-key"]
    request_ids = [first.headers["x-client-request-id"], second.headers["x-client-request-id"]]

    with DurableModelCallStore(tmp_path / "calls.sqlite3") as store:
        attempts = store.attempts(body["run_id"], call_id)
    assert [(row.model_call_id, row.request_id, row.attempt_kind) for row in attempts] == [
        (call_id, request_ids[0], "new"),
        (call_id, request_ids[1], "transport_retry"),
    ]

    events = _model_call_events(body["run_id"])
    assert len(events) == body["model_call_count"]
    (retried,) = [event for event in events if len(event["attempts"]) == 2]
    assert (retried["model_call_id"], retried["request_id"]) == (call_id, request_ids[0])
    assert retried["attempts"] == [
        {"request_id": request_ids[0], "status": "failed", "error_code": "upstream_request_error"},
        {"request_id": request_ids[1], "status": "succeeded"},
    ]
    return retried


def test_a_b1_call_refused_once_is_one_call_with_two_attempts(tmp_path, monkeypatch, cleanup) -> None:
    upstream = _RefusingUpstream(refuse={1})
    with _service(tmp_path, monkeypatch, upstream, "b1") as client:
        body = _ask(client, QUESTION)
    _assert_one_logical_call_retried(tmp_path, upstream, body, refused=1)


def test_a_b1_call_refused_twice_fails_once_and_still_records_the_retry(tmp_path, monkeypatch, cleanup) -> None:
    upstream = _RefusingUpstream(refuse={1, 2})
    with _service(tmp_path, monkeypatch, upstream, "b1") as client:
        body = _ask(client, QUESTION)

    assert (body["status"], body["error_code"], body["model_call_count"]) == ("FAILED", "upstream_request_error", 1)
    assert body["usage_total"]["status"] == "unknown" and upstream.chat_responses == []
    first, second = upstream.chat_requests
    call_id = first.headers["idempotency-key"]
    assert second.headers["idempotency-key"] == call_id and first.content == second.content
    with DurableModelCallStore(tmp_path / "calls.sqlite3") as store:
        kinds = [row.attempt_kind for row in store.attempts(body["run_id"], call_id)]
    assert kinds == ["new", "transport_retry"]
    (event,) = _model_call_events(body["run_id"])
    assert (event["status"], event["request_id"]) == ("failed", first.headers["x-client-request-id"])
    assert [attempt["request_id"] for attempt in event["attempts"]] == [
        first.headers["x-client-request-id"],
        second.headers["x-client-request-id"],
    ]


def test_a_b2_sub_agent_call_refused_once_is_one_call_with_two_attempts(tmp_path, monkeypatch, cleanup) -> None:
    # Chat calls in order: search, delegate, the two sub-agents (side by side), the answer.
    upstream = _RefusingUpstream(refuse={3})
    with _service(tmp_path, monkeypatch, upstream, "b2") as client:
        body = _ask(client, COMPOSITE)
    assert body["profile"] == "B2-multi-agent" and body["model_call_count"] == 5
    retried = _assert_one_logical_call_retried(tmp_path, upstream, body, refused=3)
    assert retried["agent"].startswith("subtask-")


def test_a_b0_call_refused_once_is_one_call_with_two_attempts(cleanup) -> None:
    upstream = _RefusingUpstream(refuse={1})
    tools, _ = fixture_tools()
    record = run_b0_single_pass(upstream.model(), tools, clarification_cases._context("run-b0-retry"), QUESTION, time_window=B0_WINDOW)

    assert upstream.waits == [1.0] and len(upstream.chat_requests) == 2 and len(upstream.chat_responses) == 1
    assert record["model_call_count"] == 1
    assert record["usage"] == {"usage_status": "known", **upstream.usage_total()}
    first, second = upstream.chat_requests
    assert first.headers["idempotency-key"] == second.headers["idempotency-key"] == record["model_call_ids"][0]
    assert first.content == second.content
    assert record["model_call_attempts"] == [
        {"request_id": first.headers["x-client-request-id"], "status": "failed", "error_code": "upstream_request_error"},
        {"request_id": second.headers["x-client-request-id"], "status": "succeeded"},
    ]


def _b0_over(outcomes):
    """B0's one call over a scripted transport: each outcome is an exception to raise or a reply's content."""

    requests: list[httpx.Request] = []
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        outcome = outcomes[len(requests) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        usage = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
        body = {"id": "chat-1", "choices": [{"message": {"content": outcome}}], "usage": usage}
        return httpx.Response(200, json=body)

    config = OpenAICompatibleConfig(base_url=BASE_URL, api_key=PLACEHOLDER_KEY, model=FAKE_UPSTREAM_MODEL)
    tools, executor = fixture_tools()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleModel(config, client=client, sleep=waits.append)
        record = run_b0_single_pass(model, tools, clarification_cases._context("run-b0-failed"), QUESTION, time_window=B0_WINDOW)
    return record, requests, waits, executor


@pytest.mark.parametrize(
    "second, error_code, second_attempt",
    [
        pytest.param(
            httpx.ConnectError("refused"),
            "upstream_request_error",
            {"status": "failed", "error_code": "upstream_request_error"},
            id="refused_twice",
        ),
        pytest.param("not json", "invalid_json", {"status": "succeeded"}, id="reply_does_not_parse"),
    ],
)
def test_a_b0_call_that_fails_after_a_retry_still_records_both_attempts(second, error_code, second_attempt) -> None:
    record, requests, waits, executor = _b0_over([httpx.ConnectError("refused"), second])

    assert (record["status"], record["error_code"], record["model_call_count"]) == ("failed", error_code, 1)
    assert waits == [1.0] and len(requests) == 2 and executor.executed == []
    first, retry = requests
    assert first.headers["idempotency-key"] == retry.headers["idempotency-key"] == record["model_call_ids"][0]
    assert first.content == retry.content
    assert record["model_call_attempts"] == [
        {"request_id": first.headers["x-client-request-id"], "status": "failed", "error_code": "upstream_request_error"},
        {"request_id": retry.headers["x-client-request-id"], **second_attempt},
    ]
