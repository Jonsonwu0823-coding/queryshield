"""Every model and query-embedding call inside a run carries ``X-Run-Id``; calls outside a run do not.

The product HTTP path runs with the Real adapters pointed at the fake upstream (FastAPI's
TestClient as the httpx client, so no port is opened); every request the adapters send is
captured with its headers.  A run id (``run-<uuid>``), a request id (a bare uuid) and a
model call id (``local-<uuid>``) never share a value, so a header filled from the wrong
source cannot pass.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from queryshield.api.main import app, get_model_provider, get_retriever_source
from queryshield.approval.service import reset_shared_state_stores
from queryshield.catalog import load_default_catalog
from queryshield.knowledge.index import build_embedding_index
from queryshield.knowledge.retrieval import HybridRetriever
from queryshield.knowledge.runtime import product_knowledge
from queryshield.providers.embedding import EmbeddingConfig, OpenAICompatibleEmbedding
from queryshield.providers.http import json_headers
from queryshield.providers.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleModel
from scripts.fake_upstream import FAKE_UPSTREAM_MODEL
from scripts.fake_upstream import app as fake_upstream_app

BASE_URL = "http://fake-upstream/v1"
PLACEHOLDER_KEY = "not-a-real-key"
REQUESTER = "run-id-a-requester"
# The adapter passes its timeout to the injected client; TestClient ignores it and warns.
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")


class _Upstream:
    """The fake upstream behind a client that keeps every request it sends."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.client = TestClient(fake_upstream_app)
        self.client.event_hooks = {"request": [self.requests.append]}

    def model(self) -> OpenAICompatibleModel:
        return OpenAICompatibleModel(
            OpenAICompatibleConfig(base_url=BASE_URL, api_key=PLACEHOLDER_KEY, model=FAKE_UPSTREAM_MODEL), client=self.client
        )

    def embedding(self) -> OpenAICompatibleEmbedding:
        config = EmbeddingConfig(
            base_url=BASE_URL, api_key=PLACEHOLDER_KEY, model=FAKE_UPSTREAM_MODEL, model_revision=FAKE_UPSTREAM_MODEL, dimensions=128
        )
        return OpenAICompatibleEmbedding(config, client=self.client)

    def sent(self, path: str) -> list[httpx.Request]:
        return [request for request in self.requests if request.url.path == f"/v1/{path}"]


@pytest.fixture()
def upstream() -> _Upstream:
    return _Upstream()


@pytest.fixture()
def retriever(upstream) -> HybridRetriever:
    """The product hybrid retriever with the Real embedding adapter; building its index is not a run."""

    embedder = upstream.embedding()
    build = build_embedding_index(product_knowledge(demo=False).snapshot, embedder, ingest_job_id="run-id-header")
    return HybridRetriever(catalog=load_default_catalog(), snapshot=build.snapshot, index=build.index, embedder=embedder)


@pytest.fixture()
def service(tmp_path, monkeypatch, upstream, retriever):
    monkeypatch.setenv("QUERYSHIELD_STATE_STORE_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    monkeypatch.delenv("QUERYSHIELD_AGENT_PROFILE", raising=False)
    monkeypatch.delenv("QUERYSHIELD_MODEL_PROTOCOL", raising=False)
    monkeypatch.delenv("QUERYSHIELD_RETRIEVAL", raising=False)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", REQUESTER)
    reset_shared_state_stores()
    model = upstream.model()
    app.dependency_overrides[get_model_provider] = lambda: model
    app.dependency_overrides[get_retriever_source] = lambda: lambda: retriever
    upstream.requests.clear()
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
    reset_shared_state_stores()


def _ask(client: TestClient, question: str, **extra) -> dict:
    return client.post("/queries", headers={"Authorization": f"Bearer {REQUESTER}"}, json={"question": question, **extra}).json()


def _run_ids(requests: list[httpx.Request]) -> list[str | None]:
    return [request.headers.get("x-run-id") for request in requests]


def _assert_tagged(requests: list[httpx.Request], run_id: str) -> None:
    assert requests, "the path called the upstream"
    assert _run_ids(requests) == [run_id] * len(requests)
    # Never the request id the same call sends.
    assert all(request.headers["x-client-request-id"] != run_id for request in requests)


@pytest.mark.parametrize("protocol", ["json", "native"])
def test_every_b1_model_call_carries_the_run_id(service, upstream, monkeypatch, protocol) -> None:
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", protocol)

    body = _ask(service, "2026年9月已支付订单总额是多少？")

    assert body["status"] == "SUCCEEDED"
    assert body["run_id"].startswith("run-")
    _assert_tagged(upstream.sent("chat/completions"), body["run_id"])


def test_the_b0_model_call_carries_the_run_id(service, upstream, monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "b0")

    body = _ask(service, "2026年9月已支付订单总额是多少？")

    assert body["status"] == "SUCCEEDED"
    chat = upstream.sent("chat/completions")
    assert len(chat) == 1
    _assert_tagged(chat, body["run_id"])


def test_the_query_embedding_of_a_run_carries_the_run_id(service, upstream) -> None:
    body = _ask(service, "退款后净额是怎么算的？")

    assert body["status"] == "SUCCEEDED"
    _assert_tagged(upstream.sent("embeddings"), body["run_id"])
    _assert_tagged(upstream.sent("chat/completions"), body["run_id"])


def test_the_calls_after_a_resume_carry_the_same_run_id(service, upstream) -> None:
    waiting = _ask(service, "2026年9月销售额是多少？")
    assert waiting["status"] == "WAITING_USER"
    upstream.requests.clear()

    resumed = service.post(
        f"/runs/{waiting['run_id']}/resume", headers={"Authorization": f"Bearer {REQUESTER}"}, json={"answer": "按支付金额统计"}
    ).json()

    assert resumed["status"] == "SUCCEEDED"
    _assert_tagged(upstream.sent("chat/completions"), waiting["run_id"])


def test_two_runs_send_their_own_run_ids(service, upstream) -> None:
    first = _ask(service, "2026年9月已支付订单总额是多少？")
    first_calls = len(upstream.requests)
    second = _ask(service, "2026年9月已支付订单总额是多少？")

    assert first["run_id"] != second["run_id"]
    assert set(_run_ids(upstream.requests[:first_calls])) == {first["run_id"]}
    assert set(_run_ids(upstream.requests[first_calls:])) == {second["run_id"]}


def test_building_the_index_and_a_probe_call_carry_no_run_id(upstream) -> None:
    embedder = upstream.embedding()
    build_embedding_index(product_knowledge(demo=False).snapshot, embedder, ingest_job_id="outside-a-run")
    upstream.model().complete([{"role": "user", "content": "ping"}])

    assert upstream.sent("embeddings") and upstream.sent("chat/completions")
    assert _run_ids(upstream.requests) == [None] * len(upstream.requests)


def test_the_header_is_the_run_id_unchanged_and_only_when_given() -> None:
    assert "X-Run-Id" not in json_headers("k", "request-1", None)
    headers = json_headers("k", "request-1", "eval-b1-0123456789abcdef0123456789abcdef-state-alias")
    assert headers["X-Run-Id"] == "eval-b1-0123456789abcdef0123456789abcdef-state-alias"
    assert headers["X-Client-Request-Id"] == "request-1"
