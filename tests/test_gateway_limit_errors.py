"""A model gateway's quota and rate limits end a run on their own error codes, with HTTP 503.

Only HTTP 429 with the gateway's ``quota_exhausted`` or ``rate_limited`` code is mapped;
any other 429 and the same codes under another status stay ``upstream_http_error``.
The chat and the embedding adapter read the error the same way.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from queryshield.agent.runtime import FAILED_ERROR_HTTP, http_status_for_run
from queryshield.api.main import app, get_model_provider, get_retriever_source
from queryshield.approval.service import reset_shared_state_stores
from queryshield.catalog import load_default_catalog
from queryshield.knowledge.index import build_embedding_index
from queryshield.knowledge.retrieval import HybridRetriever
from queryshield.knowledge.runtime import product_knowledge
from queryshield.providers.contracts import ModelProviderError
from queryshield.providers.embedding import EmbeddingConfig, EmbeddingProviderError, OpenAICompatibleEmbedding
from queryshield.providers.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleModel
from scripts.fake_upstream import FAKE_UPSTREAM_MODEL
from scripts.fake_upstream import app as fake_upstream_app

BASE_URL = "http://gateway.test/v1"
KEY = "not-a-real-key"
REQUESTER = "limit-a-requester"
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")


def _error(status: int, body: object) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body) if body is not None else httpx.Response(status, text="busy")

    return httpx.Client(transport=httpx.MockTransport(handler))


def _chat(client: httpx.Client) -> OpenAICompatibleModel:
    return OpenAICompatibleModel(OpenAICompatibleConfig(base_url=BASE_URL, api_key=KEY, model="gateway-chat"), client=client)


def _embedding(client: httpx.Client) -> OpenAICompatibleEmbedding:
    config = EmbeddingConfig(base_url=BASE_URL, api_key=KEY, model="gateway-embed", model_revision="r1", dimensions=128)
    return OpenAICompatibleEmbedding(config, client=client)


def _chat_failure(client: httpx.Client) -> ModelProviderError:
    with pytest.raises(ModelProviderError) as caught:
        _chat(client).complete([{"role": "user", "content": "hi"}])
    return caught.value


def _embedding_failure(client: httpx.Client) -> EmbeddingProviderError:
    with pytest.raises(EmbeddingProviderError) as caught:
        _embedding(client).embed(["hi"])
    return caught.value


CASES = [
    # (status, body, run error code, recorded provider error code)
    (429, {"error": {"code": "quota_exhausted", "message": "quota used up"}}, "model_quota_exhausted", "quota_exhausted"),
    (429, {"error": {"code": "rate_limited", "message": "slow down"}}, "model_rate_limited", "rate_limited"),
    # The shared reader also takes a top-level code; one rule for both adapters.
    (429, {"code": "rate_limited"}, "model_rate_limited", "rate_limited"),
    # Both present and different: error.code wins.
    (429, {"error": {"code": "quota_exhausted"}, "code": "rate_limited"}, "model_quota_exhausted", "quota_exhausted"),
    # Another provider's 429, or no code at all: its meaning is not agreed, so it stays generic.
    (429, {"error": {"code": "Throttling.RateQuota"}}, "upstream_http_error", "Throttling.RateQuota"),
    (429, {"error": {"code": "QUOTA_EXHAUSTED"}}, "upstream_http_error", "QUOTA_EXHAUSTED"),
    (429, {"error": {"message": "too many requests"}}, "upstream_http_error", None),
    (429, None, "upstream_http_error", None),
    # The gateway's codes under any other status are not its limits.
    (428, {"error": {"code": "rate_limited"}}, "upstream_http_error", "rate_limited"),
    (430, {"error": {"code": "quota_exhausted"}}, "upstream_http_error", "quota_exhausted"),
    (400, {"error": {"code": "quota_exhausted"}}, "upstream_http_error", "quota_exhausted"),
    (503, {"error": {"code": "rate_limited"}}, "upstream_http_error", "rate_limited"),
]


@pytest.mark.parametrize("failure", [_chat_failure, _embedding_failure], ids=["chat", "embedding"])
@pytest.mark.parametrize(("status", "body", "code", "provider_code"), CASES)
def test_only_the_gateway_limits_get_their_own_codes(failure, status, body, code, provider_code) -> None:
    error = failure(_error(status, body))

    assert error.code == code
    assert error.record["error_code"] == code
    assert error.record["http_status"] == status
    assert error.record.get("provider_error_code") == provider_code
    assert "quota used up" not in repr(error.record) and "slow down" not in repr(error.record)


def test_both_limit_codes_end_a_failed_run_with_503() -> None:
    assert FAILED_ERROR_HTTP["model_quota_exhausted"] == 503
    assert FAILED_ERROR_HTTP["model_rate_limited"] == 503
    assert http_status_for_run("FAILED", "model_quota_exhausted") == 503
    assert http_status_for_run("FAILED", "model_rate_limited") == 503
    assert http_status_for_run("FAILED", "upstream_http_error") == 502


# --- through the product HTTP path ----------------------------------------------------


def _limit(code: str) -> dict:
    return {"error": {"code": code, "message": "limit", "type": "rate_limit_error"}}


@pytest.fixture()
def http(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYSHIELD_STATE_STORE_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    monkeypatch.delenv("QUERYSHIELD_AGENT_PROFILE", raising=False)
    monkeypatch.delenv("QUERYSHIELD_MODEL_PROTOCOL", raising=False)
    monkeypatch.delenv("QUERYSHIELD_RETRIEVAL", raising=False)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", REQUESTER)
    reset_shared_state_stores()
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
    reset_shared_state_stores()


def _ask(client: TestClient, question: str) -> httpx.Response:
    return client.post("/queries", headers={"Authorization": f"Bearer {REQUESTER}"}, json={"question": question})


@pytest.mark.parametrize("profile", ["b0", "b1"])
@pytest.mark.parametrize(("gateway_code", "run_code"), [("quota_exhausted", "model_quota_exhausted"), ("rate_limited", "model_rate_limited")])
def test_a_limited_model_call_fails_the_sync_query_with_503(http, monkeypatch, profile, gateway_code, run_code) -> None:
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", profile)
    model = _chat(_error(429, _limit(gateway_code)))
    app.dependency_overrides[get_model_provider] = lambda: model

    response = _ask(http, "2026年9月已支付订单总额是多少？")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "FAILED"
    assert body["error"]["code"] == run_code


@pytest.mark.parametrize(("gateway_code", "run_code"), [("quota_exhausted", "model_quota_exhausted"), ("rate_limited", "model_rate_limited")])
def test_a_limited_query_embedding_fails_the_sync_query_with_503(http, gateway_code, run_code) -> None:
    """The index is built through the fake upstream; then the gateway limits the run's query embedding."""

    upstream = TestClient(fake_upstream_app)
    limited = {"on": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if limited["on"] and request.url.path.endswith("/embeddings"):
            return httpx.Response(429, json=_limit(gateway_code))
        forwarded = upstream.request(request.method, request.url.path, content=request.content, headers=request.headers)
        return httpx.Response(forwarded.status_code, content=forwarded.content, headers=forwarded.headers)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    config = EmbeddingConfig(
        base_url=BASE_URL, api_key=KEY, model=FAKE_UPSTREAM_MODEL, model_revision=FAKE_UPSTREAM_MODEL, dimensions=128
    )
    embedder = OpenAICompatibleEmbedding(config, client=client)
    build = build_embedding_index(product_knowledge(demo=False).snapshot, embedder, ingest_job_id="limit")
    retriever = HybridRetriever(catalog=load_default_catalog(), snapshot=build.snapshot, index=build.index, embedder=embedder)
    model = OpenAICompatibleModel(OpenAICompatibleConfig(base_url=BASE_URL, api_key=KEY, model=FAKE_UPSTREAM_MODEL), client=client)
    app.dependency_overrides[get_model_provider] = lambda: model
    app.dependency_overrides[get_retriever_source] = lambda: lambda: retriever
    limited["on"] = True

    response = _ask(http, "退款后净额是怎么算的？")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == run_code
