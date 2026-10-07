"""Every run response carries ``usage_total``: the run's chat-call token total, never zeros for an unknown.

The product HTTP path runs with the Real adapters pointed at the fake upstream (FastAPI's
TestClient as the httpx client, no port).  The expected totals are summed from the usage
the fake upstream itself sent in each chat response, so they do not come from the stored
run the product reads.  The fake upstream counts tokens from the request and reply bytes,
so the calls of one run have different prompt and completion counts, and the calls before
and after a pause add up to different totals.
"""

from __future__ import annotations

from dataclasses import replace
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from queryshield.api.main import _usage_total, app, get_model_provider, get_retriever_source
from queryshield.approval import service as approval_service
from queryshield.agent.runtime import B0_PROFILE, B1_PROFILE, RuntimeDependencies
from queryshield.approval.service import reset_shared_state_stores, shared_run_service
from queryshield.auth.identity import resolve_identity
from queryshield.catalog import load_default_catalog
from queryshield.knowledge.index import build_embedding_index
from queryshield.knowledge.retrieval import HybridRetriever
from queryshield.knowledge.runtime import product_knowledge
from queryshield.providers.embedding import EmbeddingConfig, EmbeddingProviderError, OpenAICompatibleEmbedding
from queryshield.providers.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleModel
from scripts.fake_upstream import FAKE_UPSTREAM_MODEL
from scripts.fake_upstream import app as fake_upstream_app

BASE_URL = "http://fake-upstream/v1"
PLACEHOLDER_KEY = "not-a-real-key"
REQUESTER = "usage-a-requester"
APPROVER = "usage-a-approver"
TOKENS = ("prompt_tokens", "completion_tokens", "total_tokens")
UNKNOWN = {"status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
SEARCH = json.dumps({"type": "tool_call", "name": "search_catalog", "arguments": {"query": "退款后净额"}})
PAID_TOTAL = "2026年9月已支付订单总额是多少？"
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")


class _Upstream:
    """The fake upstream behind a client that keeps the usage of every chat response it sent."""

    def __init__(self) -> None:
        self.chat_usage: list[dict[str, int]] = []
        self.embedding_calls = 0
        self.client = TestClient(fake_upstream_app)
        self.client.event_hooks = {"response": [self._keep]}

    def _keep(self, response: httpx.Response) -> None:
        response.read()
        if response.request.url.path == "/v1/chat/completions":
            self.chat_usage.append(response.json()["usage"])
        elif response.request.url.path == "/v1/embeddings":
            self.embedding_calls += 1

    def model(self) -> OpenAICompatibleModel:
        return OpenAICompatibleModel(
            OpenAICompatibleConfig(base_url=BASE_URL, api_key=PLACEHOLDER_KEY, model=FAKE_UPSTREAM_MODEL), client=self.client
        )

    def embedding(self) -> OpenAICompatibleEmbedding:
        config = EmbeddingConfig(
            base_url=BASE_URL, api_key=PLACEHOLDER_KEY, model=FAKE_UPSTREAM_MODEL, model_revision=FAKE_UPSTREAM_MODEL, dimensions=128
        )
        return OpenAICompatibleEmbedding(config, client=self.client)


class _Steered:
    """The upstream model; it can drop one call's usage, and ask for a catalog search once ``search`` is set."""

    def __init__(self, inner: OpenAICompatibleModel) -> None:
        self.inner = inner
        self.calls = 0
        self.drop_usage_of_call: int | None = None
        self.search = False
        self.cancel_run: dict | None = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def complete(self, messages, **kwargs):
        result = self.inner.complete(messages, **kwargs)
        self.calls += 1
        if self.calls == self.drop_usage_of_call:
            result = replace(result, usage=None, usage_status="unknown")
        if self.search:
            result = replace(result, content=SEARCH)
        if self.cancel_run is not None:
            run = self.cancel_run
            shared_run_service().cancel(run_id=run["run_id"], identity={key: run[key] for key in ("tenant_id", "principal_id", "role")})
        return result


class _QueryEmbeddingFails:
    """The upstream embedding; raises once ``error`` is set (after the index is built)."""

    def __init__(self, inner: OpenAICompatibleEmbedding) -> None:
        self.inner = inner
        self.error: Exception | None = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def embed(self, inputs, **kwargs):
        if self.error is not None:
            raise self.error
        return self.inner.embed(inputs, **kwargs)


@pytest.fixture()
def upstream() -> _Upstream:
    return _Upstream()


@pytest.fixture()
def embedder(upstream) -> _QueryEmbeddingFails:
    return _QueryEmbeddingFails(upstream.embedding())


@pytest.fixture()
def model(upstream) -> _Steered:
    return _Steered(upstream.model())


@pytest.fixture()
def service(tmp_path, monkeypatch, upstream, embedder, model):
    monkeypatch.setenv("QUERYSHIELD_STATE_STORE_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    monkeypatch.delenv("QUERYSHIELD_AGENT_PROFILE", raising=False)
    monkeypatch.delenv("QUERYSHIELD_MODEL_PROTOCOL", raising=False)
    monkeypatch.delenv("QUERYSHIELD_RETRIEVAL", raising=False)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", REQUESTER)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_APPROVER", APPROVER)
    reset_shared_state_stores()
    build = build_embedding_index(product_knowledge(demo=False).snapshot, embedder, ingest_job_id="usage-total")
    retriever = HybridRetriever(catalog=load_default_catalog(), snapshot=build.snapshot, index=build.index, embedder=embedder)
    app.dependency_overrides[get_model_provider] = lambda: model
    app.dependency_overrides[get_retriever_source] = lambda: lambda: retriever
    upstream.chat_usage.clear()
    upstream.embedding_calls = 0
    # An unhandled server error must reach the test as a 500 response, as it would reach a client.
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    app.dependency_overrides.clear()
    reset_shared_state_stores()


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _ask(client: TestClient, question: str) -> dict:
    return client.post("/queries", headers=_auth(REQUESTER), json={"question": question}).json()


def _resume(client: TestClient, run_id: str) -> dict:
    return client.post(f"/runs/{run_id}/resume", headers=_auth(REQUESTER), json={"answer": "按支付金额统计"}).json()


def _approve(client: TestClient, pending: dict) -> dict:
    return client.post(
        f"/runs/{pending['run_id']}/approval",
        headers=_auth(APPROVER),
        json={"approval_id": pending["approval_id"], "decision": "approve"},
    ).json()


def _status_total(client: TestClient, run_id: str) -> dict:
    return client.get(f"/runs/{run_id}", headers=_auth(REQUESTER)).json()["usage_total"]


def _known(usages: list[dict[str, int]]) -> dict[str, object]:
    return {"status": "known", **{name: sum(usage[name] for usage in usages) for name in TOKENS}}


def _assert_distinct(usages: list[dict[str, int]]) -> None:
    """At least two calls, each with its own prompt and completion count."""

    assert len(usages) >= 2
    for name in ("prompt_tokens", "completion_tokens"):
        assert len({usage[name] for usage in usages}) == len(usages), name


def _embedding_error() -> EmbeddingProviderError:
    code = "upstream_http_error"
    return EmbeddingProviderError(code, {"status": "failed", "error_code": code, "operation_kind": "embedding"})


# --- the stored record, as the response shows it -----------------------------------------


@pytest.mark.parametrize(("stored", "expected"), [
    # The agent's summary: ``status``, and its own counts beside the totals.
    ({"status": "known", "prompt_tokens": 70, "completion_tokens": 9, "total_tokens": 79, "model_call_count": 2,
      "known_prompt_tokens": 70, "known_completion_tokens": 9, "known_total_tokens": 79},
     {"status": "known", "prompt_tokens": 70, "completion_tokens": 9, "total_tokens": 79}),
    ({"status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
      "known_prompt_tokens": 41, "known_completion_tokens": 6, "known_total_tokens": 47}, UNKNOWN),
    ({"status": "not_run", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
     {"status": "not_run", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
    # B0's record: ``usage_status``.
    ({"usage_status": "known", "prompt_tokens": 33, "completion_tokens": 5, "total_tokens": 38},
     {"status": "known", "prompt_tokens": 33, "completion_tokens": 5, "total_tokens": 38}),
    ({"usage_status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}, UNKNOWN),
    ({"usage_status": "not_run", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
     {"status": "not_run", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
    # Nothing stored yet, and totals that do not add up.
    (None, UNKNOWN),
    ({"status": "known", "prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 17}, UNKNOWN),
    ({"usage_status": "known", "prompt_tokens": 12, "completion_tokens": None, "total_tokens": 12}, UNKNOWN),
])
def test_the_stored_usage_of_either_shape_maps_to_one_public_total(stored, expected) -> None:
    assert _usage_total({"usage": stored}) == expected


# --- through the product path ---------------------------------------------------------


@pytest.mark.parametrize("protocol", ["json", "native"])
@pytest.mark.parametrize("question", [PAID_TOTAL, "退款后净额是怎么算的？"])
def test_a_sync_query_totals_the_usage_of_its_chat_calls(service, upstream, monkeypatch, protocol, question) -> None:
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", protocol)

    body = _ask(service, question)

    assert body["status"] == "SUCCEEDED"
    _assert_distinct(upstream.chat_usage)
    assert body["model_call_count"] == len(upstream.chat_usage)
    assert body["usage_total"] == _known(upstream.chat_usage)
    assert _status_total(service, body["run_id"]) == body["usage_total"]
    result = service.get(f"/runs/{body['run_id']}/result", headers=_auth(REQUESTER)).json()
    assert result["usage_total"] == body["usage_total"]


def test_the_query_embedding_of_a_run_is_not_in_its_total(service, upstream) -> None:
    body = _ask(service, "退款后净额是怎么算的？")

    assert upstream.embedding_calls >= 1, "the run embedded its question"
    assert body["usage_total"] == _known(upstream.chat_usage)


def test_the_total_is_the_sum_of_the_per_call_list_of_the_sync_response(service, upstream) -> None:
    body = _ask(service, PAID_TOTAL)

    entries = body["usage"]
    _assert_distinct([{name: entry[name] for name in TOKENS} for entry in entries])
    assert all(entry["usage_status"] == "known" for entry in entries)
    assert body["usage_total"] == {"status": "known", **{name: sum(entry[name] for entry in entries) for name in TOKENS}}


def test_one_call_without_usage_makes_the_total_unknown(service, model) -> None:
    model.drop_usage_of_call = 2

    body = _ask(service, PAID_TOTAL)

    assert body["status"] == "SUCCEEDED"
    assert [entry["usage_status"] for entry in body["usage"]].count("unknown") == 1
    assert body["usage_total"] == UNKNOWN
    assert _status_total(service, body["run_id"]) == UNKNOWN


def test_a_resumed_run_totals_the_calls_before_and_after_the_pause(service, upstream) -> None:
    waiting = _ask(service, "2026年9月销售额是多少？")
    assert waiting["status"] == "WAITING_USER"
    before = list(upstream.chat_usage)
    assert waiting["usage_total"] == _known(before)

    resumed = _resume(service, waiting["run_id"])

    assert resumed["status"] == "SUCCEEDED"
    after = upstream.chat_usage[len(before):]
    assert before and after
    assert _known(before)["total_tokens"] != _known(after)["total_tokens"]
    _assert_distinct(before + after)
    assert resumed["usage_total"] == _known(before + after)
    assert _status_total(service, waiting["run_id"]) == _known(before + after)


def test_a_run_that_raises_on_its_first_execution_totals_its_calls_before_the_error(service, upstream, model, embedder) -> None:
    model.search = True
    embedder.error = _embedding_error()

    body = _ask(service, PAID_TOTAL)

    assert (body["status"], body["error"]["code"]) == ("FAILED", "upstream_http_error")
    assert upstream.chat_usage, "a chat call was made before the error"
    assert body["usage_total"] == _known(upstream.chat_usage)
    assert _status_total(service, body["run_id"]) == _known(upstream.chat_usage)


def test_a_run_that_raises_while_resuming_totals_the_calls_before_and_after_the_pause(service, upstream, model, embedder) -> None:
    waiting = _ask(service, "2026年9月销售额是多少？")
    before = list(upstream.chat_usage)
    assert waiting["usage_total"] == _known(before)
    model.search = True
    embedder.error = _embedding_error()

    resumed = _resume(service, waiting["run_id"])

    assert (resumed["status"], resumed["error"]["code"]) == ("FAILED", "upstream_http_error")
    assert before and upstream.chat_usage[len(before):], "calls before the pause and after it"
    assert resumed["usage_total"] == _known(upstream.chat_usage)
    assert _status_total(service, waiting["run_id"]) == _known(upstream.chat_usage)
    # The stored record says so too: the evaluation reads its status.
    assert shared_run_service().store.get_run(waiting["run_id"])["usage"]["status"] == "known"


def test_a_run_cancelled_during_a_failing_resume_totals_its_calls(service, upstream, model, embedder) -> None:
    waiting = _ask(service, "2026年9月销售额是多少？")
    before = list(upstream.chat_usage)
    assert waiting["usage_total"] == _known(before)
    model.search = True
    model.cancel_run = waiting
    embedder.error = _embedding_error()

    resumed = _resume(service, waiting["run_id"])

    assert resumed["status"] == "CANCELLED"
    assert upstream.chat_usage[len(before):], "a call after the pause"
    assert resumed["usage_total"] == _known(upstream.chat_usage)
    assert _status_total(service, waiting["run_id"]) == _known(upstream.chat_usage)


def test_an_approved_execution_keeps_the_total_of_the_pause(service, upstream) -> None:
    pending = _ask(service, "查询客户姓名")
    assert pending["status"] == "WAITING_APPROVAL"
    paused = _known(upstream.chat_usage)
    assert pending["usage_total"] == paused
    calls = len(upstream.chat_usage)

    approved = _approve(service, pending)

    assert approved["status"] == "SUCCEEDED"
    assert len(upstream.chat_usage) == calls, "an approved execution calls no model"
    assert approved["usage_total"] == paused
    assert _status_total(service, pending["run_id"]) == paused


def test_an_approved_execution_that_raises_keeps_the_total_of_the_pause(service, upstream, monkeypatch) -> None:
    pending = _ask(service, "查询客户姓名")
    paused = _known(upstream.chat_usage)
    assert pending["usage_total"] == paused

    def raises(*args, **kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(approval_service, "execute_approved_query", raises)
    approved = _approve(service, pending)

    assert approved["error"]["code"] == "execution_failed"
    assert _status_total(service, pending["run_id"]) == paused


def test_the_b0_run_reports_its_one_call(service, upstream, monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "b0")

    body = _ask(service, PAID_TOTAL)

    assert body["status"] == "SUCCEEDED"
    assert len(upstream.chat_usage) == 1
    assert body["usage_total"] == _known(upstream.chat_usage)
    assert _status_total(service, body["run_id"]) == body["usage_total"]


@pytest.mark.parametrize("profile", [B1_PROFILE, B0_PROFILE])
def test_a_run_the_agent_refuses_before_any_model_call_reports_zeros(service, upstream, model, profile) -> None:
    # HTTP refuses a question naming another tenant before it starts a run; the agents
    # refuse it again on their own, and that run is what this reads back.
    run_service = shared_run_service()
    deps = RuntimeDependencies(model=model, executor=run_service.new_executor(), retriever=None, call_store=None, profile=profile)
    run = run_service.run_sync(identity=resolve_identity(f"Bearer {REQUESTER}"), question="tenant-B 的支付金额", deps=deps)

    assert (run["status"], run["model_call_count"]) == ("DENIED", 0)
    assert upstream.chat_usage == []
    assert _status_total(service, run["run_id"]) == {"status": "not_run", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_a_fake_model_run_has_an_unknown_total(service) -> None:
    app.dependency_overrides.pop(get_model_provider)

    body = _ask(service, PAID_TOTAL)

    assert body["status"] == "SUCCEEDED" and body["model_call_count"] >= 1
    assert body["usage_total"] == UNKNOWN
