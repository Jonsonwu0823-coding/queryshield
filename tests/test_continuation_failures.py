"""A run whose execution fails ends on its error code: first execution, resume and approved execution alike.

Whatever the execution raises, the run ends FAILED on the error's code, with the HTTP
code of the shared table, the SQL it already ran counted, and CANCELLED if it was
cancelled meanwhile.  A resume (after the user's answer) or an approved execution
(after the approver's decision) must never answer 500 and stay waiting: an approval is
consumed before it executes, so a run left in WAITING_APPROVAL could never run again.
"""

from __future__ import annotations

from dataclasses import replace
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from queryshield.agent.runtime import http_status_for_run
from queryshield.api.main import app, get_model_provider, get_retriever_source
from queryshield.approval import service as approval_service
from queryshield.approval.service import pending_call_from_action, reset_shared_state_stores, shared_run_service
from queryshield.catalog import load_default_catalog
from queryshield.knowledge.index import build_embedding_index
from queryshield.knowledge.retrieval import HybridRetriever
from queryshield.knowledge.runtime import HashFeatureEmbedding, product_knowledge
from queryshield.providers.embedding import EmbeddingConfig, EmbeddingProviderError, OpenAICompatibleEmbedding
from queryshield.providers.fake_model import FakeModel
from scripts.fake_upstream import FAKE_UPSTREAM_MODEL
from scripts.fake_upstream import app as fake_upstream_app

REQUESTER = "pause-a-requester"
APPROVER = "pause-a-approver"
SEARCH = json.dumps({"type": "tool_call", "name": "search_catalog", "arguments": {"query": "退款后净额"}})
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYSHIELD_STATE_STORE_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    monkeypatch.delenv("QUERYSHIELD_AGENT_PROFILE", raising=False)
    monkeypatch.delenv("QUERYSHIELD_MODEL_PROTOCOL", raising=False)
    monkeypatch.delenv("QUERYSHIELD_RETRIEVAL", raising=False)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", REQUESTER)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_APPROVER", APPROVER)
    reset_shared_state_stores()
    # An unhandled server error must reach the test as a 500 response, as it would reach a client.
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    app.dependency_overrides.clear()
    reset_shared_state_stores()


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _ask(client: TestClient, question: str) -> dict:
    return client.post("/queries", headers=_auth(REQUESTER), json={"question": question}).json()


def _run(run_id: str) -> dict:
    return shared_run_service().store.get_run(run_id)


def _terminal_events(run_id: str) -> list[tuple[str, object]]:
    events = shared_run_service().store.events(run_id, after_event_id=0, limit=1000)
    return [(event["status"], (event["payload"] or {}).get("error_code")) for event in events if event["type"] == "terminal"]


def _cancel(run: dict) -> None:
    shared_run_service().cancel(
        run_id=run["run_id"], identity={"tenant_id": run["tenant_id"], "principal_id": run["principal_id"], "role": run["role"]}
    )


# --- resume ----------------------------------------------------------------------------


class _SearchesAfterResume(FakeModel):
    """The Fake model until ``search`` is set; then it asks for a catalog search (an embedding call)."""

    def __init__(self) -> None:
        super().__init__()
        self.search = False
        self.cancel_run: dict | None = None

    def complete(self, messages, **kwargs):
        result = super().complete(messages, **kwargs)
        if not self.search:
            return result
        if self.cancel_run is not None:
            _cancel(self.cancel_run)
        return replace(result, content=SEARCH)


class _QueryEmbeddingFails(HashFeatureEmbedding):
    """Embeds the index; raises ``error`` once it is set."""

    error: Exception | None = None

    def embed(self, inputs, **kwargs):
        if self.error is not None:
            raise self.error
        return super().embed(inputs, **kwargs)


def _retriever(embedder) -> HybridRetriever:
    build = build_embedding_index(product_knowledge(demo=False).snapshot, embedder, ingest_job_id="continuation")
    return HybridRetriever(catalog=load_default_catalog(), snapshot=build.snapshot, index=build.index, embedder=embedder)


def _waiting_user(env, model, embedder) -> dict:
    retriever = _retriever(embedder)
    app.dependency_overrides[get_model_provider] = lambda: model
    app.dependency_overrides[get_retriever_source] = lambda: lambda: retriever
    waiting = _ask(env, "2026年9月销售额是多少？")
    assert waiting["status"] == "WAITING_USER"
    model.search = True
    return waiting


def _resume(env, run_id: str) -> httpx.Response:
    return env.post(f"/runs/{run_id}/resume", headers=_auth(REQUESTER), json={"answer": "按支付金额统计"})


def _provider_error(code: str) -> EmbeddingProviderError:
    return EmbeddingProviderError(code, {"status": "failed", "error_code": code, "operation_kind": "embedding"})


@pytest.mark.parametrize("code", ["upstream_http_error", "upstream_timeout", "model_rate_limited"])
def test_a_provider_failure_while_resuming_ends_the_run_on_its_code(env, code) -> None:
    model, embedder = _SearchesAfterResume(), _QueryEmbeddingFails()
    waiting = _waiting_user(env, model, embedder)
    embedder.error = _provider_error(code)

    response = _resume(env, waiting["run_id"])

    assert response.status_code == http_status_for_run("FAILED", code)
    body = response.json()
    assert (body["status"], body["error"]["code"]) == ("FAILED", code)
    run = _run(waiting["run_id"])
    assert (run["status"], run["error_code"]) == ("FAILED", code)
    assert _terminal_events(waiting["run_id"]) == [("FAILED", code)]
    # Once ended, the run cannot be resumed again.
    assert _resume(env, waiting["run_id"]).status_code == 409


def test_an_exception_without_a_code_while_resuming_ends_the_run_as_execution_failed(env) -> None:
    model, embedder = _SearchesAfterResume(), _QueryEmbeddingFails()
    waiting = _waiting_user(env, model, embedder)
    embedder.error = RuntimeError("socket closed")

    response = _resume(env, waiting["run_id"])

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "execution_failed"
    assert _run(waiting["run_id"])["status"] == "FAILED"


def test_a_run_cancelled_during_a_failing_resume_stays_cancelled(env) -> None:
    model, embedder = _SearchesAfterResume(), _QueryEmbeddingFails()
    waiting = _waiting_user(env, model, embedder)
    model.cancel_run = _run(waiting["run_id"])
    embedder.error = _provider_error("upstream_http_error")

    response = _resume(env, waiting["run_id"])

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "run_cancelled"
    assert _run(waiting["run_id"])["status"] == "CANCELLED"
    assert "FAILED" not in [state for state, _ in _terminal_events(waiting["run_id"])]


def test_the_gateway_limiting_the_query_embedding_of_a_resume_fails_it_with_503(env) -> None:
    upstream = TestClient(fake_upstream_app)
    limited = {"on": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if limited["on"]:
            return httpx.Response(429, json={"error": {"code": "rate_limited", "message": "limit"}})
        forwarded = upstream.request(request.method, request.url.path, content=request.content, headers=request.headers)
        return httpx.Response(forwarded.status_code, content=forwarded.content, headers=forwarded.headers)

    config = EmbeddingConfig(
        base_url="http://gateway.test/v1", api_key="not-a-real-key", model=FAKE_UPSTREAM_MODEL,
        model_revision=FAKE_UPSTREAM_MODEL, dimensions=128,
    )
    embedder = OpenAICompatibleEmbedding(config, client=httpx.Client(transport=httpx.MockTransport(handler)))
    waiting = _waiting_user(env, _SearchesAfterResume(), embedder)
    limited["on"] = True

    response = _resume(env, waiting["run_id"])

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_rate_limited"
    assert _run(waiting["run_id"])["status"] == "FAILED"


# --- approved execution ------------------------------------------------------------------


class _Unavailable(Exception):
    code = "database_unavailable"


def _waiting_approval(env) -> dict:
    pending = _ask(env, "查询客户姓名")
    assert pending["status"] == "WAITING_APPROVAL"
    return pending


def _approve(env, pending: dict) -> httpx.Response:
    return env.post(
        f"/runs/{pending['run_id']}/approval",
        headers=_auth(APPROVER),
        json={"approval_id": pending["approval_id"], "decision": "approve"},
    )


@pytest.mark.parametrize(("error", "code", "http_status"), [
    (RuntimeError("connection reset"), "execution_failed", 502),
    (_Unavailable("down"), "database_unavailable", 503),
])
def test_an_approved_execution_that_raises_ends_the_run_on_its_code(env, monkeypatch, error, code, http_status) -> None:
    pending = _waiting_approval(env)

    def raises(*args, **kwargs):
        raise error

    monkeypatch.setattr(approval_service, "execute_approved_query", raises)
    response = _approve(env, pending)

    assert response.status_code == http_status
    assert response.json()["error"]["code"] == code
    run = _run(pending["run_id"])
    assert (run["status"], run["error_code"], run["sql_exec_count"]) == ("FAILED", code, 0)
    assert _terminal_events(pending["run_id"]) == [("FAILED", code)]
    # The consumed approval is a replay now: it reports the ended run and executes nothing.
    replay = _approve(env, pending)
    assert replay.status_code == http_status
    assert _run(pending["run_id"])["status"] == "FAILED"


def test_sql_already_executed_by_a_failing_approved_execution_is_counted(env, monkeypatch) -> None:
    pending = _waiting_approval(env)
    executed = approval_service.execute_approved_query

    def executes_then_raises(*args, **kwargs):
        executed(*args, **kwargs)
        raise RuntimeError("lost after the query")

    monkeypatch.setattr(approval_service, "execute_approved_query", executes_then_raises)
    response = _approve(env, pending)

    assert response.status_code == 502
    run = _run(pending["run_id"])
    assert (run["status"], run["sql_exec_count"]) == ("FAILED", 1)


def test_a_run_cancelled_during_a_failing_approved_execution_stays_cancelled(env, monkeypatch) -> None:
    pending = _waiting_approval(env)

    def cancelled_then_raises(*args, **kwargs):
        _cancel(_run(pending["run_id"]))
        raise RuntimeError("connection reset")

    monkeypatch.setattr(approval_service, "execute_approved_query", cancelled_then_raises)
    response = _approve(env, pending)

    assert response.status_code == 200
    assert response.json()["status"] == "CANCELLED"
    assert _run(pending["run_id"])["status"] == "CANCELLED"
    assert "FAILED" not in [state for state, _ in _terminal_events(pending["run_id"])]


QUERY = json.dumps({
    "type": "tool_call", "name": "query_readonly",
    "arguments": {
        "sql": "SELECT COALESCE(SUM(o.amount_fen), 0) AS gross_fen FROM orders AS o "
               "WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s",
        "params": {"0": "paid", "1": "2026-09-01T00:00:00Z", "2": "2026-10-01T00:00:00Z"},
        "metrics": ["gross_fen"], "time_window": {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"},
    },
}, ensure_ascii=False)


class _QueriesThenSearches(_SearchesAfterResume):
    """After the resume: one query (SQL runs), then a catalog search."""

    resumed_calls = 0

    def complete(self, messages, **kwargs):
        result = FakeModel.complete(self, messages, **kwargs)
        if not self.search:
            return result
        self.resumed_calls += 1
        return replace(result, content=QUERY if self.resumed_calls == 1 else SEARCH)


def test_sql_already_executed_by_a_failing_resume_is_counted(env) -> None:
    model, embedder = _QueriesThenSearches(), _QueryEmbeddingFails()
    waiting = _waiting_user(env, model, embedder)
    before = _run(waiting["run_id"])["sql_exec_count"]
    embedder.error = _provider_error("upstream_http_error")

    response = _resume(env, waiting["run_id"])

    assert response.status_code == 502
    assert model.resumed_calls == 2, "the resumed run queried, then searched"
    run = _run(waiting["run_id"])
    assert (run["status"], run["sql_exec_count"]) == ("FAILED", before + 1)


# --- first execution ---------------------------------------------------------------------


class _FirstRunModel(FakeModel):
    """A first run that queries (SQL runs) or asks for the cancel, then searches; the search's embedding fails."""

    def __init__(self, *, query_first: bool = False, cancel_first: bool = False) -> None:
        super().__init__()
        self.query_first = query_first
        self.cancel_first = cancel_first
        self.calls = 0

    def complete(self, messages, *, run_id=None, **kwargs):
        result = super().complete(messages, run_id=run_id, **kwargs)
        self.calls += 1
        if self.cancel_first and self.calls == 1:
            _cancel(_run(run_id))
        return replace(result, content=QUERY if self.query_first and self.calls == 1 else SEARCH)


def _first_run(env, model: _FirstRunModel) -> tuple[httpx.Response, str]:
    embedder = _QueryEmbeddingFails()
    retriever = _retriever(embedder)
    embedder.error = _provider_error("upstream_http_error")
    app.dependency_overrides[get_model_provider] = lambda: model
    app.dependency_overrides[get_retriever_source] = lambda: lambda: retriever
    response = env.post("/queries", headers=_auth(REQUESTER), json={"question": "2026年9月已支付订单总额是多少？"})
    return response, response.json()["run_id"]


def test_a_first_run_cancelled_before_it_fails_stays_cancelled(env) -> None:
    model = _FirstRunModel(cancel_first=True)

    response, run_id = _first_run(env, model)

    assert model.calls == 1, "the run asked to cancel, then searched"
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "run_cancelled"
    assert _run(run_id)["status"] == "CANCELLED"
    assert "FAILED" not in [state for state, _ in _terminal_events(run_id)]


def test_sql_already_executed_by_a_failing_first_run_is_counted(env) -> None:
    model = _FirstRunModel(query_first=True)

    response, run_id = _first_run(env, model)

    assert model.calls == 2, "the run queried, then searched"
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_http_error"
    run = _run(run_id)
    assert (run["status"], run["error_code"], run["sql_exec_count"]) == ("FAILED", "upstream_http_error", 1)


# --- SQL run before the pause and during the failing execution ------------------------------
# Every count here has a non-zero part from before the pause and one from this execution
# where the scenario has both, so dropping either part changes the result.

QUESTION = "2026年9月已支付订单总额是多少？"
ASK = json.dumps({"type": "ask_user", "question": "请问要按哪个币种展示？"}, ensure_ascii=False)


class _ScriptedReplies(FakeModel):
    """Replies with the given actions in order; on call ``cancel_at`` it first asks for the run's cancel."""

    def __init__(self, replies: list[str], *, cancel_at: int | None = None) -> None:
        super().__init__()
        self.replies = replies
        self.cancel_at = cancel_at
        self.calls = 0

    def complete(self, messages, *, run_id=None, **kwargs):
        result = super().complete(messages, run_id=run_id, **kwargs)
        self.calls += 1
        if self.calls == self.cancel_at:
            _cancel(_run(run_id))
        return replace(result, content=self.replies[self.calls - 1])


def _use(model: FakeModel) -> _QueryEmbeddingFails:
    embedder = _QueryEmbeddingFails()
    retriever = _retriever(embedder)
    app.dependency_overrides[get_model_provider] = lambda: model
    app.dependency_overrides[get_retriever_source] = lambda: lambda: retriever
    return embedder


def _sensitive_query(env) -> str:
    """The customer-name query the Fake model proposes, written as a model action."""

    pending = _waiting_approval(env)
    call = pending_call_from_action(shared_run_service().store.get_approval(pending["approval_id"])["action"])
    arguments = {key: value for key, value in call.items() if key != "tool" and value is not None}
    return json.dumps({"type": "tool_call", "name": call["tool"], "arguments": arguments}, ensure_ascii=False)


def _paused_after_one_query(env, status: str) -> dict:
    paused = _ask(env, QUESTION)
    assert (paused["status"], _run(paused["run_id"])["sql_exec_count"]) == (status, 1)
    return paused


def test_a_failing_resume_counts_the_sql_from_before_the_pause_and_its_own(env) -> None:
    model = _ScriptedReplies([QUERY, ASK, QUERY, SEARCH])
    embedder = _use(model)
    waiting = _paused_after_one_query(env, "WAITING_USER")
    embedder.error = _provider_error("upstream_http_error")

    response = _resume(env, waiting["run_id"])

    assert model.calls == 4, "queried and asked, then after the resume queried and searched"
    assert response.status_code == 502
    run = _run(waiting["run_id"])
    assert (run["status"], run["error_code"], run["sql_exec_count"]) == ("FAILED", "upstream_http_error", 2)


def test_a_failing_approved_execution_counts_the_sql_from_before_the_pause_and_its_own(env, monkeypatch) -> None:
    model = _ScriptedReplies([QUERY, _sensitive_query(env)])
    _use(model)
    pending = _paused_after_one_query(env, "WAITING_APPROVAL")
    executed = approval_service.execute_approved_query

    def executes_then_raises(*args, **kwargs):
        executed(*args, **kwargs)
        raise RuntimeError("lost after the query")

    monkeypatch.setattr(approval_service, "execute_approved_query", executes_then_raises)
    response = _approve(env, pending)

    assert response.status_code == 502
    run = _run(pending["run_id"])
    assert (run["status"], run["error_code"], run["sql_exec_count"]) == ("FAILED", "execution_failed", 2)


def test_a_run_cancelled_after_it_ran_sql_keeps_that_count(env) -> None:
    model = _ScriptedReplies([QUERY, SEARCH], cancel_at=2)
    embedder = _use(model)
    embedder.error = _provider_error("upstream_http_error")

    response = env.post("/queries", headers=_auth(REQUESTER), json={"question": QUESTION})

    assert model.calls == 2, "queried, then asked for the cancel and searched"
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "run_cancelled"
    run = _run(response.json()["run_id"])
    assert (run["status"], run["sql_exec_count"]) == ("CANCELLED", 1)


def test_an_approved_execution_whose_executor_cannot_be_built_ends_the_run(env, monkeypatch) -> None:
    model = _ScriptedReplies([QUERY, _sensitive_query(env)])
    _use(model)
    pending = _paused_after_one_query(env, "WAITING_APPROVAL")

    def unavailable():
        raise _Unavailable("down")

    monkeypatch.setattr(shared_run_service(), "_executor_factory", unavailable)
    response = _approve(env, pending)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "database_unavailable"
    run = _run(pending["run_id"])
    assert (run["status"], run["error_code"], run["sql_exec_count"]) == ("FAILED", "database_unavailable", 1)
    # The consumed approval only replays the ended run.
    assert _approve(env, pending).status_code == 503
    assert _run(pending["run_id"])["status"] == "FAILED"
