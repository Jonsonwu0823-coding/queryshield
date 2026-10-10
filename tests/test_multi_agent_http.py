"""B2 over HTTP with the Fake model: sync, async, the event stream and the server-only profile setting."""

from __future__ import annotations

import json

import pytest

from queryshield.approval.service import shared_run_service

from multi_agent_support import COMPOSITE
from test_http_queries import REQUESTER, ask, auth, env, wait  # noqa: F401  (env is a fixture)


@pytest.fixture()
def b2_http(env, monkeypatch):  # noqa: F811
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "b2")
    return env


def _frames(client, run_id: str, **headers) -> list[dict]:
    with client.stream("GET", f"/runs/{run_id}/events", headers={**auth(REQUESTER), **headers}) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())
    return [json.loads(line[len("data: "):]) for line in text.splitlines() if line.startswith("data: ")]


def _check_answer(body: dict) -> None:
    assert body["status"] == "SUCCEEDED" and body["profile"] == "B2-multi-agent"
    assert sorted(fact["metric_id"] for fact in body["facts"]["facts"]) == ["gross_fen", "net_fen"]
    assert body["answer_status"] == "verified"


def test_a_sync_request_runs_the_coordinator_and_its_sub_agents(b2_http) -> None:
    response = ask(b2_http, COMPOSITE)
    assert response.status_code == 200
    body = response.json()
    _check_answer(body)
    # The coordinator searches first (this server has a retriever), delegates, then answers.
    assert (body["model_call_count"], body["tool_call_count"]) == (5, 3)


def test_an_async_run_streams_every_agents_steps_with_its_label_once(b2_http) -> None:
    response = ask(b2_http, COMPOSITE, asynchronous=True)
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert wait(b2_http, run_id, {"SUCCEEDED", "FAILED"})["status"] == "SUCCEEDED"
    result = b2_http.get(f"/runs/{run_id}/result", headers=auth(REQUESTER)).json()
    _check_answer({**result, "profile": shared_run_service().store.get_run(run_id)["run_config"]["profile"]})
    frames = _frames(b2_http, run_id)
    stored = shared_run_service().store.events(run_id, after_event_id=0, limit=1000)
    assert [f["event_id"] for f in frames] == [e["event_id"] for e in stored]
    agents = [f["step"].get("agent") for f in frames if f["type"] == "agent_step"]
    assert set(agents) == {"coordinator", "subtask-1", "subtask-2"}
    assert frames[-1]["type"] == "terminal"
    # Resuming after any event gives exactly the rest, whatever agent wrote it.
    middle = frames[len(frames) // 2]["event_id"]
    rest = _frames(b2_http, run_id, **{"Last-Event-ID": str(middle)})
    assert [f["event_id"] for f in rest] == [f["event_id"] for f in frames if f["event_id"] > middle]


@pytest.mark.parametrize("setting", ["B2", "b2-multi-agent", "B2-multi-agent"])
def test_the_profile_names_the_server_accepts(env, monkeypatch, setting) -> None:  # noqa: F811
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", setting)
    _check_answer(ask(env, COMPOSITE).json())


def test_the_native_protocol_with_b2_is_refused_before_any_run(b2_http, monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", "native")
    response = ask(b2_http, COMPOSITE)
    assert response.status_code == 503 and response.json()["error"]["code"] == "invalid_model_protocol"
    assert shared_run_service().store._connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


def test_the_single_proposal_entry_refuses_delegate_as_before(b2_http) -> None:
    proposal = json.dumps({"type": "delegate", "subtasks": [{"metrics": ["gross_fen"], "time_window": {}}, {"metrics": ["net_fen"], "time_window": {}}]})
    response = b2_http.post("/query-proposals", headers=auth(REQUESTER), json={"proposal": proposal})
    assert response.status_code == 422 and response.json()["error"]["code"] == "unknown_field"


# --- the server retriever and the metadata setting -------------------------------------------------


class _RecordingRetriever:
    """The product retriever, recording each search's strategy, run and returned ids."""

    def __init__(self, retriever) -> None:
        self.retriever = retriever
        self.searches: list[tuple[str, str, list]] = []

    def search(self, query, *, context, top_k):
        result = self.retriever.search(query, context=context, top_k=top_k)
        self.searches.append((context.run_id, result.evidence.strategy_version, [item.get("id") for item in result.items]))
        return result

    def __getattr__(self, name):
        return getattr(self.retriever, name)


def _source_of(spy):
    return lambda: (lambda: spy)  # no parameters: FastAPI would read them from the request


def test_the_coordinator_searches_with_the_servers_hybrid_retriever_like_b1(env, monkeypatch) -> None:  # noqa: F811
    from queryshield.agent.runtime import product_retriever
    from queryshield.api.main import app, get_retriever_source
    from queryshield.knowledge.retrieval import HybridRetriever

    hybrid = product_retriever("fake")
    assert isinstance(hybrid, HybridRetriever)
    recorded = {}
    for profile in ("b1", "b2"):
        spy = _RecordingRetriever(hybrid)
        app.dependency_overrides[get_retriever_source] = _source_of(spy)
        monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", profile)
        body = ask(env, COMPOSITE).json()
        assert body["status"] == "SUCCEEDED"
        assert [run_id for run_id, _, _ in spy.searches] == [body["run_id"]]
        recorded[profile] = spy.searches[0][1:]
    strategy, items = recorded["b2"]
    assert recorded["b1"] == recorded["b2"] and strategy and items


def test_a_b2_run_waiting_for_the_user_is_not_resumed_once_the_server_uses_mcp(b2_http, monkeypatch) -> None:
    waiting = ask(b2_http, "2026年9月的销售额和已支付订单数分别是多少？").json()
    assert waiting["status"] == "WAITING_USER"
    store = shared_run_service().store
    before = store.events(waiting["run_id"], after_event_id=0, limit=1000)
    monkeypatch.setenv("QUERYSHIELD_METADATA_TOOLS", "mcp")
    response = b2_http.post(f"/runs/{waiting['run_id']}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额统计"})
    assert response.status_code == 503 and response.json()["error"]["code"] == "invalid_metadata_tools_configuration"
    assert store.get_run(waiting["run_id"])["status"] == "WAITING_USER"
    assert store.events(waiting["run_id"], after_event_id=0, limit=1000) == before
