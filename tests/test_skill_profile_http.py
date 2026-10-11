"""The skills profile over HTTP with the Fake model: sync, async, the event stream, resume, approval,
the server-only setting, and every model call tagged with the root run id."""

from __future__ import annotations

import json

import pytest

from queryshield.agent import runtime
from queryshield.agent.skills import load_skill_library
from queryshield.api.main import app, get_model_provider
from queryshield.approval.service import shared_run_service

from multi_agent_support import COMPOSITE
from test_http_queries import APPROVER, REQUESTER, Scripted, ask, auth, env, wait  # noqa: F401  (env is a fixture)
from test_multi_agent_http import _frames
from test_run_id_header import _ask, _assert_tagged, retriever, service, upstream  # noqa: F401  (fixtures)
from test_skill_files import BAD_FILES, fresh_skills, skill_text, write_skill  # noqa: F401  (fresh_skills is a fixture)

pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")

SKILLS_PROFILE = "B1S-agent-skills"
CLARIFY = "2026年9月的销售额和已支付订单数分别是多少？"
# The SSE step fields (api/main.py _STEP_FIELDS) with a model call's token counts.
SSE_STEP_KEYS = {"kind", "status", "error_code", "tool_name", "elapsed_ms", "model", "usage_status", "agent", "prompt_tokens", "completion_tokens", "total_tokens"}


@pytest.fixture()
def skills_http(env, monkeypatch):  # noqa: F811
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "b1s")
    return env


def _steps(run_id: str) -> list[dict]:
    return [event["payload"] for event in shared_run_service().store.events(run_id, after_event_id=0, limit=1000) if event["type"] == "agent_step"]


def _check_answer(body: dict) -> None:
    assert body["status"] == "SUCCEEDED" and body["profile"] == SKILLS_PROFILE
    assert sorted(fact["metric_id"] for fact in body["facts"]["facts"]) == ["gross_fen", "net_fen"]
    assert body["answer_status"] == "verified"


def test_a_sync_composite_question_loads_the_skill_then_queries_each_part(skills_http) -> None:
    body = ask(skills_http, COMPOSITE).json()
    _check_answer(body)
    # Load, search (this server has a retriever), one query per part, the answer.
    assert (body["model_call_count"], body["tool_call_count"]) == (5, 3)
    kinds = [step["kind"] for step in _steps(body["run_id"])]
    assert kinds[:2] == ["model_call", "skill_load"] and kinds.count("skill_load") == 1
    stored = shared_run_service().store.get_run(body["run_id"])["run_config"]
    assert stored["agent_run_config"]["skill_versions"] == ["composite-question@v1", "customer-ranking@v1"]


def test_an_async_run_streams_the_load_as_a_plain_step(skills_http) -> None:
    response = ask(skills_http, COMPOSITE, asynchronous=True)
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert wait(skills_http, run_id, {"SUCCEEDED", "FAILED"})["status"] == "SUCCEEDED"
    frames = _frames(skills_http, run_id)
    steps = [frame["step"] for frame in frames if frame["type"] == "agent_step"]
    assert all(set(step) <= SSE_STEP_KEYS for step in steps)
    assert [step for step in steps if step["kind"] == "skill_load"] == [{"kind": "skill_load", "status": "succeeded"}]
    assert frames[-1]["type"] == "terminal"


def test_a_run_waiting_for_the_user_has_its_skill_back_after_the_resume(skills_http) -> None:
    waiting = ask(skills_http, CLARIFY).json()
    assert (waiting["status"], waiting["model_call_count"]) == ("WAITING_USER", 2)
    resumed = skills_http.post(f"/runs/{waiting['run_id']}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额统计"})
    body = resumed.json()
    assert resumed.status_code == 200 and body["status"] == "SUCCEEDED"
    # The Fake loads only while the Skill's text is missing: one load in the whole run.
    assert [step["kind"] for step in _steps(waiting["run_id"])].count("skill_load") == 1
    assert sorted(fact["metric_id"] for fact in body["facts"]["facts"]) == ["gross_fen", "paid_count"]


def test_a_skills_run_resumes_on_a_server_now_set_to_b1(skills_http, monkeypatch) -> None:
    waiting = ask(skills_http, CLARIFY).json()
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "b1")
    body = skills_http.post(f"/runs/{waiting['run_id']}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额统计"}).json()
    assert body["status"] == "SUCCEEDED"
    assert shared_run_service().store.get_run(waiting["run_id"])["run_config"]["profile"] == SKILLS_PROFILE


def test_a_run_is_not_resumed_once_other_skill_versions_are_deployed(skills_http, tmp_path, monkeypatch) -> None:
    waiting = ask(skills_http, CLARIFY).json()
    store = shared_run_service().store
    before = store.events(waiting["run_id"], after_event_id=0, limit=1000)
    for name in ("composite-question", "customer-ranking"):
        write_skill(tmp_path / "skills", name, skill_text(name, version="v2"))
    monkeypatch.setattr(runtime, "product_skills", lambda: load_skill_library(tmp_path / "skills"))
    response = skills_http.post(f"/runs/{waiting['run_id']}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额统计"})
    assert response.status_code == 409 and response.json()["error"]["code"] == "checkpoint_invalid"
    assert store.get_run(waiting["run_id"])["status"] == "WAITING_USER"
    assert store.events(waiting["run_id"], after_event_id=0, limit=1000) == before


def test_after_an_approval_no_model_is_called_and_the_load_stays_recorded(skills_http) -> None:
    model = Scripted([
        json.dumps({"type": "load_skill", "skill_id": "customer-ranking"}),
        json.dumps({"type": "tool_call", "name": "query_readonly", "arguments": {"sql": "SELECT c.customer_id, c.name FROM customers AS c ORDER BY c.customer_id", "params": {}}}),
    ])
    app.dependency_overrides[get_model_provider] = lambda: model
    pending = ask(skills_http, "查询客户姓名").json()
    assert (pending["status"], pending["model_call_count"]) == ("WAITING_APPROVAL", 2)
    approved = skills_http.post(f"/runs/{pending['run_id']}/approval", headers=auth(APPROVER), json={"approval_id": pending["approval_id"], "decision": "approve"})
    assert approved.status_code == 200 and approved.json()["status"] == "SUCCEEDED"
    assert model.calls == 2
    loads = [step for step in _steps(pending["run_id"]) if step["kind"] == "skill_load"]
    assert [(step["status"], step["skill_id"]) for step in loads] == [("succeeded", "customer-ranking")]
    result = skills_http.get(f"/runs/{pending['run_id']}/result", headers=auth(REQUESTER)).json()
    assert {row["name"] for row in result["result"]["rows"]} == {"甲", "乙"}


@pytest.mark.parametrize("setting", ["b1s", "B1S", "b1s-agent-skills", SKILLS_PROFILE])
def test_the_profile_names_the_server_accepts(env, monkeypatch, setting) -> None:  # noqa: F811
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", setting)
    _check_answer(ask(env, COMPOSITE).json())


def test_the_native_protocol_with_the_skills_profile_is_refused_before_any_run(skills_http, monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", "native")
    response = ask(skills_http, COMPOSITE)
    assert response.status_code == 503 and response.json()["error"]["code"] == "invalid_model_protocol"
    assert shared_run_service().store._connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


def test_a_bad_skill_file_blocks_the_skills_profile(skills_http, tmp_path, monkeypatch, fresh_skills) -> None:  # noqa: F811
    write_skill(tmp_path / "skills", "alpha-skill", BAD_FILES["missing_field"])
    monkeypatch.setattr(runtime, "load_skill_library", lambda: load_skill_library(tmp_path / "skills"))
    response = ask(skills_http, COMPOSITE)
    assert response.status_code == 503 and response.json()["error"]["code"] == "invalid_skill_file"
    assert shared_run_service().store._connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


@pytest.mark.parametrize("action, code", [({"type": "load_skill", "skill_id": "composite-question"}, "unknown_field"), ({"type": "load_skill"}, "unknown_action")])
def test_the_single_proposal_entry_refuses_load_skill_as_before(skills_http, action, code) -> None:
    response = skills_http.post("/query-proposals", headers=auth(REQUESTER), json={"proposal": json.dumps(action)})
    assert response.status_code == 422 and response.json()["error"]["code"] == code


def test_every_call_of_a_skills_run_carries_the_root_run_id(service, upstream, monkeypatch) -> None:  # noqa: F811
    """The Real adapters against the fake upstream: the load is one more tagged chat call, counted by the run."""

    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "b1s")
    body = _ask(service, COMPOSITE)
    _check_answer(body)
    chats = upstream.sent("chat/completions")
    _assert_tagged(chats, body["run_id"])
    assert len(chats) == body["model_call_count"] == 5  # load, search, two queries, the answer
    assert body["usage_total"]["status"] == "known"
