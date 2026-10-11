"""The demo reports know the skills profile: which Skills a run loaded, the load rate, the new profile option;
and the Fake model loads the composite Skill only when the server offers Skills."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from queryshield.agent.context import build_context
from queryshield.agent.proposals import ExecutionContext
from queryshield.agent.runtime import B1S_PROFILE, product_run_config, product_skills
from queryshield.api.main import app, get_model_provider
from queryshield.catalog import load_default_catalog
from queryshield.providers.fake_model import FakeModel

from scripts import compare_demo_runs as compare
from scripts import demo_run as run

from multi_agent_support import COMPOSITE
from test_composite_demo import _record, _summary
from test_http_queries import Scripted, ask, env  # noqa: F401  (env is a fixture)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CONTEXT = ExecutionContext(run_id="run-skill-fake", tenant_id="A", principal_id="principal-A", role="requester")
SKILLS = product_skills()
CONFIG = product_run_config(B1S_PROFILE, catalog=load_default_catalog(), retriever=None)


def test_the_run_shape_lists_the_skills_a_run_loaded(env, tmp_path, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "b1s")
    loaded = ask(env, COMPOSITE).json()
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "b1")
    plain = ask(env, COMPOSITE).json()
    state = Path(tmp_path / "state.sqlite3")
    assert run._run_shape(state, loaded["run_id"]) == {"profile": B1S_PROFILE, "subtask_count": None, "skills_loaded": ["composite-question"]}
    assert run._run_shape(state, plain["run_id"]) == {"profile": "B1-bounded-agent", "subtask_count": None, "skills_loaded": []}


def test_the_run_shape_lists_only_the_loads_that_succeeded(env, tmp_path, monkeypatch) -> None:  # noqa: F811
    load = json.dumps({"type": "load_skill", "skill_id": "composite-question"})
    model = Scripted([load, load])
    app.dependency_overrides[get_model_provider] = lambda: model
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "b1s")
    body = ask(env, COMPOSITE).json()
    assert (body["status"], body["error_code"]) == ("FAILED", "skill_already_loaded")
    assert run._run_shape(Path(tmp_path / "state.sqlite3"), body["run_id"])["skills_loaded"] == ["composite-question"]


def test_the_summary_record_carries_the_loaded_skill_names_only() -> None:
    question = {"id": "CQ01", "identity": "a-requester", "kind": "composite"}
    judgement = {"hard_failures": [], "known_gaps": []}
    record = run.summary_record(question, {"skills_loaded": ["composite-question"], "facts": []}, judgement)
    assert record["skills_loaded"] == ["composite-question"]
    assert run.summary_record(question, {"facts": []}, judgement)["skills_loaded"] is None


def test_the_comparison_reports_the_skill_load_rate_and_none_for_older_summaries() -> None:
    skills = _summary(B1S_PROFILE, [
        _record("CQ01", skills_loaded=["composite-question", "customer-ranking"]),
        _record("CQ02", skills_loaded=["composite-question"]),
        _record("CQ08", skills_loaded=[]),
        _record("CQ07", run_id=None, skills_loaded=None),
    ])
    older = _summary("B1-bounded-agent", [_record("CQ01")])
    report = compare.compare([skills, older])["profiles"]
    # A run that loaded two Skills still counts once in the rate.
    assert (report[B1S_PROFILE]["skill_load_rate"], report[B1S_PROFILE]["skills_loaded"]) == (0.667, {"composite-question": 2, "customer-ranking": 1})
    assert report["B1-bounded-agent"]["skill_load_rate"] == report["B1-bounded-agent"]["skills_loaded"] == "none"


def test_the_profile_option_and_its_check_know_the_skills_profile() -> None:
    records = [{"id": "CQ01", "profile": B1S_PROFILE}]
    assert run.run_profile_check(records, "b1s") == (B1S_PROFILE, [])
    assert run.run_profile_check(records, "b1") == (B1S_PROFILE, ["CQ01:profile_mismatch"])
    for script in ("demo-local.ps1", "compose-real-demo.ps1"):
        assert "[ValidateSet('b1', 'b2', 'b1s')][string]$Profile" in (SCRIPTS / script).read_text(encoding="utf-8-sig")
    check_all = (SCRIPTS / "check-all.ps1").read_text(encoding="utf-8")
    assert '@{ Name = "demo-run-composite-b1s"; Dir = "demo-run-composite-b1s"; Script = "demo_run.py"; Args = @("--mode", "fake", "--questions", "composite", "--profile", "b1s"); Demo = $true }' in check_all


# --- the Fake model -----------------------------------------------------------------------------------


def _fake(question: str, *, loaded=(), config=CONFIG, skills=SKILLS) -> dict:
    messages = build_context(CONTEXT, question, run_config=config, parallel_available=False, retrieval_available=False, skills=skills, loaded_skills=loaded).messages
    return json.loads(FakeModel().complete(list(messages)).content)


@pytest.mark.parametrize("question", [COMPOSITE, "2026年9月的销售额和已支付订单数分别是多少？"])
def test_the_fake_loads_the_composite_skill_first_when_skills_are_offered(question) -> None:
    assert _fake(question) == {"type": "load_skill", "skill_id": "composite-question"}
    after = _fake(question, loaded=(SKILLS.get("composite-question"),))
    assert after["type"] in {"tool_call", "ask_user"}


def test_the_fake_never_loads_for_a_single_question_or_without_skills() -> None:
    assert _fake("2026年9月已支付订单总额是多少？")["type"] == "tool_call"
    assert _fake(COMPOSITE, config=None, skills=None)["type"] == "tool_call"
