"""The composite demo questions: generated with their answers, checked again on the database,
answered by the Fake under B1 and B2, judged part by part, and compared by profile."""

from __future__ import annotations

import copy
import json

import pytest

from scripts import compare_demo_runs as compare
from scripts import demo_run as run
from scripts import generate_demo_data as gen
from scripts import verify_demo_expected as verify

from multi_agent_support import run_b2, steps
from test_clarification import service  # noqa: F401  (fixture)
from test_demo_expected_answers import conn, needs_demo_database  # noqa: F401  (conn is a fixture)

DOCUMENT = json.loads(gen.COMPOSITE_PATH.read_text(encoding="utf-8"))
QUESTIONS = {q["id"]: q for q in DOCUMENT["questions"]}


# --- the question file ------------------------------------------------------------------------


def test_the_composite_file_is_generated_and_checked(tmp_path, monkeypatch, capsys) -> None:
    assert gen.render_all()[gen.COMPOSITE_PATH].encode("utf-8") == gen.COMPOSITE_PATH.read_bytes()
    edited = tmp_path / gen.COMPOSITE_PATH.name
    edited.write_text(gen.COMPOSITE_PATH.read_text(encoding="utf-8").replace("2830010", "2830011"), encoding="utf-8")
    monkeypatch.setattr(gen, "COMPOSITE_PATH", edited)
    assert gen.main(["--check"]) == 1
    assert gen.COMPOSITE_PATH.name in capsys.readouterr().out


def test_the_questions_cover_every_kind_the_comparison_needs() -> None:
    assert DOCUMENT["version"] == "demo-composite-questions-v1" and DOCUMENT["data_version"] == "commerce-demo-v1"
    assert [q["kind"] for q in DOCUMENT["questions"]].count("composite") == 6
    parts = {qid: [(f["metric_id"], f["window"]["start"][:7]) for f in q["expected"].get("facts", [])] for qid, q in QUESTIONS.items()}
    assert parts["CQ01"] == [("gross_fen", "2026-08"), ("net_fen", "2026-08")]  # one window, one part is net
    assert parts["CQ02"] == [("paid_count", "2026-07"), ("paid_count", "2026-08")]  # one metric, two windows
    assert len(parts["CQ04"]) == 3
    assert QUESTIONS["CQ05"]["kind"] == "composite_clarify" and "销售额" in QUESTIONS["CQ05"]["question"]
    assert 0 in [f["value"] for f in QUESTIONS["CQ06"]["expected"]["facts"]]  # an empty window
    assert QUESTIONS["CQ07"]["kind"] == "isolation" and QUESTIONS["CQ08"]["tenant"] == "B"
    assert all(q["request_time_window"] is None and q["fake_supported"] for q in DOCUMENT["questions"])


@needs_demo_database
def test_sql_on_the_demo_database_agrees_with_every_expected_part(conn) -> None:  # noqa: F811
    verdicts = verify.verify(conn, DOCUMENT)
    assert len(verdicts) == 16 and all(ok for _, ok, _ in verdicts), [v for v in verdicts if not v[1]]
    wrong = copy.deepcopy(DOCUMENT)
    wrong["questions"][0]["expected"]["facts"][1]["value"] += 1
    assert [qid for qid, ok, _ in verify.verify(conn, wrong) if not ok] == ["CQ01:net_fen:2026-08"]


# --- the judgement ------------------------------------------------------------------------------


def _fact(metric_id, window, value):
    return {"metric_id": metric_id, "value": value, "time_window": {**window, "timezone": "UTC"}}


def _obs(facts, **extra):
    return {"http_status": 200, "status": "SUCCEEDED", "answer_status": "verified", "facts": facts, **extra}


def _expected_facts(qid):
    return [_fact(f["metric_id"], f["window"], f["value"]) for f in QUESTIONS[qid]["expected"]["facts"]]


def test_every_expected_part_right_and_nothing_else_passes() -> None:
    outcome = run.judge_composite(QUESTIONS["CQ01"], _obs(_expected_facts("CQ01")))
    assert (outcome["hard_failures"], outcome["fact_completeness"], outcome["actual"]) == ([], 1.0, {"fact_count": 2, "found": 2})


@pytest.mark.parametrize(
    "change, failure, completeness",
    [
        (lambda facts: facts[:1], "missing_fact:net_fen", 0.5),
        (lambda facts: [facts[0], {**facts[1], "value": facts[1]["value"] + 1}], "value_mismatch:net_fen", 0.5),
        (lambda facts: [facts[0], {**facts[1], "time_window": facts[0]["time_window"] | {"start": "2026-07-01T00:00:00Z"}}], "unexpected_fact", 0.5),
        (lambda facts: facts + [_fact("paid_count", facts[0]["time_window"], 1)], "unexpected_fact", 1.0),
    ],
    ids=["missing", "wrong-value", "wrong-window", "extra"],
)
def test_a_missing_wrong_or_extra_part_fails(change, failure, completeness) -> None:
    outcome = run.judge_composite(QUESTIONS["CQ01"], _obs(change(_expected_facts("CQ01"))))
    assert failure in outcome["hard_failures"] and outcome["fact_completeness"] == completeness


def test_the_clarified_question_must_ask_the_catalog_question_first() -> None:
    first = {"http_status": 202, "status": "WAITING_USER", "pending_is_catalog_question": True}
    assert run.judge_composite_clarify(QUESTIONS["CQ05"], _obs(_expected_facts("CQ05"), first=first))["hard_failures"] == []
    skipped = run.judge_composite_clarify(QUESTIONS["CQ05"], _obs(_expected_facts("CQ05"), first={"http_status": 200, "status": "SUCCEEDED"}))
    assert skipped["hard_failures"] == ["clarify_not_triggered"]


def test_a_run_under_another_profile_than_requested_fails() -> None:
    records = [{"id": "CQ01", "profile": "B1-bounded-agent"}, {"id": "CQ07", "profile": None}]
    assert run.run_profile_check(records, None) == ("B1-bounded-agent", [])
    assert run.run_profile_check(records, "b1") == ("B1-bounded-agent", [])
    assert run.run_profile_check(records, "b2") == ("B1-bounded-agent", ["CQ01:profile_mismatch"])
    mixed = records + [{"id": "CQ02", "profile": "B2-multi-agent"}]
    assert run.run_profile_check(mixed, None) == (None, ["mixed_profiles"])


# --- the comparison -------------------------------------------------------------------------------


def _summary(profile, records):
    return {"profile": profile, "questions_set": "composite", "records": records}


def _record(qid, **fields):
    base = {"id": qid, "verdict": "pass", "run_id": f"run-{qid}", "model_call_count": 4, "tool_call_count": 2, "elapsed_ms": 100, "delegated": False, "error_code": None, "usage_total": {"status": "unknown"}}
    return {**base, **fields}


def test_the_comparison_reports_each_profile_and_says_none_for_what_is_missing() -> None:
    b1 = _summary("B1-bounded-agent", [_record("CQ01", fact_completeness=1.0, expected={"fact_count": 2}, actual={"found": 2}), _record("CQ07", run_id=None, model_call_count=None, tool_call_count=None, elapsed_ms=10, error_code="forbidden")])
    b2 = _summary("B2-multi-agent", [
        _record("CQ01", model_call_count=5, tool_call_count=3, elapsed_ms=300, delegated=True, fact_completeness=0.5, expected={"fact_count": 2}, actual={"found": 1}, verdict="fail",
                usage_total={"status": "known", "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}),
        _record("CQ02", model_call_count=1, tool_call_count=0, elapsed_ms=200, delegated=True, error_code="subtask_incomplete", verdict="fail",
                usage_total={"status": "known", "prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}),
        {"id": "CQ03", "verdict": "not_applicable"},
    ])
    report = compare.compare([b1, b2, _summary("B2-multi-agent", [])])["profiles"]
    first, second = report["B1-bounded-agent"], report["B2-multi-agent"]
    assert (first["passed"], first["questions"], first["fact_completeness"], first["delegation_rate"]) == (2, 2, 1.0, 0.0)
    assert first["input_tokens"] == first["output_tokens"] == "none"
    assert first["elapsed_ms"] == {"median": 55.0, "max": 100} and first["error_codes"] == {"forbidden": 1}
    assert (second["runs"], second["passed"], second["questions"], second["fact_completeness"]) == (2, 0, 2, 0.5)
    assert second["model_calls"] == {"total": 6, "per_run": 3.0} and second["tool_calls"] == {"total": 3, "per_run": 1.5}
    assert (second["input_tokens"], second["output_tokens"], second["delegation_rate"]) == (15, 3, 1.0)
    assert second["error_codes"] == {"subtask_incomplete": 1}
    assert compare.main(["only-one.json"]) == 2


# --- the Fake answers both profiles ----------------------------------------------------------------


@pytest.mark.parametrize("profile", ["b1", "b2"])
def test_the_fake_answers_every_part_under_both_profiles(service, monkeypatch, profile) -> None:  # noqa: F811
    from queryshield.providers.fake_model import FakeModel

    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", profile)
    svc, _ = service
    for qid in ("CQ01", "CQ02", "CQ03", "CQ04", "CQ06"):
        question = QUESTIONS[qid]
        done = run_b2(service, question["question"], FakeModel())
        assert done["status"] == "SUCCEEDED", (qid, done.get("error_code"))
        windows = {(f["metric_id"], f["time_window"]["start"]) for f in done["facts"]["facts"]}
        assert windows == {(f["metric_id"], f["window"]["start"]) for f in question["expected"]["facts"]}, qid
        kinds = [p["kind"] for p in steps(svc, done["run_id"])]
        assert ("delegation" in kinds) is (profile == "b2")
