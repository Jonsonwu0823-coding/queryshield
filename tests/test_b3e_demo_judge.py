"""B3e: every verified fact of every demo question equals the generator's own value for its tenant, metric and window."""

from __future__ import annotations

import ast
import json

from scripts import demo_run as run
from scripts import generate_demo_data as gen

QUESTIONS = {q["id"]: q for q in json.loads(gen.QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]}
DEMO_IDS = run.demo_source_ids()
JULY = {"start": "2026-07-01T00:00:00Z", "end": "2026-08-01T00:00:00Z"}
SEPTEMBER = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}


def _fact(metric_id, value, window):
    return {"metric_id": metric_id, "value": value, "time_window": {**window, "timezone": "UTC"}, "catalog_version": "catalog-v4"}


def _obs(facts, *, answer_status="verified", **extra):
    return {"http_status": 200, "status": "SUCCEEDED", "answer_status": answer_status, "facts": facts, "rows": [], "source_ids": [], "trace": [], **extra}


def _independent(tenant, metric_id, window):
    data = gen.generate()
    start, end = run._utc_seconds(window["start"]), run._utc_seconds(window["end"])
    return gen.expected_metrics(data, tenant, start, end)[metric_id]


def test_a_fact_equal_to_the_independent_value_passes() -> None:
    q = QUESTIONS["Q01"]
    obs = _obs([_fact("paid_count", q["expected"]["value"], q["window"])])
    assert run.verified_fact_counts(q, obs) == (1, 0)
    outcome = run.judge_question(q, obs, DEMO_IDS)
    assert outcome["hard_failures"] == []
    assert (outcome["verified_facts_checked"], outcome["verified_facts_mismatched"]) == (1, 0)


def test_a_fact_that_differs_is_a_hard_failure_on_any_question() -> None:
    q = QUESTIONS["Q12"]  # the isolation question judges no value of its own
    wrong = _independent("A", "gross_fen", q["window"]) + 1
    outcome = run.judge_question(q, _obs([_fact("gross_fen", wrong, q["window"])]), DEMO_IDS)
    assert "verified_fact_mismatch" in outcome["hard_failures"]
    assert (outcome["verified_facts_checked"], outcome["verified_facts_mismatched"]) == (1, 1)


def test_a_fact_in_another_window_is_recomputed_for_its_own_window() -> None:
    q = QUESTIONS["Q02"]  # August gross_fen
    right_for_september = _independent("A", "gross_fen", SEPTEMBER)
    obs = _obs([_fact("gross_fen", right_for_september, SEPTEMBER)])
    assert run.verified_fact_counts(q, obs) == (1, 0)
    outcome = run.judge_question(q, obs, DEMO_IDS)
    # The fact itself is true; the question's own rule still flags the wrong window.
    assert "verified_fact_mismatch" not in outcome["hard_failures"] and "window_mismatch" in outcome["hard_failures"]
    assert run.verified_fact_counts(q, _obs([_fact("gross_fen", q["expected"]["value"], SEPTEMBER)])) == (1, 1)


def test_no_facts_checks_nothing() -> None:
    q = QUESTIONS["Q09"]
    obs = _obs([], answer_status="unverified")
    assert run.verified_fact_counts(q, obs) == (0, 0)
    outcome = run.judge_question(q, {**obs, "sql_exec_count": 0, "source_ids": ["demo-metric-net"]}, DEMO_IDS)
    assert (outcome["verified_facts_checked"], outcome["verified_facts_mismatched"]) == (0, 0)


def test_unknown_metric_tenant_or_window_counts_as_a_mismatch() -> None:
    q = QUESTIONS["Q01"]
    assert run.verified_fact_counts(q, _obs([_fact("unknown_metric", 1, JULY)])) == (1, 1)
    assert run.verified_fact_counts(q, _obs([{"metric_id": "paid_count", "value": 124, "time_window": {"start": "bad"}}])) == (1, 1)
    assert run.verified_fact_counts({**q, "identity": "c-requester"}, _obs([_fact("paid_count", 124, JULY)])) == (1, 1)


def _q07_obs(facts=(), answer_status="unverified"):
    q = QUESTIONS["Q07"]
    rows = [{"name": q["expected"]["name"], "gross_fen": q["expected"]["value"]}]
    return _obs(list(facts), answer_status=answer_status, rows=rows, approval_seen=True, answer_contains_row_values=False)


def test_q07_name_right_unverified_without_facts_passes() -> None:
    q = QUESTIONS["Q07"]
    outcome = run.judge_question(q, _q07_obs(), DEMO_IDS)
    assert outcome["hard_failures"] == [] and outcome["known_gaps"] == []


def test_q07_the_top_customer_amount_labelled_as_the_tenant_total_is_a_hard_failure() -> None:
    q = QUESTIONS["Q07"]  # B3d final Real: 287280 reported as the July gross_fen
    obs = _q07_obs(facts=[_fact("gross_fen", q["expected"]["value"], q["window"])], answer_status="verified")
    outcome = run.judge_question(q, obs, DEMO_IDS)
    assert "verified_fact_mismatch" in outcome["hard_failures"]
    assert outcome["verified_facts_mismatched"] == 1


def test_q07_a_correct_supporting_fact_with_verified_status_is_a_known_gap() -> None:
    q = QUESTIONS["Q07"]
    total = _independent("A", "gross_fen", q["window"])
    assert total == 2_222_500
    obs = _q07_obs(facts=[_fact("gross_fen", total, q["window"])], answer_status="verified")
    outcome = run.judge_question(q, obs, DEMO_IDS)
    assert outcome["hard_failures"] == [] and outcome["known_gaps"] == ["verified_supporting_fact"]


def test_the_summary_carries_only_the_two_counts() -> None:
    q = QUESTIONS["Q07"]
    obs = _q07_obs(facts=[_fact("gross_fen", q["expected"]["value"], q["window"])], answer_status="verified")
    record = run.summary_record(q, obs, run.judge_question(q, obs, DEMO_IDS))
    assert (record["verified_facts_checked"], record["verified_facts_mismatched"]) == (1, 1)
    assert q["expected"]["name"] not in json.dumps(record, ensure_ascii=False)


def test_the_independent_values_never_come_from_queryshield() -> None:
    tree = ast.parse(gen.__file__ and open(gen.__file__, encoding="utf-8").read())
    imported = {
        alias.name if isinstance(node, ast.Import) else node.module
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name and name.startswith("queryshield") for name in imported)
