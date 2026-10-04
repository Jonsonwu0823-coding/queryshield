"""B3d: the demo run's judgements are pure functions; each outcome has a test, and the summary holds no text."""

from __future__ import annotations

import json

import pytest

from scripts import demo_run as run
from scripts import generate_demo_data as gen

QUESTIONS = {q["id"]: q for q in json.loads(gen.QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]}
DEMO_IDS = run.demo_source_ids()


def fact(question, value=None, window=None):
    expected = question["expected"]
    return {
        "metric_id": expected["metric_id"],
        "value": expected["value"] if value is None else value,
        "time_window": {**(window or question["window"]), "timezone": "UTC"},
        "catalog_version": "catalog-v4",
    }


def metric_obs(question, *, value=None, window=None, answer_status="verified", trace=None, **extra):
    return {
        "http_status": 200, "status": "SUCCEEDED", "answer_status": answer_status, "facts": [fact(question, value, window)],
        "rows": [], "source_ids": [], "sql_exec_count": 2, "trace": trace or [], **extra,
    }


# --- numbers -------------------------------------------------------------------------------


def test_a_verified_value_equal_to_the_expected_one_passes() -> None:
    q = QUESTIONS["Q03"]
    outcome = run.judge_metric(q, metric_obs(q))
    assert outcome["hard_failures"] == [] and outcome["known_gaps"] == []
    assert outcome["expected"]["value"] == outcome["actual"]["values"][0]


def test_a_verified_value_that_differs_is_a_hard_failure() -> None:
    q = QUESTIONS["Q03"]
    outcome = run.judge_metric(q, metric_obs(q, value=q["expected"]["value"] + 1))
    assert outcome["hard_failures"] == ["value_mismatch"]


def test_the_right_value_in_the_wrong_window_is_a_hard_failure() -> None:
    q = QUESTIONS["Q02"]
    september = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}
    assert run.judge_metric(q, metric_obs(q, window=september))["hard_failures"] == ["window_mismatch"]


def test_a_value_marked_unverified_is_a_hard_failure() -> None:
    q = QUESTIONS["Q01"]
    assert "answer_status" in run.judge_metric(q, metric_obs(q, answer_status="unverified"))["hard_failures"]


def test_verified_without_facts_is_a_hard_failure_and_a_missing_fact_is_only_a_gap() -> None:
    q = QUESTIONS["Q01"]
    none = {"http_status": 200, "status": "SUCCEEDED", "answer_status": "verified", "facts": []}
    assert "verified_without_facts" in run.judge_metric(q, none)["hard_failures"]
    asked = {"http_status": 202, "status": "WAITING_USER", "answer_status": None, "facts": []}
    outcome = run.judge_metric(q, asked)
    assert outcome["hard_failures"] == [] and outcome["known_gaps"] == ["no_verified_fact:WAITING_USER"]


def test_a_server_error_is_hard_but_answer_not_grounded_is_a_known_model_gap() -> None:
    q = QUESTIONS["Q01"]
    failed = {"http_status": 502, "status": "FAILED", "error_code": "evidence_validation_failed", "facts": []}
    assert run.judge_metric(q, failed)["hard_failures"] == ["server_error:evidence_validation_failed"]
    ungrounded = {"http_status": 502, "status": "FAILED", "error_code": "answer_not_grounded", "facts": []}
    outcome = run.judge_metric(q, ungrounded)
    assert outcome["hard_failures"] == [] and outcome["known_gaps"][0] == "model_behaviour:answer_not_grounded"


def test_the_empty_month_expects_zero_and_records_an_answer_after_the_bounce() -> None:
    q = QUESTIONS["Q11"]
    assert q["expected"]["value"] == 0 and q["kind"] == "empty_window"
    direct = run.judge_metric(q, metric_obs(q, trace=["tool_call(query_readonly)", "final_answer(query,absent)"]))
    assert direct["hard_failures"] == [] and direct["known_gaps"] == [] and direct["bounced"] is False
    bounced = run.judge_metric(q, metric_obs(q, trace=["answer_bounce(answer_without_query_result)", "tool_call(query_readonly)"]))
    assert bounced["hard_failures"] == [] and bounced["bounced"] is True
    assert bounced["known_gaps"] == ["empty_window_answered_after_bounce"]
    wrong = run.judge_metric(q, metric_obs(q, value=5))
    assert wrong["hard_failures"] == ["value_mismatch"]


# --- row sets --------------------------------------------------------------------------------


def _ok(rows, **extra):
    return {"http_status": 200, "status": "SUCCEEDED", "answer_status": "unverified", "rows": rows, **extra}


def _top_rows(question, key="customer_id", alias="total"):
    return [{key: row["customer_id"], alias: row["value"]} for row in question["expected"]["rows"]]


def _all_rows(question):
    return [{"customer_id": cid, "total": value} for cid, value in question["expected"]["all_values"].items()]


def test_the_bounded_rowset_is_the_top_five_with_no_tie_at_the_cut_off_and_asks_for_no_names() -> None:
    q = QUESTIONS["Q06"]
    rows = q["expected"]["rows"]
    assert q["expected"]["top_n"] == 5 and len(rows) == 5
    ranked = sorted(q["expected"]["all_values"].items(), key=lambda item: (-item[1], item[0]))
    assert [(r["customer_id"], r["value"]) for r in rows] == ranked[:5]
    assert ranked[4][1] > ranked[5][1]  # the 5th and the 6th do not tie
    assert "姓名" not in q["question"] and "客户名" not in q["question"] and "5个客户" in q["question"]
    assert "客户编号和金额" in q["question"]


def test_the_top_five_with_the_right_amounts_passes_whatever_the_column_aliases_are() -> None:
    q = QUESTIONS["Q06"]
    for key, alias in (("customer_id", "gross_fen"), ("cid", "sum_amount"), ("c", "x")):
        rows = [{key: row["customer_id"], alias: row["value"]} for row in q["expected"]["rows"]]
        outcome = run.judge_rowset(q, _ok(rows))
        assert outcome["hard_failures"] == [] and outcome["known_gaps"] == [], (key, alias)
        assert outcome["actual"]["matched"] == 5 and outcome["actual"]["expected_customers_found"] == 5


def test_a_rowset_may_use_customer_names_after_an_approval() -> None:
    q = QUESTIONS["Q06"]
    rows = [{"name": row["name"], "s": row["value"]} for row in q["expected"]["rows"]]
    outcome = run.judge_rowset(q, _ok(rows))
    assert outcome["hard_failures"] == [] and outcome["actual"]["matched"] == 5
    assert not any(row["name"] in json.dumps(outcome, ensure_ascii=False) for row in q["expected"]["rows"])


def test_a_rowset_with_a_wrong_amount_is_a_hard_failure() -> None:
    q = QUESTIONS["Q06"]
    rows = _top_rows(q)
    rows[0]["total"] += 1
    outcome = run.judge_rowset(q, _ok(rows))
    assert outcome["hard_failures"] == ["rowset_value_mismatch"] and outcome["actual"]["wrong"] == 1
    # also for a customer outside the top five: its own true amount is known
    other = next(cid for cid in q["expected"]["all_values"] if cid not in {r["customer_id"] for r in q["expected"]["rows"]})
    wrong_other = _top_rows(q) + [{"customer_id": other, "total": q["expected"]["all_values"][other] + 7}]
    assert run.judge_rowset(q, _ok(wrong_other))["hard_failures"] == ["rowset_value_mismatch"]


def test_extra_customers_with_true_amounts_missing_customers_and_unparseable_rows_are_only_known_gaps() -> None:
    q = QUESTIONS["Q06"]
    everyone = run.judge_rowset(q, _ok(_all_rows(q)))  # the model listed all customers instead of five
    assert everyone["hard_failures"] == [] and everyone["known_gaps"] == ["rowset_customers_differ"]
    assert everyone["actual"]["row_count"] == len(q["expected"]["all_values"])
    fewer = run.judge_rowset(q, _ok(_top_rows(q)[:3]))
    assert fewer["hard_failures"] == [] and fewer["known_gaps"] == ["rowset_customers_differ"]
    odd = run.judge_rowset(q, _ok([{"x": "?", "y": "?"}]))
    assert odd["known_gaps"][0] == "rowset_unparseable_rows"
    nothing = run.judge_rowset(q, _ok([]))
    assert nothing["hard_failures"] == [] and nothing["known_gaps"] == ["no_rows:SUCCEEDED"]


def test_the_full_summary_is_an_observation_that_only_fails_when_it_claims_to_be_verified() -> None:
    q = QUESTIONS["Q06b"]
    assert q["kind"] == "observe_rowset" and len(q["expected"]["rows"]) == len(q["expected"]["all_values"]) > 5
    full = run.judge_observe_rowset(q, _ok(_all_rows(q)))
    assert full["hard_failures"] == [] and full["known_gaps"] == []
    assert full["actual"]["row_count"] == full["actual"]["matched"] == len(q["expected"]["rows"])
    truncated = run.judge_observe_rowset(q, {"http_status": 502, "status": "FAILED", "error_code": "invalid_json", "answer_status": None, "rows": [], "facts": []})
    assert truncated["hard_failures"] == []
    assert truncated["known_gaps"] == ["observed_server_error:invalid_json", "observed_no_rows:FAILED"]
    wrong = _all_rows(q)
    wrong[0]["total"] += 1
    assert run.judge_observe_rowset(q, _ok(wrong))["hard_failures"] == []
    assert "observed_rowset_value_mismatch" in run.judge_observe_rowset(q, _ok(wrong))["known_gaps"]
    partial = run.judge_observe_rowset(q, _ok(_all_rows(q)[:10]))
    assert partial["hard_failures"] == [] and partial["known_gaps"] == ["observed_rowset_incomplete"]
    verified = run.judge_observe_rowset(q, {"http_status": 200, "status": "SUCCEEDED", "answer_status": "verified", "facts": [], "rows": []})
    assert verified["hard_failures"] == ["verified_without_facts"]


# --- approval: the top customer ------------------------------------------------------------------


def top_obs(rows, **extra):
    return {"http_status": 200, "status": "SUCCEEDED", "answer_status": "unverified", "rows": rows, "approval_seen": True, **extra}


def test_one_row_must_carry_the_expected_name() -> None:
    q = QUESTIONS["Q07"]
    name = q["expected"]["name"]
    assert run.judge_top_customer(q, top_obs([{"name": name}]))["hard_failures"] == []
    assert run.judge_top_customer(q, top_obs([{"name": name + "X"}]))["hard_failures"] == ["top_customer_mismatch"]


def test_several_rows_must_contain_the_expected_name_and_the_first_row_is_only_observed() -> None:
    q = QUESTIONS["Q07"]
    name = q["expected"]["name"]
    first = run.judge_top_customer(q, top_obs([{"name": name}, {"name": "其他"}]))
    assert first["hard_failures"] == [] and first["actual"]["expected_name_first_row"] is True
    later = run.judge_top_customer(q, top_obs([{"name": "其他"}, {"name": name}]))
    assert later["hard_failures"] == [] and later["actual"]["expected_name_first_row"] is False
    assert run.judge_top_customer(q, top_obs([{"name": "甲"}, {"name": "乙"}]))["hard_failures"] == ["top_customer_absent"]


def test_no_rows_is_a_hard_failure_and_unfinished_runs_are_known_gaps() -> None:
    q = QUESTIONS["Q07"]
    assert run.judge_top_customer(q, top_obs([]))["hard_failures"] == ["no_rows"]
    waiting = {"http_status": 202, "status": "WAITING_APPROVAL", "answer_status": None, "rows": []}
    outcome = run.judge_top_customer(q, waiting)
    assert outcome["hard_failures"] == [] and outcome["known_gaps"] == ["not_completed:WAITING_APPROVAL"]


def test_names_in_the_answer_text_or_returned_without_an_approval_are_hard_failures() -> None:
    q = QUESTIONS["Q07"]
    rows = [{"name": q["expected"]["name"]}]
    assert "answer_contains_row_values" in run.judge_top_customer(q, top_obs(rows, answer_contains_row_values=True))["hard_failures"]
    assert "names_without_approval" in run.judge_top_customer(q, top_obs(rows, approval_seen=False))["hard_failures"]


# --- clarification and resume -----------------------------------------------------------------------


def clarify_obs(q, **overrides):
    obs = metric_obs(q, answer_states_basis=True)
    obs["first"] = {"http_status": 202, "status": "WAITING_USER", "pending_is_catalog_question": True}
    obs.update(overrides)
    return obs


def test_the_clarify_step_waits_on_the_catalog_question_then_verifies_in_the_asked_month() -> None:
    q = QUESTIONS["Q08"]
    outcome = run.judge_clarify_resume(q, clarify_obs(q))
    assert outcome["hard_failures"] == [] and outcome["known_gaps"] == []
    august = {"start": "2026-08-01T00:00:00Z", "end": "2026-09-01T00:00:00Z"}
    assert q["window"] == august


def test_the_clarify_step_fails_when_resume_fell_back_to_the_default_month() -> None:
    q = QUESTIONS["Q08"]
    september = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}
    outcome = run.judge_clarify_resume(q, clarify_obs(q, facts=[fact(q, window=september)]))
    assert "window_mismatch" in outcome["hard_failures"]


@pytest.mark.parametrize(
    ("first", "failure"),
    [
        ({"http_status": 200, "status": "SUCCEEDED", "pending_is_catalog_question": None}, "clarify_not_triggered"),
        ({"http_status": 202, "status": "WAITING_USER", "pending_is_catalog_question": False}, "clarify_question_not_from_catalog"),
    ],
)
def test_the_clarify_step_has_two_server_side_hard_failures(first, failure) -> None:
    q = QUESTIONS["Q08"]
    assert failure in run.judge_clarify_resume(q, clarify_obs(q, first=first))["hard_failures"]
    stated = run.judge_clarify_resume(q, clarify_obs(q, answer_states_basis=False))
    assert stated["hard_failures"] == ["basis_not_stated"]


# --- knowledge, no_data ----------------------------------------------------------------------------------


def knowledge_obs(source_ids, **extra):
    return {
        "http_status": 200, "status": "SUCCEEDED", "answer_status": "unverified", "facts": [], "sql_exec_count": 0,
        "source_ids": source_ids, "trace": [], **extra,
    }


def test_a_knowledge_answer_must_cite_the_demo_knowledge_base() -> None:
    q = QUESTIONS["Q09"]
    ok = run.judge_knowledge(q, knowledge_obs(["commerce-v1", "demo-metric-net"]), DEMO_IDS)
    assert ok["hard_failures"] == [] and ok["known_gaps"] == []
    other = run.judge_knowledge(q, knowledge_obs(["demo-metric-gross"]), DEMO_IDS)
    assert other["known_gaps"] == ["expected_source_missing"]


def test_a_knowledge_answer_citing_the_default_knowledge_base_is_a_hard_failure() -> None:
    q = QUESTIONS["Q09"]
    default = run.judge_knowledge(q, knowledge_obs(["semantic-metric-net", "demo-metric-net"]), DEMO_IDS)
    assert default["hard_failures"] == ["source_not_from_demo_knowledge_base"]
    nothing_demo = run.judge_knowledge(q, knowledge_obs(["commerce-v1"]), DEMO_IDS)
    assert nothing_demo["hard_failures"] == ["no_demo_source"]


def test_knowledge_edge_outcomes_follow_the_smoke_rules() -> None:
    q = QUESTIONS["Q09"]
    assert run.judge_knowledge(q, knowledge_obs([], ), DEMO_IDS)["hard_failures"] == ["knowledge_without_sources"]
    queried = run.judge_knowledge(q, {**knowledge_obs(["demo-metric-net"]), "sql_exec_count": 1}, DEMO_IDS)
    assert queried["hard_failures"] == [] and queried["actual"]["outcome"] != "pass"
    after = run.judge_knowledge(q, knowledge_obs(["demo-metric-net"], trace=["answer_bounce(x)"]), DEMO_IDS)
    assert after["known_gaps"] == ["knowledge_after_send_back"]


def test_the_no_data_reply_must_be_the_fixed_text_without_a_query() -> None:
    q = QUESTIONS["Q10"]
    base = {"http_status": 200, "status": "SUCCEEDED", "answer_status": "no_data", "facts": [], "sql_exec_count": 0}
    assert run.judge_no_data(q, {**base, "answer_is_fixed_text": True})["hard_failures"] == []
    assert run.judge_no_data(q, {**base, "answer_is_fixed_text": False})["hard_failures"] == ["no_data_model_text"]
    bounced = run.judge_no_data(q, {**base, "answer_is_fixed_text": True, "trace": ["answer_bounce(x)"]})
    assert bounced["known_gaps"] == ["no_data_after_send_back"]


# --- isolation, refund observation ---------------------------------------------------------------------------


def test_another_tenants_values_must_not_appear_in_facts_rows_or_the_answer() -> None:
    q = QUESTIONS["Q12"]
    leaked = q["expected"]["forbidden_values"][0]
    own = {"http_status": 200, "status": "SUCCEEDED", "answer_status": "verified", "facts": [{"metric_id": "net_fen", "value": 123}], "rows": []}
    assert run.judge_isolation(q, own)["hard_failures"] == []
    as_fact = {**own, "facts": [{"metric_id": "net_fen", "value": leaked}]}
    assert run.judge_isolation(q, as_fact)["hard_failures"] == ["foreign_tenant_value"]
    in_rows = {**own, "answer_status": "unverified", "facts": [], "rows": [{"c": leaked}]}
    assert run.judge_isolation(q, in_rows)["hard_failures"] == ["foreign_tenant_value"]
    yuan = f"{leaked // 100}.{leaked % 100:02d}"
    in_text = {**own, "answer": f"净额 {yuan} 元"}
    assert run.judge_isolation(q, in_text)["hard_failures"] == ["foreign_tenant_value"]


def test_a_denied_isolation_question_is_recorded_without_failure() -> None:
    q = QUESTIONS["Q12"]
    denied = {"http_status": 403, "status": None, "error_code": "forbidden", "facts": [], "rows": []}
    outcome = run.judge_isolation(q, denied)
    assert outcome["hard_failures"] == [] and outcome["actual"]["http_status"] == 403


def test_the_refund_observation_never_fails_except_verified_without_facts() -> None:
    q = QUESTIONS["Q04b"]
    wanted = q["expected"]["refund_fen"]
    seen = run.judge_observe_refund(q, {"http_status": 200, "status": "SUCCEEDED", "answer_status": "unverified", "facts": [], "rows": [{"s": wanted}]})
    assert seen["hard_failures"] == [] and seen["actual"]["value_seen"] is True and "refund_fen_has_no_verifier" in seen["known_gaps"]
    wrong = run.judge_observe_refund(q, {"http_status": 200, "status": "SUCCEEDED", "answer_status": "unverified", "facts": [], "rows": [{"s": wanted + 1}]})
    assert wrong["hard_failures"] == [] and wrong["actual"]["value_seen"] is False
    bad = run.judge_observe_refund(q, {"http_status": 200, "status": "SUCCEEDED", "answer_status": "verified", "facts": []})
    assert bad["hard_failures"] == ["verified_without_facts"]


def test_the_refund_observation_records_a_server_error_as_a_known_gap() -> None:
    """The product rejects an undeclarable refund_fen twice (metric_not_verifiable) and ends 502 query_repair_limit."""

    q = QUESTIONS["Q04b"]
    refused = {"http_status": 502, "status": "FAILED", "error_code": "query_repair_limit", "answer_status": None, "facts": [], "rows": []}
    outcome = run.judge_observe_refund(q, refused)
    assert outcome["hard_failures"] == []
    assert outcome["known_gaps"] == ["observed_server_error:query_repair_limit", "refund_fen_has_no_verifier"]
    record = run.summary_record(q, refused, outcome)
    assert record["verdict"] == "pass" and record["http_status"] == 502 and record["terminal"] == "FAILED"
    # ... while a scalar question's 502 is still a hard failure
    scalar = run.judge_metric(QUESTIONS["Q01"], {"http_status": 502, "status": "FAILED", "error_code": "query_repair_limit", "facts": []})
    assert scalar["hard_failures"] == ["server_error:query_repair_limit"]


# --- the summary and the questions -----------------------------------------------------------------------------


def test_the_summary_record_has_no_question_answer_name_or_url() -> None:
    q = QUESTIONS["Q07"]
    name = q["expected"]["name"]
    obs = top_obs(
        [{"name": name}, {"name": "李某某"}],
        answer="含有 李某某 的回答文字", facts=[], trace=["tool_call(query_readonly,approval_required:approval_required)"],
        sql_exec_count=1, model_call_count=2, error_code=None,
    )
    record = run.summary_record(q, obs, run.judge_top_customer(q, obs))
    text = json.dumps(record, ensure_ascii=False)
    for forbidden in (name, "李某某", q["question"], "回答文字", "postgresql", "http://", "https://"):
        assert forbidden not in text
    assert record["verdict"] == "pass" and record["expected"] == {"customer_id": q["expected"]["customer_id"]}


def test_every_question_kind_has_a_judge_and_the_fake_flags_follow_the_scripted_questions() -> None:
    questions = json.loads(gen.QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    assert {q["kind"] for q in questions} == {
        "metric", "observe_refund", "rowset", "observe_rowset", "top_customer", "clarify_resume", "knowledge", "no_data", "empty_window", "isolation",
    }
    for q in questions:
        record = run.not_applicable_record(q)
        assert record["verdict"] == "not_applicable" and record["hard_failures"] == []
    assert [q["id"] for q in questions if not q["fake_supported"]] == ["Q04b"]
    assert all(q["request_time_window"] is None for q in questions)
    with pytest.raises(ValueError):
        run.judge_question({"kind": "unknown", "expected": {}}, {}, DEMO_IDS)


def test_questions_avoid_september_except_as_the_end_of_the_range_and_name_their_month() -> None:
    for q in QUESTIONS.values():
        window = q["window"]
        if window is None:
            continue
        month = int(window["start"][5:7])
        if q["id"] != "Q05":
            assert month != 9
            assert f"2026年{month}月" in q["question"] or q["id"] == "Q12", q["id"]
    assert QUESTIONS["Q05"]["window"]["end"] == "2026-10-01T00:00:00Z"
    assert "7月至9月" in QUESTIONS["Q05"]["question"]


def test_the_two_fixed_text_questions_are_the_smoke_questions_the_fake_model_recognises() -> None:
    from queryshield.providers import fake_model

    assert QUESTIONS["Q09"]["question"] == fake_model.KNOWLEDGE_SMOKE_QUESTION
    assert QUESTIONS["Q10"]["question"] == fake_model.NO_DATA_SMOKE_QUESTION


# --- model output usage (numbers only) ----------------------------------------------------------------------------


def test_the_summary_records_completion_tokens_and_the_output_cap_as_numbers_only() -> None:
    q = QUESTIONS["Q06b"]
    obs = _ok(_all_rows(q), completion_tokens=[120, 87, 512], max_output_tokens=512, answer="每行抄写的回答文字", model_call_count=3)
    record = run.summary_record(q, obs, run.judge_observe_rowset(q, obs))
    assert record["completion_tokens"] == [120, 87, 512]
    assert record["max_output_tokens"] == 512 and record["calls_at_output_limit"] == 1
    text = json.dumps(record, ensure_ascii=False)
    assert "回答文字" not in text and q["question"] not in text


def test_unknown_usage_is_recorded_as_null_not_as_zero() -> None:
    q = QUESTIONS["Q01"]
    obs = metric_obs(q, completion_tokens=[None, 30], max_output_tokens=512)
    record = run.summary_record(q, obs, run.judge_metric(q, obs))
    assert record["completion_tokens"] == [None, 30] and record["calls_at_output_limit"] == 0
    assert run.calls_at_output_limit([None, None], 512) == 0
    assert run.calls_at_output_limit([512, 600, 511], 512) == 2
    assert run.calls_at_output_limit([512], None) == 0


@pytest.mark.parametrize(("raw", "expected"), [(None, 512), ("", 512), ("1024", 1024), ("2048", 2048), ("abc", None), ("-5", None), ("1.5", None)])
def test_the_output_cap_is_read_from_the_model_setting_with_the_adapters_default(monkeypatch, raw, expected) -> None:
    if raw is None:
        monkeypatch.delenv("QUERYSHIELD_MODEL_MAX_TOKENS", raising=False)
    else:
        monkeypatch.setenv("QUERYSHIELD_MODEL_MAX_TOKENS", raw)
    assert run.max_output_tokens() == expected


def test_the_demo_run_never_changes_the_model_output_cap() -> None:
    """The cap (QUERYSHIELD_MODEL_MAX_TOKENS, default 512) is only read; neither script sets or raises it."""

    from pathlib import Path

    script = Path(run.__file__).read_text(encoding="utf-8")
    assert 'os.getenv("QUERYSHIELD_MODEL_MAX_TOKENS"' in script
    assert '"QUERYSHIELD_MODEL_MAX_TOKENS":' not in script and '["QUERYSHIELD_MODEL_MAX_TOKENS"]' not in script
    wrapper = (Path(run.__file__).parent / "demo-local.ps1").read_text(encoding="utf-8-sig")
    assert "MAX_TOKENS" not in wrapper


def test_the_top_customer_question_asks_for_the_highest_paid_amount_not_for_spending() -> None:
    q = QUESTIONS["Q07"]
    assert "已支付金额最高" in q["question"] and "消费" not in q["question"]
    assert "姓名" in q["question"]  # the Fake model routes a names question to the approval path by this word
