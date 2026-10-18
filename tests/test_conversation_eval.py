"""The multi-turn development evaluation: its cases come from the generator, and its own judgements."""

from __future__ import annotations

import json

from scripts import conversation_eval as ce
from scripts import generate_demo_data as gen


def test_the_case_file_is_the_generators_output() -> None:
    assert ce.CASES_PATH.read_text(encoding="utf-8") == ce.cases_text(gen.generate())


def test_the_cases_cover_every_follow_up_kind() -> None:
    document = json.loads(ce.CASES_PATH.read_text(encoding="utf-8"))
    conversations = document["conversations"]
    tags = {tag for item in conversations for turn in item["turns"] for tag in turn["tags"]}
    assert {
        "carry_metric", "carry_window", "last_month", "compare", "add_grouping", "carry_grouping", "these_customers", "add_rank",
        "carry_rank", "independent", "retract", "fork", "foreign_tenant", "injection", "approval_again", "not_owner", "not_finished",
        "clarify_again", "request_window", "clarify_month_alone", "clarify_no_month", "clarify_last_month",
    } <= tags
    assert 30 <= len(conversations) <= 50
    assert all(2 <= len(item["turns"]) <= 4 for item in conversations)
    assert all(item["turns"][0]["follow"] == "none" for item in conversations)


def _obs(**fields) -> dict:
    return {"http_status": 200, "status": "SUCCEEDED", "answer_status": "unverified", "facts": [], "rows": [], "answer": "", **fields}


SUBSET = {"kind": "subset_rowset", "expected": {"metric_id": "gross_fen", "values": {"c01": 100, "c02": 200}}}


def test_these_customers_must_all_be_in_the_rows_with_their_values() -> None:
    rows = [{"customer_id": "c01", "gross_fen": 100}, {"customer_id": "c02", "gross_fen": 200}, {"customer_id": "c03", "gross_fen": 5}]
    assert ce.judge_subset_rowset(SUBSET, _obs(rows=rows))["hard_failures"] == []
    assert ce.judge_subset_rowset(SUBSET, _obs(rows=rows))["known_gaps"] == []
    wrong = ce.judge_subset_rowset(SUBSET, _obs(rows=[{"customer_id": "c01", "gross_fen": 101}, rows[1]]))
    assert wrong["hard_failures"] == ["rowset_value_mismatch"]
    assert ce.judge_subset_rowset(SUBSET, _obs(rows=rows[:1]))["known_gaps"] == ["carried_customers_missing"]
    assert ce.judge_subset_rowset(SUBSET, _obs(rows=[], status="WAITING_USER"))["known_gaps"] == ["no_rows:WAITING_USER"]


def test_a_refusal_must_have_its_status_and_code_and_create_no_run() -> None:
    turn = {"kind": "rejected", "expected": {"http_status": 404, "error_code": "not_found"}}
    assert ce.judge_rejected(turn, {"http_status": 404, "error_code": "not_found", "run_id": None})["hard_failures"] == []
    assert ce.judge_rejected(turn, {"http_status": 200, "error_code": None, "run_id": "run-1"})["hard_failures"] == [
        "http_status:200", "error_code:None", "run_created",
    ]


def test_names_need_this_turns_approval() -> None:
    rows = [{"customer_id": "c1", "name": "甲"}]
    assert ce.judge_names({}, _obs(rows=rows, approval_seen=True))["hard_failures"] == []
    assert ce.judge_names({}, _obs(rows=rows))["hard_failures"] == ["names_without_approval"]
    assert ce.judge_names({}, _obs(rows=rows, approval_seen=True, answer_contains_row_values=True))["hard_failures"] == ["answer_contains_row_values"]


def _record(turn: int, **fields) -> dict:
    return {"conversation": "C1", "turn": turn, "identity_role": "requester", **fields}


def test_approval_violations_count_each_kind() -> None:
    names = frozenset({"甲"})
    clean = [_record(1, approval_id="approval-1", approval_seen=True, rows_with_names=True), _record(2, follow_up_record='{"customer_ids": ["c1"]}')]
    assert ce.approval_violations(clean, names) == []
    assert ce.approval_violations([_record(1, rows_with_names=True)], names) == ["C1.1:names_without_approval"]
    assert ce.approval_violations([_record(1, approval_id="a"), _record(2, approval_id="a")], names) == ["C1.2:approval_reused"]
    assert ce.approval_violations([_record(2, follow_up_record='{"x": "甲"}')], names) == ["C1.2:name_in_follow_up_record"]
    # An approver reads names without an approval.
    assert ce.approval_violations([_record(1, rows_with_names=True, identity_role="approver")], names) == []


def test_metrics_report_turns_conversations_wrong_carry_and_message_sizes() -> None:
    records = [
        {"conversation": "C1", "turn": 1, "verdict": "pass", "correct": True, "tags": []},
        {"conversation": "C1", "turn": 2, "verdict": "pass", "correct": True, "tags": ["independent"], "follow_up_message_bytes": 300},
        {"conversation": "C2", "turn": 1, "verdict": "pass", "correct": True, "tags": []},
        {"conversation": "C2", "turn": 2, "verdict": "fail", "correct": False, "tags": ["retract"], "follow_up_message_bytes": 320},
        {"conversation": "C3", "turn": 1, "verdict": "not_applicable", "correct": False, "tags": []},
    ]
    result = ce.metrics(records, ["C2.2:approval_reused"])
    assert result["turns"] == 4 and result["turn_accuracy"] == 0.75
    assert result["conversations"] == 2 and result["conversation_accuracy"] == 0.5
    assert (result["wrong_carry_turns"], result["wrong_carry_failed"], result["wrong_carry_rate"]) == (2, 1, 0.5)
    assert result["approval_violations"] == 1
    assert result["follow_up_message_bytes"] == {"2": 320}


def test_the_follow_target_is_the_previous_turn_or_the_named_one() -> None:
    assert ce._follow_target({"follow": "none"}, 0, []) is None
    assert ce._follow_target({"follow": "previous"}, 2, ["run-1", "run-2"]) == "run-2"
    assert ce._follow_target({"follow": "turn:1"}, 2, ["run-1", "run-2"]) == "run-1"
    assert ce._follow_target({"follow": "previous"}, 1, [None]) == "run-not-created"


def test_the_evidence_dir_is_required_to_run(capsys) -> None:
    try:
        ce.main(["--mode", "fake"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("ran without --evidence-dir")


def _fact(metric_id: str, start: str, end: str, value: int) -> dict:
    return {"metric_id": metric_id, "value": value, "time_window": {"start": start, "end": end, "timezone": "UTC"}}


SEP = ("2026-09-01T00:00:00Z", "2026-10-01T00:00:00Z")
JUL = ("2026-07-01T00:00:00Z", "2026-08-01T00:00:00Z")


def _turn(conversation_id: str, index: int) -> tuple[dict, dict]:
    document = json.loads(ce.CASES_PATH.read_text(encoding="utf-8"))
    conversation = next(item for item in document["conversations"] if item["id"] == conversation_id)
    return conversation, conversation["turns"][index]


def _metric_obs(*facts) -> dict:
    return _obs(answer_status="verified", facts=list(facts))


def test_a_metric_turn_with_the_previous_turns_fact_next_to_the_asked_one_is_a_hard_failure() -> None:
    conversation, turn = _turn("CI01", 1)  # a new question after September's paid amount
    asked = _fact("paid_count", *JUL, turn["expected"]["value"])
    previous = _fact("gross_fen", *SEP, ce.gen.expected_metrics(ce.demo_run.demo_data(), "A", *ce.gen._win(9, 9))["gross_fen"])
    assert ce.judge_turn(conversation, 1, turn, _metric_obs(asked), frozenset())["hard_failures"] == []
    judged = ce.judge_turn(conversation, 1, turn, _metric_obs(asked, previous), frozenset())
    assert judged["hard_failures"] == ["extra_fact"]

    conversation, turn = _turn("CM01", 1)  # 退款后净额呢 after September's paid amount
    asked = _fact("net_fen", *SEP, turn["expected"]["value"])
    assert ce.judge_turn(conversation, 1, turn, _metric_obs(asked), frozenset())["hard_failures"] == []
    assert ce.judge_turn(conversation, 1, turn, _metric_obs(asked, previous), frozenset())["hard_failures"] == ["extra_fact"]


AUG = ("2026-08-01T00:00:00Z", "2026-09-01T00:00:00Z")


def test_the_same_metric_for_the_previous_turns_window_too_is_an_extra_fact() -> None:
    # The most common wrong carry: "那8月呢" answered for August and again for the previous September.
    conversation, turn = _turn("CF01", 1)
    data = ce.demo_run.demo_data()
    asked = _fact("gross_fen", *AUG, turn["expected"]["value"])
    previous = _fact("gross_fen", *SEP, ce.gen.expected_metrics(data, "A", *ce.gen._win(9, 9))["gross_fen"])
    assert ce.judge_turn(conversation, 1, turn, _metric_obs(asked), frozenset())["hard_failures"] == []
    judged = ce.judge_turn(conversation, 1, turn, _metric_obs(asked, previous), frozenset())
    # judge_metric also sees two values and a second window; extra_fact names the carried one.
    assert "extra_fact" in judged["hard_failures"]
