"""B3c-1: the ask review only sends the model back on a strong signal, a time-range
ask is never pushed into a guessed window, a declaration the question contradicts
is refused before SQL, and the basis sentence always quotes the catalog phrase.
"""

from __future__ import annotations

import json

import pytest

from queryshield.agent import BoundedAgent, ModelCallStore
from queryshield.agent.context import build_context
from queryshield.agent.graph import MAX_CLARIFICATION_BOUNCES
from queryshield.agent.metric_intent import CLARIFICATION_APPLY_RULE, build_metric_binding, declaration_examples
from queryshield.agent.parallel import BranchExecution, ParallelPlan, ParallelScheduler
from queryshield.agent.runtime import http_status_for_run
from queryshield.agent.tool_execution import MetricContradictsQuestionError, check_clarification
from queryshield.catalog import load_catalog, load_default_catalog
from queryshield.catalog.phrases import (
    CONTRADICTION_ACTION,
    NOT_NEEDED_ACTION,
    check_declaration,
    metric_basis_note,
    not_needed_hint,
    read_clarifications,
    review_ask,
)
from test_b3b_clarification import (  # noqa: E402 - pytest prepend import mode puts tests/ on sys.path
    BAD_SQL,
    DEVELOPMENT_TABLE,
    GROSS_SQL,
    METRIC_BASIS_QUESTION,
    MODEL_ASK,
    ROOT,
    SEPTEMBER,
    _agent,
    _ask,
    _b0,
    _cite,
    _context,
    _query,
    _resume,
    _Scripted,
    _start,
    _tools,
    service,  # noqa: F401 - pytest fixture
)


NET_AS_GROSS_SQL = _query(GROSS_SQL, ("net_fen",))  # the model picks net_fen; SQL never runs
TIME_ASK = "请问要查哪个时间范围的支付金额？"
MB, OS, RW = "clarify.metric_basis", "clarify.order_status_scope", "clarify.refund_window"


# ---------------------------------------------------------------------------
# catalog-v4
# ---------------------------------------------------------------------------


def test_catalog_v4_is_v3_without_the_generic_markers_and_with_one_business_phrase() -> None:
    v3 = json.loads((ROOT / "fixtures" / "semantic" / "catalog-v3.json").read_text(encoding="utf-8"))
    v4 = json.loads((ROOT / "fixtures" / "semantic" / "catalog-v4.json").read_text(encoding="utf-8"))
    assert (v3["catalog_version"], v4["catalog_version"]) == ("catalog-v3", "catalog-v4")
    gross3 = next(entry for entry in v3["entries"] if entry["id"] == "metric.gross_fen")
    gross4 = next(entry for entry in v4["entries"] if entry["id"] == "metric.gross_fen")
    assert [phrase for phrase in gross4["phrases"] if phrase not in gross3["phrases"]] == ["毛额"]
    basis3 = next(rule for rule in v3["clarifications"] if rule["id"] == MB)
    basis4 = next(rule for rule in v4["clarifications"] if rule["id"] == MB)
    assert [marker for marker in basis3["ask_markers"] if marker not in basis4["ask_markers"]] == ["金额", "总额"]
    # Everything else is v3 verbatim.
    gross3["phrases"] = gross4["phrases"]
    basis3["ask_markers"] = basis4["ask_markers"]
    v3["catalog_version"] = v4["catalog_version"]
    assert v3 == v4
    assert load_catalog(ROOT / "fixtures" / "semantic" / "catalog-v3.json").catalog_version == "catalog-v3"


# ---------------------------------------------------------------------------
# The ask judgment table (ticket section 7 item 2)
# ---------------------------------------------------------------------------

# id -> (question, clarification_id, ask, B3b verdict, B3c-1 verdict).  A verdict is
# "decision" or "decision rule"; mt = model_text, cq = catalog_question, nn = not_needed.
ASK_TABLE = {
    # 1.2: time-range asks that mention the metric (B3b sent them back)
    "T1": ("2026年9月支付金额", None, TIME_ASK, "nn mb", "mt"),
    "T2": ("支付金额是多少？", None, TIME_ASK, "nn mb", "mt"),
    "T3": ("支付金额是多少？", None, "请问您想看哪个月的金额？", "nn mb", "mt"),
    "T4": ("支付金额是多少？", None, "请问是哪一个月？", "mt", "mt"),
    "T5": ("已支付订单总额是多少", None, "请问统计哪个月的总额？", "nn mb", "mt"),
    "T6": ("2026年9月已支付订单数", None, "请问要看哪个月的订单数？", "nn os", "mt"),
    "T7": ("退款后净额是多少", None, "请问是哪个月的退款后净额？", "nn mb", "mt"),
    # one-value confirmation asks (the O2 cost)
    "C1": ("2026年9月支付金额", None, "是否按支付金额统计？", "nn mb", "mt"),
    "C2": ("2026年9月已支付订单总额", None, "确认一下，是统计已支付订单总额吗？", "nn mb", "mt"),
    "C3": ("2026年9月支付金额", None, "要不要扣除退款？", "nn mb", "mt"),
    # two values: a real choice
    "V1": ("2026年9月支付金额", None, MODEL_ASK, "nn mb", "nn mb"),
    "V2": ("2026年9月支付金额", None, "要毛额还是净额？", "nn mb", "nn mb"),
    "V3": ("2026年9月已支付订单数", None, "只统计已支付，还是包含已取消？", "nn os", "nn os"),
    # a rule's own marker
    "M1": ("2026年9月支付金额", None, "请确认统计口径", "nn mb", "nn mb"),
    "M2": ("2026年9月支付金额", None, "请问按哪种统计方式？", "nn mb", "nn mb"),
    "M3": ("2026年9月已支付订单数", None, "请确认订单范围", "nn os", "nn os"),
    "M4": ("2026年9月退款后净额", None, "退款按哪个退款窗口计算？", "nn rw", "nn rw"),
    # clarification_id
    "I1": ("2026年9月支付金额", MB, METRIC_BASIS_QUESTION, "nn mb", "nn mb"),
    "I2": ("支付金额是多少？", MB, "请问要查哪个月？", "nn mb", "nn mb"),
    "I3": ("2026年9月销售额是多少？", MB, METRIC_BASIS_QUESTION, "cq mb", "cq mb"),
    # time and basis in one ask
    "B1": ("支付金额是多少？", None, "请问要查哪个月，按支付金额还是退款后净额统计？", "nn mb", "nn mb"),
    "B2": ("销售额是多少？", None, "请问要查哪个月、按支付金额还是退款后净额统计？", "cq mb", "cq mb"),
    "B3": ("支付金额是多少？", None, "请问要查哪个月的数据？统计口径是什么？", "nn mb", "nn mb"),
    # the rule is still open: ask on the catalog question
    "O1": ("2026年9月销售额是多少？", None, "请问要查哪个月的销售额？", "cq mb", "cq mb"),
    # only ambiguous wording / wording outside the table
    "O2": ("2026年9月支付金额", None, "你要看哪种销售额？", "nn mb", "mt"),
    "O3": ("2026年9月支付金额", None, "是否包含退款？", "mt", "mt"),
    "O4": ("2026年9月支付金额", None, "按gross还是net统计？", "mt", "mt"),
    # a rule the question never mentions, touched only weakly
    "S1": ("2026年9月已支付订单数", None, "要不要同时看金额？", "cq mb", "mt"),
    "S2": ("2026年9月已支付订单数", None, "要不要同时看支付金额？", "cq mb", "mt"),
}
_DECISIONS = {"mt": "model_text", "cq": "catalog_question", "nn": "not_needed"}
_RULES = {"mb": MB, "os": OS, "rw": RW}


def _verdict(text: str) -> tuple[str, str | None]:
    parts = text.split()
    return _DECISIONS[parts[0]], _RULES[parts[1]] if len(parts) > 1 else None


@pytest.mark.parametrize("row", sorted(ASK_TABLE))
def test_ask_judgment_table(row: str) -> None:
    question, clarification_id, ask, _b3b, expected = ASK_TABLE[row]
    verdict = review_ask(read_clarifications(load_default_catalog(), question), clarification_id, ask)
    assert (verdict.decision, verdict.rule.id if verdict.rule else None) == _verdict(expected)


def test_the_judgment_table_changes_exactly_the_documented_rows() -> None:
    changed = sorted(row for row, (*_, b3b, b3c1) in ASK_TABLE.items() if b3b != b3c1)
    assert changed == ["C1", "C2", "C3", "O2", "S1", "S2", "T1", "T2", "T3", "T5", "T6", "T7"]


# R1: the signal each verdict rests on (a fixed identifier in the review event).
ASK_SIGNALS = {
    "T1": "weak_only", "T2": "weak_only", "T3": "none", "T4": "none", "T5": "none", "T6": "weak_only", "T7": "weak_only",
    "C1": "weak_only", "C2": "weak_only", "C3": "weak_only",
    "V1": "two_values", "V2": "two_values", "V3": "two_values",
    "M1": "marker", "M2": "marker", "M3": "marker", "M4": "marker",
    "I1": "clarification_id", "I2": "clarification_id", "I3": "clarification_id",
    "B1": "two_values", "B2": "two_values", "B3": "marker",
    "O1": "weak_only", "O2": "weak_only", "O3": "none", "O4": "none",
    "S1": "none", "S2": "weak_only",
}


@pytest.mark.parametrize("row", sorted(ASK_TABLE))
def test_ask_verdict_names_the_signal_it_rests_on(row: str) -> None:
    question, clarification_id, ask, _b3b, _expected = ASK_TABLE[row]
    verdict = review_ask(read_clarifications(load_default_catalog(), question), clarification_id, ask)
    assert verdict.signal == ASK_SIGNALS[row]


def test_review_events_record_the_signal_of_a_bounce_and_the_recovered_ask() -> None:
    """The local Real shape: a basis ask is sent back, the model then asks for the time."""

    tools, executor = _tools()
    result = _agent(_Scripted([_ask(), _ask(TIME_ASK)]), tools).run(_context("run-b3c1-r1"), "支付金额是多少？")
    assert result.status == "waiting_user"
    assert result.action == {"type": "ask_user", "question": TIME_ASK}
    reviews = [event for event in result.events if event["kind"] == "clarification_review"]
    assert [(event["decision"], event.get("error_code"), event["signal"]) for event in reviews] == [
        ("not_needed", "clarification_not_needed", "two_values"),
        ("model_text", None, "weak_only"),
    ]
    assert executor.executed == []


# The development set (B3b record section 4): the basis ask is judged as in B3b
# (tests/test_b3b_clarification.py::test_development_set_judgment_table); a
# time-range ask naming the question's own wording now keeps the model's text,
# except where the wording leaves the basis open.
DEVELOPMENT_TIME_ASK = {
    "gross-total-fen": "mt",
    "paid-order-count": "mt",
    "join-aggregate-by-customer": "mt",
    "empty-window-zero-aggregate": "mt",
    "single-repair-budget": "mt",
    "security-cross-tenant-filter": "mt",
    "rephrased-dual-metric-count-and-gross": "mt",
    "queries-net-after-refund": "mt",
    "queries-ambiguous-sales-asks-user": "cq mb",
    "smoke-sync": "mt",
    "smoke-async": "mt",
    "smoke-approval": "mt",
}


@pytest.mark.parametrize("case", sorted(DEVELOPMENT_TABLE))
def test_development_set_time_ask_column(case: str) -> None:
    question, explicit, ambiguous, *_ = DEVELOPMENT_TABLE[case]
    wording = (explicit or ambiguous or ["数据"])[0]
    reading = read_clarifications(load_default_catalog(), question)
    verdict = review_ask(reading, None, f"请问要查哪个时间范围的{wording}？")
    assert (verdict.decision, verdict.rule.id if verdict.rule else None) == _verdict(DEVELOPMENT_TIME_ASK[case])
    # The basis ask of B3b is unchanged for every development question.
    assert review_ask(reading, None, MODEL_ASK).decision == DEVELOPMENT_TABLE[case][5]


def test_the_smoke_time_question_keeps_a_time_ask_and_still_bounces_a_basis_ask() -> None:
    reading = read_clarifications(load_default_catalog(), "支付金额是多少？")
    assert review_ask(reading, None, TIME_ASK).decision == "model_text"
    assert review_ask(reading, None, MODEL_ASK).decision == "not_needed"


def test_an_open_rule_still_gets_the_catalog_question() -> None:
    for ask in (TIME_ASK, "请问要看哪个月的销售额？", MODEL_ASK):
        verdict = review_ask(read_clarifications(load_default_catalog(), "2026年9月销售额是多少？"), None, ask)
        assert verdict.decision == "catalog_question" and verdict.rule.id == MB


# ---------------------------------------------------------------------------
# No time range: the contract and the bounce hint
# ---------------------------------------------------------------------------


def _server(**kwargs) -> dict[str, object]:
    return json.loads(build_context(_context(), "q", **kwargs).messages[0]["content"].split("\n", 1)[1])


def test_contract_says_to_ask_only_for_the_time_range_without_an_id() -> None:
    server = _server()
    assert server["metric_declaration"]["clarifications"]["apply_rule"] == CLARIFICATION_APPLY_RULE
    assert "If neither the question nor request_time_window states a time range, ask_user only for it, without clarification_id; never guess one." in CLARIFICATION_APPLY_RULE
    note = server["action_contract"]["actions"]["ask_user"]["clarification_id"]
    assert note.startswith("Required when asking about a catalog rule; omit it for a time-range ask.")


# The B3b bounce text, spelled out: R2 (controller 2026-09-30) withdrew the
# window-dependent variant of ticket 3.2 after it sent a real model on the W05
# join question (month in the question, no request window) back to asking.
B3B_NOT_NEEDED_ACTION = (
    "Do not ask_user about this rule: the question already names its value (or the rule has one value). "
    "Send a tool_call named query_readonly that declares the named metric."
)


def test_bounce_hint_is_the_b3b_text_with_or_without_a_request_window() -> None:
    catalog = load_default_catalog()
    reading = read_clarifications(catalog, "支付金额是多少？")
    rule = catalog.clarification(MB)
    without = not_needed_hint(reading, rule, None)
    with_window = not_needed_hint(reading, rule, {**SEPTEMBER, "timezone": "UTC"})
    assert without["action"] == with_window["action"] == NOT_NEEDED_ACTION == B3B_NOT_NEEDED_ACTION
    assert without["request_time_window"] is None
    assert with_window["request_time_window"] == SEPTEMBER
    assert {key: value for key, value in without.items() if key != "request_time_window"} == {
        key: value for key, value in with_window.items() if key != "request_time_window"
    }
    assert without["named_phrases"] == ["支付金额"] and without["declare_metrics"] == ["gross_fen"]


def test_a_time_ask_carrying_the_rule_id_gets_the_b3b_hint_and_a_later_time_ask_waits() -> None:
    tools, executor = _tools()
    model = _Scripted([_ask("请问要查哪个月？", MB), _ask(TIME_ASK)])
    result = _agent(model, tools).run(_context("run-b3c1-id-on-time"), "支付金额是多少？")
    assert result.status == "waiting_user"
    assert result.action == {"type": "ask_user", "question": TIME_ASK}
    hint = json.loads(next(m["content"] for m in model.messages[1] if "clarification_not_needed" in m["content"]).split("\n", 1)[1])
    assert hint["repair_hint"]["action"] == B3B_NOT_NEEDED_ACTION
    assert hint["repair_hint"]["request_time_window"] is None
    assert executor.executed == []


def test_undated_question_waits_for_the_time_then_answers_from_the_catalog_phrase(service) -> None:
    _, executor = service
    run = _start(service, "支付金额是多少？", [_ask(TIME_ASK)], window=None)
    assert run["status"] == "WAITING_USER"
    assert run["checkpoint"]["agent_checkpoint"]["waiting_question"] == TIME_ASK
    assert run["checkpoint"]["agent_checkpoint"]["waiting_clarification_id"] is None
    assert executor.executed == []
    done = _resume(service, run, "2026年9月", [_query(), _cite])
    assert done["status"] == "SUCCEEDED"
    assert [fact["metric_id"] for fact in done["facts"]["facts"]] == ["gross_fen"]
    assert done["answer"].split("\n")[1] == "口径：支付订单总额（gross_fen）；依据：问题中提到‘支付金额’。"
    assert len(executor.executed) == 1


# ---------------------------------------------------------------------------
# A declaration the question contradicts
# ---------------------------------------------------------------------------


def _check(question: str, metrics, answers=(), confirmed=()) -> str | None:
    verdict = check_declaration(read_clarifications(load_default_catalog(), question, answers, confirmed), metrics)
    return None if verdict is None else f"{verdict.kind} {verdict.rule.id}"


@pytest.mark.parametrize(
    ("question", "metrics", "answers", "confirmed"),
    [
        ("按支付金额算的销售额", ["gross_fen"], (), ()),
        ("退款后的销售额", ["net_fen"], (), ()),
        ("2026年9月已支付订单数和支付金额", ["paid_count", "gross_fen"], (), ()),
        ("2026年8月没有订单时已支付订单数和总额是多少", ["paid_count", "gross_fen"], (), ()),
        ("2026年9月支付金额", ["net_fen"], ("要退款后净额",), ()),  # the user chose another value
        ("2026年9月支付金额", ["net_fen"], (), ("net_fen",)),  # confirmed / pre-bound
        ("2026年9月销售额", ["net_fen"], (), ("net_fen",)),
        ("订单总额和净额", ["gross_fen", "net_fen"], (), ()),  # also covers wording the table does not list
        ("支付金额和退款后净额", ["net_fen"], (), ()),
    ],
)
def test_declarations_the_question_does_not_contradict_pass(question, metrics, answers, confirmed) -> None:
    assert _check(question, metrics, answers, confirmed) is None


# The development questions with their declared metrics are judged exactly as in
# B3b (never "contradicts"): tests/test_b3b_clarification.py::test_development_set_judgment_table.


@pytest.mark.parametrize(
    ("question", "metrics"),
    [
        ("2026年9月支付金额", ["net_fen"]),
        ("2026年9月毛额", ["net_fen"]),
        ("2026年9月已支付订单扣除这些订单在9月内的退款后，净额是多少？", ["gross_fen"]),
        ("2026年9月退款后净额", ["gross_fen", "paid_count"]),
    ],
)
def test_a_declaration_naming_none_of_the_questions_values_contradicts(question, metrics) -> None:
    assert _check(question, metrics) == f"contradicts {MB}"
    with pytest.raises(MetricContradictsQuestionError):
        check_clarification(read_clarifications(load_default_catalog(), question), metrics)


def test_contradiction_is_refused_before_sql_then_the_corrected_query_answers() -> None:
    tools, executor = _tools()
    model = _Scripted([NET_AS_GROSS_SQL, _query(), _cite])
    result = _agent(model, tools).run(_context("run-b3c1-contradiction"), "2026年9月支付金额", request_time_window=SEPTEMBER)
    assert result.status == "succeeded"
    assert [fact["metric_id"] for fact in result.facts["facts"]] == ["gross_fen"]
    assert result.repair_count == 0
    rejected = next(event for event in result.events if event["kind"] == "tool_call" and event["status"] == "rejected")
    assert rejected["error_code"] == "metric_contradicts_question" and rejected["clarification_id"] == MB
    bounces = [event for event in result.events if event["kind"] == "clarification_bounce"]
    assert [(event["bounce_index"], event["error_code"]) for event in bounces] == [(1, "metric_contradicts_question")]
    assert len(executor.executed) == 1
    hint = json.loads(next(m["content"] for m in model.messages[1] if "metric_contradicts_question" in m["content"]).split("\n", 1)[1])
    assert hint["repair_hint"] == {
        "action": CONTRADICTION_ACTION,
        "clarification_id": MB,
        "named_phrases": ["支付金额"],
        "declare_metrics": ["gross_fen"],
        "request_time_window": SEPTEMBER,
    }
    assert result.answer.split("\n")[1] == "口径：支付订单总额（gross_fen）；依据：问题中提到‘支付金额’。"


def test_a_second_contradiction_fails_without_sql() -> None:
    tools, executor = _tools()
    model = _Scripted([NET_AS_GROSS_SQL, NET_AS_GROSS_SQL])
    result = _agent(model, tools).run(_context("run-b3c1-contradiction-twice"), "2026年9月支付金额", request_time_window=SEPTEMBER)
    assert result.status == "failed" and result.error_code == "metric_contradicts_question"
    assert executor.executed == []
    assert http_status_for_run("FAILED", "metric_contradicts_question") == 502


@pytest.mark.parametrize(
    ("steps", "error_code"),
    [
        ([_ask(), NET_AS_GROSS_SQL], "metric_contradicts_question"),
        ([NET_AS_GROSS_SQL, _ask()], "clarification_not_needed"),
    ],
    ids=["ask-then-contradiction", "contradiction-then-ask"],
)
def test_ask_bounce_and_contradiction_share_one_bounce(steps, error_code) -> None:
    tools, executor = _tools()
    result = _agent(_Scripted(steps), tools).run(_context("run-b3c1-shared"), "2026年9月支付金额", request_time_window=SEPTEMBER)
    assert MAX_CLARIFICATION_BOUNCES == 1
    assert result.status == "failed" and result.error_code == error_code
    assert executor.executed == []


def test_single_repair_budget_keeps_its_one_repair_after_a_contradiction() -> None:
    """The injected SQL fault is still repaired once: the contradiction uses the bounce, not the repair."""

    tools, executor = _tools()
    model = _Scripted([NET_AS_GROSS_SQL, _query(BAD_SQL), _query(), _cite])
    result = _agent(model, tools).run(_context("run-b3c1-contradiction-repair"), "2026年9月支付金额", request_time_window=SEPTEMBER)
    assert result.status == "succeeded"
    assert result.repair_count == 1
    assert [event["error_code"] for event in result.events if event["kind"] == "clarification_bounce"] == [
        "metric_contradicts_question"
    ]
    assert result.model_call_count == 4


def test_b0_fails_a_contradiction_before_sql() -> None:
    record = _b0("2026年9月支付金额", NET_AS_GROSS_SQL)
    assert record["status"] == "failed" and record["error_code"] == "metric_contradicts_question"
    assert record["executed"] == [] and record["http_status"] == 502


# ---------------------------------------------------------------------------
# Parallel path (O5 and the contradiction check there)
# ---------------------------------------------------------------------------


def _parallel(metrics) -> dict[str, object]:
    return {"type": "parallel_readonly", "metric_ids": list(metrics)}


def _parallel_run(question: str, steps, run_id: str):
    tools, executor = _tools()
    branches: list[str] = []

    def runner(context, metric_id, branch_id) -> BranchExecution:
        branches.append(metric_id)
        return BranchExecution(metric_id, f"result-{metric_id}", ({metric_id: 1},), "2026-09-21T00:00:00Z")

    context = _context(run_id)
    plan = ParallelPlan.from_context(
        context, ("gross_fen", "net_fen", "paid_count"), time_window={**SEPTEMBER, "timezone": "UTC", "interval": "[start,end)"}
    )
    agent = BoundedAgent(_Scripted(steps), tools=tools, call_store=ModelCallStore(), parallel_scheduler=ParallelScheduler(runner))
    result = agent.run(context, question, parallel_plan=plan, request_time_window=SEPTEMBER)
    return result, branches, executor


def test_o5_parallel_declaration_for_ambiguous_wording_waits_on_the_catalog_question() -> None:
    result, branches, executor = _parallel_run("2026年9月销售额和订单数", [_parallel(["gross_fen", "paid_count"])], "run-b3c1-o5")
    assert result.status == "waiting_user"
    assert result.action == {"type": "ask_user", "question": METRIC_BASIS_QUESTION, "clarification_id": MB}
    assert branches == [] and executor.executed == []


def test_parallel_contradiction_is_bounced_then_fails_without_branches() -> None:
    steps = [_parallel(["net_fen", "paid_count"]), _parallel(["net_fen", "paid_count"])]
    result, branches, _ = _parallel_run("2026年9月支付金额和已支付订单数", steps, "run-b3c1-parallel")
    assert result.status == "failed" and result.error_code == "metric_contradicts_question"
    assert branches == []
    assert [event["error_code"] for event in result.events if event["kind"] == "clarification_bounce"] == [
        "metric_contradicts_question"
    ]


# ---------------------------------------------------------------------------
# O4: confirmed or pre-bound metrics
# ---------------------------------------------------------------------------


def _bound(metric_id: str):
    return (build_metric_binding(load_default_catalog(), metric_id, {**SEPTEMBER, "timezone": "UTC"}),)


def test_o4_prebound_net_makes_a_basis_ask_unneeded_then_fails_the_second() -> None:
    tools, executor = _tools()
    result = _agent(_Scripted([_ask(), _ask()]), tools).run(
        _context("run-b3c1-o4-ask"), "2026年9月销售额", metric_bindings=_bound("net_fen"), request_time_window=SEPTEMBER
    )
    assert result.status == "failed" and result.error_code == "clarification_not_needed"
    reviews = [event for event in result.events if event["kind"] == "clarification_review"]
    assert [event["decision"] for event in reviews] == ["not_needed", "not_needed"]
    assert executor.executed == []


def test_o4_prebound_gross_lets_the_gross_declaration_run() -> None:
    tools, executor = _tools()
    result = _agent(_Scripted([_query(), _cite]), tools).run(
        _context("run-b3c1-o4-declare"), "2026年9月销售额", metric_bindings=_bound("gross_fen"), request_time_window=SEPTEMBER
    )
    assert result.status == "succeeded"
    assert [fact["metric_id"] for fact in result.facts["facts"]] == ["gross_fen"]
    assert len(executor.executed) == 1


def test_o4_a_value_chosen_in_the_clarification_lets_its_declaration_run(service) -> None:
    run = _start(service, "2026年9月销售额是多少？", [_query()])
    assert run["status"] == "WAITING_USER"
    net = {"type": "tool_call", "name": "query_readonly", "arguments": declaration_examples(SEPTEMBER)["net_fen"]}
    done = _resume(service, run, "要退款后净额", [net, _cite])
    assert done["status"] == "SUCCEEDED"
    assert [fact["metric_id"] for fact in done["facts"]["facts"]] == ["net_fen"]
    assert "你在追问中选择了‘退款后净额’" in done["answer"]


# ---------------------------------------------------------------------------
# Basis sentence (O3, O10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "metric_id", "quoted"),
    [
        ("2026年9月支付金额", "gross_fen", "支付金额"),
        ("退款后的销售额", "net_fen", "退款后"),
        ("2026年9月gross_fen", "gross_fen", "gross_fen"),
        ("2026年9月毛额", "gross_fen", "毛额"),
    ],
)
def test_basis_sentence_quotes_the_catalog_phrase_with_one_template(question, metric_id, quoted) -> None:
    note = metric_basis_note(read_clarifications(load_default_catalog(), question), metric_id)
    assert f"；依据：问题中提到‘{quoted}’。" in note
    assert "问题中的‘" not in note


def test_basis_never_says_unstated_when_the_question_names_another_value() -> None:
    catalog = load_default_catalog()
    named = metric_basis_note(read_clarifications(catalog, "2026年9月支付金额"), "net_fen")
    assert "没有写明" not in named
    assert named == "口径：退款后净额（net_fen）；依据：按目录定义统计。前提：退款按同一 UTC 窗口内、已支付订单的退款计算。"
    unstated = metric_basis_note(read_clarifications(catalog, "2026年8月没有订单时已支付订单数和总额是多少"), "gross_fen")
    assert "问题中没有写明口径" in unstated


def test_order_scope_wording_is_quoted_for_paid_count() -> None:
    note = metric_basis_note(read_clarifications(load_default_catalog(), "2026年9月已支付订单有几笔？"), "paid_count")
    assert note == "口径：已支付订单数（paid_count）；依据：问题中提到‘已支付订单’。"
