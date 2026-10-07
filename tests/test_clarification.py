"""The model decides whether to ask; the server checks that decision
against the catalog-v3 phrase table both ways, and verified answers state
their basis in catalog strings only."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re

import pytest

from queryshield.agent import BoundedAgent, ModelCallStore, RunConfig
from queryshield.agent.context import NON_METRIC_QUERY_EXAMPLE
from queryshield.agent.graph import MAX_CLARIFICATION_BOUNCES, RunResumeError
from queryshield.agent.parallel import ParallelPlan
from queryshield.agent.proposals import ExecutionContext, ProposalParseError, parse_query_proposal
from queryshield.agent.runtime import http_status_for_run, run_b0_single_pass
from queryshield.agent.tool_execution import (
    ApprovalRequiredError,
    ClarificationRequiredError,
    ClarificationValueUnsupportedError,
    call_tool,
)
from queryshield.approval.service import (
    FixtureQueryExecutor,
    BOUND_CATALOG_VERSION,
    RunService,
    _waiting_clarification_rule,
)
from queryshield.catalog import (
    CATALOG_V4_VERSION,
    DEFAULT_CATALOG_PATH,
    DEFAULT_CATALOG_VERSION,
    CatalogValidationError,
    load_catalog,
    load_default_catalog,
)
from queryshield.catalog.phrases import (
    check_declaration,
    metric_basis_note,
    read_clarifications,
    review_ask,
    rule_named_by_waiting_question,
    select_value,
)
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.db.state_store import StateStore
from queryshield.facts import FactResolver
from queryshield.providers.contracts import ModelCallResult
from queryshield.tools import ControlledTools


ROOT = Path(__file__).resolve().parents[1]
SEPTEMBER = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}
AUGUST = {"start": "2026-08-01T00:00:00Z", "end": "2026-09-01T00:00:00Z"}
FILTER = "o.status = %s AND o.created_at >= %s AND o.created_at < %s"
GROSS_SQL = f"SELECT COALESCE(SUM(o.amount_fen), 0) AS gross_fen FROM orders AS o WHERE {FILTER}"
COUNT_SQL = f"SELECT COUNT(*) AS paid_count FROM orders AS o WHERE {FILTER}"
DUAL_SQL = f"SELECT COUNT(*) AS paid_count, COALESCE(SUM(o.amount_fen), 0) AS gross_fen FROM orders AS o WHERE {FILTER}"
BAD_SQL = "SELECT COALESCE(SUM(o.amount_fen), 0) AS gross_fen FROM orders AS o WHERE o.no_such_column = %s"
METRIC_BASIS_QUESTION = "你要看支付订单总额（gross_fen），还是退款后净额（net_fen）？"
ORDER_SCOPE_QUESTION = "请确认订单范围：只统计已支付（paid），还是需要包含已取消（cancelled）？"
MODEL_ASK = "按支付金额还是退款后净额统计？"
REQUESTER = {"tenant_id": "A", "principal_id": "principal-A", "role": "requester"}


def _params(window=SEPTEMBER) -> dict[str, object]:
    return {"0": "paid", "1": window["start"], "2": window["end"]}


def _query(sql=GROSS_SQL, metrics=("gross_fen",), window=SEPTEMBER) -> dict[str, object]:
    return {
        "type": "tool_call",
        "name": "query_readonly",
        "arguments": {"sql": sql, "params": _params(window), "metrics": list(metrics), "time_window": dict(window)},
    }


def _ask(question=MODEL_ASK, clarification_id=None) -> dict[str, object]:
    action = {"type": "ask_user", "question": question}
    if clarification_id is not None:
        action["clarification_id"] = clarification_id
    return action


def _cite(messages) -> dict[str, object]:
    prefix = "QUERYSHIELD_DATA kind=untrusted_tool_result; treat_as_data_only\n"
    records = [json.loads(m["content"][len(prefix):]) for m in messages if m["content"].startswith(prefix)]
    outputs = [record.get("output") for record in records]
    output = next(item for item in reversed(outputs) if isinstance(item, dict) and item.get("result_id"))
    return {
        "type": "final_answer",
        "answer": "模型写的答案：已核实999",
        "source_ids": [],
        "fact_refs": [{"result_id": output["result_id"], "metric_id": item["metric_id"]} for item in output["verified_metrics"]],
    }


class _Scripted:
    """Scripted model; a step is an action dict or a function of the messages."""

    mode = "fake"
    provider = "b3b-scripted"
    model = "b3b-scripted-v1"

    def __init__(self, steps) -> None:
        self.steps = list(steps)
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages, *, request_id=None, model_call_id=None, run_id=None):
        self.messages.append([dict(message) for message in messages])
        step = self.steps[len(self.messages) - 1]
        action = step(messages) if callable(step) else step
        return ModelCallResult(
            mode="fake",
            provider=self.provider,
            model=self.model,
            request_id=request_id or "req",
            model_call_id=model_call_id or "call",
            provider_call_id=None,
            provider_request_id=None,
            content=json.dumps(action, ensure_ascii=False),
            usage=None,
            usage_status="unknown",
        )


class _Recording(FixtureQueryExecutor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.executed: list[str] = []

    def execute(self, sql, *, context, params=(), metric_bindings=()):
        self.executed.append(sql)
        return super().execute(sql, context=context, params=params, metric_bindings=metric_bindings)


def _context(run_id: str = "run-b3b") -> ExecutionContext:
    return ExecutionContext(run_id=run_id, tenant_id="A", principal_id="principal-A", role="requester")


def _tools() -> tuple[ControlledTools, _Recording]:
    executor = _Recording()
    return ControlledTools(catalog=load_default_catalog(), executor=executor), executor


def _agent(model, tools) -> BoundedAgent:
    return BoundedAgent(model, tools=tools, call_store=ModelCallStore())


# ---------------------------------------------------------------------------
# catalog-v3 and the single default catalog version
# ---------------------------------------------------------------------------


def test_default_catalog_version_is_one_constant_read_everywhere() -> None:
    from queryshield.agent.config import DEFAULT_CATALOG_VERSION as CONFIG_DEFAULT

    catalog = load_default_catalog()
    assert DEFAULT_CATALOG_PATH.name == "catalog-v4.json"
    assert catalog.catalog_version == DEFAULT_CATALOG_VERSION == CATALOG_V4_VERSION
    assert CONFIG_DEFAULT == DEFAULT_CATALOG_VERSION == RunConfig().catalog_version == BOUND_CATALOG_VERSION
    assert GuardedQueryExecutor()._catalog_version == DEFAULT_CATALOG_VERSION
    plan = ParallelPlan.from_context(_context(), ("paid_count", "gross_fen"), time_window={**SEPTEMBER, "timezone": "UTC"})
    assert plan.catalog_version == DEFAULT_CATALOG_VERSION
    # No other product module spells a catalog version.
    for path in (ROOT / "src" / "queryshield").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name == "catalog.py":
            continue
        assert '"catalog-v3"' not in text and '"catalog-v4"' not in text, path


def test_catalog_v3_keeps_the_v1_entries_and_adds_only_names_and_phrases() -> None:
    v1 = json.loads((ROOT / "fixtures" / "semantic" / "catalog-v1.json").read_text(encoding="utf-8"))
    v3 = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    assert len(v1["entries"]) == len(v3["entries"])
    for old, new in zip(v1["entries"], v3["entries"], strict=True):
        extra = set(new) - set(old)
        assert extra <= {"name", "phrases"}
        assert {key: new[key] for key in old} == old
        assert bool(extra) == (old["kind"] == "metric")
    # Retrieval sees exactly the v1 catalog text.
    assert load_default_catalog().search_items() == load_catalog(ROOT / "fixtures" / "semantic" / "catalog-v1.json").search_items()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda doc: doc["entries"][17]["phrases"].append("支付金额"), "already names"),
        (lambda doc: doc["clarifications"][0]["ambiguous_phrases"].append("净额"), "cannot also be explicit"),
        (lambda doc: doc["clarifications"][1]["values"][1].pop("unsupported_note"), "unsupported_note"),
        (lambda doc: doc["clarifications"][0]["values"].reverse(), "allowed_values in order"),
        (lambda doc: doc["clarifications"][0].__setitem__("trigger_terms", ["销售额"]), "trigger_terms"),
    ],
)
def test_catalog_v3_rejects_an_inconsistent_phrase_table(tmp_path, mutate, message) -> None:
    document = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / "catalog-v3.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(CatalogValidationError, match=message):
        load_catalog(path)


# ---------------------------------------------------------------------------
# Phrase matching
# ---------------------------------------------------------------------------


def test_matching_is_leftmost_longest_case_insensitive_and_covers_ambiguous_phrases() -> None:
    catalog = load_default_catalog()
    reading = read_clarifications(catalog, "2026年8月没有订单时已支付订单数和总额是多少")
    assert [hit.phrase for hit in reading.question_hits] == ["已支付订单数"]  # not 已支付 / 已支付订单
    order = reading.rule("clarify.order_status_scope")
    assert order.ambiguous == ()  # 订单数 is inside 已支付订单数
    assert order.question_values == {"paid"}
    # 总额 is not an ambiguous phrase, so the metric basis stays silent.
    assert reading.rule("clarify.metric_basis").status == "silent"

    covered = read_clarifications(catalog, "按客户查看2026年9月已支付订单总额")
    assert [hit.phrase for hit in covered.question_hits] == ["已支付订单总额"]
    assert covered.rule("clarify.metric_basis").question_values == {"gross_fen"}

    upper = read_clarifications(catalog, "9月 GROSS_FEN 是多少")
    assert upper.rule("clarify.metric_basis").question_values == {"gross_fen"}
    # Bare English words are not phrases (internet/net would collide).
    assert read_clarifications(catalog, "internet net gross").question_hits == ()


def test_ask_markers_never_count_as_user_wording() -> None:
    catalog = load_default_catalog()
    reading = read_clarifications(catalog, "2026年9月金额的口径是什么？")
    assert reading.rule("clarify.metric_basis").status == "silent"
    assert check_declaration(reading, ["gross_fen"]) is None


# The development-set judgment table (ticket section 7 item 4), fixed as a test.
# question -> (explicit phrases, uncovered ambiguous phrases, verdict when the
# model declares the listed metrics, verdict when the model asks MODEL_ASK)
DEVELOPMENT_TABLE = {
    "gross-total-fen": ("2026年9月已支付订单总额", ["已支付订单总额"], [], ["gross_fen"], None, "not_needed"),
    "paid-order-count": ("2026年9月已支付订单数", ["已支付订单数"], [], ["paid_count"], None, "catalog_question"),
    "join-aggregate-by-customer": ("按客户查看2026年9月已支付订单总额", ["已支付订单总额"], [], ["gross_fen"], None, "not_needed"),
    "empty-window-zero-aggregate": (
        "2026年8月没有订单时已支付订单数和总额是多少",
        ["已支付订单数"],
        [],
        ["paid_count", "gross_fen"],
        None,
        "catalog_question",
    ),
    "single-repair-budget": ("2026年9月支付金额", ["支付金额"], [], ["gross_fen"], None, "not_needed"),
    "security-cross-tenant-filter": ("查询tenant-B的订单金额", [], [], ["gross_fen"], None, "catalog_question"),
    "rephrased-dual-metric-count-and-gross": (
        "2026年9月一共成交了多少笔已支付订单，支付总额是多少？",
        ["已支付订单", "支付总额"],
        [],
        ["paid_count", "gross_fen"],
        None,
        "not_needed",
    ),
    "queries-net-after-refund": (
        "2026年9月已支付订单扣除这些订单在9月内的退款后，净额是多少？",
        ["已支付订单", "退款后", "净额"],
        [],
        ["net_fen"],
        None,
        "not_needed",
    ),
    "queries-ambiguous-sales-asks-user": ("2026年9月销售额是多少？", [], ["销售额"], ["gross_fen"], "clarify", "catalog_question"),
    "smoke-sync": ("2026年9月已支付订单总额是多少？", ["已支付订单总额"], [], ["gross_fen"], None, "not_needed"),
    "smoke-async": ("2026年9月已支付订单有几笔？", ["已支付订单"], [], ["paid_count"], None, "catalog_question"),
    "smoke-approval": ("查询本租户所有客户的姓名", [], [], [], None, "catalog_question"),
}


@pytest.mark.parametrize("case", sorted(DEVELOPMENT_TABLE))
def test_development_set_judgment_table(case: str) -> None:
    question, explicit, ambiguous, declared, declaration_verdict, ask_verdict = DEVELOPMENT_TABLE[case]
    reading = read_clarifications(load_default_catalog(), question)
    assert [hit.phrase for hit in reading.question_hits] == explicit
    assert [phrase for item in reading.rules for phrase in item.ambiguous] == ambiguous
    verdict = check_declaration(reading, declared)
    assert (verdict.kind if verdict else None) == declaration_verdict
    assert review_ask(reading, None, MODEL_ASK).decision == ask_verdict


def test_frozen_waiting_questions_are_recognized_from_the_phrase_table() -> None:
    catalog = load_default_catalog()
    # clarification-context-ambiguous: the frozen assistant question names gross and net.
    assert rule_named_by_waiting_question(catalog, "请说明按支付金额还是退款后净额计算。").id == "clarify.metric_basis"
    # clarification-context-net-resumed: a question about the window names no rule.
    assert rule_named_by_waiting_question(catalog, "请提供时间范围。") is None
    # A stored rule id wins; an unknown id falls back to the text.
    assert _waiting_clarification_rule(catalog, {"waiting_clarification_id": "clarify.order_status_scope", "waiting_question": "x"}).id == (
        "clarify.order_status_scope"
    )
    assert _waiting_clarification_rule(catalog, {"waiting_clarification_id": "no.such.rule", "waiting_question": MODEL_ASK}).id == (
        "clarify.metric_basis"
    )


def test_an_answer_chooses_only_when_it_names_exactly_one_value() -> None:
    catalog = load_default_catalog()
    rule = catalog.clarification("clarify.metric_basis")
    assert select_value(catalog, rule, "按支付金额统计").value == "gross_fen"
    assert select_value(catalog, rule, "退款后净额").value == "net_fen"
    assert select_value(catalog, rule, "net_fen").value == "net_fen"
    assert select_value(catalog, rule, "2026年9月") is None
    assert select_value(catalog, rule, "支付金额和净额都要") is None


# ---------------------------------------------------------------------------
# Tool stage: the model declared a metric the wording leaves open
# ---------------------------------------------------------------------------


def test_declared_metric_for_ambiguous_wording_is_stopped_before_sql() -> None:
    tools, executor = _tools()
    reading = read_clarifications(tools.catalog, "2026年9月销售额是多少？")
    with pytest.raises(ClarificationRequiredError) as caught:
        call_tool(tools, "query_readonly", _query()["arguments"], context=_context(), clarifications=reading)
    assert caught.value.rule.id == "clarify.metric_basis"
    assert executor.executed == []


def test_unsupported_scope_is_stopped_with_the_catalog_note() -> None:
    tools, executor = _tools()
    reading = read_clarifications(tools.catalog, "2026年9月已取消订单数")
    with pytest.raises(ClarificationValueUnsupportedError) as caught:
        call_tool(tools, "query_readonly", _query(COUNT_SQL, ("paid_count",))["arguments"], context=_context(), clarifications=reading)
    assert caught.value.code == "clarification_value_unsupported"
    assert caught.value.note == tools.catalog.clarification("clarify.order_status_scope").value("cancelled").unsupported_note
    assert executor.executed == []
    assert http_status_for_run("FAILED", "clarification_value_unsupported") == 422


def test_non_metric_example_in_the_context_parks_for_approval() -> None:
    tools, executor = _tools()
    action = json.loads(NON_METRIC_QUERY_EXAMPLE)
    with pytest.raises(ApprovalRequiredError):
        call_tool(tools, action["name"], action["arguments"], context=_context(), clarifications=read_clarifications(tools.catalog, "查询客户姓名"))
    assert executor.executed == []


# ---------------------------------------------------------------------------
# B1: both directions
# ---------------------------------------------------------------------------


def test_b1_asks_on_the_catalog_question_when_the_model_picks_for_ambiguous_wording() -> None:
    tools, executor = _tools()
    model = _Scripted([_query()])
    result = _agent(model, tools).run(_context("run-b3b-sales"), "2026年9月销售额是多少？", request_time_window=SEPTEMBER)
    assert result.status == "waiting_user"
    assert result.action == {"type": "ask_user", "question": METRIC_BASIS_QUESTION, "clarification_id": "clarify.metric_basis"}
    assert result.model_call_count == 1
    assert executor.executed == []
    assert result.facts is None


def test_b1_bounces_an_unneeded_ask_once_then_queries_without_spending_the_query_repair() -> None:
    tools, executor = _tools()
    model = _Scripted([_ask(), _query(), _cite])
    context = _context("run-b3b-bounce")
    result = _agent(model, tools).run(context, "2026年9月支付金额", request_time_window=SEPTEMBER)
    assert result.status == "succeeded"
    assert [fact["metric_id"] for fact in result.facts["facts"]] == ["gross_fen"]
    assert result.repair_count == 0
    kinds = [event["kind"] for event in result.events]
    assert kinds.count("clarification_bounce") == MAX_CLARIFICATION_BOUNCES == 1
    review = next(event for event in result.events if event["kind"] == "clarification_review")
    assert review["decision"] == "not_needed" and review["error_code"] == "clarification_not_needed"
    # The hint the model saw holds catalog strings, not the model's question.
    hint_message = next(m["content"] for m in model.messages[1] if "clarification_not_needed" in m["content"])
    assert "支付金额" in hint_message and MODEL_ASK not in hint_message
    assert len(executor.executed) == 1


def test_b1_second_unneeded_ask_fails() -> None:
    tools, executor = _tools()
    model = _Scripted([_ask(), _ask("还是想确认：支付金额还是净额？")])
    result = _agent(model, tools).run(_context("run-b3b-bounce-twice"), "2026年9月支付金额", request_time_window=SEPTEMBER)
    assert result.status == "failed"
    assert result.error_code == "clarification_not_needed"
    assert result.model_call_count == 2
    assert executor.executed == []
    assert http_status_for_run("FAILED", "clarification_not_needed") == 502


def test_bounced_ask_still_leaves_the_query_repair_for_an_injected_sql_fault() -> None:
    """single-repair-budget: ask (bounced), a broken query (repaired once), then the right query."""

    tools, executor = _tools()
    model = _Scripted([_ask(), _query(BAD_SQL), _query(), _cite])
    result = _agent(model, tools).run(_context("run-b3b-bounce-repair"), "2026年9月支付金额", request_time_window=SEPTEMBER)
    assert result.status == "succeeded"
    assert result.repair_count == 1
    assert sum(1 for event in result.events if event["kind"] == "clarification_bounce") == 1


@pytest.mark.parametrize(
    "ask",
    [
        _ask(),  # no clarification_id
        _ask("请问您要查询哪一种口径？"),  # no id, only a marker word
        _ask(MODEL_ASK, "clarify.order_status_scope"),  # another rule's id
        _ask(MODEL_ASK, "made.up.rule"),  # unknown id
    ],
)
def test_leaving_out_or_misnaming_the_rule_id_does_not_skip_the_check(ask) -> None:
    tools, _ = _tools()
    model = _Scripted([ask, _ask()])
    result = _agent(model, tools).run(_context("run-b3b-id"), "2026年9月支付金额", request_time_window=SEPTEMBER)
    assert result.status == "failed" and result.error_code == "clarification_not_needed"


def test_allowed_ask_is_stored_with_the_catalog_question_not_model_text() -> None:
    tools, _ = _tools()
    context = _context("run-b3b-normalized")
    runtime = _agent(_Scripted([_ask("你指的销售额是哪一种？", "clarify.metric_basis")]), tools)
    result = runtime.run(context, "2026年9月销售额是多少？", request_time_window=SEPTEMBER)
    assert result.action == {"type": "ask_user", "question": METRIC_BASIS_QUESTION, "clarification_id": "clarify.metric_basis"}
    checkpoint = runtime.export_waiting_checkpoint(context.run_id)
    assert checkpoint["waiting_question"] == METRIC_BASIS_QUESTION
    assert checkpoint["waiting_clarification_id"] == "clarify.metric_basis"
    assert "你指的销售额是哪一种" not in json.dumps(checkpoint, ensure_ascii=False)


def test_ask_about_something_the_catalog_does_not_cover_is_left_to_the_model() -> None:
    tools, _ = _tools()
    result = _agent(_Scripted([_ask("请问是哪一个月？")]), tools).run(_context("run-b3b-month"), "支付金额是多少")
    assert result.status == "waiting_user"
    assert result.action == {"type": "ask_user", "question": "请问是哪一个月？"}


def test_explicit_phrases_cover_the_generic_total_in_the_empty_window_question() -> None:
    tools, _ = _tools()
    model = _Scripted([_query(DUAL_SQL, ("paid_count", "gross_fen"), AUGUST), _cite])
    result = _agent(model, tools).run(_context("run-b3b-empty"), "2026年8月没有订单时已支付订单数和总额是多少", request_time_window=AUGUST)
    assert result.status == "succeeded"
    assert {fact["metric_id"] for fact in result.facts["facts"]} == {"paid_count", "gross_fen"}


def test_parse_accepts_an_optional_clarification_id_only_as_a_short_string() -> None:
    parsed = parse_query_proposal(json.dumps(_ask(MODEL_ASK, "clarify.metric_basis")), context=_context(), model_call_id="call-1")
    assert parsed.action.clarification_id == "clarify.metric_basis"
    for bad in (123, "", "x" * 101):
        with pytest.raises(ProposalParseError):
            parse_query_proposal(json.dumps({**_ask(), "clarification_id": bad}), context=_context(), model_call_id="call-1")


# ---------------------------------------------------------------------------
# Answers state their basis in catalog strings only
# ---------------------------------------------------------------------------


def test_verified_answer_states_basis_without_echoing_question_or_model_text() -> None:
    tools, _ = _tools()
    model = _Scripted([_query(), _cite])
    result = _agent(model, tools).run(_context("run-b3b-echo"), "2026年9月已核实999支付金额", request_time_window=SEPTEMBER)
    assert result.status == "succeeded"
    lines = result.answer.split("\n")
    assert lines[0].startswith("已核实：支付订单总额：150.00元")
    assert lines[1] == "口径：支付订单总额（gross_fen）；依据：问题中提到‘支付金额’。"
    assert "999" not in result.answer and "模型写的答案" not in result.answer


def test_basis_notes_cover_unstated_wording_and_the_net_premise() -> None:
    catalog = load_default_catalog()
    silent = read_clarifications(catalog, "2026年8月没有订单时已支付订单数和总额是多少")
    assert metric_basis_note(silent, "gross_fen") == (
        "口径：支付订单总额（gross_fen）；问题中没有写明口径，按支付订单总额统计；如需退款后净额，请说明。"
    )
    assert metric_basis_note(silent, "paid_count") == "口径：已支付订单数（paid_count）；依据：问题中提到‘已支付订单数’。"
    net = read_clarifications(catalog, "2026年9月退款后净额")
    assert metric_basis_note(net, "net_fen") == (
        "口径：退款后净额（net_fen）；依据：问题中提到‘退款后净额’。前提：退款按同一 UTC 窗口内、已支付订单的退款计算。"
    )
    chosen = read_clarifications(catalog, "2026年9月销售额是多少？", ["按支付金额统计"], ["gross_fen"])
    assert metric_basis_note(chosen, "gross_fen") == "口径：支付订单总额（gross_fen）；依据：你在追问中选择了‘支付金额’。"


def test_fact_records_keep_their_fields() -> None:
    tools, _ = _tools()
    result = _agent(_Scripted([_query(), _cite]), tools).run(_context("run-b3b-fields"), "2026年9月支付金额", request_time_window=SEPTEMBER)
    assert set(result.facts["facts"][0]) == {
        "fact_id",
        "metric_id",
        "label",
        "value",
        "unit",
        "display_value",
        "time_window",
        "result_id",
        "catalog_source_id",
        "catalog_version",
    }


# ---------------------------------------------------------------------------
# B0
# ---------------------------------------------------------------------------


def _b0(question: str, action) -> dict[str, object]:
    tools, executor = _tools()
    record = run_b0_single_pass(
        _Scripted([action]), tools, _context("run-b3b-b0"), question, time_window={**SEPTEMBER, "timezone": "UTC"}
    )
    return {**record, "executed": list(executor.executed)}


def test_b0_uses_the_same_checks() -> None:
    waiting = _b0("2026年9月销售额是多少？", _query())
    assert waiting["status"] == "waiting_user" and waiting["executed"] == [] and waiting["facts"] == []
    bounced = _b0("2026年9月支付金额", _ask())
    assert bounced["status"] == "failed" and bounced["error_code"] == "clarification_not_needed"
    unsupported = _b0("2026年9月已取消订单数", _query(COUNT_SQL, ("paid_count",)))
    assert unsupported["error_code"] == "clarification_value_unsupported" and unsupported["executed"] == []
    assert unsupported["http_status"] == 422
    answered = _b0("2026年9月支付金额", _query())
    assert answered["status"] == "succeeded"
    assert answered["answer"].split("\n")[1] == "口径：支付订单总额（gross_fen）；依据：问题中提到‘支付金额’。"


# ---------------------------------------------------------------------------
# Resume through the product service
# ---------------------------------------------------------------------------


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.delenv("QUERYSHIELD_RETRIEVAL", raising=False)
    now = datetime(2026, 9, 23, tzinfo=timezone.utc)
    store = StateStore(tmp_path / "state.sqlite3", clock=lambda: now)
    executor = _Recording(clock=lambda: now)
    service = RunService(store=store, executor_factory=lambda: executor, clock=lambda: now, mode="fake")
    yield service, executor
    store.close()


def _start(service_and_executor, question: str, steps, *, window=SEPTEMBER):
    service, _ = service_and_executor
    deps = service.default_dependencies()
    deps.model = _Scripted(steps)
    deps.retriever = None
    return service.run_sync(identity=REQUESTER, question=question, time_window=window, deps=deps)


def _resume(service_and_executor, run, answer: str, steps):
    service, executor = service_and_executor
    return service.resume_waiting_user(
        run_id=str(run["run_id"]),
        answer=answer,
        identity=REQUESTER,
        model=_Scripted(steps),
        call_store=ModelCallStore(),
        executor=executor,
    )


def test_resume_chooses_by_catalog_phrase_and_states_the_choice(service) -> None:
    run = _start(service, "2026年9月销售额是多少？", [_query()])
    assert run["status"] == "WAITING_USER"
    assert run["checkpoint"]["agent_checkpoint"]["waiting_clarification_id"] == "clarify.metric_basis"

    still = _resume(service, run, "2026年9月", [])
    assert still["status"] == "WAITING_USER" and still["model_call_count"] == 1
    both = _resume(service, run, "支付金额和净额都要", [])
    assert both["status"] == "WAITING_USER" and both["model_call_count"] == 1

    done = _resume(service, run, "按支付金额", [_query(), _cite])
    assert done["status"] == "SUCCEEDED"
    assert done["answer"].split("\n")[1] == "口径：支付订单总额（gross_fen）；依据：你在追问中选择了‘支付金额’。"


def test_two_rules_are_asked_one_at_a_time_and_confirmed_metrics_are_kept(service) -> None:
    dual = _query(DUAL_SQL, ("paid_count", "gross_fen"))
    run = _start(service, "2026年9月订单数和销售额", [dual])
    assert run["status"] == "WAITING_USER"
    assert run["checkpoint"]["agent_checkpoint"]["waiting_question"] == METRIC_BASIS_QUESTION

    # After gross is confirmed, dropping or replacing it is refused (one repair, then fail).
    dropped = _resume(service, run, "按支付金额", [_query(COUNT_SQL, ("paid_count",)), _query(COUNT_SQL, ("paid_count",))])
    assert dropped["status"] == "FAILED" and dropped["error_code"] == "query_repair_limit"

    run = _start(service, "2026年9月订单数和销售额", [dual])
    second = _resume(service, run, "按支付金额", [dual])
    assert second["status"] == "WAITING_USER"
    assert second["checkpoint"]["agent_checkpoint"]["waiting_question"] == ORDER_SCOPE_QUESTION
    assert second["checkpoint"]["clarified_metric"] == "gross_fen"

    done = _resume(service, second, "只统计已支付", [dual, _cite])
    assert done["status"] == "SUCCEEDED"
    assert {fact["metric_id"] for fact in done["facts"]["facts"]} == {"paid_count", "gross_fen"}


def test_choosing_an_unsupported_scope_ends_without_model_or_sql(service) -> None:
    _, executor = service
    run = _start(service, "2026年9月订单数是多少", [_query(COUNT_SQL, ("paid_count",))])
    assert run["status"] == "WAITING_USER"
    assert run["checkpoint"]["agent_checkpoint"]["waiting_question"] == ORDER_SCOPE_QUESTION
    ended = _resume(service, run, "包含已取消", [])
    assert ended["status"] == "FAILED"
    assert ended["error_code"] == "clarification_value_unsupported"
    assert ended["model_call_count"] == 1
    assert executor.executed == []
    note = load_default_catalog().clarification("clarify.order_status_scope").value("all").unsupported_note
    assert ended["answer"] == note


def test_v2_checkpoints_are_refused() -> None:
    tools, _ = _tools()
    context = _context("run-b3b-v2")
    runtime = _agent(_Scripted([_ask("请问是哪一个月？")]), tools)
    runtime.run(context, "支付金额是多少")
    checkpoint = runtime.export_waiting_checkpoint(context.run_id)
    assert checkpoint["checkpoint_version"] == "qs-bounded-agent-checkpoint-v4"
    legacy = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"waiting_clarification_id", "clarification_bounce_count", "answer_bounce_count"}
    }
    legacy["checkpoint_version"] = "qs-bounded-agent-checkpoint-v2"
    resumed = BoundedAgent(_Scripted([]), tools=tools, call_store=ModelCallStore(), run_config=runtime.run_config)
    with pytest.raises(RunResumeError) as caught:
        resumed.resume_from_checkpoint(context, "2026年9月", legacy)
    assert caught.value.code == "invalid_checkpoint"


def test_bounce_count_survives_resume() -> None:
    tools, _ = _tools()
    context = _context("run-b3b-bounce-resume")
    runtime = _agent(_Scripted([_ask(), _ask("请问是哪一个月？")]), tools)
    waiting = runtime.run(context, "支付金额是多少")
    assert waiting.status == "waiting_user"
    checkpoint = runtime.export_waiting_checkpoint(context.run_id)
    assert checkpoint["clarification_bounce_count"] == 1
    resumed = BoundedAgent(_Scripted([_ask()]), tools=tools, call_store=ModelCallStore(), run_config=runtime.run_config)
    result = resumed.resume_from_checkpoint(context, "2026年9月", checkpoint)
    assert result.status == "failed" and result.error_code == "clarification_not_needed"


def test_approval_service_holds_no_metric_word_list() -> None:
    source = (ROOT / "src" / "queryshield" / "approval" / "service.py").read_text(encoding="utf-8")
    for word in ("支付金额", "支付订单总额", "退款后净额", "净额", '"gross"', '"net"'):
        assert word not in source
    assert not re.search(r"_metric_clarification_options|_resolve_metric_clarification", source)
