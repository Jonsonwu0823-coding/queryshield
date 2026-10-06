"""The offline FakeModel's exact output for each kind of server conversation.

Fake records, the demo and the HTTP smoke all depend on these bytes, so
every branch of the fake's decision is pinned here (json and native) before
its code is restructured.  Expected outputs are in ``fake_model_output_pins.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from queryshield.providers import fake_model as fm
from queryshield.providers.fake_model import FakeModel


PINS = json.loads(Path(__file__).with_name("fake_model_output_pins.json").read_text(encoding="utf-8"))
SEARCH = {"type": "function", "function": {"name": "search_catalog"}}
QUERY = {"type": "function", "function": {"name": "query_readonly"}}
WITH_SEARCH = [SEARCH, QUERY]
WITHOUT_SEARCH = [QUERY]


def _system(*, contract=("search_catalog", "query_readonly"), window=None, baseline=False, raw_context=None) -> dict:
    text = "You are the W05 single-pass baseline.\n" if baseline else "rules\n"
    context = {"action_contract": {"tool_names": list(contract)}}
    if window is not None:
        context["metric_declaration"] = {"request_time_window": window}
    body = raw_context if raw_context is not None else json.dumps(context, ensure_ascii=False)
    return {"role": "system", "content": text + "QUERYSHIELD_SERVER_CONTEXT\n" + body}


def _tool_result(name: str, status: str = "succeeded", output=None) -> dict:
    record = {"tool_name": name, "status": status, "output": output}
    return {"role": "user", "content": fm._TOOL_RESULT_PREFIX + json.dumps(record, ensure_ascii=False)}


def _retrieval(payload: dict) -> dict:
    return {"role": "user", "content": fm._RETRIEVAL_PREFIX + json.dumps(payload, ensure_ascii=False)}


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


VERIFIED = {"result_id": "result-1", "verified_metrics": [{"metric_id": "gross_fen"}, {"metric_id": 3}, "x"], "rows": [{"gross_fen": 1}]}
GROUPED = {"result_id": "result-2", "verified_metrics": [], "rows": [{"customer_id": "c1", "gross_fen": 5}]}

# (label, messages, native tools or None for the json protocol)
CASES = [
    ("health", [_user("provider health check")], None),
    ("query_result_verified", [_system(), _user("2026年9月支付金额"), _tool_result("query_readonly", output=VERIFIED)], None),
    ("query_result_grouped", [_system(), _user("按客户支付金额"), _tool_result("query_readonly", output=GROUPED)], WITH_SEARCH),
    ("query_result_last_one_wins", [_system(), _user("q"), _tool_result("query_readonly", output=VERIFIED), _tool_result("query_readonly", output=GROUPED)], None),
    ("query_result_failed_is_ignored", [_system(contract=()), _user("2026年9月订单数"), _tool_result("query_readonly", status="failed", output=VERIFIED)], None),
    ("query_result_without_result_id", [_system(contract=()), _user("2026年9月订单数"), _tool_result("query_readonly", output={"rows": []})], None),
    ("query_result_bad_lists", [_system(), _user("q"), _tool_result("query_readonly", output={"result_id": "r", "verified_metrics": "x", "rows": "y"})], None),
    ("undated_smoke", [_system(), _user(fm.UNDATED_SMOKE_QUESTION)], None),
    ("undated_smoke_native", [_system(), _user(fm.UNDATED_SMOKE_QUESTION)], WITH_SEARCH),
    ("undated_smoke_answered", [_system(contract=()), _user(fm.UNDATED_SMOKE_QUESTION), _user("2026年8月")], None),
    ("no_data", [_system(), _user(fm.NO_DATA_SMOKE_QUESTION)], None),
    ("no_data_native", [_system(), _user(fm.NO_DATA_SMOKE_QUESTION)], WITHOUT_SEARCH),
    ("ambiguous", [_system(), _user("2026年9月销售额是多少？")], None),
    ("ambiguous_operating", [_system(), _user("营业额")], WITH_SEARCH),
    ("ambiguous_answered_paid", [_system(contract=()), _user("2026年9月销售额是多少？"), _user("按支付金额")], None),
    ("ambiguous_answered_paid_word", [_system(contract=()), _user("2026年9月销售额"), _user("支付")], None),
    ("ambiguous_answered_net", [_system(contract=()), _user("2026年9月销售额"), _user("退款后")], None),
    ("knowledge_search_first", [_system(), _user(fm.KNOWLEDGE_SMOKE_QUESTION)], None),
    ("knowledge_search_first_native", [_system(), _user(fm.KNOWLEDGE_SMOKE_QUESTION)], WITH_SEARCH),
    ("knowledge_after_search", [_system(), _user(fm.KNOWLEDGE_SMOKE_QUESTION), _tool_result("search_catalog", output={})], None),
    ("knowledge_after_retrieval_items", [_system(), _user(fm.KNOWLEDGE_SMOKE_QUESTION), _retrieval({"items": [1]})], None),
    ("knowledge_retrieval_empty_items", [_system(), _user(fm.KNOWLEDGE_SMOKE_QUESTION), _retrieval({"items": []})], None),
    ("knowledge_after_retrieval_item", [_system(), _user(fm.KNOWLEDGE_SMOKE_QUESTION), _retrieval({"item": {"a": 1}})], None),
    ("knowledge_not_offered", [_system(contract=("query_readonly",)), _user(fm.KNOWLEDGE_SMOKE_QUESTION)], None),
    ("knowledge_not_offered_native", [_system(), _user(fm.KNOWLEDGE_SMOKE_QUESTION)], WITHOUT_SEARCH),
    ("knowledge_baseline", [_system(baseline=True), _user(fm.KNOWLEDGE_SMOKE_QUESTION)], None),
    ("search_first", [_system(), _user("2026年9月已支付订单数")], None),
    ("search_first_long_question", [_system(), _user("2026年9月已支付订单数" + "。" * 250)], None),
    ("search_first_native", [_system(), _user("2026年9月已支付订单数")], WITH_SEARCH),
    ("search_offered_but_baseline", [_system(baseline=True), _user("QUESTION\n2026年9月已支付订单总额\n\nAPPROVED_TABLE_SCHEMA\n{}")], None),
    ("names", [_system(), _user("查询本租户所有客户的姓名"), _tool_result("search_catalog", output={})], None),
    ("names_native", [_system(), _user("客户名"), _tool_result("search_catalog", output={})], WITH_SEARCH),
    ("unknown_metric", [_system(contract=()), _user("你们公司怎么样")], None),
    ("unknown_metric_native", [_system(), _user("怎么样")], WITHOUT_SEARCH),
    ("count", [_system(contract=()), _user("2026年9月有几笔订单")], None),
    ("count_and_gross", [_system(contract=()), _user("2026年8月订单数和支付金额")], None),
    ("gross_by_customer", [_system(contract=()), _user("2026年9月按客户的支付金额")], None),
    ("count_by_customer", [_system(contract=()), _user("2026年9月按客户的订单数")], None),
    ("net", [_system(contract=()), _user("2026年12月退款后净额")], None),
    ("net_native", [_system(), _user("2026年9月净额"), _tool_result("search_catalog", output={})], WITH_SEARCH),
    ("range_window", [_system(contract=()), _user("2026年7月至9月的总额")], None),
    ("range_window_reversed", [_system(contract=()), _user("2026年9月到7月的总额")], None),
    ("request_window", [_system(contract=(), window={"start": "2026-06-01T00:00:00Z", "end": "2026-07-01T00:00:00Z"}), _user("订单数")], None),
    ("request_window_incomplete", [_system(contract=(), window={"start": "2026-06-01T00:00:00Z"}), _user("订单数")], None),
    ("default_window", [_system(contract=()), _user("数量")], None),
    ("bad_server_context", [_system(raw_context="{not json"), _user("2026年9月订单数")], None),
    ("server_context_not_object", [_system(raw_context="[1]"), _user("2026年9月订单数")], None),
    ("no_user_message", [_system(contract=())], None),
    ("clarification_skips_data_messages", [_system(contract=()), _user("2026年9月销售额"), _retrieval({"items": []}), _user("净额")], None),
]


def _output(messages, tools) -> list[object]:
    result = FakeModel().complete(messages, request_id="r", model_call_id="m", tools=tools)
    calls = None if result.tool_calls is None else [[call.id, call.name, call.arguments] for call in result.tool_calls]
    return [result.content, calls, result.finish_reason]


def test_every_case_has_a_pin() -> None:
    labels = [label for label, _, _ in CASES]
    assert len(labels) == len(set(labels)) and set(labels) == set(PINS)


@pytest.mark.parametrize("label, messages, tools", CASES, ids=[label for label, _, _ in CASES])
def test_fake_output_is_pinned(label, messages, tools) -> None:
    assert _output(messages, tools) == PINS[label]


def test_fake_record_shape_and_empty_messages() -> None:
    result = FakeModel().complete([_user("provider health check")])
    assert (result.mode, result.provider, result.model, result.usage, result.usage_status) == ("fake", "fake", "fake-model", None, "unknown")
    assert result.content == "FAKE_OK" and result.tool_calls is None and result.finish_reason is None
    with pytest.raises(ValueError, match="messages must not be empty"):
        FakeModel().complete([])
