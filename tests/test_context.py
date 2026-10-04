from __future__ import annotations

import json
import pytest

from queryshield.agent import ContextBudgetError, ExecutionContext, ProposalParseError, build_context, parse_query_proposal
from queryshield.agent.context import CONTEXT_VERSION, MAX_SERVER_CONTEXT_CHARS
from queryshield.evaluation import load_development_cases


def _context() -> ExecutionContext:
    return ExecutionContext(
        run_id="run-context-1",
        tenant_id="tenant-A",
        principal_id="principal-A",
        role="requester",
    )


def test_context_keeps_identity_slots_and_data_outside_system() -> None:
    result = build_context(
        _context(),
        "请查询已支付订单数",
        confirmed_metric="paid_count",
        time_window={
            "start": "2026-09-01T00:00:00Z",
            "end": "2026-10-01T00:00:00Z",
            "timezone": "UTC",
        },
        retrieval_items=[
            {
                "id": "metric.paid_count",
                "text": "固定窗口内已支付订单数",
                "source_id": "commerce-v1",
                "version": "commerce-v1",
            }
        ],
        tool_results=[{"result_id": "result-1", "row_count": 1}],
    )

    assert result.serialized_bytes <= 24_000
    assert len(result.messages) <= 32
    assert "tenant-A" in result.messages[0]["content"]
    assert "paid_count" in result.messages[0]["content"]
    assert all(message["role"] != "system" for message in result.messages[1:])
    assert any("result-1" in message["content"] for message in result.messages)


def test_context_carries_multiple_server_metric_bindings_and_exact_ask_user_contract() -> None:
    window = {
        "start": "2026-08-01T00:00:00Z",
        "end": "2026-09-01T00:00:00Z",
        "timezone": "UTC",
    }
    result = build_context(
        _context(),
        "2026年8月已支付订单数和总额是多少",
        metric_bindings=(
            {"metric_id": "paid_count", "result_position": "paid_count", "unit": "count", "time_window": window},
            {"metric_id": "gross_fen", "result_position": "gross_fen", "unit": "fen", "time_window": window},
        ),
    )
    server = json.loads(result.messages[0]["content"].split("\n", 1)[1])
    slots = server["confirmed_slots"]["metrics"]

    assert CONTEXT_VERSION == "context-v16"
    assert [(item["metric_id"], item["result_position"], item["unit"], item["time_window"]) for item in slots] == [
        ("paid_count", "paid_count", "count", window),
        ("gross_fen", "gross_fen", "fen", window),
    ]
    ask_user = server["action_contract"]["actions"]["ask_user"]
    # Built from the catalog's first multi-option rule (catalog-v3).
    assert ask_user["valid_shape_examples"] == [
        '{"type":"ask_user","clarification_id":"clarify.metric_basis","question":"你要看支付订单总额（gross_fen），还是退款后净额（net_fen）？"}',
    ]
    assert ask_user["optional_fields"] == ["clarification_id"]
    assert "duplicate member names" in ask_user["wire_format"]
    # B3c-1 removed the instructions line repeating this (and the tool-fields
    # rule); both stay stated once in the action contract.
    assert "never repeat JSON member names" not in " ".join(server["instructions"])
    assert "Tool fields go only inside arguments" in server["action_contract"]["actions"]["tool_call"]["wire_format"]
    final_answer = server["action_contract"]["actions"]["final_answer"]
    assert final_answer["valid_shape_examples"] == [
        '{"type":"final_answer","answer":"...","source_ids":["<source_id from this run\'s results>"],"fact_refs":[{"result_id":"<result_id returned by query_readonly in this run>","metric_id":"<verified metric_id>"}]}',
        # B3c-2 R1: the no_data action, copyable as is; basis stays optional.
        '{"type":"final_answer","answer":"","source_ids":[],"fact_refs":[],"basis":"no_data"}',
    ]
    assert "basis" not in final_answer["required_fields"]
    assert "arrays, even for one item" in final_answer["wire_format"]
    assert "not bare objects" in final_answer["wire_format"]
    assert "replace example placeholders" in final_answer["fact_refs_rule"]
    # B3c-2: values still need this run's query, even when no data is expected.
    assert "Business values need this run's query first, even when no data is expected" in final_answer["fact_refs_rule"]
    assert "never state one without it" in final_answer["fact_refs_rule"]
    with pytest.raises(ProposalParseError, match="fact_refs must be a list"):
        parse_query_proposal(
            '{"type":"final_answer","answer":"150.00元","source_ids":["semantic-metric-gross"],"fact_refs":{"result_id":"result-1","metric_id":"gross_fen"}}',
            context=_context(),
            model_call_id="call-context-single-fact-object",
        )
    group_guidance = server["action_contract"]["actions"]["tool_call"]["tools"]["query_readonly"]["metric_query_guidance"]["group_by_customer"]
    assert "joining customers is REQUIRED" in group_guidance
    assert "never group orders alone" in group_guidance
    assert "o.tenant_id = c.tenant_id AND o.customer_id = c.customer_id" in group_guidance
    assert "GROUP BY o.customer_id" in group_guidance
    assert _within_message_limits(result)

    net_result = build_context(
        _context(),
        "查询2026年8月退款后净额",
        metric_bindings=(
            {"metric_id": "net_fen", "result_position": "net_fen", "unit": "fen", "time_window": window},
        ),
    )
    net_server = json.loads(net_result.messages[0]["content"].split("\n", 1)[1])
    assert "net_fen" not in net_server["action_contract"]["actions"]["tool_call"]["tools"]["query_readonly"]["metric_query_guidance"]
    assert "valid_net_fen_template" not in net_server["action_contract"]["actions"]["tool_call"]["tools"]["query_readonly"]
    assert "net_fen is declared alone" in net_server["metric_declaration"]["rule"]
    # B3c-1 removed the instructions line that repeated metric_declaration.rule.
    assert "MUST declare it in arguments.metrics" in net_server["metric_declaration"]["rule"]
    assert "metric_declaration" in " ".join(net_server["action_contract"]["workflow"])
    assert net_server["action_contract"]["actions"]["parallel_readonly"]["metric_count_range"] == [2, 3]

    with pytest.raises(ProposalParseError, match="duplicate JSON field"):
        parse_query_proposal(
            '{"type":"ask_user","question":"先问这个","question":"再问这个"}',
            context=_context(),
            model_call_id="call-context-duplicate",
        )


def _within_message_limits(result) -> bool:
    server, *others = result.messages
    return len(server["content"]) <= MAX_SERVER_CONTEXT_CHARS and all(len(message["content"]) <= 8_000 for message in others)


def test_context_drops_old_optional_messages_as_whole_units() -> None:
    old = "old-summary-" + ("旧摘要内容。" * 700)
    newer = "new-summary-" + ("新摘要内容。" * 700)
    result = build_context(
        _context(),
        "查询营业额",
        optional_summaries=[old, newer],
        tool_results=[{"result_id": "result-new", "rows": [{"gross_fen": 15000}]}],
    )

    assert result.serialized_bytes <= 24_000
    assert result.dropped_optional_ids
    assert "summary-0" in result.dropped_optional_ids
    assert "summary-0" not in result.included_optional_ids
    assert _within_message_limits(result)


def test_hard_context_overflow_stops_before_model_send() -> None:
    large_items = [
        {
            "id": f"field.orders.large-{index}",
            "text": "x" * 7_500,
            "source_id": "commerce-v1",
            "version": "commerce-v1",
        }
        for index in range(3)
    ]

    with pytest.raises(ContextBudgetError, match="context_budget_exceeded"):
        build_context(_context(), "硬约束超预算" + ("q" * 7_990), retrieval_items=large_items)


def test_development_fixture_contains_twelve_queries_and_empty_case() -> None:
    cases = load_development_cases()

    assert len(cases) == 12
    assert len({case.family_id for case in cases}) == 12
    assert any(case.relevant_source_ids == () for case in cases)
