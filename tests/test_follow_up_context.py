"""A follow-up's record of the previous run, and the data message built from it (synthetic runs)."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from queryshield.agent.context import MAX_CONTEXT_BYTES, build_context, previous_turn_content
from queryshield.agent.follow_up import (
    FOLLOW_UP_QUESTIONS,
    MAX_CARRIED_CUSTOMER_IDS,
    FollowUp,
    follow_up_from_run,
    same_as_previous_line,
)
from queryshield.agent.proposals import ExecutionContext, MetricBinding, ResultEvidence
from queryshield.catalog import load_default_catalog


CATALOG = load_default_catalog()
SEPTEMBER = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z", "timezone": "UTC"}
AUGUST = {"start": "2026-08-01T00:00:00Z", "end": "2026-09-01T00:00:00Z", "timezone": "UTC"}


def _binding(metric_id: str, window: dict[str, str] = SEPTEMBER, *, grouped: bool = False) -> MetricBinding:
    entry = CATALOG.metric(metric_id)
    return MetricBinding(
        metric_id=metric_id,
        result_position=metric_id,
        unit=str(entry.payload["unit"]),
        time_window=window,
        catalog_source_id=entry.source_id,
        catalog_version=CATALOG.catalog_version,
        grouped=grouped,
    )


def _evidence(run_id: str, rows, bindings=(), *, principal: str = "a-requester", result_id: str | None = None) -> ResultEvidence:
    context = ExecutionContext(run_id=run_id, tenant_id="A", principal_id=principal, role="requester")
    return ResultEvidence.from_server_execution(
        context,
        result_id=result_id or f"result-{run_id}-{len(rows)}",
        rows=rows,
        normalized_query="SELECT 1",
        params=(),
        observed_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        policy_version="policy",
        catalog_version=CATALOG.catalog_version,
        metric_bindings=bindings,
    )


def _run(run_id: str, question: str, evidences=(), *, role: str = "requester", approved_sql: str | None = None, earlier: FollowUp | None = None):
    principal = "a-approver" if role == "approver" else "a-requester"
    result = None
    if evidences:
        result = evidences[0].as_dict()
        if evidences[1:]:
            result["supporting_results"] = [item.as_dict() for item in evidences[1:]]
    return {
        "run_id": run_id,
        "tenant_id": "A",
        "principal_id": principal,
        "role": role,
        "question": question,
        "status": "SUCCEEDED",
        "result": result,
        "approval_id": "approval-1" if approved_sql is not None else None,
        "action": {"sql": approved_sql} if approved_sql is not None else None,
        "checkpoint": {"follow_up": earlier.as_dict()} if earlier is not None else {},
    }


def _grouped_rows(count: int) -> list[dict[str, object]]:
    return [{"customer_id": f"c{index:02d}", "gross_fen": 100 * index} for index in range(1, count + 1)]


def test_a_metric_run_carries_its_metric_window_and_question_but_no_value() -> None:
    evidence = _evidence("run-1", [{"gross_fen": 15000}], (_binding("gross_fen"),))
    follow_up = follow_up_from_run(_run("run-1", "2026年9月已支付订单总额", [evidence]))

    assert follow_up.questions == ("2026年9月已支付订单总额",)
    assert follow_up.executed == ({"metric_id": "gross_fen", "time_window": SEPTEMBER, "grouped_by_customer": False},)
    assert follow_up.customer_ids == () and follow_up.customer_count == 0
    assert "15000" not in json.dumps(follow_up.message_payload())


def test_a_grouped_run_carries_its_customer_ids() -> None:
    evidence = _evidence("run-1", _grouped_rows(3), (_binding("gross_fen", grouped=True),))
    follow_up = follow_up_from_run(_run("run-1", "按客户汇总2026年9月的已支付金额", [evidence]))

    assert follow_up.customer_ids == ("c01", "c02", "c03")
    assert follow_up.executed[0]["grouped_by_customer"] is True
    assert "300" not in json.dumps(follow_up.message_payload())


def test_more_customers_than_the_cap_carry_only_the_count() -> None:
    evidence = _evidence("run-1", _grouped_rows(MAX_CARRIED_CUSTOMER_IDS + 1), (_binding("gross_fen", grouped=True),))
    follow_up = follow_up_from_run(_run("run-1", "按客户汇总", [evidence]))

    assert follow_up.customer_ids is None
    assert follow_up.customer_count == MAX_CARRIED_CUSTOMER_IDS + 1


@pytest.mark.parametrize(
    ("sql", "rows"),
    [
        # Two output columns named customer_id: the executor's dict row keeps the later one, the name.
        ("SELECT c.customer_id, c.name AS customer_id FROM customers AS c ORDER BY c.customer_id", [{"customer_id": "甲"}, {"customer_id": "乙"}]),
        # The name under an output name that differs only in case.
        ("SELECT c.name AS Customer_Id, c.customer_id FROM customers AS c ORDER BY c.customer_id", [{"Customer_Id": "甲", "customer_id": "c1"}, {"Customer_Id": "乙", "customer_id": "c2"}]),
        ("SELECT c.name AS id, c.customer_id FROM customers AS c", [{"id": "甲", "customer_id": "c1"}, {"id": "乙", "customer_id": "c2"}]),
    ],
    ids=["same_name", "differs_in_case", "plain"],
)
def test_an_approved_run_carries_no_customer_ids_whatever_its_sql(sql, rows) -> None:
    run = _run("run-1", "查询客户姓名", [_evidence("run-1", rows)], approved_sql=sql)
    follow_up = follow_up_from_run(run)

    assert follow_up.customer_ids is None and follow_up.customer_count is None
    assert follow_up.listed_customer_names is True
    for shown in (follow_up.as_dict(), follow_up.message_payload(), follow_up.event_payload(message_bytes=0)):
        assert not {"甲", "乙"} & set(json.dumps(shown, ensure_ascii=False))
    assert "甲" not in previous_turn_content(follow_up.message_payload())


def test_an_approvers_run_carries_no_customer_ids() -> None:
    rows = [{"customer_id": "c1", "name": "甲"}]
    run = _run("run-1", "查询客户姓名", [_evidence("run-1", rows, principal="a-approver")], role="approver")
    follow_up = follow_up_from_run(run)

    # Its rows are not read: the count and whether it listed names are unknown, not zero and false.
    assert follow_up.customer_ids is None and follow_up.customer_count is None and follow_up.listed_customer_names is None
    assert "甲" not in json.dumps(follow_up.as_dict(), ensure_ascii=False)
    assert FollowUp.from_dict(json.loads(json.dumps(follow_up.as_dict()))) == follow_up


def test_every_result_of_the_previous_answer_is_carried() -> None:
    # A composite answer: gross_fen and net_fen are two results (primary and supporting).
    gross = _evidence("run-1", [{"gross_fen": 1}], (_binding("gross_fen", AUGUST),), result_id="result-gross")
    net = _evidence("run-1", [{"net_fen": 1}], (_binding("net_fen", AUGUST),), result_id="result-net")
    follow_up = follow_up_from_run(_run("run-1", "2026年8月的已支付订单总额和退款后净额分别是多少？", [gross, net]))

    assert [item["metric_id"] for item in follow_up.executed] == ["gross_fen", "net_fen"]
    assert all(item["time_window"] == AUGUST for item in follow_up.executed)


def test_a_run_without_a_result_carries_only_its_question() -> None:
    follow_up = follow_up_from_run(_run("run-1", "退款后净额是怎么算的？"))

    assert follow_up.questions == ("退款后净额是怎么算的？",) and follow_up.executed == ()


def test_a_stored_result_that_fails_its_checks_raises() -> None:
    evidence = _evidence("run-1", [{"gross_fen": 1}], (_binding("gross_fen"),))
    run = _run("run-1", "q", [evidence])
    run["result"]["row_count"] = 5

    with pytest.raises(ValueError):
        follow_up_from_run(run)


def test_the_questions_are_the_previous_runs_carried_ones_plus_its_own() -> None:
    earlier: FollowUp | None = None
    for index in range(6):
        earlier = follow_up_from_run(_run(f"run-{index}", f"问题{index}", earlier=earlier))
    assert earlier.questions == ("问题3", "问题4", "问题5")
    assert len(earlier.questions) == FOLLOW_UP_QUESTIONS


CONTEXT = ExecutionContext(run_id="run-next", tenant_id="A", principal_id="a-requester", role="requester")


def test_the_previous_turn_is_one_hard_data_message_just_before_the_question() -> None:
    follow_up = follow_up_from_run(_run("run-1", "2026年9月已支付订单总额", [_evidence("run-1", [{"gross_fen": 1}], (_binding("gross_fen"),))]))
    plain = build_context(CONTEXT, "那8月呢")
    result = build_context(CONTEXT, "那8月呢", previous_turn=follow_up.message_payload())

    messages = list(result.messages)
    index = next(i for i, message in enumerate(messages) if message["content"].startswith("QUERYSHIELD_DATA kind=previous_turn;"))
    assert messages[index + 1] == {"role": "user", "content": "那8月呢"}
    assert messages[index]["content"] == previous_turn_content(follow_up.message_payload())
    assert "previous-turn" in result.hard_message_ids
    # Everything else is the context the same question gets without a previous turn.
    assert messages[:index] + messages[index + 1:] == list(plain.messages)
    assert result.context_version == plain.context_version == "context-v16"


def test_the_context_does_not_grow_with_the_number_of_turns() -> None:
    sizes = []
    earlier: FollowUp | None = None
    for index in range(10):
        question = f"2026年9月已支付订单总额，第{index:02d}轮" + "很" * 400
        evidence = _evidence(f"run-{index}", _grouped_rows(5), (_binding("gross_fen", grouped=True),))
        earlier = follow_up_from_run(_run(f"run-{index}", question, [evidence], earlier=earlier))
        sizes.append(build_context(CONTEXT, "那8月呢", previous_turn=earlier.message_payload()).serialized_bytes)
    assert len(set(sizes[FOLLOW_UP_QUESTIONS - 1:])) == 1, sizes
    assert max(sizes) < MAX_CONTEXT_BYTES


@pytest.mark.parametrize(
    ("current", "line"),
    [
        ([_binding("gross_fen", AUGUST)], "与上一轮相同：指标 {gross}"),
        ([_binding("net_fen")], "与上一轮相同：时间 2026-09-01T00:00:00Z至2026-10-01T00:00:00Z，UTC"),
        ([_binding("gross_fen", grouped=True)], "与上一轮相同：指标 {gross}；时间 2026-09-01T00:00:00Z至2026-10-01T00:00:00Z，UTC；按客户"),
        ([_binding("paid_count", AUGUST)], None),
    ],
)
def test_the_answer_line_names_only_the_conditions_equal_to_the_previous_runs(current, line) -> None:
    previous = follow_up_from_run(_run("run-1", "q", [_evidence("run-1", _grouped_rows(2), (_binding("gross_fen", grouped=True),))]))
    rows = _grouped_rows(2) if any(binding.grouped for binding in current) else [{"gross_fen": 1}]
    evidences = [_evidence("run-2", rows, tuple(current))]
    expected = line.format(gross=CATALOG.metric_name("gross_fen")) if line else None
    assert same_as_previous_line(previous, evidences, CATALOG) == expected


def _by_order_rows() -> list[dict[str, object]]:
    return [{"order_id": "o1", "gross_fen": 100}, {"order_id": "o2", "gross_fen": 200}]


def test_a_query_grouped_by_something_else_is_not_grouped_by_customer() -> None:
    # The metric check accepts GROUP BY o.order_id: the binding is grouped, the rows have no customer_id.
    evidence = _evidence("run-1", _by_order_rows(), (_binding("gross_fen", grouped=True),))
    previous = follow_up_from_run(_run("run-1", "按订单汇总2026年9月的已支付金额", [evidence]))
    assert previous.executed[0]["grouped_by_customer"] is False

    current = [_evidence("run-2", _by_order_rows(), (_binding("gross_fen", AUGUST, grouped=True),))]
    assert same_as_previous_line(previous, current, CATALOG) == f"与上一轮相同：指标 {CATALOG.metric_name('gross_fen')}"
    by_customer = follow_up_from_run(_run("run-1", "q", [_evidence("run-1", _grouped_rows(2), (_binding("gross_fen", grouped=True),))]))
    assert "按客户" not in same_as_previous_line(by_customer, current, CATALOG)


def test_the_customer_id_column_is_found_whatever_its_case() -> None:
    rows = [{"Customer_ID": "c1", "gross_fen": 1}]
    evidence = _evidence("run-1", rows, (_binding("gross_fen", grouped=True),))
    assert follow_up_from_run(_run("run-1", "q", [evidence])).executed[0]["grouped_by_customer"] is True


def test_no_result_no_line() -> None:
    previous = follow_up_from_run(_run("run-1", "q", [_evidence("run-1", [{"gross_fen": 1}], (_binding("gross_fen"),))]))
    assert same_as_previous_line(previous, [], CATALOG) is None


def test_the_record_round_trips() -> None:
    previous = follow_up_from_run(_run("run-1", "q", [_evidence("run-1", _grouped_rows(2), (_binding("gross_fen", grouped=True),))]))
    assert FollowUp.from_dict(json.loads(json.dumps(previous.as_dict()))) == previous
