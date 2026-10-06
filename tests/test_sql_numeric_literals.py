"""Numeric literals in model-proposed SQL: only ASCII digits, and only finite decimals."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from queryshield.agent import BoundedAgent, ModelCallStore
from queryshield.agent.proposals import ExecutionContext
from queryshield.agent.tool_execution import call_tool
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.policy.sql import SQLPolicyError, parse_readonly_select
from queryshield.tools import ControlledTools, ToolError

from test_agent_graph import _ScriptedModel

HUGE_DECIMAL = "1" * 400 + ".5"
BASE = "SELECT name FROM customers WHERE tenant_id = %s"
BAD_LITERALS = {
    "superscript-limit": BASE + " LIMIT ²",
    "superscript-select": "SELECT ² FROM customers WHERE tenant_id = %s",
    "arabic-indic-limit": BASE + " LIMIT ١٠",
    "fullwidth-comparison": BASE + " AND customer_id > ３",
    "arabic-indic-fraction": "SELECT 1.٥ FROM customers WHERE tenant_id = %s",
    "overflowing-decimal": f"SELECT {HUGE_DECIMAL} FROM customers WHERE tenant_id = %s",
    "ascii-then-arabic-indic-limit": BASE + " LIMIT 1٠",
    "ascii-then-arabic-indic-select": "SELECT 1٢3 FROM customers WHERE tenant_id = %s",
}


def _code(sql: str) -> str:
    with pytest.raises(SQLPolicyError) as caught:
        parse_readonly_select(sql)
    return caught.value.code


@pytest.mark.parametrize("sql", BAD_LITERALS.values(), ids=BAD_LITERALS.keys())
def test_a_literal_with_a_non_ascii_digit_or_an_overflowing_decimal_is_invalid_sql(sql: str) -> None:
    assert _code(sql) == "invalid_sql"


def test_ordinary_numeric_literals_are_parsed_as_before() -> None:
    statement = parse_readonly_select("SELECT 12, 1.50, 0.25 FROM customers WHERE tenant_id = %s LIMIT 7")
    assert [item.expression.value for item in statement.projection] == [12, 1.5, 0.25]
    assert statement.limit == 7
    assert parse_readonly_select(BASE + " LIMIT 100").limit == 100
    assert _code(BASE + " LIMIT 101") == "limit_exceeded"
    assert _code(BASE + " LIMIT 1.5") == "limit_exceeded"


def test_every_ascii_digit_is_read_in_integers_decimals_and_limits() -> None:
    statement = parse_readonly_select("SELECT 1234567890, 0.9876543210 FROM customers WHERE tenant_id = %s LIMIT 9")
    assert [item.expression.value for item in statement.projection] == [1234567890, 0.987654321]
    assert statement.limit == 9


LARGEST_FINITE = "17976931348623157" + "0" * 292 + ".0"


def test_the_largest_finite_decimal_is_parsed() -> None:
    statement = parse_readonly_select(f"SELECT {LARGEST_FINITE} FROM customers WHERE tenant_id = %s")
    assert statement.projection[0].expression.value == 1.7976931348623157e308


def test_a_decimal_with_one_more_digit_than_the_largest_finite_one_is_refused() -> None:
    assert _code(f"SELECT {LARGEST_FINITE[:-2]}0.0 FROM customers WHERE tenant_id = %s") == "invalid_sql"


def test_an_overflowing_decimal_is_refused_with_the_existing_message() -> None:
    with pytest.raises(SQLPolicyError) as caught:
        parse_readonly_select(BAD_LITERALS["overflowing-decimal"])
    assert str(caught.value) == "invalid_sql: invalid numeric literal"


def test_an_integer_literal_past_two_to_the_fifty_third_keeps_every_digit() -> None:
    statement = parse_readonly_select("SELECT 12345678901234567891 FROM customers WHERE tenant_id = %s")
    assert statement.projection[0].expression.value == 12345678901234567891


def _tools() -> ControlledTools:
    def connect():
        raise AssertionError("a query that does not parse must not reach the database")

    executor = GuardedQueryExecutor(connect=connect, clock=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc))
    return ControlledTools(executor=executor)


def _context(run_id: str) -> ExecutionContext:
    return ExecutionContext(run_id=run_id, tenant_id="tenant-A", principal_id="principal-A", role="requester")


@pytest.mark.parametrize("sql", [BAD_LITERALS["superscript-limit"], BAD_LITERALS["overflowing-decimal"]])
def test_a_proposed_query_with_a_bad_literal_gets_a_controlled_tool_error(sql: str) -> None:
    arguments = {"sql": sql, "params": {"0": "tenant-A"}}
    with pytest.raises(ToolError) as caught:
        call_tool(_tools(), "query_readonly", arguments, context=_context("run-literal"))
    assert caught.value.code == "invalid_sql"


@pytest.mark.parametrize("sql", [BAD_LITERALS["superscript-select"], BAD_LITERALS["arabic-indic-limit"]])
def test_the_agent_run_treats_a_bad_literal_as_a_repairable_query_error(sql: str) -> None:
    proposal = json.dumps(
        {"type": "tool_call", "name": "query_readonly", "arguments": {"sql": sql, "params": {"0": "tenant-A"}}},
        ensure_ascii=False,
    )
    model = _ScriptedModel([proposal, proposal])
    agent = BoundedAgent(model, tools=_tools(), call_store=ModelCallStore())

    result = agent.run(_context("run-literal-agent"), "列出客户")

    assert (result.status, result.error_code) == ("failed", "query_repair_limit")
    assert (result.model_call_count, result.tool_call_count, result.repair_count) == (2, 2, 1)
