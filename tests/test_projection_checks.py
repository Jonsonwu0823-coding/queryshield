"""The tool-stage projection checks each have a test.

Before they were tested, replacing ``_metric_scope_filters_match`` or ``_has_bound_window``
(agent/tool_execution.py) with an always-true stub left the whole suite green.
Every case below is refused in call_tool before any SQL reaches the database.

- (a) missing ``status = 'paid'`` and (b) an unrelated extra filter are
  refused by ``_metric_scope_filters_match`` only;
- (c) a gross query without the bound window is refused by both checks (they
  overlap on the gross/paid path), so it fails only when both are stubbed;
- (d) a declared net_fen query without the order window is refused by
  ``_has_bound_window`` only (via ``_net_fen_request_matches``).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from queryshield.agent.metric_intent import build_metric_binding
from queryshield.agent.proposals import ExecutionContext
from queryshield.agent.tool_execution import call_tool
from queryshield.catalog import load_default_catalog
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.tools import ControlledTools
from queryshield.tools.semantic import ToolError


SEPTEMBER = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}


class _Cursor:
    def __init__(self, connection: "_Connection") -> None:
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def execute(self, sql, params) -> None:
        self.connection.executed.append((sql, tuple(params)))

    def fetchmany(self, size):
        return [{"gross_fen": 15000, "refund_fen": 3000}][:size]


class _Connection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def cursor(self, *, row_factory):
        return _Cursor(self)


def _tools() -> tuple[ControlledTools, _Connection]:
    connection = _Connection()
    executor = GuardedQueryExecutor(connect=lambda: connection, clock=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc))
    return ControlledTools(catalog=load_default_catalog(), executor=executor), connection


def _context() -> ExecutionContext:
    return ExecutionContext(run_id="run-b3b-o5", tenant_id="A", principal_id="principal-A", role="requester")


def _prebound_gross():
    return (build_metric_binding(load_default_catalog(), "gross_fen", SEPTEMBER),)


PREBOUND_GROSS_REFUSALS = {
    "a_missing_paid_status": (
        "SELECT COALESCE(SUM(amount_fen), 0) AS gross_fen FROM orders WHERE created_at >= %s AND created_at < %s",
        {"0": SEPTEMBER["start"], "1": SEPTEMBER["end"]},
    ),
    "b_unrelated_extra_filter": (
        "SELECT COALESCE(SUM(amount_fen), 0) AS gross_fen FROM orders "
        "WHERE status = %s AND created_at >= %s AND created_at < %s AND customer_id = %s",
        {"0": "paid", "1": SEPTEMBER["start"], "2": SEPTEMBER["end"], "3": "c1"},
    ),
    "c_missing_time_window": (
        "SELECT COALESCE(SUM(amount_fen), 0) AS gross_fen FROM orders WHERE status = %s",
        {"0": "paid"},
    ),
}


@pytest.mark.parametrize("case", sorted(PREBOUND_GROSS_REFUSALS))
def test_prebound_gross_query_with_wrong_filters_is_refused_before_sql(case: str) -> None:
    sql, params = PREBOUND_GROSS_REFUSALS[case]
    tools, connection = _tools()
    with pytest.raises(ToolError) as caught:
        call_tool(tools, "query_readonly", {"sql": sql, "params": params}, context=_context(), metric_bindings=_prebound_gross())
    assert caught.value.code == "evidence_validation_failed"
    assert connection.executed == []


def test_prebound_gross_query_with_the_bound_filters_runs() -> None:
    """Control: the same binding with the exact filters is accepted."""

    tools, connection = _tools()
    output = call_tool(
        tools,
        "query_readonly",
        {
            "sql": "SELECT COALESCE(SUM(amount_fen), 0) AS gross_fen FROM orders WHERE status = %s AND created_at >= %s AND created_at < %s",
            "params": {"0": "paid", "1": SEPTEMBER["start"], "2": SEPTEMBER["end"]},
        },
        context=_context(),
        metric_bindings=_prebound_gross(),
    )
    assert output["row_count"] == 1
    assert len(connection.executed) == 1


def test_declared_net_fen_query_without_the_order_window_is_refused_before_sql() -> None:
    tools, connection = _tools()
    with pytest.raises(ToolError) as caught:
        call_tool(
            tools,
            "query_readonly",
            {
                "sql": "SELECT COALESCE(SUM(amount_fen), 0) AS gross_fen FROM orders WHERE status = %s",
                "params": {"0": "paid"},
                "metrics": ["net_fen"],
                "time_window": SEPTEMBER,
            },
            context=_context(),
        )
    assert caught.value.code == "evidence_validation_failed"
    assert connection.executed == []
