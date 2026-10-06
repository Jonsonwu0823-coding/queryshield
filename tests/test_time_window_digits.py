"""A declared time window is written with ASCII digits; other digits are refused before any SQL runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from queryshield.agent import BoundedAgent, ModelCallStore
from queryshield.agent.metric_intent import MetricDeclarationError, normalize_time_window
from queryshield.agent.proposals import ExecutionContext
from queryshield.agent.tool_execution import call_tool
from queryshield.catalog import load_default_catalog
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.tools import ControlledTools, ToolError

from test_agent_graph import _ScriptedModel

WINDOW_FILTER = "status = %s AND created_at >= %s AND created_at < %s"
SQL = f"SELECT COALESCE(SUM(amount_fen), 0) AS gross_fen FROM orders WHERE {WINDOW_FILTER}"
END = "2026-10-01T00:00:00Z"
STARTS = {
    "fullwidth": "２０２６-09-01T00:00:00Z",
    "arabic-indic": "٢٠٢٦-09-01T00:00:00Z",
    "fullwidth-time": "2026-09-01T００:00:00Z",
    "fullwidth-month": "2026-０9-01T00:00:00Z",
    "fullwidth-day": "2026-09-０1T00:00:00Z",
    "fullwidth-minute": "2026-09-01T00:０0:00Z",
    "fullwidth-second": "2026-09-01T00:00:０0Z",
    "superscript": "2026-09-01T0²:00:00Z",
}


def _arguments(start: str) -> dict[str, object]:
    return {
        "sql": SQL,
        "params": {"0": "paid", "1": start, "2": END},
        "metrics": ["gross_fen"],
        "time_window": {"start": start, "end": END},
    }


def _tools() -> ControlledTools:
    def connect():
        raise AssertionError("a query with a refused time window must not reach the database")

    executor = GuardedQueryExecutor(connect=connect, clock=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc))
    return ControlledTools(catalog=load_default_catalog(), executor=executor)


def _context(run_id: str) -> ExecutionContext:
    return ExecutionContext(run_id=run_id, tenant_id="A", principal_id="principal-A", role="requester")


@pytest.mark.parametrize("start", STARTS.values(), ids=STARTS.keys())
def test_normalizing_a_window_with_non_ascii_digits_is_invalid_time_window(start: str) -> None:
    with pytest.raises(MetricDeclarationError) as caught:
        normalize_time_window({"start": start, "end": END})
    assert caught.value.code == "invalid_time_window"
    assert caught.value.message == "time_window.start must be YYYY-MM-DDTHH:MM:SSZ in UTC"


def test_an_ascii_window_is_normalized_as_before() -> None:
    assert normalize_time_window({"start": "2026-09-01T00:00:00Z", "end": END}) == {
        "start": "2026-09-01T00:00:00Z",
        "end": END,
        "timezone": "UTC",
    }


def test_a_window_with_a_nine_in_every_field_is_normalized_unchanged() -> None:
    window = {"start": "2029-09-19T19:59:59Z", "end": "2029-09-29T09:09:09Z"}
    assert normalize_time_window(window) == {**window, "timezone": "UTC"}


@pytest.mark.parametrize("start", STARTS.values(), ids=STARTS.keys())
def test_a_proposed_query_with_non_ascii_digits_in_its_window_is_refused_before_the_database(start: str) -> None:
    with pytest.raises(ToolError) as caught:
        call_tool(_tools(), "query_readonly", _arguments(start), context=_context("run-window"))
    assert caught.value.code == "invalid_time_window"


@pytest.mark.parametrize("start", [STARTS["fullwidth"], STARTS["arabic-indic"]], ids=["fullwidth", "arabic-indic"])
def test_the_agent_run_refuses_a_window_with_non_ascii_digits_without_running_sql(start: str) -> None:
    proposal = json.dumps(
        {"type": "tool_call", "name": "query_readonly", "arguments": _arguments(start)}, ensure_ascii=False
    )
    agent = BoundedAgent(_ScriptedModel([proposal, proposal]), tools=_tools(), call_store=ModelCallStore())

    result = agent.run(_context("run-window-agent"), "2026年9月已支付订单总额")

    assert result.error_code is not None
    assert [event["kind"] for event in result.events][:2] == ["model_call", "tool_call"]
    assert result.events[1]["error_code"] == "invalid_time_window"
