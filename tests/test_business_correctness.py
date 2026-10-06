from __future__ import annotations

from datetime import datetime, timezone

from queryshield.catalog import DEFAULT_CATALOG_VERSION
from queryshield.agent.context import (
    NET_FEN_GROSS_QUERY,
    NET_FEN_PLAN_ID,
    NET_FEN_REFUND_QUERY,
)
from queryshield.agent.proposals import ExecutionContext, MetricBinding
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.tools import ControlledTools


LEGACY_NET_FEN_SQL = (
    "SELECT COALESCE(SUM(o.amount_fen), 0) - "
    "COALESCE(SUM(r.amount_fen), 0) AS net_fen "
    "FROM orders AS o INNER JOIN refunds AS r "
    "ON o.order_id = r.order_id AND o.tenant_id = r.tenant_id "
    "WHERE o.status = 'paid' AND o.created_at >= %s "
    "AND o.created_at < %s AND r.created_at >= %s "
    "AND r.created_at < %s"
)


class _CommerceCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self._rows: list[dict[str, object]] = []

    def __enter__(self) -> _CommerceCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        self.executed.append((sql, params))
        normalized = sql.lower()
        if 'sum("o"."amount_fen")' in normalized and 'sum("r"."amount_fen")' in normalized:
            # commerce-v1: A/o1 has two refunds, so a direct join counts the
            # 10,000-fen order twice: (10,000 + 10,000) - (2,000 + 1,000).
            self._rows = [{"net_fen": 17000}]
        elif 'as "gross_fen"' in normalized:
            self._rows = [{"gross_fen": 15000}]
        elif 'as "refund_fen"' in normalized:
            self._rows = [{"refund_fen": 3000}]
        else:
            self._rows = []

    def fetchmany(self, size: int) -> list[dict[str, object]]:
        return self._rows[:size]


class _CommerceConnection:
    def __init__(self) -> None:
        self.cursor_instance = _CommerceCursor()

    def __enter__(self) -> _CommerceConnection:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def cursor(self, *, row_factory: object) -> _CommerceCursor:
        return self.cursor_instance


def _context() -> ExecutionContext:
    return ExecutionContext(
        run_id="run-t06-legacy-net-repro",
        tenant_id="A",
        principal_id="principal-t06-repro",
        role="requester",
    )


def _binding() -> MetricBinding:
    return MetricBinding(
        metric_id="net_fen",
        result_position="net_fen",
        unit="CNY_fen",
        time_window={
            "start": "2026-09-01T00:00:00Z",
            "end": "2026-10-01T00:00:00Z",
            "timezone": "UTC",
        },
        catalog_source_id="commerce-v1",
        catalog_version=DEFAULT_CATALOG_VERSION,
        plan_id=NET_FEN_PLAN_ID,
    )


def test_legacy_net_join_is_corrected_by_controlled_plan() -> None:
    """The old direct-join request is now routed through the verified plan."""

    connection = _CommerceConnection()
    tools = ControlledTools(
        executor=GuardedQueryExecutor(
            connect=lambda: connection,
            clock=lambda: datetime(2026, 9, 22, tzinfo=timezone.utc),
        )
    )

    result = tools.query_readonly(
        {
            "sql": LEGACY_NET_FEN_SQL,
            "params": {
                "0": "2026-09-01T00:00:00Z",
                "1": "2026-10-01T00:00:00Z",
                "2": "2026-09-01T00:00:00Z",
                "3": "2026-10-01T00:00:00Z",
            },
        },
        context=_context(),
        metric_bindings=(_binding(),),
    )

    # The pre-fix baseline intentionally failed this same assertion with 17,000;
    # the current server-owned plan must now make the legacy request correct.
    assert result["rows"] == [{"net_fen": 12000}]


def test_two_aggregate_template_steps_still_return_bound_net_evidence() -> None:
    """The published gross/refund steps must enter the same server-owned plan."""

    connection = _CommerceConnection()
    tools = ControlledTools(
        executor=GuardedQueryExecutor(
            connect=lambda: connection,
            clock=lambda: datetime(2026, 9, 22, tzinfo=timezone.utc),
        )
    )
    binding = (_binding(),)

    gross = tools.query_readonly(
        {
            "sql": NET_FEN_GROSS_QUERY,
            "params": {
                "0": "paid",
                "1": "2026-09-01T00:00:00Z",
                "2": "2026-10-01T00:00:00Z",
            },
        },
        context=_context(),
        metric_bindings=binding,
    )
    refund = tools.query_readonly(
        {
            "sql": NET_FEN_REFUND_QUERY,
            "params": {
                "0": "paid",
                "1": "2026-09-01T00:00:00Z",
                "2": "2026-10-01T00:00:00Z",
                "3": "2026-09-01T00:00:00Z",
                "4": "2026-10-01T00:00:00Z",
            },
        },
        context=_context(),
        metric_bindings=binding,
    )

    assert gross["rows"] == [{"net_fen": 12000}]
    assert refund["rows"] == [{"net_fen": 12000}]
    assert gross["metric_plan_id"] == NET_FEN_PLAN_ID
    assert refund["metric_plan_id"] == NET_FEN_PLAN_ID
