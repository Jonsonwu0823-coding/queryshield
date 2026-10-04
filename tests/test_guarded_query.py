from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import json

import pytest

from queryshield.agent.proposals import ExecutionContext
from queryshield.db.guarded import (
    GuardedQueryError,
    GuardedQueryExecutor,
    render_scoped_select,
)
from queryshield.policy.sql import SQLPolicyError, parse_readonly_select


class _FakeCursor:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.executed: tuple[str, tuple[object, ...]] | None = None

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        self.executed = (sql, params)

    def fetchmany(self, size: int) -> list[dict[str, object]]:
        return self.rows[:size]


class _FakeConnection:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.cursor_instance = _FakeCursor(rows)
        self.cursor_row_factory: object | None = None

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def cursor(self, *, row_factory: object) -> _FakeCursor:
        self.cursor_row_factory = row_factory
        return self.cursor_instance


def _context() -> ExecutionContext:
    return ExecutionContext(
        run_id="run-A-1",
        tenant_id="tenant-A",
        principal_id="principal-A-requester",
        role="requester",
    )


def test_render_scopes_every_table_with_server_tenant_and_keeps_values_parameterized() -> None:
    statement = parse_readonly_select(
        "SELECT o.amount_fen FROM orders AS o "
        "INNER JOIN customers AS c ON o.customer_id = c.customer_id "
        "WHERE o.status = %s"
    )

    rendered = render_scoped_select(statement, tenant_id="tenant-A", input_params=("paid",))

    assert 'SELECT * FROM "orders" WHERE "tenant_id" = %s' in rendered.sql
    assert 'SELECT * FROM "customers" WHERE "tenant_id" = %s' in rendered.sql
    assert "tenant-A" not in rendered.sql
    assert rendered.params == ("tenant-A", "tenant-A", "paid")


def test_executor_records_actual_rows_and_server_owned_evidence() -> None:
    connection = _FakeConnection([{"paid_count": 2}])
    executor = GuardedQueryExecutor(
        connect=lambda: connection,
        clock=lambda: datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc),
    )

    result = executor.execute(
        "SELECT COUNT(*) AS paid_count FROM orders WHERE tenant_id = %s",
        context=_context(),
        params=("tenant-B",),
    )

    assert result.rows == ({"paid_count": 2},)
    assert result.evidence.tenant_id == "tenant-A"
    assert result.evidence.principal_id == "principal-A-requester"
    assert result.evidence.row_count == 1
    assert connection.cursor_instance.executed is not None
    assert connection.cursor_instance.executed[1][0] == "tenant-A"
    assert connection.cursor_instance.executed[1][1] == "tenant-B"


def test_executor_normalizes_database_values_for_json_evidence() -> None:
    connection = _FakeConnection(
        [
            {
                "gross_fen": Decimal("15000"),
                "ratio": Decimal("1.25"),
                "observed_on": date(2026, 9, 18),
            }
        ]
    )
    executor = GuardedQueryExecutor(connect=lambda: connection)

    result = executor.execute(
        "SELECT COUNT(*) AS gross_fen FROM orders",
        context=_context(),
    )

    assert result.rows == (
        {
            "gross_fen": 15000,
            "ratio": "1.25",
            "observed_on": "2026-09-18",
        },
    )
    json.dumps(result.evidence.as_dict())


def test_invalid_query_and_parameter_mismatch_do_not_open_database_connection() -> None:
    opened = False

    def connect() -> _FakeConnection:
        nonlocal opened
        opened = True
        return _FakeConnection([])

    executor = GuardedQueryExecutor(connect=connect)
    with pytest.raises(SQLPolicyError):
        executor.execute("DELETE FROM orders", context=_context())
    assert opened is False

    with pytest.raises(GuardedQueryError) as error:
        executor.execute(
            "SELECT * FROM orders WHERE tenant_id = %s",
            context=_context(),
            params=(),
        )
    assert error.value.code == "parameter_mismatch"
    assert opened is False


def test_more_than_one_hundred_rows_is_not_returned_as_success() -> None:
    connection = _FakeConnection([{"order_id": index} for index in range(101)])
    executor = GuardedQueryExecutor(connect=lambda: connection)

    with pytest.raises(GuardedQueryError) as error:
        executor.execute("SELECT * FROM orders", context=_context())

    assert error.value.code == "limit_reached"
