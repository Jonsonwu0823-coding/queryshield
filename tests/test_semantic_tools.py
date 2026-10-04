from __future__ import annotations

from datetime import datetime, timezone

import pytest

from queryshield.agent.proposals import ExecutionContext
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.tools import ControlledTools, ToolError


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

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def cursor(self, *, row_factory: object) -> _FakeCursor:
        return self.cursor_instance


def _context(*, role: str = "requester", principal_id: str = "principal-A") -> ExecutionContext:
    return ExecutionContext(
        run_id="run-A-tools",
        tenant_id="tenant-A",
        principal_id=principal_id,
        role=role,
    )


def _tools(rows: list[dict[str, object]]) -> tuple[ControlledTools, _FakeConnection]:
    connection = _FakeConnection(rows)
    executor = GuardedQueryExecutor(
        connect=lambda: connection,
        clock=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc),
    )
    return ControlledTools(executor=executor), connection


def test_search_catalog_is_deterministic_and_returns_source_version() -> None:
    tools, _ = _tools([])
    result = tools.search_catalog(
        {"query": "营业额", "top_k": 3},
        context=_context(),
    )

    assert result["items"]
    assert result["items"][0]["id"] == "metric.gross_fen"
    assert set(result["items"][0]) == {"id", "text", "source_id", "version"}
    assert result["items"][0]["source_id"] == "commerce-v1"
    assert result["items"][0]["version"] == "commerce-v1"
    assert tools.search_catalog({"query": "__no_such_catalog_term__"}, context=_context()) == {"items": []}


def test_search_and_describe_enforce_argument_shapes_and_role_visibility() -> None:
    tools, _ = _tools([])
    with pytest.raises(ToolError, match="unknown_argument"):
        tools.search_catalog({"query": "订单", "tenant_id": "B"}, context=_context())

    described = tools.describe_tables({"tables": ["orders"]}, context=_context())
    assert described["tables"] == [
        {
            "name": "orders",
            "columns": ["amount_fen", "created_at", "customer_id", "order_id", "status", "tenant_id"],
            "source_id": "commerce-v1",
            "version": "commerce-v1",
        }
    ]
    with pytest.raises(ToolError, match="table_not_allowed"):
        tools.describe_tables({"tables": ["secrets"]}, context=_context())

    requester = tools.search_catalog({"query": "customers.name"}, context=_context())
    approver = tools.search_catalog(
        {"query": "customers.name"},
        context=_context(role="approver", principal_id="principal-A-approver"),
    )
    assert not any(item["id"] == "field.customers.name" for item in requester["items"])
    assert any(item["id"] == "field.customers.name" for item in approver["items"])


def test_query_readonly_uses_server_tenant_and_returns_narrow_result() -> None:
    tools, connection = _tools([{"paid_count": 2}])
    result = tools.query_readonly(
        {
            "sql": "SELECT COUNT(*) AS paid_count FROM orders WHERE status = %s",
            "params": {"0": "paid"},
        },
        context=_context(),
    )

    assert set(result) == {"rows", "row_count", "result_id", "policy_version"}
    assert result["rows"] == [{"paid_count": 2}]
    assert result["row_count"] == 1
    assert connection.cursor_instance.executed is not None
    assert connection.cursor_instance.executed[1] == ("tenant-A", "paid")
    evidence = tools.get_result_evidence(result["result_id"], context=_context())
    assert evidence.tenant_id == "tenant-A"


def test_query_readonly_rejects_unknown_tables_sensitive_values_and_identity_params() -> None:
    tools, connection = _tools([{"name": "Alice"}])
    with pytest.raises(ToolError, match="table_not_allowed"):
        tools.query_readonly(
            {"sql": "SELECT * FROM secrets", "params": {}},
            context=_context(),
        )
    with pytest.raises(ToolError, match="approval_required"):
        tools.query_readonly(
            {"sql": "SELECT c.name FROM customers AS c", "params": {}},
            context=_context(),
        )
    with pytest.raises(ToolError, match="reserved_parameter"):
        tools.query_readonly(
            {"sql": "SELECT order_id FROM orders", "params": {"tenant_id": "B"}},
            context=_context(),
        )
    assert connection.cursor_instance.executed is None


def test_query_readonly_allows_sensitive_metadata_but_not_requester_values() -> None:
    tools, connection = _tools([{"customer_id": "c1"}])
    result = tools.query_readonly(
        {"sql": "SELECT c.customer_id FROM customers AS c", "params": {}},
        context=_context(),
    )
    assert result["rows"] == [{"customer_id": "c1"}]
    assert connection.cursor_instance.executed is not None
