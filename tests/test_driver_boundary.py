from types import SimpleNamespace

import pytest
from psycopg.errors import ConnectionFailure, InsufficientPrivilege, QueryCanceled, UndefinedColumn

from queryshield.agent.tool_execution import call_tool
from queryshield.agent.proposals import ExecutionContext
from queryshield.tools.semantic import ToolError


@pytest.mark.parametrize("error_type,code", [
    (UndefinedColumn, "invalid_sql"),
    (ConnectionFailure, "database_unavailable"),
    (InsufficientPrivilege, "forbidden"),
    (QueryCanceled, "query_timeout"),
])
def test_actual_driver_errors_have_safe_distinct_codes(error_type, code):
    calls = []

    def execute(*args, **kwargs):
        calls.append(args)
        raise error_type("sensitive driver message must not be emitted")

    with pytest.raises(ToolError) as captured:
        call_tool(SimpleNamespace(call=execute), "query_readonly", {}, context=None)
    assert captured.value.code == code
    assert "sensitive" not in captured.value.message
    assert len(calls) == 1


def test_non_driver_programming_error_is_not_misclassified_as_repairable_sql():
    def execute(*args, **kwargs):
        raise RuntimeError("programming error")

    with pytest.raises(RuntimeError):
        call_tool(SimpleNamespace(call=execute), "query_readonly", {}, context=None)


@pytest.mark.parametrize(
    "sql,params,expected",
    [
        (
            "SELECT order_id FROM orders WHERE tenant_id = %s AND status = %s",
            {"0": "B", "1": "paid"},
            {"0": "A", "1": "paid"},
        ),
        (
            "SELECT order_id FROM orders WHERE %s = tenant_id AND status = %s",
            {"0": "B", "1": "paid"},
            {"0": "A", "1": "paid"},
        ),
        (
            "SELECT order_id FROM orders WHERE NOT (tenant_id = %s) AND status = %s",
            {"0": "B", "1": "paid"},
            {"0": "A", "1": "paid"},
        ),
        (
            "SELECT o.order_id FROM orders AS o INNER JOIN customers AS c "
            "ON o.tenant_id = %s AND o.customer_id = c.customer_id WHERE o.status = %s",
            {"0": "B", "1": "paid"},
            {"0": "A", "1": "paid"},
        ),
    ],
)
def test_tool_boundary_rebinds_tenant_equality_params_without_mutating_proposal(sql, params, expected):
    original = {"sql": sql, "params": dict(params)}
    received = []

    def execute(name, arguments, **kwargs):
        received.append(arguments)
        return {"status": "succeeded"}

    context = ExecutionContext(
        run_id="run-scope-bind",
        tenant_id="A",
        principal_id="principal-A",
        role="requester",
    )
    call_tool(
        SimpleNamespace(call=execute),
        "query_readonly",
        original,
        context=context,
    )

    assert received[0]["params"] == expected
    assert original["params"] == params
    assert received[0] is not original


def test_tool_boundary_does_not_rebind_tenant_column_joins_or_other_parameters():
    original = {
        "sql": (
            "SELECT o.order_id FROM orders AS o INNER JOIN customers AS c "
            "ON o.tenant_id = c.tenant_id WHERE o.status = %s"
        ),
        "params": {"0": "paid"},
    }
    received = []

    def execute(name, arguments, **kwargs):
        received.append(arguments)
        return {"status": "succeeded"}

    context = ExecutionContext(
        run_id="run-scope-bind-column-join",
        tenant_id="A",
        principal_id="principal-A",
        role="requester",
    )
    call_tool(SimpleNamespace(call=execute), "query_readonly", original, context=context)

    assert received[0]["params"] == {"0": "paid"}
