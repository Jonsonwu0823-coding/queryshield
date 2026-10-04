from __future__ import annotations

import pytest

from queryshield.policy.sql import FunctionCall, SQLPolicyError, parse_readonly_select


def test_allowed_select_is_parsed_into_a_restricted_ast() -> None:
    statement = parse_readonly_select(
        """
        SELECT o.status, COUNT(*) AS order_count,
               COALESCE(SUM(o.amount_fen), 0) AS gross_fen
        FROM orders AS o
        INNER JOIN customers AS c ON o.tenant_id = c.tenant_id
                                  AND o.customer_id = c.customer_id
        WHERE o.tenant_id = %s
          AND o.status = 'paid'
          AND o.created_at >= %s
          AND o.created_at < %s
        GROUP BY o.status
        ORDER BY order_count DESC
        LIMIT 10
        """
    )

    assert statement.referenced_tables == ("orders", "customers")
    assert statement.function_names == ("COUNT", "COALESCE", "SUM")
    assert statement.limit == 10
    assert isinstance(statement.projection[1].expression, FunctionCall)


@pytest.mark.parametrize(
    ("sql", "code"),
    [
        ("UPDATE orders SET amount_fen = 1", "statement_not_allowed"),
        (
            "WITH changed AS (DELETE FROM orders RETURNING order_id) SELECT * FROM orders",
            "unsupported_syntax",
        ),
        (
            "SELECT COUNT(*) FROM orders; DELETE FROM orders",
            "multiple_statements",
        ),
        ("SELECT relname FROM pg_catalog.pg_class", "table_not_allowed"),
        ("SELECT LOWER(status) FROM orders", "function_not_allowed"),
        ("SELECT * FROM orders LEFT JOIN customers ON orders.customer_id = customers.customer_id", "unsupported_syntax"),
        ("SELECT * FROM orders -- hide a second statement\n", "unsupported_syntax"),
    ],
)
def test_unsafe_or_unsupported_sql_is_rejected_before_database_call(
    sql: str,
    code: str,
) -> None:
    with pytest.raises(SQLPolicyError) as error:
        parse_readonly_select(sql)

    assert error.value.code == code


def test_parameterized_filter_and_single_statement_terminator_are_allowed() -> None:
    statement = parse_readonly_select(
        "SELECT COUNT(*) FROM orders WHERE tenant_id = %s AND status = %s LIMIT %s;"
    )

    assert statement.referenced_tables == ("orders",)
    assert statement.limit is not None
