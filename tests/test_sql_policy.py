from __future__ import annotations

import pytest

from queryshield.policy.sql import ColumnRef, FunctionCall, LiteralValue, ParameterRef, SQLPolicyError, parse_readonly_select


def _function_names(expression):
    """The functions an expression calls, outermost first."""

    if isinstance(expression, FunctionCall):
        yield expression.name
        for argument in expression.arguments:
            yield from _function_names(argument)


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
    assert [name for item in statement.projection for name in _function_names(item.expression)] == ["COUNT", "COALESCE", "SUM"]
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


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        (None, "SQL must be a string"),
        (b"SELECT 1", "SQL must be a string"),
        ("  \n\t ", "SQL must not be blank"),
        ("SELECT o.order_id FROM orders AS o" + " " * 3967, "SQL exceeds the 4000 character limit"),
    ],
    ids=["none", "bytes", "blank", "4001_characters"],
)
def test_the_tokenizer_refuses_input_it_does_not_read(sql: object, message: str) -> None:
    with pytest.raises(SQLPolicyError) as error:
        parse_readonly_select(sql)  # type: ignore[arg-type]

    assert (error.value.code, str(error.value)) == ("invalid_sql", f"invalid_sql: {message}")


def test_4000_characters_are_within_the_limit() -> None:
    sql = "SELECT o.order_id FROM orders AS o"
    assert parse_readonly_select(sql + " " * (4000 - len(sql))).referenced_tables == ("orders",)


def test_a_cast_is_refused_as_unsupported_syntax() -> None:
    with pytest.raises(SQLPolicyError) as error:
        parse_readonly_select("SELECT o.amount_fen::text FROM orders AS o")

    assert (error.value.code, str(error.value)) == ("unsupported_syntax", "unsupported_syntax: casts are not supported")


def test_a_question_mark_is_read_as_a_parameter_like_percent_s() -> None:
    statement = parse_readonly_select("SELECT o.order_id FROM orders AS o WHERE o.tenant_id = ? AND o.status = %s")

    assert [comparison.right for comparison in (statement.where.left, statement.where.right)] == [ParameterRef(0), ParameterRef(1)]


@pytest.mark.parametrize("tail", ["LIMIT 1.", "WHERE o.amount_fen > 1.", "WHERE o.amount_fen > 1.x"])
def test_a_number_with_a_point_and_no_digits_after_it_is_invalid(tail: str) -> None:
    with pytest.raises(SQLPolicyError) as error:
        parse_readonly_select(f"SELECT o.order_id FROM orders AS o {tail}")

    assert str(error.value) == "invalid_sql: invalid numeric literal"


def test_a_name_keeps_its_digits_and_underscores() -> None:
    statement = parse_readonly_select("SELECT o_2.order_id FROM orders AS o_2 WHERE o_2.status = %s")

    assert statement.table_aliases == {"o_2": "orders"}
    assert statement.projection[0].expression == ColumnRef(name="order_id", qualifier="o_2")


def test_true_false_and_null_are_read_as_their_values_in_any_case() -> None:
    statement = parse_readonly_select("SELECT TRUE AS t, false AS f, Null AS n FROM orders AS o")

    assert [item.expression for item in statement.projection] == [LiteralValue(True), LiteralValue(False), LiteralValue(None)]
