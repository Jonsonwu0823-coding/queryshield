from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

from psycopg.rows import dict_row

from queryshield.agent.proposals import (
    ExecutionContext,
    MetricBinding,
    ResultEvidence,
)
from queryshield.catalog.catalog import DEFAULT_CATALOG_VERSION
from queryshield.db.readonly import bind_transaction_tenant, connect_readonly
from queryshield.policy.sql import (
    MAX_RESULT_ROWS,
    SQL_POLICY_VERSION,
    BinaryExpression,
    BooleanExpression,
    ColumnRef,
    Comparison,
    Condition,
    FunctionCall,
    LiteralValue,
    NotExpression,
    ParameterRef,
    SelectItem,
    SelectStatement,
    Star,
    TableRef,
    parse_readonly_select,
)


class GuardedQueryError(ValueError):
    """A query cannot be safely executed or its result cannot be represented."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class RenderedQuery:
    sql: str
    params: tuple[object, ...]


@dataclass(frozen=True)
class GuardedQueryResult:
    evidence: ResultEvidence

    @property
    def rows(self) -> tuple[Mapping[str, object], ...]:
        return self.evidence.rows


ConnectionFactory = Callable[[], Any]
Clock = Callable[[], datetime]


def render_scoped_select(
    statement: SelectStatement,
    *,
    tenant_id: str,
    input_params: Sequence[object] = (),
) -> RenderedQuery:
    """Render only the parsed AST and add one server-owned tenant predicate per table."""
    if not isinstance(statement, SelectStatement):
        raise TypeError("statement must be a SelectStatement")
    if type(tenant_id) is not str or not tenant_id.strip():
        raise GuardedQueryError("invalid_tenant", "tenant context must be a non-empty string")
    _validate_input_params(input_params)
    _validate_qualifiers(statement)
    renderer = _Renderer(statement, tenant_id, tuple(input_params))
    return renderer.render()


class GuardedQueryExecutor:
    """Execute the parsed AST through a tenant-scoped, read-only database boundary."""

    def __init__(
        self,
        *,
        connect: ConnectionFactory = connect_readonly,
        clock: Clock | None = None,
        policy_version: str = SQL_POLICY_VERSION,
        catalog_version: str = DEFAULT_CATALOG_VERSION,
    ) -> None:
        self._connect = connect
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._policy_version = policy_version
        self._catalog_version = catalog_version

    def execute(
        self,
        sql: str,
        *,
        context: ExecutionContext,
        params: Sequence[object] = (),
        metric_bindings: Sequence[MetricBinding] = (),
    ) -> GuardedQueryResult:
        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")
        statement = parse_readonly_select(sql)
        rendered = render_scoped_select(
            statement,
            tenant_id=context.tenant_id,
            input_params=params,
        )

        read_only_default = self._connect is connect_readonly
        connection = self._connect()
        with connection:
            if read_only_default:
                bind_transaction_tenant(connection, context.tenant_id)
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(rendered.sql, rendered.params)
                rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)

        if len(rows) > MAX_RESULT_ROWS:
            raise GuardedQueryError(
                "limit_reached",
                "query returned more than the 100 row limit",
            )
        normalized_rows = _normalize_rows(rows)
        observed_at = self._clock()
        if not isinstance(observed_at, datetime) or observed_at.tzinfo is None:
            raise GuardedQueryError("invalid_clock", "observation time must be timezone-aware")

        evidence = ResultEvidence.from_server_execution(
            context,
            result_id=f"result-{uuid4()}",
            rows=normalized_rows,
            normalized_query=rendered.sql,
            params=rendered.params,
            observed_at=observed_at,
            policy_version=self._policy_version,
            catalog_version=self._catalog_version,
            metric_bindings=metric_bindings,
        )
        return GuardedQueryResult(evidence=evidence)


class _Renderer:
    def __init__(
        self,
        statement: SelectStatement,
        tenant_id: str,
        input_params: tuple[object, ...],
    ) -> None:
        self._statement = statement
        self._tenant_id = tenant_id
        self._input_params = input_params
        self._params: list[object] = []
        self._used_input_indexes: set[int] = set()

    def render(self) -> RenderedQuery:
        projection = ", ".join(self._select_item(item) for item in self._statement.projection)
        sql_parts = [f"SELECT {projection} FROM {self._scoped_table(self._statement.from_table)}"]
        for join in self._statement.joins:
            sql_parts.append(
                f" INNER JOIN {self._scoped_table(join.table)} ON {self._condition(join.condition)}"
            )
        if self._statement.where is not None:
            sql_parts.append(f" WHERE {self._condition(self._statement.where)}")
        if self._statement.group_by:
            sql_parts.append(" GROUP BY " + ", ".join(self._expression(item) for item in self._statement.group_by))
        if self._statement.order_by:
            sql_parts.append(
                " ORDER BY "
                + ", ".join(
                    f"{self._expression(item.expression)} {item.direction}"
                    for item in self._statement.order_by
                )
            )
        if self._statement.limit is not None:
            if type(self._statement.limit) is int:
                sql_parts.append(f" LIMIT {self._statement.limit}")
            else:
                sql_parts.append(f" LIMIT {self._expression(self._statement.limit)}")
        else:
            sql_parts.append(f" LIMIT {MAX_RESULT_ROWS + 1}")

        expected_inputs = max(self._used_input_indexes, default=-1) + 1
        if len(self._input_params) != expected_inputs:
            raise GuardedQueryError(
                "parameter_mismatch",
                "number of parameters does not match the parsed query",
            )
        return RenderedQuery(sql="".join(sql_parts), params=tuple(self._params))

    def _select_item(self, item: SelectItem) -> str:
        rendered = self._expression(item.expression)
        if item.alias is not None:
            rendered += f" AS {_quote_identifier(item.alias)}"
        return rendered

    def _scoped_table(self, table: TableRef) -> str:
        alias = table.alias or table.name
        self._params.append(self._tenant_id)
        return (
            f'(SELECT * FROM {_quote_identifier(table.name)} '
            f'WHERE {_quote_identifier("tenant_id")} = %s) AS {_quote_identifier(alias)}'
        )

    def _condition(self, condition: Condition) -> str:
        if isinstance(condition, Comparison):
            left = self._expression(condition.left)
            if condition.operator in {"IS NULL", "IS NOT NULL"}:
                return f"{left} {condition.operator}"
            return f"{left} {condition.operator} {self._expression(condition.right)}"
        if isinstance(condition, BooleanExpression):
            return f"({self._condition(condition.left)} {condition.operator} {self._condition(condition.right)})"
        if isinstance(condition, NotExpression):
            return f"NOT ({self._condition(condition.operand)})"
        raise GuardedQueryError("invalid_ast", "unknown condition node")

    def _expression(self, expression: object) -> str:
        if isinstance(expression, ColumnRef):
            if expression.qualifier is None:
                return _quote_identifier(expression.name)
            return f"{_quote_identifier(expression.qualifier)}.{_quote_identifier(expression.name)}"
        if isinstance(expression, LiteralValue):
            self._params.append(expression.value)
            return "%s"
        if isinstance(expression, ParameterRef):
            if expression.index >= len(self._input_params):
                raise GuardedQueryError("parameter_mismatch", "query parameter is missing")
            self._used_input_indexes.add(expression.index)
            self._params.append(self._input_params[expression.index])
            return "%s"
        if isinstance(expression, Star):
            return "*"
        if isinstance(expression, FunctionCall):
            args = ", ".join(self._expression(argument) for argument in expression.arguments)
            return f"{expression.name}({args})"
        if isinstance(expression, BinaryExpression):
            return f"({self._expression(expression.left)} {expression.operator} {self._expression(expression.right)})"
        raise GuardedQueryError("invalid_ast", "unknown expression node")


def _validate_input_params(params: Sequence[object]) -> None:
    if isinstance(params, (str, bytes)):
        raise GuardedQueryError("invalid_params", "parameters must be a sequence of scalars")
    for value in params:
        if value is not None and not isinstance(value, (date, Decimal)) and type(value) not in {
            str,
            int,
            float,
            bool,
        }:
            raise GuardedQueryError("invalid_params", "parameters must be scalar values")


def _validate_qualifiers(statement: SelectStatement) -> None:
    aliases = statement.table_aliases
    expressions: list[object] = [item.expression for item in statement.projection]
    expressions.extend(statement.group_by)
    expressions.extend(item.expression for item in statement.order_by)
    if statement.where is not None:
        _collect_condition_expressions(statement.where, expressions)
    for join in statement.joins:
        _collect_condition_expressions(join.condition, expressions)
    for expression in expressions:
        for qualifier in _collect_qualifiers(expression):
            if qualifier not in aliases:
                raise GuardedQueryError("unknown_qualifier", "column qualifier is not a known table alias")


def _collect_condition_expressions(condition: Condition, expressions: list[object]) -> None:
    if isinstance(condition, Comparison):
        expressions.extend((condition.left, condition.right))
    elif isinstance(condition, BooleanExpression):
        _collect_condition_expressions(condition.left, expressions)
        _collect_condition_expressions(condition.right, expressions)
    elif isinstance(condition, NotExpression):
        _collect_condition_expressions(condition.operand, expressions)


def _collect_qualifiers(expression: object) -> tuple[str, ...]:
    if isinstance(expression, ColumnRef):
        return (expression.qualifier,) if expression.qualifier is not None else ()
    if isinstance(expression, FunctionCall):
        return tuple(
            qualifier
            for argument in expression.arguments
            for qualifier in _collect_qualifiers(argument)
        )
    if isinstance(expression, BinaryExpression):
        return _collect_qualifiers(expression.left) + _collect_qualifiers(expression.right)
    return ()


def _normalize_rows(rows: Sequence[object]) -> tuple[Mapping[str, object], ...]:
    normalized: list[Mapping[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise GuardedQueryError("invalid_database_row", "database row is not a mapping")
        normalized.append(
            {key: _normalize_database_value(value) for key, value in row.items()}
        )
    return tuple(normalized)


def _normalize_database_value(value: object) -> object:
    """Keep server evidence JSON-safe without losing numeric precision."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.isoformat()
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise GuardedQueryError(
                "invalid_database_value", "database numeric value is not finite"
            )
        if value == value.to_integral_value():
            return int(value)
        return format(value, "f")
    if isinstance(value, Mapping):
        return {key: _normalize_database_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_database_value(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise GuardedQueryError(
        "invalid_database_value", "database value cannot be returned as JSON"
    )


def _quote_identifier(identifier: str) -> str:
    if type(identifier) is not str or not identifier or "\x00" in identifier:
        raise GuardedQueryError("invalid_identifier", "identifier is invalid")
    return '"' + identifier.replace('"', '""') + '"'
