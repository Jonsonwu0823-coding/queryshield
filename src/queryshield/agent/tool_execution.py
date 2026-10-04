"""Keep database driver failures inside the bounded Agent tool protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from psycopg import Error as DatabaseDriverError
from psycopg import OperationalError as DatabaseOperationalError

from queryshield.policy.sql import (
    BooleanExpression,
    ColumnRef,
    Comparison,
    FunctionCall,
    LiteralValue,
    NotExpression,
    ParameterRef,
    SQLPolicyError,
    SelectStatement,
    Star,
    parse_readonly_select,
)
from queryshield.agent.metric_intent import MetricDeclarationError, resolve_query_declaration
from queryshield.agent.proposals import ExecutionContext, MetricBinding, ResultEvidence
from queryshield.catalog.catalog import ALLOWED_TABLE_COLUMNS, ClarificationRule, ClarificationValue
from queryshield.catalog.phrases import ClarificationReading, check_declaration
from queryshield.db.guarded import GuardedQueryError, render_scoped_select
from queryshield.tools.semantic import ToolError, _controlled_error_detail, _params

# The guarded executor refuses results over its 100-row bound with
# GuardedQueryError("limit_reached").  Inside the Agent that is a failed query,
# not the Agent's own budget status ``limit_reached``; give it its own code.
RESULT_ROW_LIMIT_CODE = "result_row_limit"


class ApprovalRequiredError(ToolError):
    """A fully verified read of approval-protected values; nothing was executed.

    ``pending_call`` is the canonical, server-verified query_readonly call the
    approval binds to (SQL, params, declared metrics and time window).
    """

    def __init__(self, pending_call: Mapping[str, object]) -> None:
        super().__init__("approval_required", "customers.name values require the approval path")
        self.pending_call = dict(pending_call)


CLARIFICATION_REQUIRED_CODE = "clarification_required"
CLARIFICATION_VALUE_UNSUPPORTED_CODE = "clarification_value_unsupported"
METRIC_CONTRADICTS_QUESTION_CODE = "metric_contradicts_question"


class ClarificationRequiredError(ToolError):
    """The model declared a metric the question's wording leaves open.

    Nothing was executed; the run waits for the user on the catalog rule's
    fixed question.
    """

    def __init__(self, rule: ClarificationRule) -> None:
        super().__init__(CLARIFICATION_REQUIRED_CODE, "the question's wording needs a catalog clarification first")
        self.rule = rule


class ClarificationValueUnsupportedError(ToolError):
    """The question (or the user's choice) names a scope no catalog metric supports.

    Terminal and not repairable: the model may not widen a metric definition.
    ``note`` is the catalog's fixed explanation.
    """

    def __init__(self, rule: ClarificationRule, value: ClarificationValue) -> None:
        super().__init__(CLARIFICATION_VALUE_UNSUPPORTED_CODE, "the requested scope has no supported catalog metric")
        self.rule = rule
        self.value = value
        self.note = str(value.unsupported_note)


class MetricContradictsQuestionError(ToolError):
    """The question names a value of a catalog rule and the model declared another.

    Nothing was executed.  B1 sends the model back once (the same budget as
    an ask the wording settles); B0 has no repair turn and fails.
    """

    def __init__(self, rule: ClarificationRule) -> None:
        super().__init__(METRIC_CONTRADICTS_QUESTION_CODE, "the declared metric contradicts the value the question names")
        self.rule = rule


def _driver_error(exc: DatabaseDriverError) -> ToolError:
    state = exc.sqlstate
    if state is None and isinstance(exc, DatabaseOperationalError):
        # A failed connection carries no SQLSTATE; it is not a SQL error.
        return ToolError("database_unavailable", "database connection failed")
    if state in {"42601", "42703", "42803", "42P01"}:
        return ToolError("invalid_sql", "database rejected the query syntax or referenced columns")
    if state == "42501":
        return ToolError("forbidden", "database permission denied")
    if state == "57014":
        return ToolError("query_timeout", "database query was cancelled or timed out")
    if state is not None and state.startswith("08"):
        return ToolError("database_unavailable", "database connection failed")
    return ToolError("database_error", "database execution failed")


def _row_limit_error(exc: ToolError) -> ToolError:
    return ToolError(RESULT_ROW_LIMIT_CODE, "query returned more than the 100 row limit")


def call_tool(tools, name, arguments, *, context, metric_bindings=(), request_time_window=None, clarifications=None):
    """Use the existing safety facade; translate only actual driver exceptions.

    SQL syntax/name errors may use the existing one-repair budget. Connection,
    timeout and privilege failures must never trigger a SQL repair or a retry.
    Driver messages can contain connection details, so expose fixed messages.

    ``metric_bindings`` are server pre-bound bindings (a confirmed clarification
    slot).  A query_readonly call may also declare catalog ``metrics`` and a
    ``time_window``; the server builds those bindings from the catalog and they
    must pass the same projection or controlled-plan checks before execution.

    ``clarifications`` is the run's reading of the catalog phrase table
    (question, clarification answers, confirmed metrics).  A declared metric
    the wording leaves open raises ClarificationRequiredError; a scope with no
    supported metric raises ClarificationValueUnsupportedError.  Both happen
    before the approval gate and before any SQL.
    """
    execution_arguments, execution_bindings, declared_metric_ids = _resolve_call(
        tools,
        name,
        arguments,
        context=context,
        metric_bindings=metric_bindings,
        request_time_window=request_time_window,
        clarifications=clarifications,
    )
    try:
        try:
            output = tools.call(name, execution_arguments, context=context, metric_bindings=execution_bindings)
        except ToolError as exc:
            if name == "query_readonly" and exc.code == "approval_required":
                # Park only a query that passes every remaining server check;
                # a broken or unauthorized query keeps its own error code.
                _verified_read(execution_arguments, execution_bindings, context=context)
                raise ApprovalRequiredError(
                    _pending_call(execution_arguments, execution_bindings)
                ) from exc
            if name == "query_readonly" and exc.code == "limit_reached":
                raise _row_limit_error(exc) from exc
            raise
        if declared_metric_ids and isinstance(output, Mapping) and type(output.get("result_id")) is str:
            output = {
                **dict(output),
                "verified_metrics": [
                    {"metric_id": binding.metric_id.removeprefix("metric."), "result_position": binding.result_position}
                    for binding in execution_bindings
                ],
            }
        return output
    except DatabaseDriverError as exc:
        raise _driver_error(exc) from exc


def check_clarification(clarifications: ClarificationReading | None, metric_ids) -> None:
    """Raise when the phrase table says the declared metrics cannot run as asked."""

    if clarifications is None or not metric_ids:
        return
    verdict = check_declaration(clarifications, metric_ids)
    if verdict is None:
        return
    if verdict.kind == "unsupported" and verdict.value is not None:
        raise ClarificationValueUnsupportedError(verdict.rule, verdict.value)
    if verdict.kind == "contradicts":
        raise MetricContradictsQuestionError(verdict.rule)
    raise ClarificationRequiredError(verdict.rule)


def _resolve_call(tools, name, arguments, *, context, metric_bindings=(), request_time_window=None, clarifications=None):
    """Declaration and binding checks shared by call_tool and pending-call preparation."""

    execution_arguments = _bind_tenant_equality_params(arguments, context)
    execution_bindings = tuple(metric_bindings)
    declared_metric_ids: tuple[str, ...] = ()
    if name == "query_readonly":
        try:
            declaration = resolve_query_declaration(
                execution_arguments,
                catalog=getattr(tools, "catalog", None),
                request_time_window=request_time_window,
                prebound=execution_bindings,
            )
        except MetricDeclarationError as exc:
            raise ToolError(exc.code, exc.message) from exc
        execution_arguments = declaration.arguments
        execution_bindings = declaration.bindings
        declared_metric_ids = declaration.declared_metric_ids
        check_clarification(clarifications, declared_metric_ids)
        if any(binding.metric_id.removeprefix("metric.") == "net_fen" for binding in execution_bindings if isinstance(binding, MetricBinding)) and declared_metric_ids:
            if not _net_fen_request_matches(execution_arguments, execution_bindings):
                raise ToolError(
                    "evidence_validation_failed",
                    "a declared net_fen query must read only orders/refunds and filter orders.created_at by the declared window",
                )
    if name == "query_readonly" and execution_bindings:
        execution_bindings = _bind_metric_result_positions(execution_arguments, execution_bindings, context=context)
        if execution_bindings is None:
            raise ToolError(
                "evidence_validation_failed",
                "query projection or filters do not match the server-bound metric semantics; customer_id grouped results require a composite tenant_id and customer_id join whose ON clause has only those two equalities",
            )
    return execution_arguments, execution_bindings, declared_metric_ids


def prepare_pending_call(tools, arguments, *, context, request_time_window=None) -> dict[str, object]:
    """Run every server check on a query_readonly call without executing it.

    Returns the canonical call an approval binds to.  Used to materialize a
    WAITING_APPROVAL state; the live path reaches the same result in call_tool.
    """

    execution_arguments, execution_bindings, _ = _resolve_call(
        tools,
        "query_readonly",
        arguments,
        context=context,
        request_time_window=request_time_window,
    )
    _verified_read(execution_arguments, execution_bindings, context=context)
    return _pending_call(execution_arguments, execution_bindings)


def _pending_call(arguments, bindings) -> dict[str, object]:
    """Canonical approvable call: exactly what will run after approval."""

    metric_bindings = [binding for binding in bindings if isinstance(binding, MetricBinding)]
    window = dict(metric_bindings[0].time_window) if metric_bindings else None
    return {
        "tool": "query_readonly",
        "sql": str(arguments["sql"]),
        "params": {str(key): value for key, value in dict(arguments["params"]).items()},
        "metrics": [binding.metric_id.removeprefix("metric.") for binding in metric_bindings],
        "time_window": window,
    }


def _verified_read(arguments, bindings, *, context) -> tuple[str, tuple[object, ...]]:
    """The checks query_readonly makes after its sensitive-field check, without executing."""

    if not isinstance(arguments, Mapping):
        raise ToolError("invalid_arguments", "tool arguments must be an object")
    sql = arguments.get("sql")
    if type(sql) is not str or not 1 <= len(sql.strip()) <= 4000:
        raise ToolError("invalid_argument", "sql length is outside the allowed range")
    params = _params(arguments.get("params"))
    try:
        statement = parse_readonly_select(sql)
    except SQLPolicyError as exc:
        raise ToolError(exc.code, _controlled_error_detail(exc, exc.code)) from exc
    if any(table not in ALLOWED_TABLE_COLUMNS for table in statement.referenced_tables):
        raise ToolError("table_not_allowed", "query references a table outside the server allowlist")
    if any(
        isinstance(binding, MetricBinding) and binding.metric_id.removeprefix("metric.") == "net_fen"
        for binding in bindings
    ):
        raise ToolError("invalid_binding", "net_fen controlled plan accepts only orders and refunds requests")
    try:
        render_scoped_select(statement, tenant_id=context.tenant_id, input_params=params)
    except (SQLPolicyError, GuardedQueryError) as exc:
        raise ToolError(exc.code, _controlled_error_detail(exc, exc.code)) from exc
    return sql, params


def execute_approved_query(tools, pending_call: Mapping[str, object], *, context: ExecutionContext) -> ResultEvidence:
    """Execute exactly one approved pending call for the requester.

    This is not a general permission switch: the caller passes the one call
    whose digest the approval bound, and this function re-runs every server
    check except the sensitive-field gate that the approval satisfied.
    Nothing is stored in the tool facade; the evidence is returned.
    """

    if not isinstance(context, ExecutionContext):
        raise ToolError("unauthorized", "approved execution requires a server-created context")
    if not isinstance(pending_call, Mapping) or pending_call.get("tool") != "query_readonly":
        raise ToolError("approval_action_invalid", "the approved action is not a query_readonly call")
    arguments: dict[str, object] = {"sql": pending_call.get("sql"), "params": pending_call.get("params")}
    metrics = pending_call.get("metrics")
    if metrics:
        arguments["metrics"] = list(metrics) if isinstance(metrics, list) else metrics
        arguments["time_window"] = pending_call.get("time_window")
    try:
        declaration = resolve_query_declaration(arguments, catalog=getattr(tools, "catalog", None))
    except MetricDeclarationError as exc:
        raise ToolError(exc.code, exc.message) from exc
    bindings = declaration.bindings
    if bindings:
        bindings = _bind_metric_result_positions(declaration.arguments, bindings, context=context)
        if bindings is None:
            raise ToolError("evidence_validation_failed", "approved query no longer matches its metric semantics")
    sql, params = _verified_read(declaration.arguments, bindings, context=context)
    try:
        result = tools.executor.execute(sql, context=context, params=params, metric_bindings=bindings)
    except GuardedQueryError as exc:
        if exc.code == "limit_reached":
            raise ToolError(RESULT_ROW_LIMIT_CODE, "query returned more than the 100 row limit") from exc
        raise ToolError(exc.code, _controlled_error_detail(exc, exc.code)) from exc
    except SQLPolicyError as exc:
        raise ToolError(exc.code, _controlled_error_detail(exc, exc.code)) from exc
    except DatabaseDriverError as exc:
        raise _driver_error(exc) from exc
    return result.evidence


def _bind_tenant_equality_params(arguments, context):
    """Bind explicit tenant equality filters to the server-owned identity."""

    if (
        not isinstance(arguments, Mapping)
        or type(arguments.get("sql")) is not str
        or not isinstance(arguments.get("params"), Mapping)
        or type(getattr(context, "tenant_id", None)) is not str
        or not context.tenant_id.strip()
    ):
        return arguments
    params = arguments["params"]
    keys = list(params)
    if (
        any(type(key) is not str or not key.isdecimal() or str(int(key)) != key for key in keys)
        or sorted(int(key) for key in keys) != list(range(len(keys)))
        or any(value is not None and type(value) not in {str, int, float, bool} for value in params.values())
    ):
        return arguments
    try:
        statement = parse_readonly_select(arguments["sql"])
    except SQLPolicyError:
        return arguments

    indexes: set[int] = set()
    if statement.where is not None:
        _collect_tenant_equality_parameters(statement.where, indexes)
    for join in statement.joins:
        _collect_tenant_equality_parameters(join.condition, indexes)
    if not indexes or any(index >= len(params) for index in indexes):
        return arguments

    bound_arguments = dict(arguments)
    bound_params = dict(params)
    for index in indexes:
        bound_params[str(index)] = context.tenant_id
    bound_arguments["params"] = bound_params
    return bound_arguments


def _collect_tenant_equality_parameters(condition, indexes: set[int]) -> None:
    if isinstance(condition, Comparison) and condition.operator == "=":
        pairs = ((condition.left, condition.right), (condition.right, condition.left))
        for column, value in pairs:
            if (
                isinstance(column, ColumnRef)
                and column.name.casefold() == "tenant_id"
                and isinstance(value, ParameterRef)
            ):
                indexes.add(value.index)
    elif isinstance(condition, BooleanExpression):
        _collect_tenant_equality_parameters(condition.left, indexes)
        _collect_tenant_equality_parameters(condition.right, indexes)
    elif isinstance(condition, NotExpression):
        _collect_tenant_equality_parameters(condition.operand, indexes)


def _net_fen_request_matches(arguments, bindings) -> bool:
    """A declared net_fen query must name the plan's tables and its declared order window.

    The server still replaces the query with its two-aggregate plan; this keeps
    the model's SQL consistent with what it declared.  Parser errors are left to
    query_readonly so they keep their repairable codes.
    """

    binding = next(item for item in bindings if item.metric_id.removeprefix("metric.") == "net_fen")
    if not isinstance(arguments, Mapping):
        return False
    sql = arguments.get("sql")
    params = arguments.get("params")
    if type(sql) is not str or not isinstance(params, Mapping):
        return False
    keys = list(params)
    if (
        any(type(key) is not str or not key.isdecimal() or str(int(key)) != key for key in keys)
        or sorted(int(key) for key in keys) != list(range(len(keys)))
    ):
        return False
    try:
        statement = parse_readonly_select(sql)
    except SQLPolicyError:
        return True
    tables = set(statement.referenced_tables)
    if "orders" not in tables or not tables <= {"orders", "refunds"}:
        return False
    terms = _and_comparisons(statement.where)
    parameter_values = tuple(params[str(index)] for index in range(len(params)))
    return terms is not None and _has_bound_window(terms, parameter_values, binding.time_window, statement)


def _bind_metric_result_positions(arguments, metric_bindings, *, context):
    """Bind only SQL projection aliases whose expression and filters match a trusted metric."""

    bindings = tuple(metric_bindings)
    if not bindings or not isinstance(arguments, Mapping):
        return bindings
    sql = arguments.get("sql")
    params = arguments.get("params")
    if type(sql) is not str or not isinstance(params, Mapping):
        return None
    if any(not isinstance(binding, MetricBinding) for binding in bindings):
        return None
    if not any(binding.metric_id.removeprefix("metric.") in {"gross_fen", "paid_count"} for binding in bindings):
        return bindings
    keys = list(params)
    if (
        any(type(key) is not str or not key.isdecimal() or str(int(key)) != key for key in keys)
        or sorted(int(key) for key in keys) != list(range(len(keys)))
    ):
        return None
    try:
        statement = parse_readonly_select(sql)
    except SQLPolicyError:
        return bindings
    parameter_values = tuple(params[str(index)] for index in range(len(params)))
    terms = _and_comparisons(statement.where)
    if terms is None or not _metric_scope_filters_match(terms, parameter_values, statement, bindings, context.tenant_id):
        return None

    by_metric = {binding.metric_id.removeprefix("metric."): binding for binding in bindings}
    output = list(bindings)
    tables = set(statement.referenced_tables)
    if "customers" in tables:
        if not _has_trusted_customer_join(statement):
            return None
    elif statement.joins:
        return None
    if any(
        isinstance(expression, ColumnRef) and expression.name.casefold() == "customer_id"
        for expression in statement.group_by
    ) and not _is_customer_aggregate(statement):
        return None

    for metric_id, binding in by_metric.items():
        if metric_id not in {"gross_fen", "paid_count"}:
            continue
        if not _has_bound_window(terms, parameter_values, binding.time_window, statement):
            return None
        if metric_id == "gross_fen" and not _is_customer_aggregate(statement) and any(
            _resolved_table(column, statement) != "orders"
            for item in statement.projection
            for column in _columns_in(item.expression)
            if column.name.casefold() == "amount_fen"
        ):
            return None
        matches = [
            item for item in statement.projection
            if (_is_count_star(item.expression) if metric_id == "paid_count" else _is_gross_sum(item.expression, statement))
        ]
        if len(matches) != 1 or not matches[0].alias or not matches[0].alias.islower():
            return None
        index = next(i for i, item in enumerate(output) if item.metric_id.removeprefix("metric.") == metric_id)
        # B3e: a GROUP BY makes every row a per-group value, so the binding is
        # a rowset even when one row comes back (ORDER BY ... LIMIT 1).
        output[index] = replace(binding, result_position=matches[0].alias, grouped=bool(statement.group_by))
    return tuple(output)


def _and_comparisons(condition):
    if isinstance(condition, Comparison):
        return (condition,)
    if isinstance(condition, BooleanExpression) and condition.operator == "AND":
        left = _and_comparisons(condition.left)
        right = _and_comparisons(condition.right)
        return None if left is None or right is None else left + right
    return None


def _expression_value(expression, params):
    if isinstance(expression, LiteralValue):
        return expression.value
    if isinstance(expression, ParameterRef) and expression.index < len(params):
        return params[expression.index]
    return object()


def _column_table(column: ColumnRef, statement: SelectStatement) -> str | None:
    if column.qualifier is None:
        return statement.from_table.name if column.name.casefold() in {"status", "created_at", "amount_fen"} and "orders" in statement.referenced_tables else None
    aliases = {
        table.alias or table.name: table.name
        for table in (statement.from_table,) + tuple(join.table for join in statement.joins)
    }
    return aliases.get(column.qualifier)


def _metric_scope_filters_match(terms, params, statement: SelectStatement, bindings, tenant_id: str) -> bool:
    bound_windows = [binding.time_window for binding in bindings if binding.metric_id.removeprefix("metric.") in {"gross_fen", "paid_count"}]
    if not bound_windows or any(window != bound_windows[0] for window in bound_windows[1:]):
        return False
    has_paid = False
    has_start = False
    has_end = False
    for term in terms:
        matched = False
        for column, value, operator in (
            (term.left, term.right, term.operator),
            (term.right, term.left, {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(term.operator, term.operator)),
        ):
            if not isinstance(column, ColumnRef):
                continue
            field = column.name.casefold()
            if _resolved_table(column, statement) != "orders":
                continue
            actual = _expression_value(value, params)
            if field == "status" and operator == "=" and actual == "paid":
                has_paid = True
                matched = True
            elif field == "created_at" and operator == ">=" and actual == bound_windows[0].get("start"):
                has_start = True
                matched = True
            elif field == "created_at" and operator == "<" and actual == bound_windows[0].get("end"):
                has_end = True
                matched = True
            elif field == "tenant_id" and operator == "=" and actual == tenant_id:
                matched = True
        if not matched:
            return False
    return has_paid and has_start and has_end


def _has_bound_window(terms, params, time_window, statement: SelectStatement) -> bool:
    if not isinstance(time_window, Mapping) or time_window.get("timezone") != "UTC":
        return False
    found_start = False
    found_end = False
    for term in terms:
        for column, value, operator in (
            (term.left, term.right, term.operator),
            (term.right, term.left, {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(term.operator, term.operator)),
        ):
            if (
                not isinstance(column, ColumnRef)
                or column.name.casefold() != "created_at"
                or _resolved_table(column, statement) != "orders"
            ):
                continue
            actual = _expression_value(value, params)
            if operator == ">=" and actual == time_window.get("start"):
                found_start = True
            if operator == "<" and actual == time_window.get("end"):
                found_end = True
    return found_start and found_end


def _resolved_table(column: ColumnRef, statement: SelectStatement) -> str | None:
    if column.qualifier is None:
        return "orders" if "orders" in statement.referenced_tables else None
    aliases = {
        table.alias or table.name: table.name
        for table in (statement.from_table,) + tuple(join.table for join in statement.joins)
    }
    return aliases.get(column.qualifier)


def _columns_in(expression):
    if isinstance(expression, ColumnRef):
        return (expression,)
    if isinstance(expression, FunctionCall):
        return tuple(column for child in expression.arguments for column in _columns_in(child))
    return ()


def _is_count_star(expression) -> bool:
    candidate = expression
    if isinstance(candidate, FunctionCall) and candidate.name == "COALESCE":
        if len(candidate.arguments) != 2:
            return False
        left, right = candidate.arguments
        if isinstance(left, FunctionCall) and isinstance(right, LiteralValue) and right.value == 0:
            candidate = left
        elif isinstance(right, FunctionCall) and isinstance(left, LiteralValue) and left.value == 0:
            candidate = right
        else:
            return False
    return (
        isinstance(candidate, FunctionCall)
        and candidate.name == "COUNT"
        and len(candidate.arguments) == 1
        and isinstance(candidate.arguments[0], Star)
    )


def _is_gross_sum(expression, statement: SelectStatement) -> bool:
    candidate = expression
    if isinstance(candidate, FunctionCall) and candidate.name == "COALESCE":
        left, right = candidate.arguments
        if isinstance(left, FunctionCall) and left.name == "SUM" and isinstance(right, LiteralValue) and right.value == 0:
            candidate = left
        elif isinstance(right, FunctionCall) and right.name == "SUM" and isinstance(left, LiteralValue) and left.value == 0:
            candidate = right
        else:
            return False
    return (
        isinstance(candidate, FunctionCall)
        and candidate.name == "SUM"
        and len(candidate.arguments) == 1
        and isinstance(candidate.arguments[0], ColumnRef)
        and candidate.arguments[0].name.casefold() == "amount_fen"
        and _resolved_table(candidate.arguments[0], statement) == "orders"
    )


def _is_customer_aggregate(statement: SelectStatement) -> bool:
    if "customers" not in statement.referenced_tables:
        return False
    if not any(
        isinstance(expression, ColumnRef) and expression.name.casefold() == "customer_id"
        for expression in statement.group_by
    ):
        return False
    return _has_trusted_customer_join(statement)


def _has_trusted_customer_join(statement: SelectStatement) -> bool:
    if len(statement.joins) != 1 or statement.joins[0].table.name != "customers":
        return False
    conditions = _and_comparisons(statement.joins[0].condition)
    if conditions is None:
        return False
    pairs: set[frozenset[tuple[str | None, str]]] = set()
    for condition in conditions:
        if condition.operator == "=" and isinstance(condition.left, ColumnRef) and isinstance(condition.right, ColumnRef):
            pairs.add(frozenset(((condition.left.qualifier, condition.left.name.casefold()), (condition.right.qualifier, condition.right.name.casefold()))))
    aliases = {
        table.name: table.alias or table.name
        for table in (statement.from_table,) + tuple(join.table for join in statement.joins)
    }
    order_alias = aliases.get("orders")
    customer_alias = aliases.get("customers")
    expected = {
        frozenset(((order_alias, "tenant_id"), (customer_alias, "tenant_id"))),
        frozenset(((order_alias, "customer_id"), (customer_alias, "customer_id"))),
    }
    # B3e: exactly the two key equalities.  Any further ON condition narrows
    # the rows (one customer, or an orders filter) like an unbound WHERE term.
    return len(pairs) == len(conditions) and pairs == expected
