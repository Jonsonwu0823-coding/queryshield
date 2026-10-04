from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias


class SQLPolicyError(ValueError):
    """A query is outside the deliberately small read-only SQL subset."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


TokenKind = Literal[
    "word",
    "quoted_identifier",
    "number",
    "string",
    "parameter",
    "operator",
    "symbol",
    "eof",
]


@dataclass(frozen=True)
class _Token:
    kind: TokenKind
    text: str
    position: int


@dataclass(frozen=True)
class ColumnRef:
    name: str
    qualifier: str | None = None


@dataclass(frozen=True)
class LiteralValue:
    value: str | int | float | bool | None


@dataclass(frozen=True)
class ParameterRef:
    index: int


@dataclass(frozen=True)
class Star:
    pass


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: tuple[Expression, ...]


@dataclass(frozen=True)
class UnaryExpression:
    operator: str
    operand: Expression


@dataclass(frozen=True)
class BinaryExpression:
    operator: str
    left: Expression
    right: Expression


@dataclass(frozen=True)
class Comparison:
    operator: str
    left: Expression
    right: Expression


@dataclass(frozen=True)
class BooleanExpression:
    operator: Literal["AND", "OR"]
    left: Condition
    right: Condition


@dataclass(frozen=True)
class NotExpression:
    operand: Condition


Expression: TypeAlias = (
    ColumnRef
    | LiteralValue
    | ParameterRef
    | Star
    | FunctionCall
    | UnaryExpression
    | BinaryExpression
)
Condition: TypeAlias = Comparison | BooleanExpression | NotExpression


@dataclass(frozen=True)
class SelectItem:
    expression: Expression
    alias: str | None = None


@dataclass(frozen=True)
class TableRef:
    name: str
    alias: str | None = None


@dataclass(frozen=True)
class Join:
    table: TableRef
    condition: Condition


@dataclass(frozen=True)
class OrderItem:
    expression: Expression
    direction: Literal["ASC", "DESC"] = "ASC"


@dataclass(frozen=True)
class SelectStatement:
    """The parsed subset; it is an AST, not executable SQL."""

    projection: tuple[SelectItem, ...]
    from_table: TableRef
    joins: tuple[Join, ...]
    where: Condition | None
    group_by: tuple[Expression, ...]
    order_by: tuple[OrderItem, ...]
    limit: int | ParameterRef | None

    @property
    def referenced_tables(self) -> tuple[str, ...]:
        return (self.from_table.name,) + tuple(join.table.name for join in self.joins)

    @property
    def function_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for item in self.projection:
            _collect_function_names(item.expression, names)
        for expression in self.group_by:
            _collect_function_names(expression, names)
        for item in self.order_by:
            _collect_function_names(item.expression, names)
        return tuple(names)


ALLOWED_TABLES = frozenset({"customers", "orders", "refunds"})
ALLOWED_FUNCTIONS = frozenset({"SUM", "COUNT", "COALESCE"})
MAX_SQL_LENGTH = 4000
MAX_RESULT_ROWS = 100

_KEYWORDS = frozenset(
    {
        "SELECT",
        "FROM",
        "WHERE",
        "INNER",
        "JOIN",
        "ON",
        "GROUP",
        "BY",
        "ORDER",
        "LIMIT",
        "ASC",
        "DESC",
        "AS",
        "AND",
        "OR",
        "NOT",
        "IS",
        "NULL",
        "TRUE",
        "FALSE",
        "SUM",
        "COUNT",
        "COALESCE",
        "WITH",
        "UNION",
        "INTERSECT",
        "EXCEPT",
        "FOR",
        "UPDATE",
        "INSERT",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "DISTINCT",
        "HAVING",
        "LEFT",
        "RIGHT",
        "FULL",
        "CROSS",
        "OUTER",
        "IN",
        "LIKE",
        "BETWEEN",
    }
)
_MULTI_OPERATORS = frozenset({"<=", ">=", "<>", "!=", "::"})
_SYMBOLS = frozenset({"(", ")", ",", ".", "*", ";"})
_SINGLE_OPERATORS = frozenset({"=", "<", ">", "+", "-"})


def parse_readonly_select(sql: str) -> SelectStatement:
    """Parse and policy-check one supported SELECT before database execution."""
    tokens = _tokenize(sql)
    return _Parser(tokens).parse()


def _tokenize(sql: str) -> tuple[_Token, ...]:
    if type(sql) is not str:
        raise SQLPolicyError("invalid_sql", "SQL must be a string")
    if not sql.strip():
        raise SQLPolicyError("invalid_sql", "SQL must not be blank")
    if len(sql) > MAX_SQL_LENGTH:
        raise SQLPolicyError("invalid_sql", "SQL exceeds the 4000 character limit")

    tokens: list[_Token] = []
    index = 0
    while index < len(sql):
        char = sql[index]
        if char.isspace():
            index += 1
            continue

        if sql.startswith("--", index) or sql.startswith("/*", index):
            raise SQLPolicyError("unsupported_syntax", "SQL comments are not supported")

        if char.isalpha() or char == "_":
            end = index + 1
            while end < len(sql) and (sql[end].isalnum() or sql[end] == "_"):
                end += 1
            tokens.append(_Token("word", sql[index:end], index))
            index = end
            continue

        if char.isdigit():
            end = index + 1
            while end < len(sql) and sql[end].isdigit():
                end += 1
            if end < len(sql) and sql[end] == ".":
                end += 1
                fraction_start = end
                while end < len(sql) and sql[end].isdigit():
                    end += 1
                if end == fraction_start:
                    raise SQLPolicyError("invalid_sql", "invalid numeric literal")
            tokens.append(_Token("number", sql[index:end], index))
            index = end
            continue

        if char == "'":
            value, end = _read_string(sql, index)
            tokens.append(_Token("string", value, index))
            index = end
            continue

        if char == '"':
            value, end = _read_quoted_identifier(sql, index)
            tokens.append(_Token("quoted_identifier", value, index))
            index = end
            continue

        if sql.startswith("%s", index):
            tokens.append(_Token("parameter", "%s", index))
            index += 2
            continue
        if char == "?":
            tokens.append(_Token("parameter", "?", index))
            index += 1
            continue

        operator = sql[index : index + 2]
        if operator in _MULTI_OPERATORS:
            if operator == "::":
                raise SQLPolicyError("unsupported_syntax", "casts are not supported")
            tokens.append(_Token("operator", operator, index))
            index += 2
            continue
        if char in _SINGLE_OPERATORS:
            tokens.append(_Token("operator", char, index))
            index += 1
            continue
        if char in _SYMBOLS:
            tokens.append(_Token("symbol", char, index))
            index += 1
            continue

        raise SQLPolicyError("invalid_sql", "SQL contains an unsupported character")

    tokens.append(_Token("eof", "", len(sql)))
    return tuple(tokens)


def _read_string(sql: str, start: int) -> tuple[str, int]:
    index = start + 1
    value: list[str] = []
    while index < len(sql):
        char = sql[index]
        if char == "'":
            if index + 1 < len(sql) and sql[index + 1] == "'":
                value.append("'")
                index += 2
                continue
            return "".join(value), index + 1
        value.append(char)
        index += 1
    raise SQLPolicyError("invalid_sql", "unterminated string literal")


def _read_quoted_identifier(sql: str, start: int) -> tuple[str, int]:
    index = start + 1
    value: list[str] = []
    while index < len(sql):
        char = sql[index]
        if char == '"':
            if index + 1 < len(sql) and sql[index + 1] == '"':
                value.append('"')
                index += 2
                continue
            if not value:
                raise SQLPolicyError("invalid_sql", "quoted identifier must not be blank")
            return "".join(value), index + 1
        value.append(char)
        index += 1
    raise SQLPolicyError("invalid_sql", "unterminated quoted identifier")


class _Parser:
    def __init__(self, tokens: Sequence[_Token]) -> None:
        self._tokens = tokens
        self._index = 0
        self._parameter_index = 0
        self._aliases: set[str] = set()

    def parse(self) -> SelectStatement:
        if self._match_word("WITH"):
            raise SQLPolicyError("unsupported_syntax", "CTE queries are not supported")
        if not self._match_word("SELECT"):
            raise SQLPolicyError("statement_not_allowed", "only SELECT is allowed")

        projection = self._parse_projection()
        self._expect_word("FROM")
        from_table = self._parse_table()
        joins = self._parse_joins()
        self._validate_effective_aliases(from_table, joins)

        where = self._parse_condition() if self._match_word("WHERE") else None
        group_by = self._parse_group_by() if self._match_word("GROUP") else ()
        order_by = self._parse_order_by() if self._match_word("ORDER") else ()
        limit = self._parse_limit() if self._match_word("LIMIT") else None

        if self._match_symbol(";"):
            if not self._at("eof"):
                raise SQLPolicyError("multiple_statements", "multiple SQL statements are not allowed")
        elif not self._at("eof"):
            if self._current().text.upper() in {"UNION", "INTERSECT", "EXCEPT"}:
                raise SQLPolicyError("unsupported_syntax", "set operations are not supported")
            raise SQLPolicyError("unsupported_syntax", "unexpected SQL after SELECT")

        return SelectStatement(
            projection=tuple(projection),
            from_table=from_table,
            joins=tuple(joins),
            where=where,
            group_by=tuple(group_by),
            order_by=tuple(order_by),
            limit=limit,
        )

    def _validate_effective_aliases(self, from_table: TableRef, joins: Sequence[Join]) -> None:
        effective_names = [from_table.alias or from_table.name]
        effective_names.extend(join.table.alias or join.table.name for join in joins)
        if len(effective_names) != len(set(effective_names)):
            raise SQLPolicyError("invalid_sql", "table aliases must be unique")

    def _parse_projection(self) -> list[SelectItem]:
        items: list[SelectItem] = []
        while True:
            expression = self._parse_value_expression()
            alias: str | None = None
            if self._match_word("AS"):
                alias = self._parse_identifier("column alias")
            elif self._current().kind in {"word", "quoted_identifier"} and not self._is_keyword(
                self._current()
            ):
                alias = self._parse_identifier("column alias")
            items.append(SelectItem(expression=expression, alias=alias))
            if not self._match_symbol(","):
                return items

    def _parse_table(self) -> TableRef:
        name = self._parse_identifier("table name")
        if self._match_symbol("."):
            raise SQLPolicyError("table_not_allowed", "schema-qualified tables are not supported")
        if name not in ALLOWED_TABLES:
            raise SQLPolicyError("table_not_allowed", "table is not in the allowlist")

        alias: str | None = None
        if self._match_word("AS"):
            alias = self._parse_identifier("table alias")
        elif self._current().kind in {"word", "quoted_identifier"} and not self._is_keyword(
            self._current()
        ):
            alias = self._parse_identifier("table alias")
        if alias is not None:
            if alias in self._aliases:
                raise SQLPolicyError("invalid_sql", "table aliases must be unique")
            self._aliases.add(alias)
        return TableRef(name=name, alias=alias)

    def _parse_joins(self) -> list[Join]:
        joins: list[Join] = []
        while self._match_word("INNER") or self._match_word("JOIN"):
            if self._previous().text.upper() == "INNER":
                self._expect_word("JOIN")
            table = self._parse_table()
            self._expect_word("ON")
            joins.append(Join(table=table, condition=self._parse_condition()))

        if self._current().text.upper() in {"LEFT", "RIGHT", "FULL", "CROSS", "OUTER"}:
            raise SQLPolicyError("unsupported_syntax", "only INNER JOIN is supported")
        return joins

    def _parse_condition(self) -> Condition:
        return self._parse_or()

    def _parse_or(self) -> Condition:
        expression = self._parse_and()
        while self._match_word("OR"):
            expression = BooleanExpression("OR", expression, self._parse_and())
        return expression

    def _parse_and(self) -> Condition:
        expression = self._parse_not()
        while self._match_word("AND"):
            expression = BooleanExpression("AND", expression, self._parse_not())
        return expression

    def _parse_not(self) -> Condition:
        if self._match_word("NOT"):
            return NotExpression(self._parse_not())
        if self._match_symbol("("):
            expression = self._parse_condition()
            self._expect_symbol(")")
            return expression

        left = self._parse_value_expression()
        if self._current().kind != "operator" and not self._at_word("IS"):
            raise SQLPolicyError("unsupported_syntax", "WHERE and JOIN conditions need comparisons")
        if self._current().kind == "operator":
            operator = self._advance().text
            right = self._parse_value_expression()
            return Comparison(operator=operator, left=left, right=right)

        self._expect_word("IS")
        is_not = self._match_word("NOT")
        self._expect_word("NULL")
        return Comparison(
            operator="IS NOT NULL" if is_not else "IS NULL",
            left=left,
            right=LiteralValue(None),
        )

    def _parse_value_expression(self) -> Expression:
        expression = self._parse_primary()
        while self._current().text in {"+", "-"}:
            operator = self._advance().text
            expression = BinaryExpression(
                operator=operator,
                left=expression,
                right=self._parse_primary(),
            )
        return expression

    def _parse_primary(self) -> Expression:
        token = self._current()
        if self._match_symbol("("):
            expression = self._parse_value_expression()
            self._expect_symbol(")")
            return expression
        if self._match_symbol("*"):
            return Star()
        if token.kind == "parameter":
            self._advance()
            parameter = ParameterRef(self._parameter_index)
            self._parameter_index += 1
            return parameter
        if token.kind == "number":
            self._advance()
            return LiteralValue(float(token.text) if "." in token.text else int(token.text))
        if token.kind == "string":
            self._advance()
            return LiteralValue(token.text)
        if token.kind in {"word", "quoted_identifier"}:
            if token.kind == "word" and token.text.upper() in {"NULL", "TRUE", "FALSE"}:
                self._advance()
                values: dict[str, object] = {"NULL": None, "TRUE": True, "FALSE": False}
                return LiteralValue(values[token.text.upper()])
            if self._lookahead_symbol("("):
                return self._parse_function()
            return self._parse_column()
        raise SQLPolicyError("unsupported_syntax", "expected a SQL expression")

    def _parse_function(self) -> FunctionCall:
        token = self._current()
        if token.kind not in {"word", "quoted_identifier"}:
            raise SQLPolicyError("invalid_sql", "expected function name")
        self._advance()
        name = token.text.upper()
        if name not in ALLOWED_FUNCTIONS:
            raise SQLPolicyError("function_not_allowed", "function is not in the allowlist")
        self._expect_symbol("(")

        arguments: list[Expression] = []
        if self._match_symbol("*"):
            arguments.append(Star())
        elif not self._at_symbol(")"):
            while True:
                arguments.append(self._parse_value_expression())
                if not self._match_symbol(","):
                    break
        self._expect_symbol(")")

        expected_arguments = {"SUM": 1, "COUNT": 1, "COALESCE": 2}[name]
        if len(arguments) != expected_arguments:
            raise SQLPolicyError("invalid_sql", f"{name} received the wrong number of arguments")
        if name in {"SUM", "COUNT"} and len(arguments) == 1 and isinstance(arguments[0], Star):
            if name != "COUNT":
                raise SQLPolicyError("invalid_sql", "SUM does not accept wildcard input")
        return FunctionCall(name=name, arguments=tuple(arguments))

    def _parse_column(self) -> ColumnRef:
        first = self._parse_identifier("column name")
        if self._match_symbol("."):
            second = self._parse_identifier("qualified column name")
            return ColumnRef(name=second, qualifier=first)
        return ColumnRef(name=first)

    def _parse_group_by(self) -> tuple[Expression, ...]:
        self._expect_word("BY")
        expressions = [self._parse_value_expression()]
        while self._match_symbol(","):
            expressions.append(self._parse_value_expression())
        return tuple(expressions)

    def _parse_order_by(self) -> tuple[OrderItem, ...]:
        self._expect_word("BY")
        items: list[OrderItem] = []
        while True:
            expression = self._parse_value_expression()
            direction: Literal["ASC", "DESC"] = "DESC" if self._match_word("DESC") else "ASC"
            if direction == "ASC":
                self._match_word("ASC")
            items.append(OrderItem(expression=expression, direction=direction))
            if not self._match_symbol(","):
                return tuple(items)

    def _parse_limit(self) -> int | ParameterRef:
        token = self._current()
        if token.kind == "number":
            self._advance()
            value = int(token.text) if token.text.isdigit() else -1
            if not 0 <= value <= MAX_RESULT_ROWS:
                raise SQLPolicyError("limit_exceeded", "LIMIT must be between 0 and 100")
            return value
        if token.kind == "parameter":
            self._advance()
            parameter = ParameterRef(self._parameter_index)
            self._parameter_index += 1
            return parameter
        raise SQLPolicyError("invalid_sql", "LIMIT must be an integer or parameter")

    def _parse_identifier(self, field: str) -> str:
        token = self._current()
        if token.kind == "quoted_identifier":
            self._advance()
            return token.text
        if token.kind != "word" or self._is_keyword(token):
            raise SQLPolicyError("invalid_sql", f"expected {field}")
        self._advance()
        return token.text

    def _expect_word(self, word: str) -> None:
        if not self._match_word(word):
            raise SQLPolicyError("invalid_sql", f"expected {word}")

    def _expect_symbol(self, symbol: str) -> None:
        if not self._match_symbol(symbol):
            raise SQLPolicyError("invalid_sql", f"expected {symbol}")

    def _match_word(self, word: str) -> bool:
        if self._at_word(word):
            self._advance()
            return True
        return False

    def _match_symbol(self, symbol: str) -> bool:
        if self._at_symbol(symbol):
            self._advance()
            return True
        return False

    def _at_word(self, word: str) -> bool:
        token = self._current()
        return token.kind == "word" and token.text.upper() == word

    def _at_symbol(self, symbol: str) -> bool:
        return self._current().text == symbol

    def _lookahead_symbol(self, symbol: str) -> bool:
        return self._tokens[self._index + 1].text == symbol

    def _is_keyword(self, token: _Token) -> bool:
        return token.kind == "word" and token.text.upper() in _KEYWORDS

    def _at(self, kind: TokenKind) -> bool:
        return self._current().kind == kind

    def _advance(self) -> _Token:
        token = self._current()
        if not self._at("eof"):
            self._index += 1
        return token

    def _current(self) -> _Token:
        return self._tokens[self._index]

    def _previous(self) -> _Token:
        return self._tokens[self._index - 1]


def _collect_function_names(expression: Expression, names: list[str]) -> None:
    if isinstance(expression, FunctionCall):
        names.append(expression.name)
        for argument in expression.arguments:
            _collect_function_names(argument, names)
    elif isinstance(expression, UnaryExpression):
        _collect_function_names(expression.operand, names)
    elif isinstance(expression, BinaryExpression):
        _collect_function_names(expression.left, names)
        _collect_function_names(expression.right, names)
