"""B4d: POST /query-proposals applies the same sensitive-field rule as /queries.

A requester must not read ``customers.name`` through the single-proposal entry;
the approver may.  The rule has one source (``check_sensitive_access``).  It is
applied to every SQL that parses; one that does not parse is passed on to the
executor, which rejects it with the same function as before, so policy errors
keep their status, their code and the executor's attempt (the W05 frozen case
``security-mutating-sql-rejected`` relies on that).

Two counters are kept apart: ``execute_calls`` (the executor was asked) and
``statements`` (a statement reached the database connection).
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

from queryshield.agent import ExecutionContext
from queryshield.api import main as api_main
from queryshield.api.main import app, get_guarded_executor
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.tools import semantic
from queryshield.tools.semantic import ControlledTools, ToolError

REQUESTER_TOKEN = "b4d-requester-token"
APPROVER_TOKEN = "b4d-approver-token"
APPROVAL_MESSAGE = "包含客户姓名的查询要通过 /queries 发起，由同租户审批人批准"

NAME_SQL = "SELECT c.customer_id, c.name FROM customers AS c ORDER BY c.customer_id LIMIT 3"
ORDERS_SQL = "SELECT o.order_id, o.amount_fen FROM orders AS o ORDER BY o.order_id"


class _Cursor:
    def __init__(self, rows: list[dict[str, object]], log: list[tuple[str, tuple[object, ...]]]) -> None:
        self.rows = rows
        self.log = log

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        self.log.append((sql, params))

    def fetchmany(self, size: int) -> list[dict[str, object]]:
        return self.rows[:size]


class _Connection:
    def __init__(self, rows: list[dict[str, object]], log: list[tuple[str, tuple[object, ...]]]) -> None:
        self.rows = rows
        self.log = log

    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def cursor(self, *, row_factory: object) -> _Cursor:
        return _Cursor(self.rows, self.log)


class _RecordingExecutor(GuardedQueryExecutor):
    """The real guarded executor over a stub connection, with two counters."""

    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.execute_calls = 0
        self.statements: list[tuple[str, tuple[object, ...]]] = []
        super().__init__(connect=lambda: _Connection(rows or [], self.statements))

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.execute_calls += 1
        return super().execute(*args, **kwargs)


@pytest.fixture()
def tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", REQUESTER_TOKEN)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_APPROVER", APPROVER_TOKEN)
    yield
    app.dependency_overrides.clear()


def _proposal(sql: str, params: dict[str, object] | None = None) -> dict[str, str]:
    return {
        "proposal": json.dumps(
            {"type": "tool_call", "name": "query_readonly", "arguments": {"sql": sql, "params": params or {}}},
            ensure_ascii=False,
        )
    }


def _post(token: str, sql: str, params: dict[str, object] | None = None, *, executor: GuardedQueryExecutor) -> Any:
    app.dependency_overrides[get_guarded_executor] = lambda: executor
    with TestClient(app) as client:
        return client.post(
            "/query-proposals",
            headers={"Authorization": f"Bearer {token}"},
            json=_proposal(sql, params),
        )


def _assert_approval_required(response: Any, executor: _RecordingExecutor) -> None:
    assert response.status_code == 403
    body = response.json()
    assert body["error"]["code"] == "approval_required"
    assert body["error"]["message"] == APPROVAL_MESSAGE
    assert "rows" not in body and "result" not in body
    assert executor.execute_calls == 0
    assert executor.statements == []


NAME_ROWS = [{"customer_id": "c1", "name": "甲"}, {"customer_id": "c2", "name": "乙"}]


def test_requester_cannot_read_customer_names_through_the_proposal_entry(tokens) -> None:
    executor = _RecordingExecutor(NAME_ROWS)
    _assert_approval_required(_post(REQUESTER_TOKEN, NAME_SQL, executor=executor), executor)


def test_requester_select_star_from_customers_is_refused(tokens) -> None:
    executor = _RecordingExecutor(NAME_ROWS)
    _assert_approval_required(_post(REQUESTER_TOKEN, "SELECT * FROM customers", executor=executor), executor)


def test_customer_alias_star_is_outside_the_sql_subset_and_fails_closed(tokens) -> None:
    executor = _RecordingExecutor(NAME_ROWS)
    response = _post(REQUESTER_TOKEN, "SELECT c.* FROM customers AS c", executor=executor)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "invalid_sql"
    # Passed on to the executor, which rejects it before any database access.
    assert executor.execute_calls == 1
    assert executor.statements == []


def test_requester_join_selecting_customer_name_is_refused(tokens) -> None:
    executor = _RecordingExecutor([{"order_id": "o1", "name": "甲"}])
    sql = (
        "SELECT o.order_id, c.name FROM orders AS o "
        "INNER JOIN customers AS c ON o.customer_id = c.customer_id ORDER BY o.order_id"
    )
    _assert_approval_required(_post(REQUESTER_TOKEN, sql, executor=executor), executor)


@pytest.mark.parametrize(
    ("sql", "params"),
    [
        ("SELECT c.customer_id FROM customers AS c WHERE c.name = %s", {"0": "甲"}),
        ("SELECT c.customer_id FROM customers AS c ORDER BY c.name", {}),
        ("SELECT name FROM customers", {}),
        ("SELECT customers.name FROM customers", {}),
    ],
    ids=["where-qualified", "order-by", "unqualified", "table-qualified"],
)
def test_every_position_of_customer_name_is_covered_like_the_queries_path(tokens, sql, params) -> None:
    executor = _RecordingExecutor(NAME_ROWS)
    _assert_approval_required(_post(REQUESTER_TOKEN, sql, params, executor=executor), executor)


def test_approver_may_run_the_same_name_query(tokens) -> None:
    executor = _RecordingExecutor(NAME_ROWS)
    response = _post(APPROVER_TOKEN, NAME_SQL, executor=executor)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCEEDED"
    assert body["rows"] == NAME_ROWS
    assert executor.execute_calls == 1
    assert len(executor.statements) == 1


def test_requester_query_without_customer_name_is_unchanged(tokens) -> None:
    rows = [{"order_id": "o1", "amount_fen": 10000}]
    executor = _RecordingExecutor(rows)
    response = _post(REQUESTER_TOKEN, ORDERS_SQL, executor=executor)

    assert response.status_code == 200
    assert response.json()["rows"] == rows
    assert executor.execute_calls == 1
    assert len(executor.statements) == 1


def test_policy_errors_keep_their_code_and_the_executor_attempt(tokens) -> None:
    executor = _RecordingExecutor()
    response = _post(REQUESTER_TOKEN, "UPDATE orders SET amount_fen = 1", executor=executor)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "statement_not_allowed"
    # As before B4d: the executor is asked once and rejects it; nothing reaches the database.
    assert executor.execute_calls == 1
    assert executor.statements == []


def test_other_tool_errors_are_not_swallowed_as_approval(tokens, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(context: ExecutionContext, statement: object) -> None:
        raise ToolError("internal_problem", "not an approval decision")

    monkeypatch.setattr(api_main, "check_sensitive_access", broken)
    executor = _RecordingExecutor()
    app.dependency_overrides[get_guarded_executor] = lambda: executor
    with TestClient(app, raise_server_exceptions=True) as client:
        with pytest.raises(ToolError):
            client.post(
                "/query-proposals",
                headers={"Authorization": f"Bearer {REQUESTER_TOKEN}"},
                json=_proposal(ORDERS_SQL),
            )
    assert executor.execute_calls == 0


def test_the_sensitive_rule_has_a_single_source(tokens, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    original = semantic.check_sensitive_access

    def counting(context: ExecutionContext, statement: object) -> None:
        calls.append(context.role)
        original(context, statement)  # type: ignore[arg-type]

    # The entry imports the name into api.main; the tool facade looks it up in
    # tools.semantic.  Both must resolve to the one function.
    assert api_main.check_sensitive_access is original
    monkeypatch.setattr(semantic, "check_sensitive_access", counting)
    monkeypatch.setattr(api_main, "check_sensitive_access", counting)

    tools = ControlledTools(executor=_RecordingExecutor([{"order_id": "o1", "amount_fen": 1}]))
    context = ExecutionContext(run_id="run-1", tenant_id="A", principal_id="a-requester", role="requester")
    tools.query_readonly({"sql": ORDERS_SQL, "params": {}}, context=context)
    assert calls == ["requester"]

    response = _post(REQUESTER_TOKEN, ORDERS_SQL, executor=_RecordingExecutor([{"order_id": "o1", "amount_fen": 1}]))
    assert response.status_code == 200
    assert calls == ["requester", "requester"]


def test_the_old_private_name_is_gone() -> None:
    import inspect

    assert "_check_sensitive_access" not in inspect.getsource(semantic)


needs_database = pytest.mark.skipif(
    not os.environ.get("QUERYSHIELD_DATABASE_URL"),
    reason="needs the local test database",
)


@needs_database
def test_real_database_requester_is_refused_and_approver_succeeds(tokens) -> None:
    executor = GuardedQueryExecutor()

    refused = _post(REQUESTER_TOKEN, NAME_SQL, executor=executor)
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "approval_required"
    assert "rows" not in refused.json()

    allowed = _post(APPROVER_TOKEN, NAME_SQL, executor=executor)
    assert allowed.status_code == 200
    body = allowed.json()
    assert body["result"]["tenant_id"] == "A"
    assert body["rows"] and all(row["name"] for row in body["rows"])
