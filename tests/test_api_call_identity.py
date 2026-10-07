from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from psycopg.errors import GroupingError, UndefinedColumn

from queryshield.agent import DurableModelCallStore
from queryshield.api.main import app, get_guarded_executor, get_model_provider
from queryshield.approval.service import reset_shared_state_stores
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.providers.contracts import ModelProviderError
from queryshield.providers.fake_model import FakeModel
from queryshield.providers.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleModel,
)


# Sync /queries runs the product Agent runtime.  These tests keep the proposal-boundary
# requirements (call identity, pre-model tenant rejection, provider/usage
# separation, no Fake fallback, controlled SQL errors, no DB before a valid
# proposal) on the new path; response shapes follow the shared outcome table.


@pytest.fixture()
def api_env(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYSHIELD_STATE_STORE_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", "test-token-a")
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    monkeypatch.delenv("QUERYSHIELD_FAKE_DB", raising=False)
    reset_shared_state_stores()
    yield tmp_path
    app.dependency_overrides.clear()
    reset_shared_state_stores()


def _real_model(monkeypatch, contents, *, call_id_prefix="provider-call"):
    """An OpenAI-compatible adapter over a scripted transport (one reply per call)."""

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        index = min(len(calls), len(contents) - 1)
        calls.append(index)
        return httpx.Response(
            200,
            request=request,
            headers={"x-request-id": f"provider-request-{len(calls)}"},
            json={
                "id": f"{call_id_prefix}-{len(calls)}",
                "model": "demo-model",
                "choices": [{"message": {"content": contents[index]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
            },
        )

    config = OpenAICompatibleConfig(base_url="https://example.test/v1", api_key="test-secret", model="demo-model")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleModel(config, client=client), calls


def _post(question: str):
    with TestClient(app) as client:
        return client.post("/queries", headers={"Authorization": "Bearer test-token-a"}, json={"question": question})


def test_query_api_persists_server_call_identity_without_payload(api_env, monkeypatch) -> None:
    database_path = api_env / "api-model-calls.sqlite3"
    monkeypatch.setenv("QUERYSHIELD_CALL_STORE_PATH", str(database_path))
    connection = _ProposalConnection([{"gross_fen": 3000}])
    app.dependency_overrides[get_model_provider] = lambda: FakeModel()
    app.dependency_overrides[get_guarded_executor] = lambda: GuardedQueryExecutor(connect=lambda: connection)

    response = _post("2026年9月已支付订单总额")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCEEDED"
    assert body["mode"] == "fake"
    assert body["facts"]["facts"][0]["metric_id"] == "gross_fen"
    assert body["facts"]["facts"][0]["value"] == 3000
    assert body["answer"].startswith("已核实：")
    assert body["result"]["tenant_id"] == "A"
    assert body["result"]["rows"] == [{"gross_fen": 3000}]
    assert len(body["usage"]) == body["model_call_count"] >= 2
    assert all(item["usage_status"] == "unknown" for item in body["usage"])

    with DurableModelCallStore(database_path) as store:
        for item in body["usage"]:
            restored = store.get(body["run_id"], item["model_call_id"])
            assert restored.request_id
            assert store.attempts(body["run_id"], item["model_call_id"])[0].attempt_kind == "new"
        columns = {
            row[1]
            for row in store._connection.execute("PRAGMA table_info(model_call_attempts)").fetchall()
        }

    assert "prompt" not in columns
    assert "response" not in columns
    assert "api_key" not in columns


def test_query_api_rejects_explicit_foreign_tenant_before_provider_or_sql(api_env) -> None:
    connection = _ProposalConnection([])
    model = _CountingModel()
    app.dependency_overrides[get_model_provider] = lambda: model
    app.dependency_overrides[get_guarded_executor] = lambda: GuardedQueryExecutor(connect=lambda: connection)

    response = _post("查询tenant-B的订单金额")

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"
    assert model.calls == 0
    assert connection.cursor_instance.executed is None
    assert "run_id" not in response.json()


def test_query_api_real_adapter_keeps_provider_id_and_usage_separate(api_env, monkeypatch) -> None:
    connection = _ProposalConnection([{"order_id": "o1", "amount_fen": 10000}])
    app.dependency_overrides[get_guarded_executor] = lambda: GuardedQueryExecutor(connect=lambda: connection)
    model, calls = _real_model(
        monkeypatch,
        [
            '{"type":"tool_call","name":"query_readonly","arguments":{"sql":"SELECT o.order_id, o.amount_fen '
            'FROM orders AS o WHERE o.order_id = %s","params":{"0":"o1"}}}',
            '{"type":"final_answer","answer":"o1","source_ids":[],"fact_refs":[]}',
        ],
    )
    app.dependency_overrides[get_model_provider] = lambda: model

    response = _post("查询订单o1的金额，确认不会读到其他租户的同名订单。")

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "real"
    assert body["result"]["rows"] == [{"order_id": "o1", "amount_fen": 10000}]
    assert body["result"]["tenant_id"] == "A"
    assert body["facts"] is None
    assert len(body["usage"]) == len(calls) == 2
    for index, usage in enumerate(body["usage"], start=1):
        assert usage["model_call_id"] != f"provider-call-{index}"
        assert usage["provider_call_id"] == f"provider-call-{index}"
        assert usage["provider_request_id"] == f"provider-request-{index}"
        assert usage["usage_status"] == "known"
        assert usage["total_tokens"] == 22


def test_query_api_provider_error_does_not_fallback_to_fake(api_env, monkeypatch) -> None:
    def missing_configuration():
        raise ModelProviderError(
            "missing_model_configuration",
            {
                "status": "blocked",
                "mode": "real",
                "provider": "openai_compatible",
                "usage": None,
                "usage_status": "unknown",
                "error_code": "missing_model_configuration",
            },
        )

    fake_calls = []
    monkeypatch.setattr(FakeModel, "complete", lambda *args, **kwargs: fake_calls.append(1))
    app.dependency_overrides[get_model_provider] = missing_configuration

    response = _post("A租户有哪些已支付订单？")

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "missing_model_configuration"
    assert "result" not in body
    assert fake_calls == []

    # A provider failure during the run is persisted, mapped by the shared table,
    # and still never falls back to the Fake model.
    app.dependency_overrides[get_model_provider] = lambda: _FailingModel("upstream_timeout")
    response = _post("2026年9月已支付订单总额")
    body = response.json()
    assert response.status_code == 504
    assert body["status"] == "FAILED"
    assert body["error"]["code"] == "upstream_timeout"
    assert body["result"] is None
    assert fake_calls == []


def _bad_sql_run(monkeypatch, execute_error, sql: str, params: str):
    connection = _ProposalConnection([], execute_error=execute_error)
    app.dependency_overrides[get_guarded_executor] = lambda: GuardedQueryExecutor(connect=lambda: connection)
    model, calls = _real_model(
        monkeypatch,
        ['{"type":"tool_call","name":"query_readonly","arguments":{"sql":"' + sql + '","params":' + params + "}}"],
    )
    app.dependency_overrides[get_model_provider] = lambda: model
    return calls


def test_query_api_turns_unknown_database_column_into_controlled_sql_error(api_env, monkeypatch) -> None:
    calls = _bad_sql_run(monkeypatch, UndefinedColumn("column o.id does not exist"), "SELECT o.id FROM orders AS o", "{}")

    response = _post("查询订单编号。")

    body = response.json()
    # invalid_sql is repairable once; the second identical failure ends the run.
    assert response.status_code == 502
    assert body["status"] == "FAILED"
    assert body["error"]["code"] == "query_repair_limit"
    assert body["result"] is None
    assert "o.id" not in response.text and "does not exist" not in response.text
    assert len(calls) == 2
    assert [item["provider_call_id"] for item in body["usage"]] == ["provider-call-1", "provider-call-2"]
    assert all(item["usage_status"] == "known" for item in body["usage"])


def test_query_api_turns_incomplete_grouping_into_controlled_sql_error(api_env, monkeypatch) -> None:
    calls = _bad_sql_run(
        monkeypatch,
        GroupingError("column o.amount_fen must appear in GROUP BY"),
        "SELECT o.order_id, o.amount_fen, SUM(r.amount_fen) AS refund_fen FROM orders AS o "
        "INNER JOIN refunds AS r ON o.order_id = r.order_id WHERE o.order_id = %s GROUP BY o.order_id",
        '{"0":"o1"}',
    )

    response = _post("查询订单o1的退款。")

    body = response.json()
    assert response.status_code == 502
    assert body["error"]["code"] == "query_repair_limit"
    assert body["result"] is None
    assert "GROUP BY" not in response.text
    assert len(calls) == 2
    assert body["usage"][0]["provider_call_id"] == "provider-call-1"


def test_query_api_rejects_malformed_real_proposal_before_database(api_env, monkeypatch) -> None:
    opened = False

    def connect() -> _ProposalConnection:
        nonlocal opened
        opened = True
        return _ProposalConnection([])

    app.dependency_overrides[get_guarded_executor] = lambda: GuardedQueryExecutor(connect=connect)
    model, calls = _real_model(monkeypatch, ["not-json"], call_id_prefix="provider-call-invalid-json")
    app.dependency_overrides[get_model_provider] = lambda: model

    response = _post("A租户有哪些已支付订单？")

    assert response.status_code == 502
    assert response.json()["status"] == "FAILED"
    assert response.json()["error"]["code"] == "invalid_json"
    assert response.json()["usage"][0]["provider_call_id"] == "provider-call-invalid-json-1"
    assert len(calls) == 1
    assert opened is False


class _CountingModel:
    mode = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, request_id=None, model_call_id=None):
        self.calls += 1
        raise AssertionError("the model must not be called")


class _FailingModel:
    mode = "real"

    def __init__(self, code: str) -> None:
        self.code = code

    def complete(self, messages, *, request_id=None, model_call_id=None, run_id=None):
        raise ModelProviderError(
            self.code,
            {"status": "failed", "mode": "real", "provider": "scripted", "usage": None, "usage_status": "unknown", "error_code": self.code},
        )


class _ProposalCursor:
    def __init__(
        self,
        rows: list[dict[str, object]],
        *,
        execute_error: Exception | None = None,
    ) -> None:
        self.rows = rows
        self.execute_error = execute_error
        self.executed: tuple[str, tuple[object, ...]] | None = None

    def __enter__(self) -> _ProposalCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        if self.execute_error is not None:
            raise self.execute_error
        self.executed = (sql, params)

    def fetchmany(self, size: int) -> list[dict[str, object]]:
        return self.rows[:size]


class _ProposalConnection:
    def __init__(
        self,
        rows: list[dict[str, object]],
        *,
        execute_error: Exception | None = None,
    ) -> None:
        self.cursor_instance = _ProposalCursor(rows, execute_error=execute_error)

    def __enter__(self) -> _ProposalConnection:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def cursor(self, *, row_factory: object) -> _ProposalCursor:
        return self.cursor_instance


def test_query_proposal_api_executes_through_server_bound_guard(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", "test-token-a")
    connection = _ProposalConnection([{"order_id": "o1", "tenant_id": "A"}])
    app.dependency_overrides[get_guarded_executor] = lambda: GuardedQueryExecutor(
        connect=lambda: connection
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/query-proposals",
                headers={"Authorization": "Bearer test-token-a"},
                json={
                    "proposal": (
                        '{"type":"tool_call","name":"query_readonly",'
                        '"arguments":{"sql":"SELECT o.order_id, o.tenant_id '
                        'FROM orders AS o","params":{}}}'
                    )
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == [{"order_id": "o1", "tenant_id": "A"}]
    assert body["result"]["tenant_id"] == "A"
    assert body["result"]["row_count"] == 1
    assert isinstance(body["usage"][0]["model_call_id"], str)
    assert body["usage"][0]["model_call_id"]
    assert connection.cursor_instance.executed is not None
    assert connection.cursor_instance.executed[1] == ("A",)


def test_query_proposal_api_rejects_model_identity_parameter_before_database(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", "test-token-a")
    opened = False

    def connect() -> _ProposalConnection:
        nonlocal opened
        opened = True
        return _ProposalConnection([])

    app.dependency_overrides[get_guarded_executor] = lambda: GuardedQueryExecutor(connect=connect)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/query-proposals",
                headers={"Authorization": "Bearer test-token-a"},
                json={
                    "proposal": (
                        '{"type":"tool_call","name":"query_readonly",'
                        '"arguments":{"sql":"SELECT * FROM orders",'
                        '"params":{"tenant_id":"B"}}}'
                    )
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "reserved_parameter"
    assert opened is False


def test_query_proposal_api_keeps_server_tenant_when_model_filters_other_tenant(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", "test-token-a")
    connection = _ProposalConnection([])
    app.dependency_overrides[get_guarded_executor] = lambda: GuardedQueryExecutor(
        connect=lambda: connection
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/query-proposals",
                headers={"Authorization": "Bearer test-token-a"},
                json={
                    "proposal": (
                        '{"type":"tool_call","name":"query_readonly",'
                        '"arguments":{"sql":"SELECT o.order_id, o.tenant_id '
                        'FROM orders AS o WHERE o.tenant_id = %s",'
                        '"params":{"0":"B"}}}'
                    )
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == []
    assert body["result"]["tenant_id"] == "A"
    assert body["result"]["row_count"] == 0
    assert connection.cursor_instance.executed is not None
    assert connection.cursor_instance.executed[1] == ("A", "B")
