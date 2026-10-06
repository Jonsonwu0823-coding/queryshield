"""The approval/API tidy-up changes no behaviour.

These tests describe the behaviour of ``api/``, ``approval/``, ``models/`` and ``memory/`` as
a reader of the HTTP API or the stored run can see it: status codes, error codes and messages,
body keys and their order, the order of persisted events, and the order of the checks.  They
were written and run against the code before the tidy-up, then run again unchanged after it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import psycopg
import pytest
from psycopg.errors import GroupingError, QueryCanceled, UndefinedColumn
from starlette.requests import Request

from queryshield.api import main as api_main
from queryshield.api.main import app, get_guarded_executor, get_model_provider
from queryshield.agent.runtime import B1_PROFILE, RuntimeConfigurationError
from queryshield.approval.service import (
    MAX_ACTIVE_RUNS,
    ApprovalConflict,
    FixtureQueryExecutor,
    RunService,
    shared_run_service,
)
from queryshield.db.guarded import GuardedQueryError
from queryshield.policy.sql import SQLPolicyError
from queryshield.providers.contracts import ModelProviderError
from queryshield.tools.semantic import ToolError

from test_clarification import REQUESTER as REQUESTER_A  # the identity mapping, not the HTTP token
from test_clarification import _cite, _query, _resume, _start, service  # noqa: F401  (service is a fixture)
from test_http_queries import APPROVER, OTHER, REQUESTER, auth, env, wait  # noqa: F401  (env is a fixture)

UNAUTHORIZED = (401, "unauthorized", "身份认证失败")
NOT_FOUND = (404, "not_found", "任务不存在")


def assert_error(response, status: int, code: str, message: str, *, run_id: str | None = None) -> None:
    body = response.json()
    assert response.status_code == status, body
    assert list(body) == (["error"] if run_id is None else ["error", "run_id"]), body
    assert list(body["error"]) == ["code", "message", "request_id"], body
    assert (body["error"]["code"], body["error"]["message"]) == (code, message)
    assert isinstance(body["error"]["request_id"], str) and body["error"]["request_id"]
    if run_id is not None:
        assert body["run_id"] == run_id


def _raising(exc: BaseException):
    def dependency():
        raise exc

    return dependency


# ---------------------------------------------------------------------------
# One error body for every handler
# ---------------------------------------------------------------------------

HANDLED = [
    (get_model_provider, ModelProviderError("upstream_timeout", {}), 504, "upstream_timeout", "模型服务暂时不可用"),
    (get_model_provider, ModelProviderError("missing_model_configuration", {}), 503, "missing_model_configuration", "模型服务暂时不可用"),
    (get_model_provider, ModelProviderError("invalid_model_configuration", {}), 503, "invalid_model_configuration", "模型服务暂时不可用"),
    (get_model_provider, ModelProviderError("invalid_provider_mode", {}), 503, "invalid_provider_mode", "模型服务暂时不可用"),
    (get_model_provider, ModelProviderError("upstream_http_error", {}), 502, "upstream_http_error", "模型服务暂时不可用"),
    (get_model_provider, ModelProviderError("invalid_response", {}), 502, "invalid_response", "模型服务暂时不可用"),
    (get_model_provider, RuntimeConfigurationError("knowledge_unavailable", "x"), 503, "knowledge_unavailable", "服务配置不允许运行"),
    (get_guarded_executor, psycopg.OperationalError("down"), 503, "database_unavailable", "数据库暂时不可用"),
    (get_guarded_executor, QueryCanceled("slow"), 504, "query_timeout", "查询超时"),
]


@pytest.mark.parametrize("dependency, exc, status, code, message", HANDLED)
def test_exception_handlers_share_one_error_body(env, dependency, exc, status, code, message) -> None:
    app.dependency_overrides[dependency] = _raising(exc)
    response = env.post("/queries", headers=auth(REQUESTER), json={"question": "2026年9月已支付订单总额"})
    assert_error(response, status, code, message)


@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": "q", "extra": 1}, {"question": "x" * 501}])
def test_an_invalid_request_body_is_422(env, body) -> None:
    assert_error(env.post("/queries", headers=auth(REQUESTER), json=body), 422, "invalid_request", "请求参数不符合要求")


# ---------------------------------------------------------------------------
# Authentication: the same 401 on every route, and it does not outrank a bad body
# ---------------------------------------------------------------------------

ROUTES = [
    ("GET", "/runs/run-x", None),
    ("GET", "/runs/run-x/result", None),
    ("POST", "/runs/run-x/resume", {"answer": "a"}),
    ("POST", "/runs/run-x/approval", {"approval_id": "a", "decision": "approve"}),
    ("POST", "/runs/run-x/cancel", {}),
    ("GET", "/runs/run-x/events", None),
    ("GET", "/preferences/display_language", None),
    ("PUT", "/preferences/display_language", {"value": "en", "confirmed": True}),
    ("DELETE", "/preferences/display_language", None),
    ("POST", "/queries", {"question": "q"}),
    ("POST", "/query-proposals", {"proposal": "p"}),
]
BAD_BODIES = [
    ("POST", "/runs/run-x/resume", {}),
    ("POST", "/runs/run-x/approval", {}),
    ("POST", "/runs/run-x/cancel", {"x": 1}),
    ("PUT", "/preferences/display_language", {}),
    ("POST", "/queries", {}),
    ("POST", "/query-proposals", {}),
]


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer not-a-token"}, {"Authorization": "Basic abc"}])
@pytest.mark.parametrize("method, path, body", ROUTES)
def test_every_route_answers_401_without_a_valid_identity(env, method, path, body, headers) -> None:
    response = env.request(method, path, headers=headers, json=body)
    assert_error(response, *UNAUTHORIZED)


@pytest.mark.parametrize("method, path, body", BAD_BODIES)
def test_a_bad_body_is_422_even_without_authentication(env, method, path, body) -> None:
    assert_error(env.request(method, path, json=body), 422, "invalid_request", "请求参数不符合要求")


# ---------------------------------------------------------------------------
# Route-level errors: status, code and message per route
# ---------------------------------------------------------------------------


def test_unknown_runs_are_404_on_every_run_route(env) -> None:
    headers = auth(REQUESTER)
    for method, path, body in [
        ("GET", "/runs/run-x", None),
        ("POST", "/runs/run-x/resume", {"answer": "a"}),
        ("POST", "/runs/run-x/approval", {"approval_id": "a", "decision": "approve"}),
        ("POST", "/runs/run-x/cancel", {}),
        ("GET", "/runs/run-x/events", None),
    ]:
        assert_error(env.request(method, path, headers=headers, json=body), *NOT_FOUND)
    assert_error(env.get("/runs/run-x/result", headers=headers), 404, "not_found", "结果不存在")


def test_preference_routes(env) -> None:
    headers = auth(REQUESTER)
    assert_error(env.get("/preferences/nope", headers=headers), 422, "unknown_preference", "偏好键不受支持")
    assert_error(env.delete("/preferences/nope", headers=headers), 422, "unknown_preference", "偏好键不受支持")
    assert_error(
        env.put("/preferences/nope", headers=headers, json={"value": "en", "confirmed": True}),
        422, "unknown_preference", "偏好请求未通过校验",
    )
    assert_error(
        env.put("/preferences/display_language", headers=headers, json={"value": "en", "confirmed": False}),
        422, "confirmation_required", "偏好请求未通过校验",
    )
    assert_error(
        env.put("/preferences/display_language", headers=headers, json={"value": "fr", "confirmed": True}),
        422, "invalid_preference_value", "偏好请求未通过校验",
    )
    assert_error(env.get("/preferences/display_language", headers=headers), 404, "preference_not_found", "偏好不存在")
    stored = env.put("/preferences/display_language", headers=headers, json={"value": "en", "confirmed": True})
    assert stored.status_code == 200 and stored.json()["value"] == "en"
    assert env.get("/preferences/display_language", headers=headers).json() == stored.json()
    deleted = env.delete("/preferences/display_language", headers=headers)
    assert deleted.status_code == 204 and deleted.content == b""


def test_queries_prechecks(env) -> None:
    headers = auth(REQUESTER)
    assert_error(
        env.post("/queries", headers=headers, json={"question": "查询tenant-B的订单金额"}),
        403, "forbidden", "请求涉及当前身份不可访问的租户",
    )
    assert_error(
        env.post("/queries", headers={**headers, "Prefer": "return=minimal"}, json={"question": "q"}),
        400, "unsupported_preference", "只支持Prefer: respond-async",
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_full_service_is_503_on_both_paths_and_async_drops_the_request_call_store(env, monkeypatch, asynchronous) -> None:
    service = shared_run_service()
    seen = []
    for name in ("run_sync", "start_async"):
        original = getattr(service, name)

        def spy(*args, _original=original, _name=name, **kwargs):
            seen.append((_name, kwargs["deps"]))
            return _original(*args, **kwargs)

        monkeypatch.setattr(service, name, spy)
    headers = {**auth(REQUESTER), **({"Prefer": "respond-async"} if asynchronous else {})}
    monkeypatch.setattr(RunService, "_active_count", lambda self: MAX_ACTIVE_RUNS)
    response = env.post("/queries", headers=headers, json={"question": "2026年9月已支付订单总额"})
    assert_error(response, 503, "run_capacity_reached", "当前运行容量已满")
    [(name, deps)] = seen
    assert name == ("start_async" if asynchronous else "run_sync")
    assert (deps.call_store is None) is asynchronous
    assert deps.profile == B1_PROFILE and deps.model is not None and deps.executor is not None


def test_the_async_path_accepts_with_202_and_a_location(env) -> None:
    response = env.post(
        "/queries", headers={**auth(REQUESTER), "Prefer": "respond-async"}, json={"question": "2026年9月已支付订单总额"}
    )
    body = response.json()
    assert response.status_code == 202 and list(body) == ["run_id", "status"] and body["status"] == "RUNNING"
    assert response.headers["location"] == f"/runs/{body['run_id']}"
    wait(env, body["run_id"], {"SUCCEEDED"})


# ---------------------------------------------------------------------------
# /query-proposals: every failure has one status, code and message
# ---------------------------------------------------------------------------

ORDERS_SQL = "SELECT COUNT(*) AS paid_count FROM orders AS o"
NAME_SQL = "SELECT c.customer_id, c.name FROM customers AS c ORDER BY c.customer_id LIMIT 3"


def _proposal(sql: str = ORDERS_SQL, params: object = None, *, name: str = "query_readonly") -> dict[str, str]:
    arguments = {"sql": sql, "params": {} if params is None else params}
    return {"proposal": json.dumps({"type": "tool_call", "name": name, "arguments": arguments}, ensure_ascii=False)}


class _Executor:
    """Records what it was asked and then returns the fixture result or raises."""

    def __init__(self, raises: BaseException | None = None) -> None:
        self.raises = raises
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.fixture = FixtureQueryExecutor()

    def execute(self, sql, *, context, params=(), metric_bindings=()):
        self.calls.append((sql, tuple(params)))
        if self.raises is not None:
            raise self.raises
        return self.fixture.execute(sql, context=context, params=params, metric_bindings=metric_bindings)


def _propose(client, body: dict, *, token: str = REQUESTER, executor: _Executor | None = None):
    executor = executor or _Executor()
    app.dependency_overrides[get_guarded_executor] = lambda: executor
    return client.post("/query-proposals", headers=auth(token), json=body), executor


PARAMS_MESSAGE = "模型提案参数未通过校验"


@pytest.mark.parametrize(
    "params",
    [{"a": 1}, {"-1": 1}, {"01": 1}, {"00": 1}, {"٠": 1}, {"0": 1, "2": 3}, {"1": 1}, {"0": 1, "01": 2}],
)
def test_malformed_parameter_keys_are_422_invalid_field(env, params) -> None:
    response, executor = _propose(env, _proposal(params=params))
    assert_error(response, 422, "invalid_field", PARAMS_MESSAGE)
    assert executor.calls == []


@pytest.mark.parametrize(
    "params, expected",
    [
        (None, ()),
        ({"0": "a"}, ("a",)),
        ({"1": "b", "0": "a"}, ("a", "b")),
        ({str(index): f"v{index}" for index in range(11)}, tuple(f"v{index}" for index in range(11))),
    ],
)
def test_wellformed_parameter_keys_reach_the_executor_in_index_order(env, params, expected) -> None:
    response, executor = _propose(env, _proposal(params=params))
    assert response.status_code == 200, response.json()
    assert executor.calls == [(ORDERS_SQL, expected)]


def test_a_proposal_success_has_this_body(env) -> None:
    response, _ = _propose(env, _proposal())
    body = response.json()
    assert response.status_code == 200
    assert list(body) == ["run_id", "status", "rows", "result", "proposal_sha256", "usage", "mode"]
    assert body["status"] == "SUCCEEDED" and body["mode"] == "proposal_ingress"
    assert list(body["usage"][0]) == ["model_call_id"]
    assert body["rows"] == body["result"]["rows"]


def test_a_proposal_that_does_not_parse_is_422(env) -> None:
    response, executor = _propose(env, {"proposal": "not json"})
    assert response.status_code == 422 and executor.calls == []
    body = response.json()
    assert list(body) == ["error"] and body["error"]["message"] == "模型提案未通过结构校验"
    assert list(body["error"]) == ["code", "message", "request_id"]


def test_only_query_readonly_proposals_run_here(env) -> None:
    body = {"proposal": json.dumps({"type": "tool_call", "name": "describe_tables", "arguments": {"tables": ["orders"]}})}
    response, executor = _propose(env, body)
    assert_error(response, 422, "unsupported_action", "当前入口只执行query_readonly提案")
    assert executor.calls == []


EXECUTOR_FAILURES = [
    (SQLPolicyError("forbidden_statement", "no"), 403, "forbidden_statement", "查询被SQL安全策略拒绝"),
    (UndefinedColumn("no such column"), 403, "invalid_sql", "提案引用了不存在的数据库字段"),
    (GroupingError("grouping"), 403, "invalid_sql", "提案的聚合字段分组不完整"),
    (GuardedQueryError("invalid_params", "x"), 422, "invalid_params", "查询未通过受限执行器"),
]


@pytest.mark.parametrize("exc, status, code, message", EXECUTOR_FAILURES)
def test_executor_failures_map_to_one_status_code_and_message(env, exc, status, code, message) -> None:
    response, executor = _propose(env, _proposal(), executor=_Executor(exc))
    assert_error(response, status, code, message)
    assert len(executor.calls) == 1


def test_a_result_over_the_row_limit_is_a_200_with_limit_reached(env) -> None:
    response, _ = _propose(env, _proposal(), executor=_Executor(GuardedQueryError("limit_reached", "too many")))
    body = response.json()
    assert response.status_code == 200
    assert list(body) == ["run_id", "status", "error", "usage"]
    assert body["status"] == "LIMIT_REACHED"
    assert list(body["error"]) == ["code", "message", "request_id"]
    assert (body["error"]["code"], body["error"]["message"]) == ("limit_reached", "查询结果超过100行")
    assert list(body["usage"][0]) == ["model_call_id"]


def test_a_sensitive_proposal_needs_approval_for_the_requester_only(env) -> None:
    response, executor = _propose(env, _proposal(NAME_SQL))
    assert_error(response, 403, "approval_required", "包含客户姓名的查询要通过 /queries 发起，由同租户审批人批准")
    assert executor.calls == []
    approver, executor = _propose(env, _proposal(NAME_SQL), token=APPROVER)
    assert approver.status_code == 200 and len(executor.calls) == 1


def test_a_sql_that_does_not_parse_goes_to_the_executor(env) -> None:
    response, executor = _propose(
        env, _proposal("DELETE FROM customers"), executor=_Executor(SQLPolicyError("forbidden_statement", "no"))
    )
    assert_error(response, 403, "forbidden_statement", "查询被SQL安全策略拒绝")
    assert [sql for sql, _ in executor.calls] == ["DELETE FROM customers"]


def test_other_tool_errors_are_not_caught_as_approval(env, monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise ToolError("something_else", "x")

    monkeypatch.setattr(api_main, "check_sensitive_access", broken)
    app.dependency_overrides[get_guarded_executor] = lambda: _Executor()
    with pytest.raises(ToolError):
        env.post("/query-proposals", headers=auth(REQUESTER), json=_proposal())


# ---------------------------------------------------------------------------
# One outcome table for the sync and the resume response
# ---------------------------------------------------------------------------


def _request() -> Request:
    return Request({"type": "http", "method": "GET", "path": "/", "headers": []})


def _run(status: str, **extra) -> dict[str, object]:
    base = {
        "run_id": "run-1", "status": status, "tenant_id": "A", "principal_id": "p", "role": "requester",
        "question": "q", "created_at": "t0", "updated_at": "t1", "mode": "fake",
        "model_call_count": 1, "tool_call_count": 0, "sql_exec_count": 0,
    }
    return {**base, **extra}


_NO_EVENTS = SimpleNamespace(store=SimpleNamespace(events=lambda *args, **kwargs: []))

#            status, error_code -> (resume status, sync status, error code or None)
OUTCOMES = [
    ("SUCCEEDED", None, 200, 200, None),
    ("WAITING_USER", None, 200, 202, None),
    ("WAITING_APPROVAL", None, 200, 202, None),
    ("DENIED", None, 403, 403, "run_failed"),
    ("DENIED", "forbidden", 403, 403, "forbidden"),
    ("FAILED", None, 502, 502, "run_failed"),
    ("FAILED", "query_repair_limit", 502, 502, "query_repair_limit"),
    ("FAILED", "upstream_timeout", 504, 504, "upstream_timeout"),
    ("FAILED", "database_unavailable", 503, 503, "database_unavailable"),
    ("FAILED", "approval_permission_unavailable", 503, 503, "approval_permission_unavailable"),
    ("LIMIT_REACHED", None, 502, 502, "run_failed"),
    ("CANCELLED", None, 409, 409, "run_cancelled"),
    ("CANCELLED", "forbidden", 409, 409, "run_cancelled"),
    ("RUNNING", None, 502, 502, "run_failed"),
    ("CANCEL_REQUESTED", None, 502, 502, "run_failed"),
    ("USAGE_UNKNOWN", None, 502, 502, "run_failed"),
]


def _outcome(response) -> tuple[int, dict | None, str | None]:
    body = json.loads(response.body)
    error = body.get("error")
    if error is not None:
        assert list(error) == ["code", "message", "request_id"]
    return response.status_code, error, response.headers.get("location")


@pytest.mark.parametrize("status, error_code, resume_status, sync_status, code", OUTCOMES)
def test_the_resume_and_sync_responses_share_the_outcome_table(status, error_code, resume_status, sync_status, code) -> None:
    run = _run(status, error_code=error_code)

    request = _request()
    http_status, error, location = _outcome(api_main._resume_run_response(request, run))
    assert http_status == resume_status and location is None
    if code is None:
        assert error is None
    else:
        assert (error["code"], error["message"]) == (code, "续跑未完成")
        assert request.state.error_code == code

    request = _request()
    http_status, error, location = _outcome(api_main._sync_run_response(request, _NO_EVENTS, run))
    assert http_status == sync_status
    assert location == ("/runs/run-1" if sync_status == 202 else None)
    if code is None:
        assert error is None
    else:
        assert (error["code"], error["message"]) == (code, "查询未完成")
        assert request.state.error_code == code


RUN_KEYS = [
    "run_id", "status", "tenant_id", "principal_id", "role", "question", "created_at", "updated_at", "mode",
    "model_call_count", "tool_call_count", "sql_exec_count",
]


def test_the_sync_and_resume_body_keys() -> None:
    succeeded = _run("SUCCEEDED", answer="a", result={"result_id": "r1", "catalog_version": "c"})
    body = json.loads(api_main._sync_run_response(_request(), _NO_EVENTS, succeeded).body)
    assert list(body) == RUN_KEYS + ["answer", "result", "facts", "answer_status", "source_ids", "profile", "usage", "sources"]
    assert body["sources"] == [{"source_id": "r1", "version": "c"}]
    failed = json.loads(api_main._sync_run_response(_request(), _NO_EVENTS, _run("FAILED", error_code="x")).body)
    assert list(failed) == RUN_KEYS + ["error_code", "answer", "result", "facts", "answer_status", "source_ids", "profile", "usage", "sources", "error"]
    waiting = json.loads(
        api_main._sync_run_response(
            _request(),
            _NO_EVENTS,
            _run("WAITING_USER", checkpoint={"agent_checkpoint": {"waiting_question": "which?"}}),
        ).body
    )
    assert waiting["pending_question"] == "which?" and "pending_question" not in body
    resumed = json.loads(api_main._resume_run_response(_request(), _run("FAILED", error_code="x")).body)
    assert list(resumed) == RUN_KEYS + ["error_code", "answer", "result", "facts", "answer_status", "source_ids", "error"]


# ---------------------------------------------------------------------------
# /runs/{id}/events: cursor errors and the exact frames
# ---------------------------------------------------------------------------


def _finished_run(client) -> str:
    response = client.post("/queries", headers=auth(REQUESTER), json={"question": "2026年9月已支付订单总额"})
    assert response.status_code == 200 and response.json()["status"] == "SUCCEEDED"
    return response.json()["run_id"]


def _frame(event: dict) -> str:
    data = {key: event[key] for key in ("event_id", "run_id", "type", "status", "occurred_at")}
    if event.get("result_id") is not None:
        data["result_id"] = event["result_id"]
    return (
        f"id: {event['event_id']}\nevent: {event['type']}\n"
        f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )


def _stream(client, run_id: str, **headers) -> tuple[int, str]:
    with client.stream("GET", f"/runs/{run_id}/events", headers={**auth(REQUESTER), **headers}) as response:
        return response.status_code, b"".join(response.iter_bytes()).decode("utf-8")


def test_the_stream_is_a_heartbeat_then_one_frame_per_stored_event(env) -> None:
    run_id = _finished_run(env)
    events = shared_run_service().store.events(run_id, after_event_id=0, limit=1000)
    assert any(event["result_id"] is not None for event in events)
    status, text = _stream(env, run_id)
    assert status == 200
    assert text == ": heartbeat\n\n" + "".join(_frame(event) for event in events)


@pytest.mark.parametrize("cursor, first_id", [("", 1), ("  ", 1), ("0", 1), ("2", 3), (" 2 ", 3)])
def test_the_cursor_selects_the_events_after_it(env, cursor, first_id) -> None:
    run_id = _finished_run(env)
    events = shared_run_service().store.events(run_id, after_event_id=0, limit=1000)
    _, text = _stream(env, run_id, **{"Last-Event-ID": cursor})
    assert text == ": heartbeat\n\n" + "".join(_frame(event) for event in events if event["event_id"] >= first_id)


@pytest.mark.parametrize("cursor", ["abc", "-1", "1.5", "1,2", "0x1"])
def test_a_malformed_cursor_is_400(env, cursor) -> None:
    run_id = _finished_run(env)
    response = env.get(f"/runs/{run_id}/events", headers={**auth(REQUESTER), "Last-Event-ID": cursor})
    assert_error(response, 400, "invalid_event_cursor", "Last-Event-ID格式错误", run_id=run_id)


def test_a_cursor_before_the_kept_history_is_410(env) -> None:
    run_id = _finished_run(env)
    store = shared_run_service().store
    store._connection.execute("DELETE FROM events WHERE run_id = ? AND event_id < 3", (run_id,))
    for cursor in ("0", "1"):
        response = env.get(f"/runs/{run_id}/events", headers={**auth(REQUESTER), "Last-Event-ID": cursor})
        assert_error(response, 410, "event_history_expired", "事件游标已过期", run_id=run_id)
    status, text = _stream(env, run_id, **{"Last-Event-ID": "2"})
    assert status == 200 and text.startswith(": heartbeat\n\nid: 3\n")


def test_another_tenant_cannot_open_the_stream(env) -> None:
    run_id = _finished_run(env)
    assert_error(env.get(f"/runs/{run_id}/events", headers=auth(OTHER)), *NOT_FOUND)


@pytest.mark.parametrize("status", ["SUCCEEDED", "DENIED", "FAILED", "LIMIT_REACHED", "CANCELLED", "USAGE_UNKNOWN"])
def test_the_stream_ends_for_every_terminal_status(env, status) -> None:
    run_id = _finished_run(env)
    store = shared_run_service().store
    store._connection.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
    store.update_run(run_id, status=status)
    assert _stream(env, run_id) == (200, ": heartbeat\n\n")


# ---------------------------------------------------------------------------
# answer status: one normalisation for what is written and what is read back
# ---------------------------------------------------------------------------


def _stored(status="SUCCEEDED", answer="a", **checkpoint) -> dict:
    return {"status": status, "answer": answer, "checkpoint": checkpoint}


@pytest.mark.parametrize(
    "run, expected",
    [
        (_stored(status="FAILED", answer_status="verified"), (None, None)),
        (_stored(answer=None, answer_status="verified"), (None, None)),
        (_stored(), ("unverified", [])),
        ({"status": "SUCCEEDED", "answer": "a"}, ("unverified", [])),
        ({"status": "SUCCEEDED", "answer": "a", "checkpoint": "not a mapping"}, ("unverified", [])),
        (_stored(answer_status="verified", answer_source_ids=["s1", "s2"]), ("verified", ["s1", "s2"])),
        (_stored(answer_status="unverified", answer_source_ids=[]), ("unverified", [])),
        (_stored(answer_status="no_data", answer_source_ids=["s1"]), ("no_data", ["s1"])),
        (_stored(answer_status="bogus"), ("unverified", [])),
        (_stored(answer_status=None), ("unverified", [])),
        (_stored(answer_status=5), ("unverified", [])),
        (_stored(answer_status="verified", answer_source_ids=["a", 1, None, "b"]), ("verified", ["a", "b"])),
        (_stored(answer_status="verified", answer_source_ids="a"), ("verified", [])),
        (_stored(answer_status="verified", answer_source_ids=("a",)), ("verified", [])),
    ],
)
def test_answer_fields_read_back_from_the_stored_run(run, expected) -> None:
    assert api_main._answer_fields(run) == {"answer_status": expected[0], "source_ids": expected[1]}


@pytest.mark.parametrize(
    "payload, succeeded, expected",
    [
        ({"answer": "a", "answer_status": "verified"}, False, (None, None)),
        ({"answer": None, "answer_status": "verified"}, True, (None, None)),
        ({"answer": "a"}, True, ("unverified", [])),
        ({"answer": "a", "answer_status": "bogus"}, True, ("unverified", [])),
        ({"answer": "a", "answer_status": 5}, True, ("unverified", [])),
        ({"answer": "a", "answer_status": "no_data"}, True, ("no_data", [])),
        (
            {"answer": "a", "answer_status": "verified", "action": {"type": "final_answer", "source_ids": ["s", 1, "t"]}},
            True,
            ("verified", ["s", "t"]),
        ),
        (
            {"answer": "a", "answer_status": "verified", "action": {"type": "ask_user", "source_ids": ["s"]}},
            True,
            ("verified", []),
        ),
        (
            {"answer": "a", "answer_status": "verified", "action": {"type": "final_answer", "source_ids": "s"}},
            True,
            ("verified", []),
        ),
        ({"answer": "a", "answer_status": "verified", "action": "final_answer"}, True, ("verified", [])),
    ],
)
def test_the_answer_envelope_written_with_the_run(payload, succeeded, expected) -> None:
    from queryshield.approval.service import _answer_envelope

    assert _answer_envelope(payload, succeeded=succeeded) == {"answer_status": expected[0], "answer_source_ids": expected[1]}


# ---------------------------------------------------------------------------
# request models and preferences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model, field", [("QueryRequest", "question"), ("QueryProposalRequest", "proposal")])
def test_request_text_is_stripped_and_must_not_be_blank(model, field) -> None:
    from pydantic import ValidationError

    from queryshield.models import queries

    cls = getattr(queries, model)
    assert getattr(cls(**{field: "  hi \n"}), field) == "hi"
    assert getattr(cls(**{field: " " * 600 + "a"}), field) == "a"  # the length limit applies after stripping
    for blank in ("", "   ", "\t\n"):
        with pytest.raises(ValidationError) as caught:
            cls(**{field: blank})
        assert [(item["type"], item["msg"]) for item in caught.value.errors()] == [
            ("value_error", f"Value error, {field} must not be blank")
        ]
    with pytest.raises(ValidationError) as caught:
        cls(**{field: 5})
    assert [item["type"] for item in caught.value.errors()] == ["string_type"]
    with pytest.raises(ValidationError):
        cls(**{field: "x" * (501 if model == "QueryRequest" else 4001)})
    with pytest.raises(ValidationError):
        cls(**{field: "ok", "extra": 1})


def test_preference_checks_run_in_this_order() -> None:
    from queryshield.db.state_store import StateStore
    from queryshield.memory.preferences import PreferenceError, PreferenceStore

    store = PreferenceStore(StateStore(":memory:"))

    def put(**kwargs):
        base = {"tenant_id": "A", "principal_id": "p", "key": "display_language", "value": "en", "confirmed": True}
        return store.put(**{**base, **kwargs})

    def code(**kwargs) -> str:
        with pytest.raises(PreferenceError) as caught:
            put(**kwargs)
        return caught.value.code

    assert code(key="nope", confirmed=False, value="zz") == "unknown_preference"
    assert code(confirmed=False, value="zz") == "confirmation_required"
    for not_true in (False, 1, "true", None, 0):
        assert code(confirmed=not_true) == "confirmation_required"
    assert code(value="zz") == "invalid_preference_value"
    assert code(value=1) == "invalid_preference_value"
    assert code(key="answer_style", value="en") == "invalid_preference_value"
    assert put()["value"] == "en"


def test_preference_keys_are_checked_even_when_nothing_is_read() -> None:
    from queryshield.db.state_store import StateStore
    from queryshield.memory.preferences import PreferenceError, PreferenceStore

    store = PreferenceStore(StateStore(":memory:"))
    for call in (
        lambda: store.get(tenant_id="A", principal_id="p", key="nope"),
        lambda: store.delete(tenant_id="A", principal_id="p", key="nope"),
        lambda: store.apply_to_request(tenant_id="A", principal_id="p", key="nope", explicit_value="x"),
        lambda: store.apply_to_request(tenant_id="A", principal_id="p", key="nope", explicit_value=None),
    ):
        with pytest.raises(PreferenceError) as caught:
            call()
        assert caught.value.code == "unknown_preference"
    kwargs = {"tenant_id": "A", "principal_id": "p", "key": "answer_style"}
    assert store.apply_to_request(**kwargs, explicit_value="table") == "table"
    assert store.apply_to_request(**kwargs, explicit_value=None) is None
    store.put(**kwargs, value="concise", confirmed=True)
    assert store.apply_to_request(**kwargs, explicit_value=None) == "concise"
    assert store.apply_to_request(**kwargs, explicit_value="table") == "table"


# ---------------------------------------------------------------------------
# the two storage paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [(None, ":memory:"), ("", ":memory:"), ("   ", ":memory:"), (" /tmp/x.db ", "/tmp/x.db")])
def test_the_state_store_path_comes_from_one_reader(monkeypatch, value, expected) -> None:
    from queryshield.approval.service import state_path_from_env

    if value is None:
        monkeypatch.delenv("QUERYSHIELD_STATE_STORE_PATH", raising=False)
    else:
        monkeypatch.setenv("QUERYSHIELD_STATE_STORE_PATH", value)
    assert state_path_from_env() == expected


CALL_STORE_PATHS = [(None, ":memory:"), ("", ":memory:"), ("   ", ":memory:"), (" /tmp/c.db ", "/tmp/c.db")]


def _spy_on_call_stores(monkeypatch, value) -> list[str]:
    import queryshield.agent.call_store as call_store_module

    opened: list[str] = []
    real = call_store_module.DurableModelCallStore

    class Spy(real):
        def __init__(self, path, *args, **kwargs):
            opened.append(str(path))
            super().__init__(":memory:", *args, **kwargs)

    monkeypatch.setattr(call_store_module, "DurableModelCallStore", Spy)
    monkeypatch.setattr(api_main, "DurableModelCallStore", Spy)
    if value is None:
        monkeypatch.delenv("QUERYSHIELD_CALL_STORE_PATH", raising=False)
    else:
        monkeypatch.setenv("QUERYSHIELD_CALL_STORE_PATH", value)
    return opened


@pytest.mark.parametrize("value, expected", CALL_STORE_PATHS)
def test_the_request_call_store_path(monkeypatch, value, expected) -> None:
    opened = _spy_on_call_stores(monkeypatch, value)
    generator = api_main.get_call_store()
    next(generator)
    generator.close()
    assert opened == [expected]


@pytest.mark.parametrize("value, expected", CALL_STORE_PATHS)
def test_the_worker_call_store_path(monkeypatch, service, value, expected) -> None:
    opened = _spy_on_call_stores(monkeypatch, value)
    run = _start(service, "2026年9月支付金额是多少", [_query(), _cite])
    assert run["status"] == "SUCCEEDED"
    assert opened == [expected]


# ---------------------------------------------------------------------------
# What a run writes: the order and content of its events, per outcome
# ---------------------------------------------------------------------------

NAME_QUERY = {"type": "tool_call", "name": "query_readonly", "arguments": {"sql": NAME_SQL, "params": {}}}
APPROVER_A = {"tenant_id": "A", "principal_id": "approver-1", "role": "approver"}
MODEL_CALL, TOOL_CALL, ANSWER = "model_call", "tool_call", "answer"


def _seq(store, run: dict) -> list[tuple]:
    """(type, status, has result id, payload) for every stored event; model steps reduced to their kind."""

    out = []
    for event in store.events(str(run["run_id"]), after_event_id=0, limit=1000):
        payload = event["payload"]
        if event["type"] == "agent_step":
            payload = payload["kind"]
        elif "approval_id" in payload:
            assert payload["approval_id"] == run["approval_id"]
            payload = {"approval_id": "<id>"}
        out.append((event["type"], event["status"], event["result_id"] is not None, payload))
    return out


ACCEPTED = ("accepted", "RUNNING", False, {})
STARTED = ("step_started", "RUNNING", False, {"profile": "B1-bounded-agent", "step": "agent_run"})
STEP = ("agent_step", "RUNNING", False)
QUERIED = ("agent_step", "RUNNING", True, TOOL_CALL)
AGENT_RUN = [ACCEPTED, STARTED]
WAITING_APPROVAL = ("waiting", "WAITING_APPROVAL", False, {"approval_id": "<id>"})


def _envelope(run: dict, *keys: str) -> dict:
    return {key: run["checkpoint"].get(key) for key in keys}


def test_a_succeeded_run_writes_its_events_then_the_terminal_status(service) -> None:
    svc, _ = service
    run = _start(service, "2026年9月支付金额是多少", [_query(), _cite])
    assert run["status"] == "SUCCEEDED" and run["error_code"] is None
    assert _seq(svc.store, run) == AGENT_RUN + [
        (*STEP, MODEL_CALL), QUERIED, (*STEP, MODEL_CALL), (*STEP, ANSWER), ("terminal", "SUCCEEDED", False, {}),
    ]
    assert (run["model_call_count"], run["tool_call_count"], run["sql_exec_count"]) == (2, 1, 1)
    assert run["result"] is not None and run["facts"] is not None and run["answer"]
    assert _envelope(run, "status", "answer_status", "answer_source_ids", "agent_checkpoint") == {
        "status": "SUCCEEDED", "answer_status": "verified", "answer_source_ids": ["commerce-v1"], "agent_checkpoint": None,
    }


def test_a_waiting_user_run_keeps_its_checkpoint_and_writes_a_waiting_event(service) -> None:
    svc, _ = service
    run = _start(service, "2026年9月销售额是多少？", [_query()])
    assert run["status"] == "WAITING_USER" and run["answer"] is None and run["result"] is None
    assert _seq(svc.store, run) == AGENT_RUN + [
        (*STEP, MODEL_CALL), (*STEP, TOOL_CALL), ("waiting", "WAITING_USER", False, {}),
    ]
    assert _envelope(run, "status", "answer_status") == {"status": "WAITING_USER", "answer_status": None}
    assert run["checkpoint"]["agent_checkpoint"] is not None
    assert (run["model_call_count"], run["tool_call_count"], run["sql_exec_count"]) == (1, 1, 0)


def test_a_waiting_approval_run_binds_one_pending_approval(service) -> None:
    svc, executor = service
    run = _start(service, "查询客户姓名", [NAME_QUERY])
    assert run["status"] == "WAITING_APPROVAL" and run["error_code"] is None
    assert _seq(svc.store, run) == AGENT_RUN + [(*STEP, MODEL_CALL), (*STEP, TOOL_CALL), WAITING_APPROVAL]
    approval = svc.store.get_approval(run["approval_id"])
    assert approval["status"] == "PENDING" and approval["requester_principal_id"] == "principal-A"
    assert run["action"] == approval["action"] and approval["action"]["kind"] == "query_readonly"
    assert run["checkpoint"]["pre_approval_results"] == [] and executor.executed == []
    assert run["answer"] is None and run["result"] is None and run["facts"] is None


def test_a_missing_permission_source_ends_the_run_failed_without_an_approval(service, monkeypatch) -> None:
    svc, _ = service

    def unavailable(run):
        raise ApprovalConflict("approval_permission_unavailable", "none")

    monkeypatch.setattr(svc, "_approval_permission", unavailable)
    run = _start(service, "查询客户姓名", [NAME_QUERY])
    assert (run["status"], run["error_code"], run["approval_id"], run["action"]) == (
        "FAILED", "approval_permission_unavailable", None, None,
    )
    assert _seq(svc.store, run) == AGENT_RUN + [
        (*STEP, MODEL_CALL), (*STEP, TOOL_CALL), ("terminal", "FAILED", False, {"error_code": "approval_permission_unavailable"}),
    ]
    assert run["checkpoint"]["status"] == "FAILED" and "pre_approval_results" not in run["checkpoint"]


def test_a_cancel_requested_while_running_wins_before_the_result_is_committed(service) -> None:
    svc, _ = service

    def cancelling(messages):
        svc.cancel(run_id=next(iter(svc._active)), identity=REQUESTER_A)
        return _query()

    run = _start(service, "2026年9月支付金额是多少", [cancelling, _cite])
    assert run["status"] == "CANCELLED" and run["cancel_requested"] is True
    assert _seq(svc.store, run) == AGENT_RUN + [
        ("step_finished", "CANCEL_REQUESTED", False, {"cancel_requested": True}),
        (*STEP, MODEL_CALL), QUERIED, (*STEP, MODEL_CALL), (*STEP, ANSWER),
        ("step_finished", "CANCELLED", False, {"cancelled_before_commit": True}),
        ("terminal", "CANCELLED", False, {}),
    ]
    assert (run["answer"], run["result"], run["facts"]) == (None, None, None)
    assert run["checkpoint"]["status"] == "CANCELLED" and "answer_status" not in run["checkpoint"]
    assert (run["model_call_count"], run["tool_call_count"], run["sql_exec_count"]) == (2, 1, 1)


class _Tools:
    def __init__(self, record) -> None:
        self.record, self.closed = record, 0

    def close(self):
        self.closed += 1
        return self.record


@pytest.mark.parametrize("record", [None, {"session": "x"}])
def test_a_failed_execution_writes_the_metadata_record_before_the_terminal_event(service, monkeypatch, record) -> None:
    import queryshield.approval.service as service_module

    svc, _ = service
    made: list[_Tools] = []

    def tools(deps, executor=None, metadata=None):
        made.append(_Tools(record))
        return made[0]

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(service_module, "product_tools", tools)
    monkeypatch.setattr(service_module, "run_profile", boom)
    run = _start(service, "2026年9月支付金额是多少", [_query()])
    assert (run["status"], run["error_code"]) == ("FAILED", "execution_failed")
    assert _seq(svc.store, run) == AGENT_RUN + (
        [("metadata_session", "RUNNING", False, record)] if record else []
    ) + [("terminal", "FAILED", False, {"error_code": "execution_failed"})]
    assert made[0].closed == 2
    assert (run["model_call_count"], run["sql_exec_count"]) == (0, 0) and run["checkpoint"]["status"] == "RUNNING"


# ---------------------------------------------------------------------------
# Approval: what each decision and each failure writes
# ---------------------------------------------------------------------------


def _pending(service, question="查询客户姓名", steps=None):
    return _start(service, question, steps or [NAME_QUERY])


def _approve(service, run, *, decision="approve", identity=APPROVER_A, approval_id=None):
    svc, _ = service
    return svc.approve(
        run_id=run["run_id"], approval_id=approval_id or run["approval_id"], identity=identity, decision=decision
    )


def test_an_approved_query_runs_once_for_the_requester_and_keeps_the_earlier_metric_result(service) -> None:
    svc, executor = service
    run = _pending(service, "2026年9月支付金额和客户姓名", [_query(), NAME_QUERY])
    assert run["status"] == "WAITING_APPROVAL" and len(run["checkpoint"]["pre_approval_results"]) == 1
    done = _approve(service, run)
    assert done["status"] == "SUCCEEDED" and done["error_code"] is None
    assert _seq(svc.store, done) == AGENT_RUN + [
        (*STEP, MODEL_CALL), QUERIED, (*STEP, MODEL_CALL), (*STEP, TOOL_CALL), WAITING_APPROVAL,
        ("step_finished", "SUCCEEDED", True, {"step": "approved_query"}),
        ("terminal", "SUCCEEDED", True, {}),
    ]
    assert [fact["metric_id"] for fact in done["facts"]["facts"]] == ["gross_fen"]
    assert len(done["result"]["supporting_results"]) == 1
    assert (done["sql_exec_count"], done["model_call_count"], done["tool_call_count"]) == (2, 2, 2)
    assert done["answer"] and _envelope(done, "status", "answer_status", "answer_source_ids") == {
        "status": "WAITING_APPROVAL", "answer_status": "verified", "answer_source_ids": [],
    }
    assert svc.store.get_approval(run["approval_id"])["status"] == "APPROVED"
    assert len(executor.executed) == 2
    # A replay of the consumed decision returns the run and runs nothing.
    assert _approve(service, run)["status"] == "SUCCEEDED" and len(executor.executed) == 2


def test_an_approved_query_without_a_metric_is_unverified(service) -> None:
    svc, _ = service
    done = _approve(service, _pending(service))
    assert done["status"] == "SUCCEEDED" and done["facts"] is None
    assert _envelope(done, "answer_status", "answer_source_ids") == {"answer_status": "unverified", "answer_source_ids": []}
    assert done["answer"].startswith("审批通过，已执行只读查询：返回 ")


def test_a_rejected_approval_denies_the_run_without_running_the_query(service) -> None:
    svc, executor = service
    run = _pending(service)
    done = _approve(service, run, decision="reject")
    assert done["status"] == "DENIED" and executor.executed == []
    assert _seq(svc.store, done) == AGENT_RUN + [
        (*STEP, MODEL_CALL), (*STEP, TOOL_CALL), WAITING_APPROVAL, ("terminal", "DENIED", False, {"approval_id": "<id>"}),
    ]
    assert svc.store.get_approval(run["approval_id"])["status"] == "REJECTED"


@pytest.mark.parametrize(
    "exc, code",
    [
        (ToolError("invalid_argument", "x"), "invalid_argument"),
        (RuntimeConfigurationError("invalid_database_configuration", "x"), "invalid_database_configuration"),
    ],
)
def test_an_approved_query_that_fails_ends_the_run_with_the_error_code(service, monkeypatch, exc, code) -> None:
    svc, executor = service
    run = _pending(service)

    def failing(*args, **kwargs):
        raise exc

    monkeypatch.setattr(executor, "execute", failing)
    done = _approve(service, run)
    assert (done["status"], done["error_code"]) == ("FAILED", code)
    assert _seq(svc.store, done) == AGENT_RUN + [
        (*STEP, MODEL_CALL), (*STEP, TOOL_CALL), WAITING_APPROVAL, ("terminal", "FAILED", False, {"error_code": code}),
    ]
    assert done["sql_exec_count"] == 0 and (done["answer"], done["result"], done["facts"]) == (None, None, None)
    assert svc.store.get_approval(run["approval_id"])["status"] == "APPROVED"


def test_an_unexpected_error_in_an_approved_query_leaves_the_run_waiting(service, monkeypatch) -> None:
    svc, executor = service
    run = _pending(service)

    def failing(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(executor, "execute", failing)
    with pytest.raises(RuntimeError):
        _approve(service, run)
    assert svc.store.get_run(run["run_id"])["status"] == "WAITING_APPROVAL"
    assert svc.store.get_approval(run["approval_id"])["status"] == "APPROVED"
    assert svc._active_count() == 0


def test_a_cancel_during_an_approved_query_wins_before_the_result_is_committed(service, monkeypatch) -> None:
    svc, executor = service
    run = _pending(service)
    original = executor.execute

    def cancelling(*args, **kwargs):
        svc.cancel(run_id=run["run_id"], identity=REQUESTER_A)
        return original(*args, **kwargs)

    monkeypatch.setattr(executor, "execute", cancelling)
    done = _approve(service, run)
    assert done["status"] == "CANCELLED"
    assert _seq(svc.store, done) == AGENT_RUN + [
        (*STEP, MODEL_CALL), (*STEP, TOOL_CALL), WAITING_APPROVAL,
        ("terminal", "CANCELLED", False, {}),
        ("step_finished", "CANCELLED", False, {"cancelled_before_commit": True}),
        ("terminal", "CANCELLED", False, {}),
    ]
    assert (done["answer"], done["result"], done["facts"]) == (None, None, None) and done["sql_exec_count"] == 1


# ---------------------------------------------------------------------------
# Resume: what each answer writes
# ---------------------------------------------------------------------------


def test_resume_events_for_each_of_its_three_outcomes(service) -> None:
    from test_clarification import COUNT_SQL

    svc, executor = service
    waiting = _start(service, "2026年9月销售额是多少？", [_query()])
    before = _seq(svc.store, waiting)

    still = _resume(service, waiting, "2026年9月", [])
    assert still["status"] == "WAITING_USER"
    assert _seq(svc.store, still) == before + [("waiting", "WAITING_USER", False, {})]

    done = _resume(service, waiting, "按支付金额", [_query(), _cite])
    assert done["status"] == "SUCCEEDED"
    assert _seq(svc.store, done)[len(before) + 1:] == [
        (*STEP, MODEL_CALL), QUERIED, (*STEP, MODEL_CALL), (*STEP, ANSWER), ("terminal", "SUCCEEDED", False, {}),
    ]
    assert (done["model_call_count"], done["tool_call_count"]) == (3, 2)
    assert done["checkpoint"]["clarified_metric"] == "gross_fen"

    run = _start(service, "2026年9月订单数是多少", [_query(COUNT_SQL, ("paid_count",))])
    ended = _resume(service, run, "包含已取消", [])
    assert (ended["status"], ended["error_code"]) == ("FAILED", "clarification_value_unsupported")
    assert ended["answer"]
    assert _seq(svc.store, ended)[-1] == ("terminal", "FAILED", False, {"error_code": "clarification_value_unsupported"})
    assert executor.executed == [executor.executed[0]]  # only the one gross query above ran SQL


# ---------------------------------------------------------------------------
# The checks of an approval decision, in order
# ---------------------------------------------------------------------------

from queryshield.approval.service import ApprovalNotFound, ObjectNotFound, RunAuthorizationError  # noqa: E402
from queryshield.knowledge.runtime import product_knowledge  # noqa: E402
from queryshield.knowledge.snapshots import KnowledgeSnapshotRepository  # noqa: E402

PERMISSION_SOURCE = "semantic-sensitive-customer-name"
OWN_APPROVER = {"tenant_id": "A", "principal_id": "principal-A", "role": "approver"}  # the requester's own principal
FOREIGN_APPROVER = {"tenant_id": "B", "principal_id": "approver-b", "role": "approver"}
SAME_TENANT_REQUESTER = {"tenant_id": "A", "principal_id": "other-requester", "role": "requester"}


def _fails(service, run, kind: type, code: str | None, text: str | None = None, **kwargs) -> None:
    svc, executor = service
    ran = len(executor.executed)
    with pytest.raises(kind) as caught:
        _approve(service, run, **kwargs)
    if code is not None:
        assert caught.value.code == code
    if text is not None:
        assert str(caught.value) == text
    assert len(executor.executed) == ran  # nothing was executed by a refused decision


def _untouched(service, run) -> None:
    svc, _ = service
    assert svc.store.get_approval(run["approval_id"])["status"] == "PENDING"
    assert svc.store.get_run(run["run_id"])["status"] == "WAITING_APPROVAL"


def _expire(svc, run) -> None:
    svc.store._connection.execute(
        "UPDATE approvals SET expires_at = ? WHERE approval_id = ?", ("2026-09-22T00:00:00Z", run["approval_id"])
    )


def _with_permission(service):
    svc, _ = service
    svc.store.publish_snapshot(product_knowledge(demo=False).snapshot.as_dict())
    run = _pending(service)
    assert run["action"]["permission_source_id"] == PERMISSION_SOURCE
    return svc, run


def test_an_unknown_run_or_another_tenant_is_not_found_before_anything_else(service) -> None:
    svc, _ = service
    run = _pending(service)
    ghost = {**run, "run_id": "run-ghost"}
    _fails(service, ghost, ObjectNotFound, "not_found", "object is not visible")
    _fails(service, run, ObjectNotFound, "not_found", identity=FOREIGN_APPROVER)
    _fails(service, run, ObjectNotFound, "not_found", identity={**FOREIGN_APPROVER, "role": "requester"})
    _untouched(service, run)


def test_only_a_same_tenant_approver_may_decide(service) -> None:
    run = _pending(service)
    _fails(service, run, RunAuthorizationError, "forbidden", "only a same-tenant approver can decide", identity=SAME_TENANT_REQUESTER)
    _fails(service, run, RunAuthorizationError, "forbidden", identity=REQUESTER_A)
    _untouched(service, run)


def test_the_approval_must_exist_and_belong_to_the_run(service) -> None:
    run, other = _pending(service), _pending(service)
    _fails(service, run, ApprovalNotFound, "approval_not_found", approval_id="approval-ghost")
    _fails(service, run, ApprovalNotFound, "approval_not_found", approval_id=other["approval_id"])
    _untouched(service, run)


def test_the_run_must_still_be_waiting_for_approval_before_self_approval_is_judged(service) -> None:
    svc, _ = service
    run = _pending(service)
    svc.cancel(run_id=run["run_id"], identity=REQUESTER_A)
    _fails(service, run, ApprovalConflict, "invalid_run_state", "run is not waiting for approval")
    _fails(service, run, ApprovalConflict, "invalid_run_state", identity=OWN_APPROVER)
    assert svc.store.get_approval(run["approval_id"])["status"] == "PENDING"


def test_an_approver_cannot_decide_their_own_request_even_when_the_approval_is_expired(service) -> None:
    svc, _ = service
    run = _pending(service)
    _fails(service, run, RunAuthorizationError, "forbidden", "requester cannot approve its own action", identity=OWN_APPROVER)
    _expire(svc, run)
    _fails(service, run, RunAuthorizationError, "forbidden", identity=OWN_APPROVER)
    assert svc.store.get_approval(run["approval_id"])["status"] == "PENDING"


def test_an_expired_approval_is_marked_expired_and_refused(service) -> None:
    svc, _ = service
    run = _pending(service)
    _expire(svc, run)
    _fails(service, run, ApprovalConflict, "approval_stale", "approval has expired")
    assert svc.store.get_approval(run["approval_id"])["status"] == "EXPIRED"
    _fails(service, run, ApprovalConflict, "approval_stale", "approval has expired")  # the EXPIRED status itself
    assert svc.store.get_run(run["run_id"])["status"] == "WAITING_APPROVAL"


def test_a_decided_approval_is_a_replay_even_when_the_run_is_no_longer_waiting(service) -> None:
    svc, executor = service
    run = _pending(service)
    _approve(service, run, decision="reject")
    ran = len(executor.executed)
    again = _approve(service, run)
    assert again["status"] == "DENIED" and len(executor.executed) == ran


@pytest.mark.parametrize("change", ["revoke", "roles", "tenant"])
def test_a_changed_permission_refuses_the_decision_and_leaves_it_pending(service, change) -> None:
    svc, run = _with_permission(service)
    repository = KnowledgeSnapshotRepository(svc.store)
    if change == "revoke":
        repository.revoke(PERMISSION_SOURCE)
    elif change == "roles":
        svc.store.set_source_acl(PERMISSION_SOURCE, allowed_roles=("approver", "requester"))
    else:
        svc.store.set_source_acl(PERMISSION_SOURCE, tenant_scope="B")
    _fails(service, run, ApprovalConflict, "authorization_revoked")
    _untouched(service, run)


def test_an_approval_whose_snapshot_is_gone_is_refused_even_when_the_acl_is_unchanged(service) -> None:
    svc, run = _with_permission(service)
    svc.store._connection.execute(
        "UPDATE approvals SET knowledge_snapshot_id = ? WHERE approval_id = ?", ("snapshot-ghost", run["approval_id"])
    )
    _fails(service, run, ApprovalConflict, "authorization_revoked", "approval permission is no longer current")
    _untouched(service, run)


def test_an_approval_whose_acl_row_is_gone_is_refused(service) -> None:
    svc, run = _with_permission(service)
    svc.store._connection.execute("DELETE FROM knowledge_acl WHERE source_id = ?", (PERMISSION_SOURCE,))
    _fails(service, run, ApprovalConflict, "authorization_revoked")
    _untouched(service, run)


def test_the_permission_check_comes_after_expiry_and_before_the_bound_action(service) -> None:
    svc, run = _with_permission(service)
    KnowledgeSnapshotRepository(svc.store).revoke(PERMISSION_SOURCE)
    svc.store.update_run(run["run_id"], action_json=json.dumps({**run["action"], "sql": "SELECT 1"}))
    _fails(service, run, ApprovalConflict, "authorization_revoked")  # not approval_stale
    _expire(svc, run)
    _fails(service, run, ApprovalConflict, "approval_stale", "approval has expired")  # expiry first
    assert svc.store.get_approval(run["approval_id"])["status"] == "EXPIRED"


def test_a_current_permission_lets_the_decision_through(service) -> None:
    svc, run = _with_permission(service)
    assert _approve(service, run)["status"] == "SUCCEEDED"


def test_a_changed_action_is_refused_on_approve_but_a_rejection_needs_no_binding(service) -> None:
    svc, executor = service
    run = _pending(service)
    svc.store.update_run(run["run_id"], action_json=json.dumps({**run["action"], "sql": "SELECT 1"}))
    _fails(service, run, ApprovalConflict, "approval_stale", "approved action no longer matches the server-bound action")
    _untouched(service, run)
    assert _approve(service, run, decision="reject")["status"] == "DENIED"
    assert executor.executed == []


# ---------------------------------------------------------------------------
# The checks of a resume, in order
# ---------------------------------------------------------------------------

OTHER_PRINCIPAL = {**REQUESTER_A, "principal_id": "someone-else"}
OTHER_ROLE = {**REQUESTER_A, "role": "approver"}
OTHER_TENANT = {**REQUESTER_A, "tenant_id": "B"}


def _waiting_user(service):
    return _start(service, "2026年9月销售额是多少？", [_query()])


def _edit(svc, run, *, envelope=None, agent=None, config=None, raw_envelope=None) -> None:
    """Change the stored checkpoint or run configuration of a waiting run."""

    stored = svc.store.get_run(run["run_id"])
    checkpoint = json.loads(json.dumps(stored["checkpoint"]))
    if agent is not None:
        agent(checkpoint["agent_checkpoint"])
    if envelope is not None:
        envelope(checkpoint)
    fields: dict = {"checkpoint_json": raw_envelope if raw_envelope is not None else json.dumps(checkpoint)}
    if config is not None:
        run_config = json.loads(json.dumps(stored["run_config"]))
        config(run_config)
        fields["run_config_json"] = json.dumps(run_config)
    svc.store.update_run(run["run_id"], **fields)


def _resume_fails(service, run, kind: type, code: str, text: str | None = None, *, identity=REQUESTER_A) -> None:
    svc, executor = service
    with pytest.raises(kind) as caught:
        svc.resume_waiting_user(
            run_id=run["run_id"], answer="按支付金额", identity=identity, model=_NoModel(), call_store=None, executor=executor
        )
    assert caught.value.code == code
    if text is not None:
        assert str(caught.value) == text
    assert svc.store.get_run(run["run_id"])["status"] == "WAITING_USER"
    assert executor.executed == []


class _NoModel:
    mode = "fake"

    def complete(self, *args, **kwargs):  # pragma: no cover - a refused resume never reaches the model
        raise AssertionError("the model must not be called")


def test_resume_is_only_for_the_owner_and_role_of_a_waiting_run(service) -> None:
    svc, _ = service
    run = _waiting_user(service)
    for identity in (OTHER_PRINCIPAL, OTHER_ROLE, OTHER_TENANT):
        _resume_fails(service, run, ObjectNotFound, "not_found", identity=identity)
    with pytest.raises(ObjectNotFound):
        svc.resume_waiting_user(
            run_id="run-ghost", answer="a", identity=REQUESTER_A, model=_NoModel(), call_store=None, executor=None
        )
    svc.cancel(run_id=run["run_id"], identity=REQUESTER_A)
    with pytest.raises(ApprovalConflict) as caught:
        svc.resume_waiting_user(
            run_id=run["run_id"], answer="a", identity=REQUESTER_A, model=_NoModel(), call_store=None, executor=None
        )
    assert (caught.value.code, str(caught.value)) == ("invalid_run_state", "run is not waiting for user input")


@pytest.mark.parametrize(
    "edit, code, text",
    [
        ({"raw_envelope": "null"}, "checkpoint_invalid", "waiting run has no server checkpoint"),
        ({"envelope": lambda e: e.update(agent_checkpoint=None)}, "not_supported", "this profile has no interactive continuation"),
        (
            {"config": lambda c: c.update(agent_run_config="bad"), "agent": lambda a: a.update(run_config=5)},
            "checkpoint_invalid", "waiting run profile is invalid",
        ),
        (
            {"config": lambda c: c["agent_run_config"].update(profile="queryshield-w03-hybrid-v1")},
            "not_supported", "the single-pass profile cannot resume a task",
        ),
        (
            {"agent": lambda a: a["run_config"].update(prompt_version="other")},
            "checkpoint_invalid", "checkpoint profile differs from the stored run profile",
        ),
        (
            {"envelope": lambda e: e.update(clarified_metric="bogus")},
            "checkpoint_invalid", "server metric slot is invalid",
        ),
        ({"agent": lambda a: a.pop("model_call_count")}, "checkpoint_invalid", "checkpoint budget counters are missing"),
        ({"agent": lambda a: a.update(model_call_count="1")}, "checkpoint_invalid", "checkpoint budget counters do not match the run"),
        ({"agent": lambda a: a.update(tool_call_count=9)}, "checkpoint_invalid", "checkpoint budget counters do not match the run"),
        ({"agent": lambda a: a.update(events="x")}, "checkpoint_invalid", "checkpoint budget counters do not match the run"),
    ],
)
def test_each_checkpoint_check_refuses_with_its_own_code_and_text(service, edit, code, text) -> None:
    svc, _ = service
    run = _waiting_user(service)
    _edit(svc, run, **edit)
    _resume_fails(service, run, ApprovalConflict, code, text)


def test_the_resume_checks_keep_their_order(service) -> None:
    svc, _ = service
    run = _waiting_user(service)
    # profile, then checkpoint/run configuration, then the metric slot, then the budget.
    _edit(
        svc, run,
        envelope=lambda e: e.update(clarified_metric="bogus"),
        agent=lambda a: (a.pop("model_call_count"), a["run_config"].update(prompt_version="other")),
    )
    _resume_fails(service, run, ApprovalConflict, "checkpoint_invalid", "checkpoint profile differs from the stored run profile")
    _edit(svc, run, agent=lambda a: a.update(run_config=svc.store.get_run(run["run_id"])["run_config"]["agent_run_config"]))
    _resume_fails(service, run, ApprovalConflict, "checkpoint_invalid", "server metric slot is invalid")
    _edit(svc, run, envelope=lambda e: e.pop("clarified_metric"))
    _resume_fails(service, run, ApprovalConflict, "checkpoint_invalid", "checkpoint budget counters are missing")


# ---------------------------------------------------------------------------
# The metric binding a resume computes on the server (window from the question and the answer)
# ---------------------------------------------------------------------------


def _window(start: str, end: str) -> dict[str, str]:
    return {"start": f"{start}T00:00:00Z", "end": f"{end}T00:00:00Z"}


RESUME_WINDOWS = [
    ("销售额是多少？", "按支付金额，2026年12月", _window("2026-12-01", "2027-01-01")),  # December rolls into January
    ("销售额是多少？", "按支付金额 2027年1月", _window("2027-01-01", "2027-02-01")),
    ("销售额是多少？", "按支付金额 2026年03月", _window("2026-03-01", "2026-04-01")),
    ("2026年11月销售额是多少？", "按支付金额", _window("2026-11-01", "2026-12-01")),  # the month is in the question
    ("销售额是多少？", "按支付金额", _window("2026-09-01", "2026-10-01")),  # no month anywhere: the server default
]


@pytest.mark.parametrize("question, answer, window", RESUME_WINDOWS)
def test_resume_binds_the_chosen_metric_to_the_window_the_server_derives(service, question, answer, window) -> None:
    from queryshield.approval.service import BOUND_CATALOG_VERSION

    run = _start(service, question, [_query(window=window)], window=window)
    assert run["status"] == "WAITING_USER"
    done = _resume(service, run, answer, [_query(window=window), _cite])
    assert done["status"] == "SUCCEEDED", done["error_code"]
    assert done["result"]["metric_bindings"] == [
        {
            "catalog_source_id": "commerce-v1",
            "catalog_version": BOUND_CATALOG_VERSION,
            "metric_id": "gross_fen",
            "plan_id": None,
            "result_position": "gross_fen",
            "time_window": {**window, "timezone": "UTC"},
            "unit": "CNY_fen",
        }
    ]


# ---------------------------------------------------------------------------
# The Fake database that the frozen checks import from the approval service
# ---------------------------------------------------------------------------


def test_the_fake_executor_keeps_its_import_path_and_its_result_shape() -> None:
    from queryshield.agent.proposals import ExecutionContext

    executor = FixtureQueryExecutor(clock=lambda: datetime(2026, 9, 23, tzinfo=timezone.utc))
    context = ExecutionContext(run_id="r", tenant_id="A", principal_id="p", role="requester")
    result = executor.execute(
        "SELECT COUNT(*) AS paid_count, COALESCE(SUM(o.amount_fen), 0) AS gross_fen FROM orders AS o", context=context
    )
    assert executor.sql_calls == 1
    assert result.rows == ({"paid_count": 2, "gross_fen": 15000},)
    assert result.evidence.rows == result.rows and result.evidence.observed_at == datetime(2026, 9, 23, tzinfo=timezone.utc)
    names = executor.execute("SELECT c.customer_id, c.name FROM customers AS c", context=context)
    assert names.rows == ({"customer_id": "c1", "name": "甲"}, {"customer_id": "c2", "name": "乙"})
    assert executor.sql_calls == 2


# ---------------------------------------------------------------------------
# Names other code imports from these modules, and how it reaches them
# ---------------------------------------------------------------------------

IMPORTED = {
    "queryshield.api.main": [
        "app", "get_call_store", "get_guarded_executor", "get_model_provider", "get_retriever_source", "get_run_service",
        "get_agent_profile", "_answer_fields", "check_sensitive_access", "execute_query_proposal", "_application_lifespan",
    ],
    "queryshield.approval.models": ["ApprovalRequest", "EmptyRequest", "PreferenceUpdate", "ResumeRequest"],
    "queryshield.approval.service": [
        "ApprovalConflict", "ApprovalNotFound", "FixtureQueryExecutor", "MAX_ACTIVE_RUNS", "ObjectNotFound", "RunAuthorizationError",
        "RunIdentity", "RunService", "BOUND_CATALOG_VERSION", "DEFAULT_KNOWLEDGE_SNAPSHOT", "BOUND_POLICY_VERSION", "STATE_VERSION",
        "_answer_envelope", "_approved_answer", "_succeeded_without_evidence", "_waiting_clarification_rule", "action_hash",
        "build_pending_action", "pending_call_from_action", "reset_shared_state_stores", "shared_state_store",
        "shared_run_service", "state_path_from_env",
    ],
    "queryshield.memory.preferences": ["ALLOWED_PREFERENCE_VALUES", "PREFERENCE_KEYS", "PreferenceError", "PreferenceStore"],
    "queryshield.models.queries": ["QueryProposalRequest", "QueryRequest"],
}
SERVICE_METHODS = [
    "run_sync", "start_async", "resume_waiting_user", "approve", "_approve_locked", "cancel", "visible_run",
    "_approval_permission", "recover_parallel_groups", "recover", "new_executor", "default_dependencies",
]


@pytest.mark.parametrize("module, names", IMPORTED.items())
def test_the_names_other_code_imports_still_exist(module, names) -> None:
    import importlib

    imported = importlib.import_module(module)
    assert [name for name in names if not hasattr(imported, name)] == []


def test_the_service_keeps_the_methods_and_attributes_others_use() -> None:
    from queryshield.db.state_store import StateStore

    service = RunService(store=StateStore(":memory:"), mode="fake")
    assert [name for name in SERVICE_METHODS if not callable(getattr(service, name, None))] == []
    assert hasattr(service, "_approval_lock")
    for attribute in ("store", "mode", "clock", "_executor_factory", "_metadata_tools"):
        assert hasattr(service, attribute), attribute
    assert hasattr(FixtureQueryExecutor(), "sql_calls")


def test_the_evaluation_can_still_swap_the_dependencies_and_the_provider() -> None:
    """It overrides by function object and assigns ``get_model_provider`` on the module."""

    used = {
        route_dependency.call
        for route in app.routes
        if hasattr(route, "dependant")
        for route_dependency in _walk(route.dependant)
    }
    for name in ("get_call_store", "get_guarded_executor", "get_model_provider", "get_retriever_source", "get_run_service"):
        assert getattr(api_main, name) in used, name
    assert importlib_import("queryshield.api.main").app is app


def _walk(dependant):
    for sub in dependant.dependencies:
        yield sub
        yield from _walk(sub)


def importlib_import(name: str):
    import importlib

    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# The shared pieces the tidy-up introduced
# ---------------------------------------------------------------------------

INDEX_MESSAGE = "query_readonly params keys must be consecutive indexes starting at zero without gaps"


def _call(params: object):
    from queryshield.agent.proposals import ToolCallAction

    return api_main._proposal_params(ToolCallAction("query_readonly", {"sql": "SELECT 1", "params": params}))


@pytest.mark.parametrize(
    "params, expected",
    [({}, ()), ({"0": "a"}, ("a",)), ({"1": "b", "0": "a"}, ("a", "b")), ({str(i): i for i in range(11)}, tuple(range(11)))],
)
def test_proposal_params_are_the_values_in_index_order(params, expected) -> None:
    assert _call(params) == expected


@pytest.mark.parametrize("params", [{"a": 1}, {"-1": 1}, {"01": 1}, {"٠": 1}, {"0": 1, "2": 3}, {"1": 1}])
def test_proposal_params_with_other_keys_fail_with_one_message(params) -> None:
    from queryshield.agent.proposals import ProposalParseError

    with pytest.raises(ProposalParseError) as caught:
        _call(params)
    assert caught.value.code == "invalid_field" and str(caught.value) == f"invalid_field: {INDEX_MESSAGE}"


def test_normalized_answer_is_the_one_rule_for_what_is_written_and_read() -> None:
    from queryshield.approval.service import normalized_answer

    assert normalized_answer("verified", ["a", 1, "b"]) == ("verified", ["a", "b"])
    assert normalized_answer("no_data", None) == ("no_data", [])
    assert normalized_answer("bogus", ("a",)) == ("unverified", [])
    assert normalized_answer(None, []) == ("unverified", [])


def test_the_terminal_statuses_are_one_set_shared_with_the_stream() -> None:
    from queryshield.approval.service import TERMINAL_STATUSES

    assert TERMINAL_STATUSES == {"SUCCEEDED", "DENIED", "FAILED", "LIMIT_REACHED", "CANCELLED", "USAGE_UNKNOWN"}
    assert api_main.TERMINAL_STATUSES is TERMINAL_STATUSES


@pytest.mark.parametrize(
    "run, expected",
    [
        ({"status": "CANCEL_REQUESTED"}, True),
        ({"status": "CANCELLED"}, True),
        ({"status": "RUNNING", "cancel_requested": True}, True),
        ({"status": "RUNNING", "cancel_requested": False}, False),
        ({"status": "RUNNING"}, False),
        ({"status": "SUCCEEDED"}, False),
    ],
)
def test_cancel_requested(run, expected) -> None:
    from queryshield.approval.service import _cancel_requested

    assert _cancel_requested(run) is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", _window("2026-09-01", "2026-10-01")),
        ("2026年9月", _window("2026-09-01", "2026-10-01")),
        ("2026年09月", _window("2026-09-01", "2026-10-01")),
        ("2026年12月", _window("2026-12-01", "2027-01-01")),
        ("2099年12月", _window("2099-12-01", "2100-01-01")),
        ("2000年1月", _window("2000-01-01", "2000-02-01")),
        ("2026年1月 and 2026年5月", _window("2026-01-01", "2026-02-01")),  # the first month named
        ("2026年13月", _window("2026-09-01", "2026-10-01")),  # not a month: the default window
        ("1999年5月", _window("2026-09-01", "2026-10-01")),
    ],
)
def test_the_month_window(text, expected) -> None:
    from queryshield.approval.service import _month_window

    assert _month_window(text) == {**expected, "timezone": "UTC"}


@pytest.mark.parametrize("metric_id, plan_id", [("gross_fen", None), ("paid_count", None), ("net_fen", "commerce-v1.net_fen.v1")])
def test_the_clarified_binding_is_built_from_the_catalog(metric_id, plan_id) -> None:
    from queryshield.approval.service import BOUND_CATALOG_VERSION, _clarified_binding
    from queryshield.catalog import load_default_catalog

    binding = _clarified_binding(load_default_catalog(), metric_id, {"question": "2026年12月"}, "")
    assert binding.as_dict() == {
        "metric_id": metric_id,
        "result_position": metric_id,
        "unit": "count" if metric_id == "paid_count" else "CNY_fen",
        "time_window": {"start": "2026-12-01T00:00:00Z", "end": "2027-01-01T00:00:00Z", "timezone": "UTC"},
        "catalog_source_id": "commerce-v1",
        "catalog_version": BOUND_CATALOG_VERSION,
        "plan_id": plan_id,
    }
    assert list(binding.as_dict()) == [
        "metric_id", "result_position", "unit", "time_window", "catalog_source_id", "catalog_version", "plan_id",
    ]


def test_an_unknown_metric_slot_is_refused_before_the_checkpoint_text_is_read() -> None:
    from queryshield.approval.service import _clarified_binding
    from queryshield.catalog import load_default_catalog

    for metric in ("bogus", None, 5, ["gross_fen"]):
        with pytest.raises(ApprovalConflict) as caught:
            _clarified_binding(load_default_catalog(), metric, {"clarifications": 5}, "x")
        assert (caught.value.code, str(caught.value)) == ("checkpoint_invalid", "server metric slot is invalid")


def test_the_fake_executor_lives_in_its_own_module_and_is_re_exported() -> None:
    from dataclasses import FrozenInstanceError

    from queryshield.agent.proposals import ExecutionContext
    from queryshield.approval import fixture_executor, service as service_module, versions

    assert service_module.FixtureQueryExecutor is fixture_executor.FixtureQueryExecutor
    assert (service_module.BOUND_POLICY_VERSION, service_module.BOUND_CATALOG_VERSION) == (
        versions.BOUND_POLICY_VERSION, versions.BOUND_CATALOG_VERSION,
    )
    assert versions.BOUND_POLICY_VERSION == "qs-sql-v1"
    result = fixture_executor.FixtureQueryExecutor().execute(
        "SELECT COUNT(*) AS paid_count FROM orders AS o",
        context=ExecutionContext(run_id="r", tenant_id="B", principal_id="p", role="requester"),
    )
    assert isinstance(result, fixture_executor.FixtureResult) and result.rows == ({"paid_count": 1},)
    with pytest.raises(FrozenInstanceError):
        result.rows = ()


# ---------------------------------------------------------------------------
# The order of the writes: a terminal event is stored before the terminal status
# ---------------------------------------------------------------------------


def _record_writes(store, monkeypatch) -> list[tuple]:
    """Every event and every run update, in the order the service wrote them."""

    writes: list[tuple] = []
    append_event, update_run = store.append_event, store.update_run

    def appending(run_id, event_type, status, **kwargs):
        writes.append(("event", event_type, status))
        return append_event(run_id, event_type, status, **kwargs)

    def updating(run_id, **fields):
        writes.append(("update", fields.get("status")))
        return update_run(run_id, **fields)

    monkeypatch.setattr(store, "append_event", appending)
    monkeypatch.setattr(store, "update_run", updating)
    return writes


def test_a_failed_execution_stores_the_terminal_event_before_the_status(service, monkeypatch) -> None:
    import queryshield.approval.service as service_module

    svc, _ = service
    writes = _record_writes(svc.store, monkeypatch)
    monkeypatch.setattr(service_module, "run_profile", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("x")))
    _start(service, "2026年9月支付金额是多少", [_query()])
    assert writes[-2:] == [("event", "terminal", "FAILED"), ("update", "FAILED")]


def test_a_cancel_before_commit_stores_both_events_before_the_status(service, monkeypatch) -> None:
    svc, _ = service
    writes = _record_writes(svc.store, monkeypatch)

    def cancelling(messages):
        svc.cancel(run_id=next(iter(svc._active)), identity=REQUESTER_A)
        return _query()

    _start(service, "2026年9月支付金额是多少", [cancelling, _cite])
    assert writes[-3:] == [
        ("event", "step_finished", "CANCELLED"), ("event", "terminal", "CANCELLED"), ("update", "CANCELLED"),
    ]


def test_an_approved_query_stores_its_events_before_the_succeeded_status(service, monkeypatch) -> None:
    svc, executor = service
    run = _pending(service)
    writes = _record_writes(svc.store, monkeypatch)
    _approve(service, run)
    assert writes[-3:] == [
        ("event", "step_finished", "SUCCEEDED"), ("event", "terminal", "SUCCEEDED"), ("update", "SUCCEEDED"),
    ]


def test_an_approved_query_that_fails_stores_the_terminal_event_before_the_status(service, monkeypatch) -> None:
    svc, executor = service
    run = _pending(service)
    writes = _record_writes(svc.store, monkeypatch)
    monkeypatch.setattr(executor, "execute", lambda *a, **k: (_ for _ in ()).throw(ToolError("invalid_argument", "x")))
    _approve(service, run)
    assert [item for item in writes if item[0] != "event" or item[1] != "decide"][-2:] == [
        ("event", "terminal", "FAILED"), ("update", "FAILED"),
    ]


def test_a_committed_outcome_stores_its_agent_events_before_the_status(service, monkeypatch) -> None:
    svc, _ = service
    writes = _record_writes(svc.store, monkeypatch)
    _start(service, "2026年9月支付金额是多少", [_query(), _cite])
    assert writes[-2:] == [("event", "terminal", "SUCCEEDED"), ("update", "SUCCEEDED")]
    assert [item for item in writes if item[0] == "update"][-1] == ("update", "SUCCEEDED")
    assert writes.index(("event", "terminal", "SUCCEEDED")) > max(
        index for index, item in enumerate(writes) if item[:2] == ("event", "agent_step")
    )
