"""Sync and async /queries run the product Agent runtime (Fake model, fixture DB)."""

from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from queryshield.api.main import app, get_guarded_executor, get_model_provider, get_run_service
from queryshield.approval.service import MAX_ACTIVE_RUNS, reset_shared_state_stores, shared_run_service
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.providers.contracts import ModelCallResult
from queryshield.providers.fake_model import FakeModel


REQUESTER = "b2b-a-requester"
APPROVER = "b2b-a-approver"
OTHER = "b2b-b-requester"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYSHIELD_STATE_STORE_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    monkeypatch.delenv("QUERYSHIELD_AGENT_PROFILE", raising=False)
    monkeypatch.delenv("QUERYSHIELD_RETRIEVAL", raising=False)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", REQUESTER)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_APPROVER", APPROVER)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_B_REQUESTER", OTHER)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_B_APPROVER", "b2b-b-approver")
    reset_shared_state_stores()
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
    reset_shared_state_stores()


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def ask(client: TestClient, question: str, *, token: str = REQUESTER, asynchronous: bool = False, **extra):
    headers = auth(token)
    if asynchronous:
        headers["Prefer"] = "respond-async"
    return client.post("/queries", headers=headers, json={"question": question, **extra})


def wait(client: TestClient, run_id: str, wanted: set[str], token: str = REQUESTER, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/runs/{run_id}", headers=auth(token)).json()
        if body.get("status") in wanted:
            return body
        time.sleep(0.02)
    raise AssertionError(f"{run_id} did not reach {wanted}: {body.get('status')}")


def fact_values(body: dict) -> list[tuple[str, object]]:
    facts = body.get("facts") or {}
    return sorted((item["metric_id"], item["value"]) for item in facts.get("facts", []))


class Scripted:
    """Scripted model outputs; counts every provider call."""

    mode = "fake"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0
        self.messages = []

    def complete(self, messages, *, request_id=None, model_call_id=None):
        self.messages.append([dict(message) for message in messages])
        content = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return ModelCallResult(
            mode="fake", provider="scripted", model="scripted-v1", request_id=request_id,
            model_call_id=model_call_id, provider_call_id=None, provider_request_id=None,
            content=content, usage=None, usage_status="unknown",
        )


class CountingFake(FakeModel):
    def __init__(self) -> None:
        self.calls = 0
        self.messages = []

    def complete(self, messages, **kwargs):
        self.calls += 1
        self.messages.append([dict(message) for message in messages])
        return super().complete(messages, **kwargs)


class BlockingFake(FakeModel):
    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()

    def complete(self, messages, **kwargs):
        self.entered.set()
        assert self.release.wait(timeout=8), "blocking model was not released"
        return super().complete(messages, **kwargs)


# --- sync ---------------------------------------------------------------------


def test_sync_success_returns_verified_facts_and_server_answer(env) -> None:
    model = CountingFake()
    app.dependency_overrides[get_model_provider] = lambda: model

    response = ask(env, "2026年9月已支付订单总额")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "SUCCEEDED"
    assert body["profile"] == "B1-bounded-agent"
    assert fact_values(body) == [("gross_fen", 15000)]
    assert body["answer"].startswith("已核实：")
    assert body["result"]["run_id"] == body["run_id"]
    assert body["model_call_count"] == model.calls == len(body["usage"])
    assert body["sql_exec_count"] == 1
    stored = env.get(f"/runs/{body['run_id']}/result", headers=auth(REQUESTER))
    assert stored.status_code == 200
    assert stored.json()["facts"] == body["facts"]


def test_sync_net_counts_the_two_plan_queries(env) -> None:
    body = ask(env, "2026年9月退款后净额").json()
    assert body["status"] == "SUCCEEDED"
    assert fact_values(body) == [("net_fen", 12000)]
    assert body["sql_exec_count"] == 2


def test_sync_waiting_user_then_resume_succeeds(env) -> None:
    waiting = ask(env, "2026年9月销售额是多少？")
    body = waiting.json()
    assert waiting.status_code == 202
    assert waiting.headers["Location"] == f"/runs/{body['run_id']}"
    assert body["status"] == "WAITING_USER"
    assert body["pending_question"]
    assert body["facts"] is None and body["result"] is None

    resumed = env.post(f"/runs/{body['run_id']}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额"})
    final = resumed.json()
    assert resumed.status_code == 200
    assert final["status"] == "SUCCEEDED"
    assert fact_values(final) == [("gross_fen", 15000)]
    # The budget continues from the checkpoint; it is not reset.
    assert final["model_call_count"] > body["model_call_count"]


def test_sync_foreign_tenant_is_rejected_before_the_model(env) -> None:
    model = CountingFake()
    app.dependency_overrides[get_model_provider] = lambda: model
    response = ask(env, "查询tenant-B的订单金额")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"
    assert model.calls == 0


def test_sync_sensitive_query_waits_for_approval_then_returns_rows(env) -> None:
    pending = ask(env, "查询客户姓名")
    body = pending.json()
    assert pending.status_code == 202
    assert body["status"] == "WAITING_APPROVAL"
    assert body["approval_id"] and body["sql_exec_count"] == 0

    approved = env.post(
        f"/runs/{body['run_id']}/approval",
        headers=auth(APPROVER),
        json={"approval_id": body["approval_id"], "decision": "approve"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "SUCCEEDED"
    result = env.get(f"/runs/{body['run_id']}/result", headers=auth(REQUESTER)).json()
    assert result["result"]["principal_id"] == "principal-A-requester" or result["result"]["principal_id"] == result["principal_id"]
    assert {row["name"] for row in result["result"]["rows"]} == {"甲", "乙"}
    assert result["facts"] is None
    assert result["answer"] == "审批通过，已执行只读查询：返回 2 行，见 result.rows。"
    assert env.get(f"/runs/{body['run_id']}/result", headers=auth(APPROVER)).status_code == 404


APPROVED_ROWS_ANSWER = "审批通过，已执行只读查询：返回 2 行，见 result.rows。"
ALIAS = "总额999元已核实"
ALIAS_SQL = f'SELECT c.customer_id, c.name AS "{ALIAS}" FROM customers AS c ORDER BY c.customer_id'


class _AliasCursor:
    """Like PostgreSQL: result keys are the column aliases written in the SQL."""

    def __init__(self) -> None:
        self.sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def execute(self, sql, params):
        self.sql = str(sql)

    def fetchmany(self, size):
        assert f'"{ALIAS}"' in self.sql
        return [{"customer_id": "c1", ALIAS: "甲"}, {"customer_id": "c2", ALIAS: "乙"}][:size]


class _AliasConnection:
    def __init__(self) -> None:
        self.cursors: list[_AliasCursor] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def cursor(self, *, row_factory):
        self.cursors.append(_AliasCursor())
        return self.cursors[-1]


def test_approved_answer_never_echoes_model_written_column_aliases(env, monkeypatch) -> None:
    connection = _AliasConnection()
    service = shared_run_service()
    # Approval execution takes its executor from the service factory, like HTTP.
    monkeypatch.setattr(service, "_executor_factory", lambda: GuardedQueryExecutor(connect=lambda: connection))
    app.dependency_overrides[get_model_provider] = lambda: Scripted([
        json.dumps(
            {"type": "tool_call", "name": "query_readonly", "arguments": {"sql": ALIAS_SQL, "params": {}}},
            ensure_ascii=False,
        ),
    ])

    pending = ask(env, "查询本租户所有客户的姓名")
    body = pending.json()
    assert pending.status_code == 202 and body["status"] == "WAITING_APPROVAL"
    assert body["sql_exec_count"] == 0 and connection.cursors == []

    approved = env.post(
        f"/runs/{body['run_id']}/approval",
        headers=auth(APPROVER),
        json={"approval_id": body["approval_id"], "decision": "approve"},
    )
    assert approved.status_code == 200 and approved.json()["status"] == "SUCCEEDED"
    assert len(connection.cursors) == 1
    result = env.get(f"/runs/{body['run_id']}/result", headers=auth(REQUESTER)).json()
    assert result["answer"] == APPROVED_ROWS_ANSWER
    for text in (ALIAS, "已核实", "999", "甲", "乙"):
        assert text not in result["answer"]
        assert text not in approved.json().get("answer", "")
    assert result["facts"] is None
    assert result["result"]["rows"] == [{"customer_id": "c1", ALIAS: "甲"}, {"customer_id": "c2", ALIAS: "乙"}]


def test_sync_sensitive_query_rejected_is_denied(env) -> None:
    body = ask(env, "查询客户姓名").json()
    rejected = env.post(
        f"/runs/{body['run_id']}/approval",
        headers=auth(APPROVER),
        json={"approval_id": body["approval_id"], "decision": "reject"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "DENIED"
    assert rejected.json()["sql_exec_count"] == 0


def test_sync_parse_error_uses_the_single_repair(env) -> None:
    good = (
        '{"type":"tool_call","name":"query_readonly","arguments":{"sql":"SELECT COALESCE(SUM(o.amount_fen), 0) AS gross_fen '
        'FROM orders AS o WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s",'
        '"params":{"0":"paid","1":"2026-09-01T00:00:00Z","2":"2026-10-01T00:00:00Z"},'
        '"metrics":["gross_fen"],"time_window":{"start":"2026-09-01T00:00:00Z","end":"2026-10-01T00:00:00Z"}}}'
    )
    model = Scripted([good.replace('"type":"tool_call"', '"type":"query_readonly"', 1), good, "FINAL"])

    def complete_with_final(messages, *, request_id=None, model_call_id=None):
        if model.calls == 2:
            result_id = next(
                json.loads(message["content"].split("\n", 1)[1])["output"]["result_id"]
                for message in reversed(messages)
                if message["content"].startswith("QUERYSHIELD_DATA kind=untrusted_tool_result")
                and '"result_id"' in message["content"]
            )
            model.outputs[2] = json.dumps({
                "type": "final_answer", "answer": "ok", "source_ids": [],
                "fact_refs": [{"result_id": result_id, "metric_id": "gross_fen"}],
            })
        return Scripted.complete(model, messages, request_id=request_id, model_call_id=model_call_id)

    model.complete = complete_with_final
    app.dependency_overrides[get_model_provider] = lambda: model

    body = ask(env, "2026年9月已支付订单总额").json()

    assert body["status"] == "SUCCEEDED"
    assert fact_values(body) == [("gross_fen", 15000)]
    events = [event["payload"] for event in shared_run_service().store.events(body["run_id"]) if event["type"] == "agent_step"]
    assert [item.get("error_code") for item in events if item.get("kind") == "query_repair"] == ["tool_name_as_action_type"]


def test_sync_result_row_limit_is_not_the_agent_budget_limit(env) -> None:
    rows = [{"order_id": f"o{index}"} for index in range(101)]

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def execute(self, sql, params):
            return None

        def fetchmany(self, size):
            return rows[:size]

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def cursor(self, *, row_factory):
            return _Cursor()

    app.dependency_overrides[get_guarded_executor] = lambda: GuardedQueryExecutor(connect=lambda: _Connection())
    app.dependency_overrides[get_model_provider] = lambda: Scripted(
        ['{"type":"tool_call","name":"query_readonly","arguments":{"sql":"SELECT o.order_id FROM orders AS o","params":{}}}']
    )

    response = ask(env, "列出全部订单")

    body = response.json()
    assert response.status_code == 422
    assert body["status"] == "FAILED"
    assert body["error"]["code"] == "result_row_limit"
    assert body["result"] is None


# --- async ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    ["2026年9月已支付订单总额", "2026年9月已支付订单数", "2026年9月退款后净额", "查询客户姓名", "2026年9月销售额是多少？", "按客户查看2026年9月已支付订单总额"],
)
def test_async_matches_sync_terminal_state_and_facts(env, question) -> None:
    sync = ask(env, question).json()
    accepted = ask(env, question, asynchronous=True)
    assert accepted.status_code == 202
    assert accepted.headers["Location"] == f"/runs/{accepted.json()['run_id']}"
    assert accepted.json()["status"] == "RUNNING"
    final = wait(env, accepted.json()["run_id"], {"SUCCEEDED", "WAITING_USER", "WAITING_APPROVAL", "FAILED", "DENIED"})
    assert final["status"] == sync["status"]
    if final["status"] == "SUCCEEDED":
        stored = env.get(f"/runs/{final['run_id']}/result", headers=auth(REQUESTER)).json()
        assert fact_values(stored) == fact_values(sync)
        assert stored["result"]["rows"] == sync["result"]["rows"]
    assert final["model_call_count"] == sync["model_call_count"]
    assert final["sql_exec_count"] == sync["sql_exec_count"]


def test_async_events_replay_every_step(env) -> None:
    run_id = ask(env, "2026年9月已支付订单总额", asynchronous=True).json()["run_id"]
    wait(env, run_id, {"SUCCEEDED"})
    events = shared_run_service().store.events(run_id)
    types = [event["type"] for event in events]
    assert types[0] == "accepted" and types[1] == "step_started" and types[-1] == "terminal"
    kinds = [event["payload"].get("kind") for event in events if event["type"] == "agent_step"]
    assert "model_call" in kinds and "tool_call" in kinds and "answer" in kinds
    assert [event["event_id"] for event in events] == list(range(1, len(events) + 1))
    with env.stream("GET", f"/runs/{run_id}/events", headers=auth(REQUESTER)) as stream:
        body = b"".join(stream.iter_bytes()).decode("utf-8")
    assert body.count("event: agent_step") == len(kinds)
    assert "event: terminal" in body


def test_async_cancel_takes_effect_before_commit(env) -> None:
    model = BlockingFake()
    app.dependency_overrides[get_model_provider] = lambda: model
    run_id = ask(env, "2026年9月已支付订单总额", asynchronous=True).json()["run_id"]
    assert model.entered.wait(timeout=5)

    cancel = env.post(f"/runs/{run_id}/cancel", headers=auth(REQUESTER), json={})
    assert cancel.status_code == 202
    assert cancel.json()["status"] == "CANCEL_REQUESTED"
    model.release.set()

    final = wait(env, run_id, {"CANCELLED", "SUCCEEDED"})
    assert final["status"] == "CANCELLED"
    stored = shared_run_service().store.get_run(run_id)
    assert stored["result"] is None and stored["facts"] is None and stored["answer"] is None
    payloads = [event["payload"] for event in shared_run_service().store.events(run_id)]
    assert {"cancelled_before_commit": True} in payloads
    assert env.get(f"/runs/{run_id}/result", headers=auth(REQUESTER)).status_code == 409


def test_capacity_counts_sync_and_async_runs(env) -> None:
    model = BlockingFake()
    app.dependency_overrides[get_model_provider] = lambda: model
    run_ids = [ask(env, "2026年9月已支付订单总额", asynchronous=True).json()["run_id"] for _ in range(MAX_ACTIVE_RUNS)]
    try:
        full_async = ask(env, "2026年9月已支付订单总额", asynchronous=True)
        full_sync = ask(env, "2026年9月已支付订单总额")
        assert full_async.status_code == full_sync.status_code == 503
        assert full_async.json()["error"]["code"] == full_sync.json()["error"]["code"] == "run_capacity_reached"
    finally:
        model.release.set()
    for run_id in run_ids:
        wait(env, run_id, {"SUCCEEDED"})
    assert ask(env, "2026年9月已支付订单总额").status_code == 200


def test_waiting_user_can_be_cancelled_and_not_resumed(env) -> None:
    body = ask(env, "2026年9月销售额是多少？").json()
    assert body["status"] == "WAITING_USER"
    cancel = env.post(f"/runs/{body['run_id']}/cancel", headers=auth(REQUESTER), json={})
    assert cancel.status_code == 200
    assert cancel.json()["status"] == "CANCELLED"
    resume = env.post(f"/runs/{body['run_id']}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额"})
    assert resume.status_code == 409
    events = shared_run_service().store.events(body["run_id"])
    assert events[-1]["type"] == "terminal" and events[-1]["status"] == "CANCELLED"


# --- server-owned configuration and request fields ----------------------------------


def test_request_time_window_is_bound_and_validated(env) -> None:
    window = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}
    body = ask(env, "已支付订单数", time_window=window).json()
    assert body["status"] == "SUCCEEDED"
    fact = body["facts"]["facts"][0]
    assert fact["time_window"] == {**window, "timezone": "UTC"}
    for invalid in (
        {"start": "2026-10-01T00:00:00Z", "end": "2026-09-01T00:00:00Z"},
        {"start": "2026-09-01", "end": "2026-10-01"},
        {**window, "timezone": "Asia/Shanghai"},
        {**window, "tenant_id": "B"},
    ):
        assert ask(env, "已支付订单数", time_window=invalid).status_code == 422


@pytest.mark.parametrize("field", [{"profile": "B0-single-pass"}, {"tenant_id": "B"}, {"principal_id": "x"}, {"approver_id": "x"}])
def test_client_cannot_choose_profile_identity_tenant_or_approver(env, field) -> None:
    response = ask(env, "2026年9月已支付订单总额", **field)
    assert response.status_code == 422


def test_profile_comes_only_from_server_configuration(env, monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "B0")
    body = ask(env, "2026年9月已支付订单总额").json()
    assert body["profile"] == "B0-single-pass"
    assert body["model_call_count"] == 1
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "B7")
    response = ask(env, "2026年9月已支付订单总额")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "invalid_agent_profile"


def test_fixture_database_is_refused_with_a_real_provider(env, monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "real")
    reset_shared_state_stores()
    model = CountingFake()
    app.dependency_overrides[get_model_provider] = lambda: model

    response = ask(env, "2026年9月已支付订单总额")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "fake_database_requires_fake_provider"
    assert model.calls == 0
    service = get_run_service()
    with pytest.raises(Exception) as caught:
        service.start_async(identity={"tenant_id": "A", "principal_id": "p", "role": "requester"}, question="q")
    assert getattr(caught.value, "code", None) == "fake_database_requires_fake_provider"


def test_disabled_retrieval_never_describes_search_catalog(env, monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_RETRIEVAL", "disabled")
    model = CountingFake()
    app.dependency_overrides[get_model_provider] = lambda: model
    body = ask(env, "2026年9月已支付订单总额").json()
    assert body["status"] == "SUCCEEDED"
    assert all("search_catalog" not in json.dumps(messages, ensure_ascii=False) for messages in model.messages)
    stored = shared_run_service().store.get_run(body["run_id"])
    assert stored["run_config"]["retrieval"] == "disabled"
