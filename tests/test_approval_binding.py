"""An approval binds the exact server-verified query and executes it once."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from queryshield.agent.proposals import ExecutionContext
from queryshield.agent.runtime import RuntimeDependencies
from queryshield.agent.tool_execution import execute_approved_query
from queryshield.api.main import app, get_run_service
from queryshield.approval.service import (
    ApprovalConflict,
    FixtureQueryExecutor,
    RunAuthorizationError,
    RunService,
    action_hash,
    pending_call_from_action,
)
from queryshield.catalog import load_default_catalog
from queryshield.db.state_store import StateStore
from queryshield.knowledge.ingest import build_snapshot
from queryshield.knowledge.snapshots import KnowledgeSnapshotRepository
from queryshield.providers.contracts import ModelCallResult
from queryshield.providers.fake_model import FakeModel
from queryshield.tools.semantic import ControlledTools, ToolError


REQUESTER = {"tenant_id": "A", "principal_id": "a-requester", "role": "requester"}
APPROVER = {"tenant_id": "A", "principal_id": "a-approver", "role": "approver"}
OTHER_APPROVER = {"tenant_id": "B", "principal_id": "b-approver", "role": "approver"}
BASE = datetime(2026, 9, 23, tzinfo=timezone.utc)


class RecordingFixture(FixtureQueryExecutor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, sql, *, context, params=(), metric_bindings=()):
        self.executed.append((sql, tuple(params)))
        return super().execute(sql, context=context, params=params, metric_bindings=metric_bindings)


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.delenv("QUERYSHIELD_RETRIEVAL", raising=False)
    now = [BASE]
    store = StateStore(tmp_path / "state.sqlite3", clock=lambda: now[0])
    executor = RecordingFixture(clock=lambda: now[0])
    service = RunService(store=store, executor_factory=lambda: executor, clock=lambda: now[0], mode="fake")
    yield service, executor, now
    store.close()


def _deps(service, model=None) -> RuntimeDependencies:
    deps = service.default_dependencies()
    if model is not None:
        deps.model = model
    return deps


def _waiting(service, question: str = "查询客户姓名", model=None) -> dict[str, object]:
    run = service.run_sync(identity=REQUESTER, question=question, deps=_deps(service, model))
    assert run["status"] == "WAITING_APPROVAL", run.get("error_code")
    return run


def _approve(service, run, identity=APPROVER, decision="approve"):
    return service.approve(run_id=str(run["run_id"]), approval_id=str(run["approval_id"]), identity=identity, decision=decision)


def test_pending_action_is_the_verified_call_bound_to_run_identity_and_versions(harness) -> None:
    service, executor, _ = harness
    run = _waiting(service)
    approval = service.store.get_approval(str(run["approval_id"]))
    action = approval["action"]
    assert action["kind"] == "query_readonly"
    assert action["run_id"] == run["run_id"]
    assert action["tenant_id"] == "A" and action["requester_principal_id"] == "a-requester"
    assert "customers" in action["sql"] and "name" in action["sql"]
    assert approval["action_hash"] == action_hash(action) == action_hash(run["action"])
    assert executor.executed == []  # nothing ran before approval


FIELD_MUTATIONS = {
    "sql": lambda action: {**action, "sql": "SELECT c.name FROM customers AS c"},
    "params": lambda action: {**action, "params": ["c2"]},
    "metrics": lambda action: {**action, "metrics": ["paid_count"]},
    "time_window": lambda action: {**action, "time_window": {"start": "2026-08-01T00:00:00Z", "end": "2026-09-01T00:00:00Z", "timezone": "UTC"}},
    "tenant_id": lambda action: {**action, "tenant_id": "B"},
    "requester_principal_id": lambda action: {**action, "requester_principal_id": "a-approver"},
    "permission_version": lambda action: {**action, "permission_source_id": "semantic-sensitive-customer-name", "permission_version": 999},
    "run_id": lambda action: {**action, "run_id": "run-other"},
}


@pytest.mark.parametrize("field", sorted(FIELD_MUTATIONS))
@pytest.mark.parametrize("side", ["run", "approval"])
def test_any_changed_action_field_is_refused_before_execution(harness, field, side) -> None:
    service, executor, _ = harness
    run = _waiting(service)
    approval_id = str(run["approval_id"])
    mutated = FIELD_MUTATIONS[field](dict(run["action"]))
    if side == "run":
        service.store.update_run(str(run["run_id"]), action_json=json.dumps(mutated, ensure_ascii=False))
    else:
        with service.store._lock:  # test-only tampering with the stored approval record
            service.store._connection.execute(
                "UPDATE approvals SET action_json = ? WHERE approval_id = ?",
                (json.dumps(mutated, ensure_ascii=False), approval_id),
            )
    with pytest.raises(ApprovalConflict) as caught:
        _approve(service, run)
    assert caught.value.code in {"approval_stale", "authorization_revoked"}
    assert executor.executed == []
    assert service.store.get_approval(approval_id)["status"] == "PENDING"
    assert service.store.get_run(str(run["run_id"]))["status"] == "WAITING_APPROVAL"


def test_changed_permission_version_revokes_the_approval(harness) -> None:
    service, executor, _ = harness
    root = Path(__file__).resolve().parents[1] / "fixtures" / "knowledge"
    repository = KnowledgeSnapshotRepository(service.store)
    repository.publish(build_snapshot(root, root / "source_registry.json", catalog_version="catalog-v1"))
    run = _waiting(service)
    assert run["action"]["permission_source_id"] == "semantic-sensitive-customer-name"
    repository.revoke("semantic-sensitive-customer-name")
    with pytest.raises(ApprovalConflict) as caught:
        _approve(service, run)
    assert caught.value.code == "authorization_revoked"
    assert executor.executed == []


def test_expired_approval_is_refused(harness) -> None:
    service, executor, now = harness
    run = _waiting(service)
    now[0] = BASE + timedelta(seconds=601)
    with pytest.raises(ApprovalConflict) as caught:
        _approve(service, run)
    assert caught.value.code == "approval_stale"
    assert executor.executed == []


def test_requester_and_other_tenant_cannot_approve(harness) -> None:
    service, executor, _ = harness
    run = _waiting(service)
    with pytest.raises(RunAuthorizationError) as caught:
        _approve(service, run, identity={**REQUESTER, "role": "approver"})
    assert caught.value.code == "forbidden"
    with pytest.raises(RunAuthorizationError):
        _approve(service, run, identity=OTHER_APPROVER)
    assert executor.executed == []


def test_approval_executes_exactly_the_approved_action_once_for_the_requester(harness) -> None:
    service, executor, _ = harness
    run = _waiting(service)
    action = run["action"]

    approved = _approve(service, run)
    replay = _approve(service, run)
    # A later tampered run action cannot trigger a second execution either.
    service.store.update_run(str(run["run_id"]), action_json=json.dumps({**action, "sql": "SELECT c.name FROM customers AS c"}))
    replay_after_change = _approve(service, run)

    assert approved["status"] == replay["status"] == replay_after_change["status"] == "SUCCEEDED"
    assert executor.executed == [(action["sql"], tuple(action["params"]))]
    assert approved["result"]["result_id"] == replay["result"]["result_id"]
    assert approved["result"]["principal_id"] == "a-requester"
    assert approved["sql_exec_count"] == 1
    with pytest.raises(Exception):
        service.visible_run(run_id=str(run["run_id"]), identity=APPROVER, result=True)


def test_approver_cannot_read_the_result_over_http(harness, monkeypatch) -> None:
    service, _, _ = harness
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", "t-a-requester")
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_APPROVER", "t-a-approver")
    run = _waiting(service)
    _approve(service, run)
    app.dependency_overrides[get_run_service] = lambda: service
    try:
        client = TestClient(app)
        requester = client.get(f"/runs/{run['run_id']}/result", headers={"Authorization": "Bearer t-a-requester"})
        approver = client.get(f"/runs/{run['run_id']}/result", headers={"Authorization": "Bearer t-a-approver"})
        approver_status = client.get(f"/runs/{run['run_id']}", headers={"Authorization": "Bearer t-a-approver"})
    finally:
        app.dependency_overrides.clear()
    assert requester.status_code == 200
    assert approver.status_code == 404
    assert approver_status.status_code == 404


class _CountThenNames:
    """Queries a verified metric first, then asks for customer names."""

    mode = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, request_id=None, model_call_id=None, run_id=None):
        self.calls += 1
        if self.calls == 1:
            content = {
                "type": "tool_call", "name": "query_readonly",
                "arguments": {
                    "sql": "SELECT COUNT(*) AS paid_count FROM orders AS o WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s",
                    "params": {"0": "paid", "1": "2026-09-01T00:00:00Z", "2": "2026-10-01T00:00:00Z"},
                    "metrics": ["paid_count"],
                    "time_window": {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"},
                },
            }
        else:
            content = {
                "type": "tool_call", "name": "query_readonly",
                "arguments": {"sql": "SELECT c.customer_id, c.name FROM customers AS c ORDER BY c.customer_id", "params": {}},
            }
        return ModelCallResult(
            mode="fake", provider="scripted", model="count-then-names", request_id=request_id,
            model_call_id=model_call_id, provider_call_id=None, provider_request_id=None,
            content=json.dumps(content, ensure_ascii=False), usage=None, usage_status="unknown",
        )


def test_facts_verified_before_approval_are_kept_once(harness) -> None:
    from queryshield.facts.persisted import validate_persisted_run_result

    service, executor, _ = harness
    run = _waiting(service, "9月已支付订单数和客户姓名", model=_CountThenNames())
    assert len(run["checkpoint"]["pre_approval_results"]) == 1
    assert run["sql_exec_count"] == 1

    approved = _approve(service, run)
    replay = _approve(service, run)

    facts = approved["facts"]["facts"]
    assert [(item["metric_id"], item["value"]) for item in facts] == [("paid_count", 2)]
    assert facts[0]["result_id"] == run["checkpoint"]["pre_approval_results"][0]["result_id"]
    assert facts[0]["principal_id"] == "a-requester"
    assert approved["result"]["supporting_results"][0]["result_id"] == facts[0]["result_id"]
    assert approved["answer"].startswith("已核实：已支付订单数：2笔")
    assert approved["answer"].endswith("\n审批通过，已执行只读查询：返回 2 行，见 result.rows。")
    assert replay["facts"] == approved["facts"]
    assert approved["sql_exec_count"] == 2 and len(executor.executed) == 2
    validate_persisted_run_result(approved)


def test_approved_execution_reruns_every_check_except_the_sensitive_gate(harness) -> None:
    service, _, _ = harness
    run = _waiting(service)
    tools = ControlledTools(catalog=load_default_catalog(), executor=FixtureQueryExecutor())
    context = ExecutionContext(run_id=str(run["run_id"]), tenant_id="A", principal_id="a-requester", role="requester")
    call = pending_call_from_action(run["action"])
    evidence = execute_approved_query(tools, call, context=context)
    assert evidence.principal_id == "a-requester"
    for broken, code in (
        ({**call, "tool": "search_catalog"}, "approval_action_invalid"),
        ({**call, "params": {"0": "c1", "tenant_id": "B"}}, "reserved_parameter"),
        ({**call, "sql": "DELETE FROM customers"}, None),
        ({**call, "sql": "SELECT s.x FROM secrets AS s"}, None),
    ):
        with pytest.raises(ToolError) as caught:
            execute_approved_query(tools, broken, context=context)
        if code is not None:
            assert caught.value.code == code


def test_fake_model_customer_name_question_reaches_approval_not_denial(harness) -> None:
    service, _, _ = harness
    run = _waiting(service, model=FakeModel())
    events = service.store.events(str(run["run_id"]))
    tool_events = [event["payload"] for event in events if event["type"] == "agent_step" and event["payload"].get("kind") == "tool_call"]
    assert tool_events[-1]["status"] == "approval_required"
    assert events[-1]["type"] == "waiting"


class _AliasNames:
    """Asks for customer names under a model-written alias with digits and 已核实."""

    mode = "fake"
    alias = "总额999元已核实"

    def complete(self, messages, *, request_id=None, model_call_id=None, run_id=None):
        content = {
            "type": "tool_call", "name": "query_readonly",
            "arguments": {
                "sql": f'SELECT c.customer_id, c.name AS "{self.alias}" FROM customers AS c ORDER BY c.customer_id',
                "params": {},
            },
        }
        return ModelCallResult(
            mode="fake", provider="scripted", model="alias-names", request_id=request_id,
            model_call_id=model_call_id, provider_call_id=None, provider_request_id=None,
            content=json.dumps(content, ensure_ascii=False), usage=None, usage_status="unknown",
        )


class _AliasFixture(RecordingFixture):
    """Returns rows keyed by the SQL alias, as a real database does."""

    def _rows(self, sql, context, params):
        rows = super()._rows(sql, context, params)
        return tuple(
            {(_AliasNames.alias if key == "name" else key): value for key, value in row.items()} for row in rows
        )


def test_approved_answer_lists_no_column_names(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.delenv("QUERYSHIELD_RETRIEVAL", raising=False)
    store = StateStore(tmp_path / "state.sqlite3", clock=lambda: BASE)
    executor = _AliasFixture(clock=lambda: BASE)
    service = RunService(store=store, executor_factory=lambda: executor, clock=lambda: BASE, mode="fake")
    try:
        run = _waiting(service, "查询本租户所有客户的姓名", model=_AliasNames())
        approved = _approve(service, run)
    finally:
        store.close()

    assert approved["status"] == "SUCCEEDED"
    assert approved["answer"] == "审批通过，已执行只读查询：返回 2 行，见 result.rows。"
    for text in (_AliasNames.alias, "已核实", "999", "customer_id", "甲"):
        assert text not in approved["answer"]
    assert approved["facts"] is None
    assert [row[_AliasNames.alias] for row in approved["result"]["rows"]] == ["甲", "乙"]
