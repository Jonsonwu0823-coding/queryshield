from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest
from fastapi.testclient import TestClient

from queryshield.agent.context_runtime import ContextRecoveryError, compress_context, verify_restore
from queryshield.agent.parallel_durable import DurableParallelScheduler
from queryshield.agent.proposals import ExecutionContext
from queryshield.approval.service import FixtureQueryExecutor, W04RunService, reset_shared_state_stores, shared_w04_service
from queryshield.db.w04_state import StateStore
from queryshield.knowledge.ingest import build_snapshot
from queryshield.knowledge.snapshots import KnowledgeAccessError, KnowledgeIdentity, KnowledgeSnapshotRepository
from queryshield.memory.preferences import PreferenceStore


@pytest.fixture()
def w04_client(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("QUERYSHIELD_STATE_STORE_PATH", str(tmp_path / "api.sqlite"))
    monkeypatch.setenv("QUERYSHIELD_W04_FAKE_DB", "1")
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", "test-a-requester")
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_APPROVER", "test-a-approver")
    monkeypatch.setenv("QUERYSHIELD_TOKEN_B_REQUESTER", "test-b-requester")
    monkeypatch.setenv("QUERYSHIELD_TOKEN_B_APPROVER", "test-b-approver")
    reset_shared_state_stores()
    from queryshield.api.main import app

    with TestClient(app) as client:
        yield client
    reset_shared_state_stores()


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def wait_for_status(client: TestClient, run_id: str, token: str, wanted: set[str], timeout: float = 5.0) -> dict[str, object]:
    """B2b: the async worker runs the Agent, so WAITING_APPROVAL is reached after 202."""

    import time

    deadline = time.monotonic() + timeout
    body: dict[str, object] = {}
    while time.monotonic() < deadline:
        body = client.get(f"/runs/{run_id}", headers=auth(token)).json()
        if body.get("status") in wanted:
            return body
        time.sleep(0.02)
    raise AssertionError(f"run did not reach {sorted(wanted)}: {body.get('status')}")


def test_w04_approval_object_scope_and_replay(w04_client: TestClient) -> None:
    accepted = w04_client.post(
        "/queries",
        json={"question": "查询客户姓名"},
        headers={**auth("test-a-requester"), "Prefer": "respond-async"},
    )
    assert accepted.status_code == 202
    run_id = accepted.json()["run_id"]
    owner = wait_for_status(w04_client, run_id, "test-a-requester", {"WAITING_APPROVAL"})
    approval_id = owner["approval_id"]
    assert w04_client.get(f"/runs/{run_id}", headers=auth("test-b-requester")).status_code == 404
    assert w04_client.post(
        f"/runs/{run_id}/approval",
        json={"approval_id": approval_id, "decision": "approve"},
        headers=auth("test-a-requester"),
    ).status_code == 403
    approved = w04_client.post(
        f"/runs/{run_id}/approval",
        json={"approval_id": approval_id, "decision": "approve"},
        headers=auth("test-a-approver"),
    )
    replay = w04_client.post(
        f"/runs/{run_id}/approval",
        json={"approval_id": approval_id, "decision": "approve"},
        headers=auth("test-a-approver"),
    )
    assert approved.status_code == replay.status_code == 200
    assert approved.json()["result"]["result_id"] == replay.json()["result"]["result_id"]
    assert replay.json()["sql_exec_count"] == 1


def test_w04_pending_approval_is_invalidated_by_current_source_acl(w04_client: TestClient) -> None:
    service = shared_w04_service()
    root = __import__("pathlib").Path(__file__).resolve().parents[1] / "fixtures" / "knowledge"
    snapshot = build_snapshot(root, root / "source_registry.json", catalog_version="catalog-v1")
    repository = KnowledgeSnapshotRepository(service.store)
    repository.publish(snapshot)

    accepted = w04_client.post(
        "/queries",
        json={"question": "查询客户姓名"},
        headers={**auth("test-a-requester"), "Prefer": "respond-async"},
    )
    assert accepted.status_code == 202
    run_id = accepted.json()["run_id"]
    pending = wait_for_status(w04_client, run_id, "test-a-requester", {"WAITING_APPROVAL"})
    action = service.store.get_run(run_id)["action"]
    assert action["permission_source_id"] == "semantic-sensitive-customer-name"
    assert action["permission_version"] == service.store.get_source_acl("semantic-sensitive-customer-name")["acl_version"]

    repository.revoke("semantic-sensitive-customer-name")
    approved = w04_client.post(
        f"/runs/{run_id}/approval",
        json={"approval_id": pending["approval_id"], "decision": "approve"},
        headers=auth("test-a-approver"),
    )
    current = service.store.get_run(run_id)
    assert approved.status_code == 409
    assert approved.json()["error"]["code"] == "authorization_revoked"
    assert current["status"] == "WAITING_APPROVAL"
    assert current["sql_exec_count"] == 0


def test_w04_preference_context_and_restore(tmp_path) -> None:
    state = StateStore(tmp_path / "state.sqlite")
    preferences = PreferenceStore(state)
    assert preferences.put(tenant_id="A", principal_id="p", key="answer_style", value="table", confirmed=True)["version"] == 1
    assert preferences.apply_to_request(tenant_id="A", principal_id="p", key="answer_style", explicit_value="concise") == "concise"
    checkpoint = {
        "tenant_id": "A",
        "principal_id": "p",
        "versions": {"state": "v1"},
        "permission_version": "perm1",
        "result_refs": [{"result_id": "result-1", "metric_id": "paid_count"}],
    }
    context = compress_context(
        goal="g",
        metric_ids=("paid_count",),
        time_window={"start": "s", "end": "e", "timezone": "UTC"},
        constraints=("readonly",),
        result_refs=checkpoint["result_refs"],
        optional_tool_summaries=({"text": "x" * 30000},),
    )
    assert context.size_bytes <= 24000
    assert verify_restore(
        checkpoint,
        tenant_id="A",
        principal_id="p",
        current_versions={"state": "v1"},
        current_permission_version="perm1",
        approval_valid=True,
    )["reexecute_submitted_results"] is False
    with pytest.raises(ContextRecoveryError, match="current permissions"):
        verify_restore(
            checkpoint,
            tenant_id="A",
            principal_id="p",
            current_versions={"state": "v1"},
            current_permission_version="perm2",
            approval_valid=True,
        )
    state.close()


def test_w04_parallel_plan_is_durable_and_bounded(tmp_path) -> None:
    state = StateStore(tmp_path / "parallel.sqlite")
    state.create_run(run_id="run-p", tenant_id="A", principal_id="p", role="requester", question="parallel", mode="fake")
    context = ExecutionContext(run_id="run-p", tenant_id="A", principal_id="p", role="requester")
    scheduler = DurableParallelScheduler(state=state, executor_factory=FixtureQueryExecutor)
    first = scheduler.run(context, ("gross_fen", "paid_count", "net_fen"))
    second = scheduler.run(context, ("net_fen", "paid_count", "gross_fen"))
    assert first.status == second.status == "SUCCEEDED"
    assert first.peak_active <= 2
    assert second.reused is True and second.new_branch_count == 0
    group = state.get_parallel_group("run-p")
    assert group is not None
    assert all(
        branch["result"]["run_id"] == "run-p"
        and branch["result"]["tenant_id"] == "A"
        and branch["result"]["principal_id"] == "p"
        for branch in group["branches"]
    )
    state.close()


def test_w04_knowledge_acl_and_atomic_failed_publish(tmp_path) -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1] / "fixtures" / "knowledge"
    snapshot = build_snapshot(root, root / "source_registry.json", catalog_version="catalog-v1")
    state = StateStore(tmp_path / "knowledge.sqlite")
    repository = KnowledgeSnapshotRepository(state)
    repository.publish(snapshot)
    with pytest.raises(Exception):
        state.publish_snapshot({"snapshot_id": "bad", "catalog_version": "catalog-v1", "sources": [snapshot.as_dict()["sources"][0], "bad"]})
    assert repository.current()["snapshot_id"] == snapshot.snapshot_id
    repository.visible_source(
        snapshot_id=snapshot.snapshot_id,
        source_id="tenant-a-orders-overview",
        identity=KnowledgeIdentity("A", "p", "requester"),
    )
    with pytest.raises(KnowledgeAccessError):
        repository.visible_source(
            snapshot_id=snapshot.snapshot_id,
            source_id="tenant-a-orders-overview",
            identity=KnowledgeIdentity("B", "p", "requester"),
        )
    state.close()
