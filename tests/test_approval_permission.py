"""The product publishes the knowledge it uses and binds every approval to a permission version.

Only the product service (``shared_run_service``) publishes; a service built
with its own store (evaluation, checks, tests) keeps the earlier behaviour.
Publishing is idempotent and never undoes a run-time ACL change.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from queryshield.api.main import app, get_guarded_executor
from queryshield.approval.service import (
    DEFAULT_KNOWLEDGE_SNAPSHOT,
    FixtureQueryExecutor,
    RunService,
    reset_shared_state_stores,
    shared_run_service,
)
from queryshield.db.state_store import StateStore
from queryshield.knowledge.runtime import (
    DEMO_SENSITIVE_PERMISSION_SOURCE_ID,
    product_knowledge,
    shared_demo_retrieval_runtime,
    shared_retrieval_runtime,
)
from queryshield.knowledge.snapshots import KnowledgeSnapshotRepository

from test_http_queries import APPROVER, REQUESTER, auth, env  # noqa: F401  (env is a fixture)

DEFAULT_SOURCE = "semantic-sensitive-customer-name"
NAMES = "查询客户姓名"


class CountingFixture(FixtureQueryExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def execute(self, sql, **kwargs):
        self.calls += 1
        return super().execute(sql, **kwargs)


@pytest.fixture()
def counted(env, monkeypatch):
    executor = CountingFixture()
    monkeypatch.setattr(shared_run_service(), "_executor_factory", lambda: executor)
    app.dependency_overrides[get_guarded_executor] = lambda: executor
    return env, executor


def _pending(client) -> dict:
    response = client.post("/queries", headers=auth(REQUESTER), json={"question": NAMES})
    body = response.json()
    assert response.status_code == 202 and body["status"] == "WAITING_APPROVAL", body
    return body


def _approver_view(client, run_id) -> dict:
    return client.get(f"/runs/{run_id}", headers=auth(APPROVER)).json()


def _approve(client, body):
    return client.post(
        f"/runs/{body['run_id']}/approval", headers=auth(APPROVER),
        json={"approval_id": body["approval_id"], "decision": "approve"},
    )


def test_the_product_path_binds_the_permission_without_any_manual_publication(counted) -> None:
    client, executor = counted
    service = shared_run_service()
    assert service.store.get_snapshot() is None  # nothing was published by the test
    body = _pending(client)
    action = _approver_view(client, body["run_id"])["approval"]["action"]
    assert action["permission_source_id"] == DEFAULT_SOURCE
    assert action["permission_version"] == service.store.get_source_acl(DEFAULT_SOURCE)["acl_version"] == 1
    run = service.store.get_run(body["run_id"])
    approval = service.store.get_approval(body["approval_id"])
    assert run["run_config"]["knowledge_snapshot_id"] == approval["knowledge_snapshot_id"] == product_knowledge(demo=False).snapshot_id
    assert executor.calls == 0
    assert _approve(client, body).json()["status"] == "SUCCEEDED" and executor.calls == 1


@pytest.mark.parametrize("change", ["revoke", "roles", "tenant"])
def test_a_permission_change_while_waiting_refuses_the_approval(counted, change) -> None:
    client, executor = counted
    service = shared_run_service()
    body = _pending(client)
    repository = KnowledgeSnapshotRepository(service.store)
    if change == "revoke":
        repository.revoke(DEFAULT_SOURCE)
    elif change == "roles":
        service.store.set_source_acl(DEFAULT_SOURCE, allowed_roles=("approver", "requester"))
    else:
        service.store.set_source_acl(DEFAULT_SOURCE, tenant_scope="B")
    response = _approve(client, body)
    assert response.status_code == 409 and response.json()["error"]["code"] == "authorization_revoked"
    assert executor.calls == 0
    assert service.store.get_approval(body["approval_id"])["status"] == "PENDING"
    assert service.store.get_run(body["run_id"])["status"] == "WAITING_APPROVAL"


def test_publishing_the_same_content_again_keeps_every_acl_version(tmp_path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    snapshot = product_knowledge(demo=False).snapshot.as_dict()
    store.publish_snapshot(snapshot)
    before = {source["source_id"]: store.get_source_acl(source["source_id"])["acl_version"] for source in snapshot["sources"]}
    store.publish_snapshot(snapshot)
    store.publish_snapshot({**snapshot, "snapshot_id": "same-acl-other-content"})
    assert {sid: store.get_source_acl(sid)["acl_version"] for sid in before} == before
    changed = [dict(source) for source in snapshot["sources"]]
    target = next(item for item in changed if item["source_id"] == DEFAULT_SOURCE)
    target["allowed_roles"] = ["approver", "requester"]
    store.publish_snapshot({**snapshot, "snapshot_id": "changed-acl", "sources": changed})
    assert store.get_source_acl(DEFAULT_SOURCE)["acl_version"] == before[DEFAULT_SOURCE] + 1
    assert all(store.get_source_acl(sid)["acl_version"] == version for sid, version in before.items() if sid != DEFAULT_SOURCE)
    store.close()


def _restart() -> RunService:
    """A new process on the same state store: fresh caches, fresh service."""

    reset_shared_state_stores()
    return shared_run_service()


def test_a_waiting_approval_survives_a_restart(counted, monkeypatch) -> None:
    client, executor = counted
    body = _pending(client)
    version = shared_run_service().store.get_source_acl(DEFAULT_SOURCE)["acl_version"]
    service = _restart()
    assert service is not None
    monkeypatch.setattr(service, "_executor_factory", lambda: executor)
    # A first run after the restart must not disturb the version the approval is bound to.
    client.post("/queries", headers=auth(REQUESTER), json={"question": "已支付订单有几笔"})
    assert service.store.get_source_acl(DEFAULT_SOURCE)["acl_version"] == version
    response = _approve(client, body)
    assert response.status_code == 200 and response.json()["status"] == "SUCCEEDED"


def test_a_restart_never_undoes_a_revocation(counted, monkeypatch) -> None:
    client, executor = counted
    body = _pending(client)
    KnowledgeSnapshotRepository(shared_run_service().store).revoke(DEFAULT_SOURCE)
    revoked = shared_run_service().store.get_source_acl(DEFAULT_SOURCE)
    service = _restart()
    monkeypatch.setattr(service, "_executor_factory", lambda: executor)
    first = client.post("/queries", headers=auth(REQUESTER), json={"question": "已支付订单有几笔"})
    assert first.json()["status"] == "SUCCEEDED"
    assert service.store.get_source_acl(DEFAULT_SOURCE) == revoked  # still revoked, same version
    response = _approve(client, body)
    assert response.status_code == 409 and response.json()["error"]["code"] == "authorization_revoked"
    assert executor.calls == 1  # only the paid-count query ran
    # A new sensitive request cannot create an approval on a revoked source either.
    refused = client.post("/queries", headers=auth(REQUESTER), json={"question": NAMES})
    assert refused.status_code == 503 and refused.json()["error"]["code"] == "approval_permission_unavailable"


def test_new_knowledge_content_keeps_a_revoked_source_revoked(tmp_path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    snapshot = product_knowledge(demo=False).snapshot.as_dict()
    store.publish_snapshot(snapshot, keep_inactive_sources=True)
    store.set_source_acl(DEFAULT_SOURCE, status="deleted")
    revoked = store.get_source_acl(DEFAULT_SOURCE)
    store.publish_snapshot({**snapshot, "snapshot_id": "new-content"}, keep_inactive_sources=True)
    assert store.get_source_acl(DEFAULT_SOURCE) == revoked
    # An explicit publication (an operator restoring the snapshot) restores the ACL, as before.
    store.publish_snapshot(snapshot)
    assert store.get_source_acl(DEFAULT_SOURCE)["status"] == "active"
    assert store.get_source_acl(DEFAULT_SOURCE)["acl_version"] == revoked["acl_version"] + 1
    store.close()


def test_no_approval_without_a_permission_source(counted) -> None:
    client, executor = counted
    service = shared_run_service()
    client.post("/queries", headers=auth(REQUESTER), json={"question": "已支付订单有几笔"})  # publishes
    with service.store._lock:  # test-only: the permission row disappears
        service.store._connection.execute("DELETE FROM knowledge_acl WHERE source_id = ?", (DEFAULT_SOURCE,))
    response = client.post("/queries", headers=auth(REQUESTER), json={"question": NAMES})
    body = response.json()
    assert response.status_code == 503 and body["error"]["code"] == "approval_permission_unavailable"
    run = service.store.get_run(body["run_id"])
    assert run["status"] == "FAILED" and run["approval_id"] is None and run["answer"] is None
    with service.store._lock:
        count = service.store._connection.execute("SELECT COUNT(*) FROM approvals WHERE run_id = ?", (body["run_id"],)).fetchone()[0]
    assert count == 0 and executor.calls == 1


def test_the_demo_setting_binds_the_demo_permission_source(counted, monkeypatch) -> None:
    client, _ = counted
    monkeypatch.setenv("QUERYSHIELD_DEMO_DATASET", "commerce-demo-v1")
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", "postgresql://reader@localhost/queryshield_demo")
    body = _pending(client)
    action = _approver_view(client, body["run_id"])["approval"]["action"]
    assert action["permission_source_id"] == DEMO_SENSITIVE_PERMISSION_SOURCE_ID
    run = shared_run_service().store.get_run(body["run_id"])
    demo = shared_demo_retrieval_runtime("fake")
    assert run["run_config"]["knowledge_snapshot_id"] == product_knowledge(demo=True).snapshot_id == demo.index_build.base_snapshot_id
    assert run["run_config"]["agent_run_config"]["knowledge_snapshot_id"] == demo.snapshot.snapshot_id


def test_hybrid_retrieval_embeds_the_snapshot_named_at_the_top_level(counted) -> None:
    client, _ = counted
    body = client.post("/queries", headers=auth(REQUESTER), json={"question": "已支付订单有几笔"}).json()
    run_config = shared_run_service().store.get_run(body["run_id"])["run_config"]
    runtime = shared_retrieval_runtime("fake")
    assert run_config["retrieval"] == "hybrid"
    assert run_config["knowledge_snapshot_id"] == runtime.index_build.base_snapshot_id == DEFAULT_KNOWLEDGE_SNAPSHOT
    assert run_config["agent_run_config"]["knowledge_snapshot_id"] == runtime.snapshot.snapshot_id


@pytest.mark.parametrize("setting", [("QUERYSHIELD_RETRIEVAL", "catalog"), ("QUERYSHIELD_RETRIEVAL", "disabled"), ("QUERYSHIELD_AGENT_PROFILE", "b0")])
def test_without_knowledge_retrieval_the_default_ids_are_equal_and_permissions_still_publish(counted, monkeypatch, setting) -> None:
    client, _ = counted
    monkeypatch.setenv(*setting)
    body = client.post("/queries", headers=auth(REQUESTER), json={"question": "已支付订单有几笔"}).json()
    service = shared_run_service()
    run_config = service.store.get_run(body["run_id"])["run_config"]
    assert run_config["knowledge_snapshot_id"] == run_config["agent_run_config"]["knowledge_snapshot_id"] == DEFAULT_KNOWLEDGE_SNAPSHOT
    assert service.store.get_source_acl(DEFAULT_SOURCE)["status"] == "active"


def test_a_service_with_its_own_store_publishes_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.delenv("QUERYSHIELD_RETRIEVAL", raising=False)
    store = StateStore(tmp_path / "own.sqlite3")
    service = RunService(store=store, executor_factory=FixtureQueryExecutor, mode="fake")
    run = service.run_sync(identity={"tenant_id": "A", "principal_id": "a-requester", "role": "requester"}, question=NAMES)
    assert run["status"] == "WAITING_APPROVAL"
    assert store.get_snapshot() is None and store.get_source_acl(DEFAULT_SOURCE) is None
    assert run["run_config"]["knowledge_snapshot_id"] == DEFAULT_KNOWLEDGE_SNAPSHOT
    action = store.get_approval(str(run["approval_id"]))["action"]
    assert "permission_source_id" not in action and "permission_version" not in action
    store.close()


def test_http_without_lifespan_still_publishes_before_the_first_run(env) -> None:
    # TestClient used without its context manager never enters the lifespan.
    client = TestClient(app)
    body = client.post("/queries", headers=auth(REQUESTER), json={"question": NAMES}).json()
    assert body["status"] == "WAITING_APPROVAL"
    assert shared_run_service().store.get_source_acl(DEFAULT_SOURCE) is not None


def test_a_restart_never_undoes_a_role_change(counted, monkeypatch) -> None:
    client, executor = counted
    client.post("/queries", headers=auth(REQUESTER), json={"question": "已支付订单有几笔"})  # publishes
    shared_run_service().store.set_source_acl(DEFAULT_SOURCE, allowed_roles=("approver", "auditor"))
    changed = shared_run_service().store.get_source_acl(DEFAULT_SOURCE)
    service = _restart()
    monkeypatch.setattr(service, "_executor_factory", lambda: executor)
    client.post("/queries", headers=auth(REQUESTER), json={"question": "已支付订单有几笔"})
    assert service.store.get_source_acl(DEFAULT_SOURCE) == changed


def test_the_first_product_publication_keeps_an_earlier_revocation(counted) -> None:
    client, _ = counted
    service = shared_run_service()
    # Another snapshot with the same sources was published and the source revoked
    # before the product ever ran on this store (the STATE-EN05 order).
    other = product_knowledge(demo=False).snapshot.as_dict()
    service.store.publish_snapshot({**other, "snapshot_id": "operator-published"})
    service.store.set_source_acl(DEFAULT_SOURCE, status="deleted")
    revoked = service.store.get_source_acl(DEFAULT_SOURCE)
    refused = client.post("/queries", headers=auth(REQUESTER), json={"question": NAMES})
    assert service.store.get_snapshot(product_knowledge(demo=False).snapshot_id) is not None  # the product published
    assert service.store.get_source_acl(DEFAULT_SOURCE) == revoked
    assert refused.status_code == 503 and refused.json()["error"]["code"] == "approval_permission_unavailable"


def test_the_smoke_reads_only_the_permission_field_names_and_version() -> None:
    from scripts.http_smoke import approval_permission_bound

    bound = {"approval": {"action": {"permission_source_id": DEFAULT_SOURCE, "permission_version": 3, "sql": "x"}}}
    assert approval_permission_bound(bound) is True
    assert approval_permission_bound({"approval": {"action": {"sql": "x"}}}) is False
    assert approval_permission_bound({"approval": {"action": {"permission_source_id": DEFAULT_SOURCE, "permission_version": True}}}) is False
    assert approval_permission_bound({}) is False


# -- a knowledge base that cannot be loaded is not a permission problem --


def _knowledge_fails(monkeypatch, *, after: int = 0):
    """Make ``product_knowledge`` raise from call number ``after + 1`` on."""

    import queryshield.knowledge.runtime as runtime

    real = runtime.product_knowledge
    calls = {"count": 0}

    def loader(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] > after:
            raise RuntimeError("secret loader detail")
        return real(*args, **kwargs)

    monkeypatch.setattr(runtime, "product_knowledge", loader)
    return calls


def test_a_knowledge_base_that_cannot_load_is_knowledge_unavailable_before_any_run(counted, monkeypatch) -> None:
    client, executor = counted
    _knowledge_fails(monkeypatch)
    # Even a question that never needs an approval: every product run publishes the knowledge first.
    response = client.post("/queries", headers=auth(REQUESTER), json={"question": "已支付订单有几笔"})
    assert response.status_code == 503 and response.json()["error"]["code"] == "knowledge_unavailable"
    assert "secret loader detail" not in response.text
    assert executor.calls == 0


def test_the_knowledge_base_failing_when_an_approval_is_needed_is_knowledge_unavailable(counted, monkeypatch) -> None:
    client, executor = counted
    service = shared_run_service()
    calls = _knowledge_fails(monkeypatch, after=1)  # the run starts; the approval step cannot load it
    response = client.post("/queries", headers=auth(REQUESTER), json={"question": NAMES})
    body = response.json()
    assert calls["count"] >= 2
    assert response.status_code == 503 and body["error"]["code"] == "knowledge_unavailable", body
    assert "secret loader detail" not in response.text
    run = service.store.get_run(body["run_id"])
    assert run["status"] == "FAILED" and run["approval_id"] is None and run["answer"] is None
    with service.store._lock:
        count = service.store._connection.execute("SELECT COUNT(*) FROM approvals WHERE run_id = ?", (body["run_id"],)).fetchone()[0]
    assert count == 0 and executor.calls == 0


def test_a_missing_permission_source_keeps_its_own_code_when_the_knowledge_loads(counted) -> None:
    client, _ = counted
    service = shared_run_service()
    client.post("/queries", headers=auth(REQUESTER), json={"question": "已支付订单有几笔"})  # publishes
    with service.store._lock:
        service.store._connection.execute("DELETE FROM knowledge_acl WHERE source_id = ?", (DEFAULT_SOURCE,))
    response = client.post("/queries", headers=auth(REQUESTER), json={"question": NAMES})
    assert response.status_code == 503 and response.json()["error"]["code"] == "approval_permission_unavailable"
