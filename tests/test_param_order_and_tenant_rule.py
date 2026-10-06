"""One parameter-index rule and one tenant-scope rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from queryshield.approval.service import ApprovalConflict, build_pending_action
from queryshield.db.state_store import StateStore
from queryshield.knowledge.ingest import build_snapshot
from queryshield.knowledge.acl import tenant_matches
from queryshield.knowledge.snapshots import KnowledgeAccessError, KnowledgeIdentity, KnowledgeSnapshotRepository
from queryshield.policy.params import ordered_param_values


ELEVEN = {str(index): f"v{index}" for index in range(11)}


@pytest.mark.parametrize(
    "params, expected",
    [
        (ELEVEN, tuple(f"v{index}" for index in range(11))),
        ({"1": "b", "0": "a"}, ("a", "b")),
        ({}, ()),
        ({"0": "a", "2": "c"}, None),  # gap
        ({"1": "b"}, None),  # does not start at zero
        ({"00": "a"}, None),  # leading zero
        ({"01": "b", "0": "a"}, None),
        ({"a": 1}, None),  # not decimal
        ({"-1": 1}, None),
        ({"٠": 1}, None),  # non-ASCII decimal digit
        ({0: "a"}, None),  # not a string key
    ],
)
def test_ordered_param_values(params, expected):
    assert ordered_param_values(params) == expected


def _pending_call(params: dict) -> dict:
    return {"tool": "query_readonly", "sql": "SELECT 1", "params": params, "metrics": [], "time_window": None}


def _build(params: dict) -> dict:
    return build_pending_action(_pending_call(params), run_id="r", tenant_id="A", requester_principal_id="p")


def test_eleven_parameters_can_be_approved_in_index_order():
    # The previous check, sorted(params) != [str(i) ...], put "10" before "2" and rejected this.
    assert sorted(ELEVEN) != [str(index) for index in range(11)]
    assert _build(ELEVEN)["params"] == [f"v{index}" for index in range(11)]


@pytest.mark.parametrize("params", [{"0": 1, "2": 3}, {"01": 1}, {"a": 1}, {"1": 1}])
def test_malformed_parameter_keys_are_still_rejected(params):
    with pytest.raises(ApprovalConflict) as caught:
        _build(params)
    assert caught.value.code == "approval_action_invalid"


@pytest.mark.parametrize(
    "scope, tenant, visible",
    [
        ("A", "tenant-A", True),
        ("B", "tenant-A", False),
        ("A", "A", True),
        ("B", "A", False),
        ("B", "B", True),
        ("global", "A", True),
        ("global", "tenant-B", True),
    ],
)
def test_tenant_matches(scope, tenant, visible):
    assert tenant_matches(scope, tenant) is visible


@pytest.fixture
def repository(tmp_path):
    root = Path(__file__).resolve().parents[1] / "fixtures" / "knowledge"
    snapshot = build_snapshot(root, root / "source_registry.json", catalog_version="catalog-v1")
    state = StateStore(tmp_path / "knowledge.sqlite")
    repository = KnowledgeSnapshotRepository(state)
    repository.publish(snapshot)
    yield repository, snapshot.snapshot_id
    state.close()


@pytest.mark.parametrize("tenant, visible", [("A", True), ("tenant-A", True), ("B", False), ("tenant-B", False)])
def test_visible_source_follows_the_retrievers_tenant_rule(repository, tenant, visible):
    repo, snapshot_id = repository
    identity = KnowledgeIdentity(tenant, "p", "requester")
    call = lambda: repo.visible_source(snapshot_id=snapshot_id, source_id="tenant-a-orders-overview", identity=identity)  # noqa: E731
    if visible:
        assert call()["source_id"] == "tenant-a-orders-overview"
    else:
        with pytest.raises(KnowledgeAccessError):
            call()
