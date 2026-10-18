"""A follow-up of an approved names query on the real test database: no name reaches the next turn.

The fixture database ignores column aliases; PostgreSQL returns the output names the
SQL wrote, so a name can come back under ``customer_id`` (review V1).  Needs the
test database and skips without it.
"""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from queryshield.api.main import app, get_model_provider
from queryshield.approval.service import reset_shared_state_stores, shared_run_service

from test_fake_upstream_real_mode import _reachable
from test_follow_up_api import PREVIOUS_TURN, previous_turn
from test_http_queries import APPROVER, OTHER, REQUESTER, CountingFake, Scripted, auth


pytestmark = pytest.mark.skipif(not _reachable(os.environ.get("QUERYSHIELD_DATABASE_URL")), reason="needs the test database")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYSHIELD_STATE_STORE_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.delenv("QUERYSHIELD_FAKE_DB", raising=False)
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    for name in ("QUERYSHIELD_AGENT_PROFILE", "QUERYSHIELD_RETRIEVAL", "QUERYSHIELD_METADATA_TOOLS", "QUERYSHIELD_DEMO_DATASET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", REQUESTER)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_APPROVER", APPROVER)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_B_REQUESTER", OTHER)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_B_APPROVER", f"{OTHER}-approver")
    reset_shared_state_stores()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    reset_shared_state_stores()


def _names_query(sql: str) -> str:
    return json.dumps({"type": "tool_call", "name": "query_readonly", "arguments": {"sql": sql, "params": {}}}, ensure_ascii=False)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT c.customer_id, c.name FROM customers AS c ORDER BY c.customer_id",
        "SELECT c.customer_id, c.name AS customer_id FROM customers AS c ORDER BY c.customer_id",
        "SELECT c.name AS Customer_Id, c.customer_id FROM customers AS c ORDER BY c.customer_id",
        'SELECT c.name AS "customer_id", c.customer_id AS "CUSTOMER_ID" FROM customers AS c ORDER BY c.customer_id',
        "SELECT c.name AS id, c.customer_id FROM customers AS c ORDER BY c.customer_id",
    ],
    ids=["plain", "same_name", "differs_in_case", "quoted", "renamed"],
)
def test_an_approved_names_query_carries_no_name_into_the_next_turn(client, sql) -> None:
    app.dependency_overrides[get_model_provider] = lambda: Scripted([_names_query(sql)])
    pending = client.post("/queries", headers=auth(REQUESTER), json={"question": "查询客户姓名"}).json()
    assert pending["status"] == "WAITING_APPROVAL", pending
    approved = client.post(
        f"/runs/{pending['run_id']}/approval",
        headers=auth(APPROVER),
        json={"approval_id": pending["approval_id"], "decision": "approve"},
    ).json()
    assert approved["status"] == "SUCCEEDED", approved
    result = client.get(f"/runs/{pending['run_id']}/result", headers=auth(REQUESTER)).json()
    names = {
        cell for row in result["result"]["rows"] for cell in row.values()
        if isinstance(cell, str) and not cell.startswith("c")
    }
    assert names, result["result"]["rows"]

    model = CountingFake()
    app.dependency_overrides[get_model_provider] = lambda: model
    second = client.post(
        "/queries", headers=auth(REQUESTER), json={"question": "他们2026年9月的支付金额呢", "previous_run_id": pending["run_id"]}
    ).json()

    store = shared_run_service().store
    seen = json.dumps(
        {"messages": model.messages, "response": second, "run": store.get_run(second["run_id"]), "events": store.events(second["run_id"])},
        ensure_ascii=False,
        default=str,
    )
    assert PREVIOUS_TURN.strip() in seen
    assert not [name for name in names if name in seen]
    carried = previous_turn(model.messages[0])
    assert carried["customer_ids"] is None and carried["customer_count"] is None and carried["listed_customer_names"] is True
