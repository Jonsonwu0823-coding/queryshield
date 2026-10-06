"""QUERYSHIELD_METADATA_TOOLS: local by default, mcp only in the product service.

A bad value is a fixed 503 before any run exists.  A directly constructed
service (evaluation, checks, tests) stays local whatever the environment says,
and the local setting adds no field and no event anywhere.
"""

from __future__ import annotations

import json

import pytest

from queryshield.approval.service import FixtureQueryExecutor, RunService, shared_run_service
from queryshield.db.state_store import StateStore
from queryshield.mcp_metadata.launch import (
    MetadataToolsConfigurationError,
    call_timeout_seconds,
    metadata_tools_setting,
    resolve_product_config,
)
from queryshield.mcp_metadata.process import process_exists

from test_http_queries import ask, env  # noqa: F401  (env is a fixture)


IDENTITY = {"tenant_id": "A", "principal_id": "a-requester", "role": "requester"}


@pytest.mark.parametrize("value, expected", [(None, "local"), ("", "local"), ("local", "local"), (" MCP ", "mcp")])
def test_the_setting_values(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("QUERYSHIELD_METADATA_TOOLS", raising=False)
    else:
        monkeypatch.setenv("QUERYSHIELD_METADATA_TOOLS", value)
    assert metadata_tools_setting() == expected
    assert (resolve_product_config("fake") is None) == (expected == "local")


@pytest.mark.parametrize("value", ["remote", "stdio", "1", "true", "mcp-http"])
def test_any_other_setting_is_a_configuration_error(monkeypatch, value):
    monkeypatch.setenv("QUERYSHIELD_METADATA_TOOLS", value)
    with pytest.raises(MetadataToolsConfigurationError):
        metadata_tools_setting()


@pytest.mark.parametrize("value, seconds", [("", 2.0), ("1", 1.0), ("2.5", 2.5), ("10", 10.0)])
def test_the_call_timeout_setting(monkeypatch, value, seconds):
    monkeypatch.setenv("QUERYSHIELD_MCP_CALL_TIMEOUT_SECONDS", value)
    assert call_timeout_seconds() == seconds


@pytest.mark.parametrize("value", ["0.5", "11", "abc", "nan", "inf", "-2"])
def test_a_call_timeout_outside_one_to_ten_seconds_is_refused(monkeypatch, value):
    monkeypatch.setenv("QUERYSHIELD_MCP_CALL_TIMEOUT_SECONDS", value)
    with pytest.raises(MetadataToolsConfigurationError):
        call_timeout_seconds()


@pytest.mark.parametrize(
    "name, value",
    [("QUERYSHIELD_METADATA_TOOLS", "remote"), ("QUERYSHIELD_MCP_CALL_TIMEOUT_SECONDS", "60")],
)
def test_the_product_refuses_a_bad_value_with_503_before_any_run(env, monkeypatch, name, value):
    monkeypatch.setenv("QUERYSHIELD_METADATA_TOOLS", "mcp")
    monkeypatch.setenv(name, value)
    for asynchronous in (False, True):
        response = ask(env, "已支付订单有几笔", asynchronous=asynchronous)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "invalid_metadata_tools_configuration"
    assert shared_run_service()._active_count() == 0


def _no_mcp_anywhere(store, run_id):
    events = store.events(run_id, after_event_id=0, limit=1000)
    assert not [e for e in events if e["type"] == "metadata_session"]
    dumped = json.dumps(events, ensure_ascii=False)
    for marker in ("mcp_stdio", "transport", "mcp_session_id", "mcp_outcome", "mcp_request_sent"):
        assert marker not in dumped


def test_a_directly_constructed_service_stays_local_whatever_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYSHIELD_METADATA_TOOLS", "mcp")
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.delenv("QUERYSHIELD_RETRIEVAL", raising=False)
    store = StateStore(tmp_path / "own.sqlite3")
    service = RunService(store=store, executor_factory=FixtureQueryExecutor, mode="fake")
    assert service.metadata_config() is None
    run = service.run_sync(identity=IDENTITY, question="已支付订单有几笔")
    assert run["status"] == "SUCCEEDED"
    _no_mcp_anywhere(store, run["run_id"])
    store.close()


def test_the_local_setting_adds_nothing_to_the_product_records(env, monkeypatch):
    monkeypatch.delenv("QUERYSHIELD_METADATA_TOOLS", raising=False)
    for question in ("已支付订单有几笔", "退款后净额是怎么算的？", "查询客户姓名"):
        body = ask(env, question).json()
        _no_mcp_anywhere(shared_run_service().store, body["run_id"])


def test_an_explicit_mcp_service_with_catalog_search_uses_the_keyword_server(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.setenv("QUERYSHIELD_RETRIEVAL", "catalog")
    store = StateStore(tmp_path / "own.sqlite3")
    service = RunService(store=store, executor_factory=FixtureQueryExecutor, mode="fake", metadata_tools="mcp")
    run = service.run_sync(identity=IDENTITY, question="已支付订单有几笔")
    assert run["status"] == "SUCCEEDED"
    [session] = [e["payload"] for e in store.events(run["run_id"], after_event_id=0, limit=1000) if e["type"] == "metadata_session"]
    assert session["retrieval"] == "keyword" and session["knowledge_snapshot_id"] is None
    assert session["cleanup"] == "ok" and process_exists(session["server_pid"]) is False
    store.close()


def test_b0_reads_its_table_description_over_mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "b0")
    monkeypatch.delenv("QUERYSHIELD_RETRIEVAL", raising=False)
    store = StateStore(tmp_path / "own.sqlite3")
    service = RunService(store=store, executor_factory=FixtureQueryExecutor, mode="fake", metadata_tools="mcp")
    run = service.run_sync(identity=IDENTITY, question="已支付订单有几笔")
    [session] = [e["payload"] for e in store.events(run["run_id"], after_event_id=0, limit=1000) if e["type"] == "metadata_session"]
    assert session["call_count"] == 1 and session["retrieval"] == "keyword" and session["cleanup"] == "ok"
    store.close()
