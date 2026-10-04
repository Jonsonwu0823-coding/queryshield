"""B2b follow-up: an unreachable database fails fast instead of waiting for the OS TCP timeout."""

from __future__ import annotations

import socket
import time

import pytest

from queryshield.agent.proposals import ExecutionContext
from queryshield.agent.tool_execution import call_tool
from queryshield.db import readonly
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.tools.semantic import ControlledTools, ToolError


def _capture_connect(monkeypatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_connect(conninfo, **kwargs):
        captured["conninfo"] = conninfo
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", "postgresql://queryshield_ro@127.0.0.1:1/queryshield_test")
    monkeypatch.setattr(readonly.psycopg, "connect", fake_connect)
    return captured


def test_connect_readonly_sets_a_default_connect_timeout(monkeypatch) -> None:
    monkeypatch.delenv("QUERYSHIELD_DB_CONNECT_TIMEOUT", raising=False)
    captured = _capture_connect(monkeypatch)
    readonly.connect_readonly()
    assert captured["connect_timeout"] == 5
    assert "default_transaction_read_only=on" in str(captured["options"])


@pytest.mark.parametrize("value", ["1", "12", "30", " 7 "])
def test_connect_timeout_can_be_overridden_within_range(monkeypatch, value) -> None:
    monkeypatch.setenv("QUERYSHIELD_DB_CONNECT_TIMEOUT", value)
    captured = _capture_connect(monkeypatch)
    readonly.connect_readonly()
    assert captured["connect_timeout"] == int(value)


@pytest.mark.parametrize("value", ["0", "31", "-1", "abc", "5.5", "1e1", "５", "999"])
def test_invalid_connect_timeout_is_refused_not_ignored(monkeypatch, value) -> None:
    monkeypatch.setenv("QUERYSHIELD_DB_CONNECT_TIMEOUT", value)
    captured = _capture_connect(monkeypatch)
    with pytest.raises(readonly.DatabaseConfigurationError) as caught:
        readonly.connect_readonly()
    assert caught.value.code == "invalid_database_configuration"
    assert captured == {}


def test_unresponsive_database_ends_as_database_unavailable_within_the_timeout(monkeypatch) -> None:
    # A listener that accepts TCP but never answers the PostgreSQL handshake:
    # without connect_timeout the client would wait indefinitely.
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    port = server.getsockname()[1]
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", f"postgresql://queryshield_ro:x@127.0.0.1:{port}/queryshield_test")
    monkeypatch.setenv("QUERYSHIELD_DB_CONNECT_TIMEOUT", "1")
    tools = ControlledTools(executor=GuardedQueryExecutor())
    context = ExecutionContext(run_id="run-timeout", tenant_id="A", principal_id="p", role="requester")
    started = time.monotonic()
    try:
        with pytest.raises(ToolError) as caught:
            call_tool(
                tools,
                "query_readonly",
                {"sql": "SELECT COUNT(*) AS paid_count FROM orders AS o WHERE o.status = %s", "params": {"0": "paid"}},
                context=context,
            )
    finally:
        server.close()
    elapsed = time.monotonic() - started
    assert caught.value.code == "database_unavailable"
    assert elapsed < 10
