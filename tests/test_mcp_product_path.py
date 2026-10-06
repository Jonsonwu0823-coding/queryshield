"""The product with QUERYSHIELD_METADATA_TOOLS=mcp (HTTP, Fake model, fixture database).

Metadata calls go to the run's own MCP server process, queries stay local, the
run records how each call travelled and one record per session, and every
ending leaves no server process behind.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from queryshield.api.main import app, get_model_provider
from queryshield.approval.service import shared_run_service
from queryshield.knowledge.runtime import shared_retrieval_runtime
from queryshield.mcp_metadata.launch import LaunchSpec, product_launch
from queryshield.mcp_metadata.process import process_exists
from queryshield.tools.semantic import ControlledTools

from mcp_helpers import config
from test_http_queries import OTHER, REQUESTER, Scripted, ask, auth, env, wait  # noqa: F401  (env is a fixture)


DATA_QUESTION = "已支付订单有几笔"
DEFINITION_QUESTION = "退款后净额是怎么算的？"


@pytest.fixture()
def mcp(env, monkeypatch):
    monkeypatch.setenv("QUERYSHIELD_METADATA_TOOLS", "mcp")
    return env


def _events(run_id: str) -> list[dict]:
    return shared_run_service().store.events(run_id, after_event_id=0, limit=1000)


def _tool_events(run_id: str) -> list[dict]:
    return [e["payload"] for e in _events(run_id) if e["type"] == "agent_step" and e["payload"].get("kind") == "tool_call"]


def _sessions(run_id: str) -> list[dict]:
    return [e["payload"] for e in _events(run_id) if e["type"] == "metadata_session"]


def _assert_sessions_closed(run_id: str, count: int = 1) -> list[dict]:
    sessions = _sessions(run_id)
    assert len(sessions) == count, sessions
    for session in sessions:
        assert session["cleanup"] == "ok" and session["server_exited"] is True and session["cleanup_error"] is None
        assert process_exists(session["server_pid"]) is False
    return sessions


def _local_tools_forbidden(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("the local metadata tools must not run under the MCP setting")

    monkeypatch.setattr(ControlledTools, "search_catalog", refused)


def test_a_data_question_reads_metadata_over_mcp_and_queries_locally(mcp, monkeypatch):
    _local_tools_forbidden(monkeypatch)
    body = ask(mcp, DATA_QUESTION).json()
    assert body["status"] == "SUCCEEDED" and body["answer_status"] == "verified" and body["facts"]
    tools = _tool_events(body["run_id"])
    assert [(t["tool_name"], t.get("transport"), t.get("mcp_outcome")) for t in tools] == [
        ("search_catalog", "mcp_stdio", "ok"), ("query_readonly", None, None),
    ]
    assert tools[0]["mcp_request_sent"] is True
    [session] = _assert_sessions_closed(body["run_id"])
    assert session["session_id"] == tools[0]["mcp_session_id"]
    assert session["transport"] == "mcp_stdio" and session["sdk_version"] == "2.2.0"
    assert session["protocol_version"] == "2025-11-25" and session["tools_listed"] == ["search_catalog", "describe_tables"]
    assert (session["initialize"], session["list"], session["call_count"], session["retrieval"]) == ("ok", "ok", 1, "hybrid")
    assert session["knowledge_snapshot_id"] == shared_retrieval_runtime("fake").retriever.snapshot.snapshot_id
    types = [e["type"] for e in _events(body["run_id"])]
    assert types.index("metadata_session") == len(types) - 2 and types[-1] == "terminal"
    # Fixed fields only: no question or item text in the session record.
    assert DATA_QUESTION not in json.dumps(session, ensure_ascii=False)


def test_the_snapshot_ids_are_the_local_ones(env, monkeypatch):
    local = shared_run_service().store.get_run(ask(env, DATA_QUESTION).json()["run_id"])["run_config"]
    monkeypatch.setenv("QUERYSHIELD_METADATA_TOOLS", "mcp")
    remote = shared_run_service().store.get_run(ask(env, DATA_QUESTION).json()["run_id"])["run_config"]
    assert remote["knowledge_snapshot_id"] == local["knowledge_snapshot_id"]
    assert remote["agent_run_config"] == local["agent_run_config"]


def test_a_definition_answer_cites_only_sources_mcp_returned(mcp):
    body = ask(mcp, DEFINITION_QUESTION).json()
    assert body["status"] == "SUCCEEDED" and body["source_ids"]
    returned = {s for t in _tool_events(body["run_id"]) if t.get("transport") == "mcp_stdio" and t["status"] == "succeeded" for s in t["source_ids"]}
    assert set(body["source_ids"]) <= returned
    _assert_sessions_closed(body["run_id"])


KNOWLEDGE_WITHOUT_SOURCE = json.dumps(
    {"type": "final_answer", "answer": "净额是总额减退款。", "source_ids": [], "fact_refs": [], "basis": "knowledge"}, ensure_ascii=False
)


def test_the_server_search_for_a_knowledge_answer_goes_over_mcp(mcp):
    model = Scripted([KNOWLEDGE_WITHOUT_SOURCE, KNOWLEDGE_WITHOUT_SOURCE])
    app.dependency_overrides[get_model_provider] = lambda: model
    body = ask(mcp, DEFINITION_QUESTION).json()
    assert body["status"] == "SUCCEEDED" and body["source_ids"]
    [server_search] = [t for t in _tool_events(body["run_id"]) if t.get("initiated_by") == "server"]
    assert server_search["transport"] == "mcp_stdio" and server_search["mcp_outcome"] == "ok"
    assert set(body["source_ids"]) <= set(server_search["source_ids"])
    _assert_sessions_closed(body["run_id"])


def test_an_mcp_timeout_in_the_server_search_fails_with_mcp_timeout(mcp, monkeypatch):
    model = Scripted([KNOWLEDGE_WITHOUT_SOURCE, KNOWLEDGE_WITHOUT_SOURCE])
    app.dependency_overrides[get_model_provider] = lambda: model
    monkeypatch.setattr(shared_run_service(), "_metadata_tools", config(fixture="sleep", call_timeout=1.0))
    response = ask(mcp, DEFINITION_QUESTION)
    body = response.json()
    assert response.status_code == 504 and body["error"]["code"] == "mcp_timeout", body
    [server_search] = [t for t in _tool_events(body["run_id"]) if t.get("initiated_by") == "server"]
    assert (server_search["error_code"], server_search["mcp_outcome"]) == ("mcp_timeout", "timeout")
    [session] = _assert_sessions_closed(body["run_id"])
    assert session["failure_code"] == "mcp_timeout"


def test_identity_fields_from_the_model_never_reach_the_server(mcp):
    # The action parser refuses them before any tool runs (unknown_field), so no
    # session starts; the MCP host check would refuse them too (test_mcp_protocol).
    model = Scripted([json.dumps({"type": "tool_call", "name": "search_catalog", "arguments": {"query": "净额", "tenant_id": "B"}})])
    app.dependency_overrides[get_model_provider] = lambda: model
    body = ask(mcp, "净额").json()
    assert body["status"] != "SUCCEEDED" and body["error"]["code"] == "unknown_field"
    assert _sessions(body["run_id"]) == [] and _tool_events(body["run_id"]) == []


@pytest.mark.parametrize(
    "fixture, http_status, code",
    [("sleep", 504, "mcp_timeout"), ("die", 503, "mcp_unavailable"), ("extra_tool", 502, "mcp_protocol_error"), ("text_changed", 502, "mcp_result_invalid")],
)
def test_mcp_failures_fail_the_run_without_any_local_fallback(mcp, monkeypatch, fixture, http_status, code):
    _local_tools_forbidden(monkeypatch)
    monkeypatch.setattr(shared_run_service(), "_metadata_tools", config(fixture=fixture, call_timeout=1.0))
    response = ask(mcp, DATA_QUESTION)
    body = response.json()
    assert response.status_code == http_status and body["status"] == "FAILED" and body["error"]["code"] == code, body
    assert body["sql_exec_count"] == 0 and body["answer"] is None
    [tool] = _tool_events(body["run_id"])
    assert tool["tool_name"] == "search_catalog" and tool["error_code"] == code and tool["transport"] == "mcp_stdio"
    _assert_sessions_closed(body["run_id"])


def test_two_identities_at_once_get_their_own_server(mcp):
    first = ask(mcp, "Tenant A orders 已支付订单有几笔", asynchronous=True).json()
    second = ask(mcp, "Tenant B orders 已支付订单有几笔", token=OTHER, asynchronous=True).json()
    done = [wait(mcp, first["run_id"], {"SUCCEEDED"}, timeout=30), wait(mcp, second["run_id"], {"SUCCEEDED"}, token=OTHER, timeout=30)]
    assert all(run["status"] == "SUCCEEDED" for run in done)
    [a], [b] = _assert_sessions_closed(first["run_id"]), _assert_sessions_closed(second["run_id"])
    assert a["server_pid"] != b["server_pid"]
    a_sources = {s for t in _tool_events(first["run_id"]) for s in t.get("source_ids") or ()}
    b_sources = {s for t in _tool_events(second["run_id"]) for s in t.get("source_ids") or ()}
    assert "tenant-b-orders-overview" not in a_sources and "tenant-a-orders-overview" not in b_sources


def test_waiting_for_approval_closes_the_session(mcp):
    body = ask(mcp, "查询客户姓名").json()
    assert body["status"] == "WAITING_APPROVAL"
    _assert_sessions_closed(body["run_id"])
    types = [e["type"] for e in _events(body["run_id"])]
    assert types.index("metadata_session") < types.index("waiting")


def test_waiting_for_the_user_and_the_resume_each_close_their_session(mcp):
    body = ask(mcp, "销售额是多少").json()
    assert body["status"] == "WAITING_USER"
    before = _sessions(body["run_id"])  # the Fake model asks before any metadata call
    _assert_sessions_closed(body["run_id"], count=len(before))
    resumed = mcp.post(f"/runs/{body['run_id']}/resume", headers=auth(REQUESTER), json={"answer": "按支付订单总额，2026年7月"}).json()
    assert resumed["status"] == "SUCCEEDED"
    # The resume is a new execution with its own session, closed before the result.
    after = _assert_sessions_closed(body["run_id"], count=len(before) + 1)
    assert after[-1]["session_id"] not in {session["session_id"] for session in before}
    resumed_tools = [t for t in _tool_events(body["run_id"]) if t.get("mcp_session_id") == after[-1]["session_id"]]
    assert resumed_tools and all(t["transport"] == "mcp_stdio" for t in resumed_tools)


def test_a_cancelled_run_closes_its_session(mcp, monkeypatch):
    monkeypatch.setattr(shared_run_service(), "_metadata_tools", config(fixture="slow"))
    accepted = ask(mcp, DATA_QUESTION, asynchronous=True).json()
    time.sleep(0.3)  # the first metadata call is in flight
    cancelled = mcp.post(f"/runs/{accepted['run_id']}/cancel", headers=auth(REQUESTER), json={})
    assert cancelled.status_code in {200, 202}, cancelled.json()
    final = wait(mcp, accepted["run_id"], {"CANCELLED"}, timeout=30)
    assert final["status"] == "CANCELLED"
    _assert_sessions_closed(accepted["run_id"])


class _BreaksAfterTheSearch(Scripted):
    def complete(self, messages, **kwargs):
        if self.calls >= 1:
            raise RuntimeError("the model failed after the session opened")
        return super().complete(messages, **kwargs)


def test_a_model_exception_after_the_session_opened_still_closes_it(mcp):
    model = _BreaksAfterTheSearch([json.dumps({"type": "tool_call", "name": "search_catalog", "arguments": {"query": "净额"}})])
    app.dependency_overrides[get_model_provider] = lambda: model
    response = ask(mcp, "净额")
    body = response.json()
    assert body["status"] == "FAILED", body
    _assert_sessions_closed(body["run_id"])
    types = [e["type"] for e in _events(body["run_id"])]
    assert types.index("metadata_session") < types.index("terminal")


def _wrong_snapshot(ctx, retriever, mode, cwd):
    spec = product_launch(ctx, retriever, mode, cwd)
    args = tuple("--expected-snapshot-id=knowledge-v1-0000000000000000" if a.startswith("--expected-snapshot-id=") else a for a in spec.args)
    return LaunchSpec(spec.command, args, spec.env, spec.cwd, spec.retrieval, spec.knowledge_snapshot_id)


def test_an_index_that_is_not_the_runs_snapshot_never_starts(mcp, monkeypatch):
    monkeypatch.setattr(shared_run_service(), "_metadata_tools", config(launcher=_wrong_snapshot))
    response = ask(mcp, DATA_QUESTION)
    body = response.json()
    assert response.status_code == 503 and body["error"]["code"] == "mcp_unavailable"
    [session] = _sessions(body["run_id"])
    assert session["refused_reason"] == "index_mismatch" and session["initialize"] == "failed" and session["server_pid"] is None


SECRET_NAMES = (
    "QUERYSHIELD_DATABASE_URL",
    "QUERYSHIELD_MODEL_API_KEY",
    "QUERYSHIELD_MODEL_BASE_URL",
    "QUERYSHIELD_TOKEN_A_REQUESTER",
    "QUERYSHIELD_STATE_STORE_PATH",
    "QUERYSHIELD_CALL_STORE_PATH",
    "QUERYSHIELD_DEMO_DATASET",
    "QUERYSHIELD_RETRIEVAL",
    "QUERYSHIELD_METADATA_TOOLS",
)


@pytest.mark.skipif(not __import__("sys").platform.startswith("linux"), reason="reads the server's environment from /proc")
def test_the_server_process_environment_is_a_whitelist(monkeypatch):
    from mcp_helpers import context, started_session

    for name in SECRET_NAMES:
        monkeypatch.setenv(name, "secret-value")
    monkeypatch.setenv("QUERYSHIELD_EMBEDDING_API_KEY", "embedding-key")
    session = started_session(context("A"), shared_retrieval_runtime("fake").retriever)
    try:
        raw = Path(f"/proc/{session.server_pid}/environ").read_bytes().split(b"\0")
        names = {item.split(b"=", 1)[0].decode() for item in raw if item}
    finally:
        session.close()
    assert not names & set(SECRET_NAMES)
    assert not any(name.startswith("QUERYSHIELD_") for name in names)  # Fake: not even the embedding settings
    assert {"PYTHONPATH", "PYTHONSAFEPATH", "PYTHONNOUSERSITE"} <= names


def test_only_hybrid_real_passes_the_embedding_settings(monkeypatch, tmp_path):
    from mcp_helpers import context

    for name in SECRET_NAMES:
        monkeypatch.setenv(name, "secret-value")
    for name in ("QUERYSHIELD_EMBEDDING_BASE_URL", "QUERYSHIELD_EMBEDDING_API_KEY", "QUERYSHIELD_EMBEDDING_MODEL_NAME"):
        monkeypatch.setenv(name, "x")
    retriever = shared_retrieval_runtime("fake").retriever
    real_hybrid = product_launch(context("A"), retriever, "real", str(tmp_path)).env
    fake_hybrid = product_launch(context("A"), retriever, "fake", str(tmp_path)).env
    real_keyword = product_launch(context("A"), None, "real", str(tmp_path)).env
    python_only = {"PYTHONPATH", "PYTHONSAFEPATH", "PYTHONNOUSERSITE", "PYTHONUTF8", "PYTHONIOENCODING"}
    assert set(real_hybrid) == python_only | {"QUERYSHIELD_EMBEDDING_BASE_URL", "QUERYSHIELD_EMBEDDING_API_KEY", "QUERYSHIELD_EMBEDDING_MODEL_NAME"}
    assert set(fake_hybrid) == set(real_keyword) == python_only
