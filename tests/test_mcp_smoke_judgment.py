"""The MCP smoke's judgments, on built inputs (no process, no model, no database)."""

from __future__ import annotations

import pytest

from scripts.mcp_smoke import judge_product_step, judge_protocol, judge_sessions


MCP_SEARCH = {"tool_name": "search_catalog", "status": "succeeded", "transport": "mcp_stdio", "source_ids": ["semantic-metric-net", "commerce-v1"]}
LOCAL_QUERY = {"tool_name": "query_readonly", "status": "succeeded"}


def _step(step="data", *, http_status=200, status="SUCCEEDED", answer_status="verified", tools=(MCP_SEARCH, LOCAL_QUERY), sources=()):
    return judge_product_step(step, http_status=http_status, status=status, answer_status=answer_status, tool_events=list(tools), answer_source_ids=list(sources))


def test_a_data_question_over_mcp_passes():
    assert _step() == ([], [])


def test_a_data_question_without_metadata_calls_is_a_known_gap_only():
    assert _step(tools=(LOCAL_QUERY,)) == ([], ["metadata_tools_not_used_by_model"])


@pytest.mark.parametrize(
    "kwargs, failure",
    [
        ({"tools": ({**MCP_SEARCH, "transport": None}, LOCAL_QUERY)}, "metadata_call_not_over_mcp:data"),
        ({"tools": ({"tool_name": "describe_tables", "status": "succeeded"}, LOCAL_QUERY)}, "metadata_call_not_over_mcp:data"),
        ({"tools": (MCP_SEARCH, {**LOCAL_QUERY, "transport": "mcp_stdio"})}, "query_left_the_host:data"),
        ({"http_status": 503, "status": "FAILED"}, "server_error:data"),
        ({"status": "FAILED", "http_status": 422}, "not_succeeded:data"),
        ({"answer_status": "unverified"}, "data_not_verified"),
    ],
)
def test_data_hard_failures(kwargs, failure):
    failures, _ = _step(**kwargs)
    assert failure in failures


def test_a_definition_answer_must_cite_only_sources_mcp_returned():
    assert _step("definition", answer_status="unverified", tools=(MCP_SEARCH,), sources=["semantic-metric-net"]) == ([], [])
    server_search = {**MCP_SEARCH, "initiated_by": "server"}
    assert _step("definition", tools=(server_search,), sources=["commerce-v1"]) == ([], ["knowledge_after_send_back"])
    for tools, sources in [
        ((MCP_SEARCH,), ["semantic-refund-policy-v2"]),
        ((MCP_SEARCH,), []),
        (({**MCP_SEARCH, "status": "failed"},), ["semantic-metric-net"]),
        (({**MCP_SEARCH, "transport": None},), ["semantic-metric-net"]),
    ]:
        failures, _ = _step("definition", tools=tools, sources=sources)
        assert "definition_sources_not_from_mcp" in failures


def test_a_search_the_model_made_is_not_a_known_gap():
    model_search = {**MCP_SEARCH, "initiated_by": "model"}
    for tools in ((MCP_SEARCH,), (model_search,)):
        assert _step("definition", answer_status="unverified", tools=tools, sources=["semantic-metric-net"]) == ([], [])


def test_a_search_the_server_added_is_the_b2b_known_gap_and_changes_no_hard_failure():
    server_search = {**MCP_SEARCH, "initiated_by": "server"}
    # The model searched first and the server searched again after the send-back.
    failures, gaps = _step("definition", tools=(MCP_SEARCH, server_search), sources=["semantic-metric-net"])
    assert (failures, gaps) == ([], ["knowledge_after_send_back"])
    # A server search over local transport is still a hard failure, and the gap is still recorded.
    failures, gaps = _step("definition", tools=({**server_search, "transport": None},), sources=["semantic-metric-net"])
    assert failures == ["metadata_call_not_over_mcp:definition", "definition_sources_not_from_mcp"] and gaps == ["knowledge_after_send_back"]
    # A data question never records it.
    assert _step("data", tools=(server_search, LOCAL_QUERY)) == ([], [])


def test_sessions_must_be_cleaned_and_their_processes_gone():
    clean = {"server_pid": 11, "cleanup": "ok", "server_exited": True}
    assert judge_sessions([clean], gone={11: True}) == []
    assert judge_sessions([clean], gone={11: False}) == ["leftover_server_process"]
    assert judge_sessions([{**clean, "cleanup": "failed"}], gone={11: True}) == ["session_not_cleaned"]
    assert judge_sessions([{**clean, "server_exited": None}], gone={}) == ["leftover_server_process", "session_not_cleaned"]


def test_every_protocol_check_must_hold():
    assert judge_protocol({"initialize": True, "two_valid_calls": True}) == []
    assert judge_protocol({"initialize": True, "unknown_tool_is_a_protocol_error": False, "x": None}) == [
        "protocol:unknown_tool_is_a_protocol_error", "protocol:x",
    ]


# -- R1: the state store is read while the server runs; errors still write the summary --

import json  # noqa: E402
import sqlite3  # noqa: E402

import scripts.mcp_smoke as smoke  # noqa: E402


class _FakeServer:
    def __init__(self, log, *, port_closes=True):
        self.base, self.port, self.stopped = "http://127.0.0.1:1", 1, False
        self.log, self.port_closes = log, port_closes

    def stop(self):
        self.log.append("stop")
        self.stopped = True
        return self.port_closes


SESSION = {"server_pid": 4242, "cleanup": "ok", "server_exited": True}
BODIES = {
    smoke.DATA_QUESTION: {"run_id": "run-data", "status": "SUCCEEDED", "answer_status": "verified", "source_ids": ["commerce-v1"]},
    smoke.DEFINITION_QUESTION: {"run_id": "run-definition", "status": "SUCCEEDED", "answer_status": "unverified", "source_ids": ["semantic-metric-net"]},
}


def _patch_product(monkeypatch, log, *, port_closes=True, read_error=None):
    server = _FakeServer(log, port_closes=port_closes)
    monkeypatch.setattr(smoke, "_database_ready", lambda: True)
    monkeypatch.setattr(smoke, "_start_server", lambda env: server)
    monkeypatch.setattr(smoke, "_ask", lambda base, token, question: (200, BODIES[question]))

    def read(state_path, run_id):
        log.append(f"read:{run_id}:server_stopped={server.stopped}")
        if read_error is not None:
            raise read_error
        return [MCP_SEARCH, LOCAL_QUERY], [dict(SESSION)]

    def gone(pids, timeout=5.0):
        log.append(f"pids:server_stopped={server.stopped}")
        return {pid: True for pid in pids}

    monkeypatch.setattr(smoke, "_read_run", read)
    monkeypatch.setattr(smoke, "_pids_gone", gone)
    return server


def test_the_product_part_reads_the_store_before_stopping_the_server(monkeypatch):
    log: list[str] = []
    _patch_product(monkeypatch, log)
    records, failures, gaps = smoke.run_product("fake")
    assert failures == [] and gaps == [] and [record["step"] for record in records] == ["data", "definition"]
    assert log == [
        "read:run-data:server_stopped=False",
        "read:run-definition:server_stopped=False",
        "stop",
        "pids:server_stopped=True",
    ]


def test_a_port_that_still_answers_after_the_stop_is_a_hard_failure(monkeypatch):
    _patch_product(monkeypatch, [], port_closes=False)
    _, failures, _ = smoke.run_product("fake")
    assert "server_still_listening" in failures


def test_a_store_error_stops_the_server_and_still_writes_the_summary(monkeypatch, tmp_path):
    log: list[str] = []
    server = _patch_product(monkeypatch, log, read_error=sqlite3.OperationalError("disk I/O error at C:\\secret"))
    monkeypatch.setattr(smoke, "run_protocol", lambda: ({"initialize": True}, {}))
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", "postgresql://unused")
    assert smoke.main(["--mode", "fake", "--evidence-dir", str(tmp_path)]) == 1
    assert server.stopped
    summary = json.loads((tmp_path / smoke.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["status"] == "fail" and summary["hard_failures"] == ["smoke_error:product:OperationalError"]
    assert "secret" not in json.dumps(summary)


def test_any_error_in_either_part_still_writes_the_summary(monkeypatch, tmp_path):
    def protocol_fails():
        raise RuntimeError("含敏感文字 sk-secret")

    def product_fails(mode):
        raise OSError("C:\\Users\\secret")

    monkeypatch.setattr(smoke, "run_protocol", protocol_fails)
    monkeypatch.setattr(smoke, "run_product", product_fails)
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", "postgresql://unused")
    assert smoke.main(["--mode", "fake", "--evidence-dir", str(tmp_path)]) == 1
    summary = json.loads((tmp_path / smoke.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["hard_failures"] == ["smoke_error:product:OSError", "smoke_error:protocol:RuntimeError"]
    assert summary["protocol"] == {"error": "RuntimeError"} and summary["product"] == {"error": "OSError"}
    assert "secret" not in json.dumps(summary, ensure_ascii=False) and "敏感" not in json.dumps(summary, ensure_ascii=False)


def test_the_port_wait_ends_when_nothing_listens():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert smoke._wait_port_closed(port, timeout=1.0) is True
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        assert smoke._wait_port_closed(listener.getsockname()[1], timeout=0.5) is False
