"""MCP protocol over a real stdio server process (the product server module itself).

initialize -> tools/list -> tools/call -> close; argument and identity errors are
tool errors (isError), unknown tools and malformed requests are protocol
errors; timeouts and a dying server fail closed; after close the pid the server
reported is gone; stdout carries JSON-RPC messages only.
"""

from __future__ import annotations

import json
import subprocess
import time

import anyio
import mcp_types as types
from mcp.shared.exceptions import MCPError
import pytest

from queryshield.knowledge.runtime import shared_retrieval_runtime
from queryshield.mcp_metadata.launch import product_launch
from queryshield.mcp_metadata.process import process_exists
from queryshield.mcp_metadata.schemas import PROTOCOL_VERSION, TOOL_NAMES, TOOLS, tool_definitions
from queryshield.mcp_metadata.session import McpMetadataSession, McpSessionError
from queryshield.mcp_metadata.tools import McpMetadataTools, McpToolError

from mcp_helpers import config, context, raw_requests, started_session


@pytest.fixture(scope="module")
def session():
    live = started_session(context("A"), shared_retrieval_runtime("fake").retriever)
    yield live
    record = live.close()
    assert record["cleanup"] == "ok" and record["server_exited"] is True


def _payload(result) -> dict:
    [content] = result.content
    return json.loads(content.text)


def test_initialize_and_tools_list_are_exactly_the_shared_definitions(session, tmp_path):
    record = session.record
    assert record["initialize"] == "ok" and record["list"] == "ok"
    assert record["protocol_version"] == PROTOCOL_VERSION == "2025-11-25"
    assert record["tools_listed"] == list(TOOL_NAMES)
    assert record["sdk_version"] == "2.2.0" and record["server_pid"] and process_exists(record["server_pid"])
    spec = product_launch(context("A"), None, "fake", str(tmp_path))
    [listed] = anyio.run(raw_requests, spec, [lambda client: client.list_tools()])
    as_data = [tool.model_dump(by_alias=True, exclude_none=True) for tool in listed.tools]
    assert as_data == tool_definitions()
    for tool in listed.tools:
        assert tool.input_schema["additionalProperties"] is False
        assert not set(tool.input_schema["properties"]) & {"tenant_id", "principal_id", "role", "token"}
        assert tool.annotations.read_only_hint is True


@pytest.mark.parametrize(
    "name, arguments",
    [("search_catalog", {"query": "退款后净额", "top_k": 3}), ("describe_tables", {"tables": ["orders", "refunds"]})],
)
def test_valid_calls_return_structured_content_with_the_same_json_text(session, name, arguments):
    from jsonschema import Draft202012Validator

    result = session.call(name, arguments)
    assert result.is_error is False
    assert Draft202012Validator(TOOLS[name]["output_schema"]).is_valid(result.structured_content)
    assert _payload(result) == result.structured_content


INVALID = [
    ("search_catalog", {}, "missing_argument"),
    ("search_catalog", {"top_k": 3}, "missing_argument"),
    ("search_catalog", {"query": "净额", "top_k": 0}, "invalid_argument"),
    ("search_catalog", {"query": "净额", "top_k": 6}, "invalid_argument"),
    ("search_catalog", {"query": "净额", "top_k": "3"}, "invalid_argument"),
    ("search_catalog", {"query": "净额", "top_k": 3.0}, "invalid_argument"),
    ("search_catalog", {"query": "净额", "top_k": True}, "invalid_argument"),
    ("search_catalog", {"query": "   "}, "invalid_argument"),
    ("search_catalog", {"query": "x" * 201}, "invalid_argument"),
    ("describe_tables", {}, "missing_argument"),
    ("describe_tables", {"tables": []}, "invalid_argument"),
    ("describe_tables", {"tables": ["orders", "refunds", "customers", "orders2"]}, "invalid_argument"),
    ("describe_tables", {"tables": ["orders", "orders"]}, "invalid_argument"),
    ("describe_tables", {"tables": "orders"}, "invalid_argument"),
    ("describe_tables", {"tables": ["pg_shadow"]}, "table_not_allowed"),
]


@pytest.mark.parametrize("name, arguments, code", INVALID)
def test_invalid_arguments_are_tool_errors_without_data(session, name, arguments, code):
    result = session.call(name, arguments)
    assert result.is_error is True and result.structured_content is None
    assert _payload(result)["error_code"] == code


@pytest.mark.parametrize("field", ["tenant_id", "role", "principal_id", "token"])
def test_identity_fields_are_refused_and_the_session_identity_stays(session, field):
    result = session.call("search_catalog", {"query": "订单概览", "top_k": 5, field: "B"})
    assert result.is_error is True and _payload(result)["error_code"] == "unknown_argument"
    sources = {item["source_id"] for item in session.call("search_catalog", {"query": "租户 订单 概览", "top_k": 5}).structured_content["items"]}
    assert "tenant-b-orders-overview" not in sources


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
def test_unknown_tools_and_malformed_requests_are_protocol_errors(tmp_path):
    spec = product_launch(context("A"), None, "fake", str(tmp_path))

    async def bad_arguments(client):
        return await client.send_request(
            types.CallToolRequest(params=types.CallToolRequestParams.model_construct(name="search_catalog", arguments=["x"])),
            types.CallToolResult,
        )

    results = anyio.run(
        raw_requests,
        spec,
        [
            lambda client: client.call_tool("query_readonly", {"sql": "select 1", "params": {}}),
            lambda client: client.call_tool("drop_everything", {}),
            bad_arguments,
            lambda client: client.call_tool("describe_tables", {"tables": ["orders"]}),
        ],
    )
    for result in results[:3]:
        assert isinstance(result, MCPError) and result.error.code == types.INVALID_PARAMS
    # Nothing executed for the unknown tools; the session still serves the real ones.
    assert results[3].is_error is False


def test_the_host_never_sends_an_unknown_tool(session):
    with pytest.raises(McpSessionError) as caught:
        session.call("query_readonly", {"sql": "select 1", "params": {}})
    assert caught.value.code == "mcp_protocol_error"


@pytest.mark.parametrize(
    "edit, reason",
    [
        (lambda args: [item for item in args if not item.startswith("--tenant-id=")], "invalid_arguments"),
        (lambda args: [("--role=admin" if item.startswith("--role=") else item) for item in args], "invalid_role"),
        (lambda args: [("--principal-id= " if item.startswith("--principal-id=") else item) for item in args], "empty_identity"),
        (lambda args: [("--package-dir=/tmp" if item.startswith("--package-dir=") else item) for item in args], "package_mismatch"),
        (lambda args: [("--expected-snapshot-id=knowledge-v1-0000" if item.startswith("--expected-snapshot-id=") else item) for item in args], "index_mismatch"),
    ],
)
def test_a_server_without_a_valid_launch_refuses_to_start(edit, reason):
    retriever = shared_retrieval_runtime("fake").retriever
    with pytest.raises(McpSessionError) as caught:
        started_session(context("A"), retriever, args_edit=edit)
    assert caught.value.code == "mcp_unavailable"


def test_a_refusal_is_recorded_with_its_fixed_reason_and_no_process_left():
    tools = McpMetadataTools(retriever=None, metadata_config=config(launcher=_refusing_launcher))
    with pytest.raises(McpToolError) as caught:
        tools.search_catalog({"query": "净额"}, context=context("A"))
    assert caught.value.code == "mcp_unavailable"
    record = tools.close()
    assert record["refused_reason"] == "invalid_role" and record["initialize"] == "failed"
    assert record["server_pid"] is None and record["failure_code"] == "mcp_unavailable"


def _refusing_launcher(ctx, retriever, mode, cwd):
    from queryshield.mcp_metadata.launch import LaunchSpec

    spec = product_launch(ctx, retriever, mode, cwd)
    args = tuple("--role=admin" if item.startswith("--role=") else item for item in spec.args)
    return LaunchSpec(spec.command, args, spec.env, spec.cwd, spec.retrieval, spec.knowledge_snapshot_id)


@pytest.mark.parametrize("fixture", ["sleep"])
def test_a_slow_call_times_out_and_the_server_is_gone(fixture):
    tools = McpMetadataTools(retriever=None, metadata_config=config(fixture=fixture, call_timeout=1.0))
    started = time.monotonic()
    with pytest.raises(McpToolError) as caught:
        tools.search_catalog({"query": "净额"}, context=context("A"))
    elapsed = time.monotonic() - started
    assert caught.value.code == "mcp_timeout"
    record = tools.close()
    assert elapsed < 1.0 + 6.0 + record["started_ms"] / 1000
    assert record["failure_code"] == "mcp_timeout" and record["cleanup"] == "ok" and record["server_exited"] is True
    assert process_exists(record["server_pid"]) is False
    assert tools.take_metadata_call_record() == {
        "transport": "mcp_stdio", "mcp_session_id": record["session_id"], "mcp_request_sent": True, "mcp_outcome": "timeout",
    }
    # Failed closed: no second session, no local fallback.
    with pytest.raises(McpToolError) as again:
        tools.search_catalog({"query": "净额"}, context=context("A"))
    assert again.value.code == "mcp_unavailable" and tools.close()["call_count"] == 1


def test_a_server_that_dies_mid_call_is_unavailable_and_gone():
    tools = McpMetadataTools(retriever=None, metadata_config=config(fixture="die"))
    with pytest.raises(McpToolError) as caught:
        tools.describe_tables({"tables": ["orders"]}, context=context("A"))
    assert caught.value.code == "mcp_unavailable"
    record = tools.close()
    assert record["cleanup"] == "ok" and process_exists(record["server_pid"]) is False


def test_stdout_carries_only_json_rpc_messages(tmp_path):
    spec = product_launch(context("A"), shared_retrieval_runtime("fake").retriever, "fake", str(tmp_path))
    from mcp.client.stdio import get_default_environment

    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "raw", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "search_catalog", "arguments": {"query": "退款后净额", "top_k": 5}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "describe_tables", "arguments": {"tables": ["customers"]}}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "search_catalog", "arguments": {"query": "x", "role": "approver"}}},
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "nope", "arguments": {}}},
    ]
    process = subprocess.Popen(
        [spec.command, *spec.args],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**get_default_environment(), **spec.env}, cwd=spec.cwd,
    )
    lines = []
    assert process.stdin is not None and process.stdout is not None
    for message in messages:
        process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
        process.stdin.flush()
        if "id" in message:
            lines.append(process.stdout.readline())
    process.stdin.close()
    lines.extend(process.stdout.read().splitlines(keepends=True))
    process.wait(timeout=10)
    stderr = process.stderr.read().decode("utf-8").splitlines() if process.stderr else []
    parsed = [json.loads(line) for line in lines if line.strip()]
    assert [message.get("id") for message in parsed] == [1, 2, 3, 4, 5, 6]
    assert all(message["jsonrpc"] == "2.0" and ("result" in message or "error" in message) for message in parsed)
    assert parsed[5]["error"]["code"] == types.INVALID_PARAMS
    assert process.returncode == 0  # it exits by itself when stdin closes
    assert len(stderr) == 1 and stderr[0].startswith("queryshield-mcp-metadata ready pid=")


def test_close_is_idempotent_and_reports_the_server_gone():
    session = started_session(context("B"))
    pid = session.server_pid
    assert process_exists(pid) is True
    first = session.close()
    assert session.close() == first and first["cleanup"] == "ok" and process_exists(pid) is False
    with pytest.raises(McpSessionError):
        session.call("search_catalog", {"query": "净额"})


def test_sessions_never_outlive_a_closed_facade_after_tool_errors():
    tools = McpMetadataTools(retriever=None, metadata_config=config())
    with pytest.raises(Exception) as caught:
        tools.describe_tables({"tables": ["orders"], "tenant_id": "B"}, context=context("A"))
    assert getattr(caught.value, "code", None) == "unknown_argument"
    assert tools.close() is None  # nothing was sent, no process was started
    assert McpMetadataSession  # imported for the type


def test_a_session_bound_to_one_identity_refuses_another():
    tools = McpMetadataTools(retriever=None, metadata_config=config())
    try:
        tools.describe_tables({"tables": ["orders"]}, context=context("A"))
        with pytest.raises(Exception) as caught:
            tools.describe_tables({"tables": ["orders"]}, context=context("B"))
        assert getattr(caught.value, "code", None) == "unauthorized"
    finally:
        record = tools.close()
    assert record["call_count"] == 1


@pytest.mark.skipif(not __import__("sys").platform.startswith("linux"), reason="kills the host with os._exit and reads /proc")
def test_the_server_exits_when_its_host_dies(tmp_path):
    import os
    import sys

    host = (
        "import os, sys; sys.path.insert(0, %r)\n"
        "from mcp_helpers import context, started_session\n"
        "session = started_session(context('A'), cwd=%r)\n"
        "print(session.server_pid, flush=True)\n"
        "os._exit(9)\n"
    ) % (os.path.dirname(__file__), str(tmp_path))
    completed = subprocess.run([sys.executable, "-c", host], capture_output=True, text=True, timeout=60)
    pid = int(completed.stdout.strip())
    deadline = time.monotonic() + 10
    while process_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert process_exists(pid) is False  # stdin closed with the host, the server ended itself
