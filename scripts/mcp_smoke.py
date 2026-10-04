"""MCP smoke: the two metadata tools over a real stdio session, then through the product.

Protocol part (no model, no database): starts the real server process as tenant
A's requester and checks initialize, tools/list, valid calls, tool errors,
protocol errors, local/MCP equality, tenant and role boundaries, and that the
server is gone after close.  It always uses the Fake embedding.

Product part: starts uvicorn with QUERYSHIELD_METADATA_TOOLS=mcp and asks one
data question and one definition question; every metadata call must have gone
over MCP, queries stay local, and no server process may be left once uvicorn
stops.  Run by the user through scripts/mcp-local-smoke.ps1 (credentials are
prompted for there).  Output is fixed fields only: no question, item, row or
answer text, no URL and no credential.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = Path(__file__).resolve().parent
for root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

DATA_QUESTION = "已支付订单有几笔"
DEFINITION_QUESTION = "退款后净额是怎么算的？"
METADATA_TOOLS = ("search_catalog", "describe_tables")
SUMMARY_NAME = "mcp-smoke-summary.json"


# -- judgments (pure; tests call these with built inputs) ---------------------


def judge_protocol(checks: dict[str, bool]) -> list[str]:
    """Every protocol check must hold; a failed one is a hard failure by name."""

    return [f"protocol:{name}" for name, passed in sorted(checks.items()) if passed is not True]


def judge_sessions(sessions: list[dict], *, gone: dict[int, bool | None]) -> list[str]:
    failures: list[str] = []
    for session in sessions:
        if session.get("cleanup") != "ok" or session.get("server_exited") is not True:
            failures.append("session_not_cleaned")
        pid = session.get("server_pid")
        if type(pid) is int and gone.get(pid) is not True:
            failures.append("leftover_server_process")
    return sorted(set(failures))


def judge_product_step(
    step: str,
    *,
    http_status: int,
    status: object,
    answer_status: object,
    tool_events: list[dict],
    answer_source_ids: list[str],
) -> tuple[list[str], list[str]]:
    """(hard failures, known gaps) for one product question."""

    failures: list[str] = []
    gaps: list[str] = []
    if http_status >= 500:
        failures.append(f"server_error:{step}")
    metadata = [event for event in tool_events if event.get("tool_name") in METADATA_TOOLS]
    if any(event.get("transport") != "mcp_stdio" for event in metadata):
        failures.append(f"metadata_call_not_over_mcp:{step}")
    if any(event.get("transport") for event in tool_events if event.get("tool_name") == "query_readonly"):
        failures.append(f"query_left_the_host:{step}")
    if status != "SUCCEEDED":
        failures.append(f"not_succeeded:{step}")
    if step == "data":
        if status == "SUCCEEDED" and answer_status != "verified":
            failures.append("data_not_verified")
        if not metadata:
            gaps.append("metadata_tools_not_used_by_model")
    if step == "definition":
        returned = {
            source
            for event in metadata
            if event.get("status") == "succeeded" and event.get("transport") == "mcp_stdio"
            for source in event.get("source_ids") or ()
        }
        if not answer_source_ids or not set(answer_source_ids) <= returned:
            failures.append("definition_sources_not_from_mcp")
        # Same name as the B2b HTTP smoke: the server searched after the model answered without sources.
        if any(event.get("tool_name") == "search_catalog" and event.get("initiated_by") == "server" for event in tool_events):
            gaps.append("knowledge_after_send_back")
    return failures, gaps


# -- protocol part ------------------------------------------------------------


def run_protocol() -> tuple[dict[str, bool], dict[str, object]]:
    import anyio
    import mcp_types as types
    from mcp.shared.exceptions import MCPError

    from queryshield.agent.proposals import ExecutionContext
    from queryshield.knowledge.runtime import shared_retrieval_runtime
    from queryshield.mcp_metadata.launch import product_launch
    from queryshield.mcp_metadata.process import process_exists
    from queryshield.mcp_metadata.schemas import PROTOCOL_VERSION, TOOL_NAMES
    from queryshield.mcp_metadata.session import McpMetadataSession
    from queryshield.mcp_metadata.tools import McpMetadataTools
    from queryshield.mcp_metadata.launch import McpMetadataConfig
    from queryshield.tools.semantic import ControlledTools

    retriever = shared_retrieval_runtime("fake").retriever
    a_requester = ExecutionContext(run_id=f"run-mcp-smoke-{uuid4()}", tenant_id="A", principal_id="a-requester", role="requester")
    b_requester = ExecutionContext(run_id=f"run-mcp-smoke-{uuid4()}", tenant_id="B", principal_id="b-requester", role="requester")
    checks: dict[str, bool] = {}
    session = McpMetadataSession(
        product_launch(a_requester, retriever, "fake", tempfile.mkdtemp(prefix="mcp-smoke-")),
        call_timeout=2.0,
        start_timeout=20.0,
    )

    def error_code(result) -> str | None:
        if not result.is_error or len(result.content) != 1:
            return None
        return json.loads(result.content[0].text).get("error_code")

    try:
        session.start()
        record = session.record
        checks["initialize"] = record["initialize"] == "ok" and record["protocol_version"] == PROTOCOL_VERSION
        checks["tools_list_two_tools_shared_schemas"] = record["list"] == "ok" and record["tools_listed"] == list(TOOL_NAMES)
        search = session.call("search_catalog", {"query": DEFINITION_QUESTION, "top_k": 3})
        tables = session.call("describe_tables", {"tables": ["orders", "refunds"]})
        checks["two_valid_calls"] = all(
            result.is_error is False and json.loads(result.content[0].text) == result.structured_content for result in (search, tables)
        )
        invalid = [
            session.call("search_catalog", {"query": "净额", "top_k": 0}),
            session.call("search_catalog", {"top_k": 3}),
            session.call("describe_tables", {"tables": []}),
        ]
        checks["invalid_arguments_are_tool_errors"] = [error_code(result) for result in invalid] == [
            "invalid_argument", "missing_argument", "invalid_argument",
        ]
        identity = [session.call("search_catalog", {"query": "净额", field: "B"}) for field in ("tenant_id", "role")]
        checks["identity_fields_are_tool_errors"] = all(error_code(result) == "unknown_argument" for result in identity)
        checks["forbidden_table_is_a_tool_error"] = error_code(session.call("describe_tables", {"tables": ["pg_shadow"]})) == "table_not_allowed"
        local = ControlledTools(retriever=retriever)
        same = True
        for query in (DEFINITION_QUESTION, DATA_QUESTION, "Tenant A orders"):
            remote_items = session.call("search_catalog", {"query": query, "top_k": 5}).structured_content
            same = same and remote_items == local.search_catalog({"query": query, "top_k": 5}, context=a_requester)
        remote_tables = session.call("describe_tables", {"tables": ["customers"]}).structured_content
        checks["same_result_as_local_tools"] = same and remote_tables == local.describe_tables({"tables": ["customers"]}, context=a_requester)
        pid = session.server_pid
    finally:
        closed = session.close()
    checks["server_gone_after_close"] = closed["cleanup"] == "ok" and pid is not None and process_exists(pid) is False

    async def unknown_tools():
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        spec = product_launch(a_requester, None, "fake", tempfile.mkdtemp(prefix="mcp-smoke-"))
        params = StdioServerParameters(command=spec.command, args=list(spec.args), env=dict(spec.env), cwd=spec.cwd)
        outcomes = []
        with open(Path(spec.cwd) / "stderr.txt", "w+", encoding="utf-8") as errlog:
            async with stdio_client(params, errlog=errlog) as streams:
                async with ClientSession(*streams) as client:
                    await client.initialize()
                    for name in ("query_readonly", "drop_everything"):
                        try:
                            await client.call_tool(name, {})
                            outcomes.append(False)
                        except MCPError as exc:
                            outcomes.append(exc.error.code == types.INVALID_PARAMS)
        return outcomes

    checks["unknown_tool_is_a_protocol_error"] = all(anyio.run(unknown_tools))

    def sources(ctx, queries) -> set[str]:
        tools = McpMetadataTools(retriever=retriever, metadata_config=McpMetadataConfig(mode="fake"))
        try:
            return {item["source_id"] for query in queries for item in tools.search_catalog({"query": query, "top_k": 5}, context=ctx)["items"]}
        finally:
            tools.close()

    checks["b_requester_cannot_see_a_only_source"] = "tenant-a-orders-overview" not in sources(b_requester, ("Tenant A orders", "tenant orders tenant_scope"))
    checks["requester_cannot_see_approver_only_source"] = "semantic-sensitive-customer-name" not in sources(
        a_requester, ("Sensitive customer names approver", "客户姓名")
    )
    facts = {key: closed[key] for key in ("sdk_version", "protocol_version", "tools_listed", "call_count", "initialize", "list", "cleanup", "server_exited")}
    return checks, facts


# -- product part -------------------------------------------------------------
#
# The state store is read while the server still runs (WAL allows a concurrent
# reader, as in the B2b smoke).  On Windows the uvicorn process is a child of
# the venv launcher that Popen started, so after terminate() it may still hold
# the SQLite file for a moment: nothing opens the store after the stop.


class _Server:
    """A local uvicorn process; ``stop`` ends it and waits until its port no longer answers."""

    def __init__(self, process: subprocess.Popen, port: int) -> None:
        self.process = process
        self.port = port
        self.base = f"http://127.0.0.1:{port}"

    def stop(self) -> bool:
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        return _wait_port_closed(self.port)


def _database_ready() -> bool:
    from b2b_http_smoke import _database_reachable

    return _database_reachable()


def _start_server(env: dict[str, str]) -> _Server | None:
    from b2b_http_smoke import _free_port
    from urllib.error import URLError
    from urllib.request import urlopen

    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "queryshield.api.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=PROJECT_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    server = _Server(process, port)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            with urlopen(server.base + "/health", timeout=2) as response:
                if response.status == 200:
                    return server
        except (URLError, OSError):
            time.sleep(0.2)
    server.stop()
    return None


def _ask(base: str, token: str, question: str) -> tuple[int, dict]:
    from b2b_http_smoke import _http

    return _http(base, "/queries", token=token, method="POST", body={"question": question})


def _read_run(state_path: Path, run_id: str) -> tuple[list[dict], list[dict]]:
    """(tool_call event payloads, metadata_session payloads) of one run, while the server runs."""

    from queryshield.db.w04_state import StateStore

    with StateStore(state_path) as store:
        events = store.events(run_id, after_event_id=0, limit=1000)
    tool_events = [e["payload"] for e in events if e["type"] == "agent_step" and e["payload"].get("kind") == "tool_call"]
    sessions = [e["payload"] for e in events if e["type"] == "metadata_session"]
    return tool_events, sessions


def _wait_port_closed(port: int, timeout: float = 20.0) -> bool:
    """True once nothing accepts connections on the port; False if something still does at the deadline."""

    import socket

    deadline = time.monotonic() + timeout
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.2)


def _pids_gone(pids: list[int], timeout: float = 5.0) -> dict[int, bool]:
    from queryshield.mcp_metadata.process import wait_until_gone

    return {pid: wait_until_gone(pid, timeout) is True for pid in pids}


def run_product(mode: str) -> tuple[list[dict], list[str], list[str]]:
    if not _database_ready():
        return [], ["database_unreachable"], []
    workdir = Path(tempfile.mkdtemp(prefix="mcp-smoke-product-"))
    state_path = workdir / "state.sqlite3"
    requester = uuid4().hex
    env = os.environ.copy()
    for name in ("QUERYSHIELD_W04_FAKE_DB", "QUERYSHIELD_AGENT_PROFILE", "QUERYSHIELD_RETRIEVAL", "QUERYSHIELD_DEMO_DATASET"):
        env.pop(name, None)
    env.update({
        "QUERYSHIELD_METADATA_TOOLS": "mcp",
        "QUERYSHIELD_PROVIDER_MODE": mode,
        "QUERYSHIELD_STATE_STORE_PATH": str(state_path),
        "QUERYSHIELD_CALL_STORE_PATH": str(workdir / "calls.sqlite3"),
        "QUERYSHIELD_TOKEN_A_REQUESTER": requester,
        "QUERYSHIELD_TOKEN_A_APPROVER": uuid4().hex,
        "PYTHONPATH": str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
    })
    server = _start_server(env)
    if server is None:
        return [], ["server_not_healthy"], []
    records: list[dict] = []
    failures: list[str] = []
    gaps: list[str] = []
    sessions: list[dict] = []
    try:
        for step, question in (("data", DATA_QUESTION), ("definition", DEFINITION_QUESTION)):
            try:
                code, body = _ask(server.base, requester, question)
            except Exception as error:  # noqa: BLE001 - the kind only
                failures.append(f"http_error:{step}:{type(error).__name__}")
                continue
            record = {
                "step": step, "http_status": code, "status": body.get("status"), "answer_status": body.get("answer_status"),
                "error_code": (body.get("error") or {}).get("code"), "source_ids": list(body.get("source_ids") or []),
            }
            # Read while the server still runs.
            tool_events, step_sessions = _read_run(state_path, str(body.get("run_id")))
            sessions.extend(step_sessions)
            step_failures, step_gaps = judge_product_step(
                step, http_status=int(code), status=record["status"], answer_status=record["answer_status"],
                tool_events=tool_events, answer_source_ids=record["source_ids"],
            )
            failures.extend(step_failures)
            gaps.extend(step_gaps)
            record["tool_calls"] = [
                {key: event.get(key) for key in ("tool_name", "status", "error_code", "transport", "mcp_outcome", "initiated_by")} for event in tool_events
            ]
            record["sessions"] = [
                {key: session.get(key) for key in ("server_pid", "call_count", "initialize", "list", "cleanup", "server_exited", "retrieval", "protocol_version")}
                for session in step_sessions
            ]
            records.append(record)
    finally:
        if not server.stop():
            failures.append("server_still_listening")
    # After the stop: every MCP server process the runs reported must be gone.
    pids = [int(session["server_pid"]) for session in sessions if type(session.get("server_pid")) is int]
    failures.extend(judge_sessions(sessions, gone=_pids_gone(pids)))
    return records, sorted(set(failures)), sorted(set(gaps))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("real", "fake"), default="real")
    parser.add_argument("--part", choices=("all", "protocol", "product"), default="all")
    args = parser.parse_args(argv)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    from b2b_http_smoke import REQUIRED_NAMES

    summary: dict[str, object] = {"mode": args.mode, "part": args.part}
    hard_failures: list[str] = []
    known_gaps: list[str] = []
    if args.part in {"all", "protocol"}:
        try:
            checks, facts = run_protocol()
            summary["protocol"] = {"checks": checks, **facts}
            hard_failures.extend(judge_protocol(checks))
        except Exception as error:  # noqa: BLE001 - the type name only, never the text
            summary["protocol"] = {"error": type(error).__name__}
            hard_failures.append(f"smoke_error:protocol:{type(error).__name__}")
    if args.part in {"all", "product"}:
        required = REQUIRED_NAMES if args.mode == "real" else ("QUERYSHIELD_DATABASE_URL",)
        missing = [name for name in required if not os.getenv(name, "").strip()]
        if missing:
            summary["product"] = {"status": "blocked", "missing_configuration_names": missing}
            hard_failures.append("product_blocked")
        else:
            try:
                records, failures, gaps = run_product(args.mode)
                summary["product"] = {"records": records}
                hard_failures.extend(failures)
                known_gaps.extend(gaps)
            except Exception as error:  # noqa: BLE001 - the type name only, never the text
                summary["product"] = {"error": type(error).__name__}
                hard_failures.append(f"smoke_error:product:{type(error).__name__}")
    summary.update({
        "status": "pass" if not hard_failures else "fail",
        "hard_failures": sorted(set(hard_failures)),
        "known_gaps": sorted(set(known_gaps)),
        "note": "fixed fields only; no question, item, row or answer text, URL or credential",
    })
    (args.evidence_dir / SUMMARY_NAME).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "hard_failures", "known_gaps")}, ensure_ascii=False))
    return 0 if not hard_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
