"""One MCP stdio session to one metadata server process, driven from sync code.

Each session owns an asyncio event loop in a daemon thread.  One long-lived
task enters ``stdio_client`` and ``ClientSession``, runs initialize and
tools/list, then waits for close; calls run as separate tasks on the same loop.
Closing leaves both contexts, so the SDK closes the server's stdin, waits, and
ends the whole process tree; the host then confirms that the pid the server
reported is gone.  Every failure maps to one of four fixed codes and no text
from the server process.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
import sys
from threading import Lock, Thread
import time
from typing import Any
from uuid import uuid4
import weakref

from queryshield.mcp_metadata.launch import LaunchSpec
from queryshield.mcp_metadata.process import wait_until_gone
from queryshield.mcp_metadata.schemas import (
    MCP_PROTOCOL_ERROR,
    MCP_RESULT_INVALID,
    MCP_TIMEOUT,
    MCP_UNAVAILABLE,
    PROTOCOL_VERSION,
    TOOL_NAMES,
    TOOLS,
    TRANSPORT,
)
from queryshield.mcp_metadata.server import READY_PREFIX, REFUSED_PREFIX


_READY_RE = re.compile(rf"^{re.escape(READY_PREFIX)}([1-9][0-9]{{0,9}})$")
_REFUSED_RE = re.compile(rf"^{re.escape(REFUSED_PREFIX)}([a-z_]{{1,40}})$")
# The SDK's own shutdown is bounded (flush 0.5 s, stdin grace 2 s, kill 2 s, reap 2 s).
_CLOSE_TIMEOUT_SECONDS = 10.0
_EXIT_CONFIRM_SECONDS = 1.0
_OPEN_SESSIONS: "weakref.WeakSet[McpMetadataSession]" = weakref.WeakSet()
_OPEN_LOCK = Lock()


class McpSessionError(Exception):
    """A session failure with one of the four fixed MCP codes."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _sdk_version() -> str | None:
    try:
        return version("mcp")
    except PackageNotFoundError:  # pragma: no cover - the SDK is a locked dependency
        return None


def _close_open_sessions() -> None:
    with _OPEN_LOCK:
        sessions = list(_OPEN_SESSIONS)
    for session in sessions:
        session.close()


atexit.register(_close_open_sessions)


class McpMetadataSession:
    """Start, call and close one server process; ``close`` returns the session record."""

    def __init__(
        self,
        spec: LaunchSpec,
        *,
        call_timeout: float,
        start_timeout: float,
        session_id: str | None = None,
    ) -> None:
        self.spec = spec
        self.call_timeout = float(call_timeout)
        self.start_timeout = float(start_timeout)
        self.session_id = session_id or f"mcp-session-{uuid4()}"
        self._lock = Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Thread | None = None
        self._serve_future: concurrent.futures.Future | None = None
        self._ready: concurrent.futures.Future = concurrent.futures.Future()
        self._stop: asyncio.Event | None = None
        self._scope: Any = None
        self._client: Any = None
        self._errlog_path = Path(spec.cwd) / "server-stderr.txt"
        self._errlog: Any = None
        self._started_at: float | None = None
        self._closed = False
        self._broken = False
        self._record: dict[str, object] = {
            "session_id": self.session_id,
            "transport": TRANSPORT,
            "server_pid": None,
            "sdk_version": _sdk_version(),
            "protocol_version": None,
            "tools_listed": [],
            "call_count": 0,
            "initialize": "not_run",
            "list": "not_run",
            "cleanup": "not_run",
            "cleanup_error": None,
            "server_exited": None,
            "refused_reason": None,
            "failure_code": None,
            "failure_source": None,
            "started_ms": None,
            "duration_ms": None,
            "retrieval": spec.retrieval,
            "knowledge_snapshot_id": spec.knowledge_snapshot_id,
        }

    # -- lifecycle ---------------------------------------------------------

    @property
    def server_pid(self) -> int | None:
        pid = self._record["server_pid"]
        return pid if type(pid) is int else None

    @property
    def record(self) -> dict[str, object]:
        return dict(self._record)

    def start(self) -> None:
        """Start the process, initialize and check tools/list; raise McpSessionError on failure."""

        from mcp.client.stdio import StdioServerParameters

        self._started_at = time.monotonic()
        with _OPEN_LOCK:
            _OPEN_SESSIONS.add(self)
        self._errlog = open(self._errlog_path, "w+", encoding="utf-8")  # noqa: SIM115 - closed in close()
        params = StdioServerParameters(
            command=self.spec.command,
            args=list(self.spec.args),
            env=dict(self.spec.env),
            cwd=self.spec.cwd,
        )
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._loop.run_forever, name=f"queryshield-{self.session_id}", daemon=True)
        self._thread.start()
        self._serve_future = asyncio.run_coroutine_threadsafe(self._serve(params), self._loop)
        try:
            listed_names, protocol_version = self._ready.result(timeout=self.start_timeout)
        except concurrent.futures.TimeoutError:
            self._record["initialize"] = "timeout" if self._record["initialize"] == "not_run" else self._record["initialize"]
            self._fail(MCP_UNAVAILABLE)
        except McpSessionError as exc:
            self._fail(exc.code)
        except BaseException:  # noqa: BLE001 - any start failure: the server is unavailable
            self._fail(MCP_UNAVAILABLE)
        else:
            self._record["started_ms"] = int((time.monotonic() - self._started_at) * 1000)
            self._read_pid()
            if self.server_pid is None:
                self._fail(MCP_UNAVAILABLE)
            del listed_names, protocol_version

    def mark_failed(self, code: str, source: str | None = None) -> None:
        """The host rejected what this session returned; it serves nothing more.

        ``source`` names where an upstream-caused failure came from (a fixed value).
        """

        self._broken = True
        if self._record["failure_code"] is None:
            self._record["failure_code"] = code
            self._record["failure_source"] = source

    def _fail(self, code: str):
        """A start that failed: record it, close (no process or loop is left), raise."""

        self._broken = True
        if self._record["failure_code"] is None:
            self._record["failure_code"] = code
        self._read_pid()
        self.close()
        raise McpSessionError(code)

    async def _serve(self, params) -> None:
        import anyio
        from mcp.client.session import ClientSession
        from mcp.client.stdio import stdio_client

        self._stop = asyncio.Event()
        try:
            # Stopped through an anyio scope, never a native task.cancel(): the
            # SDK's shutdown is shielded against anyio cancellation only.
            with anyio.CancelScope() as self._scope:
                async with stdio_client(params, errlog=self._errlog) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as client:
                        await self._handshake(client)
                        await self._stop.wait()
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(_session_error(exc) or McpSessionError(MCP_UNAVAILABLE))
            if not isinstance(exc, (asyncio.CancelledError, Exception)):
                raise
        finally:
            self._client = None
            if not self._ready.done():
                self._ready.set_exception(McpSessionError(MCP_UNAVAILABLE))

    async def _handshake(self, client) -> None:
        try:
            initialized = await client.initialize()
        except Exception as exc:
            self._record["initialize"] = "failed"
            raise McpSessionError(MCP_UNAVAILABLE) from exc
        self._record["initialize"] = "ok"
        self._record["protocol_version"] = initialized.protocol_version
        if initialized.protocol_version != PROTOCOL_VERSION:
            raise McpSessionError(MCP_PROTOCOL_ERROR)
        try:
            listed = await client.list_tools()
        except Exception as exc:
            self._record["list"] = "failed"
            raise McpSessionError(MCP_PROTOCOL_ERROR) from exc
        names = [tool.name for tool in listed.tools]
        self._record["tools_listed"] = names[:10]
        if not _listing_matches(listed.tools):
            self._record["list"] = "mismatch"
            raise McpSessionError(MCP_PROTOCOL_ERROR)
        self._record["list"] = "ok"
        self._client = client
        self._ready.set_result((names, initialized.protocol_version))

    def _request_stop(self) -> None:
        """On the loop: a ready session leaves its contexts; a starting one is cancelled (anyio)."""

        if self._stop is not None and self._client is not None:
            self._stop.set()
        elif self._scope is not None:
            self._scope.cancel()
        elif self._stop is not None:
            self._stop.set()

    def _read_pid(self) -> None:
        if self._record["server_pid"] is not None or self._errlog is None:
            return
        try:
            lines = self._errlog_path.read_text(encoding="utf-8", errors="replace").splitlines()[:3]
        except OSError:
            return
        for line in lines:
            ready = _READY_RE.match(line.strip())
            if ready is not None:
                self._record["server_pid"] = int(ready.group(1))
                return
            refused = _REFUSED_RE.match(line.strip())
            if refused is not None:
                self._record["refused_reason"] = refused.group(1)
                return

    # -- calls -------------------------------------------------------------

    def call(self, name: str, arguments: dict[str, object]) -> Any:
        """One tools/call; returns the SDK's CallToolResult or raises McpSessionError."""

        if name not in TOOL_NAMES:
            raise McpSessionError(MCP_PROTOCOL_ERROR)
        with self._lock:
            if self._closed or self._broken or self._client is None or self._loop is None:
                raise McpSessionError(MCP_UNAVAILABLE)
            self._record["call_count"] = int(self._record["call_count"]) + 1  # type: ignore[arg-type]
            future = asyncio.run_coroutine_threadsafe(self._call(name, arguments), self._loop)
        try:
            return future.result(timeout=self.call_timeout + 1.0)
        except concurrent.futures.TimeoutError:
            future.cancel()
            self._broken = True
            self._record["failure_code"] = self._record["failure_code"] or MCP_TIMEOUT
            raise McpSessionError(MCP_TIMEOUT) from None
        except McpSessionError as exc:
            self._broken = True
            self._record["failure_code"] = self._record["failure_code"] or exc.code
            raise
        except BaseException:  # noqa: BLE001 - a cancelled or broken call
            self._broken = True
            self._record["failure_code"] = self._record["failure_code"] or MCP_UNAVAILABLE
            raise McpSessionError(MCP_UNAVAILABLE) from None

    async def _call(self, name: str, arguments: dict[str, object]) -> Any:
        import anyio
        from mcp.shared.exceptions import MCPError
        import mcp_types as types

        client = self._client
        if client is None:
            raise McpSessionError(MCP_UNAVAILABLE)
        try:
            with anyio.fail_after(self.call_timeout):
                return await client.call_tool(name, arguments)
        except TimeoutError as exc:
            raise McpSessionError(MCP_TIMEOUT) from exc
        except MCPError as exc:
            code = MCP_UNAVAILABLE if exc.error.code == types.CONNECTION_CLOSED else MCP_PROTOCOL_ERROR
            raise McpSessionError(code) from exc
        except RuntimeError as exc:
            # The SDK's own output-schema check, or a result shape we never ask for.
            raise McpSessionError(MCP_RESULT_INVALID) from exc
        except (ValueError, TypeError) as exc:
            raise McpSessionError(MCP_PROTOCOL_ERROR) from exc
        except Exception as exc:  # noqa: BLE001 - closed streams, broken pipes, a dead process
            raise McpSessionError(MCP_UNAVAILABLE) from exc

    # -- close -------------------------------------------------------------

    def close(self) -> dict[str, object]:
        """Close once: leave the SDK contexts, confirm the server pid is gone, return the record."""

        with self._lock:
            if self._closed:
                return dict(self._record)
            self._closed = True
        cleanup_error: str | None = None
        loop, future = self._loop, self._serve_future
        if loop is not None and future is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._request_stop)
            try:
                future.result(timeout=_CLOSE_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                cleanup_error = "close_timeout"
            except BaseException:  # noqa: BLE001 - the session already ended with an error
                pass
            loop.call_soon_threadsafe(loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            if self._thread is None or not self._thread.is_alive():
                loop.close()
        self._read_pid()
        pid = self.server_pid
        if pid is None:
            exited = None
            if self._record["initialize"] == "ok":
                cleanup_error = cleanup_error or "pid_unknown"
        else:
            exited = wait_until_gone(pid, _EXIT_CONFIRM_SECONDS)
            if exited is False:
                cleanup_error = cleanup_error or "process_still_alive"
        self._record["server_exited"] = exited
        if cleanup_error is not None:
            self._record["cleanup"] = "failed"
        elif pid is not None and exited is None:
            self._record["cleanup"] = "unverified"
        else:
            self._record["cleanup"] = "ok"
        self._record["cleanup_error"] = cleanup_error
        if self._errlog is not None:
            try:
                self._errlog.close()
            except OSError:
                pass
        if self._started_at is not None:
            self._record["duration_ms"] = int((time.monotonic() - self._started_at) * 1000)
        with _OPEN_LOCK:
            _OPEN_SESSIONS.discard(self)
        if cleanup_error is not None:
            sys.stderr.write(f"queryshield-mcp-metadata cleanup_failed reason={cleanup_error}\n")
        return dict(self._record)


def _session_error(exc: BaseException) -> McpSessionError | None:
    """The McpSessionError inside an (anyio) exception group, if any."""

    if isinstance(exc, McpSessionError):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            found = _session_error(inner)
            if found is not None:
                return found
    return None


def _listing_matches(tools: Any) -> bool:
    """Exactly the two tools, each with the shared input and output schema."""

    if len(tools) != len(TOOL_NAMES):
        return False
    by_name = {tool.name: tool for tool in tools}
    if set(by_name) != set(TOOL_NAMES):
        return False
    return all(
        by_name[name].input_schema == TOOLS[name]["input_schema"]
        and by_name[name].output_schema == TOOLS[name]["output_schema"]
        for name in TOOL_NAMES
    )


__all__ = ["McpMetadataSession", "McpSessionError"]
