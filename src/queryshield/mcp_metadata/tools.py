"""The run's tool facade when the product serves metadata tools over MCP.

Only ``search_catalog`` and ``describe_tables`` change: the host runs the
local argument checks first (a rejected call sends nothing), sends the
normalized arguments to the run's own server process, checks the result and
hands the graph a result rebuilt from its own records.  ``query_readonly`` and
everything else is the inherited local facade.  Any MCP failure closes the
session and fails the call; nothing falls back to the local tools.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import shutil
import tempfile
from typing import Any

from queryshield.agent.proposals import ExecutionContext
from queryshield.mcp_metadata.launch import McpMetadataConfig, product_launch
from queryshield.mcp_metadata.schemas import (
    FAILURE_SOURCE_UPSTREAM,
    MCP_ERROR_MESSAGES,
    MCP_RESULT_INVALID,
    MCP_TIMEOUT,
    MCP_UNAVAILABLE,
    TRANSPORT,
)
from queryshield.mcp_metadata.session import McpMetadataSession, McpSessionError
from queryshield.mcp_metadata.verify import (
    ResultInvalid,
    UpstreamFailure,
    checked_error,
    checked_search_items,
    checked_tables,
)
from queryshield.tools.semantic import (
    ControlledTools,
    ToolError,
    _describe_tables_arguments,
    _require_context,
    _search_catalog_arguments,
)


_OUTCOMES = {
    MCP_UNAVAILABLE: "unavailable",
    MCP_TIMEOUT: "timeout",
    "mcp_protocol_error": "protocol_error",
    MCP_RESULT_INVALID: "result_invalid",
}


class McpToolError(ToolError):
    """One of the four MCP failures; fixed text, never repairable, never a fallback."""

    def __init__(self, code: str) -> None:
        super().__init__(code, MCP_ERROR_MESSAGES[code])


@dataclass(eq=False)
class McpMetadataTools(ControlledTools):
    """ControlledTools whose two metadata tools run in this run's MCP server process."""

    metadata_config: McpMetadataConfig | None = None
    metadata_transport: str = field(default=TRANSPORT, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.metadata_config, McpMetadataConfig):
            raise TypeError("an MCP metadata configuration is required")
        self._session: McpMetadataSession | None = None
        self._session_context: ExecutionContext | None = None
        self._session_dir: str | None = None
        self._session_failed = False
        self._last_call: dict[str, object] | None = None
        self._session_record: dict[str, object] | None = None

    # -- the graph reads one record per metadata call -------------------------

    def take_metadata_call_record(self) -> dict[str, object] | None:
        record, self._last_call = self._last_call, None
        return record

    def _begin_call(self) -> dict[str, object]:
        record: dict[str, object] = {
            "transport": TRANSPORT,
            "mcp_session_id": self._session.session_id if self._session is not None else None,
            "mcp_request_sent": False,
            "mcp_outcome": "not_sent",
        }
        self._last_call = record
        return record

    # -- the two metadata tools -----------------------------------------------

    def search_catalog(self, arguments: Mapping[str, object], *, context: ExecutionContext) -> dict[str, object]:
        record = self._begin_call()
        _require_context(context)
        query, top_k = _search_catalog_arguments(arguments)
        result = self._send(record, "search_catalog", {"query": query, "top_k": top_k}, context)
        return self._checked(
            record,
            lambda: checked_search_items(
                result, top_k=top_k, context=context, catalog=self.catalog, retriever=self.retriever
            ),
            result,
        )

    def describe_tables(self, arguments: Mapping[str, object], *, context: ExecutionContext) -> dict[str, object]:
        record = self._begin_call()
        _require_context(context)
        tables = list(_describe_tables_arguments(arguments))
        expected = ControlledTools.describe_tables(self, {"tables": tables}, context=context)
        result = self._send(record, "describe_tables", {"tables": tables}, context)
        return self._checked(record, lambda: checked_tables(result, expected=expected), result)

    def _send(self, record: dict[str, object], name: str, arguments: dict[str, object], context: ExecutionContext) -> Any:
        session = self._session_for(context, record)
        record["mcp_session_id"] = session.session_id
        record["mcp_request_sent"] = True
        try:
            return session.call(name, arguments)
        except McpSessionError as exc:
            self._fail_session(record, exc.code)
            raise McpToolError(exc.code) from None

    def _checked(self, record: dict[str, object], check, result: Any) -> dict[str, object]:
        try:
            if getattr(result, "is_error", False):
                error = checked_error(result)
            else:
                output = check()
                record["mcp_outcome"] = "ok"
                return output
        except ResultInvalid:
            self._fail_session(record, MCP_RESULT_INVALID)
            raise McpToolError(MCP_RESULT_INVALID) from None
        except UpstreamFailure:
            self._fail_session(record, MCP_UNAVAILABLE, source=FAILURE_SOURCE_UPSTREAM)
            raise McpToolError(MCP_UNAVAILABLE) from None
        record["mcp_outcome"] = "tool_error"
        raise error

    # -- session --------------------------------------------------------------

    def _session_for(self, context: ExecutionContext, record: dict[str, object]) -> McpMetadataSession:
        if self._session_failed:
            raise McpToolError(MCP_UNAVAILABLE)
        if self._session is not None:
            if context != self._session_context:
                raise ToolError("unauthorized", "the metadata session is bound to another execution context")
            return self._session
        config = self.metadata_config
        assert config is not None
        self._session_dir = tempfile.mkdtemp(prefix="queryshield-mcp-session-")
        launcher = config.launcher or product_launch
        try:
            spec = launcher(context, self.retriever, config.mode, self._session_dir)
        except Exception:  # noqa: BLE001 - e.g. the index file could not be written
            self._session_failed = True
            record["mcp_outcome"] = "unavailable"
            raise McpToolError(MCP_UNAVAILABLE) from None
        session = McpMetadataSession(spec, call_timeout=config.call_timeout, start_timeout=config.start_timeout)
        self._session = session
        self._session_context = context
        record["mcp_session_id"] = session.session_id
        try:
            session.start()
        except McpSessionError as exc:
            record["mcp_request_sent"] = True
            self._fail_session(record, exc.code)
            raise McpToolError(exc.code) from None
        return session

    def _fail_session(self, record: dict[str, object], code: str, source: str | None = None) -> None:
        record["mcp_outcome"] = _OUTCOMES.get(code, "unavailable")
        self._session_failed = True
        if self._session is not None:
            self._session.mark_failed(code, source=source)
        self.close()

    def close(self) -> dict[str, object] | None:
        """Close this run's session (idempotent); the record, or None if none was opened."""

        if self._session is not None and self._session_record is None:
            self._session_record = self._session.close()
            if self._session_dir is not None:
                shutil.rmtree(self._session_dir, ignore_errors=True)
        elif self._session is None and self._session_dir is not None:
            shutil.rmtree(self._session_dir, ignore_errors=True)
        return dict(self._session_record) if self._session_record is not None else None


__all__ = ["McpMetadataTools", "McpToolError"]
