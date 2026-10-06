"""MCP stdio server for the two metadata tools (``python -m queryshield.mcp_metadata.server``).

The host starts one process per run.  Every argument comes from the host: the
run's identity, the retrieval setting, the knowledge base and, for hybrid
retrieval, the host's index file and its expected embedded snapshot id.  The
process never connects to a database and never embeds the knowledge base.

stdout carries MCP messages only.  stderr carries fixed lines only: the ready
line with this process's pid, or one refusal line before exiting non-zero.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any


READY_PREFIX = "queryshield-mcp-metadata ready pid="
REFUSED_PREFIX = "queryshield-mcp-metadata refused reason="


class _Refused(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # noqa: ARG002 - never echo arguments
        raise _Refused("invalid_arguments")


class _NoDatabaseExecutor:
    """The server process has no database boundary."""

    catalog_version = None

    def execute(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("the metadata server has no database access")


def _parse(argv: list[str] | None) -> argparse.Namespace:
    from queryshield.catalog.catalog import ALLOWED_ROLES

    parser = _Parser(add_help=False)
    for name in ("--run-id", "--tenant-id", "--principal-id", "--role", "--package-dir"):
        parser.add_argument(name, required=True)
    parser.add_argument("--retrieval", required=True, choices=("hybrid", "keyword"))
    parser.add_argument("--knowledge", required=True, choices=("default", "demo"))
    parser.add_argument("--mode", required=True, choices=("fake", "real"))
    parser.add_argument("--index-path")
    parser.add_argument("--expected-snapshot-id")
    args = parser.parse_args(argv)
    if any(not str(value).strip() for value in (args.run_id, args.tenant_id, args.principal_id)):
        raise _Refused("empty_identity")
    if args.role not in ALLOWED_ROLES:
        raise _Refused("invalid_role")
    if args.retrieval == "hybrid" and not (args.index_path and args.expected_snapshot_id):
        raise _Refused("missing_index")
    return args


def _build_tools(args: argparse.Namespace):
    import queryshield
    from queryshield.catalog import load_default_catalog
    from queryshield.tools.semantic import ControlledTools

    if Path(queryshield.__file__).resolve().parent != Path(args.package_dir).resolve():
        raise _Refused("package_mismatch")
    retriever = None
    if args.retrieval == "hybrid":
        from queryshield.knowledge.runtime import retriever_from_index_file

        try:
            retriever = retriever_from_index_file(
                args.mode,
                args.index_path,
                demo=args.knowledge == "demo",
                expected_snapshot_id=args.expected_snapshot_id,
            )
        except Exception as exc:  # noqa: BLE001 - one fixed refusal, no detail
            raise _Refused("index_mismatch") from exc
    return ControlledTools(catalog=load_default_catalog(), executor=_NoDatabaseExecutor(), retriever=retriever)


def _context(args: argparse.Namespace):
    from queryshield.agent.proposals import ExecutionContext

    return ExecutionContext(
        run_id=args.run_id,
        tenant_id=args.tenant_id,
        principal_id=args.principal_id,
        role=args.role,
    )


def _text(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def invoke_tool(tools: Any, context: Any, name: str, arguments: Mapping[str, object]) -> dict[str, Any]:
    """One metadata call as a result dict: ``{"output": ...}`` or ``{"error": {...}}``.

    Tool errors keep their code.  A failure of the upstream embedding service
    is one fixed code and text; any other exception is ``internal_error``.
    Never the exception text, and never what the upstream service returned.
    """

    from queryshield.mcp_metadata.schemas import UPSTREAM_ERROR_CODE, UPSTREAM_ERROR_MESSAGE
    from queryshield.providers.embedding import EmbeddingProviderError
    from queryshield.tools.semantic import ToolError

    method = tools.search_catalog if name == "search_catalog" else tools.describe_tables
    try:
        output = method(arguments, context=context)
    except ToolError as exc:
        return {"error": {"error_code": exc.code, "message": exc.message}}
    except EmbeddingProviderError:
        return {"error": {"error_code": UPSTREAM_ERROR_CODE, "message": UPSTREAM_ERROR_MESSAGE}}
    except Exception:  # noqa: BLE001 - fixed text, never the exception
        return {"error": {"error_code": "internal_error", "message": "the metadata tool failed"}}
    return {"output": output}


def build_server(tools: Any, context: Any):
    """The lowlevel MCP server over one bound ``ControlledTools`` and identity."""

    import anyio
    import mcp_types as types
    from mcp.server.lowlevel import Server
    from mcp.shared.exceptions import MCPError

    from queryshield.mcp_metadata.schemas import SERVER_NAME, TOOL_NAMES, tool_definitions

    listed = [types.Tool.model_validate(definition) for definition in tool_definitions()]

    async def on_list_tools(ctx, params) -> types.ListToolsResult:  # noqa: ARG001
        return types.ListToolsResult(tools=listed)

    async def on_call_tool(ctx, params) -> types.CallToolResult:  # noqa: ARG001
        if params.name not in TOOL_NAMES:
            raise MCPError(code=types.INVALID_PARAMS, message="Unknown tool", data=None)
        arguments = params.arguments if params.arguments is not None else {}
        result = await anyio.to_thread.run_sync(
            invoke_tool, tools, context, params.name, arguments, abandon_on_cancel=True
        )
        if "error" in result:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=_text(result["error"]))],
                is_error=True,
            )
        output = result["output"]
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=_text(output))],
            structured_content=output,
            is_error=False,
        )

    return Server(
        SERVER_NAME,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


def main(argv: list[str] | None = None) -> int:
    logging.disable(logging.CRITICAL)
    try:
        args = _parse(argv)
        tools = _build_tools(args)
        context = _context(args)
    except _Refused as refused:
        sys.stderr.write(f"{REFUSED_PREFIX}{refused.reason}\n")
        sys.stderr.flush()
        return 2
    except Exception:  # noqa: BLE001 - one fixed refusal line
        sys.stderr.write(f"{REFUSED_PREFIX}startup_failed\n")
        sys.stderr.flush()
        return 2

    import anyio
    from mcp.server.stdio import stdio_server

    server = build_server(tools, context)

    async def serve() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    sys.stderr.write(f"{READY_PREFIX}{os.getpid()}\n")
    sys.stderr.flush()
    try:
        anyio.run(serve)
    except Exception:  # noqa: BLE001 - stdin closed or the session broke: just exit
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
