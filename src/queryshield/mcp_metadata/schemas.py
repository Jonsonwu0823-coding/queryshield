"""The one definition of the two MCP tools: names, input and output schemas.

The server publishes exactly these definitions and the host compares the
server's tools/list against them.  The input schemas have the shapes of the
local tools: no identity field, no extra field.
"""

from __future__ import annotations

from typing import Any

from queryshield.policy.argument_limits import DESCRIBE_TABLES_RANGE, SEARCH_QUERY_MAX_CHARS, TOP_K_DEFAULT, TOP_K_RANGE


SERVER_NAME = "queryshield-metadata"
# initialize negotiates the handshake protocol version; the SDK's latest is this one.
PROTOCOL_VERSION = "2025-11-25"
TRANSPORT = "mcp_stdio"
TOOL_NAMES = ("search_catalog", "describe_tables")

_STRING = {"type": "string"}

SEARCH_CATALOG_INPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": SEARCH_QUERY_MAX_CHARS},
        "top_k": {"type": "integer", "minimum": TOP_K_RANGE[0], "maximum": TOP_K_RANGE[1], "default": TOP_K_DEFAULT},
    },
    "required": ["query"],
    "additionalProperties": False,
}

DESCRIBE_TABLES_INPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tables": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": DESCRIBE_TABLES_RANGE[0],
            "maxItems": DESCRIBE_TABLES_RANGE[1],
            "uniqueItems": True,
        },
    },
    "required": ["tables"],
    "additionalProperties": False,
}

SEARCH_CATALOG_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "maxItems": TOP_K_RANGE[1],
            "items": {
                "type": "object",
                "properties": {"id": _STRING, "text": _STRING, "source_id": _STRING, "version": _STRING},
                "required": ["id", "text", "source_id", "version"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}

DESCRIBE_TABLES_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tables": {
            "type": "array",
            "minItems": DESCRIBE_TABLES_RANGE[0],
            "maxItems": DESCRIBE_TABLES_RANGE[1],
            "items": {
                "type": "object",
                "properties": {
                    "name": _STRING,
                    "columns": {"type": "array", "items": _STRING},
                    "source_id": _STRING,
                    "version": _STRING,
                },
                "required": ["name", "columns", "source_id", "version"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["tables"],
    "additionalProperties": False,
}

TOOLS: dict[str, dict[str, Any]] = {
    "search_catalog": {
        "description": "Search the governed semantic catalog and knowledge base visible to this session's identity.",
        "input_schema": SEARCH_CATALOG_INPUT,
        "output_schema": SEARCH_CATALOG_OUTPUT,
    },
    "describe_tables": {
        "description": "Describe allowlisted tables: column names and catalog source; no database access.",
        "input_schema": DESCRIBE_TABLES_INPUT,
        "output_schema": DESCRIBE_TABLES_OUTPUT,
    },
}

# Error codes the host accepts from the server for a call.  A server error result
# with one of these codes fails the call with the same code.  The host has
# already run the local argument and identity checks before sending, so an
# honest server cannot return an argument or identity error: those codes (and
# anything unknown, ``internal_error`` included) make the result invalid, never
# a refusal.  ``retrieval_unavailable`` is the server's own retriever state,
# which the host cannot know beforehand.
KNOWN_TOOL_ERROR_CODES = frozenset({"retrieval_unavailable"})
# Fixed text per code for an error that came back from the server.
KNOWN_TOOL_ERROR_MESSAGES = {
    "retrieval_unavailable": "the metadata server could not search",
}
# The upstream service behind the server (the embedding service) failed.  The
# server sends this one fixed code and text; the host fails the call with
# ``mcp_unavailable`` and marks the session record ``failure_source: upstream``.
UPSTREAM_ERROR_CODE = "upstream_unavailable"
UPSTREAM_ERROR_MESSAGE = "the upstream service behind the metadata server is unavailable"
FAILURE_SOURCE_UPSTREAM = "upstream"

MCP_UNAVAILABLE = "mcp_unavailable"
MCP_TIMEOUT = "mcp_timeout"
MCP_PROTOCOL_ERROR = "mcp_protocol_error"
MCP_RESULT_INVALID = "mcp_result_invalid"
MCP_ERROR_CODES = frozenset({MCP_UNAVAILABLE, MCP_TIMEOUT, MCP_PROTOCOL_ERROR, MCP_RESULT_INVALID})
MCP_ERROR_MESSAGES = {
    MCP_UNAVAILABLE: "the metadata tool server is unavailable",
    MCP_TIMEOUT: "the metadata tool call timed out",
    MCP_PROTOCOL_ERROR: "the metadata tool server broke the protocol",
    MCP_RESULT_INVALID: "the metadata tool result failed the host checks",
}


def tool_definitions() -> list[dict[str, Any]]:
    """The tools/list entries as plain data (name, description, schemas, read-only hint)."""

    return [
        {
            "name": name,
            "description": TOOLS[name]["description"],
            "inputSchema": TOOLS[name]["input_schema"],
            "outputSchema": TOOLS[name]["output_schema"],
            "annotations": {"readOnlyHint": True},
        }
        for name in TOOL_NAMES
    ]
