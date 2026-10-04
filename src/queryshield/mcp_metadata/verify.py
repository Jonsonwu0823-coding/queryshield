"""Host-side checks on what an MCP metadata server returned (its output is untrusted).

A result reaches the run only after every check passes, and what reaches the
run is rebuilt from the host's own trusted records (catalog entries, index
chunks), never the server's objects, so extra fields or ``_meta`` cannot
travel.  Search items are judged with the same visibility rule the host's
retriever filters with: catalog entries by ``requires_approval``; knowledge
chunks by active status, allowed role and tenant scope of their source, with
the version of the host's snapshot and identical text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from queryshield.agent.proposals import ExecutionContext
from queryshield.mcp_metadata.schemas import (
    KNOWN_TOOL_ERROR_CODES,
    KNOWN_TOOL_ERROR_MESSAGES,
    TOOLS,
    UPSTREAM_ERROR_CODE,
)
from queryshield.tools.semantic import ToolError


class ResultInvalid(Exception):
    """The server's result failed a host check (``mcp_result_invalid``)."""


class UpstreamFailure(Exception):
    """The server reported that the upstream service behind it failed (``mcp_unavailable``)."""


def _single_text_json(result: Any) -> object:
    content = getattr(result, "content", None)
    if not isinstance(content, list) or len(content) != 1:
        raise ResultInvalid("exactly one content item is required")
    item = content[0]
    if getattr(item, "type", None) != "text" or type(getattr(item, "text", None)) is not str:
        raise ResultInvalid("the content item must be text")
    if getattr(item, "meta", None) is not None or getattr(item, "annotations", None) is not None:
        raise ResultInvalid("the content item carries extra fields")
    try:
        return json.loads(item.text)
    except ValueError as exc:
        raise ResultInvalid("the text content is not JSON") from exc


def _validate_schema(value: object, schema: Mapping[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    if not Draft202012Validator(dict(schema)).is_valid(value):
        raise ResultInvalid("structured content does not match the output schema")


def checked_error(result: Any) -> ToolError:
    """The local ToolError for a server error result with a known code.

    The server's upstream failure raises ``UpstreamFailure``; any other code
    (argument and identity codes the host already checked before sending,
    ``internal_error``, unknown codes) is an invalid result.
    """

    payload = _single_text_json(result)
    if not isinstance(payload, Mapping) or set(payload) != {"error_code", "message"}:
        raise ResultInvalid("the error result has an unexpected shape")
    code = payload.get("error_code")
    if code == UPSTREAM_ERROR_CODE:
        raise UpstreamFailure()
    if type(code) is not str or code not in KNOWN_TOOL_ERROR_CODES:
        raise ResultInvalid("the error code is not a known tool error")
    return ToolError(code, KNOWN_TOOL_ERROR_MESSAGES[code])


def _structured(name: str, result: Any) -> Mapping[str, Any]:
    if getattr(result, "meta", None) is not None:
        raise ResultInvalid("the result carries _meta")
    structured = getattr(result, "structured_content", None)
    if not isinstance(structured, Mapping):
        raise ResultInvalid("structured content is missing")
    _validate_schema(structured, TOOLS[name]["output_schema"])
    if _single_text_json(result) != structured:
        raise ResultInvalid("the text content differs from the structured content")
    return structured


def checked_search_items(
    result: Any,
    *,
    top_k: int,
    context: ExecutionContext,
    catalog: Any,
    retriever: Any | None,
) -> dict[str, object]:
    """search_catalog: every item is a visible catalog entry or a visible chunk of the host index."""

    from queryshield.knowledge.retrieval import _source_visible

    items = _structured("search_catalog", result)["items"]
    if not isinstance(items, Sequence) or len(items) > top_k:
        raise ResultInvalid("more items than top_k")
    ids = [item.get("id") for item in items]
    if len(set(ids)) != len(ids):
        raise ResultInvalid("duplicate item ids")
    entries = {entry.id: entry for entry in catalog.entries}
    chunks: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    if retriever is not None:
        chunks = {chunk.chunk_id: chunk for chunk in retriever.index.chunks}
        sources = {source.source_id: source for source in retriever.snapshot.source_records}
    trusted: list[dict[str, str]] = []
    for item in items:
        item_id = item["id"]
        entry = entries.get(item_id)
        if entry is not None:
            expected = entry.as_search_item()
            visible = not entry.requires_approval or context.role == "approver"
        elif item_id in chunks:
            chunk = chunks[item_id]
            source = sources.get(chunk.source_id)
            expected = {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "source_id": chunk.source_id,
                "version": chunk.source_version,
            }
            visible = (
                source is not None
                and source.version == chunk.source_version
                and _source_visible(source, context)
            )
        else:
            raise ResultInvalid("the item is not in the host catalog or index")
        if dict(item) != expected:
            raise ResultInvalid("the item differs from the host record")
        if not visible:
            raise ResultInvalid("the item is not visible to this identity")
        trusted.append(dict(expected))
    return {"items": trusted}


def checked_tables(result: Any, *, expected: Mapping[str, object]) -> dict[str, object]:
    """describe_tables: exactly what the host computes locally for the same request."""

    structured = _structured("describe_tables", result)
    if json.dumps(structured, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ResultInvalid("the table description differs from the host catalog")
    return json.loads(json.dumps(expected))


__all__ = ["ResultInvalid", "UpstreamFailure", "checked_error", "checked_search_items", "checked_tables"]
