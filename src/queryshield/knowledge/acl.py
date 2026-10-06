"""Who may see a knowledge source, a knowledge chunk or a catalog entry.

The one definition of the rule.  The retriever filters with it, the MCP host judges what a
server returned with it, the approval re-checks a permission source with the tenant rule, and
the retrieval side record reports it.  This module imports no retrieval code.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from queryshield.agent.proposals import ExecutionContext
    from queryshield.knowledge.ingest import SourceRecord


def tenant_matches(scope: str, tenant_id: str) -> bool:
    """Match the fixture's trusted A/B scope to the server identity only."""

    if scope == "global":
        return True
    return tenant_id == scope or tenant_id == f"tenant-{scope}"


def source_visible(source: SourceRecord, context: ExecutionContext) -> bool:
    return (
        source.status == "active"
        and context.role in source.allowed_roles
        and tenant_matches(source.tenant_scope, context.tenant_id)
    )


def chunk_visible(chunk: Any, sources: Mapping[str, SourceRecord], context: ExecutionContext) -> bool:
    """The chunk's source is in the snapshot at the chunk's version and is visible."""

    source = sources.get(chunk.source_id)
    return source is not None and source.version == chunk.source_version and source_visible(source, context)


def catalog_entry_visible(entry: Any, role: str) -> bool:
    """A catalog entry that needs approval is visible to approvers only."""

    return not entry.requires_approval or role == "approver"
