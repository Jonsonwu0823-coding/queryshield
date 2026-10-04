"""Permission-aware versioned knowledge snapshot repository for W04."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from queryshield.db.w04_state import StateStore, StateStoreError
from queryshield.knowledge.ingest import KnowledgeSnapshot


class KnowledgeAccessError(ValueError):
    def __init__(self, code: str, message: str = "knowledge source is not visible") -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class KnowledgeIdentity:
    tenant_id: str
    principal_id: str
    role: str


class KnowledgeSnapshotRepository:
    def __init__(self, state: StateStore) -> None:
        self.state = state

    def publish(self, snapshot: KnowledgeSnapshot) -> None:
        self.state.publish_snapshot(snapshot.as_dict())

    def current(self) -> dict[str, object] | None:
        return self.state.get_snapshot()

    def get_snapshot(self, snapshot_id: str) -> dict[str, object] | None:
        return self.state.get_snapshot(snapshot_id)

    def revoke(self, source_id: str) -> None:
        self.state.set_source_acl(source_id, status="deleted")

    def update_acl(
        self,
        source_id: str,
        *,
        tenant_scope: str | None = None,
        allowed_roles: tuple[str, ...] | None = None,
        status: str | None = None,
    ) -> None:
        self.state.set_source_acl(
            source_id,
            tenant_scope=tenant_scope,
            allowed_roles=allowed_roles,
            status=status,
        )

    def visible_source(
        self,
        *,
        snapshot_id: str,
        source_id: str,
        identity: KnowledgeIdentity,
    ) -> dict[str, object]:
        snapshot = self.state.get_snapshot(snapshot_id)
        if snapshot is None:
            raise KnowledgeAccessError("snapshot_not_found")
        sources = snapshot.get("sources")
        source = next(
            (item for item in sources if isinstance(item, Mapping) and item.get("source_id") == source_id),
            None,
        ) if isinstance(sources, list) else None
        acl = self.state.get_source_acl(source_id)
        if source is None or acl is None:
            raise KnowledgeAccessError("source_not_found")
        if acl["status"] != "active":
            raise KnowledgeAccessError("source_revoked")
        tenant_scope = str(acl["tenant_scope"])
        roles = tuple(str(item) for item in acl["allowed_roles"])
        if tenant_scope not in {"global", identity.tenant_id} or identity.role not in roles:
            raise KnowledgeAccessError("source_not_found")
        return dict(source)


__all__ = ["KnowledgeAccessError", "KnowledgeIdentity", "KnowledgeSnapshotRepository"]
