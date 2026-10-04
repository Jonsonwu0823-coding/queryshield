"""Centralized W04 object-scope decisions."""

from collections.abc import Mapping

from queryshield.approval.service import ObjectNotFound, W04Identity


def require_same_subject(run: Mapping[str, object], identity: Mapping[str, str]) -> W04Identity:
    subject = W04Identity.from_mapping(identity)
    if run.get("tenant_id") != subject.tenant_id or run.get("principal_id") != subject.principal_id:
        raise ObjectNotFound()
    return subject


__all__ = ["require_same_subject"]
