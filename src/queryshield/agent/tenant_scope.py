"""Recognize explicit tenant targets in user requests without reading evaluation labels."""

from __future__ import annotations

import re


_TENANT_MENTION_PATTERNS = (
    re.compile(r"(?i)(?<![A-Za-z0-9_])tenant(?:\s*-\s*|\s+)([A-Za-z0-9][A-Za-z0-9_.-]*)"),
    re.compile(r"租户\s*([A-Za-z0-9][A-Za-z0-9_.-]*)"),
)


def _tenant_key(value: str) -> str:
    return value.strip().casefold().removeprefix("tenant-")


def explicit_foreign_tenant_mentions(question: object, authenticated_tenant_id: object) -> tuple[str, ...]:
    """Return explicit tenant labels that differ from the authenticated tenant.

    The check is intentionally narrow: it recognizes explicit ``tenant-B``,
    ``tenant B`` and ``租户B`` mentions. It does not infer a tenant from SQL,
    metadata, model output, or an evaluation case's expected values.
    """

    if type(question) is not str or type(authenticated_tenant_id) is not str:
        return ()
    current_key = _tenant_key(authenticated_tenant_id)
    if not current_key:
        return ()

    mentions: list[str] = []
    for pattern in _TENANT_MENTION_PATTERNS:
        mentions.extend(match.group(1) for match in pattern.finditer(question))
    foreign: list[str] = []
    seen: set[str] = set()
    for mention in mentions:
        key = _tenant_key(mention)
        if key and key != current_key and key not in seen:
            foreign.append(mention)
            seen.add(key)
    return tuple(foreign)


def has_explicit_foreign_tenant(question: object, authenticated_tenant_id: object) -> bool:
    return bool(explicit_foreign_tenant_mentions(question, authenticated_tenant_id))


__all__ = ["explicit_foreign_tenant_mentions", "has_explicit_foreign_tenant"]
