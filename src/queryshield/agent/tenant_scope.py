"""Recognize explicit tenant targets in user requests without reading evaluation labels."""

from __future__ import annotations

import re


_CHINESE_NOUNS = "租户|商户|商家|店铺|门店|公司"
_ENGLISH_NOUNS = "tenant|merchant"
# A label is one letter, optionally followed by digits ("B", "B2"; tenant-A loses its prefix),
# never glued to a letter, digit or underscore: "租户2026年" names a month, and "show", "ID",
# "the" or "tenant-acme" are words, not labels.
_LABEL = r"(?<![A-Za-z0-9_])[A-Za-z][0-9]*(?![A-Za-z0-9_])"
_TENANT_MENTION_PATTERNS = (
    # noun first: "租户B", "商户 B", "tenant-B", "merchant B"
    re.compile(rf"(?i)(?<![A-Za-z0-9_])(?:{_ENGLISH_NOUNS})(?:\s*-\s*|\s+)({_LABEL})"),
    re.compile(rf"(?:{_CHINESE_NOUNS})\s*({_LABEL})"),
    # label first: "B租户", "B 商户", "B tenant"
    re.compile(rf"({_LABEL})\s*(?:{_CHINESE_NOUNS})"),
    re.compile(rf"(?i)({_LABEL})\s+(?:{_ENGLISH_NOUNS})(?![A-Za-z0-9_])"),
)


def _tenant_key(value: str) -> str:
    return value.strip().casefold().removeprefix("tenant-")


def explicit_foreign_tenant_mentions(question: object, authenticated_tenant_id: object) -> tuple[str, ...]:
    """Return explicit tenant labels that differ from the authenticated tenant.

    The check is intentionally narrow: a tenant noun (租户, 商户, 商家, 店铺, 门店,
    公司, tenant, merchant) next to a label of one letter and optional digits, in
    either order ("租户B", "B 商户", "tenant-B", "B2 tenant"). It does not infer a tenant from a
    business description, SQL, metadata, model output, or an evaluation case's
    expected values.
    """

    if type(question) is not str or type(authenticated_tenant_id) is not str:
        return ()
    current_key = _tenant_key(authenticated_tenant_id)
    if not current_key:
        return ()

    mentions = [match.group(1) for pattern in _TENANT_MENTION_PATTERNS for match in pattern.finditer(question)]
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
