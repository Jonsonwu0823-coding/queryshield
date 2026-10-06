"""Query words for the keyword search routes.

The catalog fallback search and the keyword retriever expand the same business
wording, so the table lives once, here, and depends on the standard library
only (the retrieval module pulls in the model providers; this one must not).
"""

from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+")
SYNONYMS: dict[str, tuple[str, ...]] = {
    "已支付订单数": ("paid_count", "paid"),
    "订单数量": ("paid_count",),
    "营业额": ("gross_fen",),
    "支付订单总额": ("gross_fen",),
    "销售额": ("gross_fen", "net_fen"),
    "退款": ("refund_fen",),
    "退款金额": ("refund_fen",),
    "净额": ("net_fen",),
    "退款后": ("net_fen",),
}


def tokens(value: str) -> set[str]:
    return set(TOKEN_RE.findall(value.lower()))


def expanded_query_terms(query: str) -> set[str]:
    terms = tokens(query)
    for phrase, replacements in SYNONYMS.items():
        if phrase in query:
            for replacement in replacements:
                terms.update(tokens(replacement))
    return terms
