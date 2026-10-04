"""The same input gives the same result, in the same order, locally and over MCP.

Identities: A requester, A approver, B requester.  Retrieval: the catalog
keyword search (catalog setting, disabled retrieval, B0) and the hybrid
retriever with the Fake embedding.  The queries reach tenant-only sources,
the approver-only source and the deleted refund policy's successor.
"""

from __future__ import annotations

import pytest

from queryshield.knowledge.runtime import shared_retrieval_runtime
from queryshield.mcp_metadata.tools import McpMetadataTools
from queryshield.tools.semantic import ControlledTools, ToolError

from mcp_helpers import config, context


QUERIES = [
    "退款后净额是怎么算的？",
    "已支付订单数",
    "营业额",
    "销售额",
    "租户 订单 概览",
    "Tenant A orders",
    "Tenant B orders",
    "tenant orders tenant_scope",
    "客户姓名",
    "Sensitive customer names approver",
    "退款政策",
    "退款 refund policy v1",
    "时间窗口 UTC",
    "orders 表有哪些列",
    "  净额  ",
    "完全无关的问题 xyz",
]
TOP_KS = (1, 3, 5)
TABLE_REQUESTS = [["orders"], ["customers", "orders"], ["refunds", "customers", "orders"], [" orders "]]
IDENTITIES = [("A", "requester"), ("A", "approver"), ("B", "requester")]


@pytest.fixture(scope="module", params=["keyword", "hybrid"])
def retriever(request):
    return None if request.param == "keyword" else shared_retrieval_runtime("fake").retriever


@pytest.mark.parametrize("tenant, role", IDENTITIES)
def test_search_and_describe_match_the_local_tools(retriever, tenant, role):
    ctx = context(tenant, role, run_id=f"run-consistency-{tenant}-{role}")
    local = ControlledTools(retriever=retriever)
    remote = McpMetadataTools(retriever=retriever, metadata_config=config())
    try:
        for query in QUERIES:
            for top_k in TOP_KS:
                arguments = {"query": query, "top_k": top_k}
                assert remote.search_catalog(arguments, context=ctx) == local.search_catalog(arguments, context=ctx), (query, top_k)
        assert remote.search_catalog({"query": "净额"}, context=ctx) == local.search_catalog({"query": "净额"}, context=ctx)
        for tables in TABLE_REQUESTS:
            arguments = {"tables": tables}
            assert remote.describe_tables(arguments, context=ctx) == local.describe_tables(arguments, context=ctx)
        for arguments in ({"query": "净额", "tenant_id": "B"}, {"query": ""}, {"query": "净额", "top_k": 9}):
            with pytest.raises(ToolError) as remote_error:
                remote.search_catalog(arguments, context=ctx)
            with pytest.raises(ToolError) as local_error:
                local.search_catalog(arguments, context=ctx)
            assert (remote_error.value.code, remote_error.value.message) == (local_error.value.code, local_error.value.message)
    finally:
        record = remote.close()
    assert record is not None and record["cleanup"] == "ok"
    assert record["retrieval"] == ("hybrid" if retriever is not None else "keyword")


def test_tenant_and_role_boundaries_hold_over_mcp():
    retriever = shared_retrieval_runtime("fake").retriever
    seen = {}
    for tenant, role in IDENTITIES:
        remote = McpMetadataTools(retriever=retriever, metadata_config=config())
        ctx = context(tenant, role)
        try:
            sources = set()
            for query in ("Tenant A orders", "Tenant B orders", "tenant orders tenant_scope", "Sensitive customer names approver"):
                sources |= {item["source_id"] for item in remote.search_catalog({"query": query, "top_k": 5}, context=ctx)["items"]}
        finally:
            remote.close()
        seen[(tenant, role)] = sources
    assert "tenant-b-orders-overview" not in seen[("A", "requester")] | seen[("A", "approver")]
    assert "tenant-a-orders-overview" not in seen[("B", "requester")]
    assert "tenant-a-orders-overview" in seen[("A", "requester")]
    assert "semantic-sensitive-customer-name" not in seen[("A", "requester")] | seen[("B", "requester")]
    assert "semantic-sensitive-customer-name" in seen[("A", "approver")]
    assert not any("semantic-refund-policy-v1" in sources for sources in seen.values())
