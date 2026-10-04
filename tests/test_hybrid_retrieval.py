from __future__ import annotations

from queryshield.agent.proposals import ExecutionContext
from queryshield.catalog import load_default_catalog
from queryshield.knowledge.index import build_embedding_index
from queryshield.knowledge.ingest import load_snapshot
from queryshield.knowledge.retrieval import HybridRetriever, KeywordSynonymRetriever
from queryshield.providers.embedding import FixedEmbedding
from queryshield.providers.rerank import FakeReranker, RerankCallRecord
from queryshield.tools import ControlledTools


SNAPSHOT_PATH = "fixtures/knowledge/snapshots/knowledge-v1-9f580dd7f887ed0a.json"


def _context(*, tenant_id: str = "tenant-A", role: str = "requester") -> ExecutionContext:
    return ExecutionContext(
        run_id=f"run-hybrid-{tenant_id}-{role}",
        tenant_id=tenant_id,
        principal_id="principal-hybrid",
        role=role,
    )


def _retriever() -> HybridRetriever:
    snapshot = load_snapshot(SNAPSHOT_PATH)
    vectors: dict[str, tuple[float, float, float]] = {}
    for chunk in snapshot.chunk_records:
        if "metric-gross" in chunk.source_id:
            vector = (1.0, 0.0, 0.0)
        elif "metric-net" in chunk.source_id:
            vector = (0.0, 1.0, 0.0)
        elif "orders" in chunk.source_id or "schema-orders" in chunk.source_id:
            vector = (0.0, 0.0, 1.0)
        elif "refund" in chunk.source_id:
            vector = (0.7, 0.3, 0.0)
        else:
            vector = (0.2, 0.2, 0.2)
        vectors[chunk.text] = vector
    vectors.update(
        {
            "营业额": (1.0, 0.0, 0.0),
            "订单": (0.0, 0.0, 1.0),
            "客户姓名": (0.2, 0.2, 0.2),
            "__unknown__": (0.0, 0.0, 0.0),
            "__low_confidence__": (1.0, -1.0, -1.0),
        }
    )
    embedder = FixedEmbedding(vectors, model_revision="fixed-hybrid-v1", dimensions=3)
    build = build_embedding_index(snapshot, embedder, ingest_job_id="ingest-hybrid-test")
    return HybridRetriever(
        catalog=load_default_catalog(),
        snapshot=build.snapshot,
        index=build.index,
        embedder=embedder,
    )


def _keyword_retriever() -> KeywordSynonymRetriever:
    snapshot = load_snapshot(SNAPSHOT_PATH)
    return KeywordSynonymRetriever(
        catalog=load_default_catalog(),
        snapshot=snapshot,
    )


def test_hybrid_search_keeps_public_shape_and_records_rrf_evidence() -> None:
    tools = ControlledTools(retriever=_retriever())
    context = _context()

    first = tools.search_catalog({"query": "营业额", "top_k": 3}, context=context)
    second = tools.search_catalog({"query": "营业额", "top_k": 3}, context=context)

    assert all(set(item) == {"id", "text", "source_id", "version"} for item in first["items"])
    first_ids = [item["id"] for item in first["items"]]
    second_ids = [item["id"] for item in second["items"]]
    assert first_ids == second_ids

    first_evidence = tools.get_retrieval_evidence(
        tools._retrieval_evidence[(context.run_id, next(iter(tools._retrieval_evidence))[1])].retrieval_id,
        context=context,
    )
    assert first_evidence.strategy_version == "hybrid-v1"
    assert first_evidence.selected_ids == tuple(first_ids)
    assert first_evidence.embedding_call_id
    assert first_evidence.embedding_actual_return["usage"]["usage_status"] == "known"
    assert "营业额" not in str(first_evidence.as_dict())


def test_hybrid_top_k_caps_repeated_catalog_source() -> None:
    result = _retriever().search("营业额", context=_context(), top_k=3)

    catalog_items = [item for item in result.items if item["source_id"] == "commerce-v1"]
    assert len(catalog_items) <= 1
    assert len(result.items) == 3
    assert len(result.evidence.ranking) >= len(result.items)


def test_auth_filter_happens_before_and_after_hybrid_retrieval() -> None:
    tools = ControlledTools(retriever=_retriever())
    tenant_a = _context(tenant_id="tenant-A")
    result = tools.search_catalog({"query": "订单", "top_k": 5}, context=tenant_a)
    evidence = tools.get_retrieval_evidence(
        next(iter(tools._retrieval_evidence))[1],
        context=tenant_a,
    )

    assert all("tenant-b" not in str(item).lower() for item in result["items"])
    assert all("tenant-b" not in candidate_id.lower() for candidate_id in evidence.visible_candidate_ids)

    requester = tools.search_catalog({"query": "客户姓名", "top_k": 5}, context=tenant_a)
    assert not any("sensitive-customer-name" in item["id"] for item in requester["items"])


def test_unknown_query_has_explicit_empty_result() -> None:
    tools = ControlledTools(retriever=_retriever())
    context = _context()

    result = tools.search_catalog({"query": "__unknown__", "top_k": 3}, context=context)

    assert result == {"items": []}
    evidence = tools.get_retrieval_evidence(
        next(iter(tools._retrieval_evidence))[1],
        context=context,
    )
    assert evidence.selected_ids == ()
    assert evidence.visible_candidate_ids


def test_keyword_synonym_baseline_uses_same_acl_without_embedding() -> None:
    context = _context(tenant_id="tenant-A")
    baseline = _keyword_retriever()

    result = baseline.search("营业额", context=context, top_k=3)
    evidence = result.evidence

    assert evidence.strategy_version == "keyword-synonym-v1"
    assert evidence.run_id == context.run_id
    assert len(result.items) <= 3
    assert evidence.embedding_call_id is None
    assert evidence.embedding_actual_return is None
    assert evidence.vector_candidate_ids == ()
    assert evidence.ranking
    assert evidence.ranking[0].keyword_rank == 1
    assert evidence.ranking[0].keyword_score is not None
    assert all("tenant-b" not in candidate_id.lower() for candidate_id in evidence.visible_candidate_ids)
    assert not any("sensitive-customer-name" in item["id"] for item in baseline.search("客户姓名", context=context, top_k=5).items)


def test_keyword_synonym_baseline_keeps_empty_query_result_explicit() -> None:
    result = _keyword_retriever().search("__unknown__", context=_context(), top_k=3)

    assert result.items == ()
    assert result.evidence.selected_ids == ()
    assert result.evidence.embedding_call_id is None
    assert result.evidence.strategy_version == "keyword-synonym-v1"


def test_low_confidence_vector_only_query_has_explicit_empty_result() -> None:
    tools = ControlledTools(retriever=_retriever())
    context = _context()

    result = tools.search_catalog({"query": "__low_confidence__", "top_k": 3}, context=context)

    assert result == {"items": []}


def test_rerank_uses_only_visible_hybrid_candidates_and_records_call() -> None:
    context = _context(tenant_id="tenant-A")
    retriever = _retriever()
    visible = retriever._visible_candidates(context)
    assert visible
    # Reverse the original ranking deterministically; scores are tied to candidate IDs.
    retriever.reranker = FakeReranker(
        {candidate_id: float(index) for index, candidate_id in enumerate(visible)}
    )

    result = retriever.search("订单", context=context, top_k=3)
    evidence = result.evidence

    assert len(result.items) == 3
    assert evidence.rerank_call_id
    assert evidence.rerank_call is not None
    assert evidence.rerank_call["status"] == "succeeded"
    assert len(evidence.rerank_call["input_candidate_ids"]) <= 10
    assert all("tenant-b" not in candidate_id.lower() for candidate_id in evidence.rerank_call["input_candidate_ids"])
    assert all(item.rerank_score is not None for item in evidence.ranking)
    ranked_ids = {item.candidate_id for item in evidence.ranking}
    assert set(evidence.selected_ids) <= ranked_ids
    assert tuple(item["id"] for item in result.items) == evidence.selected_ids


def test_failed_rerank_does_not_silently_return_pre_rerank_success() -> None:
    class _FailedReranker:
        model = "fake-failed-reranker"

        def rerank(self, query, candidates, *, top_n):
            return RerankCallRecord(
                call_id="rerank-timeout",
                status="timeout",
                model=self.model,
                input_candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
                returned_candidate_ids=(),
                scores=(),
                usage_status="unknown",
                total_tokens=None,
                error_code="timeout",
            )

    retriever = _retriever()
    retriever.reranker = _FailedReranker()

    result = retriever.search("订单", context=_context(), top_k=3)

    assert result.items == ()
    assert result.evidence.selected_ids == ()
    assert result.evidence.rerank_call is not None
    assert result.evidence.rerank_call["status"] == "timeout"
    assert result.evidence.rerank_call["total_tokens"] is None
