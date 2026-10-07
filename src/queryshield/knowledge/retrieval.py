"""Hybrid catalog/knowledge retrieval behind the semantic search tool."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from uuid import uuid4

from queryshield.agent.proposals import ExecutionContext
from queryshield.catalog.catalog import SemanticCatalog
from queryshield.catalog.search_terms import expanded_query_terms
from queryshield.knowledge.acl import catalog_entry_visible, source_visible
from queryshield.knowledge.index import EmbeddingAdapter, EmbeddingIndex, cosine_similarity
from queryshield.knowledge.ingest import KnowledgeSnapshot
from queryshield.policy.argument_limits import TOP_K_RANGE
from queryshield.providers.embedding import EmbeddingCallResult
from queryshield.providers.rerank import (
    RerankAdapter,
    RerankCallRecord,
    RerankCandidate,
    rerank_authorized_candidates,
)


RETRIEVAL_VERSION = "hybrid-v1"
KEYWORD_SYNONYM_VERSION = "keyword-synonym-v1"
RRF_K = 60
MAX_ROUTE_CANDIDATES = 10
# A conservative absolute floor for vector-only evidence.  A positive cosine
# value is not enough: unrelated real embeddings commonly have a small positive
# cosine against some corpus item.  Calibrate it on the development set, never
# per probe.
VECTOR_MIN_SIMILARITY = 0.60


class RetrievalConfigurationError(ValueError):
    """The index, snapshot and catalog cannot form one safe retrieval space."""


def _query_hash(query: str) -> str:
    return sha256(query.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RetrievalCandidate:
    candidate_id: str
    text: str
    source_id: str
    version: str
    candidate_type: str = "knowledge"

    def as_public_item(self) -> dict[str, str]:
        return {
            "id": self.candidate_id,
            "text": self.text,
            "source_id": self.source_id,
            "version": self.version,
        }


@dataclass(frozen=True)
class RetrievalRank:
    candidate_id: str
    keyword_rank: int | None
    vector_rank: int | None
    rrf_score: float
    final_rank: int
    rerank_score: float | None = None
    keyword_score: int | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "candidate_id": self.candidate_id,
            "keyword_rank": self.keyword_rank,
            "vector_rank": self.vector_rank,
            "rrf_score": self.rrf_score,
            "final_rank": self.final_rank,
            "rerank_score": self.rerank_score,
        }
        if self.keyword_score is not None:
            result["keyword_score"] = self.keyword_score
        return result


@dataclass(frozen=True)
class RetrievalEvidence:
    """Internal evidence; raw query text and chunk bodies are deliberately absent."""

    retrieval_id: str
    run_id: str
    query_sha256: str
    snapshot_id: str
    strategy_version: str
    visible_candidate_ids: tuple[str, ...]
    keyword_candidate_ids: tuple[str, ...]
    vector_candidate_ids: tuple[str, ...]
    ranking: tuple[RetrievalRank, ...]
    selected_ids: tuple[str, ...]
    embedding_call_id: str | None
    embedding_provider_call_id: str | None
    embedding_provider_request_id: str | None
    embedding_actual_return: Mapping[str, object] | None
    rerank_call_id: str | None
    rerank_call: Mapping[str, object] | None
    elapsed_ms: int

    def as_dict(self) -> dict[str, object]:
        return {
            "retrieval_id": self.retrieval_id,
            "run_id": self.run_id,
            "query_sha256": self.query_sha256,
            "snapshot_id": self.snapshot_id,
            "strategy_version": self.strategy_version,
            "visible_candidate_ids": list(self.visible_candidate_ids),
            "keyword_candidate_ids": list(self.keyword_candidate_ids),
            "vector_candidate_ids": list(self.vector_candidate_ids),
            "ranking": [item.as_dict() for item in self.ranking],
            "selected_ids": list(self.selected_ids),
            "embedding_call_id": self.embedding_call_id,
            "embedding_provider_call_id": self.embedding_provider_call_id,
            "embedding_provider_request_id": self.embedding_provider_request_id,
            "embedding_actual_return": (
                dict(self.embedding_actual_return)
                if self.embedding_actual_return is not None
                else None
            ),
            "rerank_call_id": self.rerank_call_id,
            "rerank_call": dict(self.rerank_call) if self.rerank_call is not None else None,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True)
class RetrievalResult:
    items: tuple[Mapping[str, str], ...]
    evidence: RetrievalEvidence


def _keyword_score(candidate: RetrievalCandidate, query: str, terms: set[str]) -> int:
    searchable = f"{candidate.candidate_id} {candidate.text}".lower()
    score = sum(3 if term in candidate.candidate_id.lower() else 1 for term in terms if term in searchable)
    if query.lower() in candidate.text.lower():
        score += 5
    return score


def _rank_keyword_scored(
    candidates: Mapping[str, RetrievalCandidate],
    query: str,
) -> tuple[tuple[str, int], ...]:
    terms = expanded_query_terms(query)
    scored = [
        (score, candidate_id)
        for candidate_id, candidate in candidates.items()
        if (score := _keyword_score(candidate, query, terms)) > 0
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple((candidate_id, score) for score, candidate_id in scored[:MAX_ROUTE_CANDIDATES])


def _visible_candidates(
    catalog: SemanticCatalog,
    snapshot: KnowledgeSnapshot,
    context: ExecutionContext,
    chunks: Sequence[object],
) -> dict[str, RetrievalCandidate]:
    if not isinstance(context, ExecutionContext):
        raise RetrievalConfigurationError("retrieval requires a server-created context")
    candidates: dict[str, RetrievalCandidate] = {}
    for entry in catalog.entries:
        if not catalog_entry_visible(entry, context.role):
            continue
        candidate = RetrievalCandidate(
            candidate_id=entry.id,
            text=entry.text,
            source_id=entry.source_id,
            version=entry.version,
            candidate_type="catalog",
        )
        if candidate.candidate_id in candidates:
            raise RetrievalConfigurationError("catalog candidate IDs are not unique")
        candidates[candidate.candidate_id] = candidate

    sources = {source.source_id: source for source in snapshot.source_records}
    for raw_chunk in chunks:
        chunk_id = getattr(raw_chunk, "chunk_id", None)
        source_id = getattr(raw_chunk, "source_id", None)
        source_version = getattr(raw_chunk, "source_version", None)
        chunk_text = getattr(raw_chunk, "text", None)
        source = sources.get(source_id) if type(source_id) is str else None
        if source is None or source.version != source_version:
            raise RetrievalConfigurationError("index chunk is not bound to the snapshot source")
        if not source_visible(source, context):
            continue
        if any(type(value) is not str for value in (chunk_id, chunk_text, source_version)):
            raise RetrievalConfigurationError("snapshot chunk has invalid identity or content")
        if chunk_id in candidates:
            raise RetrievalConfigurationError("catalog and knowledge candidate IDs collide")
        candidates[chunk_id] = RetrievalCandidate(
            candidate_id=chunk_id,
            text=chunk_text,
            source_id=source_id,
            version=source_version,
        )
    return candidates


def _select_top_k(
    ranking: Sequence[RetrievalRank],
    candidates: Mapping[str, RetrievalCandidate],
    top_k: int,
) -> tuple[RetrievalRank, ...]:
    selected: list[RetrievalRank] = []
    selected_catalog_sources: set[str] = set()
    for rank in ranking:
        candidate = candidates[rank.candidate_id]
        if candidate.candidate_type == "catalog":
            if candidate.source_id in selected_catalog_sources:
                continue
            selected_catalog_sources.add(candidate.source_id)
        selected.append(rank)
        if len(selected) == top_k:
            break
    return tuple(selected)


def _rank_vector(
    candidates: Mapping[str, RetrievalCandidate],
    index: EmbeddingIndex,
    query_vector: Sequence[float],
) -> tuple[str, ...]:
    chunks = {chunk.chunk_id: chunk for chunk in index.chunks}
    scored: list[tuple[float, str]] = []
    for candidate_id in candidates:
        chunk = chunks.get(candidate_id)
        if chunk is None:
            continue
        score = cosine_similarity(query_vector, chunk.vector)
        if score >= VECTOR_MIN_SIMILARITY:
            scored.append((score, candidate_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(candidate_id for _, candidate_id in scored[:MAX_ROUTE_CANDIDATES])


def _rrf_ranking(
    keyword_ids: Sequence[str],
    vector_ids: Sequence[str],
    *,
    keyword_scores: Mapping[str, int] | None = None,
) -> tuple[RetrievalRank, ...]:
    keyword_ranks = {candidate_id: rank for rank, candidate_id in enumerate(keyword_ids, start=1)}
    vector_ranks = {candidate_id: rank for rank, candidate_id in enumerate(vector_ids, start=1)}
    candidate_ids = set(keyword_ranks) | set(vector_ranks)
    scored: list[tuple[float, str]] = []
    for candidate_id in candidate_ids:
        score = 0.0
        if candidate_id in keyword_ranks:
            score += 1.0 / (RRF_K + keyword_ranks[candidate_id])
        if candidate_id in vector_ranks:
            score += 1.0 / (RRF_K + vector_ranks[candidate_id])
        scored.append((score, candidate_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(
        RetrievalRank(
            candidate_id=candidate_id,
            keyword_rank=keyword_ranks.get(candidate_id),
            vector_rank=vector_ranks.get(candidate_id),
            rrf_score=score,
            final_rank=rank,
            keyword_score=(keyword_scores or {}).get(candidate_id),
        )
        for rank, (score, candidate_id) in enumerate(scored, start=1)
    )


def _embedding_actual_return(result: EmbeddingCallResult) -> dict[str, object]:
    return {
        "input_count": 1,
        "vector_count": len(result.vectors),
        "dimensions": result.dimensions,
        "inputs_sha256": result.inputs_sha256,
        "usage": result.usage.as_dict(),
    }


def _check_search_args(query: object, top_k: object) -> None:
    if type(query) is not str or not query.strip():
        raise RetrievalConfigurationError("query must be a non-empty string")
    if type(top_k) is not int or not TOP_K_RANGE[0] <= top_k <= TOP_K_RANGE[1]:
        raise RetrievalConfigurationError("top_k must be between one and five")


def _rerank_ranking(
    reranker: RerankAdapter,
    query: str,
    ranking: tuple[RetrievalRank, ...],
    candidates: Mapping[str, RetrievalCandidate],
) -> tuple[tuple[RetrievalRank, ...], RerankCallRecord | None]:
    """Reorder the top candidates by the reranker; a failed rerank leaves no ranking at all."""

    candidate_pool = ranking[:MAX_ROUTE_CANDIDATES]
    rerank_candidates = tuple(
        RerankCandidate(
            candidate_id=rank.candidate_id,
            text=candidates[rank.candidate_id].text,
            source_id=candidates[rank.candidate_id].source_id,
            version=candidates[rank.candidate_id].version,
        )
        for rank in candidate_pool
    )
    rerank_call, _filtered_ids = rerank_authorized_candidates(
        reranker,
        query,
        rerank_candidates,
        authorized_candidate_ids=frozenset(candidates),
        # Overfetch within the existing <=10 candidate bound so
        # catalog fields from one source do not crowd out documents.
        top_n=len(rerank_candidates),
    )
    if rerank_call is None or rerank_call.status != "succeeded":
        # Do not hide a failed rerank by silently returning the pre-rerank hybrid
        # ranking as if it succeeded.
        return (), rerank_call
    ranks_by_id = {rank.candidate_id: rank for rank in candidate_pool}
    scores_by_id = dict(zip(rerank_call.returned_candidate_ids, rerank_call.scores, strict=True))
    reranked = tuple(
        RetrievalRank(
            candidate_id=candidate_id,
            keyword_rank=ranks_by_id[candidate_id].keyword_rank,
            vector_rank=ranks_by_id[candidate_id].vector_rank,
            rrf_score=ranks_by_id[candidate_id].rrf_score,
            final_rank=rank,
            rerank_score=scores_by_id[candidate_id],
            keyword_score=ranks_by_id[candidate_id].keyword_score,
        )
        for rank, candidate_id in enumerate(rerank_call.returned_candidate_ids, start=1)
    )
    return reranked, rerank_call


def _finish(
    *,
    started: float,
    retrieval_id: str,
    context: ExecutionContext,
    query: str,
    snapshot_id: str,
    strategy_version: str,
    candidates: Mapping[str, RetrievalCandidate],
    keyword_ids: Sequence[str],
    vector_ids: Sequence[str],
    ranking: tuple[RetrievalRank, ...],
    top_k: int,
    embedding: EmbeddingCallResult | None = None,
    rerank_call: RerankCallRecord | None = None,
) -> RetrievalResult:
    # Catalog entries and knowledge chunks may come from one shared source.
    # Returning several catalog fields from that source can crowd out the
    # separately versioned documents that explain the same answer. Keep the
    # ranked evidence intact, but expose at most one result per source.
    selected = _select_top_k(ranking, candidates, top_k)
    elapsed_ms = max(0, int(round((perf_counter() - started) * 1000)))
    evidence = RetrievalEvidence(
        retrieval_id=retrieval_id,
        run_id=context.run_id,
        query_sha256=_query_hash(query),
        snapshot_id=snapshot_id,
        strategy_version=strategy_version,
        visible_candidate_ids=tuple(sorted(candidates)),
        keyword_candidate_ids=tuple(keyword_ids),
        vector_candidate_ids=tuple(vector_ids),
        ranking=ranking,
        selected_ids=tuple(item.candidate_id for item in selected),
        embedding_call_id=embedding.model_call_id if embedding is not None else None,
        embedding_provider_call_id=embedding.provider_call_id if embedding is not None else None,
        embedding_provider_request_id=embedding.provider_request_id if embedding is not None else None,
        embedding_actual_return=_embedding_actual_return(embedding) if embedding is not None else None,
        rerank_call_id=rerank_call.call_id if rerank_call is not None else None,
        rerank_call=rerank_call.as_dict() if rerank_call is not None else None,
        elapsed_ms=elapsed_ms,
    )
    return RetrievalResult(
        items=tuple(candidates[item.candidate_id].as_public_item() for item in selected),
        evidence=evidence,
    )


@dataclass
class HybridRetriever:
    """Two-route retrieval over catalog entries and one versioned knowledge index."""

    catalog: SemanticCatalog
    snapshot: KnowledgeSnapshot
    index: EmbeddingIndex
    embedder: EmbeddingAdapter
    reranker: RerankAdapter | None = None

    def __post_init__(self) -> None:
        if self.index.snapshot_id != self.snapshot.snapshot_id:
            raise RetrievalConfigurationError("index and snapshot IDs do not match")
        if self.snapshot.embedding_model_revision != self.index.model_revision:
            raise RetrievalConfigurationError("snapshot and index model revisions do not match")
        if self.snapshot.embedding_dimensions != self.index.dimensions:
            raise RetrievalConfigurationError("snapshot and index dimensions do not match")
        if self.embedder.model_revision != self.index.model_revision:
            raise RetrievalConfigurationError("query embedding revision does not match the index")
        if self.embedder.dimensions != self.index.dimensions:
            raise RetrievalConfigurationError("query embedding dimensions do not match the index")

    def _visible_candidates(self, context: ExecutionContext) -> dict[str, RetrievalCandidate]:
        return _visible_candidates(self.catalog, self.snapshot, context, self.index.chunks)

    def search(
        self,
        query: str,
        *,
        context: ExecutionContext,
        top_k: int = 3,
    ) -> RetrievalResult:
        _check_search_args(query, top_k)
        started = perf_counter()
        retrieval_id = f"retrieval-{uuid4()}"
        candidates = self._visible_candidates(context)
        keyword_scored = _rank_keyword_scored(candidates, query)
        keyword_ids = tuple(candidate_id for candidate_id, _ in keyword_scored)
        embedding = self.embedder.embed(
            (query,),
            request_id=f"{retrieval_id}-request",
            model_call_id=f"{retrieval_id}-embedding",
            run_id=context.run_id,
        )
        vector_ids = _rank_vector(candidates, self.index, embedding.vectors[0])
        ranking = _rrf_ranking(keyword_ids, vector_ids, keyword_scores=dict(keyword_scored))
        rerank_call = None
        if self.reranker is not None and ranking:
            ranking, rerank_call = _rerank_ranking(self.reranker, query, ranking, candidates)
        return _finish(
            started=started,
            retrieval_id=retrieval_id,
            context=context,
            query=query,
            snapshot_id=self.index.snapshot_id,
            strategy_version=RETRIEVAL_VERSION,
            candidates=candidates,
            keyword_ids=keyword_ids,
            vector_ids=vector_ids,
            ranking=ranking,
            top_k=top_k,
            embedding=embedding,
            rerank_call=rerank_call,
        )


@dataclass
class KeywordSynonymRetriever:
    """Lexical retrieval baseline using the same visible catalog and snapshot."""

    catalog: SemanticCatalog
    snapshot: KnowledgeSnapshot

    def search(
        self,
        query: str,
        *,
        context: ExecutionContext,
        top_k: int = 3,
    ) -> RetrievalResult:
        _check_search_args(query, top_k)
        started = perf_counter()
        retrieval_id = f"retrieval-{uuid4()}"
        candidates = _visible_candidates(
            self.catalog,
            self.snapshot,
            context,
            self.snapshot.chunk_records,
        )
        keyword_scored = _rank_keyword_scored(candidates, query)
        ranking = tuple(
            RetrievalRank(
                candidate_id=candidate_id,
                keyword_rank=rank,
                vector_rank=None,
                rrf_score=0.0,
                final_rank=rank,
                keyword_score=score,
            )
            for rank, (candidate_id, score) in enumerate(keyword_scored, start=1)
        )
        return _finish(
            started=started,
            retrieval_id=retrieval_id,
            context=context,
            query=query,
            snapshot_id=self.snapshot.snapshot_id,
            strategy_version=KEYWORD_SYNONYM_VERSION,
            candidates=candidates,
            keyword_ids=tuple(candidate_id for candidate_id, _ in keyword_scored),
            vector_ids=(),
            ranking=ranking,
            top_k=top_k,
        )


__all__ = [
    "HybridRetriever",
    "KeywordSynonymRetriever",
    "KEYWORD_SYNONYM_VERSION",
    "MAX_ROUTE_CANDIDATES",
    "RETRIEVAL_VERSION",
    "RRF_K",
    "VECTOR_MIN_SIMILARITY",
    "RetrievalCandidate",
    "RetrievalConfigurationError",
    "RetrievalEvidence",
    "RetrievalRank",
    "RetrievalResult",
]
