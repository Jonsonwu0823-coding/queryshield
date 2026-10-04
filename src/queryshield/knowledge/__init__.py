"""Versioned knowledge APIs with W04 snapshot permission checks."""

__all__ = [
    "CHUNKER_VERSION",
    "KNOWLEDGE_VERSION",
    "MAX_CHUNKS",
    "MAX_FILES",
    "MAX_FILE_BYTES",
    "ChunkRecord",
    "KnowledgeImportError",
    "KnowledgeSnapshot",
    "SourceRecord",
    "SourceSpec",
    "build_snapshot",
    "EmbeddingBuildResult",
    "EmbeddingIndex",
    "IndexChunk",
    "IndexHit",
    "IndexValidationError",
    "build_embedding_index",
    "cosine_similarity",
    "load_index",
    "publish_index",
    "search_index",
    "HybridRetriever",
    "MAX_ROUTE_CANDIDATES",
    "RETRIEVAL_VERSION",
    "RRF_K",
    "RetrievalCandidate",
    "RetrievalConfigurationError",
    "RetrievalEvidence",
    "RetrievalRank",
    "RetrievalResult",
    "import_knowledge",
    "load_snapshot",
    "write_snapshot",
    "KnowledgeAccessError",
    "KnowledgeIdentity",
    "KnowledgeSnapshotRepository",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(name)
    if name in {
        "EmbeddingBuildResult",
        "EmbeddingIndex",
        "IndexChunk",
        "IndexHit",
        "IndexValidationError",
        "build_embedding_index",
        "cosine_similarity",
        "load_index",
        "publish_index",
        "search_index",
    }:
        from queryshield.knowledge import index

        return getattr(index, name)
    if name in {
        "HybridRetriever",
        "MAX_ROUTE_CANDIDATES",
        "RETRIEVAL_VERSION",
        "RRF_K",
        "RetrievalCandidate",
        "RetrievalConfigurationError",
        "RetrievalEvidence",
        "RetrievalRank",
        "RetrievalResult",
    }:
        from queryshield.knowledge import retrieval

        return getattr(retrieval, name)
    if name in {"KnowledgeAccessError", "KnowledgeIdentity", "KnowledgeSnapshotRepository"}:
        from queryshield.knowledge import snapshots

        return getattr(snapshots, name)
    from queryshield.knowledge import ingest

    return getattr(ingest, name)
