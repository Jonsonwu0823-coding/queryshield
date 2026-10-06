"""Product-side retriever assembly shared by HTTP entrypoints and evaluation.

The same snapshot, embedding index and ``HybridRetriever`` configuration are
used whether a request arrives over HTTP or through the evaluation harness.  Fake mode
uses a deterministic hashed-feature embedding that works for any text; real
mode uses the configured embedding service and never falls back to Fake.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
from threading import Lock
from typing import Any, Sequence

from queryshield.catalog.catalog import DEFAULT_CATALOG_VERSION
from queryshield.knowledge.ingest import KNOWLEDGE_VERSION, import_knowledge
from queryshield.providers.embedding import (
    EmbeddingCallResult,
    FixedEmbedding,
    OpenAICompatibleEmbedding,
    _normalize_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KNOWLEDGE_ROOT = PROJECT_ROOT / "fixtures" / "knowledge"
# The knowledge snapshot keeps the catalog label that the frozen evidence recorded,
# while facts and the run configuration use the current catalog version.  Later catalog
# versions only added names, phrasings and recognition words, so the metric definitions
# are the same; changing the label would change the frozen evaluation's identity.  The
# demo knowledge base is new and carries the current version directly.
KNOWLEDGE_CATALOG_VERSION = "catalog-v2"
# The demo knowledge base (Chinese, product only) lives outside fixtures/knowledge,
# which the frozen fixture hash covers.
DEMO_KNOWLEDGE_ROOT = PROJECT_ROOT / "fixtures" / "demo" / "knowledge"
DEMO_KNOWLEDGE_VERSION = "knowledge-demo-v1"
# The product's current catalog version (catalog-v4), read from its one constant, not spelled again.
DEMO_CATALOG_VERSION = DEFAULT_CATALOG_VERSION
# Frozen identifiers: evidence recorded these names for the hashed-feature
# Fake embedding, so the product Fake keeps them for evidence continuity.
FAKE_EMBEDDING_MODEL = "w05-hash-feature-fake-v1"
FAKE_EMBEDDING_REVISION = "w05-hash-feature-v1"
FAKE_EMBEDDING_DIMENSIONS = 128
RETRIEVAL_DISABLED_VALUES = frozenset({"disabled", "off", "none"})
RETRIEVAL_CATALOG_VALUES = frozenset({"catalog", "catalog-only"})


class _CatalogSearchOnly:
    """Server setting: search_catalog searches the semantic catalog only (the keyword
    search); no knowledge index or embedding service is used."""

    def __repr__(self) -> str:
        return "CATALOG_SEARCH_ONLY"


CATALOG_SEARCH_ONLY = _CatalogSearchOnly()


def feature_vector(text: str, *, dimensions: int = FAKE_EMBEDDING_DIMENSIONS) -> tuple[float, ...]:
    """Deterministic hashed word/character-n-gram vector for the Fake embedding."""

    words = re.findall(r"[a-z0-9_]+|[一-鿿]+", text.lower())
    features: list[str] = []
    for word in words:
        features.append(word)
        if any("一" <= char <= "鿿" for char in word):
            features.extend(
                word[index : index + width]
                for width in (2, 3)
                for index in range(max(0, len(word) - width + 1))
            )
    counts = Counter(features)
    values = [0.0] * dimensions
    for feature, count in counts.items():
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        values[index] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values) if norm else tuple(values)


class HashFeatureEmbedding(FixedEmbedding):
    """Fake embedding that computes the hashed-feature vector for any input text."""

    def __init__(self) -> None:
        super().__init__(
            {"queryshield": feature_vector("queryshield")},
            model=FAKE_EMBEDDING_MODEL,
            model_revision=FAKE_EMBEDDING_REVISION,
            dimensions=FAKE_EMBEDDING_DIMENSIONS,
        )
        self._register_lock = Lock()

    def embed(
        self,
        inputs: Sequence[str],
        *,
        request_id: str | None = None,
        model_call_id: str | None = None,
    ) -> EmbeddingCallResult:
        normalized = _normalize_inputs(inputs)
        with self._register_lock:
            for value in normalized:
                if value not in self._vectors:
                    self._vectors[value] = feature_vector(value, dimensions=self.dimensions)
            return super().embed(inputs, request_id=request_id, model_call_id=model_call_id)


@dataclass(frozen=True)
class RetrievalRuntime:
    """One built retrieval stack: snapshot, index, embedder and hybrid retriever."""

    mode: str
    snapshot: Any
    index_build: Any
    embedder: Any
    retriever: Any


def build_retrieval_runtime(
    mode: str,
    *,
    reranker: object | None = None,
    knowledge_root: Path | None = None,
    knowledge_version: str | None = None,
    catalog_version: str | None = None,
    ingest_job_prefix: str = "w05-retrieval",
) -> RetrievalRuntime:
    """Build the product retriever; Real mode calls the embedding service for every chunk."""

    from queryshield.catalog import load_default_catalog
    from queryshield.knowledge.index import build_embedding_index
    from queryshield.knowledge.retrieval import HybridRetriever

    if mode not in {"fake", "real"}:
        raise ValueError("retrieval mode must be fake or real")
    root = knowledge_root or DEFAULT_KNOWLEDGE_ROOT
    snapshot = import_knowledge(
        root,
        root / "source_registry.json",
        catalog_version=catalog_version or KNOWLEDGE_CATALOG_VERSION,
        knowledge_version=knowledge_version or KNOWLEDGE_VERSION,
    )
    embedder = HashFeatureEmbedding() if mode == "fake" else OpenAICompatibleEmbedding.from_env()
    index_build = build_embedding_index(snapshot, embedder, ingest_job_id=f"{ingest_job_prefix}-{mode}")
    retriever = HybridRetriever(
        catalog=load_default_catalog(),
        snapshot=index_build.snapshot,
        index=index_build.index,
        embedder=embedder,
        reranker=reranker,
    )
    return RetrievalRuntime(mode, index_build.snapshot, index_build, embedder, retriever)


# The approver-only source that guards customer names, per knowledge base.
# A server table: neither the model nor a request can name another source, and
# fixtures/knowledge (covered by the frozen fingerprint) is not edited.
DEFAULT_SENSITIVE_PERMISSION_SOURCE_ID = "semantic-sensitive-customer-name"
DEMO_SENSITIVE_PERMISSION_SOURCE_ID = "demo-sensitive-customer-name"


@dataclass(frozen=True)
class ProductKnowledge:
    """The knowledge base the product uses, before embedding, and its permission source.

    ``snapshot`` is built with exactly the root, knowledge version and catalog
    version of ``shared_retrieval_runtime`` / ``shared_demo_retrieval_runtime``,
    so its id is the ``base_snapshot_id`` of the hybrid retriever's embedded
    snapshot.  Building it calls no model, embedding service or database.
    """

    snapshot: Any
    sensitive_source_id: str

    @property
    def snapshot_id(self) -> str:
        return str(self.snapshot.snapshot_id)


@dataclass(frozen=True)
class _KnowledgeBase:
    root: Path
    knowledge_version: str
    catalog_version: str
    sensitive_source_id: str
    ingest_job_prefix: str


def _knowledge_base(demo: bool) -> _KnowledgeBase:
    """The server's table of the two knowledge bases; nothing a request carries picks one."""

    if demo:
        return _KnowledgeBase(
            DEMO_KNOWLEDGE_ROOT,
            DEMO_KNOWLEDGE_VERSION,
            DEMO_CATALOG_VERSION,
            DEMO_SENSITIVE_PERMISSION_SOURCE_ID,
            "demo-retrieval",
        )
    return _KnowledgeBase(
        DEFAULT_KNOWLEDGE_ROOT,
        KNOWLEDGE_VERSION,
        KNOWLEDGE_CATALOG_VERSION,
        DEFAULT_SENSITIVE_PERMISSION_SOURCE_ID,
        "w05-retrieval",
    )


def _import_base(base: _KnowledgeBase) -> Any:
    return import_knowledge(
        base.root,
        base.root / "source_registry.json",
        catalog_version=base.catalog_version,
        knowledge_version=base.knowledge_version,
    )


_PRODUCT_KNOWLEDGE: dict[bool, ProductKnowledge] = {}


def product_knowledge(*, demo: bool) -> ProductKnowledge:
    """The default or the demo knowledge base (cached per process)."""

    with _CACHE_LOCK:
        cached = _PRODUCT_KNOWLEDGE.get(demo)
        if cached is not None:
            return cached
        base = _knowledge_base(demo)
        snapshot = _import_base(base)
        if not any(source.source_id == base.sensitive_source_id for source in snapshot.source_records):
            raise ValueError("the knowledge base has no permission source for customer names")
        knowledge = ProductKnowledge(snapshot, base.sensitive_source_id)
        _PRODUCT_KNOWLEDGE[demo] = knowledge
        return knowledge


def retrieval_setting() -> str:
    """``hybrid`` (default), ``catalog`` or ``disabled``; an explicit server setting.

    A missing embedding configuration is not ``disabled``: it blocks the run.
    """

    value = os.getenv("QUERYSHIELD_RETRIEVAL", "").strip().lower()
    if value in RETRIEVAL_DISABLED_VALUES:
        return "disabled"
    if value in RETRIEVAL_CATALOG_VALUES:
        return "catalog"
    return "hybrid"


_CACHE: dict[tuple[str, ...], RetrievalRuntime] = {}
_CACHE_LOCK = Lock()


def _cache_key(mode: str, root: Path) -> tuple[str, ...]:
    registry = root / "source_registry.json"
    registry_sha = hashlib.sha256(registry.read_bytes()).hexdigest() if registry.is_file() else "missing"
    if mode == "fake":
        return (mode, str(root), registry_sha, FAKE_EMBEDDING_MODEL)
    fingerprint = tuple(
        os.getenv(name, "").strip()
        for name in (
            "QUERYSHIELD_EMBEDDING_BASE_URL",
            "QUERYSHIELD_EMBEDDING_MODEL_NAME",
            "QUERYSHIELD_EMBEDDING_MODEL_REVISION",
            "QUERYSHIELD_EMBEDDING_DIMENSIONS",
        )
    )
    return (mode, str(root), registry_sha) + fingerprint


def _shared_runtime(mode: str, *, demo: bool) -> RetrievalRuntime:
    """Build lazily on first use and cache per process; failures are not cached."""

    base = _knowledge_base(demo)
    key = _cache_key(mode, base.root)
    with _CACHE_LOCK:
        runtime = _CACHE.get(key)
        if runtime is None:
            runtime = build_retrieval_runtime(
                mode,
                knowledge_root=base.root,
                knowledge_version=base.knowledge_version,
                catalog_version=base.catalog_version,
                ingest_job_prefix=base.ingest_job_prefix,
            )
            _CACHE[key] = runtime
        return runtime


def shared_retrieval_runtime(mode: str) -> RetrievalRuntime:
    return _shared_runtime(mode, demo=False)


def shared_demo_retrieval_runtime(mode: str) -> RetrievalRuntime:
    """The demo knowledge base, cached per process under its own key (own root, own registry)."""

    return _shared_runtime(mode, demo=True)


class IndexFileMismatch(ValueError):
    """The index file does not belong to the expected embedded snapshot."""


def retriever_from_index_file(
    mode: str,
    index_path: str | Path,
    *,
    demo: bool,
    expected_snapshot_id: str,
) -> Any:
    """The hybrid retriever over an index the host already built (MCP server process).

    The knowledge content is imported from the same root and versions as
    ``shared_retrieval_runtime`` / ``shared_demo_retrieval_runtime``; the
    embedded snapshot id is recomputed from it and the index hash, and must
    equal both the index's id and ``expected_snapshot_id``.  No chunk is
    embedded again; a search embeds only its query, as in the host.
    """

    from queryshield.catalog import load_default_catalog
    from queryshield.knowledge.index import embedded_snapshot_for_index, load_index
    from queryshield.knowledge.retrieval import HybridRetriever

    if mode not in {"fake", "real"}:
        raise ValueError("retrieval mode must be fake or real")
    index = load_index(index_path)
    embedded = embedded_snapshot_for_index(_import_base(_knowledge_base(demo)), index)
    if not (embedded.snapshot_id == index.snapshot_id == expected_snapshot_id):
        raise IndexFileMismatch("the index file is not the expected embedded snapshot")
    embedder = HashFeatureEmbedding() if mode == "fake" else OpenAICompatibleEmbedding.from_env()
    if embedder.model != index.model:
        raise IndexFileMismatch("the embedding configuration does not match the index")
    # The retriever itself refuses an embedder whose revision or dimensions differ from the index.
    return HybridRetriever(
        catalog=load_default_catalog(),
        snapshot=embedded,
        index=index,
        embedder=embedder,
    )


def reset_retrieval_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
        _PRODUCT_KNOWLEDGE.clear()


__all__ = [
    "CATALOG_SEARCH_ONLY",
    "DEFAULT_SENSITIVE_PERMISSION_SOURCE_ID",
    "DEMO_SENSITIVE_PERMISSION_SOURCE_ID",
    "FAKE_EMBEDDING_MODEL",
    "HashFeatureEmbedding",
    "IndexFileMismatch",
    "ProductKnowledge",
    "RetrievalRuntime",
    "build_retrieval_runtime",
    "feature_vector",
    "product_knowledge",
    "DEMO_CATALOG_VERSION",
    "DEMO_KNOWLEDGE_ROOT",
    "DEMO_KNOWLEDGE_VERSION",
    "reset_retrieval_cache",
    "retriever_from_index_file",
    "retrieval_setting",
    "shared_demo_retrieval_runtime",
    "shared_retrieval_runtime",
]
