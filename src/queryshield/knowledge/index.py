"""Versioned local embedding index construction for the W03 knowledge snapshot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Protocol
from uuid import uuid4

from queryshield.knowledge.ingest import KnowledgeSnapshot
from queryshield.providers.embedding import EmbeddingCallResult, OperationUsage


INDEX_VERSION = "vector-index-v1"
MAX_BATCH_SIZE = 8


class IndexValidationError(ValueError):
    """An embedding index or its model metadata is not safe to publish."""


class EmbeddingAdapter(Protocol):
    model: str
    model_revision: str
    dimensions: int

    def embed(
        self,
        inputs: Sequence[str],
        *,
        request_id: str | None = None,
        model_call_id: str | None = None,
    ) -> EmbeddingCallResult:
        """Return one validated embedding operation."""


@dataclass(frozen=True)
class IndexChunk:
    chunk_id: str
    source_id: str
    source_version: str
    text: str
    vector: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "text": self.text,
            "vector": list(self.vector),
        }


@dataclass(frozen=True)
class EmbeddingIndex:
    index_version: str
    snapshot_id: str
    knowledge_version: str
    catalog_version: str
    model: str
    model_revision: str
    dimensions: int
    index_hash: str
    chunks: tuple[IndexChunk, ...]

    def __post_init__(self) -> None:
        if self.index_version != INDEX_VERSION:
            raise IndexValidationError("unsupported index version")
        if not self.snapshot_id or not self.model or not self.model_revision:
            raise IndexValidationError("index identity fields must be non-empty")
        if type(self.dimensions) is not int or self.dimensions <= 0:
            raise IndexValidationError("index dimensions must be positive")
        if not self.chunks:
            raise IndexValidationError("index must contain at least one chunk")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise IndexValidationError("index chunk IDs must be unique")
        for chunk in self.chunks:
            if len(chunk.vector) != self.dimensions:
                raise IndexValidationError("index contains a mixed vector dimension")
            if any(not math.isfinite(value) for value in chunk.vector):
                raise IndexValidationError("index contains a non-finite vector value")

    def as_dict(self) -> dict[str, object]:
        return {
            "index_version": self.index_version,
            "snapshot_id": self.snapshot_id,
            "knowledge_version": self.knowledge_version,
            "catalog_version": self.catalog_version,
            "model": self.model,
            "model_revision": self.model_revision,
            "dimensions": self.dimensions,
            "index_hash": self.index_hash,
            "chunks": [chunk.as_dict() for chunk in self.chunks],
        }


@dataclass(frozen=True)
class IndexHit:
    chunk_id: str
    source_id: str
    source_version: str
    text: str
    score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "text": self.text,
            "score": self.score,
        }


@dataclass(frozen=True)
class EmbeddingBuildResult:
    ingest_job_id: str
    base_snapshot_id: str
    snapshot: KnowledgeSnapshot
    index: EmbeddingIndex
    operation_usages: tuple[OperationUsage, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "ingest_job_id": self.ingest_job_id,
            "base_snapshot_id": self.base_snapshot_id,
            "snapshot_id": self.snapshot.snapshot_id,
            "index_hash": self.index.index_hash,
            "model": self.index.model,
            "model_revision": self.index.model_revision,
            "dimensions": self.index.dimensions,
            "chunk_count": len(self.index.chunks),
            "operation_count": len(self.operation_usages),
            "operation_usage": [usage.as_dict() for usage in self.operation_usages],
        }


def new_ingest_job_id() -> str:
    return f"ingest-{uuid4()}"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_vector(vector: Sequence[object], *, dimensions: int) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes)) or len(vector) != dimensions:
        raise IndexValidationError("embedding vector has an unexpected dimension")
    values: list[float] = []
    for value in vector:
        if type(value) not in {int, float} or isinstance(value, bool) or not math.isfinite(float(value)):
            raise IndexValidationError("embedding vector contains a non-finite value")
        values.append(float(value))
    return tuple(values)


def _embedded_snapshot(
    snapshot: KnowledgeSnapshot,
    *,
    model_revision: str,
    dimensions: int,
    index_hash: str,
) -> KnowledgeSnapshot:
    manifest = {
        "knowledge_version": snapshot.knowledge_version,
        "catalog_version": snapshot.catalog_version,
        "chunker_version": snapshot.chunker_version,
        "embedding_model_revision": model_revision,
        "embedding_dimensions": dimensions,
        "index_hash": index_hash,
        "sources": [source.as_dict() for source in snapshot.source_records],
        "chunks": [chunk.as_dict() for chunk in snapshot.chunk_records],
    }
    manifest_sha256 = _sha256(manifest)
    return replace(
        snapshot,
        snapshot_id=f"{snapshot.knowledge_version}-{manifest_sha256[:16]}",
        embedding_model_revision=model_revision,
        embedding_dimensions=dimensions,
        index_hash=index_hash,
        manifest_sha256=manifest_sha256,
    )


def build_embedding_index(
    snapshot: KnowledgeSnapshot,
    embedder: EmbeddingAdapter,
    *,
    ingest_job_id: str | None = None,
    batch_size: int = MAX_BATCH_SIZE,
) -> EmbeddingBuildResult:
    """Embed a snapshot and return a fully validated, not-yet-published index."""

    if not isinstance(snapshot, KnowledgeSnapshot):
        raise TypeError("snapshot must be a KnowledgeSnapshot")
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise IndexValidationError("batch_size must be between one and eight")
    for name in ("model", "model_revision", "dimensions", "embed"):
        if not hasattr(embedder, name):
            raise IndexValidationError(f"embedding adapter is missing {name}")
    if type(embedder.dimensions) is not int or embedder.dimensions <= 0:
        raise IndexValidationError("embedding adapter dimensions must be positive")
    if not embedder.model_revision.strip():
        raise IndexValidationError("embedding adapter revision must be non-empty")
    if not snapshot.chunk_records:
        raise IndexValidationError("cannot build an index from an empty snapshot")

    job_id = ingest_job_id or new_ingest_job_id()
    chunks = sorted(snapshot.chunk_records, key=lambda chunk: chunk.chunk_id)
    indexed_chunks: list[IndexChunk] = []
    operation_usages: list[OperationUsage] = []
    for batch_index in range(0, len(chunks), batch_size):
        batch = chunks[batch_index : batch_index + batch_size]
        texts = tuple(chunk.text for chunk in batch)
        result = embedder.embed(
            texts,
            request_id=f"{job_id}-request-{batch_index // batch_size}",
            model_call_id=f"{job_id}-embedding-{batch_index // batch_size}",
        )
        if not isinstance(result, EmbeddingCallResult):
            raise IndexValidationError("embedding adapter returned an invalid result")
        if result.model_revision != embedder.model_revision:
            raise IndexValidationError("embedding model revision changed during one ingest job")
        if result.dimensions != embedder.dimensions:
            raise IndexValidationError("embedding dimensions changed during one ingest job")
        if len(result.vectors) != len(batch):
            raise IndexValidationError("embedding count does not match the batch")
        if result.usage.model_call_id != result.model_call_id:
            raise IndexValidationError("operation usage is not bound to the embedding call")
        for chunk, vector in zip(batch, result.vectors, strict=True):
            indexed_chunks.append(
                IndexChunk(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    source_version=chunk.source_version,
                    text=chunk.text,
                    vector=_validate_vector(vector, dimensions=embedder.dimensions),
                )
            )
        operation_usages.append(result.usage)

    index_hash = _sha256(
        {
            "index_version": INDEX_VERSION,
            "model": embedder.model,
            "model_revision": embedder.model_revision,
            "dimensions": embedder.dimensions,
            "chunks": [chunk.as_dict() for chunk in indexed_chunks],
        }
    )
    embedded_snapshot = _embedded_snapshot(
        snapshot,
        model_revision=embedder.model_revision,
        dimensions=embedder.dimensions,
        index_hash=index_hash,
    )
    index = EmbeddingIndex(
        index_version=INDEX_VERSION,
        snapshot_id=embedded_snapshot.snapshot_id,
        knowledge_version=embedded_snapshot.knowledge_version,
        catalog_version=embedded_snapshot.catalog_version,
        model=embedder.model,
        model_revision=embedder.model_revision,
        dimensions=embedder.dimensions,
        index_hash=index_hash,
        chunks=tuple(indexed_chunks),
    )
    return EmbeddingBuildResult(
        ingest_job_id=job_id,
        base_snapshot_id=snapshot.snapshot_id,
        snapshot=embedded_snapshot,
        index=index,
        operation_usages=tuple(operation_usages),
    )


def publish_index(index: EmbeddingIndex, output_path: str | Path) -> Path:
    """Atomically publish a complete index after all validation has succeeded."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(index.as_dict(), ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".embedding-index-",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with open(descriptor, "w", encoding="utf-8", newline="\n", closefd=True) as handle:
            handle.write(payload)
        Path(temporary_name).replace(destination)
    except Exception:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination


def load_index(path: str | Path) -> EmbeddingIndex:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IndexValidationError(f"cannot load embedding index: {source}") from exc
    if not isinstance(document, Mapping):
        raise IndexValidationError("embedding index root must be an object")
    raw_chunks = document.get("chunks")
    if type(raw_chunks) is not list:
        raise IndexValidationError("embedding index chunks must be a list")
    try:
        chunks = tuple(
            IndexChunk(
                chunk_id=item["chunk_id"],
                source_id=item["source_id"],
                source_version=item["source_version"],
                text=item["text"],
                vector=tuple(item["vector"]),
            )
            for item in raw_chunks
        )
        index = EmbeddingIndex(
            index_version=document["index_version"],
            snapshot_id=document["snapshot_id"],
            knowledge_version=document["knowledge_version"],
            catalog_version=document["catalog_version"],
            model=document["model"],
            model_revision=document["model_revision"],
            dimensions=document["dimensions"],
            index_hash=document["index_hash"],
            chunks=chunks,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IndexValidationError("embedding index fields are invalid") from exc
    expected_hash = _sha256(
        {
            "index_version": index.index_version,
            "model": index.model,
            "model_revision": index.model_revision,
            "dimensions": index.dimensions,
            "chunks": [chunk.as_dict() for chunk in index.chunks],
        }
    )
    if expected_hash != index.index_hash:
        raise IndexValidationError("embedding index hash does not match its vectors")
    return index


def embedded_snapshot_for_index(snapshot: KnowledgeSnapshot, index: EmbeddingIndex) -> KnowledgeSnapshot:
    """The embedded snapshot a loaded index belongs to, recomputed from its content snapshot.

    The id comes from the content, the embedding revision and dimensions, and
    the index hash, exactly as ``build_embedding_index`` computes it; no
    embedding call is made.  A caller compares it with ``index.snapshot_id``.
    """

    if not isinstance(snapshot, KnowledgeSnapshot) or not isinstance(index, EmbeddingIndex):
        raise TypeError("a KnowledgeSnapshot and an EmbeddingIndex are required")
    return _embedded_snapshot(
        snapshot,
        model_revision=index.model_revision,
        dimensions=index.dimensions,
        index_hash=index.index_hash,
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if isinstance(left, (str, bytes)) or isinstance(right, (str, bytes)) or len(left) != len(right):
        raise IndexValidationError("cosine vectors must have the same dimension")
    left_values = _validate_vector(left, dimensions=len(left))
    right_values = _validate_vector(right, dimensions=len(right))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left_values, right_values, strict=True)) / (left_norm * right_norm)


def search_index(
    index: EmbeddingIndex,
    query_vector: Sequence[float],
    *,
    top_k: int = 3,
) -> tuple[IndexHit, ...]:
    if type(top_k) is not int or not 1 <= top_k <= 5:
        raise IndexValidationError("top_k must be between one and five")
    normalized_query = _validate_vector(query_vector, dimensions=index.dimensions)
    ranked = [
        IndexHit(
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            source_version=chunk.source_version,
            text=chunk.text,
            score=cosine_similarity(normalized_query, chunk.vector),
        )
        for chunk in index.chunks
    ]
    ranked.sort(key=lambda hit: (-hit.score, hit.chunk_id))
    return tuple(ranked[:top_k])


__all__ = [
    "EmbeddingAdapter",
    "EmbeddingBuildResult",
    "EmbeddingIndex",
    "INDEX_VERSION",
    "IndexChunk",
    "IndexHit",
    "IndexValidationError",
    "build_embedding_index",
    "cosine_similarity",
    "embedded_snapshot_for_index",
    "load_index",
    "new_ingest_job_id",
    "publish_index",
    "search_index",
]
