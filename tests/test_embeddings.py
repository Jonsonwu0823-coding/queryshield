from __future__ import annotations

from dataclasses import dataclass
import json
import math

import httpx
import pytest

from queryshield.knowledge.index import (
    EmbeddingIndex,
    IndexChunk,
    IndexValidationError,
    build_embedding_index,
    cosine_similarity,
    load_index,
    publish_index,
)
from queryshield.knowledge.ingest import ChunkRecord, KnowledgeSnapshot, SourceRecord
from queryshield.providers.embedding import (
    EmbeddingCallResult,
    EmbeddingConfig,
    EmbeddingProviderError,
    FixedEmbedding,
    OpenAICompatibleEmbedding,
    OperationUsage,
)


def _snapshot() -> KnowledgeSnapshot:
    source = SourceRecord(
        source_id="semantic-test",
        path="shared/test.md",
        version="2026-09-21",
        content_sha256="a" * 64,
        tenant_scope="global",
        allowed_roles=("requester", "approver"),
        status="active",
        updated_at="2026-09-21T00:00:00Z",
    )
    chunks = tuple(
        ChunkRecord(
            chunk_id=f"semantic-test@2026-09-21#{index:04d}",
            source_id="semantic-test",
            source_version="2026-09-21",
            text=text,
            text_sha256=str(index + 1) * 64,
            chunker_version="paragraph-v1",
        )
        for index, text in enumerate(("gross", "net", "refund"))
    )
    return KnowledgeSnapshot(
        snapshot_id="knowledge-v1-base",
        knowledge_version="knowledge-v1",
        catalog_version="catalog-v1",
        chunker_version="paragraph-v1",
        embedding_model_revision=None,
        embedding_dimensions=None,
        index_hash="b" * 64,
        manifest_sha256="c" * 64,
        source_records=(source,),
        chunk_records=chunks,
    )


def _fixed() -> FixedEmbedding:
    return FixedEmbedding(
        {"gross": (1.0, 0.0, 0.0), "net": (0.8, 0.6, 0.0), "refund": (0.0, 1.0, 0.0)},
        model_revision="fixed-v1",
        dimensions=3,
    )


def test_fake_build_records_revision_usage_and_deterministic_cosine_order(tmp_path) -> None:
    result = build_embedding_index(_snapshot(), _fixed(), ingest_job_id="ingest-test", batch_size=2)

    assert result.ingest_job_id == "ingest-test"
    assert result.snapshot.embedding_model_revision == "fixed-v1"
    assert result.snapshot.embedding_dimensions == 3
    assert result.index.snapshot_id == result.snapshot.snapshot_id
    assert len(result.index.chunks) == 3
    assert len(result.operation_usages) == 2
    assert all(usage.operation_kind == "embedding" for usage in result.operation_usages)
    assert all(usage.usage_status == "known" for usage in result.operation_usages)
    assert cosine_similarity((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == pytest.approx(1.0)

    query = (1.0, 0.0, 0.0)
    ranked = sorted(result.index.chunks, key=lambda chunk: (-cosine_similarity(query, chunk.vector), chunk.chunk_id))
    assert [chunk.chunk_id for chunk in ranked] == [
        "semantic-test@2026-09-21#0000",
        "semantic-test@2026-09-21#0001",
        "semantic-test@2026-09-21#0002",
    ]
    path = publish_index(result.index, tmp_path / "index.json")
    loaded = load_index(path)
    assert loaded.index_hash == result.index.index_hash
    assert loaded.chunks == result.index.chunks


def test_invalid_vector_and_mixed_revision_do_not_replace_old_index(tmp_path) -> None:
    result = build_embedding_index(_snapshot(), _fixed(), ingest_job_id="ingest-good")
    path = publish_index(result.index, tmp_path / "current.json")
    old_hash = load_index(path).index_hash

    with pytest.raises(ValueError, match="non-finite"):
        FixedEmbedding({"gross": (math.nan, 0.0, 0.0)}, dimensions=3)

    with pytest.raises(IndexValidationError, match="dimensions changed"):
        build_embedding_index(_snapshot(), _BadDimensionAdapter(), ingest_job_id="ingest-bad")
    assert load_index(path).index_hash == old_hash

    with pytest.raises(IndexValidationError, match="revision changed"):
        build_embedding_index(
            _snapshot(),
            _ChangingRevisionAdapter(),
            ingest_job_id="ingest-mixed",
            batch_size=1,
        )
    assert load_index(path).index_hash == old_hash


def test_real_embedding_adapter_validates_count_dimensions_ids_and_usage() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            request=request,
            headers={"x-request-id": "provider-request-embed-1"},
            json={
                "id": "provider-embed-1",
                "model": "embed-demo",
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0, 0.0]},
                    {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                ],
                "usage": {"total_tokens": 9},
            },
        )

    config = EmbeddingConfig(
        base_url="https://example.test/v1",
        api_key="test-secret",
        model="embed-demo",
        model_revision="embed-demo-2026-09-21",
        dimensions=3,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = OpenAICompatibleEmbedding(config, client=client).embed(
            ["gross", "refund"],
            request_id="local-embed-request",
            model_call_id="local-embed-call",
        )

    assert captured["url"] == "https://example.test/v1/embeddings"
    assert captured["payload"] == {
        "model": "embed-demo",
        "input": ["gross", "refund"],
        "dimensions": 3,
    }
    assert result.vectors == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert result.provider_call_id == "provider-embed-1"
    assert result.provider_request_id == "provider-request-embed-1"
    assert result.usage.total_tokens == 9
    assert result.usage.usage_status == "known"
    assert "test-secret" not in repr(result)


def test_real_adapter_rejects_bad_provider_vector() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "provider-embed-bad",
                "data": [{"index": 0, "embedding": [1.0, 2.0]}],
                "usage": {"total_tokens": 2},
            },
        )

    config = EmbeddingConfig(
        base_url="https://example.test/v1",
        api_key="test-secret",
        model="embed-demo",
        model_revision="embed-demo-2026-09-21",
        dimensions=3,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(EmbeddingProviderError, match="invalid_embedding_vector"):
            OpenAICompatibleEmbedding(config, client=client).embed(["gross"])


@dataclass
class _BadDimensionAdapter:
    model: str = "bad"
    model_revision: str = "bad-v1"
    dimensions: int = 2

    def embed(self, inputs, *, request_id=None, model_call_id=None) -> EmbeddingCallResult:
        call_id = model_call_id or "bad-call"
        usage = OperationUsage(
            operation_kind="embedding",
            model=self.model,
            model_revision=self.model_revision,
            model_call_id=call_id,
            provider_call_id=None,
            provider_request_id=None,
            usage_status="known",
            input_tokens=None,
            output_tokens=None,
            total_tokens=1,
            usage_source="fake",
        )
        return EmbeddingCallResult(
            mode="fake",
            provider="bad",
            model=self.model,
            model_revision=self.model_revision,
            request_id=request_id or "bad-request",
            model_call_id=call_id,
            provider_call_id=None,
            provider_request_id=None,
            inputs_sha256="d" * 64,
            vectors=tuple((1.0, 0.0, 0.0) for _ in inputs),
            dimensions=3,
            usage=usage,
        )


@dataclass
class _ChangingRevisionAdapter:
    model: str = "changing"
    model_revision: str = "changing-v1"
    dimensions: int = 3
    calls: int = 0

    def embed(self, inputs, *, request_id=None, model_call_id=None) -> EmbeddingCallResult:
        self.calls += 1
        revision = "changing-v1" if self.calls == 1 else "changing-v2"
        call_id = model_call_id or f"changing-call-{self.calls}"
        usage = OperationUsage(
            operation_kind="embedding",
            model=self.model,
            model_revision=revision,
            model_call_id=call_id,
            provider_call_id=None,
            provider_request_id=None,
            usage_status="known",
            input_tokens=None,
            output_tokens=None,
            total_tokens=1,
            usage_source="fake",
        )
        return EmbeddingCallResult(
            mode="fake",
            provider="changing",
            model=self.model,
            model_revision=revision,
            request_id=request_id or "changing-request",
            model_call_id=call_id,
            provider_call_id=None,
            provider_request_id=None,
            inputs_sha256="e" * 64,
            vectors=tuple((1.0, 0.0, 0.0) for _ in inputs),
            dimensions=3,
            usage=usage,
        )
