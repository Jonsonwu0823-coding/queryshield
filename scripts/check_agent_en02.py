"""AGENT-EN02 embedding adapter and atomic index probe."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

from queryshield.knowledge.index import build_embedding_index, load_index, publish_index
from queryshield.knowledge.ingest import KnowledgeImportError, load_snapshot, write_snapshot
from queryshield.providers.embedding import (
    EmbeddingCallResult,
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    FixedEmbedding,
    OpenAICompatibleEmbedding,
    OperationUsage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "fixtures"
    / "knowledge"
    / "snapshots"
    / "knowledge-v1-9f580dd7f887ed0a.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixed_adapter(snapshot) -> FixedEmbedding:
    vectors: dict[str, tuple[float, float, float]] = {}
    basis = ((1.0, 0.0, 0.0), (0.8, 0.6, 0.0), (0.0, 1.0, 0.0))
    for index, chunk in enumerate(sorted(snapshot.chunk_records, key=lambda item: item.chunk_id)):
        vectors[chunk.text] = basis[index % len(basis)]
    return FixedEmbedding(vectors, model_revision="fixed-embedding-v1", dimensions=3)


class _WrongDimensionAdapter(FixedEmbedding):
    def embed(self, inputs, *, request_id=None, model_call_id=None) -> EmbeddingCallResult:
        result = super().embed(inputs, request_id=request_id, model_call_id=model_call_id)
        return replace(result, dimensions=result.dimensions + 1)


class _ChangingRevisionAdapter(FixedEmbedding):
    def __init__(self, vectors):
        super().__init__(vectors, model_revision="changing-v1", dimensions=3)
        self._calls = 0

    def embed(self, inputs, *, request_id=None, model_call_id=None) -> EmbeddingCallResult:
        self._calls += 1
        result = super().embed(inputs, request_id=request_id, model_call_id=model_call_id)
        if self._calls == 1:
            return result
        revision = "changing-v2"
        usage = OperationUsage(
            operation_kind="embedding",
            model=result.model,
            model_revision=revision,
            model_call_id=result.model_call_id,
            provider_call_id=None,
            provider_request_id=None,
            usage_status="known",
            input_tokens=None,
            output_tokens=None,
            total_tokens=result.usage.total_tokens,
            usage_source="fake",
        )
        return replace(result, model_revision=revision, usage=usage)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the embedding/index probe")
    parser.add_argument("--mode", choices=("fake", "real"), required=True)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _run_fake(snapshot, output_dir: Path) -> dict[str, object]:
    adapter = _fixed_adapter(snapshot)
    result = build_embedding_index(snapshot, adapter, ingest_job_id="ingest-w03-en02-fake")
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = publish_index(result.index, output_dir / "embedding-index-v1.json")
    snapshot_path = write_snapshot(result.snapshot, output_dir)
    old_hash = load_index(index_path).index_hash

    try:
        FixedEmbedding({"bad": (float("nan"), 0.0, 0.0)}, dimensions=3)
    except ValueError as exc:
        nan_rejection = {"status": "expected_rejection", "error": str(exc)}
    else:
        raise AssertionError("NaN vector was accepted")

    try:
        build_embedding_index(snapshot, _WrongDimensionAdapter({chunk.text: (1.0, 0.0, 0.0) for chunk in snapshot.chunk_records}), ingest_job_id="ingest-w03-en02-dimension")
    except ValueError as exc:
        dimension_rejection = {"status": "expected_rejection", "error": str(exc)}
    else:
        raise AssertionError("dimension mismatch was accepted")

    changing = _ChangingRevisionAdapter(
        {chunk.text: (1.0, 0.0, 0.0) for chunk in snapshot.chunk_records}
    )
    try:
        build_embedding_index(
            snapshot,
            changing,
            ingest_job_id="ingest-w03-en02-revision",
            batch_size=1,
        )
    except ValueError as exc:
        revision_rejection = {"status": "expected_rejection", "error": str(exc)}
    else:
        raise AssertionError("mixed embedding revision was accepted")

    _assert(load_index(index_path).index_hash == old_hash, "old index was not readable after rejected builds")
    return {
        "status": "pass",
        "profile": "fake-embedding-index-v1",
        "ingest_job_id": result.ingest_job_id,
        "base_snapshot_id": result.base_snapshot_id,
        "snapshot_id": result.snapshot.snapshot_id,
        "index_path": str(index_path),
        "snapshot_path": str(snapshot_path),
        "index_hash": result.index.index_hash,
        "chunk_count": len(result.index.chunks),
        "operation_count": len(result.operation_usages),
        "operation_usage": [usage.as_dict() for usage in result.operation_usages],
        "negative_cases": {
            "nan": nan_rejection,
            "dimension": dimension_rejection,
            "mixed_revision": revision_rejection,
            "old_index_hash_after_rejection": load_index(index_path).index_hash,
        },
    }


def _run_real(snapshot, output_dir: Path) -> dict[str, object]:
    adapter = OpenAICompatibleEmbedding.from_env()
    result = build_embedding_index(snapshot, adapter, ingest_job_id="ingest-w03-en02-real")
    if any(usage.usage_status != "known" for usage in result.operation_usages):
        raise AssertionError("real embedding response did not provide known usage")
    if any(not usage.provider_call_id or not usage.provider_request_id for usage in result.operation_usages):
        raise AssertionError("real embedding response did not provide provider IDs")
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = publish_index(result.index, output_dir / "embedding-index-v1.json")
    snapshot_path = write_snapshot(result.snapshot, output_dir)
    return {
        "status": "pass",
        "profile": "real-openai-compatible-embedding-v1",
        "ingest_job_id": result.ingest_job_id,
        "base_snapshot_id": result.base_snapshot_id,
        "snapshot_id": result.snapshot.snapshot_id,
        "index_path": str(index_path),
        "snapshot_path": str(snapshot_path),
        "index_hash": result.index.index_hash,
        "chunk_count": len(result.index.chunks),
        "operation_count": len(result.operation_usages),
        "operation_usage": [usage.as_dict() for usage in result.operation_usages],
    }


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    args = _build_parser().parse_args()
    try:
        snapshot = load_snapshot(args.snapshot)
    except KnowledgeImportError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False))
        return 2

    try:
        result = _run_fake(snapshot, args.output_dir) if args.mode == "fake" else _run_real(snapshot, args.output_dir)
    except EmbeddingConfigurationError as exc:
        print(
            json.dumps(
                {"status": "blocked", "reason": exc.code, "field": exc.field_name},
                ensure_ascii=False,
            )
        )
        return 2
    except (AssertionError, EmbeddingProviderError, ValueError) as exc:
        print(json.dumps({"status": "fail", "reason": str(exc)}, ensure_ascii=False))
        return 1

    output = {
        "check_id": "AGENT-EN02",
        "mode": args.mode,
        "runtime_extension_version": "2026-09-12.runtime-v1",
        "engineering_version": "2026-09-19.engineering-v1",
        "input_snapshot": {
            "path": str(args.snapshot),
            "sha256": _sha256(args.snapshot),
            "snapshot_id": snapshot.snapshot_id,
            "chunk_count": len(snapshot.chunk_records),
        },
        "provider_mode": args.mode,
        "database_mode": "not_run",
        **result,
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
