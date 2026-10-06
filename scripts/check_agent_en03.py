"""AGENT-EN03 hybrid retrieval and tenant-bound evidence probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from queryshield.agent.proposals import ExecutionContext
from queryshield.catalog import load_default_catalog
from queryshield.knowledge.index import build_embedding_index, publish_index
from queryshield.knowledge.ingest import KnowledgeImportError, load_snapshot, write_snapshot
from queryshield.knowledge.retrieval import HybridRetriever
from queryshield.providers.embedding import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    FixedEmbedding,
    OpenAICompatibleEmbedding,
)
from queryshield.tools import ControlledTools


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "fixtures"
    / "knowledge"
    / "snapshots"
    / "knowledge-v1-9f580dd7f887ed0a.json"
)


def _fixed_adapter(snapshot) -> FixedEmbedding:
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
        }
    )
    return FixedEmbedding(vectors, model_revision="fixed-hybrid-v1", dimensions=3)


def _latest_evidence(tools: ControlledTools, context: ExecutionContext):
    matching = [
        (key, value)
        for key, value in tools._retrieval_evidence.items()
        if key[0] == context.run_id
    ]
    if not matching:
        raise AssertionError("retrieval evidence was not recorded")
    return matching[-1][1]


def _context(*, run_id: str, tenant_id: str = "tenant-A") -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        tenant_id=tenant_id,
        principal_id="principal-en03",
        role="requester",
    )


def _run_cases(tools: ControlledTools) -> dict[str, object]:
    metric_context = _context(run_id="run-en03-metric")
    first = tools.search_catalog({"query": "营业额", "top_k": 3}, context=metric_context)
    first_evidence = _latest_evidence(tools, metric_context)
    second = tools.search_catalog({"query": "营业额", "top_k": 3}, context=metric_context)
    second_evidence = _latest_evidence(tools, metric_context)
    _assert(first_evidence.selected_ids == second_evidence.selected_ids, "RRF order is not deterministic")
    _assert(first_evidence.strategy_version == "hybrid-v1", "wrong retrieval strategy version")
    _assert(first_evidence.embedding_call_id, "embedding call ID is missing")
    _assert(first_evidence.embedding_actual_return is not None, "embedding actual return is missing")

    tenant_context = _context(run_id="run-en03-tenant")
    tenant_result = tools.search_catalog({"query": "订单", "top_k": 5}, context=tenant_context)
    tenant_evidence = _latest_evidence(tools, tenant_context)
    _assert(
        all("tenant-b" not in str(item).lower() for item in tenant_result["items"]),
        "tenant A retrieval exposed a tenant B item",
    )
    _assert(
        all("tenant-b" not in item.lower() for item in tenant_evidence.visible_candidate_ids),
        "tenant B candidate survived pre-retrieval authorization filtering",
    )

    empty_context = _context(run_id="run-en03-empty")
    empty_result = tools.search_catalog({"query": "__unknown__", "top_k": 3}, context=empty_context)
    empty_evidence = _latest_evidence(tools, empty_context)
    _assert(empty_result == {"items": []}, "unknown query did not produce an explicit empty result")
    _assert(empty_evidence.selected_ids == (), "empty query unexpectedly selected a source")

    return {
        "metric": {
            "selected_ids": list(first_evidence.selected_ids),
            "items": first["items"],
            "evidence": first_evidence.as_dict(),
        },
        "metric_repeat": {"selected_ids": list(second_evidence.selected_ids), "items": second["items"]},
        "tenant_a": {
            "selected_ids": list(tenant_evidence.selected_ids),
            "evidence": tenant_evidence.as_dict(),
        },
        "empty": {
            "items": empty_result["items"],
            "evidence": empty_evidence.as_dict(),
        },
    }


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hybrid retrieval probe")
    parser.add_argument("--mode", choices=("fake", "real"), required=True)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        snapshot = load_snapshot(args.snapshot)
        embedder = _fixed_adapter(snapshot) if args.mode == "fake" else OpenAICompatibleEmbedding.from_env()
        build = build_embedding_index(
            snapshot,
            embedder,
            ingest_job_id=f"ingest-w03-en03-{args.mode}",
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        index_path = publish_index(build.index, args.output_dir / "embedding-index-v1.json")
        snapshot_path = write_snapshot(build.snapshot, args.output_dir)
        tools = ControlledTools(
            catalog=load_default_catalog(),
            retriever=HybridRetriever(
                catalog=load_default_catalog(),
                snapshot=build.snapshot,
                index=build.index,
                embedder=embedder,
            ),
        )
        cases = _run_cases(tools)
        if args.mode == "real":
            _assert(
                all(usage.usage_status == "known" for usage in build.operation_usages),
                "real index build usage is not known",
            )
            for case in cases.values():
                if not isinstance(case, dict):
                    continue
                evidence = case.get("evidence")
                if isinstance(evidence, dict):
                    _assert(evidence.get("embedding_provider_call_id"), "real provider call ID is missing")
                    _assert(evidence.get("embedding_provider_request_id"), "real provider request ID is missing")
        output = {
            "check_id": "AGENT-EN03",
            "mode": args.mode,
            "engineering_version": "2026-09-19.engineering-v1",
            "profile": f"{args.mode}-hybrid-v1",
            "snapshot_id": build.snapshot.snapshot_id,
            "index_path": str(index_path),
            "snapshot_path": str(snapshot_path),
            "index_hash": build.index.index_hash,
            "chunk_count": len(build.index.chunks),
            "index_operation_usage": [usage.as_dict() for usage in build.operation_usages],
            "cases": cases,
        }
        print(json.dumps({"status": "pass", **output}, ensure_ascii=False, sort_keys=True))
        return 0
    except KnowledgeImportError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False))
        return 2
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


if __name__ == "__main__":
    raise SystemExit(main())
