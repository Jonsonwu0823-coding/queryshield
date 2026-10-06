"""AGENT-EN04 versioned graph/context and wording-agnostic fake probe."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from queryshield.agent import (
    BoundedAgent,
    ContextBudgetError,
    ModelCallStore,
    RunConfig,
    build_context,
)
from queryshield.agent.proposals import ExecutionContext
from queryshield.catalog import load_default_catalog
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.knowledge.index import build_embedding_index
from queryshield.knowledge.ingest import load_snapshot
from queryshield.knowledge.retrieval import HybridRetriever
from queryshield.providers.contracts import ModelCallResult, ModelUsage
from queryshield.providers.embedding import FixedEmbedding
from queryshield.tools import ControlledTools


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "fixtures"
    / "knowledge"
    / "snapshots"
    / "knowledge-v1-9f580dd7f887ed0a.json"
)


class _FakeCursor:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        return None

    def fetchmany(self, size: int) -> list[dict[str, object]]:
        return self.rows[:size]


class _FakeConnection:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.cursor_instance = _FakeCursor(rows)

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def cursor(self, *, row_factory: object) -> _FakeCursor:
        return self.cursor_instance


class _ScriptedModel:
    mode = "fake"

    def __init__(self, search_query: str, *, loop: bool = False, invalid_config: bool = False) -> None:
        self.search_query = search_query
        self.loop = loop
        self.invalid_config = invalid_config
        self.model_call_ids: list[str] = []
        self.questions: list[str] = []

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        request_id: str | None = None,
        model_call_id: str | None = None,
    ) -> ModelCallResult:
        if request_id is None or model_call_id is None:
            raise AssertionError("server identities were not passed to the model")
        index = len(self.model_call_ids)
        self.model_call_ids.append(model_call_id)
        self.questions.append(messages[1]["content"])
        if self.loop or index < 2:
            content = _tool_call("search_catalog", {"query": self.search_query, "top_k": 3})
        else:
            content = _final_answer()
        if self.invalid_config:
            payload = json.loads(content)
            payload["profile"] = "model-selected-profile"
            content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return ModelCallResult(
            mode="fake",
            provider="scripted-en04",
            model="scripted-en04-model",
            request_id=request_id,
            model_call_id=model_call_id,
            provider_call_id=f"provider-call-{index + 1}",
            provider_request_id=f"provider-request-{index + 1}",
            content=content,
            usage=ModelUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            usage_status="known",
        )


def _context(run_id: str) -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        tenant_id="tenant-A",
        principal_id="principal-en04",
        role="requester",
    )


def _tool_call(name: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "name": name, "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _final_answer() -> str:
    return json.dumps(
        {
            "type": "final_answer",
            "answer": "受控查询已返回结果。",
            "source_ids": ["commerce-v1"],
            "fact_refs": [],
            "basis": "knowledge",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _fixed_embedder(snapshot) -> FixedEmbedding:
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
            "支付订单总额": (1.0, 0.0, 0.0),
            "订单": (0.0, 0.0, 1.0),
        }
    )
    return FixedEmbedding(vectors, model_revision="fixed-en04-v1", dimensions=3)


def _runtime(*, model: _ScriptedModel, config: RunConfig, snapshot, embedder) -> tuple[BoundedAgent, ControlledTools]:
    connection = _FakeConnection([{"gross_fen": 3000}])
    executor = GuardedQueryExecutor(
        connect=lambda: connection,
        clock=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc),
    )
    build = build_embedding_index(snapshot, embedder, ingest_job_id="ingest-w03-en04-fake")
    index = build.index
    catalog = load_default_catalog()
    tools = ControlledTools(
        catalog=catalog,
        executor=executor,
        retriever=HybridRetriever(
            catalog=catalog,
            snapshot=build.snapshot,
            index=index,
            embedder=embedder,
        ),
    )
    return BoundedAgent(model, tools=tools, call_store=ModelCallStore(), run_config=config), tools


def _latest_retrieval(tools: ControlledTools, run_id: str):
    matching = [value for (stored_run_id, _), value in tools._retrieval_evidence.items() if stored_run_id == run_id]
    if not matching:
        raise AssertionError("hybrid retrieval evidence was not recorded")
    return matching[-1]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _wording_cases(config: RunConfig, snapshot, embedder) -> list[dict[str, object]]:
    cases = (
        ("能否帮我看九月付费订单的营业额？", "营业额"),
        ("我只想知道已经付款的订单总金额。", "支付订单总额"),
    )
    records: list[dict[str, object]] = []
    for index, (question, search_query) in enumerate(cases):
        model = _ScriptedModel(search_query)
        runtime, tools = _runtime(model=model, config=config, snapshot=snapshot, embedder=embedder)
        result = runtime.run(_context(f"run-en04-wording-{index}"), question)
        evidence = _latest_retrieval(tools, result.run_id)
        _assert(result.status == "succeeded", "different wording did not reach an answer")
        _assert(result.model_call_count == 3, "wording case did not use the normal three model rounds")
        _assert(result.tool_call_count == 2, "wording case did not use retrieval and query slots")
        _assert(evidence.strategy_version == "hybrid-v1", "A03 hybrid retriever was not used")
        _assert(model.questions[0] == question, "the graph did not pass the original question to the model")
        _assert(result.run_config.as_dict() == config.as_dict(), "run config changed during the run")
        records.append(
            {
                "question_sha256": sha256(question.encode("utf-8")).hexdigest(),
                "status": result.status,
                "model_call_count": result.model_call_count,
                "tool_call_count": result.tool_call_count,
                "retrieval_strategy": evidence.strategy_version,
                "selected_count": len(evidence.selected_ids),
                "model_call_ids_unique": len(set(result.model_call_ids)) == 3,
            }
        )
    return records


def _hard_budget_case(config: RunConfig) -> dict[str, object]:
    hard_items = [
        {
            "id": f"hard-{index}",
            "text": "x" * 7_500,
            "source_id": "tenant-b-doc",
            "version": "v1",
        }
        for index in range(3)
    ]
    try:
        build_context(
            _context("run-en04-hard-budget"),
            "硬约束超预算" + ("q" * 7_990),
            retrieval_items=hard_items,
            run_config=config,
        )
    except ContextBudgetError as exc:
        return {"status": "expected_rejection", "code": exc.code}
    raise AssertionError("hard context over 24000 bytes was not rejected")


def _model_config_case(config: RunConfig, snapshot, embedder) -> dict[str, object]:
    model = _ScriptedModel("营业额", invalid_config=True)
    runtime, _ = _runtime(model=model, config=config, snapshot=snapshot, embedder=embedder)
    result = runtime.run(_context("run-en04-model-config"), "配置字段不应由模型决定")
    _assert(result.status == "failed", "model-selected run config was accepted")
    _assert(result.error_code == "unknown_field", "wrong model config rejection code")
    _assert(result.tool_call_count == 0, "invalid config proposal reached a tool")
    return {
        "status": result.status,
        "error_code": result.error_code,
        "tool_call_count": result.tool_call_count,
    }


def _loop_case(config: RunConfig, snapshot, embedder) -> dict[str, object]:
    model = _ScriptedModel("营业额", loop=True)
    runtime, _ = _runtime(model=model, config=config, snapshot=snapshot, embedder=embedder)
    result = runtime.run(_context("run-en04-loop"), "循环预算检查")
    model_events = [event for event in result.events if event["kind"] == "model_call"]
    tool_events = [event for event in result.events if event["kind"] == "tool_call"]
    _assert(result.status == "limit_reached", "loop did not stop at the model budget")
    _assert(result.error_code == "model_call_limit", "loop stopped for the wrong reason")
    _assert(len(model_events) == result.model_call_count == 6, "model event count is inconsistent")
    _assert(len(tool_events) == result.tool_call_count == 6, "tool event count is inconsistent")
    _assert(len(model.model_call_ids) == 6, "seventh model call was observed")
    return {
        "status": result.status,
        "error_code": result.error_code,
        "model_call_count": result.model_call_count,
        "tool_call_count": result.tool_call_count,
        "model_event_count": len(model_events),
        "tool_event_count": len(tool_events),
        "provider_observed_call_count": len(model.model_call_ids),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the versioned graph/context probe")
    parser.add_argument("--mode", choices=("fake",), required=True)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    snapshot = load_snapshot(args.snapshot)
    embedder = _fixed_embedder(snapshot)
    embedded_snapshot = build_embedding_index(
        snapshot,
        embedder,
        ingest_job_id="ingest-w03-en04-fake",
    ).snapshot
    config = RunConfig(
        profile="queryshield-w03-en04-fake-v1",
        catalog_version=embedded_snapshot.catalog_version,
        knowledge_snapshot_id=embedded_snapshot.snapshot_id,
        model_version="scripted-en04-model-v1",
        adapter_version="fake-adapter-v1",
    )
    wording = _wording_cases(config, snapshot, embedder)
    hard_budget = _hard_budget_case(config)
    model_config = _model_config_case(config, snapshot, embedder)
    loop = _loop_case(config, snapshot, embedder)
    output = {
        "status": "pass",
        "check_id": "AGENT-EN04",
        "mode": args.mode,
        "engineering_version": "2026-09-19.engineering-v1",
        "profile": "fake-versioned-graph-context-v1",
        "run_config": config.as_dict(),
        "snapshot_id": snapshot.snapshot_id,
        "wording_cases": wording,
        "hard_budget": hard_budget,
        "model_config_rejection": model_config,
        "loop": loop,
        "database_mode": "fake_connection_in_process",
        "provider_mode": "fake_fixed_embedding_and_scripted_model",
        "real_mode": "not_applicable; EN04 requires fake mode",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / "AGENT-EN04.json"
    evidence_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({**output, "evidence_path": evidence_path.as_posix()}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
