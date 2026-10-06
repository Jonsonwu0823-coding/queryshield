"""Deterministic AGENT-RT01 probe for retrieval and context budgeting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from queryshield.agent import ContextBudgetError, ExecutionContext, build_context
from queryshield.evaluation import load_development_cases, run_development_catalog_retrieval
from queryshield.tools import ControlledTools


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "fixtures" / "semantic" / "retrieval-case-v1.json"
UPSTREAM_PATHS = (
    "src/queryshield/tools/semantic.py",
    "src/queryshield/facts/facts.py",
    "src/queryshield/agent/proposals.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context() -> ExecutionContext:
    return ExecutionContext(
        run_id="run-rt01-context",
        tenant_id="tenant-A",
        principal_id="principal-A",
        role="requester",
    )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    cases = load_development_cases(FIXTURE_PATH)
    tools = ControlledTools()
    context = _context()
    retrieval_records = run_development_catalog_retrieval(tools, context=context, cases=cases)
    _assert(len(retrieval_records) == 12, "the development fixture must run twelve queries")
    _assert(all(record["top_k"] == 3 for record in retrieval_records), "top_k must be fixed at three")
    _assert(all(len(record["items"]) <= 3 for record in retrieval_records), "a result exceeded top_k")
    empty_records = [record for record in retrieval_records if record["empty_hit"]]
    _assert(empty_records, "the no-position query must preserve an empty hit")

    gross_record = next(record for record in retrieval_records if record["family_id"] == "metric-gross")
    normal = build_context(
        context,
        "请查询本租户本窗口营业额",
        confirmed_metric="gross_fen",
        time_window={
            "start": "2026-09-01T00:00:00Z",
            "end": "2026-10-01T00:00:00Z",
            "timezone": "UTC",
            "interval": "[start,end)",
        },
        retrieval_items=gross_record["items"],
        tool_results=[
            {
                "result_id": "result-rt01-gross",
                "row_count": 1,
                "rows": [{"gross_fen": 15000}],
                "policy_version": "qs-sql-v1",
            }
        ],
        optional_summaries=["较旧摘要：用户已经确认使用固定UTC窗口。"],
    )
    _assert(normal.serialized_bytes <= 24_000, "normal context exceeded the byte budget")
    _assert(len(normal.messages) <= 32, "normal context exceeded the message budget")
    _assert(any("tenant-A" in message["content"] for message in normal.messages if message["role"] == "system"), "identity was lost")
    _assert(all(message["role"] != "system" for message in normal.messages[1:]), "data was upgraded to system")
    _assert("result-rt01-gross" in json.dumps(normal.messages, ensure_ascii=False), "tool result was lost")

    oversized = build_context(
        context,
        "请查询营业额",
        optional_summaries=[
            "old-0-" + ("旧摘要。" * 1_000),
            "old-1-" + ("较新摘要。" * 1_000),
        ],
        tool_results=[{"result_id": "result-rt01-new", "row_count": 1}],
    )
    _assert(oversized.serialized_bytes <= 24_000, "trimmed context exceeded the byte budget")
    _assert("summary-0" in oversized.dropped_optional_ids, "oldest optional summary was not dropped first")
    _assert("summary-0" not in oversized.included_optional_ids, "oldest summary was only partially retained")

    hard_items = [
        {
            "id": f"hard-{index}",
            "text": "x" * 7_500,
            "source_id": "commerce-v1",
            "version": "commerce-v1",
        }
        for index in range(3)
    ]
    try:
        build_context(context, "硬约束超预算" + ("q" * 7_990), retrieval_items=hard_items)
    except ContextBudgetError as exc:
        hard_overflow = {"status": "expected_rejection", "code": exc.code}
    else:
        raise AssertionError("hard context overflow must stop before a model call")

    output = {
        "status": "pass",
        "runtime_extension_version": "2026-09-12.runtime-v1",
        "check_id": "AGENT-RT01",
        "profile": "fake-context-budget-v1",
        "mode": "fake",
        "timing_scope": "deterministic in-process; no model/provider/database call",
        "known_usage": 0,
        "unknown_usage_count": 0,
        "input_fixture": {
            "path": FIXTURE_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(FIXTURE_PATH),
            "case_count": len(cases),
        },
        "upstream_manifest": {
            path: _sha256(PROJECT_ROOT / path) for path in UPSTREAM_PATHS
        },
        "normal": {
            "query_count": len(retrieval_records),
            "empty_hit_count": len(empty_records),
            "context": normal.as_dict(),
        },
        "counterexample": {
            "oversized_optional": oversized.as_dict(),
            "hard_overflow": hard_overflow,
            "empty_hit": empty_records[0],
        },
        "database_mode": "not_run",
        "provider_mode": "not_run",
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
