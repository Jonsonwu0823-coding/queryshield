"""AGENT-RT03 shared-budget and visible-branch-event Fake probe."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

from queryshield.agent import (
    BranchExecution,
    BoundedAgent,
    GraphLimits,
    ModelCallStore,
    ParallelPlan,
    ParallelReadonlyAction,
    ParallelScheduler,
)
from queryshield.agent.proposals import ExecutionContext
from queryshield.providers.contracts import ModelCallResult, ModelUsage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_PATHS = (
    "src/queryshield/agent/graph.py",
    "src/queryshield/agent/parallel.py",
    "src/queryshield/agent/proposals.py",
)


def _context(run_id: str) -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        tenant_id="tenant-A",
        principal_id="principal-rt03",
        role="requester",
    )


def _plan(context: ExecutionContext) -> ParallelPlan:
    return ParallelPlan.from_context(
        context,
        ("gross_fen", "net_fen", "paid_count"),
        time_window={
            "start": "2026-09-01T00:00:00Z",
            "end": "2026-10-01T00:00:00Z",
            "timezone": "UTC",
            "interval": "[start,end)",
        },
    )


def _runner_factory(calls: list[str]):
    delays = {"gross_fen": 0.04, "net_fen": 0.01, "paid_count": 0.02}
    values = {"gross_fen": 15000, "net_fen": 12000, "paid_count": 2}

    def runner(context: ExecutionContext, metric_id: str, branch_id: str) -> BranchExecution:
        calls.append(metric_id)
        time.sleep(delays[metric_id])
        return BranchExecution(
            metric_id=metric_id,
            result_id=f"result-{metric_id}",
            rows=({metric_id: values[metric_id]},),
            observed_at="2026-09-21T00:00:00Z",
            elapsed_ms=int(delays[metric_id] * 1000),
        )

    return runner


class _ParallelModel:
    mode = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        request_id: str | None = None,
        model_call_id: str | None = None,
        run_id: str | None = None,
    ) -> ModelCallResult:
        self.calls += 1
        if self.calls == 1:
            content = json.dumps(
                {"type": "parallel_readonly", "metric_ids": ["gross_fen", "net_fen", "paid_count"]},
                separators=(",", ":"),
            )
        else:
            content = json.dumps(
                {"type": "final_answer", "answer": "并行结果已汇总。", "source_ids": [], "fact_refs": []},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return ModelCallResult(
            mode="fake",
            provider="rt03-scripted",
            model="rt03-scripted-model",
            request_id=request_id or f"request-{self.calls}",
            model_call_id=model_call_id or f"call-{self.calls}",
            provider_call_id=f"provider-{self.calls}",
            provider_request_id=f"provider-request-{self.calls}",
            content=content,
            usage=ModelUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            usage_status="known",
        )


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run_probe() -> dict[str, object]:
    metrics = ("gross_fen", "net_fen", "paid_count")

    insufficient_context = _context("run-rt03-budget-two")
    insufficient_calls: list[str] = []
    insufficient_scheduler = ParallelScheduler(_runner_factory(insufficient_calls))
    insufficient = insufficient_scheduler.run(
        insufficient_context,
        ParallelReadonlyAction(metrics),
        plan=_plan(insufficient_context),
        tool_budget_remaining=2,
    )
    _assert(insufficient.status == "LIMIT_REACHED", "budget two did not stop the three-branch plan")
    _assert(insufficient.new_branch_count == 0, "a branch started with insufficient shared budget")
    _assert(insufficient.peak_active == 0, "insufficient budget had an active branch")
    _assert(insufficient_calls == [], "insufficient budget called a branch runner")
    _assert(all(branch.status == "PENDING" for branch in insufficient.branches), "pending branch state is not visible")

    sufficient_context = _context("run-rt03-budget-three")
    sufficient_calls: list[str] = []
    sufficient_scheduler = ParallelScheduler(_runner_factory(sufficient_calls))
    started = time.perf_counter()
    sufficient = sufficient_scheduler.run(
        sufficient_context,
        ParallelReadonlyAction(metrics),
        plan=_plan(sufficient_context),
        tool_budget_remaining=3,
    )
    wall_elapsed_ms = int((time.perf_counter() - started) * 1000)
    branch_elapsed_sum = sum(branch.elapsed_ms for branch in sufficient.branches)
    _assert(sufficient.status == "SUCCEEDED", "budget three did not start all branches")
    _assert(sufficient.new_branch_count == 3, "budget three started the wrong number of branches")
    _assert(len(sufficient_calls) == 3, "budget three did not count branches in the shared run")
    _assert(sufficient.peak_active <= 2, "parallel peak exceeded two")
    _assert(wall_elapsed_ms < branch_elapsed_sum, "overlapping branch wall time was added per branch")

    graph_context = _context("run-rt03-graph-budget-two")
    graph_calls: list[str] = []
    graph_model = _ParallelModel()
    graph_runtime = BoundedAgent(
        graph_model,
        call_store=ModelCallStore(),
        limits=GraphLimits(max_model_calls=6, max_tool_calls=2),
        parallel_scheduler=ParallelScheduler(_runner_factory(graph_calls)),
    )
    graph_result = graph_runtime.run(
        graph_context,
        "订单总额和净额",
        parallel_plan=_plan(graph_context),
    )
    visible_events = [event for event in graph_result.events if event["kind"] == "parallel_group"]
    _assert(graph_result.status == "limit_reached", "graph did not expose shared budget limit")
    _assert(graph_result.tool_call_count == 0, "graph counted unstarted branches")
    _assert(graph_calls == [], "graph launched a branch after shared budget rejection")
    _assert(len(visible_events) == 1, "shared budget event is not visible")
    _assert(set(visible_events[0]["branch_statuses"]) == {"PENDING"}, "pending branch event is incomplete")

    return {
        "metric_ids": list(metrics),
        "insufficient_budget": {
            "remaining_tool_budget": 2,
            "status": insufficient.status,
            "error_code": insufficient.error_code,
            "new_branch_count": insufficient.new_branch_count,
            "peak_active": insufficient.peak_active,
            "branch_statuses": [branch.status for branch in insufficient.branches],
            "runner_calls": len(insufficient_calls),
        },
        "sufficient_budget": {
            "remaining_tool_budget": 3,
            "status": sufficient.status,
            "new_branch_count": sufficient.new_branch_count,
            "peak_active": sufficient.peak_active,
            "runner_calls": len(sufficient_calls),
            "wall_elapsed_ms": wall_elapsed_ms,
            "branch_elapsed_sum_ms": branch_elapsed_sum,
        },
        "graph_visible_limit": {
            "status": graph_result.status,
            "model_call_count": graph_result.model_call_count,
            "tool_call_count": graph_result.tool_call_count,
            "parallel_event": dict(visible_events[0]),
            "runner_calls": len(graph_calls),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AGENT-RT03 shared parallel budget probe")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    output = {
        "status": "pass",
        "runtime_extension_version": "2026-09-12.runtime-v1",
        "check_id": "AGENT-RT03",
        "upstream_manifest": {
            path: _sha256(PROJECT_ROOT / path) for path in UPSTREAM_PATHS
        },
        "actual": {
            "cwd": str(Path.cwd()),
            "command": " ".join([sys.executable, *sys.argv]),
            "exit_code": 0,
        },
        "mode": "fake",
        "profile": "fake-shared-parallel-budget-v1",
        "fixture": {
            "remaining_budgets": [2, 3],
            "branch_limit": 2,
            "active_wall_clock_is_not_additive": True,
        },
        "probe": _run_probe(),
        "raw_evidence_paths": [],
        "database_mode": "not_run; injected deterministic branch runner",
        "provider_mode": "not_run; no real provider call",
        "implementation_result": "candidate_self_checked",
        "learner_result": "pending",
        "hint_level": "H3",
        "next_action": "AGENT-T04",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / "AGENT-RT03.json"
    output["raw_evidence_paths"] = [evidence_path.as_posix()]
    evidence_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

