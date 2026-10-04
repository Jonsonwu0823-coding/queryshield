"""W03-RT02 deterministic Fake parallel scheduler probe."""

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
    ModelCallStore,
    ParallelPlan,
    ParallelPlanConflict,
    ParallelReadonlyAction,
    ParallelScheduler,
    ParallelValidationError,
)
from queryshield.agent.proposals import ExecutionContext
from queryshield.providers.contracts import ModelCallResult, ModelUsage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_PATHS = (
    "src/queryshield/agent/graph.py",
    "src/queryshield/agent/proposals.py",
    "src/queryshield/agent/context.py",
    "src/queryshield/agent/parallel.py",
)


def _context(run_id: str) -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        tenant_id="tenant-A",
        principal_id="principal-rt02",
        role="requester",
    )


def _plan(context: ExecutionContext, metrics: Sequence[str] = ("gross_fen", "net_fen", "paid_count")) -> ParallelPlan:
    return ParallelPlan.from_context(
        context,
        metrics,
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
        )

    return runner


class _ParallelModel:
    mode = "fake"

    def __init__(self) -> None:
        self.call_count = 0

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        request_id: str | None = None,
        model_call_id: str | None = None,
    ) -> ModelCallResult:
        self.call_count += 1
        if self.call_count == 1:
            content = json.dumps(
                {"type": "parallel_readonly", "metric_ids": ["net_fen", "gross_fen", "paid_count"]},
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
            provider="rt02-scripted",
            model="rt02-scripted-model",
            request_id=request_id or f"request-{self.call_count}",
            model_call_id=model_call_id or f"call-{self.call_count}",
            provider_call_id=f"provider-{self.call_count}",
            provider_request_id=f"provider-request-{self.call_count}",
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
    context = _context("run-rt02-direct")
    calls: list[str] = []
    scheduler = ParallelScheduler(_runner_factory(calls))
    action = ParallelReadonlyAction(("paid_count", "net_fen", "gross_fen"))
    plan = _plan(context)
    result = scheduler.run(context, action, plan=plan)
    branch_metrics = [branch.metric_id for branch in result.branches]
    expected_metrics = ["gross_fen", "net_fen", "paid_count"]
    _assert(result.status == "SUCCEEDED", "three-branch parallel run did not succeed")
    _assert(branch_metrics == expected_metrics, "results were not merged by sorted branch identity")
    _assert(result.peak_active <= 2, "peak active branch count exceeded two")
    _assert(result.completion_order != tuple(branch.branch_id for branch in result.branches), "completion order did not differ")
    _assert(result.new_branch_count == 3, "wrong number of new branch executions")
    _assert(
        {branch.rows[0][branch.metric_id] for branch in result.branches} == {15000, 12000, 2},
        "fixed aggregate oracle did not hold",
    )

    repeated = scheduler.run(context, action, plan=plan)
    _assert(repeated.reused and repeated.new_branch_count == 0, "committed group was queried again")
    _assert(len(calls) == 3, "group re-entry created additional SQL branches")

    changed_plan = _plan(context, ("gross_fen", "net_fen"))
    try:
        scheduler.run(context, ParallelReadonlyAction(("gross_fen", "net_fen")), plan=changed_plan)
    except ParallelPlanConflict as exc:
        conflict = {"status": "expected_rejection", "code": exc.code}
    else:
        raise AssertionError("changed plan was accepted for an existing run")

    invalid_calls: list[str] = []
    invalid_scheduler = ParallelScheduler(_runner_factory(invalid_calls))
    try:
        invalid_scheduler.run(context, ("gross_fen", "unknown_metric"), plan=plan)
    except ParallelValidationError as exc:
        invalid = {"status": "expected_rejection", "code": exc.code}
    else:
        raise AssertionError("unknown metric was not rejected before scheduling")
    _assert(invalid_calls == [], "invalid metric started a branch")

    graph_context = _context("run-rt02-graph")
    graph_plan = _plan(graph_context)
    graph_model = _ParallelModel()
    graph_calls: list[str] = []
    graph_runtime = BoundedAgent(
        graph_model,
        call_store=ModelCallStore(),
        parallel_scheduler=ParallelScheduler(_runner_factory(graph_calls)),
    )
    graph_result = graph_runtime.run(
        graph_context,
        "2026年9月订单总额和净额",
        parallel_plan=graph_plan,
    )
    parallel_events = [event for event in graph_result.events if event["kind"] == "parallel_group"]
    _assert(graph_result.status == "succeeded", "LangGraph parallel action did not finish")
    _assert(graph_result.model_call_count == 2, "parallel graph used an unexpected model count")
    _assert(graph_result.tool_call_count == 3, "parallel branches were not counted as tools")
    _assert(len(parallel_events) == 1 and parallel_events[0]["peak_active"] <= 2, "parallel event is incomplete")

    return {
        "input": {
            "type": "parallel_readonly",
            "metric_ids": list(action.metric_ids),
            "plan_hash": plan.plan_hash,
        },
        "direct": {
            "status": result.status,
            "branch_metrics": branch_metrics,
            "branch_ids": [branch.branch_id for branch in result.branches],
            "completion_order": list(result.completion_order),
            "peak_active": result.peak_active,
            "new_branch_count": result.new_branch_count,
            "reentry": repeated.as_dict(),
            "calls_after_reentry": len(calls),
        },
        "pre_execution_rejections": {"plan_conflict": conflict, "unknown_metric": invalid},
        "graph": {
            "status": graph_result.status,
            "model_call_count": graph_result.model_call_count,
            "tool_call_count": graph_result.tool_call_count,
            "parallel_event_count": len(parallel_events),
            "provider_observed_branch_count": len(graph_calls),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run W03-RT02 fake parallel scheduling")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    actual_command = " ".join([sys.executable, *sys.argv])
    probe = _run_probe()
    output = {
        "status": "pass",
        "runtime_extension_version": "2026-09-12.runtime-v1",
        "check_id": "W03-RT02",
        "upstream_manifest": {
            path: _sha256(PROJECT_ROOT / path) for path in UPSTREAM_PATHS
        },
        "actual": {
            "cwd": str(Path.cwd()),
            "command": actual_command,
            "exit_code": 0,
        },
        "mode": "fake",
        "profile": "fake-qs-parallel-v1",
        "fixture": {
            "metrics": ["paid_count", "gross_fen", "net_fen"],
            "expected_values": {"gross_fen": 15000, "net_fen": 12000, "paid_count": 2},
            "max_active": 2,
        },
        "probe": probe,
        "raw_evidence_paths": [],
        "database_mode": "not_run; injected deterministic branch runner",
        "provider_mode": "not_run; no model call is needed for direct scheduler",
        "implementation_result": "candidate_self_checked",
        "learner_result": "pending",
        "hint_level": "H3",
        "next_action": "W03-U03",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / "W03-RT02.json"
    output["raw_evidence_paths"] = [evidence_path.as_posix()]
    evidence_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

