from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import time

import pytest

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
from queryshield.agent.proposals import ExecutionContext, parse_query_proposal
from queryshield.providers.contracts import ModelCallResult, ModelUsage


def _context(run_id: str = "run-parallel-test") -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        tenant_id="tenant-A",
        principal_id="principal-parallel",
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


def test_parallel_action_is_strict_and_does_not_accept_server_fields() -> None:
    context = _context()
    valid = parse_query_proposal(
        json.dumps(
            {"type": "parallel_readonly", "metric_ids": ["net_fen", "gross_fen", "paid_count"]}
        ),
        context=context,
        model_call_id="local-parallel-action",
    )
    assert isinstance(valid.action, ParallelReadonlyAction)
    assert valid.action.metric_ids == ("net_fen", "gross_fen", "paid_count")

    with pytest.raises(ValueError, match="duplicate_field|invalid_field|unknown_field"):
        parse_query_proposal(
            json.dumps(
                {
                    "type": "parallel_readonly",
                    "metric_ids": ["gross_fen", "gross_fen"],
                    "tenant_id": "tenant-B",
                }
            ),
            context=context,
            model_call_id="local-parallel-invalid",
        )


def test_parallel_scheduler_merges_by_branch_id_and_reuses_committed_results() -> None:
    delays = {"gross_fen": 0.04, "net_fen": 0.01, "paid_count": 0.02}
    calls: list[str] = []

    def runner(context: ExecutionContext, metric_id: str, branch_id: str) -> BranchExecution:
        calls.append(metric_id)
        time.sleep(delays[metric_id])
        values = {"gross_fen": 15000, "net_fen": 12000, "paid_count": 2}
        return BranchExecution(
            metric_id=metric_id,
            result_id=f"result-{metric_id}",
            rows=({metric_id: values[metric_id]},),
            observed_at="2026-09-21T00:00:00Z",
        )

    context = _context()
    scheduler = ParallelScheduler(runner)
    action = ParallelReadonlyAction(("paid_count", "net_fen", "gross_fen"))
    plan = _plan(context)

    result = scheduler.run(context, action, plan=plan)

    assert result.status == "SUCCEEDED"
    assert [branch.metric_id for branch in result.branches] == ["gross_fen", "net_fen", "paid_count"]
    assert result.completion_order != tuple(branch.branch_id for branch in result.branches)
    assert result.peak_active <= 2
    assert result.new_branch_count == 3
    assert {branch.rows[0][branch.metric_id] for branch in result.branches} == {15000, 12000, 2}

    repeated = scheduler.run(context, action, plan=plan)
    assert repeated.reused is True
    assert repeated.new_branch_count == 0
    assert len(calls) == 3

    changed_plan = _plan(context, ("gross_fen", "net_fen"))
    with pytest.raises(ParallelPlanConflict, match="parallel_plan_conflict"):
        scheduler.run(context, ParallelReadonlyAction(("gross_fen", "net_fen")), plan=changed_plan)
    assert len(calls) == 3


def test_unknown_parallel_metric_is_rejected_before_a_branch_starts() -> None:
    calls: list[str] = []

    def runner(context: ExecutionContext, metric_id: str, branch_id: str) -> BranchExecution:
        calls.append(metric_id)
        return BranchExecution(metric_id, f"result-{metric_id}", (), "2026-09-21T00:00:00Z")

    context = _context("run-parallel-invalid")
    scheduler = ParallelScheduler(runner)
    plan = _plan(context)
    with pytest.raises(ParallelValidationError, match="invalid_parallel_action"):
        scheduler.run(context, ("gross_fen", "unknown_metric"), plan=plan)
    assert calls == []


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
    ) -> ModelCallResult:
        self.calls += 1
        if self.calls == 1:
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
            provider="parallel-test",
            model="parallel-test-model",
            request_id=request_id or "request-parallel",
            model_call_id=model_call_id or "local-parallel",
            provider_call_id=f"provider-{self.calls}",
            provider_request_id=f"provider-request-{self.calls}",
            content=content,
            usage=ModelUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            usage_status="known",
        )


def test_bounded_agent_can_execute_parallel_action_without_extra_model_calls() -> None:
    context = _context("run-parallel-graph")
    plan = _plan(context)

    def runner(context: ExecutionContext, metric_id: str, branch_id: str) -> BranchExecution:
        return BranchExecution(metric_id, f"result-{metric_id}", ({metric_id: 1},), "2026-09-21T00:00:00Z")

    model = _ParallelModel()
    runtime = BoundedAgent(
        model,
        call_store=ModelCallStore(),
        parallel_scheduler=ParallelScheduler(runner),
    )

    result = runtime.run(context, "订单总额和净额", parallel_plan=plan)

    assert result.status == "succeeded"
    assert result.model_call_count == 2
    assert result.tool_call_count == 3
    assert model.calls == 2
    assert any(event["kind"] == "parallel_group" for event in result.events)
