from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import threading
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
from queryshield.agent.graph import _new_state
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
        run_id: str | None = None,
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


def _never_runs(context: ExecutionContext, metric_id: str, branch_id: str) -> BranchExecution:
    raise AssertionError("no branch may start")


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        pytest.param(lambda context: ({"run_id": context.run_id}, _plan(context), None), "unauthorized", id="context_not_server_made"),
        pytest.param(lambda context: (context, {"metric_ids": ["gross_fen"]}, None), "invalid_parallel_plan", id="plan_not_server_made"),
        pytest.param(
            lambda context: (context, _plan(ExecutionContext(context.run_id, context.tenant_id, "someone-else", context.role)), None),
            "parallel_plan_conflict",
            id="plan_of_another_principal",
        ),
        pytest.param(
            lambda context: (context, _plan(ExecutionContext(context.run_id, context.tenant_id, context.principal_id, "approver")), None),
            "parallel_plan_conflict",
            id="plan_of_another_role",
        ),
        pytest.param(lambda context: (context, _plan(context), -1), "invalid_parallel_budget", id="negative_budget"),
        pytest.param(lambda context: (context, _plan(context), "3"), "invalid_parallel_budget", id="budget_not_an_int"),
    ],
)
def test_a_parallel_run_is_refused_before_any_branch_when_its_inputs_are_not_the_servers(arguments, code) -> None:
    context = _context("run-parallel-refused")
    run_context, plan, budget = arguments(context)

    with pytest.raises(ParallelValidationError) as error:
        ParallelScheduler(_never_runs).run(run_context, ("gross_fen", "net_fen", "paid_count"), plan=plan, tool_budget_remaining=budget)

    assert error.value.code == code


@pytest.mark.parametrize(
    ("execution", "code"),
    [
        pytest.param(lambda metric_id: {"metric_id": metric_id}, "invalid_branch_result", id="not_a_branch_execution"),
        pytest.param(lambda metric_id: BranchExecution("other_metric", "result-x", (), "2026-09-21T00:00:00Z"), "invalid_branch_result", id="wrong_metric"),
        pytest.param(lambda metric_id: BranchExecution(metric_id, "result-x", (5,), "2026-09-21T00:00:00Z"), "branch_failed", id="row_not_a_mapping"),
    ],
)
def test_a_branch_whose_runner_returns_something_else_fails_with_its_code(execution, code) -> None:
    context = _context("run-parallel-bad-branch")

    result = ParallelScheduler(lambda _context, metric_id, _branch_id: execution(metric_id)).run(
        context, ("gross_fen", "net_fen"), plan=_plan(context, ("gross_fen", "net_fen"))
    )

    assert result.status == "FAILED"
    assert [(branch.status, branch.error_code, branch.rows) for branch in result.branches] == [("FAILED", code, ())] * 2


def _failing_branch(context: ExecutionContext, metric_id: str, branch_id: str) -> BranchExecution:
    raise RuntimeError("branch down")


def test_a_failed_branch_fails_the_run_with_the_parallel_branch_code() -> None:
    context = _context("run-parallel-branch-down")
    runtime = BoundedAgent(_ParallelModel(), call_store=ModelCallStore(), parallel_scheduler=ParallelScheduler(_failing_branch))

    result = runtime.run(context, "订单总额和净额", parallel_plan=_plan(context))

    assert (result.status, result.error_code, result.tool_call_count, result.model_call_count) == ("failed", "parallel_branch_failed", 3, 1)


def test_a_plan_the_scheduler_refuses_fails_the_run_with_its_code() -> None:
    context = _context("run-parallel-foreign-plan")
    foreign = _plan(ExecutionContext(context.run_id, context.tenant_id, "someone-else", context.role))
    runtime = BoundedAgent(_ParallelModel(), call_store=ModelCallStore(), parallel_scheduler=ParallelScheduler(_never_runs))

    result = runtime.run(context, "订单总额和净额", parallel_plan=foreign)

    assert (result.status, result.error_code, result.tool_call_count) == ("failed", "parallel_plan_conflict", 0)


def test_the_parallel_node_without_a_parallel_proposal_fails_the_run() -> None:
    runtime = BoundedAgent(_ParallelModel(), call_store=ModelCallStore(), parallel_scheduler=ParallelScheduler(_never_runs))
    state = _new_state(_context("run-parallel-no-proposal"), "订单总额和净额", runtime.run_config, started_at=0.0)

    update = runtime._execute_parallel(state)

    assert (update["status"], update["error_code"]) == ("failed", "invalid_parallel_transition")


def test_two_branches_run_at_once_and_the_group_records_it() -> None:
    together = threading.Barrier(2, timeout=5)

    def runner(context: ExecutionContext, metric_id: str, branch_id: str) -> BranchExecution:
        together.wait()  # each branch waits for the other: both are running
        return BranchExecution(metric_id, f"result-{metric_id}", ({metric_id: 1},), "2026-09-21T00:00:00Z")

    context = _context("run-parallel-together")
    result = ParallelScheduler(runner).run(context, ("gross_fen", "net_fen"), plan=_plan(context, ("gross_fen", "net_fen")))

    assert result.status == "SUCCEEDED" and result.peak_active == 2
