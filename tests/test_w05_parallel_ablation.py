from __future__ import annotations

import pytest

from queryshield.agent.parallel_durable import DurableParallelScheduler
from queryshield.agent.proposals import ExecutionContext
from queryshield.approval.service import FixtureQueryExecutor
from queryshield.db.w04_state import StateStore


@pytest.mark.parametrize(("max_active", "expected_peak"), ((1, 1), (2, 2)))
def test_durable_parallel_scheduler_allows_only_bounded_w05_concurrency(
    max_active: int,
    expected_peak: int,
) -> None:
    state = StateStore(":memory:")
    context = ExecutionContext(
        run_id=f"w05-parallel-cap-{max_active}",
        tenant_id="A",
        principal_id="principal-A",
        role="requester",
    )
    state.create_run(
        run_id=context.run_id,
        tenant_id=context.tenant_id,
        principal_id=context.principal_id,
        role=context.role,
        question="gross and net",
        mode="fake",
    )
    scheduler = DurableParallelScheduler(
        state=state,
        executor_factory=FixtureQueryExecutor,
        max_active_branches=max_active,
    )

    result = scheduler.run(
        context,
        ("gross_fen", "net_fen"),
        time_window={
            "start": "2026-09-01T00:00:00Z",
            "end": "2026-10-01T00:00:00Z",
            "timezone": "UTC",
        },
    )

    assert result.status == "SUCCEEDED"
    assert result.peak_active <= expected_peak
    assert len(result.branches) == 2
    state.close()


def test_durable_parallel_scheduler_rejects_concurrency_above_contract() -> None:
    state = StateStore(":memory:")
    try:
        with pytest.raises(ValueError, match="max_active_branches"):
            DurableParallelScheduler(state=state, max_active_branches=3)
    finally:
        state.close()
