"""W04 durable bounded read-only parallel execution and recovery boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from threading import Lock
from uuid import uuid4

from queryshield.agent.context import NET_FEN_PLAN_ID, NET_FEN_TIME_WINDOW
from queryshield.agent.proposals import ExecutionContext, MetricBinding
from queryshield.approval.service import FixtureQueryExecutor
from queryshield.catalog.catalog import DEFAULT_CATALOG_VERSION
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.db.w04_state import StateStore, StateStoreError, utc_now
from queryshield.tools import ControlledTools, ToolError


PARALLEL_RUNTIME_VERSION = "qs-parallel-runtime-v1"
METRICS = frozenset({"paid_count", "gross_fen", "net_fen"})
MAX_ACTIVE_BRANCHES = 2


class DurableParallelError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class DurableParallelResult:
    run_id: str
    group_id: str
    plan_hash: str
    status: str
    branches: tuple[Mapping[str, object], ...]
    reused: bool
    peak_active: int
    new_branch_count: int
    sql_exec_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "runtime_version": PARALLEL_RUNTIME_VERSION,
            "run_id": self.run_id,
            "group_id": self.group_id,
            "plan_hash": self.plan_hash,
            "status": self.status,
            "branches": [dict(branch) for branch in self.branches],
            "reused": self.reused,
            "peak_active": self.peak_active,
            "new_branch_count": self.new_branch_count,
            "sql_exec_count": self.sql_exec_count,
        }


def _binding(metric_id: str, time_window: Mapping[str, str]) -> MetricBinding:
    return MetricBinding(
        metric_id=metric_id,
        result_position=metric_id,
        unit="CNY_fen" if metric_id.endswith("_fen") else "count",
        time_window=dict(time_window),
        catalog_source_id="commerce-v1",
        catalog_version=DEFAULT_CATALOG_VERSION,
        plan_id=NET_FEN_PLAN_ID if metric_id == "net_fen" else None,
    )


def _plan(context: ExecutionContext, metric_ids: Sequence[str], time_window: Mapping[str, str]) -> tuple[dict[str, object], str]:
    if not 2 <= len(metric_ids) <= 3:
        raise DurableParallelError("invalid_parallel_action", "metric_ids must contain two or three metrics")
    normalized = tuple(sorted(metric_ids))
    if len(set(normalized)) != len(normalized) or any(item not in METRICS for item in normalized):
        raise DurableParallelError("invalid_parallel_action", "metric_ids are not a unique supported set")
    if set(time_window) != {"start", "end", "timezone"}:
        raise DurableParallelError("invalid_parallel_plan", "time_window is incomplete")
    plan = {
        "parallel_version": PARALLEL_RUNTIME_VERSION,
        "metric_ids": list(normalized),
        "time_window": dict(time_window),
        "tenant_id": context.tenant_id,
        "principal_id": context.principal_id,
        "policy_version": "qs-sql-v1",
        "catalog_version": DEFAULT_CATALOG_VERSION,
    }
    digest = hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return plan, digest


class DurableParallelScheduler:
    def __init__(
        self,
        *,
        state: StateStore,
        executor_factory: Callable[[], object] | None = None,
        clock: Callable[[], datetime] = utc_now,
        max_active_branches: int = MAX_ACTIVE_BRANCHES,
    ) -> None:
        if type(max_active_branches) is not int or not 1 <= max_active_branches <= MAX_ACTIVE_BRANCHES:
            raise ValueError(f"max_active_branches must be between 1 and {MAX_ACTIVE_BRANCHES}")
        self.state = state
        self.clock = clock
        self.executor_factory = executor_factory or (lambda: GuardedQueryExecutor())
        self.max_active_branches = max_active_branches
        self._active_lock = Lock()
        self._active = 0
        self._peak = 0

    def run(
        self,
        context: ExecutionContext,
        metric_ids: Sequence[str],
        *,
        time_window: Mapping[str, str] = NET_FEN_TIME_WINDOW,
    ) -> DurableParallelResult:
        if not isinstance(context, ExecutionContext):
            raise DurableParallelError("unauthorized", "parallel execution requires server context")
        plan, plan_hash = _plan(context, metric_ids, time_window)
        existing = self.state.get_parallel_group(context.run_id)
        if existing is not None:
            if existing["plan_hash"] != plan_hash:
                raise DurableParallelError("parallel_plan_conflict", "run already has a different parallel plan")
            if existing["status"] == "SUCCEEDED":
                return self._result(existing, reused=True, peak_active=0, new_branch_count=0)
            recovered = self.recover_submitted_group(context.run_id)
            if recovered is not None:
                self._sync_run_with_group(existing, status="SUCCEEDED")
                return recovered
            self._fail_uncertain_group(existing)
            raise DurableParallelError("recovery_required", "parallel group has unconfirmed branches")

        group_id = f"parallel-{uuid4()}"
        self.state.create_parallel_group(
            group_id=group_id,
            run_id=context.run_id,
            plan_hash=plan_hash,
            plan=plan,
            metric_ids=sorted(metric_ids),
        )
        self._active = 0
        self._peak = 0
        sql_count = 0
        errors: list[str] = []

        def execute_branch(branch: Mapping[str, object]) -> tuple[str, Mapping[str, object], int]:
            branch_id = str(branch["branch_id"])
            metric_id = str(branch["metric_id"])
            self.state.update_parallel_branch(group_id, branch_id, status="RUNNING")
            with self._active_lock:
                self._active += 1
                self._peak = max(self._peak, self._active)
            try:
                value, executions = self._execute_metric(context, metric_id, time_window)
                result = {
                    "metric_id": metric_id,
                    "result_id": value["result_id"],
                    "run_id": value["run_id"],
                    "tenant_id": value["tenant_id"],
                    "principal_id": value["principal_id"],
                    "rows": value["rows"],
                    "observed_at": value["observed_at"],
                }
                self.state.update_parallel_branch(group_id, branch_id, status="SUCCEEDED", result=result)
                return branch_id, result, executions
            except Exception as exc:  # noqa: BLE001
                code = str(getattr(exc, "code", "branch_failed"))
                self.state.update_parallel_branch(group_id, branch_id, status="FAILED", error_code=code)
                raise DurableParallelError(code, "parallel branch failed") from exc
            finally:
                with self._active_lock:
                    self._active -= 1

        group = self.state.get_parallel_group(context.run_id)
        assert group is not None
        branch_rows = tuple(group["branches"])
        results: dict[str, Mapping[str, object]] = {}
        with ThreadPoolExecutor(max_workers=self.max_active_branches, thread_name_prefix="qs-w04-parallel") as pool:
            futures = [pool.submit(execute_branch, branch) for branch in branch_rows]
            for future in as_completed(futures):
                try:
                    branch_id, result, count = future.result()
                    results[branch_id] = result
                    sql_count += count
                except DurableParallelError as exc:
                    errors.append(exc.code)

        if errors:
            self.state.update_parallel_group(group_id, status="FAILED", summary={"errors": errors})
            final = self.state.get_parallel_group(context.run_id)
            assert final is not None
            return self._result(final, reused=False, peak_active=self._peak, new_branch_count=len(results), sql_count=sql_count)
        summary = {"branches": [results[key] for key in sorted(results)], "sql_exec_count": sql_count}
        self.state.update_parallel_group(group_id, status="SUCCEEDED", summary=summary)
        final = self.state.get_parallel_group(context.run_id)
        assert final is not None
        return self._result(final, reused=False, peak_active=self._peak, new_branch_count=len(results), sql_count=sql_count)

    def _execute_metric(
        self,
        context: ExecutionContext,
        metric_id: str,
        time_window: Mapping[str, str],
    ) -> tuple[dict[str, object], int]:
        executor = self.executor_factory()
        params = ("paid", time_window["start"], time_window["end"])
        if metric_id == "gross_fen":
            result = executor.execute(  # type: ignore[attr-defined]
                "SELECT COALESCE(SUM(o.amount_fen), 0) AS gross_fen FROM orders AS o WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s",
                context=context,
                params=params,
                metric_bindings=(_binding(metric_id, time_window),),
            )
            return result.evidence.as_dict(), 1
        if metric_id == "paid_count":
            result = executor.execute(  # type: ignore[attr-defined]
                "SELECT COUNT(*) AS paid_count FROM orders AS o WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s",
                context=context,
                params=params,
                metric_bindings=(_binding(metric_id, time_window),),
            )
            return result.evidence.as_dict(), 1
        if isinstance(executor, FixtureQueryExecutor):
            result = executor.execute(
                "SELECT COALESCE(SUM(o.amount_fen), 0) AS net_fen FROM orders AS o WHERE o.status = %s",
                context=context,
                params=("paid",),
                metric_bindings=(_binding(metric_id, time_window),),
            )
            return result.evidence.as_dict(), 1
        tools = ControlledTools(executor=executor)  # type: ignore[arg-type]
        output = tools.query_readonly(
            {
                "sql": "SELECT COALESCE(SUM(o.amount_fen), 0) AS net_fen FROM orders AS o WHERE o.status = %s",
                "params": {"0": "paid"},
            },
            context=context,
            metric_bindings=(_binding(metric_id, time_window),),
        )
        evidence = tools.get_result_evidence(output["result_id"], context=context)
        return evidence.as_dict(), 2

    def recover_submitted_group(self, run_id: str) -> DurableParallelResult | None:
        group = self.state.get_parallel_group(run_id)
        if group is None or group["status"] == "SUCCEEDED":
            return None
        branches = tuple(group["branches"])
        if not branches or any(branch["status"] != "SUCCEEDED" or branch.get("result") is None for branch in branches):
            return None
        summary = {"branches": [branch["result"] for branch in branches], "recovered": True, "sql_exec_count": 0}
        self.state.update_parallel_group(str(group["group_id"]), status="SUCCEEDED", summary=summary)
        final = self.state.get_parallel_group(run_id)
        assert final is not None
        return self._result(final, reused=True, peak_active=0, new_branch_count=0, sql_count=0)

    def recover_on_startup(self) -> dict[str, object]:
        """Reconcile durable parallel groups before the service accepts requests.

        Only fully persisted branch results are reusable. Any PENDING/RUNNING
        branch is uncertain after process loss and is failed closed without
        constructing an executor or dispatching SQL.
        """
        recovered: list[str] = []
        failed: list[str] = []
        reconciled: list[str] = []
        for run_id in self.state.parallel_group_run_ids():
            group = self.state.get_parallel_group(run_id)
            if group is None:
                continue
            if group["status"] == "RUNNING":
                result = self.recover_submitted_group(run_id)
                if result is not None:
                    group = self.state.get_parallel_group(run_id)
                    assert group is not None
                    self._sync_run_with_group(group, status="SUCCEEDED")
                    recovered.append(run_id)
                    continue
                self._fail_uncertain_group(group)
                failed.append(run_id)
                continue
            target = "SUCCEEDED" if group["status"] == "SUCCEEDED" else "FAILED"
            self._sync_run_with_group(group, status=target)
            reconciled.append(run_id)
        return {
            "recovered_submitted": recovered,
            "failed_uncertain": failed,
            "reconciled_terminal": reconciled,
            "executor_calls": 0,
        }

    def _fail_uncertain_group(self, group: Mapping[str, object]) -> None:
        group_id = str(group["group_id"])
        run_id = str(group["run_id"])
        for branch in group["branches"]:
            if branch["status"] in {"PENDING", "RUNNING"}:
                self.state.update_parallel_branch(
                    group_id,
                    str(branch["branch_id"]),
                    status="FAILED",
                    error_code="recovery_required",
                )
        current = self.state.get_parallel_group(run_id)
        assert current is not None
        self.state.update_parallel_group(
            group_id,
            status="FAILED",
            summary={"error_code": "recovery_required", "recovered": False},
        )
        current = self.state.get_parallel_group(run_id)
        assert current is not None
        self._sync_run_with_group(current, status="FAILED")

    def _sync_run_with_group(self, group: Mapping[str, object], *, status: str) -> None:
        run_id = str(group["run_id"])
        run = self.state.get_run(run_id)
        if run is None:
            return
        if status == "SUCCEEDED":
            summary = group.get("summary") or {
                "branches": [branch["result"] for branch in group["branches"]],
                "recovered": True,
                "sql_exec_count": 0,
            }
            result_json = json.dumps(summary, ensure_ascii=False, sort_keys=True)
            if run.get("status") != status or run.get("result") is None:
                self.state.update_run(
                    run_id,
                    status=status,
                    result_json=result_json,
                    error_code=None,
                )
        else:
            summary = group.get("summary")
            error_code = (
                str(summary.get("error_code"))
                if isinstance(summary, Mapping) and summary.get("error_code")
                else next(
                    (str(branch["error_code"]) for branch in group["branches"] if branch.get("error_code")),
                    "parallel_branch_failed",
                )
            )
            if run.get("status") != status or run.get("error_code") != error_code:
                self.state.update_run(run_id, status=status, error_code=error_code)
        if run.get("status") != status:
            self.state.append_event(
                run_id,
                "terminal",
                status,
                payload={"recovered_parallel_group": str(group["group_id"]), "error_code": None if status == "SUCCEEDED" else "recovery_required"},
            )

    def _result(self, group: Mapping[str, object], *, reused: bool, peak_active: int, new_branch_count: int, sql_count: int = 0) -> DurableParallelResult:
        return DurableParallelResult(
            run_id=str(group["run_id"]),
            group_id=str(group["group_id"]),
            plan_hash=str(group["plan_hash"]),
            status=str(group["status"]),
            branches=tuple(group["branches"]),
            reused=reused,
            peak_active=peak_active,
            new_branch_count=new_branch_count,
            sql_exec_count=sql_count,
        )


__all__ = [
    "DurableParallelError",
    "DurableParallelResult",
    "DurableParallelScheduler",
    "MAX_ACTIVE_BRANCHES",
    "PARALLEL_RUNTIME_VERSION",
]
