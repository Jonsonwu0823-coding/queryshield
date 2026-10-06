"""Bounded same-agent read-only parallel scheduling for AGENT-U02.

This module is intentionally an in-process Fake scheduler.  It owns branch
identity, plan hashing, result merging, and the peak-active invariant; a later
week may replace the branch runner with a durable database boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from hashlib import sha256
import json
from threading import Lock
from time import monotonic
from typing import Literal
from uuid import uuid4

from queryshield.catalog.catalog import DEFAULT_CATALOG_VERSION
from queryshield.agent.proposals import (
    ExecutionContext,
    PARALLEL_METRICS,
    ParallelReadonlyAction,
)
from queryshield.policy.sql import SQL_POLICY_VERSION


PARALLEL_VERSION = "qs-parallel-v1"
MAX_PARALLEL_BRANCHES = 3
MAX_ACTIVE_BRANCHES = 2


class ParallelValidationError(ValueError):
    """A parallel action or trusted plan was rejected before branch execution."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class ParallelPlanConflict(ParallelValidationError):
    def __init__(self, message: str = "a run already has a different parallel plan") -> None:
        super().__init__("parallel_plan_conflict", message)


def _require_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or len(value) > 200:
        raise ParallelValidationError("invalid_parallel_plan", f"{field} must be a non-empty string")
    return value


def _validate_metric_ids(metric_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(metric_ids, (str, bytes)) or not 2 <= len(metric_ids) <= MAX_PARALLEL_BRANCHES:
        raise ParallelValidationError("invalid_parallel_action", "metric_ids must contain two or three metrics")
    normalized = tuple(metric_ids)
    if any(type(metric_id) is not str or not metric_id.strip() for metric_id in normalized):
        raise ParallelValidationError("invalid_parallel_action", "metric IDs must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ParallelValidationError("invalid_parallel_action", "metric IDs must be unique")
    unknown = sorted(set(normalized) - PARALLEL_METRICS)
    if unknown:
        raise ParallelValidationError("invalid_parallel_action", "metric IDs contain an unsupported metric")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class ParallelPlan:
    """Server-created plan whose hash binds identity and semantic versions."""

    metric_ids: tuple[str, ...]
    time_window: Mapping[str, str]
    tenant_id: str
    principal_id: str
    role: str
    policy_version: str
    catalog_version: str
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_ids", _validate_metric_ids(self.metric_ids))
        required_window = {"start", "end", "timezone"}
        if not isinstance(self.time_window, Mapping) or not required_window <= set(self.time_window):
            raise ParallelValidationError(
                "invalid_parallel_plan",
                "time_window must contain start, end and timezone",
            )
        normalized_window: dict[str, str] = {}
        for key, value in self.time_window.items():
            normalized_window[key] = _require_text(value, field=f"time_window.{key}")
        object.__setattr__(self, "time_window", normalized_window)
        for field_name in ("tenant_id", "principal_id", "role", "policy_version", "catalog_version"):
            _require_text(getattr(self, field_name), field=field_name)
        payload = {
            "parallel_version": PARALLEL_VERSION,
            "metric_ids": list(self.metric_ids),
            "time_window": dict(sorted(normalized_window.items())),
            "tenant_id": self.tenant_id,
            "principal_id": self.principal_id,
            "role": self.role,
            "policy_version": self.policy_version,
            "catalog_version": self.catalog_version,
        }
        digest = sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "plan_hash", digest)

    @classmethod
    def from_context(
        cls,
        context: ExecutionContext,
        metric_ids: Sequence[str],
        *,
        time_window: Mapping[str, str],
        policy_version: str = SQL_POLICY_VERSION,
        catalog_version: str = DEFAULT_CATALOG_VERSION,
    ) -> ParallelPlan:
        if not isinstance(context, ExecutionContext):
            raise ParallelValidationError("unauthorized", "parallel plans require server context")
        return cls(
            metric_ids=tuple(metric_ids),
            time_window=time_window,
            tenant_id=context.tenant_id,
            principal_id=context.principal_id,
            role=context.role,
            policy_version=policy_version,
            catalog_version=catalog_version,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "parallel_version": PARALLEL_VERSION,
            "plan_hash": self.plan_hash,
            "metric_ids": list(self.metric_ids),
            "time_window": dict(self.time_window),
            "policy_version": self.policy_version,
            "catalog_version": self.catalog_version,
        }


@dataclass(frozen=True)
class BranchExecution:
    """Safe output from one injected fake branch runner."""

    metric_id: str
    result_id: str
    rows: tuple[Mapping[str, object], ...]
    observed_at: str
    error_code: str | None = None
    elapsed_ms: int = 0


BranchStatus = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED"]


@dataclass(frozen=True)
class ParallelBranchResult:
    branch_id: str
    metric_id: str
    status: BranchStatus
    result_id: str | None
    rows: tuple[Mapping[str, object], ...]
    observed_at: str | None
    error_code: str | None
    elapsed_ms: int

    def as_dict(self) -> dict[str, object]:
        return {
            "branch_id": self.branch_id,
            "metric_id": self.metric_id,
            "status": self.status,
            "result_id": self.result_id,
            "rows": [dict(row) for row in self.rows],
            "observed_at": self.observed_at,
            "error_code": self.error_code,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True)
class ParallelRunResult:
    status: Literal["SUCCEEDED", "FAILED", "LIMIT_REACHED"]
    run_id: str
    group_id: str
    plan_hash: str
    branches: tuple[ParallelBranchResult, ...]
    completion_order: tuple[str, ...]
    peak_active: int
    new_branch_count: int
    reused: bool
    error_code: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "parallel_version": PARALLEL_VERSION,
            "status": self.status,
            "run_id": self.run_id,
            "group_id": self.group_id,
            "plan_hash": self.plan_hash,
            "branches": [branch.as_dict() for branch in self.branches],
            "completion_order": list(self.completion_order),
            "peak_active": self.peak_active,
            "new_branch_count": self.new_branch_count,
            "reused": self.reused,
            "error_code": self.error_code,
        }


@dataclass
class _Group:
    run_id: str
    group_id: str
    plan_hash: str
    branch_ids: dict[str, str]
    results: dict[str, ParallelBranchResult] = field(default_factory=dict)
    completion_order: list[str] = field(default_factory=list)
    peak_active: int = 0


class ParallelGroupStore:
    """Thread-safe in-memory group/branch identity store for the Fake path."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._groups_by_run: dict[str, _Group] = {}
        self._groups_by_id: dict[str, _Group] = {}

    def get_or_create(self, context: ExecutionContext, plan: ParallelPlan) -> _Group:
        with self._lock:
            existing = self._groups_by_run.get(context.run_id)
            if existing is not None:
                if existing.plan_hash != plan.plan_hash:
                    raise ParallelPlanConflict()
                return existing
            group_id = f"parallel-{uuid4()}"
            group = _Group(
                run_id=context.run_id,
                group_id=group_id,
                plan_hash=plan.plan_hash,
                branch_ids={
                    metric_id: f"{group_id}:{metric_id}"
                    for metric_id in plan.metric_ids
                },
            )
            self._groups_by_run[context.run_id] = group
            self._groups_by_id[group_id] = group
            return group

    def peek(self, run_id: str) -> _Group | None:
        with self._lock:
            return self._groups_by_run.get(run_id)

    def save_branch(self, group: _Group, branch: ParallelBranchResult) -> None:
        with self._lock:
            group.results[branch.branch_id] = branch
            group.completion_order.append(branch.branch_id)

    def set_peak_active(self, group: _Group, peak_active: int) -> None:
        with self._lock:
            group.peak_active = max(group.peak_active, peak_active)


BranchRunner = Callable[[ExecutionContext, str, str], BranchExecution]


class ParallelScheduler:
    """Execute up to two read-only Fake branches at once and merge by branch ID."""

    def __init__(
        self,
        branch_runner: BranchRunner,
        *,
        store: ParallelGroupStore | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._branch_runner = branch_runner
        self._store = store or ParallelGroupStore()
        self._clock = clock

    @property
    def store(self) -> ParallelGroupStore:
        return self._store

    def run(
        self,
        context: ExecutionContext,
        action_or_metric_ids: ParallelReadonlyAction | Sequence[str],
        *,
        plan: ParallelPlan,
        tool_budget_remaining: int | None = None,
    ) -> ParallelRunResult:
        if not isinstance(context, ExecutionContext):
            raise ParallelValidationError("unauthorized", "parallel execution requires server context")
        metric_ids = (
            action_or_metric_ids.metric_ids
            if isinstance(action_or_metric_ids, ParallelReadonlyAction)
            else tuple(action_or_metric_ids)
        )
        normalized_metrics = _validate_metric_ids(metric_ids)
        if not isinstance(plan, ParallelPlan):
            raise ParallelValidationError("invalid_parallel_plan", "plan must be server-created")
        if (
            plan.metric_ids != normalized_metrics
            or plan.tenant_id != context.tenant_id
            or plan.principal_id != context.principal_id
            or plan.role != context.role
        ):
            raise ParallelPlanConflict("parallel action does not match the server plan")

        existing_group = self._store.peek(context.run_id)
        if existing_group is not None and existing_group.plan_hash != plan.plan_hash:
            raise ParallelPlanConflict()
        group = existing_group
        existing = tuple(group.results.values()) if group is not None else ()
        if group is not None and (
            (len(existing) == len(group.branch_ids) and all(item.status == "SUCCEEDED" for item in existing))
            or any(item.status == "FAILED" for item in existing)
        ):
            return self._result(group, run_id=context.run_id, plan_hash=plan.plan_hash, new_branch_count=0, reused=True)

        pending_metrics = (
            [metric_id for metric_id in plan.metric_ids if group.branch_ids[metric_id] not in group.results]
            if group is not None
            else list(plan.metric_ids)
        )
        if tool_budget_remaining is not None:
            if type(tool_budget_remaining) is not int or tool_budget_remaining < 0:
                raise ParallelValidationError("invalid_parallel_budget", "remaining tool budget must be non-negative")
            if tool_budget_remaining < len(pending_metrics):
                budget_group = group or _Group(
                    run_id=context.run_id,
                    group_id="",
                    plan_hash=plan.plan_hash,
                    branch_ids={},
                )
                return self._budget_result(
                    budget_group,
                    plan=plan,
                    pending_metrics=pending_metrics,
                    reused=existing_group is not None,
                )
        group = group or self._store.get_or_create(context, plan)
        active = 0
        peak_active = 0
        active_lock = Lock()

        def execute(metric_id: str) -> ParallelBranchResult:
            nonlocal active, peak_active
            branch_id = group.branch_ids[metric_id]
            with active_lock:
                active += 1
                peak_active = max(peak_active, active)
            started = self._clock()
            try:
                execution = self._branch_runner(context, metric_id, branch_id)
                if not isinstance(execution, BranchExecution) or execution.metric_id != metric_id:
                    raise ParallelValidationError("invalid_branch_result", "branch runner returned the wrong metric")
                rows = tuple(dict(row) for row in execution.rows)
                if any(not isinstance(row, Mapping) for row in rows):
                    raise ParallelValidationError("invalid_branch_result", "branch rows must be mappings")
                elapsed_ms = execution.elapsed_ms or int(max(0.0, self._clock() - started) * 1000)
                return ParallelBranchResult(
                    branch_id=branch_id,
                    metric_id=metric_id,
                    status="SUCCEEDED",
                    result_id=execution.result_id,
                    rows=rows,
                    observed_at=execution.observed_at,
                    error_code=None,
                    elapsed_ms=elapsed_ms,
                )
            except Exception as exc:
                code = exc.code if isinstance(exc, ParallelValidationError) else "branch_failed"
                return ParallelBranchResult(
                    branch_id=branch_id,
                    metric_id=metric_id,
                    status="FAILED",
                    result_id=None,
                    rows=(),
                    observed_at=None,
                    error_code=code,
                    elapsed_ms=int(max(0.0, self._clock() - started) * 1000),
                )
            finally:
                with active_lock:
                    active -= 1

        with ThreadPoolExecutor(max_workers=MAX_ACTIVE_BRANCHES, thread_name_prefix="qs-parallel") as pool:
            futures = {pool.submit(execute, metric_id): metric_id for metric_id in pending_metrics}
            for future in as_completed(futures):
                branch = future.result()
                self._store.save_branch(group, branch)

        self._store.set_peak_active(group, peak_active)
        return self._result(
            group,
            run_id=context.run_id,
            plan_hash=plan.plan_hash,
            new_branch_count=len(pending_metrics),
            reused=False,
        )

    def _budget_result(
        self,
        group: _Group,
        *,
        plan: ParallelPlan,
        pending_metrics: Sequence[str],
        reused: bool,
    ) -> ParallelRunResult:
        branch_ids = dict(group.branch_ids)
        if not branch_ids:
            branch_ids = {
                metric_id: f"uncommitted-{plan.plan_hash[:16]}:{metric_id}"
                for metric_id in plan.metric_ids
            }
        branches_by_id = dict(group.results)
        for metric_id in pending_metrics:
            branch_id = branch_ids[metric_id]
            branches_by_id.setdefault(
                branch_id,
                ParallelBranchResult(
                    branch_id=branch_id,
                    metric_id=metric_id,
                    status="PENDING",
                    result_id=None,
                    rows=(),
                    observed_at=None,
                    error_code="parallel_budget_insufficient",
                    elapsed_ms=0,
                ),
            )
        branches = tuple(
            branches_by_id[branch_id]
            for branch_id in sorted(branches_by_id)
        )
        return ParallelRunResult(
            status="LIMIT_REACHED",
            run_id=group.run_id,
            group_id=group.group_id if group.branch_ids else "",
            plan_hash=plan.plan_hash,
            branches=branches,
            completion_order=tuple(group.completion_order),
            peak_active=group.peak_active,
            new_branch_count=0,
            reused=reused,
            error_code="parallel_budget_insufficient",
        )

    def _result(
        self,
        group: _Group,
        *,
        run_id: str,
        plan_hash: str,
        new_branch_count: int,
        reused: bool,
    ) -> ParallelRunResult:
        branches = tuple(group.results[branch_id] for branch_id in sorted(group.results))
        status: Literal["SUCCEEDED", "FAILED"] = (
            "SUCCEEDED"
            if len(branches) == len(group.branch_ids) and all(branch.status == "SUCCEEDED" for branch in branches)
            else "FAILED"
        )
        return ParallelRunResult(
            status=status,
            run_id=run_id,
            group_id=group.group_id,
            plan_hash=plan_hash,
            branches=branches,
            completion_order=tuple(group.completion_order),
            peak_active=group.peak_active,
            new_branch_count=new_branch_count,
            reused=reused,
            error_code=None,
        )


__all__ = [
    "BranchExecution",
    "MAX_ACTIVE_BRANCHES",
    "MAX_PARALLEL_BRANCHES",
    "PARALLEL_VERSION",
    "ParallelBranchResult",
    "ParallelGroupStore",
    "ParallelPlan",
    "ParallelPlanConflict",
    "ParallelRunResult",
    "ParallelScheduler",
    "ParallelValidationError",
]
