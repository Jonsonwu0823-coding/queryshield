"""The multi-agent profile: a coordinator delegates 2-3 declared subtasks to sub-agents.

The coordinator is the bounded agent with one more action, ``delegate``.  A
subtask is only a metric declaration (metrics and a time window), checked like a
query_readonly declaration against the user's own question.  Each sub-agent is a
bounded agent that sees only a sentence the server writes for its subtask; it
shares the run's identity, tools, model and call records, gets a share of the
run's budget, and ends as soon as every bound metric has a result.  The
coordinator then answers from all the results, through the usual answer check.

This module holds the pure parts; the graph nodes that use them are in
:mod:`queryshield.agent.graph`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

from queryshield.agent.metric_intent import MetricDeclarationError, check_confirmed_metrics, resolve_query_declaration
from queryshield.agent.proposals import DelegateAction, MetricBinding
from queryshield.agent.tool_execution import check_clarification
from queryshield.catalog.catalog import SemanticCatalog
from queryshield.catalog.phrases import ClarificationReading
from queryshield.tools.semantic import ToolError

# A sub-agent ended because every bound metric has a result (its final_action type).
SUBTASK_COMPLETE = "subtask_complete"
# A sub-agent asked the user, needed approval, refused or answered by itself.
SUBTASK_INCOMPLETE_CODE = "subtask_incomplete"
# Not enough of the run's budget left to give every sub-agent one model and one tool call.
DELEGATION_BUDGET_CODE = "delegation_budget_insufficient"
# After delegating, the coordinator may only answer or refuse.
AFTER_DELEGATION_CODE = "invalid_action_after_delegation"

_T = TypeVar("_T")


@dataclass(frozen=True)
class AgentRole:
    """Which agent of a multi-agent run this is; B1 has none."""

    label: str
    can_delegate: bool = False
    ends_when_bound: bool = False


COORDINATOR = AgentRole("coordinator", can_delegate=True)


def subtask_role(index: int) -> AgentRole:
    return AgentRole(f"subtask-{index}", ends_when_bound=True)


@dataclass(frozen=True)
class Subtask:
    """One checked subtask: its server-built bindings share one normalized window."""

    index: int
    bindings: tuple[MetricBinding, ...]

    @property
    def time_window(self) -> dict[str, str]:
        return dict(self.bindings[0].time_window)

    @property
    def metric_ids(self) -> list[str]:
        return [binding.metric_id.removeprefix("metric.") for binding in self.bindings]


class DelegationBudgetError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def resolve_subtasks(
    action: DelegateAction,
    *,
    catalog: SemanticCatalog,
    request_time_window: Mapping[str, object] | None,
    prebound: Sequence[MetricBinding],
    clarifications: ClarificationReading | None,
) -> tuple[Subtask, ...]:
    """Check every subtask like a query_readonly declaration, then the subtasks together.

    A subtask keeps the user-confirmed bindings of its own metrics; all subtasks
    together must keep every confirmed metric.  Raises ToolError (declaration
    codes) or the clarification errors of check_clarification.
    """

    confirmed = {binding.metric_id.removeprefix("metric."): binding for binding in prebound}
    subtasks: list[Subtask] = []
    seen: set[tuple[str, str, str]] = set()
    try:
        for index, arguments in enumerate(action.subtasks, start=1):
            own = tuple(binding for metric_id, binding in confirmed.items() if metric_id in arguments["metrics"])
            declaration = resolve_query_declaration(
                arguments, catalog=catalog, request_time_window=request_time_window, prebound=own
            )
            check_clarification(clarifications, declaration.declared_metric_ids)
            subtask = Subtask(index, declaration.bindings)
            for metric_id in subtask.metric_ids:
                key = (metric_id, subtask.time_window["start"], subtask.time_window["end"])
                if key in seen:
                    raise MetricDeclarationError("invalid_metric_declaration", "subtasks must not repeat a metric and time window")
                seen.add(key)
            subtasks.append(subtask)
        check_confirmed_metrics(catalog, confirmed, {metric_id for item in subtasks for metric_id in item.metric_ids})
    except MetricDeclarationError as exc:
        raise ToolError(exc.code, exc.message) from exc
    return tuple(subtasks)


def allocate(
    *,
    max_model_calls: int,
    max_tool_calls: int,
    max_seconds: float,
    model_used: int,
    tool_used: int,
    elapsed: float,
    count: int,
) -> tuple[int, int, float]:
    """Each sub-agent's share of what the run has left: (model calls, tool calls, seconds).

    One model call stays with the coordinator for its answer; the rest is split
    evenly (rounded down), so the run's totals never exceed its limits.  The
    sub-agents run side by side, so each may use all the time that is left.
    """

    seconds = max_seconds - elapsed
    if seconds <= 0:
        raise DelegationBudgetError("wall_clock_limit", "active wall-clock limit reached")
    model_calls = (max_model_calls - model_used - 1) // count
    tool_calls = (max_tool_calls - tool_used) // count
    if model_calls < 1 or tool_calls < 1:
        raise DelegationBudgetError(DELEGATION_BUDGET_CODE, "the remaining budget cannot give every subtask a model and a tool call")
    return model_calls, tool_calls, seconds


def subtask_question(catalog: SemanticCatalog, subtask: Subtask) -> str:
    """The only question a sub-agent sees: catalog names and the window, never the user's words."""

    window = subtask.time_window
    metrics = "和".join(f"{catalog.metric_name(metric_id)}（{metric_id}）" for metric_id in subtask.metric_ids)
    return f"统计时间窗 [{window['start']}, {window['end']})（UTC）内的{metrics}。"


def run_all(calls: Sequence[Callable[[], _T]]) -> list[_T]:
    """Run every call in its own thread and wait for all of them.

    Results come back in call order.  A failure is raised only after every call
    has ended (the first one in call order), so nothing a sub-agent does can
    happen after the run has ended.
    """

    with ThreadPoolExecutor(max_workers=len(calls), thread_name_prefix="queryshield-subtask") as pool:
        futures = [pool.submit(call) for call in calls]
    return [future.result() for future in futures]


def completed(state: Mapping[str, object]) -> bool:
    action = state.get("final_action")
    return state.get("status") == "succeeded" and isinstance(action, Mapping) and action.get("type") == SUBTASK_COMPLETE


def delegation_outcome(states: Sequence[Mapping[str, object]]) -> tuple[str, str | None] | None:
    """None when every subtask completed; else the run's status and error code.

    A server refusal comes first (a denied sub-agent carries an error code only
    when the server refused its query), then a budget stop, then any other
    failure; within each, the lowest subtask number.  A policy refusal is never
    hidden behind another outcome.
    """

    failed = [state for state in states if not completed(state)]
    if not failed:
        return None
    for state in failed:
        if state.get("status") == "denied" and state.get("error_code"):
            return "denied", str(state["error_code"])
    for state in failed:
        if state.get("status") == "limit_reached":
            return "limit_reached", state.get("error_code")  # type: ignore[return-value]
    state = failed[0]
    if state.get("status") == "failed" and state.get("error_code"):
        return "failed", str(state["error_code"])
    return "failed", SUBTASK_INCOMPLETE_CODE


__all__ = [
    "AFTER_DELEGATION_CODE",
    "COORDINATOR",
    "DELEGATION_BUDGET_CODE",
    "SUBTASK_COMPLETE",
    "SUBTASK_INCOMPLETE_CODE",
    "AgentRole",
    "DelegationBudgetError",
    "Subtask",
    "allocate",
    "completed",
    "delegation_outcome",
    "resolve_subtasks",
    "run_all",
    "subtask_question",
    "subtask_role",
]
