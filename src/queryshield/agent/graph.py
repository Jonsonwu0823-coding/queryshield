"""Bounded LangGraph runtime for the W03 single-agent path.

The graph is deliberately small.  The model may propose one of the validated
actions from :mod:`queryshield.agent.proposals`; it never selects a Python
callable or owns the execution context.  Every model call gets a server-side
identity before the provider is invoked, and the graph stops before a budget
would be exceeded.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from time import monotonic
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from queryshield.agent.config import DEFAULT_RUN_CONFIG, RunConfig
from queryshield.agent.context import (
    CONTEXT_VERSION,
    REPAIRABLE_QUERY_ERROR_CODES,
    ContextBuildError,
    available_action_types,
    build_context,
)
from queryshield.agent.metric_intent import (
    MetricDeclarationError,
    answer_basis_conflict_hint,
    answer_not_grounded_hint,
    answer_without_query_hint,
    declarable_metric_ids,
    knowledge_from_server_search_hint,
    normalize_time_window,
    undeclared_metric_hint,
)
from queryshield.catalog import load_default_catalog
from queryshield.catalog.phrases import (
    ClarificationReading,
    contradiction_hint,
    not_needed_hint,
    read_clarifications,
    review_ask,
)
from queryshield.facts import FACTS_SCHEMA_VERSION, FactResolutionError, FactResolver
from queryshield.facts.facts import is_scalar_metric_result
from queryshield.facts.render import render_no_data_answer, render_verified_answer
from queryshield.mcp_metadata.schemas import MCP_ERROR_CODES
from queryshield.agent.proposals import (
    ALLOWED_TABLES,
    AskUserAction,
    DenyAction,
    ExecutionContext,
    FinalAnswerAction,
    MetricBinding,
    ModelCallStore,
    ParallelReadonlyAction,
    ProposalParseError,
    QueryProposal,
    ToolCallAction,
    ToolNameAsActionTypeError,
    parse_error_detail,
    parse_query_proposal,
    proposal_shape_summary,
)
from queryshield.providers.contracts import ModelAdapter, ModelProviderError
from queryshield.tools.semantic import ControlledTools, ToolError
from queryshield.agent.tool_execution import (
    CLARIFICATION_VALUE_UNSUPPORTED_CODE,
    ApprovalRequiredError,
    ClarificationRequiredError,
    ClarificationValueUnsupportedError,
    MetricContradictsQuestionError,
    call_tool,
    check_clarification,
)
from queryshield.agent.tenant_scope import has_explicit_foreign_tenant
from queryshield.agent.parallel import ParallelPlan, ParallelScheduler, ParallelValidationError


MAX_MODEL_CALLS = 6
MAX_TOOL_CALLS = 8
# The server's own catalog search for a knowledge answer (B3c-2 R2): the
# contract's top_k default and search_catalog's query length limit.
SERVER_SEARCH_TOP_K = 3
SERVER_SEARCH_MAX_QUERY_CHARS = 200
MAX_WALL_CLOCK_SECONDS = 60.0
MAX_QUERY_REPAIRS = 1
# An ask_user the question's wording does not need is sent back at most once
# per run.  This budget is separate from MAX_QUERY_REPAIRS on purpose (an
# exception to the B2a rule that every repairable error shares one repair): a
# bounced ask must not use up the repair a later failed query needs.  A
# declaration the question contradicts (B3c-1) shares this one bounce: two
# phrase-table corrections in one run fail the run.
MAX_CLARIFICATION_BOUNCES = 1
# A final answer nothing in this run grounds (no query for business values, a
# basis the run contradicts, or results cited before any query) is sent back
# at most once per run (B3c-2).  A third budget on purpose: it must not use
# the SQL repair (single-repair-budget expects exactly one) nor the ask
# bounce that a clarification may already have used.
MAX_ANSWER_BOUNCES = 1
CLARIFICATION_NOT_NEEDED_CODE = "clarification_not_needed"
AGENT_CHECKPOINT_VERSION = "qs-bounded-agent-checkpoint-v4"
# v3 checkpoints predate the answer bounce count and restore with 0.  W05
# prepared WAITING_USER fixtures are still written as v3.
_V3_AGENT_CHECKPOINT_VERSION = "qs-bounded-agent-checkpoint-v3"
# v2 checkpoints predate the clarification bounce count and rule id; v1 also
# predates the request-level time window.  Both restore with the defaults.
_V2_AGENT_CHECKPOINT_VERSION = "qs-bounded-agent-checkpoint-v2"
_LEGACY_AGENT_CHECKPOINT_VERSION = "qs-bounded-agent-checkpoint-v1"
_REPAIRABLE_QUERY_ERRORS = frozenset(REPAIRABLE_QUERY_ERROR_CODES)
_SECURITY_REFUSAL_ERRORS = frozenset({
    "approval_required",
    "forbidden",
    "function_not_allowed",
    "multiple_statements",
    "reserved_parameter",
    "statement_not_allowed",
    "table_not_allowed",
    "unauthorized",
    "unknown_argument",
    "unsupported_syntax",
})

RunStatus = Literal[
    "succeeded",
    "denied",
    "waiting_user",
    "waiting_approval",
    "failed",
    "limit_reached",
]
_InternalStatus = Literal[
    "running", "succeeded", "denied", "waiting_user", "waiting_approval", "failed", "limit_reached"
]


class RunResumeError(ValueError):
    """A waiting-user run cannot be resumed by the supplied server context."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class GraphLimits:
    """Server-owned online budgets for one graph invocation."""

    max_model_calls: int = MAX_MODEL_CALLS
    max_tool_calls: int = MAX_TOOL_CALLS
    max_wall_clock_seconds: float = MAX_WALL_CLOCK_SECONDS

    def __post_init__(self) -> None:
        if type(self.max_model_calls) is not int or not 1 <= self.max_model_calls <= MAX_MODEL_CALLS:
            raise ValueError("max_model_calls must be between 1 and 6")
        if type(self.max_tool_calls) is not int or not 1 <= self.max_tool_calls <= MAX_TOOL_CALLS:
            raise ValueError("max_tool_calls must be between 1 and 8")
        if type(self.max_wall_clock_seconds) not in {int, float}:
            raise ValueError("max_wall_clock_seconds must be numeric")
        if not 0 < float(self.max_wall_clock_seconds) <= MAX_WALL_CLOCK_SECONDS:
            raise ValueError("max_wall_clock_seconds must be between 0 and 60 seconds")


class _GraphState(TypedDict, total=False):
    context: ExecutionContext
    question: str
    run_config: RunConfig
    metric_bindings: tuple[MetricBinding, ...]
    request_time_window: Mapping[str, str] | None
    # Set only by the action-type repair; never persisted in checkpoints.
    retry_model: bool
    parallel_plan: ParallelPlan | None
    started_at: float
    active_elapsed_before: float
    clarifications: tuple[str, ...]
    clarification_bounce_count: int
    answer_bounce_count: int
    status: _InternalStatus
    reason: str | None
    error_code: str | None
    proposal: QueryProposal | None
    final_action: Mapping[str, object] | None
    model_call_count: int
    tool_call_count: int
    repair_count: int
    model_call_ids: tuple[str, ...]
    events: tuple[Mapping[str, object], ...]
    retrieval_items: tuple[Mapping[str, object], ...]
    tool_results: tuple[Mapping[str, object], ...]
    public_answer: str | None
    # verified / unverified / no_data for a succeeded answer, else None.
    answer_status: str | None
    facts: Mapping[str, object] | None
    context_version: str
    elapsed_ms: int


@dataclass(frozen=True)
class AgentRunResult:
    """Redacted result of one graph run; provider messages are not retained."""

    status: RunStatus
    run_id: str
    run_config: RunConfig
    reason: str | None
    error_code: str | None
    action: Mapping[str, object] | None
    answer: str | None
    facts: Mapping[str, object] | None
    model_call_ids: tuple[str, ...]
    model_call_count: int
    tool_call_count: int
    repair_count: int
    usage_summary: Mapping[str, object]
    events: tuple[Mapping[str, object], ...]
    context_version: str
    elapsed_ms: int
    answer_status: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "run_config": self.run_config.as_dict(),
            "reason": self.reason,
            "error_code": self.error_code,
            "action": dict(self.action) if self.action is not None else None,
            "answer": self.answer,
            "facts": dict(self.facts) if self.facts is not None else None,
            "model_call_ids": list(self.model_call_ids),
            "model_call_count": self.model_call_count,
            "tool_call_count": self.tool_call_count,
            "repair_count": self.repair_count,
            "usage_summary": dict(self.usage_summary),
            "events": [dict(event) for event in self.events],
            "trace": [dict(event) for event in self.events],
            "context_version": self.context_version,
            "elapsed_ms": self.elapsed_ms,
            "answer_status": self.answer_status,
        }


class BoundedAgent:
    """The unique formal W03 runtime graph.

    The graph has three meaningful nodes: ``model_decision`` validates one
    constrained JSON proposal, ``execute_tool`` dispatches only through the
    server-owned tool facade, and ``finish`` converts terminal actions into a
    result.  A tool node returns to the model node, but all transitions remain
    inside the three explicit budgets in :class:`GraphLimits`.
    """

    def __init__(
        self,
        model: ModelAdapter,
        *,
        tools: ControlledTools | None = None,
        call_store: ModelCallStore | Any | None = None,
        limits: GraphLimits | None = None,
        clock: Callable[[], float] = monotonic,
        run_config: RunConfig | None = None,
        parallel_scheduler: ParallelScheduler | None = None,
        retrieval_available: bool = True,
    ) -> None:
        if type(retrieval_available) is not bool:
            raise TypeError("retrieval_available must be a boolean")
        self.model = model
        self.tools = tools or ControlledTools()
        self.call_store = call_store or ModelCallStore()
        self.limits = limits or GraphLimits()
        self._clock = clock
        self.run_config = run_config or DEFAULT_RUN_CONFIG
        if not isinstance(self.run_config, RunConfig):
            raise TypeError("run_config must be a RunConfig")
        self.parallel_scheduler = parallel_scheduler
        # Server configuration: without a product retriever the context does
        # not describe search_catalog and the tool node refuses it.
        self.retrieval_available = retrieval_available
        self._waiting_checkpoints: dict[str, _GraphState] = {}
        self._compiled_graph = self._build_graph()

    @property
    def graph(self) -> Any:
        """Return the compiled graph for structural checks and debugging."""

        return self._compiled_graph

    def run(
        self,
        context: ExecutionContext,
        question: str,
        *,
        metric_bindings: Sequence[MetricBinding] = (),
        parallel_plan: ParallelPlan | None = None,
        request_time_window: Mapping[str, object] | None = None,
        _evaluation_initial_retrieval_items: Sequence[Mapping[str, object]] = (),
    ) -> AgentRunResult:
        """Run one request.

        ``metric_bindings`` are server pre-bound bindings (a confirmed slot);
        ``request_time_window`` is the request-level window, like a date
        picker.  Model metric declarations must agree with both.
        """

        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")
        normalized_request_window = _request_window(request_time_window)
        if has_explicit_foreign_tenant(question, context.tenant_id):
            event = {
                "kind": "authorization",
                "status": "denied",
                "error_code": "forbidden",
                "reason_code": "explicit_foreign_tenant_request",
            }
            return AgentRunResult(
                status="denied",
                run_id=context.run_id,
                run_config=self.run_config,
                reason="request targets a tenant outside the authenticated scope",
                error_code="forbidden",
                action=None,
                answer=None,
                facts=None,
                model_call_ids=(),
                model_call_count=0,
                tool_call_count=0,
                repair_count=0,
                usage_summary=_usage_summary((event,)),
                events=(event,),
                context_version=CONTEXT_VERSION,
                elapsed_ms=0,
            )
        if context.run_id in self._waiting_checkpoints:
            raise RunResumeError(
                "run_waiting_user",
                "resume the existing WAITING_USER run instead of starting a second run",
            )
        if not isinstance(_evaluation_initial_retrieval_items, Sequence) or isinstance(
            _evaluation_initial_retrieval_items, (str, bytes)
        ):
            raise TypeError("initial evaluation retrieval items must be a sequence")
        initial_retrieval_items = tuple(
            dict(item) for item in _evaluation_initial_retrieval_items if isinstance(item, Mapping)
        )
        if len(initial_retrieval_items) != len(_evaluation_initial_retrieval_items):
            raise TypeError("initial evaluation retrieval items must contain mappings")
        state: _GraphState = {
            "context": context,
            "question": question,
            "run_config": self.run_config,
            "metric_bindings": tuple(metric_bindings),
            "request_time_window": normalized_request_window,
            "parallel_plan": parallel_plan,
            "started_at": self._clock(),
            "active_elapsed_before": 0.0,
            "clarifications": (),
            "clarification_bounce_count": 0,
            "answer_bounce_count": 0,
            "status": "running",
            "reason": None,
            "error_code": None,
            "proposal": None,
            "final_action": None,
            "model_call_count": 0,
            "tool_call_count": 0,
            "repair_count": 0,
            "model_call_ids": (),
            "events": (),
            "retrieval_items": initial_retrieval_items,
            "tool_results": (),
            "public_answer": None,
            "facts": None,
            "context_version": CONTEXT_VERSION,
            "elapsed_ms": 0,
        }
        final_state = self._compiled_graph.invoke(state)
        result = self._result_from_state(final_state, run_id=context.run_id)
        if result.status == "waiting_user":
            self._waiting_checkpoints[context.run_id] = final_state
        return result

    def resume(
        self,
        context: ExecutionContext,
        answer: str,
    ) -> AgentRunResult:
        checkpoint = self._waiting_checkpoints.get(context.run_id)
        if checkpoint is None:
            raise RunResumeError(
                "invalid_run_state",
                "only a WAITING_USER run can be resumed",
            )
        original_context = checkpoint.get("context")
        if original_context != context:
            raise RunResumeError(
                "resume_context_mismatch",
                "the resume context does not match the original run identity",
            )
        return self._continue_waiting(context, answer, checkpoint)

    def export_waiting_checkpoint(self, run_id: str) -> dict[str, object]:
        """Export a server-owned WAITING_USER state for the durable run store."""

        state = self._waiting_checkpoints.get(run_id)
        if state is None:
            raise RunResumeError("invalid_run_state", "only a WAITING_USER run has a checkpoint")
        return _serialize_agent_checkpoint(state)

    def resume_from_checkpoint(
        self,
        context: ExecutionContext,
        answer: str,
        checkpoint: Mapping[str, object],
    ) -> AgentRunResult:
        """Restore and resume one persisted checkpoint after server authorization."""

        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")
        if type(answer) is not str or not answer.strip():
            raise RunResumeError("invalid_resume_input", "answer must be a non-empty string")
        if len(answer) > 8_000:
            raise RunResumeError("invalid_resume_input", "answer exceeds 8000 characters")
        state = _deserialize_agent_checkpoint(checkpoint, expected_context=context)
        if state["run_config"] != self.run_config:
            raise RunResumeError("resume_profile_mismatch", "checkpoint profile differs from the configured agent")
        self._waiting_checkpoints[context.run_id] = state
        return self._continue_waiting(context, answer, state)

    def continue_waiting_for_clarification(
        self,
        context: ExecutionContext,
        answer: str,
        checkpoint: Mapping[str, object],
    ) -> AgentRunResult:
        """Persist an incomplete clarification without calling the model again."""

        if type(answer) is not str or not answer.strip():
            raise RunResumeError("invalid_resume_input", "answer must be a non-empty string")
        if len(answer) > 8_000:
            raise RunResumeError("invalid_resume_input", "answer exceeds 8000 characters")
        state = _deserialize_agent_checkpoint(checkpoint, expected_context=context)
        waiting_action = dict(state.get("final_action") or {})
        if not str(waiting_action.get("question", "")):
            raise RunResumeError("invalid_checkpoint", "waiting question is missing")
        state["clarifications"] = tuple(state.get("clarifications", ())) + (answer.strip(),)
        elapsed = self._elapsed_seconds(state)
        state.update(
            {
                "status": "waiting_user",
                "reason": None,
                "error_code": None,
                "proposal": None,
                "final_action": waiting_action,
                "active_elapsed_before": elapsed,
                "started_at": self._clock(),
                "elapsed_ms": int(elapsed * 1000),
                "public_answer": None,
                "facts": None,
            }
        )
        self._waiting_checkpoints[context.run_id] = state
        return self._result_from_state(state, run_id=context.run_id)

    def fail_unsupported_clarification(
        self,
        context: ExecutionContext,
        answer: str,
        checkpoint: Mapping[str, object],
        *,
        note: str,
    ) -> AgentRunResult:
        """End a run whose clarification answer chose a scope no catalog metric supports.

        No model call and no SQL; the answer is the catalog's fixed note.
        """

        if type(answer) is not str or not answer.strip():
            raise RunResumeError("invalid_resume_input", "answer must be a non-empty string")
        if len(answer) > 8_000:
            raise RunResumeError("invalid_resume_input", "answer exceeds 8000 characters")
        state = _deserialize_agent_checkpoint(checkpoint, expected_context=context)
        state["clarifications"] = tuple(state.get("clarifications", ())) + (answer.strip(),)
        elapsed = self._elapsed_seconds(state)
        state.update(
            {
                "status": "failed",
                "reason": "the chosen clarification value has no supported catalog metric",
                "error_code": CLARIFICATION_VALUE_UNSUPPORTED_CODE,
                "proposal": None,
                "final_action": None,
                "elapsed_ms": int(elapsed * 1000),
                "public_answer": note,
                "facts": None,
            }
        )
        self._waiting_checkpoints.pop(context.run_id, None)
        return self._result_from_state(state, run_id=context.run_id)

    @classmethod
    def prepared_waiting_user_checkpoint(
        cls,
        context: ExecutionContext,
        question: str,
        *,
        run_config: RunConfig,
        metric_bindings: Sequence[MetricBinding] = (),
        retrieval_items: Sequence[Mapping[str, object]] = (),
        clarifications: Sequence[str] = (),
        waiting_question: str | None = None,
        request_time_window: Mapping[str, object] | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> dict[str, object]:
        """Build an initial C10 prepared state without claiming prior model calls.

        This is used by the W05 harness when loading a declared WAITING_USER
        fixture.  It is not reachable from an HTTP request.
        """

        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")
        if type(question) is not str or not question.strip():
            raise TypeError("question must be a non-empty string")
        if not isinstance(run_config, RunConfig):
            raise TypeError("run_config must be a RunConfig")
        if any(not isinstance(item, MetricBinding) for item in metric_bindings):
            raise TypeError("metric_bindings must be server-created bindings")
        items = tuple(dict(item) for item in retrieval_items if isinstance(item, Mapping))
        if len(items) != len(retrieval_items):
            raise TypeError("retrieval_items must contain mappings")
        answers = tuple(clarifications)
        if any(type(item) is not str or not item.strip() for item in answers):
            raise TypeError("clarifications must contain non-empty strings")
        pending_question = question if waiting_question is None else waiting_question
        if type(pending_question) is not str or not pending_question.strip():
            raise TypeError("waiting_question must be a non-empty string")
        state: _GraphState = {
            "context": context,
            "question": question.strip(),
            "run_config": run_config,
            "metric_bindings": tuple(metric_bindings),
            "request_time_window": _request_window(request_time_window),
            "parallel_plan": None,
            "started_at": clock(),
            "active_elapsed_before": 0.0,
            "clarifications": answers,
            "clarification_bounce_count": 0,
            "answer_bounce_count": 0,
            "status": "waiting_user",
            "reason": None,
            "error_code": None,
            "proposal": None,
            "final_action": {"type": "ask_user", "question": pending_question.strip()},
            "model_call_count": 0,
            "tool_call_count": 0,
            "repair_count": 0,
            "model_call_ids": (),
            "events": (),
            "retrieval_items": items,
            "tool_results": (),
            "public_answer": None,
            "facts": None,
            "context_version": CONTEXT_VERSION,
            "elapsed_ms": 0,
        }
        # W05 frozen-case fixtures stay on the v3 restore path (controller, B3c-2).
        return _serialize_agent_checkpoint(state, checkpoint_version=_V3_AGENT_CHECKPOINT_VERSION)

    def _continue_waiting(
        self,
        context: ExecutionContext,
        answer: str,
        checkpoint: _GraphState,
    ) -> AgentRunResult:
        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")
        if type(answer) is not str or not answer.strip():
            raise RunResumeError("invalid_resume_input", "answer must be a non-empty string")
        if len(answer) > 8_000:
            raise RunResumeError("invalid_resume_input", "answer exceeds 8000 characters")
        original_context = checkpoint.get("context")
        if original_context != context:
            raise RunResumeError(
                "resume_context_mismatch",
                "the resume context does not match the original run identity",
            )

        resumed_state: _GraphState = dict(checkpoint)
        resumed_state.update(
            {
                "status": "running",
                "reason": None,
                "error_code": None,
                "proposal": None,
                "final_action": None,
                "public_answer": None,
                "facts": None,
                "clarifications": tuple(checkpoint.get("clarifications", ())) + (answer.strip(),),
                "started_at": self._clock(),
            }
        )
        final_state = self._compiled_graph.invoke(resumed_state)
        result = self._result_from_state(final_state, run_id=context.run_id)
        if result.status == "waiting_user":
            self._waiting_checkpoints[context.run_id] = final_state
        else:
            self._waiting_checkpoints.pop(context.run_id, None)
        return result

    def _result_from_state(self, final_state: Mapping[str, object], *, run_id: str) -> AgentRunResult:
        return AgentRunResult(
            status=_result_status(final_state.get("status", "failed")),
            run_id=run_id,
            run_config=final_state.get("run_config", self.run_config),
            reason=final_state.get("reason"),
            error_code=final_state.get("error_code"),
            action=final_state.get("final_action"),
            answer=final_state.get("public_answer"),
            answer_status=final_state.get("answer_status"),
            facts=final_state.get("facts"),
            model_call_ids=tuple(final_state.get("model_call_ids", ())),
            model_call_count=int(final_state.get("model_call_count", 0)),
            tool_call_count=int(final_state.get("tool_call_count", 0)),
            repair_count=int(final_state.get("repair_count", 0)),
            usage_summary=_usage_summary(final_state.get("events", ())),
            events=tuple(final_state.get("events", ())),
            context_version=str(final_state.get("context_version", CONTEXT_VERSION)),
            elapsed_ms=int(final_state.get("elapsed_ms", 0)),
        )

    def _build_graph(self) -> Any:
        builder = StateGraph(_GraphState)
        builder.add_node("model_decision", self._model_decision)
        builder.add_node("execute_tool", self._execute_tool)
        builder.add_node("execute_parallel", self._execute_parallel)
        builder.add_node("finish", self._finish)
        builder.add_edge(START, "model_decision")
        builder.add_conditional_edges(
            "model_decision",
            self._after_model,
            {"tool": "execute_tool", "parallel": "execute_parallel", "finish": "finish", "retry": "model_decision"},
        )
        builder.add_edge("execute_tool", "model_decision")
        builder.add_edge("execute_parallel", "model_decision")
        builder.add_conditional_edges(
            "finish",
            self._after_finish,
            {"model": "model_decision", "end": END},
        )
        return builder.compile()

    def _model_decision(self, state: _GraphState) -> dict[str, object]:
        """One model step; ``retry_model`` is set only by the action-type repair and cleared here."""

        update = self._model_decision_step(state)
        if "retry_model" not in update:
            update = {**update, "retry_model": False}
        return update

    def _model_decision_step(self, state: _GraphState) -> dict[str, object]:
        if state.get("status") != "running":
            return {}
        budget_update = self._budget_update(state, kind="model")
        if budget_update is not None:
            return budget_update

        try:
            metric_bindings = tuple(state.get("metric_bindings", ()))
            confirmed_metric = None
            confirmed_time_window = None
            if len(metric_bindings) == 1 and isinstance(metric_bindings[0], MetricBinding):
                confirmed_metric = metric_bindings[0].metric_id.removeprefix("metric.")
                confirmed_time_window = dict(metric_bindings[0].time_window)
            context_result = build_context(
                state["context"],
                state["question"],
                clarifications=state.get("clarifications", ()),
                confirmed_metric=confirmed_metric,
                time_window=confirmed_time_window,
                metric_bindings=tuple(binding.as_dict() for binding in metric_bindings if isinstance(binding, MetricBinding)),
                retrieval_items=state.get("retrieval_items", ()),
                tool_results=state.get("tool_results", ()),
                run_config=state["run_config"],
                metric_catalog=getattr(self.tools, "catalog", None),
                request_time_window=state.get("request_time_window"),
                parallel_available=self._parallel_available(state),
                retrieval_available=self.retrieval_available,
            )
        except ContextBuildError as exc:
            return self._failure_update(state, code=exc.code, reason=str(exc))

        identity = self.call_store.new_call(state["context"].run_id)
        next_count = state.get("model_call_count", 0) + 1
        next_ids = state.get("model_call_ids", ()) + (identity.model_call_id,)

        try:
            result = self.model.complete(
                context_result.messages,
                request_id=identity.request_id,
                model_call_id=identity.model_call_id,
            )
        except ModelProviderError as exc:
            event = self._event(
                state,
                {
                    "kind": "model_call",
                    "status": "failed",
                    "model_call_id": identity.model_call_id,
                    "request_id": identity.request_id,
                    "error_code": exc.code,
                    **_provider_failure_fields(exc.record),
                },
            )
            return {
                "status": "failed",
                "reason": "model provider call failed",
                "error_code": exc.code,
                "model_call_count": next_count,
                "model_call_ids": next_ids,
                "events": event,
                "context_version": context_result.context_version,
                "run_config": state["run_config"],
            }

        event = self._event(
            state,
            {
                "kind": "model_call",
                "status": "succeeded",
                "model_call_id": identity.model_call_id,
                "request_id": identity.request_id,
                "provider": result.provider,
                "model": result.model,
                "provider_call_id": result.provider_call_id,
                "provider_request_id": result.provider_request_id,
                "usage_status": result.usage_status,
                "usage": result.usage.as_dict() if result.usage is not None else None,
                "content_length": len(result.content.encode("utf-8")),
                "content_sha256": sha256(result.content.encode("utf-8")).hexdigest(),
            },
        )
        try:
            proposal = parse_query_proposal(
                result.content,
                context=state["context"],
                model_call_id=identity.model_call_id,
            )
        except ProposalParseError as exc:
            # Diagnosable without storing model text: the server's fixed
            # explanation plus a value-free structure summary.
            validation_event = self._event(
                {**state, "events": event},
                {
                    "kind": "proposal_validation",
                    "status": "failed",
                    "error_code": exc.code,
                    "error_detail": parse_error_detail(exc),
                    "action_shape": proposal_shape_summary(result.content),
                    "model_call_id": identity.model_call_id,
                },
            )
            if isinstance(exc, ToolNameAsActionTypeError):
                if state.get("repair_count", 0) < MAX_QUERY_REPAIRS:
                    return self._action_type_repair(
                        state,
                        exc,
                        events=validation_event,
                        model_call_count=next_count,
                        model_call_ids=next_ids,
                        context_version=context_result.context_version,
                    )
            return {
                "status": "failed",
                "reason": str(exc),
                "error_code": exc.code,
                "model_call_count": next_count,
                "model_call_ids": next_ids,
                "events": validation_event,
                "context_version": context_result.context_version,
                "run_config": state["run_config"],
            }

        return {
            "status": "running",
            "proposal": proposal,
            "model_call_count": next_count,
            "model_call_ids": next_ids,
            "events": event,
            "context_version": context_result.context_version,
            "run_config": state["run_config"],
        }

    def _action_type_repair(
        self,
        state: _GraphState,
        exc: ToolNameAsActionTypeError,
        *,
        events: tuple[Mapping[str, object], ...],
        model_call_count: int,
        model_call_ids: tuple[str, ...],
        context_version: str,
    ) -> dict[str, object]:
        """Spend the single repair on a tool name written as the action type.

        The hint is fixed text plus the server's own tool name; it is recorded
        as an action-validation note, not as an executed tool.
        """

        repair_index = state.get("repair_count", 0) + 1
        repair_events = self._event(
            {**state, "events": events},
            {
                "kind": "query_repair",
                "status": "scheduled",
                "repair_index": repair_index,
                "error_code": exc.code,
            },
        )
        hint_record = {
            "tool_name": "action_validation",
            "status": "failed",
            "error_code": exc.code,
            "repairable": True,
            "repair_hint": {
                "action": (
                    'Resend as {"type":"tool_call","name":<tool_name>,"arguments":{...}}; type is only one of: '
                    + ", ".join(available_action_types(parallel_available=self._parallel_available(state)))
                    + "."
                ),
                "tool_name": exc.tool_name,
            },
        }
        return {
            "status": "running",
            "proposal": None,
            "retry_model": True,
            "repair_count": repair_index,
            "model_call_count": model_call_count,
            "model_call_ids": model_call_ids,
            "tool_results": state.get("tool_results", ()) + (hint_record,),
            "events": repair_events,
            "context_version": context_version,
            "run_config": state["run_config"],
        }

    def _execute_tool(self, state: _GraphState) -> dict[str, object]:
        budget_update = self._budget_update(state, kind="tool")
        if budget_update is not None:
            return budget_update

        proposal = state.get("proposal")
        if proposal is None or not isinstance(proposal.action, ToolCallAction):
            return self._failure_update(
                state,
                code="invalid_tool_transition",
                reason="the graph reached the tool node without a tool proposal",
            )

        action = proposal.action
        next_count = state.get("tool_call_count", 0) + 1
        tool_started = self._clock()
        catalog = getattr(self.tools, "catalog", None)
        input_summary = _tool_input_summary(
            action,
            declarable_metrics=declarable_metric_ids(catalog) if catalog is not None else (),
        )
        try:
            if action.name == "search_catalog" and not self.retrieval_available:
                raise ToolError("retrieval_unavailable", "this server runs without a catalog retriever")
            output = call_tool(
                self.tools,
                action.name,
                action.arguments,
                context=state["context"],
                metric_bindings=state.get("metric_bindings", ()),
                request_time_window=state.get("request_time_window"),
                clarifications=self._clarification_reading(state),
            )
        except MetricContradictsQuestionError as exc:
            return self._contradiction_update(
                state,
                exc,
                tool_call_count=next_count,
                tool_name=action.name,
                input_summary=input_summary,
                elapsed_ms=int(max(0.0, self._clock() - tool_started) * 1000),
            )
        except (ClarificationRequiredError, ClarificationValueUnsupportedError) as exc:
            return self._clarification_gate_update(
                state,
                exc,
                tool_call_count=next_count,
                tool_name=action.name,
                input_summary=input_summary,
                elapsed_ms=int(max(0.0, self._clock() - tool_started) * 1000),
            )
        except ApprovalRequiredError as exc:
            # A verified read of approval-protected values pauses the run; the
            # server binds the pending call to an approval.  Nothing executed.
            return {
                "status": "waiting_approval",
                "reason": "the verified query reads values that require approval",
                "error_code": None,
                "final_action": {"type": "approval_required", "tool_call": dict(exc.pending_call)},
                "tool_call_count": next_count,
                "events": self._event(
                    state,
                    {
                        "kind": "tool_call",
                        "status": "approval_required",
                        "tool_call_index": next_count,
                        "tool_name": action.name,
                        "error_code": exc.code,
                        "elapsed_ms": int(max(0.0, self._clock() - tool_started) * 1000),
                        "input_summary": input_summary,
                        "policy_conclusion": "approval_required",
                    },
                ),
            }
        except ToolError as exc:
            repairable = action.name == "query_readonly" and exc.code in _REPAIRABLE_QUERY_ERRORS
            next_repairs = state.get("repair_count", 0) + (1 if repairable else 0)
            failed_event = self._event(
                state,
                {
                    "kind": "tool_call",
                    "status": "failed",
                    "tool_call_index": next_count,
                    "tool_name": action.name,
                    "error_code": exc.code,
                    "error_reason": exc.message,
                    "elapsed_ms": int(max(0.0, self._clock() - tool_started) * 1000),
                    "input_summary": input_summary,
                    "policy_conclusion": "rejected",
                    **self._metadata_transport_fields(action.name),
                },
            )
            failed_record = {
                "tool_name": action.name,
                "status": "failed",
                "error_code": exc.code,
                "error_reason": exc.message,
                "repairable": repairable,
                "input_summary": input_summary,
            }
            next_results = state.get("tool_results", ()) + (failed_record,)
            if repairable and state.get("repair_count", 0) < MAX_QUERY_REPAIRS:
                repair_event = self._event(
                    {**state, "events": failed_event},
                    {
                        "kind": "query_repair",
                        "status": "scheduled",
                        "repair_index": next_repairs,
                        "error_code": exc.code,
                    },
                )
                return {
                    "status": "running",
                    "tool_call_count": next_count,
                    "repair_count": next_repairs,
                    "tool_results": next_results,
                    "events": repair_event,
                }
            if repairable:
                return {
                    "status": "failed",
                    "reason": "the query repair budget is exhausted",
                    "error_code": "query_repair_limit",
                    "tool_call_count": next_count,
                    "repair_count": state.get("repair_count", 0),
                    "tool_results": next_results,
                    "events": failed_event,
                }
            terminal_status: _InternalStatus = (
                "denied" if exc.code in _SECURITY_REFUSAL_ERRORS else "failed"
            )
            return {
                "status": terminal_status,
                "reason": str(exc),
                "error_code": exc.code,
                "tool_call_count": next_count,
                "tool_results": next_results,
                "events": failed_event,
            }

        tool_record = {
            "tool_name": action.name,
            "status": "succeeded",
            "output": output,
        }
        next_results = state.get("tool_results", ()) + (tool_record,)
        update: dict[str, object] = {
            "status": "running",
            "tool_call_count": next_count,
            "tool_results": next_results,
            "events": self._event(
                state,
                {
                    "kind": "tool_call",
                    "status": "succeeded",
                    "tool_call_index": next_count,
                    "tool_name": action.name,
                    "elapsed_ms": int(max(0.0, self._clock() - tool_started) * 1000),
                    "input_summary": input_summary,
                    "policy_conclusion": "allowed",
                    "result_id": output.get("result_id"),
                    "row_count": output.get("row_count"),
                    "source_ids": _source_ids_from_tool_output(output),
                    **self._metadata_transport_fields(action.name),
                },
            ),
        }
        if action.name == "search_catalog":
            items = output.get("items")
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
                update["retrieval_items"] = tuple(
                    dict(item) for item in items[:3] if isinstance(item, Mapping)
                )
        return update

    def _metadata_transport_fields(self, tool_name: str) -> dict[str, object]:
        """How a metadata call travelled; empty for the local facade, so local events never change."""

        if tool_name not in {"search_catalog", "describe_tables"}:
            return {}
        take = getattr(self.tools, "take_metadata_call_record", None)
        record = take() if callable(take) else None
        return dict(record) if isinstance(record, Mapping) else {}

    def _clarification_reading(self, state: Mapping[str, object]) -> ClarificationReading | None:
        """The catalog phrase table read against this run's question and answers."""

        catalog = getattr(self.tools, "catalog", None)
        if catalog is None or not catalog.has_phrase_table:
            return None
        return read_clarifications(
            catalog,
            str(state.get("question", "")),
            tuple(state.get("clarifications", ())),
            confirmed_metrics=tuple(
                binding.metric_id for binding in state.get("metric_bindings", ()) if isinstance(binding, MetricBinding)
            ),
        )

    def _clarification_gate_update(
        self,
        state: _GraphState,
        exc: ClarificationRequiredError | ClarificationValueUnsupportedError,
        *,
        tool_call_count: int,
        tool_name: str,
        input_summary: Mapping[str, object],
        elapsed_ms: int,
    ) -> dict[str, object]:
        """A declared metric stopped by the phrase table before any SQL ran.

        Wording left open: wait for the user on the catalog rule's fixed
        question, without another model call.  Unsupported scope: fail with
        the catalog's fixed note (not repairable).
        """

        event = {
            "kind": "tool_call",
            "tool_call_index": tool_call_count,
            "tool_name": tool_name,
            "error_code": exc.code,
            "clarification_id": exc.rule.id,
            "elapsed_ms": elapsed_ms,
            "input_summary": dict(input_summary),
        }
        if isinstance(exc, ClarificationRequiredError):
            elapsed = self._elapsed_seconds(state)
            return {
                "status": "waiting_user",
                "reason": None,
                "error_code": None,
                "final_action": {"type": "ask_user", "question": exc.rule.question, "clarification_id": exc.rule.id},
                "tool_call_count": tool_call_count,
                "active_elapsed_before": elapsed,
                "started_at": self._clock(),
                "events": self._event(
                    state,
                    {**event, "status": "clarification_required", "policy_conclusion": "clarification_required"},
                ),
            }
        return {
            "status": "failed",
            "reason": "the requested scope has no supported catalog metric",
            "error_code": exc.code,
            "final_action": None,
            "public_answer": exc.note,
            "tool_call_count": tool_call_count,
            "tool_results": state.get("tool_results", ())
            + ({"tool_name": tool_name, "status": "failed", "error_code": exc.code, "repairable": False},),
            "events": self._event(
                state,
                {**event, "status": "failed", "policy_conclusion": "clarification_unsupported"},
            ),
        }

    def _phrase_bounce(
        self,
        state: _GraphState,
        events: tuple[Mapping[str, object], ...],
        *,
        error_code: str,
        tool_name: str,
        hint: Mapping[str, object],
        reason: str,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Send the model back once when it disagrees with the phrase table.

        An ask the wording settles and a declaration the question contradicts
        share one bounce per run (``clarification_bounce_count``, checkpointed,
        never the query repair budget); the second disagreement fails.
        ``events`` already records the rejected action.
        """

        bounces = state.get("clarification_bounce_count", 0)
        if bounces >= MAX_CLARIFICATION_BOUNCES:
            return {
                **(extra or {}),
                "status": "failed",
                "reason": reason,
                "error_code": error_code,
                "final_action": None,
                "events": events,
            }
        return {
            **(extra or {}),
            "status": "running",
            "proposal": None,
            "final_action": None,
            "clarification_bounce_count": bounces + 1,
            "tool_results": state.get("tool_results", ())
            + (
                {
                    "tool_name": tool_name,
                    "status": "failed",
                    "error_code": error_code,
                    "repairable": True,
                    "repair_hint": dict(hint),
                },
            ),
            "events": self._event(
                {**state, "events": events},
                {"kind": "clarification_bounce", "status": "scheduled", "bounce_index": bounces + 1, "error_code": error_code},
            ),
        }

    def _contradiction_update(
        self,
        state: _GraphState,
        exc: MetricContradictsQuestionError,
        *,
        tool_call_count: int,
        tool_name: str,
        input_summary: Mapping[str, object],
        elapsed_ms: int,
    ) -> dict[str, object]:
        """A declaration the question's wording contradicts, stopped before any SQL ran."""

        events = self._event(
            state,
            {
                "kind": "tool_call",
                "status": "rejected",
                "tool_call_index": tool_call_count,
                "tool_name": tool_name,
                "error_code": exc.code,
                "clarification_id": exc.rule.id,
                "elapsed_ms": elapsed_ms,
                "input_summary": dict(input_summary),
                "policy_conclusion": "declaration_contradicts_question",
            },
        )
        reading = self._clarification_reading(state)
        hint = (
            contradiction_hint(reading, exc.rule, state.get("request_time_window"))
            if reading is not None
            else {"action": "Declare the metric the question names.", "clarification_id": exc.rule.id}
        )
        return self._phrase_bounce(
            state,
            events,
            error_code=exc.code,
            tool_name=tool_name,
            hint=hint,
            reason="the declared metric contradicts the question twice or after another bounce",
            extra={"tool_call_count": tool_call_count},
        )

    def _parallel_available(self, state: _GraphState) -> bool:
        """Parallel reads exist only with a server scheduler and a server plan for this run."""

        return self.parallel_scheduler is not None and isinstance(state.get("parallel_plan"), ParallelPlan)

    def _parallel_unavailable_repair(
        self,
        state: _GraphState,
        action: ParallelReadonlyAction,
    ) -> dict[str, object]:
        """Spend the single repair: ask for one declared query instead of a parallel group.

        Only catalog-checked declarable metric ids and the server request window
        are echoed; the note is not an executed tool and adds no tool call.
        """

        validation_events = self._event(
            state,
            {"kind": "action_validation", "status": "failed", "error_code": "parallel_unavailable"},
        )
        repair_index = state.get("repair_count", 0) + 1
        repair_events = self._event(
            {**state, "events": validation_events},
            {
                "kind": "query_repair",
                "status": "scheduled",
                "repair_index": repair_index,
                "error_code": "parallel_unavailable",
            },
        )
        declarable = declarable_metric_ids(self.tools.catalog)
        window = state.get("request_time_window")
        hint_record = {
            "tool_name": "action_validation",
            "status": "failed",
            "error_code": "parallel_unavailable",
            "repairable": True,
            "repair_hint": {
                "action": (
                    "This run cannot run parallel reads. Send one tool_call named query_readonly and declare "
                    "these metrics together in arguments.metrics with time_window (net_fen is declared alone)."
                ),
                "declare_metrics": [metric_id for metric_id in action.metric_ids if metric_id in declarable],
                "request_time_window": (
                    {"start": window["start"], "end": window["end"]} if window is not None else None
                ),
            },
        }
        return {
            "status": "running",
            "proposal": None,
            "repair_count": repair_index,
            "tool_results": state.get("tool_results", ()) + (hint_record,),
            "events": repair_events,
        }

    def _execute_parallel(self, state: _GraphState) -> dict[str, object]:
        proposal = state.get("proposal")
        if proposal is None or not isinstance(proposal.action, ParallelReadonlyAction):
            return self._failure_update(
                state,
                code="invalid_parallel_transition",
                reason="the graph reached the parallel node without a parallel proposal",
            )
        if not self._parallel_available(state):
            if state.get("repair_count", 0) < MAX_QUERY_REPAIRS:
                return self._parallel_unavailable_repair(state, proposal.action)
            return self._failure_update(
                state,
                code="parallel_unavailable",
                reason="parallel execution requires a server-owned scheduler and plan",
            )
        try:
            check_clarification(self._clarification_reading(state), proposal.action.metric_ids)
        except MetricContradictsQuestionError as exc:
            return self._contradiction_update(
                state,
                exc,
                tool_call_count=state.get("tool_call_count", 0),
                tool_name="parallel_readonly",
                input_summary={"metric_ids": list(proposal.action.metric_ids)},
                elapsed_ms=0,
            )
        except (ClarificationRequiredError, ClarificationValueUnsupportedError) as exc:
            return self._clarification_gate_update(
                state,
                exc,
                tool_call_count=state.get("tool_call_count", 0),
                tool_name="parallel_readonly",
                input_summary={"metric_ids": list(proposal.action.metric_ids)},
                elapsed_ms=0,
            )
        remaining_tool_budget = self.limits.max_tool_calls - state.get("tool_call_count", 0)
        try:
            result = self.parallel_scheduler.run(
                state["context"],
                proposal.action,
                plan=state["parallel_plan"],
                tool_budget_remaining=remaining_tool_budget,
            )
        except ParallelValidationError as exc:
            return self._failure_update(state, code=exc.code, reason=str(exc))
        next_tool_count = state.get("tool_call_count", 0) + result.new_branch_count
        event = self._event(
            state,
            {
                "kind": "parallel_group",
                "status": result.status,
                "group_id": result.group_id,
                "plan_hash": result.plan_hash,
                "branch_ids": [branch.branch_id for branch in result.branches],
                "branch_statuses": [branch.status for branch in result.branches],
                "peak_active": result.peak_active,
                "reused": result.reused,
                "error_code": result.error_code,
            },
        )
        update: dict[str, object] = {
            "tool_call_count": next_tool_count,
            "events": event,
            "tool_results": state.get("tool_results", ())
            + ({"tool_name": "parallel_readonly", "status": result.status, "output": result.as_dict()},),
        }
        if result.status == "LIMIT_REACHED":
            return {
                **update,
                "status": "limit_reached",
                "reason": "parallel branches exceed remaining tool-call budget",
                "error_code": result.error_code or "parallel_budget_insufficient",
            }
        if result.status != "SUCCEEDED":
            return {
                **update,
                "status": "failed",
                "reason": "one or more parallel branches failed",
                "error_code": "parallel_branch_failed",
            }
        return {**update, "status": "running"}

    def _finish(self, state: _GraphState) -> dict[str, object]:
        elapsed_seconds = self._elapsed_seconds(state)
        update: dict[str, object] = {"elapsed_ms": int(elapsed_seconds * 1000)}
        if state.get("status") != "running":
            return update

        proposal = state.get("proposal")
        if proposal is None:
            return {
                **update,
                "status": "failed",
                "reason": "the graph finished without a validated proposal",
                "error_code": "missing_proposal",
            }

        action = proposal.action
        if isinstance(action, FinalAnswerAction):
            try:
                public_action, public_answer, facts, answer_status = self._verified_answer(state, action)
            except (FactResolutionError, ToolError) as exc:
                failure_event = self._event(
                    state,
                    {
                        "kind": "answer_validation",
                        "status": "failed",
                        "error_code": getattr(exc, "code", "evidence_validation_failed"),
                        "field": "fact_refs",
                        **_basis_diagnostics(action),
                    },
                )
                if isinstance(exc, _UndeclaredMetricReference) and state.get("repair_count", 0) < MAX_QUERY_REPAIRS:
                    return self._answer_repair(state, exc, failure_event)
                can_bounce = (
                    state.get("answer_bounce_count", 0) < MAX_ANSWER_BOUNCES
                    # No model call left for a corrected answer: fail with the
                    # answer's own code instead of a model_call_limit stop.
                    and state.get("model_call_count", 0) < self.limits.max_model_calls
                )
                if isinstance(exc, _KnowledgeWithoutSource) and can_bounce and self.retrieval_available:
                    return self._server_search_bounce(state, action, exc, failure_event, update)
                if isinstance(exc, _ANSWER_BOUNCE_ERRORS) and can_bounce:
                    return self._answer_bounce(state, action, exc, failure_event)
                return {
                    **update,
                    "status": "failed",
                    "reason": str(exc),
                    "error_code": getattr(exc, "code", "evidence_validation_failed"),
                    "events": failure_event,
                }
            answer_event = self._event(
                state,
                {
                    "kind": "answer",
                    "status": answer_status,
                    "source_ids": list(public_action.get("source_ids", ())),
                    "fact_ref_count": len(action.fact_refs),
                    "fact_ids": [fact["fact_id"] for fact in facts.get("facts", [])],
                    "draft_status": "unverified",
                    "draft_sha256": proposal.content_sha256,
                    **_basis_diagnostics(action),
                },
            )
            return {
                **update,
                "status": "succeeded",
                "final_action": public_action,
                "public_answer": public_answer,
                "answer_status": answer_status,
                "facts": facts,
                "events": answer_event,
            }
        if isinstance(action, AskUserAction):
            return {**update, **self._reviewed_ask(state, action, elapsed_seconds)}
        if isinstance(action, DenyAction):
            return {**update, "status": "denied", "final_action": action.as_dict()}
        return {
            **update,
            "status": "failed",
            "reason": "a tool proposal bypassed the tool node",
            "error_code": "invalid_terminal_action",
        }

    def _reviewed_ask(self, state: _GraphState, action: AskUserAction, elapsed_seconds: float) -> dict[str, object]:
        """Check an ask_user against the phrase table before the run waits.

        A catalog rule the wording leaves open: wait on the rule's fixed
        question (model text is not kept).  A rule the wording already
        settles (or a single-value rule): send the model back once, then
        fail.  No catalog rule named: the model's own question, as before.
        """

        reading = self._clarification_reading(state)
        verdict = review_ask(reading, action.clarification_id, action.question) if reading is not None else None
        waiting: dict[str, object] = {
            "status": "waiting_user",
            "active_elapsed_before": elapsed_seconds,
            "started_at": self._clock(),
            "final_action": {"type": "ask_user", "question": action.question},
        }
        if verdict is None:
            return waiting
        review_event = {
            "kind": "clarification_review",
            "status": "rejected" if verdict.decision == "not_needed" else "allowed",
            "decision": verdict.decision,
            "clarification_ids": list(verdict.targeted),
            "id_status": verdict.id_status,
            # Which signal the decision rests on (fixed identifier, never ask text).
            "signal": verdict.signal,
        }
        if verdict.decision == "not_needed" and verdict.rule is not None:
            events = self._event(state, {**review_event, "error_code": CLARIFICATION_NOT_NEEDED_CODE})
            return self._phrase_bounce(
                state,
                events,
                error_code=CLARIFICATION_NOT_NEEDED_CODE,
                tool_name="ask_user",
                hint=not_needed_hint(reading, verdict.rule, state.get("request_time_window")),
                reason="the question already settles the clarification and the bounce budget is used",
            )
        if verdict.decision == "catalog_question" and verdict.rule is not None:
            waiting["final_action"] = {
                "type": "ask_user",
                "question": verdict.rule.question,
                "clarification_id": verdict.rule.id,
            }
        return {**waiting, "events": self._event(state, review_event)}

    def _answer_repair(
        self,
        state: _GraphState,
        exc: "_UndeclaredMetricReference",
        failure_event: tuple[Mapping[str, object], ...],
    ) -> dict[str, object]:
        """Spend the single repair: send the model back to query/declare, never create facts."""

        repair_index = state.get("repair_count", 0) + 1
        repair_event = self._event(
            {**state, "events": failure_event},
            {
                "kind": "query_repair",
                "status": "scheduled",
                "repair_index": repair_index,
                "error_code": exc.code,
            },
        )
        hint_record = {
            "tool_name": "final_answer",
            "status": "failed",
            "error_code": exc.code,
            "repairable": True,
            "repair_hint": undeclared_metric_hint(
                self.tools.catalog,
                exc.metric_ids,
                state.get("request_time_window"),
            ),
        }
        return {
            "status": "running",
            "proposal": None,
            "final_action": None,
            "repair_count": repair_index,
            "tool_results": state.get("tool_results", ()) + (hint_record,),
            "events": repair_event,
        }

    def _answer_bounce(
        self,
        state: _GraphState,
        action: FinalAnswerAction,
        exc: "_UngroundedAnswer | _AnswerBasisConflict | _AnswerWithoutQueryResult",
        failure_event: tuple[Mapping[str, object], ...],
        *,
        hint: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Send an ungrounded answer back once (B3c-2); never the SQL repair.

        The hint is fixed text plus the server request window; the model's
        answer text is not kept, echoed or returned.
        """

        bounce_index = state.get("answer_bounce_count", 0) + 1
        events = self._event(
            {**state, "events": failure_event},
            {
                "kind": "answer_bounce",
                "status": "scheduled",
                "bounce_index": bounce_index,
                "error_code": exc.code,
                **_basis_diagnostics(action),
            },
        )
        window = state.get("request_time_window")
        if hint is None and isinstance(exc, _AnswerWithoutQueryResult):
            hint = answer_without_query_hint(window)
        elif hint is None and isinstance(exc, _AnswerBasisConflict):
            hint = answer_basis_conflict_hint(window)
        elif hint is None:
            hint = answer_not_grounded_hint(window, retrieval_available=self.retrieval_available)
        hint_record = {
            "tool_name": "final_answer",
            "status": "failed",
            "error_code": exc.code,
            "repairable": True,
            "repair_hint": hint,
        }
        return {
            "status": "running",
            "proposal": None,
            "final_action": None,
            "answer_bounce_count": bounce_index,
            "tool_results": state.get("tool_results", ()) + (hint_record,),
            "events": events,
        }

    def _server_search_bounce(
        self,
        state: _GraphState,
        action: FinalAnswerAction,
        exc: "_KnowledgeWithoutSource",
        failure_event: tuple[Mapping[str, object], ...],
        update: Mapping[str, object],
    ) -> dict[str, object]:
        """A knowledge answer with no retrieval source: the server searches once (B3c-2 R2).

        The search uses this run's question, the same retriever, identity and
        top_k as the model's search_catalog, and counts as a tool call.  Its
        result goes to the model as an untrusted tool result, then the answer
        is sent back once for a knowledge answer from those items.  Retrieval,
        not routing: the server picks no metric or action from it.  No
        usable source (or no tool call left, or a retrieval error): fail.
        Events keep fixed identifiers only, never the question or item text.
        """

        next_count = state.get("tool_call_count", 0) + 1
        event: dict[str, object] = {
            "kind": "tool_call",
            "tool_name": "search_catalog",
            "initiated_by": "server",
            "tool_call_index": next_count,
            "input_summary": {
                "argument_keys": ["query", "top_k"],
                "top_k": SERVER_SEARCH_TOP_K,
                "query_source": "run_question",
            },
        }
        failed = {
            **update,
            "status": "failed",
            "reason": str(exc),
            "error_code": exc.code,
        }
        if state.get("tool_call_count", 0) >= self.limits.max_tool_calls:
            events = self._event(
                {**state, "events": failure_event},
                {**event, "status": "skipped", "error_code": "tool_call_limit"},
            )
            return {**failed, "events": events}
        started = self._clock()
        try:
            output = call_tool(
                self.tools,
                "search_catalog",
                {"query": str(state.get("question", ""))[:SERVER_SEARCH_MAX_QUERY_CHARS], "top_k": SERVER_SEARCH_TOP_K},
                context=state["context"],
                metric_bindings=state.get("metric_bindings", ()),
                request_time_window=state.get("request_time_window"),
                clarifications=self._clarification_reading(state),
            )
        except ToolError as tool_error:
            events = self._event(
                {**state, "events": failure_event},
                {
                    **event,
                    "status": "failed",
                    "error_code": tool_error.code,
                    "elapsed_ms": int(max(0.0, self._clock() - started) * 1000),
                    **self._metadata_transport_fields("search_catalog"),
                },
            )
            if tool_error.code in MCP_ERROR_CODES:
                # The MCP metadata server failed (MCP setting only): the run fails with that code.
                failed = {**failed, "reason": str(tool_error), "error_code": tool_error.code}
            return {**failed, "tool_call_count": next_count, "events": events}
        record = {"tool_name": "search_catalog", "status": "succeeded", "initiated_by": "server", "output": output}
        grounding = _run_retrieval_source_ids({"tool_results": (record,)}, excluded_ids=self._table_entry_ids())
        events = self._event(
            {**state, "events": failure_event},
            {
                **event,
                "status": "succeeded",
                "elapsed_ms": int(max(0.0, self._clock() - started) * 1000),
                "source_ids": _source_ids_from_tool_output(output),
                "grounding_source_count": len(grounding),
                **self._metadata_transport_fields("search_catalog"),
            },
        )
        if not grounding:
            return {**failed, "tool_call_count": next_count, "events": events}
        searched: dict[str, object] = {
            **state,
            "tool_call_count": next_count,
            "tool_results": state.get("tool_results", ()) + (record,),
        }
        items = output.get("items")
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
            searched["retrieval_items"] = tuple(dict(item) for item in items[:3] if isinstance(item, Mapping))
        # Knowledge only after the server's own search: no query / no_data menu.
        bounced = self._answer_bounce(
            searched, action, exc, events, hint=knowledge_from_server_search_hint()  # type: ignore[arg-type]
        )
        bounced["tool_call_count"] = next_count
        if "retrieval_items" in searched:
            bounced["retrieval_items"] = searched["retrieval_items"]
        return bounced

    def _no_data_metric_names(self) -> list[str]:
        catalog = getattr(self.tools, "catalog", None) or load_default_catalog()
        return [catalog.metric_name(metric_id) for metric_id in declarable_metric_ids(catalog)]

    def _after_finish(self, state: _GraphState) -> str:
        return "model" if state.get("status") == "running" else "end"

    def _verified_answer(
        self,
        state: _GraphState,
        action: FinalAnswerAction,
    ) -> tuple[dict[str, object], str, dict[str, object], str]:
        """Check the answer against its declared basis; return it with its answer_status.

        query (default): business values need this run's query (fact_refs to
        its results, or a successful non-metric query); retrieval or table
        sources alone ground nothing.  knowledge: a definition grounded by
        this run's retrieval sources, no fact_refs.  no_data: no query and no
        fact_refs; the server writes the reply.  Only a fully server-rendered
        answer with at least one fact is ``verified``.
        """

        required_fact_refs: set[tuple[str, str]] = set()
        required_rowset_refs: set[tuple[str, str]] = set()
        verified_source_ids: set[str] = set()
        run_evidence: dict[str, object] = {}
        trusted_bindings = {
            binding.metric_id.removeprefix("metric."): binding
            for binding in state.get("metric_bindings", ())
            if isinstance(binding, MetricBinding)
        }
        for record in state.get("tool_results", ()):
            if not isinstance(record, Mapping) or record.get("status") != "succeeded":
                continue
            output = record.get("output")
            if not isinstance(output, Mapping) or type(output.get("result_id")) is not str:
                continue
            try:
                evidence = self.tools.get_result_evidence(
                    str(output["result_id"]),
                    context=state["context"],
                )
            except ToolError as exc:
                raise FactResolutionError(
                    "evidence_validation_failed",
                    f"successful query result evidence could not be loaded: {exc.code}",
                ) from exc
            run_evidence[evidence.result_id] = evidence
            # Every binding on the evidence was built by the server (from a
            # pre-bound slot or a catalog-checked declaration) and verified
            # before execution, so each one must be cited by the answer.
            for observed_binding in evidence.metric_bindings:
                metric_id = observed_binding.metric_id.removeprefix("metric.")
                requested_binding = trusted_bindings.get(metric_id)
                if requested_binding is not None and (
                    observed_binding.catalog_source_id != requested_binding.catalog_source_id
                    or observed_binding.catalog_version != requested_binding.catalog_version
                    or observed_binding.unit != requested_binding.unit
                    or dict(observed_binding.time_window) != dict(requested_binding.time_window)
                    or observed_binding.plan_id != requested_binding.plan_id
                ):
                    raise FactResolutionError("evidence_validation_failed", "result metric binding changed unexpectedly")
                verified_source_ids.add(observed_binding.catalog_source_id)
                if is_scalar_metric_result(evidence):
                    required_fact_refs.add((evidence.result_id, metric_id))
                # Several rows, or one row of a grouped query (B3e): a rowset.
                elif evidence.row_count >= 1 and any(
                    observed_binding.result_position not in row for row in evidence.rows
                ):
                    raise FactResolutionError("evidence_validation_failed", "grouped result is missing a trusted metric column")
                elif evidence.row_count >= 1:
                    required_rowset_refs.add((evidence.result_id, metric_id))

        empty_facts = {"schema_version": FACTS_SCHEMA_VERSION, "facts": []}
        # knowledge and no_data answers never cite results (B3c-2).
        if action.fact_refs and action.basis != "query":
            raise _AnswerBasisConflict()
        if action.basis == "no_data":
            if _run_has_query_attempt(state):
                raise _AnswerBasisConflict()
            reply = render_no_data_answer(self._no_data_metric_names())
            # The model's own text never leaves the server, not even in the action.
            return {**action.as_dict(), "answer": reply, "source_ids": []}, reply, empty_facts, "no_data"

        # A reference to this run's own result for a metric the server never
        # bound: the model skipped the declaration.  Repairable once; no facts.
        undeclared = [
            reference.metric_id.removeprefix("metric.")
            for reference in action.fact_refs
            if reference.result_id in run_evidence
            and reference.metric_id.removeprefix("metric.")
            not in {item.metric_id.removeprefix("metric.") for item in run_evidence[reference.result_id].metric_bindings}
        ]
        if undeclared:
            raise _UndeclaredMetricReference(undeclared)
        # Citing results before any successful query in this run (e.g. copying
        # the contract placeholders).  With a successful query present, a
        # missing result_id stays a terminal evidence failure.
        if action.fact_refs and not run_evidence:
            raise _AnswerWithoutQueryResult()

        if not action.fact_refs:
            if action.basis == "knowledge":
                # Only this run's retrieval sources ground a definition; table
                # structure (describe_tables, catalog table entries) grounds nothing.
                public_source_ids = _run_retrieval_source_ids(state, excluded_ids=self._table_entry_ids())
                if not public_source_ids:
                    raise _KnowledgeWithoutSource()
            elif _run_has_successful_query(state):
                public_source_ids = verified_source_ids
            else:
                # Business values need a successful query in this run; retrieval
                # or table sources alone do not ground the model's numbers.
                raise _UngroundedAnswer()
            if required_fact_refs or required_rowset_refs:
                raise FactResolutionError(
                    "evidence_validation_failed",
                    "a server-bound result requires an exact fact or rowset reference",
                )
            # Model text: unverified, with the server's own source ids only.
            public_action = {**action.as_dict(), "source_ids": sorted(public_source_ids)}
            return public_action, action.answer, empty_facts, "unverified"

        supplied_fact_refs = {
            (reference.result_id, reference.metric_id.removeprefix("metric."))
            for reference in action.fact_refs
        }
        if not required_fact_refs.issubset(supplied_fact_refs) or not required_rowset_refs.issubset(supplied_fact_refs):
            raise FactResolutionError(
                "evidence_validation_failed",
                "final answer omitted one or more server-bound result references",
            )

        evidences = {}
        scalar_references = tuple(
            reference
            for reference in action.fact_refs
            if (reference.result_id, reference.metric_id.removeprefix("metric.")) not in required_rowset_refs
        )
        for reference in scalar_references:
            try:
                evidence = self.tools.get_result_evidence(
                    reference.result_id,
                    context=state["context"],
                )
            except ToolError as exc:
                raise FactResolutionError(
                    "evidence_validation_failed",
                    f"result evidence could not be loaded: {exc.code}",
                ) from exc
            evidences[reference.result_id] = evidence

        if scalar_references:
            envelope = FactResolver(catalog=self.tools.catalog).resolve(
                scalar_references,
                context=state["context"],
                evidences=evidences,
            )
            facts = envelope.as_dict()
            rendered = render_verified_answer(envelope.facts, clarifications=self._clarification_reading(state))
            verified_source_ids.update(fact.catalog_source_id for fact in envelope.facts)
        else:
            facts = {"schema_version": FACTS_SCHEMA_VERSION, "facts": []}
            rendered = action.answer
        public_action = action.as_dict()
        if scalar_references:
            public_action["answer"] = rendered
        public_action["source_ids"] = sorted(verified_source_ids)
        # A rowset-only answer keeps the model's summary: unverified (B3c-2 (a)).
        answer_status = "verified" if scalar_references and facts.get("facts") else "unverified"
        return public_action, rendered, facts, answer_status

    def _table_entry_ids(self) -> frozenset[str]:
        catalog = getattr(self.tools, "catalog", None) or load_default_catalog()
        return frozenset(entry.id for entry in catalog.entries if entry.kind == "table")

    def _after_model(self, state: _GraphState) -> str:
        if state.get("status") != "running":
            return "finish"
        if state.get("retry_model") is True:
            return "retry"
        proposal = state.get("proposal")
        if proposal is not None and isinstance(proposal.action, ToolCallAction):
            return "tool"
        if proposal is not None and isinstance(proposal.action, ParallelReadonlyAction):
            return "parallel"
        return "finish"

    def _budget_update(self, state: _GraphState, *, kind: Literal["model", "tool"]) -> dict[str, object] | None:
        if self._elapsed_seconds(state) >= float(self.limits.max_wall_clock_seconds):
            return self._limit_update(
                state,
                reason="active wall-clock limit reached",
                code="wall_clock_limit",
            )
        if kind == "model" and state.get("model_call_count", 0) >= self.limits.max_model_calls:
            return self._limit_update(
                state,
                reason="model-call limit reached",
                code="model_call_limit",
            )
        if kind == "tool" and state.get("tool_call_count", 0) >= self.limits.max_tool_calls:
            return self._limit_update(
                state,
                reason="tool-call limit reached",
                code="tool_call_limit",
            )
        return None

    def _limit_update(self, state: _GraphState, *, reason: str, code: str) -> dict[str, object]:
        return {
            "status": "limit_reached",
            "reason": reason,
            "error_code": code,
            "events": self._event(
                state,
                {"kind": "limit", "status": "stopped", "reason": reason, "error_code": code},
            ),
        }

    def _failure_update(self, state: _GraphState, *, code: str, reason: str) -> dict[str, object]:
        return {
            "status": "failed",
            "reason": reason,
            "error_code": code,
            "events": self._event(
                state,
                {"kind": "runtime", "status": "failed", "error_code": code},
            ),
        }

    def _event(self, state: _GraphState, event: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
        record = {
            "sequence": len(state.get("events", ())) + 1,
            "run_id": state["context"].run_id,
            "profile": state["run_config"].profile,
            "prompt_version": state["run_config"].prompt_version,
            "action_schema_version": state["run_config"].action_schema_version,
            "tool_description_version": state["run_config"].tool_description_version,
            "catalog_version": state["run_config"].catalog_version,
            "knowledge_snapshot_id": state["run_config"].knowledge_snapshot_id,
            **dict(event),
        }
        return state.get("events", ()) + (record,)

    def _elapsed_seconds(self, state: _GraphState) -> float:
        return float(state.get("active_elapsed_before", 0.0)) + max(
            0.0,
            self._clock() - state["started_at"],
        )

    def _elapsed_ms(self, state: _GraphState) -> int:
        return int(self._elapsed_seconds(state) * 1000)


def _basis_diagnostics(action: FinalAnswerAction) -> dict[str, str]:
    """The answer's server-normalized basis and how the model wrote it (B3c-2 R1).

    Both are fixed identifiers (ANSWER_BASES, BASIS_FIELD_STATES); never model text.
    """

    return {"basis": action.basis, "basis_field": action.basis_field}


class _AnswerWithoutQueryResult(FactResolutionError):
    """A final answer cites results although this run has no successful query yet."""

    def __init__(self) -> None:
        super().__init__(
            "answer_without_query_result",
            "a final answer cites results before any successful query_readonly result in this run",
        )


class _UndeclaredMetricReference(FactResolutionError):
    """A fact_ref cites this run's result for a metric that was never declared."""

    def __init__(self, metric_ids: Sequence[str]) -> None:
        super().__init__(
            "metric_not_declared",
            "a fact_ref cites a query result that has no verified binding for that metric",
        )
        self.metric_ids = tuple(metric_ids)


class _UngroundedAnswer(FactResolutionError):
    """Nothing in this run grounds the answer's declared basis (B3c-2).

    query: no fact_refs and no successful query; knowledge: no retrieval source.
    """

    def __init__(self) -> None:
        super().__init__(
            "answer_not_grounded",
            "nothing in this run grounds the final answer's basis",
        )


class _KnowledgeWithoutSource(_UngroundedAnswer):
    """A knowledge answer with no retrieval source in this run (R2: the server searches once)."""


class _AnswerBasisConflict(FactResolutionError):
    """The declared basis contradicts the run: no_data after a query, or cited results."""

    def __init__(self) -> None:
        super().__init__(
            "answer_basis_conflict",
            "the final answer's basis contradicts this run's queries or fact_refs",
        )


_ANSWER_BOUNCE_ERRORS = (_UngroundedAnswer, _AnswerBasisConflict, _AnswerWithoutQueryResult)


def _run_has_successful_query(state: Mapping[str, object]) -> bool:
    return any(
        isinstance(record, Mapping)
        and record.get("tool_name") in {"query_readonly", "parallel_readonly"}
        and record.get("status") in {"succeeded", "SUCCEEDED"}
        for record in state.get("tool_results", ())
    )


def _run_has_query_attempt(state: Mapping[str, object]) -> bool:
    """Any query in this run, whatever its outcome (a no_data answer cannot follow one)."""

    return any(
        isinstance(record, Mapping) and record.get("tool_name") in {"query_readonly", "parallel_readonly"}
        for record in state.get("tool_results", ())
    )


def _run_retrieval_source_ids(state: Mapping[str, object], *, excluded_ids: frozenset[str]) -> set[str]:
    """Retrieval sources the server put into this run: prepared context and search_catalog results.

    describe_tables output and catalog table entries are table structure, not
    a definition source, so they are left out.
    """

    def item_sources(items: object) -> set[str]:
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            return set()
        return {
            str(item["source_id"])
            for item in items
            if isinstance(item, Mapping)
            and isinstance(item.get("source_id"), str)
            and item["source_id"]
            and item.get("id") not in excluded_ids
        }

    source_ids = item_sources(state.get("retrieval_items", ()))
    for record in state.get("tool_results", ()):
        if (
            isinstance(record, Mapping)
            and record.get("tool_name") == "search_catalog"
            and record.get("status") == "succeeded"
            and isinstance(record.get("output"), Mapping)
        ):
            source_ids.update(item_sources(record["output"].get("items")))
    return source_ids


def _request_window(value: object) -> dict[str, str] | None:
    """Normalize a server-supplied request window; ``None`` means no date picker."""

    if value is None:
        return None
    try:
        return normalize_time_window(value, field="request_time_window")
    except MetricDeclarationError as exc:
        raise ValueError(exc.message) from exc


def _serialize_agent_checkpoint(
    state: Mapping[str, object],
    *,
    checkpoint_version: str = AGENT_CHECKPOINT_VERSION,
) -> dict[str, object]:
    context = state.get("context")
    run_config = state.get("run_config")
    if state.get("status") != "waiting_user" or not isinstance(context, ExecutionContext):
        raise RunResumeError("invalid_run_state", "only a WAITING_USER state can be persisted")
    if not isinstance(run_config, RunConfig):
        raise RunResumeError("invalid_checkpoint", "run configuration is missing")
    final_action = state.get("final_action")
    if not isinstance(final_action, Mapping) or final_action.get("type") != "ask_user":
        raise RunResumeError("invalid_checkpoint", "WAITING_USER action is missing")
    if checkpoint_version not in {AGENT_CHECKPOINT_VERSION, _V3_AGENT_CHECKPOINT_VERSION}:
        raise RunResumeError("invalid_checkpoint", "unsupported checkpoint version")
    checkpoint: dict[str, object] = {
        "checkpoint_version": checkpoint_version,
        "status": "WAITING_USER",
        "context": {
            "run_id": context.run_id,
            "tenant_id": context.tenant_id,
            "principal_id": context.principal_id,
            "role": context.role,
        },
        "question": state.get("question"),
        "run_config": run_config.as_dict(),
        "metric_bindings": [item.as_dict() for item in state.get("metric_bindings", ()) if isinstance(item, MetricBinding)],
        "request_time_window": dict(state["request_time_window"]) if state.get("request_time_window") else None,
        "parallel_plan": state["parallel_plan"].as_dict()
        if isinstance(state.get("parallel_plan"), ParallelPlan)
        else None,
        "active_elapsed_before": state.get("active_elapsed_before", 0.0),
        "clarifications": list(state.get("clarifications", ())),
        "model_call_count": state.get("model_call_count", 0),
        "tool_call_count": state.get("tool_call_count", 0),
        "repair_count": state.get("repair_count", 0),
        "model_call_ids": list(state.get("model_call_ids", ())),
        "events": [dict(item) for item in state.get("events", ()) if isinstance(item, Mapping)],
        "retrieval_items": [dict(item) for item in state.get("retrieval_items", ()) if isinstance(item, Mapping)],
        "tool_results": [dict(item) for item in state.get("tool_results", ()) if isinstance(item, Mapping)],
        "context_version": state.get("context_version", CONTEXT_VERSION),
        "waiting_question": str(final_action.get("question", "")),
        "waiting_clarification_id": (
            str(final_action["clarification_id"]) if type(final_action.get("clarification_id")) is str else None
        ),
        "clarification_bounce_count": state.get("clarification_bounce_count", 0),
    }
    if checkpoint_version == AGENT_CHECKPOINT_VERSION:
        # Survives resume: a run gets one answer send-back in total (B3c-2).
        checkpoint["answer_bounce_count"] = state.get("answer_bounce_count", 0)
    try:
        json.dumps(checkpoint, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RunResumeError("invalid_checkpoint", "checkpoint contains non-serializable server state") from exc
    return checkpoint


def _deserialize_agent_checkpoint(
    raw: Mapping[str, object],
    *,
    expected_context: ExecutionContext,
) -> _GraphState:
    required = {
        "checkpoint_version", "status", "context", "question", "run_config", "metric_bindings",
        "parallel_plan", "active_elapsed_before", "clarifications", "model_call_count", "tool_call_count",
        "repair_count", "model_call_ids", "events", "retrieval_items", "tool_results", "context_version",
        "waiting_question",
    }
    version = raw.get("checkpoint_version") if isinstance(raw, Mapping) else None
    if version == AGENT_CHECKPOINT_VERSION:
        required = required | {
            "request_time_window", "waiting_clarification_id", "clarification_bounce_count", "answer_bounce_count",
        }
    elif version == _V3_AGENT_CHECKPOINT_VERSION:
        required = required | {"request_time_window", "waiting_clarification_id", "clarification_bounce_count"}
    elif version == _V2_AGENT_CHECKPOINT_VERSION:
        required = required | {"request_time_window"}
    elif version != _LEGACY_AGENT_CHECKPOINT_VERSION:
        raise RunResumeError("invalid_checkpoint", "checkpoint version or state is invalid")
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise RunResumeError("invalid_checkpoint", "checkpoint fields do not match the server schema")
    if raw.get("status") != "WAITING_USER":
        raise RunResumeError("invalid_checkpoint", "checkpoint version or state is invalid")
    try:
        request_time_window = _request_window(raw.get("request_time_window"))
    except ValueError as exc:
        raise RunResumeError("invalid_checkpoint", "checkpoint request time window is invalid") from exc
    context_raw = raw.get("context")
    if not isinstance(context_raw, Mapping) or set(context_raw) != {"run_id", "tenant_id", "principal_id", "role"}:
        raise RunResumeError("invalid_checkpoint", "checkpoint identity is invalid")
    try:
        context = ExecutionContext(
            run_id=str(context_raw["run_id"]),
            tenant_id=str(context_raw["tenant_id"]),
            principal_id=str(context_raw["principal_id"]),
            role=str(context_raw["role"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RunResumeError("invalid_checkpoint", "checkpoint identity is invalid") from exc
    if context != expected_context:
        raise RunResumeError("resume_context_mismatch", "the resume context does not match the original run identity")
    if type(raw.get("question")) is not str or not str(raw["question"]).strip():
        raise RunResumeError("invalid_checkpoint", "checkpoint question is invalid")
    if type(raw.get("waiting_question")) is not str:
        raise RunResumeError("invalid_checkpoint", "checkpoint waiting prompt is invalid")
    waiting_clarification_id = raw.get("waiting_clarification_id")
    if waiting_clarification_id is not None and (
        type(waiting_clarification_id) is not str or not waiting_clarification_id.strip()
    ):
        raise RunResumeError("invalid_checkpoint", "checkpoint clarification rule is invalid")
    bounce_count = raw.get("clarification_bounce_count", 0)
    if type(bounce_count) is not int or bounce_count < 0:
        raise RunResumeError("invalid_checkpoint", "checkpoint clarification bounce count is invalid")
    answer_bounce_count = raw.get("answer_bounce_count", 0)
    if type(answer_bounce_count) is not int or answer_bounce_count < 0:
        raise RunResumeError("invalid_checkpoint", "checkpoint answer bounce count is invalid")
    config_raw = raw.get("run_config")
    if not isinstance(config_raw, Mapping):
        raise RunResumeError("invalid_checkpoint", "checkpoint run configuration is invalid")
    config_values = dict(config_raw)
    if config_values.pop("run_config_version", None) != "run-config-v1":
        raise RunResumeError("invalid_checkpoint", "checkpoint run configuration version is invalid")
    skill_versions = config_values.get("skill_versions", ())
    if type(skill_versions) is list:
        config_values["skill_versions"] = tuple(skill_versions)
    try:
        run_config = RunConfig(**config_values)
    except (TypeError, ValueError) as exc:
        raise RunResumeError("invalid_checkpoint", "checkpoint run configuration is invalid") from exc
    raw_bindings = raw.get("metric_bindings")
    if type(raw_bindings) is not list:
        raise RunResumeError("invalid_checkpoint", "checkpoint metric bindings are invalid")
    try:
        bindings = tuple(MetricBinding(**dict(item)) for item in raw_bindings if isinstance(item, Mapping))
    except (TypeError, ValueError) as exc:
        raise RunResumeError("invalid_checkpoint", "checkpoint metric bindings are invalid") from exc
    if len(bindings) != len(raw_bindings):
        raise RunResumeError("invalid_checkpoint", "checkpoint metric bindings are invalid")
    parallel_raw = raw.get("parallel_plan")
    parallel_plan = None
    if parallel_raw is not None:
        if not isinstance(parallel_raw, Mapping):
            raise RunResumeError("invalid_checkpoint", "checkpoint parallel plan is invalid")
        try:
            parallel_plan = ParallelPlan.from_context(
                context,
                parallel_raw["metric_ids"],
                time_window=parallel_raw["time_window"],
                policy_version=str(parallel_raw["policy_version"]),
                catalog_version=str(parallel_raw["catalog_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunResumeError("invalid_checkpoint", "checkpoint parallel plan is invalid") from exc
        if parallel_plan.plan_hash != parallel_raw.get("plan_hash"):
            raise RunResumeError("invalid_checkpoint", "checkpoint parallel plan hash is invalid")
    active_elapsed = raw.get("active_elapsed_before")
    if type(active_elapsed) not in {int, float} or active_elapsed < 0:
        raise RunResumeError("invalid_checkpoint", "checkpoint active time is invalid")
    counts: dict[str, int] = {}
    for name in ("model_call_count", "tool_call_count", "repair_count"):
        value = raw.get(name)
        if type(value) is not int or value < 0:
            raise RunResumeError("invalid_checkpoint", f"checkpoint {name} is invalid")
        counts[name] = value
    model_call_ids = raw.get("model_call_ids")
    if type(model_call_ids) is not list or any(type(item) is not str or not item for item in model_call_ids):
        raise RunResumeError("invalid_checkpoint", "checkpoint model call identities are invalid")
    sequences: dict[str, tuple[Mapping[str, object], ...]] = {}
    for name in ("events", "retrieval_items", "tool_results"):
        value = raw.get(name)
        if type(value) is not list or any(not isinstance(item, Mapping) for item in value):
            raise RunResumeError("invalid_checkpoint", f"checkpoint {name} are invalid")
        sequences[name] = tuple(dict(item) for item in value)
    clarifications = raw.get("clarifications")
    if type(clarifications) is not list or any(type(item) is not str or not item.strip() for item in clarifications):
        raise RunResumeError("invalid_checkpoint", "checkpoint clarifications are invalid")
    context_version = raw.get("context_version")
    if type(context_version) is not str or not context_version:
        raise RunResumeError("invalid_checkpoint", "checkpoint context version is invalid")
    return {
        "context": context,
        "question": str(raw["question"]),
        "run_config": run_config,
        "metric_bindings": bindings,
        "request_time_window": request_time_window,
        "parallel_plan": parallel_plan,
        "started_at": monotonic(),
        "active_elapsed_before": float(active_elapsed),
        "clarifications": tuple(clarifications),
        "clarification_bounce_count": bounce_count,
        "answer_bounce_count": answer_bounce_count,
        "status": "waiting_user",
        "reason": None,
        "error_code": None,
        "proposal": None,
        "final_action": {
            "type": "ask_user",
            "question": str(raw["waiting_question"]),
            **({"clarification_id": waiting_clarification_id} if waiting_clarification_id is not None else {}),
        },
        **counts,
        "model_call_ids": tuple(model_call_ids),
        **sequences,
        "public_answer": None,
        "facts": None,
        "context_version": context_version,
        "elapsed_ms": int(float(active_elapsed) * 1000),
    }


def _provider_failure_fields(record: Mapping[str, object]) -> dict[str, object]:
    """Keep only provider identifiers/status from an already-redacted record."""

    allowed = {
        "provider",
        "model",
        "provider_call_id",
        "provider_request_id",
        "usage_status",
        "usage",
        "http_status",
        "provider_error_code",
    }
    return {key: record[key] for key in allowed if key in record}


def _tool_input_summary(action: ToolCallAction, *, declarable_metrics: Sequence[str] = ()) -> dict[str, object]:
    """Record what a tool call asked for without keeping model-written text.

    Only server-known values are kept verbatim: catalog-declarable metric ids and
    allow-listed table names.  Anything else the model wrote is reduced to a count
    and a hash, and those fields appear only when such a value is present.
    """

    arguments = action.arguments
    summary: dict[str, object] = {"argument_keys": sorted(arguments)}
    if action.name == "query_readonly":
        sql = arguments.get("sql")
        params = arguments.get("params")
        if isinstance(sql, str):
            summary.update(
                {
                    "sql_length": len(sql),
                    "sql_sha256": sha256(sql.encode("utf-8")).hexdigest(),
                }
            )
        if isinstance(params, Mapping):
            summary["params_count"] = len(params)
        metrics = arguments.get("metrics")
        if isinstance(metrics, list):
            known = [item for item in metrics if _is_known(item, declarable_metrics)]
            summary["declared_metrics"] = known[:4]
            _add_other_values(summary, "declared_metrics", [item for item in metrics if not _is_known(item, declarable_metrics)])
    elif action.name == "search_catalog":
        query = arguments.get("query")
        if isinstance(query, str):
            summary.update(
                {
                    "query_length": len(query),
                    "query_sha256": sha256(query.encode("utf-8")).hexdigest(),
                }
            )
        summary["top_k"] = arguments.get("top_k", 3)
    elif action.name == "describe_tables":
        tables = arguments.get("tables")
        if isinstance(tables, list):
            known = sorted(item for item in tables if _is_known(item, ALLOWED_TABLES))
            summary.update({"table_count": len(tables), "table_names": known})
            _add_other_values(summary, "table_names", [item for item in tables if not _is_known(item, ALLOWED_TABLES)])
    return summary


def _is_known(value: object, known: Sequence[str] | frozenset[str]) -> bool:
    return type(value) is str and value in known


def _add_other_values(summary: dict[str, object], prefix: str, others: Sequence[object]) -> None:
    """Count and hash values that are not server-known; the values themselves are not kept.

    The hash is over the canonical JSON of the values ordered by their own canonical
    form, so it is stable for strings and for any other JSON value the model wrote.
    """

    if not others:
        return
    def canonical(value: object) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)

    ordered = sorted(others, key=canonical)
    summary[f"{prefix}_other_count"] = len(ordered)
    summary[f"{prefix}_other_sha256"] = sha256(canonical(ordered).encode("utf-8")).hexdigest()


def _source_ids_from_tool_output(output: Mapping[str, object]) -> list[str]:
    source_ids: set[str] = set()
    for key in ("items", "tables"):
        values = output.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for value in values:
                if isinstance(value, Mapping) and isinstance(value.get("source_id"), str):
                    source_ids.add(value["source_id"])
    return sorted(source_ids)




def _usage_summary(events: Sequence[Mapping[str, object]]) -> dict[str, object]:
    model_events = [event for event in events if event.get("kind") == "model_call"]
    known: list[Mapping[str, object]] = []
    model_call_ids: list[str] = []
    unknown_count = 0
    for event in model_events:
        call_id = event.get("model_call_id")
        if type(call_id) is str and call_id:
            model_call_ids.append(call_id)
        usage = event.get("usage")
        prompt = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
        completion = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
        total = usage.get("total_tokens") if isinstance(usage, Mapping) else None
        if (
            event.get("usage_status") == "known"
            and type(prompt) is int and prompt >= 0
            and type(completion) is int and completion >= 0
            and type(total) is int and total >= 0
            and prompt + completion == total
        ):
            known.append({
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
            })
            continue
        unknown_count += 1

    summary: dict[str, object] = {
        "status": "not_run" if not model_events else ("known" if unknown_count == 0 else "unknown"),
        "model_call_count": len(model_events),
        "known_call_count": len(known),
        "unknown_call_count": unknown_count,
        "model_call_ids": model_call_ids,
    }
    for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        summary[f"known_{field_name}"] = (
            sum(int(usage[field_name]) for usage in known) if known else None
        )
    for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        summary[field_name] = (
            None
            if not model_events or unknown_count != 0
            else sum(int(usage[field_name]) for usage in known)
        )
    return summary


def _result_status(value: object) -> RunStatus:
    if value in {"succeeded", "denied", "waiting_user", "waiting_approval", "failed", "limit_reached"}:
        return value  # type: ignore[return-value]
    return "failed"


__all__ = [
    "AgentRunResult",
    "BoundedAgent",
    "GraphLimits",
    "MAX_QUERY_REPAIRS",
    "MAX_MODEL_CALLS",
    "MAX_TOOL_CALLS",
    "MAX_WALL_CLOCK_SECONDS",
    "RunResumeError",
]
