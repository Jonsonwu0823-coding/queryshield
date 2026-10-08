"""Durable runs over the product runtime: persistence, approval, cancellation, scope.

Sync and async ``/queries`` both create a run record here and execute the same
product runtime (``queryshield.agent.runtime.run_profile``).  One commit path
persists every outcome; ``/runs/{run_id}/resume`` continues a real
WAITING_USER checkpoint, and ``/runs/{run_id}/approval`` executes exactly the
verified query a WAITING_APPROVAL run is bound to.  Both run as RUNNING, like a
first execution.  Every write that moves a run out of a waiting state or
RUNNING to start, cancel or deny it is conditional (``StateStore.transition_run``),
so a cancel and a start that overlap leave one terminal event.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import re
from threading import Lock, RLock, Thread
from typing import Any, Callable
from uuid import uuid4

from queryshield.agent.context import NET_FEN_TIME_WINDOW
from queryshield.agent.graph import RunResumeError, _usage_summary
from queryshield.agent.metric_intent import build_metric_binding
from queryshield.agent.proposals import ExecutionContext, FactRef, MetricBinding, ResultEvidence
from queryshield.agent.config import RunConfig
from queryshield.agent.runtime import (
    B1_PROFILE,
    CountingExecutor,
    CountingModel,
    RuntimeConfigurationError,
    RuntimeDependencies,
    b1_result_payload,
    bind_facts_to_context,
    build_b1_agent,
    check_fake_database_boundary,
    configured_profile,
    model_for_mode,
    outcome_for,
    product_retriever,
    product_run_config,
    product_tools,
    render_fact_records,
    retrieval_label,
    run_profile,
    tools_retriever,
)
from queryshield.agent.tool_execution import CLARIFICATION_VALUE_UNSUPPORTED_CODE, execute_approved_query
from queryshield.approval.fixture_executor import FixtureQueryExecutor
from queryshield.approval.versions import BOUND_CATALOG_VERSION, BOUND_POLICY_VERSION
from queryshield.catalog import ClarificationRule, ClarificationValue, SemanticCatalog, load_default_catalog
from queryshield.catalog.phrases import (
    ClarificationReading,
    read_clarifications,
    rule_named_by_waiting_question,
    select_value,
)
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.db.state_store import StateStore, parse_time, utc_now
from queryshield.facts import FactResolver
from queryshield.facts.facts import FACTS_SCHEMA_VERSION, is_scalar_metric_result
from queryshield.facts.persisted import evidence_from_record
from queryshield.policy.params import ordered_param_values
from queryshield.tools import ControlledTools, ToolError


STATE_VERSION = "qs-w04-state-v1"
DEFAULT_KNOWLEDGE_SNAPSHOT = "knowledge-v1-9f580dd7f887ed0a"
APPROVAL_TTL_SECONDS = 600
MAX_ACTIVE_RUNS = 2
SENSITIVE_PERMISSION_SOURCE_ID = "semantic-sensitive-customer-name"
# The product service found no active permission source for an approval.
APPROVAL_PERMISSION_UNAVAILABLE_CODE = "approval_permission_unavailable"
# The product knowledge base could not be loaded: a different cause from a missing permission source.
KNOWLEDGE_UNAVAILABLE_CODE = "knowledge_unavailable"
APPROVAL_ACTION_KIND = "query_readonly"


def _waiting_clarification_rule(
    catalog: SemanticCatalog,
    agent_checkpoint: Mapping[str, object],
) -> ClarificationRule | None:
    """The catalog rule a waiting run asks about.

    The rule id the server stored with the question when there is one;
    otherwise (older checkpoints, frozen states) the rule whose values the
    waiting question names at least twice, read from the catalog phrase table.
    """

    rule_id = agent_checkpoint.get("waiting_clarification_id")
    if type(rule_id) is str:
        try:
            rule = catalog.clarification(rule_id)
        except KeyError:
            rule = None
        if rule is not None and len(rule.values) > 1:
            return rule
    return rule_named_by_waiting_question(catalog, str(agent_checkpoint.get("waiting_question", "")))


# The product service reads QUERYSHIELD_METADATA_TOOLS; every other service is local.
METADATA_TOOLS_FROM_ENV = "environment"


class RunAuthorizationError(ValueError):
    def __init__(self, code: str, message: str = "object is not visible") -> None:
        self.code = code
        super().__init__(message)


class ObjectNotFound(RunAuthorizationError):
    def __init__(self) -> None:
        super().__init__("not_found")


class ApprovalNotFound(RunAuthorizationError):
    def __init__(self) -> None:
        super().__init__("approval_not_found")


class ApprovalConflict(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class _ResumeRefused(ApprovalConflict):
    """The graph refused the stored checkpoint or the answer (``RunResumeError``)."""


@dataclass(frozen=True)
class RunIdentity:
    tenant_id: str
    principal_id: str
    role: str

    @classmethod
    def from_mapping(cls, identity: Mapping[str, str]) -> "RunIdentity":
        return cls(
            tenant_id=str(identity["tenant_id"]),
            principal_id=str(identity["principal_id"]),
            role=str(identity["role"]),
        )


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def action_hash(action: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(action).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Approval actions: the verified query plus server-bound identity/permission
# ---------------------------------------------------------------------------


def build_pending_action(
    pending_call: Mapping[str, object],
    *,
    run_id: str,
    tenant_id: str,
    requester_principal_id: str,
    permission: Mapping[str, object] | None = None,
    permission_source_id: str = SENSITIVE_PERMISSION_SOURCE_ID,
) -> dict[str, object]:
    """Bind one server-verified query_readonly call to its run, identity and permission.

    ``action_hash`` of the returned mapping covers every field; the client and
    the model cannot supply or change any of them.
    """

    if not isinstance(pending_call, Mapping) or pending_call.get("tool") != "query_readonly":
        raise ApprovalConflict("approval_action_invalid", "only a verified query_readonly call can be approved")
    params = pending_call.get("params")
    ordered_params = ordered_param_values(params) if isinstance(params, Mapping) else None
    metrics = pending_call.get("metrics")
    window = pending_call.get("time_window")
    if (
        type(pending_call.get("sql")) is not str
        or ordered_params is None
        or not isinstance(metrics, list)
        or any(type(item) is not str for item in metrics)
        or (window is not None and not isinstance(window, Mapping))
    ):
        raise ApprovalConflict("approval_action_invalid", "the pending call is malformed")
    action: dict[str, object] = {
        "kind": APPROVAL_ACTION_KIND,
        "run_id": run_id,
        "sql": pending_call["sql"],
        # Ordered values, as the executor receives them (action format).
        "params": list(ordered_params),
        "metrics": list(metrics),
        "time_window": dict(window) if window is not None else None,
        "tenant_id": tenant_id,
        "requester_principal_id": requester_principal_id,
        "policy_version": BOUND_POLICY_VERSION,
        "catalog_version": BOUND_CATALOG_VERSION,
    }
    if permission is not None:
        action["permission_source_id"] = permission_source_id
        action["permission_version"] = int(permission["acl_version"])
    return action


def pending_call_from_action(action: Mapping[str, object]) -> dict[str, object]:
    params = action.get("params")
    return {
        "tool": "query_readonly",
        "sql": action.get("sql"),
        "params": {str(index): value for index, value in enumerate(params)} if isinstance(params, list) else params,
        "metrics": action.get("metrics"),
        "time_window": action.get("time_window"),
    }


def state_path_from_env() -> str:
    return os.getenv("QUERYSHIELD_STATE_STORE_PATH", "").strip() or ":memory:"


def call_store_path_from_env() -> str:
    return os.getenv("QUERYSHIELD_CALL_STORE_PATH", "").strip() or ":memory:"


_STORE_CACHE: dict[str, StateStore] = {}
_SERVICE_CACHE: dict[tuple[str, str, str], "RunService"] = {}
_STORE_LOCK = RLock()


def shared_state_store() -> StateStore:
    path = state_path_from_env()
    with _STORE_LOCK:
        store = _STORE_CACHE.get(path)
        if store is None:
            store = StateStore(path)
            _STORE_CACHE[path] = store
        return store


def reset_shared_state_stores() -> None:
    with _STORE_LOCK:
        stores = tuple(_STORE_CACHE.values())
        _STORE_CACHE.clear()
        _SERVICE_CACHE.clear()
    for store in stores:
        store.close()


TERMINAL_STATUSES = frozenset({"SUCCEEDED", "DENIED", "FAILED", "LIMIT_REACHED", "CANCELLED", "USAGE_UNKNOWN"})


class RunService:
    """Application-facing durable workflow service over the product runtime."""

    def __init__(
        self,
        *,
        store: StateStore | None = None,
        executor_factory: Callable[[], object] | None = None,
        clock: Callable[[], datetime] = utc_now,
        mode: str | None = None,
        product_knowledge: bool = False,
        metadata_tools: object | None = None,
    ) -> None:
        """``product_knowledge`` is set only by ``shared_run_service`` (the HTTP product).

        The product publishes the knowledge snapshot it uses and binds every
        approval to its permission source and version.  A service built with
        its own store (evaluation, checks, tests) publishes nothing and adds
        permission fields only when the store already has the ACL.
        """

        self.store = store or shared_state_store()
        self.clock = clock
        self.mode = (mode or os.getenv("QUERYSHIELD_PROVIDER_MODE", "fake")).lower()
        self._executor_factory = executor_factory or self._default_executor
        self._active_lock = Lock()
        self._active: set[str] = set()
        self._approval_lock = Lock()
        self._resume_lock = Lock()
        self._product_knowledge = product_knowledge
        self._knowledge_lock = Lock()
        self._published_snapshot_ids: set[str] = set()
        # MCP metadata tools: None = local (environment ignored), "mcp" or an
        # McpMetadataConfig (tests, explicitly), METADATA_TOOLS_FROM_ENV (the product).
        self._metadata_tools = metadata_tools

    def recover_parallel_groups(self) -> dict[str, object]:
        """Run the parallel recovery scan at application startup."""
        from queryshield.agent.parallel_durable import DurableParallelScheduler

        return DurableParallelScheduler(state=self.store).recover_on_startup()

    def _default_executor(self) -> object:
        check_fake_database_boundary(self.mode)
        if os.getenv("QUERYSHIELD_FAKE_DB", "").lower() in {"1", "true", "yes"}:
            return FixtureQueryExecutor(clock=self.clock)
        return GuardedQueryExecutor()

    def new_executor(self) -> object:
        """One database boundary for sync, async, resume and approval execution."""

        return self._executor_factory()

    def product_knowledge(self):
        """The product's knowledge base, published to the state store once.

        The snapshot id is the content id before embedding (the hybrid
        retriever's ``base_snapshot_id``): no embedding call, independent of the
        interpreter.  A snapshot id the store already has is never published
        again, so neither a restart nor a repeated publication touches its ACL
        rows; a new snapshot keeps every source that is not active in the store
        as it is (``keep_inactive_sources``).  Only the product service calls this.
        """

        from queryshield.db.readonly import DatabaseConfigurationError, check_demo_pairing, demo_dataset_enabled
        from queryshield.knowledge.runtime import product_knowledge

        # The same demo setting checks as product_retriever.
        try:
            check_demo_pairing(os.getenv("QUERYSHIELD_DATABASE_URL"))
            demo = demo_dataset_enabled()
        except DatabaseConfigurationError as exc:
            raise RuntimeConfigurationError(getattr(exc, "code", "invalid_database_configuration"), "the demo dataset setting and the database do not match") from exc
        try:
            knowledge = product_knowledge(demo=demo)
        except Exception as exc:  # noqa: BLE001 - never run without the configured knowledge base
            raise RuntimeConfigurationError(KNOWLEDGE_UNAVAILABLE_CODE, "the product knowledge base could not be loaded") from exc
        with self._knowledge_lock:
            if knowledge.snapshot_id not in self._published_snapshot_ids:
                if self.store.get_snapshot(knowledge.snapshot_id) is None:
                    self.store.publish_snapshot(knowledge.snapshot.as_dict(), keep_inactive_sources=True)
                self._published_snapshot_ids.add(knowledge.snapshot_id)
        return knowledge

    def _approval_permission(self, run: Mapping[str, object]) -> tuple[str, Mapping[str, object] | None]:
        """(permission source id, its current ACL) for an approval of this run.

        The product requires an active ACL for its knowledge base's source;
        otherwise ApprovalConflict(approval_permission_unavailable) and no
        approval is created.  A knowledge base that cannot be loaded is a
        different cause: ApprovalConflict(knowledge_unavailable).  A service with
        its own store uses the default source, with permission fields only if its ACL exists.
        """

        if not self._product_knowledge:
            return SENSITIVE_PERMISSION_SOURCE_ID, self.store.get_source_acl(SENSITIVE_PERMISSION_SOURCE_ID)
        run_config = run.get("run_config") if isinstance(run.get("run_config"), Mapping) else {}
        try:
            knowledge = self.product_knowledge()
        except RuntimeConfigurationError as exc:
            raise ApprovalConflict(KNOWLEDGE_UNAVAILABLE_CODE, "the product knowledge base is unavailable") from exc
        acl = self.store.get_source_acl(knowledge.sensitive_source_id)
        if (
            run_config.get("knowledge_snapshot_id") != knowledge.snapshot_id
            or acl is None
            or acl.get("status") != "active"
        ):
            raise ApprovalConflict(APPROVAL_PERMISSION_UNAVAILABLE_CODE, "no active permission source for this approval")
        return knowledge.sensitive_source_id, acl

    def metadata_config(self):
        """This service's MCP metadata configuration for one run, or None for local tools.

        Only ``shared_run_service`` reads the environment; a bad value is a
        configuration error (503) before any run is created.
        """

        from queryshield.mcp_metadata.launch import (
            McpMetadataConfig,
            MetadataToolsConfigurationError,
            call_timeout_seconds,
            resolve_product_config,
        )

        setting = self._metadata_tools
        if setting is None:
            return None
        if isinstance(setting, McpMetadataConfig):
            return setting
        try:
            if setting == METADATA_TOOLS_FROM_ENV:
                return resolve_product_config(self.mode)
            if setting == "mcp":
                return McpMetadataConfig(mode=self.mode, call_timeout=call_timeout_seconds())
        except MetadataToolsConfigurationError as exc:
            raise RuntimeConfigurationError(exc.code, "the metadata tools setting is invalid") from exc
        raise RuntimeConfigurationError("invalid_metadata_tools_configuration", "the metadata tools setting is invalid")

    def _close_metadata(self, tools: object, written: list[bool]) -> dict[str, object] | None:
        """Close the run's MCP session, if any; the record once (``written`` guards the event)."""

        close = getattr(tools, "close", None)
        if not callable(close):
            return None
        record = close()
        if record is None or written[0]:
            return None
        written[0] = True
        return record

    def default_dependencies(self) -> RuntimeDependencies:
        """Server-configured dependencies for callers without FastAPI injection."""

        check_fake_database_boundary(self.mode)
        profile = configured_profile()
        return RuntimeDependencies(
            model=model_for_mode(self.mode),
            executor=self._executor_factory(),
            retriever=product_retriever(self.mode) if profile == B1_PROFILE else None,
            call_store=None,
            profile=profile,
        )

    def _active_count(self) -> int:
        with self._active_lock:
            self._active = {
                run_id
                for run_id in self._active
                if (self.store.get_run(run_id) or {}).get("status") in {"RUNNING", "CANCEL_REQUESTED"}
            }
            return len(self._active)

    # -- starting runs -----------------------------------------------------

    def _reserve_run(
        self,
        subject: RunIdentity,
        question: str,
        time_window: Mapping[str, str] | None,
        deps: RuntimeDependencies,
    ) -> str:
        if self._active_count() >= MAX_ACTIVE_RUNS:
            raise ApprovalConflict("run_capacity_reached", "active run capacity is full")
        run_id = f"run-{uuid4()}"
        if self._product_knowledge:
            # The snapshot the product published and its retriever is built from.
            knowledge_snapshot_id = self.product_knowledge().snapshot_id
        else:
            current_snapshot = self.store.get_snapshot()
            knowledge_snapshot_id = (
                str(current_snapshot["snapshot_id"])
                if isinstance(current_snapshot, Mapping) and current_snapshot.get("snapshot_id")
                else DEFAULT_KNOWLEDGE_SNAPSHOT
            )
        catalog = load_default_catalog()
        run_retriever = deps.retriever if deps.profile == B1_PROFILE else None
        agent_run_config = product_run_config(deps.profile, catalog=catalog, retriever=tools_retriever(run_retriever))
        checkpoint = {
            "state_schema_version": STATE_VERSION,
            "model_call_id": None,
            "model_call_ids": [],
            "idempotency_key": f"run:{run_id}:initial",
            "request_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "request_time_window": dict(time_window) if time_window is not None else None,
            "status": "RUNNING",
        }
        self.store.create_run(
            run_id=run_id,
            tenant_id=subject.tenant_id,
            principal_id=subject.principal_id,
            role=subject.role,
            question=question,
            # The adapter that will actually answer, not an environment guess.
            mode=str(getattr(deps.model, "mode", self.mode)),
            checkpoint=checkpoint,
            run_config={
                "profile": deps.profile,
                "mode": self.mode,
                "catalog_version": BOUND_CATALOG_VERSION,
                "knowledge_snapshot_id": knowledge_snapshot_id,
                "policy_version": BOUND_POLICY_VERSION,
                "retrieval": retrieval_label(run_retriever),
                "agent_run_config": agent_run_config.as_dict(),
            },
            model_call_count=0,
        )
        with self._active_lock:
            self._active.add(run_id)
        return run_id

    def run_sync(
        self,
        *,
        identity: Mapping[str, str],
        question: str,
        time_window: Mapping[str, str] | None = None,
        deps: RuntimeDependencies | None = None,
    ) -> dict[str, object]:
        """Run inside the caller's request; counts toward the same capacity as async."""

        subject = RunIdentity.from_mapping(identity)
        metadata = self.metadata_config()
        deps = deps or self.default_dependencies()
        run_id = self._reserve_run(subject, question, time_window, deps)
        return self._execute_run(run_id, subject, question, time_window, deps, metadata)

    def start_async(
        self,
        *,
        identity: Mapping[str, str],
        question: str,
        time_window: Mapping[str, str] | None = None,
        deps: RuntimeDependencies | None = None,
    ) -> dict[str, object]:
        subject = RunIdentity.from_mapping(identity)
        metadata = self.metadata_config()
        deps = deps or self.default_dependencies()
        run_id = self._reserve_run(subject, question, time_window, deps)
        worker = Thread(
            target=self._execute_run,
            args=(run_id, subject, question, time_window, deps, metadata),
            name=f"queryshield-run-{run_id}",
            daemon=True,
        )
        accepted_run = self.store.get_run(run_id)
        worker.start()
        # Return the durable acceptance checkpoint, not a racy post-worker
        # snapshot.  The HTTP contract is 202 after RUNNING was persisted.
        return accepted_run  # type: ignore[return-value]

    def _execute_run(
        self,
        run_id: str,
        subject: RunIdentity,
        question: str,
        time_window: Mapping[str, str] | None,
        deps: RuntimeDependencies,
        metadata: object | None = None,
    ) -> dict[str, object]:
        opened_store = None
        tools: object | None = None
        counter: CountingExecutor | None = None
        model: CountingModel | None = None
        writer = _StepWriter(self.store, run_id)
        metadata_written = [False]
        try:
            self.store.append_event(run_id, "step_started", "RUNNING", payload={"step": "agent_run", "profile": deps.profile})
            call_store = deps.call_store
            if call_store is None:
                from queryshield.agent.call_store import DurableModelCallStore

                opened_store = DurableModelCallStore(call_store_path_from_env())
                call_store = opened_store
            counter = CountingExecutor(deps.executor)
            model = CountingModel(deps.model)
            run_deps = replace(deps, call_store=call_store, model=model)
            tools = product_tools(run_deps, executor=counter, metadata=metadata)
            context = ExecutionContext(
                run_id=run_id,
                tenant_id=subject.tenant_id,
                principal_id=subject.principal_id,
                role=subject.role,
            )
            profile_run = run_profile(run_deps, context, question, time_window=time_window, tools=tools, on_step=writer)
            # The run's MCP session ends with this execution, before the result commits.
            metadata_record = self._close_metadata(tools, metadata_written)
            return self._commit(
                run_id,
                profile_run.payload,
                agent=profile_run.agent,
                tools=tools,
                context=context,
                sql_executions=counter.executions,
                writer=writer,
                metadata_record=metadata_record,
            )
        except Exception as exc:  # noqa: BLE001 - durable terminal state is the boundary
            metadata_record = self._close_metadata(tools, metadata_written)
            if metadata_record is not None:
                self.store.append_event(run_id, "metadata_session", "RUNNING", payload=metadata_record)
            return self._end_on_error(
                run_id, exc, sql_exec_count=counter.executions if counter is not None else 0, writer=writer, model=model
            )
        finally:
            self._close_metadata(tools, metadata_written)
            if opened_store is not None:
                opened_store.close()
            with self._active_lock:
                self._active.discard(run_id)

    # -- one commit path for every outcome ----------------------------------

    def _commit(
        self,
        run_id: str,
        payload: Mapping[str, object],
        *,
        agent: Any,
        tools: ControlledTools,
        context: ExecutionContext,
        sql_executions: int,
        writer: "_StepWriter",
        envelope_updates: Mapping[str, object] | None = None,
        metadata_record: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        current = self.store.get_run(run_id)
        if current is None:
            raise ApprovalConflict("run_not_found", "run disappeared before result commit")
        status = str(payload.get("status"))
        error_code = payload.get("error_code")
        outcome = outcome_for(status, error_code)
        counters = _commit_counters(current, payload, sql_executions)
        envelope = _commit_envelope(run_id, current, payload, outcome, agent, envelope_updates)

        if _cancel_requested(current):
            # Cancellation wins before any result is committed; events precede the terminal status.
            envelope["status"] = "CANCELLED"
            self._append_agent_events(run_id, writer, payload, metadata_record)
            return self._finish_cancelled(run_id, checkpoint_json=_json(envelope), **counters)

        # Written after the cancel branch, so a run that is not SUCCEEDED never carries an answer status.
        envelope.update(_answer_envelope(payload, succeeded=outcome.run_status == "SUCCEEDED"))
        facts_list = _payload_facts(payload)
        result_json, facts_json = (
            self._succeeded_result_json(payload, tools, context, facts_list) if status == "succeeded" else (None, None)
        )
        permission: tuple[str, Mapping[str, object] | None] | None = None
        if status == "waiting_approval":
            try:
                permission = self._approval_permission(current)
            except ApprovalConflict as exc:
                # No approval without a bound permission: the run fails with a fixed code.
                status, error_code, outcome = "failed", exc.code, outcome_for("failed", exc.code)
                envelope["status"] = outcome.run_status
        if status == "waiting_approval":
            envelope["pre_approval_results"] = [
                evidence.as_dict() for evidence in self._verified_metric_evidences(payload, tools, context)
            ]

        update: dict[str, object] = {
            "checkpoint_json": _json(envelope),
            "result_json": result_json,
            "facts_json": facts_json,
            # A failed run has no answer, except the catalog's fixed note for an unsupported scope.
            "answer": (
                payload.get("answer")
                if status == "succeeded" or error_code == CLARIFICATION_VALUE_UNSUPPORTED_CODE
                else None
            ),
            "error_code": error_code if status not in {"succeeded", "waiting_user", "waiting_approval"} else None,
            **counters,
        }
        # Step events precede the run leaving RUNNING, a terminal event precedes the terminal status:
        # a stream reader that stops on a terminal status has received every event.
        self._append_agent_events(run_id, writer, payload, metadata_record)
        if status == "waiting_approval":
            self.store.update_run(run_id, **update)
            self._create_pending_approval(run_id, current, payload, permission)
        elif status == "waiting_user":
            self.store.update_run(run_id, status=outcome.run_status, **update)
            self.store.append_event(run_id, "waiting", "WAITING_USER")
        else:
            self.store.append_event(
                run_id,
                "terminal",
                outcome.run_status,
                payload={"error_code": error_code} if error_code else None,
            )
            self.store.update_run(run_id, status=outcome.run_status, **update)
        return self.store.get_run(run_id)  # type: ignore[return-value]

    def _finish_cancelled(self, run_id: str, **update: object) -> dict[str, object]:
        self.store.append_event(run_id, "step_finished", "CANCELLED", payload={"cancelled_before_commit": True})
        self.store.append_event(run_id, "terminal", "CANCELLED")
        self.store.update_run(run_id, status="CANCELLED", **update)
        return self.store.get_run(run_id)  # type: ignore[return-value]

    def _finish_failed(self, run_id: str, code: str, **update: object) -> dict[str, object]:
        self.store.append_event(run_id, "terminal", "FAILED", payload={"error_code": code})
        self.store.update_run(run_id, status="FAILED", error_code=code, **update)
        return self.store.get_run(run_id)  # type: ignore[return-value]

    def _end_on_error(
        self,
        run_id: str,
        exc: Exception,
        *,
        sql_exec_count: int,
        writer: "_StepWriter | None" = None,
        model: CountingModel | None = None,
    ) -> dict[str, object]:
        """End a run whose execution raised: FAILED on the error's code, unless cancelled meanwhile.

        The first execution, a resume and an approved execution all end here; left
        waiting, an approved run could never execute again (its approval is already
        consumed).  Cancellation wins, as it does before a commit.

        The counts and usage are those of the last step whose events are all stored,
        as a commit would store them; without a step of this execution the stored
        ones stay.  A chat call this execution started without a stored event makes
        the usage unknown.
        """

        run = self.store.get_run(run_id) or {}
        update: dict[str, object] = {"sql_exec_count": sql_exec_count}
        if writer is not None and writer.state is not None:
            update.update(_step_counters(writer.state))
        elif run.get("usage") is None:
            update["usage_json"] = _json(_usage_summary(()))
        if (model.calls if model is not None else 0) > (writer.model_calls_written if writer is not None else 0):
            update["usage_json"] = _json(_USAGE_LOST)
        if _cancel_requested(run):
            return self._finish_cancelled(run_id, **update)
        return self._finish_failed(run_id, _error_code(exc), **update)

    def _succeeded_result_json(
        self,
        payload: Mapping[str, object],
        tools: ControlledTools,
        context: ExecutionContext,
        facts_list: Sequence[Mapping[str, object]],
    ) -> tuple[str | None, str | None]:
        result_json: str | None = None
        evidences = self._committed_evidences(payload, tools, context, facts_list)
        if evidences:
            primary, supporting = evidences[0], evidences[1:]
            result = primary.as_dict()
            if supporting:
                result["supporting_results"] = [item.as_dict() for item in supporting]
            result_json = _json(result)
        facts_json = _json({"schema_version": FACTS_SCHEMA_VERSION, "facts": facts_list}) if facts_list else None
        return result_json, facts_json

    def _append_agent_events(
        self,
        run_id: str,
        writer: "_StepWriter",
        payload: Mapping[str, object],
        metadata_record: Mapping[str, object] | None = None,
    ) -> None:
        """The result's events the graph has not stored yet (none after a graph run), then the MCP session record."""

        writer.write(payload.get("events") or ())
        if metadata_record is not None:
            # One record per MCP session (MCP setting only), after the steps it served.
            self.store.append_event(run_id, "metadata_session", "RUNNING", payload=dict(metadata_record))

    def _committed_evidences(
        self,
        payload: Mapping[str, object],
        tools: ControlledTools,
        context: ExecutionContext,
        facts_list: Sequence[Mapping[str, object]],
    ) -> list[ResultEvidence]:
        """Fact results first (primary + supporting); otherwise the last successful query."""

        result_ids: list[str] = []
        for fact in facts_list:
            result_id = fact.get("result_id")
            if type(result_id) is str and result_id not in result_ids:
                result_ids.append(result_id)
        if not result_ids:
            successful = _successful_query_result_ids(payload)
            if successful:
                result_ids.append(successful[-1])
        evidences: list[ResultEvidence] = []
        for result_id in result_ids:
            try:
                evidences.append(tools.get_result_evidence(result_id, context=context))
            except ToolError as exc:
                raise ApprovalConflict("evidence_validation_failed", "result evidence could not be persisted") from exc
        return evidences

    def _verified_metric_evidences(
        self,
        payload: Mapping[str, object],
        tools: ControlledTools,
        context: ExecutionContext,
    ) -> list[ResultEvidence]:
        """Results of this run that are tenant-wide metric values (``is_scalar_metric_result``)."""

        evidences: list[ResultEvidence] = []
        for result_id in _successful_query_result_ids(payload):
            try:
                evidence = tools.get_result_evidence(result_id, context=context)
            except ToolError:
                continue
            if is_scalar_metric_result(evidence):
                evidences.append(evidence)
        return evidences

    def _create_pending_approval(
        self,
        run_id: str,
        run: Mapping[str, object],
        payload: Mapping[str, object],
        permission: tuple[str, Mapping[str, object] | None] | None = None,
    ) -> None:
        final_action = payload.get("action")
        pending_call = final_action.get("tool_call") if isinstance(final_action, Mapping) else None
        source_id, acl = permission or self._approval_permission(run)
        action = build_pending_action(
            pending_call if isinstance(pending_call, Mapping) else {},
            run_id=run_id,
            tenant_id=str(run["tenant_id"]),
            requester_principal_id=str(run["principal_id"]),
            permission=acl,
            permission_source_id=source_id,
        )
        run_config = run.get("run_config") if isinstance(run.get("run_config"), Mapping) else {}
        self.store.create_approval(
            approval_id=f"approval-{uuid4()}",
            run_id=run_id,
            tenant_id=str(run["tenant_id"]),
            requester_principal_id=str(run["principal_id"]),
            action_hash=action_hash(action),
            action=action,
            policy_version=BOUND_POLICY_VERSION,
            catalog_version=BOUND_CATALOG_VERSION,
            knowledge_snapshot_id=str(run_config.get("knowledge_snapshot_id") or DEFAULT_KNOWLEDGE_SNAPSHOT),
            expires_at=self.clock() + timedelta(seconds=APPROVAL_TTL_SECONDS),
        )

    # -- WAITING_USER continuation -------------------------------------------

    def resume_waiting_user(
        self,
        *,
        run_id: str,
        answer: str,
        identity: Mapping[str, str],
        model: object,
        call_store: object,
        executor: object,
        retriever: object | None = None,
    ) -> dict[str, object]:
        """Restore one owner-authorized B1 checkpoint without resetting its budget.

        After the checks, the run moves WAITING_USER -> RUNNING (only from
        WAITING_USER: a run cancelled meanwhile never executes) and executes like a
        first execution: a cancel is a request, and every exit ends the run or
        leaves it waiting again.  A checkpoint the graph refuses before any step
        returns the run to WAITING_USER.
        """

        subject = RunIdentity.from_mapping(identity)
        metadata = self.metadata_config()
        with self._resume_lock:
            run, checkpoint_envelope, agent_checkpoint, run_config = self._load_waiting_run(run_id, subject)
            # Clarified values in the server checkpoint are authoritative.  A
            # user answer may supply a bounded month, but it cannot select or
            # replace the metric binding.
            checkpoint_for_resume = dict(agent_checkpoint)
            catalog = load_default_catalog()
            choice = _resolve_clarification(catalog, checkpoint_envelope, agent_checkpoint, answer)
            if choice.clarified_metric is not None:
                binding = _clarified_binding(catalog, choice.clarified_metric, agent_checkpoint, answer)
                checkpoint_for_resume["metric_bindings"] = [binding.as_dict()]
            stored_steps = _checked_event_count(agent_checkpoint, run)
            writer = _StepWriter(self.store, run_id, written=stored_steps)
            if not self.store.transition_run(run_id, "WAITING_USER", "RUNNING", cancel_requested=0):
                return self.store.get_run(run_id)  # type: ignore[return-value]
            context = ExecutionContext(
                run_id=run_id,
                tenant_id=subject.tenant_id,
                principal_id=subject.principal_id,
                role=subject.role,
            )
            counter = CountingExecutor(executor)
            counted = CountingModel(model)
            try:
                deps = RuntimeDependencies(
                    model=counted,  # type: ignore[arg-type]
                    executor=counter,
                    retriever=retriever,
                    call_store=call_store,
                    profile=B1_PROFILE,
                )
                tools = product_tools(deps, metadata=metadata)
                agent = build_b1_agent(
                    counted,  # type: ignore[arg-type]
                    tools,
                    call_store=call_store,
                    run_config=run_config,
                    retrieval_available=retriever is not None,
                    on_step=writer,
                )
                result, metadata_record = self._resume_agent(agent, tools, choice, context, answer, checkpoint_for_resume)
                envelope_updates: dict[str, object] = {}
                if choice.selected_metric is not None:
                    envelope_updates["clarified_metric"] = choice.selected_metric
                payload = b1_result_payload(result, context, str(agent_checkpoint.get("question", run.get("question", ""))))
                envelope_updates["last_agent_result"] = result.as_dict()
                return self._commit(
                    run_id,
                    payload,
                    agent=agent,
                    tools=tools,
                    context=context,
                    sql_executions=counter.executions,
                    writer=writer,
                    envelope_updates=envelope_updates,
                    metadata_record=metadata_record,
                )
            except Exception as exc:  # noqa: BLE001 - as in a first execution, the run ends on the error's code
                # A refusal before any step leaves the run as it was; after a step, waiting
                # again would store those steps twice on the next resume.
                if (
                    isinstance(exc, _ResumeRefused)
                    and writer.written == stored_steps
                    and self.store.transition_run(run_id, "RUNNING", "WAITING_USER")
                ):
                    raise
                return self._end_on_error(
                    run_id,
                    exc,
                    sql_exec_count=int(run.get("sql_exec_count", 0)) + counter.executions,
                    writer=writer,
                    model=counted,
                )

    def _resume_agent(
        self, agent: Any, tools: object, choice: ResumeChoice, context: ExecutionContext, answer: str, checkpoint: dict[str, object]
    ):
        """The agent's result for the answer and the run's MCP session record.

        A resume is a new execution: its MCP session (if any) ends here.
        """

        metadata_written = [False]
        try:
            if choice.unsupported is not None:
                # A scope no catalog metric supports: stop with the catalog note; the model may not widen a metric.
                result = agent.fail_unsupported_clarification(
                    context, answer, checkpoint, note=str(choice.unsupported.unsupported_note)
                )
            elif choice.keep_waiting:
                result = agent.continue_waiting_for_clarification(context, answer, checkpoint)
            else:
                result = agent.resume_from_checkpoint(context, answer, checkpoint)
        except RunResumeError as exc:
            code = "not_found" if exc.code == "resume_context_mismatch" else "checkpoint_invalid"
            raise _ResumeRefused(code, "resume checkpoint could not be restored") from exc
        except Exception:
            # The run ends on this error; like a first execution, its MCP session record is kept.
            record = self._close_metadata(tools, metadata_written)
            if record is not None:
                self.store.append_event(context.run_id, "metadata_session", "RUNNING", payload=record)
            raise
        finally:
            metadata_record = self._close_metadata(tools, metadata_written)
        return result, metadata_record

    def _load_waiting_run(
        self, run_id: str, subject: RunIdentity
    ) -> tuple[dict[str, object], Mapping[str, object], Mapping[str, object], RunConfig]:
        """The owner's WAITING_USER run with its server checkpoint and run configuration.

        Checked under the resume lock, in this order; each refusal has its own code.
        This layer compares the checkpoint with the stored run; the checkpoint's own
        format is checked again when the graph restores it.
        """

        run = self.store.get_run(run_id)
        if run is None or run.get("tenant_id") != subject.tenant_id or run.get("principal_id") != subject.principal_id:
            raise ObjectNotFound()
        if run.get("role") != subject.role:
            raise ObjectNotFound()
        if run.get("status") != "WAITING_USER":
            raise ApprovalConflict("invalid_run_state", "run is not waiting for user input")
        checkpoint_envelope = run.get("checkpoint")
        if not isinstance(checkpoint_envelope, Mapping):
            raise ApprovalConflict("checkpoint_invalid", "waiting run has no server checkpoint")
        agent_checkpoint = checkpoint_envelope.get("agent_checkpoint")
        if not isinstance(agent_checkpoint, Mapping):
            raise ApprovalConflict("not_supported", "this profile has no interactive continuation")
        run_config_payload = run.get("run_config")
        config_value = run_config_payload.get("agent_run_config") if isinstance(run_config_payload, Mapping) else None
        if not isinstance(config_value, Mapping):
            config_value = agent_checkpoint.get("run_config")
        try:
            run_config = RunConfig.from_dict(dict(config_value))
        except (TypeError, ValueError) as exc:
            raise ApprovalConflict("checkpoint_invalid", "waiting run profile is invalid") from exc
        if run_config.profile != B1_PROFILE:
            raise ApprovalConflict("not_supported", "the single-pass profile cannot resume a task")
        if agent_checkpoint.get("run_config") != run_config.as_dict():
            raise ApprovalConflict("checkpoint_invalid", "checkpoint profile differs from the stored run profile")
        return run, checkpoint_envelope, agent_checkpoint, run_config

    # -- approval --------------------------------------------------------------

    def approve(
        self,
        *,
        run_id: str,
        approval_id: str,
        identity: Mapping[str, str],
        decision: str,
    ) -> dict[str, object]:
        # Serialize the check-then-execute section.  StateStore makes the
        # decision transition atomic, but the business query must only run
        # for the caller that won that transition.
        with self._approval_lock:
            return self._approve_locked(
                run_id=run_id,
                approval_id=approval_id,
                identity=identity,
                decision=decision,
            )

    def _approve_locked(
        self,
        *,
        run_id: str,
        approval_id: str,
        identity: Mapping[str, str],
        decision: str,
    ) -> dict[str, object]:
        subject = RunIdentity.from_mapping(identity)
        run = self.store.get_run(run_id)
        if run is None or run.get("tenant_id") != subject.tenant_id:
            raise ObjectNotFound()
        if subject.role != "approver":
            raise RunAuthorizationError("forbidden", "only a same-tenant approver can decide")
        approval = self.store.get_approval(approval_id)
        if approval is None or approval.get("run_id") != run_id:
            raise ApprovalNotFound()
        if approval.get("status") in {"APPROVED", "REJECTED"}:
            # A consumed decision is a durable replay.  It must not execute
            # the business query a second time even though the run is terminal.
            return self.store.get_run(run_id)  # type: ignore[return-value]
        if run.get("status") != "WAITING_APPROVAL":
            raise ApprovalConflict("invalid_run_state", "run is not waiting for approval")
        if approval.get("requester_principal_id") == subject.principal_id:
            raise RunAuthorizationError("forbidden", "requester cannot approve its own action")
        if approval.get("status") == "EXPIRED":
            raise ApprovalConflict("approval_stale", "approval has expired")
        if parse_time(str(approval["expires_at"])) <= self.clock().astimezone(timezone.utc):
            self.store.decide_approval(approval_id, approver_principal_id=subject.principal_id, decision=decision, now=self.clock())
            raise ApprovalConflict("approval_stale", "approval has expired")
        self._require_current_permission(approval, subject)
        if decision == "approve":
            # Checked before the decision is consumed: a stale binding leaves
            # the approval pending and the run waiting, with no SQL executed.
            _require_bound_action(approval, run)
        decided = self.store.decide_approval(
            approval_id,
            approver_principal_id=subject.principal_id,
            decision=decision,
            now=self.clock(),
        )
        if decided.get("status") == "REJECTED":
            # Only a run still waiting is denied: a cancel that came first stands.
            self.store.transition_run(
                run_id, "WAITING_APPROVAL", "DENIED", event=("terminal", "DENIED", {"approval_id": approval_id})
            )
            return self.store.get_run(run_id)  # type: ignore[return-value]
        if decided.get("status") == "APPROVED":
            # The consumed record must still carry the exact bound digest.
            _require_bound_action(decided, run)
            try:
                # Cancelled after the state check: the approved query never runs.
                if not self.store.transition_run(run_id, "WAITING_APPROVAL", "RUNNING", cancel_requested=0):
                    return self.store.get_run(run_id)  # type: ignore[return-value]
                return self._execute_approved(run_id, run, decided)
            finally:
                with self._active_lock:
                    self._active.discard(run_id)
        if decided.get("status") == "EXPIRED":
            raise ApprovalConflict("approval_stale", "approval has expired")
        return self.store.get_run(run_id)  # type: ignore[return-value]

    def _require_current_permission(self, approval: Mapping[str, object], subject: RunIdentity) -> None:
        """An approval bound to a permission source needs that permission, unchanged and visible to this approver."""

        pending_action = approval.get("action")
        if not (isinstance(pending_action, Mapping) and pending_action.get("permission_source_id")):
            return
        from queryshield.knowledge.snapshots import (
            KnowledgeAccessError,
            KnowledgeIdentity,
            KnowledgeSnapshotRepository,
        )

        source_id = str(pending_action["permission_source_id"])
        current_permission = self.store.get_source_acl(source_id)
        try:
            KnowledgeSnapshotRepository(self.store).visible_source(
                snapshot_id=str(approval["knowledge_snapshot_id"]),
                source_id=source_id,
                identity=KnowledgeIdentity(subject.tenant_id, subject.principal_id, subject.role),
            )
        except KnowledgeAccessError as exc:
            raise ApprovalConflict("authorization_revoked", "approval permission is no longer current") from exc
        if (
            current_permission is None
            or current_permission.get("status") != "active"
            or current_permission.get("acl_version") != pending_action.get("permission_version")
        ):
            raise ApprovalConflict("authorization_revoked", "approval permission version changed")

    def _execute_approved(
        self,
        run_id: str,
        run: Mapping[str, object],
        approval: Mapping[str, object],
    ) -> dict[str, object]:
        """Execute the single approved call; whatever it raises ends the run (the approval is consumed)."""

        counter: CountingExecutor | None = None
        try:
            counter = CountingExecutor(self._executor_factory())
            return self._run_approved(run_id, run, approval, counter)
        except Exception as exc:  # noqa: BLE001 - durable terminal state is the boundary
            executed = counter.executions if counter is not None else 0
            return self._end_on_error(run_id, exc, sql_exec_count=int(run.get("sql_exec_count", 0)) + executed)

    def _run_approved(
        self,
        run_id: str,
        run: Mapping[str, object],
        approval: Mapping[str, object],
        counter: CountingExecutor,
    ) -> dict[str, object]:
        """Run the single approved call for the requester and render server-side.

        The approver's identity only authorizes this call; the result, facts
        and answer belong to the requester.  Earlier verified metric results
        of this run (stored when it paused) are kept alongside, once each.
        """

        requester = ExecutionContext(
            run_id=run_id,
            tenant_id=str(run["tenant_id"]),
            principal_id=str(run["principal_id"]),
            role=str(run["role"]),
        )
        catalog = load_default_catalog()
        tools = ControlledTools(catalog=catalog, executor=counter)
        evidence = execute_approved_query(tools, pending_call_from_action(approval["action"]), context=requester)  # type: ignore[arg-type]
        sql_total = int(run.get("sql_exec_count", 0)) + counter.executions
        current_run = self.store.get_run(run_id) or {}
        if _cancel_requested(current_run):
            return self._finish_cancelled(run_id, sql_exec_count=sql_total)

        envelope = run.get("checkpoint") if isinstance(run.get("checkpoint"), Mapping) else {}
        prior = [
            evidence_from_record(item, run)
            for item in envelope.get("pre_approval_results", ()) or ()  # type: ignore[union-attr]
            if isinstance(item, Mapping)
        ]
        references = _approved_fact_refs(prior, evidence)
        facts = _resolve_approved_facts(catalog, requester, references, [*prior, evidence])
        supporting = [item for item in prior if any(ref.result_id == item.result_id for ref in references)]
        result = evidence.as_dict()
        if supporting:
            result["supporting_results"] = [item.as_dict() for item in supporting]
        answer = _approved_answer(
            facts,
            evidence,
            clarifications=(
                read_clarifications(catalog, str(run.get("question", ""))) if catalog.has_phrase_table else None
            ),
        )
        approved_envelope = dict(current_run.get("checkpoint") or {}) if isinstance(current_run.get("checkpoint"), Mapping) else {}
        # Server text throughout; verified only with facts.
        approved_envelope.update({"answer_status": "verified" if facts else "unverified", "answer_source_ids": []})
        self.store.append_event(run_id, "step_finished", "SUCCEEDED", result_id=evidence.result_id, payload={"step": "approved_query"})
        self.store.append_event(run_id, "terminal", "SUCCEEDED", result_id=evidence.result_id)
        self.store.update_run(
            run_id,
            status="SUCCEEDED",
            result_json=_json(result),
            facts_json=_json({"schema_version": FACTS_SCHEMA_VERSION, "facts": facts}) if facts else None,
            answer=answer,
            sql_exec_count=sql_total,
            error_code=None,
            checkpoint_json=_json(approved_envelope),
        )
        return self.store.get_run(run_id)  # type: ignore[return-value]

    # -- cancellation and visibility ----------------------------------------------

    def cancel(self, *, run_id: str, identity: Mapping[str, str]) -> dict[str, object]:
        subject = RunIdentity.from_mapping(identity)
        run = self.store.get_run(run_id)
        if run is None or run.get("tenant_id") != subject.tenant_id or run.get("principal_id") != subject.principal_id:
            raise ObjectNotFound()
        status = str(run["status"])
        if status in {"WAITING_APPROVAL", "WAITING_USER"}:
            # Nothing is executing; the waiting run ends here and cannot resume.
            moved = self.store.transition_run(run_id, status, "CANCELLED", event=("terminal", "CANCELLED", None), cancel_requested=1)
        elif status == "RUNNING":
            # The execution owns the final transition after its resource exits;
            # a running request must never be reported as CANCELLED early.
            moved = self.store.transition_run(
                run_id,
                "RUNNING",
                "CANCEL_REQUESTED",
                event=("step_finished", "CANCEL_REQUESTED", {"cancel_requested": True}),
                cancel_requested=1,
            )
        else:
            # Already cancelling or ended: nothing moves.
            if status == "USAGE_UNKNOWN":
                self.store.update_run(run_id, cancel_requested=1)
            elif status not in {"CANCEL_REQUESTED"} | TERMINAL_STATUSES:
                raise ApprovalConflict("invalid_run_state", "run cannot be cancelled in its current state")
            return self.store.get_run(run_id)  # type: ignore[return-value]
        if not moved:
            # The run moved on between the read and the write (an execution started,
            # paused or ended): decide again on what it is now.  An ended run is left as it is.
            return self.cancel(run_id=run_id, identity=identity)
        return self.store.get_run(run_id)  # type: ignore[return-value]

    def visible_run(self, *, run_id: str, identity: Mapping[str, str], result: bool = False) -> dict[str, object]:
        subject = RunIdentity.from_mapping(identity)
        run = self.store.get_run(run_id)
        if run is None or run.get("tenant_id") != subject.tenant_id:
            raise ObjectNotFound()
        if run.get("principal_id") == subject.principal_id:
            if result and run.get("result") is None and not _succeeded_without_evidence(run):
                raise ApprovalConflict("result_not_ready", "result is not ready")
            return run
        if subject.role == "approver" and run.get("status") == "WAITING_APPROVAL" and not result:
            redacted = dict(run)
            redacted["result"] = None
            redacted["facts"] = None
            redacted["answer"] = None
            return redacted
        raise ObjectNotFound()

    def recover(self, *, run_id: str) -> dict[str, object]:
        """Reload a WAITING_APPROVAL checkpoint without executing it."""

        run = self.store.get_run(run_id)
        if run is None:
            raise ObjectNotFound()
        if run.get("status") == "WAITING_APPROVAL":
            checkpoint = run.get("checkpoint") or {}
            if not isinstance(checkpoint, Mapping) or not checkpoint.get("model_call_id"):
                raise ApprovalConflict("recovery_required", "checkpoint is incomplete")
            self.store.append_event(run_id, "step_started", "WAITING_APPROVAL", payload={"recovered": True})
        return self.store.get_run(run_id)  # type: ignore[return-value]


@dataclass(frozen=True)
class ResumeChoice:
    """What a resume answer decided, read from the catalog phrase table."""

    selected_metric: str | None  # the metric the answer chose, if it chose one
    clarified_metric: str | None  # the stored metric, else the one just chosen
    keep_waiting: bool  # the answer named no value, or two: nothing is chosen
    unsupported: ClarificationValue | None  # a chosen scope no catalog metric supports


def _resolve_clarification(
    catalog: SemanticCatalog,
    envelope: Mapping[str, object],
    agent_checkpoint: Mapping[str, object],
    answer: str,
) -> ResumeChoice:
    """Which catalog rule is waiting and which value the answer chose."""

    stored_clarified_metric = envelope.get("clarified_metric")
    waiting_rule = _waiting_clarification_rule(catalog, agent_checkpoint)
    if waiting_rule is not None and any(value.value == stored_clarified_metric for value in waiting_rule.values):
        waiting_rule = None
    chosen = select_value(catalog, waiting_rule, answer) if waiting_rule is not None else None
    selected_metric = chosen.value if chosen is not None and chosen.metric == chosen.value else None
    return ResumeChoice(
        selected_metric=selected_metric,
        clarified_metric=stored_clarified_metric if stored_clarified_metric is not None else selected_metric,
        keep_waiting=waiting_rule is not None and chosen is None,
        unsupported=chosen if chosen is not None and not chosen.supported else None,
    )


def _clarification_text(agent_checkpoint: Mapping[str, object], answer: str) -> str:
    prior_clarifications = agent_checkpoint.get("clarifications", ())
    return " ".join(
        [str(agent_checkpoint.get("question", ""))]
        + [str(item) for item in prior_clarifications if type(item) is str]
        + [answer]
    )


def _month_window(text: str) -> dict[str, str]:
    """The month the text names as a half-open UTC window; the server's default window without one."""

    named = re.search(r"(?P<year>20\d{2})年(?P<month>1[0-2]|0?[1-9])月", text)
    if named is None:
        return dict(NET_FEN_TIME_WINDOW)
    year, month = int(named.group("year")), int(named.group("month"))
    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
    return {
        "start": f"{year:04d}-{month:02d}-01T00:00:00Z",
        "end": f"{next_year:04d}-{next_month:02d}-01T00:00:00Z",
        "timezone": "UTC",
    }


def _clarified_binding(
    catalog: SemanticCatalog, metric_id: object, agent_checkpoint: Mapping[str, object], answer: str
) -> MetricBinding:
    """The server-built binding for the clarified metric; only the three declarable metrics are accepted."""

    if type(metric_id) is not str or metric_id not in {"gross_fen", "net_fen", "paid_count"}:
        raise ApprovalConflict("checkpoint_invalid", "server metric slot is invalid")
    return build_metric_binding(catalog, metric_id, _month_window(_clarification_text(agent_checkpoint, answer)))


def _checked_event_count(agent_checkpoint: Mapping[str, object], run: Mapping[str, object]) -> int:
    """The number of agent events already stored; the budget counters must match the run record."""

    try:
        previous_model_calls = agent_checkpoint["model_call_count"]
        previous_tool_calls = agent_checkpoint["tool_call_count"]
        old_events = agent_checkpoint["events"]
    except KeyError as exc:
        raise ApprovalConflict("checkpoint_invalid", "checkpoint budget counters are missing") from exc
    if (
        type(previous_model_calls) is not int
        or type(previous_tool_calls) is not int
        or type(old_events) is not list
        or previous_model_calls != run.get("model_call_count")
        or previous_tool_calls != run.get("tool_call_count")
    ):
        raise ApprovalConflict("checkpoint_invalid", "checkpoint budget counters do not match the run")
    return len(old_events)


def _resolve_approved_facts(
    catalog: SemanticCatalog,
    requester: ExecutionContext,
    references: Sequence[FactRef],
    evidences: Sequence[ResultEvidence],
) -> list[dict[str, object]]:
    if not references:
        return []
    return bind_facts_to_context(
        FactResolver(catalog=catalog)
        .resolve(tuple(references), context=requester, evidences={item.result_id: item for item in evidences})
        .as_dict()["facts"],
        requester,
    )


def _approved_fact_refs(prior: Sequence[ResultEvidence], evidence: ResultEvidence) -> list[FactRef]:
    """One reference per (result, metric) of the earlier and the approved results that are tenant-wide metric values."""

    references: list[FactRef] = []
    seen: set[tuple[str, str]] = set()
    for source in [*prior, evidence]:
        # Checkpointed results are judged again here, by the same rule.
        if not is_scalar_metric_result(source):
            continue
        for binding in source.metric_bindings:
            key = (source.result_id, binding.metric_id)
            if key not in seen:
                seen.add(key)
                references.append(FactRef(result_id=source.result_id, metric_id=binding.metric_id))
    return references


def _cancel_requested(run: Mapping[str, object]) -> bool:
    return run.get("status") in {"CANCEL_REQUESTED", "CANCELLED"} or bool(run.get("cancel_requested"))


class _StepWriter:
    """Stores each agent event of one execution once, as the graph produces it.

    ``written`` counts the run's events already stored (a resume starts after the
    checkpoint's).  ``state`` is the last agent state whose events are all stored.
    """

    def __init__(self, store: StateStore, run_id: str, *, written: int = 0) -> None:
        self.store, self.run_id, self.written = store, run_id, written
        self.model_calls_written = 0
        self.state: Mapping[str, object] | None = None

    def __call__(self, state: Mapping[str, object]) -> None:
        self.write(state.get("events") or ())
        self.state = state

    def write(self, events: Sequence[Mapping[str, object]]) -> None:
        for event in events[self.written:]:
            self.store.append_event(
                self.run_id,
                "agent_step",
                "RUNNING",
                result_id=str(event["result_id"]) if event.get("result_id") else None,
                payload=dict(event),
            )
            self.written += 1
            if event.get("kind") == "model_call":
                self.model_calls_written += 1


def _step_counters(state: Mapping[str, object]) -> dict[str, object]:
    """What a commit stores from an agent state: the graph's own counters and the usage of its events."""

    return {
        "model_call_count": int(state.get("model_call_count", 0)),
        "tool_call_count": int(state.get("tool_call_count", 0)),
        "usage_json": _json(_usage_summary(state.get("events", ()))),
    }


def _commit_counters(current: Mapping[str, object], payload: Mapping[str, object], sql_executions: int) -> dict[str, object]:
    usage = payload.get("usage_summary") if isinstance(payload.get("usage_summary"), Mapping) else payload.get("usage")
    return {
        "model_call_count": int(payload.get("model_call_count", 0) or 0),
        "tool_call_count": int(payload.get("tool_call_count", 0) or 0),
        "sql_exec_count": int(current.get("sql_exec_count", 0)) + int(sql_executions),
        "usage_json": json.dumps(dict(usage), ensure_ascii=False, sort_keys=True) if isinstance(usage, Mapping) else None,
    }


def _commit_envelope(
    run_id: str,
    current: Mapping[str, object],
    payload: Mapping[str, object],
    outcome: Any,
    agent: Any,
    envelope_updates: Mapping[str, object] | None,
) -> dict[str, object]:
    model_call_ids = [str(item) for item in payload.get("model_call_ids", ()) if isinstance(item, str)]  # type: ignore[union-attr]
    envelope = dict(current.get("checkpoint") or {})
    envelope.update(dict(envelope_updates or {}))
    envelope.update(
        {
            "model_call_ids": model_call_ids,
            "model_call_id": model_call_ids[-1] if model_call_ids else envelope.get("model_call_id"),
            "status": outcome.run_status,
            "agent_checkpoint": (
                agent.export_waiting_checkpoint(run_id) if str(payload.get("status")) == "waiting_user" and agent is not None else None
            ),
        }
    )
    return envelope


def _require_bound_action(approval: Mapping[str, object], run: Mapping[str, object]) -> None:
    """The approval and the run must carry the identical, current, server-bound action."""

    action = approval.get("action")
    run_action = run.get("action")
    expected = approval.get("action_hash")
    if (
        not isinstance(action, Mapping)
        or not isinstance(run_action, Mapping)
        or type(expected) is not str
        or action_hash(action) != expected
        or action_hash(run_action) != expected
        or action.get("kind") != APPROVAL_ACTION_KIND
        or action.get("run_id") != run.get("run_id")
        or action.get("tenant_id") != run.get("tenant_id")
        or action.get("requester_principal_id") != run.get("principal_id")
        or action.get("policy_version") != BOUND_POLICY_VERSION
        or action.get("catalog_version") != BOUND_CATALOG_VERSION
    ):
        raise ApprovalConflict("approval_stale", "approved action no longer matches the server-bound action")


def _approved_answer(
    facts: Sequence[Mapping[str, object]],
    evidence: ResultEvidence,
    *,
    clarifications: ClarificationReading | None = None,
) -> str:
    """"已核实" only for resolved metric facts; row values are never copied into the answer.

    Column names are not listed either: a real database returns the SQL
    aliases the model wrote, so they are model-controlled text.  Only fixed
    server text, catalog basis notes and the server's row count go into the
    answer.
    """

    parts: list[str] = []
    if facts:
        parts.append(render_fact_records(facts, clarifications=clarifications))
    # The approved query added no fact (no metric, or a grouped metric whose
    # rows stay in result.rows): the same fixed row-count text.
    if not any(fact.get("result_id") == evidence.result_id for fact in facts):
        parts.append(f"审批通过，已执行只读查询：返回 {evidence.row_count} 行，见 result.rows。")
    return "\n".join(parts)


ANSWER_STATUSES = frozenset({"verified", "unverified", "no_data"})


def normalized_answer(status: object, source_ids: object) -> tuple[str, list[str]]:
    """The answer status and source ids as stored and as read back.

    An unknown status is kept conservative (unverified), never promoted to
    verified; source ids keep only strings.
    """

    return (
        status if status in ANSWER_STATUSES else "unverified",  # type: ignore[return-value]
        [item for item in source_ids if type(item) is str] if isinstance(source_ids, list) else [],
    )


def _answer_envelope(payload: Mapping[str, object], *, succeeded: bool) -> dict[str, object]:
    """answer_status and the server's answer source ids for the run envelope.

    Only a SUCCEEDED run with an answer has them.
    """

    if not succeeded or payload.get("answer") is None:
        return {"answer_status": None, "answer_source_ids": None}
    action = payload.get("action")
    source_ids = action.get("source_ids") if isinstance(action, Mapping) and action.get("type") == "final_answer" else None
    status, ids = normalized_answer(payload.get("answer_status"), source_ids)
    return {"answer_status": status, "answer_source_ids": ids}


def _succeeded_without_evidence(run: Mapping[str, object]) -> bool:
    """A SUCCEEDED answer that needed no query (knowledge, no_data): no result and no facts to check."""

    return run.get("status") == "SUCCEEDED" and run.get("result") is None and run.get("facts") is None


def _payload_facts(payload: Mapping[str, object]) -> list[dict[str, object]]:
    raw = payload.get("facts")
    items = raw.get("facts") if isinstance(raw, Mapping) else raw
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, Mapping)]


def _successful_query_result_ids(payload: Mapping[str, object]) -> list[str]:
    ids: list[str] = []
    events = payload.get("events")
    if isinstance(events, list):
        for event in events:
            if (
                isinstance(event, Mapping)
                and event.get("kind") == "tool_call"
                and event.get("tool_name") == "query_readonly"
                and event.get("status") == "succeeded"
                and type(event.get("result_id")) is str
            ):
                ids.append(str(event["result_id"]))
    raw_ids = payload.get("result_ids")
    if not ids and isinstance(raw_ids, list):
        ids = [str(item) for item in raw_ids if type(item) is str]
    return ids


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _error_code(exc: Exception) -> str:
    return str(getattr(exc, "code", "execution_failed"))


# The stored usage of a run that ended on an exception (the agent's summary shape).
_USAGE_LOST = {"status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


def shared_run_service() -> RunService:
    """Return one process-wide service per state path/mode/database boundary."""

    path = state_path_from_env()
    mode = os.getenv("QUERYSHIELD_PROVIDER_MODE", "fake").lower()
    fake_db = os.getenv("QUERYSHIELD_FAKE_DB", "").lower()
    key = (path, mode, fake_db)
    with _STORE_LOCK:
        service = _SERVICE_CACHE.get(key)
        if service is None:
            service = RunService(
                store=shared_state_store(),
                mode=mode,
                product_knowledge=True,
                metadata_tools=METADATA_TOOLS_FROM_ENV,
            )
            _SERVICE_CACHE[key] = service
        return service


__all__ = [
    "APPROVAL_TTL_SECONDS",
    "FixtureQueryExecutor",
    "MAX_ACTIVE_RUNS",
    "METADATA_TOOLS_FROM_ENV",
    "RunService",
    "BOUND_CATALOG_VERSION",
    "BOUND_POLICY_VERSION",
    "STATE_VERSION",
    "action_hash",
    "build_pending_action",
    "pending_call_from_action",
    "reset_shared_state_stores",
    "shared_run_service",
    "shared_state_store",
]
