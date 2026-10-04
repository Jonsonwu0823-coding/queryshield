"""W04 durable runs over the product runtime: persistence, approval, cancellation, scope.

Sync and async ``/queries`` both create a run record here and execute the same
product runtime (``queryshield.agent.runtime.run_profile``).  One commit path
persists every outcome; ``/runs/{run_id}/resume`` continues a real
WAITING_USER checkpoint, and ``/runs/{run_id}/approval`` executes exactly the
verified query a WAITING_APPROVAL run is bound to.
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

from queryshield.agent.context import NET_FEN_PLAN_ID, NET_FEN_TIME_WINDOW
from queryshield.agent.graph import RunResumeError
from queryshield.agent.proposals import ExecutionContext, FactRef, MetricBinding, ResultEvidence
from queryshield.agent.config import RunConfig
from queryshield.agent.runtime import (
    B1_PROFILE,
    CountingExecutor,
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
from queryshield.catalog import DEFAULT_CATALOG_VERSION, ClarificationRule, SemanticCatalog, load_default_catalog
from queryshield.catalog.phrases import (
    ClarificationReading,
    read_clarifications,
    rule_named_by_waiting_question,
    select_value,
)
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.db.w04_state import StateStore, parse_time, utc_now
from queryshield.facts import FactResolver
from queryshield.facts.facts import FACTS_SCHEMA_VERSION, is_scalar_metric_result
from queryshield.facts.persisted import evidence_from_record
from queryshield.tools import ControlledTools, ToolError


W04_STATE_VERSION = "qs-w04-state-v1"
W04_POLICY_VERSION = "qs-sql-v1"
W04_CATALOG_VERSION = DEFAULT_CATALOG_VERSION
W04_DEFAULT_SNAPSHOT = "knowledge-v1-9f580dd7f887ed0a"
APPROVAL_TTL_SECONDS = 600
MAX_ACTIVE_RUNS = 2
SENSITIVE_PERMISSION_SOURCE_ID = "semantic-sensitive-customer-name"
# B3e: the product service found no active permission source for an approval.
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
    otherwise (older checkpoints, frozen W05 states) the rule whose values the
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


class W04AuthorizationError(ValueError):
    def __init__(self, code: str, message: str = "object is not visible") -> None:
        self.code = code
        super().__init__(message)


class ObjectNotFound(W04AuthorizationError):
    def __init__(self) -> None:
        super().__init__("not_found")


class ApprovalNotFound(W04AuthorizationError):
    def __init__(self) -> None:
        super().__init__("approval_not_found")


class ApprovalConflict(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class W04Identity:
    tenant_id: str
    principal_id: str
    role: str

    @classmethod
    def from_mapping(cls, identity: Mapping[str, str]) -> "W04Identity":
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
    metrics = pending_call.get("metrics")
    window = pending_call.get("time_window")
    if (
        type(pending_call.get("sql")) is not str
        or not isinstance(params, Mapping)
        or sorted(params) != [str(index) for index in range(len(params))]
        or not isinstance(metrics, list)
        or any(type(item) is not str for item in metrics)
        or (window is not None and not isinstance(window, Mapping))
    ):
        raise ApprovalConflict("approval_action_invalid", "the pending call is malformed")
    action: dict[str, object] = {
        "kind": APPROVAL_ACTION_KIND,
        "run_id": run_id,
        "sql": pending_call["sql"],
        # Ordered values, as the executor receives them (W04 action format).
        "params": [params[str(index)] for index in range(len(params))],
        "metrics": list(metrics),
        "time_window": dict(window) if window is not None else None,
        "tenant_id": tenant_id,
        "requester_principal_id": requester_principal_id,
        "policy_version": W04_POLICY_VERSION,
        "catalog_version": W04_CATALOG_VERSION,
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


class FixtureQueryExecutor:
    """Deterministic W04 Fake database boundary; never used as real evidence.

    It mirrors the commerce fixture (tenant A: c1 甲 paid 10000, c2 乙 paid
    5000, refunds 3000; tenant B: c1 丙 paid 990000, refunds 10000) for the
    projections the product Fake and the server net plan use.  It ignores time
    windows; it is only selected with the Fake provider.
    """

    _CUSTOMERS = {"A": (("c1", "甲", 10000), ("c2", "乙", 5000)), "B": (("c1", "丙", 990000),)}
    _REFUND_FEN = {"A": 3000, "B": 10000}

    def __init__(self, *, clock: Callable[[], datetime] = utc_now) -> None:
        self.clock = clock
        self.sql_calls = 0

    def _rows(self, sql: str, context: ExecutionContext, params: Sequence[object]) -> tuple[dict[str, object], ...]:
        lowered = " ".join(sql.lower().split())
        projection = lowered.split(" from ", 1)[0]
        customers = self._CUSTOMERS.get(context.tenant_id, ())
        paid_total = sum(amount for _, _, amount in customers)
        if "customers" in lowered and "name" in projection:
            wanted = {value for value in params if isinstance(value, str) and re.fullmatch(r"c\d+", value)}
            selected = [item for item in customers if not wanted or item[0] in wanted]
            return tuple(
                {key: value for key, value in (("customer_id", customer_id), ("name", name)) if key in projection}
                for customer_id, name, _ in selected
            )
        if "group by" in lowered and "customer_id" in lowered:
            alias = "gross_fen" if "gross_fen" in projection else None
            return tuple(
                {"customer_id": customer_id, **({alias: amount} if alias else {})}
                for customer_id, _, amount in customers
            )
        row: dict[str, object] = {}
        if "paid_count" in projection or ("count(" in projection and "gross_fen" not in projection):
            row["paid_count"] = len(customers)
        if "gross_fen" in projection:
            row["gross_fen"] = paid_total
        if "refund_fen" in projection:
            row["refund_fen"] = self._REFUND_FEN.get(context.tenant_id, 0)
        if "net_fen" in projection:
            row["net_fen"] = paid_total - self._REFUND_FEN.get(context.tenant_id, 0)
        return (row,) if row else ()

    def execute(
        self,
        sql: str,
        *,
        context: ExecutionContext,
        params: Sequence[object] = (),
        metric_bindings: Sequence[MetricBinding] = (),
    ) -> Any:
        self.sql_calls += 1
        rows = self._rows(sql, context, tuple(params))
        evidence = ResultEvidence.from_server_execution(
            context,
            result_id=f"result-{uuid4()}",
            rows=rows,
            normalized_query=sql,
            params=tuple(params),
            observed_at=self.clock(),
            policy_version=W04_POLICY_VERSION,
            catalog_version=W04_CATALOG_VERSION,
            metric_bindings=metric_bindings,
        )
        return type("FixtureResult", (), {"evidence": evidence, "rows": rows})()


def state_path_from_env() -> str:
    configured = os.getenv("QUERYSHIELD_STATE_STORE_PATH", "").strip()
    if configured:
        return configured
    return ":memory:"


_STORE_CACHE: dict[str, StateStore] = {}
_SERVICE_CACHE: dict[tuple[str, str, str], "W04RunService"] = {}
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


_TERMINAL_STATUSES = frozenset({"SUCCEEDED", "DENIED", "FAILED", "LIMIT_REACHED", "CANCELLED", "USAGE_UNKNOWN"})


class W04RunService:
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
        """``product_knowledge`` is set only by ``shared_w04_service`` (the HTTP product).

        The product publishes the knowledge snapshot it uses and binds every
        approval to its permission source and version (B3e).  A service built
        with its own store (evaluation, checks, tests) keeps the earlier
        behaviour: it publishes nothing and adds permission fields only when
        the store already has the ACL.
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
        """Run the W04 parallel recovery scan at application startup."""
        from queryshield.agent.parallel_durable import DurableParallelScheduler

        return DurableParallelScheduler(state=self.store).recover_on_startup()

    def _default_executor(self) -> object:
        check_fake_database_boundary(self.mode)
        if os.getenv("QUERYSHIELD_W04_FAKE_DB", "").lower() in {"1", "true", "yes"}:
            return FixtureQueryExecutor(clock=self.clock)
        return GuardedQueryExecutor()

    def new_executor(self) -> object:
        """One database boundary for sync, async, resume and approval execution."""

        return self._executor_factory()

    def product_knowledge(self):
        """The product's knowledge base, published to the state store once (B3e).

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
        different cause: ApprovalConflict(knowledge_unavailable).  A service with its own store keeps the earlier
        rule (the default source, permission fields only if the ACL exists).
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

        Only ``shared_w04_service`` reads the environment; a bad value is a
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
        subject: W04Identity,
        question: str,
        time_window: Mapping[str, str] | None,
        deps: RuntimeDependencies,
    ) -> str:
        if self._active_count() >= MAX_ACTIVE_RUNS:
            raise ApprovalConflict("run_capacity_reached", "active run capacity is full")
        run_id = f"run-{uuid4()}"
        if self._product_knowledge:
            # The snapshot the product published and its retriever is built from (B3e).
            knowledge_snapshot_id = self.product_knowledge().snapshot_id
        else:
            current_snapshot = self.store.get_snapshot()
            knowledge_snapshot_id = (
                str(current_snapshot["snapshot_id"])
                if isinstance(current_snapshot, Mapping) and current_snapshot.get("snapshot_id")
                else W04_DEFAULT_SNAPSHOT
            )
        catalog = load_default_catalog()
        run_retriever = deps.retriever if deps.profile == B1_PROFILE else None
        agent_run_config = product_run_config(deps.profile, catalog=catalog, retriever=tools_retriever(run_retriever))
        checkpoint = {
            "state_schema_version": W04_STATE_VERSION,
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
                "catalog_version": W04_CATALOG_VERSION,
                "knowledge_snapshot_id": knowledge_snapshot_id,
                "policy_version": W04_POLICY_VERSION,
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

        subject = W04Identity.from_mapping(identity)
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
        subject = W04Identity.from_mapping(identity)
        metadata = self.metadata_config()
        deps = deps or self.default_dependencies()
        run_id = self._reserve_run(subject, question, time_window, deps)
        worker = Thread(
            target=self._execute_run,
            args=(run_id, subject, question, time_window, deps, metadata),
            name=f"queryshield-w04-{run_id}",
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
        subject: W04Identity,
        question: str,
        time_window: Mapping[str, str] | None,
        deps: RuntimeDependencies,
        metadata: object | None = None,
    ) -> dict[str, object]:
        opened_store = None
        tools: object | None = None
        metadata_written = [False]
        try:
            self.store.append_event(run_id, "step_started", "RUNNING", payload={"step": "agent_run", "profile": deps.profile})
            call_store = deps.call_store
            if call_store is None:
                from queryshield.agent.call_store import DurableModelCallStore

                opened_store = DurableModelCallStore(os.getenv("QUERYSHIELD_CALL_STORE_PATH", "").strip() or ":memory:")
                call_store = opened_store
            counter = CountingExecutor(deps.executor)
            run_deps = replace(deps, call_store=call_store)
            tools = product_tools(run_deps, executor=counter, metadata=metadata)
            context = ExecutionContext(
                run_id=run_id,
                tenant_id=subject.tenant_id,
                principal_id=subject.principal_id,
                role=subject.role,
            )
            profile_run = run_profile(run_deps, context, question, time_window=time_window, tools=tools)
            # The run's MCP session ends with this execution, before the result commits.
            metadata_record = self._close_metadata(tools, metadata_written)
            return self._commit(
                run_id,
                profile_run.payload,
                agent=profile_run.agent,
                tools=tools,
                context=context,
                sql_executions=counter.executions,
                metadata_record=metadata_record,
            )
        except Exception as exc:  # noqa: BLE001 - durable terminal state is the boundary
            code = _error_code(exc)
            metadata_record = self._close_metadata(tools, metadata_written)
            if metadata_record is not None:
                self.store.append_event(run_id, "metadata_session", "RUNNING", payload=metadata_record)
            self.store.append_event(run_id, "terminal", "FAILED", payload={"error_code": code})
            self.store.update_run(run_id, status="FAILED", error_code=code)
            return self.store.get_run(run_id)  # type: ignore[return-value]
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
        envelope_updates: Mapping[str, object] | None = None,
        previous_event_count: int = 0,
        metadata_record: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        current = self.store.get_run(run_id)
        if current is None:
            raise ApprovalConflict("run_not_found", "run disappeared before result commit")
        status = str(payload.get("status"))
        error_code = payload.get("error_code")
        outcome = outcome_for(status, error_code)
        sql_total = int(current.get("sql_exec_count", 0)) + int(sql_executions)
        raw_events = payload.get("events")
        new_events = (
            [dict(event) for event in raw_events[previous_event_count:] if isinstance(event, Mapping)]
            if isinstance(raw_events, list)
            else []
        )
        model_call_ids = [str(item) for item in payload.get("model_call_ids", ()) if isinstance(item, str)]  # type: ignore[union-attr]
        usage = payload.get("usage_summary") if isinstance(payload.get("usage_summary"), Mapping) else payload.get("usage")
        counters = {
            "model_call_count": int(payload.get("model_call_count", 0) or 0),
            "tool_call_count": int(payload.get("tool_call_count", 0) or 0),
            "sql_exec_count": sql_total,
            "usage_json": json.dumps(dict(usage), ensure_ascii=False, sort_keys=True) if isinstance(usage, Mapping) else None,
        }
        envelope = dict(current.get("checkpoint") or {})
        envelope.update(dict(envelope_updates or {}))
        envelope.update(
            {
                "model_call_ids": model_call_ids,
                "model_call_id": model_call_ids[-1] if model_call_ids else envelope.get("model_call_id"),
                "status": outcome.run_status,
                "agent_checkpoint": (
                    agent.export_waiting_checkpoint(run_id) if status == "waiting_user" and agent is not None else None
                ),
            }
        )

        if current.get("status") in {"CANCEL_REQUESTED", "CANCELLED"} or current.get("cancel_requested"):
            # Cancellation wins before any result is committed.  Events are
            # written before the terminal status so a stream never stops early.
            envelope["status"] = "CANCELLED"
            self._append_agent_events(run_id, new_events, metadata_record)
            self.store.append_event(run_id, "step_finished", "CANCELLED", payload={"cancelled_before_commit": True})
            self.store.append_event(run_id, "terminal", "CANCELLED")
            self.store.update_run(run_id, status="CANCELLED", checkpoint_json=_json(envelope), **counters)
            return self.store.get_run(run_id)  # type: ignore[return-value]

        # B3c-2: the answer's status and server source ids live in the envelope
        # (no table change); written after the cancel branch, so a run that is
        # not SUCCEEDED never carries one.
        envelope.update(_answer_envelope(payload, succeeded=outcome.run_status == "SUCCEEDED"))
        facts_list = _payload_facts(payload)
        result_json: str | None = None
        facts_json: str | None = None
        if status == "succeeded":
            evidences = self._committed_evidences(payload, tools, context, facts_list)
            if evidences:
                primary, supporting = evidences[0], evidences[1:]
                result = primary.as_dict()
                if supporting:
                    result["supporting_results"] = [item.as_dict() for item in supporting]
                result_json = _json(result)
            if facts_list:
                facts_json = _json({"schema_version": FACTS_SCHEMA_VERSION, "facts": facts_list})
        permission: tuple[str, Mapping[str, object] | None] | None = None
        if status == "waiting_approval":
            try:
                permission = self._approval_permission(current)
            except ApprovalConflict as exc:
                # No approval without a bound permission (B3e): the run fails with a fixed code.
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
            # A failed run returns no answer, except the catalog's fixed note
            # for an unsupported clarification scope (server text only).
            "answer": (
                payload.get("answer")
                if status == "succeeded" or error_code == CLARIFICATION_VALUE_UNSUPPORTED_CODE
                else None
            ),
            "error_code": error_code if status not in {"succeeded", "waiting_user", "waiting_approval"} else None,
            **counters,
        }
        # Every step event is written before the run leaves RUNNING, and a
        # terminal event before the terminal status, so a stream reader that
        # stops on a terminal status has already received every event.
        self._append_agent_events(run_id, new_events, metadata_record)
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

    def _append_agent_events(
        self,
        run_id: str,
        events: Sequence[Mapping[str, object]],
        metadata_record: Mapping[str, object] | None = None,
    ) -> None:
        for event in events:
            self.store.append_event(
                run_id,
                "agent_step",
                "RUNNING",
                result_id=str(event["result_id"]) if event.get("result_id") else None,
                payload=dict(event),
            )
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
            policy_version=W04_POLICY_VERSION,
            catalog_version=W04_CATALOG_VERSION,
            knowledge_snapshot_id=str(run_config.get("knowledge_snapshot_id") or W04_DEFAULT_SNAPSHOT),
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
        """Restore one owner-authorized B1 checkpoint without resetting its budget."""

        subject = W04Identity.from_mapping(identity)
        metadata = self.metadata_config()
        with self._resume_lock:
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
            # Clarified values in the server checkpoint are authoritative.  A
            # user answer may supply a bounded month, but it cannot select or
            # replace the metric binding.
            checkpoint_for_resume = dict(agent_checkpoint)
            catalog = load_default_catalog()
            stored_clarified_metric = checkpoint_envelope.get("clarified_metric")
            # Which catalog rule is waiting and which value the answer chose,
            # both read from the catalog phrase table.  An answer naming no
            # value, or two, chooses nothing and the run keeps waiting.
            waiting_rule = _waiting_clarification_rule(catalog, agent_checkpoint)
            if waiting_rule is not None and any(value.value == stored_clarified_metric for value in waiting_rule.values):
                waiting_rule = None
            chosen = select_value(catalog, waiting_rule, answer) if waiting_rule is not None else None
            selected_metric = chosen.value if chosen is not None and chosen.metric == chosen.value else None
            clarified_metric = stored_clarified_metric if stored_clarified_metric is not None else selected_metric
            keep_waiting_for_metric = waiting_rule is not None and chosen is None
            unsupported_choice = chosen if chosen is not None and not chosen.supported else None
            if clarified_metric is not None:
                if type(clarified_metric) is not str or clarified_metric not in {"gross_fen", "net_fen", "paid_count"}:
                    raise ApprovalConflict("checkpoint_invalid", "server metric slot is invalid")
                metric = catalog.metric(clarified_metric)
                prior_clarifications = agent_checkpoint.get("clarifications", ())
                clarification_text = " ".join(
                    [str(agent_checkpoint.get("question", ""))]
                    + [str(item) for item in prior_clarifications if type(item) is str]
                    + [answer]
                )
                answer_month = re.search(
                    r"(?P<year>20\d{2})年(?P<month>1[0-2]|0?[1-9])月",
                    clarification_text,
                )
                time_window = dict(NET_FEN_TIME_WINDOW)
                if answer_month is not None:
                    year = int(answer_month.group("year"))
                    month = int(answer_month.group("month"))
                    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
                    time_window = {
                        "start": f"{year:04d}-{month:02d}-01T00:00:00Z",
                        "end": f"{next_year:04d}-{next_month:02d}-01T00:00:00Z",
                        "timezone": "UTC",
                    }
                server_binding = MetricBinding(
                    metric_id=clarified_metric,
                    result_position=clarified_metric,
                    unit=str(metric.payload["unit"]),
                    time_window=time_window,
                    catalog_source_id=metric.source_id,
                    catalog_version=catalog.catalog_version,
                    plan_id=NET_FEN_PLAN_ID if clarified_metric == "net_fen" else None,
                )
                checkpoint_for_resume["metric_bindings"] = [server_binding.as_dict()]
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
            context = ExecutionContext(
                run_id=run_id,
                tenant_id=subject.tenant_id,
                principal_id=subject.principal_id,
                role=subject.role,
            )
            counter = CountingExecutor(executor)
            deps = RuntimeDependencies(
                model=model,  # type: ignore[arg-type]
                executor=counter,
                retriever=retriever,
                call_store=call_store,
                profile=B1_PROFILE,
            )
            tools = product_tools(deps, metadata=metadata)
            metadata_written = [False]
            agent = build_b1_agent(
                model,  # type: ignore[arg-type]
                tools,
                call_store=call_store,
                run_config=run_config,
                retrieval_available=retriever is not None,
            )
            try:
                if unsupported_choice is not None:
                    # A scope no catalog metric supports: stop with the
                    # catalog note; the model may not widen a metric.
                    result = agent.fail_unsupported_clarification(
                        context, answer, checkpoint_for_resume, note=str(unsupported_choice.unsupported_note)
                    )
                elif keep_waiting_for_metric:
                    result = agent.continue_waiting_for_clarification(context, answer, checkpoint_for_resume)
                else:
                    result = agent.resume_from_checkpoint(context, answer, checkpoint_for_resume)
            except RunResumeError as exc:
                code = "not_found" if exc.code == "resume_context_mismatch" else "checkpoint_invalid"
                raise ApprovalConflict(code, "resume checkpoint could not be restored") from exc
            finally:
                # A resume is a new execution: its MCP session (if any) ends here.
                metadata_record = self._close_metadata(tools, metadata_written)
            envelope_updates: dict[str, object] = {}
            if selected_metric is not None:
                envelope_updates["clarified_metric"] = selected_metric
            payload = b1_result_payload(result, context, str(agent_checkpoint.get("question", run.get("question", ""))))
            envelope_updates["last_agent_result"] = result.as_dict()
            return self._commit(
                run_id,
                payload,
                agent=agent,
                tools=tools,
                context=context,
                sql_executions=counter.executions,
                envelope_updates=envelope_updates,
                previous_event_count=len(old_events),
                metadata_record=metadata_record,
            )

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
        subject = W04Identity.from_mapping(identity)
        run = self.store.get_run(run_id)
        if run is None or run.get("tenant_id") != subject.tenant_id:
            raise ObjectNotFound()
        if subject.role != "approver":
            raise W04AuthorizationError("forbidden", "only a same-tenant approver can decide")
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
            raise W04AuthorizationError("forbidden", "requester cannot approve its own action")
        if approval.get("status") == "EXPIRED":
            raise ApprovalConflict("approval_stale", "approval has expired")
        if parse_time(str(approval["expires_at"])) <= self.clock().astimezone(timezone.utc):
            self.store.decide_approval(approval_id, approver_principal_id=subject.principal_id, decision=decision, now=self.clock())
            raise ApprovalConflict("approval_stale", "approval has expired")
        pending_action = approval.get("action")
        if isinstance(pending_action, Mapping) and pending_action.get("permission_source_id"):
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
            self.store.append_event(run_id, "terminal", "DENIED", payload={"approval_id": approval_id})
            self.store.update_run(run_id, status="DENIED")
            return self.store.get_run(run_id)  # type: ignore[return-value]
        if decided.get("status") == "APPROVED":
            # The consumed record must still carry the exact bound digest.
            _require_bound_action(decided, run)
            try:
                return self._execute_approved(run_id, run, decided)
            finally:
                with self._active_lock:
                    self._active.discard(run_id)
        if decided.get("status") == "EXPIRED":
            raise ApprovalConflict("approval_stale", "approval has expired")
        return self.store.get_run(run_id)  # type: ignore[return-value]

    def _execute_approved(
        self,
        run_id: str,
        run: Mapping[str, object],
        approval: Mapping[str, object],
    ) -> dict[str, object]:
        """Execute the single approved call for the requester and render server-side.

        The approver's identity only authorizes this call; the result, facts
        and answer belong to the requester.  Earlier verified metric results
        of this run (stored when it paused) are kept alongside, once each.
        """

        action = approval["action"]
        assert isinstance(action, Mapping)
        requester = ExecutionContext(
            run_id=run_id,
            tenant_id=str(run["tenant_id"]),
            principal_id=str(run["principal_id"]),
            role=str(run["role"]),
        )
        catalog = load_default_catalog()
        counter = CountingExecutor(self._executor_factory())
        tools = ControlledTools(catalog=catalog, executor=counter)
        try:
            evidence = execute_approved_query(tools, pending_call_from_action(action), context=requester)
        except (ToolError, RuntimeConfigurationError) as exc:
            code = getattr(exc, "code", "execution_failed")
            self.store.append_event(run_id, "terminal", "FAILED", payload={"error_code": code})
            self.store.update_run(
                run_id,
                status="FAILED",
                error_code=code,
                sql_exec_count=int(run.get("sql_exec_count", 0)) + counter.executions,
            )
            return self.store.get_run(run_id)  # type: ignore[return-value]
        sql_total = int(run.get("sql_exec_count", 0)) + counter.executions
        current_run = self.store.get_run(run_id) or {}
        if current_run.get("status") in {"CANCEL_REQUESTED", "CANCELLED"} or current_run.get("cancel_requested"):
            self.store.append_event(run_id, "step_finished", "CANCELLED", payload={"cancelled_before_commit": True})
            self.store.append_event(run_id, "terminal", "CANCELLED")
            self.store.update_run(run_id, status="CANCELLED", sql_exec_count=sql_total)
            return self.store.get_run(run_id)  # type: ignore[return-value]

        envelope = run.get("checkpoint") if isinstance(run.get("checkpoint"), Mapping) else {}
        prior: list[ResultEvidence] = []
        for item in envelope.get("pre_approval_results", ()) or ():  # type: ignore[union-attr]
            if isinstance(item, Mapping):
                prior.append(evidence_from_record(item, run))
        evidences = {item.result_id: item for item in prior}
        evidences[evidence.result_id] = evidence
        references: list[FactRef] = []
        seen: set[tuple[str, str]] = set()
        for source in prior + [evidence]:
            # Checkpointed results are judged again here, by the same rule (B3e).
            if not is_scalar_metric_result(source):
                continue
            for binding in source.metric_bindings:
                key = (source.result_id, binding.metric_id)
                if key not in seen:
                    seen.add(key)
                    references.append(FactRef(result_id=source.result_id, metric_id=binding.metric_id))
        facts: list[dict[str, object]] = []
        if references:
            facts = bind_facts_to_context(
                FactResolver(catalog=catalog).resolve(tuple(references), context=requester, evidences=evidences).as_dict()["facts"],
                requester,
            )
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
        # Server text throughout; verified only with facts (B3c-2, controller Q5).
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
        subject = W04Identity.from_mapping(identity)
        run = self.store.get_run(run_id)
        if run is None or run.get("tenant_id") != subject.tenant_id or run.get("principal_id") != subject.principal_id:
            raise ObjectNotFound()
        status = str(run["status"])
        if status == "RUNNING":
            self.store.update_run(run_id, status="CANCEL_REQUESTED", cancel_requested=1)
            self.store.append_event(run_id, "step_finished", "CANCEL_REQUESTED", payload={"cancel_requested": True})
            # The worker owns the final transition after its resource exits;
            # a running request must never be reported as CANCELLED early.
        elif status in {"WAITING_APPROVAL", "WAITING_USER"}:
            # Nothing is executing; the waiting run ends here and cannot resume.
            self.store.append_event(run_id, "terminal", "CANCELLED")
            self.store.update_run(run_id, status="CANCELLED", cancel_requested=1)
        elif status in {"CANCEL_REQUESTED"} | _TERMINAL_STATUSES:
            if status == "USAGE_UNKNOWN":
                self.store.update_run(run_id, cancel_requested=1)
        else:
            raise ApprovalConflict("invalid_run_state", "run cannot be cancelled in its current state")
        return self.store.get_run(run_id)  # type: ignore[return-value]

    def visible_run(self, *, run_id: str, identity: Mapping[str, str], result: bool = False) -> dict[str, object]:
        subject = W04Identity.from_mapping(identity)
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
        or action.get("policy_version") != W04_POLICY_VERSION
        or action.get("catalog_version") != W04_CATALOG_VERSION
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
    # rows stay in result.rows): the same fixed row-count text (B3c-2, Q5).
    if not any(fact.get("result_id") == evidence.result_id for fact in facts):
        parts.append(f"审批通过，已执行只读查询：返回 {evidence.row_count} 行，见 result.rows。")
    return "\n".join(parts)


ANSWER_STATUSES = frozenset({"verified", "unverified", "no_data"})


def _answer_envelope(payload: Mapping[str, object], *, succeeded: bool) -> dict[str, object]:
    """answer_status and the server's answer source ids for the run envelope (B3c-2).

    Only a SUCCEEDED run with an answer has them; an unknown status is kept
    conservative (unverified), never promoted to verified.
    """

    if not succeeded or payload.get("answer") is None:
        return {"answer_status": None, "answer_source_ids": None}
    status = payload.get("answer_status")
    action = payload.get("action")
    source_ids = action.get("source_ids") if isinstance(action, Mapping) and action.get("type") == "final_answer" else None
    return {
        "answer_status": status if status in ANSWER_STATUSES else "unverified",
        "answer_source_ids": [str(item) for item in source_ids if type(item) is str] if isinstance(source_ids, list) else [],
    }


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


def shared_w04_service() -> W04RunService:
    """Return one process-wide service per state path/mode/database boundary."""

    path = state_path_from_env()
    mode = os.getenv("QUERYSHIELD_PROVIDER_MODE", "fake").lower()
    fake_db = os.getenv("QUERYSHIELD_W04_FAKE_DB", "").lower()
    key = (path, mode, fake_db)
    with _STORE_LOCK:
        service = _SERVICE_CACHE.get(key)
        if service is None:
            service = W04RunService(
                store=shared_state_store(),
                mode=mode,
                product_knowledge=True,
                metadata_tools=METADATA_TOOLS_FROM_ENV,
            )
            _SERVICE_CACHE[key] = service
        return service


__all__ = [
    "APPROVAL_TTL_SECONDS",
    "MAX_ACTIVE_RUNS",
    "METADATA_TOOLS_FROM_ENV",
    "W04RunService",
    "W04_STATE_VERSION",
    "action_hash",
    "build_pending_action",
    "pending_call_from_action",
    "reset_shared_state_stores",
    "shared_w04_service",
    "shared_state_store",
]
