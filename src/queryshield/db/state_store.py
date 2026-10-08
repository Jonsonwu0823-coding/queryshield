"""Durable control-plane state.

The commerce facts remain in PostgreSQL.  This small SQLite store contains
run/checkpoint/approval/event metadata and confirmed preferences so a local
process restart can reload a safe state without pretending that an in-memory
checkpoint is durable.  It never stores credentials, prompts, model responses,
or hidden reasoning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from threading import RLock


STATE_SCHEMA_VERSION = "qs-state-v1"
MAX_EVENTS_PER_RUN = 200


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime | None = None) -> str:
    current = value or utc_now()
    if current.tzinfo is None:
        raise ValueError("state timestamps must be timezone-aware")
    return current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    result = datetime.fromisoformat(normalized)
    if result.tzinfo is None:
        raise ValueError("stored timestamps must be timezone-aware")
    return result.astimezone(timezone.utc)


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_value(value: str | None, default: object = None) -> object:
    if value is None:
        return default
    return json.loads(value)


class StateStoreError(RuntimeError):
    """A durable state operation could not be completed safely."""


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS state_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    role TEXT NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    action_json TEXT,
    approval_id TEXT,
    result_json TEXT,
    facts_json TEXT,
    answer TEXT,
    usage_json TEXT,
    checkpoint_json TEXT,
    run_config_json TEXT,
    error_code TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    sql_exec_count INTEGER NOT NULL DEFAULT 0,
    model_call_count INTEGER NOT NULL DEFAULT 0,
    tool_call_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    tenant_id TEXT NOT NULL,
    requester_principal_id TEXT NOT NULL,
    action_hash TEXT NOT NULL,
    action_json TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    catalog_version TEXT NOT NULL,
    knowledge_snapshot_id TEXT NOT NULL,
    status TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    approver_principal_id TEXT,
    decision_at TEXT,
    decision_json TEXT
);

CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    event_id INTEGER NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    result_id TEXT,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, event_id)
);

CREATE TABLE IF NOT EXISTS preferences (
    tenant_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    version INTEGER NOT NULL,
    confirmed_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, principal_id, key)
);

CREATE TABLE IF NOT EXISTS knowledge_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    catalog_version TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    published_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS knowledge_acl (
    source_id TEXT PRIMARY KEY,
    tenant_scope TEXT NOT NULL,
    allowed_roles_json TEXT NOT NULL,
    status TEXT NOT NULL,
    acl_version INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS parallel_groups (
    group_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id),
    plan_hash TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    summary_json TEXT
);

CREATE TABLE IF NOT EXISTS parallel_branches (
    group_id TEXT NOT NULL REFERENCES parallel_groups(group_id),
    branch_id TEXT NOT NULL,
    metric_id TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    error_code TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (group_id, branch_id),
    UNIQUE (group_id, metric_id)
);
"""


class StateStore:
    """Thread-safe SQLite persistence used by the service and probes."""

    def __init__(self, database_path: str | Path = ":memory:", *, clock=utc_now) -> None:
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        if self.database_path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._lock = RLock()
        self._clock = clock
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._connection.executescript(_SCHEMA_SQL)
            self._connection.execute(
                "INSERT OR IGNORE INTO state_meta(key, value) VALUES ('schema_version', ?)",
                (STATE_SCHEMA_VERSION,),
            )

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None  # type: ignore[assignment]
            if connection is not None:
                connection.close()

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _timestamp(self) -> str:
        return isoformat(self._clock())

    def create_run(
        self,
        *,
        run_id: str,
        tenant_id: str,
        principal_id: str,
        role: str,
        question: str,
        mode: str,
        checkpoint: Mapping[str, object] | None = None,
        run_config: Mapping[str, object] | None = None,
        model_call_count: int = 0,
        tool_call_count: int = 0,
    ) -> dict[str, object]:
        now = self._timestamp()
        with self._lock:
            try:
                self._connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, tenant_id, principal_id, role, question, status, mode,
                        created_at, updated_at, checkpoint_json, run_config_json,
                        model_call_count, tool_call_count
                    ) VALUES (?, ?, ?, ?, ?, 'RUNNING', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        tenant_id,
                        principal_id,
                        role,
                        question,
                        mode,
                        now,
                        now,
                        json_text(checkpoint or {}),
                        json_text(run_config or {}),
                        model_call_count,
                        tool_call_count,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise StateStoreError("run_id already exists") from exc
            self.append_event(run_id, "accepted", "RUNNING")
        return self.get_run(run_id)  # type: ignore[return-value]

    def get_run(self, run_id: str) -> dict[str, object] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return _run_from_row(row) if row is not None else None

    def run_ids_with_status(self, *statuses: str) -> tuple[str, ...]:
        """The runs in any of ``statuses``, oldest first (startup ends the ones a process exit left executing)."""

        placeholders = ", ".join("?" for _ in statuses)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT run_id FROM runs WHERE status IN ({placeholders}) ORDER BY created_at, run_id", statuses
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def update_run(self, run_id: str, **fields: object) -> dict[str, object]:
        allowed = {
            "status", "action_json", "approval_id", "result_json", "facts_json",
            "answer", "usage_json", "checkpoint_json", "run_config_json",
            "error_code", "cancel_requested", "sql_exec_count", "model_call_count",
            "tool_call_count",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise StateStoreError(f"unknown run fields: {sorted(unknown)}")
        if not fields:
            result = self.get_run(run_id)
            if result is None:
                raise StateStoreError("run was not found")
            return result
        fields["updated_at"] = self._timestamp()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = [value for value in fields.values()]
        with self._lock:
            cursor = self._connection.execute(
                f"UPDATE runs SET {assignments} WHERE run_id = ?",
                (*values, run_id),
            )
            if cursor.rowcount != 1:
                raise StateStoreError("run was not found")
        return self.get_run(run_id)  # type: ignore[return-value]

    def transition_run(
        self,
        run_id: str,
        from_status: str,
        to_status: str,
        *,
        events: Sequence[Mapping[str, object]] = (),
        approval: Mapping[str, object] | None = None,
        **fields: object,
    ) -> bool:
        """Move a run from ``from_status`` to ``to_status``; False, writing nothing, if it is in another status.

        ``events`` (``append_event``'s keyword arguments, in order) are stored first,
        in the same transaction: a reader never sees the new status without its
        events, nor the events of a transition that did not happen.  ``approval``
        (``create_approval``'s fields) is inserted first, and the run records it.
        """

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
                if row is None or row["status"] != from_status:
                    self._connection.execute("ROLLBACK")
                    return False
                if approval is not None:
                    self._insert_approval(run_id=run_id, **approval)
                    fields.update(action_json=json_text(approval["action"]), approval_id=approval["approval_id"])
                for event in events:
                    self.append_event(run_id, **event)
                self.update_run(run_id, status=to_status, **fields)
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            self._connection.execute("COMMIT")
        return True

    def append_event(
        self,
        run_id: str,
        event_type: str,
        status: str,
        *,
        result_id: str | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        with self._lock:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(event_id), 0) FROM events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            event_id = int(row[0]) + 1
            if event_id > MAX_EVENTS_PER_RUN:
                self._connection.execute(
                    "DELETE FROM events WHERE run_id = ? AND event_id <= ?",
                    (run_id, event_id - MAX_EVENTS_PER_RUN),
                )
            occurred_at = self._timestamp()
            self._connection.execute(
                """
                INSERT INTO events(run_id, event_id, type, status, occurred_at, result_id, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, event_id, event_type, status, occurred_at, result_id, json_text(payload or {})),
            )
        return {
            "event_id": event_id,
            "run_id": run_id,
            "type": event_type,
            "status": status,
            "occurred_at": occurred_at,
            "result_id": result_id,
            "payload": dict(payload or {}),
        }

    def events(
        self,
        run_id: str,
        *,
        after_event_id: int = 0,
        limit: int | None = None,
    ) -> tuple[dict[str, object], ...]:
        if limit is not None and (type(limit) is not int or limit < 1):
            raise StateStoreError("event limit must be a positive integer")
        with self._lock:
            query = """
                SELECT event_id, run_id, type, status, occurred_at, result_id, payload_json
                FROM events WHERE run_id = ? AND event_id > ? ORDER BY event_id
            """
            parameters: tuple[object, ...] = (run_id, after_event_id)
            if limit is not None:
                query += " LIMIT ?"
                parameters += (limit,)
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def event_bounds(self, run_id: str) -> tuple[int, int] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT MIN(event_id), MAX(event_id) FROM events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0]), int(row[1])

    def create_approval(
        self,
        *,
        approval_id: str,
        run_id: str,
        tenant_id: str,
        requester_principal_id: str,
        action_hash: str,
        action: Mapping[str, object],
        policy_version: str,
        catalog_version: str,
        knowledge_snapshot_id: str,
        expires_at: datetime,
    ) -> dict[str, object]:
        with self._lock:
            self._insert_approval(
                approval_id=approval_id,
                run_id=run_id,
                tenant_id=tenant_id,
                requester_principal_id=requester_principal_id,
                action_hash=action_hash,
                action=action,
                policy_version=policy_version,
                catalog_version=catalog_version,
                knowledge_snapshot_id=knowledge_snapshot_id,
                expires_at=expires_at,
            )
            self.update_run(
                run_id,
                status="WAITING_APPROVAL",
                action_json=json_text(action),
                approval_id=approval_id,
            )
            self.append_event(run_id, "waiting", "WAITING_APPROVAL", payload={"approval_id": approval_id})
        return self.get_approval(approval_id)  # type: ignore[return-value]

    def _insert_approval(
        self,
        *,
        approval_id: str,
        run_id: str,
        tenant_id: str,
        requester_principal_id: str,
        action_hash: str,
        action: Mapping[str, object],
        policy_version: str,
        catalog_version: str,
        knowledge_snapshot_id: str,
        expires_at: datetime,
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO approvals(
                    approval_id, run_id, tenant_id, requester_principal_id,
                    action_hash, action_json, policy_version, catalog_version,
                    knowledge_snapshot_id, status, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                """,
                (
                    approval_id,
                    run_id,
                    tenant_id,
                    requester_principal_id,
                    action_hash,
                    json_text(action),
                    policy_version,
                    catalog_version,
                    knowledge_snapshot_id,
                    isoformat(expires_at),
                ),
            )

    def get_approval(self, approval_id: str) -> dict[str, object] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)
            ).fetchone()
        return _approval_from_row(row) if row is not None else None

    def decide_approval(
        self,
        approval_id: str,
        *,
        approver_principal_id: str,
        decision: str,
        now: datetime | None = None,
    ) -> dict[str, object]:
        if decision not in {"approve", "reject"}:
            raise StateStoreError("invalid_approval_decision")
        current_time = now or self._clock()
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)
            ).fetchone()
            if row is None:
                raise StateStoreError("approval_not_found")
            current = _approval_from_row(row)
            if current["status"] != "PENDING":
                return current
            if parse_time(str(current["expires_at"])) <= current_time.astimezone(timezone.utc):
                self._connection.execute(
                    "UPDATE approvals SET status = 'EXPIRED', decision_at = ? WHERE approval_id = ? AND status = 'PENDING'",
                    (isoformat(current_time), approval_id),
                )
                return self.get_approval(approval_id)  # type: ignore[return-value]
            next_status = "APPROVED" if decision == "approve" else "REJECTED"
            self._connection.execute(
                """
                UPDATE approvals
                SET status = ?, approver_principal_id = ?, decision_at = ?, decision_json = ?
                WHERE approval_id = ? AND status = 'PENDING'
                """,
                (
                    next_status,
                    approver_principal_id,
                    isoformat(current_time),
                    json_text({"decision": decision}),
                    approval_id,
                ),
            )
        return self.get_approval(approval_id)  # type: ignore[return-value]

    def put_preference(
        self,
        *,
        tenant_id: str,
        principal_id: str,
        key: str,
        value: str,
        confirmed_at: datetime | None = None,
    ) -> dict[str, object]:
        timestamp = isoformat(confirmed_at or self._clock())
        with self._lock:
            row = self._connection.execute(
                "SELECT version FROM preferences WHERE tenant_id = ? AND principal_id = ? AND key = ?",
                (tenant_id, principal_id, key),
            ).fetchone()
            version = int(row[0]) + 1 if row is not None else 1
            self._connection.execute(
                """
                INSERT INTO preferences(tenant_id, principal_id, key, value, version, confirmed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, principal_id, key) DO UPDATE SET
                    value = excluded.value, version = excluded.version, confirmed_at = excluded.confirmed_at
                """,
                (tenant_id, principal_id, key, value, version, timestamp),
            )
        return self.get_preference(tenant_id, principal_id, key)  # type: ignore[return-value]

    def get_preference(self, tenant_id: str, principal_id: str, key: str) -> dict[str, object] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT key, value, version, confirmed_at FROM preferences WHERE tenant_id = ? AND principal_id = ? AND key = ?",
                (tenant_id, principal_id, key),
            ).fetchone()
        if row is None:
            return None
        return {
            "key": row["key"],
            "value": row["value"],
            "version": int(row["version"]),
            "confirmed_at": row["confirmed_at"],
        }

    def delete_preference(self, tenant_id: str, principal_id: str, key: str) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM preferences WHERE tenant_id = ? AND principal_id = ? AND key = ?",
                (tenant_id, principal_id, key),
            )

    def publish_snapshot(self, snapshot: Mapping[str, object], *, keep_inactive_sources: bool = False) -> None:
        """Publish a snapshot and its source ACL rows (idempotent).

        ``acl_version`` changes only when a source's tenant_scope, allowed_roles
        or status changes; publishing the same content again keeps it, so a
        waiting approval bound to that version stays valid.  With
        ``keep_inactive_sources`` (the product's own publication) a source that
        is not active in the store keeps its whole row: a run-time revocation is
        never undone by publishing knowledge.  An explicit publication without it
        restores the snapshot's ACL, as before.
        """

        snapshot_id = snapshot.get("snapshot_id")
        if type(snapshot_id) is not str or not snapshot_id:
            raise StateStoreError("snapshot_id is required")
        sources = snapshot.get("sources")
        if not isinstance(sources, list):
            raise StateStoreError("snapshot sources must be a list")
        # A fixed SQL fragment (never input): keep a non-active row untouched.
        keep = "knowledge_acl.status <> 'active'" if keep_inactive_sources else "0"
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.execute("UPDATE knowledge_snapshots SET active = 0")
                self._connection.execute(
                    """
                    INSERT INTO knowledge_snapshots(snapshot_id, catalog_version, manifest_json, published_at, active)
                    VALUES (?, ?, ?, ?, 1)
                    ON CONFLICT(snapshot_id) DO UPDATE SET
                        catalog_version = excluded.catalog_version,
                        manifest_json = excluded.manifest_json,
                        published_at = excluded.published_at,
                        active = 1
                    """,
                    (
                        snapshot_id,
                        str(snapshot.get("catalog_version", "catalog-v1")),
                        json_text(snapshot),
                        self._timestamp(),
                    ),
                )
                for source in sources:
                    if not isinstance(source, Mapping):
                        raise StateStoreError("snapshot source must be an object")
                    self._connection.execute(
                        f"""
                        INSERT INTO knowledge_acl(source_id, tenant_scope, allowed_roles_json, status, acl_version)
                        VALUES (?, ?, ?, ?, 1)
                        ON CONFLICT(source_id) DO UPDATE SET
                            tenant_scope = CASE WHEN {keep} THEN knowledge_acl.tenant_scope ELSE excluded.tenant_scope END,
                            allowed_roles_json = CASE WHEN {keep} THEN knowledge_acl.allowed_roles_json ELSE excluded.allowed_roles_json END,
                            status = CASE WHEN {keep} THEN knowledge_acl.status ELSE excluded.status END,
                            acl_version = CASE
                                WHEN {keep} THEN knowledge_acl.acl_version
                                WHEN knowledge_acl.tenant_scope = excluded.tenant_scope
                                     AND knowledge_acl.allowed_roles_json = excluded.allowed_roles_json
                                     AND knowledge_acl.status = excluded.status
                                THEN knowledge_acl.acl_version
                                ELSE knowledge_acl.acl_version + 1
                            END
                        """,
                        (
                            source["source_id"],
                            source["tenant_scope"],
                            json_text(source["allowed_roles"]),
                            source["status"],
                        ),
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def get_snapshot(self, snapshot_id: str | None = None) -> dict[str, object] | None:
        with self._lock:
            if snapshot_id is None:
                row = self._connection.execute(
                    "SELECT manifest_json FROM knowledge_snapshots WHERE active = 1 ORDER BY published_at DESC LIMIT 1"
                ).fetchone()
            else:
                row = self._connection.execute(
                    "SELECT manifest_json FROM knowledge_snapshots WHERE snapshot_id = ?",
                    (snapshot_id,),
                ).fetchone()
        return json_value(row[0]) if row is not None else None  # type: ignore[return-value]

    def set_source_acl(self, source_id: str, *, status: str | None = None, tenant_scope: str | None = None, allowed_roles: Sequence[str] | None = None) -> None:
        with self._lock:
            row = self._connection.execute(
                "SELECT tenant_scope, allowed_roles_json, status FROM knowledge_acl WHERE source_id = ?",
                (source_id,),
            ).fetchone()
            if row is None:
                raise StateStoreError("knowledge source was not found")
            self._connection.execute(
                """
                UPDATE knowledge_acl SET tenant_scope = ?, allowed_roles_json = ?, status = ?, acl_version = acl_version + 1
                WHERE source_id = ?
                """,
                (
                    tenant_scope or row["tenant_scope"],
                    json_text(list(allowed_roles)) if allowed_roles is not None else row["allowed_roles_json"],
                    status or row["status"],
                    source_id,
                ),
            )

    def get_source_acl(self, source_id: str) -> dict[str, object] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT source_id, tenant_scope, allowed_roles_json, status, acl_version FROM knowledge_acl WHERE source_id = ?",
                (source_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "source_id": row["source_id"],
            "tenant_scope": row["tenant_scope"],
            "allowed_roles": json.loads(row["allowed_roles_json"]),
            "status": row["status"],
            "acl_version": int(row["acl_version"]),
        }

    def create_parallel_group(self, *, group_id: str, run_id: str, plan_hash: str, plan: Mapping[str, object], metric_ids: Sequence[str]) -> None:
        now = self._timestamp()
        with self._lock:
            self._connection.execute(
                "INSERT INTO parallel_groups(group_id, run_id, plan_hash, plan_json, status, created_at, updated_at) VALUES (?, ?, ?, ?, 'RUNNING', ?, ?)",
                (group_id, run_id, plan_hash, json_text(plan), now, now),
            )
            for index, metric_id in enumerate(metric_ids, 1):
                self._connection.execute(
                    "INSERT INTO parallel_branches(group_id, branch_id, metric_id, status, updated_at) VALUES (?, ?, ?, 'PENDING', ?)",
                    (group_id, f"{group_id}-branch-{index}", metric_id, now),
                )

    def get_parallel_group(self, run_id: str) -> dict[str, object] | None:
        with self._lock:
            group = self._connection.execute(
                "SELECT * FROM parallel_groups WHERE run_id = ?", (run_id,)
            ).fetchone()
            if group is None:
                return None
            branches = self._connection.execute(
                "SELECT * FROM parallel_branches WHERE group_id = ? ORDER BY branch_id", (group["group_id"],)
            ).fetchall()
        return {
            "group_id": group["group_id"],
            "run_id": group["run_id"],
            "plan_hash": group["plan_hash"],
            "plan": json.loads(group["plan_json"]),
            "status": group["status"],
            "summary": json_value(group["summary_json"], None),
            "branches": [
                {
                    "branch_id": row["branch_id"],
                    "metric_id": row["metric_id"],
                    "status": row["status"],
                    "result": json_value(row["result_json"], None),
                    "error_code": row["error_code"],
                }
                for row in branches
            ],
        }

    def parallel_group_run_ids(self) -> tuple[str, ...]:
        """List durable parallel runs for process-start reconciliation."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT run_id FROM parallel_groups ORDER BY created_at, run_id"
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def update_parallel_branch(self, group_id: str, branch_id: str, *, status: str, result: object = None, error_code: str | None = None) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE parallel_branches SET status = ?, result_json = ?, error_code = ?, updated_at = ? WHERE group_id = ? AND branch_id = ?",
                (status, json_text(result) if result is not None else None, error_code, self._timestamp(), group_id, branch_id),
            )

    def update_parallel_group(self, group_id: str, *, status: str, summary: object = None) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE parallel_groups SET status = ?, summary_json = ?, updated_at = ? WHERE group_id = ?",
                (status, json_text(summary) if summary is not None else None, self._timestamp(), group_id),
            )


def _run_from_row(row: sqlite3.Row) -> dict[str, object]:
    result = dict(row)
    for key in ("action_json", "result_json", "facts_json", "usage_json", "checkpoint_json", "run_config_json"):
        result[key.removesuffix("_json")] = json_value(result.pop(key), None)
    result["cancel_requested"] = bool(result["cancel_requested"])
    return result


def _approval_from_row(row: sqlite3.Row) -> dict[str, object]:
    result = dict(row)
    result["action"] = json_value(result.pop("action_json"), {})
    result["decision"] = json_value(result.pop("decision_json"), None)
    return result


def _event_from_row(row: sqlite3.Row) -> dict[str, object]:
    return {
        "event_id": int(row["event_id"]),
        "run_id": row["run_id"],
        "type": row["type"],
        "status": row["status"],
        "occurred_at": row["occurred_at"],
        "result_id": row["result_id"],
        "payload": json.loads(row["payload_json"]),
    }


__all__ = [
    "MAX_EVENTS_PER_RUN",
    "STATE_SCHEMA_VERSION",
    "StateStore",
    "StateStoreError",
    "isoformat",
    "parse_time",
    "utc_now",
]
