from __future__ import annotations

from pathlib import Path
import sqlite3
from threading import RLock
from typing import Self
from uuid import uuid4

from queryshield.agent.proposals import (
    CallIdentityError,
    ModelCallIdentity,
)


class DurableModelCallStore:
    """Minimal local persistence for call identity, not a W04 run/recovery store."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = str(database_path)
        if self._database_path != ":memory:":
            Path(self._database_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(self._database_path, check_same_thread=False)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_call_attempts (
                run_id TEXT NOT NULL,
                model_call_id TEXT NOT NULL,
                attempt_index INTEGER NOT NULL,
                request_id TEXT NOT NULL,
                attempt_kind TEXT NOT NULL CHECK (attempt_kind IN ('new', 'transport_retry')),
                retry_of_model_call_id TEXT,
                PRIMARY KEY (run_id, model_call_id, attempt_index),
                UNIQUE (run_id, request_id)
            )
            """
        )
        self._connection.commit()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
                self._connection = None

    def new_call(self, run_id: str, *, request_id: str | None = None) -> ModelCallIdentity:
        identity = ModelCallIdentity(
            run_id=run_id,
            model_call_id=f"local-{uuid4()}",
            request_id=request_id or str(uuid4()),
            attempt_kind="new",
        )
        try:
            with self._lock:
                self._connection.execute(
                """
                INSERT INTO model_call_attempts
                    (run_id, model_call_id, attempt_index, request_id, attempt_kind, retry_of_model_call_id)
                VALUES (?, ?, 0, ?, 'new', NULL)
                """,
                (identity.run_id, identity.model_call_id, identity.request_id),
                )
                self._connection.commit()
        except sqlite3.IntegrityError as exc:
            raise CallIdentityError("request identity is already stored") from exc
        return identity

    def transport_retry(
        self,
        identity: ModelCallIdentity,
        *,
        request_id: str | None = None,
    ) -> ModelCallIdentity:
        with self._lock:
            canonical = self.get(identity.run_id, identity.model_call_id)
            row = self._connection.execute(
            """
            SELECT COALESCE(MAX(attempt_index), -1) + 1
            FROM model_call_attempts
            WHERE run_id = ? AND model_call_id = ?
            """,
            (canonical.run_id, canonical.model_call_id),
            ).fetchone()
            next_index = int(row[0])
            retry = ModelCallIdentity(
            run_id=canonical.run_id,
            model_call_id=canonical.model_call_id,
            request_id=request_id or str(uuid4()),
            attempt_kind="transport_retry",
            retry_of_model_call_id=canonical.model_call_id,
            )
            try:
                self._connection.execute(
                """
                INSERT INTO model_call_attempts
                    (run_id, model_call_id, attempt_index, request_id, attempt_kind, retry_of_model_call_id)
                VALUES (?, ?, ?, ?, 'transport_retry', ?)
                """,
                (
                    retry.run_id,
                    retry.model_call_id,
                    next_index,
                    retry.request_id,
                    retry.retry_of_model_call_id,
                ),
                )
                self._connection.commit()
            except sqlite3.IntegrityError as exc:
                raise CallIdentityError("request identity is already stored") from exc
            return retry

    def get(self, run_id: str, model_call_id: str) -> ModelCallIdentity:
        with self._lock:
            row = self._connection.execute(
            """
            SELECT run_id, model_call_id, request_id, attempt_kind, retry_of_model_call_id
            FROM model_call_attempts
            WHERE run_id = ? AND model_call_id = ? AND attempt_kind = 'new'
            """,
            (run_id, model_call_id),
            ).fetchone()
        if row is None:
            raise CallIdentityError("model call identity was not found")
        return _identity_from_row(row)

    def attempts(self, run_id: str, model_call_id: str) -> tuple[ModelCallIdentity, ...]:
        self.get(run_id, model_call_id)
        with self._lock:
            rows = self._connection.execute(
            """
            SELECT run_id, model_call_id, request_id, attempt_kind, retry_of_model_call_id
            FROM model_call_attempts
            WHERE run_id = ? AND model_call_id = ?
            ORDER BY attempt_index
            """,
            (run_id, model_call_id),
            ).fetchall()
        return tuple(_identity_from_row(row) for row in rows)


def _identity_from_row(row: tuple[object, ...]) -> ModelCallIdentity:
    return ModelCallIdentity(
        run_id=row[0],
        model_call_id=row[1],
        request_id=row[2],
        attempt_kind=row[3],
        retry_of_model_call_id=row[4],
    )
