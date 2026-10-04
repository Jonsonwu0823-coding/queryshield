from __future__ import annotations

import pytest

from queryshield.agent import CallIdentityError, DurableModelCallStore


def test_call_identity_survives_store_reopen_and_keeps_retry_lineage(tmp_path) -> None:
    database_path = tmp_path / "model-calls.sqlite3"

    with DurableModelCallStore(database_path) as first_store:
        original = first_store.new_call("run-1", request_id="request-1")
        retry = first_store.transport_retry(original, request_id="request-2")

    with DurableModelCallStore(database_path) as reopened_store:
        restored = reopened_store.get("run-1", original.model_call_id)
        attempts = reopened_store.attempts("run-1", original.model_call_id)

        assert restored == original
        assert retry.model_call_id == original.model_call_id
        assert [attempt.attempt_kind for attempt in attempts] == [
            "new",
            "transport_retry",
        ]
        assert attempts[1].retry_of_model_call_id == original.model_call_id

        next_call = reopened_store.new_call("run-1", request_id="request-3")
        assert next_call.model_call_id != original.model_call_id


def test_store_rejects_duplicate_request_identity(tmp_path) -> None:
    database_path = tmp_path / "model-calls.sqlite3"

    with DurableModelCallStore(database_path) as store:
        original = store.new_call("run-1", request_id="request-1")

        with pytest.raises(CallIdentityError):
            store.new_call("run-1", request_id="request-1")

        with pytest.raises(CallIdentityError):
            store.transport_retry(original, request_id="request-1")


def test_store_schema_contains_identity_only() -> None:
    with DurableModelCallStore(":memory:") as store:
        columns = {
            row[1]
            for row in store._connection.execute(
                "PRAGMA table_info(model_call_attempts)"
            ).fetchall()
        }

    assert columns == {
        "run_id",
        "model_call_id",
        "attempt_index",
        "request_id",
        "attempt_kind",
        "retry_of_model_call_id",
    }
    assert "prompt" not in columns
    assert "response" not in columns
    assert "api_key" not in columns
