import os

import psycopg
from fastapi.testclient import TestClient
from psycopg.errors import QueryCanceled

from queryshield.api.main import app, get_guarded_executor
from queryshield.auth.identity import IDENTITY_CONFIG
from queryshield.db.guarded import GuardedQueryExecutor


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def send_query(client: TestClient, authorization: str):
    return client.post(
        "/queries",
        headers={"Authorization": authorization},
        json={"question": "2026年9月已支付订单总额"},
    )


def raise_database_error(*args, **kwargs):
    raise psycopg.OperationalError("controlled database probe")


def raise_timeout(*args, **kwargs):
    raise QueryCanceled("controlled timeout probe")


def main() -> int:
    token_env = IDENTITY_CONFIG["a-requester"]["token_env"]
    token = os.getenv(token_env, "").strip()
    if not token:
        print("fault_check_blocked=identity_config_missing")
        return 2
    authorization = f"Bearer {token}"

    with TestClient(app, raise_server_exceptions=False) as client:
        app.dependency_overrides[get_guarded_executor] = lambda: GuardedQueryExecutor(
            connect=raise_database_error
        )
        database_response = send_query(client, authorization)

        require(
            database_response.status_code == 503,
            f"expected 503, got {database_response.status_code}",
        )
        require(
            database_response.json()["error"]["code"]
            == "database_unavailable",
            "wrong database error code",
        )

        app.dependency_overrides[get_guarded_executor] = lambda: GuardedQueryExecutor(
            connect=raise_timeout
        )
        timeout_response = send_query(client, authorization)

        require(
            timeout_response.status_code == 504,
            f"expected 504, got {timeout_response.status_code}",
        )
        require(
            timeout_response.json()["error"]["code"] == "query_timeout",
            "wrong timeout error code",
        )

        for response in (database_response, timeout_response):
            require("postgresql://" not in response.text, "connection URL leaked")
            require("controlled" not in response.text, "internal error leaked")
            # The Agent run is persisted as FAILED with the same code.
            require(response.json().get("status") == "FAILED", "failed run was not persisted as FAILED")
            require(isinstance(response.json().get("run_id"), str), "failed run has no run_id")

    app.dependency_overrides.clear()
    print("fault_check_pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
