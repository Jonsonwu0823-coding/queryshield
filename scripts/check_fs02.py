import os
from datetime import datetime, timezone

import psycopg
from fastapi.testclient import TestClient
from queryshield.api.main import app
from queryshield.auth.identity import IDENTITY_CONFIG


START = datetime(2026, 9, 1, tzinfo=timezone.utc)
END = datetime(2026, 10, 1, tzinfo=timezone.utc)

def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _fact(client: TestClient, authorization: str, question: str, metric_id: str) -> tuple[int, dict]:
    response = client.post(
        "/queries",
        headers={"Authorization": authorization},
        json={"question": question},
    )
    require(response.status_code == 200, f"API status={response.status_code}")
    body = response.json()
    require(body["mode"] == "fake", "FS02 did not use fake mode")
    require(body["status"] == "SUCCEEDED", "API run did not succeed")
    facts = [item for item in body["facts"]["facts"] if item["metric_id"] == metric_id]
    require(len(facts) == 1, "API returned an unexpected fact count")
    require(facts[0]["result_id"] == body["result"]["result_id"], "fact is not bound to this run's result")
    require(facts[0]["display_value"] in body["answer"], "server answer does not render the verified fact")
    return int(facts[0]["value"]), body


def fetch_summary(client: TestClient, authorization: str) -> tuple[int, int, dict]:
    """B2b: gross and net are two Agent runs; values come from verified facts."""

    gross, gross_body = _fact(client, authorization, "2026年9月已支付订单总额", "gross_fen")
    net, net_body = _fact(client, authorization, "2026年9月退款后净额", "net_fen")
    return gross, net, {"gross": gross_body, "net": net_body}


def update_order(database_url: str, amount_fen: int) -> None:
    with psycopg.connect(database_url) as conn:
        with conn.transaction():
            result = conn.execute(
                """
                UPDATE orders
                SET amount_fen = %s
                WHERE tenant_id = %s AND order_id = %s
                """,
                (amount_fen, "A", "o1"),
            )
            require(result.rowcount == 1, "temporary order update failed")


def main() -> int:
    stage = "config"
    database_url = os.getenv("QUERYSHIELD_BOOTSTRAP_DATABASE_URL")
    readonly_url = os.getenv("QUERYSHIELD_DATABASE_URL")

    if not database_url:
        print("fs02_blocked=bootstrap_database_url_missing")
        return 2
    if not readonly_url:
        print("fs02_blocked=database_url_missing")
        return 2

    token = os.getenv(IDENTITY_CONFIG["a-requester"]["token_env"], "").strip()
    if not token:
        print("fs02_blocked=identity_config_missing")
        return 2
    authorization = f"Bearer {token}"

    try:
        stage = "bootstrap_connect"
        with psycopg.connect(database_url) as conn:
            database_name = conn.execute(
                "SELECT current_database()"
            ).fetchone()[0]

            if not str(database_name).endswith("_test"):
                print("fs02_blocked=database_is_not_test_database")
                return 2

            stage = "read_original"
            with conn.transaction():
                original_amount = conn.execute(
                    """
                    SELECT amount_fen
                    FROM orders
                    WHERE tenant_id = %s
                      AND order_id = %s
                    """,
                    ("A", "o1"),
                ).fetchone()

                require(original_amount is not None, "order A/o1 missing")
                require(int(original_amount[0]) == 10000, "unexpected original amount")

            with TestClient(app, raise_server_exceptions=False) as client:
                stage = "api_before"
                before_gross, before_net, before_body = fetch_summary(
                    client, authorization
                )
                require(before_gross == 15000, "unexpected baseline gross")
                require(before_net == 12000, "unexpected baseline net")

                changed = False
                try:
                    stage = "update_order"
                    update_order(database_url, 11000)
                    changed = True
                    stage = "api_after"
                    after_gross, after_net, after_body = fetch_summary(
                        client, authorization
                    )
                finally:
                    if changed:
                        stage = "restore_order"
                        update_order(database_url, 10000)

                stage = "api_restored"
                restored_gross, restored_net, restored_body = fetch_summary(
                    client, authorization
                )

                require(after_gross == 16000, "API gross did not increase by 1000")
                require(after_net == 13000, "API net did not increase by 1000")
                require("160.00" in after_body["gross"]["answer"], "API answer did not reflect changed gross")
                require(restored_gross == before_gross, "API gross did not recover baseline")
                require(restored_net == before_net, "API net did not recover baseline")
                require(
                    restored_body["gross"]["result"]["rows"] == before_body["gross"]["result"]["rows"],
                    "API rows did not recover baseline",
                )

        print(
            "fs02_check_pass "
            f"before={before_net} "
            f"after={after_net} "
            f"restored={restored_net}"
        )
        return 0

    except psycopg.OperationalError:
        print(f"fs02_blocked=database_unavailable stage={stage}")
        return 2
    except Exception as exc:
        print(f"fs02_check_failed={type(exc).__name__}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
