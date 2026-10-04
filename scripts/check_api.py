import argparse
import os

from fastapi.testclient import TestClient

from queryshield.api.main import app
from queryshield.auth.identity import IDENTITY_CONFIG


def send_query(client: TestClient, token: str, payload: dict):
    return client.post(
        "/queries",
        headers={"Authorization": token},
        json=payload,
    )


def _fact_value(body: dict, metric_id: str) -> object:
    facts = [item for item in body["facts"]["facts"] if item["metric_id"] == metric_id]
    assert len(facts) == 1
    assert facts[0]["tenant_id"] == body["tenant_id"]
    assert facts[0]["principal_id"] == body["principal_id"]
    return facts[0]["value"]


def bearer_for(identity_name: str) -> str:
    identity = IDENTITY_CONFIG[identity_name]
    token = os.getenv(identity["token_env"], "").strip()
    if not token:
        raise RuntimeError("identity config is missing")
    return f"Bearer {token}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fake", "real"), default="fake")
    args = parser.parse_args()
    if args.mode == "real":
        print("api_check_blocked reason=real_provider_not_implemented")
        return 2

    try:
        a_requester = bearer_for("a-requester")
        b_requester = bearer_for("b-requester")
    except RuntimeError:
        print("api_check_blocked reason=identity_config_missing")
        return 2

    with TestClient(app, raise_server_exceptions=False) as client:
        normal = send_query(
            client,
            a_requester,
            {"question": "2026年9月已支付订单总额"},
        )

        normal_body = normal.json()

        assert normal.status_code == 200
        assert normal_body["mode"] == "fake"
        # B2b: /queries runs the product Agent; the value is a verified fact.
        assert normal_body["status"] == "SUCCEEDED"
        assert _fact_value(normal_body, "gross_fen") == 15000
        assert normal_body["result"]["tenant_id"] == "A"
        assert normal_body["result"]["run_id"] == normal_body["run_id"]

        extra_field = send_query(
            client,
            a_requester,
            {"question": "hello", "tenant_id": "B"},
        )

        assert extra_field.status_code == 422
        
        tenant_b = send_query(
            client,
            b_requester,
            {"question": "2026年9月已支付订单总额"},
        )

        assert tenant_b.status_code == 200
        assert _fact_value(tenant_b.json(), "gross_fen") == 990000
        assert tenant_b.json()["result"]["tenant_id"] == "B"

        unknown_token = send_query(
            client,
            f"Bearer {os.urandom(16).hex()}",
            {"question": "hello"},
        )

        assert unknown_token.status_code == 401

    print("api_check_pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
