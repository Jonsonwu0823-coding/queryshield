from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from queryshield.api.main import app  # noqa: E402


STATUS_SUCCESS = 200
STATUS_UNPROCESSABLE = 422

FS01_SQL = (
    "SELECT o.order_id, o.amount_fen FROM orders AS o "
    "WHERE o.status = %s ORDER BY o.order_id"
)
FS02_SQL = "SELECT o.order_id, o.tenant_id, o.amount_fen FROM orders AS o"
FS02_B_FILTER_SQL = FS02_SQL + " WHERE o.tenant_id = %s ORDER BY o.order_id"


def _proposal(
    sql: str,
    params: dict[str, object],
    *,
    extra_fields: dict[str, object] | None = None,
) -> str:
    payload: dict[str, object] = {
        "type": "tool_call",
        "name": "query_readonly",
        "arguments": {"sql": sql, "params": params},
    }
    if extra_fields:
        payload.update(extra_fields)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _request(client: TestClient, token: str, raw_proposal: str):
    return client.post(
        "/query-proposals",
        headers={"Authorization": f"Bearer {token}"},
        json={"proposal": raw_proposal},
    )


def _json_body(response: Any) -> dict[str, object]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _error_summary(response: Any) -> dict[str, object]:
    body = _json_body(response)
    error = body.get("error")
    error_code = error.get("code") if isinstance(error, dict) else None
    return {
        "status_code": response.status_code,
        "error_code": error_code,
        "contains_rows": "rows" in body or "result" in body,
    }


def run_fs01(client: TestClient, token: str) -> dict[str, object]:
    """Run the same question twice and change only its bound query parameter."""
    paid_response = _request(
        client,
        token,
        _proposal(FS01_SQL, {"0": "paid"}),
    )
    cancelled_response = _request(
        client,
        token,
        _proposal(FS01_SQL, {"0": "cancelled"}),
    )

    paid_body = _json_body(paid_response)
    cancelled_body = _json_body(cancelled_response)
    paid_result = paid_body.get("result")
    cancelled_result = cancelled_body.get("result")
    paid_result = paid_result if isinstance(paid_result, dict) else {}
    cancelled_result = cancelled_result if isinstance(cancelled_result, dict) else {}

    paid_rows = paid_body.get("rows")
    cancelled_rows = cancelled_body.get("rows")
    passed = (
        paid_response.status_code == STATUS_SUCCESS
        and cancelled_response.status_code == STATUS_SUCCESS
        and paid_body.get("status") == "SUCCEEDED"
        and cancelled_body.get("status") == "SUCCEEDED"
        and paid_result.get("run_id") == paid_body.get("run_id")
        and cancelled_result.get("run_id") == cancelled_body.get("run_id")
        and paid_result.get("run_id") != cancelled_result.get("run_id")
        and paid_result.get("result_id") != cancelled_result.get("result_id")
        and paid_result.get("tenant_id") == "A"
        and cancelled_result.get("tenant_id") == "A"
        and paid_result.get("principal_id") == "a-requester"
        and cancelled_result.get("principal_id") == "a-requester"
        and paid_rows == [
            {"order_id": "o1", "amount_fen": 10000},
            {"order_id": "o2", "amount_fen": 5000},
        ]
        and cancelled_rows == [{"order_id": "o3", "amount_fen": 9000}]
        and paid_result.get("row_count") == len(paid_rows or [])
        and cancelled_result.get("row_count") == len(cancelled_rows or [])
        and paid_result.get("query_sha256") == cancelled_result.get("query_sha256")
        and paid_result.get("params_sha256") != cancelled_result.get("params_sha256")
    )

    return {
        "case_id": "W02-FS01-same-question-two-runs",
        "passed": passed,
        "same_query": True,
        "changed_parameter": {"from": "paid", "to": "cancelled"},
        "runs": [
            {
                "run_id": paid_body.get("run_id"),
                "result_id": paid_result.get("result_id"),
                "tenant_id": paid_result.get("tenant_id"),
                "principal_id": paid_result.get("principal_id"),
                "rows": paid_rows,
                "row_count": paid_result.get("row_count"),
                "query_sha256": paid_result.get("query_sha256"),
                "params_sha256": paid_result.get("params_sha256"),
            },
            {
                "run_id": cancelled_body.get("run_id"),
                "result_id": cancelled_result.get("result_id"),
                "tenant_id": cancelled_result.get("tenant_id"),
                "principal_id": cancelled_result.get("principal_id"),
                "rows": cancelled_rows,
                "row_count": cancelled_result.get("row_count"),
                "query_sha256": cancelled_result.get("query_sha256"),
                "params_sha256": cancelled_result.get("params_sha256"),
            },
        ],
    }


def run_fs02(client: TestClient, token: str) -> dict[str, object]:
    """Check that model-supplied identity/result metadata cannot enter the result."""
    forged_metadata_response = _request(
        client,
        token,
        _proposal(
            FS02_SQL,
            {},
            extra_fields={
                "tenant_id": "B",
                "principal_id": "b-requester",
                "result_id": "forged-result-id",
            },
        ),
    )
    reserved_parameter_response = _request(
        client,
        token,
        _proposal(
            FS02_SQL,
            {"tenant_id": "B", "principal_id": "b-requester"},
        ),
    )
    server_scope_response = _request(
        client,
        token,
        _proposal(FS02_B_FILTER_SQL, {"0": "B"}),
    )

    forged_summary = _error_summary(forged_metadata_response)
    reserved_summary = _error_summary(reserved_parameter_response)
    server_scope_body = _json_body(server_scope_response)
    server_scope_result = server_scope_body.get("result")
    server_scope_result = (
        server_scope_result if isinstance(server_scope_result, dict) else {}
    )
    server_scope_passed = (
        server_scope_response.status_code == STATUS_SUCCESS
        and server_scope_body.get("status") == "SUCCEEDED"
        and server_scope_body.get("rows") == []
        and server_scope_result.get("tenant_id") == "A"
        and server_scope_result.get("principal_id") == "a-requester"
        and server_scope_result.get("row_count") == 0
    )

    cases = [
        {
            "case_id": "W02-FS02-forged-result-metadata",
            **forged_summary,
            "expected_status": STATUS_UNPROCESSABLE,
            "expected_error": "unknown_field",
            "passed": (
                forged_summary["status_code"] == STATUS_UNPROCESSABLE
                and forged_summary["error_code"] == "unknown_field"
                and forged_summary["contains_rows"] is False
            ),
        },
        {
            "case_id": "W02-FS02-reserved-identity-parameters",
            **reserved_summary,
            "expected_status": STATUS_UNPROCESSABLE,
            "expected_error": "reserved_parameter",
            "passed": (
                reserved_summary["status_code"] == STATUS_UNPROCESSABLE
                and reserved_summary["error_code"] == "reserved_parameter"
                and reserved_summary["contains_rows"] is False
            ),
        },
        {
            "case_id": "W02-FS02-server-tenant-wins",
            "status_code": server_scope_response.status_code,
            "error_code": None,
            "contains_rows": True,
            "expected_status": STATUS_SUCCESS,
            "expected_error": None,
            "evidence_tenant": server_scope_result.get("tenant_id"),
            "evidence_principal": server_scope_result.get("principal_id"),
            "rows": server_scope_body.get("rows"),
            "row_count": server_scope_result.get("row_count"),
            "passed": server_scope_passed,
        },
    ]
    return {
        "cases": cases,
        "passed": all(bool(case["passed"]) for case in cases),
    }


def validate_fs01_record(record: dict[str, object]) -> bool:
    return bool(record.get("passed")) and record.get("case_id") == "W02-FS01-same-question-two-runs"


def validate_fs02_record(record: dict[str, object]) -> bool:
    cases = record.get("cases")
    return bool(record.get("passed")) and isinstance(cases, list) and len(cases) == 3


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run W02 facts/state evidence checks against the guarded API."
    )
    parser.add_argument(
        "-CheckId",
        "--check-id",
        choices=("W02-FS01", "W02-FS02"),
        required=True,
    )
    parser.add_argument("-EvidenceDir", "--evidence-dir", type=Path)
    args = parser.parse_args()

    if not os.getenv("QUERYSHIELD_DATABASE_URL", "").strip():
        print("fs_probe_blocked reason=QUERYSHIELD_DATABASE_URL_missing")
        return 2
    token = os.getenv("QUERYSHIELD_TOKEN_A_REQUESTER", "").strip()
    if not token:
        print("fs_probe_blocked reason=QUERYSHIELD_TOKEN_A_REQUESTER_missing")
        return 2

    with TestClient(app) as client:
        result = run_fs01(client, token) if args.check_id == "W02-FS01" else run_fs02(client, token)

    record: dict[str, object] = {
        "suite": args.check_id,
        "tenant_id": "A",
        "proposal_source": "deterministic_hand_authored",
        "provider_status": "not_run",
        "result": result,
    }
    if args.evidence_dir is not None:
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
        (args.evidence_dir / f"{args.check_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    passed = (
        validate_fs01_record(result)
        if args.check_id == "W02-FS01"
        else validate_fs02_record(result)
    )
    if passed:
        print(f"{args.check_id.lower()}_probe_pass provider_status=not_run")
        return 0

    print(f"{args.check_id.lower()}_probe_fail provider_status=not_run")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
