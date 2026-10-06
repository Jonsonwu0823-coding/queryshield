from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from queryshield.api.main import app  # noqa: E402


@dataclass(frozen=True)
class SafeCase:
    case_id: str
    question: str
    sql: str
    params: dict[str, object]
    expected_rows: tuple[dict[str, object], ...]
    expected_summary: str
    identity: str = "requester"


@dataclass(frozen=True)
class RejectionCase:
    case_id: str
    question: str
    sql: str
    params: dict[str, object]
    expected_status: int
    expected_error: str


SAFE_CASES = (
    SafeCase(
        case_id="T05-Q1-filter-paid-orders",
        question="A租户有哪些已支付订单？",
        sql=(
            "SELECT o.order_id, o.amount_fen FROM orders AS o "
            "WHERE o.status = %s ORDER BY o.order_id"
        ),
        params={"0": "paid"},
        expected_rows=(
            {"order_id": "o1", "amount_fen": 10000},
            {"order_id": "o2", "amount_fen": 5000},
        ),
        expected_summary="过滤：A租户的paid订单为o1和o2",
    ),
    SafeCase(
        case_id="T05-Q2-aggregate-paid-orders",
        question="A租户已支付订单有几笔、总金额是多少？",
        sql=(
            "SELECT COUNT(*) AS paid_count, "
            "COALESCE(SUM(amount_fen), 0) AS gross_fen "
            "FROM orders WHERE status = %s"
        ),
        params={"0": "paid"},
        expected_rows=({"paid_count": 2, "gross_fen": 15000},),
        expected_summary="聚合：2笔，15000分",
    ),
    SafeCase(
        case_id="T05-Q3-join-customer-orders",
        question="A租户已支付订单分别属于哪些客户？",
        sql=(
            "SELECT o.order_id, c.name, o.amount_fen "
            "FROM orders AS o INNER JOIN customers AS c "
            "ON o.customer_id = c.customer_id "
            "WHERE o.status = %s ORDER BY o.order_id"
        ),
        params={"0": "paid"},
        expected_rows=(
            {"order_id": "o1", "name": "甲", "amount_fen": 10000},
            {"order_id": "o2", "name": "乙", "amount_fen": 5000},
        ),
        expected_summary="JOIN：o1属于甲，o2属于乙",
        # customers.name needs the approval path for requesters, so this
        # case runs as the same-tenant approver.
        identity="approver",
    ),
    SafeCase(
        case_id="T05-Q4-join-refund-total",
        question="A租户订单o1已经退款多少？",
        sql=(
            "SELECT o.order_id, o.amount_fen, "
            "COALESCE(SUM(r.amount_fen), 0) AS refund_fen "
            "FROM orders AS o INNER JOIN refunds AS r "
            "ON o.order_id = r.order_id "
            "WHERE o.order_id = %s "
            "GROUP BY o.order_id, o.amount_fen ORDER BY o.order_id"
        ),
        params={"0": "o1"},
        expected_rows=({"order_id": "o1", "amount_fen": 10000, "refund_fen": 3000},),
        expected_summary="JOIN+聚合：o1退款3000分",
    ),
    SafeCase(
        case_id="T05-Q5-same-order-id-is-scoped",
        question="查询订单o1的金额，确认不会读到其他租户的同名订单。",
        sql=(
            "SELECT o.order_id, o.tenant_id, o.amount_fen FROM orders AS o "
            "WHERE o.order_id = %s ORDER BY o.order_id"
        ),
        params={"0": "o1"},
        expected_rows=({"order_id": "o1", "tenant_id": "A", "amount_fen": 10000},),
        expected_summary="归属：只返回A/o1的10000分，不返回B/o1的990000分",
    ),
)


REJECTION_CASES = (
    RejectionCase(
        case_id="T05-R1-write-statement",
        question="尝试修改订单金额。",
        sql="UPDATE orders SET amount_fen = 1",
        params={},
        expected_status=403,
        expected_error="statement_not_allowed",
    ),
    RejectionCase(
        case_id="T05-R2-multiple-statements",
        question="在查询后追加删除语句。",
        sql="SELECT COUNT(*) FROM orders; DELETE FROM orders",
        params={},
        expected_status=403,
        expected_error="multiple_statements",
    ),
    RejectionCase(
        case_id="T05-R3-system-table",
        question="读取PostgreSQL系统目录。",
        sql="SELECT relname FROM pg_catalog.pg_class",
        params={},
        expected_status=403,
        expected_error="table_not_allowed",
    ),
    RejectionCase(
        case_id="T05-R4-unsupported-function",
        question="调用未加入白名单的函数。",
        sql="SELECT LOWER(status) FROM orders",
        params={},
        expected_status=403,
        expected_error="function_not_allowed",
    ),
    RejectionCase(
        case_id="T05-R5-left-join",
        question="使用当前不支持的LEFT JOIN。",
        sql=(
            "SELECT * FROM orders LEFT JOIN customers "
            "ON orders.customer_id = customers.customer_id"
        ),
        params={},
        expected_status=403,
        expected_error="unsupported_syntax",
    ),
)


TOKEN_ENV_BY_IDENTITY = {
    "requester": "QUERYSHIELD_TOKEN_A_REQUESTER",
    "approver": "QUERYSHIELD_TOKEN_A_APPROVER",
}


def _proposal(sql: str, params: dict[str, object]) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "name": "query_readonly",
            "arguments": {"sql": sql, "params": params},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _request(client: TestClient, token: str, sql: str, params: dict[str, object]):
    return client.post(
        "/query-proposals",
        headers={"Authorization": f"Bearer {token}"},
        json={"proposal": _proposal(sql, params)},
    )


def _run_safe_case(client: TestClient, token: str, case: SafeCase) -> dict[str, object]:
    response = _request(client, token, case.sql, case.params)
    try:
        body: dict[str, Any] = response.json()
    except ValueError:
        body = {}

    result = body.get("result")
    passed = (
        response.status_code == 200
        and body.get("status") == "SUCCEEDED"
        and body.get("rows") == list(case.expected_rows)
        and isinstance(result, dict)
        and result.get("tenant_id") == "A"
        and result.get("row_count") == len(case.expected_rows)
    )
    return {
        "case_id": case.case_id,
        "question": case.question,
        "kind": "safe",
        "sql": case.sql,
        "params": case.params,
        "expected_summary": case.expected_summary,
        "identity": case.identity,
        "status_code": response.status_code,
        "status": body.get("status"),
        "rows": body.get("rows"),
        "result_id": result.get("result_id") if isinstance(result, dict) else None,
        "tenant_id": result.get("tenant_id") if isinstance(result, dict) else None,
        "row_count": result.get("row_count") if isinstance(result, dict) else None,
        "passed": passed,
    }


def _run_rejection_case(
    client: TestClient, token: str, case: RejectionCase
) -> dict[str, object]:
    response = _request(client, token, case.sql, case.params)
    try:
        body: dict[str, Any] = response.json()
    except ValueError:
        body = {}
    error = body.get("error")
    actual_error = error.get("code") if isinstance(error, dict) else None
    passed = response.status_code == case.expected_status and actual_error == case.expected_error
    return {
        "case_id": case.case_id,
        "question": case.question,
        "kind": "rejection",
        "sql": case.sql,
        "params": case.params,
        "expected_status": case.expected_status,
        "expected_error": case.expected_error,
        "status_code": response.status_code,
        "error_code": actual_error,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PROPOSAL-T05 deterministic proposals against the real guarded API."
    )
    parser.add_argument("-EvidenceDir", "--evidence-dir", type=Path)
    args = parser.parse_args()

    if not os.getenv("QUERYSHIELD_DATABASE_URL", "").strip():
        print("t05_probe_blocked reason=QUERYSHIELD_DATABASE_URL_missing")
        return 2
    tokens: dict[str, str] = {}
    for identity, env_name in TOKEN_ENV_BY_IDENTITY.items():
        token = os.getenv(env_name, "").strip()
        if not token:
            print(f"t05_probe_blocked reason={env_name}_missing")
            return 2
        tokens[identity] = token

    safe_results: list[dict[str, object]] = []
    rejection_results: list[dict[str, object]] = []
    with TestClient(app) as client:
        for case in SAFE_CASES:
            safe_results.append(_run_safe_case(client, tokens[case.identity], case))
        for case in REJECTION_CASES:
            rejection_results.append(_run_rejection_case(client, tokens["requester"], case))

    record: dict[str, object] = {
        "suite": "PROPOSAL-T05",
        "tenant_id": "A",
        "proposal_source": "deterministic_hand_authored",
        "provider_status": "not_run",
        "safe_cases": safe_results,
        "rejection_cases": rejection_results,
    }
    if args.evidence_dir is not None:
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
        (args.evidence_dir / "PROPOSAL-T05.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    safe_passed = sum(bool(item["passed"]) for item in safe_results)
    rejection_passed = sum(bool(item["passed"]) for item in rejection_results)
    if safe_passed == len(SAFE_CASES) and rejection_passed == len(REJECTION_CASES):
        print(
            "t05_probe_pass "
            f"safe={safe_passed}/{len(SAFE_CASES)} "
            f"rejections={rejection_passed}/{len(REJECTION_CASES)} "
            "provider_status=not_run"
        )
        return 0

    print(
        "t05_probe_fail "
        f"safe={safe_passed}/{len(SAFE_CASES)} "
        f"rejections={rejection_passed}/{len(REJECTION_CASES)} "
        "provider_status=not_run"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
