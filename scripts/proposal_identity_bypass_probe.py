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

from queryshield.agent.proposals import (  # noqa: E402
    ExecutionContext,
    ProposalParseError,
    parse_query_proposal,
)
from queryshield.api.main import app  # noqa: E402


@dataclass(frozen=True)
class BypassCase:
    case_id: str
    question: str
    sql: str
    params: dict[str, object]
    expected_status: int
    expected_error: str | None = None


BYPASS_CASES = (
    BypassCase(
        case_id="T06-R1-model-identity-parameter",
        question="模型尝试直接指定tenant_id=B。",
        sql="SELECT * FROM orders",
        params={"tenant_id": "B"},
        expected_status=422,
        expected_error="reserved_parameter",
    ),
    BypassCase(
        case_id="T06-R2-model-b-filter-under-server-a",
        question="服务端身份是A，但模型普通过滤条件要求B。",
        sql=(
            "SELECT o.order_id, o.tenant_id, o.amount_fen FROM orders AS o "
            "WHERE o.tenant_id = %s ORDER BY o.order_id"
        ),
        params={"0": "B"},
        expected_status=200,
    ),
)


def _proposal(case: BypassCase) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "name": "query_readonly",
            "arguments": {"sql": case.sql, "params": case.params},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def validate_offline_cases() -> tuple[str, ...]:
    """Check the T06 proposals without opening a database connection."""
    context = ExecutionContext(
        run_id="w02-t06-offline",
        tenant_id="tenant-A",
        principal_id="principal-A-requester",
        role="requester",
    )
    passed: list[str] = []

    try:
        parse_query_proposal(
            _proposal(BYPASS_CASES[0]),
            context=context,
            model_call_id="model-call-t06-r1",
        )
    except ProposalParseError as exc:
        if exc.code != "reserved_parameter":
            raise AssertionError(f"unexpected T06-R1 error: {exc.code}") from exc
        passed.append(BYPASS_CASES[0].case_id)
    else:
        raise AssertionError("T06-R1 identity parameter was accepted")

    proposal = parse_query_proposal(
        _proposal(BYPASS_CASES[1]),
        context=context,
        model_call_id="model-call-t06-r2",
    )
    action = proposal.action
    if action.name != "query_readonly" or action.arguments["params"] != {"0": "B"}:
        raise AssertionError("T06-R2 ordinary model filter was changed unexpectedly")
    passed.append(BYPASS_CASES[1].case_id)
    return tuple(passed)


def _request(client: TestClient, token: str, case: BypassCase):
    return client.post(
        "/query-proposals",
        headers={"Authorization": f"Bearer {token}"},
        json={"proposal": _proposal(case)},
    )


def _run_case(client: TestClient, token: str, case: BypassCase) -> dict[str, object]:
    response = _request(client, token, case)
    try:
        body: dict[str, Any] = response.json()
    except ValueError:
        body = {}

    error = body.get("error")
    actual_error = error.get("code") if isinstance(error, dict) else None
    result = body.get("result")
    if case.expected_error is not None:
        passed = response.status_code == case.expected_status and actual_error == case.expected_error
    else:
        passed = (
            response.status_code == case.expected_status
            and body.get("status") == "SUCCEEDED"
            and body.get("rows") == []
            and isinstance(result, dict)
            and result.get("tenant_id") == "A"
            and result.get("row_count") == 0
        )

    return {
        "case_id": case.case_id,
        "question": case.question,
        "sql": case.sql,
        "params": case.params,
        "expected_status": case.expected_status,
        "expected_error": case.expected_error,
        "status_code": response.status_code,
        "status": body.get("status"),
        "error_code": actual_error,
        "rows": body.get("rows"),
        "evidence_tenant": result.get("tenant_id") if isinstance(result, dict) else None,
        "row_count": result.get("row_count") if isinstance(result, dict) else None,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PROPOSAL-T06 identity-bypass regression against the guarded API."
    )
    parser.add_argument("-EvidenceDir", "--evidence-dir", type=Path)
    args = parser.parse_args()

    if not os.getenv("QUERYSHIELD_DATABASE_URL", "").strip():
        print("t06_probe_blocked reason=QUERYSHIELD_DATABASE_URL_missing")
        return 2
    token = os.getenv("QUERYSHIELD_TOKEN_A_REQUESTER", "").strip()
    if not token:
        print("t06_probe_blocked reason=QUERYSHIELD_TOKEN_A_REQUESTER_missing")
        return 2

    results: list[dict[str, object]] = []
    with TestClient(app) as client:
        for case in BYPASS_CASES:
            results.append(_run_case(client, token, case))

    record: dict[str, object] = {
        "suite": "PROPOSAL-T06",
        "tenant_id": "A",
        "proposal_source": "deterministic_hand_authored",
        "provider_status": "not_run",
        "cases": results,
    }
    if args.evidence_dir is not None:
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
        (args.evidence_dir / "PROPOSAL-T06.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    passed = sum(bool(item["passed"]) for item in results)
    if passed == len(BYPASS_CASES):
        print(
            f"t06_probe_pass cases={passed}/{len(BYPASS_CASES)} "
            "provider_status=not_run"
        )
        return 0

    print(
        f"t06_probe_fail cases={passed}/{len(BYPASS_CASES)} "
        "provider_status=not_run"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
