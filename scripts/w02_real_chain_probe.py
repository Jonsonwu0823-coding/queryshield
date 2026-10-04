from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from queryshield.api.main import app  # noqa: E402
from w02_t05_probe import SAFE_CASES  # noqa: E402


def _missing_configuration() -> list[str]:
    names = [
        "QUERYSHIELD_MODEL_BASE_URL",
        "QUERYSHIELD_MODEL_API_KEY",
        "QUERYSHIELD_MODEL_NAME",
        "QUERYSHIELD_DATABASE_URL",
        "QUERYSHIELD_TOKEN_A_REQUESTER",
    ]
    return [name for name in names if not (os.getenv(name) or "").strip()]


def answer_matches_expected_rows(
    answer: object, expected_rows: tuple[dict[str, object], ...]
) -> bool:
    """Accept the server's row-summary wording without trusting its numbers."""
    if not isinstance(answer, str) or not answer.strip():
        return False

    compact = re.sub(r"\s+", "", answer)
    counts = re.findall(r"(?:共)?返回(?:了)?(\d+)(?:行|条(?:记录|结果)?)", compact)
    return len(counts) == 1 and int(counts[0]) == len(expected_rows)


def _run_case(client: TestClient, token: str, case: Any) -> dict[str, object]:
    response = client.post(
        "/queries",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": case.question},
    )
    try:
        body: dict[str, Any] = response.json()
    except ValueError:
        body = {}

    result = body.get("result")
    usage_items = body.get("usage")
    usage = usage_items[0] if isinstance(usage_items, list) and usage_items else None
    run_id = body.get("run_id")
    answer = body.get("answer")
    proposal_sha256 = body.get("proposal_sha256")
    actual_rows = body.get("rows")
    evidence_ok = (
        isinstance(result, dict)
        and isinstance(run_id, str)
        and bool(run_id)
        and result.get("run_id") == run_id
        and result.get("tenant_id") == "A"
        and result.get("row_count") == len(case.expected_rows)
        and result.get("rows") == actual_rows
    )
    hash_ok = (
        isinstance(proposal_sha256, str)
        and len(proposal_sha256) == 64
        and isinstance(result, dict)
        and all(
            isinstance(result.get(key), str) and len(result[key]) == 64
            for key in ("query_sha256", "params_sha256")
        )
    )
    answer_ok = answer_matches_expected_rows(answer, case.expected_rows)
    usage_ok = (
        isinstance(usage, dict)
        and isinstance(usage.get("model_call_id"), str)
        and bool(usage["model_call_id"])
        and isinstance(usage.get("provider_call_id"), str)
        and bool(usage["provider_call_id"])
        and usage.get("usage_status") in {"known", "unknown"}
    )
    if usage_ok and usage.get("usage_status") == "unknown":
        usage_ok = all(
            usage.get(key) is None
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
    if usage_ok and usage.get("usage_status") == "known":
        values = [usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
        usage_ok = (
            all(type(value) is int and value >= 0 for value in values)
            and values[0] + values[1] == values[2]
        )

    passed = (
        response.status_code == 200
        and body.get("status") == "SUCCEEDED"
        and body.get("mode") == "real"
        and actual_rows == list(case.expected_rows)
        and evidence_ok
        and hash_ok
        and answer_ok
        and usage_ok
    )
    return {
        "case_id": case.case_id,
        "question": case.question,
        "proposal_source": "real_model_adapter",
        "status_code": response.status_code,
        "status": body.get("status"),
        "mode": body.get("mode"),
        "run_id": run_id,
        "answer": answer,
        "answer_expected_row_count": len(case.expected_rows),
        "answer_check": answer_ok,
        "proposal_sha256": proposal_sha256,
        "rows": actual_rows,
        "result": result,
        "result_id": result.get("result_id") if isinstance(result, dict) else None,
        "query_sha256": result.get("query_sha256") if isinstance(result, dict) else None,
        "params_sha256": result.get("params_sha256") if isinstance(result, dict) else None,
        "result_binding_check": evidence_ok,
        "hash_check": hash_ok,
        "usage": usage,
        "error": body.get("error"),
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run five natural-language questions through the real W02 model chain."
    )
    parser.add_argument("-EvidenceDir", "--evidence-dir", type=Path)
    args = parser.parse_args()

    if os.getenv("QUERYSHIELD_PROVIDER_MODE", "").lower() != "real":
        print("real_chain_blocked reason=QUERYSHIELD_PROVIDER_MODE_not_real")
        return 2
    missing = _missing_configuration()
    if missing:
        print("real_chain_blocked reason=missing_configuration fields=" + ",".join(missing))
        return 2

    token = os.environ["QUERYSHIELD_TOKEN_A_REQUESTER"].strip()
    results: list[dict[str, object]] = []
    # Keep a failed case in the evidence denominator if an unexpected server
    # exception escapes the route; a 500 is still a failure, never a success.
    with TestClient(app, raise_server_exceptions=False) as client:
        for case in SAFE_CASES:
            results.append(_run_case(client, token, case))

    record = {
        "suite": "W02-T05-real-chain",
        "tenant_id": "A",
        "proposal_source": "real_model_adapter",
        "provider_status": "run",
        "safe_cases": results,
    }
    if args.evidence_dir is not None:
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
        (args.evidence_dir / "W02-T05-real-chain.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    passed = sum(bool(item["passed"]) for item in results)
    if passed == len(SAFE_CASES):
        print(
            "real_chain_probe_pass "
            f"safe={passed}/{len(SAFE_CASES)} provider_status=run"
        )
        return 0

    print(
        "real_chain_probe_fail "
        f"safe={passed}/{len(SAFE_CASES)} provider_status=run"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
