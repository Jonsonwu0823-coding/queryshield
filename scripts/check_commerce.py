import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from queryshield.agent.proposals import ExecutionContext
from queryshield.db.guarded import GuardedQueryExecutor
from psycopg.rows import dict_row

from queryshield.db.commerce import COMMERCE_SUMMARY_SQL
from queryshield.db.readonly import bind_transaction_tenant, connect_readonly
from queryshield.providers.fake_model import FakeModel, fen_to_yuan


START = datetime(2026, 9, 1, tzinfo=timezone.utc)
END = datetime(2026, 10, 1, tzinfo=timezone.utc)

PAID_SUMMARY_QUERY = """
SELECT COUNT(*) AS paid_count,
       COALESCE(SUM(o.amount_fen), 0) AS gross_fen
FROM orders AS o
WHERE o.status = %s
  AND o.created_at >= %s
  AND o.created_at < %s
"""

REFUND_SUMMARY_QUERY = """
SELECT COALESCE(SUM(r.amount_fen), 0) AS refund_fen
FROM refunds AS r
INNER JOIN orders AS o
  ON r.order_id = o.order_id
 AND r.tenant_id = o.tenant_id
WHERE o.status = %s
  AND o.created_at >= %s
  AND o.created_at < %s
  AND r.created_at >= %s
  AND r.created_at < %s
"""

INJECTION_CONTROL_QUERY = """
SELECT COUNT(*) AS paid_count,
       COALESCE(SUM(o.amount_fen), 0) AS gross_fen
FROM orders AS o
WHERE o.tenant_id = %s
  AND o.status = %s
  AND o.created_at >= %s
  AND o.created_at < %s
"""


def fetch_commerce_summary(tenant_id: str, start: datetime, end: datetime) -> dict[str, Any]:
    """The W01 summary query, run the way the product runs it.

    Since W04 the tables force row-level security keyed on the transaction's
    tenant, so a read without the server-bound tenant context sees no rows
    (W01-FS01 reported ``A paid_count=0`` in the cloud).  The tenant is bound in
    the same read-only transaction; the SQL and every expected number are unchanged.
    """

    with connect_readonly() as conn:
        bind_transaction_tenant(conn, tenant_id)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(COMMERCE_SUMMARY_SQL, (tenant_id, start, end, tenant_id, start, end))
            row = cur.fetchone()
    require(row is not None, "commerce summary query returned no row")
    return row


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_legacy_summary(
    tenant_id: str,
    paid_count: int,
    gross_fen: int,
    refund_fen: int,
    net_fen: int,
) -> dict[str, Any]:
    actual = fetch_commerce_summary(tenant_id, START, END)
    require(int(actual["paid_count"]) == paid_count, f"{tenant_id} paid_count={actual['paid_count']}")
    require(int(actual["gross_fen"]) == gross_fen, f"{tenant_id} gross_fen={actual['gross_fen']}")
    require(int(actual["refund_fen"]) == refund_fen, f"{tenant_id} refund_fen={actual['refund_fen']}")
    require(int(actual["net_fen"]) == net_fen, f"{tenant_id} net_fen={actual['net_fen']}")
    return actual


def check_legacy_commerce() -> None:
    tenant_a = check_legacy_summary("A", 2, 15000, 3000, 12000)
    check_legacy_summary("B", 1, 990000, 10000, 980000)
    empty = fetch_commerce_summary(
        "A",
        datetime(2027, 1, 1, tzinfo=timezone.utc),
        datetime(2027, 2, 1, tzinfo=timezone.utc),
    )
    require(int(empty["paid_count"]) == 0, "empty paid_count is not zero")
    require(int(empty["gross_fen"]) == 0, "empty gross_fen is not zero")
    require(int(empty["refund_fen"]) == 0, "empty refund_fen is not zero")
    require(int(empty["net_fen"]) == 0, "empty net_fen is not zero")
    narrow = fetch_commerce_summary(
        "A",
        datetime(2026, 9, 9, tzinfo=timezone.utc),
        datetime(2026, 9, 11, tzinfo=timezone.utc),
    )
    require(int(narrow["paid_count"]) == 0, "narrow paid_count is not zero")
    require(int(narrow["gross_fen"]) == 0, "narrow gross_fen is not zero")
    require(int(narrow["refund_fen"]) == 0, "narrow refund_fen is not zero")
    require(int(narrow["net_fen"]) == 0, "narrow net_fen is not zero")
    require(fen_to_yuan(tenant_a["net_fen"]) == "120.00 元", "unexpected display")
    answer = FakeModel().generate("2026年9月退款后净额", tenant_a)["answer"]
    require("120.00 元" in answer, f"unexpected model answer: {answer}")
    injection = fetch_commerce_summary("A' OR '1'='1", START, END)
    require(int(injection["paid_count"]) == 0, "parameterized tenant check failed")
    require(int(injection["gross_fen"]) == 0, "parameterized SQL check failed")


def _iso_param(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _guarded_query(
    executor: GuardedQueryExecutor,
    *,
    tenant_id: str,
    run_id: str,
    query_id: str,
    sql: str,
    params: tuple[object, ...],
    observed: list[dict[str, object]],
) -> dict[str, Any]:
    context = ExecutionContext(
        run_id=run_id,
        tenant_id=tenant_id,
        principal_id="w05-db01-checker",
        role="requester",
    )
    result = executor.execute(sql, context=context, params=params)
    evidence = result.evidence
    require(evidence.run_id == run_id, f"{query_id} run ownership mismatch")
    require(evidence.tenant_id == tenant_id, f"{query_id} tenant ownership mismatch")
    require(
        evidence.principal_id == context.principal_id,
        f"{query_id} principal ownership mismatch",
    )
    require(evidence.row_count == len(evidence.rows) == 1, f"{query_id} expected one aggregate row")
    actual = dict(evidence.rows[0])
    observed.append(
        {
            "query_id": query_id,
            "sql": " ".join(sql.split()),
            "params": [_iso_param(value) for value in params],
            "result_id": evidence.result_id,
            "run_id": evidence.run_id,
            "tenant_id": evidence.tenant_id,
            "principal_id": evidence.principal_id,
            "row_count": evidence.row_count,
            "rows": [dict(row) for row in evidence.rows],
            "query_sha256": evidence.query_sha256,
            "params_sha256": evidence.params_sha256,
            "policy_version": evidence.policy_version,
            "catalog_version": evidence.catalog_version,
            "observed_at": evidence.observed_at.isoformat(),
        }
    )
    return actual


def _guarded_summary(
    executor: GuardedQueryExecutor,
    *,
    tenant_id: str,
    case_id: str,
    start: datetime,
    end: datetime,
    observed: list[dict[str, object]],
) -> dict[str, int]:
    run_id = f"w05-db01-{case_id}-{tenant_id.lower()}"
    paid = _guarded_query(
        executor,
        tenant_id=tenant_id,
        run_id=run_id,
        query_id=f"{case_id}-{tenant_id}-paid",
        sql=PAID_SUMMARY_QUERY,
        params=("paid", start, end),
        observed=observed,
    )
    refunds = _guarded_query(
        executor,
        tenant_id=tenant_id,
        run_id=run_id,
        query_id=f"{case_id}-{tenant_id}-refunds",
        sql=REFUND_SUMMARY_QUERY,
        params=("paid", start, end, start, end),
        observed=observed,
    )
    gross_fen = int(paid["gross_fen"])
    refund_fen = int(refunds["refund_fen"])
    return {
        "paid_count": int(paid["paid_count"]),
        "gross_fen": gross_fen,
        "refund_fen": refund_fen,
        "net_fen": gross_fen - refund_fen,
    }


def check_guarded_commerce(observed: list[dict[str, object]]) -> None:
    executor = GuardedQueryExecutor()
    tenant_a = _guarded_summary(
        executor,
        tenant_id="A",
        case_id="september",
        start=START,
        end=END,
        observed=observed,
    )
    tenant_b = _guarded_summary(
        executor,
        tenant_id="B",
        case_id="september",
        start=START,
        end=END,
        observed=observed,
    )

    require(
        tenant_a["paid_count"] == 2,
        f"A paid_count={tenant_a['paid_count']}",
    )
    require(
        tenant_a["gross_fen"] == 15000,
        f"A gross_fen={tenant_a['gross_fen']}",
    )
    require(
        tenant_a["refund_fen"] == 3000,
        f"A refund_fen={tenant_a['refund_fen']}",
    )
    require(
        tenant_a["net_fen"] == 12000,
        f"A net_fen={tenant_a['net_fen']}",
    )
    require(
        tenant_b == {"paid_count": 1, "gross_fen": 990000, "refund_fen": 10000, "net_fen": 980000},
        f"B commerce summary={tenant_b}",
    )


    empty_start = datetime(2027, 1, 1, tzinfo=timezone.utc)
    empty_end = datetime(2027, 2, 1, tzinfo=timezone.utc)
    empty = _guarded_summary(
        executor,
        tenant_id="A",
        case_id="empty-window",
        start=empty_start,
        end=empty_end,
        observed=observed,
    )
    require(empty == {"paid_count": 0, "gross_fen": 0, "refund_fen": 0, "net_fen": 0}, f"empty summary={empty}")

    narrow_start = datetime(2026, 9, 9, tzinfo=timezone.utc)
    narrow_end = datetime(2026, 9, 11, tzinfo=timezone.utc)
    narrow = _guarded_summary(
        executor,
        tenant_id="A",
        case_id="narrow-window",
        start=narrow_start,
        end=narrow_end,
        observed=observed,
    )
    require(narrow == {"paid_count": 0, "gross_fen": 0, "refund_fen": 0, "net_fen": 0}, f"narrow summary={narrow}")

    injection = _guarded_query(
        executor,
        tenant_id="A",
        run_id="w05-db01-injection-a",
        query_id="tenant-filter-injection-control",
        sql=INJECTION_CONTROL_QUERY,
        params=("A' OR '1'='1", "paid", START, END),
        observed=observed,
    )
    require(int(injection["paid_count"]) == 0, "parameterized tenant filter check failed")
    require(int(injection["gross_fen"]) == 0, "parameterized SQL check failed")

    display = fen_to_yuan(tenant_a["net_fen"])
    require(display == "120.00 元", f"unexpected display: {display}")

    answer = FakeModel().generate(
        "2026年9月退款后净额",
        tenant_a,
    )["answer"]
    require("120.00 元" in answer, f"unexpected model answer: {answer}")


def _write_evidence(path: Path, *, status: str, observed: list[dict[str, object]], error: Exception | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "check_id": "W05-DB01",
        "probe": "commerce_fixture",
        "mode": "real_postgres_guarded_query",
        "status": status,
        "query_count": len(observed),
        "queries": observed,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_summary"] = (
            str(error)
            if isinstance(error, AssertionError)
            else "operational detail retained in the parent sanitized probe record"
        )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rls-aware", action="store_true")
    parser.add_argument("--evidence-output", type=Path)
    args = parser.parse_args()
    if not args.rls_aware:
        check_legacy_commerce()
        print("commerce_check_pass")
        return 0
    observed: list[dict[str, object]] = []
    try:
        check_guarded_commerce(observed)
    except Exception as exc:
        if args.evidence_output is not None:
            _write_evidence(args.evidence_output, status="fail", observed=observed, error=exc)
        raise
    if args.evidence_output is not None:
        _write_evidence(args.evidence_output, status="pass", observed=observed, error=None)

    print("commerce_check_pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
