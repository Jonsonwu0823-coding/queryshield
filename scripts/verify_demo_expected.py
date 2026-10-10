"""Check the demo questions' expected answers against the real demo database.

This is the second, independent computation of the expected answers (the first
is the in-memory algorithm in scripts/generate_demo_data.py).  It imports nothing
from queryshield or from the generator: it reads the committed question lists (demo and composite)
(tenant, window, expected value), runs hand-written SQL on the database and
compares.  Exit code 0 when every value agrees, 1 on any difference, 2 when it
cannot run (no URL, wrong database name, database unreachable).

Environment: QUERYSHIELD_DEMO_DATABASE_URL (read-only role) or
QUERYSHIELD_DEMO_BOOTSTRAP_DATABASE_URL (owner).  The tenant is bound with
set_config inside each read-only transaction and also filtered explicitly.
Prints question ids and verdicts only; never the URL, never a customer name.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "demo" / "demo-questions-v1.json"
COMPOSITE_PATH = QUESTIONS_PATH.with_name("demo-composite-questions-v1.json")

# Hand-written; deliberately not the product's join-based plan.  A refund counts
# when its own time and its paid order's time are both inside [lo, hi).
METRICS_SQL = """
WITH paid AS (
    SELECT order_id, amount_fen
    FROM orders
    WHERE tenant_id = %(tenant)s AND status = 'paid'
      AND created_at >= %(lo)s::timestamptz AND created_at < %(hi)s::timestamptz
)
SELECT
    (SELECT count(*) FROM paid) AS paid_count,
    (SELECT coalesce(sum(amount_fen), 0) FROM paid) AS gross_fen,
    (SELECT coalesce(sum(r.amount_fen), 0) FROM refunds AS r
      WHERE r.tenant_id = %(tenant)s
        AND r.created_at >= %(lo)s::timestamptz AND r.created_at < %(hi)s::timestamptz
        AND EXISTS (SELECT 1 FROM paid AS p WHERE p.order_id = r.order_id)) AS refund_fen
"""
CUSTOMER_SQL = """
SELECT customer_id, sum(amount_fen) AS gross_fen
FROM orders
WHERE tenant_id = %(tenant)s AND status = 'paid'
  AND created_at >= %(lo)s::timestamptz AND created_at < %(hi)s::timestamptz
GROUP BY customer_id
"""
NAME_SQL = "SELECT name FROM customers WHERE tenant_id = %(tenant)s AND customer_id = %(customer_id)s"
COUNT_SQL = {
    "customers": "SELECT count(*) FROM customers WHERE tenant_id = %(tenant)s",
    "orders": "SELECT count(*) FROM orders WHERE tenant_id = %(tenant)s",
    "refunds": "SELECT count(*) FROM refunds WHERE tenant_id = %(tenant)s",
}


def database_name(url: str) -> str:
    segments = [segment for segment in urlsplit(url).path.split("/") if segment]
    return unquote(segments[-1]) if len(segments) == 1 else ""


def _run(conn, tenant: str, query: str, **params):
    """One read-only transaction with the tenant bound; returns all rows."""

    with conn.transaction():
        conn.execute("SELECT set_config('queryshield.tenant_id', %s, true)", (tenant,))
        return conn.execute(query, {"tenant": tenant, **params}).fetchall()


def metrics(conn, tenant: str, window: dict[str, str]) -> dict[str, int]:
    ((paid_count, gross, refund),) = _run(conn, tenant, METRICS_SQL, lo=window["start"], hi=window["end"])
    return {"paid_count": int(paid_count), "gross_fen": int(gross), "refund_fen": int(refund), "net_fen": int(gross) - int(refund)}


def customer_gross(conn, tenant: str, window: dict[str, str]) -> dict[str, int]:
    rows = _run(conn, tenant, CUSTOMER_SQL, lo=window["start"], hi=window["end"])
    return {customer_id: int(total) for customer_id, total in rows}


def verify(conn, document: dict) -> list[tuple[str, bool, str]]:
    verdicts: list[tuple[str, bool, str]] = []

    def record(qid: str, ok: bool, detail: str = "") -> None:
        verdicts.append((qid, ok, detail))

    for tenant, expected in sorted(document.get("table_counts", {}).items()):
        for table, count in expected.items():
            ((actual,),) = _run(conn, tenant, COUNT_SQL[table])
            record(f"rows:{tenant}:{table}", int(actual) == count, f"expected {count}, got {int(actual)}")

    for q in document["questions"]:
        qid, kind, tenant, window, expected = q["id"], q["kind"], q["tenant"], q.get("window"), q["expected"]
        if kind in {"composite", "composite_clarify"}:
            for fact in expected["facts"]:
                actual = metrics(conn, tenant, fact["window"])[fact["metric_id"]]
                record(f"{qid}:{fact['metric_id']}:{fact['window']['start'][:7]}", actual == fact["value"], f"expected {fact['value']}, got {actual}")
        elif kind in {"metric", "empty_window", "clarify_resume"}:
            actual = metrics(conn, tenant, window)[expected["metric_id"]]
            record(qid, actual == expected["value"], f"{expected['metric_id']}: expected {expected['value']}, got {actual}")
        elif kind == "observe_refund":
            actual = metrics(conn, tenant, window)["refund_fen"]
            record(qid, actual == expected["refund_fen"], f"refund_fen: expected {expected['refund_fen']}, got {actual}")
        elif kind in {"rowset", "observe_rowset"}:
            actual_rows = customer_gross(conn, tenant, window)
            wanted = {row["customer_id"]: row["value"] for row in expected["rows"]}
            if kind == "rowset":
                # The top-N customers by paid amount, highest first; the N-th and the next must not tie.
                ranked = sorted(actual_rows.items(), key=lambda item: (-item[1], item[0]))
                top = dict(ranked[: expected["top_n"]])
                cut_off_ok = len(ranked) > expected["top_n"] and ranked[expected["top_n"] - 1][1] > ranked[expected["top_n"]][1]
                record(qid, top == wanted and cut_off_ok, f"top {expected['top_n']} customers and amounts, no tie at the cut-off")
            else:
                record(qid, actual_rows == wanted, f"{len(wanted)} expected customers, {len(actual_rows)} found")
            record(f"{qid}:all_values", actual_rows == {k: int(v) for k, v in expected["all_values"].items()}, "per-customer amounts")
            names_ok = all(
                _run(conn, tenant, NAME_SQL, customer_id=row["customer_id"]) == [(row["name"],)] for row in expected["rows"]
            )
            record(f"{qid}:names", names_ok, "customer names match")
        elif kind == "top_customer":
            totals = customer_gross(conn, tenant, window)
            ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
            lead_ok = len(ranked) > 1 and ranked[0][1] * 100 >= ranked[1][1] * 105
            record(
                qid,
                ranked[0][0] == expected["customer_id"] and ranked[0][1] == expected["value"] and lead_ok,
                "top customer id, amount and a lead of at least 5% over the second",
            )
            record(f"{qid}:name", _run(conn, tenant, NAME_SQL, customer_id=expected["customer_id"]) == [(expected["name"],)], "name matches")
        elif kind == "isolation":
            # The forbidden values are tenant B's; A's own data must not contain them as answers.
            b = metrics(conn, "B", window)
            record(qid, sorted({b["net_fen"], b["gross_fen"]}) == sorted(expected["forbidden_values"]), "tenant B's net_fen and gross_fen")
    return verdicts


def main() -> int:
    url = os.getenv("QUERYSHIELD_DEMO_DATABASE_URL") or os.getenv("QUERYSHIELD_DEMO_BOOTSTRAP_DATABASE_URL") or ""
    if not url:
        print("verify_demo_expected blocked: no demo database URL is configured")
        return 2
    try:
        name = database_name(url)
    except ValueError:  # malformed URL; never echo it
        print("verify_demo_expected blocked: the database URL cannot be parsed")
        return 2
    if not name.endswith("_demo"):
        print("verify_demo_expected blocked: the database name must end with _demo")
        return 2
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in (QUESTIONS_PATH, COMPOSITE_PATH)]
    import psycopg

    try:
        with psycopg.connect(url, connect_timeout=5, options="-c default_transaction_read_only=on") as conn:
            verdicts = [verdict for document in documents for verdict in verify(conn, document)]
    except psycopg.Error as exc:
        print(f"verify_demo_expected blocked: {type(exc).__name__}")
        return 2
    failed = [item for item in verdicts if not item[1]]
    for qid, ok, detail in verdicts:
        print(f"{qid}: {'ok' if ok else 'MISMATCH'}" + ("" if ok else f" ({detail})"))
    print(f"checked={len(verdicts)} mismatches={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
