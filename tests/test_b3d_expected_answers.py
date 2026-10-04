"""B3d: the expected answers are computed twice (generator in memory, hand-written SQL on PostgreSQL) and agree.

These tests need the demo database (docs/b3d-demo-data.md); without it they skip, like the
other database tests.  They run against the real database, never a fake.
"""

from __future__ import annotations

import copy
import json
import os
from urllib.parse import urlsplit, urlunsplit

import pytest

from scripts import bootstrap_demo_db as bootstrap
from scripts import generate_demo_data as gen
from scripts import verify_demo_expected as verify

DEMO_NAME = "queryshield_demo"


def _with_database(url_env: str) -> str | None:
    url = os.environ.get(url_env)
    if not url:
        return None
    return urlunsplit(urlsplit(url)._replace(path=f"/{DEMO_NAME}"))


def _reachable(url: str | None) -> bool:
    if not url:
        return False
    import psycopg

    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001 - any failure means "not available here"
        return False


READONLY_URL = _with_database("QUERYSHIELD_DATABASE_URL")
ADMIN_URL = _with_database("QUERYSHIELD_BOOTSTRAP_DATABASE_URL")
needs_demo_database = pytest.mark.skipif(
    not _reachable(READONLY_URL), reason="needs the demo database (docs/b3d-demo-data.md, section 2)"
)
needs_demo_admin = pytest.mark.skipif(not _reachable(ADMIN_URL), reason="needs the demo database owner connection")


@pytest.fixture()
def conn():
    import psycopg

    with psycopg.connect(READONLY_URL, options="-c default_transaction_read_only=on") as connection:
        yield connection


@needs_demo_database
def test_verification_by_sql_agrees_with_the_committed_expected_answers(monkeypatch, capsys) -> None:
    monkeypatch.setenv("QUERYSHIELD_DEMO_DATABASE_URL", READONLY_URL)
    assert verify.main() == 0
    out = capsys.readouterr().out
    assert "mismatches=0" in out and "MISMATCH" not in out
    assert "Q03: ok" in out and "Q05: ok" in out and "Q07: ok" in out
    assert "Q06: ok" in out and "Q06b: ok" in out and "Q06:all_values: ok" in out


@needs_demo_database
def test_verification_notices_a_wrong_expected_value(conn) -> None:
    document = json.loads(gen.QUESTIONS_PATH.read_text(encoding="utf-8"))
    broken = copy.deepcopy(document)
    next(q for q in broken["questions"] if q["id"] == "Q03")["expected"]["value"] += 1
    verdicts = {qid: ok for qid, ok, _ in verify.verify(conn, broken)}
    assert verdicts["Q03"] is False and verdicts["Q02"] is True


@needs_demo_database
def test_verification_notices_a_wrong_top_five_and_a_wrong_full_summary(conn) -> None:
    document = json.loads(gen.QUESTIONS_PATH.read_text(encoding="utf-8"))
    broken = copy.deepcopy(document)
    q06 = next(q for q in broken["questions"] if q["id"] == "Q06")
    q06["expected"]["rows"][0]["value"] += 1
    q06b = next(q for q in broken["questions"] if q["id"] == "Q06b")
    q06b["expected"]["rows"][3]["value"] += 1
    verdicts = {qid: ok for qid, ok, _ in verify.verify(conn, broken)}
    assert verdicts["Q06"] is False and verdicts["Q06b"] is False
    swapped = copy.deepcopy(document)
    rows = next(q for q in swapped["questions"] if q["id"] == "Q06")["expected"]["rows"]
    rows[4] = {**rows[4], "customer_id": "c39", "name": rows[4]["name"]}  # a customer who never ordered
    assert {qid: ok for qid, ok, _ in verify.verify(conn, swapped)}["Q06"] is False


WINDOWS = [(5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (7, 9), (6, 9), (8, 9)]


@needs_demo_database
@pytest.mark.parametrize("tenant", ["A", "B"])
@pytest.mark.parametrize(("first", "last"), WINDOWS)
def test_both_algorithms_agree_for_every_metric_over_many_windows(conn, tenant, first, last) -> None:
    data = gen.generate()
    start, end = gen.month_start(first), gen.month_end(last)
    in_memory = gen.expected_metrics(data, tenant, start, end)
    on_database = verify.metrics(conn, tenant, {"start": gen.iso(start), "end": gen.iso(end)})
    assert on_database == in_memory


@needs_demo_database
def test_the_database_rows_match_the_generator_and_rls_isolates_the_tenants(conn) -> None:
    document = json.loads(gen.QUESTIONS_PATH.read_text(encoding="utf-8"))
    for tenant, counts in document["table_counts"].items():
        with conn.transaction():
            conn.execute("SELECT set_config('queryshield.tenant_id', %s, true)", (tenant,))
            for table, expected in counts.items():
                assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == expected
                assert conn.execute(f"SELECT count(*) FROM {table} WHERE tenant_id <> %s", (tenant,)).fetchone()[0] == 0
    with conn.transaction():  # no tenant bound: the forced RLS policy hides every row
        assert conn.execute("SELECT count(*) FROM orders").fetchone()[0] == 0


@needs_demo_database
def test_the_read_only_role_cannot_write_to_the_demo_database(conn) -> None:
    import psycopg

    with pytest.raises(psycopg.Error):
        with conn.transaction():
            conn.execute("DELETE FROM refunds")


@needs_demo_admin
def test_bootstrap_is_repeatable_and_loads_exactly_the_generated_rows(capsys) -> None:
    first = bootstrap.bootstrap(ADMIN_URL)
    second = bootstrap.bootstrap(ADMIN_URL)
    assert first == second == bootstrap.expected_counts()
    out = capsys.readouterr().out
    assert out.count("bootstrap_demo_db_ok database=queryshield_demo") == 2
    assert "postgresql" not in out and "password" not in out.lower()
