"""Create (or reload) the QueryShield demo database tables and rows.

Only databases whose name ends with ``_demo`` are accepted; ``_test`` is refused.
The database itself must already exist (see docs/demo-data.md for who runs
CREATE DATABASE).  In ONE transaction this script

  1. applies migrations/001_commerce_v1.sql (idempotent),
  2. turns row level security off for the three tables (the owner loads data),
  3. TRUNCATEs refunds, orders and customers and loads fixtures/demo/commerce-demo-v1.sql,
  4. counts the rows and fails (rolling everything back) if they differ from the
     generator's counts in fixtures/demo/demo-questions-v1.json,
  5. applies migrations/002_rls.sql (ENABLE and FORCE RLS plus policies),
  6. grants CONNECT, USAGE and SELECT to the existing read-only role queryshield_ro.

It never creates a role and never sets a password.  The reload is exact, so a
rerun ends in the same state.  It prints the database name and row counts only:
never the URL or any credential.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATHS = (
    PROJECT_ROOT / "migrations" / "001_commerce_v1.sql",
    PROJECT_ROOT / "migrations" / "002_rls.sql",
)
DEMO_SQL_PATH = PROJECT_ROOT / "fixtures" / "demo" / "commerce-demo-v1.sql"
QUESTIONS_PATH = PROJECT_ROOT / "fixtures" / "demo" / "demo-questions-v1.json"
URL_ENV = "QUERYSHIELD_DEMO_BOOTSTRAP_DATABASE_URL"
READONLY_ROLE = "queryshield_ro"
TABLES = ("customers", "orders", "refunds")
RLS_RESET = "".join(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;\n" for table in TABLES)


class BootstrapRefused(RuntimeError):
    """A guard failed; the message never contains the URL or a credential."""


def database_name(url: str) -> str:
    """The database name from the URL path only (query string and %-encoding handled)."""

    try:
        path = urlsplit(url).path
    except ValueError as exc:  # malformed URL; never echo it
        raise BootstrapRefused("the database URL cannot be parsed") from exc
    segments = [segment for segment in path.split("/") if segment]
    return unquote(segments[-1]) if len(segments) == 1 else ""


def check_demo_database_name(name: str) -> str:
    if not name:
        raise BootstrapRefused("the database URL has no single database name")
    if name.endswith("_test"):
        raise BootstrapRefused("refusing a _test database: the demo data never goes into the test database")
    if not name.endswith("_demo"):
        raise BootstrapRefused("the demo database name must end with _demo")
    return name


def expected_counts() -> dict[str, dict[str, int]]:
    document = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return document["table_counts"]


def _counts(conn) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for table, key in (("customers", "customers"), ("orders", "orders"), ("refunds", "refunds")):
        for tenant, count in conn.execute(f"SELECT tenant_id, count(*) FROM {table} GROUP BY tenant_id").fetchall():
            result.setdefault(tenant, {})[key] = int(count)
    return result


def bootstrap(url: str) -> dict[str, dict[str, int]]:
    import psycopg
    from psycopg import sql

    name = check_demo_database_name(database_name(url))
    migrations = [path.read_text(encoding="utf-8") for path in MIGRATION_PATHS]
    demo_sql = DEMO_SQL_PATH.read_text(encoding="utf-8")
    if any(not text.strip() for text in migrations) or not demo_sql.strip():
        raise BootstrapRefused("a migration or the demo data file is empty")
    expected = expected_counts()

    with psycopg.connect(url, cursor_factory=psycopg.ClientCursor) as conn:
        # The URL is only a name: confirm which database this really is.
        if conn.execute("SELECT current_database()").fetchone()[0] != name:
            raise BootstrapRefused("the connected database is not the one named in the URL")
        conn.execute(migrations[0])
        conn.execute(RLS_RESET)
        conn.execute("TRUNCATE refunds, orders, customers")
        conn.execute(demo_sql)
        counts = _counts(conn)  # RLS is off here, so the counts see every row
        if counts != expected:
            raise BootstrapRefused("loaded row counts differ from the generator's counts; nothing was committed")
        conn.execute(migrations[1])
        if conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (READONLY_ROLE,)).fetchone() is None:
            raise BootstrapRefused(f"the read-only role {READONLY_ROLE} does not exist; create it first (README)")
        conn.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(name), sql.Identifier(READONLY_ROLE)))
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(READONLY_ROLE)))
        conn.execute(
            sql.SQL("GRANT SELECT ON TABLE customers, orders, refunds TO {}").format(sql.Identifier(READONLY_ROLE))
        )
    print(
        f"bootstrap_demo_db_ok database={name} fixture={DEMO_SQL_PATH.stem} "
        + " ".join(
            f"{tenant}=customers:{c['customers']}/orders:{c['orders']}/refunds:{c['refunds']}"
            for tenant, c in sorted(counts.items())
        )
        + " "
        + " ".join(f"{table}={sum(c[table] for c in counts.values())}" for table in TABLES)
    )
    return counts


def main() -> int:
    url = os.getenv(URL_ENV, "")
    if not url:
        print(f"{URL_ENV} is not configured")
        return 2
    try:
        bootstrap(url)
    except BootstrapRefused as exc:
        print(f"bootstrap_demo_db_refused: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001 - driver messages can hold the URL; print the type only
        print(f"bootstrap_demo_db_failed: {type(exc).__name__}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
