from pathlib import Path
import os
from urllib.parse import urlsplit

import psycopg

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MIGRATION_PATHS = (
    PROJECT_ROOT / "migrations" / "001_commerce_v1.sql",
    PROJECT_ROOT / "migrations" / "002_rls.sql",
)
FIXTURE_PATH = PROJECT_ROOT / "fixtures" / "commerce-v1.sql"


def get_bootstrap_url() -> str:
    url = os.getenv("QUERYSHIELD_BOOTSTRAP_DATABASE_URL")

    if not url:
        raise RuntimeError(
            "QUERYSHIELD_BOOTSTRAP_DATABASE_URL is not configured"
        )

    return url


def main() -> None:
    url = get_bootstrap_url()

    try:
        database_name = urlsplit(url).path.rsplit("/", 1)[-1]
    except ValueError as exc:  # malformed URL; never echo it
        raise RuntimeError("bootstrap database URL cannot be parsed") from exc

    if not database_name.endswith("_test"):
        raise RuntimeError("bootstrap database must end with _test")

    migration_sqls = tuple(path.read_text(encoding="utf-8") for path in MIGRATION_PATHS)
    fixture_sql = FIXTURE_PATH.read_text(encoding="utf-8")

    if any(not migration_sql.strip() for migration_sql in migration_sqls):
        raise RuntimeError("migration file is empty")

    if not fixture_sql.strip():
        raise RuntimeError("fixture file is empty")

    bootstrap_rls_reset = """
        ALTER TABLE customers DISABLE ROW LEVEL SECURITY;
        ALTER TABLE orders DISABLE ROW LEVEL SECURITY;
        ALTER TABLE refunds DISABLE ROW LEVEL SECURITY;
    """

    with psycopg.connect(
        url,
        cursor_factory=psycopg.ClientCursor,
    ) as conn:
        conn.execute(migration_sqls[0])
        # The fixture is loaded before FORCE RLS is enabled.  Disabling RLS
        # here also keeps the test bootstrap rerunnable against an existing
        # database that was bootstrapped by an earlier attempt.
        conn.execute(bootstrap_rls_reset)
        conn.execute(fixture_sql)
        for migration_sql in migration_sqls[1:]:
            conn.execute(migration_sql)

    print(
        "bootstrap_db_ok "
        f"database={database_name} "
        f"fixture={FIXTURE_PATH.stem}"
    )


if __name__ == "__main__":
    main()
