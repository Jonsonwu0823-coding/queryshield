"""Create the two QueryShield roles and the test and/or demo database, then load them.

Shared by Compose (the one-shot ``setup`` service) and CI.  It connects as a PostgreSQL
superuser, given by QUERYSHIELD_SUPERUSER_DATABASE_URL, and

  * creates the roles queryshield (owner: NOSUPERUSER NOCREATEDB NOCREATEROLE BYPASSRLS)
    and queryshield_ro (NOSUPERUSER NOBYPASSRLS NOINHERIT) if missing, and on EVERY run
    sets both passwords from QUERYSHIELD_ADMIN_PASSWORD and QUERYSHIELD_RO_PASSWORD, so
    the environment is the only source of truth;
  * --test: creates queryshield_test (owner queryshield) and runs scripts/bootstrap_db.py
    (the same migrations and seed), then grants the read-only role SELECT;
  * --demo: creates queryshield_demo and runs scripts/bootstrap_demo_db.py (tables, data,
    row counts, row level security, grants);
  * makes the read-only role's transactions read only by default.

It is idempotent.  Neither bootstrap script is changed or copied: they are called.  The
output names databases and counts only, never a URL, a password or a driver message.

Exit codes: 0 done, 1 failed, 2 refused (missing configuration, a guard failed).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from urllib.parse import quote, urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPERUSER_URL_ENV = "QUERYSHIELD_SUPERUSER_DATABASE_URL"
ADMIN_PASSWORD_ENV = "QUERYSHIELD_ADMIN_PASSWORD"
RO_PASSWORD_ENV = "QUERYSHIELD_RO_PASSWORD"
ADMIN_ROLE = "queryshield"
READONLY_ROLE = "queryshield_ro"
TEST_DATABASE = "queryshield_test"
DEMO_DATABASE = "queryshield_demo"
ALLOWED_DATABASES = (TEST_DATABASE, DEMO_DATABASE)
ROLE_OPTIONS = {
    ADMIN_ROLE: "LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION BYPASSRLS",
    READONLY_ROLE: "LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT",
}


class SetupRefused(RuntimeError):
    """A guard failed; the message never holds a URL or a credential."""


def check_database_name(name: str) -> str:
    """Only the two fixed names; the bootstrap scripts additionally require _test / _demo."""

    if name not in ALLOWED_DATABASES:
        raise SetupRefused(f"database {name!r} is not one of {', '.join(ALLOWED_DATABASES)}")
    return name


def database_url(superuser_url: str, *, user: str, password: str, database: str) -> str:
    """The superuser URL's host, port and query, with another user, password and database."""

    try:
        parts = urlsplit(superuser_url)
        host = parts.hostname or ""
        port = parts.port
    except ValueError as exc:  # never echo the URL
        raise SetupRefused("the superuser database URL cannot be parsed") from exc
    if parts.scheme not in {"postgresql", "postgres"} or not host:
        raise SetupRefused("the superuser database URL must be postgresql://user:password@host:port/database")
    netloc_host = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc_host = f"{netloc_host}:{port}"
    netloc = f"{quote(user, safe='')}:{quote(password, safe='')}@{netloc_host}"
    return urlunsplit((parts.scheme, netloc, "/" + quote(database, safe=""), parts.query, ""))


def _configuration() -> tuple[str, str, str]:
    values = [os.environ.get(name, "") for name in (SUPERUSER_URL_ENV, ADMIN_PASSWORD_ENV, RO_PASSWORD_ENV)]
    missing = [name for name, value in zip((SUPERUSER_URL_ENV, ADMIN_PASSWORD_ENV, RO_PASSWORD_ENV), values) if not value.strip()]
    if missing:
        raise SetupRefused("missing configuration: " + ", ".join(missing))
    if any("\x00" in value for value in values):
        raise SetupRefused("configuration values must not contain NUL")
    return values[0], values[1], values[2]


def keep_credentials_out_of_server_logs(conn) -> None:
    """A failed ALTER ROLE would otherwise copy its statement, password included, into the server log."""

    conn.execute("SET log_min_error_statement = panic")
    conn.execute("SET log_statement = 'none'")


def ensure_roles(conn, *, admin_password: str, readonly_password: str) -> None:
    """Create the roles when missing; always set their options and passwords."""

    from psycopg import sql

    for role, password in ((ADMIN_ROLE, admin_password), (READONLY_ROLE, readonly_password)):
        exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
        if exists is None:
            conn.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(role)))
        # ALTER ROLE ... PASSWORD cannot take a server-side parameter, so the literal is quoted here.
        conn.execute(
            sql.SQL("ALTER ROLE {} {} PASSWORD {}").format(
                sql.Identifier(role), sql.SQL(ROLE_OPTIONS[role]), sql.Literal(password)
            )
        )


def ensure_database(conn, name: str) -> None:
    """Create the database owned by the admin role, or make sure that role owns it."""

    from psycopg import sql

    check_database_name(name)
    exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
    if exists is None:
        conn.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(sql.Identifier(name), sql.Identifier(ADMIN_ROLE)))
    else:
        conn.execute(sql.SQL("ALTER DATABASE {} OWNER TO {}").format(sql.Identifier(name), sql.Identifier(ADMIN_ROLE)))


def grant_readonly(conn, database: str) -> None:
    """The test database's grants (bootstrap_demo_db.py does its own)."""

    from psycopg import sql

    conn.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(database), sql.Identifier(READONLY_ROLE)))
    conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(READONLY_ROLE)))
    conn.execute(
        sql.SQL("GRANT SELECT ON TABLE customers, orders, refunds TO {}").format(sql.Identifier(READONLY_ROLE))
    )


def make_readonly_default(conn) -> None:
    from psycopg import sql

    conn.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(sql.Identifier(READONLY_ROLE)))


def _bootstrap_test(superuser_url: str, admin_password: str) -> None:
    from scripts import bootstrap_db

    url = database_url(superuser_url, user=ADMIN_ROLE, password=admin_password, database=TEST_DATABASE)
    previous = os.environ.get("QUERYSHIELD_BOOTSTRAP_DATABASE_URL")
    os.environ["QUERYSHIELD_BOOTSTRAP_DATABASE_URL"] = url
    try:
        bootstrap_db.main()
    finally:
        if previous is None:
            os.environ.pop("QUERYSHIELD_BOOTSTRAP_DATABASE_URL", None)
        else:
            os.environ["QUERYSHIELD_BOOTSTRAP_DATABASE_URL"] = previous


def _bootstrap_demo(superuser_url: str, admin_password: str) -> None:
    from scripts import bootstrap_demo_db

    url = database_url(superuser_url, user=ADMIN_ROLE, password=admin_password, database=DEMO_DATABASE)
    try:
        bootstrap_demo_db.bootstrap(url)
    except bootstrap_demo_db.BootstrapRefused as exc:
        raise SetupRefused(str(exc)) from exc


def setup(*, test: bool, demo: bool) -> None:
    import psycopg

    superuser_url, admin_password, readonly_password = _configuration()
    databases = [name for name, wanted in ((TEST_DATABASE, test), (DEMO_DATABASE, demo)) if wanted]
    for name in databases:
        check_database_name(name)
    server_url = database_url(superuser_url, user=_superuser_name(superuser_url), password=_superuser_password(superuser_url), database="postgres")
    with psycopg.connect(server_url, autocommit=True, connect_timeout=10) as conn:
        keep_credentials_out_of_server_logs(conn)
        ensure_roles(conn, admin_password=admin_password, readonly_password=readonly_password)
        for name in databases:
            ensure_database(conn, name)
    if test:
        _bootstrap_test(superuser_url, admin_password)
        test_url = database_url(superuser_url, user=_superuser_name(superuser_url), password=_superuser_password(superuser_url), database=TEST_DATABASE)
        with psycopg.connect(test_url, autocommit=True, connect_timeout=10) as conn:
            grant_readonly(conn, TEST_DATABASE)
    if demo:
        _bootstrap_demo(superuser_url, admin_password)
    with psycopg.connect(server_url, autocommit=True, connect_timeout=10) as conn:
        make_readonly_default(conn)


def _split_superuser_url(url: str):
    try:
        return urlsplit(url)
    except ValueError as exc:  # never echo the URL
        raise SetupRefused("the superuser database URL cannot be parsed") from exc


def _superuser_name(url: str) -> str:
    from urllib.parse import unquote

    return unquote(_split_superuser_url(url).username or "")


def _superuser_password(url: str) -> str:
    from urllib.parse import unquote

    return unquote(_split_superuser_url(url).password or "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help=f"create and load {TEST_DATABASE}")
    parser.add_argument("--demo", action="store_true", help=f"create and load {DEMO_DATABASE}")
    args = parser.parse_args(argv)
    if not (args.test or args.demo):
        print("setup_databases_refused: give --test and/or --demo")
        return 2
    try:
        setup(test=args.test, demo=args.demo)
    except SetupRefused as exc:
        print(f"setup_databases_refused: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001 - driver messages can hold the URL; print the type only
        print(f"setup_databases_failed: {type(exc).__name__}")
        return 1
    names = ",".join(name for name, wanted in ((TEST_DATABASE, args.test), (DEMO_DATABASE, args.demo)) if wanted)
    print(f"setup_databases_ok roles={ADMIN_ROLE},{READONLY_ROLE} databases={names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
