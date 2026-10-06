"""Scripts/setup_databases.py: names, quoting and secrecy are tested without a database;
the real setup (create, load, idempotent, password reset) runs when QUERYSHIELD_SETUP_TEST_SUPERUSER_URL
points at a throw-away PostgreSQL (the cloud's scratch cluster).  CI exercises the real script in its own
steps instead: it runs it twice against the job's PostgreSQL."""

from __future__ import annotations

import os
from urllib.parse import urlsplit, unquote

import pytest

from scripts import setup_databases as setup

# Opt-in, and deliberately NOT the variable setup_databases.py reads: the real test below changes the
# passwords of both roles, which would break every other test that uses the cluster's own roles.
# Point it at a throw-away PostgreSQL only; the test hands the value to the script itself.
THROWAWAY_URL_ENV = "QUERYSHIELD_SETUP_TEST_SUPERUSER_URL"
SUPERUSER_URL = os.environ.get(THROWAWAY_URL_ENV)


class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows


class _FakeConnection:
    """Records the SQL; role_exists / database_exists decide what the existence queries return."""

    def __init__(self, role_exists=True, database_exists=True):
        self.executed: list[str] = []
        self.role_exists = role_exists
        self.database_exists = database_exists

    def execute(self, query, params=None):
        text = query if isinstance(query, str) else query.as_string(None)
        self.executed.append(text if params is None else f"{text} {params!r}")
        if "FROM pg_roles" in text:
            return _Cursor((1,) if self.role_exists else None)
        if "FROM pg_database" in text:
            return _Cursor((1,) if self.database_exists else None)
        return _Cursor(None)


def test_only_the_two_fixed_database_names_are_accepted() -> None:
    assert setup.check_database_name("queryshield_test") == "queryshield_test"
    assert setup.check_database_name("queryshield_demo") == "queryshield_demo"
    for bad in ("postgres", "queryshield", "other_test", "queryshield_test; DROP DATABASE x", '"queryshield_demo"', ""):
        with pytest.raises(setup.SetupRefused):
            setup.check_database_name(bad)
        with pytest.raises(setup.SetupRefused):
            setup.ensure_database(_FakeConnection(), bad)


def test_roles_are_created_when_missing_and_always_get_their_options_and_passwords() -> None:
    conn = _FakeConnection(role_exists=False)
    setup.ensure_roles(conn, admin_password="adm'pw", readonly_password='ro"pw')
    text = "\n".join(conn.executed)
    assert text.count('CREATE ROLE "queryshield"') == 1 and text.count('CREATE ROLE "queryshield_ro"') == 1
    assert 'ALTER ROLE "queryshield" LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION BYPASSRLS PASSWORD' in text
    assert 'ALTER ROLE "queryshield_ro" LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT PASSWORD' in text
    assert "'adm''pw'" in text, "the password literal is quoted"
    existing = _FakeConnection(role_exists=True)
    setup.ensure_roles(existing, admin_password="a", readonly_password="b")
    joined = "\n".join(existing.executed)
    assert "CREATE ROLE" not in joined and joined.count("ALTER ROLE") == 2, "an existing role still gets its password reset"


def test_database_is_created_or_its_owner_is_fixed() -> None:
    created = _FakeConnection(database_exists=False)
    setup.ensure_database(created, "queryshield_demo")
    assert 'CREATE DATABASE "queryshield_demo" OWNER "queryshield"' in "\n".join(created.executed)
    existing = _FakeConnection(database_exists=True)
    setup.ensure_database(existing, "queryshield_test")
    assert 'ALTER DATABASE "queryshield_test" OWNER TO "queryshield"' in "\n".join(existing.executed)


def test_server_logs_are_told_to_keep_statements_out() -> None:
    conn = _FakeConnection()
    setup.keep_credentials_out_of_server_logs(conn)
    assert conn.executed == ["SET log_min_error_statement = panic", "SET log_statement = 'none'"]


def test_database_url_swaps_user_password_and_database_and_quotes_them() -> None:
    url = setup.database_url("postgresql://postgres:old@db:5432/postgres?sslmode=disable", user="queryshield", password="p@ss/w:rd", database="queryshield_demo")
    parts = urlsplit(url)
    assert (parts.hostname, parts.port, parts.path, parts.query) == ("db", 5432, "/queryshield_demo", "sslmode=disable")
    assert parts.username == "queryshield" and unquote(parts.password) == "p@ss/w:rd"
    assert "old" not in url
    assert urlsplit(setup.database_url("postgresql://x:y@[::1]:5433/postgres", user="u", password="p", database="d")).hostname == "::1"
    for bad in ("not a url", "mysql://u:p@h/db", "postgresql:///db"):
        with pytest.raises(setup.SetupRefused):
            setup.database_url(bad, user="u", password="p", database="d")


def test_missing_configuration_is_refused_by_name_and_prints_no_value(monkeypatch, capsys) -> None:
    for name in (setup.SUPERUSER_URL_ENV, setup.ADMIN_PASSWORD_ENV, setup.RO_PASSWORD_ENV):
        monkeypatch.delenv(name, raising=False)
    assert setup.main(["--demo"]) == 2
    out = capsys.readouterr().out
    assert all(name in out for name in (setup.SUPERUSER_URL_ENV, setup.ADMIN_PASSWORD_ENV, setup.RO_PASSWORD_ENV))
    assert setup.main([]) == 2


def test_a_failing_connection_prints_the_exception_type_only(monkeypatch, capsys) -> None:
    monkeypatch.setenv(setup.SUPERUSER_URL_ENV, "postgresql://postgres:very-secret-superuser@127.0.0.1:1/postgres")
    monkeypatch.setenv(setup.ADMIN_PASSWORD_ENV, "very-secret-admin")
    monkeypatch.setenv(setup.RO_PASSWORD_ENV, "very-secret-readonly")
    assert setup.main(["--test", "--demo"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("setup_databases_failed: ")
    assert "very-secret" not in out and "postgresql://" not in out and "127.0.0.1" not in out


# --- real database (superuser connection) ------------------------------------------------------

needs_superuser = pytest.mark.skipif(not SUPERUSER_URL, reason=f"needs {THROWAWAY_URL_ENV} (a throw-away PostgreSQL)")


def _connect(user: str, password: str, database: str):
    import psycopg

    return psycopg.connect(setup.database_url(SUPERUSER_URL, user=user, password=password, database=database), connect_timeout=5)


@needs_superuser
def test_real_setup_is_idempotent_and_the_environment_owns_the_passwords(monkeypatch, capsys) -> None:
    import psycopg

    monkeypatch.setenv(setup.SUPERUSER_URL_ENV, SUPERUSER_URL)
    monkeypatch.setenv(setup.ADMIN_PASSWORD_ENV, "admin-pw-one")
    monkeypatch.setenv(setup.RO_PASSWORD_ENV, "ro-pw-one")
    assert setup.main(["--test", "--demo"]) == 0
    assert setup.main(["--test", "--demo"]) == 0, "a second run succeeds"
    out = capsys.readouterr().out
    assert "setup_databases_ok" in out and "admin-pw" not in out and "ro-pw" not in out

    with _connect("queryshield_ro", "ro-pw-one", "queryshield_demo") as conn:
        assert conn.execute("SHOW default_transaction_read_only").fetchone() == ("on",)
        conn.execute("SELECT set_config('queryshield.tenant_id', 'A', false)")
        assert conn.execute("SELECT count(*) FROM orders").fetchone()[0] == 600
    with _connect("queryshield_ro", "ro-pw-one", "queryshield_test") as conn:
        conn.execute("SELECT set_config('queryshield.tenant_id', 'A', false)")
        assert conn.execute("SELECT count(*) FROM orders").fetchone()[0] > 0
    with _connect("queryshield", "admin-pw-one", "queryshield_demo") as conn:
        owner = conn.execute("SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname = current_database()").fetchone()[0]
        assert owner == "queryshield"
        row = conn.execute("SELECT rolsuper, rolcreatedb, rolcreaterole, rolbypassrls FROM pg_roles WHERE rolname = 'queryshield'").fetchone()
        assert row == (False, False, False, True)
        ro_row = conn.execute("SELECT rolsuper, rolbypassrls, rolinherit FROM pg_roles WHERE rolname = 'queryshield_ro'").fetchone()
        assert ro_row == (False, False, False)

    # Changing the environment and running again changes the passwords: the old ones stop working.
    monkeypatch.setenv(setup.RO_PASSWORD_ENV, "ro-pw-two")
    assert setup.main(["--demo"]) == 0
    with pytest.raises(psycopg.OperationalError):
        _connect("queryshield_ro", "ro-pw-one", "queryshield_demo")
    with _connect("queryshield_ro", "ro-pw-two", "queryshield_demo"):
        pass
