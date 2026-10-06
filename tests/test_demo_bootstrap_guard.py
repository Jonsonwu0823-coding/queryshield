"""The demo database bootstrap only runs against a *_demo database and never prints a URL or credential."""

from __future__ import annotations

import pytest

from scripts import bootstrap_demo_db as bootstrap

SECRET = "pa55w0rd-should-never-print"


def _url(name: str, suffix: str = "") -> str:
    return f"postgresql://queryshield:{SECRET}@127.0.0.1:5433/{name}{suffix}"


@pytest.mark.parametrize(
    ("url", "name"),
    [
        (_url("queryshield_demo"), "queryshield_demo"),
        (_url("queryshield%5Fdemo"), "queryshield_demo"),
        (_url("queryshield_demo", "?sslmode=disable&connect_timeout=5"), "queryshield_demo"),
        (f"postgresql://u:{SECRET}@h", ""),
        (f"postgresql://u:{SECRET}@h/a/b", ""),
    ],
)
def test_database_name_comes_from_the_path_only(url, name) -> None:
    assert bootstrap.database_name(url) == name


def test_demo_names_are_accepted() -> None:
    assert bootstrap.check_demo_database_name("queryshield_demo") == "queryshield_demo"
    assert bootstrap.check_demo_database_name("another_demo") == "another_demo"


@pytest.mark.parametrize("name", ["queryshield_test", "x_demo_test", "queryshield", "demo", "queryshield_demo2", ""])
def test_everything_else_is_refused_and_a_test_database_has_its_own_message(name) -> None:
    with pytest.raises(bootstrap.BootstrapRefused) as caught:
        bootstrap.check_demo_database_name(name)
    if name.endswith("_test"):
        assert "_test" in str(caught.value)


def test_a_test_database_url_is_refused_before_any_connection(monkeypatch, capsys) -> None:
    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("connected to a refused database")

    import psycopg

    monkeypatch.setattr(psycopg, "connect", explode)
    monkeypatch.setenv(bootstrap.URL_ENV, _url("queryshield_test"))
    assert bootstrap.main() == 2
    out = capsys.readouterr().out
    assert "bootstrap_demo_db_refused" in out and SECRET not in out and "postgresql" not in out and "127.0.0.1" not in out


@pytest.mark.parametrize("name", ["queryshield", "queryshield_prod", "postgres"])
def test_a_non_demo_database_url_is_refused(monkeypatch, capsys, name) -> None:
    monkeypatch.setenv(bootstrap.URL_ENV, _url(name))
    assert bootstrap.main() == 2
    out = capsys.readouterr().out
    assert SECRET not in out and "postgresql" not in out


def test_without_a_url_nothing_runs(monkeypatch, capsys) -> None:
    monkeypatch.delenv(bootstrap.URL_ENV, raising=False)
    assert bootstrap.main() == 2
    assert bootstrap.URL_ENV in capsys.readouterr().out


def test_a_connection_failure_prints_the_error_type_only(monkeypatch, capsys) -> None:
    monkeypatch.setenv(bootstrap.URL_ENV, _url("queryshield_demo").replace("5433", "1"))  # nothing listens on port 1
    assert bootstrap.main() == 1
    out = capsys.readouterr().out
    assert out.startswith("bootstrap_demo_db_failed: ")
    assert SECRET not in out and "127.0.0.1" not in out and "postgresql" not in out


def test_the_script_never_creates_a_role_or_sets_a_password() -> None:
    source = open(bootstrap.__file__, encoding="utf-8").read().upper()
    for word in ("CREATE ROLE", "CREATE USER", "ALTER ROLE", "PASSWORD", "CREATE DATABASE", "DROP DATABASE"):
        assert word not in source.replace('"""', "").split("DEF BOOTSTRAP")[1], word


def test_the_load_is_one_transaction_with_a_count_check_before_the_policies() -> None:
    source = open(bootstrap.__file__, encoding="utf-8").read()
    body = source.split("def bootstrap(")[1]
    assert body.count("psycopg.connect(") == 1
    order = [body.index(token) for token in ("TRUNCATE", "conn.execute(demo_sql)", "counts != expected", "migrations[1]", "GRANT CONNECT")]
    assert order == sorted(order)


# --- the PowerShell wrappers (static properties; they cannot be run in the cloud) -----------------------------


def _ps1(name: str) -> str:
    from pathlib import Path

    raw = (Path(bootstrap.__file__).parent / name).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM, like the other wrappers (Windows PowerShell 5.1)
    return raw.decode("utf-8-sig")


def test_the_demo_wrappers_only_touch_the_demo_database_and_restore_the_environment() -> None:
    for name in ("bootstrap-demo-db.ps1", "demo-local.ps1"):
        text = _ps1(name)
        assert "queryshield_test" not in text, name
        assert "Read-Host" in text and "-AsSecureString" in text  # hidden input
        assert "finally" in text and "SetEnvironmentVariable($name, $previousValues[$name]" in text
        assert "Write-Output $env:QUERYSHIELD" not in text and "Write-Host $env:QUERYSHIELD" not in text
    boot = _ps1("bootstrap-demo-db.ps1")
    assert "EndsWith('_demo')" in boot and "bootstrap_demo_db.py" in boot
    demo = _ps1("demo-local.ps1")
    assert "queryshield_demo" in demo and "demo_run.py" in demo
    # The demo setting is given to the server by demo_run.py, never exported from the user's session.
    assert "QUERYSHIELD_DEMO_DATASET = $null" in demo
    assert "env:QUERYSHIELD_DEMO_DATASET =" not in demo.replace("env:QUERYSHIELD_DEMO_DATASET = $null", "")
    assert "demo-$mode-$stamp" in demo
