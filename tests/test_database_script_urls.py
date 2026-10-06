"""The operations scripts refuse a database URL that cannot be parsed, without printing it or connecting."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import setup_databases

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRET = "secret-password-for-test"
UNPARSABLE = f"postgresql://user:{SECRET}@[::1/queryshield_test"


def _run(script: str, *args: str, **env: str) -> subprocess.CompletedProcess:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("QUERYSHIELD_")}
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), *args],
        env={**clean, "PYTHONPATH": str(PROJECT_ROOT / "src"), **env},
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _assert_quiet(done: subprocess.CompletedProcess) -> None:
    assert SECRET not in done.stdout + done.stderr


def test_verify_demo_expected_refuses_an_unparsable_url_with_exit_2() -> None:
    done = _run("verify_demo_expected.py", QUERYSHIELD_DEMO_DATABASE_URL=UNPARSABLE.replace("_test", "_demo"))
    assert done.returncode == 2, done.stdout + done.stderr
    assert done.stdout.strip() == "verify_demo_expected blocked: the database URL cannot be parsed"
    _assert_quiet(done)


def test_bootstrap_db_refuses_an_unparsable_url_by_its_own_runtime_error() -> None:
    done = _run("bootstrap_db.py", QUERYSHIELD_BOOTSTRAP_DATABASE_URL=UNPARSABLE)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "RuntimeError: bootstrap database URL cannot be parsed" in done.stderr
    _assert_quiet(done)


def test_setup_databases_refuses_an_unparsable_superuser_url_with_exit_2() -> None:
    done = _run(
        "setup_databases.py",
        "--test",
        QUERYSHIELD_SUPERUSER_DATABASE_URL=UNPARSABLE.replace("_test", "postgres"),
        QUERYSHIELD_ADMIN_PASSWORD="admin-password-for-test",
        QUERYSHIELD_RO_PASSWORD="readonly-password-for-test",
    )
    assert done.returncode == 2, done.stdout + done.stderr
    assert done.stdout.strip() == "setup_databases_refused: the superuser database URL cannot be parsed"
    _assert_quiet(done)


def test_bootstrap_demo_db_still_refuses_an_unparsable_url_as_before() -> None:
    done = _run("bootstrap_demo_db.py", QUERYSHIELD_DEMO_BOOTSTRAP_DATABASE_URL=UNPARSABLE.replace("_test", "_demo"))
    assert done.returncode == 2, done.stdout + done.stderr
    assert done.stdout.strip() == "bootstrap_demo_db_refused: the database URL cannot be parsed"
    _assert_quiet(done)


@pytest.mark.parametrize("read", [setup_databases._superuser_name, setup_databases._superuser_password])
def test_the_superuser_name_and_password_readers_refuse_an_unparsable_url(read) -> None:
    with pytest.raises(setup_databases.SetupRefused) as caught:
        read(UNPARSABLE)
    assert str(caught.value) == "the superuser database URL cannot be parsed"
    assert SECRET not in str(caught.value)
