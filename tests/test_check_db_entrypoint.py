from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _clean_database_env() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("QUERYSHIELD_DATABASE_URL", None)
    source_root = str(PROJECT_ROOT / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else os.pathsep.join((source_root, existing_pythonpath))
    )
    return environment


def test_check_db_entrypoint_reaches_missing_config_in_new_process() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/check_db.py"],
        cwd=PROJECT_ROOT,
        env=_clean_database_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "db_check_blocked=RuntimeError" in completed.stdout
    assert "circular import" not in completed.stderr.lower()
    assert "partially initialized" not in completed.stderr.lower()


def test_check_db_program_error_remains_fail(monkeypatch, capsys) -> None:
    from scripts import check_db

    def raise_program_error():
        raise ValueError("synthetic program error")

    monkeypatch.setattr(check_db, "connect_readonly", raise_program_error)

    assert check_db.main() == 1
    captured = capsys.readouterr()
    assert "db_check_failed=ValueError" in captured.out
