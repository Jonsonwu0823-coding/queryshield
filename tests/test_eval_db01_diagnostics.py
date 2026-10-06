from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import check_eval


@pytest.mark.parametrize(
    ("child_results", "failed_probe", "expected_assertion", "expected_failure_class", "expected_calls"),
    [
        (
            [
                SimpleNamespace(
                    returncode=1,
                    stdout=json.dumps(
                        {
                            "check_id": "STATE-DB01",
                            "status": "fail",
                            "mode": "fake",
                            "reason": "tenant A fixture counts=(0,0,0)",
                        }
                    ),
                    stderr="",
                )
            ],
            "readonly_role",
            "AssertionError: tenant A fixture counts=(0,0,0)",
            "database_identity_access_or_fixture_assertion",
            1,
        ),
        (
            [
                SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "check_id": "STATE-DB01",
                            "status": "pass",
                            "mode": "fake",
                            "details": {
                                "database": "queryshield_test",
                                "user": "queryshield_ro",
                                "tenant_a_counts": {"customers": 2, "orders": 3, "refunds": 2},
                            },
                        }
                    ),
                    stderr="",
                ),
                SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr=(
                        "Traceback (most recent call last):\n"
                        "AssertionError: expected net_fen=12000, got 11999; "
                        "password=probe-secret "
                        "postgresql://queryshield_ro:dsn-secret@127.0.0.1:5433/queryshield_test\n"
                    ),
                ),
            ],
            "commerce_fixture",
            "AssertionError: expected net_fen=12000, got 11999; password=<redacted> <redacted-endpoint>",
            "business_fixture_assertion",
            2,
        ),
    ],
)
def test_eval_db01_records_failing_child_and_sanitized_assertion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    child_results: list[SimpleNamespace],
    failed_probe: str,
    expected_assertion: str,
    expected_failure_class: str,
    expected_calls: int,
) -> None:
    monkeypatch.setenv(
        "QUERYSHIELD_DATABASE_URL",
        "postgresql://queryshield_ro:environment-secret@127.0.0.1:5433/queryshield_test",
    )
    calls: list[str] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        script_name = Path(command[1]).name
        calls.append(script_name)
        return child_results[len(calls) - 1]

    monkeypatch.setattr(check_eval.subprocess, "run", fake_run)

    result = check_eval._check_db01(tmp_path)

    assert result["status"] == "fail"
    assert result["failed_probe"] == failed_probe
    assert result["failed_assertion"] == expected_assertion
    assert result["failure_class"] == expected_failure_class
    assert len(calls) == expected_calls
    assert calls[-1] == ("check_state.py" if failed_probe == "readonly_role" else "check_commerce.py")

    details = json.loads((tmp_path / "database-probes.json").read_text(encoding="utf-8"))
    assert details["failed_probe"] == failed_probe
    assert details["probes"][failed_probe]["exit_code"] == 1
    assert details["probes"][failed_probe]["assertion"] == expected_assertion
    assert details["probes"][failed_probe]["failure_class"] == expected_failure_class
    serialized = json.dumps(details, ensure_ascii=False)
    assert "environment-secret" not in serialized
    assert "probe-secret" not in serialized
    assert "dsn-secret" not in serialized


def test_eval_db01_keeps_connection_failure_blocked_and_classified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "QUERYSHIELD_DATABASE_URL",
        "postgresql://queryshield_ro:environment-secret@127.0.0.1:5433/queryshield_test",
    )
    monkeypatch.setattr(
        check_eval.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2,
            stdout=json.dumps(
                {
                    "check_id": "STATE-DB01",
                    "status": "blocked",
                    "mode": "fake",
                    "reason": "real_postgresql_unavailable=OperationalError",
                }
            ),
            stderr="",
        ),
    )

    result = check_eval._check_db01(tmp_path)

    assert result["status"] == "blocked"
    assert result["failed_probe"] == "readonly_role"
    assert result["probe_exit_code"] == 2
    assert result["failure_class"] == "connection_authentication_or_database_runtime"
    assert result["failure_summary"] == [
        "db_check_blocked=real_postgresql_unavailable=OperationalError"
    ]


def test_eval_db01_uses_rls_aware_probe_and_records_tenant_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "QUERYSHIELD_DATABASE_URL",
        "postgresql://queryshield_ro:environment-secret@127.0.0.1:5433/queryshield_test",
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        commands.append(command)
        if Path(command[1]).name == "check_state.py":
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "check_id": "STATE-DB01",
                        "status": "pass",
                        "mode": "fake",
                        "details": {
                            "database": "queryshield_test",
                            "user": "queryshield_ro",
                            "rls_tables": 3,
                            "tenant_a_counts": {"customers": 2, "orders": 3, "refunds": 2},
                            "tenant_b_counts": {"customers": 1, "orders": 1, "refunds": 1},
                            "missing_context_counts": {"customers": 0, "orders": 0, "refunds": 0},
                        },
                    }
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="commerce_check_pass\n", stderr="")

    monkeypatch.setattr(check_eval.subprocess, "run", fake_run)

    result = check_eval._check_db01(tmp_path)

    assert result["status"] == "pass"
    assert [Path(command[1]).name for command in commands] == [
        "check_state.py",
        "check_commerce.py",
    ]
    commerce_command = commands[1]
    assert "--rls-aware" in commerce_command
    assert "--evidence-output" in commerce_command
    assert Path(commerce_command[commerce_command.index("--evidence-output") + 1]) == (
        tmp_path / "commerce-fixture.json"
    )
    assert commands[0][2:6] == ["--check-id", "STATE-DB01", "--mode", "fake"]
    assert "--output-dir" in commands[0]
    details = json.loads((tmp_path / "database-probes.json").read_text(encoding="utf-8"))
    observed = details["probes"]["readonly_role"]["database_result"]
    assert observed["tenant_a_counts"] == {"customers": 2, "orders": 3, "refunds": 2}
    assert observed["tenant_b_counts"] == {"customers": 1, "orders": 1, "refunds": 1}
    assert observed["missing_context_counts"] == {"customers": 0, "orders": 0, "refunds": 0}
    assert details["failed_probe"] is None
