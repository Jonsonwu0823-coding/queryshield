from __future__ import annotations

import json
import os
import subprocess

import pytest

from repo_layout import needs_register
from queryshield.approval.service import reset_shared_state_stores
from scripts.check_w04 import _assert_parallel_metric_oracle, check_en05, check_x01


def test_rt01_fixed_metric_oracle_rejects_wrong_value() -> None:
    expected = {"paid_count": 2, "gross_fen": 15000, "net_fen": 12000}
    actual = {"paid_count": 2, "gross_fen": 15000, "net_fen": 999}

    with pytest.raises(AssertionError, match="net_fen expected=12000 actual=999"):
        _assert_parallel_metric_oracle(actual, expected, label="RT01 negative control")


@needs_register
def test_x01_ignores_inherited_git_dir_and_still_checks_project_tag(monkeypatch, tmp_path) -> None:
    unrelated_repo = tmp_path / "unrelated.git"
    empty_git_config = tmp_path / "empty.gitconfig"
    empty_git_config.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_git_config))
    for name in list(os.environ):
        upper = name.upper()
        if upper in {"GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS"} or (
            (upper.startswith("GIT_CONFIG_KEY_") or upper.startswith("GIT_CONFIG_VALUE_"))
            and upper.rsplit("_", 1)[-1].isdigit()
        ):
            monkeypatch.delenv(name, raising=False)
    git_env = os.environ.copy()
    for name in (
        "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE", "GIT_CEILING_DIRECTORIES",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
    ):
        git_env.pop(name, None)
    subprocess.run(
        ["git", "init", "--bare", "--quiet", str(unrelated_repo)],
        env=git_env,
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.setenv("GIT_DIR", str(unrelated_repo))

    details = check_x01(tmp_path)

    assert details["upstream_tag"] == "qs-w03-accepted-20260922"
    assert "GIT_DIR" in details["repository_overrides_cleared"]
    assert details["git_safe_directory_scoped_to_project_repo"] is True


def test_en05_cross_feature_scenarios_and_fs02_mapping_fixture(tmp_path) -> None:
    # This is only a checker unit-test dependency fixture.  It is stored under
    # pytest's disposable tmp_path and is never acceptance/real-restart evidence.
    (tmp_path / "W04-FS02-details.json").write_text(
        json.dumps({
            "database_mode": "real_postgresql",
            "committed_result_replay": "same result/facts/answer and persisted SQL count across new process; no re-dispatch",
            "processes": [{"pid": 101}, {"pid": 102}, {"pid": 103}],
        }),
        encoding="utf-8",
    )
    try:
        details = check_en05(tmp_path)
        assert details["normal_baseline"]["status"] == "SUCCEEDED"
        assert details["approval_revocation"]["error_code"] == "authorization_revoked"
        assert details["approval_revocation"]["sql_exec_count"] == 0
        assert details["tool_text_injection"]["preference_absent"] is True
        replayed = details["cancel_sse_reconnect"]["terminal_replayed"]
        assert replayed["type"] == "terminal"
        assert replayed["status"] == "CANCELLED"
        assert replayed["event_id"] > details["cancel_sse_reconnect"]["last_event_id"]
        assert details["cancel_sse_reconnect"]["sql_exec_count_before_after"] == [0, 0]
        assert details["real_restart_committed_result"]["source_check_id"] == "W04-FS02"
        assert set(details["scenario_mapping"]) == {
            "revocation", "tool_text_injection", "cancel_reconnect", "normal_baseline", "committed_restart"
        }
    finally:
        reset_shared_state_stores()
