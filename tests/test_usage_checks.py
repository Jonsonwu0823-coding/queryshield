"""The agent usage checks verify invariants, and load the same way from every entry point."""

from __future__ import annotations

from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from scripts import check_agent_usage
from scripts.check_agent_usage import assert_known_usage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _summary(**overrides: object) -> dict[str, object]:
    summary: dict[str, object] = {
        "status": "known",
        "model_call_count": 3,
        "known_call_count": 3,
        "unknown_call_count": 0,
        "model_call_ids": ["local-a", "local-b", "local-c"],
        "known_prompt_tokens": 33,
        "known_completion_tokens": 12,
        "known_total_tokens": 45,
        "prompt_tokens": 33,
        "completion_tokens": 12,
        "total_tokens": 45,
    }
    summary.update(overrides)
    return summary


def _check(summary: dict[str, object]) -> dict[str, object]:
    return assert_known_usage(summary, calls=3, prompt_tokens=33, completion_tokens=12, total_tokens=45)


def test_current_summary_shape_passes_and_extra_fields_are_reported_not_fatal() -> None:
    assert _check(_summary())["ignored_extra_fields"] == []
    record = _check(_summary(future_field="anything"))
    assert record["ignored_extra_fields"] == ["future_field"]


@pytest.mark.parametrize(
    "override",
    [
        {"model_call_count": 2},
        {"known_call_count": 2, "unknown_call_count": 1},
        {"unknown_call_count": 1},
        {"status": "unknown"},
        {"prompt_tokens": 34},
        {"completion_tokens": 11},
        {"total_tokens": 46},
        {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        {"model_call_ids": ["local-a", "local-b"]},
        {"model_call_ids": ["local-a", "local-a", "local-c"]},
        {"model_call_ids": ["local-a", "", "local-c"]},
        {"model_call_ids": "local-a,local-b,local-c"},
        {"known_total_tokens": 44},
        {"known_prompt_tokens": None},
    ],
)
def test_wrong_count_total_or_call_id_fails(override: dict[str, object]) -> None:
    with pytest.raises(AssertionError, match="unexpected usage summary"):
        _check(_summary(**override))


def test_numbers_are_not_relaxed_for_the_expected_side() -> None:
    with pytest.raises(AssertionError):
        assert_known_usage(_summary(), calls=3, prompt_tokens=33, completion_tokens=12, total_tokens=44)
    with pytest.raises(AssertionError):
        assert_known_usage(_summary(total_tokens=True), calls=3, prompt_tokens=33, completion_tokens=12, total_tokens=45)


def test_missing_summary_is_rejected() -> None:
    with pytest.raises(AssertionError):
        assert_known_usage(None, calls=3, prompt_tokens=1, completion_tokens=1, total_tokens=2)  # type: ignore[arg-type]


def test_checked_field_list_matches_the_agent_summary_fields() -> None:
    from queryshield.agent.graph import _usage_summary

    events = [
        {"kind": "model_call", "model_call_id": f"id-{i}", "usage_status": "known",
         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        for i in range(2)
    ]
    assert set(_usage_summary(events)) == check_agent_usage.CHECKED_FIELDS


@pytest.mark.parametrize("script", ["check_agent_t05.py", "check_agent_fs.py"])
def test_check_scripts_load_the_usage_module_under_runpy(script: str) -> None:
    namespace = runpy.run_path(str(SCRIPTS / script))
    assert namespace["assert_known_usage"].__module__ == "check_agent_usage"


def test_direct_run_from_another_directory_resolves_the_usage_module(tmp_path: Path) -> None:
    for script, extra in (("check_agent_t05.py", []), ("check_agent_fs.py", ["--check-id", "AGENT-FS01", "--mode", "fake"])):
        completed = subprocess.run(
            [sys.executable, str(SCRIPTS / script), "--output-dir", str(tmp_path), *extra],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr[-500:]
