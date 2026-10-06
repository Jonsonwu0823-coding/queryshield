from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_multi_run_summary.py"


def _module() -> dict[str, object]:
    return runpy.run_path(str(SCRIPT))


def _record(
    case_id: str,
    profile: str,
    status: str,
    *,
    critical=None,
    error_code=None,
    terminal_state="SUCCEEDED",
    prompt="qs-system-prompt-v21",
):
    return {
        "case_id": case_id,
        "profile": profile,
        "split": "development",
        "critical_question_id": critical,
        "judged_status": status,
        "error_code": error_code,
        "terminal_state": terminal_state,
        "run_config": {"prompt_version": prompt},
        "execution_events": [{"kind": "model_call", "prompt_version": prompt}],
    }


def _root(
    tmp_path: Path,
    name: str,
    *,
    b1_empty_window: str,
    manifest: str = "manifest-a",
    prompt="qs-system-prompt-v21",
    failure_terminal_state: str = "FAILED",
) -> Path:
    root = tmp_path / name
    suite = root / "full-real"
    suite.mkdir(parents=True)
    (suite / "summary.json").write_text(
        json.dumps({"overall_status": "fail", "checks": [{"check_id": "EVAL-FS02", "status": "fail"}, {"check_id": "EVAL-R05", "status": "pass"}]}),
        encoding="utf-8",
    )
    (suite / "source-manifest.txt").write_text(manifest + "\n", encoding="utf-8")
    raw = {
        "dataset_split": "development",
        "raw_records": {
            "B0": [_record("gross-total-fen", "B0", "pass", critical="gross-total", prompt=prompt)],
            "B1": [
                _record("gross-total-fen", "B1", "pass", critical="gross-total", prompt=prompt),
                _record(
                    "empty-window-zero-aggregate",
                    "B1",
                    b1_empty_window,
                    critical="empty-window",
                    error_code=None if b1_empty_window == "pass" else "parallel_unavailable",
                    terminal_state="SUCCEEDED" if b1_empty_window == "pass" else failure_terminal_state,
                    prompt=prompt,
                ),
            ],
        },
    }
    (suite / "stateful-real-raw.json").write_text(json.dumps(raw), encoding="utf-8")
    (suite / "stateful-real-comparison-report.json").write_text(
        json.dumps({"profile_reports": {"B0": {"metrics": {"security_violations": {"numerator": 0}}}, "B1": {"metrics": {"security_violations": {"numerator": 0}}}}}),
        encoding="utf-8",
    )
    # Material the summary must never read or count.
    sealed = root / "t05"
    sealed.mkdir()
    (sealed / "stateful-real-raw.json").write_text(
        json.dumps({"dataset_split": "development", "raw_records": {"B1": [_record("sealed-case", "B1", "pass")]}}),
        encoding="utf-8",
    )
    (suite / "task-raw-by-profile.json.aesgcm").write_bytes(b"\x00encrypted")
    (suite / "notes-holdout.json").write_text("{}", encoding="utf-8")
    return root


def test_summary_counts_passes_per_case_across_roots(tmp_path) -> None:
    roots = [
        _root(tmp_path, "run-1", b1_empty_window="pass"),
        _root(tmp_path, "run-2", b1_empty_window="fail"),
        _root(tmp_path, "run-3", b1_empty_window="pass"),
    ]
    summary = _module()["summarize_roots"](roots)

    assert summary["consistent_candidate"] is True and summary["warnings"] == []
    run = summary["runs"][0]["suites"]["full-real"]
    assert run["checks"] == {"EVAL-FS02": "fail", "EVAL-R05": "pass"}
    assert run["security_violations"] == 0
    assert run["manifest_sha256"] and run["prompt_versions"] == ["qs-system-prompt-v21"]
    empty_window = next(item for item in summary["critical_cases"] if item["case_id"] == "empty-window-zero-aggregate")
    assert (empty_window["runs"], empty_window["passes"], empty_window["failure_error_codes"]) == (3, 2, ["parallel_unavailable"])
    assert all(item["case_id"] != "sealed-case" for item in summary["cases"])
    assert list(summary["runs"][0]["suites"]) == ["full-real"]


def test_summary_flags_mixed_candidates(tmp_path) -> None:
    roots = [
        _root(tmp_path, "run-a", b1_empty_window="pass", manifest="manifest-a"),
        _root(tmp_path, "run-b", b1_empty_window="pass", manifest="manifest-b", prompt="qs-system-prompt-v20"),
    ]
    summary = _module()["summarize_roots"](roots)
    assert summary["consistent_candidate"] is False
    assert {item["kind"] for item in summary["warnings"]} == {"manifest_mismatch", "prompt_version_mismatch"}


def test_summary_never_opens_sealed_or_encrypted_files(tmp_path, monkeypatch) -> None:
    roots = [_root(tmp_path, "run-1", b1_empty_window="pass")]
    opened: list[str] = []
    original_text, original_bytes = Path.read_text, Path.read_bytes

    def read_text(self, *args, **kwargs):
        opened.append(str(self))
        return original_text(self, *args, **kwargs)

    def read_bytes(self, *args, **kwargs):
        opened.append(str(self))
        return original_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    _module()["summarize_roots"](roots)
    assert opened
    assert not any("t05" in path or "aesgcm" in path or "holdout" in path for path in opened)


def test_summary_rejects_missing_roots(tmp_path, capsys) -> None:
    assert _module()["main"]([str(tmp_path / "missing")]) == 2
    assert "missing_roots" in capsys.readouterr().out


def test_summary_ignores_non_development_raw(tmp_path) -> None:
    root = _root(tmp_path, "run-1", b1_empty_window="pass")
    raw_path = root / "full-real" / "stateful-real-raw.json"
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    payload["dataset_split"] = "holdout"
    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _module()["summarize_roots"]([root])["cases"] == []


@pytest.mark.parametrize("name", ["summary.json", "EVAL-FS02.json", "stateful-supplement-real-raw.json", "source-manifest.txt"])
def test_summary_allow_list_names(name) -> None:
    module = _module()
    allowed = (
        name in {"summary.json", "source-manifest.txt"}
        or module["_CHECK_RESULT"].fullmatch(name)
        or module["_RAW"].fullmatch(name)
        or module["_REPORT"].fullmatch(name)
    )
    assert allowed


def test_summary_failures_carry_terminal_state(tmp_path) -> None:
    roots = [
        _root(tmp_path, "run-1", b1_empty_window="fail", failure_terminal_state="WAITING_USER"),
        _root(tmp_path, "run-2", b1_empty_window="fail", failure_terminal_state="FAILED"),
        _root(tmp_path, "run-3", b1_empty_window="pass"),
    ]
    summary = _module()["summarize_roots"](roots)
    empty_window = next(item for item in summary["critical_cases"] if item["case_id"] == "empty-window-zero-aggregate")
    assert (empty_window["runs"], empty_window["passes"]) == (3, 1)
    assert [(item["root"], item["terminal_state"]) for item in empty_window["failures"]] == [
        (str(roots[0]), "WAITING_USER"),
        (str(roots[1]), "FAILED"),
    ]
    assert all(item["error_code"] == "parallel_unavailable" for item in empty_window["failures"])
    passing = next(item for item in summary["cases"] if item["case_id"] == "gross-total-fen" and item["profile"] == "B1")
    assert passing["failures"] == []
