"""STATE-X01 and EVAL-X01 read the upstream records from this repository only.

Each test builds a throw-away Git repository with the same layout (a ``queryshield/``
project directory plus ``control/evidence/upstream/``), so nothing here needs files
outside the checkout, and negative controls can change any single input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess

import pytest

from repo_layout import needs_register
from scripts import check_state

pytestmark = needs_register

REAL_REPO = Path(__file__).resolve().parents[2]
CHECK_EVAL = REAL_REPO / "queryshield" / "scripts" / "check_eval.py"
UPSTREAM = Path("control/evidence/upstream")
EVAL_SOURCES = (
    "src/queryshield/evaluation/state_cases.py",
    "src/queryshield/evaluation/state_oracle.py",
    "src/queryshield/evaluation/profile_runner.py",
    "src/queryshield/providers/rerank.py",
    "evals/development/state-cases-v1.json",
    "evals/development/state-cases-v2.json",
    "evals/development/state-cases-v3.json",
    "evals/development/state-cases-v4.json",
    "evals/development/execution-profiles-v1.json",
    "evals/development/retrieval-cases-v1.json",
    "scripts/check_eval.py",
    "scripts/check.ps1",
)
GIT_IDENTITY = ("-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={repo.as_posix()}", *GIT_IDENTITY, *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _clean_git_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY", "GIT_NAMESPACE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    project = root / "queryshield"
    for relative in (*EVAL_SOURCES, "docs/w03-a04-versioned-runtime.md", "scripts/check_state.py"):
        (project / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL_REPO / "queryshield" / relative, project / relative)
    (root / UPSTREAM).mkdir(parents=True)
    for name in ("W03-accepted-source-manifest.txt", "W04-accepted-source-manifest.txt", "W04-TASKS.snapshot.md"):
        shutil.copyfile(REAL_REPO / UPSTREAM / name, root / UPSTREAM / name)
    _git(root, "init", "--quiet")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "assets")
    introducing = _git(root, "rev-parse", "HEAD")

    register = json.loads((REAL_REPO / UPSTREAM / "accepted-assets.json").read_text(encoding="utf-8"))
    for asset in register["assets"]:
        if asset["id"] == "W04-A05":
            template = asset["revisions"][0]
            asset["revisions"] = [
                {**template, "sha256": _sha(project / "scripts/check_state.py"), "introduced_in": introducing, "repo_path": asset["repo_path"]}
            ]
    for tag in register["tags"].values():
        tag["commit"] = introducing
    for tag in register["tags"]:
        _git(root, "tag", tag, introducing)
    (root / UPSTREAM / "accepted-assets.json").write_text(json.dumps(register, indent=2) + "\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "register")
    return root


def _register(repo: Path) -> dict:
    return json.loads((repo / UPSTREAM / "accepted-assets.json").read_text(encoding="utf-8"))


def _write_register(repo: Path, register: dict) -> None:
    (repo / UPSTREAM / "accepted-assets.json").write_text(json.dumps(register, indent=2) + "\n", encoding="utf-8")


def _eval_x01(repo: Path) -> dict:
    return runpy.run_path(str(CHECK_EVAL))["_check_upstream_versions"](repo)


def _state_x01(monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path) -> dict:
    monkeypatch.setattr(check_state, "REPO_ROOT", repo)
    return check_state.check_x01(tmp_path)


def test_real_repository_passes_both_x01_checks_without_outside_files(tmp_path: Path) -> None:
    result = runpy.run_path(str(CHECK_EVAL))["_check_upstream_versions"]()
    assert result["status"] == "pass"
    assert len(result["accepted_assets_sha256"]) == 64
    assert result["accepted_assets_sha256"] == _sha(REAL_REPO / UPSTREAM / "accepted-assets.json")
    assert check_state.check_x01(tmp_path)["upstream_tag"] == "qs-w03-accepted-20260922"


def test_synthetic_repository_passes_and_records_register_hash(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _eval_x01(repo)
    assert result["status"] == "pass"
    assert result["accepted_assets_sha256"] == _sha(repo / UPSTREAM / "accepted-assets.json")
    revisions = result["accepted_upstream_asset_checks"]["W04-A05"]["revisions_verified"]
    assert len(revisions) == 1
    assert _state_x01(monkeypatch, repo, tmp_path)["accepted_assets_sha256"] == result["accepted_assets_sha256"]


def test_asset_changed_without_registering_it_fails(repo: Path) -> None:
    path = repo / "queryshield/scripts/check_state.py"
    path.write_bytes(path.read_bytes() + b"# unregistered change\n")
    with pytest.raises(AssertionError, match="W04-A05 current source differs"):
        _eval_x01(repo)


def test_runtime_doc_asset_changed_fails(repo: Path) -> None:
    path = repo / "queryshield/docs/w03-a04-versioned-runtime.md"
    path.write_bytes(path.read_bytes() + b"\nchanged\n")
    with pytest.raises(AssertionError, match="W03-A04 current source differs"):
        _eval_x01(repo)


def test_register_edited_without_the_file_fails(repo: Path) -> None:
    register = _register(repo)
    register["assets"][1]["revisions"][0]["sha256"] = "0" * 64
    _write_register(repo, register)
    with pytest.raises(AssertionError, match="W04-A05 current source differs"):
        _eval_x01(repo)


def test_forged_revision_naming_a_commit_with_other_content_fails(repo: Path) -> None:
    path = repo / "queryshield/scripts/check_state.py"
    path.write_bytes(path.read_bytes() + b"# forged\n")
    register = _register(repo)
    register["assets"][1]["revisions"][0]["sha256"] = _sha(path)
    _write_register(repo, register)
    with pytest.raises(AssertionError, match="revision 0 hash does not match the file at its introducing commit"):
        _eval_x01(repo)


def test_revision_commit_missing_from_the_clone_is_a_failure_with_a_hint(repo: Path) -> None:
    register = _register(repo)
    register["assets"][1]["revisions"][0]["introduced_in"] = "1" * 40
    _write_register(repo, register)
    result = _eval_x01(repo)
    assert result["status"] == "fail"
    assert result["git_error_class"] == "revision_not_found"
    assert "git fetch" in result["hint"]


def test_revision_commit_outside_head_history_fails(repo: Path) -> None:
    orphan = _git(repo, "commit-tree", "HEAD^{tree}", "-m", "orphan")
    register = _register(repo)
    register["assets"][1]["revisions"][0]["introduced_in"] = orphan
    _write_register(repo, register)
    result = _eval_x01(repo)
    assert result["status"] == "fail"
    assert result["git_error_class"] == "commit_not_in_history"


def test_tampered_upstream_record_copy_fails(repo: Path) -> None:
    path = repo / UPSTREAM / "W04-accepted-source-manifest.txt"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(AssertionError, match="source manifest bytes differ"):
        _eval_x01(repo)


def test_missing_upstream_record_is_blocked_not_pass(repo: Path) -> None:
    (repo / UPSTREAM / "W03-accepted-source-manifest.txt").unlink()
    result = _eval_x01(repo)
    assert result["status"] == "blocked"


def test_eval_x01_missing_tag_is_not_a_pass(repo: Path) -> None:
    _git(repo, "tag", "-d", "qs-w04-accepted-20260923")
    result = _eval_x01(repo)
    assert result["status"] == "fail"
    assert result["git_error_class"] == "tag_not_found"
    assert "git fetch --tags" in result["hint"]


def test_eval_x01_tag_on_another_commit_fails(repo: Path) -> None:
    _git(repo, "tag", "-f", "qs-w03-accepted-20260922", "HEAD")
    result = _eval_x01(repo)
    assert result["status"] == "fail"
    assert result["failed_assertion"] == "accepted_tag_resolves_to_recorded_commit"


def test_state_x01_missing_tag_fails_with_fetch_hint(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git(repo, "tag", "-d", "qs-w03-accepted-20260922")
    with pytest.raises(AssertionError, match="git fetch --tags origin"):
        _state_x01(monkeypatch, repo, tmp_path)


def test_state_x01_tag_on_another_commit_fails(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git(repo, "tag", "-f", "qs-w03-accepted-20260922", "HEAD")
    with pytest.raises(AssertionError, match="does not resolve to the recorded commit"):
        _state_x01(monkeypatch, repo, tmp_path)


def test_state_x01_tampered_task_entry_copy_fails(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = repo / UPSTREAM / "W04-TASKS.snapshot.md"
    path.write_bytes(path.read_bytes() + b"\nextra\n")
    with pytest.raises(AssertionError, match="task-entry copy differs from the hash pinned"):
        _state_x01(monkeypatch, repo, tmp_path)


def test_state_x01_missing_register_fails(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (repo / UPSTREAM / "accepted-assets.json").unlink()
    with pytest.raises(AssertionError, match="register is missing"):
        _state_x01(monkeypatch, repo, tmp_path)


def _rename_asset(repo: Path) -> str:
    """Move check_state.py under another name; the register follows with the old paths kept where they belong."""

    _git(repo, "mv", "queryshield/scripts/check_state.py", "queryshield/scripts/check_renamed.py")
    _git(repo, "commit", "--quiet", "-m", "rename the asset")
    register = _register(repo)
    asset = register["assets"][1]
    old_repo_path = asset["repo_path"]
    asset["revisions"][0]["repo_path"] = old_repo_path
    asset["path"], asset["repo_path"] = "scripts/check_renamed.py", "queryshield/scripts/check_renamed.py"
    _write_register(repo, register)
    return old_repo_path


def test_renamed_asset_passes_when_the_register_keeps_each_older_path(repo: Path) -> None:
    _rename_asset(repo)
    result = _eval_x01(repo)
    assert result["status"] == "pass"
    assert len(result["accepted_upstream_asset_checks"]["W04-A05"]["revisions_verified"]) == 1


def test_revision_naming_a_path_that_did_not_hold_the_file_fails(repo: Path) -> None:
    _rename_asset(repo)
    register = _register(repo)
    register["assets"][1]["revisions"][0]["repo_path"] = "queryshield/scripts/check_renamed.py"
    _write_register(repo, register)
    result = _eval_x01(repo)
    assert result["status"] == "fail"
    assert result["failed_assertion"] == "registered_revision_readable_at_introducing_commit"


def test_accepted_path_must_name_the_entry_of_the_accepted_record(repo: Path) -> None:
    register = _register(repo)
    register["assets"][1]["accepted"]["path"] = "scripts/not_in_the_record.py"
    _write_register(repo, register)
    with pytest.raises(AssertionError, match="accepted baseline is not the hash"):
        _eval_x01(repo)


@pytest.mark.parametrize("where", ("accepted", "revision"))
def test_optional_path_fields_must_be_non_empty_strings(repo: Path, where: str) -> None:
    register = _register(repo)
    asset = register["assets"][1]
    (asset["accepted"] if where == "accepted" else asset["revisions"][0])["path" if where == "accepted" else "repo_path"] = ""
    _write_register(repo, register)
    with pytest.raises(AssertionError, match="is not a non-empty string"):
        _eval_x01(repo)
