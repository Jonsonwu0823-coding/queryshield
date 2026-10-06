"""Names and text carry no week or ticket label.

A file name, directory name or line of text says what the code is for, not which week or ticket
produced it.  This scans what the public repository contains (the project directory without
``docs/w03-*.md``, plus the CI workflow) for week labels (``W01``-``W12``) and ticket labels
(``B2b``, ``B3c-1``, ``C1 验收`` ...).  A label that is a value on purpose (written into records
or hashes, sent to the model, the name of a git tag or of a past run) is listed in
``name_label_exceptions.json`` with its exact string, how many times it occurs and why.

A failure prints the path, the line and the label found, never the line itself.
"""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path
import re
import subprocess

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = Path(".github") / "workflows" / "queryshield-ci.yml"
EXCEPTIONS = Path(__file__).with_name("name_label_exceptions.json")

WEEK = re.compile(r"(?<![A-Za-z0-9])[Ww](?:0[1-9]|1[0-2])(?![0-9])")
TICKET_B = re.compile(r"(?<![A-Za-z0-9])[Bb](?:1-[12]|[2-4][a-e](?:-?[12])?)(?![A-Za-z0-9])")
TICKET_CS = re.compile(r"(?<![A-Za-z0-9])[CS][1-9](?=[ :]?(?:验收|返工|计划|实现|review|rework|plan)|:)")
CONTENT_PATTERNS = (WEEK, TICKET_B, TICKET_CS)
PATH_LABEL = re.compile(r"(?<![a-z0-9])(?:w(?:0[1-9]|1[0-2])|b1-[12]|b[2-4][a-e](?:-?[12])?|c[1-5]|s[1-5])(?![a-z0-9])")
SKIPPED_DIRS = {".venv", "__pycache__", ".pytest_cache", ".git", "build", "dist"}


class FileListError(RuntimeError):
    """git is there but could not list the tracked files; the scan does not guess."""


def tracked_files(repo: Path, *pathspec: str, cwd: Path) -> list[str]:
    """Paths git tracks under ``cwd``.  ``safe.directory`` is scoped to this repository only."""

    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}  # no GIT_DIR or index pointing elsewhere
    listed = subprocess.run(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "ls-files", "-z", "--", *pathspec],
        cwd=cwd,
        env=env,
        capture_output=True,
        check=False,
    )
    if listed.returncode != 0:
        reason = listed.stderr.decode("utf-8", "replace").strip().splitlines()[:1]
        raise FileListError(f"git ls-files failed in {repo.name} (exit {listed.returncode}): {reason[0] if reason else 'no message'}")
    return [n for n in listed.stdout.decode("utf-8").split("\0") if n]


def walked_files(root: Path) -> list[str]:
    """Every file under ``root``, for a copy that is not a git work tree (a downloaded archive)."""

    return [
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and not (set(p.relative_to(root).parts) & SKIPPED_DIRS) and not p.name.endswith(".pyc")
    ]


def scanned_files(root: Path = PROJECT_ROOT) -> list[tuple[Path, str]]:
    """(file, path as the exception table writes it) for every file the public repository holds.

    In a git work tree only the files git tracks are read: local evidence, raw records and ``.env``
    sit untracked next to them and are never opened.  Without git (an archive) every file is the project's.
    """

    repo = next((base for base in (root, root.parent) if (base / ".git").exists()), None)
    if repo is None:
        names = walked_files(root)
    else:
        names = tracked_files(repo, cwd=root)
        if repo != root and not any(n == WORKFLOW.as_posix() for n in names):
            if tracked_files(repo, WORKFLOW.as_posix(), cwd=repo):
                names.append(WORKFLOW.as_posix())
    own = {EXCEPTIONS.relative_to(PROJECT_ROOT).as_posix(), Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix()}
    files = []
    for name in sorted(names):
        if fnmatch.fnmatch(name, "docs/w03-*.md") or name in own:  # the two own files list labels on purpose
            continue
        path = root / name if (root / name).is_file() else (root.parent / name)
        if path.is_file():
            files.append((path, name))
    return files


def load_exceptions() -> list[dict]:
    return json.loads(EXCEPTIONS.read_text(encoding="utf-8"))["exceptions"]


def mask(text: str, rel: str, exceptions: list[dict]) -> str:
    """Replace every excepted literal that applies to ``rel`` by blanks of the same width."""

    for entry in sorted(exceptions, key=lambda e: len(e["literal"]), reverse=True):
        if fnmatch.fnmatchcase(rel, entry["file"]):
            text = text.replace(entry["literal"], "\0" * len(entry["literal"]))
    return text


def label_hits(text: str) -> list[tuple[int, str]]:
    hits = []
    for number, line in enumerate(text.split("\n"), start=1):
        for pattern in CONTENT_PATTERNS:
            hits.extend((number, m.group(0)) for m in pattern.finditer(line))
    return hits


def path_hits(rel: str) -> list[str]:
    return [m.group(0) for part in rel.split("/") for m in PATH_LABEL.finditer(part.lower())]


def read(path: Path) -> str | None:
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return None


def label_problems(root: Path, exceptions: list[dict]) -> list[str]:
    """One line per label found under ``root``: ``path: path has 'x'`` or ``path:line: label``."""

    problems = []
    for path, rel in scanned_files(root):
        problems.extend(f"{rel}: path has {token!r}" for token in path_hits(rel))
        text = read(path)
        if text is not None:
            problems.extend(f"{rel}:{number}: {token}" for number, token in label_hits(mask(text, rel, exceptions)))
    return problems


def test_no_week_or_ticket_label_is_left_outside_the_exceptions() -> None:
    problems = label_problems(PROJECT_ROOT, load_exceptions())
    assert not problems, "week or ticket labels:\n" + "\n".join(problems[:60])


def test_every_exception_occurs_exactly_as_often_as_it_says() -> None:
    exceptions = load_exceptions()
    files = [(rel, read(path)) for path, rel in scanned_files()]
    wrong = []
    for entry in exceptions:
        found = sum(text.count(entry["literal"]) for rel, text in files if text and fnmatch.fnmatchcase(rel, entry["file"]))
        if found != entry["count"]:
            wrong.append(f"{entry['file']} [{entry['class']}]: declared {entry['count']}, found {found}")
    assert not wrong, "exception counts differ:\n" + "\n".join(wrong[:60])


def test_every_exception_names_its_reason() -> None:
    for entry in load_exceptions():
        assert entry["literal"] and entry["file"] and entry["reason"].strip(), entry["file"]
        assert entry["count"] >= 1, f"{entry['file']}: an exception nothing needs"


@pytest.mark.parametrize(
    "text",
    ["W05 holdout", "see w03_notes", "run B3e", "B3c-1 follow-up", "b2b_http_smoke", "Week W12.", "C1 验收", "S2: tidy"],
)
def test_the_scan_finds_labels(text: str) -> None:
    assert label_hits(text), text


@pytest.mark.parametrize(
    "text",
    ["commerce-v1", "catalog-v4", "B0 and B1 profiles", "0b3d1c9f", "SHA c1d2e3", "W5", "W130", "STATE-X01", "C1 is a customer"],
)
def test_the_scan_leaves_ordinary_text_alone(text: str) -> None:
    assert not label_hits(text), text


def test_the_scan_finds_labels_in_paths() -> None:
    assert path_hits("tests/test_b2b_runtime.py") and path_hits("scripts/check_w05.py") and path_hits("docs/w03-notes.md")
    assert not path_hits("tests/test_shared_runtime.py") and not path_hits("evals/development/state-cases-v4.json")


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "kept.txt").write_text("tracked\n", encoding="utf-8")
    (repo / "local-evidence").mkdir()
    (repo / "local-evidence" / "raw.json").write_text('{"label": "W05"}\n', encoding="utf-8")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
    subprocess.run([*git, "add", "kept.txt"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], cwd=repo, check=True)
    return repo


def test_in_a_git_work_tree_only_tracked_files_are_listed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = make_repo(tmp_path)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere"))  # must not redirect the listing
    assert [rel for _, rel in scanned_files(repo)] == ["kept.txt"]


def test_a_git_failure_is_an_error_and_the_directory_is_not_walked(tmp_path: Path) -> None:
    broken = tmp_path / "project"
    (broken / ".git").mkdir(parents=True)  # looks like a work tree, git cannot use it
    (broken / "local-evidence").mkdir()
    (broken / "local-evidence" / "raw.json").write_text('{"label": "W05"}\n', encoding="utf-8")
    with pytest.raises(FileListError, match="git ls-files failed"):
        scanned_files(broken)


def test_outside_a_git_work_tree_the_directory_is_walked(tmp_path: Path) -> None:
    root = tmp_path / "project"
    for name in ("a.txt", "docs/w03-notes.md", ".venv/x.txt", "pkg/__pycache__/m.pyc", "pkg/m.py"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x\n", encoding="utf-8")
    assert [rel for _, rel in scanned_files(root)] == ["a.txt", "pkg/m.py"]


def commit_layout(tmp_path: Path, tracked: dict[str, str], untracked: dict[str, str]) -> Path:
    """A development layout: the git repository is ``repo``, the project is ``repo/project``."""

    repo = tmp_path / "repo"
    for files in (tracked, untracked):
        for name, text in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text, encoding="utf-8")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
    subprocess.run([*git, "add", "--", *tracked], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], cwd=repo, check=True)
    return repo / "project"


PROJECT_FILES = {
    "project/README.md": "see W05\n",
    "project/docs/a.md": "see W05\n",
    "project/docs/w03-x.md": "see W05\n",
    "project/fixtures/f.json": '{"label": "W05"}\n',
    "project/src/m.py": "# W05\n",
    "project/scripts/s.ps1": "# W05\n",
    "project/tests/test_b2b_x.py": "# W05\n",
    ".github/workflows/queryshield-ci.yml": "# W05\n",
}


def test_the_scan_covers_every_kind_of_tracked_file_and_reports_paths_and_text(tmp_path: Path) -> None:
    project = commit_layout(tmp_path, PROJECT_FILES, {"project/local-evidence/raw.json": '{"label": "W05"}\n'})
    expected = sorted(rel.removeprefix("project/") if rel.startswith("project/") else rel for rel in PROJECT_FILES if rel != "project/docs/w03-x.md")
    assert sorted(rel for _, rel in scanned_files(project)) == expected
    problems = label_problems(project, [])
    for rel in expected:
        assert any(line.startswith(f"{rel}:") and line.endswith("W05") for line in problems), rel
    assert "tests/test_b2b_x.py: path has 'b2b'" in problems
    assert not any("local-evidence" in line or "w03-x" in line for line in problems)


def test_a_workflow_git_does_not_track_is_not_scanned(tmp_path: Path) -> None:
    tracked = {k: v for k, v in PROJECT_FILES.items() if k != ".github/workflows/queryshield-ci.yml"}
    project = commit_layout(tmp_path, tracked, {".github/workflows/queryshield-ci.yml": "# W05\n"})
    assert ".github/workflows/queryshield-ci.yml" not in [rel for _, rel in scanned_files(project)]
