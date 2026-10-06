"""What the public repository must look like, checked on the files that are in it.

The license, the places the local helper scripts write their evidence, what git ignores, the CI
runner pin and the wording of the skip reasons.  Nothing here needs the development repository, so
the file passes in both layouts (see repo_layout.py).
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pytest

from repo_layout import REPO_ROOT, STANDALONE_WORKFLOW, WORKFLOW_RELATIVE
from test_container_files import SUMMARY_JSON, _fake_docker, _SHELL, needs_posix, needs_powershell

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
EVIDENCE_ROOT_SCRIPTS = ("http-local-smoke.ps1", "demo-local.ps1", "mcp-local-smoke.ps1", "eval-local-real.ps1")
COMPOSE_SCRIPT = "compose-real-demo.ps1"
LOCAL_SCRIPTS = (*EVIDENCE_ROOT_SCRIPTS, COMPOSE_SCRIPT)
# Written in pieces so that this file itself does not contain what the checks look for.
CLOUD_SESSION_DIR = "." + "cloud"
WORK_ORDER = "工" + "单"
WEEKLY_TASKS_DIR = "01_" + "每周任务"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _workflow_text() -> str:
    for path in (STANDALONE_WORKFLOW, REPO_ROOT / WORKFLOW_RELATIVE):
        if path.is_file():
            return _text(path)
    pytest.skip("the CI workflow is not in this checkout")


# --- license -----------------------------------------------------------------------------------------


def test_license_is_the_standard_mit_text_with_the_copyright_line() -> None:
    lines = _text(PROJECT_ROOT / "LICENSE").splitlines()
    assert lines[0] == "MIT License" and lines[1] == "" and lines[2] == "Copyright (c) 2026 Jiawei Wu"
    text = "\n".join(lines)
    assert "Permission is hereby granted, free of charge, to any person obtaining a copy" in text
    assert 'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND' in text


def test_readme_ends_with_a_license_section_that_links_the_license_file() -> None:
    readme = _text(PROJECT_ROOT / "README.md")
    section = readme[readme.index("\n## 许可证"):]
    assert "MIT" in section and "](LICENSE)" in section
    assert "\n## " not in section[1:], "the license section is the last one"


# --- the local helper scripts ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", LOCAL_SCRIPTS)
def test_local_scripts_no_longer_reach_outside_the_project_for_their_evidence(name: str) -> None:
    text = _text(SCRIPTS / name)
    assert "workspaceRoot" not in text and WEEKLY_TASKS_DIR not in text
    assert "evidence\\engineering" not in text and "Split-Path -Parent (Split-Path -Parent $projectRoot)" not in text
    assert 'Join-Path $projectRoot "evidence"' in text


@pytest.mark.parametrize("name", EVIDENCE_ROOT_SCRIPTS)
def test_four_local_scripts_take_an_optional_evidence_root(name: str) -> None:
    text = _text(SCRIPTS / name)
    param_block = text[text.index("param("): text.index("\n)", text.index("param("))]
    assert "[string]$EvidenceRoot" in param_block and "EvidenceDir" not in param_block
    assert "GetUnresolvedProviderPathFromPSPath($EvidenceRoot)" in text


def test_the_compose_demo_script_keeps_its_evidence_dir_parameter() -> None:
    text = _text(SCRIPTS / COMPOSE_SCRIPT)
    param_block = text[text.index("param("): text.index("\n)", text.index("param("))]
    assert "[string]$EvidenceDir" in param_block and "EvidenceRoot" not in param_block


def test_the_obsolete_week_four_rework_runner_is_gone() -> None:
    assert not (SCRIPTS / "run_w04_rework_secure.ps1").exists()


@needs_powershell
@needs_posix
def test_the_compose_demo_script_writes_its_default_evidence_inside_the_project(tmp_path) -> None:
    import os
    import shutil

    project = tmp_path / "queryshield"
    (project / "scripts").mkdir(parents=True)
    shutil.copyfile(SCRIPTS / COMPOSE_SCRIPT, project / "scripts" / COMPOSE_SCRIPT)
    (project / ".env").write_text("", encoding="utf-8")
    _fake_docker(tmp_path / "bin")
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}", "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT": "1"}
    before = set(tmp_path.parent.iterdir())
    completed = subprocess.run(
        [_SHELL, "-NoProfile", "-File", str(project / "scripts" / COMPOSE_SCRIPT), "-FakeDryRun"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    runs = list((project / "evidence").glob("compose-fake-*"))
    assert len(runs) == 1, "one run directory under <project>/evidence"
    for name in ("demo-summary.json", "walkthrough-summary.json"):
        assert (runs[0] / name).read_text(encoding="utf-8").strip() == SUMMARY_JSON
    assert set(tmp_path.parent.iterdir()) == before, "nothing is created outside the project's own directory"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["bin", "queryshield"]


# --- git ignore, CI pin, wording -------------------------------------------------------------------------


def test_gitignore_keeps_local_evidence_encrypted_material_and_raw_records_out() -> None:
    lines = set(_text(PROJECT_ROOT / ".gitignore").splitlines())
    for pattern in ("/evidence/", "*.enc", "*.aesgcm", "*.dpapi", "*-raw.json", CLOUD_SESSION_DIR + "/", ".env", ".env.*", "!.env.example"):
        assert pattern in lines, pattern


def test_ci_jobs_run_on_a_pinned_ubuntu_release() -> None:
    text = _workflow_text()
    runners = re.findall(r"^\s+runs-on:\s*(\S+)\s*$", text, re.MULTILINE)
    assert runners == ["ubuntu-24.04", "ubuntu-24.04"], runners
    assert "ubuntu-latest" not in text


@pytest.mark.parametrize("name", ("test_rls_checks.py", "test_demo_expected_answers.py", "test_demo_walkthrough.py"))
def test_skip_reasons_do_not_point_at_the_cloud_session_file(name: str) -> None:
    text = _text(PROJECT_ROOT / "tests" / name)
    assert CLOUD_SESSION_DIR not in text
    reasons = re.findall(r'reason="([^"]*)"', text)
    assert reasons and all(CLOUD_SESSION_DIR not in reason for reason in reasons)


def test_the_demo_data_doc_uses_neutral_wording_and_the_new_evidence_location() -> None:
    text = _text(PROJECT_ROOT / "docs" / "demo-data.md")
    assert WORK_ORDER not in text
    assert "往上两级" not in text and "evidence\\engineering" not in text
    assert "evidence\\demo-" in text and "-EvidenceRoot" in text
