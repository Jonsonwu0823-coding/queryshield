"""Check.ps1 on Linux (path rules) and check-all.ps1 (exclusion list matches what check.ps1 registers).

The text assertions need no PowerShell.  The behavioural ones run when pwsh (or powershell)
is installed, which is always the case on GitHub's Linux runners.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys

import pytest

from repo_layout import REGISTER, REGISTER_SKIP_REASON, WORKFLOW_RELATIVE, layout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
CHECK_PS1 = PROJECT_ROOT / "scripts" / "check.ps1"
CHECK_ALL_PS1 = PROJECT_ROOT / "scripts" / "check-all.ps1"
SHELL = shutil.which("pwsh") or shutil.which("powershell")
needs_powershell = pytest.mark.skipif(SHELL is None, reason="needs PowerShell (pwsh); GitHub's Linux runners have it")
needs_posix = pytest.mark.skipif(sys.platform == "win32", reason="POSIX paths and executables")


def _text(path: Path) -> str:
    return path.read_text(encoding="ascii")


def _registry(text: str) -> dict[str, list[str]]:
    """Check ids per suite from check.ps1's ``$script:allRequiredCheckIds = switch`` block."""

    start = text.index("$script:allRequiredCheckIds = switch")
    block = text[start: text.index("$script:requiredCheckIds = @(", start)]
    suites: dict[str, list[str]] = {}
    for name in ("DB-SMOKE", "AGENT", "STATE", "EVAL"):
        match = re.search(r'"' + re.escape(name) + r'" \{ @\(([^)]*)\) \}', block)
        assert match, f"{name} is not registered on one line any more"
        suites[name] = re.findall(r'"([^"]+)"', match.group(1))
    proposal_block = re.search(r'"PROPOSAL" \{(.*?)\n    \}\n', block, re.S)
    assert proposal_block
    suites["PROPOSAL"] = sorted(set(re.findall(r'"(PROPOSAL-[A-Z0-9-]+)"', proposal_block.group(1))))
    default = re.search(r"default \{ @\(([^)]*)\) \}", block)
    assert default
    suites["BASE"] = re.findall(r'"([^"]+)"', default.group(1))
    return suites


def _excluded_checks(text: str) -> list[dict[str, str]]:
    block = text[text.index("$script:ExcludedChecks = @("): text.index("$script:RegisterRelativePath")]
    items = []
    for match in re.finditer(r'@\{ Id = "([^"]+)"; Suite = "([^"]+)"; Kind = "([^"]+)"; Reason = "([^"]*)" \}', block):
        items.append({"id": match.group(1), "suite": match.group(2), "kind": match.group(3), "reason": match.group(4)})
    return items


def _powershell(args: list[str], *, cwd: Path = PROJECT_ROOT, env: dict[str, str] | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    full_env = {**os.environ, "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT": "1", **(env or {})}
    return subprocess.run([SHELL, "-NoProfile", *args], cwd=cwd, env=full_env, capture_output=True, text=True, timeout=timeout, check=False)


# --- text assertions --------------------------------------------------------------


def test_both_scripts_are_plain_ascii_lf_without_a_bom() -> None:
    for path in (CHECK_PS1, CHECK_ALL_PS1):
        data = path.read_bytes()
        assert not data.startswith(b"\xef\xbb\xbf"), f"{path.name} must not have a BOM"
        assert b"\r" not in data, f"{path.name} must use LF"
        data.decode("ascii")


def test_exclusion_list_is_registered_by_check_ps1_with_a_reason_each() -> None:
    registry = _registry(_text(CHECK_PS1))
    excluded = _excluded_checks(_text(CHECK_ALL_PS1))
    assert {item["id"] for item in excluded} == {"EVAL-R01", "EVAL-R06", "STATE-X01", "EVAL-X01"}
    for item in excluded:
        assert item["id"] in registry[item["suite"]], f"{item['id']} is not registered by check.ps1 for {item['suite']}"
        assert len(item["reason"]) > 40, f"{item['id']} needs a real reason"
        assert item["kind"] in {"excluded", "conditional"}
    # -CheckIds only works for AGENT, STATE and EVAL, so only those suites can carry exclusions.
    assert {item["suite"] for item in excluded} <= {"AGENT", "STATE", "EVAL"}


def test_check_all_runs_every_suite_check_ps1_registers() -> None:
    registry = _registry(_text(CHECK_PS1))
    match = re.search(r"\$script:CheckSuites = @\(([^)]*)\)", _text(CHECK_ALL_PS1))
    assert match
    suites = re.findall(r'"([^"]+)"', match.group(1))
    assert sorted(suites) == sorted(registry), "check-all must run exactly the suites check.ps1 registers"
    # What it invokes plus what it excludes is the whole registry, no more and no less.
    excluded = {item["id"]: item["suite"] for item in _excluded_checks(_text(CHECK_ALL_PS1))}
    registered_total = {check for ids in registry.values() for check in ids}
    invoked = registered_total - set(excluded)
    assert invoked | set(excluded) == registered_total and not invoked & set(excluded)
    assert len(registered_total) == sum(len(ids) for ids in registry.values()), "a check id is registered twice"


def test_check_all_reads_the_registry_with_a_pattern_that_matches_check_ps1() -> None:
    script = _text(CHECK_ALL_PS1)
    assert "Get-RegisteredCheckIds" in script and r'\{\s*@\(([^)]*)\)\s*\}' in script
    for suite in ("STATE", "EVAL"):
        assert re.search(r'"' + suite + r'"\s*\{\s*@\(([^)]*)\)\s*\}', _text(CHECK_PS1))


def test_check_ps1_windows_behaviour_text_is_unchanged() -> None:
    text = _text(CHECK_PS1)
    assert 'Join-Path $projectRoot ".venv\\Scripts\\python.exe"' in text
    assert r"-match '^[A-Za-z]:[\\/]'" in text and r"-match '^\\\\'" in text
    assert "fixtures\\knowledge" not in text, "paths are joined without backslash literals"
    assert text.index("[System.PlatformID]::Win32NT") < text.index('$python = if ($PythonPath)')
    assert "$IsWindows" not in text.replace("no $IsWindows", ""), "Windows PowerShell 5.1 has no $IsWindows"


def test_check_ps1_clears_the_mcp_setting_before_the_first_probe_and_restores_it() -> None:
    text = _text(CHECK_PS1)
    clear = text.index("Remove-Item -LiteralPath Env:QUERYSHIELD_METADATA_TOOLS -ErrorAction SilentlyContinue\n$env:QUERYSHIELD_PROVIDER_MODE")
    assert clear < text.index('if ($Suite -eq "STATE") {\n    Invoke-StateSuite')
    summary = text[text.index("function Write-Summary"):]
    assert summary.index("Restore-MetadataToolsSetting") < summary.index("Write-SourceManifest")


# --- behaviour (PowerShell) ---------------------------------------------------------


@needs_powershell
def test_check_all_parses_without_errors() -> None:
    script = (
        "$tokens = $null; $errors = $null; "
        f"[void][System.Management.Automation.Language.Parser]::ParseFile('{CHECK_ALL_PS1}', [ref]$tokens, [ref]$errors); "
        "$errors.Count"
    )
    completed = _powershell(["-Command", script])
    assert completed.stdout.strip() == "0", completed.stdout + completed.stderr


@needs_powershell
@needs_posix
@pytest.mark.parametrize(
    ("name", "args", "reason"),
    [
        ("unknown_check_id", ["-Suite", "EVAL", "-Mode", "fake", "-Database", "postgres", "-CheckIds", "EVAL-UNKNOWN"], "unknown_CheckIds=EVAL-UNKNOWN"),
        ("missing_mode", ["-Suite", "EVAL", "-Database", "postgres", "-CheckIds", "EVAL-R01"], "EVAL_Mode_must_be_explicit"),
    ],
)
def test_the_two_inputs_eval_x02_sends_are_refused_for_their_own_reason(tmp_path, name, args, reason) -> None:
    """On Linux they used to exit 2 only because the evidence path looked relative to the drive-letter rule."""

    evidence = tmp_path / name
    completed = _powershell(["-File", str(CHECK_PS1), *args, "-EvidenceDir", str(evidence)], timeout=60)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    blocked = (evidence / "check-blocked.txt").read_text(encoding="utf-8-sig")
    assert reason in blocked
    assert "evidence_dir_must_be_absolute" not in blocked and "evidence_dir_must_be_absolute" not in completed.stdout


@needs_powershell
@needs_posix
def test_a_relative_evidence_dir_is_still_refused_on_linux() -> None:
    completed = _powershell(["-File", str(CHECK_PS1), "-Suite", "DB-SMOKE", "-EvidenceDir", "relative/path"], timeout=60)
    assert completed.returncode == 2
    assert "evidence_dir_must_be_absolute" in completed.stdout


@needs_powershell
@needs_posix
def test_default_python_on_linux_is_the_venv_bin_python(tmp_path) -> None:
    """A copy of check.ps1 in a project without .venv is blocked as python_missing; with .venv/bin/python it runs."""

    scripts = tmp_path / "project" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(CHECK_PS1, scripts / "check.ps1")
    env = {"QUERYSHIELD_DATABASE_URL": "postgresql://x@127.0.0.1:1/x"}
    blocked = tmp_path / "blocked"
    completed = _powershell(["-File", str(scripts / "check.ps1"), "-Suite", "DB-SMOKE", "-Mode", "fake", "-EvidenceDir", str(blocked)], env=env, timeout=60)
    assert completed.returncode == 2
    assert "python_missing" in (blocked / "check-blocked.txt").read_text(encoding="utf-8-sig")

    stub = _stub_python(tmp_path / "project" / ".venv" / "bin", name="python")
    assert stub.name == "python"
    ran = tmp_path / "ran"
    completed = _powershell(["-File", str(scripts / "check.ps1"), "-Suite", "DB-SMOKE", "-Mode", "fake", "-EvidenceDir", str(ran)], env=env, timeout=60)
    # It got past input validation and tried the runtime preflight (which the empty copy does not have).
    assert not (ran / "check-blocked.txt").exists(), "the .venv/bin/python default was not found"
    assert "CHECK-RUNTIME" in completed.stdout


def _stub_python(directory: Path, name: str = "stub-python") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stub = directory / name
    stub.write_text('#!/bin/sh\necho "metadata=[${QUERYSHIELD_METADATA_TOOLS-unset}]"\n', encoding="ascii")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return stub


@needs_powershell
@needs_posix
def test_m11_probes_see_no_metadata_setting_and_the_caller_gets_it_back(tmp_path) -> None:
    stub = _stub_python(tmp_path)
    evidence = tmp_path / "evidence"
    script = (
        "$env:QUERYSHIELD_METADATA_TOOLS = 'mcp'; "
        f"& '{CHECK_PS1}' -Suite DB-SMOKE -Mode fake -PythonPath '{stub}' -EvidenceDir '{evidence}' | Out-Null; "
        "'exit=' + $LASTEXITCODE; 'after=' + $env:QUERYSHIELD_METADATA_TOOLS"
    )
    completed = _powershell(["-Command", script], env={"QUERYSHIELD_DATABASE_URL": "postgresql://x@127.0.0.1:1/x"}, timeout=60)
    assert "exit=0" in completed.stdout and "after=mcp" in completed.stdout, completed.stdout + completed.stderr
    assert (evidence / "DB-SMOKE.txt").read_text(encoding="utf-8-sig").strip() == "metadata=[unset]"


@needs_powershell
@needs_posix
@pytest.mark.parametrize(
    ("suite", "check_id", "reason"),
    [
        ("STATE", "STATE-R03", "STATE-R03_requires_STATE_fake_model_boundary"),
        ("EVAL", "EVAL-FS01", "EVAL-FS01_requires_EVAL_fake_harness"),
    ],
)
def test_a_fake_only_check_in_real_mode_records_why_it_is_not_applicable(tmp_path, suite, check_id, reason) -> None:
    """The reason names the check; ``$checkId_requires_...`` used to read as one undefined variable and came out empty."""

    args = ["-Suite", suite, "-Mode", "real", "-Database", "postgres", "-CheckIds", check_id]
    completed = _powershell(["-File", str(CHECK_PS1), *args, "-PythonPath", sys.executable, "-EvidenceDir", str(tmp_path)], timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evidence = (tmp_path / f"{check_id}.txt").read_text(encoding="utf-8-sig").strip()
    assert evidence == f"not_applicable reason={reason}"


@needs_powershell
@needs_posix
def test_m11_leaves_an_unset_variable_unset(tmp_path) -> None:
    stub = _stub_python(tmp_path)
    script = (
        "Remove-Item Env:QUERYSHIELD_METADATA_TOOLS -ErrorAction SilentlyContinue; "
        f"& '{CHECK_PS1}' -Suite DB-SMOKE -Mode fake -PythonPath '{stub}' -EvidenceDir '{tmp_path / 'evidence'}' | Out-Null; "
        "'defined=' + (Test-Path Env:QUERYSHIELD_METADATA_TOOLS)"
    )
    completed = _powershell(["-Command", script], env={"QUERYSHIELD_DATABASE_URL": "postgresql://x@127.0.0.1:1/x"}, timeout=60)
    assert "defined=False" in completed.stdout, completed.stdout + completed.stderr


@needs_powershell
@needs_posix
def test_check_all_is_blocked_without_database_settings_and_prints_no_url(tmp_path) -> None:
    env = {name: "" for name in ("QUERYSHIELD_DATABASE_URL", "QUERYSHIELD_BOOTSTRAP_DATABASE_URL")}
    completed = _powershell(["-File", str(CHECK_ALL_PS1), "-EvidenceDir", str(tmp_path / "evidence"), "-NonInteractive"], env=env, timeout=60)
    assert completed.returncode == 2
    assert "configuration_missing" in completed.stdout and "postgresql://" not in completed.stdout


@needs_powershell
@needs_posix
def test_check_all_refuses_a_relative_evidence_dir() -> None:
    completed = _powershell(["-File", str(CHECK_ALL_PS1), "-EvidenceDir", "relative"], timeout=60)
    assert completed.returncode == 2 and "evidence_dir_must_be_absolute" in completed.stdout


# --- X01 applicability, the tests' own temporary directory, progress lines -------------------------


def test_acceptance_tag_names_are_read_from_the_register_not_written_in_the_script() -> None:
    text = _text(CHECK_ALL_PS1)
    assert not re.search(r"qs-w0\d-", text), "tag names come from control/evidence/upstream/accepted-assets.json"
    assert "$document.tags.PSObject.Properties" in text
    assert "safe.directory=$safeDirectory" in text, "git is told to trust this repository root only"
    if layout() != "development":
        pytest.skip(REGISTER_SKIP_REASON)
    register = json.loads((REPO_ROOT / "control" / "evidence" / "upstream" / "accepted-assets.json").read_text(encoding="utf-8"))
    assert set(register["tags"]) and all(name.startswith("qs-w0") for name in register["tags"])


def test_full_tests_get_their_own_basetemp_outside_the_evidence_dir_and_it_is_removed() -> None:
    text = _text(CHECK_ALL_PS1)
    assert '"--basetemp", $testBaseTemp' in text
    assert "[System.IO.Path]::GetTempPath()" in text and "[Guid]::NewGuid()" in text
    assert "Remove-Item -LiteralPath $testBaseTemp -Recurse -Force" in text
    assert "$testBaseTemp = Join-Path $EvidenceDir" not in text


def test_every_step_prints_a_line_before_it_starts_and_the_test_step_says_how_long() -> None:
    text = _text(CHECK_ALL_PS1)
    assert 'Write-StepStart -Name "tests" -Note "the full test suite: about 2 to 5 minutes' in text
    assert text.count("Write-StepStart -Name") >= 4
    assert 'Write-StepStart -Name $Suite' in text and 'Write-StepStart -Name $smoke.Name' in text


def _stub_git(directory: Path, exit_code_for: dict[str, int], default: int = 0) -> Path:
    """A git that records its arguments and answers per tag name."""

    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "git-calls.txt"
    branches = "".join(f'  *"refs/tags/{tag}^"*) exit {code} ;;\n' for tag, code in exit_code_for.items())
    script = f'#!/bin/sh\necho "$@" >> "{log}"\ncase "$*" in\n{branches}  *) exit {default} ;;\nesac\n'
    stub = directory / "git"
    stub.write_text(script, encoding="ascii")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return stub


def _x01_applicability(tmp_path: Path, *, tags: list[str] | None, git_dir: Path | None, path_without_git: bool = False, standalone: bool = False) -> dict:
    """Run Test-X01Applicable (extracted from check-all.ps1 with the PowerShell parser) in a scratch repository."""

    repo = tmp_path / "repo"
    if tags is not None:
        register = repo / "control" / "evidence" / "upstream" / "accepted-assets.json"
        register.parent.mkdir(parents=True, exist_ok=True)
        register.write_text(json.dumps({"tags": {name: {"commit": "0" * 40} for name in tags}}), encoding="utf-8")
    else:
        repo.mkdir(parents=True, exist_ok=True)
    project = repo / "queryshield"
    project.mkdir(parents=True, exist_ok=True)
    if standalone:
        workflow = project / ".github" / "workflows" / "queryshield-ci.yml"
        workflow.parent.mkdir(parents=True, exist_ok=True)
        workflow.write_text("name: CI\n", encoding="utf-8")
    path = os.environ["PATH"] if not path_without_git else str(Path(SHELL).parent)
    if git_dir is not None:
        path = f"{git_dir}{os.pathsep}{path}"
    script = f"""
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('{CHECK_ALL_PS1}', [ref]$tokens, [ref]$errors)
$functions = $ast.FindAll({{ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -in @('Test-X01Applicable', 'Read-JsonFile') }}, $true)
foreach ($function in $functions) {{ Invoke-Expression $function.Extent.Text }}
$repoRoot = '{repo}'
$projectRoot = '{project}'
$script:RegisterRelativePath = 'control/evidence/upstream/accepted-assets.json'
$script:StandaloneWorkflowRelativePath = '.github/workflows/queryshield-ci.yml'
$result = Test-X01Applicable
[ordered]@{{ Applicable = $result.Applicable; Failure = $result.Failure; Reason = $result.Reason }} | ConvertTo-Json -Compress
"""
    completed = _powershell(["-Command", script], env={"PATH": path}, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


@needs_powershell
@needs_posix
def test_x01_is_not_applicable_without_a_register(tmp_path) -> None:
    result = _x01_applicability(tmp_path, tags=None, git_dir=None)
    assert result["Applicable"] is False and result["Failure"] is False and "register" in result["Reason"]


@needs_powershell
@needs_posix
def test_x01_is_not_applicable_when_the_project_directory_is_itself_the_repository_root(tmp_path) -> None:
    """The standalone layout (the CI workflow is inside the project directory) never looks above itself: a register in a
    parent directory is ignored and git is not asked about tags."""

    git_dir = tmp_path / "bin"
    _stub_git(git_dir, {})
    result = _x01_applicability(tmp_path, tags=["qs-w03-accepted-20260922"], git_dir=git_dir, standalone=True)
    assert result["Applicable"] is False and result["Failure"] is False and "register" in result["Reason"]
    assert not (git_dir / "git-calls.txt").exists(), "git was not run"


@needs_powershell
@needs_posix
def test_x01_in_the_development_layout_is_unchanged_by_the_standalone_rule(tmp_path) -> None:
    git_dir = tmp_path / "bin"
    _stub_git(git_dir, {})
    result = _x01_applicability(tmp_path, tags=["qs-w03-accepted-20260922"], git_dir=git_dir)
    assert result == {"Applicable": True, "Failure": False, "Reason": ""}
    assert (git_dir / "git-calls.txt").exists()


def test_check_all_and_the_test_layout_rule_name_the_same_workflow_file() -> None:
    assert f'$script:StandaloneWorkflowRelativePath = "{WORKFLOW_RELATIVE.as_posix()}"' in _text(CHECK_ALL_PS1)


@needs_powershell
@needs_posix
def test_x01_looks_up_the_tags_the_register_lists_and_trusts_only_this_repository(tmp_path) -> None:
    git_dir = tmp_path / "bin"
    _stub_git(git_dir, {})
    result = _x01_applicability(tmp_path, tags=["qs-w03-accepted-20260922", "qs-w04-accepted-20260923"], git_dir=git_dir)
    assert result == {"Applicable": True, "Failure": False, "Reason": ""}
    calls = (git_dir / "git-calls.txt").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2 and all(f"safe.directory={tmp_path / 'repo'}" in call and "rev-parse --verify --quiet" in call for call in calls)
    assert "refs/tags/qs-w03-accepted-20260922^{commit}" in calls[0] and "refs/tags/qs-w04-accepted-20260923^{commit}" in calls[1]
    assert not any("qs-w05" in call for call in calls), "the tag names are the register's"
    assert not any(" tag --list" in call for call in calls)


@needs_powershell
@needs_posix
def test_x01_with_a_missing_listed_tag_is_not_applicable_and_names_it(tmp_path) -> None:
    git_dir = tmp_path / "bin"
    _stub_git(git_dir, {"qs-w04-accepted-20260923": 1})
    result = _x01_applicability(tmp_path, tags=["qs-w03-accepted-20260922", "qs-w04-accepted-20260923"], git_dir=git_dir)
    assert result["Applicable"] is False and result["Failure"] is False
    assert "qs-w04-accepted-20260923" in result["Reason"] and "qs-w03" not in result["Reason"]


@needs_powershell
@needs_posix
@pytest.mark.parametrize("code", [128, 2, 255])
def test_x01_with_a_git_that_cannot_run_is_a_failure_not_a_missing_tag(tmp_path, code) -> None:
    """Dubious ownership and similar make git exit 128 (not 1): that must never read as 'tag missing, not applicable'."""

    git_dir = tmp_path / "bin"
    _stub_git(git_dir, {}, default=code)
    result = _x01_applicability(tmp_path, tags=["qs-w03-accepted-20260922"], git_dir=git_dir)
    assert result["Failure"] is True and result["Applicable"] is True
    assert "could not read the repository" in result["Reason"] and str(code) in result["Reason"]
    assert "missing" not in result["Reason"]


@needs_powershell
@needs_posix
def test_x01_without_git_installed_is_a_failure(tmp_path) -> None:
    result = _x01_applicability(tmp_path, tags=["qs-w03-accepted-20260922"], git_dir=None, path_without_git=True)
    assert result["Failure"] is True and "git is not available" in result["Reason"]


@needs_powershell
@needs_posix
def test_x01_with_an_unreadable_register_is_a_failure(tmp_path) -> None:
    register = tmp_path / "repo" / "control" / "evidence" / "upstream" / "accepted-assets.json"
    register.parent.mkdir(parents=True)
    register.write_text("{not json", encoding="utf-8")
    result = _x01_applicability(tmp_path, tags=None, git_dir=None)
    assert result["Failure"] is True and "JSON" in result["Reason"]


@needs_powershell
@needs_posix
def test_the_full_tests_run_with_a_private_basetemp_and_progress_lines_come_first(tmp_path) -> None:
    """A stub interpreter shows the arguments pytest would get; the run then fails later steps, which is fine here."""

    stub = tmp_path / "stub-python"
    stub.write_text('#!/bin/sh\necho "args: $@"\n', encoding="ascii")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    evidence = tmp_path / "evidence"
    env = {"QUERYSHIELD_DATABASE_URL": "postgresql://x@127.0.0.1:1/queryshield_test", "QUERYSHIELD_BOOTSTRAP_DATABASE_URL": "postgresql://x@127.0.0.1:1/queryshield_test", "TMPDIR": str(tmp_path / "systemp")}
    (tmp_path / "systemp").mkdir()
    completed = _powershell(["-File", str(CHECK_ALL_PS1), "-EvidenceDir", str(evidence), "-PythonPath", str(stub), "-NonInteractive"], env=env, timeout=240)
    lines = completed.stdout.splitlines()
    running = next(index for index, line in enumerate(lines) if line.startswith("check_all running step=tests"))
    finished = next(index for index, line in enumerate(lines) if line.startswith("check_all step=tests"))
    assert running < finished and "minutes" in lines[running]
    recorded = (evidence / "tests" / "pytest-output.txt").read_text(encoding="utf-8-sig")
    match = re.search(r"--basetemp (\S+)", recorded)
    assert match, recorded
    basetemp = Path(match.group(1))
    assert basetemp.parent == tmp_path / "systemp" and basetemp.name.startswith("qs-check-all-")
    assert evidence not in basetemp.parents, "not inside the evidence directory"
    assert not basetemp.exists(), "removed after the run"
    assert any(line.startswith("check_all running step=EVAL") for line in lines) and any(line.startswith("check_all running step=smoke-http") for line in lines)


def test_an_x01_failure_becomes_a_failed_step_of_its_own_and_the_x01_checks_still_run() -> None:
    text = _text(CHECK_ALL_PS1)
    assert re.search(r'if \(\$x01\.Failure\) \{\s+Add-Step -Name "X01-applicability" -Status "fail"', text)
    # Applicable stays true on a failure, so Invoke-CheckSuite keeps X01 in the run (its own evidence).
    assert text.count("Applicable = $true; Failure = $true") >= 4
    assert "Applicable = $false; Failure = $true" not in text
