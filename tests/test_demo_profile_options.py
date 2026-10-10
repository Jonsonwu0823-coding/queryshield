"""compose-real-demo.ps1 -Profile / -Questions: passed on only when given, so the default command line is unchanged.

Runs the script's copy next to the fake docker of tests/test_container_files.py (-FakeDryRun).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from test_container_files import REAL_DEMO_PS1, _SHELL, _fake_docker, needs_posix, needs_powershell

DEMO_CALL = "scripts/demo_run.py --mode fake --base-url http://127.0.0.1:8000 --model-protocol json --evidence-dir /tmp/demo-run"


def _run(tmp_path: Path, *arguments: str, session_profile: str | None = None) -> tuple[list[str], list[str]]:
    project = tmp_path / "queryshield"
    (project / "scripts").mkdir(parents=True)
    shutil.copyfile(REAL_DEMO_PS1, project / "scripts" / "compose-real-demo.ps1")
    (project / ".env").write_text("", encoding="utf-8")
    _, log = _fake_docker(tmp_path / "bin")
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}", "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT": "1"}
    env.pop("QUERYSHIELD_AGENT_PROFILE", None)
    if session_profile is not None:
        env["QUERYSHIELD_AGENT_PROFILE"] = session_profile
    command = [_SHELL, "-NoProfile", "-File", str(project / "scripts" / "compose-real-demo.ps1"), "-FakeDryRun", "-EvidenceDir", str(tmp_path / "evidence"), *arguments]
    completed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    up_env = (tmp_path / "bin" / "up-env.txt").read_text(encoding="utf-8").splitlines()
    return log.read_text(encoding="utf-8").splitlines(), up_env


def _demo_call(calls: list[str]) -> str:
    (call,) = [call for call in calls if "scripts/demo_run.py" in call]
    return call


@needs_powershell
@needs_posix
def test_without_the_options_the_demo_command_is_unchanged_and_the_app_runs_its_default(tmp_path) -> None:
    calls, up_env = _run(tmp_path, session_profile="b0")
    assert _demo_call(calls).endswith(DEMO_CALL)
    # A profile left in the session does not reach the app.
    assert not any(line.startswith("QUERYSHIELD_AGENT_PROFILE=") for line in up_env)


@needs_powershell
@needs_posix
def test_the_options_reach_the_app_and_the_demo_run(tmp_path) -> None:
    calls, up_env = _run(tmp_path, "-Profile", "b2", "-Questions", "composite")
    assert _demo_call(calls).endswith(DEMO_CALL + " --profile b2 --questions composite")
    assert "QUERYSHIELD_AGENT_PROFILE=b2" in up_env
