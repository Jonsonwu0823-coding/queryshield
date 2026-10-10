"""compose-real-demo.ps1 through a model gateway, run against a fake docker.

The gateway mode starts the stack with compose.yaml and compose.model-gateway.yaml, runs only the demo
questions and copies only their summary, so every run the gateway billed is one run_id in that summary.
Wrong arguments and a missing network stop it before any container starts; the key stays in the process.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_container_files import PROJECT_ROOT, REAL_DEMO_PS1, SUMMARY_JSON, _SHELL, _fake_docker, needs_posix, needs_powershell

FILES = "compose -f compose.yaml -f compose.model-gateway.yaml "
GATEWAY_URL = "http://model-gateway.test:9000/v1"
NETWORK = "shared-net.test_1"
# Recognisable and without the sk- prefix a Bailian key has: it must never show up in output or docker's arguments.
FAKE_KEY = "gw-issued-7c41e9d2"
GATEWAY_NAMES = ("GATEWAY_BASE_URL", "GATEWAY_NETWORK")
WATCHED = tuple(f"QUERYSHIELD_{name}" for name in (*GATEWAY_NAMES, "MODEL_API_KEY", "EMBEDDING_API_KEY", "PROVIDER_MODE", "MODEL_NAME", "AGENT_PROFILE"))
# Any Bailian address: the gateway mode refuses the combination before looking at it.
BAILIAN_URL = "https://bailian.example/compatible-mode/v1"


def _run(tmp_path: Path, arguments: str, *, key: str | None = FAKE_KEY, evidence: bool = True, previous: dict | None = None, **docker):
    """Run the script's copy in a scratch project next to a fake docker, from a PowerShell session that then
    reports which watched variables differ from before the run (by name only, never the value).

    CHANGED compares values with unset and empty alike; LEFT_DEFINED is the strict check for a variable that
    did not exist before the run but exists (even empty) after it.
    """

    project = tmp_path / "queryshield"
    (project / "scripts").mkdir(parents=True)
    shutil.copyfile(REAL_DEMO_PS1, project / "scripts" / "compose-real-demo.ps1")
    (project / ".env").write_text("", encoding="utf-8")
    _, log = _fake_docker(tmp_path / "bin", **docker)
    target = tmp_path / "evidence"
    if evidence:
        arguments += f" -EvidenceDir '{target}'"
    watched = ", ".join(f"'{name}'" for name in WATCHED)
    command = (
        f"$names = @({watched}); $before = @{{}}; foreach ($n in $names) {{ $before[$n] = [Environment]::GetEnvironmentVariable($n, 'Process') }}; "
        f"& '{project / 'scripts' / 'compose-real-demo.ps1'}' {arguments}; $code = $LASTEXITCODE; "
        "foreach ($n in $names) { if ([string][Environment]::GetEnvironmentVariable($n, 'Process') -ne [string]$before[$n]) { Write-Output ('CHANGED ' + $n) } }; "
        "foreach ($n in $names) { if ($null -eq $before[$n] -and (Test-Path -LiteralPath ('Env:' + $n))) { Write-Output ('LEFT_DEFINED ' + $n) } }; "
        "exit $code"
    )
    env = {name: value for name, value in os.environ.items() if not name.startswith("QUERYSHIELD_")}
    env.update({"PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}", "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT": "1"})
    if key is not None:
        env["QUERYSHIELD_MODEL_API_KEY"] = key
    env.update(previous or {})
    completed = subprocess.run(
        [_SHELL, "-NoProfile", "-Command", command],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False, stdin=subprocess.DEVNULL,
    )
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return completed, (target if evidence else project / "evidence"), calls


def _compose_calls(calls: list[str]) -> list[str]:
    return [call for call in calls if call.startswith("compose")]


def _compose_environment(tmp_path: Path) -> dict:
    """The QUERYSHIELD_ variables (keys aside) docker compose up was given."""

    lines = (tmp_path / "bin" / "up-env.txt").read_text(encoding="utf-8").splitlines()
    return dict(line.split("=", 1) for line in lines)


GOOD = f"-GatewayBaseUrl '{GATEWAY_URL}/' -GatewayNetwork '{NETWORK}'"


@needs_powershell
@needs_posix
def test_the_gateway_mode_names_both_files_on_every_compose_call_and_runs_only_the_demo(tmp_path) -> None:
    previous = {"QUERYSHIELD_GATEWAY_NETWORK": "network-before-the-run", "QUERYSHIELD_GATEWAY_BASE_URL": "http://before.test/v1"}
    completed, evidence, calls = _run(tmp_path, GOOD, previous=previous)
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    compose = _compose_calls(calls)
    assert compose and all(call.startswith(FILES) for call in compose), compose
    assert compose[0] == FILES + "up -d --build" and compose[-1] == FILES + "down", compose
    assert calls.index(f"network inspect {NETWORK}") < calls.index(compose[0]), "the network is checked before anything starts"
    assert any("scripts/demo_run.py --mode real --base-url http://127.0.0.1:8000 --model-protocol json" in call for call in compose)
    assert not any("demo_walkthrough.py" in call for call in calls)
    assert sorted(path.name for path in evidence.iterdir()) == ["demo-summary.json"]
    assert (evidence / "demo-summary.json").read_text(encoding="utf-8").strip() == SUMMARY_JSON
    # docker compose got the key, the gateway variables and the same model profile as the Bailian mode.
    assert (tmp_path / "bin" / "up-key.txt").read_text(encoding="utf-8").strip() == FAKE_KEY
    given = _compose_environment(tmp_path)
    assert {name: given.get(name) for name in (
        "QUERYSHIELD_GATEWAY_BASE_URL", "QUERYSHIELD_GATEWAY_NETWORK", "QUERYSHIELD_PROVIDER_MODE", "QUERYSHIELD_MODEL_NAME",
        "QUERYSHIELD_EMBEDDING_MODEL_NAME", "QUERYSHIELD_EMBEDDING_MODEL_REVISION", "QUERYSHIELD_EMBEDDING_DIMENSIONS",
    )} == {
        "QUERYSHIELD_GATEWAY_BASE_URL": GATEWAY_URL, "QUERYSHIELD_GATEWAY_NETWORK": NETWORK, "QUERYSHIELD_PROVIDER_MODE": "real",
        "QUERYSHIELD_MODEL_NAME": "qwen-plus", "QUERYSHIELD_EMBEDDING_MODEL_NAME": "text-embedding-v4",
        "QUERYSHIELD_EMBEDDING_MODEL_REVISION": "text-embedding-v4", "QUERYSHIELD_EMBEDDING_DIMENSIONS": "1024",
    }
    assert FAKE_KEY not in output and not any(FAKE_KEY in call for call in calls)
    assert "CHANGED" not in output, "every variable the script set is restored when it ends"


@needs_powershell
@needs_posix
def test_variables_the_session_did_not_have_are_removed_again_not_left_empty(tmp_path) -> None:
    """Newer PowerShell leaves a variable defined (empty) when it is set to "" or $null; the script must remove it."""

    completed, _, _ = _run(tmp_path, GOOD)
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "LEFT_DEFINED" not in output, output


@needs_powershell
@needs_posix
def test_an_https_gateway_address_reaches_compose_without_its_default_port(tmp_path) -> None:
    completed, _, _ = _run(tmp_path, f"-GatewayBaseUrl 'https://model-gateway.test:443/v1/' -GatewayNetwork '{NETWORK}'")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _compose_environment(tmp_path)["QUERYSHIELD_GATEWAY_BASE_URL"] == "https://model-gateway.test/v1"


@needs_powershell
@needs_posix
def test_the_gateway_mode_writes_its_default_evidence_inside_the_project(tmp_path) -> None:
    completed, _, _ = _run(tmp_path, GOOD, evidence=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    runs = list((tmp_path / "queryshield" / "evidence").glob("compose-gateway-*"))
    assert len(runs) == 1 and sorted(path.name for path in runs[0].iterdir()) == ["demo-summary.json"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["bin", "queryshield"]


# What the script says for each kind of wrong argument.
BOTH = "needs both -GatewayBaseUrl and -GatewayNetwork"
OTHER_MODE = "does not take -BailianBaseUrl or -FakeDryRun"
ADDRESS = "address of the gateway on the shared network, ending in /v1"
BAD_NETWORK = "network name starts with a letter or digit"
NO_KEY = "API Key is not available"
BAD_KEY = "API Key format is invalid"
WRONG_ARGUMENTS = {
    "address only": (f"-GatewayBaseUrl '{GATEWAY_URL}'", FAKE_KEY, BOTH),
    "network only": (f"-GatewayNetwork '{NETWORK}'", FAKE_KEY, BOTH),
    "with the Bailian address": (f"{GOOD} -BailianBaseUrl '{BAILIAN_URL}'", FAKE_KEY, OTHER_MODE),
    "with the Fake dry run": (f"{GOOD} -FakeDryRun", FAKE_KEY, OTHER_MODE),
    "not an absolute address": (f"-GatewayBaseUrl 'gateway v1' -GatewayNetwork '{NETWORK}'", FAKE_KEY, ADDRESS),
    "not http or https": (f"-GatewayBaseUrl 'ftp://model-gateway.test/v1' -GatewayNetwork '{NETWORK}'", FAKE_KEY, ADDRESS),
    # In two pieces, so that the public export's scan does not read it as a credential.
    "user and password": ("-GatewayBaseUrl 'http://user:" + f"pass@model-gateway.test:9000/v1' -GatewayNetwork '{NETWORK}'", FAKE_KEY, ADDRESS),
    "user only": (f"-GatewayBaseUrl 'http://user@model-gateway.test:9000/v1' -GatewayNetwork '{NETWORK}'", FAKE_KEY, ADDRESS),
    "query": (f"-GatewayBaseUrl '{GATEWAY_URL}?route=a' -GatewayNetwork '{NETWORK}'", FAKE_KEY, ADDRESS),
    "fragment": (f"-GatewayBaseUrl '{GATEWAY_URL}#top' -GatewayNetwork '{NETWORK}'", FAKE_KEY, ADDRESS),
    "path is not /v1": (f"-GatewayBaseUrl 'http://model-gateway.test:9000/v2' -GatewayNetwork '{NETWORK}'", FAKE_KEY, ADDRESS),
    "no path": (f"-GatewayBaseUrl 'http://model-gateway.test:9000' -GatewayNetwork '{NETWORK}'", FAKE_KEY, ADDRESS),
    "network starts with a dash": (f"-GatewayBaseUrl '{GATEWAY_URL}' -GatewayNetwork '-net'", FAKE_KEY, BAD_NETWORK),
    "network has a space": (f"-GatewayBaseUrl '{GATEWAY_URL}' -GatewayNetwork 'shared net'", FAKE_KEY, BAD_NETWORK),
    "network has a slash": (f"-GatewayBaseUrl '{GATEWAY_URL}' -GatewayNetwork 'shared/net'", FAKE_KEY, BAD_NETWORK),
    # PowerShell's "`n": a regular expression ending in $ would let the trailing newline through.
    "network ends with a newline": (f"-GatewayBaseUrl '{GATEWAY_URL}' -GatewayNetwork \"shared-net`n\"", FAKE_KEY, BAD_NETWORK),
    "no key and non-interactive": (f"{GOOD} -NonInteractive", None, NO_KEY),
    "key with a space": (GOOD, "gw-issued 7c41", BAD_KEY),
    "key with an asterisk": (GOOD, "gw-issued-****", BAD_KEY),
}


@needs_powershell
@needs_posix
@pytest.mark.parametrize("case", sorted(WRONG_ARGUMENTS))
def test_wrong_gateway_arguments_exit_2_before_any_docker_call(tmp_path, case) -> None:
    arguments, key, message = WRONG_ARGUMENTS[case]
    completed, evidence, calls = _run(tmp_path, arguments, key=key)
    output = completed.stdout + completed.stderr
    assert completed.returncode == 2, output
    assert message in completed.stdout, output
    assert calls == [], calls
    assert not evidence.exists()
    assert key is None or key not in output
    assert "CHANGED" not in output


@needs_powershell
@needs_posix
def test_a_missing_gateway_network_exits_2_without_starting_the_stack(tmp_path) -> None:
    completed, _, calls = _run(tmp_path, GOOD, network_exit=1)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "Start the model gateway first" in completed.stdout
    assert calls == [f"network inspect {NETWORK}"], calls


@needs_powershell
@needs_posix
def test_a_demo_failure_through_the_gateway_gives_exit_code_1_and_still_copies_the_summary_and_stops(tmp_path) -> None:
    completed, evidence, calls = _run(tmp_path, GOOD, exec_exit=1)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert sorted(path.name for path in evidence.iterdir()) == ["demo-summary.json"]
    assert _compose_calls(calls)[-1] == FILES + "down"


def _script() -> str:
    return REAL_DEMO_PS1.read_text(encoding="utf-8-sig")


def test_the_gateway_variables_are_restored_like_the_others() -> None:
    text = _script()
    touched = text[text.index("$touchedNames = @("): text.index("\n)", text.index("$touchedNames = @("))]
    for name in GATEWAY_NAMES:
        assert f'"QUERYSHIELD_{name}"' in touched, name


def test_the_operations_doc_writes_the_gateway_command_with_placeholders() -> None:
    assert re.search(r"-GatewayBaseUrl 'http://<[^>]+>/v1' -GatewayNetwork '<[^>]+>'", (PROJECT_ROOT / "docs" / "operations.md").read_text(encoding="utf-8"))
