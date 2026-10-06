"""Static checks of the container files and the CI workflow (no Docker needed).

They encode what the independent review looks for: pinned images and actions, no credential
literals anywhere, nothing but the service published to the host, the evaluation data and
tests kept out of the image, and the files agreeing with each other.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re

import pytest
import yaml

from repo_layout import STANDALONE_WORKFLOW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
COMPOSE = PROJECT_ROOT / "compose.yaml"
OVERRIDE = PROJECT_ROOT / "deploy" / "ci-tmp-volume.compose.yaml"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "queryshield-ci.yml"
FALLBACK_WORKFLOW = PROJECT_ROOT / "deploy" / "queryshield-ci.yml"

SECRET_NAME = re.compile(r"(PASSWORD|TOKEN|SECRET|API_KEY)", re.IGNORECASE)
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}")


def _workflow_path() -> Path:
    # The workflow lives under .github/workflows of the repository root: inside this directory when it is
    # the repository root itself, one level up in the development repository.  If pushing it there was
    # refused it is kept in deploy/.
    if STANDALONE_WORKFLOW.exists():
        return STANDALONE_WORKFLOW
    return WORKFLOW if WORKFLOW.exists() else FALLBACK_WORKFLOW


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _code(path: Path) -> str:
    """The file without comment lines."""

    return "\n".join(line for line in _text(path).splitlines() if not line.lstrip().startswith("#"))


class _ComposeLoader(yaml.SafeLoader):
    """Compose's !reset tag is not standard YAML."""


_ComposeLoader.add_constructor("!reset", lambda loader, node: [])


def _compose() -> dict:
    return yaml.safe_load(_text(COMPOSE))


def _is_reference_or_empty(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip().strip("\"'")
    return text == "" or text.startswith("$") or text.startswith("<") or text.startswith("${{")


# --- Dockerfile and .dockerignore ------------------------------------------------------------------


def test_dockerfile_pins_the_base_image_by_tag_and_digest_and_runs_unprivileged() -> None:
    text = _code(DOCKERFILE)
    from_lines = [line for line in text.splitlines() if line.startswith("FROM ")]
    assert len(from_lines) == 1 and "python:3.14.3-slim-bookworm" in from_lines[0] and DIGEST.search(from_lines[0])
    users = [line.split()[1] for line in text.splitlines() if line.startswith("USER ")]
    assert users and users[-1] not in {"root", "0"}
    assert "HEALTHCHECK" in text and "curl" not in text.lower() and "urllib.request" in text
    assert "--no-deps -e ." in text and "requirements.lock" in text
    assert "0.0.0.0" in text and "--workers" not in text


def test_dockerfile_copies_an_explicit_list_and_never_the_evaluation_data_or_tests() -> None:
    text = _text(DOCKERFILE)
    copies = [line.split(None, 1)[1] for line in text.replace("\\\n", " ").splitlines() if line.startswith("COPY ")]
    assert copies and not any(line.strip().startswith(". ") for line in copies), "no COPY . : the context is never copied wholesale"
    assert "ADD " not in text
    joined = " ".join(copies)
    for kept_out in ("evals", "tests", "docs", ".env", "README"):
        assert kept_out not in joined, f"{kept_out} must not be copied into the image"
    for needed in ("src", "fixtures", "migrations"):
        assert re.search(rf"(^|\s){needed}\s", joined + " ")


def test_every_script_the_image_needs_is_copied() -> None:
    copied_line = next(line for line in _text(DOCKERFILE).replace("\\\n", " ").splitlines() if line.startswith("COPY scripts/"))
    copied = {Path(item).name for item in copied_line.split() if item.startswith("scripts/")}
    for name in copied:
        assert (PROJECT_ROOT / "scripts" / name).is_file(), f"scripts/{name} listed in the Dockerfile does not exist"
    # Whatever these entry points import from scripts/ must be in the image too.
    pending, seen = ["setup_databases.py", "demo_walkthrough.py", "demo_run.py", "new_env.py"], set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        assert name in copied, f"scripts/{name} is needed by the image's scripts but is not copied"
        tree = ast.parse((PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "scripts":
                pending.extend(f"{alias.name}.py" for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("scripts."):
                pending.append(node.module.split(".", 1)[1] + ".py")


def test_dockerignore_keeps_credentials_evaluation_data_and_state_out_of_the_context() -> None:
    lines = [line.strip() for line in _text(DOCKERIGNORE).splitlines() if line.strip() and not line.startswith("#")]
    for required in (".env", ".env.*", "!.env.example", ".venv", ".git", "evals", "tests", "*.sqlite3", "evidence", "**/__pycache__"):
        assert required in lines, f".dockerignore must list {required}"
    assert lines.index(".env.*") < lines.index("!.env.example")


# --- compose.yaml -----------------------------------------------------------------------------------


def test_only_the_service_is_published_and_only_on_loopback() -> None:
    services = _compose()["services"]
    assert set(services) == {"db", "setup", "app"}
    assert "ports" not in services["db"] and "ports" not in services["setup"]
    assert len(services["app"]["ports"]) == 1 and services["app"]["ports"][0].startswith("127.0.0.1:")
    assert services["app"]["ports"][0].endswith(":8000") and "${QUERYSHIELD_HTTP_PORT:-8000}" in services["app"]["ports"][0]


def test_images_are_pinned_and_the_service_is_locked_down() -> None:
    services = _compose()["services"]
    assert "postgres:16." in services["db"]["image"] and DIGEST.search(services["db"]["image"])
    app = services["app"]
    assert app["init"] is True and app["read_only"] is True and "/tmp" in app["tmpfs"]
    assert float(str(app["stop_grace_period"]).rstrip("s")) >= 30
    assert app["depends_on"]["setup"]["condition"] == "service_completed_successfully"
    assert services["setup"]["depends_on"]["db"]["condition"] == "service_healthy" and services["setup"]["restart"] == "no"
    assert services["db"]["healthcheck"]["test"][-1].startswith("pg_isready")
    assert any(volume.startswith("qs_state:") for volume in app["volumes"])


def test_the_app_uses_the_readonly_role_on_the_named_demo_database_and_the_demo_setting() -> None:
    env = _compose()["services"]["app"]["environment"]
    assert env["QUERYSHIELD_DATABASE_URL"].startswith("postgresql://queryshield_ro:${QUERYSHIELD_RO_PASSWORD:?")
    assert env["QUERYSHIELD_DATABASE_URL"].endswith("@db:5432/queryshield_demo")
    assert env["QUERYSHIELD_DEMO_DATASET"] == "commerce-demo-v1"
    assert env["QUERYSHIELD_PROVIDER_MODE"] == "${QUERYSHIELD_PROVIDER_MODE:-fake}"
    assert env["QUERYSHIELD_STATE_STORE_PATH"].startswith("/var/lib/queryshield/") and env["QUERYSHIELD_CALL_STORE_PATH"].startswith("/var/lib/queryshield/")
    # Model, embedding, MCP and output-limit settings are passed through from the shell only (null = not set).
    for name in (
        "QUERYSHIELD_MODEL_BASE_URL", "QUERYSHIELD_MODEL_API_KEY", "QUERYSHIELD_MODEL_NAME", "QUERYSHIELD_MODEL_MAX_TOKENS",
        "QUERYSHIELD_EMBEDDING_BASE_URL", "QUERYSHIELD_EMBEDDING_API_KEY", "QUERYSHIELD_EMBEDDING_MODEL_NAME",
        "QUERYSHIELD_EMBEDDING_MODEL_REVISION", "QUERYSHIELD_EMBEDDING_DIMENSIONS", "QUERYSHIELD_METADATA_TOOLS",
    ):
        assert name in env and env[name] is None, f"{name} must be passed through, not given a value"


def test_required_variables_stop_compose_with_their_name_and_match_the_env_example_and_generator() -> None:
    from scripts import new_env

    required = set(re.findall(r"\$\{([A-Z_]+):\?", _text(COMPOSE)))
    assert required == set(new_env.PASSWORD_VARIABLES) | set(new_env.TOKEN_VARIABLES)
    example = {line.partition("=")[0] for line in _text(ENV_EXAMPLE).splitlines() if "=" in line and not line.startswith("#")}
    assert example == required
    assert all(f"${{{name}:?{name} is not set" in _text(COMPOSE) for name in required), "each message names its variable"
    plain = set(re.findall(r"\$\{([A-Z_]+)\}", _text(COMPOSE)))
    assert not plain & required, f"{sorted(plain & required)} are used without the :? check somewhere"


def test_no_credential_literal_in_the_container_files_or_the_workflow() -> None:
    compose = _compose()
    for service in compose["services"].values():
        for key, value in (service.get("environment") or {}).items():
            if SECRET_NAME.search(key):
                assert _is_reference_or_empty(value), f"{key} must come from the environment, not a literal"
    for url in (value for service in compose["services"].values() for value in (service.get("environment") or {}).values() if isinstance(value, str) and "://" in value):
        userinfo = re.match(r"[a-z]+://[^:@/]+:([^@]*)@", url)
        assert userinfo and userinfo.group(1).startswith("${"), "a database URL takes its password from a variable"
    for path in (DOCKERFILE, COMPOSE, OVERRIDE, _workflow_path()):
        for number, line in enumerate(_text(path).splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            line = re.sub(r"\$\{[^}]*\}", "${}", line)  # a ${VAR...} reference, whatever its message says
            match = re.search(r"([A-Za-z_]*(?:PASSWORD|TOKEN|SECRET|API_KEY)[A-Za-z_]*)\s*[:=]\s*(\S+)", line)
            if match:
                assert _is_reference_or_empty(match.group(2)), f"{path.name}:{number} sets {match.group(1)} to a literal"
    example_values = [line.partition("=")[2] for line in _text(ENV_EXAMPLE).splitlines() if "=" in line and not line.startswith("#")]
    assert all(value == "<generated>" for value in example_values)


def test_the_ci_override_moves_tmp_to_a_volume() -> None:
    app = yaml.load(_text(OVERRIDE), Loader=_ComposeLoader)["services"]["app"]
    assert "!reset" in _text(OVERRIDE) and any(volume.endswith(":/tmp") for volume in app["volumes"])
    assert not (OVERRIDE.parent / "queryshield-ci.yml").exists() or _workflow_path() == FALLBACK_WORKFLOW


# --- the workflow --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(_text(_workflow_path()))


def test_workflow_triggers_permissions_and_concurrency(workflow) -> None:
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"pull_request", "push"} and triggers["push"]["branches"] == ["main"]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is True and "github.ref" in workflow["concurrency"]["group"]
    assert set(workflow["jobs"]) == {"checks", "compose"}
    assert workflow["jobs"]["checks"]["timeout-minutes"] == 15 and workflow["jobs"]["compose"]["timeout-minutes"] == 10  # measured: about 5 and 2 minutes


def test_actions_are_pinned_to_full_commit_shas_with_the_version_in_a_comment() -> None:
    text = _text(_workflow_path())
    uses = re.findall(r"^\s*(?:-\s*)?uses:\s*(\S+)(.*)$", text, re.MULTILINE)
    assert uses
    for target, rest in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", target), f"{target} must be pinned to a full commit SHA"
        assert re.search(r"#\s*v\d", rest), f"{target} needs its version in a comment"
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in text
    assert "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97" in text
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in text


def test_workflow_uses_no_repository_secret_and_masks_what_it_generates() -> None:
    text = _code(_workflow_path())
    # GitHub's secrets context (Python's own `secrets` module is what generates the credentials).
    assert not re.search(r"\$\{\{[^}]*\bsecrets\b", text) and "GITHUB_TOKEN" not in text
    assert text.count("::add-mask::") >= 2
    assert "token_urlsafe" in text and "new_env.py" in text


def test_workflow_names_the_project_directory_once() -> None:
    text = _text(_workflow_path())
    if _workflow_path() == STANDALONE_WORKFLOW:
        # This directory is the repository root: the one place that names the project directory says ".".
        assert len(re.findall(r"^\s+working-directory: \.\s*$", text, re.MULTILINE)) == 1
        assert "working-directory: queryshield" not in text
    else:
        assert text.count("working-directory: queryshield") == 1
    assert not re.findall(r"\bqueryshield/", text), "no other path mentions the project directory"


def test_checks_job_fetches_history_and_tags_creates_both_databases_and_uses_the_shared_entry(workflow) -> None:
    steps = workflow["jobs"]["checks"]["steps"]
    checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout"))
    assert checkout["with"] == {"fetch-depth": 0, "fetch-tags": True}
    runs = "\n".join(step.get("run", "") for step in steps)
    assert "setup_databases.py --test --demo" in runs and "check-all.ps1" in runs and "pwsh" in runs
    assert "postgres:16." in runs and DIGEST.search(runs) and "127.0.0.1:" in runs
    assert "python -m venv .venv" in runs and "--no-deps -e ." in runs


def test_artifacts_never_include_raw_records(workflow) -> None:
    uploads = [step for job in workflow["jobs"].values() for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact")]
    assert len(uploads) == 2
    for step in uploads:
        paths = step["with"]["path"].split("\n")
        assert all("*-raw.json" not in line or line.lstrip().startswith("!") for line in paths)
        assert not any(line.rstrip("/").endswith("qs-evidence") for line in paths), "a whole evidence directory is never uploaded"
    checks_paths = uploads[0]["with"]["path"]
    assert "!${{ runner.temp }}/qs-evidence/**/*-raw.json" in checks_paths


def test_compose_job_runs_both_demo_passes_the_tmp_volume_checks_and_always_cleans_up(workflow) -> None:
    steps = workflow["jobs"]["compose"]["steps"]
    runs = "\n".join(step.get("run", "") for step in steps)
    for expected in (
        "new_env.py", "docker compose up -d --build", "demo_walkthrough.py --mode fake --metadata-tools local",
        "demo_walkthrough.py --mode fake --metadata-tools mcp", "demo_run.py --mode fake --base-url", "--no-deps app",
        "queryshield-mcp-index-", "Application shutdown complete", "ExitCode", "docker compose up -d\n",
    ):
        assert expected in runs, expected
    assert '"$exit_code" = "143"' in runs, "143 is a normal stop, 137 is not"
    accepted = [line for line in runs.splitlines() if 'exit_code" = ' in line]
    assert accepted and not any("137" in line for line in accepted), "an exit code of 137 (killed after the grace period) must never be accepted"
    assert "--wait" not in runs, "a one-shot service is checked explicitly instead"
    last = steps[-1]
    assert last.get("if") == "always()" and "down -v" in last["run"]
    assert any(step.get("if") == "failure()" and "logs" in step.get("run", "") for step in steps)


# --- the local real-model script -----------------------------------------------------------------------


def test_compose_real_demo_script_keeps_keys_in_the_process_and_has_a_bom_for_its_non_ascii_path() -> None:
    path = PROJECT_ROOT / "scripts" / "compose-real-demo.ps1"
    data = path.read_bytes()
    text = data.decode("utf-8-sig")
    assert b"\r" not in data
    if any(ord(char) > 127 for char in text):
        assert data.startswith(b"\xef\xbb\xbf"), "a .ps1 with non-ASCII characters needs a BOM for Windows PowerShell 5.1"
    assert "-AsSecureString" in text and "SetEnvironmentVariable($name, $previousValues[$name]" in text
    assert "Set-Content" not in text and "Out-File" not in text, "no key is ever written to a file"
    assert re.search(r"sk-[A-Za-z0-9]{6,}", text) is None
    assert "demo_run.py" in text and "--base-url" in text and "demo_walkthrough.py" in text
    assert '"compose", "down"' in text and '"-v"' not in text.split('"compose", "down"')[1].split("\n")[0]
    assert "demo-raw.json" in text and "raw.json" not in text.split("$copies = @(")[1].split(")\n    foreach")[0]


# --- the operations document ------------------------------------------------------------------------------


def test_operations_doc_covers_the_limits_the_variables_and_does_not_link_control() -> None:
    text = _text(PROJECT_ROOT / "docs" / "operations.md")
    assert "control/" not in text and "../control" not in text, "the public repository has no control/ directory"
    for tag in ("K6", "H7", "H2", "G6", "N6", "D-2", "M11"):
        assert tag in text, f"limitation {tag} is missing"
    for name in re.findall(r"\$\{([A-Z_]+):\?", _text(COMPOSE)):
        assert name in text, f"{name} is not described"
    for name in _compose()["services"]["app"]["environment"]:
        assert name in text, f"{name} is not in the environment table"
    for command in ("docker compose down -v", "docker compose up -d --no-deps app", "new_env.py", "demo_walkthrough.py", "check-all.ps1"):
        assert command in text
    # Every code location it cites exists.
    for symbol, relative in (
        ("_approval_lock", "src/queryshield/approval/service.py"), ("state_path_from_env", "src/queryshield/approval/service.py"),
        ("get_call_store", "src/queryshield/api/main.py"), ("cleanup_index_dir", "src/queryshield/mcp_metadata/launch.py"),
        ("check_demo_pairing", "src/queryshield/db/readonly.py"), ("database_names_from_url", "src/queryshield/db/readonly.py"),
        ("QUERYSHIELD_MODEL_MAX_TOKENS", "src/queryshield/providers/openai_compatible.py"),
    ):
        assert symbol in text and symbol in _text(PROJECT_ROOT / relative), f"{symbol} is not in {relative}"


# --- compose-real-demo.ps1 run against a fake docker (R2) ---------------------------------------------------

REAL_DEMO_PS1 = PROJECT_ROOT / "scripts" / "compose-real-demo.ps1"
_SHELL = __import__("shutil").which("pwsh") or __import__("shutil").which("powershell")
needs_powershell = pytest.mark.skipif(_SHELL is None, reason="needs PowerShell (pwsh); GitHub's Linux runners have it")
needs_posix = pytest.mark.skipif(__import__("sys").platform == "win32", reason="the fake docker is a shell script")

SUMMARY_JSON = '{"status": "pass", "hard_failures": [], "known_gaps": []}'


def _fake_docker(directory: Path, *, up_exit: int = 0, exec_exit: int = 0) -> tuple[Path, Path]:
    """A docker whose `compose up` prints BuildKit-style build-log lines on STANDARD OUTPUT, then answers per sub-command."""

    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "docker-calls.txt"
    script = f"""#!/bin/sh
echo "$@" >> "{log}"
case "$1 $2" in
  "compose up")
    echo "#1 [internal] load build definition from Dockerfile"
    echo "#2 [internal] load metadata for docker.io/library/python"
    echo "#3 DONE 0.0s"
    exit {up_exit} ;;
  "compose down") exit 0 ;;
  "compose ps") echo "abc123container"; exit 0 ;;
  "compose exec")
    case "$*" in
      *" cat "*) echo '{SUMMARY_JSON}'; exit 0 ;;
      *) exit {exec_exit} ;;
    esac ;;
esac
case "$1" in
  inspect) echo "healthy"; exit 0 ;;
esac
exit 0
"""
    fake = directory / "docker"
    fake.write_text(script, encoding="ascii")
    fake.chmod(fake.stat().st_mode | 0o111)
    return fake, log


def _run_real_demo(tmp_path: Path, *, up_exit: int = 0, exec_exit: int = 0):
    """The script's copy sits in a scratch project (with an empty .env, so nothing is generated) next to a fake docker."""

    import os
    import shutil
    import subprocess

    project = tmp_path / "queryshield"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(REAL_DEMO_PS1, scripts / "compose-real-demo.ps1")
    (project / ".env").write_text("", encoding="utf-8")
    _, log = _fake_docker(tmp_path / "bin", up_exit=up_exit, exec_exit=exec_exit)
    evidence = tmp_path / "evidence"
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}", "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT": "1"}
    completed = subprocess.run(
        [_SHELL, "-NoProfile", "-File", str(scripts / "compose-real-demo.ps1"), "-FakeDryRun", "-EvidenceDir", str(evidence)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    return completed, evidence, log


@needs_powershell
@needs_posix
def test_the_real_model_script_does_not_mistake_build_output_for_a_failed_compose_up(tmp_path) -> None:
    """docker compose up --build writes its build log to stdout; a function that returns that output together with the
    exit code turned `(Invoke-Docker ...) -ne 0` into a non-empty array, so a successful up was reported as failed."""

    before = (PROJECT_ROOT / ".env").exists()
    completed, evidence, log = _run_real_demo(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "compose up failed" not in completed.stdout and "sanitized failure" not in completed.stdout
    assert "load build definition" in completed.stdout, "the build log still reaches the screen"
    for name in ("demo-summary.json", "walkthrough-summary.json"):
        assert (evidence / name).read_text(encoding="utf-8").strip() == SUMMARY_JSON
    calls = log.read_text(encoding="utf-8").splitlines()
    assert calls[0] == "compose up -d --build" and calls[-1] == "compose down", calls
    assert any("scripts/demo_run.py --mode fake --base-url http://127.0.0.1:8000" in call for call in calls)
    assert any("scripts/demo_walkthrough.py --mode fake" in call for call in calls)
    assert (PROJECT_ROOT / ".env").exists() == before, "the test never creates .env in the real project"


@needs_powershell
@needs_posix
def test_a_failing_compose_up_is_still_reported_and_the_stack_is_taken_down(tmp_path) -> None:
    completed, evidence, log = _run_real_demo(tmp_path, up_exit=1)
    assert completed.returncode == 2
    assert "compose up failed" in completed.stdout
    assert not evidence.exists() or not list(evidence.glob("*.json"))
    assert log.read_text(encoding="utf-8").splitlines()[-1] == "compose down"


@needs_powershell
@needs_posix
def test_a_demo_failure_inside_the_container_gives_exit_code_1_and_still_copies_the_summaries(tmp_path) -> None:
    completed, evidence, _ = _run_real_demo(tmp_path, exec_exit=1)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert (evidence / "demo-summary.json").exists() and (evidence / "walkthrough-summary.json").exists()


def test_ps1_functions_that_call_an_external_command_either_capture_its_output_or_send_it_to_the_screen() -> None:
    """A function must not leave a native command's output in the pipeline and then also return an exit code (R2)."""

    real_demo = REAL_DEMO_PS1.read_text(encoding="utf-8-sig")
    docker_function = real_demo[real_demo.index("function Invoke-Docker"): real_demo.index("function Get-DockerText")]
    assert "& docker @Arguments | Out-Host" in docker_function and "return [int]$LASTEXITCODE" in docker_function
    for path in (REAL_DEMO_PS1, PROJECT_ROOT / "scripts" / "check-all.ps1"):
        text = path.read_text(encoding="utf-8-sig")
        for function in re.finditer(r"^function ([\w-]+) \{.*?^\}", text, re.S | re.M):
            body = function.group(0)
            for call in re.finditer(r"^(.*)& (\$\w+|docker|git)\b.*$", body, re.M):
                line = call.group(0)
                captured = "@(" in line or "$output =" in line or "[void]" in line or "| Out-Host" in line or "$null" in line
                assert captured, f"{path.name}: {function.group(1)} runs an external command without taking its output: {line.strip()}"
