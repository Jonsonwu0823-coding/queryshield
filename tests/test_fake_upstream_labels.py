"""Runs through the fake upstream are never real-model evidence, and say so in their summaries."""

from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

from queryshield.db.state_store import StateStore
from queryshield.knowledge.runtime import FAKE_EMBEDDING_DIMENSIONS
from repo_layout import PROJECT_ROOT, REPO_ROOT, STANDALONE_WORKFLOW, WORKFLOW_RELATIVE
from scripts import check_eval
from scripts import demo_run
from scripts import http_smoke as smoke
from scripts.fake_upstream import FAKE_UPSTREAM_MODEL, evidence_failures

REAL_NAME = "provider-model-x"


@pytest.mark.parametrize(
    ("mode", "names", "expected"),
    [
        ("real", [REAL_NAME], []),
        ("real", [REAL_NAME, FAKE_UPSTREAM_MODEL], ["fake_upstream_in_real_mode"]),
        ("real", [FAKE_UPSTREAM_MODEL], ["fake_upstream_in_real_mode"]),
        ("fake-upstream", [FAKE_UPSTREAM_MODEL], []),
        ("fake-upstream", [FAKE_UPSTREAM_MODEL, FAKE_UPSTREAM_MODEL], []),
        ("fake-upstream", [FAKE_UPSTREAM_MODEL, REAL_NAME], ["model_not_fake_upstream"]),
        ("fake-upstream", [REAL_NAME], ["model_not_fake_upstream"]),
        ("fake-upstream", [], ["model_not_fake_upstream"]),
        ("fake", ["fake-model"], []),
    ],
)
def test_evidence_failures(mode, names, expected) -> None:
    assert evidence_failures(mode, names) == expected


def _store(tmp_path, runs: dict[str, list[str]]):
    path = tmp_path / "state.sqlite3"
    with StateStore(path) as store:
        for run_id, models in runs.items():
            store.create_run(run_id=run_id, tenant_id="A", principal_id="p", role="requester", question="q", mode="real")
            for model in models:
                store.append_event(run_id, "agent_step", "RUNNING", payload={"kind": "model_call", "status": "succeeded", "model": model})
            store.append_event(run_id, "agent_step", "RUNNING", payload={"kind": "tool_call", "model": "not-a-model-call"})
    return path


def test_recorded_model_names_reads_the_model_calls_of_the_given_runs_and_the_embedding_model(tmp_path) -> None:
    path = _store(tmp_path, {"run-1": [FAKE_UPSTREAM_MODEL], "run-2": [FAKE_UPSTREAM_MODEL, REAL_NAME], "run-other": ["fake-model"]})
    environ = {"QUERYSHIELD_EMBEDDING_MODEL_NAME": " embedding-x "}
    assert smoke.recorded_model_names(path, ["run-1", "run-2"], environ) == sorted([FAKE_UPSTREAM_MODEL, REAL_NAME, "embedding-x"])
    assert smoke.recorded_model_names(path, ["run-1"], {}) == [FAKE_UPSTREAM_MODEL]
    assert smoke._store_run_ids(path) == ["run-1", "run-2", "run-other"]


def test_model_labels_mark_real_and_fake_upstream_runs_and_leave_a_fake_run_alone(tmp_path) -> None:
    path = _store(tmp_path, {"run-1": [FAKE_UPSTREAM_MODEL]})
    environ = {"QUERYSHIELD_EMBEDDING_MODEL_NAME": FAKE_UPSTREAM_MODEL}
    assert smoke.model_labels("fake-upstream", path, ["run-1"], environ) == ([FAKE_UPSTREAM_MODEL], [])
    assert smoke.model_labels("real", path, ["run-1"], environ) == ([FAKE_UPSTREAM_MODEL], ["fake_upstream_in_real_mode"])
    # A fake run adds no field and reads no store (this path does not exist).
    assert smoke.model_labels("fake", tmp_path / "missing" / "state.sqlite3", ["run-1"], environ) == (None, [])


def test_the_fake_upstream_mode_runs_the_server_in_real_mode_with_the_real_configuration() -> None:
    assert smoke.SERVER_MODE == {"fake": "fake", "real": "real", "fake-upstream": "real"}
    assert smoke.required_names("fake-upstream") == smoke.required_names("real") == smoke.REQUIRED_NAMES
    assert smoke.required_names("fake") == ("QUERYSHIELD_DATABASE_URL",)


def _configured(monkeypatch, model: str, embedding: str) -> None:
    for name in smoke.REQUIRED_NAMES:
        monkeypatch.setenv(name, "set")
    monkeypatch.setenv("QUERYSHIELD_MODEL_NAME", model)
    monkeypatch.setenv("QUERYSHIELD_EMBEDDING_MODEL_NAME", embedding)


@pytest.mark.parametrize(("model", "embedding", "wrong"), [
    (REAL_NAME, FAKE_UPSTREAM_MODEL, ["QUERYSHIELD_MODEL_NAME"]),
    (FAKE_UPSTREAM_MODEL, REAL_NAME, ["QUERYSHIELD_EMBEDDING_MODEL_NAME"]),
    (f"{FAKE_UPSTREAM_MODEL}-x", FAKE_UPSTREAM_MODEL, ["QUERYSHIELD_MODEL_NAME"]),  # the whole name, not a part of it
    (f"x-{FAKE_UPSTREAM_MODEL}", FAKE_UPSTREAM_MODEL, ["QUERYSHIELD_MODEL_NAME"]),  # not a suffix either
])
def test_fake_upstream_mode_is_blocked_before_anything_runs_unless_both_names_are_the_fake_upstreams(
    tmp_path, monkeypatch, capsys, model, embedding, wrong
) -> None:
    _configured(monkeypatch, model, embedding)
    assert smoke.names_not_fake_upstream(os.environ) == wrong
    assert demo_run.main(["--mode", "fake-upstream", "--evidence-dir", str(tmp_path / "demo")]) == 2
    assert json.loads(capsys.readouterr().out) == {"status": "blocked", "reason": "model_names_not_fake_upstream", "names": wrong}
    monkeypatch.setattr(sys, "argv", ["http_smoke.py", "--mode", "fake-upstream", "--evidence-dir", str(tmp_path / "smoke")])
    assert smoke.main() == 2
    assert json.loads(capsys.readouterr().out) == {"status": "blocked", "reason": "model_names_not_fake_upstream", "names": wrong}
    assert not (tmp_path / "demo").exists() and not (tmp_path / "smoke").exists()


def test_the_fake_upstream_names_pass_the_preflight() -> None:
    environ = {"QUERYSHIELD_MODEL_NAME": FAKE_UPSTREAM_MODEL, "QUERYSHIELD_EMBEDDING_MODEL_NAME": f" {FAKE_UPSTREAM_MODEL} "}
    assert smoke.names_not_fake_upstream(environ) == []


def _run_check_eval(monkeypatch, tmp_path, mode: str, returned_models: set[str], embedding: str, status: str = "pass"):
    monkeypatch.setattr(check_eval._RecordingModelAdapter, "returned_models", set(returned_models))
    monkeypatch.setattr(check_eval, "run_check", lambda check_id, mode, evidence_dir: {"status": status, "check_id": check_id, "mode": mode})
    monkeypatch.setenv("QUERYSHIELD_EMBEDDING_MODEL_NAME", embedding)
    monkeypatch.setattr(sys, "argv", ["check_eval.py", "--check-id", "EVAL-R05", "--mode", mode, "--evidence-dir", str(tmp_path)])
    code = check_eval.main()
    return code, json.loads((tmp_path / "EVAL-R05.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(("returned", "embedding"), [({FAKE_UPSTREAM_MODEL}, REAL_NAME), ({REAL_NAME}, FAKE_UPSTREAM_MODEL)])
def test_a_real_evaluation_through_the_fake_upstream_fails(monkeypatch, tmp_path, capsys, returned, embedding) -> None:
    code, result = _run_check_eval(monkeypatch, tmp_path, "real", returned, embedding)
    assert code == 1
    assert result["status"] == "fail" and result["evidence_failures"] == ["fake_upstream_in_real_mode"]


def test_a_real_evaluation_with_real_models_and_a_fake_evaluation_are_unchanged(monkeypatch, tmp_path, capsys) -> None:
    assert _run_check_eval(monkeypatch, tmp_path, "real", {REAL_NAME}, "embedding-x") == (0, {"status": "pass", "check_id": "EVAL-R05", "mode": "real"})
    assert _run_check_eval(monkeypatch, tmp_path, "fake", {FAKE_UPSTREAM_MODEL}, FAKE_UPSTREAM_MODEL) == (0, {"status": "pass", "check_id": "EVAL-R05", "mode": "fake"})


def test_the_recording_adapter_remembers_every_returned_model_name(monkeypatch) -> None:
    from queryshield.providers.fake_model import FakeModel

    monkeypatch.setattr(check_eval._RecordingModelAdapter, "returned_models", set())
    check_eval._RecordingModelAdapter(FakeModel()).complete([{"role": "user", "content": "你好，你能做什么？"}])
    assert check_eval._RecordingModelAdapter.returned_models == {"fake-model"}


# The adapter passes its timeout to the injected client; TestClient ignores it and warns.
@pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")
def test_the_recording_adapter_keeps_the_model_name_from_the_response_not_the_configured_one(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from queryshield.providers.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleModel
    from scripts.fake_upstream import app

    config = OpenAICompatibleConfig(base_url="http://fake-upstream/v1", api_key="not-a-real-key", model=REAL_NAME)
    adapter = check_eval._RecordingModelAdapter(OpenAICompatibleModel(config, client=TestClient(app)))
    monkeypatch.setattr(check_eval._RecordingModelAdapter, "returned_models", set())
    adapter.complete([{"role": "user", "content": "你好，你能做什么？"}])
    assert check_eval._RecordingModelAdapter.returned_models == {FAKE_UPSTREAM_MODEL}


# --- the Compose override and the CI step ------------------------------------------------------------


def _override() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "compose.fake-upstream.yaml").read_text(encoding="utf-8"))


def test_the_override_runs_the_fake_upstream_unpublished_and_points_app_at_it_in_real_mode() -> None:
    services = _override()["services"]
    upstream = services["fake-upstream"]
    assert "ports" not in upstream, "only the other services reach the fake upstream"
    assert upstream["read_only"] is True and upstream["image"] == "queryshield-app:local"
    command = upstream["command"]
    assert command[:2] == ["python", "scripts/fake_upstream.py"]
    port = command[command.index("--port") + 1]
    assert port == "8000", "the image's health check probes :8000/health"

    app = services["app"]
    assert app["depends_on"] == {"fake-upstream": {"condition": "service_healthy"}}
    environment = app["environment"]
    assert environment["QUERYSHIELD_PROVIDER_MODE"] == "real"
    for name in ("QUERYSHIELD_MODEL_BASE_URL", "QUERYSHIELD_EMBEDDING_BASE_URL"):
        assert environment[name] == f"http://fake-upstream:{port}/v1"
    assert smoke.names_not_fake_upstream(environment) == []
    assert environment["QUERYSHIELD_EMBEDDING_MODEL_REVISION"] == FAKE_UPSTREAM_MODEL
    assert environment["QUERYSHIELD_EMBEDDING_DIMENSIONS"] == str(FAKE_EMBEDDING_DIMENSIONS)
    # The only credential-like values are the documented placeholder.
    secret_like = {name: value for name, value in environment.items() if any(part in name for part in ("PASSWORD", "TOKEN", "SECRET", "API_KEY"))}
    assert secret_like == {"QUERYSHIELD_MODEL_API_KEY": "not-a-real-key", "QUERYSHIELD_EMBEDDING_API_KEY": "not-a-real-key"}


def test_the_ci_compose_job_runs_the_fake_upstream_demo_and_removes_the_service() -> None:
    workflow = STANDALONE_WORKFLOW if STANDALONE_WORKFLOW.is_file() else REPO_ROOT / WORKFLOW_RELATIVE
    steps = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["compose"]["steps"]
    step = next(item for item in steps if "compose.fake-upstream.yaml" in item.get("run", ""))
    run = step["run"]
    assert "demo_run.py --mode fake-upstream --base-url http://127.0.0.1:8000" in run
    assert "rm -sf fake-upstream" in run and "trap " in run
    assert '"qs-fake-upstream-v1"' in run
    names = [item["name"] for item in steps]
    assert names.index(step["name"]) < names.index("Show the service log when a step failed")


def test_fake_and_fake_upstream_demo_runs_skip_the_questions_the_fake_model_does_not_script() -> None:
    questions = json.loads(demo_run.QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    unscripted = [question for question in questions if not question.get("fake_supported")]
    assert unscripted and len(questions) - len(unscripted) == 13
    for question in questions:
        assert demo_run.fake_scripted_only("real", question) is False
        for mode in ("fake", "fake-upstream"):
            assert demo_run.fake_scripted_only(mode, question) is (question in unscripted)


# The adapter passes its timeout to the injected client; TestClient ignores it and warns.
@pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")
def test_a_real_evaluation_whose_embedding_alone_answers_from_the_fake_upstream_fails(monkeypatch, tmp_path, capsys) -> None:
    """The configured embedding name is a real one; only the response names the fake upstream."""

    from fastapi.testclient import TestClient

    from queryshield.providers.embedding import EmbeddingConfig, OpenAICompatibleEmbedding
    from scripts.fake_upstream import app

    monkeypatch.setenv("QUERYSHIELD_EMBEDDING_BASE_URL", "http://fake-upstream/v1")
    monkeypatch.setenv("QUERYSHIELD_EMBEDDING_API_KEY", "not-a-real-key")
    monkeypatch.setenv("QUERYSHIELD_EMBEDDING_MODEL_NAME", "provider-embedding-x")
    monkeypatch.setenv("QUERYSHIELD_EMBEDDING_MODEL_REVISION", "provider-embedding-x-r1")
    monkeypatch.setenv("QUERYSHIELD_EMBEDDING_DIMENSIONS", str(FAKE_EMBEDDING_DIMENSIONS))
    upstream = TestClient(app)
    monkeypatch.setattr(
        OpenAICompatibleEmbedding, "from_env", classmethod(lambda cls, client=None: cls(EmbeddingConfig.from_env(), client=upstream))
    )
    monkeypatch.setattr(check_eval._RecordingModelAdapter, "returned_models", {REAL_NAME})

    _cases, _snapshot, index_build, _embedder, _retriever = check_eval._build_retrieval_runtime("real", retrieval_cases=())
    assert index_build.index.model == "provider-embedding-x"  # the index keeps the configured name
    assert check_eval._RecordingModelAdapter.returned_models == {REAL_NAME, FAKE_UPSTREAM_MODEL}
    monkeypatch.setattr(check_eval, "run_check", lambda check_id, mode, evidence_dir: {"status": "pass", "check_id": check_id, "mode": mode})
    monkeypatch.setattr(sys, "argv", ["check_eval.py", "--check-id", "EVAL-R05", "--mode", "real", "--evidence-dir", str(tmp_path)])

    assert check_eval.main() == 1
    result = json.loads((tmp_path / "EVAL-R05.json").read_text(encoding="utf-8"))
    assert result["status"] == "fail" and result["evidence_failures"] == ["fake_upstream_in_real_mode"]
