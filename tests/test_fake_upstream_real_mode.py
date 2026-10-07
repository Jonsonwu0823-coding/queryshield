"""A real-mode smoke or demo run that reaches the fake upstream fails, whatever model names it was configured with.

The configured names are deliberately not the fake upstream's, so only the model name the
runs actually recorded (the fake upstream answers ``qs-fake-upstream-v1`` in every response)
can reveal it.  Each test starts the fake upstream on a real port, because the scripts start
their own server process that calls it over HTTP.  Both need PostgreSQL and skip without it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

import pytest

from repo_layout import PROJECT_ROOT
from scripts import demo_run
from scripts import http_smoke as smoke
from scripts.fake_upstream import FAKE_UPSTREAM_MODEL

CONFIGURED_NAME = "provider-model-x"  # not the fake upstream's name
DEMO_QUESTION_IDS = ("Q01", "Q02")  # two questions the Fake model scripts keep the demo run short


def _reachable(url: str | None) -> bool:
    if not url:
        return False
    import psycopg

    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001 - any failure means "not available here"
        return False


def _demo_url() -> str | None:
    url = os.environ.get("QUERYSHIELD_DATABASE_URL")
    return urlunsplit(urlsplit(url)._replace(path="/queryshield_demo")) if url else None


needs_database = pytest.mark.skipif(not _reachable(os.environ.get("QUERYSHIELD_DATABASE_URL")), reason="needs the test database")
needs_demo_database = pytest.mark.skipif(not _reachable(_demo_url()), reason="needs the demo database (docs/demo-data.md, section 2)")


@pytest.fixture()
def real_mode_against_the_fake_upstream(monkeypatch):
    """Start the fake upstream and point the real-mode model and embedding settings at it, under other names."""

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "fake_upstream.py"), "--host", "127.0.0.1", "--port", str(port)],
        cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("the fake upstream did not start")
                time.sleep(0.2)
        base_url = f"http://127.0.0.1:{port}/v1"
        settings = {
            "QUERYSHIELD_MODEL_BASE_URL": base_url,
            "QUERYSHIELD_EMBEDDING_BASE_URL": base_url,
            "QUERYSHIELD_MODEL_NAME": CONFIGURED_NAME,
            "QUERYSHIELD_EMBEDDING_MODEL_NAME": CONFIGURED_NAME,
            "QUERYSHIELD_EMBEDDING_MODEL_REVISION": CONFIGURED_NAME,
            "QUERYSHIELD_EMBEDDING_DIMENSIONS": "128",
            "QUERYSHIELD_MODEL_API_KEY": "not-a-real-key",
            "QUERYSHIELD_EMBEDDING_API_KEY": "not-a-real-key",
        }
        for name, value in settings.items():
            monkeypatch.setenv(name, value)
        yield
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


def _assert_marked_fake(summary: dict) -> None:
    assert summary["mode"] == "real"
    assert summary["status"] == "fail"
    assert summary["hard_failures"] == ["fake_upstream_in_real_mode"], "every step itself passed; only the label fails"
    assert summary["model_names"] == sorted([FAKE_UPSTREAM_MODEL, CONFIGURED_NAME])


@needs_database
def test_a_real_mode_smoke_through_the_fake_upstream_fails_on_the_recorded_model_name(
    real_mode_against_the_fake_upstream, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["http_smoke.py", "--mode", "real", "--evidence-dir", str(tmp_path)])
    assert smoke.main() == 1
    _assert_marked_fake(json.loads((tmp_path / "http-smoke-summary.json").read_text(encoding="utf-8")))


@needs_demo_database
def test_a_real_mode_demo_run_through_the_fake_upstream_fails_on_the_recorded_model_name(
    real_mode_against_the_fake_upstream, tmp_path, monkeypatch
) -> None:
    document = json.loads(demo_run.QUESTIONS_PATH.read_text(encoding="utf-8"))
    document["questions"] = [question for question in document["questions"] if question["id"] in DEMO_QUESTION_IDS]
    assert [question["id"] for question in document["questions"]] == list(DEMO_QUESTION_IDS)
    assert all(question.get("fake_supported") for question in document["questions"])
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(demo_run, "QUESTIONS_PATH", questions)
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", _demo_url())

    assert demo_run.main(["--mode", "real", "--evidence-dir", str(tmp_path / "evidence")]) == 1
    summary = json.loads((tmp_path / "evidence" / "demo-summary.json").read_text(encoding="utf-8"))
    _assert_marked_fake(summary)
    assert [record["id"] for record in summary["records"]] == list(DEMO_QUESTION_IDS)
    assert all(record["verdict"] == "pass" for record in summary["records"])
