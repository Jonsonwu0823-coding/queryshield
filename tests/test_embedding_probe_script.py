"""The embedding probe script reports a failed provider call as a failure instead of crashing."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "check_agent_en02.py"


def test_a_failed_real_embedding_call_is_reported_as_a_failure(tmp_path) -> None:
    """Port 1 on this machine refuses the connection, so no embedding service is called."""

    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
        "QUERYSHIELD_EMBEDDING_BASE_URL": "http://127.0.0.1:1/v1",
        "QUERYSHIELD_EMBEDDING_API_KEY": "key-for-test",
        "QUERYSHIELD_EMBEDDING_MODEL_NAME": "model-for-test",
        "QUERYSHIELD_EMBEDDING_MODEL_REVISION": "revision-for-test",
        "QUERYSHIELD_EMBEDDING_DIMENSIONS": "8",
    }
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "real", "--output-dir", str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert "NameError" not in done.stderr, done.stderr
    assert done.returncode == 1, done.stdout + done.stderr
    assert json.loads(done.stdout)["status"] == "fail"
