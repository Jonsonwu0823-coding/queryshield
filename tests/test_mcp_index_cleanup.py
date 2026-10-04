"""The host's MCP index directory goes away when the application stops normally.

Linux: uvicorn runs the application's shutdown phase on SIGTERM and then
re-raises the signal, so ``atexit`` never runs there; the directory is removed
in the lifespan shutdown.  Windows ``terminate()`` is a hard kill and is not
covered (``docs/mcp.md``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

from queryshield.mcp_metadata import launch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFINITION_QUESTION = "退款后净额是怎么算的？"
TOKEN = "index-cleanup-requester"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _index_dirs(root: Path) -> list[Path]:
    return sorted(root.glob("queryshield-mcp-index-*"))


def test_cleanup_removes_the_index_directory_and_can_run_twice(monkeypatch, tmp_path):
    from queryshield.knowledge.runtime import shared_retrieval_runtime

    launch.cleanup_index_dir()  # nothing of an earlier test is left behind
    monkeypatch.setattr(launch.tempfile, "tempdir", str(tmp_path))
    path = launch.published_index_path(shared_retrieval_runtime("fake").retriever)
    [directory] = _index_dirs(tmp_path)
    assert path.parent == directory and path.is_file()
    launch.cleanup_index_dir()
    assert not directory.exists()
    launch.cleanup_index_dir()  # idempotent


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="SIGTERM semantics of uvicorn on Linux")
def test_a_normal_stop_by_sigterm_leaves_no_index_directory(tmp_path):
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    env = os.environ.copy()
    for name in ("QUERYSHIELD_AGENT_PROFILE", "QUERYSHIELD_RETRIEVAL", "QUERYSHIELD_DEMO_DATASET", "QUERYSHIELD_DATABASE_URL"):
        env.pop(name, None)
    env.update(
        {
            "QUERYSHIELD_METADATA_TOOLS": "mcp",
            "QUERYSHIELD_W04_FAKE_DB": "1",
            "QUERYSHIELD_PROVIDER_MODE": "fake",
            "QUERYSHIELD_STATE_STORE_PATH": str(tmp_path / "state.sqlite3"),
            "QUERYSHIELD_CALL_STORE_PATH": str(tmp_path / "calls.sqlite3"),
            "QUERYSHIELD_TOKEN_A_REQUESTER": TOKEN,
            "QUERYSHIELD_TOKEN_A_APPROVER": "index-cleanup-approver",
            "QUERYSHIELD_TOKEN_B_REQUESTER": "index-cleanup-b-requester",
            "QUERYSHIELD_TOKEN_B_APPROVER": "index-cleanup-b-approver",
            "TMPDIR": str(scratch),  # the index directory lands where the test can see it
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
        }
    )
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "queryshield.api.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=PROJECT_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                with urlopen(base + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (URLError, OSError):
                assert process.poll() is None and time.monotonic() < deadline, "the server did not start"
                time.sleep(0.2)
        request = Request(
            base + "/queries",
            data=json.dumps({"question": DEFINITION_QUESTION}).encode(),
            headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        )
        with urlopen(request, timeout=60) as response:
            assert response.status == 200
        assert len(_index_dirs(scratch)) == 1  # the MCP search wrote the host's index
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    assert _index_dirs(scratch) == []
    assert not list(scratch.glob("queryshield-mcp-session-*"))
