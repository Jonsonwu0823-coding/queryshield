"""B2b follow-up: the local HTTP smoke reports blocked/timeouts and always writes its summary."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import b2b_http_smoke as smoke  # noqa: E402


def _silent_listener() -> socket.socket:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    return server


def test_unreachable_database_is_blocked_before_the_server_starts(tmp_path, monkeypatch, capsys) -> None:
    closed = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    closed.bind(("127.0.0.1", 0))
    port = closed.getsockname()[1]
    closed.close()
    url = f"postgresql://queryshield_ro:secret-password@127.0.0.1:{port}/queryshield_test"
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", url)
    started = []
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *args, **kwargs: started.append(args))
    monkeypatch.setattr(sys, "argv", ["b2b_http_smoke.py", "--mode", "fake", "--evidence-dir", str(tmp_path)])

    assert smoke.main() == 2

    output = capsys.readouterr().out
    assert json.loads(output.splitlines()[0]) == {"status": "blocked", "reason": "database_unreachable"}
    assert "secret-password" not in output and "postgresql://" not in output and "Traceback" not in output
    assert started == []
    summary = json.loads((tmp_path / "b2b-http-smoke-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked" and summary["reason"] == "database_unreachable"


def test_http_timeout_is_a_step_stop_not_a_traceback() -> None:
    server = _silent_listener()
    try:
        with pytest.raises(smoke._StepStopped) as caught:
            smoke._http(f"http://127.0.0.1:{server.getsockname()[1]}", "/queries", token="t", method="POST", body={}, timeout=0.5)
    finally:
        server.close()
    assert caught.value.kind == "http_timeout"


def test_timeout_stops_later_steps_and_still_writes_the_summary(tmp_path, monkeypatch, capsys) -> None:
    class _Process:
        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    class _Health:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    calls = []

    def timed_out(*args, **kwargs):
        calls.append(args[1])
        raise smoke._StepStopped("http_timeout")

    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", "postgresql://unused")
    monkeypatch.setattr(smoke, "_database_reachable", lambda: True)
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *args, **kwargs: _Process())
    monkeypatch.setattr(smoke, "urlopen", lambda *args, **kwargs: _Health())
    monkeypatch.setattr(smoke, "_http", timed_out)
    monkeypatch.setattr(sys, "argv", ["b2b_http_smoke.py", "--mode", "fake", "--evidence-dir", str(tmp_path)])

    assert smoke.main() == 1

    assert calls == ["/queries"]  # no later step was attempted
    summary = json.loads((tmp_path / "b2b-http-smoke-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "fail"
    assert summary["hard_failures"] == ["http_timeout:sync_success"]
    assert summary["records"] == [{"step": "sync_success", "error": "http_timeout"}]
    assert "Traceback" not in capsys.readouterr().out


def test_failed_step_record_shows_parse_detail_and_shape(tmp_path, capsys) -> None:
    from queryshield.db.w04_state import StateStore

    state_path = tmp_path / "state.sqlite3"
    shape = {"json": "valid", "type": "tool_call", "name": "query_readonly", "arguments": {"params": "array"}}
    with StateStore(state_path) as store:
        store.create_run(
            run_id="run-1", tenant_id="A", principal_id="p", role="requester", question="q",
            mode="real", checkpoint={}, run_config={},
        )
        store.append_event("run-1", "agent_step", "RUNNING", payload={
            "kind": "proposal_validation", "status": "failed", "error_code": "invalid_field",
            "error_detail": "params must be an object", "action_shape": shape,
        })

    record = smoke._record("approval_request", 502, {"run_id": "run-1", "status": "FAILED"}, state_path)

    assert record["parse_failures"] == [
        {"error_code": "invalid_field", "error_detail": "params must be an object", "action_shape": shape}
    ]
    assert "params must be an object" in capsys.readouterr().out
