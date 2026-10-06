"""The demo walkthrough's judgements are pure functions, and a wrong verified value fails them.

The end-to-end tests drive a real uvicorn process on the demo database (needs_demo_database), like
the CI "checks" job; everything else runs without a database.
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
from uuid import uuid4

import pytest

from scripts import demo_run as run
from scripts import demo_walkthrough as walk
from scripts import generate_demo_data as gen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = {q["id"]: q for q in json.loads(gen.QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]}

GOOD_APPROVAL = {
    "request_http": 202, "request_status": "WAITING_APPROVAL", "approval_permission_bound": True,
    "cross_tenant_http": 404, "cross_tenant_error": "not_found",
    "requester_http": 403, "requester_error": "forbidden",
    "approve_http": 200, "approve_status": "SUCCEEDED",
    "row_count": 40, "answer_contains_row_values": False, "answer_status": "unverified", "fact_count": 0,
}


def _fact(question, value=None, metric=None, window=None):
    expected = question["expected"]
    return {
        "metric_id": metric or expected["metric_id"],
        "value": expected["value"] if value is None else value,
        "time_window": {**(window or question["window"]), "timezone": "UTC"},
    }


# --- the verified-fact rule (the mutation the ticket asks for) -------------------------


def test_a_correct_verified_fact_matches_the_generators_value() -> None:
    question = QUESTIONS["Q02"]
    assert run.verified_fact_counts(question, {"facts": [_fact(question)]}) == (1, 0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda q: _fact(q, value=q["expected"]["value"] + 1),  # one fen off
        lambda q: _fact(q, metric="paid_count"),  # right number, wrong metric
        lambda q: _fact(q, window={"start": "2026-07-01T00:00:00Z", "end": "2026-08-01T00:00:00Z"}),  # wrong window
    ],
    ids=["value", "metric", "window"],
)
def test_a_changed_verified_fact_is_a_mismatch_and_a_hard_failure(mutation) -> None:
    question = QUESTIONS["Q02"]
    obs = {
        "http_status": 200, "status": "SUCCEEDED", "answer_status": "verified", "facts": [mutation(question)],
        "rows": [], "source_ids": [], "sql_exec_count": 2, "trace": [],
    }
    assert run.verified_fact_counts(question, obs) == (1, 1)
    outcome = run.judge_question(question, obs, run.demo_source_ids())
    assert "verified_fact_mismatch" in outcome["hard_failures"]


def test_the_same_value_for_the_other_tenant_is_a_mismatch() -> None:
    question = QUESTIONS["Q02"]  # tenant A
    other = {**question, "identity": "b-requester"}
    assert run.verified_fact_counts(other, {"facts": [_fact(question)]}) == (1, 1)


# --- scenario judgements ----------------------------------------------------------------


def test_the_expected_approval_sequence_passes() -> None:
    assert walk.judge_approval_scenario(GOOD_APPROVAL)["hard_failures"] == []


@pytest.mark.parametrize(
    ("change", "failure"),
    [
        ({"cross_tenant_http": 403, "cross_tenant_error": "forbidden"}, "cross_tenant_approval_not_404"),
        ({"cross_tenant_http": 200}, "cross_tenant_approval_not_404"),
        ({"requester_http": 200}, "requester_approval_not_403"),
        ({"requester_error": "not_found"}, "requester_approval_not_403"),
        ({"approve_http": 403}, "approval_not_executed"),
        ({"row_count": 0}, "approval_no_rows"),
        ({"answer_contains_row_values": True}, "answer_contains_row_values"),
        ({"answer_status": "verified"}, "approval_answer_status"),
        ({"fact_count": 1}, "approval_rows_became_facts"),
        ({"approval_permission_bound": False}, "approval_permission_unbound"),
    ],
)
def test_each_wrong_step_of_the_approval_sequence_is_a_hard_failure(change, failure) -> None:
    assert failure in walk.judge_approval_scenario({**GOOD_APPROVAL, **change})["hard_failures"]


def test_an_approval_that_never_starts_is_one_hard_failure() -> None:
    result = walk.judge_approval_scenario({"request_http": 200, "request_status": "SUCCEEDED"})
    assert result["hard_failures"] == ["approval_not_triggered"]


def test_time_scenario_requires_the_ask_and_a_verified_resume() -> None:
    first = {"http_status": 202, "status": "WAITING_USER", "error_code": None}
    resumed = {"http_status": 200, "status": "SUCCEEDED", "answer_status": "verified"}
    good = walk.judge_time_scenario(first, [], False, resumed, True, True, True)
    assert good["hard_failures"] == [] and good["outcome"] == "waiting"
    for index, name in ((4, "time_resume_basis_not_stated"), (5, "time_resume_window"), (6, "time_resume")):
        args = [first, [], False, resumed, True, True, True]
        args[index] = False
        assert name in walk.judge_time_scenario(*args)["hard_failures"]
    # The server rewriting the model's time question into the catalog's basis question is a hard failure.
    assert "time_ask_rewritten" in walk.judge_time_scenario(first, [], True, None, False, False, False)["hard_failures"]
    # A model that picked the month itself is a known gap, not a failure.
    guessed = walk.judge_time_scenario({"http_status": 200, "status": "SUCCEEDED", "error_code": None}, [], False, None, False, False, False)
    assert guessed["hard_failures"] == [] and guessed["known_gaps"] == ["time_window_guessed_by_model"]


def test_async_scenario_needs_202_then_a_verified_result_with_facts() -> None:
    final = {"status": "SUCCEEDED", "answer_status": "verified", "facts": [{"metric_id": "paid_count"}]}
    assert walk.judge_async_scenario(202, "RUNNING", final)["hard_failures"] == []
    assert "async_not_accepted" in walk.judge_async_scenario(200, "RUNNING", final)["hard_failures"]
    assert "async_no_facts" in walk.judge_async_scenario(202, "RUNNING", {**final, "facts": []})["hard_failures"]
    assert "async_not_succeeded" in walk.judge_async_scenario(202, "RUNNING", {"status": "FAILED"})["hard_failures"]


def test_metadata_expectation() -> None:
    assert walk.metadata_expectation_failures("any", 0) == [] and walk.metadata_expectation_failures("any", 3) == []
    assert walk.metadata_expectation_failures("mcp", 0) == ["metadata_session_missing"]
    assert walk.metadata_expectation_failures("mcp", 2) == []
    assert walk.metadata_expectation_failures("local", 1) == ["metadata_session_unexpected"]
    assert walk.metadata_expectation_failures("local", 0) == []


def test_fake_mode_turns_a_known_gap_into_a_failure() -> None:
    assert walk.fake_gaps_are_failures("fake", [], ["knowledge_after_send_back"]) == (["fake_gap:knowledge_after_send_back"], [])
    assert walk.fake_gaps_are_failures("real", [], ["knowledge_after_send_back"]) == ([], ["knowledge_after_send_back"])


def test_event_stream_parsing_and_mcp_run_selection() -> None:
    lines = [": heartbeat", "", "id: 1", "event: run_created", "data: {}", "", "id: 2", "event: metadata_session", "data: {}", ""]
    assert walk.parse_sse_event_types(lines) == ["run_created", "metadata_session"]
    assert walk.runs_with_metadata_session({"a": ["x", "metadata_session"], "b": ["x"]}) == ["a"]


def test_scenario_record_has_fixed_fields_only() -> None:
    record = walk.scenario_record("S1", http_codes=[200], terminal="SUCCEEDED", answer_status="verified", checked=1, mismatched=0, hard=[], gaps=[], runs=1)
    assert set(record) == {
        "scenario", "http_codes", "terminal", "answer_status", "verified_facts_checked", "verified_facts_mismatched",
        "run_count", "hard_failures", "known_gaps", "verdict",
    }
    assert walk.scenario_record("S1", http_codes=[200], terminal=None, answer_status=None, checked=1, mismatched=1, hard=["verified_fact_mismatch"], gaps=[], runs=1)["verdict"] == "fail"


# --- blocked before anything runs -------------------------------------------------------


def test_missing_or_repeated_tokens_block(monkeypatch) -> None:
    with pytest.raises(walk.Blocked):
        walk.read_tokens({})
    same = {name: "same" for name in walk.TOKEN_ENVIRONMENT.values()}
    with pytest.raises(walk.Blocked):
        walk.read_tokens(same)
    different = {name: f"t-{index}" for index, name in enumerate(walk.TOKEN_ENVIRONMENT.values())}
    assert set(walk.read_tokens(different)) == set(walk.TOKEN_ENVIRONMENT)


def test_an_unreadable_state_store_blocks_instead_of_skipping_the_trace(tmp_path, monkeypatch, capsys) -> None:
    for index, name in enumerate(walk.TOKEN_ENVIRONMENT.values()):
        monkeypatch.setenv(name, f"secret-token-{index}")
    monkeypatch.delenv(walk.STATE_PATH_ENV, raising=False)
    assert walk.main(["--base-url", "http://127.0.0.1:1"]) == 2
    out = capsys.readouterr().out
    assert json.loads(out.splitlines()[-1])["status"] == "blocked" and "secret-token" not in out
    assert walk.main(["--base-url", "http://127.0.0.1:1", "--state-path", str(tmp_path / "missing.sqlite3")]) == 2
    present = tmp_path / "state.sqlite3"
    present.write_bytes(b"")
    assert walk.main(["--base-url", "http://127.0.0.1:1", "--state-path", str(present)]) == 2  # no service answers /health
    assert "did not answer" in capsys.readouterr().out


# --- demo_run --base-url -------------------------------------------------------------------


def test_demo_run_base_url_needs_the_tokens_and_a_readable_state_store(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", "postgresql://u:secret-db-password@127.0.0.1:1/queryshield_demo")
    for name in run.TOKEN_ENVIRONMENT.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(run.STATE_PATH_ENV, raising=False)
    started = []
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs: started.append(args))

    assert run.main(["--mode", "fake", "--base-url", "http://127.0.0.1:1", "--evidence-dir", str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "missing_tokens"
    for index, name in enumerate(run.TOKEN_ENVIRONMENT.values()):
        monkeypatch.setenv(name, f"token-{index}")
    assert run.main(["--mode", "fake", "--base-url", "http://127.0.0.1:1", "--evidence-dir", str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "state_store_unreadable"
    assert started == [], "--base-url never starts a server"
    assert "secret-db-password" not in capsys.readouterr().out


def test_demo_run_tokens_come_from_the_environment() -> None:
    tokens, problems = run.tokens_from_environment({name: f"v-{i}" for i, name in enumerate(run.TOKEN_ENVIRONMENT.values())})
    assert problems == [] and len(set(tokens.values())) == 4
    assert run.tokens_from_environment({})[0] is None
    assert run.tokens_from_environment({name: "x" for name in run.TOKEN_ENVIRONMENT.values()})[1] == ["tokens_must_be_four_different_values"]


# --- end to end against a real server on the demo database -------------------------------------


def _demo_url() -> str | None:
    url = os.environ.get("QUERYSHIELD_DATABASE_URL")
    return urlunsplit(urlsplit(url)._replace(path="/queryshield_demo")) if url else None


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


needs_demo_database = pytest.mark.skipif(not _reachable(_demo_url()), reason="needs the demo database (docs/demo-data.md, section 2)")


class _Server:
    def __init__(self, tmp_path: Path, metadata_tools: str | None) -> None:
        self.tokens = {name: uuid4().hex for name in walk.TOKEN_ENVIRONMENT.values()}
        self.state = tmp_path / "state.sqlite3"
        env = {key: value for key, value in os.environ.items() if key not in {"QUERYSHIELD_METADATA_TOOLS", "QUERYSHIELD_FAKE_DB"}}
        env.update(
            {
                "QUERYSHIELD_DATABASE_URL": _demo_url(), "QUERYSHIELD_DEMO_DATASET": "commerce-demo-v1", "QUERYSHIELD_PROVIDER_MODE": "fake",
                "QUERYSHIELD_STATE_STORE_PATH": str(self.state), "QUERYSHIELD_CALL_STORE_PATH": str(tmp_path / "calls.sqlite3"),
                "PYTHONPATH": str(PROJECT_ROOT / "src"), **self.tokens,
            }
        )
        if metadata_tools:
            env["QUERYSHIELD_METADATA_TOOLS"] = metadata_tools
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "queryshield.api.main:app", "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=PROJECT_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            try:
                with urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2):
                    return
            except OSError:
                time.sleep(0.3)
        self.stop()
        raise RuntimeError("the test server did not start")

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()


@pytest.fixture()
def serve(tmp_path, monkeypatch):
    servers: list[_Server] = []

    def start(metadata_tools: str | None = None) -> _Server:
        server = _Server(tmp_path, metadata_tools)
        servers.append(server)
        for name, token in server.tokens.items():
            monkeypatch.setenv(name, token)
        monkeypatch.setenv(walk.STATE_PATH_ENV, str(server.state))
        return server

    yield start
    for server in servers:
        server.stop()


@needs_demo_database
def test_walkthrough_passes_against_a_real_fake_server_and_writes_no_text(serve, tmp_path, capsys) -> None:
    server = serve()
    assert walk.main(["--base-url", server.base, "--mode", "fake", "--metadata-tools", "local", "--evidence-dir", str(tmp_path / "evidence")]) == 0
    printed = capsys.readouterr().out
    assert "场景 7" in printed and "原始查询结果，未核实" in printed and "404" in printed and "403" in printed
    summary_text = (tmp_path / "evidence" / "walkthrough-summary.json").read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    assert summary["status"] == "pass" and [s["scenario"] for s in summary["scenarios"]] == [
        "S1_verified_sync", "S2_clarify_resume", "S3_time_request", "S4_no_data", "S5_knowledge", "S6_async", "S7_approval",
    ]
    assert summary["verified_fact_check"]["checked"] >= 4 and summary["verified_fact_check"]["mismatched"] == 0
    assert summary["mcp_run_count"] == 0
    # No question, answer text or customer name; no token.
    for forbidden in ("销售额", "已核实：", "何文博", *server.tokens.values()):
        assert forbidden not in summary_text


@needs_demo_database
def test_walkthrough_sees_the_mcp_setting_and_the_expectation_is_enforced(serve, tmp_path, capsys) -> None:
    server = serve("mcp")
    assert walk.main(["--base-url", server.base, "--metadata-tools", "mcp", "--evidence-dir", str(tmp_path / "ev1")]) == 0
    summary = json.loads((tmp_path / "ev1" / "walkthrough-summary.json").read_text(encoding="utf-8"))
    assert summary["mcp_run_count"] >= 4 and "S1_verified_sync" in summary["mcp_scenarios"]
    assert walk.main(["--base-url", server.base, "--metadata-tools", "local", "--evidence-dir", str(tmp_path / "ev2")]) == 1
    assert "metadata_session_unexpected" in capsys.readouterr().out


@needs_demo_database
def test_demo_run_drives_a_running_service_like_the_one_it_starts_itself(serve, tmp_path, monkeypatch) -> None:
    server = serve()
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", _demo_url())  # inside the service container this is the demo database
    assert run.main(["--mode", "fake", "--base-url", server.base, "--evidence-dir", str(tmp_path / "external")]) == 0
    external = json.loads((tmp_path / "external" / "demo-summary.json").read_text(encoding="utf-8"))
    assert external["status"] == "pass" and external["verified_fact_check"]["mismatched"] == 0 and external["verified_fact_check"]["checked"] > 0


# --- Every verified fact is compared on every path ----------------------------------------------------------------

import ast  # noqa: E402

SEPTEMBER = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}


def _generator_value(metric_id: str, tenant: str = "A", window: dict = SEPTEMBER) -> int:
    start, end = run._utc_seconds(window["start"]), run._utc_seconds(window["end"])
    return gen.expected_metrics(run.demo_data(), tenant, start, end)[metric_id]


def _september_fact(metric_id: str = "gross_fen", *, delta: int = 0) -> dict:
    return {"metric_id": metric_id, "value": _generator_value(metric_id) + delta, "time_window": {**SEPTEMBER, "timezone": "UTC"}}


def _response(facts: list[dict], status: str = "SUCCEEDED", **extra) -> dict:
    return {"run_id": "run-1", "status": status, "answer_status": "verified", "answer": "text", "facts": {"facts": facts}, **extra}


def _walkthrough(monkeypatch, mode: str = "real", http=None) -> "walk.Walkthrough":
    """A Walkthrough with a stand-in for HTTP and for the service's state store (no database, no server)."""

    instance = object.__new__(walk.Walkthrough)
    instance.base, instance.state_path, instance.mode, instance.metadata_tools = "http://stub", Path("/nonexistent/state.sqlite3"), mode, "any"
    instance.tokens = {identity: f"token-{identity}" for identity in walk.demo_run.TOKEN_ENVIRONMENT}
    instance.records, instance.run_tokens, instance.run_scenario = [], {}, {}
    instance.no_data_reply, instance.catalog_question, instance.demo_ids = "", "", frozenset()
    instance._http = http
    monkeypatch.setattr(walk.smoke, "_clarification_reviews", lambda *args, **kwargs: [])
    monkeypatch.setattr(walk.smoke, "_action_trace", lambda *args, **kwargs: [])
    monkeypatch.setattr(walk.demo_run, "_completion_tokens", lambda *args, **kwargs: [])
    return instance


def test_scenario_3_compares_the_facts_when_the_model_guessed_the_time_window(monkeypatch, capsys) -> None:
    """The first answer is already SUCCEEDED (the model picked September itself): its fact is still compared."""

    first = _response([_september_fact()])
    walkthrough = _walkthrough(monkeypatch, http=lambda identity, path, **kwargs: (200, first))
    walkthrough.undated()
    record = walkthrough.records[0]
    assert (record["verified_facts_checked"], record["verified_facts_mismatched"]) == (1, 0)
    assert record["known_gaps"] == ["time_window_guessed_by_model"] and record["hard_failures"] == []
    assert record["verdict"] == "pass" and record["http_codes"] == [200]


def test_scenario_3_a_wrong_value_in_a_guessed_window_is_a_hard_failure(monkeypatch, capsys) -> None:
    wrong = _response([_september_fact(delta=1)])
    walkthrough = _walkthrough(monkeypatch, http=lambda identity, path, **kwargs: (200, wrong))
    walkthrough.undated()
    record = walkthrough.records[0]
    assert (record["verified_facts_checked"], record["verified_facts_mismatched"]) == (1, 1)
    assert "verified_fact_mismatch" in record["hard_failures"] and record["verdict"] == "fail"
    assert record["known_gaps"] == ["time_window_guessed_by_model"], "the known gap is still recorded"


def test_scenario_3_still_compares_on_the_ask_and_resume_path(monkeypatch, capsys) -> None:
    waiting = {"run_id": "run-1", "status": "WAITING_USER", "pending_question": "请问要查哪个时间范围的支付金额？"}
    resumed = _response([_september_fact(delta=1)], answer="口径：支付订单总额（gross_fen）；依据：问题中提到‘支付金额’。")

    def http(identity, path, **kwargs):
        return (202, waiting) if path == "/queries" else (200, resumed)

    walkthrough = _walkthrough(monkeypatch, http=http)
    walkthrough.undated()
    record = walkthrough.records[0]
    assert (record["verified_facts_checked"], record["verified_facts_mismatched"]) == (1, 1)
    assert "verified_fact_mismatch" in record["hard_failures"] and record["http_codes"] == [202, 200]


def test_the_invariant_fails_a_scenario_that_has_facts_it_did_not_compare(monkeypatch, capsys) -> None:
    walkthrough = _walkthrough(monkeypatch)
    walkthrough._finish(
        "S_x", [200], {"status": "SUCCEEDED", "answer_status": "verified"},
        facts=[_september_fact(), _september_fact("paid_count")], checked=1, mismatched=0, hard=[], gaps=[], runs=1,
    )
    record = walkthrough.records[0]
    assert record["hard_failures"] == ["verified_fact_unchecked"] and record["verdict"] == "fail"
    # Compared all of them: fine.
    walkthrough._finish("S_y", [200], {"status": "SUCCEEDED"}, facts=[_september_fact()], checked=1, mismatched=0, hard=[], gaps=[], runs=1)
    assert walkthrough.records[1]["hard_failures"] == []


def test_unchecked_fact_failures_is_exact() -> None:
    assert walk.unchecked_fact_failures(0, 0) == [] and walk.unchecked_fact_failures(3, 3) == []
    assert walk.unchecked_fact_failures(1, 0) == ["verified_fact_unchecked"]
    assert walk.unchecked_fact_failures(2, 1) == ["verified_fact_unchecked"]


def test_every_scenario_ends_through_finish_with_its_facts_and_the_count_it_compared() -> None:
    """_finish needs `facts` and `checked` as keywords, and every call site passes them: no scenario can skip the invariant."""

    tree = ast.parse((PROJECT_ROOT / "scripts" / "demo_walkthrough.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_finish"]
    assert len(calls) == 5, "demo questions (S1, S2, S4, S5), S3, S6 and two exits of S7"
    for call in calls:
        assert {"facts", "checked"} <= {keyword.arg for keyword in call.keywords}
    walkthrough = object.__new__(walk.Walkthrough)
    with pytest.raises(TypeError):
        walkthrough._finish("S_x", [200], {}, 0, 0, [], [], 1)  # the old positional form no longer exists


def test_a_demo_question_scenario_goes_through_the_invariant_too(monkeypatch, capsys) -> None:
    question = QUESTIONS["Q02"]
    obs = {
        "http_status": 200, "status": "SUCCEEDED", "answer_status": "verified", "answer": "x", "facts": [_fact(question)],
        "rows": [], "source_ids": [], "sql_exec_count": 2, "trace": [], "run_id": "run-1",
    }
    walkthrough = _walkthrough(monkeypatch)
    walkthrough.questions = QUESTIONS
    monkeypatch.setattr(walk.demo_run, "_run_question", lambda *args, **kwargs: (dict(obs), {}))
    walkthrough.demo_question("S1_verified_sync", 1, "t", "Q02")
    assert (walkthrough.records[0]["verified_facts_checked"], walkthrough.records[0]["hard_failures"]) == (1, [])
    # A judgement that compared nothing although the answer has a verified fact fails the scenario.
    monkeypatch.setattr(walk.demo_run, "judge_question", lambda *args, **kwargs: {"verified_facts_checked": 0, "verified_facts_mismatched": 0, "hard_failures": [], "known_gaps": []})
    walkthrough.demo_question("S1_verified_sync", 1, "t", "Q02")
    assert walkthrough.records[1]["hard_failures"] == ["verified_fact_unchecked"]


def test_scenario_6_prints_a_status_line_only_when_the_state_changes(monkeypatch, capsys) -> None:
    statuses = ["RUNNING"] * 5 + ["SUCCEEDED"]
    polled = iter(statuses)
    final = _response([_september_fact("paid_count")], run_id="run-1")

    def http(identity, path, **kwargs):
        if path == "/queries":
            return 202, {"run_id": "run-1", "status": "RUNNING"}
        if path.endswith("/result"):
            return 200, final
        return 200, {"run_id": "run-1", "status": next(polled)}

    walkthrough = _walkthrough(monkeypatch, http=http)
    monkeypatch.setattr(walk.time, "sleep", lambda seconds: None)
    walkthrough.async_run()
    lines = capsys.readouterr().out.splitlines()
    assert [line for line in lines if line.startswith("查状态：")] == ["查状态：HTTP 200 状态=RUNNING", "查状态：HTTP 200 状态=SUCCEEDED"]
    assert "  一共查了 6 次状态" in lines
    record = walkthrough.records[0]
    assert (record["verified_facts_checked"], record["verified_facts_mismatched"], record["hard_failures"]) == (1, 0, [])
    assert set(record) == {
        "scenario", "http_codes", "terminal", "answer_status", "verified_facts_checked", "verified_facts_mismatched",
        "run_count", "hard_failures", "known_gaps", "verdict",
    }, "the summary fields did not change"
