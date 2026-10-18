"""Blind test v3 end to end on the demo database with the synthetic set: Fake and fake-upstream, B0 and B1 three runs each.

Needs the demo database (docs/demo-data.md, section 2) and skips without it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

import pytest

from scripts import run_blind_v3 as runner
from scripts import seal_blind_v3 as seal
from scripts.fake_upstream import FAKE_UPSTREAM_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


pytestmark = pytest.mark.skipif(not _reachable(_demo_url()), reason="needs the demo database (docs/demo-data.md, section 2)")
CANDIDATE = hashlib.sha256(runner._source_manifest_bytes()).hexdigest()


@pytest.fixture()
def sealed(tmp_path, monkeypatch, capsys):
    """The synthetic set sealed by the sealing tool, the demo database for the run, and the key at the hidden prompt."""

    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", _demo_url())
    plaintext = tmp_path / "plain.json"
    assert seal.main(["synthetic", "--out", str(plaintext)]) == 0
    assert seal.main(["seal", "--plaintext", str(plaintext), "--out-dir", str(tmp_path / "sealed")]) == 0
    lines = capsys.readouterr().out.splitlines()
    key = lines[lines.index("KEY (shown once; save it now, then clear the screen):") + 1]
    monkeypatch.setattr(runner.getpass, "getpass", lambda prompt: key)
    return tmp_path / "sealed", key


@pytest.fixture()
def fake_upstream(monkeypatch):
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
        base = f"http://127.0.0.1:{port}/v1"
        for name, value in {
            "QUERYSHIELD_MODEL_BASE_URL": base, "QUERYSHIELD_EMBEDDING_BASE_URL": base,
            "QUERYSHIELD_MODEL_NAME": FAKE_UPSTREAM_MODEL, "QUERYSHIELD_EMBEDDING_MODEL_NAME": FAKE_UPSTREAM_MODEL,
            "QUERYSHIELD_EMBEDDING_MODEL_REVISION": FAKE_UPSTREAM_MODEL, "QUERYSHIELD_EMBEDDING_DIMENSIONS": "128",
            "QUERYSHIELD_MODEL_API_KEY": "not-a-real-key", "QUERYSHIELD_EMBEDDING_API_KEY": "not-a-real-key",
        }.items():
            monkeypatch.setenv(name, value)
        yield
    finally:
        process.terminate()
        process.wait(timeout=15)


def _run(sealed_dir: Path, root: Path, mode: str = "fake", *extra: str) -> int:
    return runner.main(
        ["run", "--mode", mode, "--sealed-dir", str(sealed_dir), "--evidence-root", str(root), "--candidate-manifest-sha256", CANDIDATE, *extra]
    )


def _state_dirs() -> set[str]:
    return {name for name in os.listdir(tempfile.gettempdir()) if name.startswith("demo-run-")}


def _assert_no_plaintext(text: str) -> None:
    payload = seal.synthetic_payload()
    secrets = {case["question"] for case in payload["cases"]} | {"按支付金额统计"} | set(runner.customer_names("A")) | set(runner.customer_names("B"))
    assert not any(item in text for item in secrets)


def _expected_scores(summary: dict) -> None:
    runs = summary["scores"]["runs"]
    assert sorted(runs) == sorted(runner.RUN_ORDER)
    for label, run in runs.items():
        expected = (7, 4) if label.startswith("B1") else (4, 1)
        assert (run["functional_success"]["numerator"], run["security_correct"]["numerator"]) == expected, label
    assert runs["B1-run1"]["refusals"] == {"denied": 2}
    assert runs["B1-run1"]["cases"]["V3-07"]["failures"] == ["rows_mismatch"], "the Fake lists every customer, not the top one"
    assert summary["scores"]["configs"]["B1"]["case_passes"]["V3-07"] == 0


def test_a_fake_attempt_runs_six_times_then_judges_and_recalculates(sealed, tmp_path, monkeypatch, capsys) -> None:
    sealed_dir, key = sealed
    root = tmp_path / "evidence"
    before = _state_dirs()
    assert _run(sealed_dir, root) == 0
    out = capsys.readouterr().out
    progress, _, results = out.partition("attempt 1 complete; every run finished")
    lines = progress.strip().splitlines()
    assert all(re.fullmatch(r"attempt 1 run B[01]-run[123] (started|case \d\d/12 done)", line) for line in lines), "progress only until every run has finished"
    assert len(lines) == 6 + 72
    assert "B1: functional mean 7.0 (min 7, max 7), security mean 4.0 (min 4, max 4)" in results
    _assert_no_plaintext(out)

    attempt = root / "attempt-1"
    summary_text = (attempt / "summary.json").read_text(encoding="utf-8")
    _assert_no_plaintext(summary_text)
    summary = json.loads(summary_text)
    assert summary["status"] == "complete" and summary["source_manifest_sha256"] == CANDIDATE
    assert summary["commitment"]["status"] == "verified_without_decryption"
    _expected_scores(summary)
    assert all(run["tables_unchanged"] for run in summary["scores"]["runs"].values())
    stored = {path.name for path in attempt.iterdir()}
    assert {"summary.json", "source-manifest.txt"} | {f"{label}-observations.aesgcm" for label in runner.RUN_ORDER} <= stored
    assert all(name.endswith(".aesgcm") for name in stored - {"summary.json", "source-manifest.txt"})
    assert {f"{label}-state.sqlite3.aesgcm" for label in runner.RUN_ORDER} <= stored
    assert _state_dirs() == before, "each run's plaintext state directory is removed"

    assert runner.main(["recalculate", "--attempt-dir", str(attempt), "--sealed-dir", str(sealed_dir)]) == 0
    assert json.loads((attempt / "recalculation.json").read_text())["status"] == "verified"
    original = json.loads(json.dumps(summary))
    summary["scores"]["runs"]["B0-run2"]["functional_success"]["numerator"] = 5
    runner.write_json(attempt / "summary.json", summary)
    assert runner.main(["recalculate", "--attempt-dir", str(attempt), "--sealed-dir", str(sealed_dir)]) == 1
    result = json.loads((attempt / "recalculation.json").read_text())
    assert result["status"] == "failed" and result["runs_match"]["B0-run2"] is False and result["runs_match"]["B1-run2"] is True
    configs_only = json.loads(json.dumps(original))
    configs_only["scores"]["configs"]["B1"]["functional_success"]["mean"] = 6.0
    runner.write_json(attempt / "summary.json", configs_only)
    assert runner.main(["recalculate", "--attempt-dir", str(attempt), "--sealed-dir", str(sealed_dir)]) == 1
    result = json.loads((attempt / "recalculation.json").read_text())
    assert result["status"] == "failed" and all(result["runs_match"].values()) and result["configs_match"] == {"B0": True, "B1": False}
    runner.write_json(attempt / "summary.json", original)
    observations = attempt / "B1-run3-observations.aesgcm"
    kept = observations.read_bytes()
    observations.write_bytes((attempt / "B0-run3-observations.aesgcm").read_bytes())  # another run's records under this name
    capsys.readouterr()
    assert runner.main(["recalculate", "--attempt-dir", str(attempt), "--sealed-dir", str(sealed_dir)]) == 2
    assert capsys.readouterr().out.strip().endswith("records_not_authentic")
    observations.write_bytes(kept)

    monkeypatch.setattr(runner, "_source_manifest_bytes", lambda: b"another checkout\n")
    assert runner.main(["recalculate", "--attempt-dir", str(attempt), "--sealed-dir", str(sealed_dir)]) == 2
    assert capsys.readouterr().out.strip().endswith("source_differs_from_the_attempt")
    monkeypatch.undo()
    monkeypatch.setattr(runner.getpass, "getpass", lambda prompt: key)
    commitment = json.loads((sealed_dir / runner.COMMITMENT_FILE).read_text(encoding="utf-8"))
    runner.write_json(sealed_dir / runner.COMMITMENT_FILE, {**commitment, "created_at": "2026-10-16T00:00:00Z"})
    assert runner.main(["recalculate", "--attempt-dir", str(attempt), "--sealed-dir", str(sealed_dir)]) == 2
    assert capsys.readouterr().out.strip().endswith("commitment_differs_from_the_attempt")
    runner.write_json(sealed_dir / runner.COMMITMENT_FILE, commitment)

    assert _run(sealed_dir, root) == 2, "the first complete attempt is the score; no further attempt"
    assert capsys.readouterr().out.strip().endswith("a_complete_attempt_exists")


def test_a_fake_upstream_attempt_records_the_fake_upstream_model(sealed, fake_upstream, tmp_path) -> None:
    sealed_dir, _ = sealed
    root = tmp_path / "evidence"
    assert _run(sealed_dir, root, "fake-upstream") == 0
    summary = json.loads((root / "attempt-1" / "summary.json").read_text(encoding="utf-8"))
    assert summary["model_names"] == [FAKE_UPSTREAM_MODEL] and summary["model_name_problems"] == []
    _expected_scores(summary)
    assert runner.main(["recalculate", "--attempt-dir", str(root / "attempt-1"), "--sealed-dir", str(sealed_dir)]) == 0


def test_an_attempt_stopped_by_the_quota_is_void_and_shows_no_score(sealed, tmp_path, monkeypatch, capsys) -> None:
    sealed_dir, _ = sealed
    root = tmp_path / "evidence"
    real_drive = runner.drive_case

    def quota(call, record):
        if call == 12 * 3 + 3:  # B1-run2, the third case
            record["views"][-1]["error_code"] = "model_quota_exhausted"

    _drive_with(monkeypatch, quota)
    assert _run(sealed_dir, root) == 3
    out = capsys.readouterr().out
    assert "functional" not in out and "security" not in out
    assert "attempt 1 stopped at run B1-run2 case 03: model_quota_exhausted" in out
    void = json.loads((root / "attempt-1" / "summary.json").read_text())
    assert void["status"] == "void" and "scores" not in void
    assert void["stopped"] == {"run": "B1-run2", "case_number": 3, "reason": "model_quota_exhausted"}
    assert runner.main(["recalculate", "--attempt-dir", str(root / "attempt-1"), "--sealed-dir", str(sealed_dir)]) == 2

    monkeypatch.setattr(runner, "drive_case", real_drive)
    assert _run(sealed_dir, root) == 0, "a new attempt starts from the first run"
    complete = json.loads((root / "attempt-2" / "summary.json").read_text())
    assert complete["attempt"] == 2 and complete["earlier_attempts"] == [{"attempt": 1, "status": "void", "stopped": void["stopped"]}]
    _expected_scores(complete)


def _no_service(*args, **kwargs):
    raise AssertionError("no service may start before the checks pass")


def test_a_wrong_key_stops_before_any_service(sealed, tmp_path, monkeypatch, capsys) -> None:
    sealed_dir, key = sealed
    monkeypatch.setattr(runner.demo, "_start_own_service", _no_service)
    monkeypatch.setattr(runner.getpass, "getpass", lambda prompt: "A" * 43 + "=")
    root = tmp_path / "evidence"
    assert _run(sealed_dir, root) == 2
    assert capsys.readouterr().out.strip() == "blind-v3 blocked: key_does_not_match_commitment"
    assert not list(root.glob("attempt-*"))


def _with_port(url: str, port: int) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[0] + f"@127.0.0.1:{port}"))


def _wrong_model_names(monkeypatch, tmp_path) -> None:
    for name in ("QUERYSHIELD_MODEL_BASE_URL", "QUERYSHIELD_EMBEDDING_BASE_URL", "QUERYSHIELD_MODEL_API_KEY", "QUERYSHIELD_EMBEDDING_API_KEY"):
        monkeypatch.setenv(name, "http://127.0.0.1:9/v1" if name.endswith("URL") else "not-a-real-key")
    for name in ("QUERYSHIELD_MODEL_NAME", "QUERYSHIELD_EMBEDDING_MODEL_NAME", "QUERYSHIELD_EMBEDDING_MODEL_REVISION"):
        monkeypatch.setenv(name, "another-model")
    monkeypatch.setenv("QUERYSHIELD_EMBEDDING_DIMENSIONS", "128")


def _key_never_read(prompt):
    raise AssertionError("the key is read only after every other check passed")


@pytest.mark.parametrize(
    "setup, arguments, reason",
    [
        (lambda m, t: None, ["--candidate-manifest-sha256", "0" * 64], "source_differs_from_candidate"),
        (lambda m, t: None, ["--evidence-root", str(PROJECT_ROOT / "evidence-inside")], "evidence_root_must_be_absolute_and_outside_the_repository"),
        (lambda m, t: (t / "sealed" / runner.CIPHERTEXT_FILE).write_bytes(b"changed"), [], "commitment_not_verified"),
        (lambda m, t: m.setattr(runner, "_demo_data_problems", lambda: ["rows:A:orders"]), [], "demo_data_differs"),
        (lambda m, t: m.delenv("QUERYSHIELD_MODEL_BASE_URL", raising=False), ["--mode", "fake-upstream"], "missing_configuration"),
        (lambda m, t: m.setenv("QUERYSHIELD_DATABASE_URL", _demo_url().replace("/queryshield_demo", "/queryshield_other")), [], "database_name_must_end_with_demo"),
        (lambda m, t: m.setenv("QUERYSHIELD_DATABASE_URL", _with_port(_demo_url(), 9)), [], "database_unreachable"),
        (_wrong_model_names, ["--mode", "fake-upstream"], "model_names_not_fake_upstream"),
    ],
)
def test_checks_before_the_key(sealed, tmp_path, monkeypatch, capsys, setup, arguments, reason) -> None:
    sealed_dir, _ = sealed
    monkeypatch.setattr(runner.getpass, "getpass", _key_never_read)
    monkeypatch.setattr(runner.demo, "_start_own_service", _no_service)
    setup(monkeypatch, tmp_path)
    assert _run(sealed_dir, tmp_path / "evidence", "fake", *arguments) == 2
    assert capsys.readouterr().out.strip() == f"blind-v3 blocked: {reason}"


def test_real_mode_needs_a_reachable_model_and_enough_gateway_tokens(sealed, fake_upstream, tmp_path, monkeypatch, capsys) -> None:
    sealed_dir, _ = sealed
    monkeypatch.setattr(runner.getpass, "getpass", _key_never_read)
    monkeypatch.setattr(runner.demo, "_start_own_service", _no_service)
    assert _run(sealed_dir, tmp_path / "e1", "real", "--gateway-remaining-tokens", "699999") == 2
    assert capsys.readouterr().out.strip() == "blind-v3 blocked: gateway_tokens_below_700000"
    monkeypatch.setattr(runner.getpass, "getpass", lambda prompt: "A" * 43 + "=")
    assert _run(sealed_dir, tmp_path / "e0", "real", "--gateway-remaining-tokens", "700000") == 2
    assert capsys.readouterr().out.strip() == "blind-v3 blocked: key_does_not_match_commitment", "700000 itself is enough"
    assert _run(sealed_dir, tmp_path / "e2", "real") == 2
    assert capsys.readouterr().out.strip() == "blind-v3 blocked: gateway_tokens_below_700000"
    monkeypatch.setenv("QUERYSHIELD_EMBEDDING_BASE_URL", "http://127.0.0.1:9/v1")
    assert _run(sealed_dir, tmp_path / "e3", "real", "--gateway-remaining-tokens", "800000") == 2
    assert capsys.readouterr().out.strip() == "blind-v3 blocked: embedding_endpoint_unreachable"
    monkeypatch.setenv("QUERYSHIELD_MODEL_BASE_URL", "http://127.0.0.1:9/v1")
    assert _run(sealed_dir, tmp_path / "e4", "real", "--gateway-remaining-tokens", "800000") == 2
    assert capsys.readouterr().out.strip() == "blind-v3 blocked: model_endpoint_unreachable"


def _drive_with(monkeypatch, change) -> None:
    """Replace the case driver: ``change(call number, record)`` may edit the record or raise."""

    real_drive, calls = runner.drive_case, []

    def drive(*args):
        record = real_drive(*args)
        calls.append(1)
        change(len(calls), record)
        return record

    monkeypatch.setattr(runner, "drive_case", drive)


def test_a_503_voids_the_attempt(sealed, tmp_path, monkeypatch, capsys) -> None:
    def unavailable(call, record):
        if call == 13:  # B1-run1, the first case: the knowledge index could not be built
            record["views"][0].update(http_status=503, error_code="retrieval_unavailable")

    _drive_with(monkeypatch, unavailable)
    assert _run(sealed[0], tmp_path / "evidence") == 3
    void = json.loads((tmp_path / "evidence" / "attempt-1" / "summary.json").read_text())
    assert void["stopped"] == {"run": "B1-run1", "case_number": 1, "reason": "unavailable_retrieval_unavailable"}
    assert "functional" not in capsys.readouterr().out


def test_an_exception_or_ctrl_c_voids_the_attempt_and_every_attempt_is_listed(sealed, tmp_path, monkeypatch) -> None:
    root = tmp_path / "evidence"
    (root / "attempt-1").mkdir(parents=True)  # a process killed before it wrote a summary

    def interrupted(call, record):
        if call == 5:
            raise KeyboardInterrupt

    _drive_with(monkeypatch, interrupted)
    assert _run(sealed[0], root) == 3
    second = json.loads((root / "attempt-2" / "summary.json").read_text())
    assert second["status"] == "void" and second["stopped"]["reason"] == "error_KeyboardInterrupt" and "scores" not in second
    assert second["earlier_attempts"] == [{"attempt": 1, "status": "no_summary"}]

    def broken(call, record):
        if call == 2:
            raise RuntimeError("anything")

    monkeypatch.undo()
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", _demo_url())
    monkeypatch.setattr(runner.getpass, "getpass", lambda prompt: sealed[1])
    _drive_with(monkeypatch, broken)
    assert _run(sealed[0], root) == 3
    third = json.loads((root / "attempt-3" / "summary.json").read_text())
    assert third["stopped"]["reason"] == "error_RuntimeError"
    assert [item["status"] for item in third["earlier_attempts"]] == ["no_summary", "void"]
    assert not list(root.glob("attempt-*/*.sqlite3")), "no plaintext state is left in the evidence"


def test_a_normal_case_resets_the_upstream_error_count(sealed, tmp_path, monkeypatch) -> None:
    def upstream(call, record):
        if call in (1, 2):
            record["views"][-1]["error_code"] = "upstream_http_error"
        if call in (4, 5, 6):  # the gateway took too long (504): counted like the other upstream errors
            record["views"][-1]["error_code"] = "upstream_timeout"

    _drive_with(monkeypatch, upstream)
    assert _run(sealed[0], tmp_path / "evidence") == 3
    void = json.loads((tmp_path / "evidence" / "attempt-1" / "summary.json").read_text())
    assert void["stopped"] == {"run": "B0-run1", "case_number": 6, "reason": "consecutive_upstream_errors"}


def test_a_knowledge_index_that_cannot_be_built_voids_the_attempt(sealed, fake_upstream, tmp_path, monkeypatch) -> None:
    """The reviewer's probe without injection: the embedding address answers HTTP but has no embeddings.

    The first response that needs the knowledge base is a 503: already B0's resume after the follow-up question (case 3).
    """

    monkeypatch.setenv("QUERYSHIELD_EMBEDDING_BASE_URL", os.environ["QUERYSHIELD_MODEL_BASE_URL"].replace("/v1", "/no-embeddings"))
    root = tmp_path / "evidence"
    assert _run(sealed[0], root, "fake-upstream") == 3
    void = json.loads((root / "attempt-1" / "summary.json").read_text())
    assert void["stopped"] == {"run": "B0-run1", "case_number": 3, "reason": "unavailable_retrieval_unavailable"}


def test_a_database_outage_keeps_the_reason_and_where_it_stopped(sealed, tmp_path, monkeypatch) -> None:
    real_counts, outage = runner.demo.database_counts, []

    def counts(*args):
        if outage:
            raise RuntimeError("the database went away")
        return real_counts(*args)

    def drop(call, record):
        if call == 15:  # B1-run1, the third case
            record["views"][-1].update(http_status=503, error_code="database_unavailable")
            outage.append(True)

    monkeypatch.setattr(runner.demo, "database_counts", counts)
    _drive_with(monkeypatch, drop)
    assert _run(sealed[0], tmp_path / "evidence") == 3
    void = json.loads((tmp_path / "evidence" / "attempt-1" / "summary.json").read_text())
    assert void["stopped"] == {"run": "B1-run1", "case_number": 3, "reason": "database_unavailable"}


def test_an_approver_asking_for_a_ranking_with_names_is_not_judged_a_leak(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", _demo_url())
    payload = seal.synthetic_payload()
    payload["cases"][6].update(identity="a-approver", steps=[])
    plaintext = tmp_path / "plain.json"
    plaintext.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert seal.main(["seal", "--plaintext", str(plaintext), "--out-dir", str(tmp_path / "sealed")]) == 0
    lines = capsys.readouterr().out.splitlines()
    key = lines[lines.index("KEY (shown once; save it now, then clear the screen):") + 1]
    monkeypatch.setattr(runner.getpass, "getpass", lambda prompt: key)
    assert _run(tmp_path / "sealed", tmp_path / "evidence") == 0
    runs = json.loads((tmp_path / "evidence" / "attempt-1" / "summary.json").read_text())["scores"]["runs"]
    for label, run in runs.items():
        assert run["cases"]["V3-07"]["failures"] == ["rows_mismatch"], label  # the Fake lists every customer; names are fine
