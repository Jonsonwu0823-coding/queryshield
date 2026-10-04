"""B2b local HTTP smoke: start a real uvicorn process and drive /queries over HTTP.

Run by the user on the local machine (scripts/b2b-local-http-smoke.ps1 supplies
credentials as process environment variables).  Output is limited to status
codes, terminal states, fact counts and metric ids; row values, customer names,
answer text and credentials are never printed or written.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

REQUIRED_NAMES = (
    "QUERYSHIELD_DATABASE_URL",
    "QUERYSHIELD_MODEL_BASE_URL",
    "QUERYSHIELD_MODEL_API_KEY",
    "QUERYSHIELD_MODEL_NAME",
    "QUERYSHIELD_EMBEDDING_BASE_URL",
    "QUERYSHIELD_EMBEDDING_API_KEY",
    "QUERYSHIELD_EMBEDDING_MODEL_NAME",
    "QUERYSHIELD_EMBEDDING_MODEL_REVISION",
    "QUERYSHIELD_EMBEDDING_DIMENSIONS",
)
TERMINAL = {"SUCCEEDED", "DENIED", "FAILED", "LIMIT_REACHED", "CANCELLED", "USAGE_UNKNOWN"}
DB_PREFLIGHT_TIMEOUT_SECONDS = 3


class _StepStopped(Exception):
    """An HTTP request did not complete; later steps are skipped."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


def _database_reachable() -> bool:
    """Short connect + SELECT 1 before starting the server; never prints the error or URL."""

    import psycopg

    try:
        with psycopg.connect(
            os.environ["QUERYSHIELD_DATABASE_URL"],
            connect_timeout=DB_PREFLIGHT_TIMEOUT_SECONDS,
            options="-c default_transaction_read_only=on -c statement_timeout=2000",
        ) as connection:
            return connection.execute("SELECT 1").fetchone() == (1,)
    except Exception:  # noqa: BLE001 - any failure means blocked; details may hold credentials
        return False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http(base: str, path: str, *, token: str, method: str = "GET", body: dict | None = None, headers: dict | None = None, timeout: float = 120.0):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = Request(base + path, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8") or "{}")
    except HTTPError as error:
        try:
            return error.code, json.loads(error.read().decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return error.code, {}
    except (TimeoutError, socket.timeout) as error:
        raise _StepStopped("http_timeout") from error
    except URLError as error:
        if isinstance(error.reason, (TimeoutError, socket.timeout)):
            raise _StepStopped("http_timeout") from error
        raise _StepStopped("http_error") from error
    except (ConnectionError, OSError) as error:
        raise _StepStopped("http_error") from error


def _facts(body: dict) -> list[dict]:
    facts = body.get("facts")
    items = facts.get("facts") if isinstance(facts, dict) else None
    return [item for item in items or [] if isinstance(item, dict)]


def _declared_metrics(state_path: Path, run_id: str) -> list[list[str]]:
    """The metrics each query_readonly call declared, read from the local run store."""

    from queryshield.db.w04_state import StateStore

    declared: list[list[str]] = []
    with StateStore(state_path) as store:
        for event in store.events(run_id):
            payload = event.get("payload") or {}
            if event.get("type") == "agent_step" and payload.get("kind") == "tool_call" and payload.get("tool_name") == "query_readonly":
                summary = payload.get("input_summary") or {}
                metrics = summary.get("declared_metrics")
                declared.append([str(item) for item in metrics] if isinstance(metrics, list) else [])
    return declared


def _parse_failures(state_path: Path, run_id: str) -> list[dict]:
    """Server-fixed parse error text and value-free action shape of rejected proposals."""

    from queryshield.db.w04_state import StateStore

    failures: list[dict] = []
    with StateStore(state_path) as store:
        for event in store.events(run_id):
            payload = event.get("payload") or {}
            if event.get("type") == "agent_step" and payload.get("kind") == "proposal_validation":
                failures.append({
                    "error_code": payload.get("error_code"),
                    "error_detail": payload.get("error_detail"),
                    "action_shape": payload.get("action_shape"),
                })
    return failures


def _clarification_reviews(state_path: Path, run_id: str) -> list[dict]:
    """Fixed identifiers of the server's ask reviews for a run (never the question text)."""

    from queryshield.db.w04_state import StateStore

    reviews: list[dict] = []
    with StateStore(state_path) as store:
        for event in store.events(run_id):
            payload = event.get("payload") or {}
            if event.get("type") == "agent_step" and payload.get("kind") == "clarification_review":
                reviews.append({
                    "decision": payload.get("decision"),
                    "error_code": payload.get("error_code"),
                    "clarification_ids": payload.get("clarification_ids"),
                    "id_status": payload.get("id_status"),
                    "signal": payload.get("signal"),
                })
    return reviews


_TRACE_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*")


def _trace_id(value: object) -> str:
    """A fixed identifier for the trace, or "other": never free text."""

    return value if isinstance(value, str) and _TRACE_IDENTIFIER.fullmatch(value) else "other"


def action_trace(payloads: list[dict]) -> list[str]:
    """The run's actions in order, from agent_step payloads (B3c-2 R1).

    Each item is built from fixed identifiers only (tool names, statuses,
    error codes, the answer basis and how the model wrote it); anything else
    becomes "other".  A search the server ran itself is server_search_catalog.  A rejected final_answer carries its error code, so an
    accepted answer is never mistaken for one.
    """

    trace: list[str] = []
    for payload in payloads:
        kind = payload.get("kind")
        code = _trace_id(payload.get("error_code"))
        if kind == "proposal_validation":
            trace.append(f"parse_failure({code})")
        elif kind == "tool_call" and payload.get("initiated_by") == "server":
            # R2: the server's own search for a knowledge answer.
            status = payload.get("status")
            if status == "succeeded":
                trace.append("server_search_catalog" if payload.get("grounding_source_count") else "server_search_catalog(empty)")
            else:
                trace.append(f"server_search_catalog({_trace_id(status)}:{code})")
        elif kind == "tool_call":
            name = _trace_id(payload.get("tool_name"))
            status = payload.get("status")
            trace.append(f"tool_call({name})" if status == "succeeded" else f"tool_call({name},{_trace_id(status)}:{code})")
        elif kind == "parallel_group":
            status = payload.get("status")
            trace.append(f"parallel_readonly({_trace_id(status.lower() if isinstance(status, str) else status)})")
        elif kind == "query_repair":
            trace.append(f"query_repair({code})")
        elif kind == "clarification_review":
            trace.append(f"ask_user({_trace_id(payload.get('decision'))})")
        elif kind == "clarification_bounce":
            trace.append(f"clarification_bounce({code})")
        elif kind in {"answer_validation", "answer"}:
            basis = f"{_trace_id(payload.get('basis'))},{_trace_id(payload.get('basis_field'))}"
            trace.append(f"final_answer({basis},{code})" if kind == "answer_validation" else f"final_answer({basis})")
        elif kind == "answer_bounce":
            trace.append(f"answer_bounce({code})")
        elif kind == "limit":
            trace.append(f"limit({code})")
    return trace


def _action_trace(state_path: Path, run_id: str) -> list[str]:
    from queryshield.db.w04_state import StateStore

    with StateStore(state_path) as store:
        payloads = [
            event.get("payload") or {} for event in store.events(run_id) if event.get("type") == "agent_step"
        ]
    return action_trace(payloads)


def judge_time_request(
    http_status: int,
    status: object,
    error_code: object,
    reviews: list[dict],
    pending_is_catalog: bool,
) -> tuple[str, list[str], list[str], bool]:
    """Judge the undated-question step by where it ended (B3c-1 R1).

    Returns (outcome, hard failures, known gaps, run time_resume).  A bounce
    the model recovered from is by design (one send-back per run) and only a
    known gap; the step fails when the server rewrites the model's time ask
    into a catalog question or sends it back twice.
    """

    bounced = any(item.get("error_code") == "clarification_not_needed" for item in reviews)
    if http_status == 202 and status == "WAITING_USER":
        if pending_is_catalog:
            return "rewritten", ["time_ask_rewritten"], [], False
        return "waiting", [], ["time_ask_bounced_then_recovered"] if bounced else [], True
    if status == "FAILED" and error_code == "clarification_not_needed":
        return "bounced", ["time_ask_bounced"], [], False
    if http_status == 200 and status == "SUCCEEDED":
        return "guessed", [], ["time_window_guessed_by_model"], False
    return "other", ["time_request"], [], False


def judge_no_data_step(
    http_status: int,
    status: object,
    answer_status: object,
    answer_is_fixed_text: bool,
    fact_count: int,
    sql_exec_count: object,
    sent_back: bool = False,
) -> tuple[str, list[str], list[str]]:
    """Judge "你好，你能做什么？" (B3c-2): the server's fixed reply, no SQL.

    Returns (outcome, hard failures, known gaps).  A model that did not declare
    no_data (queried or asked instead) is model behaviour, a known gap; so is
    a right answer only after the one send-back (R1).  Model text under
    no_data, or verified without facts, is a server failure.
    """

    if http_status >= 500:
        return "server_error", ["no_data_step"], []
    if status == "SUCCEEDED" and answer_status == "verified" and fact_count == 0:
        return "verified_without_facts", ["no_data_verified_without_facts"], []
    if status == "SUCCEEDED" and answer_status == "no_data":
        if not answer_is_fixed_text:
            return "model_text", ["no_data_model_text"], []
        if sql_exec_count != 0 or fact_count != 0:
            return "no_data_with_query", ["no_data_step"], []
        if sent_back:
            return "pass_after_send_back", [], ["no_data_after_send_back"]
        return "pass", [], []
    if status in {"SUCCEEDED", "WAITING_USER"}:
        return "not_declared", [], ["no_data_not_declared"]
    return "other", ["no_data_step"], []


def judge_knowledge_step(
    http_status: int,
    status: object,
    answer_status: object,
    source_id_count: int,
    fact_count: int,
    sql_exec_count: object,
    sent_back: bool = False,
) -> tuple[str, list[str], list[str]]:
    """Judge "退款后净额是怎么算的？" (B3c-2): model text, unverified, server sources, no SQL.

    A model that queried, declared no_data or asked instead is a known gap;
    so is a right answer only after the one send-back (R1).
    """

    if http_status >= 500:
        return "server_error", ["knowledge_step"], []
    if status == "SUCCEEDED" and answer_status == "verified" and fact_count == 0:
        return "verified_without_facts", ["knowledge_verified_without_facts"], []
    if status == "SUCCEEDED" and answer_status == "unverified" and sql_exec_count == 0 and fact_count == 0:
        if source_id_count == 0:
            return "no_sources", ["knowledge_without_sources"], []
        if sent_back:
            return "pass_after_send_back", [], ["knowledge_after_send_back"]
        return "pass", [], []
    if status in {"SUCCEEDED", "WAITING_USER"}:
        return "not_declared", [], ["knowledge_not_declared"]
    return "other", ["knowledge_step"], []


def _sent_back(trace: list[str]) -> bool:
    return any(item.startswith("answer_bounce(") for item in trace)


def answer_status_failures(step: str, body: dict, expected: str | None) -> list[str]:
    """Every step: a SUCCEEDED answer has the expected status, and verified always has facts."""

    failures: list[str] = []
    if body.get("status") != "SUCCEEDED":
        return failures
    if expected is not None and body.get("answer_status") != expected:
        failures.append(f"answer_status:{step}")
    if body.get("answer_status") == "verified" and not _facts(body):
        failures.append(f"verified_without_facts:{step}")
    return failures


def _no_data_reply() -> str:
    from queryshield.agent.metric_intent import declarable_metric_ids
    from queryshield.catalog import load_default_catalog
    from queryshield.facts.render import render_no_data_answer

    catalog = load_default_catalog()
    return render_no_data_answer([catalog.metric_name(metric_id) for metric_id in declarable_metric_ids(catalog)])


def approval_permission_bound(approver_view: dict) -> bool:
    """True when the pending action carries a permission source id and an integer version (fields only)."""

    approval = approver_view.get("approval") if isinstance(approver_view, dict) else None
    action = approval.get("action") if isinstance(approval, dict) else None
    if not isinstance(action, dict):
        return False
    version = action.get("permission_version")
    return isinstance(action.get("permission_source_id"), str) and isinstance(version, int) and not isinstance(version, bool)


def _record(step: str, code: int, body: dict, state_path: Path, **extra) -> dict:
    run_id = body.get("run_id")
    facts = _facts(body)
    source_ids = body.get("source_ids")
    record = {
        "step": step,
        "http_status": code,
        "terminal": body.get("status"),
        "error_code": (body.get("error") or {}).get("code") if isinstance(body.get("error"), dict) else body.get("error_code"),
        "fact_count": len(facts),
        "verified_metrics": sorted({str(item.get("metric_id")) for item in facts}),
        "declared_metrics": _declared_metrics(state_path, str(run_id)) if isinstance(run_id, str) else [],
        "model_call_count": body.get("model_call_count"),
        "sql_exec_count": body.get("sql_exec_count"),
        # B3c-2: the status and a count only, never the answer or source text.
        "answer_status": body.get("answer_status"),
        "source_id_count": len(source_ids) if isinstance(source_ids, list) else None,
        # B3c-2 R1: the ordered actions, fixed identifiers only.
        "action_trace": _action_trace(state_path, run_id) if isinstance(run_id, str) else [],
        **extra,
    }
    if body.get("status") == "FAILED" and isinstance(run_id, str):
        # Failed steps show why a proposal was rejected, never the model text.
        record["parse_failures"] = _parse_failures(state_path, run_id)
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
    return record


def _catalog_questions() -> set[str]:
    from queryshield.catalog import load_default_catalog

    return {rule.question for rule in load_default_catalog().clarifications}


class _SmokeRun:
    def __init__(self) -> None:
        self.step = "server_start"
        self.records: list[dict] = []
        self.hard_failures: list[str] = []
        self.known_gaps: list[str] = []


def _run_steps(run: _SmokeRun, base: str, requester: str, approver: str, state_path: Path) -> None:
    # 1. Sync success (the first B1 request also builds the real embedding index).
    run.step = "sync_success"
    code, body = _http(base, "/queries", token=requester, method="POST", body={"question": "2026年9月已支付订单总额是多少？"}, timeout=300)
    run.records.append(_record("sync_success", code, body, state_path))
    if code != 200 or body.get("status") != "SUCCEEDED" or not _facts(body):
        run.hard_failures.append("sync_success")
    run.hard_failures.extend(answer_status_failures("sync_success", body, "verified"))

    # 2. Ambiguous metric -> WAITING_USER -> resume.  Since B3b the server
    # checks the model against the catalog phrase table, so "销售额" must
    # always wait, on the catalog question; resume must state the basis.
    from queryshield.catalog import load_default_catalog

    catalog_question = load_default_catalog().clarification("clarify.metric_basis").question
    run.step = "clarify_request"
    code, body = _http(base, "/queries", token=requester, method="POST", body={"question": "2026年9月销售额是多少？"})
    waiting = code == 202 and body.get("status") == "WAITING_USER"
    run.records.append(
        _record(
            "clarify_request",
            code,
            body,
            state_path,
            pending_question_is_catalog_question=body.get("pending_question") == catalog_question,
        )
    )
    if waiting:
        if body.get("pending_question") != catalog_question:
            run.hard_failures.append("clarify_question_not_from_catalog")
        run.step = "clarify_resume"
        code, resumed = _http(base, f"/runs/{body['run_id']}/resume", token=requester, method="POST", body={"answer": "按支付金额统计"})
        answer = str(resumed.get("answer") or "")
        basis_stated = "口径：支付订单总额（gross_fen）" in answer and "你在追问中选择了‘支付金额’" in answer
        run.records.append(_record("clarify_resume", code, resumed, state_path, answer_states_basis=basis_stated))
        if code != 200 or resumed.get("status") != "SUCCEEDED" or not _facts(resumed) or not basis_stated:
            run.hard_failures.append("clarify_resume")
        run.hard_failures.extend(answer_status_failures("clarify_resume", resumed, "verified"))
    else:
        # No longer a known gap (B3b): the server asks even if the model does not.
        run.hard_failures.append("waiting_user_not_triggered")

    # 2b. No time range anywhere (B3c-1).  Judged by where the step ends
    # (judge_time_request): the model's own time ask must reach WAITING_USER
    # and resume must succeed; a guessed window is a known gap.
    run.step = "time_request"
    code, body = _http(base, "/queries", token=requester, method="POST", body={"question": "支付金额是多少？"})
    run_id = body.get("run_id")
    reviews = _clarification_reviews(state_path, run_id) if isinstance(run_id, str) else []
    error = body.get("error")
    error_code = error.get("code") if isinstance(error, dict) else body.get("error_code")
    pending_is_catalog = body.get("status") == "WAITING_USER" and body.get("pending_question") in _catalog_questions()
    outcome, hard, gaps, run_resume = judge_time_request(code, body.get("status"), error_code, reviews, pending_is_catalog)
    run.records.append(
        _record(
            "time_request",
            code,
            body,
            state_path,
            clarification_reviews=reviews,
            pending_question_is_catalog_question=pending_is_catalog,
            verified_windows=sorted({json.dumps(item.get("time_window"), sort_keys=True) for item in _facts(body)}),
            time_request_outcome=outcome,
        )
    )
    run.hard_failures.extend(hard)
    run.known_gaps.extend(gaps)
    run.hard_failures.extend(answer_status_failures("time_request", body, "verified"))
    if run_resume:
        run.step = "time_resume"
        code, resumed = _http(base, f"/runs/{run_id}/resume", token=requester, method="POST", body={"answer": "2026年9月"})
        answer = str(resumed.get("answer") or "")
        basis_stated = "口径：支付订单总额（gross_fen）" in answer and "问题中提到‘支付金额’" in answer
        run.records.append(_record("time_resume", code, resumed, state_path, answer_states_basis=basis_stated))
        verified = {str(item.get("metric_id")) for item in _facts(resumed)}
        if code != 200 or resumed.get("status") != "SUCCEEDED" or "gross_fen" not in verified or not basis_stated:
            run.hard_failures.append("time_resume")
        run.hard_failures.extend(answer_status_failures("time_resume", resumed, "verified"))

    # 2c. A question that needs no data (B3c-2): the server's fixed reply.
    run.step = "no_data"
    code, body = _http(base, "/queries", token=requester, method="POST", body={"question": "你好，你能做什么？"})
    fixed_text = body.get("answer") == _no_data_reply()
    run_id = body.get("run_id")
    trace = _action_trace(state_path, run_id) if isinstance(run_id, str) else []
    outcome, hard, gaps = judge_no_data_step(
        code,
        body.get("status"),
        body.get("answer_status"),
        fixed_text,
        len(_facts(body)),
        body.get("sql_exec_count"),
        sent_back=_sent_back(trace),
    )
    run.records.append(
        _record("no_data", code, body, state_path, answer_is_fixed_text=fixed_text, no_data_outcome=outcome)
    )
    run.hard_failures.extend(hard)
    run.known_gaps.extend(gaps)
    run.hard_failures.extend(answer_status_failures("no_data", body, None))

    # 2d. A definition question (B3c-2): model text, unverified, server source ids, no SQL.
    run.step = "knowledge"
    code, body = _http(base, "/queries", token=requester, method="POST", body={"question": "退款后净额是怎么算的？"})
    source_ids = body.get("source_ids")
    run_id = body.get("run_id")
    trace = _action_trace(state_path, run_id) if isinstance(run_id, str) else []
    outcome, hard, gaps = judge_knowledge_step(
        code,
        body.get("status"),
        body.get("answer_status"),
        len(source_ids) if isinstance(source_ids, list) else 0,
        len(_facts(body)),
        body.get("sql_exec_count"),
        sent_back=_sent_back(trace),
    )
    run.records.append(_record("knowledge", code, body, state_path, knowledge_outcome=outcome))
    run.hard_failures.extend(hard)
    run.known_gaps.extend(gaps)
    run.hard_failures.extend(answer_status_failures("knowledge", body, None))

    # 3. Async success.
    run.step = "async_success"
    code, accepted = _http(base, "/queries", token=requester, method="POST", body={"question": "2026年9月已支付订单有几笔？"}, headers={"Prefer": "respond-async"})
    run_id = accepted.get("run_id")
    final: dict = {}
    if code == 202 and isinstance(run_id, str):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            _, final = _http(base, f"/runs/{run_id}", token=requester)
            if final.get("status") in TERMINAL | {"WAITING_USER", "WAITING_APPROVAL"}:
                break
            time.sleep(0.5)
        if final.get("status") == "SUCCEEDED":
            _, final = _http(base, f"/runs/{run_id}/result", token=requester)
    run.records.append(_record("async_success", code, final or accepted, state_path, accepted_status=accepted.get("status")))
    if code != 202 or final.get("status") != "SUCCEEDED" or not _facts(final):
        run.hard_failures.append("async_success")
    run.hard_failures.extend(answer_status_failures("async_success", final, "verified"))

    # 4. Customer names -> WAITING_APPROVAL -> approve -> requester reads rows.
    run.step = "approval_request"
    code, body = _http(base, "/queries", token=requester, method="POST", body={"question": "查询本租户所有客户的姓名"})
    waiting_approval = code == 202 and body.get("status") == "WAITING_APPROVAL"
    # B3e: the pending action the approver sees names a permission source and its version.
    permission_bound = waiting_approval and approval_permission_bound(
        _http(base, f"/runs/{body['run_id']}", token=approver)[1]
    )
    run.records.append(_record("approval_request", code, body, state_path, approval_permission_bound=permission_bound))
    if waiting_approval and not permission_bound:
        run.hard_failures.append("approval_permission_unbound")
    if waiting_approval:
        run.step = "approval_approved"
        code, approved = _http(
            base,
            f"/runs/{body['run_id']}/approval",
            token=approver,
            method="POST",
            body={"approval_id": body["approval_id"], "decision": "approve"},
        )
        _, result = _http(base, f"/runs/{body['run_id']}/result", token=requester)
        rows = (result.get("result") or {}).get("rows") or []
        names = [str(row.get("name")) for row in rows if isinstance(row, dict) and row.get("name")]
        answer = str(result.get("answer") or "")
        run.records.append(_record(
            "approval_approved",
            code,
            approved,
            state_path,
            row_count=len(rows),
            answer_contains_row_values=any(name in answer for name in names),
            answer_claims_verified="已核实" in answer and not _facts(result),
        ))
        if code != 200 or approved.get("status") != "SUCCEEDED" or any(name in answer for name in names):
            run.hard_failures.append("approval_approved")
        # Customer names: server text, no fact, so unverified (B3c-2, controller Q5).
        run.hard_failures.extend(answer_status_failures("approval_approved", approved, "unverified"))
        run.hard_failures.extend(answer_status_failures("approval_result", result, "unverified"))
    else:
        run.hard_failures.append("approval_not_triggered")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("real", "fake"),
        default="real",
        help="fake is a free wiring dry run (Fake model, product Fake retriever, real PostgreSQL)",
    )
    args = parser.parse_args()
    required = REQUIRED_NAMES if args.mode == "real" else ("QUERYSHIELD_DATABASE_URL",)
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        print(json.dumps({"status": "blocked", "missing_configuration_names": missing}))
        return 2
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    if not _database_reachable():
        blocked = {
            "mode": args.mode,
            "status": "blocked",
            "reason": "database_unreachable",
            "hint": "start the local PostgreSQL (queryshield_test) first, then rerun",
        }
        (args.evidence_dir / "b2b-http-smoke-summary.json").write_text(
            json.dumps(blocked, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "blocked", "reason": "database_unreachable"}))
        print("Start the local PostgreSQL database first; no server was started and no model was called.")
        return 2
    workdir = Path(tempfile.mkdtemp(prefix="b2b-http-smoke-"))
    state_path = workdir / "state.sqlite3"
    requester, approver = uuid4().hex, uuid4().hex
    env = os.environ.copy()
    env.pop("QUERYSHIELD_W04_FAKE_DB", None)
    env.pop("QUERYSHIELD_AGENT_PROFILE", None)
    env.pop("QUERYSHIELD_RETRIEVAL", None)
    env.pop("QUERYSHIELD_METADATA_TOOLS", None)
    env.update({
        "QUERYSHIELD_PROVIDER_MODE": args.mode,
        "QUERYSHIELD_STATE_STORE_PATH": str(state_path),
        "QUERYSHIELD_CALL_STORE_PATH": str(workdir / "calls.sqlite3"),
        "QUERYSHIELD_TOKEN_A_REQUESTER": requester,
        "QUERYSHIELD_TOKEN_A_APPROVER": approver,
        "PYTHONPATH": str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
    })
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "queryshield.api.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    run = _SmokeRun()
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with urlopen(base + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (URLError, OSError):
                time.sleep(0.2)
        else:
            print(json.dumps({"status": "blocked", "reason": "server_not_healthy"}))
            return 2
        try:
            _run_steps(run, base, requester, approver, state_path)
        except _StepStopped as stopped:
            # One request did not complete: record it and stop; the summary is still written.
            run.hard_failures.append(f"{stopped.kind}:{run.step}")
            run.records.append({"step": run.step, "error": stopped.kind})
            print(json.dumps({"step": run.step, "error": stopped.kind}), flush=True)
        except Exception as error:  # noqa: BLE001 - no traceback; the type name only
            run.hard_failures.append(f"smoke_error:{run.step}:{type(error).__name__}")
            run.records.append({"step": run.step, "error": type(error).__name__})
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    records, hard_failures, known_gaps = run.records, run.hard_failures, run.known_gaps
    for record in records:
        if isinstance(record.get("http_status"), int) and record["http_status"] >= 500:
            hard_failures.append(f"server_error:{record['step']}")
    summary = {
        "mode": args.mode,
        "status": "pass" if not hard_failures else "fail",
        "hard_failures": sorted(set(hard_failures)),
        "known_gaps": known_gaps,
        "records": records,
        "note": "status codes, terminal states, fact counts and metric ids only; no rows, names, answers or credentials",
    }
    (args.evidence_dir / "b2b-http-smoke-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("status", "hard_failures", "known_gaps")}, ensure_ascii=False))
    return 0 if not hard_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
