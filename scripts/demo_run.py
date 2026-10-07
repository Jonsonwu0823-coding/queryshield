"""Demo run: start a real uvicorn process on the demo database and drive the demo questions over HTTP.

Fake mode (free): the Fake model answers the questions it knows; the others are
``not_applicable``.  Verified values are still compared with the expected answers,
so this is the product's own verification path computing the answers again.
Every question also gets one rule: each verified fact must equal the value the
generator computes for the question's tenant and the fact's metric and window.
Real mode (the user runs it locally, scripts/demo-local.ps1): the real model.

The server process gets ``QUERYSHIELD_DEMO_DATASET`` from this script, never from
the caller's environment; the database name must end with ``_demo``.

Evidence (two files in --evidence-dir):
  demo-summary.json  fixed fields and numbers only: question id, identity, HTTP
                         status, terminal state, answer_status, action trace, expected
                         and actual values, verdict.  No question text, answer text,
                         customer name, URL or credential.
  demo-raw.json      question text, answer text and row values (customer names).
                         Local use only; the controller does not open it.

Exit codes: 0 pass, 1 a hard failure, 2 blocked (configuration, database, server).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scripts import http_smoke as smoke  # noqa: E402  (reuses its HTTP helpers and judgements)
# Standard library only and imports nothing from queryshield: the independent values every
# verified fact is compared with never come from the product's SQL building.
from scripts import generate_demo_data as gen  # noqa: E402

QUESTIONS_PATH = PROJECT_ROOT / "fixtures" / "demo" / "demo-questions-v1.json"
DEMO_KNOWLEDGE_REGISTRY = PROJECT_ROOT / "fixtures" / "demo" / "knowledge" / "source_registry.json"
DEMO_DATASET = "commerce-demo-v1"
TOKEN_ENVIRONMENT = {
    "a-requester": "QUERYSHIELD_TOKEN_A_REQUESTER",
    "a-approver": "QUERYSHIELD_TOKEN_A_APPROVER",
    "b-requester": "QUERYSHIELD_TOKEN_B_REQUESTER",
    "b-approver": "QUERYSHIELD_TOKEN_B_APPROVER",
}
STATE_PATH_ENV = "QUERYSHIELD_STATE_STORE_PATH"
APPROVER_OF = {"a-requester": "a-approver", "b-requester": "b-approver"}
TENANT_OF = {"a-requester": "A", "b-requester": "B"}
# A model that answers without querying is a documented model-behaviour gap, not a server bug.
MODEL_BEHAVIOUR_ERROR_CODES = frozenset({"answer_not_grounded"})
_CUSTOMER_ID = re.compile(r"c\d{2,}")


# --- observation helpers (pure) ------------------------------------------------


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def row_ints(row: Mapping[str, object]) -> list[int]:
    """Integer cells of one row; digit strings count (numeric columns may arrive as text)."""

    values: list[int] = []
    for cell in row.values():
        if _is_int(cell):
            values.append(int(cell))
        elif isinstance(cell, str) and re.fullmatch(r"-?\d+", cell.strip()):
            values.append(int(cell.strip()))
    return values


def row_customer(row: Mapping[str, object], names: Mapping[str, str]) -> str | None:
    """The customer a row is about: a c## id cell, or a name cell found in ``names``."""

    for cell in row.values():
        if isinstance(cell, str) and _CUSTOMER_ID.fullmatch(cell.strip()):
            return cell.strip()
    for cell in row.values():
        if isinstance(cell, str) and cell in names:
            return names[cell]
    return None


def fact_values(facts: list[dict], metric_id: str) -> list[int]:
    return [int(item["value"]) for item in facts if item.get("metric_id") == metric_id and _is_int(item.get("value"))]


def fact_windows(facts: list[dict], metric_id: str) -> list[dict[str, str]]:
    result = []
    for item in facts:
        window = item.get("time_window")
        if item.get("metric_id") == metric_id and isinstance(window, Mapping):
            result.append({"start": str(window.get("start")), "end": str(window.get("end"))})
    return result


_DEMO_DATA: list = []


def demo_data():
    """The generator's in-memory rows (algorithm 1), built once per process."""

    if not _DEMO_DATA:
        _DEMO_DATA.append(gen.generate())
    return _DEMO_DATA[0]


def _utc_seconds(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(parsed.timestamp())


def verified_fact_counts(question: Mapping[str, object], obs: Mapping[str, object], data=None) -> tuple[int, int]:
    """(checked, mismatched) for every verified fact of one question (all questions).

    Each fact's value must equal the generator's own value for the question's tenant,
    the fact's metric and the fact's own time window.  A fact that cannot be recomputed
    (unknown metric, tenant or window) counts as a mismatch.
    """

    facts = obs.get("facts") or []
    tenant = TENANT_OF.get(str(question.get("identity")))
    checked = mismatched = 0
    for fact in facts:
        checked += 1
        window = fact.get("time_window") if isinstance(fact.get("time_window"), Mapping) else {}
        start, end = _utc_seconds(window.get("start")), _utc_seconds(window.get("end"))
        if tenant is None or start is None or end is None or not _is_int(fact.get("value")):
            mismatched += 1
            continue
        expected = gen.expected_metrics(data or demo_data(), tenant, start, end).get(str(fact.get("metric_id")))
        if expected is None or expected != fact["value"]:
            mismatched += 1
    return checked, mismatched


def max_output_tokens() -> int | None:
    """The run's output cap for the model: QUERYSHIELD_MODEL_MAX_TOKENS, default 512 (the adapter's default)."""

    raw = os.getenv("QUERYSHIELD_MODEL_MAX_TOKENS", "512").strip() or "512"
    return int(raw) if re.fullmatch(r"\d{1,5}", raw) else None


def calls_at_output_limit(completion_tokens: list, limit: int | None) -> int:
    """How many model calls used the whole output cap (their JSON may have been cut off)."""

    return sum(1 for value in completion_tokens if limit is not None and isinstance(value, int) and value >= limit)


def _outcome(**fields) -> dict:
    return {"hard_failures": [], "known_gaps": [], **fields}


def _common_failures(obs: Mapping[str, object]) -> tuple[list[str], list[str]]:
    """Checks every question gets: server errors, and verified answers without facts."""

    hard: list[str] = []
    gaps: list[str] = []
    http = obs.get("http_status")
    if isinstance(http, int) and http >= 500:
        if obs.get("error_code") in MODEL_BEHAVIOUR_ERROR_CODES:
            gaps.append(f"model_behaviour:{obs.get('error_code')}")
        else:
            hard.append(f"server_error:{obs.get('error_code') or http}")
    if obs.get("status") == "SUCCEEDED" and obs.get("answer_status") == "verified" and not obs.get("facts"):
        hard.append("verified_without_facts")
    return hard, gaps


def _observation_failures(obs: Mapping[str, object]) -> tuple[list[str], list[str]]:
    """Observation questions: a server error is only recorded; the one hard failure is verified without facts."""

    hard: list[str] = []
    gaps: list[str] = []
    http = obs.get("http_status")
    if isinstance(http, int) and http >= 500:
        gaps.append(f"observed_server_error:{obs.get('error_code') or http}")
    if obs.get("status") == "SUCCEEDED" and obs.get("answer_status") == "verified" and not obs.get("facts"):
        hard.append("verified_without_facts")
    return hard, gaps


# --- judgements (pure) ---------------------------------------------------------


def judge_metric(question: Mapping[str, object], obs: Mapping[str, object]) -> dict:
    """A verified scalar: the value and the window must equal the expected ones.

    A model that did not produce a verified fact (asked, answered without querying)
    is a known gap; a verified fact with a different value or window is a hard failure.
    """

    expected = question["expected"]
    metric_id, wanted = expected["metric_id"], expected["value"]
    hard, gaps = _common_failures(obs)
    facts = obs.get("facts") or []
    values = fact_values(facts, metric_id)
    outcome = _outcome(expected={"metric_id": metric_id, "value": wanted}, actual={"values": values})
    if values:
        if values != [wanted]:
            hard.append("value_mismatch")
        windows = fact_windows(facts, metric_id)
        if question.get("window") and any(w != question["window"] for w in windows):
            hard.append("window_mismatch")
        if obs.get("answer_status") != "verified":
            hard.append("answer_status")
    elif not hard:
        gaps.append(f"no_verified_fact:{obs.get('status')}")
    outcome["bounced"] = any(item.startswith("answer_bounce(") for item in obs.get("trace") or [])
    if question["kind"] == "empty_window" and outcome["bounced"] and not hard:
        gaps.append("empty_window_answered_after_bounce")  # The "query first" bounce
    outcome["hard_failures"], outcome["known_gaps"] = hard, gaps
    return outcome


def _rowset_stats(question: Mapping[str, object], obs: Mapping[str, object]) -> dict:
    """Compare returned rows with the true per-customer amounts (by customer id; aliases do not matter)."""

    expected = question["expected"]
    wanted = {row["customer_id"]: row["value"] for row in expected["rows"]}
    truth = {key: int(value) for key, value in (expected.get("all_values") or wanted).items()}
    names = {row["name"]: row["customer_id"] for row in expected["rows"]}
    rows = obs.get("rows") or []
    wrong = unknown = unparseable = 0
    seen: set[str] = set()
    for row in rows:
        customer = row_customer(row, names)
        ints = row_ints(row)
        if customer is None or not ints:
            unparseable += 1
        elif customer not in truth:
            unknown += 1
        else:
            seen.add(customer)
            if truth[customer] not in ints:
                wrong += 1
    return {
        "wanted": set(wanted), "rows": rows, "row_count": len(rows), "seen": seen, "wrong": wrong, "unknown": unknown,
        "unparseable": unparseable,
    }


def judge_rowset(question: Mapping[str, object], obs: Mapping[str, object]) -> dict:
    """The bounded row set (top-N customers): every returned customer's amount must be that customer's true amount.

    A wrong amount is a hard failure.  Missing or extra customers, unparseable rows or no rows
    (a model that copied too many rows, asked a question, ...) are known gaps.
    """

    hard, gaps = _common_failures(obs)
    stats = _rowset_stats(question, obs)
    if stats["wrong"]:
        hard.append("rowset_value_mismatch")
    if stats["rows"] and stats["unparseable"]:
        gaps.append("rowset_unparseable_rows")
    if stats["rows"] and (stats["unknown"] or stats["seen"] != stats["wanted"]) and not stats["wrong"]:
        gaps.append("rowset_customers_differ")
    if not stats["rows"] and not hard:
        gaps.append(f"no_rows:{obs.get('status')}")
    return _outcome(
        expected={"customers": len(stats["wanted"])},
        actual={
            "row_count": stats["row_count"], "matched": len(stats["seen"]) - stats["wrong"], "wrong": stats["wrong"],
            "unknown": stats["unknown"], "unparseable": stats["unparseable"], "expected_customers_found": len(stats["seen"] & stats["wanted"]),
        },
        hard_failures=hard,
        known_gaps=gaps,
    )


def judge_observe_rowset(question: Mapping[str, object], obs: Mapping[str, object]) -> dict:
    """Q06b, the full per-customer summary: only recorded (rows are copied by the model and limited by its output cap)."""

    hard, gaps = _observation_failures(obs)
    stats = _rowset_stats(question, obs)
    if stats["wrong"]:
        gaps.append("observed_rowset_value_mismatch")
    if stats["rows"] and (stats["unparseable"] or stats["unknown"] or stats["seen"] != stats["wanted"]):
        gaps.append("observed_rowset_incomplete")
    if not stats["rows"]:
        gaps.append(f"observed_no_rows:{obs.get('status')}")
    return _outcome(
        expected={"customers": len(stats["wanted"])},
        actual={
            "row_count": stats["row_count"], "matched": len(stats["seen"]) - stats["wrong"], "wrong": stats["wrong"],
            "unknown": stats["unknown"], "unparseable": stats["unparseable"], "terminal": obs.get("status"),
            "answer_status": obs.get("answer_status"),
        },
        hard_failures=hard,
        known_gaps=gaps,
    )


def judge_top_customer(question: Mapping[str, object], obs: Mapping[str, object]) -> dict:
    """Customer names need approval; rows are checked after the approver ran the query.

    One row: the name must be the expected one.  Several rows: the expected name must
    be among them (whether it is the first row is only observed).  No rows: hard failure.
    Names are compared here and never written to the summary.

    The top customer's amount is a grouped value, never a verified tenant metric,
    so the answer is expected unverified with no fact.  A fact that is there anyway is
    checked by the rule every question gets (verified_fact_mismatch); a verified answer
    whose facts all equal the independent values (e.g. an earlier tenant total kept as a
    supporting fact) is a known gap.
    """

    expected = question["expected"]
    hard, gaps = _common_failures(obs)
    rows = obs.get("rows") or []
    names = [str(cell) for row in rows for key, cell in row.items() if key == "name" and isinstance(cell, str)]
    first_row_matches = bool(rows) and rows[0].get("name") == expected["name"]
    outcome = _outcome(
        expected={"customer_id": expected["customer_id"]},
        actual={"row_count": len(rows), "expected_name_in_rows": expected["name"] in names, "expected_name_first_row": first_row_matches},
    )
    if obs.get("status") != "SUCCEEDED":
        gaps.append(f"not_completed:{obs.get('status')}")
    elif not rows:
        hard.append("no_rows")
    elif not names:
        gaps.append("name_column_not_returned")
    elif len(rows) == 1 and names != [expected["name"]]:
        hard.append("top_customer_mismatch")
    elif len(rows) > 1 and expected["name"] not in names:
        hard.append("top_customer_absent")
    if obs.get("answer_contains_row_values"):
        hard.append("answer_contains_row_values")
    if not obs.get("approval_seen") and obs.get("status") == "SUCCEEDED" and names:
        hard.append("names_without_approval")
    if obs.get("status") == "SUCCEEDED" and obs.get("answer_status") == "verified" and obs.get("facts"):
        if verified_fact_counts(question, obs)[1] == 0:
            gaps.append("verified_supporting_fact")
    outcome["hard_failures"], outcome["known_gaps"] = hard, gaps
    return outcome


def judge_clarify_resume(question: Mapping[str, object], obs: Mapping[str, object]) -> dict:
    """"销售额": the server must ask the catalog question; resume must verify the chosen basis in the asked month."""

    hard, gaps = _common_failures(obs)
    first = obs.get("first") or {}
    if not (first.get("http_status") == 202 and first.get("status") == "WAITING_USER"):
        hard.append("clarify_not_triggered")
    elif not first.get("pending_is_catalog_question"):
        hard.append("clarify_question_not_from_catalog")
    resumed = judge_metric({**question, "kind": "metric"}, obs)
    if "answer_states_basis" in obs and not obs["answer_states_basis"] and not resumed["hard_failures"]:
        hard.append("basis_not_stated")
    if first.get("status") == "WAITING_USER" and obs.get("status") != "SUCCEEDED" and not resumed["hard_failures"]:
        resumed["known_gaps"].append(f"resume_not_completed:{obs.get('status')}")
    return _outcome(
        expected=resumed["expected"],
        actual=resumed["actual"],
        hard_failures=sorted(set(hard + resumed["hard_failures"])),
        known_gaps=sorted(set(gaps + resumed["known_gaps"])),
    )


def judge_knowledge(question: Mapping[str, object], obs: Mapping[str, object], demo_source_ids: frozenset[str]) -> dict:
    """A definition question: model text, unverified, no SQL, sources from the DEMO knowledge base."""

    outcome, hard, gaps = smoke.judge_knowledge_step(
        obs.get("http_status") or 0,
        obs.get("status"),
        obs.get("answer_status"),
        len(obs.get("source_ids") or []),
        len(obs.get("facts") or []),
        obs.get("sql_exec_count"),
        sent_back=any(item.startswith("answer_bounce(") for item in obs.get("trace") or []),
    )
    hard, gaps = list(hard), list(gaps)
    source_ids = list(obs.get("source_ids") or [])
    if outcome in {"pass", "pass_after_send_back"}:
        # "commerce-v1" is the catalog's own source id; every other id must be a demo document.
        if any(item not in demo_source_ids and item != "commerce-v1" for item in source_ids):
            hard.append("source_not_from_demo_knowledge_base")
        elif not any(item in demo_source_ids for item in source_ids):
            hard.append("no_demo_source")
        elif question["expected"]["expected_source_id"] not in source_ids:
            gaps.append("expected_source_missing")
    return _outcome(
        expected={"expected_source_id": question["expected"]["expected_source_id"]},
        actual={"outcome": outcome, "source_id_count": len(source_ids)},
        hard_failures=hard,
        known_gaps=gaps,
    )


def judge_no_data(question: Mapping[str, object], obs: Mapping[str, object]) -> dict:
    outcome, hard, gaps = smoke.judge_no_data_step(
        obs.get("http_status") or 0,
        obs.get("status"),
        obs.get("answer_status"),
        bool(obs.get("answer_is_fixed_text")),
        len(obs.get("facts") or []),
        obs.get("sql_exec_count"),
        sent_back=any(item.startswith("answer_bounce(") for item in obs.get("trace") or []),
    )
    return _outcome(expected={}, actual={"outcome": outcome}, hard_failures=list(hard), known_gaps=list(gaps))


def judge_isolation(question: Mapping[str, object], obs: Mapping[str, object]) -> dict:
    """Asking tenant B's data as tenant A: none of B's values may come back; any terminal state is recorded."""

    forbidden = set(question["expected"]["forbidden_values"])
    hard, gaps = _common_failures(obs)
    seen = {int(item["value"]) for item in obs.get("facts") or [] if _is_int(item.get("value"))}
    seen |= {value for row in obs.get("rows") or [] for value in row_ints(row)}
    leaked = sorted(forbidden & seen)
    answer = str(obs.get("answer") or "")
    shown = {text for value in forbidden for text in (str(value), f"{value // 100}.{value % 100:02d}")}
    if leaked or any(text in answer for text in shown):
        hard.append("foreign_tenant_value")
    return _outcome(
        expected={"forbidden_value_count": len(forbidden)},
        actual={"leaked_value_count": len(leaked), "terminal": obs.get("status"), "http_status": obs.get("http_status")},
        hard_failures=hard,
        known_gaps=gaps,
    )


def judge_observe_refund(question: Mapping[str, object], obs: Mapping[str, object]) -> dict:
    """Q04b: refund_fen has no server-side verifier.  Only record (also a 5xx); only verified-without-facts is hard."""

    wanted = question["expected"]["refund_fen"]
    hard, gaps = _observation_failures(obs)
    ints = {value for row in obs.get("rows") or [] for value in row_ints(row)}
    ints |= {int(item["value"]) for item in obs.get("facts") or [] if _is_int(item.get("value"))}
    gaps.append("refund_fen_has_no_verifier")
    return _outcome(
        expected={"refund_fen": wanted},
        actual={"terminal": obs.get("status"), "answer_status": obs.get("answer_status"), "value_seen": wanted in ints, "ints_seen": len(ints)},
        hard_failures=hard,
        known_gaps=gaps,
    )


def judge_question(question: Mapping[str, object], obs: Mapping[str, object], demo_source_ids: frozenset[str]) -> dict:
    """The kind's own judgement, then the rule every question gets: each verified
    fact equals the independently computed value for its tenant, metric and window."""

    outcome = _judge_kind(question, obs, demo_source_ids)
    checked, mismatched = verified_fact_counts(question, obs)
    outcome["verified_facts_checked"], outcome["verified_facts_mismatched"] = checked, mismatched
    if mismatched and "verified_fact_mismatch" not in outcome["hard_failures"]:
        outcome["hard_failures"] = [*outcome["hard_failures"], "verified_fact_mismatch"]
    return outcome


def _judge_kind(question: Mapping[str, object], obs: Mapping[str, object], demo_source_ids: frozenset[str]) -> dict:
    kind = question["kind"]
    if kind in {"metric", "empty_window"}:
        return judge_metric(question, obs)
    if kind == "rowset":
        return judge_rowset(question, obs)
    if kind == "observe_rowset":
        return judge_observe_rowset(question, obs)
    if kind == "top_customer":
        return judge_top_customer(question, obs)
    if kind == "clarify_resume":
        return judge_clarify_resume(question, obs)
    if kind == "knowledge":
        return judge_knowledge(question, obs, demo_source_ids)
    if kind == "no_data":
        return judge_no_data(question, obs)
    if kind == "isolation":
        return judge_isolation(question, obs)
    if kind == "observe_refund":
        return judge_observe_refund(question, obs)
    raise ValueError(f"unknown question kind: {kind}")


def summary_record(question: Mapping[str, object], obs: Mapping[str, object], judgement: Mapping[str, object]) -> dict:
    """The summary line: fixed fields and numbers only (no text, names, URLs)."""

    facts = obs.get("facts") or []
    hard = list(judgement["hard_failures"])
    return {
        "id": question["id"],
        "identity": question["identity"],
        "kind": question["kind"],
        # To match the gateway's per-run record (X-Run-Id): a random run id, and fixed states and numbers.
        "run_id": obs.get("run_id"),
        "http_status": obs.get("http_status"),
        "terminal": obs.get("status"),
        "error_code": obs.get("error_code"),
        "answer_status": obs.get("answer_status"),
        "fact_count": len(facts),
        "verified_metrics": sorted({str(item.get("metric_id")) for item in facts}),
        "sql_exec_count": obs.get("sql_exec_count"),
        "model_call_count": obs.get("model_call_count"),
        "usage_total": obs.get("usage_total"),
        "action_trace": list(obs.get("trace") or []),
        # Numbers only: completion tokens of each model call (null when the provider gave none) and the cap.
        "completion_tokens": list(obs.get("completion_tokens") or []),
        "max_output_tokens": obs.get("max_output_tokens"),
        "calls_at_output_limit": calls_at_output_limit(list(obs.get("completion_tokens") or []), obs.get("max_output_tokens")),
        "approval_seen": bool(obs.get("approval_seen")),
        "verified_facts_checked": judgement.get("verified_facts_checked", 0),
        "verified_facts_mismatched": judgement.get("verified_facts_mismatched", 0),
        "expected": judgement.get("expected"),
        "actual": judgement.get("actual"),
        "hard_failures": hard,
        "known_gaps": list(judgement["known_gaps"]),
        "verdict": "fail" if hard else "pass",
    }


def fake_scripted_only(mode: str, question: Mapping[str, object]) -> bool:
    """Fake and fake-upstream runs skip the questions the Fake model does not script."""

    return mode != "real" and not question.get("fake_supported")


def not_applicable_record(question: Mapping[str, object]) -> dict:
    return {
        "id": question["id"],
        "identity": question["identity"],
        "kind": question["kind"],
        "verdict": "not_applicable",
        "reason": "the Fake model does not script this question",
        "hard_failures": [],
        "known_gaps": [],
    }


# --- HTTP driving ---------------------------------------------------------------


def _error_code(body: Mapping[str, object]) -> object:
    error = body.get("error")
    return error.get("code") if isinstance(error, Mapping) else body.get("error_code")


def _completion_tokens(state_path: Path, run_id: str) -> list:
    """Completion tokens of every model call of the run, from the run's stored events (None = unknown)."""

    from queryshield.db.state_store import StateStore

    tokens: list = []
    with StateStore(state_path) as store:
        for event in store.events(run_id):
            payload = event.get("payload") or {}
            if event.get("type") == "agent_step" and payload.get("kind") == "model_call":
                usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
                value = usage.get("completion_tokens")
                tokens.append(int(value) if _is_int(value) else None)
    return tokens


def _observe(code: int, body: Mapping[str, object], state_path: Path, fixed_no_data_reply: str) -> dict:
    run_id = body.get("run_id")
    result = body.get("result") if isinstance(body.get("result"), Mapping) else {}
    rows = [row for row in (result.get("rows") or []) if isinstance(row, Mapping)]
    source_ids = body.get("source_ids")
    answer = str(body.get("answer") or "")
    return {
        "http_status": code,
        "status": body.get("status"),
        "error_code": _error_code(body),
        "answer_status": body.get("answer_status"),
        "answer": answer,
        "answer_is_fixed_text": answer == fixed_no_data_reply,
        "facts": smoke._facts(body),
        "rows": rows,
        "source_ids": [item for item in source_ids if isinstance(item, str)] if isinstance(source_ids, list) else [],
        "sql_exec_count": body.get("sql_exec_count"),
        "model_call_count": body.get("model_call_count"),
        "trace": smoke._action_trace(state_path, run_id) if isinstance(run_id, str) else [],
        "completion_tokens": _completion_tokens(state_path, run_id) if isinstance(run_id, str) else [],
        "max_output_tokens": max_output_tokens(),
        "run_id": run_id,
        "usage_total": body.get("usage_total"),
    }


def _run_question(base: str, tokens: Mapping[str, str], question: Mapping[str, object], state_path: Path, no_data_reply: str, catalog_question: str) -> tuple[dict, dict]:
    """Drive one question; returns (observation, raw record)."""

    token = tokens[question["identity"]]
    body: dict[str, object] = {"question": question["question"]}
    if question.get("request_time_window"):
        body["time_window"] = question["request_time_window"]
    code, response = smoke._http(base, "/queries", token=token, method="POST", body=body, timeout=300)
    obs = _observe(code, response, state_path, no_data_reply)
    raw = {"question": question["question"], "steps": []}
    if question["kind"] == "clarify_resume":
        obs["first"] = {
            "http_status": code,
            "status": response.get("status"),
            "pending_is_catalog_question": response.get("pending_question") == catalog_question,
        }
        if response.get("status") == "WAITING_USER":
            code, response = smoke._http(
                base, f"/runs/{response['run_id']}/resume", token=token, method="POST", body={"answer": question["resume_answer"]}
            )
            obs.update(_observe(code, response, state_path, no_data_reply))
            obs["answer_states_basis"] = "口径：支付订单总额（gross_fen）" in obs["answer"] and "你在追问中选择了‘支付金额’" in obs["answer"]
    if response.get("status") == "WAITING_APPROVAL":
        approver = tokens[APPROVER_OF[question["identity"]]]
        code, approved = smoke._http(
            base,
            f"/runs/{response['run_id']}/approval",
            token=approver,
            method="POST",
            body={"approval_id": response["approval_id"], "decision": "approve"},
        )
        _, result = smoke._http(base, f"/runs/{response['run_id']}/result", token=token)
        obs.update(_observe(code if result.get("status") != "SUCCEEDED" else 200, result or approved, state_path, no_data_reply))
        obs["approval_seen"] = True
        names = [str(row.get("name")) for row in obs["rows"] if isinstance(row.get("name"), str)]
        obs["answer_contains_row_values"] = any(name in obs["answer"] for name in names)
    raw["answer"] = obs.get("answer")
    raw["rows"] = obs.get("rows")
    return obs, raw


# --- fixed (offline) checks --------------------------------------------------------


def database_counts(url: str, tenants: list[str]) -> dict[str, dict[str, int]]:
    """Row counts per tenant through the read-only role with the tenant bound (RLS on)."""

    import psycopg

    counts: dict[str, dict[str, int]] = {}
    with psycopg.connect(url, connect_timeout=5, options="-c default_transaction_read_only=on") as conn:
        for tenant in tenants:
            with conn.transaction():
                conn.execute("SELECT set_config('queryshield.tenant_id', %s, true)", (tenant,))
                counts[tenant] = {
                    table: int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
                    for table in ("customers", "orders", "refunds")
                }
    return counts


def knowledge_checks(document: Mapping[str, object]) -> list[dict]:
    """Snapshot builds with the demo versions; every probe's expected source is found, every isolation probe holds (Fake embedding)."""

    from queryshield.agent.proposals import ExecutionContext
    from queryshield.knowledge.runtime import (
        DEMO_CATALOG_VERSION,
        DEMO_KNOWLEDGE_VERSION,
        reset_retrieval_cache,
        shared_demo_retrieval_runtime,
    )

    reset_retrieval_cache()
    runtime = shared_demo_retrieval_runtime("fake")
    snapshot = runtime.snapshot
    results = [
        {
            "check": "demo_snapshot",
            "ok": snapshot.knowledge_version == DEMO_KNOWLEDGE_VERSION and snapshot.catalog_version == DEMO_CATALOG_VERSION,
            "knowledge_version": snapshot.knowledge_version,
            "catalog_version": snapshot.catalog_version,
            "chunks": len(snapshot.chunk_records),
            "sources": len(snapshot.source_records),
        }
    ]

    def search(identity: str, query: str) -> list[str]:
        tenant, role = identity.split("-")
        context = ExecutionContext(run_id="demo-probe", tenant_id=tenant.upper(), principal_id=identity, role=role)
        return [item["source_id"] for item in runtime.retriever.search(query, context=context, top_k=3).items]

    for probe in document["retrieval_probes"]:
        found = search(probe["identity"], probe["query"])
        results.append({"check": f"probe:{probe['id']}", "ok": probe["expected_source_id"] in found})
    for probe in document["isolation_probes"]:
        found = search(probe["identity"], probe["query"])
        results.append({"check": f"isolation:{probe['id']}", "ok": not set(found) & set(probe["forbidden_source_ids"])})
    return results


def demo_source_ids() -> frozenset[str]:
    document = json.loads(DEMO_KNOWLEDGE_REGISTRY.read_text(encoding="utf-8"))
    return frozenset(item["source_id"] for item in document["sources"])


# --- main ---------------------------------------------------------------------------


def tokens_from_environment(environ: Mapping[str, str]) -> tuple[dict[str, str] | None, list[str]]:
    """The four identity tokens a RUNNING service was started with (--base-url), and the names missing or repeated."""

    tokens = {identity: environ.get(name, "").strip() for identity, name in TOKEN_ENVIRONMENT.items()}
    problems = [name for identity, name in TOKEN_ENVIRONMENT.items() if not tokens[identity]]
    if not problems and len(set(tokens.values())) != len(tokens):
        problems = ["tokens_must_be_four_different_values"]
    return (None if problems else tokens), problems


def _blocked(reason: str, **extra) -> int:
    print(json.dumps({"status": "blocked", "reason": reason, **extra}, ensure_ascii=False))
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(smoke.SERVER_MODE), default="real", help="fake-upstream: the Real adapters against the fake upstream, fake-supported questions only")
    parser.add_argument(
        "--model-protocol",
        choices=("json", "native"),
        default="json",
        help="how the model returns its decision; with --base-url, the running service's setting",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="drive a service that is already running instead of starting one (for example "
        "'docker compose exec app python scripts/demo_run.py --base-url http://127.0.0.1:8000'); "
        "tokens come from the QUERYSHIELD_TOKEN_* variables and the action trace from "
        f"{STATE_PATH_ENV}, so run it where the service's state store is readable",
    )
    args = parser.parse_args(argv)

    missing = [name for name in smoke.required_names(args.mode) if not os.getenv(name, "").strip()]
    if missing:
        return _blocked("missing_configuration", missing_configuration_names=missing)
    if args.mode == "fake-upstream" and smoke.names_not_fake_upstream(os.environ):
        return _blocked("model_names_not_fake_upstream", names=smoke.names_not_fake_upstream(os.environ))
    external = None
    if args.base_url:
        tokens, problems = tokens_from_environment(os.environ)
        if tokens is None:
            return _blocked("missing_tokens", names=problems)
        configured_state = os.environ.get(STATE_PATH_ENV, "").strip()
        if not configured_state or not Path(configured_state).is_file() or not os.access(configured_state, os.R_OK):
            # The action trace is read from the service's state store; never skip its judgements silently.
            return _blocked("state_store_unreadable", hint=f"set {STATE_PATH_ENV} to a store this process can read")
        external = (tokens, Path(configured_state), args.base_url.rstrip("/"))
    from queryshield.db.readonly import DEMO_DATABASE_SUFFIX, database_names_from_url

    names = database_names_from_url(os.environ["QUERYSHIELD_DATABASE_URL"])
    if not names or not all(name.endswith(DEMO_DATABASE_SUFFIX) for name in names):
        return _blocked("database_name_must_end_with_demo")
    if not smoke._database_reachable():
        return _blocked("database_unreachable", hint="create the demo database first (docs/demo-data.md)")
    args.evidence_dir.mkdir(parents=True, exist_ok=True)

    document = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = document["questions"]
    setup: list[dict] = []
    counts = database_counts(os.environ["QUERYSHIELD_DATABASE_URL"], sorted(document["table_counts"]))
    setup.append({"check": "table_counts", "ok": counts == document["table_counts"], "counts": counts})
    setup += knowledge_checks(document)

    from queryshield.agent.metric_intent import declarable_metric_ids
    from queryshield.catalog import load_default_catalog
    from queryshield.facts.render import render_no_data_answer

    catalog = load_default_catalog()
    no_data_reply = render_no_data_answer([catalog.metric_name(metric_id) for metric_id in declarable_metric_ids(catalog)])
    catalog_question = catalog.clarification("clarify.metric_basis").question
    demo_ids = demo_source_ids()

    process = None
    if external is not None:
        tokens, state_path, base = external
    else:
        workdir = Path(tempfile.mkdtemp(prefix="demo-run-"))
        state_path = workdir / "state.sqlite3"
        tokens = {name: uuid4().hex for name in ("a-requester", "a-approver", "b-requester", "b-approver")}
        env = os.environ.copy()
        for name in ("QUERYSHIELD_FAKE_DB", "QUERYSHIELD_AGENT_PROFILE", "QUERYSHIELD_RETRIEVAL", "QUERYSHIELD_METADATA_TOOLS"):
            env.pop(name, None)
        env.update(
            {
                "QUERYSHIELD_DEMO_DATASET": DEMO_DATASET,
                "QUERYSHIELD_PROVIDER_MODE": smoke.SERVER_MODE[args.mode],
                "QUERYSHIELD_MODEL_PROTOCOL": args.model_protocol,
                "QUERYSHIELD_STATE_STORE_PATH": str(state_path),
                "QUERYSHIELD_CALL_STORE_PATH": str(workdir / "calls.sqlite3"),
                "QUERYSHIELD_TOKEN_A_REQUESTER": tokens["a-requester"],
                "QUERYSHIELD_TOKEN_A_APPROVER": tokens["a-approver"],
                "QUERYSHIELD_TOKEN_B_REQUESTER": tokens["b-requester"],
                "QUERYSHIELD_TOKEN_B_APPROVER": tokens["b-approver"],
                "PYTHONPATH": str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
            }
        )
        port = smoke._free_port()
        base = f"http://127.0.0.1:{port}"
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "queryshield.api.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    records: list[dict] = []
    raw_records: list[dict] = []
    run_ids: list[str] = []
    fact_catalog_versions: set[str] = set()
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
            return _blocked("server_not_healthy")
        for question in questions:
            if fake_scripted_only(args.mode, question):
                records.append(not_applicable_record(question))
                continue
            try:
                obs, raw = _run_question(base, tokens, question, state_path, no_data_reply, catalog_question)
                judgement = judge_question(question, obs, demo_ids)
                if isinstance(obs.get("run_id"), str):
                    run_ids.append(obs["run_id"])
                fact_catalog_versions |= {str(fact.get("catalog_version")) for fact in obs["facts"] if fact.get("catalog_version")}
                record = summary_record(question, obs, judgement)
                if "bounced" in judgement:
                    record["answered_after_bounce"] = judgement["bounced"]
                records.append(record)
                raw_records.append({"id": question["id"], **raw})
            except smoke._StepStopped as stop:
                stopped = f"{stop.kind}:{question['id']}"
                records.append({"id": question["id"], "identity": question["identity"], "kind": question["kind"], "verdict": "fail", "hard_failures": [stopped], "known_gaps": []})
                break
            print(json.dumps({key: records[-1].get(key) for key in ("id", "verdict", "hard_failures", "known_gaps")}, ensure_ascii=False), flush=True)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    # Versions as recorded by the server for these runs (never the URL): the knowledge snapshot
    # label (catalog-v2 for the frozen knowledge base, catalog-v4 for the demo one) next to the
    # catalog version of the facts and run configuration (O8).
    from queryshield.db.state_store import StateStore

    snapshot_ids: set[str] = set()
    with StateStore(state_path) as store:
        for run_id in run_ids:
            config = ((store.get_run(run_id) or {}).get("run_config") or {}).get("agent_run_config") or {}
            if config.get("knowledge_snapshot_id"):
                snapshot_ids.add(str(config["knowledge_snapshot_id"]))
    hard_failures = sorted({f"{r['id']}:{item}" for r in records for item in r.get("hard_failures", [])})
    hard_failures += [f"setup:{item['check']}" for item in setup if not item.get("ok")]
    model_names, failures = smoke.model_labels(args.mode, state_path, run_ids, os.environ)
    hard_failures += failures
    summary = {
        "mode": args.mode,
        "model_protocol": args.model_protocol,
        "data_version": document["data_version"],
        "questions_version": document["version"],
        "status": "pass" if not hard_failures else "fail",
        "hard_failures": hard_failures,
        "known_gaps": sorted({f"{r['id']}:{item}" for r in records for item in r.get("known_gaps", [])}),
        "model_max_output_tokens": max_output_tokens(),
        "setup_checks": setup,
        "records": records,
        "verified_fact_check": {
            "checked": sum(int(r.get("verified_facts_checked") or 0) for r in records),
            "mismatched": sum(int(r.get("verified_facts_mismatched") or 0) for r in records),
        },
        "versions": {
            "knowledge_version": next((item.get("knowledge_version") for item in setup if item["check"] == "demo_snapshot"), None),
            "knowledge_snapshot_catalog_version": next((item.get("catalog_version") for item in setup if item["check"] == "demo_snapshot"), None),
            "fact_catalog_versions": sorted(fact_catalog_versions),
            "run_config_knowledge_snapshot_ids": sorted(snapshot_ids),
        },
        "note": "ids, status codes, terminal states, counts and numbers only; no question or answer text, customer names, URLs or credentials",
    }
    if model_names is not None:
        summary["model_names"] = model_names
    (args.evidence_dir / "demo-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.evidence_dir / "demo-raw.json").write_text(json.dumps(raw_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "hard_failures", "known_gaps")}, ensure_ascii=False))
    return 0 if not hard_failures else 1


if __name__ == "__main__":
    sys.exit(main())
