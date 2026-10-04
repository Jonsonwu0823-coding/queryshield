"""Run independently seeded W05 state cases through profile-specific adapters."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import Any
from uuid import uuid4

from queryshield.evaluation.report import build_w05_comparison_report
from queryshield.evaluation.state_cases import W05StateCase, canonical_sha256
from queryshield.evaluation.state_oracle import judge_state_case


_UNKNOWN_USAGE = {
    "usage_status": "unknown",
    "prompt_tokens": None,
    "completion_tokens": None,
    "total_tokens": None,
}


def _missing_observation(error_code: str) -> dict[str, object]:
    return {
        "status": "unknown",
        "http_status": None,
        "terminal_state": "UNKNOWN",
        "facts": [],
        "invariants": {},
        "side_effects": {
            "model_calls": 0,
            "readonly_queries": 0,
            "fact_count": 0,
            "write_statements": 0,
            "cross_tenant_rows": 0,
            "unauthorized_facts": 0,
        },
        "usage": dict(_UNKNOWN_USAGE),
        "elapsed_ms": None,
        "execution_status": "not_run",
        "not_run_reason": error_code,
    }


def run_w05_stateful_suite(
    cases: Sequence[W05StateCase],
    run_profile: Callable[[W05StateCase, str, str], Mapping[str, object]],
    *,
    metadata: Mapping[str, object],
    profiles: Sequence[str] = ("B0", "B1"),
) -> dict[str, object]:
    """Execute every supplied frozen case for B0/B1 and retain each outcome.

    The runner deliberately has no fixed 20-case assumption: the development
    split uses 20 rows today, while the same interface accepts all 32 frozen
    cases when the sealed 12 are first opened by T05.
    """

    if not isinstance(cases, Sequence) or not 1 <= len(cases) <= 32:
        raise ValueError("W05 replay requires between one and 32 frozen task cases")
    if any(not isinstance(case, W05StateCase) for case in cases):
        raise TypeError("all replay cases must be W05StateCase values")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("replay case ids must be unique")
    if tuple(profiles) != ("B0", "B1"):
        raise ValueError("W05 requires B0 followed by B1 for each independently seeded case")

    raw_by_profile: dict[str, list[dict[str, object]]] = {"B0": [], "B1": []}
    for case in cases:
        initial = case.case["initial"]
        action = case.case["action"]
        state_hash = canonical_sha256(initial)
        input_hash = canonical_sha256({"initial": initial, "action": action})
        for profile in profiles:
            profile_run_id = f"w05-{profile.lower()}-{uuid4().hex}"
            started = perf_counter()
            try:
                response = run_profile(case, profile, profile_run_id)
                if not isinstance(response, Mapping):
                    raise TypeError("profile adapter must return a mapping")
                observation_value = response.get("observation", response)
                if not isinstance(observation_value, Mapping):
                    raise TypeError("profile adapter observation must be a mapping")
                observation = dict(observation_value)
                execution = dict(response)
                execution.pop("observation", None)
            except TimeoutError as exc:
                observation = _missing_observation("profile_timeout")
                observation["side_effects"] = {name: None for name in observation["side_effects"]}
                execution = {"execution_error_type": type(exc).__name__}
                observation.update({"status": "timeout", "execution_status": "executed", "not_run_reason": None})
            except Exception as exc:  # keep a case-shaped failure in the frozen denominator
                observation = {
                    "status": "failed",
                    "http_status": None,
                    "terminal_state": "FAILED",
                    "facts": [],
                    "invariants": {},
                    "side_effects": {
                        "model_calls": None,
                        "readonly_queries": None,
                        "fact_count": None,
                        "write_statements": None,
                        "cross_tenant_rows": None,
                        "unauthorized_facts": None,
                    },
                    "usage": dict(_UNKNOWN_USAGE),
                    "elapsed_ms": None,
                    "execution_status": "executed",
                    "error_code": "profile_adapter_failed",
                }
                execution = {"execution_error_type": type(exc).__name__}

            elapsed_ms = observation.get("elapsed_ms")
            if elapsed_ms is None:
                observation["elapsed_ms"] = max(0, int((perf_counter() - started) * 1000))
            if "usage" not in observation:
                observation["usage"] = dict(_UNKNOWN_USAGE)
            if "execution_status" not in observation:
                observation["execution_status"] = "executed"

            record: dict[str, object] = {
                "case_id": case.case_id,
                "family_id": case.family_id,
                "classification": case.classification,
                "critical_question_id": case.critical_question_id,
                "split": str(case.case["split"]),
                "profile": profile,
                "profile_run_id": profile_run_id,
                "input_sha256": input_hash,
                "initial_state": dict(initial),
                "action_input": dict(action),
                "initial_state_sha256": state_hash,
                **dict(observation),
                **execution,
            }
            try:
                judgment = judge_state_case(case, record)
            except Exception as exc:  # a broken observation remains visible as a failed judgement
                judgment = {
                    "case_id": case.case_id,
                    "family_id": case.family_id,
                    "classification": case.classification,
                    "critical_question_id": case.critical_question_id,
                    "judged_status": "fail",
                    "functional_success": False,
                    "security_correct": False,
                    "mismatches": ["oracle_error"],
                    "oracle_error_type": type(exc).__name__,
                    "usage_status": "unknown",
                    "elapsed_ms": record.get("elapsed_ms"),
                }
            record["judgment"] = judgment
            record["judged_status"] = judgment["judged_status"]
            raw_by_profile[profile].append(record)

    report = build_w05_comparison_report(
        cases,
        raw_by_profile,
        metadata={**dict(metadata), "frozen_case_count": len(cases)},
    )
    profile_summaries: dict[str, dict[str, object]] = {}
    for profile in profiles:
        rows = raw_by_profile[profile]
        report_usage = report["profile_reports"][profile]["metrics"]["usage"]
        execution_counts = Counter(str(row.get("execution_status", "unknown")) for row in rows)
        outcome_counts = Counter(str(row.get("status", "unknown")) for row in rows)
        usage_counts = Counter(
            str(row["usage"].get("usage_status", "unknown"))
            if isinstance(row.get("usage"), Mapping)
            else "unknown"
            for row in rows
        )
        known_token_rows = [
            row["usage"]["total_tokens"]
            for row in rows
            if isinstance(row.get("usage"), Mapping)
            and row["usage"].get("usage_status") == "known"
            and type(row["usage"].get("total_tokens")) is int
        ]
        profile_summaries[profile] = {
            "case_count": len(rows),
            "executed_count": execution_counts.get("executed", 0),
            "blocked_count": execution_counts.get("blocked", 0),
            "not_run_count": execution_counts.get("not_run", 0),
            "outcomes": dict(sorted(outcome_counts.items())),
            "pass_count": sum(row.get("judged_status") == "pass" for row in rows),
            "fail_count": sum(row.get("judged_status") == "fail" for row in rows),
            "blocked_judgment_count": sum(row.get("judged_status") == "blocked" for row in rows),
            "not_run_judgment_count": sum(row.get("judged_status") == "not_run" for row in rows),
            "known_total_tokens": sum(known_token_rows) if len(known_token_rows) == len(rows) else None,
            "known_usage_case_count": usage_counts.get("known", 0),
            "unknown_usage_case_count": usage_counts.get("unknown", 0),
            "model_call_usage": report_usage["model_calls"],
            "usage_phases": report_usage["phases"],
            "usage_reconciliation": report_usage["reconciliation"],
        }
    critical_b1 = [
        row for row in raw_by_profile["B1"] if row.get("critical_question_id") is not None
    ]
    record_coverage_complete = all(
        len(raw_by_profile[profile]) == len(cases)
        and {str(row["case_id"]) for row in raw_by_profile[profile]} == set(case_ids)
        for profile in profiles
    )
    product_execution_complete = record_coverage_complete and all(
        row.get("execution_status") == "executed"
        for profile in profiles
        for row in raw_by_profile[profile]
    )
    return {
        "report": report,
        "raw_records": raw_by_profile,
        "summary": {
            "frozen_case_count": len(cases),
            "record_coverage_complete": record_coverage_complete,
            "product_execution_complete": product_execution_complete,
            "profile_summaries": profile_summaries,
            "b1_critical_question_count": len(critical_b1),
            "b1_critical_pass_count": sum(row.get("judged_status") == "pass" for row in critical_b1),
            "b1_critical_fail_count": sum(row.get("judged_status") == "fail" for row in critical_b1),
            "b1_critical_blocked_count": sum(row.get("judged_status") == "blocked" for row in critical_b1),
            "b1_critical_not_run_count": sum(row.get("judged_status") == "not_run" for row in critical_b1),
        },
    }


__all__ = ["run_w05_stateful_suite"]
