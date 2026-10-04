"""Raw-bound W05 B0/B1 quality, safety, latency and usage reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from queryshield.evaluation.state_cases import W05StateCase
from queryshield.evaluation.state_oracle import recompute_w05_metrics


def build_w05_comparison_report(
    cases: Sequence[W05StateCase],
    profile_records: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Build a report whose numerators and denominators are derived from raw rows.

    Missing rows remain in each frozen denominator. A two-case smoke result is
    marked partial and cannot be confused with the complete 20-case result.
    """

    if set(profile_records) != {"B0", "B1"}:
        raise ValueError("W05 report requires raw B0 and B1 records")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    frozen_ids = {case.case_id for case in cases}
    if len(frozen_ids) != len(cases) or not frozen_ids:
        raise ValueError("the W05 case manifest must have unique case IDs")

    reports: dict[str, dict[str, object]] = {}
    judged_by_profile: dict[str, dict[str, Mapping[str, object]]] = {}
    for profile in ("B0", "B1"):
        raw = tuple(profile_records[profile])
        metrics = recompute_w05_metrics(cases, raw)
        present = {str(row.get("case_id")) for row in raw}
        if len(present) != len(raw):
            raise ValueError(f"{profile} raw case IDs are missing or duplicated")
        judged = {
            str(row["case_id"]): row
            for row in metrics["per_case"]
            if isinstance(row, Mapping) and type(row.get("case_id")) is str
        }
        judged_by_profile[profile] = judged
        unexecuted = sorted(
            str(row.get("case_id"))
            for row in raw
            if row.get("execution_status") in {"blocked", "not_run", "missing"}
            or row.get("status") in {"blocked", "not_run", "missing"}
        )
        coverage_complete = present == frozen_ids
        execution_complete = coverage_complete and not unexecuted
        reports[profile] = {
            "raw_record_count": len(raw),
            "recorded_case_ids": sorted(present),
            "missing_case_ids": sorted(frozen_ids - present),
            "unexecuted_case_ids": unexecuted,
            "coverage_complete": coverage_complete,
            "execution_complete": execution_complete,
            "evaluation_status": "complete" if execution_complete else ("not_run" if not raw else "partial"),
            "metrics": metrics,
            "raw_records": [dict(row) for row in raw],
        }

    paired = []
    for case in cases:
        outcomes = {
            profile: {
                "judged_status": judged_by_profile[profile][case.case_id]["judged_status"],
                "mismatches": judged_by_profile[profile][case.case_id]["mismatches"],
            }
            for profile in ("B0", "B1")
        }
        paired.append({"case_id": case.case_id, "family_id": case.family_id, "profiles": outcomes})

    b0_functional = reports["B0"]["metrics"]["functional_success"]
    b1_functional = reports["B1"]["metrics"]["functional_success"]
    b0_security = reports["B0"]["metrics"]["security_correct"]
    b1_security = reports["B1"]["metrics"]["security_correct"]
    return {
        "report_schema": "w05-comparison-report-v1",
        "evaluation_status": (
            "complete"
            if all(reports[profile]["execution_complete"] for profile in ("B0", "B1"))
            else "partial_or_not_run"
        ),
        "metadata": dict(metadata),
        "frozen_case_count": len(cases),
        "profile_reports": reports,
        "paired_outcomes": paired,
        "differences": {
            "functional_success_rate_b1_minus_b0": {
                "b0_numerator": b0_functional["numerator"],
                "b1_numerator": b1_functional["numerator"],
                "denominator": b0_functional["denominator"],
                "difference": (
                    b1_functional["value"] - b0_functional["value"]
                    if b1_functional["value"] is not None and b0_functional["value"] is not None
                    else None
                ),
            },
            "security_correct_rate_b1_minus_b0": {
                "b0_numerator": b0_security["numerator"],
                "b1_numerator": b1_security["numerator"],
                "denominator": b0_security["denominator"],
                "difference": (
                    b1_security["value"] - b0_security["value"]
                    if b1_security["value"] is not None and b0_security["value"] is not None
                    else None
                ),
            },
        },
        "interpretation_boundary": "descriptive finite-set comparison; no statistical significance or production claim",
    }


__all__ = ["build_w05_comparison_report"]
