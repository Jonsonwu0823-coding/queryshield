"""Summarise several local W05 evidence roots (development data only).

Read-only: it opens only a fixed allow-list of evidence file names under each
root's suite directories, never calls a model or database, and never reads
secrets, sealed/holdout material or encrypted files.  Directories whose name
mentions t05 or holdout are skipped, and raw records whose dataset split is not
``development`` are ignored.

Usage:
    python scripts/w05_multi_run_summary.py ROOT [ROOT ...] [--output PATH]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


_SKIPPED_DIR_PARTS = ("t05", "holdout", "sealed")
_CHECK_RESULT = re.compile(r"W05-[A-Z0-9-]+\.json")
_RAW = re.compile(r"(w05-stateful|w05-stateful-supplement|w05-b2a-smoke)-(fake|real)-raw\.json")
_REPORT = re.compile(r"(w05-stateful|w05-stateful-supplement|w05-b2a-smoke)-(fake|real)-comparison-report\.json")


def _allowed_dir(path: Path) -> bool:
    name = path.name.lower()
    return path.is_dir() and not path.is_symlink() and not any(part in name for part in _SKIPPED_DIR_PARTS)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _record_prompt_versions(record: Mapping[str, Any]) -> set[str]:
    versions: set[str] = set()
    run_config = record.get("run_config")
    if isinstance(run_config, Mapping) and isinstance(run_config.get("prompt_version"), str):
        versions.add(run_config["prompt_version"])
    for event in record.get("execution_events") or ():
        if isinstance(event, Mapping) and isinstance(event.get("prompt_version"), str):
            versions.add(event["prompt_version"])
    return versions


def summarize_suite(suite: Path) -> dict[str, Any]:
    """One suite directory (e.g. full-real, smoke-b2a-real) of one evidence root."""

    result: dict[str, Any] = {
        "checks": {},
        "overall_status": None,
        "security_violations": 0,
        "manifest_sha256": None,
        "prompt_versions": [],
        "cases": [],
    }
    summary = suite / "summary.json"
    if summary.is_file():
        payload = _read_json(summary)
        if isinstance(payload, Mapping):
            result["overall_status"] = payload.get("overall_status")
            for check in payload.get("checks") or ():
                if isinstance(check, Mapping) and isinstance(check.get("check_id"), str):
                    result["checks"][check["check_id"]] = check.get("status")
    prompt_versions: set[str] = set()
    for path in sorted(suite.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        name = path.name
        if name == "source-manifest.txt":
            result["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif _CHECK_RESULT.fullmatch(name):
            payload = _read_json(path)
            if isinstance(payload, Mapping) and isinstance(payload.get("check_id"), str):
                result["checks"].setdefault(payload["check_id"], payload.get("status"))
        elif _REPORT.fullmatch(name):
            payload = _read_json(path)
            reports = payload.get("profile_reports") if isinstance(payload, Mapping) else None
            for profile in ("B0", "B1"):
                metrics = (reports or {}).get(profile, {}).get("metrics", {}) if isinstance(reports, Mapping) else {}
                violations = metrics.get("security_violations") if isinstance(metrics, Mapping) else None
                if isinstance(violations, Mapping) and type(violations.get("numerator")) is int:
                    result["security_violations"] += violations["numerator"]
        elif (match := _RAW.fullmatch(name)) is not None:
            payload = _read_json(path)
            if not isinstance(payload, Mapping) or payload.get("dataset_split") != "development":
                continue
            raw_records = payload.get("raw_records")
            if not isinstance(raw_records, Mapping):
                continue
            for profile in ("B0", "B1"):
                for record in raw_records.get(profile) or ():
                    if not isinstance(record, Mapping) or record.get("split", "development") != "development":
                        continue
                    prompt_versions |= _record_prompt_versions(record)
                    result["cases"].append(
                        {
                            "case_set": match.group(1),
                            "profile": profile,
                            "case_id": record.get("case_id"),
                            "critical_question_id": record.get("critical_question_id"),
                            "judged_status": record.get("judged_status"),
                            "error_code": record.get("error_code"),
                            "terminal_state": record.get("terminal_state"),
                        }
                    )
    result["prompt_versions"] = sorted(prompt_versions)
    return result


def summarize_roots(roots: list[Path]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    tallies: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    manifests: dict[str, set[str]] = defaultdict(set)
    prompts: dict[str, set[str]] = defaultdict(set)
    for root in roots:
        suites = {suite.name: summarize_suite(suite) for suite in sorted(root.iterdir()) if _allowed_dir(suite)}
        runs.append({"root": str(root), "suites": {name: {k: v for k, v in data.items() if k != "cases"} for name, data in suites.items()}})
        for suite_name, data in suites.items():
            if data["manifest_sha256"]:
                manifests[suite_name].add(data["manifest_sha256"])
            prompts[suite_name].update(data["prompt_versions"])
            for case in data["cases"]:
                key = (suite_name, case["case_set"], case["profile"], str(case["case_id"]))
                tally = tallies.setdefault(
                    key,
                    {
                        "suite": suite_name,
                        "case_set": case["case_set"],
                        "profile": case["profile"],
                        "case_id": case["case_id"],
                        "critical_question_id": case["critical_question_id"],
                        "runs": 0,
                        "passes": 0,
                        "failure_error_codes": [],
                        "failures": [],
                    },
                )
                tally["runs"] += 1
                if case["judged_status"] == "pass":
                    tally["passes"] += 1
                else:
                    tally["failure_error_codes"].append(case["error_code"])
                    tally["failures"].append(
                        {"root": str(root), "error_code": case["error_code"], "terminal_state": case["terminal_state"]}
                    )
    warnings = []
    for suite_name, values in sorted(manifests.items()):
        if len(values) > 1:
            warnings.append({"kind": "manifest_mismatch", "suite": suite_name, "values": sorted(values)})
    for suite_name, values in sorted(prompts.items()):
        if len(values) > 1:
            warnings.append({"kind": "prompt_version_mismatch", "suite": suite_name, "values": sorted(values)})
    cases = sorted(tallies.values(), key=lambda item: (item["suite"], item["case_set"], item["profile"], str(item["case_id"])))
    return {
        "root_count": len(roots),
        "consistent_candidate": not warnings,
        "warnings": warnings,
        "runs": runs,
        "cases": cases,
        "critical_cases": [item for item in cases if item["critical_question_id"] is not None],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    missing = [str(root) for root in args.roots if not root.is_dir()]
    if missing:
        print(json.dumps({"status": "blocked", "missing_roots": missing}, ensure_ascii=False))
        return 2
    summary = summarize_roots(list(args.roots))
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
