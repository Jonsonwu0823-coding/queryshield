"""Summarise several local evidence roots (development data only).

Read-only: it opens only a fixed allow-list of evidence file names under each
root's suite directories, never calls a model or database, and never reads
secrets, sealed/holdout material or encrypted files.  Directories whose name
mentions t05 or holdout are skipped, and raw records whose dataset split is not
``development`` are ignored.

For the C1 protocol comparison it also reports, from fields the records already
have, each suite's model protocol and per-profile passes and cost, and per case
the model/tool call counts and parse error codes.  A metric a record lacks is
reported as "unavailable"; no model text is ever output.

Usage:
    python scripts/eval_multi_run_summary.py ROOT [ROOT ...] [--output PATH]
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from queryshield.agent.config import NATIVE_ADAPTER_VERSION, NATIVE_VERSIONS  # noqa: E402


_SKIPPED_DIR_PARTS = ("t05", "holdout", "sealed")
_CHECK_RESULT = re.compile(r"EVAL-[A-Z0-9-]+\.json")
_RAW = re.compile(r"(stateful|stateful-supplement|smoke)-(fake|real)-raw\.json")
_REPORT = re.compile(r"(stateful|stateful-supplement|smoke)-(fake|real)-comparison-report\.json")


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


def _model_call_events(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = record.get("execution_events") or ()
    return [event for event in events if isinstance(event, Mapping) and event.get("kind") == "model_call"]


def _call_counts(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("execution_metrics") or record.get("side_effects") or {}


def _model_calls(record: Mapping[str, Any]) -> Any:
    return _call_counts(record).get("model_calls")


def _record_protocol(record: Mapping[str, Any]) -> str | None:
    """The protocol of a record's model calls; None when it made none (e.g. a fixture case).

    A resumed run's record has no run_config, so its model_call events decide
    (events carry the action schema version, not the adapter version).  A B0
    record that called the model has neither, so its protocol is unknown.
    """

    events = _model_call_events(record)
    if not events and not _model_calls(record):
        return None
    run_config = record.get("run_config")
    if run_config:
        return "native" if run_config.get("adapter_version") == NATIVE_ADAPTER_VERSION else "json"
    if not events:
        return None
    return "native" if events[0].get("action_schema_version") == NATIVE_VERSIONS["action_schema_version"] else "json"


def _record_tokens(record: Mapping[str, Any]) -> tuple[int, int] | None:
    """Prompt and completion tokens of a record's model calls; None if one call's usage is missing.

    B1 records carry usage per model_call event; a failed call returned no
    usage and adds nothing (a record whose calls all failed adds 0).  B0
    records have no events, only the record usage.
    """

    events = _model_call_events(record)
    if events:
        usages = [event.get("usage") for event in events if event.get("status") == "succeeded"]
    elif _model_calls(record):
        usage = record.get("usage") or {}
        usages = [usage if usage.get("usage_status") == "known" else None]
    else:
        return 0, 0
    if not all(
        isinstance(usage, Mapping) and type(usage.get("prompt_tokens")) is int and type(usage.get("completion_tokens")) is int
        for usage in usages
    ):
        return None
    return sum(usage["prompt_tokens"] for usage in usages), sum(usage["completion_tokens"] for usage in usages)


def _record_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    """Comparison metrics a raw record already carries; None where it has none."""

    counts = _call_counts(record)
    tokens = _record_tokens(record)
    return {
        "model_calls": counts.get("model_calls"),
        "tool_calls": counts.get("tool_calls"),
        "parse_error_codes": [
            event.get("error_code")
            for event in record.get("execution_events") or ()
            if isinstance(event, Mapping) and event.get("kind") == "proposal_validation"
        ],
        "prompt_tokens": tokens[0] if tokens is not None else None,
        "completion_tokens": tokens[1] if tokens is not None else None,
        "elapsed_ms": record.get("elapsed_ms"),
    }


def _profile_totals(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Per profile: passes, critical passes and summed cost ("unavailable" if any record lacks it)."""

    totals: dict[str, Any] = {}
    for profile in ("B0", "B1"):
        rows = [case for case in cases if case["profile"] == profile]
        critical = [case for case in rows if case["critical_question_id"] is not None]
        entry: dict[str, Any] = {
            "cases": len(rows),
            "passes": sum(case["judged_status"] == "pass" for case in rows),
            "critical_cases": len(critical),
            "critical_passes": sum(case["judged_status"] == "pass" for case in critical),
        }
        for key in ("model_calls", "tool_calls", "prompt_tokens", "completion_tokens", "elapsed_ms"):
            values = [case[key] for case in rows]
            numeric = values and all(type(value) in (int, float) for value in values)
            entry[key] = sum(values) if numeric else "unavailable"
        totals[profile] = entry
    return totals


def summarize_suite(suite: Path) -> dict[str, Any]:
    """One suite directory (e.g. full-real, smoke-real) of one evidence root."""

    result: dict[str, Any] = {
        "checks": {},
        "overall_status": None,
        "security_violations": 0,
        "manifest_sha256": None,
        "prompt_versions": [],
        "prompt_versions_by_protocol": {},
        "model_protocols": [],
        "profiles": {},
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
    by_protocol: dict[str, set[str]] = defaultdict(set)
    protocols: set[str] = set()
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
                    versions = _record_prompt_versions(record)
                    prompt_versions |= versions
                    # Records without a model call have no protocol (fixture cases).
                    protocol = _record_protocol(record)
                    if protocol is not None:
                        by_protocol[protocol] |= versions
                        if profile == "B1":
                            protocols.add(protocol)
                    result["cases"].append(
                        {
                            "case_set": match.group(1),
                            "profile": profile,
                            "case_id": record.get("case_id"),
                            "critical_question_id": record.get("critical_question_id"),
                            "judged_status": record.get("judged_status"),
                            "error_code": record.get("error_code"),
                            "terminal_state": record.get("terminal_state"),
                            **_record_metrics(record),
                        }
                    )
    result["prompt_versions"] = sorted(prompt_versions)
    result["prompt_versions_by_protocol"] = {protocol: sorted(values) for protocol, values in sorted(by_protocol.items())}
    result["model_protocols"] = sorted(protocols)
    result["profiles"] = _profile_totals(result["cases"])
    return result


def summarize_roots(roots: list[Path]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    tallies: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    manifests: dict[str, set[str]] = defaultdict(set)
    # The protocols' prompt versions differ by design (B0 is always json), so versions are compared per protocol.
    prompts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for root in roots:
        suites = {suite.name: summarize_suite(suite) for suite in sorted(root.iterdir()) if _allowed_dir(suite)}
        runs.append({"root": str(root), "suites": {name: {k: v for k, v in data.items() if k != "cases"} for name, data in suites.items()}})
        for suite_name, data in suites.items():
            if data["manifest_sha256"]:
                manifests[suite_name].add(data["manifest_sha256"])
            for protocol, versions in data["prompt_versions_by_protocol"].items():
                prompts[(suite_name, protocol)].update(versions)
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
                        "model_calls": [],
                        "tool_calls": [],
                        "parse_error_codes": [],
                    },
                )
                tally["runs"] += 1
                for metric in ("model_calls", "tool_calls", "parse_error_codes"):
                    tally[metric].append(case[metric])
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
    for (suite_name, protocol), values in sorted(prompts.items()):
        if len(values) > 1:
            warnings.append({"kind": "prompt_version_mismatch", "suite": suite_name, "model_protocol": protocol, "values": sorted(values)})
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
