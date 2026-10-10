"""Compare demo runs by Agent profile: read two or more demo-summary.json files, print one JSON report.

Usage:
  python scripts/compare_demo_runs.py <demo-summary.json> <demo-summary.json> [...]

Each summary is grouped by the profile its runs actually used (the summary's
``profile``, read from the server's state store by scripts/demo_run.py).  Per
profile: runs, questions, passes, fact completeness (composite questions), model
and tool calls, input and output tokens, wall time (median and maximum), how
often the run delegated, and the error codes.  Anything the records do not have
is reported as "none", never as 0.  Only the summaries are read, never raw
evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from statistics import median
import sys

NONE = "none"


def _ints(records: Sequence[Mapping[str, object]], key: str) -> list[int]:
    return [int(r[key]) for r in records if type(r.get(key)) is int]


def _total_and_mean(values: list[int], count: int) -> dict[str, object] | str:
    if not values or len(values) != count:
        return NONE
    return {"total": sum(values), "per_run": round(sum(values) / count, 2)}


def _tokens(records: Sequence[Mapping[str, object]], name: str) -> int | str:
    """The sum over every run, or "none" when any run's usage is not known."""

    usages = [r.get("usage_total") for r in records]
    if not usages or any(not isinstance(u, Mapping) or u.get("status") != "known" or type(u.get(name)) is not int for u in usages):
        return NONE
    return sum(int(u[name]) for u in usages)  # type: ignore[index]


def _completeness(records: Sequence[Mapping[str, object]]) -> float | str:
    composite = [r for r in records if "fact_completeness" in r]
    expected = sum(int((r.get("expected") or {}).get("fact_count", 0)) for r in composite)
    found = sum(int((r.get("actual") or {}).get("found", 0)) for r in composite)
    return round(found / expected, 3) if expected else NONE


def profile_report(summaries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    records = [r for s in summaries for r in s.get("records", []) if r.get("verdict") != "not_applicable"]
    runs = [r for r in records if r.get("run_id")]
    elapsed = _ints(records, "elapsed_ms")
    return {
        "runs": len(summaries),
        "questions_sets": sorted({str(s.get("questions_set")) for s in summaries}),
        "questions": len(records),
        "passed": sum(1 for r in records if r.get("verdict") == "pass"),
        "fact_completeness": _completeness(records),
        "model_calls": _total_and_mean(_ints(runs, "model_call_count"), len(runs)),
        "tool_calls": _total_and_mean(_ints(runs, "tool_call_count"), len(runs)),
        "input_tokens": _tokens(runs, "prompt_tokens"),
        "output_tokens": _tokens(runs, "completion_tokens"),
        "elapsed_ms": {"median": median(elapsed), "max": max(elapsed)} if elapsed else NONE,
        "delegation_rate": round(sum(1 for r in runs if r.get("delegated")) / len(runs), 3) if runs else NONE,
        "error_codes": dict(sorted(Counter(str(r["error_code"]) for r in records if r.get("error_code")).items())),
    }


def compare(summaries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = {}
    for summary in summaries:
        groups.setdefault(str(summary.get("profile") or NONE), []).append(summary)
    return {"profiles": {profile: profile_report(items) for profile, items in sorted(groups.items())}}


def main(argv: list[str] | None = None) -> int:
    paths = [Path(item) for item in (sys.argv[1:] if argv is None else argv)]
    if len(paths) < 2:
        print("usage: compare_demo_runs.py <demo-summary.json> <demo-summary.json> [...]", file=sys.stderr)
        return 2
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    print(json.dumps(compare(summaries), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
