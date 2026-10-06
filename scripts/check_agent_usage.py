"""Usage-summary invariants shared by the checks (T05, FS01, FS02).

The agent's usage summary has grown fields since it was introduced (``model_call_ids`` and the
``known_*`` totals).  The checks therefore verify what a usage summary must
guarantee instead of comparing the whole dictionary, and they never relax a number:
call counts, known/unknown split and the token totals stay exact.  Fields this
module does not know are ignored on purpose and reported in the returned record.

Import from a check script with the scripts directory on ``sys.path`` (the check
scripts add it themselves), so direct runs, ``runpy`` loads and ``check.ps1``
all resolve the same module.
"""

from __future__ import annotations

from collections.abc import Mapping

TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
CHECKED_FIELDS = frozenset(
    {
        "status",
        "model_call_count",
        "known_call_count",
        "unknown_call_count",
        "model_call_ids",
        "known_prompt_tokens",
        "known_completion_tokens",
        "known_total_tokens",
        *TOKEN_FIELDS,
    }
)


def _fail(summary: object, reason: str) -> AssertionError:
    return AssertionError(f"unexpected usage summary ({reason}): {summary}")


def assert_known_usage(
    summary: Mapping[str, object],
    *,
    calls: int,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> dict[str, object]:
    """Require ``calls`` model calls, all with known usage and exact token totals."""

    if not isinstance(summary, Mapping):
        raise _fail(summary, "not a mapping")
    expected = {
        "status": "known",
        "model_call_count": calls,
        "known_call_count": calls,
        "unknown_call_count": 0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    for field, value in expected.items():
        actual = summary.get(field)
        if type(actual) is not type(value) or actual != value:
            raise _fail(summary, f"{field} expected={value!r} actual={actual!r}")
    if prompt_tokens + completion_tokens != total_tokens:
        raise _fail(summary, "expected token totals are inconsistent")
    # The summary also lists the call ids and the known-token totals.
    call_ids = summary.get("model_call_ids")
    if not isinstance(call_ids, list) or len(call_ids) != calls:
        raise _fail(summary, "model_call_ids must list one id per model call")
    if any(type(call_id) is not str or not call_id for call_id in call_ids) or len(set(call_ids)) != len(call_ids):
        raise _fail(summary, "model_call_ids must be distinct non-empty strings")
    for field in TOKEN_FIELDS:
        known = summary.get(f"known_{field}")
        if type(known) is not int or known != summary[field]:
            raise _fail(summary, f"known_{field} must equal {field} when every call is known")
    return {
        "invariants_checked": sorted(CHECKED_FIELDS),
        "ignored_extra_fields": sorted(set(summary) - CHECKED_FIELDS),
    }
