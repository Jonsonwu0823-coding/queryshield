"""Usage status normalization shared by W05 runtime and product observations."""

from __future__ import annotations

from collections.abc import Mapping


def usage_status_record(
    summary: Mapping[str, object] | None,
    *,
    expected_call_count: int,
) -> dict[str, object]:
    """Project runtime ``status`` into the case oracle's ``usage_status`` form.

    Counts and token arithmetic must agree before a usage value is reported as
    known. Tokens from known calls are never exposed as a complete total when
    another call is unknown.
    """

    if type(expected_call_count) is not int or expected_call_count < 0:
        raise ValueError("expected_call_count must be a non-negative integer")
    status = summary.get("status", summary.get("usage_status")) if isinstance(summary, Mapping) else None
    recorded_call_count = summary.get("model_call_count") if isinstance(summary, Mapping) else None
    if expected_call_count == 0 and status == "not_run" and recorded_call_count in (None, 0):
        return {
            "usage_status": "not_run",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    if (
        expected_call_count > 0
        and (recorded_call_count is None or recorded_call_count == expected_call_count)
        and status == "known"
        and isinstance(summary, Mapping)
    ):
        prompt = summary.get("prompt_tokens")
        completion = summary.get("completion_tokens")
        total = summary.get("total_tokens")
        if (
            type(prompt) is int and prompt >= 0
            and type(completion) is int and completion >= 0
            and type(total) is int and total >= 0
            and prompt + completion == total
        ):
            return {
                "usage_status": "known",
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
            }
    return {
        "usage_status": "unknown",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


__all__ = ["usage_status_record"]
