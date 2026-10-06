"""Token usage is known only when every number is a plain non-negative integer and the parts add up.

An unknown call is never counted as zero: the totals stay empty (None) as soon as one call is unknown.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from queryshield.agent.graph import _usage_summary
from queryshield.agent.runtime import usage_record


def _call(call_id, usage, status="known", kind="model_call"):
    return {"kind": kind, "model_call_id": call_id, "usage_status": status, "usage": usage}


def _tokens(prompt, completion, total):
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


GOOD = _tokens(10, 2, 12)
KEYS = [
    "status", "model_call_count", "known_call_count", "unknown_call_count", "model_call_ids",
    "known_prompt_tokens", "known_completion_tokens", "known_total_tokens",
    "prompt_tokens", "completion_tokens", "total_tokens",
]


def _summary(status, calls, known, unknown, ids, known_tokens, tokens):
    values = [status, calls, known, unknown, ids, *known_tokens, *tokens]
    return dict(zip(KEYS, values, strict=True))


NONE3 = (None, None, None)


def test_no_model_call_is_not_run_with_empty_totals() -> None:
    assert _usage_summary(()) == _summary("not_run", 0, 0, 0, [], NONE3, NONE3)
    assert _usage_summary(({"kind": "tool_call"},)) == _summary("not_run", 0, 0, 0, [], NONE3, NONE3)


def test_known_calls_add_up() -> None:
    events = [_call("a", GOOD), {"kind": "tool_call"}, _call("b", _tokens(1, 1, 2))]
    assert _usage_summary(events) == _summary("known", 2, 2, 0, ["a", "b"], (11, 3, 14), (11, 3, 14))
    assert list(_usage_summary(events)) == KEYS


def test_one_unknown_call_empties_the_totals_but_keeps_the_known_part() -> None:
    events = [_call("a", GOOD), _call("b", None, "unknown"), _call("c", GOOD)]
    assert _usage_summary(events) == _summary("unknown", 3, 2, 1, ["a", "b", "c"], (20, 4, 24), NONE3)


def test_only_unknown_calls_have_no_known_part() -> None:
    assert _usage_summary([_call("a", None, "unknown")]) == _summary("unknown", 1, 0, 1, ["a"], NONE3, NONE3)


@pytest.mark.parametrize(
    "usage",
    [
        _tokens(10, 2, 13),
        _tokens(-1, 2, 1),
        _tokens(True, 1, 2),
        _tokens(10.0, 2.0, 12.0),
        _tokens("10", 2, 12),
        _tokens(10, 2, None),
        {"prompt_tokens": 10, "completion_tokens": 2},
        "12",
        None,
    ],
    ids=["sum-differs", "negative", "bool", "float", "text", "none", "missing-field", "not-a-mapping", "no-usage"],
)
def test_malformed_usage_is_unknown_even_when_the_status_says_known(usage) -> None:
    assert _usage_summary([_call("a", usage)]) == _summary("unknown", 1, 0, 1, ["a"], NONE3, NONE3)


def test_known_numbers_do_not_count_when_the_status_is_not_known() -> None:
    assert _usage_summary([_call("a", GOOD, "unknown")]) == _summary("unknown", 1, 0, 1, ["a"], NONE3, NONE3)
    assert _usage_summary([_call("a", GOOD, None)]) == _summary("unknown", 1, 0, 1, ["a"], NONE3, NONE3)


def test_a_missing_call_id_is_left_out_of_the_id_list_but_the_call_is_counted() -> None:
    events = [_call(None, GOOD), _call("", GOOD), _call(7, GOOD), _call("d", GOOD)]
    assert _usage_summary(events) == _summary("known", 4, 4, 0, ["d"], (40, 8, 48), (40, 8, 48))


# --- the single-pass record --------------------------------------------------------

UNKNOWN_RECORD = {"usage_status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


def _model_call(usage, status="known"):
    return SimpleNamespace(usage=None if usage is None else SimpleNamespace(**usage), usage_status=status)


def test_a_known_single_call_keeps_its_numbers() -> None:
    assert usage_record(_model_call(GOOD)) == {"usage_status": "known", **GOOD}
    assert list(usage_record(_model_call(GOOD))) == ["usage_status", "prompt_tokens", "completion_tokens", "total_tokens"]


@pytest.mark.parametrize(
    "call",
    [
        _model_call(None),
        _model_call(GOOD, "unknown"),
        _model_call(_tokens(10, 2, 13)),
        _model_call(_tokens(-1, 2, 1)),
        _model_call(_tokens(True, 1, 2)),
        _model_call(_tokens(10, 2, None)),
        SimpleNamespace(),
    ],
    ids=["no-usage", "status-unknown", "sum-differs", "negative", "bool", "none", "no-attributes"],
)
def test_anything_else_is_unknown_and_never_zero(call) -> None:
    assert usage_record(call) == UNKNOWN_RECORD
