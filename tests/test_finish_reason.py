"""The finish_reason a provider returns is recorded only if it is one of the four known values."""

from __future__ import annotations

import pytest

from queryshield.providers.openai_compatible import _parse_response

from test_native_tools import _complete

MISSING = object()


def _finish_reason(value: object):
    choice: dict[str, object] = {"message": {"content": ""}}
    if value is not MISSING:
        choice["finish_reason"] = value
    return _parse_response({"id": "call-1", "choices": [choice]}, native=True)[4]


@pytest.mark.parametrize("value", ["stop", "tool_calls", "length", "content_filter"])
def test_a_known_finish_reason_is_kept(value) -> None:
    assert _finish_reason(value) == value


@pytest.mark.parametrize("value", [MISSING, None])
def test_a_missing_or_null_finish_reason_is_none(value) -> None:
    assert _finish_reason(value) is None


@pytest.mark.parametrize(
    "value",
    ["function_call", "STOP", " stop", "stop ", "", "x" * 10_000, 7, 1.5, True, ["stop"], {"stop": 1}],
    ids=["unknown", "upper", "lead-space", "trail-space", "empty", "10000-chars", "int", "float", "bool", "list", "dict"],
)
def test_any_other_finish_reason_is_recorded_as_other(value) -> None:
    assert _finish_reason(value) == "<other>"


def test_an_unknown_finish_reason_reaches_the_call_record_as_other() -> None:
    result, _ = _complete({"content": "text"}, finish_reason="secret-" + "x" * 5000)
    assert result.finish_reason == "<other>"
    assert result.to_redacted_record()["finish_reason"] == "<other>"
