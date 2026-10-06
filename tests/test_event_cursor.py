"""The Last-Event-ID resume cursor: ASCII digits only, bounded length, never an exception."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from queryshield.api.main import _event_cursor


class _Store:
    def event_bounds(self, run_id):
        return None


def _cursor(value: str | None):
    request = SimpleNamespace(state=SimpleNamespace())
    return _event_cursor(request, SimpleNamespace(store=_Store()), "run-1", value)


@pytest.mark.parametrize(
    "value, expected",
    [(None, 0), ("", 0), ("   ", 0), ("0", 0), ("7", 7), (" 42 ", 42), ("9" * 18, 10**18 - 1)],
)
def test_a_well_formed_cursor_is_read(value, expected) -> None:
    assert _cursor(value) == expected


@pytest.mark.parametrize(
    "value",
    ["²", "1²", "１２", "٣", "9" * 19, "9" * 5000, "abc", "-1", "1.5", "0x1"],
)
def test_any_other_cursor_is_the_400_invalid_event_cursor(value) -> None:
    response = _cursor(value)
    assert response.status_code == 400
    assert b'"code":"invalid_event_cursor"' in response.body
    assert "Last-Event-ID格式错误".encode() in response.body
