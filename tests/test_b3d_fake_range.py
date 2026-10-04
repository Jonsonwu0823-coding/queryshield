"""B3d: the Fake model reads "YYYY年M月至N月" as one window; every single-month and default behaviour is unchanged."""

from __future__ import annotations

import pytest

from queryshield.providers import fake_model as fm


@pytest.mark.parametrize(
    ("text", "window"),
    [
        ("2026年7月至9月的退款后净额", ("2026-07-01T00:00:00Z", "2026-10-01T00:00:00Z")),
        ("2026年7月到9月", ("2026-07-01T00:00:00Z", "2026-10-01T00:00:00Z")),
        ("2026年10月至12月", ("2026-10-01T00:00:00Z", "2027-01-01T00:00:00Z")),
        ("2026年8月 ～ 8月", ("2026-08-01T00:00:00Z", "2026-09-01T00:00:00Z")),
        ("2026年6月-7月", ("2026-06-01T00:00:00Z", "2026-08-01T00:00:00Z")),
    ],
)
def test_a_month_range_is_one_half_open_window(text, window) -> None:
    assert fm._window(text, None) == {"start": window[0], "end": window[1]}


@pytest.mark.parametrize("text", ["2026年9月至7月", "2026年7月", "支付金额是多少", "2026年7月和9月"])
def test_non_ranges_are_not_range_windows(text) -> None:
    assert fm._range_window(text) is None


def test_single_months_request_windows_and_the_default_are_as_before() -> None:
    assert fm._window("2026年8月已支付订单数", None) == {"start": "2026-08-01T00:00:00Z", "end": "2026-09-01T00:00:00Z"}
    assert fm._window("2026年12月", None) == {"start": "2026-12-01T00:00:00Z", "end": "2027-01-01T00:00:00Z"}
    request = {"start": "2026-02-01T00:00:00Z", "end": "2026-03-01T00:00:00Z"}
    assert fm._window("订单数", request) == request
    assert fm._window("订单数", None) == dict(fm._DEFAULT_WINDOW)


def test_a_range_in_the_question_makes_a_net_fen_call_over_the_whole_window() -> None:
    call = fm._query_call("2026年7月至9月的退款后净额是多少？", "", None)
    assert call["metrics"] == ["net_fen"]
    assert call["time_window"] == {"start": "2026-07-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}
    assert call["params"] == {"0": "paid", "1": "2026-07-01T00:00:00Z", "2": "2026-10-01T00:00:00Z"}
