"""The Fake model's follow-up rules: the slots a question leaves unsaid come from the previous turn."""

from __future__ import annotations

import json

from queryshield.agent.context import previous_turn_content
from queryshield.providers import fake_model as fm


SEPTEMBER = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z", "timezone": "UTC"}
AUGUST = {"start": "2026-08-01T00:00:00Z", "end": "2026-09-01T00:00:00Z"}


def _previous(*, metric: str = "gross_fen", grouped: bool = False, names: bool = False, window=SEPTEMBER) -> dict:
    return {
        "note": "n",
        "questions": ["q"],
        "executed": [{"metric_id": metric, "time_window": window, "grouped_by_customer": grouped}] if metric else [],
        "customer_ids": ["c1", "c2"],
        "customer_count": 2,
        "listed_customer_names": names,
    }


def _plan(question: str, **previous) -> dict | None:
    return fm._follow_up_plan(question, "", _previous(**previous), None)


def _windows(plan) -> list[dict[str, str]]:
    return [window for _, window in plan["parts"]]


def test_a_question_naming_its_own_metric_and_month_is_new() -> None:
    assert _plan("2026年7月已支付订单数是多少？", grouped=True) is None


def test_a_month_alone_keeps_the_metric_and_takes_the_year_of_the_previous_window() -> None:
    plan = _plan("那8月呢")
    assert plan["parts"] == [(["gross_fen"], AUGUST)] and plan["by_customer"] is False


def test_a_metric_alone_keeps_the_window() -> None:
    plan = _plan("退款后净额呢")
    assert plan["parts"] == [(["net_fen"], {"start": SEPTEMBER["start"], "end": SEPTEMBER["end"]})]


def test_last_month_is_the_month_before_the_previous_window() -> None:
    assert _windows(_plan("那上个月呢")) == [AUGUST]
    january = {"start": "2026-01-01T00:00:00Z", "end": "2026-02-01T00:00:00Z", "timezone": "UTC"}
    assert _windows(_plan("上个月呢", window=january)) == [{"start": "2025-12-01T00:00:00Z", "end": "2026-01-01T00:00:00Z"}]


def test_compared_with_last_month_queries_both_months() -> None:
    assert _windows(_plan("比上个月多多少")) == [{"start": SEPTEMBER["start"], "end": SEPTEMBER["end"]}, AUGUST]


def test_grouping_is_kept_unless_the_question_drops_it() -> None:
    assert _plan("那8月呢", grouped=True)["by_customer"] is True
    assert _plan("不按客户了，8月总的呢", grouped=True)["by_customer"] is False


def test_these_customers_group_by_customer_even_with_a_full_question() -> None:
    plan = _plan("这几位客户2026年7月的支付金额呢")
    assert plan["by_customer"] is True and _windows(plan) == [{"start": "2026-07-01T00:00:00Z", "end": "2026-08-01T00:00:00Z"}]


def test_names_are_asked_again_when_the_previous_turn_listed_them() -> None:
    assert _plan("那8月呢", metric="", names=True) == {"names": True}
    assert _plan("姓名呢") == {"names": True}
    # A metric word asks for the metric, not for names again.
    assert _plan("他们的支付金额呢", metric="", names=True) is None


def test_after_a_turn_without_a_metric_the_question_names_metric_and_month() -> None:
    plan = _plan("他们2026年8月的支付金额呢", metric="", names=True)
    assert plan == {"parts": [(["gross_fen"], AUGUST)], "by_customer": True}


def test_nothing_to_carry_is_a_new_question() -> None:
    assert _plan("那8月呢", metric="") is None


def _messages(question: str, previous: dict) -> list[dict[str, str]]:
    return [{"role": "user", "content": previous_turn_content(previous)}, {"role": "user", "content": question}]


def test_the_model_queries_the_carried_metric_for_the_new_month() -> None:
    action = json.loads(fm._fake_content(_messages("那8月呢", _previous())))
    assert action["name"] == "query_readonly"
    assert action["arguments"]["metrics"] == ["gross_fen"] and action["arguments"]["time_window"] == AUGUST


def test_a_comparison_answers_after_both_months_are_queried() -> None:
    action = json.loads(fm._fake_content(_messages("比上个月多多少", _previous())))
    assert action["arguments"]["time_window"]["start"] == SEPTEMBER["start"]


def test_without_a_previous_turn_a_bare_month_names_no_metric() -> None:
    assert json.loads(fm._fake_content([{"role": "user", "content": "那8月呢"}]))["type"] == "ask_user"
