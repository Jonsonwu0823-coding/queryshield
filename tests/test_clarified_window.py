"""The window a clarified metric is bound to: a month alone, a follow-up's previous window, Chinese month names."""

from __future__ import annotations

import json

import pytest

from queryshield.agent.context import NET_FEN_TIME_WINDOW
from queryshield.api.main import app, get_model_provider
from queryshield.approval.service import _month_window

from test_follow_up_api import facts, follow, previous_turn, succeeded
from test_http_queries import REQUESTER, CountingFake, Scripted, ask, auth, env  # noqa: F401  (env is a fixture)


def _month(year: int, month: int) -> dict[str, str]:
    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
    return {"start": f"{year:04d}-{month:02d}-01T00:00:00Z", "end": f"{next_year:04d}-{next_month:02d}-01T00:00:00Z", "timezone": "UTC"}


JULY_2026 = _month(2026, 7)
JANUARY_2025 = _month(2025, 1)
CHINESE = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"]


def single(text: str) -> dict[str, str]:
    return _month_window(text, NET_FEN_TIME_WINDOW, follow_up=False)


def following(text: str, base: dict[str, str] = JULY_2026) -> dict[str, str]:
    return _month_window(text, base, follow_up=True)


# --- rule 1: a year and a month -------------------------------------------------


@pytest.mark.parametrize("month", range(1, 13))
def test_a_year_and_a_month_in_numerals_or_characters_is_that_month(month) -> None:
    assert single(f"2025年{month}月销售额是多少？") == _month(2025, month)
    assert single(f"2025年{CHINESE[month - 1]}月销售额是多少？") == _month(2025, month)
    assert following(f"那2025年{CHINESE[month - 1]}月的销售额呢") == _month(2025, month)


def test_a_year_and_a_month_wins_over_the_previous_window() -> None:
    assert following("那2026年8月的销售额呢 按支付金额统计") == _month(2026, 8)


# --- rule 2: a month alone --------------------------------------------------------


@pytest.mark.parametrize("month", range(1, 13))
def test_a_month_alone_is_that_month_in_the_base_year(month) -> None:
    assert single(f"{month}月的销售额是多少？") == _month(2026, month)
    assert single(f"{CHINESE[month - 1]}月的销售额是多少？") == _month(2026, month)
    # A follow-up takes the year of the previous turn's window, never another one.
    assert following(f"那{month}月的销售额呢", JANUARY_2025) == _month(2025, month)
    assert following(f"那{CHINESE[month - 1]}月的销售额呢", JANUARY_2025) == _month(2025, month)


# --- rule 3: a follow-up's last month ---------------------------------------------


def test_a_follow_ups_last_month_is_the_month_before_the_previous_window() -> None:
    assert following("那上个月的销售额呢") == _month(2026, 6)
    assert following("上一个月的销售额呢") == _month(2026, 6)
    assert following("那上个月的销售额呢", JANUARY_2025) == _month(2024, 12)


# --- rule 4: no month at all --------------------------------------------------------


def test_no_month_is_the_base_window() -> None:
    assert following("那销售额呢 按支付金额统计") == JULY_2026
    assert single("销售额是多少？ 按支付金额统计") == NET_FEN_TIME_WINDOW


# --- rule 5: any other wording ------------------------------------------------------


@pytest.mark.parametrize("text", ["这个月的销售额", "下个月的销售额", "前两个月的销售额", "上上个月的销售额"])
def test_other_wordings_keep_the_default_window(text) -> None:
    assert following(text) == NET_FEN_TIME_WINDOW
    assert single(text) == NET_FEN_TIME_WINDOW


def test_a_single_questions_last_month_keeps_the_default_window() -> None:
    assert single("上个月的销售额是多少？") == NET_FEN_TIME_WINDOW


# --- nothing else is read as a month --------------------------------------------------


@pytest.mark.parametrize("text", ["18月的销售额", "十八月的销售额", "13月的销售额", "0月的销售额", "第108月"])
def test_numbers_that_are_not_a_month_are_not_read_as_one(text) -> None:
    assert single(text) == NET_FEN_TIME_WINDOW
    assert following(text) == NET_FEN_TIME_WINDOW


def test_a_year_without_a_month_is_not_a_month() -> None:
    assert following("2028年的销售额呢") == JULY_2026
    assert single("2028年的销售额") == NET_FEN_TIME_WINDOW


def test_the_year_is_never_read_from_other_digits() -> None:
    # "12026年8月" is not 2026年8月; its 8月 follows 年, so it is not a month alone either.
    assert single("12026年8月的销售额") == NET_FEN_TIME_WINDOW


# --- over HTTP ---------------------------------------------------------------------------


def _confirmed_window(messages) -> dict[str, str]:
    content = next(item["content"] for item in messages if item["role"] == "system" and item["content"].startswith("QUERYSHIELD_SERVER_CONTEXT\n"))
    return json.loads(content.split("\n", 1)[1])["confirmed_slots"]["time_window"]


def test_a_single_question_with_a_month_alone_is_bound_to_that_month(env) -> None:
    app.dependency_overrides[get_model_provider] = lambda: CountingFake()
    waiting = ask(env, "8月的销售额是多少？").json()
    assert waiting["status"] == "WAITING_USER"
    model = Scripted([json.dumps({"type": "deny", "reason": "probe"})])
    app.dependency_overrides[get_model_provider] = lambda: model
    env.post(f"/runs/{waiting['run_id']}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额"})

    assert _confirmed_window(model.messages[0]) == _month(2026, 8)


@pytest.mark.parametrize(
    ("question", "month"),
    [("那8月的销售额呢", 8), ("那八月的销售额呢", 8), ("那销售额呢", 7), ("那上个月的销售额呢", 6), ("那2026年9月的销售额呢", 9)],
)
def test_a_follow_up_clarification_is_bound_from_the_previous_window(env, question, month) -> None:
    model = CountingFake()
    app.dependency_overrides[get_model_provider] = lambda: model
    first = succeeded(ask(env, "2026年7月已支付订单总额是多少？"))
    waiting = follow(env, first["run_id"], question).json()
    assert waiting["status"] == "WAITING_USER"
    calls = len(model.messages)
    resumed = env.post(f"/runs/{waiting['run_id']}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额"})

    assert _confirmed_window(model.messages[calls]) == _month(2026, month)
    if question != "那八月的销售额呢":  # the Fake model reads months in numerals only
        assert facts(succeeded(resumed)) == [("gross_fen", _month(2026, month)["start"], 15000)]



def _clarified_window_of(env, model, previous_run_id: str, question: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    """The windows the follow-up carried, and the window its clarified metric is bound to."""

    waiting = follow(env, previous_run_id, question).json()
    assert waiting["status"] == "WAITING_USER", waiting
    carried = [item["time_window"] for item in previous_turn(model.messages[-1])["executed"]]
    calls = len(model.messages)
    env.post(f"/runs/{waiting['run_id']}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额"})
    return carried, _confirmed_window(model.messages[calls])


def test_a_follow_up_of_a_turn_without_a_window_is_bound_as_a_single_question(env) -> None:
    model = CountingFake()
    app.dependency_overrides[get_model_provider] = lambda: model
    knowledge = succeeded(ask(env, "退款后净额是怎么算的？"))

    # Its "上个月" has no previous window to count from: rule 5, the default window.
    assert _clarified_window_of(env, model, knowledge["run_id"], "那上个月的销售额呢") == ([], NET_FEN_TIME_WINDOW)


def test_a_follow_up_of_a_turn_with_two_windows_falls_back_to_the_first(env) -> None:
    model = CountingFake()
    app.dependency_overrides[get_model_provider] = lambda: model
    july = succeeded(ask(env, "2026年7月已支付订单总额是多少？"))
    compared = succeeded(follow(env, july["run_id"], "比上个月多多少"))

    assert _clarified_window_of(env, model, compared["run_id"], "那销售额呢") == ([JULY_2026, _month(2026, 6)], JULY_2026)
