"""B3d: the demo data generator is reproducible, honest about its edge cases, and its SQL is parseable."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

import pytest

from scripts import generate_demo_data as gen

DEMO_DIR = Path(gen.DEMO_DIR)


@pytest.fixture(scope="module")
def data():
    return gen.generate()


def test_splitmix64_matches_the_reference_vector() -> None:
    # Published SplitMix64 outputs for seed 0.
    rng = gen.Rng(0)
    assert [rng.next() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_two_generations_are_identical(data) -> None:
    again = gen.generate()
    assert again == data
    assert gen.render_sql(again) == gen.render_sql(data)


def test_committed_files_are_byte_identical_to_a_fresh_run() -> None:
    outputs = gen.render_all()
    for path, text in outputs.items():
        assert path.read_bytes() == text.encode("utf-8"), path.name
    assert gen.main(["--check"]) == 0


def test_check_mode_fails_when_a_file_differs(tmp_path, monkeypatch, capsys) -> None:
    changed = tmp_path / "commerce-demo-v1.sql"
    changed.write_text("-- edited\n", encoding="utf-8")
    monkeypatch.setattr(gen, "SQL_PATH", changed)
    assert gen.main(["--check"]) == 1
    assert "commerce-demo-v1.sql" in capsys.readouterr().out


def test_a_different_seed_gives_different_data(data) -> None:
    assert gen.generate(gen.SEED + 1) != data


def test_scale_and_edge_cases_match_the_specification(data) -> None:
    for tenant, spec in gen.TENANT_SPECS.items():
        counts = gen.edge_counts(data, tenant)
        assert counts["customers"] == spec["customers"]
        assert counts["customers_without_orders"] == spec["inactive"]
        assert counts["orders"] == sum(spec["orders"].values())
        assert counts["cancelled_orders"] == sum(spec["cancelled"].values())
        assert counts["paid_orders"] == counts["orders"] - counts["cancelled_orders"]
        assert counts["paid_orders_with_refunds"] == sum(spec["refund_orders"].values())
        assert counts["full_refund_orders"] == spec["full"]
        assert counts["partial_refund_orders"] == spec["partial"]
        assert counts["multi_refund_orders"] == spec["multi"]
        assert counts["refunds_on_cancelled_orders"] == sum(spec["cancelled_refunds"].values())
        assert counts["cross_month_refund_orders"] >= sum(spec["cross"].values())
        assert counts["october_refunds_on_september_orders"] >= spec["cross"][9]
        per_month = Counter(gen._month_of(o.created) for o in data.orders if o.tenant == tenant)
        assert dict(per_month) == spec["orders"]


def test_the_description_states_the_same_counts(data) -> None:
    text = (DEMO_DIR / "commerce-demo-v1.md").read_text(encoding="utf-8")
    for tenant in gen.TENANT_SPECS:
        counts = gen.edge_counts(data, tenant)
        assert str(counts["refunds"]) in text and str(counts["cross_month_refund_orders"]) in text
    assert "取消订单上的退款" in text and "不计入 refund_fen" in text
    assert "commerce-v1" in text and "不是数据行" in text


def test_money_is_non_negative_integer_fen_and_times_are_whole_utc_seconds(data) -> None:
    for order in data.orders:
        assert type(order.amount_fen) is int and order.amount_fen >= 0
        assert type(order.created) is int
    for refund in data.refunds:
        assert type(refund.amount_fen) is int and refund.amount_fen >= 0
        assert type(refund.created) is int
    assert "." not in " ".join(re.findall(r"\b\d+\.\d+\b", gen.render_sql(data)))  # no decimal amounts


def test_orders_and_refunds_stay_inside_the_window_rules(data) -> None:
    assert min(o.created for o in data.orders) >= gen.month_start(6)
    assert max(o.created for o in data.orders) < gen.month_end(9)
    assert max(r.created for r in data.refunds) <= gen._ts(2026, 10, 31, 23, 59, 59)
    assert not any(gen.month_start(5) <= o.created < gen.month_end(5) for o in data.orders)
    assert not any(gen.month_start(5) <= r.created < gen.month_end(5) for r in data.refunds)
    orders = {(o.tenant, o.order_id): o for o in data.orders}
    totals: Counter = Counter()
    for refund in data.refunds:
        assert refund.created >= orders[(refund.tenant, refund.order_id)].created
        totals[(refund.tenant, refund.order_id)] += refund.amount_fen
    assert all(total <= orders[key].amount_fen for key, total in totals.items())


def test_boundary_orders_sit_on_the_edges_of_the_half_open_month(data) -> None:
    for tenant in gen.TENANT_SPECS:
        boundary = [o for o in data.orders if o.tenant == tenant and o.boundary]
        assert Counter(o.boundary for o in boundary) == {"first_second": 3, "last_second": 4}
        for order in boundary:
            month = gen._month_of(order.created)
            edge = gen.month_start(month) if order.boundary == "first_second" else gen.month_end(month) - 1
            assert order.created == edge and order.status == "paid"


def test_cross_month_refunds_change_the_single_month_answer(data) -> None:
    for tenant in gen.TENANT_SPECS:
        for month in (7, 8, 9):
            start, end = gen.month_start(month), gen.month_end(month)
            counted = gen.expected_metrics(data, tenant, start, end)["refund_fen"]
            in_window_only = sum(r.amount_fen for r in data.refunds if r.tenant == tenant and start <= r.created < end)
            assert counted < in_window_only, (tenant, month)
        wide = gen.expected_metrics(data, tenant, gen.month_start(7), gen.month_end(9))["refund_fen"]
        monthly = sum(gen.expected_metrics(data, tenant, gen.month_start(m), gen.month_end(m))["refund_fen"] for m in (7, 8, 9))
        assert wide > monthly  # the cross-month refunds count in the Jul-Sep window


def test_refunds_on_cancelled_orders_never_count(data) -> None:
    cancelled = {(o.tenant, o.order_id) for o in data.orders if o.status == "cancelled"}
    refunds = [r for r in data.refunds if (r.tenant, r.order_id) in cancelled]
    assert len(refunds) == 4  # A 3, B 1
    window = (gen.month_start(6), gen.month_end(9))
    for tenant in gen.TENANT_SPECS:
        on_cancelled = sum(
            r.amount_fen for r in data.refunds
            if r.tenant == tenant and (r.tenant, r.order_id) in cancelled and window[0] <= r.created < window[1]
        )
        on_paid = sum(
            r.amount_fen for r in data.refunds
            if r.tenant == tenant and (r.tenant, r.order_id) not in cancelled and window[0] <= r.created < window[1]
        )
        assert on_cancelled > 0  # the trap exists inside the window
        assert gen.expected_metrics(data, tenant, *window)["refund_fen"] == on_paid


def test_top_customers_lead_by_five_percent_and_the_q07_leader_is_unique(data) -> None:
    for tenant in gen.TENANT_SPECS:
        for month in gen.MONTHS:
            ranked = sorted(gen.customer_gross(data, tenant, gen.month_start(month), gen.month_end(month)).values(), reverse=True)
            assert ranked[0] * 100 >= ranked[1] * 105
    questions = {q["id"]: q for q in gen.build_questions(data)["questions"]}
    totals = gen.customer_gross(data, "A", gen.month_start(7), gen.month_end(7))
    assert questions["Q07"]["expected"]["customer_id"] == max(totals, key=totals.get)


def test_validate_rejects_data_that_breaks_its_own_rules(data) -> None:
    oversized = gen.DemoData(
        data.customers, data.orders, data.refunds + (gen.Refund("A", "r9999", data.refunds[0].order_id, 10**9, data.refunds[0].created),)
    )
    with pytest.raises(AssertionError):
        gen.validate(oversized)
    may = gen.Order("A", "o9999", "c01", "paid", 100, gen.month_start(5) + 5)
    with pytest.raises(AssertionError):
        gen.validate(gen.DemoData(data.customers, data.orders + (may,), data.refunds))


def test_customer_names_are_unique_within_a_tenant_and_the_files_are_utf8_lf(data) -> None:
    names = [(c.tenant, c.name) for c in data.customers]
    assert len(set(names)) == len(names)
    for path in (gen.SQL_PATH, gen.DOC_PATH, gen.QUESTIONS_PATH):
        raw = path.read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")
        raw.decode("utf-8")


def test_the_sql_file_header_names_the_version_and_the_seed() -> None:
    head = gen.SQL_PATH.read_text(encoding="utf-8").splitlines()[:3]
    assert "commerce-demo-v1" in head[0] and str(gen.SEED) in head[1]
    assert json.loads(gen.QUESTIONS_PATH.read_text(encoding="utf-8"))["data_version"] == "commerce-demo-v1"


# --- a third, independent reading of the SQL text (parses the committed file, not the generator) ---------


def _parse_sql(text: str):
    orders, refunds = [], []
    for match in re.finditer(r"\('([AB])', '(o\d+)', '(c\d+)', '(paid|cancelled)', (\d+), '([^']+)'\)", text):
        orders.append(match.groups())
    for match in re.finditer(r"\('([AB])', '(r\d+)', '(o\d+)', (\d+), '([^']+)'\)", text):
        refunds.append(match.groups())
    return orders, refunds


def test_expected_answers_agree_with_a_plain_parse_of_the_sql_file() -> None:
    text = gen.SQL_PATH.read_text(encoding="utf-8")
    orders, refunds = _parse_sql(text)
    document = json.loads(gen.QUESTIONS_PATH.read_text(encoding="utf-8"))
    paid_by_key = {(t, o): (int(a), ts) for t, o, _c, s, a, ts in orders if s == "paid"}
    for question in document["questions"]:
        if question["kind"] not in {"metric", "empty_window", "clarify_resume"}:
            continue
        tenant, window, expected = question["tenant"], question["window"], question["expected"]
        # ISO UTC strings of one fixed shape compare correctly as text
        inside = lambda ts: window["start"] <= ts < window["end"]  # noqa: E731
        paid = {key: v for key, v in paid_by_key.items() if key[0] == tenant and inside(v[1])}
        gross = sum(amount for amount, _ in paid.values())
        refund = sum(int(a) for t, _r, o, a, ts in refunds if t == tenant and (t, o) in paid and inside(ts))
        actual = {"paid_count": len(paid), "gross_fen": gross, "net_fen": gross - refund}[expected["metric_id"]]
        assert actual == expected["value"], question["id"]


# --- Q06 (bounded row set), Q06b (full summary), Q07 -----------------------------------------------------------------


def test_q06_is_the_top_five_customers_of_june_and_q06b_is_the_full_summary(data) -> None:
    questions = {q["id"]: q for q in gen.build_questions(data)["questions"]}
    june = gen.customer_gross(data, "A", gen.month_start(6), gen.month_end(6))
    ranked = sorted(june.items(), key=lambda item: (-item[1], item[0]))
    q06, q06b = questions["Q06"], questions["Q06b"]
    assert gen.TOP_N == 5 and q06["kind"] == "rowset" and q06b["kind"] == "observe_rowset"
    assert [(r["customer_id"], r["value"]) for r in q06["expected"]["rows"]] == ranked[:5]
    assert ranked[4][1] > ranked[5][1]
    assert q06["expected"]["all_values"] == q06b["expected"]["all_values"] == dict(sorted(june.items()))
    assert len(q06b["expected"]["rows"]) == len(june) > 5
    assert q06["window"] == q06b["window"] == {"start": "2026-06-01T00:00:00Z", "end": "2026-07-01T00:00:00Z"}
    assert "姓名" not in q06["question"] and "客户名" not in q06["question"]
    assert q06["question"] != q06b["question"] and "5个客户" in q06["question"]


def test_validate_rejects_a_tie_at_the_top_five_cut_off(data) -> None:
    june = gen.customer_gross(data, "A", gen.month_start(6), gen.month_end(6))
    ranked = sorted(june.items(), key=lambda item: (-item[1], item[0]))
    (_, fifth), (sixth_id, sixth) = ranked[4], ranked[5]
    tie = gen.Order("A", "o9999", sixth_id, "paid", fifth - sixth, gen.month_start(6) + 100)
    with pytest.raises(AssertionError, match="must not tie"):
        gen.validate(gen.DemoData(data.customers, data.orders + (tie,), data.refunds))


def test_q07_asks_for_the_highest_paid_amount_and_its_leader_is_the_same_by_net(data) -> None:
    q07 = next(q for q in gen.build_questions(data)["questions"] if q["id"] == "Q07")
    assert q07["question"] == "2026年7月已支付金额最高的客户姓名是什么？"
    totals = gen.customer_gross(data, "A", gen.month_start(7), gen.month_end(7))
    assert q07["expected"]["customer_id"] == max(totals, key=totals.get)
