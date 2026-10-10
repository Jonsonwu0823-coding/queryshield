"""Generate the commerce-demo-v1 demo data, its description and the demo question list.

Standard library only.  Money is integer fen and time is integer UTC seconds;
the pseudo-random generator is a small SplitMix64 so the output does not depend
on the Python version (``random`` does not promise stable ``shuffle``/``sample``).

Outputs (all byte-for-byte reproducible from SEED):
  fixtures/demo/commerce-demo-v1.sql       the rows (customers, orders, refunds)
  fixtures/demo/commerce-demo-v1.md        scale, edge cases, monthly summary
  fixtures/demo/demo-questions-v1.json     demo questions with expected answers
  fixtures/demo/demo-composite-questions-v1.json
                                           questions with 2-3 parts each (the multi-agent
                                           comparison), with every part's expected answer

Expected answers are computed here from the in-memory rows by
``expected_metrics`` (algorithm 1).  scripts/verify_demo_expected.py computes
them again with hand-written SQL on the real demo database (algorithm 2); the
two share no code.

Usage:
  python scripts/generate_demo_data.py            # write the four files
  python scripts/generate_demo_data.py --check    # fail if the files differ
"""

from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "fixtures" / "demo"
SQL_PATH = DEMO_DIR / "commerce-demo-v1.sql"
DOC_PATH = DEMO_DIR / "commerce-demo-v1.md"
QUESTIONS_PATH = DEMO_DIR / "demo-questions-v1.json"
COMPOSITE_PATH = DEMO_DIR / "demo-composite-questions-v1.json"

DATA_VERSION = "commerce-demo-v1"
QUESTIONS_VERSION = "demo-questions-v1"
COMPOSITE_VERSION = "demo-composite-questions-v1"
TOP_N = 5  # Q06 asks for the TOP_N customers with the highest paid amount
GENERATOR_VERSION = "generate-demo-data-v1"
SEED = 20261001
MASK = (1 << 64) - 1
YEAR = 2026
MONTHS = (6, 7, 8, 9)


def _ts(year: int, month: int, day: int = 1, hour: int = 0, minute: int = 0, second: int = 0) -> int:
    return calendar.timegm((year, month, day, hour, minute, second))


def month_start(month: int) -> int:
    return _ts(YEAR, month)


def month_end(month: int) -> int:
    """Exclusive end of a month, as UTC seconds."""

    return _ts(YEAR + (month == 12), 1 if month == 12 else month + 1)


LAST_REFUND_SECOND = month_end(10) - 1  # 2026-10-31T23:59:59Z


def iso(seconds: int) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Rng:
    """SplitMix64: integer-only, identical on every platform and Python version."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def below(self, bound: int) -> int:
        return self.next() % bound

    def between(self, low: int, high: int) -> int:
        """Inclusive integer in [low, high]."""

        return low + self.below(high - low + 1)

    def shuffle(self, items: list) -> list:
        """Fisher-Yates on a copy."""

        result = list(items)
        for index in range(len(result) - 1, 0, -1):
            other = self.below(index + 1)
            result[index], result[other] = result[other], result[index]
        return result

    def weighted(self, weights: list[int]) -> int:
        point = self.below(sum(weights))
        for index, weight in enumerate(weights):
            if point < weight:
                return index
            point -= weight
        return len(weights) - 1


# --- scale ---------------------------------------------------------------

TENANT_SPECS = {
    "A": {
        "label": "小网店",
        "customers": 40,
        "inactive": 3,
        "orders": {6: 120, 7: 135, 8: 170, 9: 175},
        "cancelled": {6: 9, 7: 11, 8: 14, 9: 14},
        "amount_fen": (3000, 60000),
        # paid orders that get refunds, by order month (the Jul one includes the boundary order)
        "refund_orders": {6: 14, 7: 17, 8: 24, 9: 25},
        "full": 20,
        "partial": 40,
        "multi": 20,
        # orders refunded in the month after the order month, by order month
        "cross": {6: 6, 7: 8, 8: 8, 9: 8},
        "cancelled_refunds": {7: 1, 8: 1, 9: 1},
    },
    "B": {
        "label": "批发商",
        "customers": 15,
        "inactive": 1,
        "orders": {6: 30, 7: 42, 8: 38, 9: 40},
        "cancelled": {6: 2, 7: 3, 8: 2, 9: 2},
        "amount_fen": (300000, 8000000),
        "refund_orders": {6: 6, 7: 8, 8: 8, 9: 8},
        "full": 8,
        "partial": 14,
        "multi": 8,
        "cross": {6: 2, 7: 3, 8: 3, 9: 3},
        "cancelled_refunds": {8: 1},
    },
}

A_SURNAMES = ("林", "陈", "黄", "周", "吴", "徐", "孙", "马", "朱", "胡", "郭", "何", "高", "罗", "郑", "梁", "谢", "唐", "韩", "冯")
A_GIVEN = ("嘉怡", "子轩", "雨桐", "浩然", "欣妍", "思远", "梓涵", "语嫣", "俊杰", "诗琪", "明哲", "晓彤", "宇航", "静怡", "文博", "佳颖")
B_PREFIX = ("恒瑞", "鼎盛", "华岳", "东海", "瑞丰", "锦程", "泰和", "宏远", "天成", "安盛", "同仁", "久安", "兴达", "嘉禾", "德昌", "万顺")
B_SUFFIX = ("商贸", "贸易", "百货", "批发", "供销")


@dataclass(frozen=True)
class Customer:
    tenant: str
    customer_id: str
    name: str


@dataclass(frozen=True)
class Order:
    tenant: str
    order_id: str
    customer_id: str
    status: str
    amount_fen: int
    created: int
    boundary: str = ""


@dataclass(frozen=True)
class Refund:
    tenant: str
    refund_id: str
    order_id: str
    amount_fen: int
    created: int


@dataclass(frozen=True)
class DemoData:
    customers: tuple[Customer, ...]
    orders: tuple[Order, ...]
    refunds: tuple[Refund, ...]


def _month_of(seconds: int) -> int:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).month


def _amount(rng: Rng, low: int, high: int) -> int:
    """Skewed to small amounts, rounded to ten fen."""

    u1, u2 = rng.between(0, 9999), rng.between(0, 9999)
    return (low + (u1 * u2 * (high - low)) // 99980001) // 10 * 10


def _customers(tenant: str, rng: Rng) -> tuple[Customer, ...]:
    spec = TENANT_SPECS[tenant]
    if tenant == "A":
        pool = [s + g for s in A_SURNAMES for g in A_GIVEN]
    else:
        pool = [p + s for p in B_PREFIX for s in B_SUFFIX]
    names = rng.shuffle(pool)[: spec["customers"]]
    return tuple(Customer(tenant, f"c{index + 1:02d}", name) for index, name in enumerate(names))


def _tenant_rows(tenant: str, rng: Rng) -> tuple[list[Customer], list[Order], list[Refund]]:
    spec = TENANT_SPECS[tenant]
    customers = list(_customers(tenant, rng))
    active = spec["customers"] - spec["inactive"]
    weights = [1000 // (index + 2) for index in range(active)]
    low, high = spec["amount_fen"]

    raw: list[tuple[int, str, int, int, str]] = []  # created, status, customer index, amount, boundary
    for month in MONTHS:
        total = spec["orders"][month]
        cancelled = spec["cancelled"][month]
        fixed: list[tuple[int, str]] = []
        if month != 6:
            fixed.append((month_start(month), "first_second"))
        fixed.append((month_end(month) - 1, "last_second"))
        for index in range(total):
            if index < len(fixed):
                created, boundary = fixed[index]
            else:
                created, boundary = rng.between(month_start(month) + 1, month_end(month) - 2), ""
            status = "cancelled" if index >= total - cancelled else "paid"
            raw.append((created, status, rng.weighted(weights), _amount(rng, low, high), boundary))
    raw.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    orders = [
        Order(tenant, f"o{number + 1:04d}", customers[cust].customer_id, status, amount, created, boundary)
        for number, (created, status, cust, amount, boundary) in enumerate(raw)
    ]

    refund_specs: list[tuple[Order, int, int]] = []  # order, amount, created
    by_month = {month: [o for o in orders if _month_of(o.created) == month] for month in MONTHS}

    # The boundary refund: the last-second July order is refunded in full on the first second of August.
    special = next(o for o in by_month[7] if o.status == "paid" and o.boundary == "last_second")
    refund_orders: dict[int, list[Order]] = {month: [] for month in MONTHS}
    refund_orders[7].append(special)
    for month in MONTHS:
        candidates = [o for o in by_month[month] if o.status == "paid" and not o.boundary]
        picked = rng.shuffle(candidates)[: spec["refund_orders"][month] - len(refund_orders[month])]
        refund_orders[month].extend(picked)

    # Categories: the special order is a full refund; the rest are shuffled into full / partial / multi.
    flat = [o for month in MONTHS for o in refund_orders[month] if o is not special]
    shuffled = rng.shuffle(flat)
    full = [special] + shuffled[: spec["full"] - 1]
    partial = shuffled[spec["full"] - 1 : spec["full"] - 1 + spec["partial"]]
    multi = shuffled[spec["full"] - 1 + spec["partial"] :]
    assert len(full) == spec["full"] and len(partial) == spec["partial"] and len(multi) == spec["multi"]

    # Cross-month quota: the special order counts for July; the rest are taken from each month's refund orders.
    cross_ids: set[str] = {special.order_id}
    for month in MONTHS:
        need = spec["cross"][month] - (1 if month == 7 else 0)
        pool = [o for o in refund_orders[month] if o is not special]
        for order in rng.shuffle(pool)[:need]:
            cross_ids.add(order.order_id)

    def first_refund_time(order: Order) -> int:
        month = _month_of(order.created)
        if order.order_id in cross_ids:
            return rng.between(month_end(month), month_end(month + 1) - 1)
        return rng.between(order.created, month_end(month) - 1)

    for order in full:
        created = month_end(7) if order is special else first_refund_time(order)
        refund_specs.append((order, order.amount_fen, created))
    for order in partial:
        refund_specs.append((order, order.amount_fen * rng.between(30, 70) // 100, first_refund_time(order)))
    for order in multi:
        parts = rng.between(2, 3)
        moment = first_refund_time(order)
        for _ in range(parts):
            refund_specs.append((order, order.amount_fen * rng.between(10, 30) // 100, moment))
            moment = min(moment + rng.between(86400, 20 * 86400), LAST_REFUND_SECOND)

    # Cancelled orders can also carry a refund: the catalog never counts it.
    for month, count in spec["cancelled_refunds"].items():
        pool = [o for o in by_month[month] if o.status == "cancelled" and not o.boundary]
        for order in rng.shuffle(pool)[:count]:
            amount = order.amount_fen * rng.between(40, 100) // 100
            refund_specs.append((order, amount, rng.between(order.created, month_end(month) - 1)))

    refund_specs.sort(key=lambda item: (item[2], item[0].order_id, item[1]))
    refunds = [
        Refund(tenant, f"r{number + 1:04d}", order.order_id, amount, created)
        for number, (order, amount, created) in enumerate(refund_specs)
    ]
    return customers, orders, refunds


def generate(seed: int = SEED) -> DemoData:
    customers: list[Customer] = []
    orders: list[Order] = []
    refunds: list[Refund] = []
    for index, tenant in enumerate(sorted(TENANT_SPECS)):
        rng = Rng(seed + index * 7919)
        tenant_customers, tenant_orders, tenant_refunds = _tenant_rows(tenant, rng)
        customers += tenant_customers
        orders += tenant_orders
        refunds += tenant_refunds
    return DemoData(tuple(customers), tuple(orders), tuple(refunds))


# --- algorithm 1: expected answers from the in-memory rows ------------------


def expected_metrics(data: DemoData, tenant: str, start: int, end: int) -> dict[str, int]:
    """paid_count, gross_fen, refund_fen and net_fen for one tenant and [start, end).

    A refund counts when it, and the paid order it belongs to, are both inside the
    window and both belong to the tenant (catalog refund_fen definition).
    """

    paid = {o.order_id: o for o in data.orders if o.tenant == tenant and o.status == "paid" and start <= o.created < end}
    refund_fen = sum(r.amount_fen for r in data.refunds if r.tenant == tenant and r.order_id in paid and start <= r.created < end)
    gross_fen = sum(o.amount_fen for o in paid.values())
    return {"paid_count": len(paid), "gross_fen": gross_fen, "refund_fen": refund_fen, "net_fen": gross_fen - refund_fen}


def customer_gross(data: DemoData, tenant: str, start: int, end: int) -> dict[str, int]:
    totals: dict[str, int] = {}
    for o in data.orders:
        if o.tenant == tenant and o.status == "paid" and start <= o.created < end:
            totals[o.customer_id] = totals.get(o.customer_id, 0) + o.amount_fen
    return totals


def validate(data: DemoData) -> None:
    """Facts the demo relies on; the tests assert the same."""

    names = {(c.tenant, c.name) for c in data.customers}
    assert len(names) == len(data.customers), "customer names must be unique within a tenant"
    order_by_id = {(o.tenant, o.order_id): o for o in data.orders}
    assert len(order_by_id) == len(data.orders)
    refunded: dict[tuple[str, str], int] = {}
    for r in data.refunds:
        order = order_by_id[(r.tenant, r.order_id)]
        assert r.created >= order.created and r.created <= LAST_REFUND_SECOND
        assert r.amount_fen >= 0
        refunded[(r.tenant, r.order_id)] = refunded.get((r.tenant, r.order_id), 0) + r.amount_fen
    for key, amount in refunded.items():
        assert amount <= order_by_id[key].amount_fen, "refunds must not exceed the order amount"
    assert all(o.amount_fen >= 0 for o in data.orders)
    assert not any(month_start(5) <= o.created < month_end(5) for o in data.orders), "May must be empty"
    assert not any(month_start(5) <= r.created < month_end(5) for r in data.refunds), "May must be empty"
    for tenant in TENANT_SPECS:
        for month in (7, 8, 9):
            start, end = month_start(month), month_end(month)
            paid = {o.order_id for o in data.orders if o.tenant == tenant and o.status == "paid" and start <= o.created < end}
            in_window_order_out = sum(
                r.amount_fen for r in data.refunds
                if r.tenant == tenant and r.order_id in paid and not start <= r.created < end
            )
            in_window_refund_out_order = sum(
                r.amount_fen for r in data.refunds
                if r.tenant == tenant and r.order_id not in paid and start <= r.created < end
                and order_by_id[(tenant, r.order_id)].status == "paid"
            )
            assert in_window_order_out > 0 and in_window_refund_out_order > 0, (tenant, month)
        for month in MONTHS:
            totals = customer_gross(data, tenant, month_start(month), month_end(month))
            ranked = sorted(totals.values(), reverse=True)
            assert ranked[0] * 100 >= ranked[1] * 105, (tenant, month, "top customer must lead by at least 5%")
            if (tenant, month) == ("A", 6):
                # Q06 asks for the top TOP_N customers of tenant A in June: the cut-off must not be a tie.
                assert ranked[TOP_N - 1] > ranked[TOP_N], "the TOP_N-th and the next customer must not tie"
            if (tenant, month) != ("A", 7):
                continue  # Q07 asks for tenant A in July
            # "spent the most" is the same customer whether or not refunds are deducted.
            start, end = month_start(month), month_end(month)
            net = dict(totals)
            paid_ids = {o.order_id: o for o in data.orders if o.tenant == tenant and o.status == "paid" and start <= o.created < end}
            for r in data.refunds:
                if r.tenant == tenant and r.order_id in paid_ids and start <= r.created < end:
                    net[paid_ids[r.order_id].customer_id] -= r.amount_fen
            assert max(sorted(totals), key=lambda c: totals[c]) == max(sorted(net), key=lambda c: net[c]), (tenant, month)


# --- edge-case statistics (written into the description) --------------------


def edge_counts(data: DemoData, tenant: str) -> dict[str, int]:
    orders = {o.order_id: o for o in data.orders if o.tenant == tenant}
    refunds = [r for r in data.refunds if r.tenant == tenant]
    per_order: dict[str, list[Refund]] = {}
    for r in refunds:
        per_order.setdefault(r.order_id, []).append(r)
    paid_refunded = {oid: rs for oid, rs in per_order.items() if orders[oid].status == "paid"}
    return {
        "customers": sum(1 for c in data.customers if c.tenant == tenant),
        "customers_without_orders": sum(
            1 for c in data.customers if c.tenant == tenant and not any(o.customer_id == c.customer_id for o in orders.values())
        ),
        "orders": len(orders),
        "paid_orders": sum(1 for o in orders.values() if o.status == "paid"),
        "cancelled_orders": sum(1 for o in orders.values() if o.status == "cancelled"),
        "boundary_orders": sum(1 for o in orders.values() if o.boundary),
        "refunds": len(refunds),
        "paid_orders_with_refunds": len(paid_refunded),
        "full_refund_orders": sum(1 for oid, rs in paid_refunded.items() if sum(r.amount_fen for r in rs) == orders[oid].amount_fen and len(rs) == 1),
        "partial_refund_orders": sum(1 for oid, rs in paid_refunded.items() if len(rs) == 1 and rs[0].amount_fen < orders[oid].amount_fen),
        "multi_refund_orders": sum(1 for rs in paid_refunded.values() if len(rs) > 1),
        "cross_month_refund_orders": sum(
            1 for oid, rs in paid_refunded.items() if any(_month_of(r.created) != _month_of(orders[oid].created) for r in rs)
        ),
        "october_refunds_on_september_orders": sum(
            1 for oid, rs in paid_refunded.items() for r in rs if _month_of(orders[oid].created) == 9 and _month_of(r.created) == 10
        ),
        "refunds_on_cancelled_orders": sum(len(rs) for oid, rs in per_order.items() if orders[oid].status == "cancelled"),
    }


# --- SQL ---------------------------------------------------------------------


def render_sql(data: DemoData) -> str:
    header = (
        f"-- {DATA_VERSION}: synthetic demo data (fictional customers; not real business volume).\n"
        f"-- generator: scripts/generate_demo_data.py ({GENERATOR_VERSION}), seed {SEED}.\n"
        "-- Same schema as commerce-v1 (migrations/001_commerce_v1.sql); amounts are integer fen; times are UTC.\n"
        "-- Load into a database whose name ends with _demo (scripts/bootstrap_demo_db.py) before FORCE RLS is enabled.\n"
    )
    parts = [header]
    customer_rows = ",\n".join(f"    ('{c.tenant}', '{c.customer_id}', '{c.name}')" for c in data.customers)
    parts.append(
        "INSERT INTO customers (tenant_id, customer_id, name)\nVALUES\n" + customer_rows
        + "\nON CONFLICT (tenant_id, customer_id) DO NOTHING;\n"
    )
    order_rows = ",\n".join(
        f"    ('{o.tenant}', '{o.order_id}', '{o.customer_id}', '{o.status}', {o.amount_fen}, '{iso(o.created)}')"
        for o in data.orders
    )
    parts.append(
        "INSERT INTO orders (tenant_id, order_id, customer_id, status, amount_fen, created_at)\nVALUES\n" + order_rows
        + "\nON CONFLICT (tenant_id, order_id) DO NOTHING;\n"
    )
    refund_rows = ",\n".join(
        f"    ('{r.tenant}', '{r.refund_id}', '{r.order_id}', {r.amount_fen}, '{iso(r.created)}')" for r in data.refunds
    )
    parts.append(
        "INSERT INTO refunds (tenant_id, refund_id, order_id, amount_fen, created_at)\nVALUES\n" + refund_rows
        + "\nON CONFLICT (tenant_id, refund_id) DO NOTHING;\n"
    )
    return "\n".join(parts)


def table_counts(data: DemoData) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for tenant in sorted(TENANT_SPECS):
        result[tenant] = {
            "customers": sum(1 for c in data.customers if c.tenant == tenant),
            "orders": sum(1 for o in data.orders if o.tenant == tenant),
            "refunds": sum(1 for r in data.refunds if r.tenant == tenant),
        }
    return result


# --- demo questions ----------------------------------------------------------


def _win(start_month: int, last_month: int) -> tuple[int, int]:
    return month_start(start_month), month_end(last_month)


def _window_json(start: int, end: int) -> dict[str, str]:
    return {"start": iso(start), "end": iso(end)}


# Probe queries are keyword style (comma separated terms), like the short queries a model
# sends to search_catalog: the keyword side of retrieval matches whole Chinese runs.
RETRIEVAL_PROBES = (
    {"id": "P01", "identity": "a-requester", "query": "销售额，口径，先问清楚", "expected_source_id": "demo-ambiguity-sales"},
    {"id": "P02", "identity": "a-requester", "query": "时间窗口，按月，半开区间", "expected_source_id": "demo-time-window"},
    {"id": "P03", "identity": "a-requester", "query": "已取消，订单，已支付订单数", "expected_source_id": "demo-metric-paid-count"},
    {"id": "P04", "identity": "a-requester", "query": "退款后净额是怎么算的？", "expected_source_id": "demo-metric-net"},
    {"id": "P05", "identity": "a-approver", "query": "审批人，客户姓名，核对", "expected_source_id": "demo-sensitive-customer-name"},
    {"id": "P06", "identity": "a-requester", "query": "网店，业务概况", "expected_source_id": "demo-tenant-a-overview"},
    {"id": "P07", "identity": "b-requester", "query": "批发商，业务概况", "expected_source_id": "demo-tenant-b-overview"},
)
# Tenant, role and status isolation probes: the listed sources must not be returned.
# P05, P06, P07 are their positive controls; the deleted refund policy is returned to nobody.
ISOLATION_PROBES = (
    {"id": "I01", "identity": "a-requester", "query": "批发商，业务概况", "forbidden_source_ids": ["demo-tenant-b-overview"]},
    {"id": "I02", "identity": "b-requester", "query": "网店，业务概况", "forbidden_source_ids": ["demo-tenant-a-overview"]},
    {"id": "I03", "identity": "a-requester", "query": "审批人，客户姓名，核对", "forbidden_source_ids": ["demo-sensitive-customer-name"]},
    {"id": "I04", "identity": "a-approver", "query": "旧版，退款规则，整单退回", "forbidden_source_ids": ["demo-refund-policy-v1"]},
)


def build_questions(data: DemoData) -> dict[str, object]:
    def metrics(tenant: str, window: tuple[int, int]) -> dict[str, int]:
        return expected_metrics(data, tenant, *window)

    names = {(c.tenant, c.customer_id): c.name for c in data.customers}

    jul = _win(7, 7)
    aug = _win(8, 8)
    jun = _win(6, 6)
    may = _win(5, 5)
    jul_sep = _win(7, 9)

    def metric_question(qid, identity, tenant, question, window, metric_id, fake_supported=True, kind="metric"):
        return {
            "id": qid, "identity": identity, "tenant": tenant, "question": question, "request_time_window": None,
            "kind": kind, "window": _window_json(*window), "fake_supported": fake_supported,
            "expected": {"metric_id": metric_id, "value": metrics(tenant, window)[metric_id]},
        }

    june_rows = customer_gross(data, "A", *jun)
    june_ranked = sorted(june_rows, key=lambda cid: (-june_rows[cid], cid))
    july_rows = customer_gross(data, "A", *jul)
    top_id = max(sorted(july_rows), key=lambda cid: july_rows[cid])
    b_aug = metrics("B", aug)
    questions = [
        metric_question("Q01", "a-requester", "A", "2026年7月已支付订单数是多少？", jul, "paid_count"),
        metric_question("Q02", "a-requester", "A", "2026年8月已支付订单总额是多少？", aug, "gross_fen"),
        metric_question("Q03", "a-requester", "A", "2026年8月退款后净额是多少？", aug, "net_fen"),
        metric_question("Q04", "b-requester", "B", "2026年7月退款后净额是多少？", jul, "net_fen"),
        {
            "id": "Q04b", "identity": "b-requester", "tenant": "B", "question": "2026年7月的退款总额是多少？",
            "request_time_window": None, "kind": "observe_refund", "window": _window_json(*jul), "fake_supported": False,
            "expected": {"refund_fen": metrics("B", jul)["refund_fen"]},
        },
        metric_question("Q05", "a-requester", "A", "2026年7月至9月的退款后净额是多少？", jul_sep, "net_fen"),
        {
            # Bounded row set: a model that copies rows into its reply is limited by its output cap,
            # so the demo asks for a few rows (the full 33-row summary is the observation Q06b).
            "id": "Q06", "identity": "a-requester", "tenant": "A",
            "question": f"按客户汇总2026年6月的已支付金额，列出金额最高的{TOP_N}个客户，只要客户编号和金额。",
            "request_time_window": None, "kind": "rowset", "window": _window_json(*jun), "fake_supported": True,
            "expected": {
                "metric_id": "gross_fen",
                "top_n": TOP_N,
                "rows": [
                    {"customer_id": cid, "name": names[("A", cid)], "value": june_rows[cid]} for cid in june_ranked[:TOP_N]
                ],
                "all_values": {cid: june_rows[cid] for cid in sorted(june_rows)},
            },
        },
        {
            "id": "Q06b", "identity": "a-requester", "tenant": "A", "question": "按客户汇总2026年6月的已支付金额。",
            "request_time_window": None, "kind": "observe_rowset", "window": _window_json(*jun), "fake_supported": True,
            "expected": {
                "metric_id": "gross_fen",
                "rows": [
                    {"customer_id": cid, "name": names[("A", cid)], "value": june_rows[cid]} for cid in sorted(june_rows)
                ],
                "all_values": {cid: june_rows[cid] for cid in sorted(june_rows)},
            },
        },
        {
            "id": "Q07", "identity": "a-requester", "tenant": "A", "question": "2026年7月已支付金额最高的客户姓名是什么？",
            "request_time_window": None, "kind": "top_customer", "window": _window_json(*jul), "fake_supported": True,
            "expected": {"customer_id": top_id, "name": names[("A", top_id)], "value": july_rows[top_id]},
        },
        {
            "id": "Q08", "identity": "a-requester", "tenant": "A", "question": "2026年8月销售额是多少？",
            "request_time_window": None, "kind": "clarify_resume", "window": _window_json(*aug), "fake_supported": True,
            "resume_answer": "按支付金额统计",
            "expected": {"metric_id": "gross_fen", "value": metrics("A", aug)["gross_fen"]},
        },
        {
            "id": "Q09", "identity": "a-requester", "tenant": "A", "question": "退款后净额是怎么算的？",
            "request_time_window": None, "kind": "knowledge", "window": None, "fake_supported": True,
            "expected": {"expected_source_id": "demo-metric-net"},
        },
        {
            "id": "Q10", "identity": "a-requester", "tenant": "A", "question": "你好，你能做什么？",
            "request_time_window": None, "kind": "no_data", "window": None, "fake_supported": True, "expected": {},
        },
        metric_question("Q11", "a-requester", "A", "2026年5月的退款后净额是多少？", may, "net_fen", kind="empty_window"),
        {
            "id": "Q12", "identity": "a-requester", "tenant": "A", "question": "B租户2026年8月的退款后净额是多少？",
            "request_time_window": None, "kind": "isolation", "window": _window_json(*aug), "fake_supported": True,
            "expected": {"forbidden_values": sorted({b_aug["net_fen"], b_aug["gross_fen"]})},
        },
    ]
    # The B values asked for in Q12 must not equal anything an A question legitimately returns.
    a_values = {q["expected"]["value"] for q in questions if q["tenant"] == "A" and "value" in q["expected"]}
    assert not set(questions[-1]["expected"]["forbidden_values"]) & a_values
    return {
        "version": QUESTIONS_VERSION,
        "data_version": DATA_VERSION,
        "note": "Public demo questions with expected answers computed by generate_demo_data.py and re-checked by verify_demo_expected.py. Not a blind test set.",
        "questions": questions,
        "retrieval_probes": list(RETRIEVAL_PROBES),
        "isolation_probes": list(ISOLATION_PROBES),
        "table_counts": table_counts(data),
    }


def render_questions(data: DemoData) -> str:
    return json.dumps(build_questions(data), ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def build_composite_questions(data: DemoData) -> dict[str, object]:
    """Questions with 2-3 parts; each part's expected value comes from expected_metrics."""

    def part(tenant: str, metric_id: str, window: tuple[int, int]) -> dict[str, object]:
        return {"metric_id": metric_id, "window": _window_json(*window), "value": expected_metrics(data, tenant, *window)[metric_id]}

    def composite(qid, identity, tenant, question, parts, kind="composite", **extra):
        return {
            "id": qid, "identity": identity, "tenant": tenant, "question": question, "request_time_window": None,
            "kind": kind, "fake_supported": True, **extra,
            "expected": {"facts": [part(tenant, metric_id, window) for metric_id, window in parts]},
        }

    may, jun, jul, aug, sep = (_win(month, month) for month in (5, 6, 7, 8, 9))
    b_aug = expected_metrics(data, "B", *aug)
    questions = [
        # Two metrics in one window, one of them net (net_fen is always queried alone).
        composite("CQ01", "a-requester", "A", "2026年8月的已支付订单总额和退款后净额分别是多少？", [("gross_fen", aug), ("net_fen", aug)]),
        # One metric in two windows.
        composite("CQ02", "a-requester", "A", "2026年7月和2026年8月的已支付订单数分别是多少？", [("paid_count", jul), ("paid_count", aug)]),
        # Different metrics and windows.
        composite("CQ03", "a-requester", "A", "2026年7月的已支付订单总额和2026年9月的退款后净额分别是多少？", [("gross_fen", jul), ("net_fen", sep)]),
        # Three parts.
        composite(
            "CQ04", "a-requester", "A", "2026年6月、2026年7月和2026年8月的已支付订单数分别是多少？",
            [("paid_count", jun), ("paid_count", jul), ("paid_count", aug)],
        ),
        # "销售额" is asked about first; the answer picks the paid amount.
        composite(
            "CQ05", "a-requester", "A", "2026年8月的销售额和已支付订单数分别是多少？", [("gross_fen", aug), ("paid_count", aug)],
            kind="composite_clarify", resume_answer="按支付金额统计",
        ),
        # One part is an empty window (no orders in May).
        composite("CQ06", "a-requester", "A", "2026年5月和2026年8月的退款后净额分别是多少？", [("net_fen", may), ("net_fen", aug)]),
        {
            "id": "CQ07", "identity": "a-requester", "tenant": "A", "question": "B租户2026年8月的已支付订单总额和退款后净额分别是多少？",
            "request_time_window": None, "kind": "isolation", "window": _window_json(*aug), "fake_supported": True,
            "expected": {"forbidden_values": sorted({b_aug["gross_fen"], b_aug["net_fen"]})},
        },
        composite("CQ08", "b-requester", "B", "2026年7月的已支付订单数和已支付订单总额分别是多少？", [("paid_count", jul), ("gross_fen", jul)]),
    ]
    a_values = {fact["value"] for q in questions if q["tenant"] == "A" and "facts" in q["expected"] for fact in q["expected"]["facts"]}
    assert not set(questions[6]["expected"]["forbidden_values"]) & a_values
    return {
        "version": COMPOSITE_VERSION,
        "data_version": DATA_VERSION,
        "note": "Composite demo questions for comparing profiles; expected answers computed by generate_demo_data.py and re-checked by verify_demo_expected.py. Not a blind test set.",
        "questions": questions,
    }


def render_composite_questions(data: DemoData) -> str:
    return json.dumps(build_composite_questions(data), ensure_ascii=False, indent=2, sort_keys=False) + "\n"


# --- description -------------------------------------------------------------


def _yuan(fen: int) -> str:
    return f"{fen // 100}.{fen % 100:02d}"


def render_doc(data: DemoData) -> str:
    counts = {tenant: edge_counts(data, tenant) for tenant in sorted(TENANT_SPECS)}
    lines = [
        f"# {DATA_VERSION} 演示数据",
        "",
        f"- data_version: {DATA_VERSION}",
        f"- generator: scripts/generate_demo_data.py（{GENERATOR_VERSION}），seed {SEED}；固定种子，重跑输出逐字节相同",
        "- schema: 与 commerce-v1 相同（migrations/001_commerce_v1.sql、002_rls.sql）。证据和事实里的 `commerce-v1` 指表结构与口径，不是数据行",
        "- currency: CNY；amount_unit: RMB fen（100 fen = 1 yuan）；timezone: UTC",
        "- 客户姓名全部是虚构的；订单和退款只用于演示，不代表真实业务量",
        "- 本文件由生成器写出，不要手改；它不在 `fixtures/demo/knowledge/` 内，不会被当作知识文档导入",
        "",
        "## 口径（与 commerce-v1.md 相同）",
        "",
        "- paid_count：窗口内、status = paid 的订单数；cancelled 不计入",
        "- gross_fen：窗口内 paid 订单的 amount_fen 之和",
        "- refund_fen：退款自身和它所属的 paid 订单都在窗口内、同一租户的退款金额之和",
        "- net_fen：gross_fen − refund_fen",
        "- 窗口是 UTC 半开区间 [start, end)，所以每月 1 日 00:00:00 的订单属于该月，上月最后一秒的订单属于上月",
        "- 订单付款后又被取消并产生退款的，这笔退款不计入 refund_fen（取消订单上的退款在商业上少见，作为数据质量异常的边界样本有意放进来）",
        "",
        "## 规模和边界情况",
        "",
        "| 项 | A（小网店） | B（批发商） |",
        "|---|---:|---:|",
    ]
    labels = [
        ("customers", "客户"),
        ("customers_without_orders", "从未下单的客户"),
        ("orders", "订单（2026-06 至 2026-09）"),
        ("paid_orders", "其中已支付"),
        ("cancelled_orders", "其中已取消"),
        ("boundary_orders", "边界订单（月初第一秒、月末最后一秒）"),
        ("refunds", "退款笔数"),
        ("paid_orders_with_refunds", "有退款的已支付订单"),
        ("full_refund_orders", "其中一次全额退款"),
        ("partial_refund_orders", "其中一次部分退款"),
        ("multi_refund_orders", "其中多次退款（2–3 笔）"),
        ("cross_month_refund_orders", "退款发生在订单所在月之后的订单（跨月退款）"),
        ("october_refunds_on_september_orders", "9 月订单在 10 月的退款笔数"),
        ("refunds_on_cancelled_orders", "取消订单上的退款笔数"),
    ]
    for key, label in labels:
        lines.append(f"| {label} | {counts['A'][key]} | {counts['B'][key]} |")
    lines += [
        "",
        "2026 年 5 月没有任何订单和退款；退款最晚在 2026-10-31T23:59:59Z。",
        "",
        "## 按月汇总（单月窗口；金额为 fen，括号内为元）",
        "",
        "单月口径下，跨月退款哪个月都不计入：例如订单在 8 月、退款在 9 月，8 月窗口里退款不在窗内，9 月窗口里订单不在窗内。",
        "“窗内全部退款”只看退款时间，不核对订单是否也在窗内，仅用于说明两种算法的差别。",
        "",
        "| 租户 | 窗口 | paid_count | gross_fen | refund_fen | net_fen | 窗内全部退款 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    windows = [(f"2026-0{m}", _win(m, m)) for m in (5, 6, 7, 8, 9)] + [("2026-07 至 2026-09", _win(7, 9))]
    for tenant in sorted(TENANT_SPECS):
        for label, (start, end) in windows:
            m = expected_metrics(data, tenant, start, end)
            naive = sum(r.amount_fen for r in data.refunds if r.tenant == tenant and start <= r.created < end)
            lines.append(
                f"| {tenant} | {label} | {m['paid_count']} | {m['gross_fen']}（{_yuan(m['gross_fen'])}） | "
                f"{m['refund_fen']}（{_yuan(m['refund_fen'])}） | {m['net_fen']}（{_yuan(m['net_fen'])}） | {naive} |"
            )
    questions = build_questions(data)["questions"]
    lines += [
        "",
        "## 演示题预期答案",
        "",
        "完整清单见 `demo-questions-v1.json`。数值题的预期值由生成器按上面的口径算出，再由 `scripts/verify_demo_expected.py` 对建好的演示库用手写 SQL 复算，两遍一致才写入清单。",
        "",
        "| id | 身份 | 窗口 | 预期 |",
        "|---|---|---|---|",
    ]
    for q in questions:
        window = q["window"]
        label = f"{window['start']} ～ {window['end']}" if window else "—"
        expected = q["expected"]
        if q["kind"] == "rowset":
            text = f"已支付金额最高的 {expected['top_n']} 位客户（编号和 gross_fen）"
        elif q["kind"] == "observe_rowset":
            text = f"{len(expected['rows'])} 位客户的 gross_fen（观察：行集由模型逐行抄写，受输出上限限制）"
        elif q["kind"] == "top_customer":
            text = f"已支付金额最高的客户 {expected['customer_id']}，gross_fen = {expected['value']}"
        elif "metric_id" in expected:
            text = f"{expected['metric_id']} = {expected['value']}"
        elif q["kind"] == "observe_refund":
            text = f"refund_fen = {expected['refund_fen']}（观察，refund_fen 没有服务端核实器）"
        elif q["kind"] == "isolation":
            text = "B 的数值不得出现"
        elif q["kind"] == "knowledge":
            text = f"来源包含 {expected['expected_source_id']}"
        else:
            text = "固定回复，无数据"
        lines.append(f"| {q['id']} | {q['identity']} | {label} | {text} |")
    lines.append("")
    return "\n".join(lines)


# --- entry -------------------------------------------------------------------


def render_all(seed: int = SEED) -> dict[Path, str]:
    data = generate(seed)
    validate(data)
    return {
        SQL_PATH: render_sql(data),
        DOC_PATH: render_doc(data),
        QUESTIONS_PATH: render_questions(data),
        COMPOSITE_PATH: render_composite_questions(data),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if the committed files differ from a fresh run")
    args = parser.parse_args(argv)
    outputs = render_all()
    if args.check:
        stale = [path.name for path, text in outputs.items() if not path.is_file() or path.read_bytes() != text.encode("utf-8")]
        if stale:
            print("demo files differ from the generator output: " + ", ".join(stale))
            return 1
        print("demo files match the generator output")
        return 0
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for path, text in outputs.items():
        path.write_bytes(text.encode("utf-8"))
        print(f"wrote {path.relative_to(PROJECT_ROOT).as_posix()} ({len(text.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
