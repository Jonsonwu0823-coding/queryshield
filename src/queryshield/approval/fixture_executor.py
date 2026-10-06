"""The deterministic Fake database boundary of the checks.

It lives in ``src/`` only because the frozen ``scripts/check_state.py`` (and ``check_eval.py``) import
it from ``queryshield.approval.service``, which re-exports it, and because the product selects it
with ``QUERYSHIELD_FAKE_DB=1`` and the Fake provider.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Callable
from uuid import uuid4

from queryshield.agent.proposals import ExecutionContext, MetricBinding, ResultEvidence
from queryshield.approval.versions import BOUND_CATALOG_VERSION, BOUND_POLICY_VERSION
from queryshield.db.state_store import utc_now


@dataclass(frozen=True)
class FixtureResult:
    evidence: ResultEvidence
    rows: tuple[dict[str, object], ...]


class FixtureQueryExecutor:
    """Deterministic Fake database boundary; never used as real evidence.

    It mirrors the commerce fixture (tenant A: c1 甲 paid 10000, c2 乙 paid
    5000, refunds 3000; tenant B: c1 丙 paid 990000, refunds 10000) for the
    projections the product Fake and the server net plan use.  It ignores time
    windows; it is only selected with the Fake provider.
    """

    _CUSTOMERS = {"A": (("c1", "甲", 10000), ("c2", "乙", 5000)), "B": (("c1", "丙", 990000),)}
    _REFUND_FEN = {"A": 3000, "B": 10000}

    def __init__(self, *, clock: Callable[[], datetime] = utc_now) -> None:
        self.clock = clock
        self.sql_calls = 0

    def _rows(self, sql: str, context: ExecutionContext, params: Sequence[object]) -> tuple[dict[str, object], ...]:
        lowered = " ".join(sql.lower().split())
        projection = lowered.split(" from ", 1)[0]
        customers = self._CUSTOMERS.get(context.tenant_id, ())
        paid_total = sum(amount for _, _, amount in customers)
        if "customers" in lowered and "name" in projection:
            wanted = {value for value in params if isinstance(value, str) and re.fullmatch(r"c\d+", value)}
            selected = [item for item in customers if not wanted or item[0] in wanted]
            return tuple(
                {key: value for key, value in (("customer_id", customer_id), ("name", name)) if key in projection}
                for customer_id, name, _ in selected
            )
        if "group by" in lowered and "customer_id" in lowered:
            alias = "gross_fen" if "gross_fen" in projection else None
            return tuple(
                {"customer_id": customer_id, **({alias: amount} if alias else {})}
                for customer_id, _, amount in customers
            )
        row: dict[str, object] = {}
        if "paid_count" in projection or ("count(" in projection and "gross_fen" not in projection):
            row["paid_count"] = len(customers)
        if "gross_fen" in projection:
            row["gross_fen"] = paid_total
        if "refund_fen" in projection:
            row["refund_fen"] = self._REFUND_FEN.get(context.tenant_id, 0)
        if "net_fen" in projection:
            row["net_fen"] = paid_total - self._REFUND_FEN.get(context.tenant_id, 0)
        return (row,) if row else ()

    def execute(
        self,
        sql: str,
        *,
        context: ExecutionContext,
        params: Sequence[object] = (),
        metric_bindings: Sequence[MetricBinding] = (),
    ) -> FixtureResult:
        self.sql_calls += 1
        rows = self._rows(sql, context, tuple(params))
        evidence = ResultEvidence.from_server_execution(
            context,
            result_id=f"result-{uuid4()}",
            rows=rows,
            normalized_query=sql,
            params=tuple(params),
            observed_at=self.clock(),
            policy_version=BOUND_POLICY_VERSION,
            catalog_version=BOUND_CATALOG_VERSION,
            metric_bindings=metric_bindings,
        )
        return FixtureResult(evidence=evidence, rows=rows)
