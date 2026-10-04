from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from queryshield.db.guarded import render_scoped_select
from queryshield.policy.sql import parse_readonly_select
from scripts import check_commerce


class FixtureGuardedExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, tuple[object, ...]]] = []

    def execute(self, sql: str, *, context: object, params: tuple[object, ...]):
        self.calls.append((sql, context, params))
        tenant_id = context.tenant_id
        start = params[1] if sql != check_commerce.INJECTION_CONTROL_QUERY else params[2]
        end = params[2] if sql != check_commerce.INJECTION_CONTROL_QUERY else params[3]
        empty_window = start.year == 2027
        narrow_window = start == datetime(2026, 9, 9, tzinfo=timezone.utc)

        if sql == check_commerce.INJECTION_CONTROL_QUERY:
            rows = ({"paid_count": 0, "gross_fen": 0},)
        elif sql == check_commerce.PAID_SUMMARY_QUERY:
            if empty_window or narrow_window:
                rows = ({"paid_count": 0, "gross_fen": 0},)
            elif tenant_id == "A":
                rows = ({"paid_count": 2, "gross_fen": 15000},)
            else:
                rows = ({"paid_count": 1, "gross_fen": 990000},)
        else:
            assert sql == check_commerce.REFUND_SUMMARY_QUERY
            if empty_window or narrow_window:
                rows = ({"refund_fen": 0},)
            elif tenant_id == "A":
                rows = ({"refund_fen": 3000},)
            else:
                rows = ({"refund_fen": 10000},)

        evidence = SimpleNamespace(
            result_id=f"result-{len(self.calls)}",
            run_id=context.run_id,
            tenant_id=tenant_id,
            principal_id=context.principal_id,
            rows=rows,
            row_count=len(rows),
            query_sha256="a" * 64,
            params_sha256="b" * 64,
            policy_version="qs-sql-v1",
            catalog_version="catalog-v1",
            observed_at=datetime(2026, 9, 24, tzinfo=timezone.utc),
        )
        return SimpleNamespace(evidence=evidence)


def test_w05_commerce_preflight_uses_guarded_tenant_runs_and_records_actual_rows(
    monkeypatch,
) -> None:
    executor = FixtureGuardedExecutor()
    monkeypatch.setattr(check_commerce, "GuardedQueryExecutor", lambda: executor)
    observed: list[dict[str, object]] = []

    check_commerce.check_guarded_commerce(observed)

    assert len(executor.calls) == len(observed) == 9
    assert {context.tenant_id for _, context, _ in executor.calls} == {"A", "B"}
    assert all(context.principal_id == "w05-db01-checker" for _, context, _ in executor.calls)
    assert all(context.role == "requester" for _, context, _ in executor.calls)
    assert len({context.run_id for _, context, _ in executor.calls}) == 5
    assert observed[0]["rows"] == [{"paid_count": 2, "gross_fen": 15000}]
    assert observed[-1]["rows"] == [{"paid_count": 0, "gross_fen": 0}]
    assert observed[0]["result_id"] == "result-1"
    assert observed[0]["tenant_id"] == "A"
    assert observed[0]["run_id"] == "w05-db01-september-a"


def test_w05_commerce_injection_control_is_a_bound_filter_under_server_tenant_scope() -> None:
    statement = parse_readonly_select(check_commerce.INJECTION_CONTROL_QUERY)
    injected = "A' OR '1'='1"

    rendered = render_scoped_select(
        statement,
        tenant_id="A",
        input_params=(injected, "paid", check_commerce.START, check_commerce.END),
    )

    assert 'SELECT * FROM "orders" WHERE "tenant_id" = %s' in rendered.sql
    assert injected not in rendered.sql
    assert rendered.params == (
        0,
        "A",
        injected,
        "paid",
        check_commerce.START,
        check_commerce.END,
    )
