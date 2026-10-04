"""B3a: the W01 database checks read with the server-bound tenant context (RLS since W04)."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from types import SimpleNamespace

import pytest

from scripts import check_commerce, check_db, check_db_smoke

START = datetime(2026, 9, 1, tzinfo=timezone.utc)
END = datetime(2026, 10, 1, tzinfo=timezone.utc)


class _Cursor:
    def __init__(self, log: list[tuple[str, object]]) -> None:
        self.log = log

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: object = None) -> None:
        self.log.append(("query", params))

    def fetchone(self) -> dict[str, int]:
        return {"paid_count": 2, "gross_fen": 15000, "refund_fen": 3000, "net_fen": 12000}


class _Connection:
    def __init__(self) -> None:
        self.log: list[tuple[str, object]] = []
        self.tenant: str | None = None

    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def cursor(self, *, row_factory: object) -> _Cursor:
        return _Cursor(self.log)

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> SimpleNamespace:
        if "set_config" in sql:
            self.tenant = str(params[0])
            self.log.append(("bind", self.tenant))
            return SimpleNamespace(fetchone=lambda: (self.tenant,))
        # An RLS-forced table shows a tenant only its own rows and nothing without context.
        rows = check_db.TENANT_FIXTURE_COUNTS.get(self.tenant, (0, 0, 0))
        self.log.append(("count", self.tenant))
        return SimpleNamespace(fetchone=lambda: rows)

    def rollback(self) -> None:
        self.tenant = None
        self.log.append(("rollback", None))


def test_commerce_summary_binds_the_tenant_before_the_query(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _Connection()
    monkeypatch.setattr(check_commerce, "connect_readonly", lambda: connection)

    summary = check_commerce.fetch_commerce_summary("A", START, END)

    assert summary["paid_count"] == 2
    assert connection.log[0] == ("bind", "A")
    assert connection.log[1][0] == "query" and connection.log[1][1][0] == "A"


def test_commerce_summary_without_a_row_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _Connection()
    monkeypatch.setattr(_Cursor, "fetchone", lambda self: None)
    monkeypatch.setattr(check_commerce, "connect_readonly", lambda: connection)
    with pytest.raises(AssertionError, match="returned no row"):
        check_commerce.fetch_commerce_summary("A", START, END)


def test_tenant_bound_counts_read_each_tenant_in_its_own_transaction() -> None:
    connection = _Connection()

    counts = check_db.tenant_bound_counts(connection)

    assert counts == check_db.TENANT_FIXTURE_COUNTS
    assert [tuple(sum(v[i] for v in counts.values()) for i in range(3))] == [(3, 4, 3)]
    binds = [entry for entry in connection.log if entry[0] == "bind"]
    assert [tenant for _, tenant in binds] == ["A", "B"]
    assert [entry for entry in connection.log if entry[0] == "count"] == [("count", "A"), ("count", "B")]


def test_unbound_read_sees_nothing_which_is_what_the_old_checks_tripped_over() -> None:
    connection = _Connection()
    assert connection.execute(check_db.COUNT_SQL).fetchone() == (0, 0, 0)


needs_database = pytest.mark.skipif(
    not os.environ.get("QUERYSHIELD_DATABASE_URL"),
    reason="needs the queryshield_test database (README, 不用 Compose 的本机安装; set QUERYSHIELD_DATABASE_URL)",
)


@needs_database
def test_w01_db_checks_pass_against_the_rls_fixture_database(capsys: pytest.CaptureFixture[str]) -> None:
    assert check_db.main() == 0
    assert check_db_smoke.main() == 0
    check_commerce.check_legacy_commerce()
    out = capsys.readouterr().out
    assert "write_probe_cleanup=fixture_counts_unchanged" in out
    assert "fixture_counts=customers:3 orders:4 refunds:3" in out
