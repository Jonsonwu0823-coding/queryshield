from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from queryshield.agent.proposals import ExecutionContext, FactRef, MetricBinding  # noqa: E402
from queryshield.catalog import DEFAULT_CATALOG_VERSION  # noqa: E402
from queryshield.db.guarded import GuardedQueryExecutor  # noqa: E402
from queryshield.facts import FactResolver  # noqa: E402
from queryshield.tools import ControlledTools, ToolError  # noqa: E402


class _Cursor:
    def __init__(self) -> None:
        self.executed: tuple[str, tuple[object, ...]] | None = None

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        self.executed = (sql, params)

    def fetchmany(self, size: int) -> list[dict[str, object]]:
        return [{"paid_count": 2}]


class _Connection:
    def __init__(self) -> None:
        self.cursor_instance = _Cursor()

    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def cursor(self, *, row_factory: object) -> _Cursor:
        return self.cursor_instance


def main() -> int:
    context = ExecutionContext(
        run_id="run-t02-probe",
        tenant_id="tenant-A",
        principal_id="principal-A-requester",
        role="requester",
    )
    connection = _Connection()
    tools = ControlledTools(
        executor=GuardedQueryExecutor(
            connect=lambda: connection,
            clock=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc),
        )
    )

    search = tools.search_catalog({"query": "营业额", "top_k": 3}, context=context)
    description = tools.describe_tables({"tables": ["orders"]}, context=context)
    query = tools.query_readonly(
        {
            "sql": "SELECT COUNT(*) AS paid_count FROM orders WHERE status = %s",
            "params": {"0": "paid"},
        },
        context=context,
        metric_bindings=(
            MetricBinding(
                metric_id="paid_count",
                result_position="paid_count",
                unit="count",
                time_window={
                    "start": "2026-09-01T00:00:00Z",
                    "end": "2026-10-01T00:00:00Z",
                    "timezone": "UTC",
                },
                catalog_source_id="commerce-v1",
                catalog_version=DEFAULT_CATALOG_VERSION,
            ),
        ),
    )
    evidence = tools.get_result_evidence(query["result_id"], context=context)
    facts = FactResolver().resolve(
        (FactRef(result_id=evidence.result_id, metric_id="paid_count"),),
        context=context,
        evidences={evidence.result_id: evidence},
    )

    denials: list[str] = []
    for label, arguments in (
        ("unknown_table", {"tables": ["secrets"]}),
        ("sensitive_value", {"sql": "SELECT c.name FROM customers AS c", "params": {}}),
    ):
        try:
            if label == "unknown_table":
                tools.describe_tables(arguments, context=context)
            else:
                tools.query_readonly(arguments, context=context)
        except ToolError as exc:
            denials.append(f"{label}:{exc.code}")
        else:
            raise AssertionError(f"expected denial: {label}")

    assert search["items"][0]["id"] == "metric.gross_fen"
    assert description["tables"][0]["columns"] == [
        "amount_fen",
        "created_at",
        "customer_id",
        "order_id",
        "status",
        "tenant_id",
    ]
    assert query["rows"] == [{"paid_count": 2}]
    assert connection.cursor_instance.executed is not None
    assert connection.cursor_instance.executed[1] == ("tenant-A", "paid")
    assert facts.facts[0].display_value == "2笔"
    assert set(denials) == {"unknown_table:table_not_allowed", "sensitive_value:approval_required"}
    print(
        json.dumps(
            {
                "status": "pass",
                "search_first_id": search["items"][0]["id"],
                "describe_tables": description["tables"],
                "query": {
                    "row_count": query["row_count"],
                    "result_id": query["result_id"],
                    "policy_version": query["policy_version"],
                    "server_params": list(connection.cursor_instance.executed[1]),
                },
                "facts": facts.as_dict(),
                "denials": sorted(denials),
                "database_mode": "injected_fake_connection_only",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
