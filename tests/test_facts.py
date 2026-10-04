from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from queryshield.catalog import DEFAULT_CATALOG_VERSION
from queryshield.agent.context import NET_FEN_PLAN_ID
from queryshield.agent.proposals import ExecutionContext, FactRef, MetricBinding, ResultEvidence
from queryshield.facts import FactResolutionError, FactResolver, FACTS_SCHEMA_VERSION


def _context(*, run_id: str = "run-A-facts", principal_id: str = "principal-A") -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        tenant_id="tenant-A",
        principal_id=principal_id,
        role="requester",
    )


def _evidence(context: ExecutionContext, *, result_id: str = "result-A-net") -> ResultEvidence:
    return ResultEvidence.from_server_execution(
        context,
        result_id=result_id,
        rows=({"net_fen": 12000},),
        normalized_query="SELECT net_fen FROM orders",
        params={},
        observed_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
        policy_version="qs-sql-v1",
        catalog_version=DEFAULT_CATALOG_VERSION,
        metric_bindings=(
            MetricBinding(
                metric_id="net_fen",
                result_position="net_fen",
                unit="CNY_fen",
                time_window={
                    "start": "2026-09-01T00:00:00Z",
                    "end": "2026-10-01T00:00:00Z",
                    "timezone": "UTC",
                },
                catalog_source_id="commerce-v1",
                catalog_version=DEFAULT_CATALOG_VERSION,
                plan_id=NET_FEN_PLAN_ID,
            ),
        ),
        metric_plan_id=NET_FEN_PLAN_ID,
    )


def test_fact_resolver_uses_actual_rows_and_server_binding() -> None:
    context = _context()
    evidence = _evidence(context)
    first = FactResolver().resolve(
        (FactRef(result_id=evidence.result_id, metric_id="net_fen"),),
        context=context,
        evidences={evidence.result_id: evidence},
    )
    second = FactResolver().resolve(
        (FactRef(result_id=evidence.result_id, metric_id="net_fen"),),
        context=context,
        evidences={evidence.result_id: evidence},
    )

    assert first.schema_version == FACTS_SCHEMA_VERSION
    assert first.as_dict() == second.as_dict()
    assert first.facts[0].value == 12000
    assert first.facts[0].unit == "CNY_fen"
    assert first.facts[0].display_value == "120.00元"
    assert first.facts[0].time_window["timezone"] == "UTC"
    json.dumps(first.as_dict(), ensure_ascii=False)


def test_fact_resolver_rejects_wrong_run_or_untrusted_metric_binding() -> None:
    context = _context()
    evidence = _evidence(context)
    ref = FactRef(result_id=evidence.result_id, metric_id="net_fen")
    with pytest.raises(FactResolutionError, match="evidence_validation_failed"):
        FactResolver().resolve((ref,), context=_context(run_id="another-run"), evidences={evidence.result_id: evidence})

    with pytest.raises(FactResolutionError, match="evidence_validation_failed"):
        FactResolver().resolve(
            (FactRef(result_id=evidence.result_id, metric_id="gross_fen"),),
            context=context,
            evidences={evidence.result_id: evidence},
        )


def test_fact_resolver_rejects_unknown_result_and_model_value_fields() -> None:
    context = _context()
    with pytest.raises(FactResolutionError, match="evidence_validation_failed"):
        FactResolver().resolve(
            (FactRef(result_id="result-missing", metric_id="net_fen"),),
            context=context,
            evidences={},
        )
