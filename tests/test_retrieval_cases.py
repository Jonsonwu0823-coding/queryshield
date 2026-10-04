from __future__ import annotations

from queryshield.agent.proposals import ExecutionContext
from queryshield.evaluation import load_development_cases, run_development_catalog_retrieval
from queryshield.tools import ControlledTools


def test_catalog_development_queries_use_fixed_top_k_and_keep_empty_hits() -> None:
    cases = load_development_cases()
    records = run_development_catalog_retrieval(
        ControlledTools(),
        context=ExecutionContext(
            run_id="run-eval-1",
            tenant_id="tenant-A",
            principal_id="principal-A",
            role="requester",
        ),
        cases=cases,
    )

    assert len(records) == 12
    assert all(record["top_k"] == 3 for record in records)
    assert any(record["empty_hit"] for record in records)
    assert all(len(record["items"]) <= 3 for record in records)
