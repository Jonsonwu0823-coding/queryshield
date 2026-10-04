from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from queryshield.agent.proposals import (
    ExecutionContext,
    MetricBinding,
    ModelCallStore,
    ProposalParseError,
    ResultEvidence,
    ToolCallAction,
    parse_query_proposal,
)


def _context() -> ExecutionContext:
    return ExecutionContext(
        run_id="run-A-1",
        tenant_id="tenant-A",
        principal_id="principal-A-requester",
        role="requester",
    )


def test_valid_tool_proposal_is_bound_to_server_context() -> None:
    raw = json.dumps(
        {
            "type": "tool_call",
            "name": "search_catalog",
            "arguments": {"query": "paid orders", "top_k": 3},
        }
    )

    proposal = parse_query_proposal(
        raw,
        context=_context(),
        model_call_id="local-call-1",
    )

    assert isinstance(proposal.action, ToolCallAction)
    assert proposal.action.arguments == {"query": "paid orders", "top_k": 3}
    assert proposal.run_id == "run-A-1"
    assert proposal.context.tenant_id == "tenant-A"
    assert proposal.context.principal_id == "principal-A-requester"
    assert proposal.model_call_id == "local-call-1"
    assert "content" not in proposal.as_dict()
    assert len(proposal.content_sha256) == 64


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("{", "invalid_json"),
        (
            '{"type":"tool_call","name":"unknown_tool","arguments":{}}',
            "unknown_action",
        ),
        ('{"type":"tool_call","name":"search_catalog"}', "missing_field"),
        (
            '{"type":"tool_call","name":"search_catalog","arguments":{"query":"x","tenant_id":"B"}}',
            "unknown_field",
        ),
        (
            '{"type":"tool_call","name":"query_readonly","arguments":{"sql":123,"params":{}}}',
            "invalid_field",
        ),
        (
            '{"type":"final_answer","answer":"ok","source_ids":[],"fact_refs":[],"tenant_id":"B"}',
            "unknown_field",
        ),
    ],
)
def test_invalid_proposals_are_rejected_before_execution(raw: str, code: str) -> None:
    with pytest.raises(ProposalParseError) as error:
        parse_query_proposal(raw, context=_context(), model_call_id="local-call-1")

    assert error.value.code == code


def test_missing_nested_field_is_named_without_relaxing_the_parser() -> None:
    with pytest.raises(ProposalParseError, match=r"missing_field.*query"):
        parse_query_proposal(
            '{"type":"tool_call","name":"search_catalog","arguments":{}}',
            context=_context(),
            model_call_id="local-missing-query",
        )


def test_unknown_field_is_named_without_relaxing_the_parser() -> None:
    with pytest.raises(ProposalParseError, match=r"unknown_field.*reasoning"):
        parse_query_proposal(
            '{"type":"ask_user","question":"需要澄清","reasoning":"hidden"}',
            context=_context(),
            model_call_id="local-unknown-reasoning",
        )


def test_tool_arguments_must_be_nested_under_arguments() -> None:
    with pytest.raises(ProposalParseError, match=r"unknown_field.*query"):
        parse_query_proposal(
            '{"type":"tool_call","name":"search_catalog","query":"退款后净额"}',
            context=_context(),
            model_call_id="local-top-level-query",
        )


def test_transport_retry_reuses_logical_call_but_next_call_is_new() -> None:
    store = ModelCallStore()
    first = store.new_call("run-A-1", request_id="request-1")
    retry = store.transport_retry(first, request_id="request-2")
    next_call = store.new_call("run-A-1", request_id="request-3")

    assert retry.model_call_id == first.model_call_id
    assert retry.request_id != first.request_id
    assert retry.attempt_kind == "transport_retry"
    assert store.get("run-A-1", first.model_call_id) == first
    assert len(store.attempts("run-A-1", first.model_call_id)) == 2
    assert next_call.model_call_id != first.model_call_id


def test_result_evidence_uses_server_context_and_actual_row_count() -> None:
    evidence = ResultEvidence.from_server_execution(
        _context(),
        result_id="result-A-1",
        rows=({"paid_count": 2},),
        normalized_query="SELECT paid_count FROM commerce_summary",
        params={"window_start": "2026-09-01T00:00:00Z"},
        observed_at=datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc),
        policy_version="qs-sql-v1",
        catalog_version="catalog-v1",
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
                catalog_version="catalog-v1",
            ),
        ),
    )

    record = evidence.as_dict()
    assert record["tenant_id"] == "tenant-A"
    assert record["principal_id"] == "principal-A-requester"
    assert record["row_count"] == 1
    assert record["rows"] == [{"paid_count": 2}]
    assert record["metric_bindings"][0]["metric_id"] == "paid_count"
