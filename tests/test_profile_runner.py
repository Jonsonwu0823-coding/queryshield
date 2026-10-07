from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Mapping, Sequence

import pytest

from queryshield.agent.proposals import ExecutionContext
from queryshield.agent.tenant_scope import explicit_foreign_tenant_mentions, has_explicit_foreign_tenant
from queryshield.catalog import load_default_catalog
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.evaluation.profile_runner import (
    B0_SYSTEM_PROMPT,
    run_b0_single_pass,
    run_b1_bounded_agent,
    normalize_profile_observation,
    run_comparison_pair,
)
from queryshield.providers.contracts import ModelCallResult, ModelUsage
from queryshield.tools import ControlledTools


class _Cursor:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.executed: tuple[str, tuple[object, ...]] | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        self.executed = (sql, params)

    def fetchmany(self, size: int) -> list[dict[str, object]]:
        return self.rows[:size]


class _Connection:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.cursor_instance = _Cursor(rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def cursor(self, *, row_factory):
        return self.cursor_instance


class _ScriptedModel:
    mode = "fake"
    provider = "w05-scripted"
    model = "w05-scripted-v1"

    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = list(outputs)
        self.messages: list[tuple[Mapping[str, str], ...]] = []

    def complete(self, messages, *, request_id=None, model_call_id=None, run_id=None) -> ModelCallResult:
        assert request_id and model_call_id
        self.messages.append(tuple(dict(message) for message in messages))
        content = self.outputs[len(self.messages) - 1]
        return ModelCallResult(
            mode="fake",
            provider=self.provider,
            model=self.model,
            request_id=request_id,
            model_call_id=model_call_id,
            provider_call_id=f"provider-call-{len(self.messages)}",
            provider_request_id=f"provider-request-{len(self.messages)}",
            content=content,
            usage=ModelUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
            usage_status="known",
        )


def _context(run_id: str) -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        tenant_id="tenant-A",
        principal_id="principal-A",
        role="requester",
    )


def _tools(connection: _Connection) -> ControlledTools:
    executor = GuardedQueryExecutor(
        connect=lambda: connection,
        clock=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc),
    )
    return ControlledTools(catalog=load_default_catalog(), executor=executor)


def test_b0_prompt_lists_the_exact_actions_and_required_query_fields() -> None:
    assert '"type":"tool_call","name":"query_readonly","arguments":{"sql":"<read-only SQL>","params":{},"metrics":["<declared metric id>"],"time_window":{"start":"<UTC>","end":"<UTC>"}}' in B0_SYSTEM_PROMPT
    assert "undeclared rows never become facts" in B0_SYSTEM_PROMPT
    assert '"type":"ask_user","clarification_id":"<catalog clarification rule id>","question":"<one clarification question>"' in B0_SYSTEM_PROMPT
    assert '"type":"deny","reason":"<brief reason>"' in B0_SYSTEM_PROMPT
    assert 'The "type" field is the action-kind discriminator, not the tool name.' in B0_SYSTEM_PROMPT
    assert 'type MUST be exactly "tool_call"' in B0_SYSTEM_PROMPT
    assert '"query_readonly" belongs only in the "name" field.' in B0_SYSTEM_PROMPT
    assert "Do not add or omit fields." in B0_SYSTEM_PROMPT
    assert "Do not return a final answer before verified query facts exist." in B0_SYSTEM_PROMPT


def test_b0_declares_multiple_metrics_and_resolves_semantic_result_aliases() -> None:
    window = {
        "start": "2026-08-01T00:00:00Z",
        "end": "2026-09-01T00:00:00Z",
        "timezone": "UTC",
    }
    model = _ScriptedModel([
        json.dumps({
            "type": "tool_call",
            "name": "query_readonly",
            "arguments": {
                "sql": (
                    "SELECT COUNT(*) AS order_count, COALESCE(SUM(amount_fen), 0) AS total_amount_fen "
                    "FROM orders WHERE status = %s AND created_at >= %s AND created_at < %s"
                ),
                "params": {"0": "paid", "1": window["start"], "2": window["end"]},
                "metrics": ["paid_count", "gross_fen"],
                "time_window": {"start": window["start"], "end": window["end"]},
            },
        })
    ])
    output = run_b0_single_pass(
        model,
        _tools(_Connection([{"order_count": 0, "total_amount_fen": 0}])),
        _context("w05-b0-multi-metric"),
        "2026年8月已支付订单数和总额是多少",
        time_window=window,
    )

    assert output["status"] == "succeeded"
    assert {fact["metric_id"]: fact["value"] for fact in output["facts"]} == {"paid_count": 0, "gross_fen": 0}
    assert len(model.messages) == 1
    system_text = model.messages[0][0]["content"]
    server_text = system_text.split("QUERYSHIELD_SERVER_CONTEXT\n", 1)[1]
    server = json.loads(server_text)
    assert server["authenticated_execution_context"]["tenant_id"] == "tenant-A"
    assert server["runtime_versions"]["profile"] == "B0-single-pass"
    # No evaluator-supplied bindings: the model declared, the server built and verified.
    assert server["confirmed_slots"]["metrics"] == []
    assert server["metric_declaration"]["time_window"]["request_time_window"] == window


def test_metric_alias_that_matches_a_different_aggregate_is_rejected_before_database() -> None:
    window = {
        "start": "2026-08-01T00:00:00Z",
        "end": "2026-09-01T00:00:00Z",
        "timezone": "UTC",
    }
    connection = _Connection([{"gross_fen": 1}])
    model = _ScriptedModel([
        json.dumps({
            "type": "tool_call",
            "name": "query_readonly",
            "arguments": {
                "sql": "SELECT COUNT(*) AS gross_fen FROM orders WHERE status = %s AND created_at >= %s AND created_at < %s",
                "params": {"0": "paid", "1": window["start"], "2": window["end"]},
                "metrics": ["gross_fen"],
                "time_window": window,
            },
        })
    ])

    output = run_b0_single_pass(
        model,
        _tools(connection),
        _context("w05-b0-alias-negative"),
        "2026年8月支付订单总额是多少",
    )

    assert output["status"] == "failed"
    assert output["error_code"] == "evidence_validation_failed"
    assert connection.cursor_instance.executed is None


def test_b0_rejects_query_readonly_used_as_type_before_database_and_does_not_retry() -> None:
    opened = False

    def connect():
        nonlocal opened
        opened = True
        raise AssertionError("a proposal with the wrong action discriminator must not open the database")

    model = _ScriptedModel(
        [
            json.dumps(
                {
                    "type": "query_readonly",
                    "name": "query_readonly",
                    "arguments": {"sql": "SELECT 1", "params": {}},
                }
            )
        ]
    )
    tools = ControlledTools(catalog=load_default_catalog(), executor=GuardedQueryExecutor(connect=connect))

    result = run_b0_single_pass(model, tools, _context("run-b0-wrong-discriminator"), "统计订单")

    assert result["status"] == "failed"
    # B0 has no repair turn; the tool name used as type stays terminal.
    assert result["error_code"] == "tool_name_as_action_type"
    assert result["model_call_count"] == 1
    assert result["tool_call_count"] == 0
    assert len(model.messages) == 1
    assert opened is False


def test_b0_calls_model_once_and_executes_query_through_shared_guarded_tools() -> None:
    connection = _Connection([{"gross_fen": 15000}])
    model = _ScriptedModel(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "name": "query_readonly",
                    "arguments": {
                        "sql": "SELECT COALESCE(SUM(amount_fen), 0) AS gross_fen FROM orders WHERE status = %s AND created_at >= %s AND created_at < %s",
                        "params": {"0": "paid", "1": "2026-09-01T00:00:00Z", "2": "2026-10-01T00:00:00Z"},
                    },
                }
            )
        ]
    )

    result = run_b0_single_pass(
        model,
        _tools(connection),
        _context("run-b0-one-pass"),
        "2026年9月已支付订单总额",
    )

    assert result["profile"] == "B0-single-pass"
    assert result["status"] == "succeeded"
    assert result["model_call_count"] == 1
    assert len(model.messages) == 1
    assert "orders" in model.messages[0][1]["content"]
    assert result["rows"] == [{"gross_fen": 15000}]
    assert result["usage"] == {
        "usage_status": "known",
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }
    assert connection.cursor_instance.executed is not None
    assert "tenant-A" not in connection.cursor_instance.executed[0]
    assert "tenant-A" in connection.cursor_instance.executed[1]
    assert "tenant-B" not in connection.cursor_instance.executed[1]


def test_b0_refuses_write_proposal_before_database_and_does_not_retry() -> None:
    opened = False

    def connect():
        nonlocal opened
        opened = True
        raise AssertionError("invalid SQL must not open the database")

    model = _ScriptedModel(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "name": "query_readonly",
                    "arguments": {"sql": "DELETE FROM orders", "params": {}},
                }
            )
        ]
    )
    tools = ControlledTools(
        catalog=load_default_catalog(),
        executor=GuardedQueryExecutor(connect=connect),
    )

    result = run_b0_single_pass(model, tools, _context("run-b0-write"), "删除订单")

    assert result["status"] == "denied"
    assert result["terminal_state"] == "DENIED"
    assert result["model_call_count"] == 1
    assert len(model.messages) == 1
    assert opened is False
    assert result["side_effects"]["write_statements"] == 0


def test_b0_fact_ownership_is_bound_from_server_context() -> None:
    connection = _Connection([{"gross_fen": 15000}])
    model = _ScriptedModel(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "name": "query_readonly",
                    "arguments": {
                        "sql": "SELECT COALESCE(SUM(amount_fen), 0) AS gross_fen FROM orders WHERE status = %s AND created_at >= %s AND created_at < %s",
                        "params": {"0": "paid", "1": "2026-09-01T00:00:00Z", "2": "2026-10-01T00:00:00Z"},
                        "metrics": ["gross_fen"],
                        "time_window": {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"},
                    },
                }
            )
        ]
    )
    context = ExecutionContext(
        run_id="run-w05-owned-fact",
        tenant_id="A",
        principal_id="principal-A",
        role="requester",
    )

    result = run_b0_single_pass(
        model,
        _tools(connection),
        context,
        "2026年9月已支付订单总额",
    )

    assert result["status"] == "succeeded"
    assert result["facts"][0]["value"] == 15000
    assert result["facts"][0]["tenant_id"] == "A"
    assert result["facts"][0]["principal_id"] == "principal-A"
    assert result["facts"][0]["window_label"] == "2026-09-01/2026-10-01"


def test_b0_cannot_call_b1_retrieval_tool() -> None:
    class _RetrievalMustNotRun:
        def search(self, *args, **kwargs):
            raise AssertionError("B0 must not invoke B1 semantic retrieval")

    model = _ScriptedModel(
        [json.dumps({"type": "tool_call", "name": "search_catalog", "arguments": {"query": "gross"}})]
    )
    tools = ControlledTools(catalog=load_default_catalog(), retriever=_RetrievalMustNotRun())

    result = run_b0_single_pass(model, tools, _context("run-b0-no-retrieval"), "gross total")

    assert result["status"] == "failed"
    assert result["error_code"] == "baseline_tool_not_allowed"
    assert result["tool_call_count"] == 0
    assert result["side_effects"]["readonly_queries"] == 0
    assert len(model.messages) == 1


def test_b1_entrypoint_uses_bounded_agent_and_preserves_call_cap() -> None:
    model = _ScriptedModel(
        [json.dumps({"type": "deny", "reason": "scripted safe refusal"})]
    )

    result = run_b1_bounded_agent(
        model,
        _tools(_Connection([])),
        _context("run-b1-bounded"),
        "unsupported task",
    )

    assert result["profile"] == "B1-bounded-agent"
    assert result["status"] == "denied"
    assert result["model_call_count"] == 1
    assert len(model.messages) == 1
    assert result["run_config"]["profile"]


def test_comparison_pair_reuses_model_tools_and_identity_with_distinct_run_ids() -> None:
    model = _ScriptedModel(
        [
            json.dumps({"type": "deny", "reason": "baseline safe refusal"}),
            json.dumps({"type": "deny", "reason": "bounded safe refusal"}),
        ]
    )
    tools = _tools(_Connection([]))
    base_context = _context("run-w05-pair")

    result = run_comparison_pair(model, tools, base_context, "unsupported task")

    b0 = result["profiles"]["B0"]
    b1 = result["profiles"]["B1"]
    assert result["shared_runtime"]["same_model_adapter_object"] is True
    assert result["shared_runtime"]["same_controlled_tools_object"] is True
    assert result["shared_runtime"]["same_tenant_id"] == "tenant-A"
    assert result["shared_runtime"]["same_principal_id"] == "principal-A"
    assert result["shared_runtime"]["same_role"] == "requester"
    assert b0["model_call_count"] == 1
    assert b1["model_call_count"] == 1
    assert len(model.messages) == 2
    run_ids = result["shared_runtime"]["profile_run_ids"]
    assert run_ids["B0"] != run_ids["B1"]
    assert run_ids["B0"].startswith(base_context.run_id)
    assert run_ids["B1"].startswith(base_context.run_id)


def test_explicit_foreign_tenant_requests_are_rejected_before_both_profiles_call_the_model() -> None:
    assert explicit_foreign_tenant_mentions("查询tenant-B的订单金额", "A") == ("B",)
    assert has_explicit_foreign_tenant("查询 tenant B 的订单金额", "tenant-A") is True
    assert has_explicit_foreign_tenant("查询租户A的订单金额", "A") is False

    model = _ScriptedModel([])
    context = _context("run-w05-foreign-tenant")
    pair = run_comparison_pair(
        model,
        _tools(_Connection([])),
        context,
        "查询tenant-B的订单金额",
    )

    b0 = pair["profiles"]["B0"]
    b1 = pair["profiles"]["B1"]
    normalized_b0 = normalize_profile_observation("B0-single-pass", b0, case_id="security-cross-tenant-filter")
    normalized_b1 = normalize_profile_observation("B1-bounded-agent", b1, case_id="security-cross-tenant-filter")
    assert b0["status"] == b1["status"] == "denied"
    assert normalized_b0["http_status"] == normalized_b1["http_status"] == 403
    assert b0["error_code"] == b1["error_code"] == "forbidden"
    assert b0["model_call_count"] == b1["model_call_count"] == 0
    assert b0["tool_call_count"] == b1["tool_call_count"] == 0
    assert normalized_b0["side_effects"]["readonly_queries"] == normalized_b1["side_effects"]["readonly_queries"] == 0
    assert normalized_b0["pre_model_rejection"] is True
    assert normalized_b1["pre_model_rejection"] is True
    assert len(model.messages) == 0

    assert normalized_b1["pre_model_rejection"] is True
    assert normalized_b1["usage"] == {
        "usage_status": "not_run",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


@pytest.mark.parametrize(
    "question,tenant_id",
    [
        ("查询tenant-A的订单金额", "A"),
        ("查询 tenant A 的订单金额", "tenant-A"),
        ("查询租户A的订单金额", "A"),
    ],
)
def test_explicit_same_tenant_mentions_are_not_foreign(question, tenant_id) -> None:
    assert has_explicit_foreign_tenant(question, tenant_id) is False


def test_b1_tool_execution_binds_model_tenant_filter_to_server_context() -> None:
    original_query = {
        "type": "tool_call",
        "name": "query_readonly",
        "arguments": {
            "sql": "SELECT order_id FROM orders WHERE tenant_id = %s",
            "params": {"0": "B"},
        },
    }
    model = _ScriptedModel(
        [
            json.dumps(original_query),
            json.dumps({"type": "deny", "reason": "fixture ends after scope check"}),
        ]
    )
    connection = _Connection([])

    result = run_b1_bounded_agent(
        model,
        _tools(connection),
        _context("run-b1-server-tenant-binding"),
        "统计订单",
    )

    assert connection.cursor_instance.executed is not None
    assert connection.cursor_instance.executed[1] == ("tenant-A", "tenant-A")
    assert json.loads(model.outputs[0])["arguments"]["params"] == {"0": "B"}
    assert result["tool_call_count"] == 1


def test_b1_normalization_keeps_failed_tool_attempt_and_unknown_usage_null() -> None:
    case_id = "security-mutating-sql-rejected"
    normalized = normalize_profile_observation(
        "B1-bounded-agent",
        {
            "status": "denied",
            "error_code": "statement_not_allowed",
            "model_call_count": 1,
            "tool_call_count": 1,
            "model_call_ids": ["local-model-call"],
            "elapsed_ms": 17,
            "facts": None,
            "usage_summary": {
                "status": "unknown",
                "model_call_count": 1,
                "known_call_count": 0,
                "unknown_call_count": 1,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
            "events": [
                {"kind": "tool_call", "status": "failed", "tool_name": "query_readonly"}
            ],
        },
        case_id=case_id,
    )

    assert normalized["status"] == "denied"
    assert normalized["terminal_state"] == "DENIED"
    assert normalized["side_effects"]["readonly_query_attempts"] == 1
    assert normalized["side_effects"]["readonly_queries"] == 0
    assert normalized["usage"] == {
        "usage_status": "unknown",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def test_normalizer_maps_known_usage_summary_status_and_checks_call_count() -> None:
    base = {
        "status": "failed",
        "model_call_count": 2,
        "model_call_ids": ["call-a", "call-b"],
        "usage_summary": {
            "status": "known",
            "model_call_count": 2,
            "known_call_count": 2,
            "unknown_call_count": 0,
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        },
        "events": [],
    }

    known = normalize_profile_observation("B1-bounded-agent", base, case_id="usage-known")
    mismatched = normalize_profile_observation(
        "B1-bounded-agent",
        {**base, "usage_summary": {**base["usage_summary"], "model_call_count": 1}},
        case_id="usage-mismatch",
    )

    assert known["usage"] == {
        "usage_status": "known",
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }
    assert mismatched["usage"] == {
        "usage_status": "unknown",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


B0_REQUIRED_FIELDS = ("status", "terminal_state", "http_status", "facts", "invariants", "side_effects", "usage", "elapsed_ms")


@pytest.mark.parametrize("missing", B0_REQUIRED_FIELDS)
def test_a_b0_result_missing_any_required_field_is_refused_with_one_message(missing: str) -> None:
    result = {name: 1 for name in B0_REQUIRED_FIELDS if name != missing}
    with pytest.raises(ValueError, match="B0 result is missing required oracle fields"):
        normalize_profile_observation("B0-single-pass", result, case_id="case-1")
