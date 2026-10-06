from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import re

from queryshield.catalog import DEFAULT_CATALOG_VERSION
from queryshield.agent import BoundedAgent, GraphLimits, ModelCallStore
from queryshield.agent.config import RunConfig
from queryshield.agent.context import NET_FEN_PLAN_ID, NET_FEN_TIME_WINDOW
from queryshield.agent.graph import _usage_summary
from queryshield.agent.proposals import ExecutionContext, MetricBinding
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.providers.contracts import ModelCallResult, ModelUsage
from queryshield.catalog import load_default_catalog
from queryshield.tools import ControlledTools


class _FakeCursor:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        normalized = sql.lower()
        if 'as "gross_fen"' in normalized:
            self.rows = [{"gross_fen": 15000}]
        elif 'as "refund_fen"' in normalized:
            self.rows = [{"refund_fen": 3000}]

    def fetchmany(self, size: int) -> list[dict[str, object]]:
        return self.rows[:size]


class _FakeConnection:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.cursor_instance = _FakeCursor(rows)

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def cursor(self, *, row_factory: object) -> _FakeCursor:
        return self.cursor_instance


class _ScriptedModel:
    mode = "fake"

    def __init__(self, contents: Sequence[str]) -> None:
        self.contents = list(contents)
        self.messages: list[tuple[Mapping[str, str], ...]] = []
        self.model_call_ids: list[str] = []

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        request_id: str | None = None,
        model_call_id: str | None = None,
    ) -> ModelCallResult:
        assert request_id is not None
        assert model_call_id is not None
        index = len(self.messages)
        self.messages.append(tuple(dict(message) for message in messages))
        self.model_call_ids.append(model_call_id)
        return ModelCallResult(
            mode="fake",
            provider="scripted-test",
            model="scripted-model",
            request_id=request_id,
            model_call_id=model_call_id,
            provider_call_id=f"provider-call-{index + 1}",
            provider_request_id=f"provider-request-{index + 1}",
            content=self.contents[index],
            usage=ModelUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            usage_status="known",
        )


def _context(run_id: str) -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        tenant_id="tenant-A",
        principal_id="principal-A",
        role="requester",
    )


def _tools() -> ControlledTools:
    connection = _FakeConnection([{"gross_fen": 3000}])
    executor = GuardedQueryExecutor(
        connect=lambda: connection,
        clock=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc),
    )
    return ControlledTools(executor=executor)


def _facts_tools() -> ControlledTools:
    connection = _FakeConnection([{"net_fen": 12000}])
    executor = GuardedQueryExecutor(
        connect=lambda: connection,
        clock=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc),
    )
    return ControlledTools(executor=executor)


def _tool_call(name: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "name": name, "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def test_graph_runs_retrieval_query_answer_with_unique_model_calls() -> None:
    model = _ScriptedModel(
        [
            _tool_call("search_catalog", {"query": "营业额", "top_k": 3}),
            _tool_call(
                "query_readonly",
                {
                    "sql": "SELECT COALESCE(SUM(amount_fen), 0) AS gross_fen FROM orders WHERE status = %s",
                    "params": {"0": "paid"},
                },
            ),
            json.dumps(
                {
                    "type": "final_answer",
                    "answer": "已从受控查询返回结果。",
                    "source_ids": ["commerce-v1"],
                    "fact_refs": [],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]
    )
    runtime = BoundedAgent(model, tools=_tools(), call_store=ModelCallStore())

    result = runtime.run(_context("run-graph-normal"), "2026年9月已支付订单总额")

    assert result.status == "succeeded"
    assert result.model_call_count == 3
    assert result.tool_call_count == 2
    assert len(result.model_call_ids) == 3
    assert len(set(result.model_call_ids)) == 3
    assert model.model_call_ids == list(result.model_call_ids)
    assert result.action is not None
    assert result.action["type"] == "final_answer"
    assert [event["kind"] for event in result.events] == [
        "model_call",
        "tool_call",
        "model_call",
        "tool_call",
        "model_call",
        "answer",
    ]
    assert all("content" not in event for event in result.events)
    assert all(
        message["role"] != "system" or "QUERYSHIELD_DATA" not in message["content"]
        for messages in model.messages
        for message in messages
    )
    assert set(runtime.graph.get_graph().nodes) >= {
        "model_decision",
        "execute_tool",
        "finish",
    }


def test_graph_stops_before_the_seventh_model_call_on_a_deliberate_loop() -> None:
    loop_action = _tool_call("search_catalog", {"query": "营业额", "top_k": 1})
    model = _ScriptedModel([loop_action] * 7)
    runtime = BoundedAgent(model, tools=_tools(), call_store=ModelCallStore())

    result = runtime.run(_context("run-graph-loop"), "故意循环测试")

    assert result.status == "limit_reached"
    assert result.error_code == "model_call_limit"
    assert result.model_call_count == 6
    assert result.tool_call_count == 6
    assert len(model.model_call_ids) == 6
    assert len(set(model.model_call_ids)) == 6
    assert result.events[-1]["kind"] == "limit"
    assert result.events[-1]["error_code"] == "model_call_limit"


def test_waiting_user_resume_preserves_run_and_adds_clarification() -> None:
    model = _ScriptedModel(
        [
            json.dumps(
                {"type": "ask_user", "question": "销售额按毛额还是净额？"},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            _tool_call(
                "query_readonly",
                {
                    "sql": "SELECT COALESCE(SUM(amount_fen), 0) AS net_fen FROM orders WHERE status = %s",
                    "params": {"0": "paid"},
                },
            ),
            json.dumps(
                {
                    "type": "final_answer",
                    "answer": "净额查询已完成。",
                    "source_ids": ["commerce-v1"],
                    "fact_refs": [],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]
    )
    context = _context("run-graph-clarification")
    runtime = BoundedAgent(model, tools=_tools(), call_store=ModelCallStore())

    waiting = runtime.run(context, "销售额")
    assert waiting.status == "waiting_user"
    # The ask names catalog rule clarify.metric_basis (销售额 is open), so
    # the server keeps the catalog question, never the model's text.
    assert waiting.action == {
        "type": "ask_user",
        "question": "你要看支付订单总额（gross_fen），还是退款后净额（net_fen）？",
        "clarification_id": "clarify.metric_basis",
    }
    assert waiting.model_call_count == 1
    assert waiting.tool_call_count == 0
    assert waiting.usage_summary["status"] == "known"
    assert waiting.usage_summary["total_tokens"] == 12
    checkpoint = runtime.export_waiting_checkpoint(context.run_id)
    assert checkpoint["model_call_count"] == 1
    assert checkpoint["model_call_ids"] == list(waiting.model_call_ids)
    assert checkpoint["events"][0]["usage_status"] == "known"

    restored_runtime = BoundedAgent(model, tools=_tools(), call_store=ModelCallStore(), run_config=runtime.run_config)
    result = restored_runtime.resume_from_checkpoint(context, "净额", checkpoint)

    assert result.status == "succeeded"
    assert result.run_id == context.run_id
    assert result.model_call_count == 3
    assert result.tool_call_count == 1
    assert result.repair_count == 0
    assert len(result.events) == 6  # + the clarification_review of the first ask
    assert result.model_call_ids[0] == waiting.model_call_ids[0]
    assert result.usage_summary["status"] == "known"
    assert result.usage_summary["model_call_count"] == 3
    assert result.usage_summary["total_tokens"] == 36
    assert any(
        message["role"] == "user" and message["content"] == "净额"
        for messages in model.messages[1:]
        for message in messages
    )


def test_query_repair_is_allowed_once_for_structural_error() -> None:
    model = _ScriptedModel(
        [
            _tool_call(
                "query_readonly",
                {
                    "sql": "SELECT missing.amount_fen FROM orders AS o",
                    "params": {},
                },
            ),
            _tool_call(
                "query_readonly",
                {
                    "sql": "SELECT o.amount_fen FROM orders AS o",
                    "params": {},
                },
            ),
            json.dumps(
                {
                    "type": "final_answer",
                    "answer": "修复后的查询完成。",
                    "source_ids": ["commerce-v1"],
                    "fact_refs": [],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]
    )
    runtime = BoundedAgent(model, tools=_tools(), call_store=ModelCallStore())

    result = runtime.run(_context("run-graph-repair-once"), "订单金额")

    assert result.status == "succeeded"
    assert result.repair_count == 1
    assert result.model_call_count == 3
    assert result.tool_call_count == 2
    assert [event["kind"] for event in result.events] == [
        "model_call",
        "tool_call",
        "query_repair",
        "model_call",
        "tool_call",
        "model_call",
        "answer",
    ]


def test_unsupported_sql_shape_uses_one_bounded_repair_and_exposes_fixed_reason() -> None:
    model = _ScriptedModel(
        [
            _tool_call(
                "query_readonly",
                {
                    "sql": "SELECT o.amount_fen FROM orders AS o LEFT JOIN refunds AS r ON o.order_id = r.order_id",
                    "params": {},
                },
            ),
            _tool_call(
                "query_readonly",
                {"sql": "SELECT o.amount_fen FROM orders AS o", "params": {}},
            ),
            json.dumps(
                {
                    "type": "final_answer",
                    "answer": "修复后的查询完成。",
                    "source_ids": ["commerce-v1"],
                    "fact_refs": [],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]
    )
    runtime = BoundedAgent(model, tools=_tools(), call_store=ModelCallStore())

    result = runtime.run(_context("run-graph-unsupported-repair"), "订单金额")

    assert result.status == "succeeded"
    assert result.repair_count == 1
    assert result.model_call_count == 3
    assert result.tool_call_count == 2
    assert any(
        "only INNER JOIN is supported" in message["content"]
        for message in model.messages[1]
        if message["role"] == "user"
    )


def test_query_repair_stops_after_one_repeat() -> None:
    bad_query = _tool_call(
        "query_readonly",
        {
            "sql": "SELECT missing.amount_fen FROM orders AS o",
            "params": {},
        },
    )
    model = _ScriptedModel([bad_query, bad_query, _tool_call("search_catalog", {"query": "never-called"})])
    runtime = BoundedAgent(model, tools=_tools(), call_store=ModelCallStore())

    result = runtime.run(_context("run-graph-repair-limit"), "订单金额")

    assert result.status == "failed"
    assert result.error_code == "query_repair_limit"
    assert result.model_call_count == 2
    assert result.tool_call_count == 2
    assert result.repair_count == 1


def test_security_refusal_does_not_enter_query_repair() -> None:
    model = _ScriptedModel(
        [
            _tool_call(
                "query_readonly",
                {"sql": "SELECT * FROM secrets", "params": {}},
            ),
            _tool_call("search_catalog", {"query": "never-called"}),
        ]
    )
    runtime = BoundedAgent(model, tools=_tools(), call_store=ModelCallStore())

    result = runtime.run(_context("run-graph-security-refusal"), "读取秘密表")

    assert result.status == "denied"
    assert result.error_code == "table_not_allowed"
    assert result.model_call_count == 1
    assert result.tool_call_count == 1
    assert result.repair_count == 0


def test_final_answer_is_server_verified_and_trace_keeps_only_summaries() -> None:
    model = _ScriptedModel(
        [
            _tool_call("search_catalog", {"query": "净额", "top_k": 3}),
            _tool_call(
                "query_readonly",
                {
                    "sql": "SELECT COALESCE(SUM(amount_fen), 0) AS net_fen FROM orders WHERE status = %s",
                    "params": {"0": "paid"},
                },
            ),
            json.dumps(
                {
                    "type": "final_answer",
                    "answer": "模型草稿：净额是120.00元。",
                    "source_ids": ["spoofed-source"],
                    "fact_refs": [{"result_id": "placeholder", "metric_id": "net_fen"}],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]
    )

    # The model must reference the actual query result ID, not invent one.
    original_complete = model.complete

    def complete_with_result_ref(*args: object, **kwargs: object) -> ModelCallResult:
        if len(model.messages) == 2:
            messages = args[0]
            message_text = "\n".join(str(message["content"]) for message in messages)  # type: ignore[index]
            match = re.search(r'"result_id":"([^"]+)"', message_text)
            assert match is not None
            result_id = match.group(1)
            model.contents[2] = json.dumps(
                {
                    "type": "final_answer",
                    "answer": "模型草稿：净额是120.00元。",
                    "source_ids": ["spoofed-source"],
                    "fact_refs": [{"result_id": result_id, "metric_id": "net_fen"}],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return original_complete(*args, **kwargs)

    model.complete = complete_with_result_ref  # type: ignore[method-assign]
    runtime = BoundedAgent(model, tools=_facts_tools(), call_store=ModelCallStore())
    runtime_result = runtime.run(
        _context("run-graph-verified-fact"),
        "2026年9月退款后净额",
        metric_bindings=(
            MetricBinding(
                metric_id="net_fen",
                result_position="net_fen",
                unit="CNY_fen",
                time_window={
                    **NET_FEN_TIME_WINDOW,
                },
                catalog_source_id="commerce-v1",
                catalog_version=DEFAULT_CATALOG_VERSION,
                plan_id=NET_FEN_PLAN_ID,
            ),
        ),
    )

    assert runtime_result.status == "succeeded"
    assert runtime_result.answer == (
        "已核实：退款后净额：120.00元（2026-09-01T00:00:00Z至2026-10-01T00:00:00Z，UTC）\n"
        "口径：退款后净额（net_fen）；依据：问题中提到‘退款后净额’。前提：退款按同一 UTC 窗口内、已支付订单的退款计算。"
    )
    assert runtime_result.action is not None
    assert runtime_result.action["source_ids"] == ["commerce-v1"]
    assert runtime_result.facts is not None
    assert runtime_result.facts["facts"][0]["value"] == 12000
    assert runtime_result.usage_summary["total_tokens"] == 36
    system_payload = json.loads(model.messages[0][0]["content"].split("\n", 1)[1])
    assert system_payload["confirmed_slots"] == {
        "metric": "net_fen",
        "time_window": dict(NET_FEN_TIME_WINDOW),
        "metrics": [{
            "metric_id": "net_fen",
            "result_position": "net_fen",
            "unit": "CNY_fen",
            "time_window": dict(NET_FEN_TIME_WINDOW),
        }],
    }
    trace_text = json.dumps(runtime_result.events, ensure_ascii=False)
    assert "模型草稿：净额是120.00元。" not in trace_text
    assert "SELECT COALESCE(SUM(amount_fen), 0) AS net_fen FROM orders WHERE status = %s" not in trace_text


def test_b1_prompt_receives_all_server_confirmed_metric_bindings() -> None:
    catalog = load_default_catalog()
    window = {"start": "2026-08-01T00:00:00Z", "end": "2026-09-01T00:00:00Z", "timezone": "UTC"}
    bindings = tuple(
        MetricBinding(
            metric_id=metric_id,
            result_position=metric_id,
            unit=str(catalog.metric(metric_id).payload["unit"]),
            time_window=window,
            catalog_source_id=catalog.metric(metric_id).source_id,
            catalog_version=catalog.catalog_version,
        )
        for metric_id in ("paid_count", "gross_fen")
    )
    model = _ScriptedModel(['{"type":"ask_user","question":"请补充统计范围。"}'])
    result = BoundedAgent(model, tools=_tools(), run_config=RunConfig(profile="B1-bounded-agent")).run(
        _context("run-multiple-metric-context"),
        "2026年8月已支付订单数和总额是多少",
        metric_bindings=bindings,
    )

    assert result.status == "waiting_user"
    server = json.loads(model.messages[0][0]["content"].split("\n", 1)[1])
    assert [item["metric_id"] for item in server["confirmed_slots"]["metrics"]] == ["paid_count", "gross_fen"]
    assert all(item["time_window"] == window for item in server["confirmed_slots"]["metrics"])


def test_b1_scalar_metric_cannot_finish_without_a_verified_fact_reference() -> None:
    catalog = load_default_catalog()
    entry = catalog.metric("gross_fen")
    window = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z", "timezone": "UTC"}
    binding = MetricBinding(
        metric_id="gross_fen",
        result_position="gross_fen",
        unit=str(entry.payload["unit"]),
        time_window=window,
        catalog_source_id=entry.source_id,
        catalog_version=catalog.catalog_version,
    )
    model = _ScriptedModel([
        _tool_call(
            "query_readonly",
            {
                "sql": "SELECT COALESCE(SUM(amount_fen), 0) AS gross_fen FROM orders WHERE status = %s AND created_at >= %s AND created_at < %s",
                "params": {"0": "paid", "1": window["start"], "2": window["end"]},
            },
        ),
        json.dumps({"type": "final_answer", "answer": "金额是150元。", "source_ids": [], "fact_refs": []}),
    ])
    result = BoundedAgent(model, tools=_tools(), run_config=RunConfig(profile="B1-bounded-agent")).run(
        _context("run-missing-required-fact"),
        "2026年9月支付金额是多少",
        metric_bindings=(binding,),
    )

    assert result.status == "failed"
    assert result.error_code == "evidence_validation_failed"
    assert result.answer is None
    assert result.facts is None


def test_resume_checkpoint_keeps_nonzero_fake_history_ids_usage_and_budget() -> None:
    """Synthetic IDs exercise the exhausted-budget negative control only."""
    context = _context("run-resume-history-budget")
    run_config = RunConfig(profile="B1-bounded-agent")
    checkpoint = BoundedAgent.prepared_waiting_user_checkpoint(
        context,
        "销售额是多少？",
        run_config=run_config,
    )
    history_ids = [f"fake-history-call-{index}" for index in range(6)]
    checkpoint["model_call_count"] = 6
    checkpoint["tool_call_count"] = 2
    checkpoint["model_call_ids"] = history_ids
    checkpoint["events"] = [
        {
            "kind": "model_call",
            "status": "succeeded",
            "model_call_id": call_id,
            "usage_status": "known",
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
        for call_id in history_ids
    ]
    model = _ScriptedModel([])
    agent = BoundedAgent(
        model,
        tools=_tools(),
        call_store=ModelCallStore(),
        limits=GraphLimits(max_model_calls=6, max_tool_calls=8, max_wall_clock_seconds=60),
        run_config=run_config,
    )

    result = agent.resume_from_checkpoint(context, "2026年9月", checkpoint)

    assert result.status == "limit_reached"
    assert result.error_code == "model_call_limit"
    assert result.model_call_count == 6
    assert result.tool_call_count == 2
    assert list(result.model_call_ids) == history_ids
    assert len(model.messages) == 0
    assert result.usage_summary["status"] == "known"
    assert result.usage_summary["total_tokens"] == 18


def test_unresolved_metric_clarification_preserves_nonzero_fake_history_without_calling_model() -> None:
    context = _context("run-resume-clarification-history")
    run_config = RunConfig(profile="B1-bounded-agent")
    checkpoint = BoundedAgent.prepared_waiting_user_checkpoint(
        context,
        "销售额是多少？",
        run_config=run_config,
        waiting_question="请说明按支付金额还是退款后净额计算。",
    )
    checkpoint["model_call_count"] = 1
    checkpoint["tool_call_count"] = 2
    checkpoint["model_call_ids"] = ["fake-history-call-1"]
    checkpoint["events"] = [{
        "kind": "model_call",
        "status": "succeeded",
        "model_call_id": "fake-history-call-1",
        "usage_status": "known",
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }]
    model = _ScriptedModel([])
    agent = BoundedAgent(
        model,
        tools=_tools(),
        call_store=ModelCallStore(),
        run_config=run_config,
    )

    result = agent.continue_waiting_for_clarification(context, "2026年9月", checkpoint)

    assert result.status == "waiting_user"
    assert result.model_call_count == 1
    assert result.tool_call_count == 2
    assert list(result.model_call_ids) == ["fake-history-call-1"]
    assert result.usage_summary["status"] == "known"
    assert result.usage_summary["total_tokens"] == 6
    assert not model.messages


def test_usage_summary_keeps_known_consumption_from_failed_call_and_unknown_call_null() -> None:
    summary = _usage_summary((
        {
            "kind": "model_call",
            "status": "failed",
            "model_call_id": "failed-call-with-known-usage",
            "usage_status": "known",
            "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
        },
        {
            "kind": "model_call",
            "status": "failed",
            "model_call_id": "failed-call-with-unknown-usage",
            "usage_status": "unknown",
            "usage": None,
        },
    ))

    assert summary["status"] == "unknown"
    assert summary["model_call_ids"] == ["failed-call-with-known-usage", "failed-call-with-unknown-usage"]
    assert summary["known_call_count"] == 1
    assert summary["unknown_call_count"] == 1
    assert summary["known_total_tokens"] == 12
    assert summary["total_tokens"] is None
