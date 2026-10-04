"""Deterministic W03-T03 probe for the bounded LangGraph runtime."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path

from queryshield.agent import BoundedAgent, GraphLimits, ModelCallStore
from queryshield.agent.proposals import ExecutionContext
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.providers.contracts import ModelCallResult, ModelUsage
from queryshield.tools import ControlledTools


class _FakeCursor:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        return None

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
        self.model_call_ids: list[str] = []
        self.message_roles: list[list[str]] = []

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        request_id: str | None = None,
        model_call_id: str | None = None,
    ) -> ModelCallResult:
        if request_id is None or model_call_id is None:
            raise AssertionError("the server must create request and model call identities")
        index = len(self.model_call_ids)
        self.model_call_ids.append(model_call_id)
        self.message_roles.append([str(message["role"]) for message in messages])
        return ModelCallResult(
            mode="fake",
            provider="scripted-t03",
            model="scripted-t03-model",
            request_id=request_id,
            model_call_id=model_call_id,
            provider_call_id=f"provider-call-{index + 1}",
            provider_request_id=f"provider-request-{index + 1}",
            content=self.contents[index],
            usage=ModelUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            usage_status="known",
        )


class _StepClock:
    def __init__(self, values: Sequence[float]) -> None:
        self.values = list(values)
        self.last = self.values[-1] if self.values else 0.0

    def __call__(self) -> float:
        if self.values:
            self.last = self.values.pop(0)
        return self.last


def _context(run_id: str) -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        tenant_id="tenant-A",
        principal_id="principal-t03",
        role="requester",
    )


def _tools() -> ControlledTools:
    connection = _FakeConnection([{"gross_fen": 3000}])
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


def _final_answer() -> str:
    return json.dumps(
        {
            "type": "final_answer",
            "answer": "受控查询已返回结果。",
            "source_ids": ["commerce-v1"],
            "fact_refs": [],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _normal_case() -> dict[str, object]:
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
            _final_answer(),
        ]
    )
    runtime = BoundedAgent(model, tools=_tools(), call_store=ModelCallStore())
    result = runtime.run(_context("run-t03-normal"), "2026年9月已支付订单总额")
    _assert(result.status == "succeeded", "normal graph did not finish with an answer")
    _assert(result.model_call_count == 3, "normal graph did not use three model rounds")
    _assert(result.tool_call_count == 2, "normal graph did not execute retrieval and query")
    _assert(len(set(result.model_call_ids)) == 3, "model call IDs are not unique")
    _assert(model.model_call_ids == list(result.model_call_ids), "provider saw different call IDs")
    _assert(all("content" not in event for event in result.events), "raw model content leaked into events")
    _assert(
        all(
            "system" not in roles[1:]
            for roles in model.message_roles
        ),
        "retrieval or tool data was promoted to the system role",
    )
    return {
        "status": result.status,
        "model_call_count": result.model_call_count,
        "tool_call_count": result.tool_call_count,
        "model_call_ids": list(result.model_call_ids),
        "events": [dict(event) for event in result.events],
        "action_type": result.action["type"] if result.action else None,
        "message_roles": model.message_roles,
    }


def _model_cap_case() -> dict[str, object]:
    loop_action = _tool_call("search_catalog", {"query": "营业额", "top_k": 1})
    model = _ScriptedModel([loop_action] * 7)
    runtime = BoundedAgent(model, tools=_tools(), call_store=ModelCallStore())
    result = runtime.run(_context("run-t03-model-cap"), "故意循环测试")
    _assert(result.status == "limit_reached", "model loop did not stop")
    _assert(result.error_code == "model_call_limit", "wrong loop stop reason")
    _assert(result.model_call_count == 6, "the sixth model call was not the final allowed call")
    _assert(len(model.model_call_ids) == 6, "a seventh model call was made")
    return {
        "status": result.status,
        "error_code": result.error_code,
        "model_call_count": result.model_call_count,
        "tool_call_count": result.tool_call_count,
        "provider_observed_call_count": len(model.model_call_ids),
        "last_event": dict(result.events[-1]),
    }


def _tool_cap_case() -> dict[str, object]:
    loop_action = _tool_call("search_catalog", {"query": "营业额", "top_k": 1})
    model = _ScriptedModel([loop_action] * 4)
    runtime = BoundedAgent(
        model,
        tools=_tools(),
        call_store=ModelCallStore(),
        limits=GraphLimits(max_model_calls=6, max_tool_calls=2),
    )
    result = runtime.run(_context("run-t03-tool-cap"), "工具预算测试")
    _assert(result.status == "limit_reached", "tool cap did not stop the loop")
    _assert(result.error_code == "tool_call_limit", "wrong tool stop reason")
    _assert(result.tool_call_count == 2, "the tool cap was exceeded")
    return {
        "status": result.status,
        "error_code": result.error_code,
        "model_call_count": result.model_call_count,
        "tool_call_count": result.tool_call_count,
        "provider_observed_call_count": len(model.model_call_ids),
    }


def _wall_clock_case() -> dict[str, object]:
    model = _ScriptedModel([_tool_call("search_catalog", {"query": "营业额", "top_k": 1})])
    # Initial timestamp is zero; the next budget check observes 61 seconds.
    runtime = BoundedAgent(
        model,
        tools=_tools(),
        call_store=ModelCallStore(),
        clock=_StepClock([0.0, 0.0, 61.0]),
    )
    result = runtime.run(_context("run-t03-wall-clock"), "墙钟预算测试")
    _assert(result.status == "limit_reached", "wall-clock cap did not stop the graph")
    _assert(result.error_code == "wall_clock_limit", "wrong wall-clock stop reason")
    _assert(result.model_call_count == 1, "wall-clock case did not make its first model call")
    _assert(result.tool_call_count == 0, "tool ran after the wall-clock budget expired")
    return {
        "status": result.status,
        "error_code": result.error_code,
        "model_call_count": result.model_call_count,
        "tool_call_count": result.tool_call_count,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the W03 bounded LangGraph probe")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    normal = _normal_case()
    model_cap = _model_cap_case()
    tool_cap = _tool_cap_case()
    wall_clock = _wall_clock_case()
    output = {
        "status": "pass",
        "check_id": "W03-T03",
        "engineering_version": "2026-09-19.engineering-v1",
        "profile": "fake-bounded-langgraph-v1",
        "graph_nodes": ["model_decision", "execute_tool", "finish"],
        "limits": {
            "max_model_calls": 6,
            "max_tool_calls": 8,
            "max_wall_clock_seconds": 60,
        },
        "normal": normal,
        "model_cap": model_cap,
        "tool_cap": tool_cap,
        "wall_clock": wall_clock,
        "database_mode": "fake_connection_in_process",
        "provider_mode": "fake_scripted_model",
        "real_mode": "not_run; requires configured real model and remains a separate evidence path",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / "W03-T03.json"
    evidence_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({**output, "evidence_path": evidence_path.as_posix()}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

