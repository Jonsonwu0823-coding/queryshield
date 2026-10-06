"""Deterministic AGENT-T04 probe for clarification and bounded query repair."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from queryshield.agent import BoundedAgent, ModelCallStore  # noqa: E402
from queryshield.agent.proposals import ExecutionContext  # noqa: E402
from queryshield.db.guarded import GuardedQueryExecutor  # noqa: E402
from queryshield.providers.contracts import ModelCallResult, ModelUsage  # noqa: E402
from queryshield.tools import ControlledTools  # noqa: E402


class _FakeCursor:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        self.executed.append((sql, params))

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


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


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
        if request_id is None or model_call_id is None:
            raise AssertionError("the server must allocate model call identity first")
        index = len(self.messages)
        if index >= len(self.contents):
            raise AssertionError("the probe made an unexpected model call")
        self.messages.append(tuple(dict(message) for message in messages))
        self.model_call_ids.append(model_call_id)
        return ModelCallResult(
            mode="fake",
            provider="scripted-t04",
            model="scripted-t04-model",
            request_id=request_id,
            model_call_id=model_call_id,
            provider_call_id=f"provider-t04-{index + 1}",
            provider_request_id=f"provider-request-t04-{index + 1}",
            content=self.contents[index],
            usage=ModelUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
            usage_status="known",
        )


def _context(run_id: str) -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        tenant_id="tenant-A",
        principal_id="principal-A",
        role="requester",
    )


def _tools(rows: list[dict[str, object]]) -> tuple[ControlledTools, _FakeConnection]:
    connection = _FakeConnection(rows)
    executor = GuardedQueryExecutor(
        connect=lambda: connection,
        clock=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc),
    )
    return ControlledTools(executor=executor), connection


def _tool_call(name: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "name": name, "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _final_answer(answer: str) -> str:
    return json.dumps(
        {
            "type": "final_answer",
            "answer": answer,
            "source_ids": ["commerce-v1"],
            "fact_refs": [],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _run_probe() -> dict[str, object]:
    clarification_model = _ScriptedModel(
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
            _final_answer("净额为120.00元。"),
        ]
    )
    clarification_tools, _ = _tools([{"net_fen": 12000}])
    clarification_clock = _Clock()
    clarification_context = _context("run-t04-clarification")
    clarification_agent = BoundedAgent(
        clarification_model,
        tools=clarification_tools,
        call_store=ModelCallStore(),
        clock=clarification_clock,
    )
    waiting = clarification_agent.run(clarification_context, "销售额")
    if waiting.status != "waiting_user":
        raise AssertionError(f"expected WAITING_USER, got {waiting.as_dict()}")
    clarification_clock.value = 120.0
    resumed = clarification_agent.resume(clarification_context, "净额")
    if resumed.status != "succeeded":
        raise AssertionError(f"expected resumed success, got {resumed.as_dict()}")
    if resumed.elapsed_ms >= 1_000:
        raise AssertionError("waiting time was incorrectly charged to the active wall clock")
    if not any(
        message["role"] == "user" and message["content"] == "净额"
        for messages in clarification_model.messages[1:]
        for message in messages
    ):
        raise AssertionError("the clarification was not passed as a user message")

    repair_model = _ScriptedModel(
        [
            _tool_call(
                "query_readonly",
                {"sql": "SELECT missing.amount_fen FROM orders AS o", "params": {}},
            ),
            _tool_call(
                "query_readonly",
                {"sql": "SELECT o.amount_fen FROM orders AS o", "params": {}},
            ),
            _final_answer("修复后的查询完成。"),
        ]
    )
    repair_tools, repair_connection = _tools([{"amount_fen": 12000}])
    repair_result = BoundedAgent(
        repair_model,
        tools=repair_tools,
        call_store=ModelCallStore(),
    ).run(_context("run-t04-repair"), "订单金额")
    if repair_result.status != "succeeded" or repair_result.repair_count != 1:
        raise AssertionError(f"expected one successful repair, got {repair_result.as_dict()}")
    if len(repair_connection.cursor_instance.executed) != 1:
        raise AssertionError("the rejected query should not reach the database")

    repeated_bad = _tool_call(
        "query_readonly",
        {"sql": "SELECT missing.amount_fen FROM orders AS o", "params": {}},
    )
    repeated_model = _ScriptedModel([repeated_bad, repeated_bad, _final_answer("not-called")])
    repeated_tools, _ = _tools([{"amount_fen": 12000}])
    repeated_result = BoundedAgent(
        repeated_model,
        tools=repeated_tools,
        call_store=ModelCallStore(),
    ).run(_context("run-t04-repair-limit"), "订单金额")
    if repeated_result.status != "failed" or repeated_result.error_code != "query_repair_limit":
        raise AssertionError(f"expected repair limit, got {repeated_result.as_dict()}")
    if repeated_result.model_call_count != 2 or repeated_result.tool_call_count != 2:
        raise AssertionError("a repeated repair must stop without a third model call")

    security_model = _ScriptedModel(
        [
            _tool_call("query_readonly", {"sql": "SELECT * FROM secrets", "params": {}}),
            _final_answer("not-called"),
        ]
    )
    security_tools, security_connection = _tools([])
    security_result = BoundedAgent(
        security_model,
        tools=security_tools,
        call_store=ModelCallStore(),
    ).run(_context("run-t04-security"), "读取秘密表")
    if security_result.status != "denied" or security_result.error_code != "table_not_allowed":
        raise AssertionError(f"expected security denial, got {security_result.as_dict()}")
    if security_result.model_call_count != 1 or security_result.tool_call_count != 1:
        raise AssertionError("a security refusal must not enter query repair")
    if security_connection.cursor_instance.executed:
        raise AssertionError("a security refusal must not reach the database")

    return {
        "clarification": {
            "initial_status": waiting.status,
            "resumed_status": resumed.status,
            "run_id_preserved": resumed.run_id == clarification_context.run_id,
            "model_call_count": resumed.model_call_count,
            "tool_call_count": resumed.tool_call_count,
            "active_elapsed_ms_after_120s_wait": resumed.elapsed_ms,
            "clarification_visible_as_user_message": True,
        },
        "repair_once": {
            "status": repair_result.status,
            "repair_count": repair_result.repair_count,
            "model_call_count": repair_result.model_call_count,
            "tool_call_count": repair_result.tool_call_count,
            "rejected_query_database_exec_count": len(repair_connection.cursor_instance.executed),
            "repair_event_count": sum(event["kind"] == "query_repair" for event in repair_result.events),
        },
        "repair_limit": {
            "status": repeated_result.status,
            "error_code": repeated_result.error_code,
            "model_call_count": repeated_result.model_call_count,
            "tool_call_count": repeated_result.tool_call_count,
        },
        "security_refusal": {
            "status": security_result.status,
            "error_code": security_result.error_code,
            "model_call_count": security_result.model_call_count,
            "tool_call_count": security_result.tool_call_count,
            "database_exec_count": len(security_connection.cursor_instance.executed),
        },
    }


def _source_manifest() -> dict[str, str]:
    paths = [
        Path("src/queryshield/agent/context.py"),
        Path("src/queryshield/agent/graph.py"),
        Path("src/queryshield/agent/proposals.py"),
        Path("src/queryshield/tools/semantic.py"),
    ]
    return {
        str(path).replace("\\", "/"): hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "AGENT-T04.json"
    command = f"{sys.executable} {Path(__file__).as_posix()} --output-dir {output_dir.as_posix()}"
    actual = _run_probe()
    evidence = {
        "runtime_extension_version": "2026-09-12.runtime-v1",
        "check_id": "AGENT-T04",
        "upstream_manifest": _source_manifest(),
        "cwd": str(PROJECT_ROOT),
        "command": command,
        "exit_code": 0,
        "fixture": {
            "clarification_question": "销售额",
            "clarification_answer": "净额",
            "repairable_error": "unknown_qualifier",
            "security_error": "table_not_allowed",
            "max_query_repairs": 1,
        },
        "expected": {
            "ambiguous_metric": "WAITING_USER then resume same run",
            "one_repair": "one structural query error may continue",
            "repeated_repair": "second structural error stops",
            "security_refusal": "no model retry and no database execution",
            "waiting_time": "not charged to active wall clock",
        },
        "actual": actual,
        "mode": "fake",
        "profile": "fake-clarification-query-repair-v1",
        "provider_mode": "not_run; injected scripted model",
        "database_mode": "not_run; injected deterministic connection",
        "raw_evidence_paths": [str(evidence_path).replace("\\", "/")],
        "implementation_result": "candidate_self_checked",
        "learner_result": "pending",
        "hint_level": "H3",
        "next_action": "AGENT-T05",
        "status": "pass",
    }
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
