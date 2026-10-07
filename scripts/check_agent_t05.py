"""Deterministic AGENT-T05 probe for trace, usage aggregation and verified facts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from check_agent_usage import assert_known_usage  # noqa: E402
from queryshield.agent import BoundedAgent, ModelCallStore  # noqa: E402
from queryshield.agent.context import NET_FEN_PLAN_ID, NET_FEN_TIME_WINDOW  # noqa: E402
from queryshield.agent.proposals import ExecutionContext, MetricBinding  # noqa: E402
from queryshield.catalog import DEFAULT_CATALOG_VERSION  # noqa: E402
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


class _FactsModel:
    mode = "fake"

    def __init__(self, *, bad_fact_ref: bool = False) -> None:
        self.bad_fact_ref = bad_fact_ref
        self.messages: list[tuple[Mapping[str, str], ...]] = []

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        request_id: str | None = None,
        model_call_id: str | None = None,
        run_id: str | None = None,
    ) -> ModelCallResult:
        if request_id is None or model_call_id is None:
            raise AssertionError("the server must allocate model call identity first")
        index = len(self.messages)
        copied = tuple(dict(message) for message in messages)
        self.messages.append(copied)
        if index == 0:
            content = json.dumps(
                {"type": "tool_call", "name": "search_catalog", "arguments": {"query": "净额", "top_k": 3}},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        elif index == 1:
            content = json.dumps(
                {
                    "type": "tool_call",
                    "name": "query_readonly",
                    "arguments": {
                        "sql": "SELECT COALESCE(SUM(amount_fen), 0) AS net_fen FROM orders WHERE status = %s",
                        "params": {"0": "paid"},
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        else:
            match = re.search(r'"result_id":"([^"]+)"', "\n".join(message["content"] for message in copied))
            result_id = "result-not-from-this-run" if self.bad_fact_ref else (match.group(1) if match else "")
            content = json.dumps(
                {
                    "type": "final_answer",
                    "answer": "模型草稿：净额是120.00元。",
                    "source_ids": ["spoofed-source"],
                    "fact_refs": [{"result_id": result_id, "metric_id": "net_fen"}],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return ModelCallResult(
            mode="fake",
            provider="scripted-t05",
            model="scripted-t05-model",
            request_id=request_id,
            model_call_id=model_call_id,
            provider_call_id=f"provider-t05-{index + 1}",
            provider_request_id=f"provider-request-t05-{index + 1}",
            content=content,
            usage=ModelUsage(prompt_tokens=11, completion_tokens=4, total_tokens=15),
            usage_status="known",
        )


def _context(run_id: str) -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        tenant_id="tenant-A",
        principal_id="principal-A",
        role="requester",
    )


def _tools() -> tuple[ControlledTools, _FakeConnection]:
    connection = _FakeConnection([{"net_fen": 12000}])
    executor = GuardedQueryExecutor(
        connect=lambda: connection,
        clock=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc),
    )
    return ControlledTools(executor=executor), connection


def _binding() -> MetricBinding:
    return MetricBinding(
        metric_id="net_fen",
        result_position="net_fen",
        unit="CNY_fen",
        time_window={
            **NET_FEN_TIME_WINDOW,
        },
        catalog_source_id="commerce-v1",
        catalog_version=DEFAULT_CATALOG_VERSION,
        plan_id=NET_FEN_PLAN_ID,
    )


def _run_probe() -> dict[str, object]:
    tools, connection = _tools()
    model = _FactsModel()
    result = BoundedAgent(model, tools=tools, call_store=ModelCallStore()).run(
        _context("run-t05-facts"),
        "2026年9月退款后净额",
        metric_bindings=(_binding(),),
    )
    if result.status != "succeeded":
        raise AssertionError(f"expected verified answer, got {result.as_dict()}")
    # A catalog basis note follows the verified line.
    verified_line, _, basis = (result.answer or "").partition("\n")
    if verified_line != "已核实：退款后净额：120.00元（2026-09-01T00:00:00Z至2026-10-01T00:00:00Z，UTC）" or not basis.startswith("口径：退款后净额（net_fen）"):
        raise AssertionError(f"unexpected server-rendered answer: {result.answer}")
    if result.action is None or result.action["source_ids"] != ["commerce-v1"]:
        raise AssertionError("public source IDs must come from verified facts")
    if result.facts is None:
        raise AssertionError("verified facts envelope is missing")
    facts = result.facts["facts"]
    if not isinstance(facts, list) or len(facts) != 1:
        raise AssertionError("expected one verified fact")
    fact = facts[0]
    if fact["value"] != 12000 or fact["unit"] != "CNY_fen" or fact["display_value"] != "120.00元":
        raise AssertionError(f"unexpected fact fields: {fact}")

    model_events = [event for event in result.events if event["kind"] == "model_call"]
    tool_events = [event for event in result.events if event["kind"] == "tool_call"]
    answer_events = [event for event in result.events if event["kind"] == "answer"]
    if len(model_events) != 3 or len(tool_events) != 2 or len(answer_events) != 1:
        raise AssertionError("trace does not contain the expected call steps")
    if answer_events[0].get("source_ids") != ["commerce-v1"]:
        raise AssertionError("answer trace must use server-verified source IDs")
    trace_text = json.dumps(result.events, ensure_ascii=False)
    if "模型草稿：净额是120.00元。" in trace_text or "SELECT COALESCE(SUM(amount_fen), 0) AS net_fen FROM orders WHERE status = %s" in trace_text:
        raise AssertionError("trace leaked model content or raw SQL")
    assert_known_usage(result.usage_summary, calls=3, prompt_tokens=33, completion_tokens=12, total_tokens=45)
    if not all(event.get("policy_conclusion") == "allowed" for event in tool_events):
        raise AssertionError("tool policy conclusions were not recorded")
    if len(connection.cursor_instance.executed) != 2:
        raise AssertionError("the controlled net plan should execute two guarded aggregate queries")

    bad_tools, _ = _tools()
    bad_result = BoundedAgent(
        _FactsModel(bad_fact_ref=True),
        tools=bad_tools,
        call_store=ModelCallStore(),
    ).run(
        _context("run-t05-bad-fact"),
        "2026年9月退款后净额",
        metric_bindings=(_binding(),),
    )
    if bad_result.status != "failed" or bad_result.error_code != "evidence_validation_failed":
        raise AssertionError(f"expected field-level fact failure, got {bad_result.as_dict()}")
    if bad_result.events[-1].get("kind") != "answer_validation" or bad_result.events[-1].get("field") != "fact_refs":
        raise AssertionError("fact validation failure was not traceable to fact_refs")
    if bad_result.answer is not None or bad_result.facts is not None:
        raise AssertionError("invalid evidence must not produce a public answer")

    return {
        "verified_answer": {
            "status": result.status,
            "answer": result.answer,
            "facts": facts,
            "model_call_count": result.model_call_count,
            "tool_call_count": result.tool_call_count,
            "trace_kinds": [event["kind"] for event in result.events],
            "usage_summary": dict(result.usage_summary),
            "policy_conclusions": [event.get("policy_conclusion") for event in tool_events],
            "database_exec_count": len(connection.cursor_instance.executed),
        },
        "invalid_fact_reference": {
            "status": bad_result.status,
            "error_code": bad_result.error_code,
            "last_trace_step": {
                "kind": bad_result.events[-1]["kind"],
                "field": bad_result.events[-1]["field"],
            },
            "public_answer": bad_result.answer,
        },
    }


def _source_manifest() -> dict[str, str]:
    paths = [
        Path("src/queryshield/agent/graph.py"),
        Path("src/queryshield/agent/proposals.py"),
        Path("src/queryshield/facts/facts.py"),
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
    evidence_path = output_dir / "AGENT-T05.json"
    evidence = {
        "runtime_extension_version": "2026-09-12.runtime-v1",
        "check_id": "AGENT-T05",
        "upstream_manifest": _source_manifest(),
        "cwd": str(PROJECT_ROOT),
        "command": f"{sys.executable} {Path(__file__).as_posix()} --output-dir {output_dir.as_posix()}",
        "exit_code": 0,
        "fixture": {
            "tenant_id": "tenant-A",
            "metric_id": "net_fen",
            "rows": [{"net_fen": 12000}],
            "usage_each_call": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        },
        "expected": {
            "verified_facts": "server renders value/unit/time_window from rows and binding",
            "trace": "each model/tool call is individually observable without raw content or SQL",
            "usage": "three known calls sum once to 33/12/45",
            "invalid_fact": "field-level evidence_validation_failed without public answer",
        },
        "actual": _run_probe(),
        "mode": "fake",
        "profile": "fake-trace-facts-usage-v1",
        "provider_mode": "not_run; injected scripted model",
        "database_mode": "not_run; injected deterministic connection",
        "raw_evidence_paths": [str(evidence_path).replace("\\", "/")],
        "implementation_result": "candidate_self_checked",
        "learner_result": "pending",
        "hint_level": "H3",
        "next_action": "AGENT-T06",
        "status": "pass",
    }
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
