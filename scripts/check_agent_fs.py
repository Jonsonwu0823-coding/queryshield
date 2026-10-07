"""AGENT-FS01/FS02 checks for server-verified facts and usage retention.

The fake mode deliberately injects a deterministic database connection and a
scripted model.  It exercises the formal Agent path, including the
server-owned ResultEvidence lookup, rather than calling FactResolver alone.
The real PostgreSQL boundary is checked separately by AGENT-DB01.
"""

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
from queryshield.agent.context import NET_FEN_PLAN_ID  # noqa: E402
from queryshield.agent.context import NET_FEN_TIME_WINDOW  # noqa: E402
from queryshield.agent.proposals import ExecutionContext, MetricBinding  # noqa: E402
from queryshield.catalog import DEFAULT_CATALOG_VERSION  # noqa: E402
from queryshield.db.guarded import GuardedQueryExecutor  # noqa: E402
from queryshield.providers.contracts import ModelCallResult, ModelUsage  # noqa: E402
from queryshield.tools import ControlledTools  # noqa: E402


EXPECTED_USAGE = {
    "status": "known",
    "model_call_count": 3,
    "known_call_count": 3,
    "unknown_call_count": 0,
    "prompt_tokens": 27,
    "completion_tokens": 9,
    "total_tokens": 36,
}


class _FakeCursor:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.default_rows = rows
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        self.executed.append((sql, params))
        normalized = sql.lower()
        target = self.default_rows[0].get("net_fen") if self.default_rows else 0
        if 'as "gross_fen"' in normalized:
            self.rows = [{"gross_fen": target}]
        elif 'as "refund_fen"' in normalized:
            self.rows = [{"refund_fen": 0}]
        else:
            self.rows = self.default_rows

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


_NET_QUERY_SQL = "SELECT COALESCE(SUM(amount_fen), 0) AS net_fen FROM orders WHERE status = %s"
# A query the server accepts for a bound gross_fen metric (tenant, status and window
# filters, an aliased SUM(amount_fen) projection).  The server rejects a
# query whose projection does not match the bound metric before any fact exists, so
# the gross-as-net forgery has to be exercised with a query the server accepts.
_GROSS_QUERY = (
    "SELECT COALESCE(SUM(o.amount_fen), 0) AS gross_fen FROM orders AS o "
    "WHERE o.tenant_id = %s AND o.status = %s AND o.created_at >= %s AND o.created_at < %s",
    {"0": "tenant-A", "1": "paid", "2": "2026-09-01T00:00:00Z", "3": "2026-10-01T00:00:00Z"},
)


class _FactsModel:
    mode = "fake"

    def __init__(
        self,
        *,
        final_result_id: str = "__actual__",
        final_metric_id: str = "net_fen",
        query: tuple[str, Mapping[str, object]] = (_NET_QUERY_SQL, {"0": "paid"}),
    ) -> None:
        self.final_result_id = final_result_id
        self.final_metric_id = final_metric_id
        self.query_sql, self.query_params = query
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
                {
                    "type": "tool_call",
                    "name": "search_catalog",
                    "arguments": {"query": "净额", "top_k": 3},
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        elif index == 1:
            content = json.dumps(
                {
                    "type": "tool_call",
                    "name": "query_readonly",
                    "arguments": {
                        "sql": self.query_sql,
                        "params": dict(self.query_params),
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        else:
            result_ids = re.findall(r'"result_id":"([^"]+)"', "\n".join(message["content"] for message in copied))
            actual_result_id = result_ids[-1] if result_ids else ""
            result_id = actual_result_id if self.final_result_id == "__actual__" else self.final_result_id
            content = json.dumps(
                {
                    "type": "final_answer",
                    "answer": "模型草稿：净额是1200元。",
                    "source_ids": ["spoofed-source"],
                    "fact_refs": [{"result_id": result_id, "metric_id": self.final_metric_id}],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return ModelCallResult(
            mode="fake",
            provider="scripted-w03-fs",
            model="scripted-w03-fs-model",
            request_id=request_id,
            model_call_id=model_call_id,
            provider_call_id=f"provider-w03-fs-{index + 1}",
            provider_request_id=f"provider-request-w03-fs-{index + 1}",
            content=content,
            usage=ModelUsage(prompt_tokens=9, completion_tokens=3, total_tokens=12),
            usage_status="known",
        )


def _context(run_id: str) -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        tenant_id="tenant-A",
        principal_id="principal-A",
        role="requester",
    )


def _binding(
    *,
    metric_id: str = "net_fen",
    result_position: str = "net_fen",
    unit: str = "CNY_fen",
    start: str = "2026-09-01",
    end: str = "2026-10-01",
    use_controlled_plan: bool = True,
) -> MetricBinding:
    plan_id = NET_FEN_PLAN_ID if metric_id == "net_fen" and use_controlled_plan else None
    return MetricBinding(
        metric_id=metric_id,
        result_position=result_position,
        unit=unit,
        time_window={"start": f"{start}T00:00:00Z", "end": f"{end}T00:00:00Z", "timezone": NET_FEN_TIME_WINDOW["timezone"]},
        catalog_source_id="commerce-v1",
        catalog_version=DEFAULT_CATALOG_VERSION,
        plan_id=plan_id,
    )


def _tools(rows: list[dict[str, object]]) -> tuple[ControlledTools, _FakeConnection]:
    connection = _FakeConnection(rows)
    executor = GuardedQueryExecutor(
        connect=lambda: connection,
        clock=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc),
    )
    return ControlledTools(executor=executor), connection


def _result_id(result: object) -> str:
    action = getattr(result, "action")
    if not isinstance(action, Mapping):
        raise AssertionError("final action is missing")
    refs = action.get("fact_refs")
    if not isinstance(refs, list) or len(refs) != 1 or not isinstance(refs[0], Mapping):
        raise AssertionError("final action has no single fact reference")
    value = refs[0].get("result_id")
    if not isinstance(value, str) or not value:
        raise AssertionError("fact reference result_id is missing")
    return value


def _run(
    *,
    run_id: str,
    rows: list[dict[str, object]],
    binding: MetricBinding,
    final_result_id: str = "__actual__",
    final_metric_id: str = "net_fen",
    query: tuple[str, Mapping[str, object]] | None = None,
) -> tuple[object, _FakeConnection]:
    tools, connection = _tools(rows)
    model = _FactsModel(
        final_result_id=final_result_id,
        final_metric_id=final_metric_id,
        **({} if query is None else {"query": query}),
    )
    result = BoundedAgent(model, tools=tools, call_store=ModelCallStore()).run(
        _context(run_id),
        "2026年9月退款后净额",
        metric_bindings=(binding,),
    )
    return result, connection


def _assert_usage(result: object, *, model_calls: int = 3) -> None:
    # Every scripted call reports the same known usage, so the totals scale with the calls.
    per_call = {name: EXPECTED_USAGE[name] // EXPECTED_USAGE["model_call_count"] for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
    assert_known_usage(
        result.usage_summary,
        calls=model_calls,
        prompt_tokens=per_call["prompt_tokens"] * model_calls,
        completion_tokens=per_call["completion_tokens"] * model_calls,
        total_tokens=per_call["total_tokens"] * model_calls,
    )
    if result.model_call_count != model_calls:
        raise AssertionError(f"unexpected model call count: {result.model_call_count}")


def _run_fs01() -> dict[str, object]:
    cases = (
        ("positive_net", 12000, "2026-09-01", "2026-10-01", "120.00元"),
        ("empty_window_zero", 0, "2026-11-01", "2026-12-01", "0.00元"),
        ("changed_result", 13000, "2026-10-01", "2026-11-01", "130.00元"),
    )
    records: list[dict[str, object]] = []
    for label, value, start, end, display in cases:
        result, connection = _run(
            run_id=f"run-w03-fs01-{label}",
            rows=[{"net_fen": value}],
            binding=_binding(start=start, end=end),
        )
        expected_answer = f"已核实：退款后净额：{display}（{start}T00:00:00Z至{end}T00:00:00Z，UTC）"
        trace_text = json.dumps(result.events, ensure_ascii=False)
        passed = (
            result.status == "succeeded"
            # A catalog basis note follows the verified line.
            and (result.answer or "").partition("\n")[0] == expected_answer
            and (result.answer or "").partition("\n")[2].startswith("口径：退款后净额（net_fen）")
            and result.facts is not None
            and result.facts["facts"][0]["value"] == value
            and result.facts["facts"][0]["display_value"] == display
            and "1200元" not in (result.answer or "")
            and "模型草稿：净额是1200元。" not in trace_text
            and result.tool_call_count == 2
            and len(connection.cursor_instance.executed) == 2
        )
        _assert_usage(result)
        if not passed:
            raise AssertionError(f"FS01 case failed: {label}; {result.as_dict()}")
        records.append(
            {
                "label": label,
                "status": result.status,
                "answer": result.answer,
                "value": value,
                "display_value": display,
                "run_id": result.run_id,
                "result_id": _result_id(result),
                "model_call_count": result.model_call_count,
                "tool_call_count": result.tool_call_count,
                "usage_summary": dict(result.usage_summary),
                "database_exec_count": len(connection.cursor_instance.executed),
            }
        )

    if len({record["run_id"] for record in records}) != 3 or len({record["result_id"] for record in records}) != 3:
        raise AssertionError("FS01 re-queries must create distinct run/result records")
    return {"status": "pass", "cases": records, "provider_mode": "scripted_fake", "database_mode": "injected_connection"}


def _invalid_case(
    *,
    label: str,
    rows: list[dict[str, object]],
    binding: MetricBinding,
    final_result_id: str = "__actual__",
    final_metric_id: str = "net_fen",
    query: tuple[str, Mapping[str, object]] | None = None,
    expected_error_code: str = "evidence_validation_failed",
    expected_repairs: int = 0,
) -> dict[str, object]:
    model_calls = 3 + expected_repairs
    result, connection = _run(
        run_id=f"run-w03-fs02-{label}",
        rows=rows,
        binding=binding,
        final_result_id=final_result_id,
        final_metric_id=final_metric_id,
        query=query,
    )
    last_event = result.events[-1] if result.events else {}
    passed = (
        result.status == "failed"
        and result.error_code == expected_error_code
        and result.answer is None
        and result.facts is None
        and result.repair_count == expected_repairs
        and result.model_call_count == model_calls
        and result.tool_call_count == 2
        and last_event.get("kind") == "answer_validation"
        and last_event.get("field") == "fact_refs"
        and len(connection.cursor_instance.executed)
        == (2 if binding.plan_id == NET_FEN_PLAN_ID and binding.unit == "CNY_fen" and binding.metric_id == "net_fen" else 1)
    )
    _assert_usage(result, model_calls=model_calls)
    if not passed:
        raise AssertionError(f"FS02 case failed: {label}; {result.as_dict()}")
    return {
        "label": label,
        "status": result.status,
        "error_code": result.error_code,
        "public_answer": result.answer,
        "facts": result.facts,
        "model_call_count": result.model_call_count,
        "tool_call_count": result.tool_call_count,
        "usage_summary": dict(result.usage_summary),
        "database_exec_count": len(connection.cursor_instance.executed),
        "last_trace": {"kind": last_event.get("kind"), "field": last_event.get("field")},
    }


def _run_fs02() -> dict[str, object]:
    foreign_result, _ = _run(
        run_id="run-w03-fs02-foreign-source",
        rows=[{"net_fen": 12000}],
        binding=_binding(),
    )
    foreign_result_id = _result_id(foreign_result)
    cases = [
        _invalid_case(
            label="unknown_result",
            rows=[{"net_fen": 12000}],
            binding=_binding(),
            final_result_id="result-not-from-this-run",
        ),
        _invalid_case(
            label="foreign_result",
            rows=[{"net_fen": 12000}],
            binding=_binding(),
            final_result_id=foreign_result_id,
        ),
        _invalid_case(
            label="gross_as_net",
            rows=[{"gross_fen": 15000}],
            binding=_binding(metric_id="gross_fen", result_position="gross_fen"),
            query=_GROSS_QUERY,
            # A fact reference to a metric the run never declared or bound is a
            # repairable metric_not_declared error (one repair, then the run fails).  The
            # forged fact still never reaches a public answer or facts envelope.
            expected_error_code="metric_not_declared",
            expected_repairs=1,
        ),
        _invalid_case(
            label="wrong_type",
            rows=[{"net_fen": "12000"}],
            binding=_binding(use_controlled_plan=False),
        ),
        _invalid_case(
            label="wrong_unit",
            rows=[{"net_fen": 12000}],
            binding=_binding(unit="USD_cent"),
        ),
    ]
    return {
        "status": "pass",
        "cases": cases,
        "provider_mode": "scripted_fake",
        "database_mode": "injected_connection",
        "usage_boundary": "invalid fact references retain every known model-call usage record; only a reference to an undeclared metric gets the single B2a repair",
    }


def _source_manifest() -> dict[str, str]:
    paths = (
        Path("src/queryshield/agent/__init__.py"),
        Path("src/queryshield/agent/graph.py"),
        Path("src/queryshield/facts/facts.py"),
        Path("src/queryshield/tools/semantic.py"),
        Path(__file__).relative_to(PROJECT_ROOT),
    )
    return {
        str(path).replace("\\", "/"): hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-id", required=True, choices=("AGENT-FS01", "AGENT-FS02"))
    parser.add_argument("--mode", choices=("fake", "real"), default="fake")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / f"{args.check_id}.json"
    if args.mode == "real":
        evidence = {
            "check_id": args.check_id,
            "mode": "real",
            "status": "not_applicable",
            "reason": "AGENT-FS01/FS02 deterministic fact-boundary cases are required in fake mode; real PostgreSQL/model multi-step evidence is AGENT-DB01 and AGENT-T06",
            "real_not_substituted": True,
            "raw_evidence_paths": [str(evidence_path).replace("\\", "/")],
        }
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
        return 0

    actual = _run_fs01() if args.check_id == "AGENT-FS01" else _run_fs02()
    evidence = {
        "runtime_extension_version": "2026-09-12.runtime-v1",
        "extension_version": "2026-09-09.facts-state-v1",
        "check_id": args.check_id,
        "cwd": str(PROJECT_ROOT),
        "command": f"{sys.executable} {Path(__file__).as_posix()} --check-id {args.check_id} --mode fake --output-dir {output_dir.as_posix()}",
        "source_manifest": _source_manifest(),
        "mode": "fake",
        "profile": "fake-server-facts-boundary-v1",
        "expected": {
            "AGENT-FS01": "server renders 12000 fen as 120.00 yuan, preserves empty/change re-query and known usage",
            "AGENT-FS02": "unknown/foreign/gross-as-net/wrong-type/wrong-unit evidence is rejected without answer/facts or repair",
        }[args.check_id],
        "actual": actual,
        "implementation_result": "candidate_self_checked",
        "learner_result": "pending",
        "hint_level": "H3",
        "raw_evidence_paths": [str(evidence_path).replace("\\", "/")],
        "status": "pass",
    }
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
