from __future__ import annotations

from pathlib import Path
import runpy
from types import SimpleNamespace
import pytest
from psycopg.errors import ConnectionFailure, InsufficientPrivilege, QueryCanceled, UndefinedColumn

from repo_layout import needs_register
from queryshield.approval.service import FixtureQueryExecutor
from queryshield.evaluation.state_cases import load_w05_development_cases
from queryshield.tools.semantic import ToolError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECK_W05 = PROJECT_ROOT / "scripts" / "check_w05.py"


def _probe_globals(function_name: str) -> dict[str, object]:
    namespace = runpy.run_path(str(CHECK_W05))
    function = namespace[function_name]
    return function.__globals__


def test_stateful_ready_runtime_records_snapshot_catalog_version(tmp_path, monkeypatch) -> None:
    globals_ = _probe_globals("_run_stateful_w05_development")
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", "postgresql://test.invalid/queryshield")
    monkeypatch.setitem(globals_, "_check_db01", lambda _path: {"status": "pass", "check_id": "W05-DB01"})

    snapshot = SimpleNamespace(
        snapshot_id="snapshot-test",
        catalog_version="catalog-test",
        manifest_sha256="manifest-test",
    )
    index_build = SimpleNamespace(index=SimpleNamespace(index_hash="index-test", model="embedding-test"))
    monkeypatch.setitem(
        globals_,
        "_build_retrieval_runtime",
        lambda _mode: (None, snapshot, index_build, object(), object()),
    )
    captured: dict[str, object] = {}

    def fake_suite(_cases, _run_profile, *, metadata):
        captured.update(metadata)
        return {
            "summary": {
                "product_execution_complete": True,
                "b1_critical_question_count": 8,
                "b1_critical_pass_count": 8,
                "b1_critical_fail_count": 0,
                "b1_critical_blocked_count": 0,
                "b1_critical_not_run_count": 0,
            },
            "report": {
                "evaluation_status": "pass",
                "profile_reports": {
                    profile: {
                        "metrics": {
                            "security_correct": {"numerator": 8, "denominator": 8},
                            "security_violations": {"numerator": 0, "denominator": 8},
                        }
                    }
                    for profile in ("B0", "B1")
                },
            },
            "raw_records": [],
        }

    monkeypatch.setitem(globals_, "run_w05_stateful_suite", fake_suite)

    result = globals_["_run_stateful_w05_development"]("fake", tmp_path)

    assert result["status"] == "pass"
    shared_config = captured["shared_runtime_configuration"]
    assert shared_config["catalog_version"] == "catalog-test"
    assert shared_config["knowledge_snapshot_id"] == "snapshot-test"


def test_parallel_postgres_smoke_creates_parent_run_before_parallel_group(tmp_path, monkeypatch) -> None:
    globals_ = _probe_globals("_run_parallel_postgres_smoke")
    monkeypatch.setitem(globals_, "GuardedQueryExecutor", FixtureQueryExecutor)
    monkeypatch.setitem(globals_, "_write_json", lambda *_args, **_kwargs: None)

    result = globals_["_run_parallel_postgres_smoke"](tmp_path)

    assert result["status"] == "pass"
    assert result["metric_rows"] == {
        "gross_fen": [{"gross_fen": 15000}],
        "net_fen": [{"net_fen": 12000}],
        "paid_count": [{"paid_count": 2}],
    }


@needs_register
def test_x01_ignores_parent_git_repository_routing(tmp_path, monkeypatch) -> None:
    globals_ = _probe_globals("_check_upstream_versions")
    monkeypatch.setenv("GIT_DIR", str(tmp_path))

    result = globals_["_check_upstream_versions"]()

    assert result["status"] == "pass"
    assert "GIT_DIR" in result["repository_overrides_cleared"]
    assert result["w04_accepted_commit"] == "f2dfacf47c3a5dba32454e22e56ae4741c80541f"


def test_stateful_quality_gate_requires_all_eight_b1_critical_questions() -> None:
    globals_ = _probe_globals("_stateful_replay_failure_reasons")

    failures = globals_["_stateful_replay_failure_reasons"](
        {
            "b1_critical_question_count": 8,
            "b1_critical_pass_count": 7,
            "b1_critical_fail_count": 1,
            "b1_critical_blocked_count": 0,
            "b1_critical_not_run_count": 0,
        },
        security_violations=0,
    )

    assert failures == ["b1_critical_questions_not_all_pass"]


def test_stateful_quality_gate_requires_all_security_cases_for_both_profiles() -> None:
    globals_ = _probe_globals("_stateful_replay_failure_reasons")
    summary = {
        "b1_critical_question_count": 8,
        "b1_critical_pass_count": 8,
        "b1_critical_fail_count": 0,
        "b1_critical_blocked_count": 0,
        "b1_critical_not_run_count": 0,
    }
    report = {
        "profile_reports": {
            "B0": {"metrics": {"security_correct": {"numerator": 8, "denominator": 8}}},
            "B1": {"metrics": {"security_correct": {"numerator": 7, "denominator": 8}}},
        }
    }

    failures = globals_["_stateful_replay_failure_reasons"](
        summary, security_violations=0, report=report
    )

    assert failures == ["b1_security_cases_not_all_correct"]


def test_profile_call_budget_is_shared_by_functional_and_security_cases() -> None:
    globals_ = _probe_globals("_expected_case_observation")
    cases = {case.case_id: case for case in load_w05_development_cases()}
    functional = cases["gross-total-fen"]
    functional_observation = globals_["_expected_case_observation"](functional)
    functional_observation["evaluation_profile"] = "B1"
    functional_observation["side_effects"]["model_calls"] = 3
    functional_observation["execution_metrics"] = {
        "model_calls": 3,
        "tool_calls": 1,
        "active_seconds": 0.1,
    }

    functional_result = globals_["judge_state_case"](functional, functional_observation)

    assert functional_result["judged_status"] == "pass"

    security = cases["tool-text-injection-untrusted-instruction"]
    security_observation = globals_["_expected_case_observation"](security)
    security_observation["evaluation_profile"] = "B1"
    security_observation["side_effects"]["model_calls"] = 1
    security_observation["execution_metrics"] = {
        "model_calls": 1,
        "tool_calls": 1,
        "active_seconds": 0.1,
    }

    security_result = globals_["judge_state_case"](security, security_observation)

    assert security_result["judged_status"] == "pass"

    security_observation["side_effects"]["model_calls"] = 7
    security_observation["execution_metrics"]["model_calls"] = 7
    over_budget = globals_["judge_state_case"](security, security_observation)
    assert over_budget["judged_status"] == "fail"
    assert "profile_budget.model_calls" in over_budget["mismatches"]

    functional_observation["evaluation_profile"] = "B0"
    functional_observation["side_effects"]["model_calls"] = 2
    functional_observation["execution_metrics"]["model_calls"] = 2
    b0_over_budget = globals_["judge_state_case"](functional, functional_observation)
    assert b0_over_budget["judged_status"] == "fail"
    assert "profile_budget.model_calls" in b0_over_budget["mismatches"]


def test_stateful_b1_uses_registered_profile_and_runs_product_path() -> None:
    script_globals = _probe_globals("_build_retrieval_runtime")
    product_globals = _probe_globals("run_w05_product_case")
    case = next(case for case in load_w05_development_cases() if case.case_id == "gross-total-fen")
    model = script_globals["_RecordingModelAdapter"](script_globals["W05StateFakeModel"](case))
    runtime = script_globals["_build_retrieval_runtime"]("fake")

    def executor_factory(records):
        return script_globals["_RecordingQueryExecutor"](FixtureQueryExecutor(), records)

    result = product_globals["run_w05_product_case"](
        case,
        "B1",
        "w05-b1-regression",
        mode="fake",
        model=model,
        retriever=runtime[4],
        recording_executor_factory=executor_factory,
    )

    observation = result["observation"]
    assert observation["status"] == "succeeded"
    assert observation["facts"][0]["value"] == 15000
    assert observation["retrieval_records"]
    assert observation["model_call_records"][0]["response_shape"]["action_name"] == "search_catalog"
    from queryshield.evaluation.w05_provenance import classify_w05_source_lineage

    lineage = classify_w05_source_lineage(observation)
    assert lineage["status"] == "pass", lineage["errors"]
    assert lineage["source_paths"]["direct_database_catalog"]
    assert lineage["source_paths"]["current_run_retrieval_to_model"]
    assert all(item["run_id"] == observation["profile_run_id"] for item in lineage["source_paths"]["current_run_retrieval_to_model"])


def test_w05_server_prefetch_retrieval_reaches_same_run_answer_context() -> None:
    script_globals = _probe_globals("_build_retrieval_runtime")
    product_globals = _probe_globals("run_w05_product_case")
    case = next(case for case in load_w05_development_cases() if case.case_id == "gross-total-fen")
    model = script_globals["_RecordingModelAdapter"](script_globals["W05StateFakeModel"](case))
    runtime = script_globals["_build_retrieval_runtime"]("fake")

    def executor_factory(records):
        return script_globals["_RecordingQueryExecutor"](FixtureQueryExecutor(), records)

    product = product_globals["run_w05_product_case"](
        case,
        "B1",
        "w05-server-prefetch-regression",
        mode="fake",
        model=model,
        retriever=runtime[4],
        recording_executor_factory=executor_factory,
        server_prefetch_retrieval=True,
    )
    observation = product["observation"]
    lineage = script_globals["classify_w05_source_lineage"](observation)
    retrieved_documents = [
        item for item in lineage["source_paths"]["current_run_retrieval_to_model"]
        if item["source_kind"] == "knowledge_document"
    ]

    assert observation["status"] == "succeeded"
    assert observation["retrieval_orchestration"]["orchestrator"] == "server_prefetch_before_bounded_agent"
    assert observation["retrieval_orchestration"]["run_id"] == observation["profile_run_id"]
    assert observation["retrieval_orchestration"]["retrieval_id"] == observation["retrieval_records"][0]["retrieval_id"]
    assert retrieved_documents
    assert lineage["status"] == "pass", lineage["errors"]
    controls = script_globals["_check_source_lineage_negative_controls"](observation)
    assert controls["status"] == "pass"
    assert observation["usage"]["usage_status"] == "unknown"


@pytest.mark.parametrize("error_kind", ["facade", "driver"])
def test_stateful_b1_single_repair_executes_and_binds_the_actual_query_result(error_kind) -> None:
    script_globals = _probe_globals("_build_retrieval_runtime")
    product_globals = _probe_globals("run_w05_product_case")
    case = next(
        case
        for case in load_w05_development_cases()
        if case.case_id == "tool-text-injection-untrusted-instruction"
    )
    model = script_globals["_RecordingModelAdapter"](script_globals["W05StateFakeModel"](case))
    runtime = script_globals["_build_retrieval_runtime"]("fake")

    class FailFirstQueryExecutor:
        def __init__(self):
            self.calls = 0
            self.fixture = FixtureQueryExecutor()

        def execute(self, sql, *, context, params=(), metric_bindings=()):
            self.calls += 1
            if self.calls == 1:
                if error_kind == "driver":
                    raise UndefinedColumn("deliberately invalid first projection")
                raise ToolError("invalid_sql", "fixture rejects the deliberately invalid first projection")
            return self.fixture.execute(
                sql,
                context=context,
                params=params,
                metric_bindings=metric_bindings,
            )

    def executor_factory(records):
        return script_globals["_RecordingQueryExecutor"](FailFirstQueryExecutor(), records)

    result = product_globals["run_w05_product_case"](
        case,
        "B1",
        "w05-b1-single-repair-regression",
        mode="fake",
        model=model,
        retriever=runtime[4],
        recording_executor_factory=executor_factory,
    )

    observation = result["observation"]
    assert observation["status"] == "succeeded"
    assert observation["side_effects"]["repair_calls"] == 1
    assert [record["status"] for record in result["sql_records"]] == ["failed", "succeeded"]
    if error_kind == "driver":
        assert result["sql_records"][0]["sqlstate"] == "42703"
        assert result["sql_records"][0]["error_type"] == "UndefinedColumn"
    else:
        assert result["sql_records"][0]["error_code"] == "invalid_sql"
    assert "created_at >= %s" in result["sql_records"][0]["sql"]
    assert result["sql_records"][0]["sql"] == result["sql_records"][1]["sql"]
    assert "missing_amount" not in result["sql_records"][1]["sql"]
    assert observation["facts"][0]["value"] == 15000
    assert observation["facts"][0]["result_id"] == result["sql_records"][1]["result_id"]
    assert observation["usage"]["usage_status"] == "unknown"
    assert observation["usage"]["total_tokens"] is None
    assert observation["side_effects"]["model_calls"] == 3
    assert len(observation["model_call_records"]) == 3


@pytest.mark.parametrize("profile,error_type,expected_code", [
    ("B0", UndefinedColumn, "invalid_sql"),
    ("B1", ConnectionFailure, "database_unavailable"),
    ("B1", QueryCanceled, "query_timeout"),
    ("B1", InsufficientPrivilege, "forbidden"),
])
def test_stateful_sql_errors_keep_actual_calls_without_unapproved_repair(profile, error_type, expected_code):
    globals_ = _probe_globals("_build_retrieval_runtime")
    case = next(
        case
        for case in load_w05_development_cases()
        if case.case_id == "tool-text-injection-untrusted-instruction"
    )
    model = globals_["_RecordingModelAdapter"](globals_["W05StateFakeModel"](case))
    runtime = globals_["_build_retrieval_runtime"]("fake")

    class FailedDatabase:
        def execute(self, *args, **kwargs):
            raise error_type("driver body must stay private")

    result = globals_["run_w05_product_case"](
        case, profile, f"driver-{profile}", mode="fake", model=model, retriever=runtime[4],
        recording_executor_factory=lambda records: globals_["_RecordingQueryExecutor"](FailedDatabase(), records),
    )
    observation = result["observation"]
    assert observation["error_code"] == expected_code
    assert observation["side_effects"]["model_calls"] == 1
    assert observation["side_effects"]["repair_calls"] == 0
    assert len(observation["model_call_records"]) == 1
    assert len(observation["sql_records"]) == 1
    assert observation["execution_metrics"]["readonly_query_attempts"] == 1
    assert observation["side_effects"]["readonly_queries"] == 0
    assert observation["facts"] == []
    assert observation["usage"]["usage_status"] == "unknown"
    assert "driver body" not in str(observation)


def test_supplement_judgement_blocks_security_always_and_functional_only_in_fake() -> None:
    reasons = _probe_globals("_supplement_failure_reasons")["_supplement_failure_reasons"]
    summary = {
        "profile_summaries": {
            "B0": {"case_count": 2, "pass_count": 2},
            "B1": {"case_count": 2, "pass_count": 1},
        }
    }
    assert reasons(summary, 0, mode="fake") == ["b1_supplement_cases_not_all_pass"]
    assert reasons(summary, 0, mode="real") == []
    assert reasons(summary, 1, mode="real") == ["forbidden_security_side_effects_observed"]


def test_declared_metrics_are_read_from_recorded_query_proposals() -> None:
    declared = _probe_globals("_declared_metrics_in_records")["_declared_metrics_in_records"]
    records = [
        {"provider_output": '{"type":"tool_call","name":"search_catalog","arguments":{"query":"q"}}'},
        {"provider_output": '{"type":"tool_call","name":"query_readonly","arguments":{"sql":"SELECT 1","params":{}}}'},
        {"provider_output": '{"type":"tool_call","name":"query_readonly","arguments":{"sql":"SELECT 1","params":{},"metrics":["net_fen"]}}'},
        {"provider_output": "not json"},
    ]
    assert declared(records) == [[], ["net_fen"]]


def test_b2a_smoke_covers_queries_critical_questions_injection_and_supplement() -> None:
    globals_ = _probe_globals("_run_w05_b2a_smoke")
    frozen = {case.case_id: case for case in load_w05_development_cases()}
    smoke = set(globals_["_W05_SMOKE_FROZEN_CASE_IDS"])
    queries_critical = {
        case_id for case_id, case in frozen.items()
        if case.critical_question_id is not None and case.case["action"]["entrypoint"] == "/queries"
    }
    assert queries_critical <= smoke
    assert smoke - queries_critical == {"tool-text-injection-untrusted-instruction"}
    assert len(smoke) == 6
