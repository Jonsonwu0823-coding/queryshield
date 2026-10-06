from __future__ import annotations

import json
from pathlib import Path
import sys

from scripts import check_eval
from queryshield.providers.contracts import ModelCallResult, ModelUsage


def test_probe_failure_emits_safe_frame_locations_without_exception_text(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    secret_text = "api_key=must-not-leak"

    def fail_probe(*_args, **_kwargs):
        raise AssertionError(secret_text)

    monkeypatch.setattr(check_eval, "run_check", fail_probe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_eval.py",
            "--check-id",
            "EVAL-R05",
            "--mode",
            "real",
            "--evidence-dir",
            str(tmp_path.resolve()),
        ],
    )

    exit_code = check_eval.main()
    output = capsys.readouterr()
    combined_output = output.out + output.err

    assert exit_code == 1
    assert "sanitized_trace_frame file=scripts/check_eval.py" in combined_output
    assert "sanitized_trace_frame file=tests/test_eval_failure_diagnostics.py" in combined_output
    assert secret_text not in combined_output

    record = json.loads((tmp_path / "EVAL-R05.json").read_text(encoding="utf-8"))
    assert record["status"] == "fail"
    assert record["error_type"] == "AssertionError"
    assert record["reason"] == "probe_failed; sanitized frame locations are emitted on stderr"
    assert record["provider_execution_status"] == "unknown"
    assert record["usage"] == {
        "usage_status": "unknown",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    assert secret_text not in json.dumps(record)


def test_r05_provenance_failure_persists_profile_observations_and_unknown_usage(
    tmp_path: Path,
) -> None:
    model_record = {
        "model_call_id": "call-b0-1",
        "proposal_type": "tool_call",
        "proposal": {"type": "tool_call", "name": "query_readonly", "sql": "SELECT 1"},
        "usage_status": "unknown",
        "usage": None,
    }
    query_record = {
        "run_id": "run-b0",
        "tenant_id": "A",
        "status": "succeeded",
        "statement_kind": "SELECT",
        "rows": [],
    }
    result = check_eval._r05_provenance_failure(
        tmp_path,
        scenario="success_candidate",
        profile="B0",
        run_id="run-b0",
        expected_tenant_id="A",
        normalized_observation={"status": "failed", "rows": []},
        raw_run_record={"answer": "未生成答案", "facts": []},
        query_records=[query_record],
        model_records=[model_record],
        all_model_records=[model_record],
        all_query_records=[query_record],
        completed_scenarios={},
    )

    assert result["failed_assertion"] == "r05_success_tenant_result_query_provenance"
    assert result["profile"] == "B0"
    assert result["observed"]["matching_tenant_nonempty_query_count"] == 0
    assert result["provider_execution_status"] == "real_calls_recorded_before_probe_failure"
    assert result["usage"] == {
        "usage_status": "unknown",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }

    partial = json.loads((tmp_path / "real-success-and-safety-replay.partial.json").read_text(encoding="utf-8"))
    assert partial["failed_profile"]["query_execution_records"][0]["rows"] == []
    assert partial["failed_profile"]["model_call_records"][0]["proposal"]["sql"] == "SELECT 1"
    assert partial["usage"]["total_tokens"] is None


def test_r05_usage_summary_sums_only_when_every_call_usage_is_known() -> None:
    known = [
        {"usage_status": "known", "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
        {"usage_status": "known", "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}},
    ]
    mixed = known + [{"usage_status": "unknown", "usage": None}]

    assert check_eval._r05_usage_summary(known) == {
        "usage_status": "known",
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }
    assert check_eval._r05_usage_summary(mixed) == {
        "usage_status": "unknown",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def test_recording_model_adapter_keeps_action_shape_without_raw_response() -> None:
    class Delegate:
        mode = "real"
        provider = "test-provider"
        model = "test-model"

        def complete(self, _messages, *, request_id=None, model_call_id=None):
            return ModelCallResult(
                mode="real",
                provider=self.provider,
                model=self.model,
                request_id=request_id or "request-test",
                model_call_id=model_call_id or "call-test",
                provider_call_id="provider-call-test",
                provider_request_id="provider-request-test",
                content=json.dumps({
                    "type": "tool_call",
                    "name": "query_readonly",
                    "arguments": {"sql": "SELECT 1"},
                }),
                usage=ModelUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                usage_status="known",
            )

    adapter = check_eval._RecordingModelAdapter(Delegate())
    adapter.complete([], request_id="request-test", model_call_id="call-test")

    record = adapter.records[0]
    assert record["response_shape"] == {
        "json_status": "valid",
        "payload_kind": "dict",
        "top_level_keys": ["arguments", "name", "type"],
        "action_type": "tool_call",
        "action_name": "query_readonly",
        "argument_keys": ["sql"],
    }
    assert "content" not in record
    assert record["proposal"]["params"] is None


def test_recording_model_adapter_records_tool_name_misused_as_type_without_raw_response() -> None:
    class Delegate:
        mode = "real"
        provider = "test-provider"
        model = "test-model"

        def complete(self, _messages, *, request_id=None, model_call_id=None):
            return ModelCallResult(
                mode="real",
                provider=self.provider,
                model=self.model,
                request_id=request_id or "request-type-mismatch",
                model_call_id=model_call_id or "call-type-mismatch",
                provider_call_id="provider-call-type-mismatch",
                provider_request_id="provider-request-type-mismatch",
                content=json.dumps({
                    "type": "query_readonly",
                    "name": "query_readonly",
                    "arguments": {"sql": "SELECT 1", "params": {}},
                }),
                usage=ModelUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                usage_status="known",
            )

    adapter = check_eval._RecordingModelAdapter(Delegate())
    adapter.complete([], request_id="request-type-mismatch", model_call_id="call-type-mismatch")

    record = adapter.records[0]
    assert record["response_shape"] == {
        "json_status": "valid",
        "payload_kind": "dict",
        "top_level_keys": ["arguments", "name", "type"],
        "action_type": "query_readonly",
        "action_name": "query_readonly",
        "argument_keys": ["params", "sql"],
    }
    assert "content" not in record
    assert "proposal" not in record


def test_recording_model_adapter_keeps_redacted_provider_failure_record() -> None:
    from queryshield.providers.contracts import ModelProviderError

    failure_record = {
        "status": "failed",
        "mode": "real",
        "provider": "openai_compatible",
        "model": "qwen-plus",
        "request_id": "request-502",
        "model_call_id": "call-502",
        "provider_call_id": None,
        "provider_request_id": "provider-request-502",
        "stream": False,
        "content_present": False,
        "usage": None,
        "usage_status": "unknown",
        "error_code": "upstream_http_error",
        "http_status": 502,
    }

    class Delegate:
        mode = "real"
        provider = "openai_compatible"
        model = "qwen-plus"

        def complete(self, _messages, *, request_id=None, model_call_id=None):
            raise ModelProviderError("upstream_http_error", failure_record)

    adapter = check_eval._RecordingModelAdapter(Delegate())
    try:
        adapter.complete([], request_id="request-502", model_call_id="call-502")
    except ModelProviderError as exc:
        assert exc.code == "upstream_http_error"
    else:
        raise AssertionError("provider failure must still propagate to the runtime")

    assert adapter.records == [failure_record]
    assert adapter.records[0]["http_status"] == 502
    assert adapter.records[0]["provider_request_id"] == "provider-request-502"
    assert adapter.records[0]["usage_status"] == "unknown"
    assert "content" not in adapter.records[0]


def test_real_r05_retains_b0_failure_and_runs_security_scenario(monkeypatch, tmp_path: Path) -> None:
    required = {
        "QUERYSHIELD_DATABASE_URL": "unused-local-dsn",
        "QUERYSHIELD_MODEL_BASE_URL": "https://provider.invalid/v1",
        "QUERYSHIELD_MODEL_API_KEY": "test-only-not-a-secret",
        "QUERYSHIELD_MODEL_NAME": "test-model",
    }
    for name, value in required.items():
        monkeypatch.setenv(name, value)

    class ProviderFactory:
        @classmethod
        def from_env(cls):
            return object()

    import queryshield.providers.openai_compatible as compatible

    monkeypatch.setattr(compatible, "OpenAICompatibleModel", ProviderFactory)

    class RecordingModel:
        provider = "test-provider"
        model = "test-model"
        records: list[dict[str, object]] = []

    model = RecordingModel()
    monkeypatch.setattr(check_eval, "_RecordingModelAdapter", lambda _delegate: model)

    class NoDatabaseCall:
        def execute(self, *_args, **_kwargs):
            raise AssertionError("the stubbed profile runner owns query observations")

    recorders = []
    original_recorder = check_eval._RecordingQueryExecutor

    def make_recorder(delegate):
        recorder = original_recorder(delegate)
        recorders.append(recorder)
        return recorder

    monkeypatch.setattr(check_eval, "_RecordingQueryExecutor", make_recorder)
    monkeypatch.setattr(check_eval, "GuardedQueryExecutor", NoDatabaseCall)
    observed_scenarios: list[str] = []

    def make_profile(short: str, scenario: str, call_ids: list[str]) -> dict[str, object]:
        is_positive = scenario == "success_candidate"
        is_b0_parse_failure = is_positive and short == "B0"
        return {
            "profile": "B0-single-pass" if short == "B0" else "B1-bounded-agent",
            "status": "failed" if is_b0_parse_failure else ("succeeded" if is_positive else "denied"),
            "terminal_state": "FAILED" if is_b0_parse_failure else ("SUCCEEDED" if is_positive else "DENIED"),
            "error_code": "missing_field" if is_b0_parse_failure else None,
            "http_status": None,
            "answer": "verified test answer" if is_positive and short == "B1" else None,
            "rows": [{"gross_fen": 15000}] if is_positive and short == "B1" else [],
            "facts": {"facts": [{"value": 15000, "tenant_id": "A"}]} if is_positive and short == "B1" else [],
            "invariants": {},
            "side_effects": {
                "model_calls": len(call_ids),
                "readonly_queries": 1 if is_positive and short == "B1" else 0,
                "fact_count": 1 if is_positive and short == "B1" else 0,
                "write_statements": 0,
                "cross_tenant_rows": 0,
                "unauthorized_facts": 0,
            },
            "usage": {"usage_status": "known", "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "usage_summary": {"status": "known", "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model_call_count": len(call_ids),
            "tool_call_count": 1 if is_positive and short == "B1" else 0,
            "repair_count": 0,
            "model_call_ids": call_ids,
            "elapsed_ms": 5,
            "events": [
                {"kind": "tool_call", "tool_name": "query_readonly", "status": "succeeded"}
            ] if is_positive and short == "B1" else [],
            "trace": [],
        }

    def fake_pair(_model, tools, context, question, *, time_window=None):
        scenario = "success_candidate" if "success_candidate" in context.run_id else "security_failure"
        observed_scenarios.append(scenario)
        run_ids = {"B0": f"{context.run_id}-b0", "B1": f"{context.run_id}-b1"}
        profiles = {}
        for short in ("B0", "B1"):
            count = 1 if short == "B0" or scenario == "security_failure" else 2
            call_ids = [f"{scenario}-{short}-{index}" for index in range(count)]
            for index, call_id in enumerate(call_ids):
                model.records.append({
                    "model_call_id": call_id,
                    "provider_call_id": f"provider-{call_id}",
                    "usage_status": "known",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "response_shape": {"json_status": "valid", "payload_kind": "dict"},
                })
            profile = make_profile(short, scenario, call_ids)
            profiles[short] = profile
            if scenario == "success_candidate" and short == "B1":
                recorders[0].records.append({
                    "run_id": run_ids[short],
                    "tenant_id": "A",
                    "principal_id": "principal-A",
                    "result_id": "result-b1-positive",
                    "status": "succeeded",
                    "statement_kind": "SELECT",
                    "rows": [{"gross_fen": 15000}],
                })
        return {
            "profiles": profiles,
            "shared_runtime": {"profile_run_ids": run_ids},
        }

    monkeypatch.setattr(check_eval, "run_comparison_pair", fake_pair)

    result = check_eval._run_real_r05(tmp_path.resolve())

    assert observed_scenarios == ["success_candidate", "security_failure"]
    assert result["status"] == "fail"
    assert result["scenario_count"] == 2
    assert result["scenarios"]["success_candidate"]["profiles"]["B1"]["query_execution_records"][0]["rows"] == [{"gross_fen": 15000}]
    assert any(
        failure.get("failed_assertion") == "r05_success_tenant_result_query_provenance"
        and failure.get("profile") == "B0"
        for failure in result["failed_assertions"]
    )
    full_evidence = json.loads((tmp_path / "real-success-and-safety-replay.json").read_text(encoding="utf-8"))
    assert "security_failure" in full_evidence["scenarios"]
