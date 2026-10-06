"""The native protocol through the context, the graph, resume, HTTP and the evaluation.

Native mode changes only how the model returns its decision; the server
rebuilds the context each call, validates with the json parser and keeps the
checkpoint format.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import itertools
import json
import re
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from queryshield.agent.config import DEFAULT_RUN_CONFIG, NATIVE_VERSIONS, RunConfig
from queryshield.agent.context import MAX_SERVER_CONTEXT_CHARS, NO_DATA_ACTION, build_context, native_tools
from queryshield.agent.graph import BoundedAgent, RunResumeError
from queryshield.agent.metric_intent import (
    DEFINITION_ANSWER_SHAPE,
    DEFINITION_SEARCH_ACTION,
    answer_not_grounded_hint,
    declaration_examples,
)
from queryshield.agent.proposals import ExecutionContext
from queryshield.agent.runtime import (
    B0_PROFILE,
    B1_PROFILE,
    RuntimeConfigurationError,
    product_run_config,
)
from queryshield.api.main import app, get_model_provider
from queryshield.approval.service import FixtureQueryExecutor, shared_run_service
from queryshield.catalog import load_default_catalog
from queryshield.evaluation.comparison import build_comparison_profiles
from queryshield.evaluation.state_cases import load_state_cases
from queryshield.evaluation.state_oracle import judge_state_case
from queryshield.evaluation.stateful_product import StateCaseFakeModel, run_product_case
from queryshield.knowledge.runtime import shared_retrieval_runtime
from queryshield.providers.contracts import ModelCallResult, NativeToolCall, native_call_for
from queryshield.providers.fake_model import FakeModel
from queryshield.tools.semantic import ControlledTools

from test_http_queries import APPROVER, REQUESTER, ask, auth, env, fact_values, wait  # noqa: F401  (env is a fixture)
from test_metric_intent import SEPTEMBER_UTC, WORST_SERVER_CONTEXT_CHARS, _longest_real_identities
from test_stateful_replay import _run_fixture_state_path


NATIVE = replace(DEFAULT_RUN_CONFIG, **NATIVE_VERSIONS)
CONTEXT = ExecutionContext(run_id="run-golden", tenant_id="A", principal_id="principal-A", role="requester")
WINDOW = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z", "timezone": "UTC"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _system(config: RunConfig = NATIVE, **kwargs) -> dict:
    built = build_context(CONTEXT, "2026年9月已支付订单总额", run_config=config, parallel_available=False, **kwargs)
    return json.loads(built.messages[0]["content"].split("\n", 1)[1])


# --- Context -----------------------------------------------------------------

# sha256 of the native rendered messages (same method as the context-v16 pins
# in test_shared_runtime), keyed by (request window, retrieval); and of the tools.
_NATIVE_CONTEXT_SHA256 = {
    (False, False): "faa57709e5c6eee7078db10b0d8a9f4301a40d765b85388ad449569ae8c2d416",
    (False, True): "309c41e29aa80408ace01fbc76b611bf1fdfdfec2616ab0b8b4a95ebebad897e",
    (True, False): "eabf5347ea66a5b71cd556165067c63e7b6e89c84fe250252727ad2ff7d7551e",
    (True, True): "1a0662ece4d6502ede6c775cb134b4e8a0701eae99e265ad7a45247b29aa88e0",
}
_NATIVE_TOOLS_SHA256 = {
    False: "8be5cf48d08772b9b354fc48868da4bfe12e296ac082af36510d8178cc760aa1",
    True: "0afcefcecf99e09d260f66d8a6b0a3a46d706e297935c78b30c1939c5bac3c80",
}


@pytest.mark.parametrize("has_window, retrieval", list(_NATIVE_CONTEXT_SHA256))
def test_native_messages_and_tools_are_pinned(has_window, retrieval) -> None:
    result = build_context(
        CONTEXT,
        "2026年9月已支付订单总额",
        request_time_window=WINDOW if has_window else None,
        parallel_available=False,
        retrieval_available=retrieval,
        run_config=NATIVE,
    )
    assert result.context_version == "context-v16"
    assert _sha256([dict(message) for message in result.messages]) == _NATIVE_CONTEXT_SHA256[(has_window, retrieval)]
    assert _sha256(native_tools(retrieval_available=retrieval)) == _NATIVE_TOOLS_SHA256[retrieval]


def test_native_contract_keeps_every_semantic_rule_and_drops_the_wire_format() -> None:
    native = _system(request_time_window=WINDOW)
    json_contract = _system(DEFAULT_RUN_CONFIG, request_time_window=WINDOW)["action_contract"]
    contract = native["action_contract"]
    text = json.dumps(contract, ensure_ascii=False)

    assert set(contract) == {"output", "functions", "workflow"}
    for wire in ("valid_shape_examples", "type_value", "forbidden_extra_fields", "required_fields", '"type":"tool_call"', "parallel_readonly"):
        assert wire not in text
    # Each semantic rule is the json contract's own value, not a copy.
    query = contract["functions"]["query_readonly"]
    json_query = json_contract["actions"]["tool_call"]["tools"]["query_readonly"]
    for key in ("allowed_tables", "table_name_rule", "supported_sql_shape", "metric_query_guidance", "repair_hints", "bounded_repair", "params"):
        assert query[key] == json_query[key]
    assert query["example_arguments"] == declaration_examples(WINDOW)["paid_count_and_gross_fen"]
    assert query["non_metric_rows"]["rule"] == json_query["non_metric_rows"]["rule"]
    assert query["non_metric_rows"]["example"] == json.loads(json_query["non_metric_rows"]["example"])["arguments"]
    assert contract["functions"]["ask_user"]["clarification_id"] == json_contract["actions"]["ask_user"]["clarification_id"]
    for key in ("basis", "fact_refs_rule"):
        assert contract["functions"]["final_answer"][key] == json_contract["actions"]["final_answer"][key]
    assert contract["workflow"] == json_contract["workflow"]
    assert native["metric_declaration"] == _system(DEFAULT_RUN_CONFIG, request_time_window=WINDOW)["metric_declaration"]
    assert native["runtime_versions"] == NATIVE.as_dict()
    # Rules live in one place: no function description repeats one.
    descriptions = " ".join(tool["function"]["description"] for tool in native_tools(retrieval_available=True))
    for rule in (json_query["table_name_rule"], json_contract["actions"]["final_answer"]["fact_refs_rule"], *contract["workflow"]):
        assert rule not in descriptions


def test_native_without_retriever_never_mentions_search_catalog() -> None:
    built = build_context(CONTEXT, "q", retrieval_available=False, run_config=NATIVE)
    assert "search_catalog" not in built.messages[0]["content"]
    assert "search_catalog" not in json.dumps(native_tools(retrieval_available=False))


def test_native_renders_quoted_json_actions_as_function_calls_and_keeps_the_record() -> None:
    hint = answer_not_grounded_hint(None, retrieval_available=True)
    record = {"tool_name": "final_answer", "status": "failed", "error_code": "answer_not_grounded", "repair_hint": hint}
    native = build_context(CONTEXT, "q", tool_results=[record], run_config=NATIVE).messages[-1]["content"]
    plain = build_context(CONTEXT, "q", tool_results=[record]).messages[-1]["content"]

    for action in (NO_DATA_ACTION, DEFINITION_SEARCH_ACTION, DEFINITION_ANSWER_SHAPE):
        assert json.dumps(action, ensure_ascii=False) in plain
        assert json.dumps(action, ensure_ascii=False) not in native
    assert '\\"type\\"' not in native
    assert "final_answer {" in native and "search_catalog {" in native
    assert record["repair_hint"] is hint and hint["no_data_action"] == NO_DATA_ACTION


def test_native_worst_case_system_message_fits_the_policy_cap() -> None:
    context = ExecutionContext(**_longest_real_identities())
    december = {"start": "2026-12-01T00:00:00Z", "end": "2027-01-01T00:00:00Z", "timezone": "UTC"}
    metric_ids = ("paid_count", "gross_fen", "net_fen")
    sizes = {}
    for parallel, retrieval, window, binding_count, confirmed in itertools.product(
        (True, False), (True, False), (None, SEPTEMBER_UTC, december), range(4), (None, *metric_ids)
    ):
        bindings = [
            {"metric_id": metric_id, "result_position": metric_id, "unit": "CNY_fen", "time_window": window or SEPTEMBER_UTC}
            for metric_id in metric_ids[:binding_count]
        ]
        built = build_context(
            context,
            "q",
            request_time_window=window,
            metric_bindings=bindings,
            confirmed_metric=confirmed,
            time_window=window,
            parallel_available=parallel,
            retrieval_available=retrieval,
            run_config=NATIVE,
        )
        sizes[(parallel, retrieval, window is not None, binding_count, confirmed)] = len(built.messages[0]["content"])
    worst = max(sizes, key=sizes.get)
    assert sizes[worst] <= WORST_SERVER_CONTEXT_CHARS == MAX_SERVER_CONTEXT_CHARS - 500, (worst, sizes[worst])


# --- Run configuration ------------------------------------------------------


def test_protocol_setting_selects_b1_versions_and_b0_stays_json(monkeypatch) -> None:
    catalog = load_default_catalog()
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", " Native ")
    b1 = product_run_config(B1_PROFILE, catalog=catalog, retriever=None)
    b0 = product_run_config(B0_PROFILE, catalog=catalog, retriever=None)
    assert b1.model_protocol == "native" and b1.as_dict() | NATIVE_VERSIONS == b1.as_dict()
    assert b0.model_protocol == "json" and b0.adapter_version == DEFAULT_RUN_CONFIG.adapter_version
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", "")
    assert product_run_config(B1_PROFILE, catalog=catalog, retriever=None).model_protocol == "json"


@pytest.mark.parametrize("profile", [B0_PROFILE, B1_PROFILE])
def test_unknown_protocol_is_blocked_not_json(monkeypatch, profile) -> None:
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", "xml")
    with pytest.raises(RuntimeConfigurationError) as caught:
        product_run_config(profile, catalog=load_default_catalog(), retriever=None)
    assert caught.value.code == "invalid_model_protocol"


def test_eval_comparison_moves_protocol_versions_to_the_profiles(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", "native")
    real = build_comparison_profiles(PROJECT_ROOT, provider_mode="real", model_name="m")
    fake = build_comparison_profiles(PROJECT_ROOT, provider_mode="fake", model_name="m")
    assert "adapter_version" not in real["shared_configuration"]
    b0, b1 = real["profiles"]
    assert (b0["adapter_version"], b1["adapter_version"]) == (DEFAULT_RUN_CONFIG.adapter_version, NATIVE.adapter_version)
    assert b1["prompt_version"] == NATIVE.prompt_version
    assert {profile["adapter_version"] for profile in fake["profiles"]} == {DEFAULT_RUN_CONFIG.adapter_version}


# --- Graph ------------------------------------------------------------------


def _agent(model, config: RunConfig) -> BoundedAgent:
    tools = ControlledTools(catalog=load_default_catalog(), executor=FixtureQueryExecutor(), retriever=shared_retrieval_runtime("fake").retriever)
    return BoundedAgent(model, tools=tools, run_config=config)


class _ToolsSeen(FakeModel):
    def __init__(self) -> None:
        self.tools_seen: list[object] = []

    def complete(self, messages, **kwargs):
        self.tools_seen.append(kwargs.get("tools"))
        return super().complete(messages, **kwargs)


@pytest.mark.parametrize(
    "question",
    ["2026年9月已支付订单总额", "2026年9月退款后净额", "退款后净额是怎么算的？", "你好，你能做什么？", "支付金额是多少？", "2026年9月销售额是多少？"],
)
def test_fake_model_makes_the_same_decisions_in_both_protocols(question) -> None:
    results = {}
    for config in (DEFAULT_RUN_CONFIG, NATIVE):
        model = _ToolsSeen()
        results[config.model_protocol] = (_agent(model, config).run(CONTEXT, question), model.tools_seen)
    (json_run, json_tools), (native_run, native_tools_seen) = results["json"], results["native"]
    assert json_tools == [None] * len(json_tools)
    assert native_tools_seen and all(tools == native_tools(retrieval_available=True) for tools in native_tools_seen)
    for field in ("status", "error_code", "action", "facts", "answer", "model_call_count", "tool_call_count", "answer_status"):
        assert _without_ids(getattr(native_run, field)) == _without_ids(getattr(json_run, field)), field
    assert native_run.run_config == NATIVE


def _without_ids(value: object) -> object:
    """Result and fact ids are fresh per run."""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return re.sub(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}|fact-[0-9a-f]{24}", "<id>", text)


class _Native:
    """Scripted native replies: each item is (content, tool_calls)."""

    mode = "fake"
    provider = "scripted-native"
    model = "scripted-native-v1"

    def __init__(self, replies) -> None:
        self.replies = list(replies)
        self.calls = 0

    def complete(self, messages, *, request_id=None, model_call_id=None, tools=None):
        assert tools is not None
        content, calls = self.replies[self.calls]
        self.calls += 1
        return ModelCallResult(
            mode="fake", provider=self.provider, model=self.model, request_id=request_id, model_call_id=model_call_id,
            provider_call_id=None, provider_request_id=None, content=content, usage=None, usage_status="unknown",
            tool_calls=calls, finish_reason="stop",
        )


def _events(result, kind: str) -> list[dict]:
    return [dict(event) for event in result.events if event.get("kind") == kind]


@pytest.mark.parametrize(
    "calls, code",
    [
        ((), "native_tool_call_missing"),
        ((NativeToolCall("a", "deny", '{"reason":"x"}'), NativeToolCall("b", "deny", '{"reason":"y"}')), "native_multiple_tool_calls"),
        ((NativeToolCall("a", "ask_user", '{"type":"deny","question":"q"}'),), "unknown_field"),
    ],
)
def test_native_parse_failure_ends_the_run_without_repair(calls, code) -> None:
    model = _Native([("some text", calls)])
    result = _agent(model, NATIVE).run(CONTEXT, "2026年9月已支付订单总额")
    assert (result.status, result.error_code, result.model_call_count, result.repair_count) == ("failed", code, 1, 0)
    assert [event["error_code"] for event in _events(result, "proposal_validation")] == [code]


def test_native_events_record_call_shape_never_arguments_or_content() -> None:
    content = "I will run a query; secret marker in content"
    query = NativeToolCall("a", "query_readonly", json.dumps({"sql": "SELECT c.customer_id, c.name FROM customers AS c", "params": {}}))
    result = _agent(_Native([(content, (query,))]), NATIVE).run(CONTEXT, "查询客户姓名")
    call = _events(result, "model_call")[0]
    assert call["finish_reason"] == "stop" and call["content_length"] == len(content)
    assert call["tool_calls"] == [
        {"name": "query_readonly", "arguments_length": len(query.arguments), "arguments_sha256": hashlib.sha256(query.arguments.encode()).hexdigest()}
    ]
    events = json.dumps([dict(event) for event in result.events], ensure_ascii=False)
    assert "secret marker" not in events and query.arguments not in events
    # The content beside a call is ignored: the call decided the action.
    assert result.status == "waiting_approval" and result.action["tool_call"]["tool"] == "query_readonly"


def test_native_lone_surrogate_escape_is_denied_like_json() -> None:
    call = NativeToolCall("a", "deny", '{"reason":"\\ud800"}')
    native = _agent(_Native([("", (call,))]), NATIVE).run(CONTEXT, "2026年9月已支付订单总额")
    json_run = _agent(_Scripted(['{"type":"deny","reason":"\\ud800"}']), DEFAULT_RUN_CONFIG).run(CONTEXT, "2026年9月已支付订单总额")
    assert native.status == json_run.status == "denied"


class _Scripted(FakeModel):
    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)

    def complete(self, messages, **kwargs):
        return replace(super().complete(messages, **kwargs), content=self.outputs.pop(0))


def test_json_events_have_no_native_fields() -> None:
    result = _agent(FakeModel(), DEFAULT_RUN_CONFIG).run(CONTEXT, "2026年9月已支付订单总额")
    assert all("tool_calls" not in event and "finish_reason" not in event for event in _events(result, "model_call"))


@pytest.mark.parametrize("started, resumed", [(NATIVE, DEFAULT_RUN_CONFIG), (DEFAULT_RUN_CONFIG, NATIVE)])
def test_checkpoint_of_one_protocol_does_not_resume_under_the_other(started, resumed) -> None:
    agent = _agent(FakeModel(), started)
    waiting = agent.run(CONTEXT, "支付金额是多少？")
    assert waiting.status == "waiting_user"
    checkpoint = agent.export_waiting_checkpoint(CONTEXT.run_id)
    with pytest.raises(RunResumeError) as caught:
        _agent(FakeModel(), resumed).resume_from_checkpoint(CONTEXT, "2026年9月", checkpoint)
    assert caught.value.code == "resume_profile_mismatch"
    assert _agent(FakeModel(), started).resume_from_checkpoint(CONTEXT, "2026年9月", checkpoint).status == "succeeded"


# --- HTTP -------------------------------------------------------------------


@pytest.fixture()
def native_env(env, monkeypatch):
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", "native")
    model = _ToolsSeen()
    app.dependency_overrides[get_model_provider] = lambda: model
    return env, model


def _stored_config(run_id: str) -> dict:
    return shared_run_service().store.get_run(run_id)["run_config"]["agent_run_config"]


@pytest.mark.parametrize("question", ["2026年9月已支付订单总额", "2026年9月退款后净额", "按客户查看2026年9月已支付订单总额"])
def test_http_sync_and_async_native_match_json(env, monkeypatch, question) -> None:
    json_body = ask(env, question).json()
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", "native")
    model = _ToolsSeen()
    app.dependency_overrides[get_model_provider] = lambda: model
    native = ask(env, question).json()
    accepted = ask(env, question, asynchronous=True).json()
    final = wait(env, accepted["run_id"], {"SUCCEEDED", "FAILED", "WAITING_USER", "WAITING_APPROVAL"})
    for body in (native, final):
        assert body["status"] == json_body["status"] == "SUCCEEDED"
        assert body["model_call_count"] == json_body["model_call_count"]
    assert fact_values(native) == fact_values(json_body)
    stored = env.get(f"/runs/{final['run_id']}/result", headers=auth(REQUESTER)).json()
    assert fact_values(stored) == fact_values(json_body)
    assert model.tools_seen and all(tools is not None for tools in model.tools_seen)
    assert _stored_config(native["run_id"])["adapter_version"] == NATIVE.adapter_version


def test_http_resume_continues_in_the_protocol_the_run_started_with(native_env, monkeypatch) -> None:
    client, model = native_env
    waiting = ask(client, "2026年9月销售额是多少？").json()
    assert waiting["status"] == "WAITING_USER"
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", "json")
    before = len(model.tools_seen)
    resumed = client.post(f"/runs/{waiting['run_id']}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额"}).json()
    assert resumed["status"] == "SUCCEEDED" and fact_values(resumed) == [("gross_fen", 15000)]
    assert len(model.tools_seen) > before and all(tools is not None for tools in model.tools_seen[before:])
    assert _stored_config(waiting["run_id"])["adapter_version"] == NATIVE.adapter_version


def test_http_resume_rejects_a_checkpoint_whose_protocol_differs_from_the_stored_run(native_env) -> None:
    client, _ = native_env
    waiting = ask(client, "2026年9月销售额是多少？").json()
    store = shared_run_service().store
    run = store.get_run(waiting["run_id"])
    checkpoint = dict(run["checkpoint"])
    checkpoint["agent_checkpoint"] = {**checkpoint["agent_checkpoint"], "run_config": DEFAULT_RUN_CONFIG.as_dict()}
    store.update_run(waiting["run_id"], checkpoint_json=json.dumps(checkpoint, ensure_ascii=False))
    response = client.post(f"/runs/{waiting['run_id']}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额"})
    assert response.status_code == 409 and response.json()["error"]["code"] == "checkpoint_invalid"


def test_http_native_sensitive_query_waits_for_approval_then_runs(native_env) -> None:
    client, model = native_env
    pending = ask(client, "查询客户姓名").json()
    assert pending["status"] == "WAITING_APPROVAL" and pending["sql_exec_count"] == 0
    approved = client.post(
        f"/runs/{pending['run_id']}/approval", headers=auth(APPROVER), json={"approval_id": pending["approval_id"], "decision": "approve"}
    )
    assert approved.status_code == 200 and approved.json()["status"] == "SUCCEEDED"
    result = client.get(f"/runs/{pending['run_id']}/result", headers=auth(REQUESTER)).json()
    assert {row["name"] for row in result["result"]["rows"]} == {"甲", "乙"}
    assert model.tools_seen and all(tools is not None for tools in model.tools_seen)


def test_http_unknown_protocol_is_503_before_any_model_call(env, monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", "xml")
    model = _ToolsSeen()
    app.dependency_overrides[get_model_provider] = lambda: model
    response = ask(env, "2026年9月已支付订单总额")
    assert response.status_code == 503 and response.json()["error"]["code"] == "invalid_model_protocol"
    assert model.tools_seen == []


# --- evaluation -------------------------------------------------------------


class _RealStateModel(StateCaseFakeModel):
    """The evaluation script under a real-mode label, recording whether tools came."""

    mode = "real"

    def complete(self, messages, *, request_id=None, model_call_id=None, tools=None):
        self.tools_seen = getattr(self, "tools_seen", []) + [tools]
        result = super().complete(messages, request_id=request_id, model_call_id=model_call_id)
        if tools is None:
            return result
        name, arguments = native_call_for(json.loads(result.content))
        return replace(result, content="", tool_calls=(NativeToolCall("w05", name, json.dumps(arguments)),), finish_reason="tool_calls")


@pytest.mark.parametrize("model_class, protocol", [(StateCaseFakeModel, "json"), (_RealStateModel, "native")])
def test_eval_fake_stays_json_and_only_a_real_model_follows_the_setting(monkeypatch, model_class, protocol) -> None:
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", "native")
    case = next(item for item in load_state_cases() if item.case_id == "gross-total-fen")
    model = model_class(case)
    model._search_issued = True
    product = run_product_case(
        case,
        "B1",
        f"run-c1-w05-{protocol}",
        mode="fake",
        model=model,
        retriever=SimpleNamespace(snapshot=SimpleNamespace(snapshot_id="test-snapshot", source_records=())),
        recording_executor_factory=lambda records: FixtureQueryExecutor(),
    )
    observation = product["observation"]
    assert RunConfig.from_dict(observation["run_config"]).model_protocol == protocol
    assert observation["model_call_records"]
    if protocol == "native":
        assert all(tools is not None for tools in model.tools_seen)
        assert all(record["tool_call_count"] == 1 for record in observation["model_call_records"])


def test_eval_resume_route_fake_stays_json_under_the_native_setting(monkeypatch) -> None:
    # _state_path_observation: the evaluation's own Fake takes no tools, so following the setting would fail the case.
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", "native")
    case = next(item for item in load_state_cases() if item.case_id == "clarification-context-net-resumed")
    result = _run_fixture_state_path(case, profile="B1", model=StateCaseFakeModel(case))
    assert RunConfig.from_dict(result["initial_agent_checkpoint"]["run_config"]).model_protocol == "json"
    observation = result["observation"]
    assert observation["status"] == "succeeded"
    assert judge_state_case(case, observation)["judged_status"] == "pass"


# --- Summary script ---------------------------------------------------------


def test_summary_reports_protocol_passes_and_cost_without_model_text(tmp_path) -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "eval_multi_run_summary.py"))

    def record(case_id, profile, status, config, *, usage, codes=()):
        return {
            "case_id": case_id, "profile": profile, "split": "development", "critical_question_id": "q" if case_id == "c1" else None,
            "judged_status": status, "error_code": None, "terminal_state": "SUCCEEDED", "run_config": config.as_dict(),
            "execution_metrics": {"model_calls": 2, "tool_calls": 1}, "elapsed_ms": 10, "usage": usage,
            "execution_events": [{"kind": "proposal_validation", "error_code": code} for code in codes],
            "raw_content": "MODEL TEXT",
        }

    roots = []
    for name, config, usage in (
        ("json", DEFAULT_RUN_CONFIG, {"usage_status": "unknown"}),
        ("native", NATIVE, {"usage_status": "known", "prompt_tokens": 100, "completion_tokens": 7}),
    ):
        suite = tmp_path / name / "full-real"
        suite.mkdir(parents=True)
        raw = {"dataset_split": "development", "raw_records": {
            "B0": [record("c1", "B0", "pass", DEFAULT_RUN_CONFIG, usage=usage)],
            "B1": [record("c1", "B1", "pass", config, usage=usage), record("c2", "B1", "fail", config, usage=usage, codes=("native_tool_call_missing",))],
        }}
        (suite / "stateful-real-raw.json").write_text(json.dumps(raw), encoding="utf-8")
        roots.append(tmp_path / name)

    summary = module["summarize_roots"](roots)
    json_run, native_run = (run["suites"]["full-real"] for run in summary["runs"])
    assert (json_run["model_protocols"], native_run["model_protocols"]) == (["json"], ["native"])
    assert native_run["profiles"]["B1"] == {
        "cases": 2, "passes": 1, "critical_cases": 1, "critical_passes": 1,
        "model_calls": 4, "tool_calls": 2, "prompt_tokens": 200, "completion_tokens": 14, "elapsed_ms": 20,
    }
    assert json_run["profiles"]["B1"]["prompt_tokens"] == "unavailable"
    c2 = next(item for item in summary["cases"] if item["case_id"] == "c2")
    assert c2["parse_error_codes"] == [["native_tool_call_missing"], ["native_tool_call_missing"]]
    assert summary["consistent_candidate"] is True
    assert "MODEL TEXT" not in json.dumps(summary)


def test_summary_takes_protocol_and_tokens_from_the_model_calls(tmp_path) -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "eval_multi_run_summary.py"))
    native_versions = {key: NATIVE.as_dict()[key] for key in ("prompt_version", "action_schema_version", "tool_description_version")}

    def call(status="succeeded", usage=(1000, 50)):
        tokens = None if usage is None else {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}
        return {"kind": "model_call", "status": status, "usage": tokens, **native_versions}

    def record(case_id, profile, *, config=None, events=(), calls=0, usage=None):
        return {
            "case_id": case_id, "profile": profile, "split": "development", "critical_question_id": None,
            "judged_status": "pass", "error_code": None, "terminal_state": "SUCCEEDED",
            "execution_metrics": {"model_calls": calls, "tool_calls": 0}, "elapsed_ms": 1,
            "execution_events": list(events), **({"run_config": config.as_dict()} if config else {}), **({"usage": usage} if usage else {}),
        }

    def summarize(b1_records):
        suite = tmp_path / str(len(list(tmp_path.iterdir()))) / "full-real"
        suite.mkdir(parents=True)
        # The real B0 shape: a model call, but no run_config and no events.
        b0 = record("b0", "B0", calls=1, usage={"usage_status": "known", "prompt_tokens": 300, "completion_tokens": 20})
        raw = {"dataset_split": "development", "raw_records": {"B0": [b0], "B1": b1_records}}
        (suite / "stateful-real-raw.json").write_text(json.dumps(raw), encoding="utf-8")
        return module["summarize_roots"]([suite.parent])["runs"][0]["suites"]["full-real"]

    b1 = [
        record("query", "B1", config=NATIVE, events=[call(), call(status="failed", usage=None), call(usage=(2000, 70))], calls=3),
        record("resumed", "B1", events=[call(usage=(500, 9))], calls=1),  # no run_config: the events decide
        record("fixture", "B1"),  # no model call: no protocol, no usage, adds nothing
    ]
    suite = summarize(b1)
    assert suite["model_protocols"] == ["native"]
    assert suite["prompt_versions_by_protocol"] == {"native": [NATIVE.prompt_version]}
    assert (suite["profiles"]["B1"]["prompt_tokens"], suite["profiles"]["B1"]["completion_tokens"]) == (3500, 129)
    assert (suite["profiles"]["B0"]["prompt_tokens"], suite["profiles"]["B0"]["completion_tokens"]) == (300, 20)

    b1[1] = record("resumed", "B1", events=[call(usage=None)], calls=1)  # a succeeded call without usage
    assert summarize(b1)["profiles"]["B1"]["prompt_tokens"] == "unavailable"

    # A record whose only call failed: the provider returned no usage, so it adds 0.
    only_failed = [record("timed-out", "B1", config=NATIVE, events=[call(status="failed", usage=None)], calls=1)]
    totals = summarize(only_failed)["profiles"]["B1"]
    assert (totals["prompt_tokens"], totals["completion_tokens"]) == (0, 0)
