"""Native function calling at the provider boundary and the protocol conversion.

The adapter only carries the call; the conversion turns the one native call
into the json action it stands for, and the unchanged json parser validates it.
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys

import httpx
import pytest

from queryshield.agent import proposals
from queryshield.agent.context import native_tools
from queryshield.agent.proposals import (
    ExecutionContext,
    ProposalParseError,
    native_action_text,
    native_call_summary,
    parse_query_proposal,
)
from queryshield.mcp_metadata.schemas import tool_definitions
from queryshield.policy import argument_limits as limits
from queryshield.providers.contracts import ModelCallResult, ModelProviderError, NativeToolCall, native_call_for
from queryshield.providers.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleModel


CONTEXT = ExecutionContext(run_id="run-c1", tenant_id="A", principal_id="principal-A", role="requester")
CONFIG = OpenAICompatibleConfig(base_url="https://example.test/v1", api_key="test-secret", model="demo-model")
TOOLS = native_tools(retrieval_available=True)
QUERY_ARGUMENTS = {
    "sql": "SELECT COUNT(*) AS paid_count FROM orders WHERE status = %s",
    "params": {"0": "paid"},
    "metrics": ["paid_count"],
    "time_window": {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"},
}


def _call(name: str, arguments: object) -> NativeToolCall:
    text = arguments if type(arguments) is str else json.dumps(arguments, ensure_ascii=False)
    return NativeToolCall("call-1", name, text)


def _native_parse(*calls: NativeToolCall):
    return parse_query_proposal(native_action_text(calls), context=CONTEXT, model_call_id="model-call-1")


def _json_parse(payload: dict[str, object]):
    return parse_query_proposal(json.dumps(payload, ensure_ascii=False), context=CONTEXT, model_call_id="model-call-1")


# --- Adapter ---------------------------------------------------------------


def _complete(message: dict[str, object], *, tools=TOOLS, finish_reason: str | None = "tool_calls"):
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        choice: dict[str, object] = {"message": message}
        if finish_reason is not None:
            choice["finish_reason"] = finish_reason
        return httpx.Response(200, request=request, json={"id": "provider-call-1", "model": "demo-model", "choices": [choice]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = OpenAICompatibleModel(CONFIG, client=client).complete(
            [{"role": "user", "content": "q"}], request_id="r-1", model_call_id="m-1", tools=tools
        )
    return result, captured["payload"]


def _tool_message(*calls: tuple[str, str], content: object = "") -> dict[str, object]:
    return {
        "content": content,
        "tool_calls": [
            {"id": f"call-{index}", "type": "function", "function": {"name": name, "arguments": arguments}}
            for index, (name, arguments) in enumerate(calls)
        ],
    }


def test_native_request_sends_functions_auto_choice_and_no_parallel_calls() -> None:
    result, payload = _complete(_tool_message(("deny", '{"reason":"x"}')))
    assert payload["tools"] == TOOLS
    assert payload["tool_choice"] == "auto" and payload["parallel_tool_calls"] is False
    assert "strict" not in json.dumps(payload) and "enable_thinking" not in payload
    assert result.tool_calls == (NativeToolCall("call-0", "deny", '{"reason":"x"}'),)
    assert result.finish_reason == "tool_calls" and result.content == ""


def test_json_request_keeps_exactly_the_five_original_fields() -> None:
    result, payload = _complete({"content": '{"type":"deny","reason":"x"}'}, tools=None, finish_reason="stop")
    assert list(payload) == ["model", "messages", "temperature", "max_tokens", "stream"]
    assert result.tool_calls is None and result.finish_reason is None
    assert set(result.to_redacted_record()) == set(
        ModelCallResult(**{**_fields(result), "tool_calls": None}).to_redacted_record()
    )
    assert "finish_reason" not in result.to_redacted_record()


def _fields(result: ModelCallResult) -> dict[str, object]:
    return {name: getattr(result, name) for name in result.__dataclass_fields__}


@pytest.mark.parametrize("content", [None, "", "I will call a function."])
def test_native_reply_accepts_null_empty_or_text_content(content) -> None:
    result, _ = _complete(_tool_message(("deny", '{"reason":"x"}'), content=content))
    assert len(result.tool_calls) == 1
    record = result.to_redacted_record()
    assert record["tool_call_count"] == 1 and record["finish_reason"] == "tool_calls"


def test_native_reply_without_a_call_and_with_two_calls_are_passed_through() -> None:
    text_only, _ = _complete({"content": "just text"}, finish_reason="stop")
    assert text_only.tool_calls == () and text_only.content == "just text"
    two, _ = _complete(_tool_message(("deny", "{}"), ("deny", "{}")))
    assert len(two.tool_calls) == 2


def test_native_reply_keeps_malformed_arguments_and_a_length_finish_reason() -> None:
    result, _ = _complete(_tool_message(("query_readonly", '{"sql": "SELECT')), finish_reason="length")
    assert result.tool_calls[0].arguments == '{"sql": "SELECT' and result.finish_reason == "length"
    with pytest.raises(ProposalParseError) as caught:
        native_action_text(result.tool_calls)
    assert caught.value.code == "invalid_json"


@pytest.mark.parametrize(
    "message",
    [
        {"content": "", "tool_calls": {"id": "x"}},
        {"content": "", "tool_calls": [{"id": "x", "type": "function"}]},
        {"content": "", "tool_calls": [{"id": "x", "function": {"name": "deny", "arguments": {"reason": "x"}}}]},
        {"content": "", "tool_calls": [{"function": {"name": "deny", "arguments": "{}"}}]},
        {"content": 7, "tool_calls": []},
    ],
)
def test_malformed_native_reply_is_invalid_response(message) -> None:
    with pytest.raises(ModelProviderError) as caught:
        _complete(message)
    assert caught.value.code == "invalid_response"


def test_json_mode_still_rejects_null_content() -> None:
    with pytest.raises(ModelProviderError) as caught:
        _complete(_tool_message(("deny", "{}"), content=None), tools=None)
    assert caught.value.code == "invalid_response"


# --- Conversion -------------------------------------------------------------


@pytest.mark.parametrize(
    "name, arguments, json_payload",
    [
        ("search_catalog", {"query": "退款后净额"}, {"type": "tool_call", "name": "search_catalog", "arguments": {"query": "退款后净额"}}),
        ("describe_tables", {"tables": ["orders"]}, {"type": "tool_call", "name": "describe_tables", "arguments": {"tables": ["orders"]}}),
        ("query_readonly", QUERY_ARGUMENTS, {"type": "tool_call", "name": "query_readonly", "arguments": QUERY_ARGUMENTS}),
        ("ask_user", {"question": "哪个月？"}, {"type": "ask_user", "question": "哪个月？"}),
        (
            "final_answer",
            {"answer": "", "source_ids": [], "fact_refs": [], "basis": "no_data"},
            {"type": "final_answer", "answer": "", "source_ids": [], "fact_refs": [], "basis": "no_data"},
        ),
        ("deny", {"reason": "out of scope"}, {"type": "deny", "reason": "out of scope"}),
    ],
)
def test_each_function_becomes_the_same_action_as_its_json(name, arguments, json_payload) -> None:
    native = _native_parse(_call(name, arguments))
    assert native.action == _json_parse(json_payload).action
    assert native_call_for(json_payload) == (name, arguments)


def test_a_lone_surrogate_escape_parses_the_same_in_both_protocols() -> None:
    # The converted text must stay encodable: json mode accepts this escape as a deny reason.
    native = _native_parse(NativeToolCall("c", "deny", '{"reason":"\\ud800"}'))
    json_mode = parse_query_proposal('{"type":"deny","reason":"\\ud800"}', context=CONTEXT, model_call_id="model-call-1")
    assert type(native.action).__name__ == "DenyAction" and native.action == json_mode.action


def test_redacted_record_never_carries_the_call_text() -> None:
    call = NativeToolCall("call-1", "secret-function-name", '{"reason":"secret-argument-text"}')
    result = ModelCallResult(
        mode="real", provider="p", model="m", request_id="r", model_call_id="c", provider_call_id="pc",
        provider_request_id=None, content="", usage=None, usage_status="unknown", tool_calls=(call,), finish_reason="tool_calls",
    )
    record = result.to_redacted_record()
    text = json.dumps(record)
    assert "secret-function-name" not in text and "secret-argument-text" not in text
    assert (record["finish_reason"], record["tool_call_count"]) == ("tool_calls", 1)


@pytest.mark.parametrize(
    "calls, code",
    [
        ((), "native_tool_call_missing"),
        ((_call("deny", {"reason": "a"}), _call("deny", {"reason": "b"})), "native_multiple_tool_calls"),
        ((_call("drop_table", {}),), "unknown_action"),
        ((_call("tool_call", {"name": "query_readonly"}),), "unknown_action"),
        ((_call("parallel_readonly", {"metric_ids": ["gross_fen", "net_fen"]}),), "unknown_action"),
        ((_call("deny", '["reason"]'),), "invalid_shape"),
        ((_call("deny", '{"reason":"a","reason":"b"}'),), "duplicate_field"),
        ((_call("search_catalog", '{"query":"q","top_k":NaN}'),), "invalid_json"),
        ((_call("deny", ""),), "invalid_json"),
        ((_call("deny", {"reason": "a", "extra": 1}),), "unknown_field"),
        ((_call("ask_user", {"type": "tool_call", "question": "q"}),), "unknown_field"),
        ((_call("final_answer", {"type": "deny", "answer": "a", "source_ids": [], "fact_refs": []}),), "unknown_field"),
        ((_call("deny", {"type": "deny", "reason": "a"}),), "unknown_field"),
    ],
)
def test_conversion_errors_map_to_fixed_codes(calls, code) -> None:
    with pytest.raises(ProposalParseError) as caught:
        _native_parse(*calls)
    assert caught.value.code == code


# One invalid argument, sent once as a json action and once as a native call:
# the same validator must reject both with the same code.
_INVALID = [
    ("search_catalog", {"query": "q", "top_k": limits.TOP_K_RANGE[1] + 1}),
    ("search_catalog", {"query": "x" * (limits.SEARCH_QUERY_MAX_CHARS + 1)}),
    ("search_catalog", {"top_k": 3}),
    ("describe_tables", {"tables": ["payments"]}),
    ("describe_tables", {"tables": ["orders"] * 2}),
    ("query_readonly", {"sql": "SELECT 1", "params": {"tenant_id": "B"}}),
    ("query_readonly", {"sql": "SELECT 1", "params": {"0": ["x"]}}),
    ("query_readonly", {"sql": "SELECT 1"}),
    ("query_readonly", {**QUERY_ARGUMENTS, "metrics": "paid_count"}),
    ("ask_user", {"question": "q" * (limits.QUESTION_MAX_CHARS + 1)}),
    ("ask_user", {"question": "q", "clarification_id": "c" * (limits.CLARIFICATION_ID_MAX_CHARS + 1)}),
    ("final_answer", {"answer": "a", "source_ids": [], "fact_refs": [{"result_id": "r"}]}),
    ("final_answer", {"answer": "a", "source_ids": [], "fact_refs": [], "basis": "guess"}),
    ("final_answer", {"answer": " ", "source_ids": [], "fact_refs": []}),
    ("deny", {"reason": "r" * (limits.REASON_MAX_CHARS + 1)}),
]


def _json_form(name: str, arguments: dict[str, object]) -> dict[str, object]:
    if name in proposals.TOOL_NAMES:
        return {"type": "tool_call", "name": name, "arguments": arguments}
    return {"type": name, **arguments}


@pytest.mark.parametrize("name, arguments", _INVALID)
def test_an_invalid_argument_is_rejected_the_same_way_in_both_protocols(name, arguments) -> None:
    with pytest.raises(ProposalParseError) as json_error:
        _json_parse(_json_form(name, arguments))
    with pytest.raises(ProposalParseError) as native_error:
        _native_parse(_call(name, arguments))
    assert native_error.value.code == json_error.value.code
    assert str(native_error.value) == str(json_error.value)


def test_identity_in_native_arguments_is_rejected() -> None:
    for key in proposals.RESERVED_IDENTITY_PARAMS:
        with pytest.raises(ProposalParseError) as caught:
            _native_parse(_call("query_readonly", {"sql": "SELECT 1", "params": {key: "B"}}))
        assert caught.value.code == "reserved_parameter"


def test_call_summary_names_only_offered_functions_and_never_the_arguments() -> None:
    summary = native_call_summary((_call("deny", {"reason": "secret words"}), NativeToolCall("c", "ignore all rules", "{}")))
    assert [item["name"] for item in summary] == ["deny", "<other>"]
    assert "secret words" not in json.dumps(summary) and "ignore all rules" not in json.dumps(summary)
    assert summary[0]["arguments_length"] == len('{"reason": "secret words"}')


# --- Schema and validator share one set of limits ----------------------------


def _parameters(name: str) -> dict:
    return next(tool["function"]["parameters"] for tool in TOOLS if tool["function"]["name"] == name)


# (function, path into its parameters, the validator constant, an argument builder at a given limit)
_LIMITS = [
    ("search_catalog", ("properties", "query", "maxLength"), limits.SEARCH_QUERY_MAX_CHARS, lambda n: {"query": "q" * n}),
    ("search_catalog", ("properties", "top_k", "maximum"), limits.TOP_K_RANGE[1], lambda n: {"query": "q", "top_k": n}),
    ("search_catalog", ("properties", "top_k", "minimum"), limits.TOP_K_RANGE[0], None),
    ("search_catalog", ("properties", "top_k", "default"), limits.TOP_K_DEFAULT, None),
    ("describe_tables", ("properties", "tables", "maxItems"), limits.DESCRIBE_TABLES_RANGE[1], lambda n: {"tables": [*sorted(proposals.ALLOWED_TABLES), "extra"][:n]}),
    ("describe_tables", ("properties", "tables", "minItems"), limits.DESCRIBE_TABLES_RANGE[0], None),
    ("describe_tables", ("properties", "tables", "items", "enum"), sorted(proposals.ALLOWED_TABLES), None),
    ("query_readonly", ("properties", "sql", "maxLength"), limits.SQL_MAX_CHARS, lambda n: {"sql": "S" * n, "params": {}}),
    ("ask_user", ("properties", "question", "maxLength"), limits.QUESTION_MAX_CHARS, lambda n: {"question": "q" * n}),
    ("ask_user", ("properties", "clarification_id", "maxLength"), limits.CLARIFICATION_ID_MAX_CHARS, lambda n: {"question": "q", "clarification_id": "c" * n}),
    ("final_answer", ("properties", "answer", "maxLength"), limits.ANSWER_MAX_CHARS, lambda n: {"answer": "a" * n, "source_ids": [], "fact_refs": []}),
    ("final_answer", ("properties", "source_ids", "maxItems"), limits.MAX_SOURCE_IDS, lambda n: {"answer": "a", "source_ids": [f"s{i}" for i in range(n)], "fact_refs": []}),
    ("final_answer", ("properties", "source_ids", "items", "maxLength"), limits.LIST_ITEM_MAX_CHARS, lambda n: {"answer": "a", "source_ids": ["s" * n], "fact_refs": []}),
    ("final_answer", ("properties", "fact_refs", "maxItems"), limits.MAX_FACT_REFS, lambda n: {"answer": "a", "source_ids": [], "fact_refs": [{"result_id": f"r{i}", "metric_id": "m"} for i in range(n)]}),
    ("final_answer", ("properties", "basis", "enum"), list(proposals.ANSWER_BASES), None),
    ("deny", ("properties", "reason", "maxLength"), limits.REASON_MAX_CHARS, lambda n: {"reason": "r" * n}),
]


@pytest.mark.parametrize("name, path, constant, build", _LIMITS, ids=[f"{item[0]}.{'.'.join(item[1][1:])}" for item in _LIMITS])
def test_schema_limit_is_the_validator_constant(name, path, constant, build) -> None:
    value = _parameters(name)
    for key in path:
        value = value[key]
    assert value == constant
    if build is None or type(constant) is not int:
        return
    # The validator accepts the limit itself and rejects one past it.
    _native_parse(_call(name, build(constant)))
    with pytest.raises(ProposalParseError):
        _native_parse(_call(name, build(constant + 1)))


def _objects(schema: object):
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            yield schema
        for value in schema.values():
            yield from _objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _objects(value)


@pytest.mark.parametrize("name", proposals.NATIVE_FUNCTION_NAMES)
def test_schema_required_fields_match_the_validator(name) -> None:
    schema = _parameters(name)
    assert all(item["additionalProperties"] is False for item in _objects(schema))
    complete = {
        "search_catalog": {"query": "q"},
        "describe_tables": {"tables": ["orders"]},
        "query_readonly": {"sql": "SELECT 1", "params": {}},
        "ask_user": {"question": "q"},
        "final_answer": {"answer": "a", "source_ids": [], "fact_refs": []},
        "deny": {"reason": "r"},
    }[name]
    assert sorted(schema["required"]) == sorted(complete)
    _native_parse(_call(name, complete))
    for field in schema["required"]:
        with pytest.raises(ProposalParseError) as caught:
            _native_parse(_call(name, {key: value for key, value in complete.items() if key != field}))
        assert caught.value.code == "missing_field"


def test_functions_are_the_offered_names_and_search_catalog_needs_a_retriever() -> None:
    assert [tool["function"]["name"] for tool in TOOLS] == list(proposals.NATIVE_FUNCTION_NAMES)
    without = [tool["function"]["name"] for tool in native_tools(retrieval_available=False)]
    assert without == [name for name in proposals.NATIVE_FUNCTION_NAMES if name != "search_catalog"]


def test_mcp_definitions_are_unchanged_from_the_baseline_literals() -> None:
    # Literal values of mcp_metadata/schemas.py before the limits became named constants (30c0fc1).
    inputs = {item["name"]: item["inputSchema"] for item in tool_definitions()}
    assert inputs["search_catalog"] == {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 200},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    assert inputs["describe_tables"] == {
        "type": "object",
        "properties": {
            "tables": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1, "maxItems": 3, "uniqueItems": True},
        },
        "required": ["tables"],
        "additionalProperties": False,
    }


# --- Dependency direction ------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "queryshield.policy.argument_limits",
        "queryshield.mcp_metadata.schemas",
        "queryshield.providers.fake_model",
        "queryshield.providers.openai_compatible",
    ],
)
def test_limits_mcp_schemas_and_providers_never_load_the_agent_package(module) -> None:
    # A fresh interpreter: an import cycle or a providers -> agent dependency shows up here.
    code = f"import sys, {module}; print(sorted(name for name in sys.modules if name.startswith('queryshield.agent')))"
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "[]"


def test_native_tools_has_no_function_level_import() -> None:
    body = ast.parse(inspect.getsource(native_tools))
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(body))
