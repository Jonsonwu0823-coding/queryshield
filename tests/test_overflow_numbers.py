"""A number that overflows to infinity is rejected as not strict JSON, in both protocols."""

from __future__ import annotations

import json

import pytest

from queryshield.agent.proposals import (
    ExecutionContext,
    ProposalParseError,
    native_action_text,
    parse_query_proposal,
)
from queryshield.providers.contracts import NativeToolCall

from test_native_runtime import NATIVE, CONTEXT as RUN_CONTEXT, _Native, _agent, _events

CONTEXT = ExecutionContext(run_id="run-A-1", tenant_id="tenant-A", principal_id="principal-A", role="requester")
QUERY = '{"sql":"SELECT 1","params":{"0":1e400}}'
SEARCH = '{"query":"paid orders","top_k":1e400}'
NEGATIVE_QUERY = '{"sql":"SELECT 1","params":{"0":-1e400}}'
NEGATIVE_SEARCH = '{"query":"paid orders","top_k":-1e400}'
CASES = [
    ("query_readonly", QUERY),
    ("search_catalog", SEARCH),
    ("query_readonly", NEGATIVE_QUERY),
    ("search_catalog", NEGATIVE_SEARCH),
]
IDS = ["query", "search", "negative-query", "negative-search"]


def _json_mode_error(name: str, arguments: str) -> ProposalParseError:
    raw = f'{{"type":"tool_call","name":"{name}","arguments":{arguments}}}'
    with pytest.raises(ProposalParseError) as caught:
        parse_query_proposal(raw, context=CONTEXT, model_call_id="local-call-1")
    return caught.value


def _native_error(name: str, arguments: str) -> ProposalParseError:
    with pytest.raises(ProposalParseError) as caught:
        parse_query_proposal(
            native_action_text([NativeToolCall(id="call-1", name=name, arguments=arguments)]),
            context=CONTEXT,
            model_call_id="local-call-1",
        )
    return caught.value


@pytest.mark.parametrize("name, arguments", CASES, ids=IDS)
def test_json_mode_rejects_an_overflowing_number_as_invalid_json(name, arguments) -> None:
    error = _json_mode_error(name, arguments)
    assert (error.code, str(error)) == ("invalid_json", "invalid_json: model content is not strict JSON")


@pytest.mark.parametrize("name, arguments", CASES, ids=IDS)
def test_native_mode_rejects_it_while_converting_the_function_call(name, arguments) -> None:
    with pytest.raises(ProposalParseError) as caught:
        native_action_text([NativeToolCall(id="call-1", name=name, arguments=arguments)])
    assert caught.value.code == "invalid_json"


@pytest.mark.parametrize("name, arguments", CASES, ids=IDS)
def test_both_protocols_give_the_same_error_for_the_same_input(name, arguments) -> None:
    assert _json_mode_error(name, arguments).code == _native_error(name, arguments).code == "invalid_json"


def test_a_finite_float_and_a_tiny_float_are_still_read() -> None:
    raw = json.dumps({"type": "tool_call", "name": "search_catalog", "arguments": {"query": "x", "top_k": 3}})
    assert parse_query_proposal(raw, context=CONTEXT, model_call_id="local-call-1").action.arguments["top_k"] == 3
    assert parse_query_proposal(
        '{"type":"tool_call","name":"query_readonly","arguments":{"sql":"SELECT 1","params":{"0":1e-400,"1":1.5,"2":-2.5e10}}}',
        context=CONTEXT,
        model_call_id="local-call-1",
    ).action.arguments["params"] == {"0": 0.0, "1": 1.5, "2": -2.5e10}


def test_a_native_run_stops_on_the_overflowing_number_and_records_the_shape_of_the_content_only() -> None:
    call = NativeToolCall("a", "query_readonly", QUERY)
    result = _agent(_Native([("", (call,))]), NATIVE).run(RUN_CONTEXT, "查询客户姓名")
    assert (result.status, result.error_code, result.model_call_count) == ("failed", "invalid_json", 1)
    (event,) = _events(result, "proposal_validation")
    assert (event["error_code"], event["action_shape"]) == ("invalid_json", {"json": "invalid"})
