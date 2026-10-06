"""The two evaluation recording wrappers keep a native call as the json action it stands for.

Under native function calling the provider's content is empty and the decision
is a function call; both wrappers must record the converted json action (the
same text the product parses), not the empty content.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from queryshield.evaluation.stateful_product import _RecordingEvaluationModel
from queryshield.providers.contracts import ModelCallResult, NativeToolCall
from scripts import check_eval


QUERY_ARGUMENTS = {"sql": "SELECT COUNT(*) AS paid_count FROM orders AS o", "params": {"0": "paid"}}


class _NativeDelegate:
    mode = "real"
    provider = "native-test"
    model = "native-model"

    def __init__(self, calls: tuple[NativeToolCall, ...], content: str = "") -> None:
        self.calls = calls
        self.content = content

    def complete(self, _messages, *, request_id=None, model_call_id=None, **_options):
        return ModelCallResult(
            mode="real", provider=self.provider, model=self.model, request_id=request_id, model_call_id=model_call_id,
            provider_call_id="provider-call", provider_request_id=None, content=self.content, usage=None,
            usage_status="unknown", tool_calls=self.calls, finish_reason="tool_calls",
        )


DENY = (NativeToolCall("c1", "deny", '{"reason":"不能查"}'),)
QUERY = (NativeToolCall("c1", "query_readonly", json.dumps(QUERY_ARGUMENTS)),)
DENY_TEXT = '{"type":"deny","reason":"\\u4e0d\\u80fd\\u67e5"}'
QUERY_TEXT = json.dumps({"type": "tool_call", "name": "query_readonly", "arguments": QUERY_ARGUMENTS}, separators=(",", ":"))


def _evaluation_record(calls, content=""):
    records: list[dict[str, object]] = []
    model = _RecordingEvaluationModel(_NativeDelegate(calls, content), records, {})
    model.complete([{"role": "user", "content": "q"}], request_id="request-1", model_call_id="call-1", tools=[])
    return records[0]


def _script_record(calls, content=""):
    adapter = check_eval._RecordingModelAdapter(_NativeDelegate(calls, content))
    adapter.complete([{"role": "user", "content": "q"}], request_id="request-1", model_call_id="call-1", tools=[])
    return adapter.records[0]


@pytest.mark.parametrize("calls, text, proposal_type", [(DENY, DENY_TEXT, "deny"), (QUERY, QUERY_TEXT, "tool_call")])
def test_evaluation_wrapper_records_the_converted_action(calls, text, proposal_type) -> None:
    record = _evaluation_record(calls)
    assert record["raw_content"] == text
    assert record["content_length"] == len(text.encode("utf-8"))
    assert record["content_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert record["proposal_type"] == proposal_type
    assert record["response_shape"]["action_type"] == proposal_type
    assert (record["finish_reason"], record["tool_call_count"]) == ("tool_calls", 1)


def test_evaluation_wrapper_keeps_the_content_when_no_action_converts() -> None:
    record = _evaluation_record((), content="plain text")
    assert record["raw_content"] == "plain text"
    assert "proposal_type" not in record
    assert (record["finish_reason"], record["tool_call_count"]) == ("tool_calls", 0)


def test_script_wrapper_records_the_converted_deny() -> None:
    record = _script_record(DENY)
    assert record["provider_output"] == DENY_TEXT
    assert record["proposal_type"] == "deny"
    assert record["response_shape"] == {
        "json_status": "valid", "payload_kind": "dict", "top_level_keys": ["reason", "type"], "action_type": "deny",
    }
    assert (record["finish_reason"], record["tool_call_count"]) == ("tool_calls", 1)


def test_script_wrapper_records_the_converted_query() -> None:
    record = _script_record(QUERY)
    assert record["provider_output"] == QUERY_TEXT
    assert "proposal_type" not in record
    assert record["proposal"] == {"type": "tool_call", "name": "query_readonly", **QUERY_ARGUMENTS}
    assert record["response_shape"]["action_name"] == "query_readonly"


def test_script_wrapper_keeps_the_content_when_no_action_converts() -> None:
    record = _script_record((NativeToolCall("a", "deny", "{}"), NativeToolCall("b", "deny", "{}")), content="text")
    assert record["provider_output"] == "text"
    assert record["response_shape"]["json_status"] == "invalid"


@pytest.mark.parametrize("protocol", ["json", "native"])
def test_only_a_real_model_follows_the_protocol_setting(monkeypatch, protocol) -> None:
    from dataclasses import replace

    from queryshield.agent.config import DEFAULT_RUN_CONFIG, NATIVE_VERSIONS
    from queryshield.evaluation.comparison import build_comparison_profiles
    from queryshield.evaluation.stateful_product import evaluation_run_config

    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", protocol)
    native = replace(DEFAULT_RUN_CONFIG, **NATIVE_VERSIONS)
    assert evaluation_run_config(DEFAULT_RUN_CONFIG, "fake") is DEFAULT_RUN_CONFIG
    assert evaluation_run_config(DEFAULT_RUN_CONFIG, "real") == (native if protocol == "native" else DEFAULT_RUN_CONFIG)
    for mode, expected in (("fake", DEFAULT_RUN_CONFIG), ("real", evaluation_run_config(DEFAULT_RUN_CONFIG, "real"))):
        b1 = build_comparison_profiles(".", provider_mode=mode, model_name="m")["profiles"][1]
        assert (b1["prompt_version"], b1["adapter_version"]) == (expected.prompt_version, expected.adapter_version)
