"""A proposal whose JSON escapes decode to a lone surrogate fails validation; the run does not crash.

``\\ud800`` is valid JSON syntax, but the string it decodes to has no UTF-8 form: hashing,
storing or sending it raises ``UnicodeEncodeError``.  The model's text itself is plain ASCII,
so the call record is fine; the parser rejects the decoded proposal with ``invalid_json``,
the same code as any other content that is not strict JSON.  Both protocols and B0 share
that parser.
"""

from __future__ import annotations

import json

import pytest

from agent_core_scenarios import B0_WINDOW, NATIVE, agent_for, fixture_tools
import test_clarification as clarification_cases
from queryshield.agent.proposals import ProposalParseError, native_action_text, parse_query_proposal
from queryshield.agent.runtime import run_b0_single_pass
from queryshield.providers.contracts import ModelCallResult, ModelUsage, NativeToolCall

QUESTION = "2026年9月已支付订单总额是多少？"
LONE = "\\ud800"  # the six ASCII characters of a JSON escape
SQL_WITH_LONE = f'"SELECT SUM(o.amount_fen) AS gross_fen FROM orders AS o {LONE}"'
JSON_TEXT = (
    '{"type":"tool_call","name":"query_readonly","arguments":{"sql":' + SQL_WITH_LONE
    + ',"params":{},"metrics":["gross_fen"]}}'
)
NATIVE_ARGUMENTS = '{"sql":' + SQL_WITH_LONE + ',"params":{},"metrics":["gross_fen"]}'
USAGE = ModelUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


class _OneReply:
    mode = "fake"

    def __init__(self, content: str, tool_calls: tuple[NativeToolCall, ...] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls

    def complete(self, messages, *, request_id=None, model_call_id=None, run_id=None, tools=None):
        return ModelCallResult(
            mode="fake",
            provider="surrogate-scripted",
            model="surrogate-scripted-v1",
            request_id=request_id,
            model_call_id=model_call_id,
            provider_call_id="call-1",
            provider_request_id="request-1",
            content=self.content,
            usage=USAGE,
            usage_status="known",
            tool_calls=self.tool_calls,
            finish_reason="tool_calls" if self.tool_calls is not None else None,
        )


def _assert_validation_failure(result: dict) -> None:
    assert (result["status"], result["error_code"]) == ("failed", "invalid_json")
    kinds = [event["kind"] for event in result["events"]]
    assert kinds.count("model_call") == 1 and kinds[-1] == "proposal_validation"
    assert result["events"][-1]["error_detail"] == "model content has a lone surrogate escape"
    assert result["usage_summary"]["status"] == "known" and result["usage_summary"]["total_tokens"] == 15
    json.dumps(result, ensure_ascii=False).encode("utf-8")  # the whole result can be stored


def test_json_protocol() -> None:
    tools, executor = fixture_tools()
    result = agent_for(_OneReply(JSON_TEXT), tools).run(
        clarification_cases._context(), QUESTION, request_time_window=clarification_cases.SEPTEMBER
    )
    _assert_validation_failure(result.as_dict())
    assert executor.executed == []


def test_native_protocol() -> None:
    tools, executor = fixture_tools()
    reply = _OneReply("", (NativeToolCall("call-1", "query_readonly", NATIVE_ARGUMENTS),))
    result = agent_for(reply, tools, config=NATIVE).run(
        clarification_cases._context(), QUESTION, request_time_window=clarification_cases.SEPTEMBER
    )
    _assert_validation_failure(result.as_dict())
    assert executor.executed == []


def test_b0() -> None:
    tools, executor = fixture_tools()
    record = run_b0_single_pass(
        _OneReply(JSON_TEXT), tools, clarification_cases._context(), QUESTION, time_window=B0_WINDOW
    )
    assert (record["status"], record["error_code"]) == ("failed", "invalid_json")
    assert record["usage"]["usage_status"] == "known"
    assert executor.executed == []


@pytest.mark.parametrize("text", ['{"type":"deny","reason":"' + LONE + '"}', '{"' + LONE + '":1}', '{"type":"x' + LONE + '"}'])
def test_the_parser_rejects_it_in_any_string(text) -> None:
    with pytest.raises(ProposalParseError) as caught:
        parse_query_proposal(text, context=clarification_cases._context(), model_call_id="local-call")
    assert caught.value.code == "invalid_json"


def test_a_native_call_is_rejected_before_it_is_rewritten() -> None:
    with pytest.raises(ProposalParseError) as caught:
        native_action_text((NativeToolCall("call-1", "query_readonly", NATIVE_ARGUMENTS),))
    assert caught.value.code == "invalid_json"


def test_a_surrogate_pair_is_an_ordinary_character() -> None:
    text = '{"type":"deny","reason":"\\ud83d\\ude00"}'
    proposal = parse_query_proposal(text, context=clarification_cases._context(), model_call_id="local-call")
    assert proposal.action.reason == "\U0001f600"
