"""The fake upstream's own contract: both endpoints, both protocols, the usage rule, its fake marks."""

from __future__ import annotations

import json
import math
import re

from fastapi.testclient import TestClient
import pytest

from queryshield.agent.context import native_tools
from queryshield.knowledge.runtime import feature_vector
from queryshield.providers.fake_model import FakeModel
from scripts import fake_upstream

QUESTION = "2026年9月已支付订单总额是多少？"
CHAT = "/v1/chat/completions"
EMBEDDINGS = "/v1/embeddings"
HEX = re.compile(r"[0-9a-f]{32}")
# Chinese text in a description: the tools JSON is counted with ensure_ascii=False.
TOOLS = [{"type": "function", "function": {"name": "query_readonly", "description": "只读查询", "parameters": {"type": "object"}}}]


@pytest.fixture()
def client() -> TestClient:
    return TestClient(fake_upstream.app)


def _expected_tokens(text: str) -> int:
    """The documented rule, written independently of the server: 4 UTF-8 bytes per token, rounded up."""

    return math.ceil(len(text.encode("utf-8")) / 4)


def _chat(client: TestClient, messages: list[dict], **extra):
    return client.post(CHAT, json={"model": "any-name", "messages": messages, "temperature": 0, "stream": False, **extra})


def _messages(*contents: str) -> list[dict[str, str]]:
    return [{"role": "system" if index == 0 else "user", "content": text} for index, text in enumerate(contents)]


def test_json_protocol_returns_the_fake_models_text_with_finish_reason_stop(client) -> None:
    messages = _messages("system text", QUESTION)
    response = _chat(client, messages)
    assert response.status_code == 200
    body = response.json()
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": FakeModel().complete(messages).content}
    assert choice["finish_reason"] == "stop"
    assert body["object"] == "chat.completion" and type(body["created"]) is int


def test_native_protocol_returns_one_function_call_with_finish_reason_tool_calls(client) -> None:
    messages = _messages("system text", QUESTION)
    tools = native_tools(retrieval_available=False)
    body = _chat(client, messages, tools=tools, tool_choice="auto", parallel_tool_calls=False).json()
    expected = FakeModel().complete(messages, tools=tools).tool_calls
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    calls = choice["message"]["tool_calls"]
    assert len(calls) == len(expected) == 1
    assert set(calls[0]) == {"id", "type", "function"}
    assert calls[0]["type"] == "function"
    assert calls[0]["id"].startswith("call_") and HEX.fullmatch(calls[0]["id"][len("call_"):])
    assert calls[0]["function"] == {"name": expected[0].name, "arguments": expected[0].arguments}
    assert isinstance(calls[0]["function"]["arguments"], str)


def test_chat_usage_counts_each_message_and_the_tools_json_by_utf8_bytes_rounded_up(client) -> None:
    # 4 bytes, 5 bytes and 6 bytes (two Chinese characters): per piece 1 + 2 + 2.  Counting the
    # pieces together (15 bytes -> 4), characters instead of bytes (你好 -> 1), rounding down
    # or another divisor all give a different number.
    contents = ("abcd", "abcde", "你好")
    expected_prompt = sum(_expected_tokens(text) for text in contents)
    assert expected_prompt == 5

    plain = _chat(client, _messages(*contents)).json()
    content = plain["choices"][0]["message"]["content"]
    assert plain["usage"] == {
        "prompt_tokens": expected_prompt,
        "completion_tokens": _expected_tokens(content),
        "total_tokens": expected_prompt + _expected_tokens(content),
    }

    native = _chat(client, _messages(*contents), tools=TOOLS).json()
    function = native["choices"][0]["message"]["tool_calls"][0]["function"]
    tools_json = json.dumps(TOOLS, ensure_ascii=False, separators=(",", ":"))
    completion = _expected_tokens(function["name"]) + _expected_tokens(function["arguments"])
    assert native["usage"] == {
        "prompt_tokens": expected_prompt + _expected_tokens(tools_json),
        "completion_tokens": completion,
        "total_tokens": expected_prompt + _expected_tokens(tools_json) + completion,
    }


def test_every_response_is_marked_fake_and_has_a_new_id(client) -> None:
    first = _chat(client, _messages("s", QUESTION))
    second = _chat(client, _messages("s", QUESTION))
    embedded = client.post(EMBEDDINGS, json={"model": "m", "input": "x"})
    for response in (first, second, embedded):
        assert response.headers["X-Fake-Upstream"] == "qs-fake-upstream-v1"
        assert response.json()["model"] == "qs-fake-upstream-v1"
        assert re.fullmatch(r"fake-req-[0-9a-f]{32}", response.headers["x-request-id"])
    ids = [response.json()["id"] for response in (first, second)]
    assert all(re.fullmatch(r"fake-chatcmpl-[0-9a-f]{32}", value) for value in ids)
    assert ids[0] != ids[1], "the same request twice must not reuse an id"
    assert re.fullmatch(r"fake-embd-[0-9a-f]{32}", embedded.json()["id"])
    assert first.headers["x-request-id"] != second.headers["x-request-id"]


def test_errors_and_unknown_paths_are_marked_fake_too(client) -> None:
    for response in (client.post(CHAT, content=b"not json"), client.get("/v1/models"), client.get("/health")):
        assert response.headers["X-Fake-Upstream"] == "qs-fake-upstream-v1"
    assert client.get("/health").json() == {"status": "ok"}


def test_embeddings_are_the_fake_embedding_vectors_in_input_order(client) -> None:
    texts = ["退款后净额", "paid orders", "支付金额"]
    body = client.post(EMBEDDINGS, json={"model": "m", "input": texts, "dimensions": 128}).json()
    assert [item["index"] for item in body["data"]] == [0, 1, 2]
    for item, text in zip(body["data"], texts):
        assert item["object"] == "embedding"
        assert len(item["embedding"]) == 128
        assert tuple(item["embedding"]) == feature_vector(text)
    used = sum(_expected_tokens(text) for text in texts)
    assert body["usage"] == {"prompt_tokens": used, "total_tokens": used}
    single = client.post(EMBEDDINGS, json={"model": "m", "input": "支付金额"}).json()
    assert tuple(single["data"][0]["embedding"]) == feature_vector("支付金额")
    assert single["usage"] == {"prompt_tokens": 3, "total_tokens": 3}  # 12 bytes


SENTINEL = "SENTINEL-must-not-echo"
INVALID_CHAT = [
    (b"not json", "body"),
    (b"[]", "body"),
    ({"messages": _messages(QUESTION)}, "model"),
    ({"model": " ", "messages": _messages(QUESTION)}, "model"),
    ({"model": SENTINEL}, "messages"),
    ({"model": "m", "messages": []}, "messages"),
    ({"model": "m", "messages": SENTINEL}, "messages"),
    ({"model": "m", "messages": [{"role": "user"}]}, "messages"),
    ({"model": "m", "messages": [{"role": "user", "content": 1}]}, "messages"),
    ({"model": "m", "messages": [SENTINEL]}, "messages"),
    ({"model": "m", "messages": _messages(QUESTION), "tools": SENTINEL}, "tools"),
    ({"model": "m", "messages": _messages(QUESTION), "tools": []}, "tools"),
    ({"model": "m", "messages": _messages(QUESTION), "tools": [SENTINEL]}, "tools"),
    ({"model": "m", "messages": _messages(QUESTION), "tools": [{"type": "code", "function": {"name": "x"}}]}, "tools"),
    ({"model": "m", "messages": _messages(QUESTION), "tools": [{"type": "function", "function": {"name": 1}}]}, "tools"),
    ({"model": "m", "messages": _messages(QUESTION), "stream": True}, "stream"),
    ({"model": "m", "messages": _messages(QUESTION), "stream": None}, "stream"),
]
INVALID_EMBEDDINGS = [
    ({"input": "x"}, "model"),
    ({"model": "m"}, "input"),
    ({"model": "m", "input": ""}, "input"),
    ({"model": "m", "input": []}, "input"),
    ({"model": "m", "input": ["x", ""]}, "input"),
    ({"model": "m", "input": ["x", 1]}, "input"),
    ({"model": "m", "input": "x", "dimensions": 127}, "dimensions"),
    ({"model": "m", "input": "x", "dimensions": 129}, "dimensions"),
    ({"model": "m", "input": "x", "dimensions": 1024}, "dimensions"),
    ({"model": "m", "input": "x", "dimensions": "128"}, "dimensions"),
    ({"model": "m", "input": "x", "dimensions": 128.0}, "dimensions"),
    ({"model": "m", "input": "x", "dimensions": None}, "dimensions"),
]


@pytest.mark.parametrize(("path", "payload", "param"), [(CHAT, *case) for case in INVALID_CHAT] + [(EMBEDDINGS, *case) for case in INVALID_EMBEDDINGS])
def test_an_invalid_request_is_a_400_with_an_openai_error_body(client, path, payload, param) -> None:
    response = client.post(path, content=payload) if isinstance(payload, bytes) else client.post(path, json=payload)
    assert response.status_code == 400
    assert response.json() == {
        "error": {"message": f"invalid or missing field: {param}", "type": "invalid_request_error", "param": param, "code": "invalid_request"}
    }
    assert SENTINEL not in response.text
    assert response.headers["X-Fake-Upstream"] == "qs-fake-upstream-v1"


def test_the_authorization_header_changes_nothing(client) -> None:
    secret = "credential-value-never-read"
    payload = {"model": "m", "messages": _messages("s", QUESTION), "tools": TOOLS}
    answers = []
    for headers in ({}, {"Authorization": f"Bearer {secret}"}, {"Authorization": "garbage"}):
        response = client.post(CHAT, json=payload, headers=headers)
        assert response.status_code == 200 and secret not in response.text
        body = response.json()
        body.pop("id"), body.pop("created")
        for call in body["choices"][0]["message"]["tool_calls"]:
            call.pop("id")
        answers.append(body)
    assert answers[0] == answers[1] == answers[2]
