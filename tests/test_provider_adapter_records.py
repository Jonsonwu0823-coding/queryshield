"""What the three HTTP provider adapters send and record, branch by branch.

Every request field, every success and failure record (with key order) and
every error code is pinned here, so a refactor of the adapters can be shown
not to change anything a provider or an evidence reader sees.  The adapters
run only in real mode; these tests reach them through httpx.MockTransport.
"""

from __future__ import annotations

import json
import re

import httpx
import pytest

from queryshield.knowledge.index import IndexValidationError, build_embedding_index
from queryshield.knowledge.ingest import ChunkRecord, KnowledgeSnapshot, SourceRecord
from queryshield.providers.contracts import ModelProviderError, NativeToolCall
from queryshield.providers.embedding import (
    EmbeddingCallResult,
    EmbeddingConfig,
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    FixedEmbedding,
    OpenAICompatibleEmbedding,
    OperationUsage,
)
from queryshield.providers.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleModel
from queryshield.providers.rerank import (
    HttpRerankAdapter,
    RerankCandidate,
    RerankInputError,
    rerank_authorized_candidates,
)


MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "问题"}]
TOOLS = [{"type": "function", "function": {"name": "deny", "parameters": {"type": "object"}}}]
CHAT_CONFIG = OpenAICompatibleConfig(
    base_url="https://example.test/v1/", api_key="test-secret", model="demo-model", timeout_seconds=7.0, max_tokens=64
)
EMBED_CONFIG = EmbeddingConfig(
    base_url="https://example.test/v1", api_key="test-secret", model="embed-demo", model_revision="rev-1", dimensions=2
)


class _Recorder:
    """A MockTransport handler that keeps the one request and answers with a fixed response or error."""

    def __init__(self, response=None, *, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        status, body, headers = self.response
        if isinstance(body, bytes):
            return httpx.Response(status, request=request, content=body, headers=headers)
        return httpx.Response(status, request=request, json=body, headers=headers)


def _headers(request: httpx.Request, names: tuple[str, ...]) -> list[tuple[str, str]]:
    return [(name, request.headers[name]) for name in names if name in request.headers]


# --- Chat completions --------------------------------------------------------


def _chat(response=None, *, error=None, tools=None, config=CHAT_CONFIG):
    recorder = _Recorder(response, error=error)
    with httpx.Client(transport=httpx.MockTransport(recorder)) as client:
        try:
            result = OpenAICompatibleModel(config, client=client).complete(
                MESSAGES, request_id="local-request", model_call_id="server-call", tools=tools
            )
        except ModelProviderError as exc:
            return exc, recorder
    return result, recorder


def _chat_body(**overrides):
    body = {
        "id": " provider-call ",
        "model": "served-model",
        "choices": [{"message": {"content": " 回答 "}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    body.update(overrides)
    return body


def _failed_chat_record(error_code: str, **extra) -> list[tuple[str, object]]:
    record = [
        ("status", "failed"),
        ("mode", "real"),
        ("provider", "openai_compatible"),
        ("model", "demo-model"),
        ("request_id", "local-request"),
        ("model_call_id", "server-call"),
        ("provider_call_id", None),
        ("provider_request_id", extra.pop("provider_request_id", None)),
        ("stream", False),
        ("content_present", False),
        ("usage", None),
        ("usage_status", "unknown"),
        ("error_code", error_code),
    ]
    return record + list(extra.items())


def test_chat_request_url_headers_and_payload_are_fixed() -> None:
    _result, recorder = _chat((200, _chat_body(), {}))
    (request,) = recorder.requests
    assert request.method == "POST"
    assert str(request.url) == "https://example.test/v1/chat/completions"
    assert _headers(request, ("accept", "authorization", "content-type", "x-client-request-id")) == [
        ("accept", "application/json"),
        ("authorization", "Bearer test-secret"),
        ("content-type", "application/json"),
        ("x-client-request-id", "local-request"),
    ]
    assert json.loads(request.content) == {
        "model": "demo-model",
        "messages": MESSAGES,
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
    }
    assert list(json.loads(request.content)) == ["model", "messages", "temperature", "max_tokens", "stream"]
    assert request.extensions["timeout"] == {"connect": 7.0, "read": 7.0, "write": 7.0, "pool": 7.0}


def test_chat_request_body_bytes_and_header_order_are_fixed() -> None:
    _result, recorder = _chat((200, _chat_body(), {}))
    (request,) = recorder.requests
    assert request.content == (
        b'{"model":"demo-model","messages":[{"role":"system","content":"s"},'
        b'{"role":"user","content":"\xe9\x97\xae\xe9\xa2\x98"}],"temperature":0,"max_tokens":64,"stream":false}'
    )
    ours = (b"Accept", b"Authorization", b"Content-Type", b"X-Client-Request-Id")
    assert [(name, value) for name, value in request.headers.raw if name in ours] == [
        (b"Accept", b"application/json"),
        (b"Authorization", b"Bearer test-secret"),
        (b"Content-Type", b"application/json"),
        (b"X-Client-Request-Id", b"local-request"),
    ]


def test_adapters_without_an_injected_client_use_the_configured_timeout(monkeypatch) -> None:
    # The product passes no client: each call opens its own, and its timeout must be the configured one.
    recorder = _Recorder((200, _chat_body(), {}))
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **options: real_client(transport=httpx.MockTransport(recorder), **options))
    OpenAICompatibleModel(CHAT_CONFIG).complete(MESSAGES, request_id="r", model_call_id="m")
    recorder.response = (200, _embed_body(), {})
    OpenAICompatibleEmbedding(EMBED_CONFIG).embed(["a", "b"], request_id="r", model_call_id="m")
    assert [request.extensions["timeout"]["read"] for request in recorder.requests] == [7.0, 30.0]


def test_chat_native_request_adds_tools_after_the_json_fields() -> None:
    body = _chat_body(choices=[{"message": {"content": None, "tool_calls": []}, "finish_reason": "stop"}])
    _result, recorder = _chat((200, body, {}), tools=TOOLS)
    payload = json.loads(recorder.requests[0].content)
    assert list(payload) == [
        "model", "messages", "temperature", "max_tokens", "stream", "tools", "tool_choice", "parallel_tool_calls"
    ]
    assert payload["tools"] == TOOLS and payload["tool_choice"] == "auto" and payload["parallel_tool_calls"] is False


def test_chat_endpoint_keeps_a_full_chat_completions_url() -> None:
    config = OpenAICompatibleConfig(base_url="https://example.test/v1/chat/completions", api_key="k", model="m")
    assert config.endpoint == "https://example.test/v1/chat/completions"
    assert OpenAICompatibleConfig(base_url="http://example.test", api_key="k", model="m").endpoint == (
        "http://example.test/chat/completions"
    )


def test_chat_json_success_result_and_record() -> None:
    result, _ = _chat((200, _chat_body(), {"x-request-id": "  provider-request  "}))
    assert (result.provider_call_id, result.content, result.model) == ("provider-call", "回答", "served-model")
    assert (result.tool_calls, result.finish_reason) == (None, None)
    assert list(result.to_redacted_record().items()) == [
        ("status", "succeeded"),
        ("mode", "real"),
        ("provider", "openai_compatible"),
        ("model", "served-model"),
        ("request_id", "local-request"),
        ("model_call_id", "server-call"),
        ("provider_call_id", "provider-call"),
        ("provider_request_id", "provider-request"),
        ("stream", False),
        ("content_present", True),
        ("content_length", 6),
        ("content_sha256", "f3f512648742020fbd700fa8f0576babd8f7ec8e56d3dcd38e28be5b99287db3"),
        ("usage", {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
        ("usage_status", "known"),
    ]


def test_chat_native_success_keeps_calls_finish_reason_and_blank_content() -> None:
    message = {
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "deny", "arguments": '{"reason":"x"}'}}],
    }
    result, _ = _chat((200, _chat_body(choices=[{"message": message, "finish_reason": "tool_calls"}]), {}), tools=TOOLS)
    assert result.content == ""
    assert result.tool_calls == (NativeToolCall("c1", "deny", '{"reason":"x"}'),)
    assert result.finish_reason == "tool_calls"
    record = result.to_redacted_record()
    assert list(record)[-2:] == ["finish_reason", "tool_call_count"]
    assert (record["finish_reason"], record["tool_call_count"]) == ("tool_calls", 1)


@pytest.mark.parametrize(
    "header, expected",
    [({"x-request-id": "   "}, None), ({}, None), ({"x-request-id": "r-1"}, "r-1")],
)
def test_chat_provider_request_id_is_stripped_or_none(header, expected) -> None:
    result, _ = _chat((200, _chat_body(), header))
    assert result.provider_request_id == expected


@pytest.mark.parametrize(
    "model, expected",
    [("  ", "demo-model"), (7, "demo-model"), (" served ", "served")],
)
def test_chat_response_model_falls_back_to_the_configured_name(model, expected) -> None:
    result, _ = _chat((200, _chat_body(model=model), {}))
    assert result.model == expected


@pytest.mark.parametrize(
    "usage, expected",
    [
        (None, (None, "unknown")),
        ("absent", (None, "unknown")),
        ({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, ({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "known")),
    ],
)
def test_chat_usage_known_or_unknown(usage, expected) -> None:
    body = _chat_body()
    if usage == "absent":
        del body["usage"]
    else:
        body["usage"] = usage
    result, _ = _chat((200, body, {}))
    assert (result.usage.as_dict() if result.usage else None, result.usage_status) == expected


@pytest.mark.parametrize(
    "usage",
    [
        [],
        "5",
        {"prompt_tokens": 3, "completion_tokens": 2},
        {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 6},
        {"prompt_tokens": -1, "completion_tokens": 6, "total_tokens": 5},
        {"prompt_tokens": True, "completion_tokens": 4, "total_tokens": 5},
        {"prompt_tokens": 3.0, "completion_tokens": 2, "total_tokens": 5},
        {"prompt_tokens": "3", "completion_tokens": 2, "total_tokens": 5},
    ],
)
def test_chat_inconsistent_usage_fails_the_call(usage) -> None:
    error, _ = _chat((200, _chat_body(usage=usage), {"x-request-id": "p"}))
    assert error.code == "invalid_usage"
    assert list(error.record.items()) == _failed_chat_record("invalid_usage", provider_request_id="p")


@pytest.mark.parametrize(
    "body, native",
    [
        ([], False),
        (_chat_body(id=None), False),
        (_chat_body(id="  "), False),
        (_chat_body(id=3), False),
        (_chat_body(choices=[]), False),
        (_chat_body(choices={}), False),
        (_chat_body(choices=["x"]), False),
        (_chat_body(choices=[{}]), False),
        (_chat_body(choices=[{"message": "x"}]), False),
        (_chat_body(choices=[{"message": {"content": "  "}}]), False),
        (_chat_body(choices=[{"message": {"content": None}}]), False),
        (_chat_body(choices=[{"message": {"content": 5}}]), False),
        (_chat_body(choices=[{"message": {"content": ["a"]}}]), False),
        (_chat_body(choices={"0": {"message": {"content": "x"}}}), False),
        (_chat_body(choices=5), False),
        (_chat_body(choices=[{"message": {"content": 1}}]), True),
        (_chat_body(choices=[{"message": {"content": "", "tool_calls": {}}}]), True),
        (_chat_body(choices=[{"message": {"content": "", "tool_calls": ["x"]}}]), True),
        (_chat_body(choices=[{"message": {"content": "", "tool_calls": [{"id": "c"}]}}]), True),
        (_chat_body(choices=[{"message": {"content": "", "tool_calls": [{"id": "c", "function": "deny"}]}}]), True),
        (_chat_body(choices=[{"message": {"content": "", "tool_calls": [{"id": 1, "function": {"name": "n", "arguments": "{}"}}]}}]), True),
        (_chat_body(choices=[{"message": {"content": "", "tool_calls": [{"id": "c", "function": {"name": "n", "arguments": {}}}]}}]), True),
    ],
)
def test_chat_malformed_json_body_is_an_invalid_response(body, native) -> None:
    error, _ = _chat((200, body, {"x-request-id": "p"}), tools=TOOLS if native else None)
    assert error.code == "invalid_response"
    assert list(error.record.items()) == _failed_chat_record("invalid_response", provider_request_id="p")


def test_chat_body_that_is_not_json_is_an_invalid_response() -> None:
    error, _ = _chat((200, b"not json", {"x-request-id": "p"}))
    assert error.code == "invalid_response"
    assert list(error.record.items()) == _failed_chat_record("invalid_response", provider_request_id="p")
    assert isinstance(error.__cause__, ValueError)


def test_chat_native_reply_without_tool_calls_and_a_non_string_finish_reason() -> None:
    body = _chat_body(choices=[{"message": {"content": " text "}, "finish_reason": 3}])
    result, _ = _chat((200, body, {}), tools=TOOLS)
    assert (result.tool_calls, result.finish_reason, result.content) == ((), "<other>", "text")


@pytest.mark.parametrize(
    "error, code",
    [
        (httpx.ReadTimeout("slow"), "upstream_timeout"),
        (httpx.ConnectTimeout("slow"), "upstream_timeout"),
        (httpx.ConnectError("refused"), "upstream_request_error"),
        (httpx.RemoteProtocolError("broken"), "upstream_request_error"),
    ],
)
def test_chat_transport_failures(error, code) -> None:
    result, _ = _chat(error=error)
    assert result.code == code and str(result) == f"model provider call failed: {code}"
    assert list(result.record.items()) == _failed_chat_record(code)
    assert result.__cause__ is not None


@pytest.mark.parametrize(
    "status, body, provider_error_code",
    [
        (500, {"error": {"code": "server_busy", "message": "secret prompt"}}, "server_busy"),
        (429, {"code": 1301, "message": "x"}, "1301"),
        (400, {"error": {"code": "bad code with spaces"}}, None),
        (401, {"error": "text"}, None),
        (404, b"<html>", None),
        (302, {"code": "Throttling.RateQuota"}, "Throttling.RateQuota"),
        (300, {}, None),
        (199, {}, None),
        (503, {"error": {"code": "x" * 65}}, None),
        (502, {"error": {"code": True}}, None),
        (500, ["code"], None),
    ],
)
def test_chat_http_errors_record_status_and_only_a_short_provider_code(status, body, provider_error_code) -> None:
    error, _ = _chat((status, body, {"x-request-id": " p "}))
    assert error.code == "upstream_http_error"
    extra = {"provider_request_id": "p", "http_status": status}
    if provider_error_code is not None:
        extra["provider_error_code"] = provider_error_code
    assert list(error.record.items()) == _failed_chat_record("upstream_http_error", **extra)
    assert error.__cause__ is None


def _blocked_chat_record(code: str) -> dict[str, object]:
    return {
        "status": "blocked",
        "mode": "real",
        "provider": "openai_compatible",
        "error_code": code,
        "usage": None,
        "usage_status": "unknown",
    }


@pytest.mark.parametrize(
    "fields, code",
    [
        ({"base_url": "  "}, "missing_model_configuration"),
        ({"base_url": "ftp://example.test"}, "invalid_model_configuration"),
        ({"base_url": "https://"}, "invalid_model_configuration"),
        ({"base_url": "example.test/v1"}, "invalid_model_configuration"),
        ({"base_url": "https://user:pw@example.test"}, "invalid_model_configuration"),
        ({"base_url": "https://user@example.test"}, "invalid_model_configuration"),
        ({"api_key": ""}, "missing_model_configuration"),
        ({"model": " "}, "missing_model_configuration"),
        ({"timeout_seconds": 0}, "invalid_model_configuration"),
        ({"timeout_seconds": 121}, "invalid_model_configuration"),
        ({"timeout_seconds": float("nan")}, "invalid_model_configuration"),
        ({"max_tokens": 0}, "invalid_model_configuration"),
        ({"max_tokens": 2049}, "invalid_model_configuration"),
        ({"max_tokens": True}, "invalid_model_configuration"),
    ],
)
def test_chat_configuration_errors_are_blocked_records(fields, code) -> None:
    values = {"base_url": "https://example.test", "api_key": "k", "model": "m", **fields}
    with pytest.raises(ModelProviderError) as caught:
        OpenAICompatibleConfig(**values)
    assert caught.value.code == code
    assert list(caught.value.record.items()) == list(_blocked_chat_record(code).items())


def test_chat_base_url_that_urlparse_rejects_is_a_blocked_record() -> None:
    with pytest.raises(ModelProviderError) as caught:
        OpenAICompatibleConfig(base_url="http://[::1", api_key="k", model="m")
    assert caught.value.code == "invalid_model_configuration"
    assert list(caught.value.record.items()) == list(_blocked_chat_record("invalid_model_configuration").items())


@pytest.mark.parametrize(
    "env, code",
    [
        ({}, "missing_model_configuration"),
        ({"QUERYSHIELD_MODEL_TIMEOUT_SECONDS": "soon"}, "invalid_model_configuration"),
        ({"QUERYSHIELD_MODEL_MAX_TOKENS": "1.5"}, "invalid_model_configuration"),
    ],
)
def test_chat_configuration_from_env_errors(monkeypatch, env, code) -> None:
    base = {} if not env else {
        "QUERYSHIELD_MODEL_BASE_URL": "https://example.test",
        "QUERYSHIELD_MODEL_API_KEY": "k",
        "QUERYSHIELD_MODEL_NAME": "m",
    }
    for name in (
        "QUERYSHIELD_MODEL_BASE_URL", "QUERYSHIELD_MODEL_API_KEY", "QUERYSHIELD_MODEL_NAME",
        "QUERYSHIELD_MODEL_TIMEOUT_SECONDS", "QUERYSHIELD_MODEL_MAX_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in {**base, **env}.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ModelProviderError) as caught:
        OpenAICompatibleConfig.from_env()
    assert caught.value.code == code
    assert caught.value.record == _blocked_chat_record(code)


@pytest.mark.parametrize(
    "messages, text",
    [
        ([], "messages must be a non-empty sequence"),
        ("hello", "messages must be a non-empty sequence"),
        (["x"], "each message must be a mapping"),
        ([{"role": "user", "content": 1}], "message role and content must be strings"),
        ([{"content": "x"}], "message role and content must be strings"),
    ],
)
def test_chat_messages_are_checked_before_any_request(messages, text) -> None:
    recorder = _Recorder((200, _chat_body(), {}))
    with httpx.Client(transport=httpx.MockTransport(recorder)) as client:
        with pytest.raises(ValueError) as caught:
            OpenAICompatibleModel(CHAT_CONFIG, client=client).complete(messages)
    assert str(caught.value) == text and not recorder.requests


def test_chat_messages_send_only_role_and_content() -> None:
    recorder = _Recorder((200, _chat_body(), {}))
    with httpx.Client(transport=httpx.MockTransport(recorder)) as client:
        OpenAICompatibleModel(CHAT_CONFIG, client=client).complete([{"role": "user", "content": "q", "name": "n"}])
    assert json.loads(recorder.requests[0].content)["messages"] == [{"role": "user", "content": "q"}]
    assert re.fullmatch(r"[0-9a-f-]{36}", recorder.requests[0].headers["x-client-request-id"])


# --- Embeddings --------------------------------------------------------------


def _embed(response=None, *, error=None, inputs=("a", "b")):
    recorder = _Recorder(response, error=error)
    with httpx.Client(transport=httpx.MockTransport(recorder)) as client:
        try:
            result = OpenAICompatibleEmbedding(EMBED_CONFIG, client=client).embed(
                list(inputs), request_id="local-request", model_call_id="server-call"
            )
        except EmbeddingProviderError as exc:
            return exc, recorder
    return result, recorder


def _embed_body(**overrides):
    body = {
        "id": " embed-call ",
        "data": [{"index": 1, "embedding": [0, 1.5]}, {"index": 0, "embedding": [1, 0]}],
        "usage": {"total_tokens": 4},
    }
    body.update(overrides)
    return body


def _failed_embed_record(error_code: str, **extra) -> list[tuple[str, object]]:
    record = [
        ("status", "failed"),
        ("mode", "real"),
        ("provider", "openai_compatible_embedding"),
        ("model", "embed-demo"),
        ("model_revision", "rev-1"),
        ("request_id", "local-request"),
        ("model_call_id", "server-call"),
        ("provider_call_id", None),
        ("provider_request_id", extra.pop("provider_request_id", None)),
        ("operation_kind", "embedding"),
        ("usage_status", "unknown"),
        ("usage", None),
        ("error_code", error_code),
    ]
    return record + list(extra.items())


def test_embedding_request_url_headers_and_payload_are_fixed() -> None:
    _result, recorder = _embed((200, _embed_body(), {}))
    (request,) = recorder.requests
    assert str(request.url) == "https://example.test/v1/embeddings"
    assert _headers(request, ("accept", "authorization", "content-type", "x-client-request-id")) == [
        ("accept", "application/json"),
        ("authorization", "Bearer test-secret"),
        ("content-type", "application/json"),
        ("x-client-request-id", "local-request"),
    ]
    assert list(json.loads(request.content).items()) == [("model", "embed-demo"), ("input", ["a", "b"]), ("dimensions", 2)]
    assert request.extensions["timeout"] == {"connect": 30.0, "read": 30.0, "write": 30.0, "pool": 30.0}


def test_embedding_endpoint_keeps_a_full_embeddings_url() -> None:
    config = EmbeddingConfig(base_url="https://e.test/v1/embeddings/", api_key="k", model="m", model_revision="r", dimensions=1)
    assert config.endpoint == "https://e.test/v1/embeddings"


def test_embedding_success_orders_vectors_by_index_and_keeps_usage() -> None:
    result, _ = _embed((200, _embed_body(), {"x-request-id": " provider-request "}))
    assert result.vectors == ((1.0, 0.0), (0.0, 1.5))
    assert (result.provider_call_id, result.provider_request_id) == ("embed-call", " provider-request ")
    assert result.inputs_sha256 == "0473ef2dc0d324ab659d3580c1134e9d812035905c4781fdd6d529b0c6860e13"
    assert list(result.usage.as_dict().items()) == [
        ("operation_kind", "embedding"),
        ("model", "embed-demo"),
        ("model_revision", "rev-1"),
        ("model_call_id", "server-call"),
        ("provider_call_id", "embed-call"),
        ("provider_request_id", " provider-request "),
        ("usage_status", "known"),
        ("input_tokens", None),
        ("output_tokens", None),
        ("total_tokens", 4),
        ("usage_source", "provider"),
        ("gateway_managed", False),
    ]
    assert (result.mode, result.provider, result.dimensions) == ("real", "openai_compatible_embedding", 2)


@pytest.mark.parametrize("header, expected", [({"x-request-id": ""}, None), ({}, None)])
def test_embedding_provider_request_id_is_kept_as_sent_or_none(header, expected) -> None:
    result, _ = _embed((200, _embed_body(), header))
    assert result.provider_request_id == expected


@pytest.mark.parametrize("usage, expected", [(None, (None, "unknown")), ("absent", (None, "unknown")), ({"total_tokens": 0}, (0, "known"))])
def test_embedding_usage_known_or_unknown(usage, expected) -> None:
    body = _embed_body()
    if usage == "absent":
        del body["usage"]
    else:
        body["usage"] = usage
    result, _ = _embed((200, body, {}))
    assert (result.usage.total_tokens, result.usage.usage_status) == expected


@pytest.mark.parametrize(
    "body, code",
    [
        ([], "invalid_response"),
        (_embed_body(id=None), "invalid_response"),
        (_embed_body(id=" "), "invalid_response"),
        (_embed_body(data=None), "invalid_response"),
        (_embed_body(data=[{"index": 0, "embedding": [1, 0]}]), "invalid_response"),
        (_embed_body(data=["x", "y"]), "invalid_response"),
        (_embed_body(data=[{"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [1, 0]}]), "invalid_response"),
        (_embed_body(data=[{"index": 0, "embedding": [1, 0]}, {"index": 2, "embedding": [1, 0]}]), "invalid_response"),
        (_embed_body(data=[{"index": 0, "embedding": [1, 0]}, {"index": True, "embedding": [1, 0]}]), "invalid_response"),
        (_embed_body(data=[{"index": 0, "embedding": [1, 0]}, {"index": -1, "embedding": [1, 0]}]), "invalid_response"),
        (_embed_body(data=[{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": "10"}]), "invalid_response"),
        (_embed_body(data=[{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": {"a": 1}}]), "invalid_response"),
        (_embed_body(data=[{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": [1]}]), "invalid_embedding_vector"),
        (_embed_body(data=[{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": [1, 0, 0]}]), "invalid_embedding_vector"),
        (_embed_body(data=[{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": [1, None]}]), "invalid_embedding_vector"),
        (_embed_body(data=[{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": [1, True]}]), "invalid_embedding_vector"),
        (_embed_body(data=[{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": [1, "2"]}]), "invalid_embedding_vector"),
        (_embed_body(usage=[]), "invalid_usage"),
        (_embed_body(usage={}), "invalid_usage"),
        (_embed_body(usage={"total_tokens": -1}), "invalid_usage"),
        (_embed_body(usage={"total_tokens": False}), "invalid_usage"),
        (_embed_body(usage={"total_tokens": 1.0}), "invalid_usage"),
    ],
)
def test_embedding_malformed_body_fails_with_a_fixed_code(body, code) -> None:
    error, _ = _embed((200, body, {"x-request-id": "p"}))
    assert error.code == code and str(error) == f"embedding provider call failed: {code}"
    assert list(error.record.items()) == _failed_embed_record(code, provider_request_id="p")


def test_embedding_non_finite_vector_value_fails() -> None:
    body = b'{"id":"e","data":[{"index":0,"embedding":[1,0]},{"index":1,"embedding":[NaN,0]}]}'
    error, _ = _embed((200, body, {}))
    assert error.code == "invalid_embedding_vector"


def test_embedding_body_that_is_not_json_is_an_invalid_response() -> None:
    error, _ = _embed((200, b"{", {"x-request-id": "p"}))
    assert error.code == "invalid_response"
    assert list(error.record.items()) == _failed_embed_record("invalid_response", provider_request_id="p")


@pytest.mark.parametrize(
    "error, code",
    [(httpx.ReadTimeout("slow"), "upstream_timeout"), (httpx.ConnectError("refused"), "upstream_request_error")],
)
def test_embedding_transport_failures(error, code) -> None:
    result, _ = _embed(error=error)
    assert result.code == code
    assert list(result.record.items()) == _failed_embed_record(code)


@pytest.mark.parametrize("status", [503, 300, 199])
def test_embedding_http_error_records_the_status_and_the_provider_code(status) -> None:
    error, _ = _embed((status, {"error": {"code": "busy"}}, {"x-request-id": " p "}))
    assert error.code == "upstream_http_error"
    assert list(error.record.items()) == _failed_embed_record(
        "upstream_http_error", provider_request_id=" p ", http_status=status, provider_error_code="busy"
    )


def test_embedding_inputs_of_exactly_32kib_are_sent() -> None:
    inputs = ("汉" * 4000, "汉" * 4000, "汉" * 2922 + "ab")
    assert sum(len(value.encode("utf-8")) for value in inputs) == 32 * 1024
    body = _embed_body(data=[{"index": index, "embedding": [1, 0]} for index in range(3)])
    result, recorder = _embed((200, body, {}), inputs=inputs)
    assert len(result.vectors) == 3 and len(recorder.requests) == 1


@pytest.mark.parametrize(
    "inputs, text",
    [
        ([], "embedding inputs must contain one to eight strings"),
        ("abc", "embedding inputs must contain one to eight strings"),
        (["a"] * 9, "embedding inputs must contain one to eight strings"),
        (["a", " "], "embedding input 1 must be a non-empty string"),
        (["a" * 4001], "embedding input 0 exceeds 4000 characters"),
        (["汉" * 3000] * 4, "embedding inputs exceed 32KiB UTF-8"),
        (["汉" * 4000, "汉" * 4000, "汉" * 2923], "embedding inputs exceed 32KiB UTF-8"),
    ],
)
def test_embedding_inputs_are_checked_before_any_request(inputs, text) -> None:
    recorder = _Recorder((200, _embed_body(), {}))
    with httpx.Client(transport=httpx.MockTransport(recorder)) as client:
        with pytest.raises(ValueError) as caught:
            OpenAICompatibleEmbedding(EMBED_CONFIG, client=client).embed(inputs)
    assert str(caught.value) == text and not recorder.requests


@pytest.mark.parametrize(
    "fields, code, field_name",
    [
        ({"base_url": " "}, "missing_embedding_configuration", "base_url"),
        ({"base_url": "file:///x"}, "invalid_embedding_configuration", "base_url"),
        ({"base_url": "https://u:p@e.test"}, "invalid_embedding_configuration", "base_url"),
        ({"api_key": ""}, "missing_embedding_configuration", "api_key"),
        ({"model": ""}, "missing_embedding_configuration", "model"),
        ({"model_revision": " "}, "missing_embedding_configuration", "model_revision"),
        ({"dimensions": 0}, "invalid_embedding_configuration", "dimensions"),
        ({"dimensions": 16_385}, "invalid_embedding_configuration", "dimensions"),
        ({"timeout_seconds": float("inf")}, "invalid_embedding_configuration", "timeout"),
    ],
)
def test_embedding_configuration_errors(fields, code, field_name) -> None:
    values = {"base_url": "https://e.test", "api_key": "k", "model": "m", "model_revision": "r", "dimensions": 2, **fields}
    with pytest.raises(EmbeddingConfigurationError) as caught:
        EmbeddingConfig(**values)
    assert (caught.value.code, caught.value.field_name, str(caught.value)) == (code, field_name, f"{code}: {field_name}")


def test_fixed_embedding_usage_counts_inputs() -> None:
    result = FixedEmbedding({"a": [1.0, 0.0], "b": [0, 1]}).embed(["b", "a"])
    assert result.vectors == ((0.0, 1.0), (1.0, 0.0))
    assert (result.usage.usage_status, result.usage.total_tokens, result.usage.usage_source) == ("known", 2, "fake")
    assert (result.request_id, result.model_call_id) == ("fake-embedding-request-1", "fake-embedding-call-1")


@pytest.mark.parametrize(
    "vector, text",
    [
        ([1.0], "embedding vector 0 has an unexpected dimension"),
        ([1.0, 0.0, 0.0], "embedding vector 0 has an unexpected dimension"),
        ("ab", "embedding vector 0 has an unexpected dimension"),
        ([1.0, float("nan")], "embedding vector 0 contains a non-finite value"),
        ([1.0, float("-inf")], "embedding vector 0 contains a non-finite value"),
        ([1.0, True], "embedding vector 0 contains a non-finite value"),
        ([1.0, None], "embedding vector 0 contains a non-finite value"),
    ],
)
def test_fixed_embedding_rejects_bad_vectors_with_the_embedding_message(vector, text) -> None:
    with pytest.raises(ValueError) as caught:
        FixedEmbedding({"a": [1.0, 0.0], "b": vector}, dimensions=2)
    assert type(caught.value) is ValueError and str(caught.value) == text


def test_provider_vector_value_beyond_float_range_is_an_invalid_vector() -> None:
    body = _embed_body(data=[{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": [1, 10**400]}])
    error, _ = _embed((200, body, {}))
    assert error.code == "invalid_embedding_vector"


def _index_snapshot() -> KnowledgeSnapshot:
    source = SourceRecord(
        source_id="s", path="shared/s.md", version="v1", content_sha256="a" * 64, tenant_scope="global",
        allowed_roles=("requester",), status="active", updated_at="2026-09-21T00:00:00Z",
    )
    chunk = ChunkRecord(
        chunk_id="s@v1#0000", source_id="s", source_version="v1", text="gross", text_sha256="1" * 64,
        chunker_version="paragraph-v1",
    )
    return KnowledgeSnapshot(
        snapshot_id="k", knowledge_version="k", catalog_version="c", chunker_version="paragraph-v1",
        embedding_model_revision=None, embedding_dimensions=None, index_hash="b" * 64, manifest_sha256="c" * 64,
        source_records=(source,), chunk_records=(chunk,),
    )


class _VectorEmbedder:
    model = "m"
    model_revision = "r"
    dimensions = 2

    def __init__(self, vector) -> None:
        self.vector = vector

    def embed(self, inputs, *, request_id=None, model_call_id=None):
        usage = OperationUsage("embedding", "m", "r", model_call_id, None, None, "unknown", None, None, None, "fake")
        return EmbeddingCallResult(
            mode="fake", provider="p", model="m", model_revision="r", request_id=request_id, model_call_id=model_call_id,
            provider_call_id=None, provider_request_id=None, inputs_sha256="x", vectors=(self.vector,), dimensions=2,
            usage=usage,
        )


@pytest.mark.parametrize(
    "vector, text",
    [
        ((1.0,), "embedding vector has an unexpected dimension"),
        ((1.0, 0.0, 0.0), "embedding vector has an unexpected dimension"),
        ("ab", "embedding vector has an unexpected dimension"),
        ((1.0, float("nan")), "embedding vector contains a non-finite value"),
        ((1.0, False), "embedding vector contains a non-finite value"),
        ((1.0, "1"), "embedding vector contains a non-finite value"),
    ],
)
def test_index_rejects_bad_vectors_with_the_index_message(vector, text) -> None:
    with pytest.raises(IndexValidationError) as caught:
        build_embedding_index(_index_snapshot(), _VectorEmbedder(vector), ingest_job_id="job")
    assert str(caught.value) == text


def test_index_converts_vector_values_to_float() -> None:
    build = build_embedding_index(_index_snapshot(), _VectorEmbedder((1, 0)), ingest_job_id="job")
    assert build.index.chunks[0].vector == (1.0, 0.0)
    assert all(type(value) is float for value in build.index.chunks[0].vector)


# --- Rerank ------------------------------------------------------------------


def _candidates(count: int = 2) -> tuple[RerankCandidate, ...]:
    return tuple(RerankCandidate(f"c{index}", f"text {index}", f"s{index}", "v1") for index in range(count))


def _rerank(response=None, *, error=None, model="rerank-demo", top_n=2):
    recorder = _Recorder(response, error=error)
    adapter = HttpRerankAdapter("https://r.test/rerank", "test-secret", model, timeout_seconds=4.0, transport=httpx.MockTransport(recorder))
    record = adapter.rerank("问题", _candidates(), top_n=top_n)
    return record, recorder


def _without_call_id(record) -> list[tuple[str, object]]:
    data = record.as_dict()
    assert re.fullmatch(r"rerank-[0-9a-f-]{36}", data.pop("call_id"))
    return list(data.items())


def test_rerank_request_url_headers_and_payload_are_fixed() -> None:
    record, recorder = _rerank((200, {"results": [{"index": 1, "relevance_score": 0.5}]}, {}))
    (request,) = recorder.requests
    assert str(request.url) == "https://r.test/rerank"
    assert request.headers["authorization"] == "Bearer test-secret"
    assert request.headers["x-client-call-id"] == record.call_id
    assert list(json.loads(request.content).items()) == [
        ("model", "rerank-demo"),
        ("query", "问题"),
        ("documents", ["text 0", "text 1"]),
        ("top_n", 2),
        ("return_documents", False),
    ]
    assert request.extensions["timeout"] == {"connect": 4.0, "read": 4.0, "write": 4.0, "pool": 4.0}


def test_qwen3_rerank_payload_has_no_return_documents() -> None:
    _record, recorder = _rerank((200, {"results": [{"index": 0, "relevance_score": 1}]}, {}), model="qwen3-rerank")
    assert list(json.loads(recorder.requests[0].content)) == ["model", "query", "documents", "top_n"]


@pytest.mark.parametrize(
    "usage, expected",
    [
        ({"total_tokens": 7}, ("known", 7)),
        ({"input_tokens": 3}, ("known", 3)),
        ({"prompt_tokens": 3, "completion_tokens": 2}, ("known", 5)),
        ({"total_tokens": -1}, ("unknown", None)),
        ("x", ("unknown", None)),
        (None, ("unknown", None)),
    ],
)
def test_rerank_success_record(usage, expected) -> None:
    body = {"object": "list", "results": [{"index": 1, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.9}], "usage": usage}
    record, _ = _rerank((200, body, {}))
    assert _without_call_id(record) == [
        ("status", "succeeded"),
        ("model", "rerank-demo"),
        ("input_candidate_ids", ["c0", "c1"]),
        ("returned_candidate_ids", ["c0", "c1"]),
        ("scores", [0.9, 0.5]),
        ("usage_status", expected[0]),
        ("total_tokens", expected[1]),
        ("error_code", None),
    ]


def test_rerank_success_record_keeps_every_input_id_when_fewer_return() -> None:
    record, _ = _rerank((200, {"results": [{"index": 1, "relevance_score": 2}]}, {}))
    assert (record.input_candidate_ids, record.returned_candidate_ids, record.scores) == (("c0", "c1"), ("c1",), (2.0,))


def test_rerank_equal_scores_keep_candidate_order() -> None:
    body = {"results": [{"index": 1, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.5}]}
    record, _ = _rerank((200, body, {}))
    assert (record.returned_candidate_ids, record.scores) == (("c0", "c1"), (0.5, 0.5))


def test_duplicate_candidate_ids_are_refused_before_sending() -> None:
    recorder = _Recorder((200, {}, {}))
    adapter = HttpRerankAdapter("https://r.test/rerank", "k", "m", transport=httpx.MockTransport(recorder))
    twice = (RerankCandidate("c0", "a", "s0", "v1"), RerankCandidate("c0", "b", "s1", "v1"))
    with pytest.raises(RerankInputError, match="^candidate IDs must be unique$"):
        adapter.rerank("q", twice, top_n=1)
    with pytest.raises(RerankInputError, match="^candidate IDs must be unique$"):
        rerank_authorized_candidates(adapter, "q", twice, authorized_candidate_ids=frozenset({"c0"}))
    assert not recorder.requests


def _failed_rerank(status: str, error_code: str) -> list[tuple[str, object]]:
    return [
        ("status", status),
        ("model", "rerank-demo"),
        ("input_candidate_ids", ["c0", "c1"]),
        ("returned_candidate_ids", []),
        ("scores", []),
        ("usage_status", "unknown"),
        ("total_tokens", None),
        ("error_code", error_code),
    ]


@pytest.mark.parametrize(
    "response, error, status, error_code",
    [
        ((503, {"results": []}, {}), None, "failed", "http_503"),
        ((301, {}, {}), None, "failed", "http_301"),
        ((300, {}, {}), None, "failed", "http_300"),
        ((199, {}, {}), None, "failed", "http_199"),
        ((200, {"results": [{"index": 0, "relevance_score": 1}, {"index": 1, "relevance_score": 1}, {"index": 0, "relevance_score": 1}]}, {}), None, "failed", "invalid_response:results must contain between one and top_n entries"),
        ((200, {"results": [{"index": True, "relevance_score": 1}]}, {}), None, "failed", "invalid_response:result index is out of range"),
        ((200, {"results": [{"index": 0, "relevance_score": True}]}, {}), None, "failed", "invalid_response:relevance_score must be finite numeric"),
        ((200, {"results": [{"index": 0, "relevance_score": 1, "v": 1}]}, {}), None, "failed", "invalid_response:result entry has unsupported fields"),
        (None, httpx.ReadTimeout("slow"), "timeout", "timeout"),
        (None, httpx.ConnectError("refused"), "failed", "transport_error"),
        ((200, b"not json", {}), None, "failed", "invalid_response:response body is not valid JSON"),
        ((200, b"\xff\xfe\x00", {}), None, "failed", "invalid_response:response body is not valid JSON"),
        ((200, {"results": [], "extra": 1}, {}), None, "failed", "invalid_response:response root has unsupported fields"),
        ((200, {"object": "x", "results": []}, {}), None, "failed", "invalid_response:response object must be list"),
        ((200, {"results": []}, {}), None, "failed", "invalid_response:results must contain between one and top_n entries"),
        ((200, {"results": [{"index": 5, "relevance_score": 1}]}, {}), None, "failed", "invalid_response:result index is out of range"),
        ((200, {"results": [{"index": 0, "relevance_score": "1"}]}, {}), None, "failed", "invalid_response:relevance_score must be finite numeric"),
    ],
)
def test_rerank_failure_records(response, error, status, error_code) -> None:
    record, _ = _rerank(response, error=error)
    assert _without_call_id(record) == _failed_rerank(status, error_code)


def test_rerank_rejects_a_bad_request_before_sending() -> None:
    recorder = _Recorder((200, {}, {}))
    adapter = HttpRerankAdapter("https://r.test/rerank", "k", "m", transport=httpx.MockTransport(recorder))
    with pytest.raises(RerankInputError, match="top_n must be between one and candidate count"):
        adapter.rerank("q", _candidates(), top_n=3)
    assert not recorder.requests


class _CountingReranker:
    model = "counting"

    def __init__(self) -> None:
        self.seen: list[tuple[str, ...]] = []

    def rerank(self, query, candidates, *, top_n):
        self.seen.append(tuple(candidate.candidate_id for candidate in candidates))
        return "called"


def test_unauthorized_candidates_never_reach_the_adapter() -> None:
    adapter = _CountingReranker()
    record, filtered = rerank_authorized_candidates(adapter, "q", _candidates(3), authorized_candidate_ids=frozenset({"c1"}), top_n=3)
    assert (record, filtered, adapter.seen) == ("called", ("c0", "c2"), [("c1",)])
    record, filtered = rerank_authorized_candidates(adapter, "q", _candidates(2), authorized_candidate_ids=set())
    assert (record, filtered, len(adapter.seen)) == (None, ("c0", "c1"), 1)


@pytest.mark.parametrize("authorized", [["c0"], "c0", ("c0",), {"c0": 1}])
def test_authorized_ids_must_be_a_set(authorized) -> None:
    with pytest.raises(RerankInputError, match="authorized_candidate_ids must be a set"):
        rerank_authorized_candidates(_CountingReranker(), "q", _candidates(), authorized_candidate_ids=authorized)


def test_more_than_ten_authorized_candidates_are_refused() -> None:
    candidates = _candidates(11)
    with pytest.raises(RerankInputError, match="at most ten authorized candidates may be reranked"):
        rerank_authorized_candidates(
            _CountingReranker(), "q", candidates, authorized_candidate_ids=frozenset(c.candidate_id for c in candidates)
        )
