from __future__ import annotations

import json

import httpx
import pytest

from queryshield.providers.contracts import ModelProviderError
from queryshield.providers.fake_model import FakeModel
from queryshield.providers.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleModel,
)


def test_fake_and_real_adapters_expose_the_same_complete_shape() -> None:
    result = FakeModel().complete([{"role": "user", "content": "OK"}])
    record = result.to_redacted_record()

    assert record["mode"] == "fake"
    assert record["status"] == "succeeded"
    assert record["stream"] is False
    assert record["usage"] is None
    assert record["usage_status"] == "unknown"


def test_real_adapter_sends_one_non_streaming_request_and_keeps_usage() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        captured["authorization"] = request.headers["authorization"]
        return httpx.Response(
            200,
            request=request,
            headers={"x-request-id": "provider-request-1"},
            json={
                "id": "provider-call-1",
                "model": "demo-model",
                "choices": [{"message": {"content": "OK"}}],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                },
            },
        )

    config = OpenAICompatibleConfig(
        base_url="https://example.test/v1",
        api_key="test-secret",
        model="demo-model",
        timeout_seconds=3.0,
        max_tokens=16,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = OpenAICompatibleModel(config, client=client).complete(
            [{"role": "user", "content": "Reply with OK; do not use business data."}],
            request_id="local-request-1",
            model_call_id="server-call-1",
        )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["stream"] is False
    assert payload["model"] == "demo-model"
    assert "business data" in payload["messages"][0]["content"]
    assert captured["authorization"] == "Bearer test-secret"

    record = result.to_redacted_record()
    assert record["mode"] == "real"
    assert record["request_id"] == "local-request-1"
    assert record["model_call_id"] == "server-call-1"
    assert record["provider_call_id"] == "provider-call-1"
    assert record["provider_request_id"] == "provider-request-1"
    assert record["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
    }
    assert record["usage_status"] == "known"
    assert "test-secret" not in json.dumps(record)
    assert record["content_present"] is True


def test_missing_usage_is_recorded_as_unknown_not_zero() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "provider-call-2",
                "choices": [{"message": {"content": "OK"}}],
            },
        )

    config = OpenAICompatibleConfig(
        base_url="https://example.test/v1",
        api_key="test-secret",
        model="demo-model",
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = OpenAICompatibleModel(config, client=client).complete(
            [{"role": "user", "content": "OK"}]
        )

    record = result.to_redacted_record()
    assert record["usage"] is None
    assert record["usage_status"] == "unknown"


def test_provider_error_does_not_store_response_body_or_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            request=request,
            json={"error": {"message": "secret response body must not be recorded"}},
        )

    config = OpenAICompatibleConfig(
        base_url="https://example.test/v1",
        api_key="test-secret",
        model="demo-model",
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ModelProviderError) as error:
            OpenAICompatibleModel(config, client=client).complete(
                [{"role": "user", "content": "OK"}]
            )

    record = error.value.record
    encoded = json.dumps(record)
    assert record["status"] == "failed"
    assert record["error_code"] == "upstream_http_error"
    assert record["http_status"] == 401
    assert record["usage"] is None
    assert "secret response body" not in encoded
    assert "test-secret" not in encoded


def _http_error_record(status: int, **response_kwargs) -> dict[str, object]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, request=request, **response_kwargs)

    config = OpenAICompatibleConfig(base_url="https://example.test/v1", api_key="test-secret", model="demo-model")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ModelProviderError) as error:
            OpenAICompatibleModel(config, client=client).complete([{"role": "user", "content": "OK"}])
    return error.value.record


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"error": {"code": "AccessDenied.Unpurchased", "message": "secret text must not be recorded"}}, "AccessDenied.Unpurchased"),
        ({"code": "Throttling", "message": "secret text must not be recorded"}, "Throttling"),
        ({"error": {"code": 40301, "message": "secret text must not be recorded"}}, "40301"),
    ],
    ids=["openai_style", "top_level", "integer"],
)
def test_provider_error_code_is_recorded_without_message(body, expected) -> None:
    record = _http_error_record(403, json=body)
    assert record["http_status"] == 403
    assert record["provider_error_code"] == expected
    assert "secret text" not in json.dumps(record)


@pytest.mark.parametrize(
    "response_kwargs",
    [
        {"json": {"error": {"code": "x" * 65}}},
        {"json": {"error": {"code": "bad code; drop table"}}},
        {"json": {"error": {"code": {"nested": "object"}}}},
        {"json": {"error": {"message": "no code here"}}},
        {"json": ["not", "an", "object"]},
        {"text": "<html>forbidden sk-secret-looking</html>"},
    ],
    ids=["too_long", "illegal_characters", "non_scalar", "missing", "non_object", "non_json"],
)
def test_provider_error_code_is_dropped_when_not_a_short_identifier(response_kwargs) -> None:
    record = _http_error_record(403, **response_kwargs)
    assert "provider_error_code" not in record
    assert record["error_code"] == "upstream_http_error"
    assert "sk-secret-looking" not in json.dumps(record)
