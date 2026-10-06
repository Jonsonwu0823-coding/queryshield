from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import re
from typing import Any, Mapping, Sequence

import httpx

from queryshield.providers.contracts import (
    ModelCallResult,
    ModelProviderError,
    ModelUsage,
    NativeToolCall,
    new_local_call_id,
    new_request_id,
    usage_is_consistent,
)
from queryshield.providers.http import base_url_is_valid, endpoint_url, json_headers, post_json


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Non-secret settings for one OpenAI-compatible Chat Completions endpoint."""

    base_url: str
    api_key: str = field(repr=False)
    model: str
    timeout_seconds: float = 15.0
    max_tokens: int = 512

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise _configuration_error("missing_model_configuration")
        if not base_url_is_valid(self.base_url):
            raise _configuration_error("invalid_model_configuration")
        if not self.api_key:
            raise _configuration_error("missing_model_configuration")
        if not self.model.strip():
            raise _configuration_error("missing_model_configuration")
        if (
            not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
            or self.timeout_seconds > 120
        ):
            raise _configuration_error("invalid_model_configuration")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 2048:
            raise _configuration_error("invalid_model_configuration")

    @classmethod
    def from_env(cls) -> OpenAICompatibleConfig:
        required = ("QUERYSHIELD_MODEL_BASE_URL", "QUERYSHIELD_MODEL_API_KEY", "QUERYSHIELD_MODEL_NAME")
        if any(not (os.getenv(name) or "").strip() for name in required):
            raise _configuration_error("missing_model_configuration")

        timeout_text = os.getenv("QUERYSHIELD_MODEL_TIMEOUT_SECONDS", "15")
        max_tokens_text = os.getenv("QUERYSHIELD_MODEL_MAX_TOKENS", "512")
        try:
            timeout_seconds = float(timeout_text)
        except ValueError as exc:
            raise _configuration_error("invalid_model_configuration") from exc
        try:
            max_tokens = int(max_tokens_text)
        except ValueError as exc:
            raise _configuration_error("invalid_model_configuration") from exc
        return cls(
            base_url=os.environ["QUERYSHIELD_MODEL_BASE_URL"].strip(),
            api_key=os.environ["QUERYSHIELD_MODEL_API_KEY"],
            model=os.environ["QUERYSHIELD_MODEL_NAME"].strip(),
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
        )

    @property
    def endpoint(self) -> str:
        return endpoint_url(self.base_url, "/chat/completions")


class OpenAICompatibleModel:
    """One non-streaming provider call with no SDK retry policy."""

    mode = "real"
    provider = "openai_compatible"

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._client = client

    @classmethod
    def from_env(
        cls,
        *,
        client: httpx.Client | None = None,
    ) -> OpenAICompatibleModel:
        return cls(OpenAICompatibleConfig.from_env(), client=client)

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        request_id: str | None = None,
        model_call_id: str | None = None,
        tools: Sequence[Mapping[str, object]] | None = None,
    ) -> ModelCallResult:
        normalized_messages = _normalize_messages(messages)
        local_request_id = request_id or new_request_id()
        server_call_id = model_call_id or new_local_call_id()

        def fail(code: str, **record_fields: Any) -> ModelProviderError:
            return ModelProviderError(
                code,
                _failed_record(
                    self.config, request_id=local_request_id, model_call_id=server_call_id, error_code=code, **record_fields
                ),
            )

        payload = self._payload(normalized_messages, tools)
        headers = json_headers(self.config.api_key, local_request_id)
        try:
            response = post_json(self._client, self.config.endpoint, payload, headers, self.config.timeout_seconds)
        except httpx.TimeoutException as exc:
            raise fail("upstream_timeout") from exc
        except httpx.RequestError as exc:
            raise fail("upstream_request_error") from exc

        provider_request_id = _header_value(response, "x-request-id")
        if response.status_code < 200 or response.status_code >= 300:
            raise fail(
                "upstream_http_error",
                provider_request_id=provider_request_id,
                http_status=response.status_code,
                provider_error_code=_provider_error_code(response),
            )
        try:
            body = response.json()
            provider_call_id, content, usage, tool_calls, finish_reason = _parse_response(body, native=tools is not None)
        except _InvalidProviderResponse as exc:
            raise fail(exc.code, provider_request_id=provider_request_id) from exc
        except ValueError as exc:  # the body is not JSON
            raise fail("invalid_response", provider_request_id=provider_request_id) from exc

        return ModelCallResult(
            mode="real",
            provider=self.provider,
            model=_response_model(body, self.config.model),
            request_id=local_request_id,
            model_call_id=server_call_id,
            provider_call_id=provider_call_id,
            provider_request_id=provider_request_id,
            content=content,
            usage=usage,
            usage_status="known" if usage is not None else "unknown",
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    def _payload(
        self, messages: list[dict[str, str]], tools: Sequence[Mapping[str, object]] | None
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if tools is not None:
            # Qwen accepts only auto/none ("required" is unsupported); one call per turn.
            payload.update(tools=list(tools), tool_choice="auto", parallel_tool_calls=False)
        return payload


class _InvalidProviderResponse(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _normalize_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes)) or not messages:
        raise ValueError("messages must be a non-empty sequence")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("each message must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("message role and content must be strings")
        normalized.append({"role": role, "content": content})
    return normalized


_FINISH_REASONS = ("stop", "tool_calls", "length", "content_filter")


def _parse_response(
    body: object, *, native: bool
) -> tuple[str, str, ModelUsage | None, tuple[NativeToolCall, ...] | None, str | None]:
    if not isinstance(body, Mapping):
        raise _InvalidProviderResponse("invalid_response")
    provider_call_id = body.get("id")
    if not isinstance(provider_call_id, str) or not provider_call_id.strip():
        raise _InvalidProviderResponse("invalid_response")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _InvalidProviderResponse("invalid_response")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise _InvalidProviderResponse("invalid_response")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise _InvalidProviderResponse("invalid_response")
    content = message.get("content")
    if not native:
        if not isinstance(content, str) or not content.strip():
            raise _InvalidProviderResponse("invalid_response")
        return provider_call_id.strip(), content.strip(), _parse_usage(body.get("usage")), None, None
    # A native reply carries its decision in tool_calls; content may be "" or null.
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise _InvalidProviderResponse("invalid_response")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and finish_reason not in _FINISH_REASONS:  # an unknown value is not recorded
        finish_reason = "<other>"
    return (
        provider_call_id.strip(),
        content.strip(),
        _parse_usage(body.get("usage")),
        _parse_tool_calls(message.get("tool_calls")),
        finish_reason,
    )


def _parse_tool_calls(value: object) -> tuple[NativeToolCall, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _InvalidProviderResponse("invalid_response")
    calls = []
    for item in value:
        function = item.get("function") if isinstance(item, Mapping) else None
        if not isinstance(function, Mapping):
            raise _InvalidProviderResponse("invalid_response")
        fields = (item.get("id"), function.get("name"), function.get("arguments"))
        if not all(isinstance(field, str) for field in fields):
            raise _InvalidProviderResponse("invalid_response")
        calls.append(NativeToolCall(*fields))
    return tuple(calls)


def _parse_usage(raw_usage: object) -> ModelUsage | None:
    if raw_usage is None:
        return None
    if not isinstance(raw_usage, Mapping):
        raise _InvalidProviderResponse("invalid_usage")
    prompt, completion, total = (raw_usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
    if not usage_is_consistent(prompt, completion, total):
        raise _InvalidProviderResponse("invalid_usage")
    return ModelUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def _response_model(body: Mapping[str, object], fallback: str) -> str:
    model = body.get("model")
    return model.strip() if isinstance(model, str) and model.strip() else fallback


def _header_value(response: httpx.Response, name: str) -> str | None:
    value = response.headers.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _failed_record(
    config: OpenAICompatibleConfig,
    *,
    request_id: str,
    model_call_id: str,
    error_code: str,
    provider_request_id: str | None = None,
    http_status: int | None = None,
    provider_error_code: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "status": "failed",
        "mode": "real",
        "provider": "openai_compatible",
        "model": config.model,
        "request_id": request_id,
        "model_call_id": model_call_id,
        "provider_call_id": None,
        "provider_request_id": provider_request_id,
        "stream": False,
        "content_present": False,
        "usage": None,
        "usage_status": "unknown",
        "error_code": error_code,
    }
    if http_status is not None:
        record["http_status"] = http_status
    if provider_error_code is not None:
        record["provider_error_code"] = provider_error_code
    return record


_PROVIDER_ERROR_CODE = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def _provider_error_code(response: Any) -> str | None:
    """The provider's machine error code from a JSON error body, or None.

    Only ``error.code`` (OpenAI style) or a top-level ``code`` is read, and
    only when it is a short identifier.  The message and any other text are
    never recorded: they can echo prompts, keys or endpoints.
    """

    try:
        body = response.json()
    except Exception:  # non-JSON or unreadable body
        return None
    if not isinstance(body, Mapping):
        return None
    error = body.get("error")
    code = error.get("code") if isinstance(error, Mapping) else None
    if code is None:
        code = body.get("code")
    if type(code) is int:
        code = str(code)
    if type(code) is not str or not _PROVIDER_ERROR_CODE.fullmatch(code):
        return None
    return code


def _configuration_error(code: str) -> ModelProviderError:
    return ModelProviderError(
        code,
        {
            "status": "blocked",
            "mode": "real",
            "provider": "openai_compatible",
            "error_code": code,
            "usage": None,
            "usage_status": "unknown",
        },
    )
