from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import httpx

from queryshield.providers.contracts import (
    ModelCallResult,
    ModelProviderError,
    ModelUsage,
    new_local_call_id,
    new_request_id,
)


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
            raise _configuration_error("missing_model_configuration", "base_url")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise _configuration_error("invalid_model_configuration", "base_url")
        if parsed.username or parsed.password:
            raise _configuration_error("invalid_model_configuration", "base_url")
        if not self.api_key:
            raise _configuration_error("missing_model_configuration", "api_key")
        if not self.model.strip():
            raise _configuration_error("missing_model_configuration", "model")
        if (
            not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
            or self.timeout_seconds > 120
        ):
            raise _configuration_error("invalid_model_configuration", "timeout")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 2048:
            raise _configuration_error("invalid_model_configuration", "max_tokens")

    @classmethod
    def from_env(cls) -> OpenAICompatibleConfig:
        missing = [
            name
            for name in (
                "QUERYSHIELD_MODEL_BASE_URL",
                "QUERYSHIELD_MODEL_API_KEY",
                "QUERYSHIELD_MODEL_NAME",
            )
            if not (os.getenv(name) or "").strip()
        ]
        if missing:
            raise _configuration_error("missing_model_configuration", missing[0])

        timeout_text = os.getenv("QUERYSHIELD_MODEL_TIMEOUT_SECONDS", "15")
        max_tokens_text = os.getenv("QUERYSHIELD_MODEL_MAX_TOKENS", "512")
        try:
            timeout_seconds = float(timeout_text)
        except ValueError as exc:
            raise _configuration_error(
                "invalid_model_configuration", "timeout"
            ) from exc
        try:
            max_tokens = int(max_tokens_text)
        except ValueError as exc:
            raise _configuration_error(
                "invalid_model_configuration", "max_tokens"
            ) from exc
        return cls(
            base_url=os.environ["QUERYSHIELD_MODEL_BASE_URL"].strip(),
            api_key=os.environ["QUERYSHIELD_MODEL_API_KEY"],
            model=os.environ["QUERYSHIELD_MODEL_NAME"].strip(),
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
        )

    @property
    def endpoint(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"


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
    ) -> ModelCallResult:
        normalized_messages = _normalize_messages(messages)
        local_request_id = request_id or new_request_id()
        server_call_id = model_call_id or new_local_call_id()
        payload = {
            "model": self.config.model,
            "messages": normalized_messages,
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "X-Client-Request-Id": local_request_id,
        }

        try:
            response = self._post(payload, headers)
        except httpx.TimeoutException as exc:
            raise ModelProviderError(
                "upstream_timeout",
                _failed_record(
                    self.config,
                    request_id=local_request_id,
                    model_call_id=server_call_id,
                    error_code="upstream_timeout",
                ),
            ) from exc
        except httpx.RequestError as exc:
            raise ModelProviderError(
                "upstream_request_error",
                _failed_record(
                    self.config,
                    request_id=local_request_id,
                    model_call_id=server_call_id,
                    error_code="upstream_request_error",
                ),
            ) from exc

        provider_request_id = _header_value(response, "x-request-id")
        if response.status_code < 200 or response.status_code >= 300:
            raise ModelProviderError(
                "upstream_http_error",
                _failed_record(
                    self.config,
                    request_id=local_request_id,
                    model_call_id=server_call_id,
                    provider_request_id=provider_request_id,
                    error_code="upstream_http_error",
                    http_status=response.status_code,
                    provider_error_code=_provider_error_code(response),
                ),
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise ModelProviderError(
                "invalid_response",
                _failed_record(
                    self.config,
                    request_id=local_request_id,
                    model_call_id=server_call_id,
                    provider_request_id=provider_request_id,
                    error_code="invalid_response",
                ),
            ) from exc

        try:
            provider_call_id, content, usage = _parse_response(body)
        except _InvalidProviderResponse as exc:
            raise ModelProviderError(
                exc.code,
                _failed_record(
                    self.config,
                    request_id=local_request_id,
                    model_call_id=server_call_id,
                    provider_request_id=provider_request_id,
                    error_code=exc.code,
                ),
            ) from exc

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
        )

    def _post(
        self,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> httpx.Response:
        if self._client is not None:
            return self._client.post(
                self.config.endpoint,
                headers=headers,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            return client.post(
                self.config.endpoint,
                headers=headers,
                json=payload,
            )


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


def _parse_response(body: object) -> tuple[str, str, ModelUsage | None]:
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
    if not isinstance(content, str) or not content.strip():
        raise _InvalidProviderResponse("invalid_response")
    return provider_call_id.strip(), content.strip(), _parse_usage(body.get("usage"))


def _parse_usage(raw_usage: object) -> ModelUsage | None:
    if raw_usage is None:
        return None
    if not isinstance(raw_usage, Mapping):
        raise _InvalidProviderResponse("invalid_usage")
    values: list[int] = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw_usage.get(key)
        if type(value) is not int or value < 0:
            raise _InvalidProviderResponse("invalid_usage")
        values.append(value)
    if values[0] + values[1] != values[2]:
        raise _InvalidProviderResponse("invalid_usage")
    return ModelUsage(
        prompt_tokens=values[0],
        completion_tokens=values[1],
        total_tokens=values[2],
    )


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


def _configuration_error(code: str, _field: str) -> ModelProviderError:
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
