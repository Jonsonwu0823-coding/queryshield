"""Controlled embedding adapters for the W03 knowledge index.

Chat completion remains exposed by ``OpenAICompatibleModel.complete``.  This
module adds an explicit embedding operation with its own configuration,
revision, dimensions, IDs and usage record; it never silently falls back to a
chat model or a local vector when real configuration is missing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import os
from typing import Literal
from urllib.parse import urlparse

import httpx

from queryshield.providers.contracts import new_local_call_id, new_request_id


EmbeddingMode = Literal["fake", "real"]
UsageStatus = Literal["known", "unknown"]


class EmbeddingConfigurationError(ValueError):
    """The independent embedding configuration is missing or invalid."""

    def __init__(self, code: str, field_name: str) -> None:
        self.code = code
        self.field_name = field_name
        super().__init__(f"{code}: {field_name}")


class EmbeddingProviderError(RuntimeError):
    """Embedding failure with a safe record that contains no credentials."""

    def __init__(self, code: str, record: Mapping[str, object]) -> None:
        self.code = code
        self.record = dict(record)
        super().__init__(f"embedding provider call failed: {code}")


@dataclass(frozen=True)
class OperationUsage:
    """One independently metered embedding operation."""

    operation_kind: Literal["embedding"]
    model: str
    model_revision: str
    model_call_id: str
    provider_call_id: str | None
    provider_request_id: str | None
    usage_status: UsageStatus
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    usage_source: Literal["provider", "fake"]
    gateway_managed: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "operation_kind": self.operation_kind,
            "model": self.model,
            "model_revision": self.model_revision,
            "model_call_id": self.model_call_id,
            "provider_call_id": self.provider_call_id,
            "provider_request_id": self.provider_request_id,
            "usage_status": self.usage_status,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "usage_source": self.usage_source,
            "gateway_managed": self.gateway_managed,
        }


@dataclass(frozen=True)
class EmbeddingCallResult:
    """Validated vectors and the redacted operation evidence for one call."""

    mode: EmbeddingMode
    provider: str
    model: str
    model_revision: str
    request_id: str
    model_call_id: str
    provider_call_id: str | None
    provider_request_id: str | None
    inputs_sha256: str
    vectors: tuple[tuple[float, ...], ...]
    dimensions: int
    usage: OperationUsage

    def to_redacted_record(self) -> dict[str, object]:
        vector_bytes = json.dumps(
            self.vectors,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "status": "succeeded",
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "model_revision": self.model_revision,
            "request_id": self.request_id,
            "model_call_id": self.model_call_id,
            "provider_call_id": self.provider_call_id,
            "provider_request_id": self.provider_request_id,
            "input_count": len(self.vectors),
            "dimensions": self.dimensions,
            "inputs_sha256": self.inputs_sha256,
            "vectors_sha256": sha256(vector_bytes).hexdigest(),
            "usage": self.usage.as_dict(),
        }


@dataclass(frozen=True)
class EmbeddingConfig:
    """Non-secret settings for one OpenAI-compatible embeddings endpoint."""

    base_url: str
    api_key: str = field(repr=False)
    model: str
    model_revision: str
    dimensions: int
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise EmbeddingConfigurationError("missing_embedding_configuration", "base_url")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise EmbeddingConfigurationError("invalid_embedding_configuration", "base_url")
        if parsed.username or parsed.password:
            raise EmbeddingConfigurationError("invalid_embedding_configuration", "base_url")
        if not self.api_key:
            raise EmbeddingConfigurationError("missing_embedding_configuration", "api_key")
        if not self.model.strip():
            raise EmbeddingConfigurationError("missing_embedding_configuration", "model")
        if not self.model_revision.strip():
            raise EmbeddingConfigurationError("missing_embedding_configuration", "model_revision")
        if type(self.dimensions) is not int or not 1 <= self.dimensions <= 16_384:
            raise EmbeddingConfigurationError("invalid_embedding_configuration", "dimensions")
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 120:
            raise EmbeddingConfigurationError("invalid_embedding_configuration", "timeout")

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        required = {
            "base_url": "QUERYSHIELD_EMBEDDING_BASE_URL",
            "api_key": "QUERYSHIELD_EMBEDDING_API_KEY",
            "model": "QUERYSHIELD_EMBEDDING_MODEL_NAME",
            "model_revision": "QUERYSHIELD_EMBEDDING_MODEL_REVISION",
            "dimensions": "QUERYSHIELD_EMBEDDING_DIMENSIONS",
        }
        for field_name, env_name in required.items():
            if not (os.getenv(env_name) or "").strip():
                raise EmbeddingConfigurationError("missing_embedding_configuration", field_name)
        try:
            dimensions = int(os.environ[required["dimensions"]])
        except ValueError as exc:
            raise EmbeddingConfigurationError("invalid_embedding_configuration", "dimensions") from exc
        timeout_text = os.getenv("QUERYSHIELD_EMBEDDING_TIMEOUT_SECONDS", "30")
        try:
            timeout_seconds = float(timeout_text)
        except ValueError as exc:
            raise EmbeddingConfigurationError("invalid_embedding_configuration", "timeout") from exc
        return cls(
            base_url=os.environ[required["base_url"]].strip(),
            api_key=os.environ[required["api_key"]],
            model=os.environ[required["model"]].strip(),
            model_revision=os.environ[required["model_revision"]].strip(),
            dimensions=dimensions,
            timeout_seconds=timeout_seconds,
        )

    @property
    def endpoint(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/embeddings"):
            return base_url
        return f"{base_url}/embeddings"


def _normalize_inputs(inputs: Sequence[str]) -> tuple[str, ...]:
    if isinstance(inputs, (str, bytes)) or not 1 <= len(inputs) <= 8:
        raise ValueError("embedding inputs must contain one to eight strings")
    normalized: list[str] = []
    total_bytes = 0
    for index, value in enumerate(inputs):
        if type(value) is not str or not value.strip():
            raise ValueError(f"embedding input {index} must be a non-empty string")
        if len(value) > 4_000:
            raise ValueError(f"embedding input {index} exceeds 4000 characters")
        total_bytes += len(value.encode("utf-8"))
        normalized.append(value)
    if total_bytes > 32 * 1024:
        raise ValueError("embedding inputs exceed 32KiB UTF-8")
    return tuple(normalized)


def _inputs_hash(inputs: Sequence[str]) -> str:
    payload = json.dumps(list(inputs), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def _validate_vectors(
    vectors: Sequence[Sequence[object]],
    *,
    expected_count: int,
    dimensions: int,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(vectors, (str, bytes)) or len(vectors) != expected_count:
        raise ValueError("embedding response count does not match input count")
    normalized: list[tuple[float, ...]] = []
    for index, vector in enumerate(vectors):
        if isinstance(vector, (str, bytes)) or len(vector) != dimensions:
            raise ValueError(f"embedding vector {index} has an unexpected dimension")
        values: list[float] = []
        for value in vector:
            if type(value) not in {int, float} or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"embedding vector {index} contains a non-finite value")
            values.append(float(value))
        normalized.append(tuple(values))
    return tuple(normalized)


def _operation_usage(
    *,
    model: str,
    model_revision: str,
    model_call_id: str,
    provider_call_id: str | None,
    provider_request_id: str | None,
    total_tokens: int | None,
    usage_source: Literal["provider", "fake"],
) -> OperationUsage:
    if total_tokens is not None and (type(total_tokens) is not int or total_tokens < 0):
        raise ValueError("embedding total_tokens must be a non-negative integer or null")
    return OperationUsage(
        operation_kind="embedding",
        model=model,
        model_revision=model_revision,
        model_call_id=model_call_id,
        provider_call_id=provider_call_id,
        provider_request_id=provider_request_id,
        usage_status="known" if total_tokens is not None else "unknown",
        input_tokens=None,
        output_tokens=None,
        total_tokens=total_tokens,
        usage_source=usage_source,
    )


class FixedEmbedding:
    """A test-only fixed-vector adapter; it is never selected by real config."""

    mode = "fake"
    provider = "fixed_embedding"

    def __init__(
        self,
        vectors: Mapping[str, Sequence[float]],
        *,
        model: str = "fixed-embedding",
        model_revision: str = "fixed-embedding-v1",
        dimensions: int | None = None,
    ) -> None:
        if not vectors:
            raise ValueError("fixed embedding vectors must not be empty")
        inferred_dimensions = dimensions or len(next(iter(vectors.values())))
        if type(inferred_dimensions) is not int or inferred_dimensions <= 0:
            raise ValueError("fixed embedding dimensions must be positive")
        self.model = model
        self.model_revision = model_revision
        self.dimensions = inferred_dimensions
        self._vectors = {
            text: _validate_vectors(
                [vector],
                expected_count=1,
                dimensions=inferred_dimensions,
            )[0]
            for text, vector in vectors.items()
        }
        if any(type(text) is not str or not text for text in self._vectors):
            raise ValueError("fixed embedding keys must be non-empty strings")
        self._sequence = 0

    def embed(
        self,
        inputs: Sequence[str],
        *,
        request_id: str | None = None,
        model_call_id: str | None = None,
    ) -> EmbeddingCallResult:
        normalized = _normalize_inputs(inputs)
        missing = [value for value in normalized if value not in self._vectors]
        if missing:
            raise ValueError("fixed embedding input is not registered")
        self._sequence += 1
        local_request_id = request_id or f"fake-embedding-request-{self._sequence}"
        server_call_id = model_call_id or f"fake-embedding-call-{self._sequence}"
        vectors = tuple(self._vectors[value] for value in normalized)
        usage = _operation_usage(
            model=self.model,
            model_revision=self.model_revision,
            model_call_id=server_call_id,
            provider_call_id=None,
            provider_request_id=None,
            total_tokens=len(normalized),
            usage_source="fake",
        )
        return EmbeddingCallResult(
            mode="fake",
            provider=self.provider,
            model=self.model,
            model_revision=self.model_revision,
            request_id=local_request_id,
            model_call_id=server_call_id,
            provider_call_id=None,
            provider_request_id=None,
            inputs_sha256=_inputs_hash(normalized),
            vectors=vectors,
            dimensions=self.dimensions,
            usage=usage,
        )


class OpenAICompatibleEmbedding:
    """One non-streaming OpenAI-compatible embedding request."""

    mode = "real"
    provider = "openai_compatible_embedding"

    def __init__(
        self,
        config: EmbeddingConfig,
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
    ) -> OpenAICompatibleEmbedding:
        return cls(EmbeddingConfig.from_env(), client=client)

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def model_revision(self) -> str:
        return self.config.model_revision

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    def embed(
        self,
        inputs: Sequence[str],
        *,
        request_id: str | None = None,
        model_call_id: str | None = None,
    ) -> EmbeddingCallResult:
        normalized = _normalize_inputs(inputs)
        local_request_id = request_id or new_request_id()
        server_call_id = model_call_id or new_local_call_id()
        payload = {
            "model": self.config.model,
            "input": list(normalized),
            "dimensions": self.config.dimensions,
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
            raise EmbeddingProviderError(
                "upstream_timeout",
                _failed_record(self, local_request_id, server_call_id, "upstream_timeout"),
            ) from exc
        except httpx.RequestError as exc:
            raise EmbeddingProviderError(
                "upstream_request_error",
                _failed_record(self, local_request_id, server_call_id, "upstream_request_error"),
            ) from exc

        provider_request_id = response.headers.get("x-request-id") or None
        if response.status_code < 200 or response.status_code >= 300:
            raise EmbeddingProviderError(
                "upstream_http_error",
                _failed_record(
                    self,
                    local_request_id,
                    server_call_id,
                    "upstream_http_error",
                    provider_request_id=provider_request_id,
                    http_status=response.status_code,
                ),
            )
        try:
            body = response.json()
            provider_call_id, vectors, total_tokens = _parse_embedding_response(
                body,
                expected_count=len(normalized),
                dimensions=self.config.dimensions,
            )
        except (ValueError, TypeError) as exc:
            code = getattr(exc, "code", "invalid_response")
            raise EmbeddingProviderError(
                code,
                _failed_record(
                    self,
                    local_request_id,
                    server_call_id,
                    code,
                    provider_request_id=provider_request_id,
                ),
            ) from exc

        usage = _operation_usage(
            model=self.config.model,
            model_revision=self.config.model_revision,
            model_call_id=server_call_id,
            provider_call_id=provider_call_id,
            provider_request_id=provider_request_id,
            total_tokens=total_tokens,
            usage_source="provider",
        )
        return EmbeddingCallResult(
            mode="real",
            provider=self.provider,
            model=self.config.model,
            model_revision=self.config.model_revision,
            request_id=local_request_id,
            model_call_id=server_call_id,
            provider_call_id=provider_call_id,
            provider_request_id=provider_request_id,
            inputs_sha256=_inputs_hash(normalized),
            vectors=vectors,
            dimensions=self.config.dimensions,
            usage=usage,
        )

    def _post(self, payload: dict[str, object], headers: dict[str, str]) -> httpx.Response:
        if self._client is not None:
            return self._client.post(
                self.config.endpoint,
                headers=headers,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            return client.post(self.config.endpoint, headers=headers, json=payload)


class _InvalidEmbeddingResponse(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _parse_embedding_response(
    body: object,
    *,
    expected_count: int,
    dimensions: int,
) -> tuple[str, tuple[tuple[float, ...], ...], int | None]:
    if not isinstance(body, Mapping):
        raise _InvalidEmbeddingResponse("invalid_response")
    provider_call_id = body.get("id")
    if not isinstance(provider_call_id, str) or not provider_call_id.strip():
        raise _InvalidEmbeddingResponse("invalid_response")
    raw_data = body.get("data")
    if type(raw_data) is not list or len(raw_data) != expected_count:
        raise _InvalidEmbeddingResponse("invalid_response")
    indexed: dict[int, Sequence[object]] = {}
    for item in raw_data:
        if not isinstance(item, Mapping):
            raise _InvalidEmbeddingResponse("invalid_response")
        index = item.get("index")
        vector = item.get("embedding")
        if type(index) is not int or not 0 <= index < expected_count or index in indexed:
            raise _InvalidEmbeddingResponse("invalid_response")
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
            raise _InvalidEmbeddingResponse("invalid_response")
        indexed[index] = vector
    if set(indexed) != set(range(expected_count)):
        raise _InvalidEmbeddingResponse("invalid_response")
    try:
        vectors = _validate_vectors(
            [indexed[index] for index in range(expected_count)],
            expected_count=expected_count,
            dimensions=dimensions,
        )
    except ValueError as exc:
        raise _InvalidEmbeddingResponse("invalid_embedding_vector") from exc

    total_tokens: int | None = None
    if "usage" in body and body["usage"] is not None:
        usage = body["usage"]
        if not isinstance(usage, Mapping):
            raise _InvalidEmbeddingResponse("invalid_usage")
        raw_total = usage.get("total_tokens")
        if type(raw_total) is not int or raw_total < 0:
            raise _InvalidEmbeddingResponse("invalid_usage")
        total_tokens = raw_total
    return provider_call_id.strip(), vectors, total_tokens


def _failed_record(
    adapter: OpenAICompatibleEmbedding,
    request_id: str,
    model_call_id: str,
    error_code: str,
    *,
    provider_request_id: str | None = None,
    http_status: int | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "status": "failed",
        "mode": "real",
        "provider": adapter.provider,
        "model": adapter.config.model,
        "model_revision": adapter.config.model_revision,
        "request_id": request_id,
        "model_call_id": model_call_id,
        "provider_call_id": None,
        "provider_request_id": provider_request_id,
        "operation_kind": "embedding",
        "usage_status": "unknown",
        "usage": None,
        "error_code": error_code,
    }
    if http_status is not None:
        record["http_status"] = http_status
    return record


__all__ = [
    "EmbeddingCallResult",
    "EmbeddingConfig",
    "EmbeddingConfigurationError",
    "EmbeddingMode",
    "EmbeddingProviderError",
    "FixedEmbedding",
    "OpenAICompatibleEmbedding",
    "OperationUsage",
]
