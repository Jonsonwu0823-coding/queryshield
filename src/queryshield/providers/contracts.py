from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Mapping, Protocol, Sequence
from uuid import uuid4


ModelMode = Literal["fake", "real"]
UsageStatus = Literal["known", "unknown"]


@dataclass(frozen=True)
class ModelUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ModelCallResult:
    """Private result of one provider call; its record is safe to persist."""

    mode: ModelMode
    provider: str
    model: str
    request_id: str
    model_call_id: str
    provider_call_id: str | None
    provider_request_id: str | None
    content: str
    usage: ModelUsage | None
    usage_status: UsageStatus

    def to_redacted_record(self) -> dict[str, object]:
        content_bytes = self.content.encode("utf-8")
        return {
            "status": "succeeded",
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "request_id": self.request_id,
            "model_call_id": self.model_call_id,
            "provider_call_id": self.provider_call_id,
            "provider_request_id": self.provider_request_id,
            "stream": False,
            "content_present": bool(self.content),
            "content_length": len(content_bytes),
            "content_sha256": sha256(content_bytes).hexdigest(),
            "usage": self.usage.as_dict() if self.usage is not None else None,
            "usage_status": self.usage_status,
        }


class ModelProviderError(RuntimeError):
    """Provider failure carrying a safe, already-redacted evidence record."""

    def __init__(self, code: str, record: Mapping[str, object]) -> None:
        self.code = code
        self.record = dict(record)
        super().__init__(f"model provider call failed: {code}")


class ModelAdapter(Protocol):
    mode: ModelMode

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        request_id: str | None = None,
        model_call_id: str | None = None,
    ) -> ModelCallResult:
        """Run one non-streaming call without exposing provider credentials.

        ``model_call_id`` is created by the server and is intentionally
        separate from any identifier returned by the upstream provider.
        """


def new_request_id() -> str:
    return str(uuid4())


def new_local_call_id() -> str:
    return f"local-{uuid4()}"
