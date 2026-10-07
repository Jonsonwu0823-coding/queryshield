from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
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


def usage_is_consistent(prompt: object, completion: object, total: object) -> bool:
    """Token counts count as known only when each is a plain non-negative integer and the parts add up."""

    return all(type(value) is int and value >= 0 for value in (prompt, completion, total)) and prompt + completion == total


@dataclass(frozen=True)
class NativeToolCall:
    """One native function call exactly as the provider returned it (arguments unparsed)."""

    id: str
    name: str
    arguments: str


def native_call_for(payload: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    """The function name and arguments for a server-written json action.

    The inverse of agent.proposals.native_action_text.
    """
    if payload["type"] == "tool_call":
        return str(payload["name"]), dict(payload["arguments"])  # type: ignore[call-overload]
    return str(payload["type"]), {key: value for key, value in payload.items() if key != "type"}


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
    # None for a json-protocol call; a tuple (maybe empty) when tools were sent.
    tool_calls: tuple[NativeToolCall, ...] | None = None
    finish_reason: str | None = None

    def to_redacted_record(self) -> dict[str, object]:
        content_bytes = self.content.encode("utf-8")
        record: dict[str, object] = {
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
        if self.tool_calls is not None:
            record["finish_reason"] = self.finish_reason
            record["tool_call_count"] = len(self.tool_calls)
        return record


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
        run_id: str | None = None,
        tools: Sequence[Mapping[str, object]] | None = None,
    ) -> ModelCallResult:
        """Run one non-streaming call without exposing provider credentials.

        With ``tools`` the call uses native function calling and the result
        carries ``tool_calls``; without it the request is the json protocol's.

        ``model_call_id`` is created by the server and is intentionally
        separate from any identifier returned by the upstream provider.
        """


def finite_float(value: object) -> float | None:
    """``value`` as a float when it is a plain int or float that is finite as a float, else None.

    A bool is not a number here, and an int too large for a float (over 308 digits) is not finite.
    """

    if type(value) not in {int, float}:
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def new_request_id() -> str:
    return str(uuid4())


def new_local_call_id() -> str:
    return f"local-{uuid4()}"
