"""Deterministic W04 context compression and restore checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json


CONTEXT_RUNTIME_VERSION = "qs-context-runtime-v1"
MAX_CONTEXT_BYTES = 24_000
MAX_RESULT_REFERENCES = 32


class ContextRecoveryError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class CompressedContext:
    payload: Mapping[str, object]
    removed_optional_count: int
    size_bytes: int

    def as_dict(self) -> dict[str, object]:
        return dict(self.payload)


def _encoded(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compress_context(
    *,
    goal: str,
    metric_ids: Sequence[str],
    time_window: Mapping[str, str],
    constraints: Sequence[str],
    result_refs: Sequence[Mapping[str, object]],
    optional_tool_summaries: Sequence[Mapping[str, object]] = (),
    retrieved_items: Sequence[Mapping[str, object]] = (),
    max_bytes: int = MAX_CONTEXT_BYTES,
) -> CompressedContext:
    if type(goal) is not str or not goal.strip():
        raise ContextRecoveryError("invalid_context", "goal must be non-empty")
    if set(time_window) != {"start", "end", "timezone"}:
        raise ContextRecoveryError("invalid_context", "time_window is incomplete")
    hard = {
        "runtime_version": CONTEXT_RUNTIME_VERSION,
        "goal": goal,
        "metric_ids": list(metric_ids),
        "time_window": dict(time_window),
        "constraints": list(constraints),
        "result_refs": [dict(item) for item in result_refs],
    }
    if len(hard["result_refs"]) > MAX_RESULT_REFERENCES:
        raise ContextRecoveryError("invalid_context", "too many result references")
    optional = [
        {"kind": "tool_summary", "value": dict(item)} for item in optional_tool_summaries
    ] + [
        {"kind": "retrieval_data", "value": dict(item)} for item in retrieved_items
    ]
    removed = 0
    while True:
        candidate = {**hard, "optional_data": optional}
        size = len(_encoded(candidate))
        if size <= max_bytes:
            return CompressedContext(candidate, removed, size)
        if not optional:
            raise ContextRecoveryError("context_budget_exceeded", "hard context exceeds the configured budget")
        optional.pop(0)
        removed += 1


def verify_restore(
    checkpoint: Mapping[str, object],
    *,
    tenant_id: str,
    principal_id: str,
    current_versions: Mapping[str, str],
    current_permission_version: str,
    approval_valid: bool,
) -> dict[str, object]:
    required = {"tenant_id", "principal_id", "versions", "permission_version", "result_refs"}
    if not required <= set(checkpoint):
        raise ContextRecoveryError("recovery_required", "checkpoint is missing restore fields")
    if checkpoint["tenant_id"] != tenant_id or checkpoint["principal_id"] != principal_id:
        raise ContextRecoveryError("not_found", "checkpoint is outside the current subject")
    stored_versions = checkpoint["versions"]
    if not isinstance(stored_versions, Mapping):
        raise ContextRecoveryError("recovery_required", "checkpoint versions are invalid")
    for key, value in stored_versions.items():
        if current_versions.get(str(key)) != value:
            raise ContextRecoveryError("recovery_required", "a required runtime version is unavailable")
    if checkpoint["permission_version"] != current_permission_version:
        raise ContextRecoveryError("authorization_revoked", "current permissions changed")
    if not approval_valid:
        raise ContextRecoveryError("approval_stale", "approval is no longer valid")
    return {
        "status": "compatible",
        "tenant_id": tenant_id,
        "principal_id": principal_id,
        "result_refs": [dict(item) for item in checkpoint["result_refs"]],
        "reexecute_submitted_results": False,
    }


__all__ = [
    "CONTEXT_RUNTIME_VERSION",
    "CompressedContext",
    "ContextRecoveryError",
    "MAX_CONTEXT_BYTES",
    "compress_context",
    "verify_restore",
]
