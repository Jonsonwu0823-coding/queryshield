"""Frozen, fair B0/B1 evaluation profiles sharing one safety boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from queryshield.agent.config import (
    ACTION_SCHEMA_VERSION,
    DEFAULT_ADAPTER_VERSION,
    DEFAULT_RUN_CONFIG,
    TOOL_DESCRIPTION_VERSION,
)
from queryshield.evaluation.state_cases import canonical_sha256
from queryshield.evaluation.stateful_product import evaluation_run_config


COMPARISON_VERSION = "w05-b0-b1-comparison-v1"
B0_PROFILE = "B0-single-pass"
B1_PROFILE = "B1-bounded-agent"
COMMON_SECURITY_BOUNDARY = {
    "identity": "server-resolved-authenticated-principal-v1",
    "executor": "GuardedQueryExecutor",
    "sql_policy": "parse-readonly-select-plus-server-tenant-scope-v1",
    "catalog_version": "catalog-v1+catalog-v2-overlay",
    "knowledge_version": "knowledge-v1",
    "fixture_version": "commerce-v1",
    "facts": "FactResolver",
    "approval": "W04RunService",
    "max_result_rows": 100,
    "database_access": "postgresql-read-only-queryshield-ro",
}


@dataclass(frozen=True)
class ComparisonProfile:
    profile_id: str
    workflow: str
    max_model_calls: int
    temperature: int
    max_output_tokens: int
    uses_semantic_retrieval: bool
    supports_clarification_and_repair: bool
    prompt_version: str
    prompt_source_sha256: str
    action_schema_version: str
    tool_description_version: str
    adapter_version: str
    security_boundary_sha256: str
    shared_configuration_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "workflow": self.workflow,
            "max_model_calls": self.max_model_calls,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "uses_semantic_retrieval": self.uses_semantic_retrieval,
            "supports_clarification_and_repair": self.supports_clarification_and_repair,
            "prompt_version": self.prompt_version,
            "prompt_source_sha256": self.prompt_source_sha256,
            "action_schema_version": self.action_schema_version,
            "tool_description_version": self.tool_description_version,
            "adapter_version": self.adapter_version,
            "security_boundary_sha256": self.security_boundary_sha256,
            "shared_configuration_sha256": self.shared_configuration_sha256,
        }


def _fixture_sha256(project_root: str | Path) -> str:
    root = Path(project_root)
    files = [
        root / "fixtures" / "commerce-v1.md",
        root / "migrations" / "001_commerce_v1.sql",
        root / "fixtures" / "semantic" / "catalog-v1.json",
        root / "fixtures" / "semantic" / "catalog-v2.json",
    ]
    files.extend(
        path for path in (root / "fixtures" / "knowledge").rglob("*") if path.is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(files):
        if not path.is_file():
            raise ValueError(f"required fixture file is missing: {path.name}")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _source_sha256(project_root: str | Path, relative_path: str) -> str:
    path = Path(project_root) / relative_path
    if not path.is_file():
        raise ValueError(f"required profile source is missing: {relative_path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_comparison_profiles(
    project_root: str | Path,
    *,
    provider_mode: str,
    model_name: str,
    max_output_tokens: int = 512,
    temperature: int = 0,
    model_endpoint_fingerprint: str | None = None,
    rerank_enabled: bool = False,
) -> dict[str, object]:
    """Bind the only allowed workflow differences while sharing common controls."""

    if provider_mode not in {"fake", "real"}:
        raise ValueError("provider_mode must be fake or real")
    if type(model_name) is not str or not model_name.strip():
        raise ValueError("model_name must be a non-empty safe model identifier")
    if type(temperature) is not int or temperature != 0:
        raise ValueError("comparison fixes temperature at zero")
    if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 2048:
        raise ValueError("max_output_tokens must be between 1 and 2048")
    if model_endpoint_fingerprint is not None and (
        type(model_endpoint_fingerprint) is not str
        or len(model_endpoint_fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in model_endpoint_fingerprint)
    ):
        raise ValueError("model_endpoint_fingerprint must be a SHA256 digest, never a URL or key")

    boundary_digest = canonical_sha256(COMMON_SECURITY_BOUNDARY)
    shared = {
        "comparison_version": COMPARISON_VERSION,
        "provider_mode": provider_mode,
        "model_name": model_name,
        "model_endpoint_fingerprint": model_endpoint_fingerprint,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "fixture_sha256": _fixture_sha256(project_root),
        "security_boundary_sha256": boundary_digest,
        "same_model_adapter_instance": True,
        "same_database_role_and_fixture": True,
        "same_sampling_configuration": True,
        "b0_prompt_source_sha256": _source_sha256(project_root, "src/queryshield/agent/runtime.py"),
        "b1_prompt_source_sha256": _source_sha256(project_root, "src/queryshield/agent/context.py"),
    }
    shared_digest = canonical_sha256(shared)
    # The model protocol is the native-calling experiment's variable, so its
    # versions are per profile, not shared: B0 is always json, B1 follows
    # evaluation_run_config.
    b1_config = evaluation_run_config(DEFAULT_RUN_CONFIG, provider_mode)
    profiles = (
        ComparisonProfile(
            profile_id=B0_PROFILE,
            workflow="one model generation from question and approved table schema, then one guarded execution",
            max_model_calls=1,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            uses_semantic_retrieval=False,
            supports_clarification_and_repair=False,
            prompt_version="w05-b0-single-pass-v1",
            prompt_source_sha256=shared["b0_prompt_source_sha256"],
            action_schema_version=ACTION_SCHEMA_VERSION,
            tool_description_version=TOOL_DESCRIPTION_VERSION,
            adapter_version=DEFAULT_ADAPTER_VERSION,
            security_boundary_sha256=boundary_digest,
            shared_configuration_sha256=shared_digest,
        ),
        ComparisonProfile(
            profile_id=B1_PROFILE,
            workflow="versioned semantic retrieval and bounded agent over the same guarded tools",
            max_model_calls=6,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            uses_semantic_retrieval=True,
            supports_clarification_and_repair=True,
            prompt_version=b1_config.prompt_version,
            prompt_source_sha256=shared["b1_prompt_source_sha256"],
            action_schema_version=b1_config.action_schema_version,
            tool_description_version=b1_config.tool_description_version,
            adapter_version=b1_config.adapter_version,
            security_boundary_sha256=boundary_digest,
            shared_configuration_sha256=shared_digest,
        ),
    )
    if len({profile.security_boundary_sha256 for profile in profiles}) != 1:
        raise AssertionError("B0 and B1 must share one security boundary")
    if len({profile.shared_configuration_sha256 for profile in profiles}) != 1:
        raise AssertionError("B0 and B1 must share one model/database/fixture configuration")
    return {
        "comparison_version": COMPARISON_VERSION,
        "provider_mode": provider_mode,
        "shared_configuration": shared,
        "profiles": [profile.as_dict() for profile in profiles],
        "profile_strategy_diff": {
            "B0": ["single generation", "no retrieval", "no clarification or repair"],
            "B1": ["bounded agent loop", "versioned semantic retrieval", "clarification and one bounded repair"],
        },
        "rerank_enabled": bool(rerank_enabled),
    }


__all__ = [
    "B0_PROFILE",
    "B1_PROFILE",
    "COMMON_SECURITY_BOUNDARY",
    "COMPARISON_VERSION",
    "ComparisonProfile",
    "build_comparison_profiles",
]
