"""Server-owned, versioned configuration for one W03 graph run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from queryshield.catalog.catalog import DEFAULT_CATALOG_VERSION


RUN_CONFIG_VERSION: Final = "run-config-v1"
SYSTEM_PROMPT_VERSION: Final = "qs-system-prompt-v27"
ACTION_SCHEMA_VERSION: Final = "qs-action-schema-v5"
TOOL_DESCRIPTION_VERSION: Final = "qs-tool-descriptions-v15"
DEFAULT_PROFILE: Final = "queryshield-w03-hybrid-v1"
DEFAULT_KNOWLEDGE_SNAPSHOT_ID: Final = "knowledge-v1-9f580dd7f887ed0a"
DEFAULT_MODEL_VERSION: Final = "server-model-v1"
DEFAULT_ADAPTER_VERSION: Final = "model-adapter-v1"


def _version_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or len(value) > 200:
        raise ValueError(f"{field} must be a non-empty version string")
    return value


@dataclass(frozen=True)
class RunConfig:
    """Immutable configuration selected by the server, never by the model."""

    profile: str = DEFAULT_PROFILE
    prompt_version: str = SYSTEM_PROMPT_VERSION
    action_schema_version: str = ACTION_SCHEMA_VERSION
    tool_description_version: str = TOOL_DESCRIPTION_VERSION
    catalog_version: str = DEFAULT_CATALOG_VERSION
    knowledge_snapshot_id: str = DEFAULT_KNOWLEDGE_SNAPSHOT_ID
    skill_versions: tuple[str, ...] = ()
    model_version: str = DEFAULT_MODEL_VERSION
    adapter_version: str = DEFAULT_ADAPTER_VERSION

    def __post_init__(self) -> None:
        fields = (
            ("profile", self.profile),
            ("prompt_version", self.prompt_version),
            ("action_schema_version", self.action_schema_version),
            ("tool_description_version", self.tool_description_version),
            ("catalog_version", self.catalog_version),
            ("knowledge_snapshot_id", self.knowledge_snapshot_id),
            ("model_version", self.model_version),
            ("adapter_version", self.adapter_version),
        )
        for field, value in fields:
            _version_text(value, field=field)
        if type(self.skill_versions) is not tuple:
            raise ValueError("skill_versions must be a tuple")
        normalized = tuple(_version_text(value, field="skill_versions[]") for value in self.skill_versions)
        if len(set(normalized)) != len(normalized):
            raise ValueError("skill_versions must not contain duplicates")
        object.__setattr__(self, "skill_versions", normalized)

    def as_dict(self) -> dict[str, object]:
        return {
            "run_config_version": RUN_CONFIG_VERSION,
            "profile": self.profile,
            "prompt_version": self.prompt_version,
            "action_schema_version": self.action_schema_version,
            "tool_description_version": self.tool_description_version,
            "catalog_version": self.catalog_version,
            "knowledge_snapshot_id": self.knowledge_snapshot_id,
            "skill_versions": list(self.skill_versions),
            "model_version": self.model_version,
            "adapter_version": self.adapter_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RunConfig":
        if not isinstance(value, dict):
            raise ValueError("run configuration must be an object")
        fields = {
            "run_config_version",
            "profile",
            "prompt_version",
            "action_schema_version",
            "tool_description_version",
            "catalog_version",
            "knowledge_snapshot_id",
            "skill_versions",
            "model_version",
            "adapter_version",
        }
        if set(value) != fields or value.get("run_config_version") != RUN_CONFIG_VERSION:
            raise ValueError("run configuration fields or version are invalid")
        skills = value.get("skill_versions")
        if type(skills) is not list:
            raise ValueError("skill_versions must be a list in persisted configuration")
        return cls(
            profile=value["profile"],
            prompt_version=value["prompt_version"],
            action_schema_version=value["action_schema_version"],
            tool_description_version=value["tool_description_version"],
            catalog_version=value["catalog_version"],
            knowledge_snapshot_id=value["knowledge_snapshot_id"],
            skill_versions=tuple(skills),
            model_version=value["model_version"],
            adapter_version=value["adapter_version"],
        )


DEFAULT_RUN_CONFIG = RunConfig()


__all__ = [
    "ACTION_SCHEMA_VERSION",
    "DEFAULT_ADAPTER_VERSION",
    "DEFAULT_CATALOG_VERSION",
    "DEFAULT_KNOWLEDGE_SNAPSHOT_ID",
    "DEFAULT_MODEL_VERSION",
    "DEFAULT_PROFILE",
    "DEFAULT_RUN_CONFIG",
    "RUN_CONFIG_VERSION",
    "SYSTEM_PROMPT_VERSION",
    "TOOL_DESCRIPTION_VERSION",
    "RunConfig",
]
