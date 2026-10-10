"""Server-owned, versioned configuration for one graph run."""

from __future__ import annotations

from dataclasses import dataclass, fields
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
NATIVE_ADAPTER_VERSION: Final = "model-adapter-native-v1"
# A native function-calling run differs from json only in how the model
# returns its decision; these versions mark it, so a checkpoint of one protocol
# never resumes under the other.
NATIVE_VERSIONS: Final = {
    "prompt_version": "qs-system-prompt-native-v2",
    "action_schema_version": "qs-action-schema-native-v1",
    "tool_description_version": "qs-tool-descriptions-native-v2",
    "adapter_version": NATIVE_ADAPTER_VERSION,
}

# The multi-agent profile: its coordinator's contract adds the delegate action
# and its subtask agents share the run's configuration, so a checkpoint of one
# profile never resumes under the other.
MULTI_AGENT_VERSIONS: Final = {
    "prompt_version": "qs-system-prompt-multi-agent-v1",
    "action_schema_version": "qs-action-schema-multi-agent-v1",
}


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
        for field in fields(self):
            if field.name != "skill_versions":
                _version_text(getattr(self, field.name), field=field.name)
        if type(self.skill_versions) is not tuple:
            raise ValueError("skill_versions must be a tuple")
        normalized = tuple(_version_text(value, field="skill_versions[]") for value in self.skill_versions)
        if len(set(normalized)) != len(normalized):
            raise ValueError("skill_versions must not contain duplicates")
        object.__setattr__(self, "skill_versions", normalized)

    @property
    def model_protocol(self) -> str:
        return "native" if self.adapter_version == NATIVE_ADAPTER_VERSION else "json"

    def as_dict(self) -> dict[str, object]:
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        return {"run_config_version": RUN_CONFIG_VERSION, **values, "skill_versions": list(self.skill_versions)}

    @classmethod
    def from_dict(cls, value: object) -> "RunConfig":
        if not isinstance(value, dict):
            raise ValueError("run configuration must be an object")
        names = [field.name for field in fields(cls)]
        if set(value) != {"run_config_version", *names} or value.get("run_config_version") != RUN_CONFIG_VERSION:
            raise ValueError("run configuration fields or version are invalid")
        skills = value.get("skill_versions")
        if type(skills) is not list:
            raise ValueError("skill_versions must be a list in persisted configuration")
        return cls(**{**{name: value[name] for name in names}, "skill_versions": tuple(skills)})


DEFAULT_RUN_CONFIG = RunConfig()


__all__ = [
    "ACTION_SCHEMA_VERSION",
    "DEFAULT_ADAPTER_VERSION",
    "DEFAULT_CATALOG_VERSION",
    "DEFAULT_KNOWLEDGE_SNAPSHOT_ID",
    "DEFAULT_MODEL_VERSION",
    "DEFAULT_PROFILE",
    "DEFAULT_RUN_CONFIG",
    "NATIVE_ADAPTER_VERSION",
    "MULTI_AGENT_VERSIONS",
    "NATIVE_VERSIONS",
    "RUN_CONFIG_VERSION",
    "SYSTEM_PROMPT_VERSION",
    "TOOL_DESCRIPTION_VERSION",
    "RunConfig",
]
