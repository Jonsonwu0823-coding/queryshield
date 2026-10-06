"""Frozen runtime budgets bound to evaluation profiles, never labels."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path


PROFILE_BUDGET_VERSION = "w05-execution-profile-budgets-v1"
_EXPECTED_PROFILES = {"B0", "B1"}


class ProfileBudgetError(ValueError):
    """The profile budget file is missing or malformed."""


def load_profile_budgets(path: str | Path | None = None) -> dict[str, object]:
    budget_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[3] / "evals" / "development" / "execution-profiles-v1.json"
    )
    try:
        document = json.loads(budget_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProfileBudgetError("cannot load execution profile budgets") from exc
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "profiles",
        "pre_model_rejections_require_zero_calls",
        "classification_independent",
    }:
        raise ProfileBudgetError("profile budget fields mismatch")
    if document["schema_version"] != PROFILE_BUDGET_VERSION:
        raise ProfileBudgetError("unsupported profile budget version")
    profiles = document["profiles"]
    if not isinstance(profiles, dict) or set(profiles) != _EXPECTED_PROFILES:
        raise ProfileBudgetError("profile budget set mismatch")
    for name, expected in {
        "B0": {"max_model_calls": 1, "max_tool_calls": 1, "max_active_seconds": None},
        "B1": {"max_model_calls": 6, "max_tool_calls": 8, "max_active_seconds": 60},
    }.items():
        actual = profiles.get(name)
        if not isinstance(actual, Mapping) or dict(actual) != expected:
            raise ProfileBudgetError(f"{name} budget does not match the approved contract")
    if document["pre_model_rejections_require_zero_calls"] is not True:
        raise ProfileBudgetError("pre-model rejections must keep a zero-call assertion")
    if document["classification_independent"] is not True:
        raise ProfileBudgetError("profile budgets must not depend on task classification")
    return document


__all__ = ["PROFILE_BUDGET_VERSION", "ProfileBudgetError", "load_profile_budgets"]
