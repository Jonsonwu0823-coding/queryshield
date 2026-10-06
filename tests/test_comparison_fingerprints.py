"""The B0 and B1 prompt fingerprints hash the files that define each profile's prompt."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys

from queryshield.agent.context import build_context
from queryshield.agent.runtime import _b0_messages
from queryshield.evaluation.comparison import build_comparison_profiles, canonical_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256_of_module_file(function) -> str:
    module = sys.modules[function.__module__]
    return hashlib.sha256(Path(inspect.getsourcefile(module)).read_bytes()).hexdigest()


def test_the_b0_fingerprint_is_the_hash_of_the_file_that_defines_the_b0_prompt() -> None:
    assert hasattr(sys.modules[_b0_messages.__module__], "B0_SYSTEM_PROMPT")
    b0, _ = build_comparison_profiles(PROJECT_ROOT, provider_mode="fake", model_name="m")["profiles"]
    assert b0["prompt_source_sha256"] == _sha256_of_module_file(_b0_messages)


def test_the_b1_fingerprint_is_the_hash_of_the_file_that_defines_the_context_builder() -> None:
    _, b1 = build_comparison_profiles(PROJECT_ROOT, provider_mode="fake", model_name="m")["profiles"]
    assert b1["prompt_source_sha256"] == _sha256_of_module_file(build_context)


def test_the_shared_configuration_digest_is_recomputed_from_the_b0_fingerprint() -> None:
    manifest = build_comparison_profiles(PROJECT_ROOT, provider_mode="fake", model_name="m")
    shared = dict(manifest["shared_configuration"])
    assert shared["b0_prompt_source_sha256"] == _sha256_of_module_file(_b0_messages)
    assert {profile["shared_configuration_sha256"] for profile in manifest["profiles"]} == {canonical_sha256(shared)}
