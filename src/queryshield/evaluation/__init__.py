"""Small, versioned evaluation fixtures used by the W03 runtime checks."""

from queryshield.evaluation.state_cases import (
    StateCaseError,
    W05StateCase,
    canonical_sha256,
    load_w05_development_cases,
    resolve_w05_actor_fixture,
    resolve_w05_principal_fixture,
    w05_development_manifest,
)
from queryshield.evaluation.w05_retrieval import (
    W05RetrievalCase,
    W05RetrievalCaseError,
    load_w05_retrieval_cases,
    load_w05_versioned_gold,
    score_w05_retrieval,
    validate_w05_gold_source_versions,
    w05_retrieval_manifest,
)
from queryshield.evaluation.state_oracle import (
    StateOracleError,
    judge_state_case,
    recompute_w05_metrics,
)
from queryshield.evaluation.sealed_holdout import HoldoutSealError, verify_w05_holdout_seal
from queryshield.evaluation.w05_holdout import (
    W05HoldoutDataset,
    W05HoldoutPayloadError,
    decrypt_and_load_w05_holdout,
)
from queryshield.evaluation.comparison import (
    B0_PROFILE,
    B1_PROFILE,
    COMPARISON_VERSION,
    ComparisonProfile,
    build_w05_comparison_profiles,
)
from queryshield.evaluation.w05_runner import (
    run_b0_single_pass,
    run_b1_bounded_agent,
    normalize_profile_observation,
    run_w05_comparison_pair,
)
from queryshield.evaluation.report import build_w05_comparison_report
from queryshield.evaluation.stateful_replay import run_w05_stateful_suite
from queryshield.evaluation.retrieval import (
    RETRIEVAL_CASE_VERSION,
    RetrievalCase,
    load_development_cases,
    run_development_catalog_retrieval,
)

__all__ = [
    "StateCaseError",
    "W05StateCase",
    "canonical_sha256",
    "load_w05_development_cases",
    "resolve_w05_actor_fixture",
    "resolve_w05_principal_fixture",
    "w05_development_manifest",
    "W05RetrievalCase",
    "W05RetrievalCaseError",
    "load_w05_retrieval_cases",
    "load_w05_versioned_gold",
    "score_w05_retrieval",
    "validate_w05_gold_source_versions",
    "w05_retrieval_manifest",
    "StateOracleError",
    "judge_state_case",
    "recompute_w05_metrics",
    "HoldoutSealError",
    "W05HoldoutDataset",
    "W05HoldoutPayloadError",
    "decrypt_and_load_w05_holdout",
    "verify_w05_holdout_seal",
    "B0_PROFILE",
    "B1_PROFILE",
    "COMPARISON_VERSION",
    "ComparisonProfile",
    "build_w05_comparison_profiles",
    "run_b0_single_pass",
    "run_b1_bounded_agent",
    "normalize_profile_observation",
    "run_w05_comparison_pair",
    "build_w05_comparison_report",
    "run_w05_stateful_suite",
    "RETRIEVAL_CASE_VERSION",
    "RetrievalCase",
    "load_development_cases",
    "run_development_catalog_retrieval",
]
