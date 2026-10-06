"""Small, versioned evaluation fixtures used by the runtime checks."""

from queryshield.evaluation.state_cases import (
    StateCaseError,
    StateCase,
    canonical_sha256,
    load_state_cases,
    resolve_actor_fixture,
    resolve_principal_fixture,
    state_case_manifest,
)
from queryshield.evaluation.source_retrieval import (
    SourceRetrievalCase,
    SourceRetrievalCaseError,
    load_source_retrieval_cases,
    load_versioned_retrieval_gold,
    score_source_retrieval,
    validate_gold_source_versions,
    source_retrieval_manifest,
)
from queryshield.evaluation.state_oracle import (
    StateOracleError,
    judge_state_case,
    recompute_metrics,
)
from queryshield.evaluation.sealed_holdout import HoldoutSealError, verify_holdout_seal
from queryshield.evaluation.holdout_dataset import (
    HoldoutDataset,
    HoldoutPayloadError,
    decrypt_and_load_holdout,
)
from queryshield.evaluation.comparison import (
    B0_PROFILE,
    B1_PROFILE,
    COMPARISON_VERSION,
    ComparisonProfile,
    build_comparison_profiles,
)
from queryshield.evaluation.profile_runner import (
    run_b0_single_pass,
    run_b1_bounded_agent,
    normalize_profile_observation,
    run_comparison_pair,
)
from queryshield.evaluation.report import build_comparison_report
from queryshield.evaluation.stateful_replay import run_stateful_suite
from queryshield.evaluation.retrieval import (
    RETRIEVAL_CASE_VERSION,
    RetrievalCase,
    load_development_cases,
    run_development_catalog_retrieval,
)

__all__ = [
    "StateCaseError",
    "StateCase",
    "canonical_sha256",
    "load_state_cases",
    "resolve_actor_fixture",
    "resolve_principal_fixture",
    "state_case_manifest",
    "SourceRetrievalCase",
    "SourceRetrievalCaseError",
    "load_source_retrieval_cases",
    "load_versioned_retrieval_gold",
    "score_source_retrieval",
    "validate_gold_source_versions",
    "source_retrieval_manifest",
    "StateOracleError",
    "judge_state_case",
    "recompute_metrics",
    "HoldoutSealError",
    "HoldoutDataset",
    "HoldoutPayloadError",
    "decrypt_and_load_holdout",
    "verify_holdout_seal",
    "B0_PROFILE",
    "B1_PROFILE",
    "COMPARISON_VERSION",
    "ComparisonProfile",
    "build_comparison_profiles",
    "run_b0_single_pass",
    "run_b1_bounded_agent",
    "normalize_profile_observation",
    "run_comparison_pair",
    "build_comparison_report",
    "run_stateful_suite",
    "RETRIEVAL_CASE_VERSION",
    "RetrievalCase",
    "load_development_cases",
    "run_development_catalog_retrieval",
]
