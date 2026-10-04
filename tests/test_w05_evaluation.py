from __future__ import annotations

import json
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from queryshield.evaluation import (
    StateCaseError,
    W05RetrievalCaseError,
    StateOracleError,
    HoldoutSealError,
    B0_PROFILE,
    B1_PROFILE,
    build_w05_comparison_report,
    build_w05_comparison_profiles,
    canonical_sha256,
    load_w05_development_cases,
    load_w05_retrieval_cases,
    load_w05_versioned_gold,
    judge_state_case,
    recompute_w05_metrics,
    resolve_w05_actor_fixture,
    resolve_w05_principal_fixture,
    score_w05_retrieval,
    validate_w05_gold_source_versions,
    w05_development_manifest,
    w05_retrieval_manifest,
    verify_w05_holdout_seal,
)
from queryshield.evaluation.profile_budgets import load_w05_profile_budgets


def test_w05_state_fixture_freezes_c10_pairs_quota_and_eight_critical_questions() -> None:
    cases = load_w05_development_cases()
    manifest = w05_development_manifest()

    assert len(cases) == manifest["case_count"] == 20
    assert manifest["functional_count"] == 12
    assert manifest["security_count"] == 8
    assert len(manifest["paired_family_ids"]) == 6
    assert len(manifest["critical_question_ids"]) == 8
    assert len(manifest["cases"]) == 20
    assert len(manifest["case_config_sha256"]) == 64
    assert all(len(row["input_sha256"]) == 64 for row in manifest["cases"])
    assert all(len(row["expected_sha256"]) == 64 for row in manifest["cases"])
    assert all(row["case_config_sha256"] == manifest["case_config_sha256"] for row in manifest["cases"])


def test_w05_erratum_manifest_preserves_predecessors_and_binds_profile_budgets() -> None:
    project_root = Path(__file__).resolve().parents[1]
    eval_root = project_root / "evals" / "w05"
    manifest = w05_development_manifest()
    expected_hashes = {
        "state-cases-v1.json": "6660153a05b88c781a3fce2186e950f0f96039a9f58bc630f0929d5fdf733a9f",
        "state-cases-v2.json": "1a0f1d9d526d8d37441a65c8b128738c7b650429a8a1471f59277c2311cb4e45",
        "state-cases-v3.json": "1b97b6dd6bc65ab5711aca70564b63b94afbdce9c8128bf0ea7ba99ff59489b2",
        "state-cases-v4.json": "ce3d64220c68cb358aa56aaced231ca001b66ed1c17f3c4b5628c80cdcb459d0",
        "execution-profiles-v1.json": "8fc58aa45a8528c51f10040aa51e6427bda720b2c637cef84ae6dc1b0b7359da",
    }
    actual_hashes = {
        name: hashlib.sha256((eval_root / name).read_bytes()).hexdigest()
        for name in expected_hashes
    }

    assert actual_hashes == expected_hashes
    assert manifest["dataset_revision"] == "w05-development-erratum-r3"
    assert Path(manifest["path"]).name == "state-cases-v4.json"
    assert manifest["sha256"] == expected_hashes["state-cases-v4.json"]
    assert Path(manifest["predecessor_path"]).name == "state-cases-v3.json"
    assert manifest["predecessor_sha256"] == expected_hashes["state-cases-v3.json"]
    assert manifest["execution_profile_budgets_sha256"] == expected_hashes["execution-profiles-v1.json"]

    budgets = load_w05_profile_budgets()
    assert budgets["classification_independent"] is True
    assert budgets["pre_model_rejections_require_zero_calls"] is True
    assert budgets["profiles"]["B0"] == {
        "max_model_calls": 1,
        "max_tool_calls": 1,
        "max_active_seconds": None,
    }
    assert budgets["profiles"]["B1"] == {
        "max_model_calls": 6,
        "max_tool_calls": 8,
        "max_active_seconds": 60,
    }


def test_principal_fixture_maps_to_commerce_tenant_key_and_rejects_cross_tenant_actor() -> None:
    assert resolve_w05_principal_fixture("tenant-A/requester-A") == {
        "tenant_id": "A",
        "principal_id": "principal-A",
        "role": "requester",
    }
    assert resolve_w05_actor_fixture("requester-A", tenant_id="A") == {
        "tenant_id": "A",
        "principal_id": "principal-A",
        "role": "requester",
    }
    assert resolve_w05_actor_fixture("approver-A", tenant_id="A")["role"] == "approver"
    with pytest.raises(StateCaseError, match="same commerce tenant"):
        resolve_w05_principal_fixture("tenant-A/requester-B")
    with pytest.raises(StateCaseError, match="initialized tenant"):
        resolve_w05_actor_fixture("requester-B", tenant_id="A")


def test_state_loader_rejects_changed_split_or_duplicate_case_id(tmp_path) -> None:
    source = json.loads(
        (Path(__file__).resolve().parents[1] / "evals" / "w05" / "state-cases-v3.json").read_text(
            encoding="utf-8"
        )
    )
    source["cases"][0]["case"]["split"] = "holdout"
    changed = tmp_path / "invalid-state-cases.json"
    changed.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(StateCaseError, match="split must be development"):
        load_w05_development_cases(changed)

    source = json.loads(
        (Path(__file__).resolve().parents[1] / "evals" / "w05" / "state-cases-v3.json").read_text(
            encoding="utf-8"
        )
    )
    source["cases"][1]["case"]["case_id"] = source["cases"][0]["case"]["case_id"]
    changed.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(StateCaseError, match="case_id values must be unique"):
        load_w05_development_cases(changed)


def test_development_manifest_binds_safe_shared_configuration_digest() -> None:
    digest = "a" * 64
    manifest = w05_development_manifest(shared_runtime_config_sha256=digest)

    assert manifest["shared_runtime_config_sha256"] == digest
    with pytest.raises(StateCaseError, match="lowercase SHA256"):
        w05_development_manifest(shared_runtime_config_sha256="not-a-digest")


def test_b0_b1_profiles_share_model_database_fixture_and_security_hash() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = build_w05_comparison_profiles(
        project_root,
        provider_mode="fake",
        model_name="fake-model",
        model_endpoint_fingerprint="a" * 64,
    )
    b0, b1 = manifest["profiles"]

    assert b0["profile_id"] == B0_PROFILE
    assert b1["profile_id"] == B1_PROFILE
    assert b0["security_boundary_sha256"] == b1["security_boundary_sha256"]
    assert b0["shared_configuration_sha256"] == b1["shared_configuration_sha256"]
    assert b0["max_model_calls"] == 1
    assert b1["max_model_calls"] > b0["max_model_calls"]
    assert b0["uses_semantic_retrieval"] is False
    assert b1["uses_semantic_retrieval"] is True


def test_retrieval_fixture_and_versioned_gold_bind_catalog_v2_snapshot() -> None:
    cases = load_w05_retrieval_cases()
    gold = load_w05_versioned_gold()
    manifest = w05_retrieval_manifest()

    assert len(cases) == manifest["case_count"] == 12
    assert manifest["catalog_version"] == "catalog-v2"
    assert set(gold) == {case.family_id for case in cases}
    assert all(version for sources in gold.values() for _, version in sources)
    assert next(case for case in cases if case.family_id == "empty-hit").relevant_source_ids == ()


def test_retrieval_scorer_keeps_failed_timeout_and_empty_gold_in_query_denominator() -> None:
    cases = load_w05_retrieval_cases()[-2:]
    gold = load_w05_versioned_gold()
    records = [
        {"family_id": cases[0].family_id, "status": "timeout"},
        {"family_id": cases[1].family_id, "status": "succeeded", "items": []},
    ]

    report = score_w05_retrieval(cases, records, {case.family_id: gold[case.family_id] for case in cases})

    assert report["case_count"] == 2
    assert report["hit_at_3"] == {"numerator": 0, "denominator": 2, "value": 0.0}
    assert report["recall_at_3_macro"]["case_denominator"] == 2
    assert report["failed_count"] == 1
    assert report["empty_hit_count"] == 2
    assert [item["status"] for item in report["per_case"]] == ["timeout", "succeeded"]


def test_retrieval_scorer_requires_exact_source_version_and_reports_stale_match() -> None:
    case = load_w05_retrieval_cases()[0]
    gold = load_w05_versioned_gold()[case.family_id]
    source_id, _ = gold[0]

    report = score_w05_retrieval(
        (case,),
        ({"family_id": case.family_id, "status": "succeeded", "items": [{"source_id": source_id, "version": "stale-version"}]},),
        {case.family_id: gold},
    )

    assert report["hit_at_3"]["numerator"] == 0
    assert report["wrong_version_count"] == 1
    assert report["per_case"][0]["wrong_version_returned"] is True


def test_retrieval_scorer_rejects_gold_family_gap_and_duplicate_record() -> None:
    cases = load_w05_retrieval_cases()[:1]
    gold = load_w05_versioned_gold()
    with pytest.raises(W05RetrievalCaseError, match="cover exactly"):
        score_w05_retrieval(cases, (), {})
    with pytest.raises(W05RetrievalCaseError, match="duplicate retrieval record"):
        score_w05_retrieval(
            cases,
            ({"family_id": cases[0].family_id, "status": "failed"},) * 2,
            {cases[0].family_id: gold[cases[0].family_id]},
        )


def test_retrieval_gold_rejects_same_source_with_stale_version() -> None:
    gold = load_w05_versioned_gold()

    with pytest.raises(W05RetrievalCaseError, match="stale"):
        validate_w05_gold_source_versions(gold, {"semantic-metric-gross": "2026-09-20"})

    matched = validate_w05_gold_source_versions(
        {"metric-gross": (("semantic-metric-gross", "2026-09-21"),), "empty-hit": ()},
        {"semantic-metric-gross": "2026-09-21"},
    )
    assert matched["gold_family_count"] == 2
    assert matched["gold_source_count"] == 1


def test_canonical_hash_rejects_non_finite_floats() -> None:
    with pytest.raises(ValueError):
        canonical_sha256({"not-finite": float("nan")})


def test_state_oracle_accepts_exact_facts_and_order_independent_rows() -> None:
    case = next(case for case in load_w05_development_cases() if case.case_id == "join-aggregate-by-customer")
    expected = deepcopy(case.case["expected"])
    rows = expected["facts"][0]["rows"]
    observed = {
        "status": "succeeded",
        "http_status": expected["http_status"],
        "terminal_state": expected["terminal_state"],
        "facts": [],
        "rows": list(reversed(rows)),
        "rowset_metadata": {
            "unit": expected["facts"][0]["unit"],
            "time_window": expected["facts"][0]["time_window"],
            "tenant_id": expected["facts"][0]["tenant_id"],
        },
        "invariants": expected["invariants"],
        "side_effects": {**expected["allowed_side_effects"], **expected["forbidden_side_effects"]},
        "usage": {"usage_status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "elapsed_ms": 12,
    }

    result = judge_state_case(case, observed)

    assert result["judged_status"] == "pass"
    assert result["critical_question_id"] == "join-aggregate"


def test_state_oracle_rejects_wrong_amount_and_forbidden_side_effect() -> None:
    case = next(case for case in load_w05_development_cases() if case.case_id == "security-mutating-sql-rejected")
    expected = deepcopy(case.case["expected"])
    observed = {
        "status": "succeeded",
        "http_status": expected["http_status"],
        "terminal_state": expected["terminal_state"],
        "facts": [],
        "invariants": expected["invariants"],
        "side_effects": {**expected["allowed_side_effects"], "write_statements": 1},
        "usage": {"usage_status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }

    result = judge_state_case(case, observed)

    assert result["judged_status"] == "fail"
    assert "forbidden_side_effects.write_statements" in result["mismatches"]


def test_state_oracle_checks_expected_terminal_status_and_unknown_usage_shape() -> None:
    case = next(case for case in load_w05_development_cases() if case.case_id == "security-mutating-sql-rejected")
    expected = deepcopy(case.case["expected"])
    observed = {
        "status": "succeeded",
        "http_status": expected["http_status"],
        "terminal_state": expected["terminal_state"],
        "facts": [],
        "invariants": expected["invariants"],
        "side_effects": {**expected["allowed_side_effects"], **expected["forbidden_side_effects"]},
        "usage": {"usage_status": "unknown", "prompt_tokens": None, "total_tokens": None},
    }

    result = judge_state_case(case, observed)

    assert result["judged_status"] == "fail"
    assert any(mismatch.startswith("terminal-observation") for mismatch in result["mismatches"])
    assert "usage_fields_missing" in result["mismatches"]


def test_state_oracle_treats_fact_order_as_irrelevant_but_preserves_values() -> None:
    case = next(case for case in load_w05_development_cases() if case.case_id == "empty-window-zero-aggregate")
    expected = deepcopy(case.case["expected"])
    observed = {
        "status": "succeeded",
        "http_status": expected["http_status"],
        "terminal_state": expected["terminal_state"],
        "facts": list(reversed(expected["facts"])),
        "invariants": expected["invariants"],
        "side_effects": {**expected["allowed_side_effects"], **expected["forbidden_side_effects"]},
        "usage": {"usage_status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }

    assert judge_state_case(case, observed)["judged_status"] == "pass"
    observed["facts"][0]["value"] = 1
    assert judge_state_case(case, observed)["judged_status"] == "fail"


def test_state_oracle_accepts_waiting_approval_and_checks_exact_invariants() -> None:
    case = next(case for case in load_w05_development_cases() if case.case_id == "approval-expiry-stale")
    expected = case.case["expected"]
    observed = {
        "status": "waiting_approval",
        "http_status": expected["http_status"],
        "terminal_state": expected["terminal_state"],
        "facts": [],
        "invariants": expected["invariants"],
        "side_effects": {**expected["allowed_side_effects"], **expected["forbidden_side_effects"]},
        "usage": {"usage_status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }

    assert judge_state_case(case, observed)["judged_status"] == "pass"
    observed["invariants"] = {**expected["invariants"], "no_retry_bypass": False}
    result = judge_state_case(case, observed)
    assert result["judged_status"] == "fail"
    assert "invariants" in result["mismatches"]


@pytest.mark.parametrize(
    ("case_id", "profile", "model_calls", "tool_calls", "active_seconds", "expected_mismatch"),
    [
        ("gross-total-fen", "B0", 2, 1, 0.1, "profile_budget.model_calls"),
        ("security-cross-tenant-filter", "B1", 7, 8, 10, "profile_budget.model_calls"),
        ("gross-total-fen", "B1", 6, 8, 60, None),
    ],
)
def test_profile_budgets_are_enforced_independent_of_case_classification(
    case_id: str,
    profile: str,
    model_calls: int,
    tool_calls: int,
    active_seconds: float,
    expected_mismatch: str | None,
) -> None:
    case = next(case for case in load_w05_development_cases() if case.case_id == case_id)
    expected = case.case["expected"]
    terminal_state = expected["terminal_state"]
    status = {
        "SUCCEEDED": "succeeded",
        "DENIED": "denied",
        "WAITING_USER": "waiting_user",
        "WAITING_APPROVAL": "waiting_approval",
        "FAILED": "failed",
    }[terminal_state]
    effects = {
        **expected["allowed_side_effects"],
        **expected["forbidden_side_effects"],
        "model_calls": model_calls,
        "tool_calls": tool_calls,
    }
    observation = {
        "status": status,
        "http_status": expected["http_status"],
        "terminal_state": terminal_state,
        "facts": deepcopy(expected["facts"]),
        "invariants": deepcopy(expected["invariants"]),
        "side_effects": effects,
        "usage": {
            "usage_status": "unknown",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
        "elapsed_ms": int(active_seconds * 1000),
        "evaluation_profile": profile,
        "execution_metrics": {
            "model_calls": model_calls,
            "tool_calls": tool_calls,
            "active_seconds": active_seconds,
        },
    }

    result = judge_state_case(case, observation)
    if expected_mismatch is None:
        assert result["judged_status"] == "pass"
    else:
        assert result["judged_status"] == "fail"
        assert expected_mismatch in result["mismatches"]


def test_recomputed_metrics_keep_missing_timeout_and_unknown_cost_in_denominator() -> None:
    cases = load_w05_development_cases()[:3]
    raw = [
        {
            "case_id": cases[0].case_id,
            "status": "timeout",
            "http_status": None,
            "terminal_state": "UNKNOWN",
            "facts": [],
            "side_effects": {"model_calls": 1},
            "usage": {"usage_status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "elapsed_ms": 30000,
        }
    ]

    report = recompute_w05_metrics(cases, raw)

    assert report["case_count"] == 3
    assert report["functional_success"]["denominator"] == sum(case.classification == "functional" for case in cases)
    assert report["model_calls"]["known_total"] == 1
    assert report["model_calls"]["case_denominator"] == 3
    assert report["usage"]["known_total_tokens"] is None
    assert report["usage"]["unknown_case_count"] == 3
    assert report["latency_ms"]["case_denominator"] == 3
    assert sum(row["judged_status"] == "fail" for row in report["per_case"]) == 3


def test_recomputed_usage_reconciles_known_failed_call_unknown_call_and_phases() -> None:
    case = load_w05_development_cases()[0]
    known_id = "prep-failed-call-known-usage"
    unknown_id = "action-call-unknown-usage"
    raw = [{
        "case_id": case.case_id,
        "status": "failed",
        "model_call_ids": [known_id, unknown_id],
        "model_call_records": [
            {
                "model_call_id": known_id,
                "status": "failed",
                "usage_status": "known",
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
            {
                "model_call_id": unknown_id,
                "status": "failed",
                "usage_status": "unknown",
                "usage": None,
            },
        ],
        "execution_metrics": {"model_calls": 2},
        "side_effects": {"model_calls": 2},
        "usage": {"usage_status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "usage_phases": {
            "preparation": {"status": "known", "model_call_count": 1, "model_call_ids": [known_id]},
            "action": {"status": "unknown", "model_call_count": 1, "model_call_ids": [unknown_id]},
            "cumulative": {"status": "unknown", "model_call_count": 2, "model_call_ids": [known_id, unknown_id]},
        },
    }]

    report = recompute_w05_metrics([case], raw)
    calls = report["usage"]["model_calls"]
    phases = report["usage"]["phases"]
    reconciliation = report["usage"]["reconciliation"]

    assert calls["status"] == "unknown"
    assert calls["model_call_count"] == 2
    assert calls["known_call_count"] == 1
    assert calls["unknown_call_count"] == 1
    assert calls["known_total_tokens"] == 7
    assert calls["total_tokens"] is None
    assert phases["preparation"]["status"] == "known"
    assert phases["preparation"]["total_tokens"] == 7
    assert phases["action"]["status"] == "unknown"
    assert phases["cumulative"]["model_call_ids"] == sorted([known_id, unknown_id])
    assert reconciliation["preparation_union_action_matches_cumulative"] is True
    assert reconciliation["provider_records_cover_observed_ids"] is True
    assert reconciliation["declared_ids_match_provider_record_ids"] is True
    assert reconciliation["reported_call_count_matches_ids"] is True


def test_blocked_product_case_stays_in_denominator_without_becoming_product_failure() -> None:
    cases = load_w05_development_cases()[:1]
    raw = [{
        "case_id": cases[0].case_id,
        "status": "blocked",
        "execution_status": "blocked",
        "not_run_reason": "PostgreSQL configuration is missing",
        "usage": {"usage_status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }]

    report = recompute_w05_metrics(cases, raw)

    assert report["case_count"] == 1
    assert report["functional_success"]["denominator"] == 1
    assert report["functional_success"]["numerator"] == 0
    assert report["blocked_case_count"] == 1
    assert report["judged_failure_count"] == 0
    assert report["per_case"][0]["judged_status"] == "blocked"


def test_recomputed_metrics_reject_raw_rows_outside_frozen_dataset() -> None:
    with pytest.raises(StateOracleError, match="unknown cases"):
        recompute_w05_metrics(
            load_w05_development_cases()[:1],
            ({"case_id": "not-in-frozen-data"},),
        )


def test_w05_report_keeps_unrun_cases_in_denominators_and_marks_partial() -> None:
    cases = load_w05_development_cases()
    raw = {
        "B0": [
            {
                "case_id": cases[0].case_id,
                "status": "timeout",
                "http_status": None,
                "terminal_state": "UNKNOWN",
                "facts": [],
                "invariants": {},
                "side_effects": {"model_calls": 1},
                "usage": {"usage_status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                "elapsed_ms": None,
            }
        ],
        "B1": [],
    }

    report = build_w05_comparison_report(cases, raw, metadata={"provider_mode": "fake"})

    assert report["evaluation_status"] == "partial_or_not_run"
    b0 = report["profile_reports"]["B0"]
    assert b0["evaluation_status"] == "partial"
    assert b0["missing_case_ids"] == sorted(case.case_id for case in cases[1:])
    assert b0["metrics"]["case_count"] == 20
    assert b0["metrics"]["functional_success"]["denominator"] == 12
    assert b0["metrics"]["security_correct"]["denominator"] == 8
    assert b0["metrics"]["usage"]["known_total_tokens"] is None
    assert b0["metrics"]["usage"]["unknown_case_count"] == 20
    assert report["differences"]["functional_success_rate_b1_minus_b0"]["denominator"] == 12


def test_holdout_seal_verification_checks_commitments_without_decrypting(tmp_path) -> None:
    cases = load_w05_development_cases()
    retrieval_cases = load_w05_retrieval_cases()
    ciphertext = b"AES-GCM ciphertext fixture"
    ciphertext_path = tmp_path / "holdout.enc"
    ciphertext_path.write_bytes(ciphertext)
    task_hashes = [hashlib.sha256(f"sealed-task-{index}".encode()).hexdigest() for index in range(11)]
    retrieval_hashes = [hashlib.sha256(f"sealed-retrieval-{index}".encode()).hexdigest() for index in range(6)]
    metadata = {
        "schema_versions": {
            "task_cases": "state-case-v1",
            "retrieval_queries": "retrieval-case-v1",
            "case_manifest": "w05-sealed-case-manifest-v1",
        },
        "catalog_versions": {"business_fixture": "commerce-v1", "catalog": "catalog-v2", "knowledge": "knowledge-v1"},
        "paired_family_count": 1,
        "novel_pair_vs_C10": True,
        "paired_family_id_sha256": "c" * 64,
        "c10_development_pair_count": 6,
        "manifest_row_count": 18,
        "manifest_required_per_row_fields_present": True,
        "cipher": "AES-256-GCM",
        "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
        "plaintext_payload_sha256": "a" * 64,
        "case_manifest_sha256": "b" * 64,
        "split_counts": {
            "task_cases": {"functional": 8, "security": 4, "holdout": 12},
            "retrieval_queries": {"holdout": 6},
        },
        "family_id_commitments_sha256": {
            "task_cases": task_hashes,
            "retrieval_queries": retrieval_hashes,
        },
        "development_family_overlap_counts": {
            "task_cases": {"development_family_count": 14, "overlap_count": 0},
            "retrieval_queries": {"development_family_count": 12, "overlap_count": 0},
        },
    }
    metadata_path = tmp_path / "holdout.meta.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = verify_w05_holdout_seal(metadata_path, ciphertext_path, cases, retrieval_cases)

    assert result["status"] == "verified_without_decryption"
    assert result["payload_opened"] is False
    assert result["task_functional_count"] == 8
    assert result["task_security_count"] == 4
    assert result["task_family_overlap_count"] == 0
    assert result["paired_family_count"] == 1
    metadata["ciphertext_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(HoldoutSealError, match="ciphertext SHA256"):
        verify_w05_holdout_seal(metadata_path, ciphertext_path, cases, retrieval_cases)
