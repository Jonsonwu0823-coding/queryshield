from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import pytest

from queryshield.evaluation import (
    W05HoldoutPayloadError,
    canonical_sha256,
    decrypt_and_load_w05_holdout,
    load_w05_development_cases,
    load_w05_retrieval_cases,
    load_w05_versioned_gold,
)
from queryshield.evaluation.stateful_replay import run_w05_stateful_suite


def test_synthetic_holdout_drives_generic_task_and_three_strategy_retrieval_runners(tmp_path) -> None:
    payload, _rows = _synthetic_payload()
    metadata_path, ciphertext_path, key = _seal_synthetic_payload(tmp_path, payload)
    dataset = decrypt_and_load_w05_holdout(
        key,
        metadata_path=metadata_path,
        ciphertext_path=ciphertext_path,
        development_cases=load_w05_development_cases(),
        development_retrieval_cases=load_w05_retrieval_cases(),
    )

    def failed_profile(case, profile, run_id):
        return {
            "observation": {
                "status": "failed",
                "http_status": 502,
                "terminal_state": "FAILED",
                "facts": [],
                "invariants": {},
                "side_effects": {
                    "model_calls": 0,
                    "readonly_queries": 0,
                    "fact_count": 0,
                    "write_statements": 0,
                    "cross_tenant_rows": 0,
                    "unauthorized_facts": 0,
                },
                "usage": {
                    "usage_status": "unknown",
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
                "elapsed_ms": 0,
                "execution_status": "executed",
                "error_code": "synthetic_control_failure",
            }
        }

    task_suite = run_w05_stateful_suite(
        dataset.task_cases,
        failed_profile,
        metadata={"split": "synthetic-holdout", "mode": "fake"},
    )
    assert task_suite["report"]["evaluation_status"] == "complete"
    for profile in ("B0", "B1"):
        report = task_suite["report"]["profile_reports"][profile]
        assert report["metrics"]["case_count"] == 12
        assert report["metrics"]["functional_success"]["denominator"] == 8
        assert report["metrics"]["security_correct"]["denominator"] == 4
        assert report["metrics"]["usage"]["unknown_case_count"] == 12

    import sys

    project = Path(__file__).resolve().parents[1]
    scripts = str(project / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import check_w05

    runtime = check_w05._build_retrieval_runtime("fake", retrieval_cases=dataset.retrieval_cases)
    matrix = check_w05._run_retrieval_matrix(
        "fake",
        runtime=runtime,
        reranker=check_w05._OverlapFakeReranker(),
        cases=dataset.retrieval_cases,
        gold_by_family_id=dataset.gold_by_family_id,
    )
    assert {item["strategy"] for item in matrix} == {"keyword-synonym", "hybrid", "hybrid+rerank"}
    assert all(item["metrics"]["case_count"] == 6 for item in matrix)
    assert all(item["metrics"]["hit_at_3"]["denominator"] == 6 for item in matrix)


def test_synthetic_t05_raw_can_be_recalculated_without_repeating_provider_calls(tmp_path) -> None:
    import sys

    project = Path(__file__).resolve().parents[1]
    scripts = str(project / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import check_w05
    import run_w05_t05

    payload, _rows = _synthetic_payload()
    metadata_path, ciphertext_path, key = _seal_synthetic_payload(tmp_path, payload)
    dataset = decrypt_and_load_w05_holdout(
        key,
        metadata_path=metadata_path,
        ciphertext_path=ciphertext_path,
        development_cases=load_w05_development_cases(),
        development_retrieval_cases=load_w05_retrieval_cases(),
    )

    def failed_profile(case, profile, run_id):
        return {
            "observation": {
                "status": "failed",
                "http_status": 502,
                "terminal_state": "FAILED",
                "facts": [],
                "invariants": {},
                "side_effects": {"model_calls": 0, "readonly_queries": 0, "fact_count": 0},
                "usage": {
                    "usage_status": "unknown",
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
                "elapsed_ms": 0,
                "execution_status": "executed",
                "error_code": "synthetic_control_failure",
            }
        }

    task_suite = run_w05_stateful_suite(
        dataset.task_cases,
        failed_profile,
        metadata={"split": "holdout", "mode": "fake"},
    )
    runtime = check_w05._build_retrieval_runtime("fake", retrieval_cases=dataset.retrieval_cases)
    retrieval_matrix = check_w05._run_retrieval_matrix(
        "fake",
        runtime=runtime,
        reranker=check_w05._OverlapFakeReranker(),
        cases=dataset.retrieval_cases,
        gold_by_family_id=dataset.gold_by_family_id,
    )

    previous = tmp_path / "synthetic-t05-raw"
    previous.mkdir()
    source_manifest = run_w05_t05._source_manifest_bytes()
    source_sha = hashlib.sha256(source_manifest).hexdigest()
    (previous / "source-manifest.txt").write_bytes(source_manifest)
    (previous / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "payload_opened": True,
                "source_manifest_sha256": source_sha,
                "dataset_commitments": {
                    "payload_sha256": dataset.payload_sha256,
                    "case_manifest_sha256": dataset.case_manifest_sha256,
                    "ciphertext_sha256": dataset.ciphertext_sha256,
                    "metadata_sha256": dataset.metadata_sha256,
                    "task_case_count": 12,
                    "retrieval_query_count": 6,
                },
            }
        ),
        encoding="utf-8",
    )
    (previous / "task-raw-by-profile.json").write_text(json.dumps(task_suite["raw_records"]), encoding="utf-8")
    (previous / "task-comparison-report.json").write_text(json.dumps(task_suite["report"]), encoding="utf-8")
    t05_retrieval = []
    for config in retrieval_matrix:
        copied = dict(config)
        copied["config_id"] = config["config_id"].replace("fake", "real")
        t05_retrieval.append(copied)
    (previous / "retrieval-comparison-matrix.json").write_text(json.dumps(t05_retrieval), encoding="utf-8")

    result = run_w05_t05._recalculate_existing_t05(
        source_manifest_sha256=source_sha,
        previous_evidence_dir=previous,
        key=key,
        metadata_path=metadata_path,
        ciphertext_path=ciphertext_path,
        output_dir=tmp_path / "recalculation",
    )

    assert result["status"] == "verified"
    assert result["provider_calls_repeated"] is False
    assert all(result["task_metrics_match_stored_report"].values())
    assert all(result["retrieval_metrics_match_stored_report"].values())


def test_t05_requires_all_19_current_source_real_check_ids_before_unseal(tmp_path) -> None:
    import sys

    project = Path(__file__).resolve().parents[1]
    scripts = str(project / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_w05_t05

    expected_manifest = "a" * 64
    ids = sorted(run_w05_t05._W05_IDS)
    summary = {
        "suite": "W05",
        "mode": "real",
        "scope": "full",
        "overall_status": "pass",
        "source_manifest_sha256": expected_manifest,
        "unexecuted_checks": [],
        "all_required_check_ids": ids,
        "checks": [
            {
                "check_id": check_id,
                "status": "not_applicable"
                if check_id in run_w05_t05._FAKE_ONLY_REAL_NOT_APPLICABLE
                else "pass",
            }
            for check_id in ids
        ],
    }
    summary_path = tmp_path / "full-real-summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = run_w05_t05._verify_full_real_summary(summary_path, expected_manifest)

    assert result["status"] == "verified"
    assert result["check_ids"] == 19
    summary_path.write_bytes(b"\xef\xbb\xbf" + json.dumps(summary).encode("utf-8"))
    bom_result = run_w05_t05._verify_full_real_summary(summary_path, expected_manifest)
    assert bom_result["status"] == "verified"
    assert bom_result["summary_sha256"] == hashlib.sha256(summary_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source manifest"):
        run_w05_t05._verify_full_real_summary(summary_path, "b" * 64)


@pytest.mark.parametrize(
    ("invalid_case", "expected_error"),
    [
        ("partial_scope", "not a passing full run"),
        ("failed_overall_status", "not a passing full run"),
        ("failed_check_status", "does not cover the 19 check IDs cleanly"),
        ("missing_check_record", "has no check records"),
        ("missing_required_id", "required-ID declaration is incomplete"),
        ("unexecuted_check", "not a passing full run"),
    ],
)
def test_t05_bom_aware_reader_preserves_full_real_summary_rejections(
    tmp_path, invalid_case: str, expected_error: str
) -> None:
    import sys

    project = Path(__file__).resolve().parents[1]
    scripts = str(project / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_w05_t05

    expected_manifest = "a" * 64
    ids = sorted(run_w05_t05._W05_IDS)
    summary = {
        "suite": "W05",
        "mode": "real",
        "scope": "full",
        "overall_status": "pass",
        "source_manifest_sha256": expected_manifest,
        "unexecuted_checks": [],
        "all_required_check_ids": ids,
        "checks": [
            {
                "check_id": check_id,
                "status": "not_applicable"
                if check_id in run_w05_t05._FAKE_ONLY_REAL_NOT_APPLICABLE
                else "pass",
            }
            for check_id in ids
        ],
    }
    if invalid_case == "partial_scope":
        summary["scope"] = "partial"
    elif invalid_case == "failed_overall_status":
        summary["overall_status"] = "fail"
    elif invalid_case == "failed_check_status":
        summary["checks"][0]["status"] = "fail"
    elif invalid_case == "missing_check_record":
        summary["checks"].pop()
    elif invalid_case == "missing_required_id":
        summary["all_required_check_ids"].pop()
    elif invalid_case == "unexecuted_check":
        summary["unexecuted_checks"] = [ids[0]]
    else:
        raise AssertionError(f"unhandled test case: {invalid_case}")

    summary_path = tmp_path / "invalid-full-real-summary.json"
    summary_path.write_bytes(b"\xef\xbb\xbf" + json.dumps(summary).encode("utf-8"))

    with pytest.raises(ValueError, match=expected_error):
        run_w05_t05._verify_full_real_summary(summary_path, expected_manifest)


def _synthetic_payload() -> tuple[dict[str, object], list[dict[str, object]]]:
    dev_tasks = load_w05_development_cases()
    functional = [case for case in dev_tasks if case.classification == "functional"][:8]
    security = [case for case in dev_tasks if case.classification == "security"][:4]
    selected = [*functional, *security]
    task_cases: list[dict[str, object]] = []
    for index, source in enumerate(selected):
        case = json.loads(json.dumps(source.case))
        case["case_id"] = f"synthetic-holdout-task-{index:02d}"
        case["family_id"] = (
            "synthetic-new-paired-family"
            if index in {0, 8}
            else f"synthetic-holdout-family-{index:02d}"
        )
        case["split"] = "holdout"
        classification = "functional" if index < 8 else "security"
        task_cases.append(
            {"case": case, "classification": classification, "critical_question_id": None}
        )

    source_queries = load_w05_retrieval_cases()[:6]
    dev_gold = load_w05_versioned_gold()
    retrieval_cases: list[dict[str, object]] = []
    gold: dict[str, list[dict[str, str]]] = {}
    for index, source in enumerate(source_queries):
        family_id = f"synthetic-holdout-retrieval-{index:02d}"
        retrieval_cases.append(
            {
                "query": f"Synthetic public query {index}",
                "relevant_source_ids": list(source.relevant_source_ids),
                "family_id": family_id,
                "split": "holdout",
                "catalog_version": "catalog-v2",
            }
        )
        gold[family_id] = [
            {"source_id": source_id, "version": version}
            for source_id, version in dev_gold[source.family_id]
        ]
    payload: dict[str, object] = {
        "schema_version": "w05-holdout-payload-v1",
        "task_cases": task_cases,
        "retrieval_queries": {
            "schema_version": "w05-retrieval-evaluation-v1",
            "catalog_version": "catalog-v2",
            "cases": retrieval_cases,
        },
        "retrieval_gold": {
            "schema_version": "w05-retrieval-gold-v1",
            "catalog_version": "catalog-v2",
            "gold_by_family_id": gold,
        },
    }
    rows: list[dict[str, object]] = []
    for item in task_cases:
        case = item["case"]
        rows.append(
            {
                "artifact_type": "task_case",
                "case_id": case["case_id"],
                "family_id": case["family_id"],
                "split": "holdout",
                "classification": item["classification"],
                "input_sha256": canonical_sha256(
                    {"initial": case["initial"], "action": case["action"]}
                ),
                "expected_sha256": canonical_sha256(case["expected"]),
            }
        )
    for case in retrieval_cases:
        rows.append(
            {
                "artifact_type": "retrieval_query",
                "case_id": case["family_id"],
                "family_id": case["family_id"],
                "split": "holdout",
                "classification": "retrieval",
                "input_sha256": canonical_sha256(case["query"]),
                "expected_sha256": canonical_sha256(gold[case["family_id"]]),
            }
        )
    rows.sort(key=lambda row: (row["artifact_type"], row["case_id"]))
    manifest = {"schema_version": "w05-sealed-case-manifest-v1", "rows": rows}
    payload["case_manifest"] = manifest
    return payload, rows


def _seal_synthetic_payload(tmp_path: Path, payload: dict[str, object]) -> tuple[Path, Path, str]:
    key = AESGCM.generate_key(bit_length=256)
    nonce = bytes(range(12))
    plaintext = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encrypted = AESGCM(key).encrypt(nonce, plaintext, None)
    ciphertext, tag = encrypted[:-16], encrypted[-16:]
    metadata = {
        "schema_versions": {
            "task_cases": "state-case-v1",
            "retrieval_queries": "retrieval-case-v1",
            "case_manifest": "w05-sealed-case-manifest-v1",
        },
        "catalog_versions": {
            "business_fixture": "commerce-v1",
            "catalog": "catalog-v2",
            "knowledge": "knowledge-v1",
        },
        "paired_family_count": 1,
        "novel_pair_vs_C10": True,
        "paired_family_id_sha256": hashlib.sha256(
            b"synthetic-new-paired-family"
        ).hexdigest(),
        "c10_development_pair_count": 6,
        "manifest_row_count": 18,
        "manifest_required_per_row_fields_present": True,
        "cipher": "AES-256-GCM",
        "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
        "plaintext_payload_sha256": hashlib.sha256(plaintext).hexdigest(),
        "case_manifest_sha256": canonical_sha256(payload["case_manifest"]),
        "split_counts": {
            "task_cases": {"functional": 8, "security": 4, "holdout": 12},
            "retrieval_queries": {"holdout": 6},
        },
        "family_id_commitments_sha256": {
            "task_cases": sorted(
                {
                    hashlib.sha256(row["family_id"].encode("utf-8")).hexdigest()
                    for row in payload["case_manifest"]["rows"]
                    if row["artifact_type"] == "task_case"
                }
            ),
            "retrieval_queries": sorted(
                {
                    hashlib.sha256(row["family_id"].encode("utf-8")).hexdigest()
                    for row in payload["case_manifest"]["rows"]
                    if row["artifact_type"] == "retrieval_query"
                }
            ),
        },
        "development_family_overlap_counts": {
            "task_cases": {"development_family_count": 14, "overlap_count": 0},
            "retrieval_queries": {"development_family_count": 12, "overlap_count": 0},
        },
        "nonce_base64": base64.b64encode(nonce).decode("ascii"),
        "tag_base64": base64.b64encode(tag).decode("ascii"),
    }
    metadata_path = tmp_path / "synthetic-holdout.meta.json"
    ciphertext_path = tmp_path / "synthetic-holdout.enc"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    ciphertext_path.write_bytes(ciphertext)
    return metadata_path, ciphertext_path, base64.b64encode(key).decode("ascii")


def test_synthetic_sealed_holdout_loads_in_memory_and_checks_all_18_manifest_rows(tmp_path) -> None:
    payload, _rows = _synthetic_payload()
    metadata_path, ciphertext_path, key = _seal_synthetic_payload(tmp_path, payload)

    loaded = decrypt_and_load_w05_holdout(
        key,
        metadata_path=metadata_path,
        ciphertext_path=ciphertext_path,
        development_cases=load_w05_development_cases(),
        development_retrieval_cases=load_w05_retrieval_cases(),
    )

    assert len(loaded.task_cases) == 12
    assert Counter(case.classification for case in loaded.task_cases) == {"functional": 8, "security": 4}
    assert len(loaded.retrieval_cases) == 6
    assert set(loaded.gold_by_family_id) == {case.family_id for case in loaded.retrieval_cases}
    assert loaded.case_manifest_sha256 == json.loads(metadata_path.read_text(encoding="utf-8"))["case_manifest_sha256"]
    assert set(path.name for path in tmp_path.iterdir()) == {metadata_path.name, ciphertext_path.name}


def test_synthetic_sealed_holdout_rejects_wrong_key_without_emitting_plaintext(tmp_path) -> None:
    payload, _rows = _synthetic_payload()
    metadata_path, ciphertext_path, _key = _seal_synthetic_payload(tmp_path, payload)
    wrong_key = base64.b64encode(bytes([7]) * 32).decode("ascii")

    with pytest.raises(W05HoldoutPayloadError, match="authentication failed"):
        decrypt_and_load_w05_holdout(
            wrong_key,
            metadata_path=metadata_path,
            ciphertext_path=ciphertext_path,
            development_cases=load_w05_development_cases(),
            development_retrieval_cases=load_w05_retrieval_cases(),
        )
    assert len(list(tmp_path.iterdir())) == 2


def test_synthetic_sealed_holdout_rejects_manifest_case_mismatch(tmp_path) -> None:
    payload, _rows = _synthetic_payload()
    payload["case_manifest"]["rows"][0]["expected_sha256"] = "0" * 64
    metadata_path, ciphertext_path, key = _seal_synthetic_payload(tmp_path, payload)

    with pytest.raises(W05HoldoutPayloadError, match="manifest rows do not match"):
        decrypt_and_load_w05_holdout(
            key,
            metadata_path=metadata_path,
            ciphertext_path=ciphertext_path,
            development_cases=load_w05_development_cases(),
            development_retrieval_cases=load_w05_retrieval_cases(),
        )
