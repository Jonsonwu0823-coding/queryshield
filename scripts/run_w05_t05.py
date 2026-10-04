"""Run the first real evaluation of the sealed W05 holdout (T05 only)."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import check_w05 as checks  # noqa: E402
from queryshield.db.guarded import GuardedQueryExecutor  # noqa: E402
from queryshield.evaluation import (  # noqa: E402
    canonical_sha256,
    decrypt_and_load_w05_holdout,
    build_w05_comparison_report,
    load_w05_development_cases,
    load_w05_retrieval_cases,
    load_w05_versioned_gold,
    run_w05_stateful_suite,
    score_w05_retrieval,
    validate_w05_gold_source_versions,
)
from queryshield.providers.openai_compatible import OpenAICompatibleModel  # noqa: E402
from queryshield.providers.rerank import HttpRerankAdapter  # noqa: E402
from queryshield.evaluation.stateful_product import run_w05_product_case  # noqa: E402


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_REAL_CONFIG = (
    "QUERYSHIELD_DATABASE_URL",
    "QUERYSHIELD_MODEL_BASE_URL",
    "QUERYSHIELD_MODEL_API_KEY",
    "QUERYSHIELD_MODEL_NAME",
    "QUERYSHIELD_EMBEDDING_BASE_URL",
    "QUERYSHIELD_EMBEDDING_API_KEY",
    "QUERYSHIELD_EMBEDDING_MODEL_NAME",
    "QUERYSHIELD_EMBEDDING_MODEL_REVISION",
    "QUERYSHIELD_EMBEDDING_DIMENSIONS",
    "QUERYSHIELD_RERANK_URL",
    "QUERYSHIELD_RERANK_API_KEY",
    "QUERYSHIELD_RERANK_MODEL_NAME",
)
_W05_IDS = {
    *(f"W05-R{number:02}" for number in range(1, 8)),
    *(f"W05-X{number:02}" for number in range(1, 4)),
    "W05-DB01",
    "W05-FS01",
    "W05-FS02",
    *(f"W05-RT{number:02}" for number in range(1, 4)),
    *(f"W05-EN{number:02}" for number in range(1, 4)),
}
_FAKE_ONLY_REAL_NOT_APPLICABLE = {"W05-FS01", "W05-RT01", "W05-RT02", "W05-RT03", "W05-EN01"}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_manifest_bytes() -> bytes:
    excluded_dirs = {
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".vscode",
        ".idea",
        "build",
        "dist",
    }
    excluded_files = {"*.pyc", "*.pyo", "*.coverage", "*.code-workspace"}
    roots = ("src", "scripts", "migrations", "fixtures", "tests", "docs", "evals")
    paths: set[str] = set()
    for root in roots:
        root_path = PROJECT_ROOT / root
        if not root_path.is_dir():
            continue
        for path in root_path.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if any(part in excluded_dirs or part.endswith(".egg-info") for part in relative.split("/")):
                continue
            if any(path.match(pattern) for pattern in excluded_files):
                continue
            paths.add(relative)
    for top_level in ("pyproject.toml", "README.md", "requirements.lock", ".gitignore"):
        if (PROJECT_ROOT / top_level).is_file():
            paths.add(top_level)
    lines = [
        f"{hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()}  {relative}"
        for relative in sorted(paths)
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _verify_full_real_summary(path: Path, expected_manifest: str) -> dict[str, object]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("the linked full W05 Real summary cannot be loaded") from exc
    if not isinstance(summary, Mapping):
        raise ValueError("the linked full W05 Real summary is malformed")
    if (
        summary.get("suite") != "W05"
        or summary.get("mode") != "real"
        or summary.get("scope") != "full"
        or summary.get("overall_status") != "pass"
        or summary.get("source_manifest_sha256") != expected_manifest
        or summary.get("unexecuted_checks") != []
    ):
        raise ValueError("the linked full W05 Real run is not a passing full run for this source manifest")
    results = summary.get("checks")
    if type(results) is not list or len(results) != len(_W05_IDS):
        raise ValueError("the linked full W05 Real summary has no check records")
    result_by_id = {
        item.get("check_id"): item.get("status")
        for item in results
        if isinstance(item, Mapping) and type(item.get("check_id")) is str
    }
    if (
        len(result_by_id) != len(results)
        or set(result_by_id) != _W05_IDS
        or any(status not in {"pass", "not_applicable"} for status in result_by_id.values())
        or any(
            result_by_id[check_id] != ("not_applicable" if check_id in _FAKE_ONLY_REAL_NOT_APPLICABLE else "pass")
            for check_id in _W05_IDS
        )
    ):
        raise ValueError("the linked full W05 Real summary does not cover the 19 check IDs cleanly")
    required_ids = summary.get("all_required_check_ids")
    if type(required_ids) is not list or len(required_ids) != len(_W05_IDS) or set(required_ids) != _W05_IDS:
        raise ValueError("the linked full W05 Real summary required-ID declaration is incomplete")
    return {
        "status": "verified",
        "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_manifest_sha256": expected_manifest,
        "check_ids": len(result_by_id),
        "checks": result_by_id,
    }


def _recalculate_existing_t05(
    *,
    source_manifest_sha256: str,
    previous_evidence_dir: Path,
    key: str,
    metadata_path: Path,
    ciphertext_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    previous_evidence_dir = previous_evidence_dir.expanduser()
    if not previous_evidence_dir.is_absolute() or not previous_evidence_dir.is_dir():
        raise ValueError("existing T05 evidence directory must be absolute and readable")
    try:
        previous_summary = json.loads((previous_evidence_dir / "summary.json").read_text(encoding="utf-8"))
        previous_manifest = (previous_evidence_dir / "source-manifest.txt").read_bytes()
        task_raw = json.loads((previous_evidence_dir / "task-raw-by-profile.json").read_text(encoding="utf-8"))
        stored_task_report = json.loads((previous_evidence_dir / "task-comparison-report.json").read_text(encoding="utf-8"))
        stored_retrieval = json.loads((previous_evidence_dir / "retrieval-comparison-matrix.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("existing T05 evidence artifacts are incomplete or invalid") from exc
    if (
        not isinstance(previous_summary, Mapping)
        or previous_summary.get("status") != "complete"
        or previous_summary.get("payload_opened") is not True
        or previous_summary.get("source_manifest_sha256") != source_manifest_sha256
        or hashlib.sha256(previous_manifest).hexdigest() != source_manifest_sha256
        or not isinstance(task_raw, Mapping)
        or set(task_raw) != {"B0", "B1"}
        or not isinstance(stored_task_report, Mapping)
        or not isinstance(stored_retrieval, list)
    ):
        raise ValueError("existing T05 evidence does not bind to this complete candidate run")

    development_cases = load_w05_development_cases()
    development_retrieval_cases = load_w05_retrieval_cases()
    dataset = decrypt_and_load_w05_holdout(
        key,
        metadata_path=metadata_path,
        ciphertext_path=ciphertext_path,
        development_cases=development_cases,
        development_retrieval_cases=development_retrieval_cases,
    )
    old_commitments = previous_summary.get("dataset_commitments")
    if not isinstance(old_commitments, Mapping) or any(
        old_commitments.get(name) != expected
        for name, expected in (
            ("payload_sha256", dataset.payload_sha256),
            ("case_manifest_sha256", dataset.case_manifest_sha256),
            ("ciphertext_sha256", dataset.ciphertext_sha256),
            ("metadata_sha256", dataset.metadata_sha256),
            ("task_case_count", 12),
            ("retrieval_query_count", 6),
        )
    ):
        raise ValueError("existing T05 raw evidence is not bound to the current sealed dataset commitments")
    raw_by_profile = {
        profile: rows
        for profile, rows in task_raw.items()
        if profile in {"B0", "B1"} and type(rows) is list
    }
    if set(raw_by_profile) != {"B0", "B1"}:
        raise ValueError("existing T05 task raw records do not include both profiles")
    recalculated_task_report = build_w05_comparison_report(
        dataset.task_cases,
        raw_by_profile,
        metadata={
            "split": "holdout",
            "mode": "real",
            "source_manifest_sha256": source_manifest_sha256,
            "task_manifest_sha256": dataset.case_manifest_sha256,
            "payload_sha256": dataset.payload_sha256,
            "recalculated_from_existing_raw": True,
        },
    )
    stored_profiles = stored_task_report.get("profile_reports")
    if not isinstance(stored_profiles, Mapping):
        raise ValueError("existing T05 task report is missing profile metrics")
    task_profile_match = {}
    for profile in ("B0", "B1"):
        stored_profile = stored_profiles.get(profile)
        stored_metrics = stored_profile.get("metrics") if isinstance(stored_profile, Mapping) else None
        task_profile_match[profile] = stored_metrics == recalculated_task_report["profile_reports"][profile]["metrics"]

    expected_configs = {
        "retrieval-keyword-synonym-real-v1": "keyword-synonym",
        "B1-hybrid-real-v1": "hybrid",
        "B1-hybrid-rerank-real-v1": "hybrid+rerank",
    }
    retrieval_match: dict[str, bool] = {}
    recalculated_retrieval: dict[str, object] = {}
    if len(stored_retrieval) != 3:
        raise ValueError("existing T05 retrieval evidence must contain three configurations")
    for config in stored_retrieval:
        if not isinstance(config, Mapping):
            raise ValueError("existing T05 retrieval configuration record is malformed")
        config_id = config.get("config_id")
        if config_id not in expected_configs or config_id in recalculated_retrieval:
            raise ValueError("existing T05 retrieval evidence has missing or duplicate configurations")
        if config.get("strategy") != expected_configs[config_id]:
            raise ValueError("existing T05 retrieval strategy identity is inconsistent")
        records = config.get("raw_records")
        if type(records) is not list:
            raise ValueError("existing T05 retrieval raw records are missing")
        metrics = score_w05_retrieval(dataset.retrieval_cases, records, dataset.gold_by_family_id, top_k=3)
        recalculated_retrieval[config_id] = metrics
        retrieval_match[config_id] = config.get("metrics") == metrics
    if set(recalculated_retrieval) != set(expected_configs):
        raise ValueError("existing T05 retrieval evidence is missing a required configuration")

    metrics_match = all(task_profile_match.values()) and all(retrieval_match.values())
    result = {
        "status": "verified" if metrics_match else "failed",
        "source_manifest_sha256": source_manifest_sha256,
        "dataset_commitments": {
            "payload_sha256": dataset.payload_sha256,
            "case_manifest_sha256": dataset.case_manifest_sha256,
            "ciphertext_sha256": dataset.ciphertext_sha256,
            "metadata_sha256": dataset.metadata_sha256,
            "task_case_count": len(dataset.task_cases),
            "retrieval_query_count": len(dataset.retrieval_cases),
        },
        "task_metrics_match_stored_report": task_profile_match,
        "recalculated_task_report": recalculated_task_report,
        "retrieval_metrics_match_stored_report": retrieval_match,
        "recalculated_retrieval_metrics": recalculated_retrieval,
        "provider_calls_repeated": False,
        "plaintext_saved": False,
    }
    _write_json(output_dir / "independent-recalculation.json", result)
    return result


def _failure_location(exc: BaseException) -> list[dict[str, object]]:
    root = PROJECT_ROOT.resolve()
    result = []
    for frame in traceback.extract_tb(exc.__traceback__)[-8:]:
        file_path = Path(frame.filename)
        try:
            display = file_path.resolve().relative_to(root).as_posix()
        except (OSError, RuntimeError, ValueError):
            display = file_path.name
        result.append({"file": display, "line": frame.lineno, "function": frame.name})
    return result


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run W05 T05 on the sealed holdout after candidate freeze.")
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--full-real-summary", type=Path)
    parser.add_argument("--recalculate-from", type=Path)
    parser.add_argument("--metadata", type=Path, default=checks.HOLDOUT_METADATA)
    parser.add_argument("--ciphertext", type=Path, default=checks.HOLDOUT_CIPHERTEXT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    evidence_dir = args.evidence_dir.expanduser()
    if not evidence_dir.is_absolute():
        print("W05-T05 requires a new absolute EvidenceDir.")
        return 2
    if not _SHA256_RE.fullmatch(args.candidate_manifest_sha256):
        print("W05-T05 candidate manifest digest must be lowercase SHA256.")
        return 2
    try:
        evidence_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("W05-T05 EvidenceDir already exists; choose a new absolute directory.")
        return 2
    except OSError:
        print("W05-T05 EvidenceDir could not be created.")
        return 2

    summary: dict[str, object] = {
        "suite": "W05-T05",
        "mode": "real",
        "status": "blocked",
        "payload_opened": False,
        "source_manifest_sha256": None,
        "full_real_source_summary": None,
        "missing_configuration_names": [],
    }
    source_manifest = _source_manifest_bytes()
    source_manifest_path = evidence_dir / "source-manifest.txt"
    source_manifest_path.write_bytes(source_manifest)
    source_digest = hashlib.sha256(source_manifest).hexdigest()
    summary["source_manifest_sha256"] = source_digest
    if source_digest != args.candidate_manifest_sha256:
        summary["reason"] = "current source does not match the frozen candidate manifest"
        _write_json(evidence_dir / "summary.json", summary)
        print("W05-T05 blocked: current source differs from the supplied frozen candidate manifest.")
        return 2

    if args.recalculate_from is not None:
        print("W05-T05 recalculation uses the unseal key only in memory and does not repeat provider calls.")
        key = getpass.getpass("Enter the original T05 unseal key from its custodian (hidden): ")
        if not key.strip():
            summary["reason"] = "the independent evaluator did not receive an unseal key"
            _write_json(evidence_dir / "summary.json", summary)
            print("W05-T05 recalculation blocked: unseal key was not entered.")
            return 2
        try:
            recalculation = _recalculate_existing_t05(
                source_manifest_sha256=source_digest,
                previous_evidence_dir=args.recalculate_from,
                key=key,
                metadata_path=args.metadata,
                ciphertext_path=args.ciphertext,
                output_dir=evidence_dir,
            )
        except Exception as exc:
            summary.update(
                {
                    "status": "failed",
                    "reason": "existing T05 raw evidence could not be independently recalculated",
                    "failure_type": type(exc).__name__,
                    "failure_location": _failure_location(exc),
                }
            )
            _write_json(evidence_dir / "summary.json", summary)
            print("W05-T05 recalculation failed; provider calls were not repeated.")
            return 1
        finally:
            key = ""
        summary.update(
            {
                "status": recalculation["status"],
                "payload_opened": True,
                "recalculation_source": str(args.recalculate_from),
                "provider_calls_repeated": False,
                "plaintext_saved": False,
                "metrics_match_stored_evidence": recalculation["status"] == "verified",
            }
        )
        _write_json(evidence_dir / "summary.json", summary)
        print(f"W05-T05 recalculation {summary['status']}; evidence={evidence_dir}")
        return 0 if summary["status"] == "verified" else 1

    if args.full_real_summary is None:
        summary["reason"] = "a matching complete W05 Real summary is required before the first holdout run"
        _write_json(evidence_dir / "summary.json", summary)
        print("W05-T05 blocked: supply the matching full Real summary.")
        return 2

    try:
        full_real = _verify_full_real_summary(args.full_real_summary, source_digest)
    except ValueError as exc:
        summary["reason"] = str(exc)
        _write_json(evidence_dir / "summary.json", summary)
        print("W05-T05 blocked: linked full Real evidence does not match this frozen candidate.")
        return 2
    summary["full_real_source_summary"] = full_real

    missing = [name for name in _REQUIRED_REAL_CONFIG if not os.environ.get(name, "").strip()]
    if missing:
        summary["missing_configuration_names"] = missing
        summary["reason"] = "real W05 configuration is not present in this process"
        _write_json(evidence_dir / "summary.json", summary)
        print("W05-T05 blocked; missing configuration names: " + ", ".join(missing))
        return 2

    database = checks._check_db01(evidence_dir / "preflight-db")
    _write_json(evidence_dir / "database-preflight.json", database)
    if database.get("status") != "pass":
        summary["database_preflight_status"] = database.get("status")
        summary["reason"] = "required real PostgreSQL preflight did not pass"
        _write_json(evidence_dir / "summary.json", summary)
        print("W05-T05 blocked/failed at PostgreSQL preflight; no holdout key was requested.")
        return 2 if database.get("status") == "blocked" else 1

    development_cases = load_w05_development_cases()
    development_retrieval_cases = load_w05_retrieval_cases()
    try:
        seal = checks.verify_w05_holdout_seal(
            args.metadata,
            args.ciphertext,
            development_cases,
            development_retrieval_cases,
        )
    except Exception:
        summary["reason"] = "metadata-only holdout seal verification failed"
        _write_json(evidence_dir / "summary.json", summary)
        print("W05-T05 failed metadata-only seal verification; no holdout key was requested.")
        return 1

    print("W05-T05 is ready to open the sealed dataset in memory; no key, query, or gold is written to evidence.")
    key = getpass.getpass("Enter the T05 unseal key supplied directly by the holdout custodian (hidden, base64 or hex): ")
    if not key.strip():
        summary["reason"] = "the independent evaluator did not receive an unseal key"
        _write_json(evidence_dir / "summary.json", summary)
        print("W05-T05 blocked: unseal key was not entered.")
        return 2

    try:
        dataset = decrypt_and_load_w05_holdout(
            key,
            metadata_path=args.metadata,
            ciphertext_path=args.ciphertext,
            development_cases=development_cases,
            development_retrieval_cases=development_retrieval_cases,
        )
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "payload_opened": False,
                "reason": "decryption or strict payload validation failed",
                "failure_type": type(exc).__name__,
                "failure_location": _failure_location(exc),
                "seal_verification": seal,
            }
        )
        _write_json(evidence_dir / "summary.json", summary)
        print("W05-T05 failed decryption/schema validation; no plaintext was saved.")
        return 1
    finally:
        key = ""

    try:
        active_versions = checks._active_source_versions()
        gold_validation = validate_w05_gold_source_versions(dataset.gold_by_family_id, active_versions)
        reranker = HttpRerankAdapter.from_env()
        runtime = checks._build_retrieval_runtime("real", retrieval_cases=dataset.retrieval_cases)
        model = checks._RecordingModelAdapter(OpenAICompatibleModel.from_env())
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "payload_opened": True,
                "reason": "real provider or catalog setup failed before evaluation",
                "failure_type": type(exc).__name__,
                "failure_location": _failure_location(exc),
                "dataset_commitments": {
                    "task_case_count": len(dataset.task_cases),
                    "retrieval_query_count": len(dataset.retrieval_cases),
                    "case_manifest_sha256": dataset.case_manifest_sha256,
                    "payload_sha256": dataset.payload_sha256,
                },
            }
        )
        _write_json(evidence_dir / "summary.json", summary)
        print("W05-T05 failed provider/catalog preparation; no plaintext or secret was printed.")
        return 1

    def recording_executor_factory(records):
        return checks._RecordingQueryExecutor(GuardedQueryExecutor(), records)

    def run_profile(case, profile, run_id):
        result = run_w05_product_case(
            case,
            profile,
            run_id,
            mode="real",
            model=model,
            retriever=runtime[4],
            recording_executor_factory=recording_executor_factory,
        )
        observation = result.get("observation")
        if isinstance(observation, Mapping):
            call_ids = set(observation.get("model_call_ids", ()))
            result = dict(result)
            result["model_call_records"] = [
                dict(record) for record in model.records if record.get("model_call_id") in call_ids
            ]
            result["model_provider"] = model.provider
            result["model_name"] = model.model
        return result

    try:
        task_suite = run_w05_stateful_suite(
            dataset.task_cases,
            run_profile,
            metadata={
                "split": "holdout",
                "mode": "real",
                "source_manifest_sha256": source_digest,
                "task_manifest_sha256": dataset.case_manifest_sha256,
                "payload_sha256": dataset.payload_sha256,
            },
        )
        retrieval_matrix = checks._run_retrieval_matrix(
            "real",
            runtime=runtime,
            reranker=reranker,
            cases=dataset.retrieval_cases,
            gold_by_family_id=dataset.gold_by_family_id,
        )
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "payload_opened": True,
                "reason": "T05 product execution failed before all reports were finalized",
                "failure_type": type(exc).__name__,
                "failure_location": _failure_location(exc),
                "dataset_commitments": {
                    "task_case_count": len(dataset.task_cases),
                    "retrieval_query_count": len(dataset.retrieval_cases),
                    "case_manifest_sha256": dataset.case_manifest_sha256,
                    "payload_sha256": dataset.payload_sha256,
                },
            }
        )
        _write_json(evidence_dir / "summary.json", summary)
        print("W05-T05 failed during product execution; partial evidence was retained locally.")
        return 1

    checks._write_json(
        evidence_dir / "task-raw-by-profile.json",
        task_suite["raw_records"],
    )
    checks._write_json(evidence_dir / "task-comparison-report.json", task_suite["report"])
    checks._write_json(evidence_dir / "chat-provider-calls.json", model.records)
    checks._write_json(
        evidence_dir / "retrieval-comparison-matrix.json",
        retrieval_matrix,
    )
    checks._write_json(
        evidence_dir / "shared-retrieval-index-preparation.json",
        checks._retrieval_shared_index_preparation(
            runtime,
            configuration_ids=[
                "retrieval-keyword-synonym-real-v1",
                "B1-hybrid-real-v1",
                "B1-hybrid-rerank-real-v1",
            ],
        ),
    )
    task_report = task_suite["report"]
    profile_counts = {
        profile: {
            "execution_complete": task_report["profile_reports"][profile]["execution_complete"],
            "case_count": task_report["profile_reports"][profile]["metrics"]["case_count"],
            "functional_success": task_report["profile_reports"][profile]["metrics"]["functional_success"],
            "security_correct": task_report["profile_reports"][profile]["metrics"]["security_correct"],
            "unknown_usage_cases": task_report["profile_reports"][profile]["metrics"]["usage"]["unknown_case_count"],
            "known_total_tokens": task_report["profile_reports"][profile]["metrics"]["usage"]["known_total_tokens"],
        }
        for profile in ("B0", "B1")
    }
    retrieval_counts = {
        configuration["config_id"]: {
            "case_count": configuration["metrics"]["case_count"],
            "hit_at_3": configuration["metrics"]["hit_at_3"],
            "recall_at_3_macro": configuration["metrics"]["recall_at_3_macro"],
            "empty_hit_count": configuration["metrics"]["empty_hit_count"],
            "failed_count": configuration["metrics"]["failed_count"],
        }
        for configuration in retrieval_matrix
    }
    complete = (
        all(value["execution_complete"] and value["case_count"] == 12 for value in profile_counts.values())
        and len(retrieval_matrix) == 3
        and all(value["case_count"] == 6 for value in retrieval_counts.values())
    )
    summary.update(
        {
            "status": "complete" if complete else "partial",
            "payload_opened": True,
            "candidate_manifest_sha256": source_digest,
            "source_manifest_sha256": source_digest,
            "full_real_source_summary": full_real,
            "seal_verification": seal,
            "dataset_commitments": {
                "payload_sha256": dataset.payload_sha256,
                "case_manifest_sha256": dataset.case_manifest_sha256,
                "ciphertext_sha256": dataset.ciphertext_sha256,
                "metadata_sha256": dataset.metadata_sha256,
                "task_case_count": len(dataset.task_cases),
                "task_functional_count": 8,
                "task_security_count": 4,
                "retrieval_query_count": len(dataset.retrieval_cases),
                "gold_source_validation": gold_validation,
            },
            "provider_profile": {
                "chat_model": model.model,
                "embedding_model": runtime[2].index.model,
                "embedding_revision": runtime[2].index.model_revision,
                "embedding_dimensions": runtime[2].index.dimensions,
                "rerank_model": reranker.model,
                "configuration_sha256": canonical_sha256(
                    {
                        "chat_model": model.model,
                        "chat_endpoint_sha256": hashlib.sha256(os.environ["QUERYSHIELD_MODEL_BASE_URL"].rstrip("/").encode("utf-8")).hexdigest(),
                        "embedding_model": runtime[2].index.model,
                        "embedding_revision": runtime[2].index.model_revision,
                        "embedding_dimensions": runtime[2].index.dimensions,
                        "embedding_endpoint_sha256": hashlib.sha256(os.environ["QUERYSHIELD_EMBEDDING_BASE_URL"].rstrip("/").encode("utf-8")).hexdigest(),
                        "rerank_model": reranker.model,
                        "rerank_endpoint_sha256": hashlib.sha256(os.environ["QUERYSHIELD_RERANK_URL"].rstrip("/").encode("utf-8")).hexdigest(),
                    }
                ),
                "credentials_recorded": False,
            },
            "task_metrics": profile_counts,
            "retrieval_metrics": retrieval_counts,
            "raw_evidence_paths": [
                "task-raw-by-profile.json",
                "task-comparison-report.json",
                "chat-provider-calls.json",
                "retrieval-comparison-matrix.json",
                "shared-retrieval-index-preparation.json",
            ],
            "failure_or_unknown_rows_retained": True,
            "unknown_usage_is_null": True,
            "interpretation_boundary": "one independent frozen T05 run; descriptive finite-set evidence only",
        }
    )
    _write_json(evidence_dir / "summary.json", summary)
    print(
        "W05-T05 "
        + summary["status"]
        + f"; task_cases=12/profile, retrieval_queries=6 x 3 configs; evidence={evidence_dir}"
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
