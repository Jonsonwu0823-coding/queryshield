"""Versioned retrieval gold and denominator-preserving metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path


SOURCE_RETRIEVAL_SET_VERSION = "w05-retrieval-evaluation-v1"
SOURCE_RETRIEVAL_GOLD_VERSION = "w05-retrieval-gold-v1"
EXPECTED_CATALOG_VERSION = "catalog-v2"


class SourceRetrievalCaseError(ValueError):
    """A retrieval case or versioned gold manifest is invalid."""


@dataclass(frozen=True)
class SourceRetrievalCase:
    query: str
    relevant_source_ids: tuple[str, ...]
    family_id: str
    split: str
    catalog_version: str


def _object(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise SourceRetrievalCaseError(f"{where} must be an object")
    return value


def _read_json(path: Path, *, where: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceRetrievalCaseError(f"cannot load {where}: {path}") from exc
    return _object(value, where=where)


def load_source_retrieval_cases(
    path: str | Path | None = None,
) -> tuple[SourceRetrievalCase, ...]:
    fixture_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[3] / "evals" / "development" / "retrieval-cases-v1.json"
    )
    document = _read_json(fixture_path, where="retrieval fixture")
    if set(document) != {"schema_version", "catalog_version", "cases"}:
        raise SourceRetrievalCaseError("retrieval root fields mismatch")
    if document.get("schema_version") != SOURCE_RETRIEVAL_SET_VERSION:
        raise SourceRetrievalCaseError("unsupported retrieval set schema")
    if document.get("catalog_version") != EXPECTED_CATALOG_VERSION:
        raise SourceRetrievalCaseError("retrieval set must bind catalog-v2")
    raw_cases = document.get("cases")
    if type(raw_cases) is not list or len(raw_cases) != 12:
        raise SourceRetrievalCaseError("Retrieval evaluation requires exactly twelve development retrieval cases")
    parsed: list[SourceRetrievalCase] = []
    families: set[str] = set()
    required = {"query", "relevant_source_ids", "family_id", "split", "catalog_version"}
    for index, raw in enumerate(raw_cases):
        case = _object(raw, where=f"cases[{index}]")
        if set(case) != required:
            raise SourceRetrievalCaseError(f"cases[{index}] fields mismatch")
        for name in ("query", "family_id", "split", "catalog_version"):
            value = case.get(name)
            if type(value) is not str or not value.strip():
                raise SourceRetrievalCaseError(f"cases[{index}].{name} must be non-empty")
        source_ids = case.get("relevant_source_ids")
        if type(source_ids) is not list or any(type(value) is not str or not value.strip() for value in source_ids):
            raise SourceRetrievalCaseError(f"cases[{index}].relevant_source_ids must be a list of strings")
        if case["split"] != "development" or case["catalog_version"] != EXPECTED_CATALOG_VERSION:
            raise SourceRetrievalCaseError(f"cases[{index}] is not a catalog-v2 development case")
        if case["family_id"] in families:
            raise SourceRetrievalCaseError("development retrieval family_id values must be unique")
        families.add(str(case["family_id"]))
        parsed.append(
            SourceRetrievalCase(
                query=str(case["query"]),
                relevant_source_ids=tuple(source_ids),
                family_id=str(case["family_id"]),
                split="development",
                catalog_version=EXPECTED_CATALOG_VERSION,
            )
        )
    return tuple(parsed)


def load_versioned_retrieval_gold(
    path: str | Path | None = None,
) -> dict[str, tuple[tuple[str, str], ...]]:
    gold_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[3] / "evals" / "development" / "retrieval-gold-v1.json"
    )
    document = _read_json(gold_path, where="versioned retrieval gold")
    if set(document) != {"schema_version", "catalog_version", "gold_by_family_id"}:
        raise SourceRetrievalCaseError("retrieval gold root fields mismatch")
    if document.get("schema_version") != SOURCE_RETRIEVAL_GOLD_VERSION:
        raise SourceRetrievalCaseError("unsupported retrieval gold schema")
    if document.get("catalog_version") != EXPECTED_CATALOG_VERSION:
        raise SourceRetrievalCaseError("gold must bind catalog-v2")
    raw_gold = _object(document.get("gold_by_family_id"), where="gold_by_family_id")
    result: dict[str, tuple[tuple[str, str], ...]] = {}
    for family_id, raw_sources in raw_gold.items():
        if type(family_id) is not str or not family_id.strip() or type(raw_sources) is not list:
            raise SourceRetrievalCaseError("gold family ids and source lists are invalid")
        sources: list[tuple[str, str]] = []
        for index, raw_source in enumerate(raw_sources):
            source = _object(raw_source, where=f"gold_by_family_id.{family_id}[{index}]")
            if set(source) != {"source_id", "version"}:
                raise SourceRetrievalCaseError("gold source must contain source_id and version only")
            source_id, version = source.get("source_id"), source.get("version")
            if any(type(value) is not str or not value.strip() for value in (source_id, version)):
                raise SourceRetrievalCaseError("gold source_id and version must be non-empty strings")
            pair = (str(source_id), str(version))
            if pair in sources:
                raise SourceRetrievalCaseError("duplicate versioned gold source")
            sources.append(pair)
        result[family_id] = tuple(sources)
    return result


def validate_gold_source_versions(
    gold_by_family_id: Mapping[str, Sequence[tuple[str, str]]],
    active_source_versions: Mapping[str, str],
) -> dict[str, object]:
    """Require every non-empty gold source to match one active source version."""

    if not isinstance(gold_by_family_id, Mapping) or not isinstance(active_source_versions, Mapping):
        raise SourceRetrievalCaseError("gold and active source versions must be mappings")
    stale: list[dict[str, str]] = []
    gold_source_count = 0
    for family_id, sources in gold_by_family_id.items():
        if type(family_id) is not str or not family_id.strip():
            raise SourceRetrievalCaseError("gold family id is invalid")
        for source_id, version in sources:
            if type(source_id) is not str or type(version) is not str:
                raise SourceRetrievalCaseError("gold source/version pair is invalid")
            gold_source_count += 1
            if active_source_versions.get(source_id) != version:
                stale.append({"family_id": family_id, "source_id": source_id, "version": version})
    if stale:
        raise SourceRetrievalCaseError(f"gold sources are missing, inactive or stale: {stale}")
    return {
        "gold_family_count": len(gold_by_family_id),
        "gold_source_count": gold_source_count,
        "active_source_count": len(active_source_versions),
        "source_version_matches": True,
    }


def score_source_retrieval(
    cases: Sequence[SourceRetrievalCase],
    records: Sequence[Mapping[str, object]],
    gold_by_family_id: Mapping[str, Sequence[tuple[str, str]]],
    *,
    top_k: int = 3,
) -> dict[str, object]:
    """Score every case, treating a missing/failed record and an empty gold as zero."""

    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer from 1 through 10")
    case_ids = [case.family_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise SourceRetrievalCaseError("case family_id values must be unique")
    record_by_family: dict[str, Mapping[str, object]] = {}
    for record in records:
        family_id = record.get("family_id")
        if type(family_id) is not str or family_id not in set(case_ids):
            raise SourceRetrievalCaseError("retrieval record references an unknown family_id")
        if family_id in record_by_family:
            raise SourceRetrievalCaseError("duplicate retrieval record for a family_id")
        record_by_family[family_id] = record
    if set(gold_by_family_id) != set(case_ids):
        raise SourceRetrievalCaseError("versioned gold must cover exactly the evaluated families")

    case_results: list[dict[str, object]] = []
    hit_count = 0
    empty_hit_count = 0
    failed_count = 0
    version_mismatch_count = 0
    recall_sum = Fraction(0, 1)
    relevant_total = 0
    relevant_found_total = 0

    for case in cases:
        gold = tuple(gold_by_family_id[case.family_id])
        record = record_by_family.get(case.family_id)
        items: list[object] = []
        status = "failed" if record is None else str(record.get("status", "succeeded"))
        if record is not None and status not in {"failed", "timeout", "unknown"}:
            raw_items = record.get("items")
            if type(raw_items) is list:
                items = raw_items[:top_k]
            else:
                status = "failed"
        else:
            items = []
        if status in {"failed", "timeout", "unknown"}:
            failed_count += 1
            items = []

        returned: set[tuple[str, str]] = set()
        returned_ids: set[str] = set()
        for raw_item in items:
            if not isinstance(raw_item, Mapping):
                continue
            source_id, version = raw_item.get("source_id"), raw_item.get("version")
            if type(source_id) is str:
                returned_ids.add(source_id)
                if type(version) is str:
                    returned.add((source_id, version))
        matched = set(gold) & returned
        wrong_version = any(source_id in returned_ids and (source_id, version) not in returned for source_id, version in gold)
        version_mismatch_count += int(wrong_version)
        hit = bool(matched)
        hit_count += int(hit)
        empty_hit_count += int(not items)
        relevant_total += len(gold)
        relevant_found_total += len(matched)
        case_recall = Fraction(len(matched), len(gold)) if gold else Fraction(0, 1)
        recall_sum += case_recall
        case_results.append(
            {
                "family_id": case.family_id,
                "status": status,
                "hit_at_3": hit,
                "recall_at_3": {
                    "numerator": case_recall.numerator,
                    "denominator": case_recall.denominator,
                },
                "gold_source_versions": [
                    {"source_id": source_id, "version": version} for source_id, version in gold
                ],
                "returned_count_at_3": len(items),
                "empty_hit": not items,
                "wrong_version_returned": wrong_version,
            }
        )

    total = len(cases)
    macro_recall = recall_sum / total if total else Fraction(0, 1)
    return {
        "top_k": top_k,
        "case_count": total,
        "hit_at_3": {
            "numerator": hit_count,
            "denominator": total,
            "value": hit_count / total if total else None,
        },
        "recall_at_3_macro": {
            "numerator": macro_recall.numerator,
            "denominator": macro_recall.denominator,
            "case_denominator": total,
            "value": float(macro_recall) if total else None,
        },
        "recall_at_3_micro": {
            "numerator": relevant_found_total,
            "denominator": relevant_total,
            "value": relevant_found_total / relevant_total if relevant_total else None,
        },
        "empty_hit_count": empty_hit_count,
        "failed_count": failed_count,
        "wrong_version_count": version_mismatch_count,
        "per_case": case_results,
    }


def source_retrieval_manifest(
    cases_path: str | Path | None = None,
    gold_path: str | Path | None = None,
) -> dict[str, object]:
    case_file = (
        Path(cases_path)
        if cases_path is not None
        else Path(__file__).resolve().parents[3] / "evals" / "development" / "retrieval-cases-v1.json"
    )
    gold_file = (
        Path(gold_path)
        if gold_path is not None
        else Path(__file__).resolve().parents[3] / "evals" / "development" / "retrieval-gold-v1.json"
    )
    cases = load_source_retrieval_cases(case_file)
    gold = load_versioned_retrieval_gold(gold_file)
    if set(gold) != {case.family_id for case in cases}:
        raise SourceRetrievalCaseError("versioned gold and development query families differ")
    return {
        "schema_version": SOURCE_RETRIEVAL_SET_VERSION,
        "catalog_version": EXPECTED_CATALOG_VERSION,
        "case_count": len(cases),
        "family_ids": sorted(case.family_id for case in cases),
        "cases_sha256": hashlib.sha256(case_file.read_bytes()).hexdigest(),
        "versioned_gold_sha256": hashlib.sha256(gold_file.read_bytes()).hexdigest(),
    }


__all__ = [
    "EXPECTED_CATALOG_VERSION",
    "SOURCE_RETRIEVAL_GOLD_VERSION",
    "SOURCE_RETRIEVAL_SET_VERSION",
    "SourceRetrievalCase",
    "SourceRetrievalCaseError",
    "load_source_retrieval_cases",
    "load_versioned_retrieval_gold",
    "validate_gold_source_versions",
    "score_source_retrieval",
    "source_retrieval_manifest",
]
