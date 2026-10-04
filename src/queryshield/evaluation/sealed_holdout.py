"""Verify W05 holdout commitments without opening or decrypting the dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re

from queryshield.evaluation.state_cases import W05StateCase
from queryshield.evaluation.w05_retrieval import W05RetrievalCase


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class HoldoutSealError(ValueError):
    """The sealed holdout artifact or its metadata commitment is invalid."""


def verify_w05_holdout_seal(
    metadata_path: str | Path,
    ciphertext_path: str | Path,
    development_cases: Sequence[W05StateCase],
    development_retrieval_cases: Sequence[W05RetrievalCase],
) -> dict[str, object]:
    """Check counts, ciphertext digest and hashed-family separation only."""

    metadata_file = Path(metadata_path)
    ciphertext_file = Path(ciphertext_path)
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        ciphertext_digest = hashlib.sha256(ciphertext_file.read_bytes()).hexdigest()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HoldoutSealError("holdout metadata or ciphertext is unavailable") from exc
    if not isinstance(metadata, Mapping):
        raise HoldoutSealError("holdout metadata root must be an object")
    if ciphertext_digest != metadata.get("ciphertext_sha256"):
        raise HoldoutSealError("holdout ciphertext SHA256 does not match its metadata")
    if metadata.get("cipher") != "AES-256-GCM":
        raise HoldoutSealError("holdout must be encrypted with AES-256-GCM")
    schemas = metadata.get("schema_versions")
    catalog_versions = metadata.get("catalog_versions")
    counts = metadata.get("split_counts")
    family_hashes = metadata.get("family_id_commitments_sha256", metadata.get("family_id_sha256"))
    if (
        not isinstance(schemas, Mapping)
        or not isinstance(catalog_versions, Mapping)
        or not isinstance(counts, Mapping)
        or not isinstance(family_hashes, Mapping)
    ):
        raise HoldoutSealError("holdout schema, catalog, counts or family commitments are missing")
    if (
        schemas.get("task_cases") != "state-case-v1"
        or schemas.get("retrieval_queries") != "retrieval-case-v1"
        or schemas.get("case_manifest") != "w05-sealed-case-manifest-v1"
    ):
        raise HoldoutSealError("holdout case schemas are unsupported")
    if (
        catalog_versions.get("business_fixture") != "commerce-v1"
        or catalog_versions.get("catalog") != "catalog-v2"
        or catalog_versions.get("knowledge") != "knowledge-v1"
    ):
        raise HoldoutSealError("holdout is not bound to the frozen W05 source versions")
    paired_hash = metadata.get("paired_family_id_sha256")
    if (
        metadata.get("paired_family_count") != 1
        or metadata.get("novel_pair_vs_C10") is not True
        or metadata.get("c10_development_pair_count") != 6
        or metadata.get("manifest_row_count") != 18
        or metadata.get("manifest_required_per_row_fields_present") is not True
        or type(paired_hash) is not str
        or not _SHA256_RE.fullmatch(paired_hash)
    ):
        raise HoldoutSealError("holdout manifest lacks the required novel paired-family commitment")
    task_counts = counts.get("task_cases")
    retrieval_counts = counts.get("retrieval_queries")
    task_families = family_hashes.get("task_cases")
    retrieval_families = family_hashes.get("retrieval_queries")
    if not isinstance(task_counts, Mapping) or not isinstance(retrieval_counts, Mapping):
        raise HoldoutSealError("holdout split counts are malformed")
    if task_counts.get("functional") != 8 or task_counts.get("security") != 4 or task_counts.get("holdout") != 12:
        raise HoldoutSealError("holdout task quota must be 8 functional plus 4 security")
    if retrieval_counts.get("holdout") != 6:
        raise HoldoutSealError("holdout retrieval quota must be six queries")
    if not isinstance(task_families, list) or not isinstance(retrieval_families, list):
        raise HoldoutSealError("holdout family commitments must be lists")
    if any(type(value) is not str or not _SHA256_RE.fullmatch(value) for value in task_families + retrieval_families):
        raise HoldoutSealError("holdout family commitment is not a SHA256 digest")
    if len(set(task_families)) != len(task_families) or len(set(retrieval_families)) != len(retrieval_families):
        raise HoldoutSealError("holdout family commitments contain duplicates")
    if len(task_families) < 2 or len(retrieval_families) != 6:
        raise HoldoutSealError("holdout family counts are incomplete")

    manifest_digest = metadata.get("case_manifest_sha256")
    payload_digest = metadata.get("plaintext_payload_sha256")
    if any(
        type(value) is not str or not _SHA256_RE.fullmatch(value)
        for value in (manifest_digest, payload_digest)
    ):
        raise HoldoutSealError("holdout plaintext or case manifest commitment is missing")
    overlap_counts = metadata.get("development_family_overlap_counts")
    if not isinstance(overlap_counts, Mapping):
        raise HoldoutSealError("holdout development-overlap summary is missing")
    development_task_hashes = {
        hashlib.sha256(case.family_id.encode("utf-8")).hexdigest()
        for case in development_cases
    }
    development_retrieval_hashes = {
        hashlib.sha256(case.family_id.encode("utf-8")).hexdigest()
        for case in development_retrieval_cases
    }
    for split_name in ("task_cases", "retrieval_queries"):
        split_overlap = overlap_counts.get(split_name)
        if not isinstance(split_overlap, Mapping) or split_overlap.get("overlap_count") != 0:
            raise HoldoutSealError(f"holdout {split_name} overlap commitment is not zero")
    if overlap_counts["task_cases"].get("development_family_count") != len(development_task_hashes):
        raise HoldoutSealError("holdout task overlap metadata has an outdated development count")
    if overlap_counts["retrieval_queries"].get("development_family_count") != len(development_retrieval_hashes):
        raise HoldoutSealError("holdout retrieval overlap metadata has an outdated development count")

    task_overlap = development_task_hashes & set(task_families)
    retrieval_overlap = development_retrieval_hashes & set(retrieval_families)
    if task_overlap or retrieval_overlap:
        raise HoldoutSealError("development and holdout family hashes overlap")
    return {
        "status": "verified_without_decryption",
        "cipher": metadata["cipher"],
        "ciphertext_sha256": ciphertext_digest,
        "plaintext_payload_sha256": payload_digest,
        "case_manifest_sha256": manifest_digest,
        "metadata_sha256": hashlib.sha256(metadata_file.read_bytes()).hexdigest(),
        "task_case_count": 12,
        "task_functional_count": 8,
        "task_security_count": 4,
        "task_family_count": len(task_families),
        "task_family_overlap_count": 0,
        "paired_family_count": 1,
        "novel_pair_vs_C10": True,
        "paired_family_id_sha256": paired_hash,
        "manifest_row_count": 18,
        "retrieval_query_count": 6,
        "retrieval_family_overlap_count": 0,
        "payload_opened": False,
    }


__all__ = ["HoldoutSealError", "verify_w05_holdout_seal"]
