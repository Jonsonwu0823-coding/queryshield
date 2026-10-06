"""Strict, in-memory loader for the independently sealed holdout."""

from __future__ import annotations

import base64
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from queryshield.evaluation.sealed_holdout import HoldoutSealError, verify_holdout_seal
from queryshield.evaluation.state_cases import (
    StateCaseError,
    StateCase,
    _validate_case,
    canonical_sha256,
)
from queryshield.evaluation.source_retrieval import (
    EXPECTED_CATALOG_VERSION,
    SourceRetrievalCase,
    SOURCE_RETRIEVAL_GOLD_VERSION,
    SOURCE_RETRIEVAL_SET_VERSION,
    SourceRetrievalCaseError,
)


HOLDOUT_PAYLOAD_VERSION = "w05-holdout-payload-v1"
HOLDOUT_MANIFEST_VERSION = "w05-sealed-case-manifest-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_ROW_FIELDS = frozenset(
    {
        "artifact_type",
        "case_id",
        "family_id",
        "split",
        "classification",
        "input_sha256",
        "expected_sha256",
    }
)


class HoldoutPayloadError(ValueError):
    """A decrypted holdout does not match the frozen payload contract."""


@dataclass(frozen=True)
class HoldoutDataset:
    task_cases: tuple[StateCase, ...]
    retrieval_cases: tuple[SourceRetrievalCase, ...]
    gold_by_family_id: Mapping[str, tuple[tuple[str, str], ...]]
    payload_sha256: str
    case_manifest_sha256: str
    ciphertext_sha256: str
    metadata_sha256: str


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HoldoutPayloadError("holdout JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise HoldoutPayloadError("holdout JSON contains a non-finite number")


def _strict_json(raw: bytes) -> object:
    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HoldoutPayloadError("decrypted holdout payload is not strict UTF-8 JSON") from exc


def _decode_key(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        key = value
    elif type(value) is str:
        text = value.strip()
        try:
            if re.fullmatch(r"[0-9a-fA-F]{64}", text):
                key = bytes.fromhex(text)
            else:
                key = base64.b64decode(text, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise HoldoutPayloadError("holdout key must be 32-byte hex or strict base64") from exc
    else:
        raise HoldoutPayloadError("holdout key format is invalid")
    if len(key) != 32:
        raise HoldoutPayloadError("holdout key must decode to 32 bytes")
    return key


def _decode_metadata_bytes(metadata: Mapping[str, object], field: str, expected: int) -> bytes:
    encoded = metadata.get(field)
    if type(encoded) is not str:
        raise HoldoutPayloadError(f"holdout metadata {field} is missing")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise HoldoutPayloadError(f"holdout metadata {field} is invalid base64") from exc
    if len(decoded) != expected:
        raise HoldoutPayloadError(f"holdout metadata {field} has an invalid length")
    return decoded


def _mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise HoldoutPayloadError(f"{where} must be an object")
    return value


def _exact(value: Mapping[str, object], required: frozenset[str], *, where: str) -> None:
    if set(value) != required:
        raise HoldoutPayloadError(f"{where} fields mismatch")


def _family_digest(family_id: str) -> str:
    return hashlib.sha256(family_id.encode("utf-8")).hexdigest()


def _parse_task_cases(raw: object) -> tuple[StateCase, ...]:
    if type(raw) is not list or len(raw) != 12:
        raise HoldoutPayloadError("holdout requires exactly twelve task cases")
    parsed: list[StateCase] = []
    for index, raw_wrapper in enumerate(raw):
        wrapper = _mapping(raw_wrapper, where=f"task_cases[{index}]")
        _exact(
            wrapper,
            frozenset({"case", "classification", "critical_question_id"}),
            where=f"task_cases[{index}]",
        )
        try:
            case = _validate_case(wrapper["case"], index=index, expected_split="holdout")
        except StateCaseError as exc:
            raise HoldoutPayloadError(f"task_cases[{index}] violates state-case-v1") from exc
        classification = wrapper.get("classification")
        if type(classification) is not str or classification not in {"functional", "security"}:
            raise HoldoutPayloadError(f"task_cases[{index}].classification is invalid")
        critical = wrapper.get("critical_question_id")
        if critical is not None and (type(critical) is not str or not critical.strip()):
            raise HoldoutPayloadError(f"task_cases[{index}].critical_question_id is invalid")
        parsed.append(StateCase(case, str(classification), critical))

    if len({case.case_id for case in parsed}) != 12:
        raise HoldoutPayloadError("holdout task case ids must be unique")
    if Counter(case.classification for case in parsed) != Counter({"functional": 8, "security": 4}):
        raise HoldoutPayloadError("holdout task quota must be 8 functional plus 4 security")
    family_members: dict[str, list[StateCase]] = {}
    for case in parsed:
        family_members.setdefault(case.family_id, []).append(case)
    paired = [members for members in family_members.values() if len(members) == 2]
    if len(family_members) != 11 or len(paired) != 1 or any(len(members) not in {1, 2} for members in family_members.values()):
        raise HoldoutPayloadError("holdout requires one paired family and ten single-case families")
    if Counter(case.classification for case in paired[0]) != Counter({"functional": 1, "security": 1}):
        raise HoldoutPayloadError("the paired holdout family must include one functional and one security case")
    return tuple(parsed)


def _parse_retrieval(payload: object) -> tuple[tuple[SourceRetrievalCase, ...], dict[str, tuple[tuple[str, str], ...]]]:
    payload_root = _mapping(payload, where="holdout payload")
    queries = _mapping(payload_root.get("retrieval_queries"), where="retrieval_queries")
    _exact(queries, frozenset({"schema_version", "catalog_version", "cases"}), where="retrieval_queries")
    if queries.get("schema_version") != SOURCE_RETRIEVAL_SET_VERSION or queries.get("catalog_version") != EXPECTED_CATALOG_VERSION:
        raise HoldoutPayloadError("holdout retrieval queries must use retrieval-case-v1 and catalog-v2")
    raw_cases = queries.get("cases")
    required_case_fields = frozenset({"query", "relevant_source_ids", "family_id", "split", "catalog_version"})
    if type(raw_cases) is not list or len(raw_cases) != 6:
        raise HoldoutPayloadError("holdout requires exactly six retrieval queries")
    cases: list[SourceRetrievalCase] = []
    for index, raw_case in enumerate(raw_cases):
        case = _mapping(raw_case, where=f"retrieval_queries.cases[{index}]")
        _exact(case, required_case_fields, where=f"retrieval_queries.cases[{index}]")
        if any(type(case.get(key)) is not str or not case[key].strip() for key in ("query", "family_id")):
            raise HoldoutPayloadError(f"retrieval_queries.cases[{index}] query and family_id must be non-empty")
        if case.get("split") != "holdout" or case.get("catalog_version") != EXPECTED_CATALOG_VERSION:
            raise HoldoutPayloadError(f"retrieval_queries.cases[{index}] has the wrong split or catalog")
        relevant = case.get("relevant_source_ids")
        if type(relevant) is not list or any(type(item) is not str or not item.strip() for item in relevant):
            raise HoldoutPayloadError(f"retrieval_queries.cases[{index}].relevant_source_ids is invalid")
        cases.append(
            SourceRetrievalCase(
                query=str(case["query"]),
                relevant_source_ids=tuple(relevant),
                family_id=str(case["family_id"]),
                split="holdout",
                catalog_version=EXPECTED_CATALOG_VERSION,
            )
        )
    families = [case.family_id for case in cases]
    if len(set(families)) != 6:
        raise HoldoutPayloadError("holdout retrieval family ids must be unique")

    gold_doc = _mapping(payload_root.get("retrieval_gold"), where="retrieval_gold")
    _exact(gold_doc, frozenset({"schema_version", "catalog_version", "gold_by_family_id"}), where="retrieval_gold")
    if gold_doc.get("schema_version") != SOURCE_RETRIEVAL_GOLD_VERSION or gold_doc.get("catalog_version") != EXPECTED_CATALOG_VERSION:
        raise HoldoutPayloadError("holdout retrieval gold must use retrieval-gold-v1 and catalog-v2")
    raw_gold = _mapping(gold_doc.get("gold_by_family_id"), where="retrieval_gold.gold_by_family_id")
    if set(raw_gold) != set(families):
        raise HoldoutPayloadError("holdout retrieval gold must cover exactly the six query families")
    gold: dict[str, tuple[tuple[str, str], ...]] = {}
    for family_id, raw_sources in raw_gold.items():
        if type(raw_sources) is not list:
            raise HoldoutPayloadError("holdout retrieval gold sources must be lists")
        parsed_sources: list[tuple[str, str]] = []
        for source_index, raw_source in enumerate(raw_sources):
            source = _mapping(raw_source, where=f"retrieval_gold.{family_id}[{source_index}]")
            _exact(source, frozenset({"source_id", "version"}), where=f"retrieval_gold.{family_id}[{source_index}]")
            if any(type(source.get(name)) is not str or not source[name].strip() for name in ("source_id", "version")):
                raise HoldoutPayloadError("holdout gold source id and version must be non-empty strings")
            pair = (str(source["source_id"]), str(source["version"]))
            if pair in parsed_sources:
                raise HoldoutPayloadError("holdout gold contains a duplicate source version")
            parsed_sources.append(pair)
        gold[family_id] = tuple(parsed_sources)
    return tuple(cases), gold


def _expected_manifest_rows(
    task_cases: Sequence[StateCase],
    retrieval_cases: Sequence[SourceRetrievalCase],
    gold: Mapping[str, Sequence[tuple[str, str]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in task_cases:
        rows.append(
            {
                "artifact_type": "task_case",
                "case_id": item.case_id,
                "family_id": item.family_id,
                "split": "holdout",
                "classification": item.classification,
                "input_sha256": canonical_sha256(
                    {"initial": item.case["initial"], "action": item.case["action"]}
                ),
                "expected_sha256": canonical_sha256(item.case["expected"]),
            }
        )
    for item in retrieval_cases:
        rows.append(
            {
                "artifact_type": "retrieval_query",
                "case_id": item.family_id,
                "family_id": item.family_id,
                "split": "holdout",
                "classification": "retrieval",
                "input_sha256": canonical_sha256(item.query),
                "expected_sha256": canonical_sha256(
                    [{"source_id": source_id, "version": version} for source_id, version in gold[item.family_id]]
                ),
            }
        )
    return sorted(rows, key=lambda row: (str(row["artifact_type"]), str(row["case_id"])))


def _parse_manifest(
    raw: object,
    *,
    task_cases: Sequence[StateCase],
    retrieval_cases: Sequence[SourceRetrievalCase],
    gold: Mapping[str, Sequence[tuple[str, str]]],
    expected_sha256: object,
) -> str:
    manifest = _mapping(raw, where="case_manifest")
    _exact(manifest, frozenset({"schema_version", "rows"}), where="case_manifest")
    if manifest.get("schema_version") != HOLDOUT_MANIFEST_VERSION:
        raise HoldoutPayloadError("holdout case manifest schema is unsupported")
    rows = manifest.get("rows")
    if type(rows) is not list or len(rows) != 18:
        raise HoldoutPayloadError("holdout case manifest must contain 18 rows")
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, where=f"case_manifest.rows[{index}]")
        _exact(row, _MANIFEST_ROW_FIELDS, where=f"case_manifest.rows[{index}]")
        for key in ("artifact_type", "case_id", "family_id", "split", "classification", "input_sha256", "expected_sha256"):
            if type(row.get(key)) is not str or not row[key].strip():
                raise HoldoutPayloadError(f"case_manifest.rows[{index}] has an invalid {key}")
        if row.get("split") != "holdout" or any(
            not _SHA256_RE.fullmatch(str(row.get(key))) for key in ("input_sha256", "expected_sha256")
        ):
            raise HoldoutPayloadError(f"case_manifest.rows[{index}] has an invalid split or digest")
    expected_rows = _expected_manifest_rows(task_cases, retrieval_cases, gold)
    if rows != expected_rows:
        raise HoldoutPayloadError("case manifest rows do not match the decrypted cases and gold")
    digest = canonical_sha256(manifest)
    if type(expected_sha256) is not str or not _SHA256_RE.fullmatch(expected_sha256) or digest != expected_sha256:
        raise HoldoutPayloadError("case manifest digest does not match its sealed metadata")
    return digest


def _parse_payload(
    payload: object,
    *,
    metadata: Mapping[str, object],
    development_cases: Sequence[StateCase],
    development_retrieval_cases: Sequence[SourceRetrievalCase],
    payload_sha256: str,
    ciphertext_sha256: str,
    metadata_sha256: str,
) -> HoldoutDataset:
    root = _mapping(payload, where="holdout payload")
    _exact(root, frozenset({"schema_version", "task_cases", "retrieval_queries", "retrieval_gold", "case_manifest"}), where="holdout payload")
    if root.get("schema_version") != HOLDOUT_PAYLOAD_VERSION:
        raise HoldoutPayloadError("holdout payload schema is unsupported")
    task_cases = _parse_task_cases(root.get("task_cases"))
    retrieval_cases, gold = _parse_retrieval(root)

    task_families = {case.family_id for case in task_cases}
    retrieval_families = {case.family_id for case in retrieval_cases}
    development_task_families = {case.family_id for case in development_cases}
    development_retrieval_families = {case.family_id for case in development_retrieval_cases}
    if task_families & development_task_families or retrieval_families & development_retrieval_families:
        raise HoldoutPayloadError("holdout families overlap the development set")
    family_commitments = metadata.get("family_id_commitments_sha256")
    if not isinstance(family_commitments, Mapping):
        raise HoldoutPayloadError("holdout family commitments are missing")
    committed_task = family_commitments.get("task_cases")
    committed_retrieval = family_commitments.get("retrieval_queries")
    if (
        type(committed_task) is not list
        or type(committed_retrieval) is not list
        or set(committed_task) != {_family_digest(family) for family in task_families}
        or set(committed_retrieval) != {_family_digest(family) for family in retrieval_families}
    ):
        raise HoldoutPayloadError("holdout family ids do not match the sealed commitments")
    paired_family = next(
        family for family in task_families if sum(case.family_id == family for case in task_cases) == 2
    )
    if _family_digest(paired_family) != metadata.get("paired_family_id_sha256"):
        raise HoldoutPayloadError("paired holdout family does not match its sealed commitment")

    manifest_digest = _parse_manifest(
        root.get("case_manifest"),
        task_cases=task_cases,
        retrieval_cases=retrieval_cases,
        gold=gold,
        expected_sha256=metadata.get("case_manifest_sha256"),
    )
    return HoldoutDataset(
        task_cases=task_cases,
        retrieval_cases=retrieval_cases,
        gold_by_family_id=gold,
        payload_sha256=payload_sha256,
        case_manifest_sha256=manifest_digest,
        ciphertext_sha256=ciphertext_sha256,
        metadata_sha256=metadata_sha256,
    )


def decrypt_and_load_holdout(
    key: str | bytes,
    *,
    metadata_path: str | Path,
    ciphertext_path: str | Path,
    development_cases: Sequence[StateCase],
    development_retrieval_cases: Sequence[SourceRetrievalCase],
) -> HoldoutDataset:
    """Verify, decrypt, validate, and discard plaintext without writing it to disk."""

    metadata_file = Path(metadata_path)
    ciphertext_file = Path(ciphertext_path)
    try:
        metadata_bytes = metadata_file.read_bytes()
        ciphertext = ciphertext_file.read_bytes()
        metadata = json.loads(metadata_bytes.decode("utf-8", errors="strict"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError, HoldoutPayloadError) as exc:
        raise HoldoutPayloadError("holdout metadata or ciphertext is unavailable") from exc
    if not isinstance(metadata, Mapping):
        raise HoldoutPayloadError("holdout metadata root must be an object")
    try:
        seal = verify_holdout_seal(
            metadata_file,
            ciphertext_file,
            development_cases,
            development_retrieval_cases,
        )
    except HoldoutSealError as exc:
        raise HoldoutPayloadError("holdout seal verification failed") from exc
    nonce = _decode_metadata_bytes(metadata, "nonce_base64", 12)
    tag = _decode_metadata_bytes(metadata, "tag_base64", 16)
    key_bytes = _decode_key(key)
    try:
        plaintext = AESGCM(key_bytes).decrypt(nonce, ciphertext + tag, None)
    except (InvalidTag, ValueError) as exc:
        raise HoldoutPayloadError("holdout authentication failed") from exc
    payload_digest = hashlib.sha256(plaintext).hexdigest()
    if payload_digest != metadata.get("plaintext_payload_sha256"):
        raise HoldoutPayloadError("holdout plaintext digest does not match its metadata")
    payload = _strict_json(plaintext)
    result = _parse_payload(
        payload,
        metadata=metadata,
        development_cases=development_cases,
        development_retrieval_cases=development_retrieval_cases,
        payload_sha256=payload_digest,
        ciphertext_sha256=str(seal["ciphertext_sha256"]),
        metadata_sha256=str(seal["metadata_sha256"]),
    )
    del plaintext, payload, key_bytes
    return result


__all__ = [
    "HOLDOUT_MANIFEST_VERSION",
    "HOLDOUT_PAYLOAD_VERSION",
    "HoldoutDataset",
    "HoldoutPayloadError",
    "decrypt_and_load_holdout",
]
