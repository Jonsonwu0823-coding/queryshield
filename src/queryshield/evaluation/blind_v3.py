"""Blind test v3: the sealed case format, its commitment, and opening it in memory.

The cases are HTTP conversations on the demo data (commerce-demo-v1): one question,
then up to three steps taken when the run pauses (an answer to the follow-up
question, or an approval decision by a given identity), and the expected end.
The plaintext is encrypted with AES-256-GCM under a key the sealing tool makes;
the commitment file holds digests and counts only, plus an HMAC of a fixed text
so a wrong key is caught before any model call or record is read.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from queryshield.evaluation.holdout_dataset import HoldoutPayloadError, _decode_key, _strict_json


PAYLOAD_VERSION = "blind-v3-payload-v1"
DATA_VERSION = "commerce-demo-v1"
TOOL_VERSION = "blind-v3-tools-1"
CASE_IDS = tuple(f"V3-{number:02}" for number in range(1, 13))
CASE_COUNTS = {"functional": 8, "security": 4}
IDENTITIES = ("a-requester", "a-approver", "b-requester", "b-approver")
METRICS = frozenset({"paid_count", "gross_fen", "refund_fen", "net_fen"})
OUTCOMES = frozenset({"answer", "clarify", "waiting_approval", "refuse", "fixed_reply", "definition"})
ROWS_FIELDS = ("rows", "rows_window", "rows_metric")  # a ranking: customers by one metric over one window
COMMITMENT_FIELDS = frozenset(
    {
        "schema_version", "case_counts", "ciphertext_sha256", "key_check_hmac_sha256", "data_version",
        "demo_sql_sha256", "near_duplicates_flagged", "tool_version", "tool_source_sha256", "created_at",
    }
)
_KEY_CHECK_TEXT = b"queryshield blind-v3 key check"
_NONCE_BYTES = 12


class BlindSetError(ValueError):
    """The sealed set, its commitment or a key does not match the v3 format.

    Messages name fields and case ids only, never question or answer text.
    """


@dataclass(frozen=True)
class BlindCase:
    case_id: str
    classification: str
    identity: str
    question: str
    steps: tuple[Mapping[str, str], ...]
    expected: Mapping[str, object]

    @property
    def tenant(self) -> str:
        return self.identity[0].upper()

    @property
    def asker_is_approver(self) -> bool:
        """An approver who asks sees customer names without an approval (the product never pauses for one)."""

        return self.identity.endswith("approver")


def _object(value: object, fields: set[str], where: str, optional: set[str] = frozenset()) -> Mapping[str, object]:
    if not isinstance(value, dict) or not fields <= set(value) <= fields | optional:
        raise BlindSetError(f"{where} must have exactly the fields {sorted(fields)}")
    return value


def _text(value: object, where: str) -> str:
    if type(value) is not str or not value.strip():
        raise BlindSetError(f"{where} must be a non-empty string")
    return value


def _choice(value: object, allowed, where: str) -> str:
    """A string among ``allowed`` (a list or a dict as the value is reported, not raised as TypeError)."""

    if type(value) is not str or value not in allowed:
        raise BlindSetError(f"{where} must be one of {sorted(allowed)}")
    return value


def _utc(value: object, where: str) -> datetime:
    try:
        return datetime.strptime(_text(value, where), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise BlindSetError(f"{where} must be YYYY-MM-DDTHH:MM:SSZ") from exc


def _int(value: object, where: str) -> int:
    if type(value) is not int:
        raise BlindSetError(f"{where} must be an integer")
    return value


def _list(value: object, where: str) -> list:
    if type(value) is not list:
        raise BlindSetError(f"{where} must be a list")
    return value


def _window(value: object, where: str) -> Mapping[str, str]:
    window = _object(value, {"start", "end"}, where)
    start = _utc(window["start"], f"{where}.start")
    end = _utc(window["end"], f"{where}.end")
    if not start < end:
        raise BlindSetError(f"{where}.start must be before its end")
    return window


def _step(raw: object, where: str) -> Mapping[str, str]:
    on = _choice(raw.get("on") if isinstance(raw, dict) else None, ("WAITING_USER", "WAITING_APPROVAL"), f"{where}.on")
    if on == "WAITING_USER":
        step = _object(raw, {"on", "reply"}, where)
        _text(step["reply"], f"{where}.reply")
    else:
        step = _object(raw, {"on", "actor", "decision"}, where)
        _choice(step["actor"], IDENTITIES, f"{where}.actor")
        _choice(step["decision"], ("approve", "reject"), f"{where}.decision")
    return dict(step)


def _expected(raw: object, where: str) -> Mapping[str, object]:
    outcome = _choice(raw.get("outcome") if isinstance(raw, dict) else None, OUTCOMES, f"{where}.outcome")
    if outcome == "definition":
        _text(_object(raw, {"outcome", "source_id"}, where)["source_id"], f"{where}.source_id")
    elif outcome == "answer":
        expected = _object(raw, {"outcome"}, where, optional=set(ROWS_FIELDS) | {"facts"})
        if not expected.get("facts") and not expected.get("rows"):
            raise BlindSetError(f"{where} of an answer needs facts or rows")
        if any(field in expected for field in ROWS_FIELDS) and not all(field in expected for field in ROWS_FIELDS):
            raise BlindSetError(f"{where}: rows, rows_window and rows_metric go together")
        for index, fact in enumerate(_list(expected.get("facts", []), f"{where}.facts")):
            at = f"{where}.facts[{index}]"
            fact = _object(fact, {"metric_id", "window", "value"}, at)
            _choice(fact["metric_id"], METRICS, f"{at}.metric_id")
            _window(fact["window"], f"{at}.window")
            _int(fact["value"], f"{at}.value")
        for index, row in enumerate(_list(expected.get("rows", []), f"{where}.rows")):
            at = f"{where}.rows[{index}]"
            row = _object(row, {"customer_id", "value"}, at)
            _text(row["customer_id"], f"{at}.customer_id")
            _int(row["value"], f"{at}.value")
        if "rows" in expected:
            if not expected["rows"]:
                raise BlindSetError(f"{where}.rows must not be empty")
            _window(expected["rows_window"], f"{where}.rows_window")
            _choice(expected["rows_metric"], METRICS, f"{where}.rows_metric")
    else:
        _object(raw, {"outcome"}, where)
    return raw


def parse_payload(plaintext: bytes) -> tuple[BlindCase, ...]:
    """The twelve cases of a strict blind-v3-payload-v1 document (checked once, here)."""

    try:
        document = _strict_json(plaintext)
    except HoldoutPayloadError as exc:
        raise BlindSetError("payload is not strict UTF-8 JSON") from exc
    root = _object(document, {"schema_version", "data_version", "cases"}, "payload")
    if root["schema_version"] != PAYLOAD_VERSION or root["data_version"] != DATA_VERSION:
        raise BlindSetError(f"payload must be {PAYLOAD_VERSION} on {DATA_VERSION}")
    cases = []
    for index, raw in enumerate(_list(root["cases"], "payload.cases")):
        where = f"cases[{index}]"
        case = _object(raw, {"case_id", "classification", "identity", "question", "steps", "expected"}, where, optional={"note"})
        if len(_list(case["steps"], f"{where}.steps")) > 3:
            raise BlindSetError(f"{where}.steps must hold at most three steps")
        cases.append(
            BlindCase(
                case_id=_choice(case["case_id"], CASE_IDS, f"{where}.case_id"),
                classification=_choice(case["classification"], CASE_COUNTS, f"{where}.classification"),
                identity=_choice(case["identity"], IDENTITIES, f"{where}.identity"),
                question=_text(case["question"], f"{where}.question"),
                steps=tuple(_step(step, f"{where}.steps[{n}]") for n, step in enumerate(case["steps"])),
                expected=_expected(case["expected"], f"{where}.expected"),
            )
        )
    if sorted(case.case_id for case in cases) != list(CASE_IDS):
        raise BlindSetError(f"payload must hold exactly the cases {CASE_IDS[0]} to {CASE_IDS[-1]}")
    if Counter(case.classification for case in cases) != Counter(CASE_COUNTS):
        raise BlindSetError("payload must hold 8 functional and 4 security cases")
    return tuple(sorted(cases, key=lambda case: case.case_id))


def new_key() -> bytes:
    """A fresh 32-byte key (made by the sealing tool, shown once in base64)."""

    return os.urandom(32)


def key_check(key: bytes) -> str:
    return hmac.new(key, _KEY_CHECK_TEXT, hashlib.sha256).hexdigest()


def encrypt(key: bytes, aad: str, data: bytes) -> bytes:
    """nonce + AES-256-GCM ciphertext and tag, bound to ``aad``."""

    nonce = os.urandom(_NONCE_BYTES)
    return nonce + AESGCM(key).encrypt(nonce, data, aad.encode("utf-8"))


def decrypt(key: bytes, aad: str, blob: bytes) -> bytes:
    try:
        return AESGCM(key).decrypt(blob[:_NONCE_BYTES], blob[_NONCE_BYTES:], aad.encode("utf-8"))
    except (InvalidTag, ValueError) as exc:
        raise BlindSetError(f"authentication failed for {aad}") from exc


def commitment(
    *, ciphertext: bytes, key: bytes, demo_sql_sha256: str, near_duplicates_flagged: int, tool_source_sha256: str, created_at: str
) -> dict[str, object]:
    return {
        "schema_version": PAYLOAD_VERSION,
        "case_counts": dict(CASE_COUNTS),
        "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
        "key_check_hmac_sha256": key_check(key),
        "data_version": DATA_VERSION,
        "demo_sql_sha256": demo_sql_sha256,
        "near_duplicates_flagged": near_duplicates_flagged,
        "tool_version": TOOL_VERSION,
        "tool_source_sha256": tool_source_sha256,
        "created_at": created_at,
    }


def verify_commitment(commitment_path: Path, ciphertext_path: Path, demo_sql_sha256: str) -> dict[str, object]:
    """Check the commitment against the ciphertext and the current demo data, without a key."""

    try:
        raw = commitment_path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        ciphertext = ciphertext_path.read_bytes()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BlindSetError("the commitment or the ciphertext cannot be read") from exc
    if not isinstance(document, dict) or set(document) != COMMITMENT_FIELDS:
        raise BlindSetError("the commitment fields do not match the v3 format")
    if (
        document["schema_version"] != PAYLOAD_VERSION
        or document["case_counts"] != CASE_COUNTS
        or document["data_version"] != DATA_VERSION
        or document["tool_version"] != TOOL_VERSION
    ):
        raise BlindSetError("the commitment is not a v3 set of 8 functional and 4 security cases")
    if document["ciphertext_sha256"] != hashlib.sha256(ciphertext).hexdigest():
        raise BlindSetError("the ciphertext does not match its commitment")
    if document["demo_sql_sha256"] != demo_sql_sha256:
        raise BlindSetError("the demo data differs from the data the set was sealed against")
    return {
        "status": "verified_without_decryption",
        "commitment_sha256": hashlib.sha256(raw).hexdigest(),
        "ciphertext_sha256": document["ciphertext_sha256"],
        "near_duplicates_flagged": document["near_duplicates_flagged"],
        "created_at": document["created_at"],
    }


def check_key(key_text: str, commitment_path: Path) -> bytes:
    """The key bytes, if they match the commitment's check value; otherwise stop."""

    try:
        key = _decode_key(key_text)
    except HoldoutPayloadError as exc:
        raise BlindSetError("the key must be 32 bytes in base64 or hex") from exc
    expected = json.loads(commitment_path.read_text(encoding="utf-8"))["key_check_hmac_sha256"]
    if not hmac.compare_digest(key_check(key), str(expected)):
        raise BlindSetError("the key does not match the commitment")
    return key


def open_sealed(key: bytes, ciphertext_path: Path) -> tuple[BlindCase, ...]:
    """Decrypt and parse in memory; the plaintext is never written."""

    return parse_payload(decrypt(key, PAYLOAD_VERSION, ciphertext_path.read_bytes()))
