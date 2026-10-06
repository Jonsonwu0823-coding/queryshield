"""Strict state-case-v1 loading and dataset integrity checks."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re


STATE_CASE_VERSION = "state-case-v1"
STATE_CASE_SET_VERSION = "w05-state-evaluation-v1"
SUPPLEMENT_CASE_SET_VERSION = "w05-state-evaluation-supplement-v1"
_SUPPLEMENT_FILE = "state-cases-supplement-v1.json"
CONTRACT_VERSION = "2026-09-06.practice-v3"
EXTENSION_VERSION = "2026-09-09.facts-state-v1"
FIXTURE_VERSION = "commerce-v1"

_CASE_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "family_id",
        "split",
        "system",
        "layer",
        "contract_version",
        "extension_version",
        "fixture_version",
        "initial",
        "action",
        "expected",
    }
)
_INITIAL_FIELDS = frozenset(
    {
        "principal_fixture",
        "clock_utc",
        "messages",
        "run_state",
        "result_fixtures",
        "approval_fixtures",
    }
)
_ACTION_FIELDS = frozenset({"entrypoint", "parameters", "actor"})
_EXPECTED_FIELDS = frozenset(
    {
        "http_status",
        "terminal_state",
        "facts",
        "allowed_side_effects",
        "forbidden_side_effects",
        "usage",
        "invariants",
    }
)
_PAIR_CLASSES: dict[str, Counter[str]] = {
    "clarification-context": Counter({"functional": 2}),
    "approval-expiry": Counter({"functional": 1, "security": 1}),
    "approval-action-binding": Counter({"functional": 1, "security": 1}),
    "result-origin": Counter({"functional": 1, "security": 1}),
    "result-owner": Counter({"functional": 1, "security": 1}),
    "tool-text-injection": Counter({"functional": 1, "security": 1}),
}
_CRITICAL_QUESTION_IDS = frozenset(
    {
        "gross-total",
        "paid-count",
        "net-result",
        "join-aggregate",
        "empty-window",
        "tenant-isolation",
        "ambiguity-clarification",
        "single-repair",
    }
)
_FROZEN_DATASET_CONFIG = {
    "case_schema": STATE_CASE_VERSION,
    "case_set": STATE_CASE_SET_VERSION,
    "dataset_revision": "w05-development-erratum-r3",
    "contract_version": CONTRACT_VERSION,
    "extension_version": EXTENSION_VERSION,
    "fixture_version": FIXTURE_VERSION,
    "system": "QS",
    "catalog_version": "catalog-v2",
    "knowledge_version": "knowledge-v1",
    "case_generator": "explicit-json-v1",
}


class StateCaseError(ValueError):
    """The case set is malformed or violates its frozen split contract."""


_PRINCIPAL_FIXTURE_RE = re.compile(r"^tenant-([A-Z])/requester-([A-Z])$")
_ACTOR_FIXTURE_RE = re.compile(r"^(requester|approver)-([A-Z])$")


def resolve_principal_fixture(reference: str) -> dict[str, str]:
    """Resolve the frozen fixture alias to the actual commerce-v1 identity key."""

    if type(reference) is not str:
        raise StateCaseError("principal fixture reference must be a string")
    match = _PRINCIPAL_FIXTURE_RE.fullmatch(reference)
    if match is None or match.group(1) != match.group(2):
        raise StateCaseError("principal fixture must bind the requester to the same commerce tenant")
    tenant = match.group(1)
    return {"tenant_id": tenant, "principal_id": f"principal-{tenant}", "role": "requester"}


def resolve_actor_fixture(reference: str, *, tenant_id: str) -> dict[str, str]:
    """Resolve requester/approver fixture names without accepting another tenant."""

    if type(reference) is not str or type(tenant_id) is not str:
        raise StateCaseError("actor fixture and tenant must be strings")
    match = _ACTOR_FIXTURE_RE.fullmatch(reference)
    if match is None or match.group(2) != tenant_id:
        raise StateCaseError("actor fixture must belong to the initialized tenant")
    role, tenant = match.groups()
    principal_id = f"principal-{tenant}" if role == "requester" else f"approver-{tenant}"
    return {"tenant_id": tenant, "principal_id": principal_id, "role": role}


@dataclass(frozen=True)
class StateCase:
    case: Mapping[str, object]
    classification: str
    critical_question_id: str | None

    @property
    def case_id(self) -> str:
        return str(self.case["case_id"])

    @property
    def family_id(self) -> str:
        return str(self.case["family_id"])


def canonical_sha256(value: object) -> str:
    """Hash JSON data using stable UTF-8 bytes and reject non-finite numbers."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise StateCaseError(f"{where} must be an object with string keys")
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    required: frozenset[str],
    *,
    where: str,
) -> None:
    if set(value) != required:
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        raise StateCaseError(f"{where} fields mismatch: missing={missing}, extra={extra}")


def _utc_datetime(value: object, *, where: str) -> datetime:
    if type(value) is not str or not value.strip():
        raise StateCaseError(f"{where} must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StateCaseError(f"{where} must be an ISO UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise StateCaseError(f"{where} must include a UTC timezone")
    return parsed


def _validate_fixture_rows(value: object, *, where: str, expected_fields: frozenset[str]) -> None:
    if type(value) is not list:
        raise StateCaseError(f"{where} must be a list")
    aliases: set[str] = set()
    for index, raw in enumerate(value):
        row = _require_mapping(raw, where=f"{where}[{index}]")
        _require_exact_fields(row, expected_fields, where=f"{where}[{index}]")
        alias = row.get("alias")
        if type(alias) is not str or not alias.strip() or alias in aliases:
            raise StateCaseError(f"{where}[{index}].alias must be unique and non-empty")
        aliases.add(alias)
        if "approved_at" in row:
            _utc_datetime(row["approved_at"], where=f"{where}[{index}].approved_at")
        canonical_sha256(row)


def _validate_case(
    raw: object,
    *,
    index: int,
    expected_split: str = "development",
) -> Mapping[str, object]:
    where = f"cases[{index}].case"
    case = _require_mapping(raw, where=where)
    _require_exact_fields(case, _CASE_FIELDS, where=where)
    if case.get("schema_version") != STATE_CASE_VERSION:
        raise StateCaseError(f"{where}.schema_version must be {STATE_CASE_VERSION}")
    for field in (
        "case_id",
        "family_id",
        "split",
        "system",
        "layer",
        "contract_version",
        "extension_version",
        "fixture_version",
    ):
        value = case.get(field)
        if type(value) is not str or not value.strip():
            raise StateCaseError(f"{where}.{field} must be a non-empty string")
    if case["split"] != expected_split:
        raise StateCaseError(f"{where}.split must be {expected_split}")
    if case["system"] != "QS" or case["layer"] not in {"agent", "harness"}:
        raise StateCaseError(f"{where} must target the QS agent or harness layer")
    if case["contract_version"] != CONTRACT_VERSION:
        raise StateCaseError(f"{where}.contract_version is unsupported")
    if case["extension_version"] != EXTENSION_VERSION:
        raise StateCaseError(f"{where}.extension_version is unsupported")
    if case["fixture_version"] != FIXTURE_VERSION:
        raise StateCaseError(f"{where}.fixture_version is unsupported")

    initial = _require_mapping(case.get("initial"), where=f"{where}.initial")
    _require_exact_fields(initial, _INITIAL_FIELDS, where=f"{where}.initial")
    for field in ("principal_fixture", "clock_utc"):
        value = initial.get(field)
        if type(value) is not str or not value.strip():
            raise StateCaseError(f"{where}.initial.{field} must be a non-empty string")
    _utc_datetime(initial["clock_utc"], where=f"{where}.initial.clock_utc")
    messages = initial.get("messages")
    if type(messages) is not list:
        raise StateCaseError(f"{where}.initial.messages must be a list")
    for message_index, raw_message in enumerate(messages):
        message = _require_mapping(raw_message, where=f"{where}.initial.messages[{message_index}]")
        _require_exact_fields(message, frozenset({"role", "content"}), where=f"{where}.initial.messages[{message_index}]")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise StateCaseError(f"{where}.initial.messages[{message_index}].role is invalid")
        if type(message.get("content")) is not str or not message["content"].strip():
            raise StateCaseError(f"{where}.initial.messages[{message_index}].content must be non-empty")
    _validate_fixture_rows(
        initial.get("result_fixtures"),
        where=f"{where}.initial.result_fixtures",
        expected_fields=frozenset({"alias", "result_id", "run_id", "tenant_id", "principal_id", "metric_id", "value", "unit", "window", "source"}),
    )
    _validate_fixture_rows(
        initial.get("approval_fixtures"),
        where=f"{where}.initial.approval_fixtures",
        expected_fields=frozenset({"alias", "tenant_id", "approver_id", "query_digest", "action_digest", "approved_at"}),
    )
    run_state = initial.get("run_state")
    if not isinstance(run_state, dict):
        raise StateCaseError(f"{where}.initial.run_state must be an object")
    canonical_sha256(run_state)

    action = _require_mapping(case.get("action"), where=f"{where}.action")
    _require_exact_fields(action, _ACTION_FIELDS, where=f"{where}.action")
    if any(type(action.get(field)) is not str or not action[field].strip() for field in ("entrypoint", "actor")):
        raise StateCaseError(f"{where}.action entrypoint and actor must be non-empty strings")
    if not isinstance(action.get("parameters"), dict):
        raise StateCaseError(f"{where}.action.parameters must be an object")
    canonical_sha256(action["parameters"])

    expected = _require_mapping(case.get("expected"), where=f"{where}.expected")
    _require_exact_fields(expected, _EXPECTED_FIELDS, where=f"{where}.expected")
    status = expected.get("http_status")
    if status is not None and (type(status) is not int or not 100 <= status <= 599):
        raise StateCaseError(f"{where}.expected.http_status must be an HTTP status or null")
    terminal = expected.get("terminal_state")
    if type(terminal) is not str or not terminal.strip():
        raise StateCaseError(f"{where}.expected.terminal_state must be a non-empty string")
    if type(expected.get("facts")) is not list:
        raise StateCaseError(f"{where}.expected.facts must be a list")
    canonical_sha256(expected["facts"])
    for field in ("allowed_side_effects", "forbidden_side_effects", "usage", "invariants"):
        if not isinstance(expected.get(field), dict):
            raise StateCaseError(f"{where}.expected.{field} must be an object")
    usage = expected["usage"]
    if usage.get("unknown_values_are_null") is not True:
        raise StateCaseError(f"{where}.expected.usage must preserve unknown values as null")
    return case


def load_state_cases(path: str | Path | None = None) -> tuple[StateCase, ...]:
    fixture_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[3] / "evals" / "development" / "state-cases-v4.json"
    )
    try:
        document = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StateCaseError(f"cannot load state-case fixture: {fixture_path}") from exc
    if not isinstance(document, dict):
        raise StateCaseError("case-set root must be an object")
    root_fields = {
        "schema_version",
        "contract_version",
        "extension_version",
        "fixture_version",
        "paired_family_ids",
        "critical_question_ids",
        "cases",
    }
    _require_exact_fields(document, frozenset(root_fields), where="case-set")
    if document["schema_version"] != STATE_CASE_SET_VERSION:
        raise StateCaseError("unsupported case-set schema")
    if document["contract_version"] != CONTRACT_VERSION:
        raise StateCaseError("unsupported contract version")
    if document["extension_version"] != EXTENSION_VERSION:
        raise StateCaseError("unsupported extension version")
    if document["fixture_version"] != FIXTURE_VERSION:
        raise StateCaseError("unsupported fixture version")
    raw_cases = document.get("cases")
    if type(raw_cases) is not list:
        raise StateCaseError("case-set cases must be a list")

    parsed: list[StateCase] = []
    for index, wrapper_raw in enumerate(raw_cases):
        wrapper = _require_mapping(wrapper_raw, where=f"cases[{index}]")
        _require_exact_fields(
            wrapper,
            frozenset({"case", "classification", "critical_question_id"}),
            where=f"cases[{index}]",
        )
        case = _validate_case(wrapper.get("case"), index=index)
        classification = wrapper.get("classification")
        if classification not in {"functional", "security"}:
            raise StateCaseError(f"cases[{index}].classification must be functional or security")
        critical = wrapper.get("critical_question_id")
        if critical is not None and (type(critical) is not str or not critical.strip()):
            raise StateCaseError(f"cases[{index}].critical_question_id must be a string or null")
        parsed.append(StateCase(case, str(classification), critical))

    ids = [item.case_id for item in parsed]
    if len(ids) != len(set(ids)):
        raise StateCaseError("case_id values must be unique")
    if len(parsed) != 20:
        raise StateCaseError("development set must contain exactly 20 task cases")
    categories = Counter(item.classification for item in parsed)
    if categories != Counter({"functional": 12, "security": 8}):
        raise StateCaseError(f"development quotas mismatch: {dict(categories)}")

    families: dict[str, list[StateCase]] = defaultdict(list)
    for item in parsed:
        families[item.family_id].append(item)
    declared_pairs = document.get("paired_family_ids")
    if type(declared_pairs) is not list or len(declared_pairs) != 6 or len(set(declared_pairs)) != 6:
        raise StateCaseError("exactly six unique paired development families must be declared")
    if set(declared_pairs) != set(_PAIR_CLASSES):
        raise StateCaseError("paired development families do not match C10")
    for family_id, expected_classes in _PAIR_CLASSES.items():
        members = families.get(family_id, [])
        if len(members) != 2 or Counter(item.classification for item in members) != expected_classes:
            raise StateCaseError(f"paired family {family_id} does not match its C10 classifications")
    if any(len(members) != 1 for family_id, members in families.items() if family_id not in _PAIR_CLASSES):
        raise StateCaseError("unpaired development cases must have unique family_id values")

    critical_ids = [
        item.critical_question_id for item in parsed if item.critical_question_id is not None
    ]
    declared_critical = document.get("critical_question_ids")
    if (
        type(declared_critical) is not list
        or set(declared_critical) != _CRITICAL_QUESTION_IDS
        or len(declared_critical) != len(_CRITICAL_QUESTION_IDS)
        or Counter(critical_ids) != Counter(_CRITICAL_QUESTION_IDS)
    ):
        raise StateCaseError("the eight predeclared C7 critical questions are missing or duplicated")
    return tuple(parsed)


def state_case_manifest(
    path: str | Path | None = None,
    *,
    shared_runtime_config_sha256: str | None = None,
) -> dict[str, object]:
    """Return content-addressed developer-set metadata without changing the fixture."""

    fixture_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[3] / "evals" / "development" / "state-cases-v4.json"
    )
    cases = load_state_cases(fixture_path)
    if shared_runtime_config_sha256 is not None and (
        type(shared_runtime_config_sha256) is not str
        or len(shared_runtime_config_sha256) != 64
        or any(char not in "0123456789abcdef" for char in shared_runtime_config_sha256)
    ):
        raise StateCaseError("shared_runtime_config_sha256 must be a lowercase SHA256 digest")
    raw_bytes = fixture_path.read_bytes()
    budget_path = Path(__file__).resolve().parents[3] / "evals" / "development" / "execution-profiles-v1.json"
    frozen_config_sha256 = canonical_sha256(_FROZEN_DATASET_CONFIG)
    case_entries = []
    for item in cases:
        case = item.case
        case_entries.append(
            {
                "case_id": item.case_id,
                "family_id": item.family_id,
                "split": case["split"],
                "classification": item.classification,
                "critical_question_id": item.critical_question_id,
                "case_config_sha256": frozen_config_sha256,
                "input_sha256": canonical_sha256(
                    {"initial": case["initial"], "action": case["action"]}
                ),
                "expected_sha256": canonical_sha256(case["expected"]),
            }
        )
    return {
        "schema_version": STATE_CASE_SET_VERSION,
        "dataset_revision": _FROZEN_DATASET_CONFIG["dataset_revision"],
        "path": str(fixture_path),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "predecessor_path": str(fixture_path.with_name("state-cases-v3.json")),
        "predecessor_sha256": hashlib.sha256(fixture_path.with_name("state-cases-v3.json").read_bytes()).hexdigest(),
        "execution_profile_budgets_path": str(budget_path),
        "execution_profile_budgets_sha256": hashlib.sha256(budget_path.read_bytes()).hexdigest(),
        "case_config_sha256": frozen_config_sha256,
        "shared_runtime_config_sha256": shared_runtime_config_sha256,
        "case_count": len(cases),
        "functional_count": sum(item.classification == "functional" for item in cases),
        "security_count": sum(item.classification == "security" for item in cases),
        "paired_family_ids": sorted(_PAIR_CLASSES),
        "critical_question_ids": sorted(_CRITICAL_QUESTION_IDS),
        "cases": sorted(case_entries, key=lambda entry: str(entry["case_id"])),
    }


def load_supplement_cases(path: str | Path | None = None) -> tuple[StateCase, ...]:
    """Load development cases added after the frozen 20-case set.

    Supplement cases are replayed and reported separately, so the frozen set's
    manifest, quotas and historical denominators never change.  They must be
    functional, non-critical and must not reuse a frozen case or family id.
    """

    fixture_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[3] / "evals" / "development" / _SUPPLEMENT_FILE
    )
    try:
        document = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StateCaseError(f"cannot load supplement fixture: {fixture_path}") from exc
    if not isinstance(document, dict):
        raise StateCaseError("supplement root must be an object")
    _require_exact_fields(
        document,
        frozenset({"schema_version", "contract_version", "extension_version", "fixture_version", "supplement_of", "cases"}),
        where="supplement",
    )
    if document["schema_version"] != SUPPLEMENT_CASE_SET_VERSION:
        raise StateCaseError("unsupported supplement schema")
    if (
        document["contract_version"] != CONTRACT_VERSION
        or document["extension_version"] != EXTENSION_VERSION
        or document["fixture_version"] != FIXTURE_VERSION
    ):
        raise StateCaseError("supplement versions do not match the frozen development set")
    if document["supplement_of"] != "state-cases-v4.json":
        raise StateCaseError("supplement must extend state-cases-v4.json")
    raw_cases = document["cases"]
    if type(raw_cases) is not list or not raw_cases:
        raise StateCaseError("supplement cases must be a non-empty list")
    frozen = load_state_cases()
    frozen_ids = {item.case_id for item in frozen}
    frozen_families = {item.family_id for item in frozen}
    parsed: list[StateCase] = []
    for index, wrapper_raw in enumerate(raw_cases):
        wrapper = _require_mapping(wrapper_raw, where=f"cases[{index}]")
        _require_exact_fields(
            wrapper,
            frozenset({"case", "classification", "critical_question_id"}),
            where=f"cases[{index}]",
        )
        case = _validate_case(wrapper.get("case"), index=index)
        if wrapper.get("classification") != "functional" or wrapper.get("critical_question_id") is not None:
            raise StateCaseError(f"cases[{index}] supplement cases must be functional and non-critical")
        parsed.append(StateCase(case, "functional", None))
    ids = [item.case_id for item in parsed]
    families = [item.family_id for item in parsed]
    if len(set(ids)) != len(ids) or len(set(families)) != len(families):
        raise StateCaseError("supplement case_id and family_id values must be unique")
    if set(ids) & frozen_ids or set(families) & frozen_families:
        raise StateCaseError("supplement cases must not reuse frozen case or family ids")
    return tuple(parsed)


def supplement_case_manifest(path: str | Path | None = None) -> dict[str, object]:
    """Content-addressed metadata for the supplement set."""

    fixture_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[3] / "evals" / "development" / _SUPPLEMENT_FILE
    )
    cases = load_supplement_cases(fixture_path)
    return {
        "schema_version": SUPPLEMENT_CASE_SET_VERSION,
        "path": str(fixture_path),
        "sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "supplement_of": "state-cases-v4.json",
        "case_count": len(cases),
        "cases": sorted(
            (
                {
                    "case_id": item.case_id,
                    "family_id": item.family_id,
                    "classification": item.classification,
                    "input_sha256": canonical_sha256({"initial": item.case["initial"], "action": item.case["action"]}),
                    "expected_sha256": canonical_sha256(item.case["expected"]),
                }
                for item in cases
            ),
            key=lambda entry: str(entry["case_id"]),
        ),
    }


__all__ = [
    "CONTRACT_VERSION",
    "EXTENSION_VERSION",
    "FIXTURE_VERSION",
    "STATE_CASE_VERSION",
    "SUPPLEMENT_CASE_SET_VERSION",
    "load_supplement_cases",
    "supplement_case_manifest",
    "STATE_CASE_SET_VERSION",
    "StateCaseError",
    "StateCase",
    "resolve_actor_fixture",
    "resolve_principal_fixture",
    "canonical_sha256",
    "load_state_cases",
    "state_case_manifest",
]
