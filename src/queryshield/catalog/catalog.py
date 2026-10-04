from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


CATALOG_VERSION = "catalog-v1"
CATALOG_V2_VERSION = "catalog-v2"
CATALOG_V3_VERSION = "catalog-v3"
# catalog-v4 (B3c-1): the v3 phrase table without the generic ask markers
# "金额"/"总额", plus the business phrase "毛额" for gross_fen.
CATALOG_V4_VERSION = "catalog-v4"
# Versions that carry the phrase table (metric names/phrases, rule values).
PHRASE_TABLE_VERSIONS = frozenset({CATALOG_V3_VERSION, CATALOG_V4_VERSION})
FIXTURE_VERSION = "commerce-v1"
ALLOWED_ROLES = frozenset({"requester", "approver"})
ALLOWED_TABLE_COLUMNS: dict[str, frozenset[str]] = {
    "customers": frozenset({"tenant_id", "customer_id", "name"}),
    "orders": frozenset(
        {"tenant_id", "order_id", "customer_id", "status", "amount_fen", "created_at"}
    ),
    "refunds": frozenset(
        {"tenant_id", "refund_id", "order_id", "amount_fen", "created_at"}
    ),
}
ALLOWED_METRICS = frozenset({"paid_count", "gross_fen", "refund_fen", "net_fen"})


class CatalogValidationError(ValueError):
    """The catalog is not a safe, versioned description of the known schema."""


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    kind: str
    text: str
    source_id: str
    version: str
    table: str | None
    column: str | None
    field_refs: tuple[str, ...]
    access_scope: str
    requires_approval: bool
    payload: Mapping[str, object]

    def as_search_item(self) -> dict[str, str]:
        """Return only the public C3 search shape; richer metadata stays server-side."""
        return {
            "id": self.id,
            "text": self.text,
            "source_id": self.source_id,
            "version": self.version,
        }


@dataclass(frozen=True)
class ClarificationValue:
    """One allowed value of a clarification rule (catalog-v3).

    ``metric`` is the catalog metric that value means (a metric-valued rule)
    or the metric that declaring it implies (``paid`` <- ``paid_count``).
    ``phrases`` are the value's own explicit phrases; the phrases of
    ``metric`` also name the value.
    """

    value: str
    metric: str | None
    phrases: tuple[str, ...]
    definition: str | None
    supported: bool
    unsupported_note: str | None


@dataclass(frozen=True)
class ClarificationRule:
    id: str
    trigger_terms: tuple[str, ...]
    condition: str
    question: str
    allowed_values: tuple[str, ...]
    payload: Mapping[str, object]
    # catalog-v3 phrase table; empty for catalog-v1 rules.
    ambiguous_phrases: tuple[str, ...] = ()
    ask_markers: tuple[str, ...] = ()
    values: tuple[ClarificationValue, ...] = ()

    def value(self, value: str) -> ClarificationValue:
        for item in self.values:
            if item.value == value:
                return item
        raise KeyError(value)


@dataclass(frozen=True)
class CatalogOverlay:
    catalog_version: str
    base_catalog_version: str
    fixture_version: str
    source_id: str
    entries: tuple[CatalogEntry, ...]


@dataclass(frozen=True)
class SemanticCatalog:
    catalog_version: str
    fixture_version: str
    source_id: str
    source_files: tuple[str, ...]
    time_window: Mapping[str, str]
    access_scope: Mapping[str, object]
    entries: tuple[CatalogEntry, ...]
    clarifications: tuple[ClarificationRule, ...]

    def entry(self, entry_id: str) -> CatalogEntry:
        for item in self.entries:
            if item.id == entry_id:
                return item
        raise KeyError(entry_id)

    def metric(self, metric_id: str) -> CatalogEntry:
        item = self.entry(f"metric.{metric_id}")
        if item.kind != "metric":
            raise KeyError(metric_id)
        return item

    def search_items(self) -> tuple[dict[str, str], ...]:
        return tuple(item.as_search_item() for item in self.entries)

    def clarification(self, rule_id: str) -> ClarificationRule:
        for rule in self.clarifications:
            if rule.id == rule_id:
                return rule
        raise KeyError(rule_id)

    def metric_name(self, metric_id: str) -> str:
        """The catalog display name (catalog-v3), else the metric id."""

        name = self.metric(metric_id).payload.get("name")
        return name if type(name) is str and name else metric_id

    def metric_phrases(self, metric_id: str) -> tuple[str, ...]:
        phrases = self.metric(metric_id).payload.get("phrases")
        return tuple(phrases) if type(phrases) is list else ()

    @property
    def has_phrase_table(self) -> bool:
        return any(rule.values for rule in self.clarifications)


def _duplicate_check(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CatalogValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _as_string(value: object, *, path: str) -> str:
    if type(value) is not str or not value.strip():
        raise CatalogValidationError(f"{path} must be a non-empty string")
    return value


def _as_string_list(value: object, *, path: str, minimum: int = 1) -> tuple[str, ...]:
    if type(value) is not list or len(value) < minimum:
        raise CatalogValidationError(f"{path} must be a list with at least {minimum} item(s)")
    items = tuple(_as_string(item, path=f"{path}[]") for item in value)
    if len(set(items)) != len(items):
        raise CatalogValidationError(f"{path} must not contain duplicates")
    return items


def _as_mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CatalogValidationError(f"{path} must be an object")
    return value


def _validate_field_ref(field_ref: str, *, path: str) -> None:
    parts = field_ref.split(".")
    if len(parts) != 2:
        raise CatalogValidationError(f"{path} must be table.column")
    table, column = parts
    if table not in ALLOWED_TABLE_COLUMNS or column not in ALLOWED_TABLE_COLUMNS[table]:
        raise CatalogValidationError(f"{path} points outside the commerce-v1 schema")


def validate_catalog_document(document: Mapping[str, object]) -> None:
    """Validate the T01 asset before a later retrieval/tool layer can use it."""
    if not isinstance(document, Mapping):
        raise CatalogValidationError("catalog root must be an object")

    required_root = {
        "catalog_version",
        "fixture_version",
        "source",
        "time_window",
        "access_scope",
        "entries",
        "clarifications",
    }
    missing = required_root - set(document)
    if missing:
        raise CatalogValidationError(f"catalog root is missing: {sorted(missing)}")

    version = _as_string(document["catalog_version"], path="catalog_version")
    if version != CATALOG_VERSION and version not in PHRASE_TABLE_VERSIONS:
        raise CatalogValidationError("catalog_version must be catalog-v1, catalog-v3 or catalog-v4")
    if _as_string(document["fixture_version"], path="fixture_version") != FIXTURE_VERSION:
        raise CatalogValidationError("fixture_version must be commerce-v1")

    source = _as_mapping(document["source"], path="source")
    source_id = _as_string(source.get("source_id"), path="source.source_id")
    source_version = _as_string(source.get("version"), path="source.version")
    if source_id != FIXTURE_VERSION or source_version != FIXTURE_VERSION:
        raise CatalogValidationError("source must identify the commerce-v1 fixture")
    source_files = _as_string_list(source.get("files"), path="source.files", minimum=3)

    time_window = _as_mapping(document["time_window"], path="time_window")
    for key in ("start", "end", "timezone", "interval"):
        _as_string(time_window.get(key), path=f"time_window.{key}")
    if time_window["timezone"] != "UTC" or time_window["interval"] != "[start,end)":
        raise CatalogValidationError("time_window must use the fixed UTC half-open interval")

    access_scope = _as_mapping(document["access_scope"], path="access_scope")
    roles = _as_string_list(access_scope.get("roles"), path="access_scope.roles", minimum=1)
    if set(roles) != ALLOWED_ROLES:
        raise CatalogValidationError("access_scope.roles must be requester and approver")
    if access_scope.get("tenant_key") != "tenant_id":
        raise CatalogValidationError("access_scope.tenant_key must be tenant_id")
    if access_scope.get("database_access") != "guarded_query_executor_only":
        raise CatalogValidationError("catalog must point business data to the guarded executor")
    if access_scope.get("read_only") is not True:
        raise CatalogValidationError("catalog access must be read-only")
    if access_scope.get("model_may_supply_identity") is not False:
        raise CatalogValidationError("model identity fields must remain server-owned")

    raw_entries = document["entries"]
    if type(raw_entries) is not list or len(raw_entries) < 10:
        raise CatalogValidationError("entries must contain at least ten catalog records")

    seen_ids: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        path = f"entries[{index}]"
        entry = _as_mapping(raw_entry, path=path)
        entry_id = _as_string(entry.get("id"), path=f"{path}.id")
        if entry_id in seen_ids:
            raise CatalogValidationError(f"duplicate catalog entry id: {entry_id}")
        seen_ids.add(entry_id)

        kind = _as_string(entry.get("kind"), path=f"{path}.kind")
        if kind not in {"table", "field", "metric"}:
            raise CatalogValidationError(f"{path}.kind is unsupported")
        _as_string(entry.get("text"), path=f"{path}.text")
        if entry.get("source_id") != source_id or entry.get("version") != source_version:
            raise CatalogValidationError(f"{path} has an untraceable source/version")
        if entry.get("access_scope") != "tenant_scoped_readonly":
            raise CatalogValidationError(f"{path}.access_scope is not tenant scoped")
        if type(entry.get("requires_approval")) is not bool:
            raise CatalogValidationError(f"{path}.requires_approval must be boolean")

        table = entry.get("table")
        column = entry.get("column")
        if kind in {"table", "field"}:
            table_name = _as_string(table, path=f"{path}.table")
            if table_name not in ALLOWED_TABLE_COLUMNS:
                raise CatalogValidationError(f"{path}.table is not in commerce-v1")
            if kind == "field":
                column_name = _as_string(column, path=f"{path}.column")
                if column_name not in ALLOWED_TABLE_COLUMNS[table_name]:
                    raise CatalogValidationError(f"{path}.column is not in the real table")
            elif column is not None:
                raise CatalogValidationError(f"{path}.column must be null for a table entry")
        elif table is not None or column is not None:
            raise CatalogValidationError(f"{path} metric cannot have table/column fields")

        field_refs = _as_string_list(entry.get("field_refs"), path=f"{path}.field_refs")
        for ref in field_refs:
            _validate_field_ref(ref, path=f"{path}.field_refs[]")
        if kind == "metric":
            metric_id = entry_id.removeprefix("metric.")
            if not entry_id.startswith("metric.") or metric_id not in ALLOWED_METRICS:
                raise CatalogValidationError(f"{path}.id must be a supported metric")
            if _as_string(entry.get("unit"), path=f"{path}.unit") not in {"count", "CNY_fen"}:
                raise CatalogValidationError(f"{path}.unit is unsupported")
        if "source_locator" not in entry:
            raise CatalogValidationError(f"{path}.source_locator is required")
        _as_string(entry["source_locator"], path=f"{path}.source_locator")
        if version in PHRASE_TABLE_VERSIONS and kind == "metric":
            _as_string(entry.get("name"), path=f"{path}.name")
            _as_string_list(entry.get("phrases"), path=f"{path}.phrases")

    raw_clarifications = document["clarifications"]
    if type(raw_clarifications) is not list or len(raw_clarifications) < 3:
        raise CatalogValidationError("at least three clarification rules are required")
    seen_clarifications: set[str] = set()
    for index, raw_rule in enumerate(raw_clarifications):
        path = f"clarifications[{index}]"
        rule = _as_mapping(raw_rule, path=path)
        rule_id = _as_string(rule.get("id"), path=f"{path}.id")
        if rule_id in seen_clarifications:
            raise CatalogValidationError(f"duplicate clarification id: {rule_id}")
        seen_clarifications.add(rule_id)
        if version == CATALOG_VERSION:
            _as_string_list(rule.get("trigger_terms"), path=f"{path}.trigger_terms")
        elif "trigger_terms" in rule:
            raise CatalogValidationError(f"{path}.trigger_terms is replaced by the catalog-v3 phrase table")
        _as_string(rule.get("condition"), path=f"{path}.condition")
        _as_string(rule.get("question"), path=f"{path}.question")
        _as_string_list(rule.get("allowed_values"), path=f"{path}.allowed_values")
        _as_string(rule.get("resolution"), path=f"{path}.resolution")
    if version in PHRASE_TABLE_VERSIONS:
        _validate_phrase_table(raw_entries, raw_clarifications)


def _validate_phrase_table(raw_entries: list[object], raw_rules: list[object]) -> None:
    """catalog-v3: every phrase names exactly one thing; values are well formed."""

    metric_ids = {
        str(entry["id"]).removeprefix("metric.")
        for entry in raw_entries
        if isinstance(entry, Mapping) and entry.get("kind") == "metric"
    }
    explicit_owner: dict[str, str] = {}

    def claim(phrase: str, owner: str, path: str) -> None:
        key = phrase.casefold()
        if key in explicit_owner and explicit_owner[key] != owner:
            raise CatalogValidationError(f"{path} phrase {phrase!r} already names {explicit_owner[key]}")
        explicit_owner[key] = owner

    for entry in raw_entries:
        if isinstance(entry, Mapping) and entry.get("kind") == "metric":
            for phrase in entry["phrases"]:
                claim(phrase, str(entry["id"]), f"{entry['id']}.phrases")

    ambiguous_all: set[str] = set()
    for index, raw_rule in enumerate(raw_rules):
        path = f"clarifications[{index}]"
        rule = _as_mapping(raw_rule, path=path)
        allowed = list(rule["allowed_values"])
        ambiguous = rule.get("ambiguous_phrases")
        if type(ambiguous) is not list or any(type(item) is not str or not item.strip() for item in ambiguous):
            raise CatalogValidationError(f"{path}.ambiguous_phrases must be a list of phrases")
        if len(allowed) > 1 and not ambiguous:
            raise CatalogValidationError(f"{path} with several values needs ambiguous_phrases")
        ambiguous_all.update(item.casefold() for item in ambiguous)
        _as_string_list(rule.get("ask_markers"), path=f"{path}.ask_markers")
        values = rule.get("values")
        if type(values) is not list or [
            item.get("value") if isinstance(item, Mapping) else None for item in values
        ] != allowed:
            raise CatalogValidationError(f"{path}.values must describe allowed_values in order")
        for value_index, raw_value in enumerate(values):
            value_path = f"{path}.values[{value_index}]"
            value = _as_mapping(raw_value, path=value_path)
            unknown = set(value) - {"value", "metric", "phrases", "definition", "supported", "unsupported_note"}
            if unknown:
                raise CatalogValidationError(f"{value_path} has unknown fields: {sorted(unknown)}")
            metric = value.get("metric")
            if metric is not None and metric not in metric_ids:
                raise CatalogValidationError(f"{value_path}.metric must be a catalog metric")
            if value["value"] in metric_ids and metric != value["value"]:
                raise CatalogValidationError(f"{value_path} metric value must name itself")
            own = value.get("phrases", [])
            if metric is None or own:
                own = list(_as_string_list(own, path=f"{value_path}.phrases"))
            for phrase in own:
                claim(phrase, f"{rule['id']}={value['value']}", f"{value_path}.phrases")
            if metric is None:
                _as_string(value.get("definition"), path=f"{value_path}.definition")
            supported = value.get("supported", True)
            if type(supported) is not bool:
                raise CatalogValidationError(f"{value_path}.supported must be boolean")
            if not supported:
                _as_string(value.get("unsupported_note"), path=f"{value_path}.unsupported_note")
                if metric is not None:
                    raise CatalogValidationError(f"{value_path} an unsupported value cannot name a metric")
            elif "unsupported_note" in value:
                raise CatalogValidationError(f"{value_path}.unsupported_note is only for unsupported values")
    overlap = ambiguous_all & set(explicit_owner)
    if overlap:
        raise CatalogValidationError(f"ambiguous phrases cannot also be explicit: {sorted(overlap)}")


def validate_catalog_overlay_document(document: Mapping[str, object]) -> None:
    """Validate a small catalog-v2 delta without duplicating catalog-v1."""
    if not isinstance(document, Mapping):
        raise CatalogValidationError("catalog overlay root must be an object")
    required = {
        "catalog_version",
        "base_catalog_version",
        "fixture_version",
        "source",
        "entries",
    }
    missing = required - set(document)
    if missing:
        raise CatalogValidationError(f"catalog overlay is missing: {sorted(missing)}")
    if document["catalog_version"] != CATALOG_V2_VERSION:
        raise CatalogValidationError("catalog overlay must be catalog-v2")
    if document["base_catalog_version"] != CATALOG_VERSION:
        raise CatalogValidationError("catalog overlay must extend catalog-v1")
    if document["fixture_version"] != FIXTURE_VERSION:
        raise CatalogValidationError("catalog overlay must use commerce-v1")

    source = _as_mapping(document["source"], path="overlay.source")
    source_id = _as_string(source.get("source_id"), path="overlay.source.source_id")
    source_version = _as_string(source.get("version"), path="overlay.source.version")
    if source_id != FIXTURE_VERSION or source_version != FIXTURE_VERSION:
        raise CatalogValidationError("catalog-v2 source must identify commerce-v1")
    _as_string_list(source.get("files"), path="overlay.source.files", minimum=1)

    entries = document["entries"]
    if type(entries) is not list or not entries:
        raise CatalogValidationError("catalog overlay entries must not be empty")
    seen_ids: set[str] = set()
    for index, raw_entry in enumerate(entries):
        path = f"overlay.entries[{index}]"
        entry = _as_mapping(raw_entry, path=path)
        entry_id = _as_string(entry.get("id"), path=f"{path}.id")
        if entry_id in seen_ids:
            raise CatalogValidationError(f"duplicate overlay entry id: {entry_id}")
        seen_ids.add(entry_id)
        if entry_id != "metric.refund_fen" or entry.get("kind") != "metric":
            raise CatalogValidationError("catalog-v2 currently only registers metric.refund_fen")
        _as_string(entry.get("text"), path=f"{path}.text")
        if entry.get("source_id") != source_id or entry.get("version") != source_version:
            raise CatalogValidationError(f"{path} has an untraceable source/version")
        _as_string(entry.get("source_locator"), path=f"{path}.source_locator")
        field_refs = _as_string_list(entry.get("field_refs"), path=f"{path}.field_refs")
        for ref in field_refs:
            _validate_field_ref(ref, path=f"{path}.field_refs[]")
        if entry.get("unit") != "CNY_fen":
            raise CatalogValidationError("catalog-v2 refund_fen must use CNY_fen")
        _as_string(entry.get("definition"), path=f"{path}.definition")
        fixed_values = entry.get("fixed_tenant_values")
        if not isinstance(fixed_values, Mapping) or set(fixed_values) != {"A", "B"}:
            raise CatalogValidationError(
                f"{path}.fixed_tenant_values must contain exactly A and B"
            )
        if fixed_values["A"] != 3000 or fixed_values["B"] != 10000:
            raise CatalogValidationError(
                f"{path}.fixed_tenant_values must be A=3000 and B=10000 fen"
            )
        if any(type(value) is not int or value < 0 for value in fixed_values.values()):
            raise CatalogValidationError(f"{path}.fixed_tenant_values must be non-negative integers")
        if entry.get("access_scope") != "tenant_scoped_readonly":
            raise CatalogValidationError(f"{path}.access_scope is not tenant scoped")
        if type(entry.get("requires_approval")) is not bool:
            raise CatalogValidationError(f"{path}.requires_approval must be boolean")


def _parse_entry(raw_entry: Mapping[str, object]) -> CatalogEntry:
    return CatalogEntry(
        id=_as_string(raw_entry["id"], path="entry.id"),
        kind=_as_string(raw_entry["kind"], path="entry.kind"),
        text=_as_string(raw_entry["text"], path="entry.text"),
        source_id=_as_string(raw_entry["source_id"], path="entry.source_id"),
        version=_as_string(raw_entry["version"], path="entry.version"),
        table=raw_entry.get("table") if isinstance(raw_entry.get("table"), str) else None,
        column=raw_entry.get("column") if isinstance(raw_entry.get("column"), str) else None,
        field_refs=tuple(raw_entry["field_refs"]),
        access_scope=_as_string(raw_entry["access_scope"], path="entry.access_scope"),
        requires_approval=raw_entry["requires_approval"] is True,
        payload=raw_entry,
    )


def _parse_clarification(raw_rule: Mapping[str, object]) -> ClarificationRule:
    values = tuple(
        ClarificationValue(
            value=str(item["value"]),
            metric=item.get("metric") if type(item.get("metric")) is str else None,
            phrases=tuple(item.get("phrases", ())),
            definition=item.get("definition") if type(item.get("definition")) is str else None,
            supported=item.get("supported", True) is True,
            unsupported_note=item.get("unsupported_note") if type(item.get("unsupported_note")) is str else None,
        )
        for item in raw_rule.get("values", ())
    )
    return ClarificationRule(
        id=_as_string(raw_rule["id"], path="clarification.id"),
        trigger_terms=tuple(raw_rule.get("trigger_terms", ())),
        condition=_as_string(raw_rule["condition"], path="clarification.condition"),
        question=_as_string(raw_rule["question"], path="clarification.question"),
        allowed_values=tuple(raw_rule["allowed_values"]),
        payload=raw_rule,
        ambiguous_phrases=tuple(raw_rule.get("ambiguous_phrases", ())),
        ask_markers=tuple(raw_rule.get("ask_markers", ())),
        values=values,
    )


def load_catalog(path: str | Path) -> SemanticCatalog:
    catalog_path = Path(path)
    try:
        document = json.loads(
            catalog_path.read_text(encoding="utf-8"), object_pairs_hook=_duplicate_check
        )
    except CatalogValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CatalogValidationError(f"cannot load catalog: {catalog_path}") from exc

    validate_catalog_document(document)
    source = document["source"]
    entries = tuple(_parse_entry(item) for item in document["entries"])
    clarifications = tuple(_parse_clarification(item) for item in document["clarifications"])
    return SemanticCatalog(
        catalog_version=document["catalog_version"],
        fixture_version=document["fixture_version"],
        source_id=source["source_id"],
        source_files=tuple(source["files"]),
        time_window=document["time_window"],
        access_scope=document["access_scope"],
        entries=entries,
        clarifications=clarifications,
    )


def load_catalog_overlay(path: str | Path) -> CatalogOverlay:
    overlay_path = Path(path)
    try:
        document = json.loads(
            overlay_path.read_text(encoding="utf-8"), object_pairs_hook=_duplicate_check
        )
    except CatalogValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CatalogValidationError(f"cannot load catalog overlay: {overlay_path}") from exc

    validate_catalog_overlay_document(document)
    source = document["source"]
    entries = tuple(_parse_entry(item) for item in document["entries"])
    return CatalogOverlay(
        catalog_version=document["catalog_version"],
        base_catalog_version=document["base_catalog_version"],
        fixture_version=document["fixture_version"],
        source_id=source["source_id"],
        entries=entries,
    )


# The product catalog.  Every default catalog version in the product
# (executor evidence, run configuration, approvals) reads this constant; a test
# pins it to the version inside DEFAULT_CATALOG_PATH.
DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[3] / "fixtures" / "semantic" / "catalog-v4.json"
DEFAULT_CATALOG_VERSION = CATALOG_V4_VERSION


def load_default_catalog() -> SemanticCatalog:
    return load_catalog(DEFAULT_CATALOG_PATH)
