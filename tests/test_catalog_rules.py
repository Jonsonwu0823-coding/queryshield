"""Catalog loading and validation messages, and the phrase-table texts, pinned before they are tidied.

Every validation message is part of what an operator sees when a catalog file is
wrong, and every phrase-table text reaches the model or the answer, so they are
compared verbatim.  The mutation tables start from the shipped catalog files.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import pytest

from queryshield.catalog import load_default_catalog
from queryshield.catalog.catalog import (
    CatalogValidationError,
    load_catalog,
    load_catalog_overlay,
    validate_catalog_document,
    validate_catalog_overlay_document,
)
from queryshield.catalog.phrases import (
    check_declaration,
    contradiction_hint,
    metric_basis_note,
    not_needed_hint,
    read_clarifications,
    review_ask,
    rule_named_by_waiting_question,
    select_value,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "semantic"
_PATH = re.compile(r"[^.\[\]]+|\[\d+\]")


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _locate(document, path: str):
    parts = _PATH.findall(path)
    node = document
    for part in parts[:-1]:
        node = node[int(part[1:-1])] if part.startswith("[") else node[part]
    last = parts[-1]
    return node, (int(last[1:-1]) if last.startswith("[") else last)


def put(path: str, value):
    def apply(document) -> None:
        node, key = _locate(document, path)
        node[key] = value

    return apply


def drop(path: str):
    def apply(document) -> None:
        node, key = _locate(document, path)
        del node[key]

    return apply


# (label, catalog file, change, the message the validator gives)
DOCUMENT_CASES = [
    ('missing_entries', 'catalog-v4.json', drop("entries"), "catalog root is missing: ['entries']"),
    ('missing_two', 'catalog-v4.json', lambda d: (d.pop("source"), d.pop("time_window")), "catalog root is missing: ['source', 'time_window']"),
    ('bad_version', 'catalog-v4.json', put("catalog_version", "catalog-v9"), 'catalog_version must be catalog-v1, catalog-v3 or catalog-v4'),
    ('blank_version', 'catalog-v4.json', put("catalog_version", " "), 'catalog_version must be a non-empty string'),
    ('bad_fixture', 'catalog-v4.json', put("fixture_version", "commerce-v2"), 'fixture_version must be commerce-v1'),
    ('source_not_object', 'catalog-v4.json', put("source", []), 'source must be an object'),
    ('source_id', 'catalog-v4.json', put("source.source_id", "other"), 'source must identify the commerce-v1 fixture'),
    ('source_version', 'catalog-v4.json', put("source.version", "other"), 'source must identify the commerce-v1 fixture'),
    ('source_files_short', 'catalog-v4.json', put("source.files", ["a", "b"]), 'source.files must be a list with at least 3 item(s)'),
    ('source_files_dupes', 'catalog-v4.json', put("source.files", ["a", "a", "b"]), 'source.files must not contain duplicates'),
    ('source_files_not_list', 'catalog-v4.json', put("source.files", "a"), 'source.files must be a list with at least 3 item(s)'),
    ('window_missing_interval', 'catalog-v4.json', drop("time_window.interval"), 'time_window.interval must be a non-empty string'),
    ('window_timezone', 'catalog-v4.json', put("time_window.timezone", "CST"), 'time_window must use the fixed UTC half-open interval'),
    ('window_interval', 'catalog-v4.json', put("time_window.interval", "[start,end]"), 'time_window must use the fixed UTC half-open interval'),
    ('access_not_object', 'catalog-v4.json', put("access_scope", 1), 'access_scope must be an object'),
    ('roles', 'catalog-v4.json', put("access_scope.roles", ["requester"]), 'access_scope.roles must be requester and approver'),
    ('roles_empty', 'catalog-v4.json', put("access_scope.roles", []), 'access_scope.roles must be a list with at least 1 item(s)'),
    ('tenant_key', 'catalog-v4.json', put("access_scope.tenant_key", "org_id"), 'access_scope.tenant_key must be tenant_id'),
    ('database_access', 'catalog-v4.json', put("access_scope.database_access", "direct"), 'catalog must point business data to the guarded executor'),
    ('read_only', 'catalog-v4.json', put("access_scope.read_only", False), 'catalog access must be read-only'),
    ('identity', 'catalog-v4.json', put("access_scope.model_may_supply_identity", True), 'model identity fields must remain server-owned'),
    ('entries_few', 'catalog-v4.json', lambda d: d.__setitem__("entries", d["entries"][:9]), 'entries must contain at least ten catalog records'),
    ('entries_not_list', 'catalog-v4.json', put("entries", {}), 'entries must contain at least ten catalog records'),
    ('entry_not_object', 'catalog-v4.json', put("entries[2]", "x"), 'entries[2] must be an object'),
    ('entry_dup_id', 'catalog-v4.json', put("entries[2].id", "field.customers.tenant_id"), 'duplicate catalog entry id: field.customers.tenant_id'),
    ('entry_id_blank', 'catalog-v4.json', put("entries[2].id", ""), 'entries[2].id must be a non-empty string'),
    ('entry_kind', 'catalog-v4.json', put("entries[2].kind", "view"), 'entries[2].kind is unsupported'),
    ('entry_text', 'catalog-v4.json', put("entries[2].text", "  "), 'entries[2].text must be a non-empty string'),
    ('entry_source', 'catalog-v4.json', put("entries[2].source_id", "other"), 'entries[2] has an untraceable source/version'),
    ('entry_version', 'catalog-v4.json', put("entries[2].version", "other"), 'entries[2] has an untraceable source/version'),
    ('entry_access', 'catalog-v4.json', put("entries[2].access_scope", "public"), 'entries[2].access_scope is not tenant scoped'),
    ('entry_approval', 'catalog-v4.json', put("entries[2].requires_approval", "no"), 'entries[2].requires_approval must be boolean'),
    ('entry_table_unknown', 'catalog-v4.json', put("entries[2].table", "invoices"), 'entries[2].table is not in commerce-v1'),
    ('entry_table_blank', 'catalog-v4.json', put("entries[0].table", ""), 'entries[0].table must be a non-empty string'),
    ('entry_table_column', 'catalog-v4.json', put("entries[0].column", "name"), 'entries[0].column must be null for a table entry'),
    ('field_column_unknown', 'catalog-v4.json', put("entries[2].column", "nickname"), 'entries[2].column is not in the real table'),
    ('field_column_missing', 'catalog-v4.json', put("entries[2].column", None), 'entries[2].column must be a non-empty string'),
    ('metric_with_table', 'catalog-v4.json', put("entries[15].table", "orders"), 'entries[15] metric cannot have table/column fields'),
    ('metric_with_column', 'catalog-v4.json', put("entries[15].column", "status"), 'entries[15] metric cannot have table/column fields'),
    ('field_refs_missing', 'catalog-v4.json', drop("entries[2].field_refs"), 'entries[2].field_refs must be a list with at least 1 item(s)'),
    ('field_refs_dupes', 'catalog-v4.json', put("entries[2].field_refs", ["customers.name", "customers.name"]), 'entries[2].field_refs must not contain duplicates'),
    ('field_ref_shape', 'catalog-v4.json', put("entries[2].field_refs", ["customers"]), 'entries[2].field_refs[] must be table.column'),
    ('field_ref_outside', 'catalog-v4.json', put("entries[2].field_refs", ["customers.nickname"]), 'entries[2].field_refs[] points outside the commerce-v1 schema'),
    ('metric_unsupported', 'catalog-v4.json', put("entries[15].id", "metric.bogus"), 'entries[15].id must be a supported metric'),
    ('metric_no_prefix', 'catalog-v4.json', put("entries[15].id", "paid_count"), 'entries[15].id must be a supported metric'),
    ('metric_unit_missing', 'catalog-v4.json', drop("entries[15].unit"), 'entries[15].unit must be a non-empty string'),
    ('metric_unit', 'catalog-v4.json', put("entries[15].unit", "USD"), 'entries[15].unit is unsupported'),
    ('locator_missing', 'catalog-v4.json', drop("entries[2].source_locator"), 'entries[2].source_locator is required'),
    ('locator_blank', 'catalog-v4.json', put("entries[2].source_locator", ""), 'entries[2].source_locator must be a non-empty string'),
    ('metric_name_missing', 'catalog-v4.json', drop("entries[15].name"), 'entries[15].name must be a non-empty string'),
    ('metric_phrases_empty', 'catalog-v4.json', put("entries[15].phrases", []), 'entries[15].phrases must be a list with at least 1 item(s)'),
    ('clarifications_few', 'catalog-v4.json', lambda d: d.__setitem__("clarifications", d["clarifications"][:2]), 'at least three clarification rules are required'),
    ('clarification_not_object', 'catalog-v4.json', put("clarifications[0]", 3), 'clarifications[0] must be an object'),
    ('clarification_dup', 'catalog-v4.json', put("clarifications[1].id", "clarify.metric_basis"), 'duplicate clarification id: clarify.metric_basis'),
    ('clarification_id_blank', 'catalog-v4.json', put("clarifications[1].id", " "), 'clarifications[1].id must be a non-empty string'),
    ('clarification_trigger_terms', 'catalog-v4.json', put("clarifications[0].trigger_terms", ["x"]), 'clarifications[0].trigger_terms is replaced by the catalog-v3 phrase table'),
    ('clarification_condition', 'catalog-v4.json', put("clarifications[0].condition", ""), 'clarifications[0].condition must be a non-empty string'),
    ('clarification_question', 'catalog-v4.json', drop("clarifications[0].question"), 'clarifications[0].question must be a non-empty string'),
    ('clarification_allowed', 'catalog-v4.json', put("clarifications[0].allowed_values", []), 'clarifications[0].allowed_values must be a list with at least 1 item(s)'),
    ('clarification_resolution', 'catalog-v4.json', put("clarifications[0].resolution", ""), 'clarifications[0].resolution must be a non-empty string'),
    ('ambiguous_type', 'catalog-v4.json', put("clarifications[0].ambiguous_phrases", "x"), 'clarifications[0].ambiguous_phrases must be a list of phrases'),
    ('ambiguous_blank_item', 'catalog-v4.json', put("clarifications[0].ambiguous_phrases", [" "]), 'clarifications[0].ambiguous_phrases must be a list of phrases'),
    ('ambiguous_missing_for_many', 'catalog-v4.json', put("clarifications[0].ambiguous_phrases", []), 'clarifications[0] with several values needs ambiguous_phrases'),
    ('ask_markers_missing', 'catalog-v4.json', drop("clarifications[0].ask_markers"), 'clarifications[0].ask_markers must be a list with at least 1 item(s)'),
    ('values_order', 'catalog-v4.json', put("clarifications[0].values", [{"value": "net_fen", "metric": "net_fen"}, {"value": "gross_fen", "metric": "gross_fen"}]), 'clarifications[0].values must describe allowed_values in order'),
    ('values_not_list', 'catalog-v4.json', put("clarifications[0].values", None), 'clarifications[0].values must describe allowed_values in order'),
    ('value_not_object', 'catalog-v4.json', put("clarifications[0].values[0]", "x"), 'clarifications[0].values must describe allowed_values in order'),
    ('value_unknown_field', 'catalog-v4.json', put("clarifications[0].values[0].colour", "x"), "clarifications[0].values[0] has unknown fields: ['colour']"),
    ('value_metric_unknown', 'catalog-v4.json', put("clarifications[0].values[0].metric", "bogus"), 'clarifications[0].values[0].metric must be a catalog metric'),
    ('value_names_itself', 'catalog-v4.json', put("clarifications[0].values[0].metric", "net_fen"), 'clarifications[0].values[0] metric value must name itself'),
    ('phrase_claimed_twice', 'catalog-v4.json', put("clarifications[0].values[0].phrases", ["退款后净额"]), "clarifications[0].values[0].phrases phrase '退款后净额' already names metric.net_fen"),
    ('value_phrases_bad', 'catalog-v4.json', put("clarifications[0].values[0].phrases", [""]), 'clarifications[0].values[0].phrases[] must be a non-empty string'),
    ('value_supported_type', 'catalog-v4.json', put("clarifications[0].values[0].supported", "yes"), 'clarifications[0].values[0].supported must be boolean'),
    ('value_unsupported_no_note', 'catalog-v4.json', put("clarifications[0].values[0].supported", False), 'clarifications[0].values[0].unsupported_note must be a non-empty string'),
    ('value_unsupported_with_metric', 'catalog-v4.json', lambda d: d["clarifications"][0]["values"][0].update({"supported": False, "unsupported_note": "n"}), 'clarifications[0].values[0] an unsupported value cannot name a metric'),
    ('value_note_for_supported', 'catalog-v4.json', put("clarifications[0].values[0].unsupported_note", "n"), 'clarifications[0].values[0].unsupported_note is only for unsupported values'),
    ('ambiguous_also_explicit', 'catalog-v4.json', put("clarifications[0].ambiguous_phrases", ["销售额", "毛额"]), "ambiguous phrases cannot also be explicit: ['毛额']"),
    ('v1_trigger_terms_missing', 'catalog-v1.json', drop("clarifications[0].trigger_terms"), 'clarifications[0].trigger_terms must be a list with at least 1 item(s)'),
    ('v1_trigger_terms_empty', 'catalog-v1.json', put("clarifications[0].trigger_terms", []), 'clarifications[0].trigger_terms must be a list with at least 1 item(s)'),
    ('v1_bad_version_phrase_free', 'catalog-v1.json', put("catalog_version", "catalog-v2"), 'catalog_version must be catalog-v1, catalog-v3 or catalog-v4'),
]

OVERLAY_CASES = [
    ('missing', 'catalog-v2.json', drop("entries"), "catalog overlay is missing: ['entries']"),
    ('version', 'catalog-v2.json', put("catalog_version", "catalog-v3"), 'catalog overlay must be catalog-v2'),
    ('base', 'catalog-v2.json', put("base_catalog_version", "catalog-v3"), 'catalog overlay must extend catalog-v1'),
    ('fixture', 'catalog-v2.json', put("fixture_version", "x"), 'catalog overlay must use commerce-v1'),
    ('source_not_object', 'catalog-v2.json', put("source", None), 'overlay.source must be an object'),
    ('source_id', 'catalog-v2.json', put("source.source_id", "x"), 'catalog-v2 source must identify commerce-v1'),
    ('source_version', 'catalog-v2.json', put("source.version", "x"), 'catalog-v2 source must identify commerce-v1'),
    ('source_files_empty', 'catalog-v2.json', put("source.files", []), 'overlay.source.files must be a list with at least 1 item(s)'),
    ('entries_empty', 'catalog-v2.json', put("entries", []), 'catalog overlay entries must not be empty'),
    ('entries_not_list', 'catalog-v2.json', put("entries", {}), 'catalog overlay entries must not be empty'),
    ('entry_not_object', 'catalog-v2.json', put("entries[0]", 1), 'overlay.entries[0] must be an object'),
    ('entry_id', 'catalog-v2.json', put("entries[0].id", "metric.gross_fen"), 'catalog-v2 currently only registers metric.refund_fen'),
    ('entry_kind', 'catalog-v2.json', put("entries[0].kind", "field"), 'catalog-v2 currently only registers metric.refund_fen'),
    ('entry_dup', 'catalog-v2.json', lambda d: d["entries"].append(copy.deepcopy(d["entries"][0])), 'duplicate overlay entry id: metric.refund_fen'),
    ('entry_text', 'catalog-v2.json', put("entries[0].text", ""), 'overlay.entries[0].text must be a non-empty string'),
    ('entry_source', 'catalog-v2.json', put("entries[0].source_id", "x"), 'overlay.entries[0] has an untraceable source/version'),
    ('entry_version', 'catalog-v2.json', put("entries[0].version", "x"), 'overlay.entries[0] has an untraceable source/version'),
    ('entry_locator', 'catalog-v2.json', put("entries[0].source_locator", " "), 'overlay.entries[0].source_locator must be a non-empty string'),
    ('entry_field_refs_missing', 'catalog-v2.json', drop("entries[0].field_refs"), 'overlay.entries[0].field_refs must be a list with at least 1 item(s)'),
    ('entry_field_refs_dupes', 'catalog-v2.json', put("entries[0].field_refs", ["orders.status", "orders.status"]), 'overlay.entries[0].field_refs must not contain duplicates'),
    ('entry_field_ref_shape', 'catalog-v2.json', put("entries[0].field_refs", ["orders"]), 'overlay.entries[0].field_refs[] must be table.column'),
    ('entry_field_ref_outside', 'catalog-v2.json', put("entries[0].field_refs", ["orders.nickname"]), 'overlay.entries[0].field_refs[] points outside the commerce-v1 schema'),
    ('entry_unit', 'catalog-v2.json', put("entries[0].unit", "count"), 'catalog-v2 refund_fen must use CNY_fen'),
    ('entry_definition', 'catalog-v2.json', put("entries[0].definition", ""), 'overlay.entries[0].definition must be a non-empty string'),
    ('fixed_not_object', 'catalog-v2.json', put("entries[0].fixed_tenant_values", [1]), 'overlay.entries[0].fixed_tenant_values must contain exactly A and B'),
    ('fixed_keys', 'catalog-v2.json', put("entries[0].fixed_tenant_values", {"A": 3000}), 'overlay.entries[0].fixed_tenant_values must contain exactly A and B'),
    ('fixed_values', 'catalog-v2.json', put("entries[0].fixed_tenant_values", {"A": 1, "B": 10000}), 'overlay.entries[0].fixed_tenant_values must be A=3000 and B=10000 fen'),
    ('fixed_float', 'catalog-v2.json', put("entries[0].fixed_tenant_values", {"A": 3000.0, "B": 10000}), 'overlay.entries[0].fixed_tenant_values must be non-negative integers'),
    ('entry_access', 'catalog-v2.json', put("entries[0].access_scope", "public"), 'overlay.entries[0].access_scope is not tenant scoped'),
    ('entry_approval', 'catalog-v2.json', put("entries[0].requires_approval", 1), 'overlay.entries[0].requires_approval must be boolean'),
]


@pytest.mark.parametrize("label, name, change, message", DOCUMENT_CASES, ids=[case[0] for case in DOCUMENT_CASES])
def test_catalog_validation_reports_each_fault_with_its_message(label, name, change, message) -> None:
    document = _load(name)
    change(document)
    with pytest.raises(CatalogValidationError) as caught:
        validate_catalog_document(document)
    assert str(caught.value) == message


@pytest.mark.parametrize("label, name, change, message", OVERLAY_CASES, ids=[case[0] for case in OVERLAY_CASES])
def test_overlay_validation_reports_each_fault_with_its_message(label, name, change, message) -> None:
    document = _load(name)
    change(document)
    with pytest.raises(CatalogValidationError) as caught:
        validate_catalog_overlay_document(document)
    assert str(caught.value) == message


def test_a_value_definition_may_be_blank_when_the_value_names_a_metric() -> None:
    document = _load("catalog-v4.json")
    put("clarifications[1].values[0].definition", "")(document)
    validate_catalog_document(document)


def test_the_first_fault_in_validation_order_is_the_one_reported() -> None:
    document = _load("catalog-v4.json")
    put("time_window.timezone", "CST")(document)
    put("source.version", "other")(document)
    put("catalog_version", "catalog-v9")(document)
    with pytest.raises(CatalogValidationError, match="catalog_version must be"):
        validate_catalog_document(document)
    put("catalog_version", "catalog-v4")(document)
    with pytest.raises(CatalogValidationError, match="source must identify"):
        validate_catalog_document(document)
    put("source.version", "commerce-v1")(document)
    with pytest.raises(CatalogValidationError, match="fixed UTC half-open"):
        validate_catalog_document(document)
    put("time_window.timezone", "UTC")(document)
    put("entries[3].kind", "view")(document)
    put("entries[3].text", "")(document)
    put("entries[1].requires_approval", "no")(document)
    with pytest.raises(CatalogValidationError, match=r"entries\[1\]\.requires_approval"):
        validate_catalog_document(document)
    put("entries[1].requires_approval", False)(document)
    with pytest.raises(CatalogValidationError, match=r"entries\[3\]\.kind is unsupported"):
        validate_catalog_document(document)


@pytest.mark.parametrize("bad", [[], "x", None, 3])
def test_a_root_that_is_not_an_object_is_refused(bad) -> None:
    with pytest.raises(CatalogValidationError) as caught:
        validate_catalog_document(bad)
    assert str(caught.value) == "catalog root must be an object"
    with pytest.raises(CatalogValidationError) as caught:
        validate_catalog_overlay_document(bad)
    assert str(caught.value) == "catalog overlay root must be an object"


def test_loaders_report_unreadable_files_and_duplicate_keys_with_fixed_messages(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    for loader, label in ((load_catalog, "catalog"), (load_catalog_overlay, "catalog overlay")):
        with pytest.raises(CatalogValidationError) as caught:
            loader(missing)
        assert str(caught.value) == f"cannot load {label}: {missing}"
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not json", encoding="utf-8")
        with pytest.raises(CatalogValidationError) as caught:
            loader(str(bad_json))
        assert str(caught.value) == f"cannot load {label}: {bad_json}"
        bad_bytes = tmp_path / "bytes.json"
        bad_bytes.write_bytes(b"\xff\xfe\x00")
        with pytest.raises(CatalogValidationError) as caught:
            loader(bad_bytes)
        assert str(caught.value) == f"cannot load {label}: {bad_bytes}"
        duplicate = tmp_path / "duplicate.json"
        duplicate.write_text('{"catalog_version": "catalog-v4", "catalog_version": "catalog-v4"}', encoding="utf-8")
        with pytest.raises(CatalogValidationError) as caught:
            loader(duplicate)
        assert str(caught.value) == "duplicate JSON key: catalog_version"
        not_an_object = tmp_path / "list.json"
        not_an_object.write_text("[]", encoding="utf-8")
        with pytest.raises(CatalogValidationError) as caught:
            loader(not_an_object)
        assert str(caught.value) == ("catalog root must be an object" if label == "catalog" else "catalog overlay root must be an object")


def test_the_shipped_catalogs_load_into_the_objects_the_product_reads() -> None:
    catalog = load_default_catalog()
    assert catalog.catalog_version == "catalog-v4" and catalog.source_id == "commerce-v1"
    assert [entry.id for entry in catalog.entries][-4:] == [
        "metric.paid_count",
        "metric.gross_fen",
        "metric.refund_fen",
        "metric.net_fen",
    ]
    customer_name = catalog.entry("field.customers.name")
    assert (customer_name.table, customer_name.column, customer_name.requires_approval) == ("customers", "name", True)
    assert catalog.entry("table.orders").column is None and catalog.entry("table.orders").requires_approval is False
    rule = catalog.clarification("clarify.metric_basis")
    assert [(value.value, value.metric, value.phrases, value.definition, value.supported, value.unsupported_note) for value in rule.values] == [
        ("gross_fen", "gross_fen", (), None, True, None),
        ("net_fen", "net_fen", (), None, True, None),
    ]
    assert rule.trigger_terms == () and rule.allowed_values == ("gross_fen", "net_fen")
    scope = catalog.clarification("clarify.order_status_scope")
    cancelled = scope.value("cancelled")
    assert (cancelled.supported, bool(cancelled.unsupported_note), cancelled.metric) == (False, True, None)
    refund = catalog.clarification("clarify.refund_window").values[0]
    assert refund.metric == "net_fen" and refund.definition and refund.supported
    v1 = load_catalog(FIXTURES / "catalog-v1.json")
    assert v1.clarifications[0].trigger_terms and v1.clarifications[0].values == ()
    overlay = load_catalog_overlay(FIXTURES / "catalog-v2.json")
    assert overlay.catalog_version == "catalog-v2" and overlay.source_id == "commerce-v1"
    assert overlay.entries[0].requires_approval is False and overlay.entries[0].field_refs[0] == "orders.status"


# --------------------------------------------------------------------------
# The phrase table: the basis note of a verified answer.
# --------------------------------------------------------------------------

_READINGS = {
    "gross_named": ("2026年7月的支付订单总额是多少", (), ()),
    "net_named": ("7月退款后净额", (), ()),
    "sales_open": ("7月的销售额是多少", (), ()),
    "sales_answer_gross": ("7月的销售额是多少", ("支付订单总额",), ()),
    "sales_answer_net": ("7月的销售额是多少", ("退款后净额",), ()),
    "sales_confirmed_net": ("7月的销售额是多少", (), ("net_fen",)),
    "sales_confirmed_gross": ("7月的销售额是多少", (), ("metric.gross_fen",)),
    "count_named": ("7月已支付订单数", (), ()),
    "count_and_gross": ("7月已支付订单数和支付订单总额", (), ()),
    "gross_and_net": ("支付订单总额和退款后净额", (), ()),
    "nothing": ("你好", (), ()),
    "status_scope": ("已取消订单有多少", (), ()),
}
_METRICS = ["paid_count", "gross_fen", "refund_fen", "net_fen"]
WINDOW = {"start": "2026-07-01T00:00:00Z", "end": "2026-08-01T00:00:00Z", "timezone": "UTC"}


def _reading(label: str):
    question, answers, confirmed = _READINGS[label]
    return read_clarifications(load_default_catalog(), question, answers, confirmed)


_NOTES = {
    ('gross_named', 'paid_count') : '口径：已支付订单数（paid_count）；依据：按目录定义统计。',
    ('gross_named', 'gross_fen') : '口径：支付订单总额（gross_fen）；依据：问题中提到‘支付订单总额’。',
    ('gross_named', 'refund_fen') : '口径：退款金额（refund_fen）；依据：按目录定义统计。',
    ('gross_named', 'net_fen') : '口径：退款后净额（net_fen）；依据：按目录定义统计。前提：退款按同一 UTC 窗口内、已支付订单的退款计算。',
    ('net_named', 'paid_count') : '口径：已支付订单数（paid_count）；依据：按目录定义统计。',
    ('net_named', 'gross_fen') : '口径：支付订单总额（gross_fen）；依据：按目录定义统计。',
    ('net_named', 'refund_fen') : '口径：退款金额（refund_fen）；依据：按目录定义统计。',
    ('net_named', 'net_fen') : '口径：退款后净额（net_fen）；依据：问题中提到‘退款后净额’。前提：退款按同一 UTC 窗口内、已支付订单的退款计算。',
    ('sales_open', 'paid_count') : '口径：已支付订单数（paid_count）；依据：按目录定义统计。',
    ('sales_open', 'gross_fen') : '口径：支付订单总额（gross_fen）；问题中没有写明口径，按支付订单总额统计；如需退款后净额，请说明。',
    ('sales_open', 'refund_fen') : '口径：退款金额（refund_fen）；依据：按目录定义统计。',
    ('sales_open', 'net_fen') : '口径：退款后净额（net_fen）；问题中没有写明口径，按退款后净额统计；如需支付订单总额，请说明。前提：退款按同一 UTC 窗口内、已支付订单的退款计算。',
    ('sales_answer_gross', 'paid_count') : '口径：已支付订单数（paid_count）；依据：按目录定义统计。',
    ('sales_answer_gross', 'gross_fen') : '口径：支付订单总额（gross_fen）；依据：你在追问中选择了‘支付订单总额’。',
    ('sales_answer_gross', 'refund_fen') : '口径：退款金额（refund_fen）；依据：按目录定义统计。',
    ('sales_answer_gross', 'net_fen') : '口径：退款后净额（net_fen）；依据：按目录定义统计。前提：退款按同一 UTC 窗口内、已支付订单的退款计算。',
    ('sales_answer_net', 'paid_count') : '口径：已支付订单数（paid_count）；依据：按目录定义统计。',
    ('sales_answer_net', 'gross_fen') : '口径：支付订单总额（gross_fen）；依据：按目录定义统计。',
    ('sales_answer_net', 'refund_fen') : '口径：退款金额（refund_fen）；依据：按目录定义统计。',
    ('sales_answer_net', 'net_fen') : '口径：退款后净额（net_fen）；依据：你在追问中选择了‘退款后净额’。前提：退款按同一 UTC 窗口内、已支付订单的退款计算。',
    ('sales_confirmed_net', 'paid_count') : '口径：已支付订单数（paid_count）；依据：按目录定义统计。',
    ('sales_confirmed_net', 'gross_fen') : '口径：支付订单总额（gross_fen）；依据：按目录定义统计。',
    ('sales_confirmed_net', 'refund_fen') : '口径：退款金额（refund_fen）；依据：按目录定义统计。',
    ('sales_confirmed_net', 'net_fen') : '口径：退款后净额（net_fen）；依据：你在追问中选择了‘退款后净额’。前提：退款按同一 UTC 窗口内、已支付订单的退款计算。',
    ('sales_confirmed_gross', 'paid_count') : '口径：已支付订单数（paid_count）；依据：按目录定义统计。',
    ('sales_confirmed_gross', 'gross_fen') : '口径：支付订单总额（gross_fen）；依据：你在追问中选择了‘支付订单总额’。',
    ('sales_confirmed_gross', 'refund_fen') : '口径：退款金额（refund_fen）；依据：按目录定义统计。',
    ('sales_confirmed_gross', 'net_fen') : '口径：退款后净额（net_fen）；依据：按目录定义统计。前提：退款按同一 UTC 窗口内、已支付订单的退款计算。',
    ('count_named', 'paid_count') : '口径：已支付订单数（paid_count）；依据：问题中提到‘已支付订单数’。',
    ('count_named', 'gross_fen') : '口径：支付订单总额（gross_fen）；问题中没有写明口径，按支付订单总额统计；如需退款后净额，请说明。',
    ('count_named', 'refund_fen') : '口径：退款金额（refund_fen）；依据：按目录定义统计。',
    ('count_named', 'net_fen') : '口径：退款后净额（net_fen）；问题中没有写明口径，按退款后净额统计；如需支付订单总额，请说明。前提：退款按同一 UTC 窗口内、已支付订单的退款计算。',
    ('count_and_gross', 'paid_count') : '口径：已支付订单数（paid_count）；依据：问题中提到‘已支付订单数’。',
    ('count_and_gross', 'gross_fen') : '口径：支付订单总额（gross_fen）；依据：问题中提到‘支付订单总额’。',
    ('count_and_gross', 'refund_fen') : '口径：退款金额（refund_fen）；依据：按目录定义统计。',
    ('count_and_gross', 'net_fen') : '口径：退款后净额（net_fen）；依据：按目录定义统计。前提：退款按同一 UTC 窗口内、已支付订单的退款计算。',
    ('gross_and_net', 'paid_count') : '口径：已支付订单数（paid_count）；依据：按目录定义统计。',
    ('gross_and_net', 'gross_fen') : '口径：支付订单总额（gross_fen）；依据：问题中提到‘支付订单总额’。',
    ('gross_and_net', 'refund_fen') : '口径：退款金额（refund_fen）；依据：按目录定义统计。',
    ('gross_and_net', 'net_fen') : '口径：退款后净额（net_fen）；依据：问题中提到‘退款后净额’。前提：退款按同一 UTC 窗口内、已支付订单的退款计算。',
    ('nothing', 'paid_count') : '口径：已支付订单数（paid_count）；依据：按目录定义统计。',
    ('nothing', 'gross_fen') : '口径：支付订单总额（gross_fen）；问题中没有写明口径，按支付订单总额统计；如需退款后净额，请说明。',
    ('nothing', 'refund_fen') : '口径：退款金额（refund_fen）；依据：按目录定义统计。',
    ('nothing', 'net_fen') : '口径：退款后净额（net_fen）；问题中没有写明口径，按退款后净额统计；如需支付订单总额，请说明。前提：退款按同一 UTC 窗口内、已支付订单的退款计算。',
    ('status_scope', 'paid_count') : '口径：已支付订单数（paid_count）；依据：按目录定义统计。',
    ('status_scope', 'gross_fen') : '口径：支付订单总额（gross_fen）；问题中没有写明口径，按支付订单总额统计；如需退款后净额，请说明。',
    ('status_scope', 'refund_fen') : '口径：退款金额（refund_fen）；依据：按目录定义统计。',
    ('status_scope', 'net_fen') : '口径：退款后净额（net_fen）；问题中没有写明口径，按退款后净额统计；如需支付订单总额，请说明。前提：退款按同一 UTC 窗口内、已支付订单的退款计算。',
}


def test_the_basis_note_of_every_metric_in_every_reading_is_unchanged() -> None:
    actual = {(label, metric): metric_basis_note(_reading(label), metric) for label in _READINGS for metric in _METRICS}
    assert actual == _NOTES


_NOT_NEEDED = (
    "Do not ask_user about this rule: the question already names its value (or the rule has one value). "
    "Send a tool_call named query_readonly that declares the named metric."
)
_CONTRADICTION = (
    "Nothing was executed: the question names this rule's value and the declaration picked another one. "
    "Declare the metric the question names (declare_metrics) instead."
)


@pytest.mark.parametrize(
    "label, phrases, metrics",
    [
        ("gross_named", ["支付订单总额"], ["gross_fen"]),
        ("gross_and_net", ["支付订单总额", "退款后净额"], ["gross_fen", "net_fen"]),
        ("sales_answer_gross", ["支付订单总额"], ["gross_fen"]),
        ("sales_confirmed_net", [], ["net_fen"]),
    ],
)
@pytest.mark.parametrize("window", [None, WINDOW])
def test_repair_hints_are_fixed_text_plus_catalog_strings(label, phrases, metrics, window) -> None:
    catalog = load_default_catalog()
    rule = catalog.clarification("clarify.metric_basis")
    reading = _reading(label)
    echo = {"start": WINDOW["start"], "end": WINDOW["end"]} if window is not None else None
    content = {"clarification_id": "clarify.metric_basis", "named_phrases": phrases, "declare_metrics": metrics}
    expected_not_needed = {"action": _NOT_NEEDED, **content, "request_time_window": echo}
    expected_contradiction = {"action": _CONTRADICTION, **content, "request_time_window": echo}
    assert not_needed_hint(reading, rule, window) == expected_not_needed
    assert contradiction_hint(reading, rule, window) == expected_contradiction
    assert list(not_needed_hint(reading, rule, window)) == ["action", "clarification_id", "named_phrases", "declare_metrics", "request_time_window"]


@pytest.mark.parametrize(
    "label, clarification_id, text, expected",
    [
        ("sales_open", None, "你要看哪个口径？", ("catalog_question", "clarify.metric_basis", ("clarify.metric_basis",), "absent", "marker")),
        ("gross_named", None, "你要看哪个口径？", ("not_needed", "clarify.metric_basis", ("clarify.metric_basis",), "absent", "marker")),
        ("gross_named", None, "你要看哪个时间范围？", ("model_text", None, (), "absent", "none")),
        ("gross_named", "clarify.metric_basis", "你要看哪个时间范围？", ("not_needed", "clarify.metric_basis", ("clarify.metric_basis",), "declared", "clarification_id")),
        ("nothing", None, "你要看支付订单总额还是退款后净额？", ("catalog_question", "clarify.metric_basis", ("clarify.metric_basis",), "absent", "two_values")),
        ("nothing", "clarify.unknown", "随便问问", ("model_text", None, (), "unknown", "none")),
        ("status_scope", None, "统计方式是什么？", ("catalog_question", "clarify.metric_basis", ("clarify.metric_basis",), "absent", "marker")),
        ("sales_open", None, "你要看什么时间", ("model_text", None, (), "absent", "none")),
    ],
)
def test_review_ask_decisions_are_unchanged(label, clarification_id, text, expected) -> None:
    verdict = review_ask(_reading(label), clarification_id, text)
    assert (
        verdict.decision,
        verdict.rule.id if verdict.rule else None,
        verdict.targeted,
        verdict.id_status,
        verdict.signal,
    ) == expected


@pytest.mark.parametrize(
    "label, metric_ids, expected",
    [
        ("gross_named", ["gross_fen"], None),
        ("gross_named", ["net_fen"], ("contradicts", "clarify.metric_basis", None)),
        ("gross_named", ["gross_fen", "net_fen"], None),
        ("sales_open", ["gross_fen"], ("clarify", "clarify.metric_basis", None)),
        ("sales_open", ["net_fen"], ("clarify", "clarify.metric_basis", None)),
        ("sales_answer_gross", ["gross_fen"], None),
        ("sales_answer_gross", ["net_fen"], None),
        ("count_named", ["paid_count"], None),
        ("count_named", ["gross_fen"], None),
        ("status_scope", ["paid_count"], ("unsupported", "clarify.order_status_scope", "cancelled")),
        ("nothing", ["gross_fen"], None),
        ("gross_and_net", ["gross_fen"], None),
    ],
)
def test_declaration_verdicts_are_unchanged(label, metric_ids, expected) -> None:
    verdict = check_declaration(_reading(label), metric_ids)
    assert (None if verdict is None else (verdict.kind, verdict.rule.id, verdict.value.value if verdict.value else None)) == expected


def test_the_phrase_index_reads_hits_signals_and_chosen_values_unchanged() -> None:
    catalog = load_default_catalog()
    rule = catalog.clarification("clarify.metric_basis")
    index = _reading("gross_named").index
    assert (rule_named_by_waiting_question(catalog, "你要看支付订单总额还是退款后净额") or rule).id == "clarify.metric_basis"
    assert rule_named_by_waiting_question(catalog, "你要看支付订单总额还是退款后净额").id == "clarify.metric_basis"
    assert rule_named_by_waiting_question(catalog, "7月销售额") is None
    chosen = {answer: select_value(catalog, rule, answer) for answer in ("支付订单总额", "退款后净额", "支付订单总额和退款后净额", "随便")}
    assert {answer: value.value if value else None for answer, value in chosen.items()} == {
        "支付订单总额": "gross_fen",
        "退款后净额": "net_fen",
        "支付订单总额和退款后净额": None,
        "随便": None,
    }
    assert index.ask_signals("你要看支付订单总额，还是退款后净额？") == {"clarify.metric_basis": ("two_values",)}
    assert index.ask_signals("统计方式") == {"clarify.metric_basis": ("marker",)}
    assert index.ask_signals("销售额") == {}
    assert index.rules_named_by_ask("销售额 统计方式 退款后") == frozenset({"clarify.metric_basis"})
    text = "销售额和营业额，已支付订单总额"
    assert index.uncovered_ambiguous(text, index.explicit_hits(text)) == (
        ("clarify.metric_basis", "销售额"),
        ("clarify.metric_basis", "营业额"),
    )
    hits = index.explicit_hits("7月已支付订单数和退款后净额")
    assert [(hit.start, hit.end, hit.phrase, sorted(hit.metrics), sorted(hit.values)) for hit in hits] == [
        (2, 8, "已支付订单数", ["paid_count"], [("clarify.order_status_scope", "paid")]),
        (9, 14, "退款后净额", ["net_fen"], [("clarify.metric_basis", "net_fen"), ("clarify.refund_window", "same_window_paid_orders")]),
    ]


def test_an_ambiguous_phrase_is_reported_in_its_catalog_spelling(tmp_path) -> None:
    document = _load("catalog-v4.json")
    put("clarifications[0].ambiguous_phrases", ["销售额", "Total Sales"])(document)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    catalog = load_catalog(path)
    index = read_clarifications(catalog, "").index
    text = "what are the TOTAL SALES and the 销售额"
    assert index.uncovered_ambiguous(text, index.explicit_hits(text)) == (
        ("clarify.metric_basis", "销售额"),
        ("clarify.metric_basis", "Total Sales"),
    )
