from __future__ import annotations

import json
from pathlib import Path

import pytest

from queryshield.catalog import (
    CATALOG_VERSION,
    CATALOG_V4_VERSION,
    DEFAULT_CATALOG_PATH,
    DEFAULT_CATALOG_VERSION,
    CATALOG_V2_VERSION,
    CatalogValidationError,
    load_default_catalog,
    load_catalog,
    load_catalog_overlay,
)


CATALOG_PATH = Path(__file__).parents[1] / "fixtures" / "semantic" / "catalog-v1.json"
CATALOG_V2_PATH = Path(__file__).parents[1] / "fixtures" / "semantic" / "catalog-v2.json"


def test_catalog_v1_is_traceable_and_search_shape_is_narrow() -> None:
    catalog = load_default_catalog()

    # The product default is catalog-v4 (B3c-1); its version constant is the one
    # every default executor/run config/approval reads.
    assert catalog.catalog_version == DEFAULT_CATALOG_VERSION == CATALOG_V4_VERSION
    assert DEFAULT_CATALOG_PATH.name == "catalog-v4.json"
    assert load_catalog(CATALOG_PATH).catalog_version == CATALOG_VERSION
    assert catalog.fixture_version == "commerce-v1"
    assert len(catalog.entries) >= 10
    assert len(catalog.clarifications) >= 3
    assert catalog.access_scope["database_access"] == "guarded_query_executor_only"
    assert catalog.access_scope["model_may_supply_identity"] is False

    items = catalog.search_items()
    assert all(set(item) == {"id", "text", "source_id", "version"} for item in items)
    assert all(item["source_id"] == "commerce-v1" for item in items)
    assert all(item["version"] == "commerce-v1" for item in items)
    assert all((CATALOG_PATH.parents[2] / source_file).exists() for source_file in catalog.source_files)


def test_catalog_field_refs_are_present_in_the_formal_migration() -> None:
    catalog = load_default_catalog()
    migration = (CATALOG_PATH.parents[2] / "migrations" / "001_commerce_v1.sql").read_text(
        encoding="utf-8"
    )

    for table in ("customers", "orders", "refunds"):
        start = migration.index(f"CREATE TABLE IF NOT EXISTS {table} (")
        end = migration.index(");", start)
        table_definition = migration[start:end]
        for item in catalog.entries:
            if item.kind == "field" and item.table == table:
                assert f"    {item.column} " in table_definition


def test_metric_definitions_point_only_to_real_commerce_fields() -> None:
    catalog = load_default_catalog()

    assert catalog.metric("paid_count").payload["unit"] == "count"
    assert catalog.metric("gross_fen").payload["field_refs"] == [
        "orders.status",
        "orders.amount_fen",
        "orders.created_at",
    ]
    assert set(catalog.metric("net_fen").payload["derived_from"]) == {
        "metric.gross_fen",
        "metric.refund_fen",
    }


def test_catalog_contains_fixed_ambiguity_rules() -> None:
    catalog = load_default_catalog()
    rules = {rule.id: rule for rule in catalog.clarifications}

    assert set(rules) == {
        "clarify.metric_basis",
        "clarify.order_status_scope",
        "clarify.refund_window",
    }
    assert rules["clarify.metric_basis"].allowed_values == ("gross_fen", "net_fen")
    assert "WAITING_USER" in rules["clarify.metric_basis"].payload["resolution"]
    assert rules["clarify.refund_window"].allowed_values == ("same_window_paid_orders",)


def test_catalog_v2_registers_refund_metric_without_changing_v1() -> None:
    overlay = load_catalog_overlay(CATALOG_V2_PATH)

    assert overlay.catalog_version == CATALOG_V2_VERSION
    assert overlay.base_catalog_version == CATALOG_VERSION
    assert len(overlay.entries) == 1
    assert overlay.entries[0].id == "metric.refund_fen"
    assert overlay.entries[0].payload["unit"] == "CNY_fen"
    assert overlay.entries[0].payload["fixed_tenant_values"] == {"A": 3000, "B": 10000}
    assert overlay.entries[0].payload["source_locator"] == "fixtures/commerce-v1.md:17-21"


def test_catalog_rejects_unknown_schema_field(tmp_path: Path) -> None:
    document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    document["entries"][5]["column"] = "id"
    path = tmp_path / "invalid-catalog.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(CatalogValidationError, match="not in the real table"):
        load_catalog(path)


def test_catalog_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-catalog.json"
    path.write_text('{"catalog_version":"catalog-v1","catalog_version":"catalog-v1"}', encoding="utf-8")

    with pytest.raises(CatalogValidationError, match="duplicate JSON key"):
        load_catalog(path)
