"""A catalog-v1 clarification rule cannot carry phrase-table values; the shipped catalogs still load."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from queryshield.catalog.catalog import CatalogValidationError, load_catalog, load_catalog_overlay, validate_catalog_document

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "semantic"
VALUES = [{"value": "gross", "metric": "gross_fen", "phrases": ["营业额"]}]


def _v1() -> dict:
    return json.loads((FIXTURES / "catalog-v1.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("values", [VALUES, [], None], ids=["list", "empty-list", "null"])
def test_a_catalog_v1_rule_with_values_is_rejected(values) -> None:
    document = _v1()
    document["clarifications"][1]["values"] = copy.deepcopy(values)
    with pytest.raises(CatalogValidationError) as caught:
        validate_catalog_document(document)
    assert str(caught.value) == "clarifications[1].values needs the catalog-v3 phrase table"


@pytest.mark.parametrize("name", ["catalog-v1.json", "catalog-v3.json", "catalog-v4.json"])
def test_the_shipped_catalogs_still_load(name) -> None:
    load_catalog(FIXTURES / name)


def test_the_shipped_catalog_v2_overlay_still_loads() -> None:
    load_catalog_overlay(FIXTURES / "catalog-v2.json")
