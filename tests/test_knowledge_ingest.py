from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from queryshield.knowledge.ingest import (
    MAX_FILE_BYTES,
    KnowledgeImportError,
    import_knowledge,
)


PROJECT_ROOT = Path(__file__).parents[1]
KNOWLEDGE_ROOT = PROJECT_ROOT / "fixtures" / "knowledge"
REGISTRY_PATH = KNOWLEDGE_ROOT / "source_registry.json"


def _snapshot() -> object:
    return import_knowledge(
        KNOWLEDGE_ROOT,
        REGISTRY_PATH,
        catalog_version="catalog-v2",
    )


def _by_source(items: tuple[object, ...]) -> dict[str, tuple[object, ...]]:
    grouped: dict[str, list[object]] = {}
    for item in items:
        grouped.setdefault(item.source_id, []).append(item)
    return {source_id: tuple(values) for source_id, values in grouped.items()}


def test_knowledge_import_is_deterministic_and_acl_is_registry_authored() -> None:
    first = _snapshot()
    second = _snapshot()

    assert first.as_dict() == second.as_dict()
    assert first.snapshot_id.startswith("knowledge-v1-")
    assert first.manifest_sha256
    assert first.index_hash
    assert len(first.source_records) == 14
    assert len(first.chunk_records) >= 13
    assert all(len(item.text) <= 800 for item in first.chunk_records)

    sources = {item.source_id: item for item in first.source_records}
    assert sources["semantic-refund-policy-v1"].status == "deleted"
    assert not [
        item for item in first.chunk_records if item.source_id == "semantic-refund-policy-v1"
    ]
    assert sources["tenant-a-orders-overview"].path == "tenant-a/orders-overview.md"
    assert sources["tenant-b-orders-overview"].path == "tenant-b/orders-overview.md"
    assert sources["tenant-a-orders-overview"].tenant_scope == "A"
    assert sources["tenant-b-orders-overview"].tenant_scope == "B"
    assert sources["tenant-a-orders-overview"].allowed_roles == ("requester", "approver")


def test_changing_one_paragraph_only_changes_one_source_and_its_chunks(tmp_path: Path) -> None:
    copied_root = tmp_path / "knowledge"
    shutil.copytree(KNOWLEDGE_ROOT, copied_root)
    changed_path = copied_root / "tenant-a" / "orders-overview.md"
    changed_path.write_text(
        changed_path.read_text(encoding="utf-8").replace(
            "Paid means the order status is `PAID`; a model or document cannot widen that scope.",
            "Paid means the order status is `PAID`; this changed paragraph is versioned.",
        ),
        encoding="utf-8",
    )
    registry_path = copied_root / "source_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for source in registry["sources"]:
        if source["source_id"] == "tenant-a-orders-overview":
            source["version"] = "2026-09-22"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    original = _snapshot()
    changed = import_knowledge(copied_root, registry_path, catalog_version="catalog-v2")
    original_sources = {item.source_id: item for item in original.source_records}
    changed_sources = {item.source_id: item for item in changed.source_records}
    changed_source_ids = {
        source_id
        for source_id in original_sources
        if original_sources[source_id] != changed_sources[source_id]
    }
    assert changed_source_ids == {"tenant-a-orders-overview"}

    original_chunks = _by_source(original.chunk_records)
    changed_chunks = _by_source(changed.chunk_records)
    for source_id in original_sources:
        if source_id != "tenant-a-orders-overview":
            assert original_chunks.get(source_id, ()) == changed_chunks.get(source_id, ())
    assert original_chunks["tenant-a-orders-overview"] != changed_chunks["tenant-a-orders-overview"]


def test_import_rejects_path_escape(tmp_path: Path) -> None:
    copied_root = tmp_path / "knowledge"
    shutil.copytree(KNOWLEDGE_ROOT, copied_root)
    registry_path = copied_root / "source_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["sources"][0]["path"] = "../outside.md"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(KnowledgeImportError, match="escapes"):
        import_knowledge(copied_root, registry_path, catalog_version="catalog-v2")


def test_import_rejects_oversized_file(tmp_path: Path) -> None:
    copied_root = tmp_path / "knowledge"
    shutil.copytree(KNOWLEDGE_ROOT, copied_root)
    (copied_root / "shared" / "metric-gross.md").write_text(
        "x" * (MAX_FILE_BYTES + 1), encoding="utf-8"
    )

    with pytest.raises(KnowledgeImportError, match="file limit"):
        import_knowledge(copied_root, copied_root / "source_registry.json", catalog_version="catalog-v2")
