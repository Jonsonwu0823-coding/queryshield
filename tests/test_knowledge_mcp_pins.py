"""The knowledge-base and MCP tidy-up changes no behaviour.

These tests describe ``knowledge/`` and ``mcp_metadata/`` as a reader of their outputs can see
them: the hashes and ids of snapshots and indexes, the bytes of the files they write, the
retrieval evidence, who may see which source, the error messages of the import, and the
record of an MCP session after each kind of failure.  They were written and run against the
code before the tidy-up, then run again unchanged after it.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from queryshield.agent.proposals import ExecutionContext
from queryshield.catalog import load_default_catalog
from queryshield.knowledge.index import (
    EmbeddingIndex,
    IndexValidationError,
    build_embedding_index,
    embedded_snapshot_for_index,
    load_index,
    publish_index,
)
from queryshield.knowledge.ingest import (
    KnowledgeImportError,
    build_snapshot,
    import_knowledge,
    load_snapshot,
    write_snapshot,
)
from queryshield.db.state_store import StateStore
from queryshield.knowledge.retrieval import (
    HybridRetriever,
    RetrievalConfigurationError,
    _visible_candidates,
)
from queryshield.knowledge.runtime import (
    DEFAULT_KNOWLEDGE_ROOT,
    DEFAULT_SENSITIVE_PERMISSION_SOURCE_ID,
    DEMO_CATALOG_VERSION,
    DEMO_KNOWLEDGE_ROOT,
    DEMO_KNOWLEDGE_VERSION,
    DEMO_SENSITIVE_PERMISSION_SOURCE_ID,
    IndexFileMismatch,
    build_retrieval_runtime,
    product_knowledge,
    reset_retrieval_cache,
    retriever_from_index_file,
    shared_demo_retrieval_runtime,
    shared_retrieval_runtime,
)
from queryshield.knowledge.snapshots import KnowledgeAccessError, KnowledgeIdentity, KnowledgeSnapshotRepository
from queryshield.mcp_metadata import session as session_module
from queryshield.mcp_metadata import verify
from queryshield.mcp_metadata.launch import LaunchSpec
from queryshield.mcp_metadata.session import McpMetadataSession, McpSessionError
from queryshield.mcp_metadata.tools import McpMetadataTools, McpToolError
from queryshield.providers.embedding import FixedEmbedding
from queryshield.providers.rerank import FakeReranker, RerankCallRecord
from queryshield.tools.semantic import ControlledTools, ToolError

from mcp_helpers import config, context, started_session
from test_hybrid_retrieval import _context, _keyword_retriever, _retriever


# --- snapshot and index identity ---------------------------------------------------------------


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _registry_entry(source_id: str, path: str, **overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "source_id": source_id,
        "path": path,
        "version": "v1",
        "tenant_scope": "global",
        "allowed_roles": ["requester", "approver"],
        "status": "active",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    entry.update(overrides)
    return entry


def _write_tree(root: Path, files: dict[str, str], sources: list[dict[str, object]]) -> None:
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8", newline="")
    (root / "source_registry.json").write_text(json.dumps({"sources": sources}), encoding="utf-8")


@pytest.fixture
def knowledge_tree(tmp_path: Path) -> Path:
    _write_tree(
        tmp_path,
        {
            "a.md": "Gross revenue is paid orders.\n\nSecond paragraph about gross.\r\n\r\n" + "x" * 900,
            "b.txt": "Net revenue is gross minus refunds.",
            "gone.md": "deleted source text",
        },
        [
            _registry_entry("src-a", "a.md"),
            _registry_entry("src-b", "b.txt", version="v2", tenant_scope="A", allowed_roles=["approver"], updated_at="2026-01-02T00:00:00Z"),
            _registry_entry("src-gone", "gone.md", tenant_scope="B", allowed_roles=["requester"], status="deleted", updated_at="2026-01-03T00:00:00Z"),
        ],
    )
    return tmp_path


def _fixed_embedder(snapshot) -> FixedEmbedding:
    chunks = sorted(snapshot.chunk_records, key=lambda chunk: chunk.chunk_id)
    vectors = {chunk.text: (1.0, float(index % 3), 0.5) for index, chunk in enumerate(chunks)}
    return FixedEmbedding(vectors, model_revision="fixed-s2", dimensions=3)


def test_snapshot_ids_and_hashes_are_pinned(knowledge_tree: Path) -> None:
    snapshot = build_snapshot(knowledge_tree, "source_registry.json", catalog_version="catalog-x", knowledge_version="kv-test")

    assert snapshot.snapshot_id == "kv-test-8754053eb914bb1c"
    assert snapshot.manifest_sha256 == "8754053eb914bb1cfdd724b2fff7fbfc297d7e0bd8b2b83807df1b4a20e1b46d"
    assert snapshot.index_hash == "f7ff37c59fa5586bed7710874cc12eb31fda7801bbe7801e8fb5092056cdc58a"
    assert len(snapshot.chunk_records) == 5  # the deleted source has a record and no chunk
    assert [source.source_id for source in snapshot.source_records] == ["src-a", "src-b", "src-gone"]
    # The formula, written out once more here: the id is the knowledge version and the first 16 hex digits.
    manifest = {
        "knowledge_version": "kv-test",
        "catalog_version": "catalog-x",
        "chunker_version": "paragraph-v1",
        "embedding_model_revision": None,
        "embedding_dimensions": None,
        "index_hash": snapshot.index_hash,
        "sources": [item.as_dict() for item in snapshot.source_records],
        "chunks": [item.as_dict() for item in snapshot.chunk_records],
    }
    assert _sha(_canonical(manifest)) == snapshot.manifest_sha256
    chunks_only = {"chunker_version": "paragraph-v1", "chunks": [item.as_dict() for item in snapshot.chunk_records]}
    assert _sha(_canonical(chunks_only)) == snapshot.index_hash


def test_import_knowledge_and_build_snapshot_give_the_same_snapshot(knowledge_tree: Path) -> None:
    left = import_knowledge(knowledge_tree, knowledge_tree / "source_registry.json", catalog_version="catalog-x", knowledge_version="kv-test")
    right = build_snapshot(knowledge_tree, "source_registry.json", catalog_version="catalog-x", knowledge_version="kv-test")
    assert left == right
    default = import_knowledge(knowledge_tree, knowledge_tree / "source_registry.json", catalog_version="catalog-x")
    assert default.snapshot_id.startswith("knowledge-v1-")


def test_the_demo_knowledge_snapshot_is_pinned() -> None:
    snapshot = import_knowledge(
        DEMO_KNOWLEDGE_ROOT,
        DEMO_KNOWLEDGE_ROOT / "source_registry.json",
        catalog_version=DEMO_CATALOG_VERSION,
        knowledge_version=DEMO_KNOWLEDGE_VERSION,
    )
    assert snapshot.snapshot_id == "knowledge-demo-v1-142c2433a63cc6a8"
    assert snapshot.manifest_sha256.startswith("142c2433a63cc6a8")
    assert snapshot.index_hash == "145b88a6c39f5494dc4557680170c04e530a1311f15a48257475e11622e3335b"


def test_the_embedded_snapshot_and_index_hash_are_pinned(knowledge_tree: Path) -> None:
    snapshot = build_snapshot(knowledge_tree, "source_registry.json", catalog_version="catalog-x", knowledge_version="kv-test")
    embedder = _fixed_embedder(snapshot)
    build = build_embedding_index(snapshot, embedder, ingest_job_id="ingest-s2")

    assert build.snapshot.snapshot_id == "kv-test-8f5ff9c9ef45f08d"
    assert build.snapshot.manifest_sha256 == "8f5ff9c9ef45f08d2f59740c8a4d60011d7d78b9a106ac20d08f82a1a05aea66"
    assert build.index.index_hash == "8787b0d9be45c528b7089cff4916483d65eb634c057fb04ed4fa87c2feb92353"
    assert build.snapshot.index_hash == build.index.index_hash
    assert build.snapshot.embedding_model_revision == "fixed-s2" and build.snapshot.embedding_dimensions == 3
    assert build.index.snapshot_id == build.snapshot.snapshot_id and build.base_snapshot_id == snapshot.snapshot_id
    assert build.ingest_job_id == "ingest-s2" and len(build.operation_usages) == 1
    # The formulas, written out once more here.
    index_input = {
        "index_version": "vector-index-v1",
        "model": embedder.model,
        "model_revision": "fixed-s2",
        "dimensions": 3,
        "chunks": [chunk.as_dict() for chunk in build.index.chunks],
    }
    assert _sha(_canonical(index_input)) == build.index.index_hash
    manifest = {
        "knowledge_version": "kv-test",
        "catalog_version": "catalog-x",
        "chunker_version": "paragraph-v1",
        "embedding_model_revision": "fixed-s2",
        "embedding_dimensions": 3,
        "index_hash": build.index.index_hash,
        "sources": [item.as_dict() for item in snapshot.source_records],
        "chunks": [item.as_dict() for item in snapshot.chunk_records],
    }
    assert f"kv-test-{_sha(_canonical(manifest))[:16]}" == build.snapshot.snapshot_id
    assert embedded_snapshot_for_index(snapshot, build.index).snapshot_id == build.snapshot.snapshot_id


def test_index_build_batches_and_orders_chunks_by_id(knowledge_tree: Path) -> None:
    snapshot = build_snapshot(knowledge_tree, "source_registry.json", catalog_version="catalog-x", knowledge_version="kv-test")
    one = build_embedding_index(snapshot, _fixed_embedder(snapshot), ingest_job_id="ingest-s2", batch_size=1)
    many = build_embedding_index(snapshot, _fixed_embedder(snapshot), ingest_job_id="ingest-s2")
    assert [chunk.chunk_id for chunk in one.index.chunks] == sorted(chunk.chunk_id for chunk in snapshot.chunk_records)
    assert len(one.operation_usages) == 5 and len(many.operation_usages) == 1
    assert one.index.index_hash == many.index.index_hash


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size must be between one and eight"),
        ({"batch_size": 9}, "batch_size must be between one and eight"),
        ({"batch_size": True}, "batch_size must be between one and eight"),
    ],
)
def test_index_build_refuses_bad_arguments(knowledge_tree: Path, kwargs: dict, message: str) -> None:
    snapshot = build_snapshot(knowledge_tree, "source_registry.json", catalog_version="catalog-x", knowledge_version="kv-test")
    with pytest.raises(IndexValidationError, match=message):
        build_embedding_index(snapshot, _fixed_embedder(snapshot), **kwargs)


def test_index_build_refuses_an_empty_snapshot(knowledge_tree: Path) -> None:
    snapshot = build_snapshot(knowledge_tree, "source_registry.json", catalog_version="catalog-x", knowledge_version="kv-test")
    with pytest.raises(IndexValidationError, match="cannot build an index from an empty snapshot"):
        build_embedding_index(replace(snapshot, chunk_records=()), _fixed_embedder(snapshot))


def test_a_loaded_index_is_checked_against_its_hash(knowledge_tree: Path, tmp_path: Path) -> None:
    snapshot = build_snapshot(knowledge_tree, "source_registry.json", catalog_version="catalog-x", knowledge_version="kv-test")
    build = build_embedding_index(snapshot, _fixed_embedder(snapshot), ingest_job_id="ingest-s2")
    path = publish_index(build.index, tmp_path / "out" / "index.json")
    assert load_index(path) == build.index

    document = json.loads(path.read_text(encoding="utf-8"))
    document["chunks"][0]["vector"][0] = 0.25
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(IndexValidationError, match="embedding index hash does not match its vectors"):
        load_index(path)
    document["chunks"][0]["vector"] = [0.25, 0.0]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(IndexValidationError, match="embedding index fields are invalid"):
        load_index(path)
    document["index_version"] = "other"
    document["chunks"][0]["vector"] = [0.25, 0.0, 0.0]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(IndexValidationError, match="embedding index fields are invalid"):
        load_index(path)


def test_the_files_written_are_pinned_and_leave_no_temporary_file(knowledge_tree: Path, tmp_path: Path) -> None:
    snapshot = build_snapshot(knowledge_tree, "source_registry.json", catalog_version="catalog-x", knowledge_version="kv-test")
    build = build_embedding_index(snapshot, _fixed_embedder(snapshot), ingest_job_id="ingest-s2")

    snapshot_dir = tmp_path / "snapshots" / "deeper"
    written = write_snapshot(snapshot, snapshot_dir)
    assert written == snapshot_dir / "kv-test-8754053eb914bb1c.json"
    assert _sha(written.read_bytes()) == "a82b67e0bf366c5448dce8f2eea9d0aae57b4ae67627f0542cc1493ed3becf66"
    assert written.read_bytes() == (json.dumps(snapshot.as_dict(), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    assert load_snapshot(written) == snapshot
    assert sorted(item.name for item in snapshot_dir.iterdir()) == [written.name]

    index_path = publish_index(build.index, tmp_path / "indexes" / "deeper" / "index.json")
    assert _sha(index_path.read_bytes()) == "e2d3fcecf552fa32add1de2b57bedc8e5b886e260925471b066e0882bd922e62"
    assert index_path.read_bytes() == (json.dumps(build.index.as_dict(), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    assert sorted(item.name for item in index_path.parent.iterdir()) == ["index.json"]

    # Writing again replaces the file in place.
    assert write_snapshot(snapshot, snapshot_dir) == written
    assert publish_index(build.index, index_path) == index_path
    assert len(list(snapshot_dir.iterdir())) == 1 and len(list(index_path.parent.iterdir())) == 1


def test_a_failed_write_keeps_the_old_file_and_removes_the_temporary_one(knowledge_tree: Path, tmp_path: Path, monkeypatch) -> None:
    snapshot = build_snapshot(knowledge_tree, "source_registry.json", catalog_version="catalog-x", knowledge_version="kv-test")
    build = build_embedding_index(snapshot, _fixed_embedder(snapshot), ingest_job_id="ingest-s2")
    snapshot_path = write_snapshot(snapshot, tmp_path / "snapshots")
    index_path = publish_index(build.index, tmp_path / "indexes" / "index.json")
    before = (snapshot_path.read_bytes(), index_path.read_bytes())

    def failing_replace(*args, **kwargs):
        raise PermissionError("no replace")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(PermissionError):
        write_snapshot(snapshot, tmp_path / "snapshots")
    with pytest.raises(PermissionError):
        publish_index(build.index, index_path)
    monkeypatch.undo()

    assert (snapshot_path.read_bytes(), index_path.read_bytes()) == before
    assert [item.name for item in snapshot_path.parent.iterdir()] == [snapshot_path.name]
    assert [item.name for item in index_path.parent.iterdir()] == ["index.json"]


# --- vectors and the cosine ------------------------------------------------------------------


def _index_with(vectors: list[tuple], dimensions: int = 3) -> EmbeddingIndex:
    from queryshield.knowledge.index import IndexChunk

    chunks = tuple(
        IndexChunk(chunk_id=f"c{number}", source_id="s", source_version="v1", text="t", vector=vector)
        for number, vector in enumerate(vectors)
    )
    return EmbeddingIndex(
        index_version="vector-index-v1",
        snapshot_id="snap",
        knowledge_version="kv",
        catalog_version="cv",
        model="m",
        model_revision="r",
        dimensions=dimensions,
        index_hash="h",
        chunks=chunks,
    )


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([(1.0, 0.0)], "index contains a mixed vector dimension"),
        ([(1.0, 0.0, 0.0, 0.0)], "index contains a mixed vector dimension"),
        ([(1.0, 0.0, float("nan"))], "index contains a non-finite vector value"),
        ([(1.0, float("inf"), 0.0)], "index contains a non-finite vector value"),
        ([(1.0, 0.0, 0.0), (0.0, float("-inf"), 0.0)], "index contains a non-finite vector value"),
    ],
)
def test_an_index_refuses_a_vector_of_the_wrong_size_or_with_a_non_finite_value(vectors, message) -> None:
    with pytest.raises(IndexValidationError, match=message):
        _index_with(vectors)
    assert _index_with([(1, 0, 0.5)]).chunks[0].vector == (1, 0, 0.5)  # ints are numbers


@pytest.mark.parametrize("value", [True, "1.0", None])
def test_an_index_refuses_a_vector_value_that_is_not_a_number(value) -> None:
    with pytest.raises((IndexValidationError, TypeError)):
        _index_with([(1.0, 0.0, value)])


def test_cosine_similarity_is_pinned() -> None:
    from queryshield.knowledge.index import cosine_similarity

    assert cosine_similarity((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == 0.0
    assert cosine_similarity((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)) == 0.0
    assert cosine_similarity((1.0, 2.0, 3.0), (0.0, 0.0, 0.0)) == 0.0
    assert cosine_similarity((1.0, 2.0, 3.0), (3.0, 2.0, 1.0)) == 10.0 / 14.0
    left, right = (0.1, 0.2, 0.3), (0.3, 0.2, 0.1)
    import math

    expected = (0.1 * 0.3 + 0.2 * 0.2 + 0.3 * 0.1) / (math.sqrt(0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3) * math.sqrt(0.3 * 0.3 + 0.2 * 0.2 + 0.1 * 0.1))
    assert cosine_similarity(left, right) == expected
    with pytest.raises(IndexValidationError, match="cosine vectors must have the same dimension"):
        cosine_similarity((1.0, 0.0), (1.0, 0.0, 0.0))


# --- the registry and the files an import reads ------------------------------------------------


def _import(root: Path, registry: str | Path = "source_registry.json") -> object:
    return build_snapshot(root, registry, catalog_version="catalog-x", knowledge_version="kv-test")


def _registry_case(tmp_path: Path, sources: list[dict[str, object]], files: dict[str, str] | None = None) -> Path:
    _write_tree(tmp_path, files if files is not None else {"a.md": "text", "b.md": "other"}, sources)
    return tmp_path


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda e: e[0].update(source_id="Bad Id"), r"registry\.sources\[0\]\.source_id has an invalid format"),
        (lambda e: e[0].update(source_id=""), r"registry\.sources\[0\]\.source_id must be a non-empty string"),
        (lambda e: e[0].pop("path"), r"registry\.sources\[0\]\.path must be a non-empty string"),
        (lambda e: e[0].update(path="../a.md"), r"registry\.sources\[0\]\.path escapes the configured knowledge root"),
        (lambda e: e[0].update(path="missing.md"), r"registry\.sources\[0\]\.path escapes the configured knowledge root"),
        (lambda e: e[0].update(path=os.path.abspath("/etc/passwd")), r"registry\.sources\[0\]\.path must not be absolute"),
        (lambda e: e[0].update(path="sub"), r"registry\.sources\[0\]\.path must point to a file"),
        (lambda e: e[1].update(path="a.md"), r"duplicate registry path: a\.md"),
        (lambda e: e[1].update(source_id="src-a"), r"duplicate source version: src-a/v1"),
        (lambda e: e[0].update(tenant_scope="C"), r"registry\.sources\[0\]\.tenant_scope is unsupported"),
        (lambda e: e[0].update(allowed_roles="requester"), r"registry\.sources\[0\]\.allowed_roles must be a non-empty list"),
        (lambda e: e[0].update(allowed_roles=[]), r"registry\.sources\[0\]\.allowed_roles must be a non-empty list"),
        (lambda e: e[0].update(allowed_roles=["requester", "requester"]), r"registry\.sources\[0\]\.allowed_roles must not contain duplicates"),
        (lambda e: e[0].update(allowed_roles=["auditor"]), r"registry\.sources\[0\]\.allowed_roles contains an unknown role"),
        (lambda e: e[0].update(status="archived"), r"registry\.sources\[0\]\.status must be active or deleted"),
        (lambda e: e[0].update(updated_at="2026-01-01T00:00:00+08:00"), r"registry\.sources\[0\]\.updated_at must be UTC"),
        (lambda e: e[0].pop("version"), r"registry\.sources\[0\]\.version must be a non-empty string"),
        (lambda e: e.__setitem__(0, "not an object"), r"registry\.sources\[0\] must be an object"),
    ],
)
def test_registry_entries_are_refused_with_their_messages(tmp_path: Path, mutate, message: str) -> None:
    (tmp_path / "sub").mkdir()
    sources = [_registry_entry("src-a", "a.md"), _registry_entry("src-b", "b.md")]
    mutate(sources)
    _registry_case(tmp_path, sources)
    with pytest.raises(KnowledgeImportError, match=message):
        _import(tmp_path)


def test_registry_documents_are_refused_with_their_messages(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("text", encoding="utf-8")
    registry = tmp_path / "source_registry.json"

    def refused(text: str, message: str) -> None:
        registry.write_text(text, encoding="utf-8")
        with pytest.raises(KnowledgeImportError, match=message):
            _import(tmp_path)

    refused("not json", "cannot read source registry")
    refused('{"sources": [], "sources": []}', "duplicate JSON key: sources")
    refused("[]", "registry must be an object")
    refused("{}", r"registry\.sources must be a non-empty list")
    refused('{"sources": []}', r"registry\.sources must be a non-empty list")
    too_many = [_registry_entry(f"src-{index:03d}", "a.md") for index in range(101)]
    refused(json.dumps({"sources": too_many}), "registry contains more than 100 files")
    registry.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(KnowledgeImportError, match="cannot read source registry"):
        _import(tmp_path)


def test_the_registry_must_be_inside_the_root_and_exist(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.md").write_text("text", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"sources": [_registry_entry("src-a", "a.md")]}), encoding="utf-8")
    with pytest.raises(KnowledgeImportError, match="source registry must be inside the configured root"):
        _import(root, outside)
    with pytest.raises(KnowledgeImportError, match="source registry must be inside the configured root"):
        _import(root, "missing.json")
    with pytest.raises(KnowledgeImportError, match="source_root must be a directory"):
        _import(root / "a.md")


def test_files_the_registry_does_not_list_are_refused(tmp_path: Path) -> None:
    _registry_case(tmp_path, [_registry_entry("src-a", "a.md")], {"a.md": "text", "extra.txt": "unlisted", "note.pdf": "ignored"})
    with pytest.raises(KnowledgeImportError, match=r"unregistered knowledge files are not allowed: \['extra\.txt'\]"):
        _import(tmp_path)


def test_a_file_that_is_too_big_empty_or_not_utf8_is_refused(tmp_path: Path) -> None:
    sources = [_registry_entry("src-a", "a.md")]
    _registry_case(tmp_path, sources, {"a.md": "x" * (256 * 1024 + 1)})
    with pytest.raises(KnowledgeImportError, match=r"a\.md exceeds the 262144-byte file limit"):
        _import(tmp_path)
    (tmp_path / "a.md").write_text(" \r\n \n", encoding="utf-8")
    with pytest.raises(KnowledgeImportError, match=r"a\.md is empty"):
        _import(tmp_path)
    (tmp_path / "a.md").write_bytes(b"\xff\xfe\xfa")
    with pytest.raises(KnowledgeImportError, match=r"a\.md is not valid UTF-8"):
        _import(tmp_path)


def test_too_many_chunks_are_refused(tmp_path: Path) -> None:
    text = "\n\n".join(f"paragraph {index}" for index in range(201))
    _registry_case(tmp_path, [_registry_entry("src-a", "a.md")], {"a.md": text})
    with pytest.raises(KnowledgeImportError, match="snapshot contains more than 200 chunks"):
        _import(tmp_path)


def test_a_registry_inside_a_directory_is_found_relative_to_the_root_first(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("text", encoding="utf-8")
    (tmp_path / "reg").mkdir()
    (tmp_path / "reg" / "list.json").write_text(json.dumps({"sources": [_registry_entry("src-a", "a.md")]}), encoding="utf-8")
    assert _import(tmp_path, "reg/list.json").source_records[0].path == "a.md"
    assert _import(tmp_path, tmp_path / "reg" / "list.json").source_records[0].path == "a.md"


def test_load_snapshot_refuses_unreadable_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeImportError, match="cannot load knowledge snapshot"):
        load_snapshot(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(KnowledgeImportError, match="knowledge snapshot root must be an object"):
        load_snapshot(bad)
    bad.write_text(json.dumps({"sources": {}, "chunks": []}), encoding="utf-8")
    with pytest.raises(KnowledgeImportError, match="knowledge snapshot fields are invalid"):
        load_snapshot(bad)
    bad.write_text(json.dumps({"sources": [], "chunks": []}), encoding="utf-8")
    with pytest.raises(KnowledgeImportError, match="knowledge snapshot fields are invalid"):
        load_snapshot(bad)


def test_the_command_line_import_prints_one_json_line_and_exits_with_its_code(knowledge_tree: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    from queryshield.knowledge import ingest

    out = tmp_path / "snapshots"
    argv = ["ingest", "--source-root", str(knowledge_tree), "--registry", str(knowledge_tree / "source_registry.json"), "--snapshot-dir", str(out), "--catalog-version", "catalog-x"]
    monkeypatch.setattr("sys.argv", argv)
    assert ingest.main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert list(printed) == ["status", "snapshot_id", "catalog_version", "source_count", "active_chunk_count", "manifest_sha256", "output_path"]
    assert printed["status"] == "pass" and printed["source_count"] == 3 and printed["active_chunk_count"] == 5
    assert printed["snapshot_id"].startswith("knowledge-v1-") and Path(printed["output_path"]).parent == out
    assert [item.name for item in out.iterdir()] == [f"{printed['snapshot_id']}.json"]

    (knowledge_tree / "extra.md").write_text("unlisted", encoding="utf-8")
    assert ingest.main() == 2
    blocked = json.loads(capsys.readouterr().out)
    assert list(blocked) == ["status", "error"] and blocked["status"] == "blocked"
    assert "unregistered knowledge files are not allowed" in blocked["error"]


# --- retrieval ---------------------------------------------------------------------------------


def _normalized(result) -> str:
    evidence = result.evidence.as_dict()
    retrieval_id = evidence.pop("retrieval_id")
    evidence.pop("elapsed_ms")
    text = json.dumps({"items": list(result.items), "evidence": evidence}, ensure_ascii=False, sort_keys=True)
    if evidence["rerank_call_id"] is not None:
        text = text.replace(evidence["rerank_call_id"], "RERANK-ID")
    return text.replace(retrieval_id, "RID")


def _digest(result) -> str:
    return _sha(_normalized(result).encode("utf-8"))[:16]


class _FailedReranker:
    model = "fake-failed-reranker"

    def rerank(self, query, candidates, *, top_n):
        return RerankCallRecord(
            call_id="rerank-timeout",
            status="timeout",
            model=self.model,
            input_candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
            returned_candidate_ids=(),
            scores=(),
            usage_status="unknown",
            total_tokens=None,
            error_code="timeout",
        )


def test_hybrid_evidence_is_pinned_per_identity_and_query() -> None:
    retriever = _retriever()

    first = retriever.search("营业额", context=_context(), top_k=3)
    assert _digest(first) == "c41c38bd1d679187"
    assert [item["id"] for item in first.items] == [
        "semantic-metric-gross@2026-09-21#0001",
        "metric.gross_fen",
        "semantic-metric-gross@2026-09-21#0000",
    ]
    assert first.evidence.strategy_version == "hybrid-v1" and first.evidence.snapshot_id == retriever.index.snapshot_id
    assert first.evidence.embedding_actual_return["input_count"] == 1
    assert _digest(retriever.search("客户姓名", context=_context(role="approver"), top_k=5)) == "1417d36505d7952b"
    assert _digest(retriever.search("订单", context=_context(tenant_id="tenant-B"), top_k=5)) == "133382047c0db8b9"
    assert _digest(retriever.search("__unknown__", context=_context(), top_k=3)) == "6692ad826ae53bd5"


def test_rerank_evidence_is_pinned() -> None:
    retriever = _retriever()
    context = _context()
    visible = retriever._visible_candidates(context)
    retriever.reranker = FakeReranker({candidate_id: float(index) for index, candidate_id in enumerate(visible)})
    assert _digest(retriever.search("订单", context=context, top_k=3)) == "47f0ab1c8d02da64"
    retriever.reranker = _FailedReranker()
    failed = retriever.search("订单", context=context, top_k=3)
    assert _digest(failed) == "04be3278f4ba20e8"
    assert failed.items == () and failed.evidence.ranking == ()


def test_keyword_baseline_evidence_is_pinned() -> None:
    baseline = _keyword_retriever()
    assert _digest(baseline.search("营业额", context=_context(), top_k=3)) == "7bdb9585bfd2d2e1"
    assert _digest(baseline.search("客户姓名", context=_context(role="approver"), top_k=5)) == "819764198e6e4d3e"


@pytest.mark.parametrize("query", ["", "   ", None, 3])
@pytest.mark.parametrize("make", [_retriever, _keyword_retriever])
def test_both_searches_refuse_a_bad_query(make, query) -> None:
    with pytest.raises(RetrievalConfigurationError, match="query must be a non-empty string"):
        make().search(query, context=_context())


@pytest.mark.parametrize("top_k", [0, 6, True, "3", None])
@pytest.mark.parametrize("make", [_retriever, _keyword_retriever])
def test_both_searches_refuse_a_bad_top_k(make, top_k) -> None:
    with pytest.raises(RetrievalConfigurationError, match="top_k must be between one and five"):
        make().search("订单", context=_context(), top_k=top_k)


def test_a_query_is_checked_before_top_k_and_the_context_after_both() -> None:
    with pytest.raises(RetrievalConfigurationError, match="query must be a non-empty string"):
        _retriever().search("", context=_context(), top_k=0)
    with pytest.raises(RetrievalConfigurationError, match="retrieval requires a server-created context"):
        _retriever().search("订单", context=SimpleNamespace(run_id="r", tenant_id="A", role="requester"))


def test_a_retriever_must_pair_an_index_with_its_snapshot() -> None:
    retriever = _retriever()
    with pytest.raises(RetrievalConfigurationError, match="index and snapshot IDs do not match"):
        HybridRetriever(retriever.catalog, replace(retriever.snapshot, snapshot_id="other"), retriever.index, retriever.embedder)
    with pytest.raises(RetrievalConfigurationError, match="snapshot and index model revisions do not match"):
        HybridRetriever(retriever.catalog, replace(retriever.snapshot, embedding_model_revision="other"), retriever.index, retriever.embedder)
    with pytest.raises(RetrievalConfigurationError, match="snapshot and index dimensions do not match"):
        HybridRetriever(retriever.catalog, replace(retriever.snapshot, embedding_dimensions=4), retriever.index, retriever.embedder)


# --- who may see which source -------------------------------------------------------------------


def _acl_retriever() -> HybridRetriever:
    """Sources with every kind of scope and role; one is flipped to deleted after the index is built."""

    import tempfile

    root = Path(tempfile.mkdtemp(prefix="qs-s2-acl-"))
    names = {
        "s-global": {},
        "s-a": {"tenant_scope": "A"},
        "s-b": {"tenant_scope": "B"},
        "s-approver": {"allowed_roles": ["approver"]},
        "s-flip": {},
    }
    _write_tree(
        root,
        {f"{name}.md": f"text of {name}" for name in names},
        [_registry_entry(name, f"{name}.md", **overrides) for name, overrides in names.items()],
    )
    snapshot = import_knowledge(root, root / "source_registry.json", catalog_version="catalog-x", knowledge_version="kv-acl")
    embedder = FixedEmbedding(
        {chunk.text: (1.0, 0.0, 0.0) for chunk in snapshot.chunk_records} | {"query": (1.0, 0.0, 0.0)},
        model_revision="fixed-acl",
        dimensions=3,
    )
    build = build_embedding_index(snapshot, embedder, ingest_job_id="ingest-acl")
    flipped = tuple(
        replace(source, status="deleted") if source.source_id == "s-flip" else source for source in build.snapshot.source_records
    )
    return HybridRetriever(
        catalog=load_default_catalog(), snapshot=replace(build.snapshot, source_records=flipped), index=build.index, embedder=embedder
    )


def _chunk_sources(retriever: HybridRetriever, tenant_id: str, role: str) -> list[str]:
    context = ExecutionContext(run_id="run-acl", tenant_id=tenant_id, principal_id="principal-acl", role=role)
    candidates = retriever._visible_candidates(context)
    return sorted(item.source_id for item in candidates.values() if item.candidate_type == "knowledge")


@pytest.mark.parametrize(
    ("tenant_id", "role", "expected"),
    [
        ("A", "requester", ["s-a", "s-global"]),
        ("tenant-A", "requester", ["s-a", "s-global"]),
        ("A", "approver", ["s-a", "s-approver", "s-global"]),
        ("tenant-A", "approver", ["s-a", "s-approver", "s-global"]),
        ("B", "requester", ["s-b", "s-global"]),
        ("tenant-B", "approver", ["s-approver", "s-b", "s-global"]),
        ("a", "requester", ["s-global"]),
        ("tenant-a", "requester", ["s-global"]),
        ("tenant-tenant-A", "requester", ["s-global"]),
        ("C", "requester", ["s-global"]),
        ("tenant-C", "approver", ["s-approver", "s-global"]),
        ("A", "auditor", []),
    ],
)
def test_a_chunk_is_visible_by_source_status_role_and_tenant(tenant_id: str, role: str, expected: list[str]) -> None:
    assert _chunk_sources(_acl_retriever(), tenant_id, role) == expected


def test_catalog_entries_that_need_approval_are_visible_to_approvers_only() -> None:
    retriever = _acl_retriever()
    catalog_ids = {entry.id for entry in retriever.catalog.entries}
    open_ids = {entry.id for entry in retriever.catalog.entries if not entry.requires_approval}
    assert open_ids and open_ids < catalog_ids

    def visible(role: str) -> set[str]:
        context = ExecutionContext(run_id="run-acl", tenant_id="A", principal_id="principal-acl", role=role)
        return {key for key, item in retriever._visible_candidates(context).items() if item.candidate_type == "catalog"}

    assert visible("requester") == open_ids
    assert visible("approver") == catalog_ids
    assert visible("auditor") == open_ids  # any other role is treated as a requester


def test_an_index_chunk_must_be_bound_to_a_snapshot_source_before_its_visibility_counts() -> None:
    retriever = _acl_retriever()
    context = ExecutionContext(run_id="run-acl", tenant_id="A", principal_id="principal-acl", role="requester")
    snapshot, catalog = retriever.snapshot, retriever.catalog

    def chunk(**overrides):
        values = {"chunk_id": "s-global@v1#0000", "source_id": "s-global", "source_version": "v1", "text": "t"}
        values.update(overrides)
        return SimpleNamespace(**values)

    def refused(raw, message):
        with pytest.raises(RetrievalConfigurationError, match=message):
            _visible_candidates(catalog, snapshot, context, [raw])

    refused(chunk(source_id="unknown"), "index chunk is not bound to the snapshot source")
    refused(chunk(source_id=7), "index chunk is not bound to the snapshot source")
    refused(chunk(source_version="v2"), "index chunk is not bound to the snapshot source")
    # An invisible source is still checked for the binding, but its content is never looked at.
    refused(chunk(source_id="s-b", source_version="v9"), "index chunk is not bound to the snapshot source")
    assert "x" not in _visible_candidates(catalog, snapshot, context, [chunk(source_id="s-b", chunk_id=None, text=None)])
    # A visible source needs a string id and text.
    refused(chunk(chunk_id=None), "snapshot chunk has invalid identity or content")
    refused(chunk(text=3), "snapshot chunk has invalid identity or content")
    refused(chunk(chunk_id="metric.gross_fen"), "catalog and knowledge candidate IDs collide")
    refused_twice = [chunk(), chunk()]
    # Chunks of the same id: the second one collides with the first.
    with pytest.raises(RetrievalConfigurationError, match="catalog and knowledge candidate IDs collide"):
        _visible_candidates(catalog, snapshot, context, refused_twice)


def test_catalog_candidate_ids_must_be_unique() -> None:
    retriever = _acl_retriever()
    context = ExecutionContext(run_id="run-acl", tenant_id="A", principal_id="principal-acl", role="approver")
    first = retriever.catalog.entries[0]
    duplicated = SimpleNamespace(entries=(first, first))
    with pytest.raises(RetrievalConfigurationError, match="catalog candidate IDs are not unique"):
        _visible_candidates(duplicated, retriever.snapshot, context, [])


def test_the_retrieval_item_visibility_record_is_pinned() -> None:
    retriever = _acl_retriever()
    source = {item.source_id: item for item in retriever.snapshot.source_records}["s-a"]
    chunk = SimpleNamespace(source_id="s-a", source_version="v1")
    item = {"id": "s-a@v1#0000", "source_id": "s-a", "version": "v1"}
    requester = ExecutionContext(run_id="r", tenant_id="tenant-A", principal_id="p", role="requester")
    visibility = ControlledTools._retrieval_item_visibility

    passed = visibility(item, context=requester, chunk=chunk, source=source, catalog_entry=None)
    assert passed == {
        "candidate_type": "knowledge_document",
        "source_active": True,
        "version_matches_snapshot": True,
        "tenant_visible": True,
        "role_visible": True,
        "passed": True,
    }
    other = replace(requester, tenant_id="tenant-B")
    assert visibility(item, context=other, chunk=chunk, source=source, catalog_entry=None)["tenant_visible"] is False
    deleted = replace(source, status="deleted")
    assert visibility(item, context=requester, chunk=chunk, source=deleted, catalog_entry=None)["passed"] is False
    narrow = replace(source, allowed_roles=("approver",))
    assert visibility(item, context=requester, chunk=chunk, source=narrow, catalog_entry=None)["role_visible"] is False
    assert visibility({**item, "version": "v9"}, context=requester, chunk=chunk, source=source, catalog_entry=None)["version_matches_snapshot"] is False

    entries = {entry.id: entry for entry in retriever.catalog.entries}
    open_entry = next(entry for entry in entries.values() if not entry.requires_approval)
    gated_entry = next(entry for entry in entries.values() if entry.requires_approval)
    for entry, role, role_visible in (
        (open_entry, "requester", True),
        (gated_entry, "requester", False),
        (gated_entry, "approver", True),
    ):
        context = replace(requester, role=role)
        record = visibility(entry.as_search_item(), context=context, chunk=None, source=None, catalog_entry=entry)
        assert record == {
            "candidate_type": "catalog_retrieval_candidate",
            "source_active": True,
            "version_matches_snapshot": True,
            "tenant_visible": True,
            "role_visible": role_visible,
            "passed": role_visible,
        }
    unknown = visibility(item, context=requester, chunk=None, source=None, catalog_entry=None)
    assert unknown["candidate_type"] == "unknown" and unknown["passed"] is False


def _search_result(items: list[dict[str, str]]):
    text = json.dumps({"items": items}, ensure_ascii=False, sort_keys=True)
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text, meta=None, annotations=None)],
        structured_content={"items": items},
        meta=None,
        is_error=False,
    )


def test_the_host_judges_a_returned_chunk_by_the_hosts_own_rules() -> None:
    retriever = _acl_retriever()
    catalog = retriever.catalog
    chunks = {chunk.source_id: chunk for chunk in retriever.index.chunks}

    def item(source_id: str) -> dict[str, str]:
        chunk = chunks[source_id]
        return {"id": chunk.chunk_id, "text": chunk.text, "source_id": chunk.source_id, "version": chunk.source_version}

    def check(source_id: str, tenant_id: str, role: str):
        ctx = ExecutionContext(run_id="r", tenant_id=tenant_id, principal_id="p", role=role)
        return verify.checked_search_items(_search_result([item(source_id)]), top_k=3, context=ctx, catalog=catalog, retriever=retriever)

    assert check("s-a", "tenant-A", "requester") == {"items": [item("s-a")]}
    assert check("s-global", "B", "requester") == {"items": [item("s-global")]}
    assert check("s-approver", "A", "approver") == {"items": [item("s-approver")]}
    for source_id, tenant_id, role in (
        ("s-a", "tenant-B", "requester"),
        ("s-approver", "A", "requester"),
        ("s-flip", "A", "requester"),
        ("s-b", "A", "approver"),
    ):
        with pytest.raises(verify.ResultInvalid, match="the item is not visible to this identity"):
            check(source_id, tenant_id, role)

    # A source whose version moved on is not visible either.
    moved = tuple(replace(source, version="v9") if source.source_id == "s-a" else source for source in retriever.snapshot.source_records)
    moved_retriever = replace(retriever, snapshot=replace(retriever.snapshot, source_records=moved))
    with pytest.raises(verify.ResultInvalid, match="the item is not visible to this identity"):
        verify.checked_search_items(
            _search_result([item("s-a")]),
            top_k=3,
            context=ExecutionContext(run_id="r", tenant_id="A", principal_id="p", role="requester"),
            catalog=catalog,
            retriever=moved_retriever,
        )

    gated = next(entry for entry in catalog.entries if entry.requires_approval)
    for role, passes in (("requester", False), ("approver", True)):
        ctx = ExecutionContext(run_id="r", tenant_id="A", principal_id="p", role=role)
        call = lambda: verify.checked_search_items(  # noqa: E731
            _search_result([gated.as_search_item()]), top_k=3, context=ctx, catalog=catalog, retriever=retriever
        )
        if passes:
            assert call() == {"items": [gated.as_search_item()]}
        else:
            with pytest.raises(verify.ResultInvalid, match="the item is not visible to this identity"):
                call()


# --- the snapshot repository the approval re-checks a permission source with --------------------


@pytest.fixture
def repository(tmp_path: Path):
    root = DEFAULT_KNOWLEDGE_ROOT
    snapshot = build_snapshot(root, root / "source_registry.json", catalog_version="catalog-v1")
    state = StateStore(tmp_path / "knowledge.sqlite")
    repository = KnowledgeSnapshotRepository(state)
    repository.publish(snapshot)
    yield repository, snapshot
    state.close()


def _access_code(repository, snapshot_id, source_id, tenant_id="A", role="requester") -> str:
    with pytest.raises(KnowledgeAccessError) as caught:
        repository.visible_source(snapshot_id=snapshot_id, source_id=source_id, identity=KnowledgeIdentity(tenant_id, "p", role))
    assert str(caught.value) == "knowledge source is not visible"
    return caught.value.code


def test_the_repository_reads_back_what_was_published(repository) -> None:
    repo, snapshot = repository
    assert repo.current()["snapshot_id"] == snapshot.snapshot_id
    assert repo.get_snapshot(snapshot.snapshot_id)["snapshot_id"] == snapshot.snapshot_id
    assert repo.get_snapshot("knowledge-v1-0000") is None


def test_visible_source_gives_the_stored_source_or_one_fixed_code(repository) -> None:
    repo, snapshot = repository
    source_id = "tenant-a-orders-overview"
    visible = repo.visible_source(snapshot_id=snapshot.snapshot_id, source_id=source_id, identity=KnowledgeIdentity("tenant-A", "p", "requester"))
    assert visible == next(item for item in snapshot.as_dict()["sources"] if item["source_id"] == source_id)

    assert _access_code(repo, "knowledge-v1-0000", source_id) == "snapshot_not_found"
    assert _access_code(repo, snapshot.snapshot_id, "no-such-source") == "source_not_found"
    assert _access_code(repo, snapshot.snapshot_id, source_id, tenant_id="B") == "source_not_found"
    assert _access_code(repo, snapshot.snapshot_id, source_id, tenant_id="a") == "source_not_found"
    assert _access_code(repo, snapshot.snapshot_id, source_id, role="auditor") == "source_not_found"
    # A source that is open to everyone in the tenant scope "global" stays visible to every tenant.
    assert repo.visible_source(snapshot_id=snapshot.snapshot_id, source_id="semantic-metric-gross", identity=KnowledgeIdentity("B", "p", "requester"))


def test_a_revoked_source_is_reported_before_tenant_and_role_are_looked_at(repository) -> None:
    repo, snapshot = repository
    source_id = "tenant-a-orders-overview"
    repo.revoke(source_id)
    assert _access_code(repo, snapshot.snapshot_id, source_id) == "source_revoked"
    assert _access_code(repo, snapshot.snapshot_id, source_id, tenant_id="B", role="auditor") == "source_revoked"


def test_a_changed_acl_is_what_visible_source_judges(repository) -> None:
    repo, snapshot = repository
    source_id = "tenant-a-orders-overview"
    repo.state.set_source_acl(source_id, allowed_roles=("approver",))
    assert _access_code(repo, snapshot.snapshot_id, source_id) == "source_not_found"
    assert repo.visible_source(snapshot_id=snapshot.snapshot_id, source_id=source_id, identity=KnowledgeIdentity("A", "p", "approver"))
    repo.state.set_source_acl(source_id, tenant_scope="B")
    assert _access_code(repo, snapshot.snapshot_id, source_id, role="approver") == "source_not_found"
    assert repo.visible_source(snapshot_id=snapshot.snapshot_id, source_id=source_id, identity=KnowledgeIdentity("tenant-B", "p", "approver"))


# --- the product knowledge base and the retrievers built over it ---------------------------------


@pytest.fixture
def fresh_cache():
    reset_retrieval_cache()
    yield
    reset_retrieval_cache()


def test_the_product_knowledge_is_the_default_or_the_demo_base_and_is_cached(fresh_cache) -> None:
    default = product_knowledge(demo=False)
    demo = product_knowledge(demo=True)
    assert default.snapshot_id == "knowledge-v1-9f580dd7f887ed0a"
    assert default.sensitive_source_id == DEFAULT_SENSITIVE_PERMISSION_SOURCE_ID == "semantic-sensitive-customer-name"
    assert demo.snapshot_id == "knowledge-demo-v1-142c2433a63cc6a8"
    assert demo.sensitive_source_id == DEMO_SENSITIVE_PERMISSION_SOURCE_ID == "demo-sensitive-customer-name"
    assert product_knowledge(demo=False) is default and product_knowledge(demo=True) is demo
    assert default.snapshot.catalog_version == "catalog-v2" and demo.snapshot.catalog_version == DEMO_CATALOG_VERSION


def test_a_knowledge_base_without_the_permission_source_is_refused(fresh_cache, monkeypatch) -> None:
    import queryshield.knowledge.runtime as runtime

    monkeypatch.setattr(runtime, "DEFAULT_SENSITIVE_PERMISSION_SOURCE_ID", "no-such-source")
    with pytest.raises(ValueError, match="the knowledge base has no permission source for customer names"):
        product_knowledge(demo=False)
    # A failure is not cached.
    monkeypatch.undo()
    assert product_knowledge(demo=False).sensitive_source_id == "semantic-sensitive-customer-name"


def test_the_shared_runtimes_are_built_once_per_mode_and_base(fresh_cache) -> None:
    default = shared_retrieval_runtime("fake")
    demo = shared_demo_retrieval_runtime("fake")
    assert shared_retrieval_runtime("fake") is default and shared_demo_retrieval_runtime("fake") is demo
    assert default is not demo
    assert default.index_build.ingest_job_id == "w05-retrieval-fake"
    assert demo.index_build.ingest_job_id == "demo-retrieval-fake"
    assert default.snapshot.catalog_version == "catalog-v2" and demo.snapshot.catalog_version == DEMO_CATALOG_VERSION
    assert default.snapshot.knowledge_version == "knowledge-v1" and demo.snapshot.knowledge_version == DEMO_KNOWLEDGE_VERSION
    assert default.index_build.base_snapshot_id == product_knowledge(demo=False).snapshot_id
    assert demo.index_build.base_snapshot_id == product_knowledge(demo=True).snapshot_id
    assert default.mode == "fake" and default.retriever.snapshot is default.snapshot
    with pytest.raises(ValueError, match="retrieval mode must be fake or real"):
        build_retrieval_runtime("other")


def test_a_failed_runtime_build_is_not_cached(fresh_cache, monkeypatch) -> None:
    import queryshield.knowledge.runtime as runtime

    calls = []
    real_build = runtime.build_retrieval_runtime

    def failing(mode, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("build failed")

    monkeypatch.setattr(runtime, "build_retrieval_runtime", failing)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="build failed"):
            shared_retrieval_runtime("fake")
    monkeypatch.setattr(runtime, "build_retrieval_runtime", real_build)
    assert shared_retrieval_runtime("fake").mode == "fake" and len(calls) == 2


@pytest.mark.parametrize("demo", [False, True])
def test_a_retriever_over_an_index_file_is_the_hosts_retriever(fresh_cache, tmp_path: Path, demo: bool) -> None:
    host = shared_demo_retrieval_runtime("fake") if demo else shared_retrieval_runtime("fake")
    path = publish_index(host.index_build.index, tmp_path / "index.json")
    retriever = retriever_from_index_file("fake", path, demo=demo, expected_snapshot_id=host.snapshot.snapshot_id)

    assert retriever.snapshot == host.snapshot
    assert retriever.index == host.index_build.index
    ctx = ExecutionContext(run_id="r", tenant_id="A", principal_id="p", role="requester")
    assert retriever.search("退款", context=ctx, top_k=3).items == host.retriever.search("退款", context=ctx, top_k=3).items
    with pytest.raises(IndexFileMismatch, match="the index file is not the expected embedded snapshot"):
        retriever_from_index_file("fake", path, demo=demo, expected_snapshot_id="knowledge-v1-0000")
    with pytest.raises(IndexFileMismatch, match="the index file is not the expected embedded snapshot"):
        retriever_from_index_file("fake", path, demo=not demo, expected_snapshot_id=host.snapshot.snapshot_id)
    with pytest.raises(ValueError, match="retrieval mode must be fake or real"):
        retriever_from_index_file("other", path, demo=demo, expected_snapshot_id=host.snapshot.snapshot_id)


@pytest.mark.parametrize(
    ("model", "revision", "dimensions"),
    [
        ("w05-hash-feature-fake-v1", "other-revision", 128),
        ("other-model", "w05-hash-feature-v1", 128),
        ("w05-hash-feature-fake-v1", "w05-hash-feature-v1", 64),
    ],
)
def test_an_index_built_by_another_embedder_is_refused_by_the_server_with_one_fixed_reason(
    fresh_cache, tmp_path: Path, model: str, revision: str, dimensions: int
) -> None:
    """The server maps this failure to ``index_mismatch`` whatever the exception behind it."""

    from queryshield.knowledge.runtime import feature_vector
    from queryshield.mcp_metadata import server

    snapshot = product_knowledge(demo=False).snapshot
    other = FixedEmbedding(
        {chunk.text: feature_vector(chunk.text, dimensions=dimensions) for chunk in snapshot.chunk_records},
        model=model,
        model_revision=revision,
        dimensions=dimensions,
    )
    build = build_embedding_index(snapshot, other, ingest_job_id="ingest-other")
    path = publish_index(build.index, tmp_path / "index.json")
    args = SimpleNamespace(
        package_dir=str(Path(__import__("queryshield").__file__).resolve().parent),
        retrieval="hybrid",
        knowledge="default",
        mode="fake",
        index_path=str(path),
        expected_snapshot_id=build.snapshot.snapshot_id,
    )
    with pytest.raises(server._Refused) as caught:
        server._build_tools(args)
    assert caught.value.reason == "index_mismatch"
    with pytest.raises(ValueError):  # the index file does not match, or the retriever refuses the embedder
        retriever_from_index_file("fake", path, demo=False, expected_snapshot_id=build.snapshot.snapshot_id)


def test_a_retriever_refuses_an_embedder_that_does_not_match_its_index() -> None:
    retriever = _retriever()
    other_revision = FixedEmbedding({"q": (1.0, 0.0, 0.0)}, model_revision="other", dimensions=3)
    other_dimensions = FixedEmbedding({"q": (1.0, 0.0)}, model_revision="fixed-hybrid-v1", dimensions=2)
    with pytest.raises(RetrievalConfigurationError, match="query embedding revision does not match the index"):
        HybridRetriever(retriever.catalog, retriever.snapshot, retriever.index, other_revision)
    with pytest.raises(RetrievalConfigurationError, match="query embedding dimensions do not match the index"):
        HybridRetriever(retriever.catalog, retriever.snapshot, retriever.index, other_dimensions)


# --- the MCP session record --------------------------------------------------------------------

_VOLATILE = ("session_id", "server_pid", "started_ms", "duration_ms", "sdk_version", "knowledge_snapshot_id")


def _stable(record: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in record.items() if key not in _VOLATILE}


def _record_of(*, fixture, retriever, kind, timeout):
    tools = McpMetadataTools(retriever=retriever, metadata_config=config(fixture=fixture, call_timeout=timeout))
    outcome: object = "ok"
    try:
        if kind == "search":
            tools.search_catalog({"query": "退款后净额", "top_k": 3}, context=context("A"))
        else:
            tools.describe_tables({"tables": ["orders"]}, context=context("A"))
    except ToolError as exc:
        outcome = (type(exc).__name__, exc.code)
    call = tools.take_metadata_call_record()
    record = tools.close()
    return outcome, {key: value for key, value in call.items() if key != "mcp_session_id"}, _stable(record)


_BASE = {
    "cleanup": "ok",
    "cleanup_error": None,
    "failure_source": None,
    "initialize": "ok",
    "list": "ok",
    "protocol_version": "2025-11-25",
    "refused_reason": None,
    "server_exited": True,
    "tools_listed": ["search_catalog", "describe_tables"],
    "transport": "mcp_stdio",
}
_CALL = {"mcp_request_sent": True, "transport": "mcp_stdio"}


@pytest.mark.parametrize(
    ("fixture", "hybrid", "kind", "timeout", "outcome", "mcp_outcome", "record_edit"),
    [
        ("sleep", False, "search", 1.0, ("McpToolError", "mcp_timeout"), "timeout", {"call_count": 1, "failure_code": "mcp_timeout", "retrieval": "keyword"}),
        ("die", False, "describe", 2.0, ("McpToolError", "mcp_unavailable"), "unavailable", {"call_count": 1, "failure_code": "mcp_unavailable", "retrieval": "keyword"}),
        ("upstream", True, "search", 2.0, ("McpToolError", "mcp_unavailable"), "unavailable", {"call_count": 1, "failure_code": "mcp_unavailable", "failure_source": "upstream", "retrieval": "hybrid"}),
        ("plain_exception", True, "search", 2.0, ("McpToolError", "mcp_result_invalid"), "result_invalid", {"call_count": 1, "failure_code": "mcp_result_invalid", "retrieval": "hybrid"}),
        ("text_changed", True, "search", 2.0, ("McpToolError", "mcp_result_invalid"), "result_invalid", {"call_count": 1, "failure_code": "mcp_result_invalid", "retrieval": "hybrid"}),
        (
            "extra_tool", False, "search", 2.0, ("McpToolError", "mcp_protocol_error"), "protocol_error",
            {"call_count": 0, "failure_code": "mcp_protocol_error", "retrieval": "keyword", "list": "mismatch", "tools_listed": ["search_catalog", "describe_tables", "query_readonly"]},
        ),
        (None, True, "search", 2.0, "ok", "ok", {"call_count": 1, "failure_code": None, "retrieval": "hybrid"}),
        ("server_error_retrieval_unavailable", True, "search", 2.0, ("ToolError", "retrieval_unavailable"), "tool_error", {"call_count": 1, "failure_code": None, "retrieval": "hybrid"}),
    ],
)
def test_the_session_record_after_each_kind_of_outcome(fixture, hybrid, kind, timeout, outcome, mcp_outcome, record_edit) -> None:
    retriever = shared_retrieval_runtime("fake").retriever if hybrid else None
    got_outcome, call, record = _record_of(fixture=fixture, retriever=retriever, kind=kind, timeout=timeout)
    assert got_outcome == outcome
    assert call == {**_CALL, "mcp_outcome": mcp_outcome}
    assert record == {**_BASE, **record_edit}


@pytest.mark.parametrize(
    ("command", "args", "start_timeout", "initialize"),
    [("sleep", ("30",), 1.0, "timeout"), ("true", (), 5.0, "failed"), ("/nonexistent-binary", (), 5.0, "not_run")],
)
def test_a_server_that_never_starts_is_unavailable_and_closed(tmp_path: Path, command, args, start_timeout, initialize) -> None:
    spec = LaunchSpec(command, args, {}, str(tmp_path), "keyword", None)
    session = McpMetadataSession(spec, call_timeout=1.0, start_timeout=start_timeout)
    with pytest.raises(McpSessionError) as caught:
        session.start()
    assert caught.value.code == "mcp_unavailable"
    assert _stable(session.record) == {
        "call_count": 0,
        "cleanup": "ok",
        "cleanup_error": None,
        "failure_code": "mcp_unavailable",
        "failure_source": None,
        "initialize": initialize,
        "list": "not_run",
        "protocol_version": None,
        "refused_reason": None,
        "retrieval": "keyword",
        "server_exited": None,
        "tools_listed": [],
        "transport": "mcp_stdio",
    }
    assert session._broken is True and session._closed is True
    with pytest.raises(McpSessionError) as again:
        session.call("search_catalog", {"query": "x"})
    assert again.value.code == "mcp_unavailable"


async def _raise(exc: BaseException):
    raise exc


async def _hang():
    await asyncio.sleep(5)


@pytest.mark.parametrize(
    ("replacement", "code"),
    [
        (lambda *a: _raise(McpSessionError("mcp_protocol_error")), "mcp_protocol_error"),
        (lambda *a: _raise(McpSessionError("mcp_result_invalid")), "mcp_result_invalid"),
        (lambda *a: _raise(RuntimeError("anything")), "mcp_unavailable"),
        (lambda *a: _hang(), "mcp_timeout"),
    ],
)
def test_a_failed_call_breaks_the_session_and_records_the_first_code(replacement, code) -> None:
    session = started_session(context("A"), call_timeout=0.3)
    try:
        session._call = replacement
        with pytest.raises(McpSessionError) as caught:
            session.call("search_catalog", {"query": "x"})
        assert caught.value.code == code
        assert session._broken is True
        assert (session.record["failure_code"], session.record["failure_source"], session.record["call_count"]) == (code, None, 1)
        with pytest.raises(McpSessionError) as again:
            session.call("search_catalog", {"query": "x"})
        assert again.value.code == "mcp_unavailable"
        assert (session.record["failure_code"], session.record["call_count"]) == (code, 1)
    finally:
        session.close()


def test_the_first_failure_marked_on_a_session_wins() -> None:
    session = started_session(context("A"))
    try:
        session.mark_failed("mcp_result_invalid", source="upstream")
        session.mark_failed("mcp_timeout", source="other")
        assert (session.record["failure_code"], session.record["failure_source"]) == ("mcp_result_invalid", "upstream")
        with pytest.raises(McpSessionError) as caught:
            session.call("search_catalog", {"query": "x"})
        assert caught.value.code == "mcp_unavailable"
        assert session.record["call_count"] == 0
    finally:
        record = session.close()
    assert (record["failure_code"], record["failure_source"], record["cleanup"]) == ("mcp_result_invalid", "upstream", "ok")


def test_a_call_to_a_name_the_session_does_not_list_is_a_protocol_error_before_anything_is_sent() -> None:
    session = started_session(context("A"))
    try:
        with pytest.raises(McpSessionError) as caught:
            session.call("query_readonly", {})
        assert caught.value.code == "mcp_protocol_error"
        assert session.record["call_count"] == 0 and session._broken is False
    finally:
        session.close()


@pytest.mark.parametrize(
    ("edit", "cleanup", "error", "exited", "stderr"),
    [
        ("alive", "failed", "process_still_alive", False, True),
        ("unknown", "unverified", None, None, False),
        ("no_pid", "failed", "pid_unknown", None, True),
    ],
)
def test_the_close_record_for_each_way_a_process_exit_can_be_judged(monkeypatch, capsys, edit, cleanup, error, exited, stderr) -> None:
    session = started_session(context("A"))
    if edit == "alive":
        monkeypatch.setattr(session_module, "wait_until_gone", lambda pid, seconds: False)
    elif edit == "unknown":
        monkeypatch.setattr(session_module, "wait_until_gone", lambda pid, seconds: None)
    else:
        session._record["server_pid"] = None
        monkeypatch.setattr(session, "_read_pid", lambda: None)
    record = session.close()
    assert (record["cleanup"], record["cleanup_error"], record["server_exited"]) == (cleanup, error, exited)
    assert isinstance(record["duration_ms"], int)
    assert session.close() == record
    line = f"queryshield-mcp-metadata cleanup_failed reason={error}\n"
    assert (line in capsys.readouterr().err) is stderr
    assert session not in session_module._OPEN_SESSIONS


def test_the_facade_closes_its_session_once_and_removes_its_directory() -> None:
    tools = McpMetadataTools(retriever=None, metadata_config=config())
    tools.describe_tables({"tables": ["orders"]}, context=context("A"))
    directory = Path(tools._session_dir)
    assert directory.exists()
    first = tools.close()
    assert first is not None and first["cleanup"] == "ok" and not directory.exists()
    assert tools.close() == first
    assert tools.take_metadata_call_record() is not None
    assert tools.take_metadata_call_record() is None


def test_an_mcp_tools_object_needs_a_configuration() -> None:
    with pytest.raises(TypeError, match="an MCP metadata configuration is required"):
        McpMetadataTools(retriever=None)


# --- the MCP server and its tool definitions ----------------------------------------------------


def test_the_tool_definitions_are_pinned() -> None:
    from queryshield.mcp_metadata.schemas import DESCRIBE_TABLES_INPUT, SEARCH_CATALOG_INPUT, tool_definitions
    from queryshield.mcp_metadata.schemas import DESCRIBE_TABLES_OUTPUT, SEARCH_CATALOG_OUTPUT

    assert _sha(_canonical(tool_definitions())) == "f58804350f912bac9891e217a060f7ff1e26a6faedfcc2cf717328e9606be34b"
    assert SEARCH_CATALOG_INPUT["properties"]["top_k"] == {"type": "integer", "minimum": 1, "maximum": 5, "default": 3}
    assert SEARCH_CATALOG_OUTPUT["properties"]["items"]["maxItems"] == 5
    assert DESCRIBE_TABLES_INPUT["properties"]["tables"]["minItems"] == 1 and DESCRIBE_TABLES_INPUT["properties"]["tables"]["maxItems"] == 3
    assert DESCRIBE_TABLES_OUTPUT["properties"]["tables"]["minItems"] == 1 and DESCRIBE_TABLES_OUTPUT["properties"]["tables"]["maxItems"] == 3


@pytest.mark.parametrize("role", ["requester", "approver"])
def test_the_server_accepts_the_two_roles_and_refuses_any_other(role) -> None:
    from queryshield.mcp_metadata import server

    base = ["--run-id=r", "--tenant-id=A", "--principal-id=p", "--package-dir=/x", "--retrieval=keyword", "--knowledge=default", "--mode=fake"]
    assert server._parse([*base, f"--role={role}"]).role == role
    for bad in ("admin", "Requester", ""):
        with pytest.raises(server._Refused) as caught:
            server._parse([*base, f"--role={bad}"])
        assert caught.value.reason == "invalid_role"
    with pytest.raises(server._Refused) as caught:
        server._parse([*base[:-3], "--retrieval=hybrid", "--knowledge=default", "--mode=fake", "--role=requester"])
    assert caught.value.reason == "missing_index"


@pytest.mark.parametrize("field", ["--run-id", "--tenant-id", "--principal-id"])
def test_the_server_refuses_a_blank_identity_with_its_own_reason(field: str) -> None:
    from queryshield.mcp_metadata import server

    values = {"--run-id": "r", "--tenant-id": "A", "--principal-id": "p"}
    values[field] = "  "
    argv = [f"{name}={value}" for name, value in values.items()]
    argv += ["--package-dir=/x", "--retrieval=keyword", "--knowledge=default", "--mode=fake", "--role=requester"]
    with pytest.raises(server._Refused) as caught:
        server._parse(argv)
    assert caught.value.reason == "empty_identity"


def test_the_import_roles_are_the_two_the_catalog_allows() -> None:
    from queryshield.catalog.catalog import ALLOWED_ROLES

    assert ALLOWED_ROLES == frozenset({"requester", "approver"})


# --- knowledge/acl.py: the one visibility rule --------------------------------------------------


def test_the_acl_module_does_not_load_the_retrieval_code() -> None:
    import subprocess
    import sys

    code = "import sys, queryshield.knowledge.acl; sys.exit(1 if 'queryshield.knowledge.retrieval' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


@pytest.mark.parametrize(
    ("scope", "tenant_id", "visible"),
    [
        ("global", "A", True),
        ("global", "anything", True),
        ("A", "A", True),
        ("A", "tenant-A", True),
        ("A", "B", False),
        ("A", "a", False),
        ("tenant-A", "A", False),
        ("tenant-A", "tenant-A", True),
        ("B", "tenant-A", False),
    ],
)
def test_tenant_matches_is_the_scope_or_its_tenant_prefixed_form(scope: str, tenant_id: str, visible: bool) -> None:
    from queryshield.knowledge.acl import tenant_matches

    assert tenant_matches(scope, tenant_id) is visible


def _source(**overrides):
    from queryshield.knowledge.ingest import SourceRecord

    values = dict(
        source_id="s",
        path="s.md",
        version="v1",
        content_sha256="a" * 64,
        tenant_scope="A",
        allowed_roles=("requester",),
        status="active",
        updated_at="2026-01-01T00:00:00Z",
    )
    values.update(overrides)
    return SourceRecord(**values)


def _ctx(tenant_id: str = "A", role: str = "requester") -> ExecutionContext:
    return ExecutionContext(run_id="r", tenant_id=tenant_id, principal_id="p", role=role)


@pytest.mark.parametrize(
    ("source_edit", "context", "visible"),
    [
        ({}, _ctx(), True),
        ({}, _ctx(role="approver"), False),
        ({"status": "deleted"}, _ctx(), False),
        ({"status": "pending"}, _ctx(), False),
        ({"allowed_roles": ("approver",)}, _ctx(), False),
        ({"allowed_roles": ("approver",)}, _ctx(role="approver"), True),
        ({"allowed_roles": ("requester", "approver")}, _ctx(role="approver"), True),
        ({}, _ctx(tenant_id="B"), False),
        ({"tenant_scope": "global"}, _ctx(tenant_id="B"), True),
        ({"tenant_scope": "A"}, _ctx(tenant_id="tenant-A"), True),
        ({"tenant_scope": "global", "status": "deleted"}, _ctx(), False),
    ],
)
def test_source_visible_needs_active_status_role_and_tenant(source_edit, context, visible: bool) -> None:
    from queryshield.knowledge.acl import source_visible

    assert source_visible(_source(**source_edit), context) is visible


def test_chunk_visible_needs_a_source_of_the_same_version_that_is_visible() -> None:
    from queryshield.knowledge.acl import chunk_visible

    sources = {"s": _source()}
    chunk = SimpleNamespace(source_id="s", source_version="v1")
    assert chunk_visible(chunk, sources, _ctx()) is True
    assert chunk_visible(SimpleNamespace(source_id="s", source_version="v2"), sources, _ctx()) is False
    assert chunk_visible(SimpleNamespace(source_id="other", source_version="v1"), sources, _ctx()) is False
    assert chunk_visible(chunk, sources, _ctx(tenant_id="B")) is False
    assert chunk_visible(chunk, {"s": _source(status="deleted")}, _ctx()) is False
    assert chunk_visible(chunk, {}, _ctx()) is False


def test_catalog_entry_visible_hides_approval_entries_from_everyone_but_approvers() -> None:
    from queryshield.knowledge.acl import catalog_entry_visible

    open_entry = SimpleNamespace(requires_approval=False)
    gated_entry = SimpleNamespace(requires_approval=True)
    assert catalog_entry_visible(open_entry, "requester") is True
    assert catalog_entry_visible(open_entry, "approver") is True
    assert catalog_entry_visible(gated_entry, "requester") is False
    assert catalog_entry_visible(gated_entry, "approver") is True
    assert catalog_entry_visible(gated_entry, "auditor") is False
    assert catalog_entry_visible(open_entry, "auditor") is True
