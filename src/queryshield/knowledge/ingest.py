from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from queryshield.catalog.catalog import ALLOWED_ROLES


KNOWLEDGE_VERSION = "knowledge-v1"
CHUNKER_VERSION = "paragraph-v1"
MAX_FILE_BYTES = 256 * 1024
MAX_FILES = 100
MAX_CHUNKS = 200
MAX_CHUNK_CHARS = 800
CHUNK_OVERLAP_CHARS = 100
ALLOWED_SUFFIXES = frozenset({".md", ".txt"})
ALLOWED_TENANT_SCOPES = frozenset({"global", "A", "B"})
_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9.-]+$")


class KnowledgeImportError(ValueError):
    """A trusted knowledge import cannot produce a safe deterministic snapshot."""


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    path: str
    version: str
    tenant_scope: str
    allowed_roles: tuple[str, ...]
    status: str
    updated_at: str


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    path: str
    version: str
    content_sha256: str
    tenant_scope: str
    allowed_roles: tuple[str, ...]
    status: str
    updated_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "path": self.path,
            "version": self.version,
            "content_sha256": self.content_sha256,
            "tenant_scope": self.tenant_scope,
            "allowed_roles": list(self.allowed_roles),
            "status": self.status,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source_id: str
    source_version: str
    text: str
    text_sha256: str
    chunker_version: str

    def as_dict(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "chunker_version": self.chunker_version,
        }


@dataclass(frozen=True)
class KnowledgeSnapshot:
    snapshot_id: str
    knowledge_version: str
    catalog_version: str
    chunker_version: str
    embedding_model_revision: str | None
    embedding_dimensions: int | None
    index_hash: str
    manifest_sha256: str
    source_records: tuple[SourceRecord, ...]
    chunk_records: tuple[ChunkRecord, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "knowledge_version": self.knowledge_version,
            "catalog_version": self.catalog_version,
            "chunker_version": self.chunker_version,
            "embedding_model_revision": self.embedding_model_revision,
            "embedding_dimensions": self.embedding_dimensions,
            "index_hash": self.index_hash,
            "manifest_sha256": self.manifest_sha256,
            "sources": [item.as_dict() for item in self.source_records],
            "chunks": [item.as_dict() for item in self.chunk_records],
        }


def _duplicate_check(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise KnowledgeImportError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def with_identity(snapshot: KnowledgeSnapshot) -> KnowledgeSnapshot:
    """The snapshot with its manifest hash and id computed from its other fields.

    The id is the knowledge version and the first 16 hex digits of the manifest hash, so it
    changes with the content, the catalog version, the embedding revision and the index hash.
    """

    manifest = {
        "knowledge_version": snapshot.knowledge_version,
        "catalog_version": snapshot.catalog_version,
        "chunker_version": snapshot.chunker_version,
        "embedding_model_revision": snapshot.embedding_model_revision,
        "embedding_dimensions": snapshot.embedding_dimensions,
        "index_hash": snapshot.index_hash,
        "sources": [item.as_dict() for item in snapshot.source_records],
        "chunks": [item.as_dict() for item in snapshot.chunk_records],
    }
    manifest_sha256 = sha256_hex(canonical_bytes(manifest))
    return replace(
        snapshot,
        snapshot_id=f"{snapshot.knowledge_version}-{manifest_sha256[:16]}",
        manifest_sha256=manifest_sha256,
    )


def atomic_write_text(destination: Path, text: str, *, temporary_prefix: str) -> None:
    """Write beside the destination and replace it, so a failure leaves the old file and no temporary one."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=temporary_prefix, suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary_name, destination)
    except Exception:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _string(value: object, *, path: str) -> str:
    if type(value) is not str or not value.strip():
        raise KnowledgeImportError(f"{path} must be a non-empty string")
    return value


def _string_list(value: object, *, path: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise KnowledgeImportError(f"{path} must be a non-empty list")
    items = tuple(_string(item, path=f"{path}[]") for item in value)
    if len(set(items)) != len(items):
        raise KnowledgeImportError(f"{path} must not contain duplicates")
    return items


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise KnowledgeImportError(f"{path} must be an object")
    return value


def _resolve_inside(root: Path, relative_path: str, *, path: str) -> Path:
    candidate_path = Path(relative_path)
    if candidate_path.is_absolute() or candidate_path.drive:
        raise KnowledgeImportError(f"{path} must not be absolute")
    try:
        candidate = (root / candidate_path).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError) as exc:
        raise KnowledgeImportError(f"{path} escapes the configured knowledge root") from exc
    if not candidate.is_file():
        raise KnowledgeImportError(f"{path} must point to a file")
    if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        raise KnowledgeImportError(f"{path} must be Markdown or plain text")
    return candidate


def _registry_sources(root: Path, registry_path: Path) -> list[object]:
    try:
        registry = registry_path.resolve(strict=True)
        registry.relative_to(root)
    except (OSError, ValueError) as exc:
        raise KnowledgeImportError("source registry must be inside the configured root") from exc
    try:
        document = json.loads(
            registry.read_text(encoding="utf-8"), object_pairs_hook=_duplicate_check
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise KnowledgeImportError(f"cannot read source registry: {registry}") from exc

    raw_sources = _mapping(document, path="registry").get("sources")
    if type(raw_sources) is not list or not raw_sources:
        raise KnowledgeImportError("registry.sources must be a non-empty list")
    if len(raw_sources) > MAX_FILES:
        raise KnowledgeImportError(f"registry contains more than {MAX_FILES} files")
    return raw_sources


def _acl_fields(source: Mapping[str, object], path: str) -> tuple[str, tuple[str, ...], str]:
    tenant_scope = _string(source.get("tenant_scope"), path=f"{path}.tenant_scope")
    if tenant_scope not in ALLOWED_TENANT_SCOPES:
        raise KnowledgeImportError(f"{path}.tenant_scope is unsupported")
    allowed_roles = _string_list(source.get("allowed_roles"), path=f"{path}.allowed_roles")
    if not set(allowed_roles) <= ALLOWED_ROLES:
        raise KnowledgeImportError(f"{path}.allowed_roles contains an unknown role")
    status = _string(source.get("status"), path=f"{path}.status")
    if status not in {"active", "deleted"}:
        raise KnowledgeImportError(f"{path}.status must be active or deleted")
    return tenant_scope, allowed_roles, status


def _reject_unregistered_files(root: Path, registered_paths: set[str]) -> None:
    discovered_paths: set[str] = set()
    for file in root.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        try:
            discovered_paths.add(file.resolve(strict=True).relative_to(root).as_posix())
        except (OSError, ValueError) as exc:
            raise KnowledgeImportError(
                f"knowledge file escapes the configured root: {file}"
            ) from exc
    unknown_paths = discovered_paths - registered_paths
    if unknown_paths:
        raise KnowledgeImportError(
            f"unregistered knowledge files are not allowed: {sorted(unknown_paths)}"
        )


def _load_registry(root: Path, registry_path: Path) -> tuple[SourceSpec, ...]:
    specs: list[SourceSpec] = []
    seen_paths: set[str] = set()
    seen_versions: set[tuple[str, str]] = set()
    for index, raw_source in enumerate(_registry_sources(root, registry_path)):
        path = f"registry.sources[{index}]"
        source = _mapping(raw_source, path=path)
        source_id = _string(source.get("source_id"), path=f"{path}.source_id")
        if not _SOURCE_ID_RE.fullmatch(source_id):
            raise KnowledgeImportError(f"{path}.source_id has an invalid format")
        relative_path = _string(source.get("path"), path=f"{path}.path")
        resolved_path = _resolve_inside(root, relative_path, path=f"{path}.path")
        normalized_path = resolved_path.relative_to(root).as_posix()
        if normalized_path in seen_paths:
            raise KnowledgeImportError(f"duplicate registry path: {normalized_path}")
        seen_paths.add(normalized_path)
        version = _string(source.get("version"), path=f"{path}.version")
        if (source_id, version) in seen_versions:
            raise KnowledgeImportError(f"duplicate source version: {source_id}/{version}")
        seen_versions.add((source_id, version))
        tenant_scope, allowed_roles, status = _acl_fields(source, path)
        updated_at = _string(source.get("updated_at"), path=f"{path}.updated_at")
        if not updated_at.endswith("Z"):
            raise KnowledgeImportError(f"{path}.updated_at must be UTC")
        specs.append(
            SourceSpec(
                source_id=source_id,
                path=normalized_path,
                version=version,
                tenant_scope=tenant_scope,
                allowed_roles=allowed_roles,
                status=status,
                updated_at=updated_at,
            )
        )
    _reject_unregistered_files(root, seen_paths)
    return tuple(sorted(specs, key=lambda item: (item.source_id, item.version)))


def _normalize_text(raw_text: str) -> str:
    return raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _chunk_text(text: str) -> tuple[str, ...]:
    paragraphs = tuple(part.strip() for part in re.split(r"\n\s*\n", text) if part.strip())
    chunks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= MAX_CHUNK_CHARS:
            chunks.append(paragraph)
            continue
        start = 0
        while start < len(paragraph):
            end = min(start + MAX_CHUNK_CHARS, len(paragraph))
            chunks.append(paragraph[start:end])
            if end == len(paragraph):
                break
            start = end - CHUNK_OVERLAP_CHARS
    return tuple(chunks)


def _read_source_text(root: Path, spec: SourceSpec) -> str:
    raw_bytes = (root / spec.path).read_bytes()
    if len(raw_bytes) > MAX_FILE_BYTES:
        raise KnowledgeImportError(
            f"{spec.path} exceeds the {MAX_FILE_BYTES}-byte file limit"
        )
    try:
        text = _normalize_text(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise KnowledgeImportError(f"{spec.path} is not valid UTF-8") from exc
    if not text:
        raise KnowledgeImportError(f"{spec.path} is empty")
    return text


def _chunk_records(spec: SourceSpec, text: str) -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id=f"{spec.source_id}@{spec.version}#{index:04d}",
            source_id=spec.source_id,
            source_version=spec.version,
            text=chunk_text,
            text_sha256=sha256_hex(chunk_text.encode("utf-8")),
            chunker_version=CHUNKER_VERSION,
        )
        for index, chunk_text in enumerate(_chunk_text(text))
    ]


def build_snapshot(
    source_root: str | Path,
    registry_path: str | Path,
    *,
    catalog_version: str,
    knowledge_version: str = KNOWLEDGE_VERSION,
) -> KnowledgeSnapshot:
    """Build a deterministic snapshot; no model or database call is made."""

    root = Path(source_root).resolve(strict=True)
    if not root.is_dir():
        raise KnowledgeImportError("source_root must be a directory")
    registry = Path(registry_path)
    if not registry.is_absolute():
        rooted_registry = (root / registry).resolve()
        registry = rooted_registry if rooted_registry.exists() else registry.resolve()

    source_records: list[SourceRecord] = []
    chunk_records: list[ChunkRecord] = []
    for spec in _load_registry(root, registry):
        text = _read_source_text(root, spec)
        source_records.append(
            SourceRecord(
                source_id=spec.source_id,
                path=spec.path,
                version=spec.version,
                content_sha256=sha256_hex(text.encode("utf-8")),
                tenant_scope=spec.tenant_scope,
                allowed_roles=spec.allowed_roles,
                status=spec.status,
                updated_at=spec.updated_at,
            )
        )
        if spec.status != "deleted":
            chunk_records.extend(_chunk_records(spec, text))

    if len(chunk_records) > MAX_CHUNKS:
        raise KnowledgeImportError(f"snapshot contains more than {MAX_CHUNKS} chunks")
    chunks = tuple(chunk_records)
    index_hash = sha256_hex(
        canonical_bytes({"chunker_version": CHUNKER_VERSION, "chunks": [item.as_dict() for item in chunks]})
    )
    return with_identity(
        KnowledgeSnapshot(
            snapshot_id="",
            knowledge_version=knowledge_version,
            catalog_version=catalog_version,
            chunker_version=CHUNKER_VERSION,
            embedding_model_revision=None,
            embedding_dimensions=None,
            index_hash=index_hash,
            manifest_sha256="",
            source_records=tuple(source_records),
            chunk_records=chunks,
        )
    )


import_knowledge = build_snapshot


def load_snapshot(path: str | Path) -> KnowledgeSnapshot:
    """Load and structurally validate a previously published JSON snapshot."""

    snapshot_path = Path(path)
    try:
        document = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise KnowledgeImportError(f"cannot load knowledge snapshot: {snapshot_path}") from exc
    if not isinstance(document, Mapping):
        raise KnowledgeImportError("knowledge snapshot root must be an object")
    try:
        raw_sources = document["sources"]
        raw_chunks = document["chunks"]
        if type(raw_sources) is not list or type(raw_chunks) is not list:
            raise KnowledgeImportError("knowledge snapshot sources/chunks must be lists")
        sources = tuple(
            SourceRecord(
                source_id=item["source_id"],
                path=item["path"],
                version=item["version"],
                content_sha256=item["content_sha256"],
                tenant_scope=item["tenant_scope"],
                allowed_roles=tuple(item["allowed_roles"]),
                status=item["status"],
                updated_at=item["updated_at"],
            )
            for item in raw_sources
        )
        chunks = tuple(
            ChunkRecord(
                chunk_id=item["chunk_id"],
                source_id=item["source_id"],
                source_version=item["source_version"],
                text=item["text"],
                text_sha256=item["text_sha256"],
                chunker_version=item["chunker_version"],
            )
            for item in raw_chunks
        )
        return KnowledgeSnapshot(
            snapshot_id=document["snapshot_id"],
            knowledge_version=document["knowledge_version"],
            catalog_version=document["catalog_version"],
            chunker_version=document["chunker_version"],
            embedding_model_revision=document.get("embedding_model_revision"),
            embedding_dimensions=document.get("embedding_dimensions"),
            index_hash=document["index_hash"],
            manifest_sha256=document["manifest_sha256"],
            source_records=sources,
            chunk_records=chunks,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise KnowledgeImportError("knowledge snapshot fields are invalid") from exc


def write_snapshot(snapshot: KnowledgeSnapshot, output_dir: str | Path) -> Path:
    destination = Path(output_dir) / f"{snapshot.snapshot_id}.json"
    atomic_write_text(
        destination,
        json.dumps(snapshot.as_dict(), ensure_ascii=False, indent=2) + "\n",
        temporary_prefix=".knowledge-snapshot-",
    )
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic QueryShield knowledge snapshot")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--catalog-version", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        snapshot = import_knowledge(
            args.source_root,
            args.registry,
            catalog_version=args.catalog_version,
        )
        output_path = write_snapshot(snapshot, args.snapshot_dir)
    except KnowledgeImportError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "pass",
                "snapshot_id": snapshot.snapshot_id,
                "catalog_version": snapshot.catalog_version,
                "source_count": len(snapshot.source_records),
                "active_chunk_count": len(snapshot.chunk_records),
                "manifest_sha256": snapshot.manifest_sha256,
                "output_path": str(output_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
