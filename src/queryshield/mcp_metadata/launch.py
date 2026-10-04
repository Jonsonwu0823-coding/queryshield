"""Server settings and the host-built launch of one MCP metadata server process.

Everything in a launch comes from the run's ``ExecutionContext`` and the
server's own configuration; nothing comes from model output.  The child gets
a whitelisted environment: no database URL, model key, token or state path.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from threading import Lock
from typing import Any, Callable

from queryshield.agent.proposals import ExecutionContext


METADATA_TOOLS_ENV = "QUERYSHIELD_METADATA_TOOLS"
CALL_TIMEOUT_ENV = "QUERYSHIELD_MCP_CALL_TIMEOUT_SECONDS"
METADATA_TOOLS_CONFIGURATION_ERROR = "invalid_metadata_tools_configuration"
DEFAULT_CALL_TIMEOUT_SECONDS = 2.0
CALL_TIMEOUT_RANGE = (1.0, 10.0)
# Start, initialize and tools/list; Windows starts Python and imports the SDK slowly.
START_TIMEOUT_SECONDS = 20.0
SERVER_MODULE = "queryshield.mcp_metadata.server"
EMBEDDING_ENV_NAMES = (
    "QUERYSHIELD_EMBEDDING_BASE_URL",
    "QUERYSHIELD_EMBEDDING_API_KEY",
    "QUERYSHIELD_EMBEDDING_MODEL_NAME",
    "QUERYSHIELD_EMBEDDING_MODEL_REVISION",
    "QUERYSHIELD_EMBEDDING_DIMENSIONS",
)
PACKAGE_DIR = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_DIR.parent


class MetadataToolsConfigurationError(ValueError):
    code = METADATA_TOOLS_CONFIGURATION_ERROR


def metadata_tools_setting(value: str | None = None) -> str:
    """``local`` (default) or ``mcp``; any other value is a configuration error."""

    raw = os.getenv(METADATA_TOOLS_ENV, "") if value is None else value
    setting = raw.strip().lower()
    if setting in {"", "local"}:
        return "local"
    if setting == "mcp":
        return "mcp"
    raise MetadataToolsConfigurationError("QUERYSHIELD_METADATA_TOOLS must be local or mcp")


def call_timeout_seconds(value: str | None = None) -> float:
    raw = os.getenv(CALL_TIMEOUT_ENV, "") if value is None else value
    if not raw.strip():
        return DEFAULT_CALL_TIMEOUT_SECONDS
    try:
        seconds = float(raw.strip())
    except ValueError as exc:
        raise MetadataToolsConfigurationError("the MCP call timeout must be a number of seconds") from exc
    if not math.isfinite(seconds) or not CALL_TIMEOUT_RANGE[0] <= seconds <= CALL_TIMEOUT_RANGE[1]:
        raise MetadataToolsConfigurationError("the MCP call timeout is outside 1-10 seconds")
    return seconds


@dataclass(frozen=True)
class LaunchSpec:
    """One server process: command, arguments, environment additions, working directory."""

    command: str
    args: tuple[str, ...]
    env: dict[str, str]
    cwd: str
    retrieval: str
    knowledge_snapshot_id: str | None = None


@dataclass(frozen=True)
class McpMetadataConfig:
    """What the product service decided for one run (setting already resolved)."""

    mode: str
    call_timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS
    start_timeout: float = START_TIMEOUT_SECONDS
    # Tests may start a fixture server instead; the product never sets this.
    launcher: Callable[[ExecutionContext, Any, str, str], LaunchSpec] | None = field(default=None, compare=False)


def resolve_product_config(mode: str) -> McpMetadataConfig | None:
    """The product's metadata setting: None for local, a config for MCP (raises on bad values)."""

    if metadata_tools_setting() != "mcp":
        return None
    return McpMetadataConfig(mode=mode, call_timeout=call_timeout_seconds())


# -- index files: one per embedded snapshot, in this process's own temp dir --

_INDEX_LOCK = Lock()
_INDEX_DIR: Path | None = None
_INDEX_FILES: dict[str, Path] = {}


def cleanup_index_dir() -> None:
    """Remove this process's index directory (idempotent).

    Called when the application shuts down and again at interpreter exit.  On
    Linux uvicorn re-raises SIGTERM after the shutdown phase, so ``atexit``
    alone never runs there.
    """

    global _INDEX_DIR
    with _INDEX_LOCK:
        if _INDEX_DIR is not None:
            shutil.rmtree(_INDEX_DIR, ignore_errors=True)
        _INDEX_DIR = None
        _INDEX_FILES.clear()


def published_index_path(retriever: Any) -> Path:
    """Write the run's own index once per snapshot id; the server loads it, never re-embeds."""

    from queryshield.knowledge.index import publish_index

    global _INDEX_DIR
    snapshot_id = str(retriever.snapshot.snapshot_id)
    with _INDEX_LOCK:
        path = _INDEX_FILES.get(snapshot_id)
        if path is not None and path.is_file():
            return path
        if _INDEX_DIR is None or not _INDEX_DIR.is_dir():
            _INDEX_DIR = Path(tempfile.mkdtemp(prefix="queryshield-mcp-index-"))
            atexit.register(cleanup_index_dir)
        path = publish_index(retriever.index, _INDEX_DIR / f"index-{snapshot_id}.json")
        _INDEX_FILES[snapshot_id] = path
        return path


def _knowledge_label(retriever: Any) -> str:
    from queryshield.knowledge.runtime import DEMO_KNOWLEDGE_VERSION

    return "demo" if getattr(retriever.snapshot, "knowledge_version", None) == DEMO_KNOWLEDGE_VERSION else "default"


def child_environment(*, hybrid_real: bool) -> dict[str, str]:
    """Additions to the SDK's default inherited variables; nothing else reaches the child."""

    env = {
        "PYTHONPATH": str(SOURCE_ROOT),
        "PYTHONSAFEPATH": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    if hybrid_real:
        env.update({name: os.environ[name] for name in EMBEDDING_ENV_NAMES if name in os.environ})
    return env


def product_launch(context: ExecutionContext, retriever: Any, mode: str, cwd: str) -> LaunchSpec:
    """The product server process for this run's identity and this run's retriever.

    ``retriever`` is what the local facade would search with: a hybrid
    retriever, or None for the catalog keyword search (catalog setting,
    retrieval disabled, B0).
    """

    if not isinstance(context, ExecutionContext):
        raise TypeError("a server-created execution context is required")
    if mode not in {"fake", "real"}:
        raise ValueError("mode must be fake or real")
    args = [
        "-m",
        SERVER_MODULE,
        f"--run-id={context.run_id}",
        f"--tenant-id={context.tenant_id}",
        f"--principal-id={context.principal_id}",
        f"--role={context.role}",
        f"--package-dir={PACKAGE_DIR}",
        f"--mode={mode}",
    ]
    snapshot_id = None
    if retriever is None:
        args += ["--retrieval=keyword", "--knowledge=default"]
        retrieval = "keyword"
    else:
        snapshot_id = str(retriever.snapshot.snapshot_id)
        args += [
            "--retrieval=hybrid",
            f"--knowledge={_knowledge_label(retriever)}",
            f"--index-path={published_index_path(retriever)}",
            f"--expected-snapshot-id={snapshot_id}",
        ]
        retrieval = "hybrid"
    return LaunchSpec(
        command=sys.executable,
        args=tuple(args),
        env=child_environment(hybrid_real=retriever is not None and mode == "real"),
        cwd=cwd,
        retrieval=retrieval,
        knowledge_snapshot_id=snapshot_id,
    )


__all__ = [
    "CALL_TIMEOUT_ENV",
    "DEFAULT_CALL_TIMEOUT_SECONDS",
    "LaunchSpec",
    "METADATA_TOOLS_CONFIGURATION_ERROR",
    "METADATA_TOOLS_ENV",
    "McpMetadataConfig",
    "MetadataToolsConfigurationError",
    "START_TIMEOUT_SECONDS",
    "call_timeout_seconds",
    "child_environment",
    "cleanup_index_dir",
    "metadata_tools_setting",
    "product_launch",
    "published_index_path",
    "resolve_product_config",
]
