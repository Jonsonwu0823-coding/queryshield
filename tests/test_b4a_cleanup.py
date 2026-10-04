"""B4a: pre-application cleanup (O3, M2, M3, M10, and the server's exception mapping for M1)."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import tempfile

import pytest

from queryshield.agent.proposals import ExecutionContext
from queryshield.knowledge.retrieval import _source_visible, _tenant_matches
from queryshield.tools.semantic import ControlledTools, ToolError

from mcp_helpers import context, started_session

SRC = Path(__file__).resolve().parents[1] / "src" / "queryshield"


# -- O3: no tenant-less database reader in the product package ------------------


def test_commerce_module_has_no_direct_database_readers():
    import queryshield.db.commerce as commerce

    for name in ("fetch_paid_summary", "fetch_commerce_summary", "PAID_SUMMARY_SQL", "connect_readonly", "dict_row"):
        assert not hasattr(commerce, name), name
    assert "FROM paid_orders" in commerce.COMMERCE_SUMMARY_SQL  # check_commerce.py still uses the SQL constant


def test_no_product_module_imports_the_removed_readers():
    offenders = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if any(name in path.read_text(encoding="utf-8") for name in ("fetch_paid_summary", "fetch_commerce_summary"))
    ]
    assert offenders == []


# -- M2: nothing in src/ exists only for a test fixture ---------------------------


def test_the_metadata_server_has_no_result_hook():
    from queryshield.mcp_metadata import server

    for function in (server.main, server.build_server):
        assert "result_hook" not in inspect.signature(function).parameters
    assert not hasattr(server, "ResultHook")
    assert all("result_hook" not in path.read_text(encoding="utf-8") for path in (SRC / "mcp_metadata").glob("*.py"))


# -- M1: how the server's own call maps exceptions (the host side is in test_mcp_host_checks) --


class _Tools:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def describe_tables(self, arguments, *, context):
        raise self.error

    search_catalog = describe_tables


def _invoke(error: Exception):
    from queryshield.mcp_metadata.server import invoke_tool

    return invoke_tool(_Tools(error), context("A"), "search_catalog", {"query": "x"})


def test_an_embedding_failure_is_one_fixed_upstream_error_without_its_details():
    from queryshield.providers.embedding import EmbeddingProviderError

    error = EmbeddingProviderError("upstream_http_error", {"url": "https://secret.example/v1", "detail": "SECRET"})
    assert _invoke(error) == {
        "error": {"error_code": "upstream_unavailable", "message": "the upstream service behind the metadata server is unavailable"}
    }


def test_other_exceptions_and_tool_errors_keep_their_own_mapping():
    assert _invoke(RuntimeError("SECRET")) == {"error": {"error_code": "internal_error", "message": "the metadata tool failed"}}
    assert _invoke(ToolError("retrieval_unavailable", "the configured retrieval strategy is invalid")) == {
        "error": {"error_code": "retrieval_unavailable", "message": "the configured retrieval strategy is invalid"}
    }


# -- M3: one tenant rule for the retriever and for the side record -------------------


def _visibility(scope, tenant, *, role="requester"):
    ctx = ExecutionContext(run_id="run-x", tenant_id=tenant, principal_id="p", role=role)
    source = SimpleNamespace(source_id="s1", version="v1", status="active", allowed_roles=("requester", "approver"), tenant_scope=scope)
    chunk = SimpleNamespace(chunk_id="s1#0", source_id="s1", source_version="v1")
    item = {"id": "s1#0", "source_id": "s1", "version": "v1", "text": "t"}
    return ControlledTools._retrieval_item_visibility(item, context=ctx, chunk=chunk, source=source, catalog_entry=None), source, ctx


@pytest.mark.parametrize(
    "scope, tenant, visible",
    [
        ("A", "A", True),
        ("global", "A", True),
        ("global", "B", True),
        ("B", "A", False),
        ("A", "tenant-A", True),  # the retriever matches an identity "tenant-A" to scope "A"
        ("tenant-A", "A", False),  # ...and not the other way round
        ("tenant-A", "tenant-A", True),
        (None, "A", False),
    ],
)
def test_the_side_record_uses_the_retrievers_tenant_rule(scope, tenant, visible):
    record, source, ctx = _visibility(scope, tenant)
    assert record["tenant_visible"] is visible
    if isinstance(scope, str):
        assert _tenant_matches(scope, tenant) is visible
        assert _source_visible(source, ctx) is visible  # the retriever's whole visibility rule agrees
        assert record["passed"] is visible


# -- M10: the test helpers leave no directory behind -----------------------------------------


def _helper_dirs() -> set[str]:
    return {path.name for path in Path(tempfile.gettempdir()).glob("queryshield-mcp-test-*")}


def test_a_started_session_removes_its_directory_when_closed():
    before = _helper_dirs()
    session = started_session(context("A"))
    directory = Path(session.spec.cwd)
    assert directory.is_dir() and directory.name not in before
    record = session.close()
    assert record["cleanup"] == "ok"
    assert not directory.exists()
    assert _helper_dirs() <= before


def test_a_session_that_fails_to_start_removes_its_directory():
    before = _helper_dirs()
    with pytest.raises(Exception):
        started_session(context("A"), args_edit=lambda args: [a for a in args if not a.startswith("--tenant-id=")])
    assert _helper_dirs() <= before


def test_a_caller_supplied_directory_is_left_to_the_caller(tmp_path):
    session = started_session(context("A"), cwd=tmp_path)
    try:
        assert Path(session.spec.cwd) == tmp_path
    finally:
        session.close()
    assert tmp_path.is_dir()
