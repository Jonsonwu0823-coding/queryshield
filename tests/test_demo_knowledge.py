"""The Chinese demo knowledge base builds, stays inside the import limits, is consistent with catalog-v4,
holds no business numbers, and isolates tenants and roles."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from queryshield.agent.proposals import ExecutionContext
from queryshield.catalog import load_default_catalog
from queryshield.knowledge import ingest
from queryshield.knowledge.runtime import (
    DEFAULT_KNOWLEDGE_ROOT,
    DEMO_CATALOG_VERSION,
    DEMO_KNOWLEDGE_ROOT,
    DEMO_KNOWLEDGE_VERSION,
    reset_retrieval_cache,
    shared_demo_retrieval_runtime,
)
from scripts import generate_demo_data as gen

REGISTRY = DEMO_KNOWLEDGE_ROOT / "source_registry.json"
QUESTIONS = json.loads(gen.QUESTIONS_PATH.read_text(encoding="utf-8"))
METRICS = {"paid_count", "gross_fen", "refund_fen", "net_fen"}
# Identifiers that may contain a digit.
ALLOWED_TOKENS = ("catalog-v4", "knowledge-demo-v1", "commerce-v1")


@pytest.fixture(scope="module")
def snapshot():
    return ingest.import_knowledge(
        DEMO_KNOWLEDGE_ROOT, REGISTRY, catalog_version=DEMO_CATALOG_VERSION, knowledge_version=DEMO_KNOWLEDGE_VERSION
    )


@pytest.fixture(scope="module")
def runtime():
    reset_retrieval_cache()
    yield shared_demo_retrieval_runtime("fake")
    reset_retrieval_cache()


def _context(identity: str) -> ExecutionContext:
    tenant, role = identity.split("-")
    return ExecutionContext(run_id="demo-test", tenant_id=tenant.upper(), principal_id=identity, role=role)


def _sources(runtime, identity: str, query: str, top_k: int = 3) -> list[str]:
    return [item["source_id"] for item in runtime.retriever.search(query, context=_context(identity), top_k=top_k).items]


def _texts() -> dict[str, str]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {item["source_id"]: (DEMO_KNOWLEDGE_ROOT / item["path"]).read_text(encoding="utf-8") for item in registry["sources"]}


def test_the_demo_snapshot_builds_with_its_own_versions_and_stays_inside_the_limits(snapshot) -> None:
    assert snapshot.knowledge_version == "knowledge-demo-v1"
    assert snapshot.snapshot_id.startswith("knowledge-demo-v1-")
    assert snapshot.catalog_version == "catalog-v4"
    assert 0 < len(snapshot.chunk_records) <= ingest.MAX_CHUNKS
    assert len(snapshot.source_records) <= ingest.MAX_FILES
    assert all(len(chunk.text) <= ingest.MAX_CHUNK_CHARS for chunk in snapshot.chunk_records)
    # Deleted sources are recorded but not indexed.
    deleted = {s.source_id for s in snapshot.source_records if s.status == "deleted"}
    assert deleted == {"demo-refund-policy-v1"}
    assert not {c.source_id for c in snapshot.chunk_records} & deleted


def test_every_markdown_file_is_registered_and_the_registry_is_valid(snapshot) -> None:
    on_disk = {p.relative_to(DEMO_KNOWLEDGE_ROOT).as_posix() for p in DEMO_KNOWLEDGE_ROOT.rglob("*.md")}
    assert on_disk == {record.path for record in snapshot.source_records}
    assert all(record.source_id.startswith("demo-") for record in snapshot.source_records)
    assert all(re.fullmatch(r"[a-z0-9][a-z0-9.-]+", record.source_id) for record in snapshot.source_records)


def test_an_unregistered_file_would_be_refused(tmp_path) -> None:
    import shutil

    copy = tmp_path / "knowledge"
    shutil.copytree(DEMO_KNOWLEDGE_ROOT, copy)
    (copy / "shared" / "stray.md").write_text("未登记的文件", encoding="utf-8")
    with pytest.raises(ingest.KnowledgeImportError):
        ingest.import_knowledge(copy, copy / "source_registry.json", catalog_version="catalog-v4")


def test_the_demo_knowledge_base_is_outside_the_frozen_fixture_directory() -> None:
    assert DEFAULT_KNOWLEDGE_ROOT not in DEMO_KNOWLEDGE_ROOT.parents
    assert DEMO_KNOWLEDGE_ROOT.parent.name == "demo"
    # The generator's description (with numbers) is next to, not inside, the knowledge root.
    assert gen.DOC_PATH.parent == DEMO_KNOWLEDGE_ROOT.parent


def test_the_knowledge_documents_contain_no_business_numbers() -> None:
    quantity = re.compile(r"[一二三四五六七八九十百千万两零]+(?:元|笔|分|家|单|次|天|小时|%|％)|[二三四五六七八九十百千万两]+个|百分之")
    for source_id, text in _texts().items():
        stripped = text
        for token in ALLOWED_TOKENS:
            stripped = stripped.replace(token, "")
        assert not re.search(r"[0-9０-９]", stripped), source_id
        assert not quantity.search(stripped), source_id


def test_documents_only_name_catalog_metrics_and_cover_each_of_them() -> None:
    catalog = load_default_catalog()
    assert catalog.catalog_version == DEMO_CATALOG_VERSION
    texts = _texts()
    named = {m for text in texts.values() for m in re.findall(r"\b([a-z]+_(?:count|fen))\b", text)}
    assert named <= METRICS | {"amount_fen"}  # amount_fen is a column, not a metric
    for metric_id, source_id in (
        ("paid_count", "demo-metric-paid-count"),
        ("gross_fen", "demo-metric-gross"),
        ("refund_fen", "demo-metric-refund"),
        ("net_fen", "demo-metric-net"),
    ):
        assert metric_id in texts[source_id]
        assert catalog.metric_name(metric_id) in texts[source_id]


def test_the_ambiguity_document_matches_the_catalog_clarification_rule() -> None:
    rule = load_default_catalog().clarification("clarify.metric_basis")
    text = _texts()["demo-ambiguity-sales"]
    assert all(phrase in text for phrase in rule.ambiguous_phrases)
    assert all(value in text for value in rule.allowed_values)


def test_the_data_dictionary_lists_the_real_columns_and_explains_commerce_v1() -> None:
    text = _texts()["demo-data-dictionary"]
    for column in ("tenant_id", "customer_id", "name", "order_id", "status", "amount_fen", "created_at", "refund_id"):
        assert column in text
    assert "commerce-v1" in text and "不是数据行" in text
    for table in ("customers", "orders", "refunds"):
        assert table in text


def test_the_refund_document_says_what_the_server_does_not_verify() -> None:
    text = _texts()["demo-metric-refund"]
    assert "不单独核实" in text and "取消" in text


def test_the_approver_document_is_about_checking_a_request_not_about_values() -> None:
    text = _texts()["demo-sensitive-customer-name"]
    assert all(point in text for point in ("SQL", "租户", "审批人"))


def test_every_expected_source_is_found_by_the_fake_embedding(runtime) -> None:
    for probe in QUESTIONS["retrieval_probes"]:
        assert probe["expected_source_id"] in _sources(runtime, probe["identity"], probe["query"]), probe["id"]
    knowledge_question = next(q for q in QUESTIONS["questions"] if q["kind"] == "knowledge")
    assert knowledge_question["expected"]["expected_source_id"] in _sources(runtime, "a-requester", knowledge_question["question"])


def test_tenant_a_cannot_retrieve_tenant_bs_document_and_the_other_way_round(runtime) -> None:
    assert "demo-tenant-b-overview" in _sources(runtime, "b-requester", "批发商，业务概况")
    assert "demo-tenant-b-overview" not in _sources(runtime, "a-requester", "批发商，业务概况", top_k=5)
    assert "demo-tenant-b-overview" not in _sources(runtime, "a-approver", "批发商，业务概况", top_k=5)
    assert "demo-tenant-a-overview" in _sources(runtime, "a-requester", "网店，业务概况")
    assert "demo-tenant-a-overview" not in _sources(runtime, "b-requester", "网店，业务概况", top_k=5)
    assert "demo-tenant-a-overview" not in _sources(runtime, "b-approver", "网店，业务概况", top_k=5)


def test_requesters_cannot_retrieve_the_approver_document_but_approvers_can(runtime) -> None:
    query = "审批人，客户姓名，核对"
    assert "demo-sensitive-customer-name" in _sources(runtime, "a-approver", query)
    assert "demo-sensitive-customer-name" in _sources(runtime, "b-approver", query)
    assert "demo-sensitive-customer-name" not in _sources(runtime, "a-requester", query, top_k=5)
    assert "demo-sensitive-customer-name" not in _sources(runtime, "b-requester", query, top_k=5)


def test_the_deleted_refund_policy_is_returned_to_nobody(runtime) -> None:
    for identity in ("a-requester", "a-approver", "b-requester", "b-approver"):
        for query in ("旧版，退款规则，整单退回", "整单退回，不支持部分退款"):
            assert "demo-refund-policy-v1" not in _sources(runtime, identity, query, top_k=5)


def test_every_isolation_probe_in_the_question_list_holds(runtime) -> None:
    for probe in QUESTIONS["isolation_probes"]:
        found = _sources(runtime, probe["identity"], probe["query"], top_k=5)
        assert not set(found) & set(probe["forbidden_source_ids"]), probe["id"]


def test_demo_source_ids_do_not_overlap_the_default_knowledge_base() -> None:
    default = json.loads((DEFAULT_KNOWLEDGE_ROOT / "source_registry.json").read_text(encoding="utf-8"))
    assert not {s["source_id"] for s in default["sources"]} & set(_texts())
