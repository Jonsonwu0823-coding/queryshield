"""The demo setting, its pairing with the database name, and its isolation from the default product path."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from queryshield.agent.runtime import RuntimeConfigurationError, product_retriever
from queryshield.api.main import app, get_retriever_source
from queryshield.approval.service import reset_shared_state_stores, shared_run_service
from queryshield.db import readonly
from queryshield.db.readonly import (
    DatabaseConfigurationError,
    check_demo_pairing,
    database_names_from_url,
    demo_dataset_enabled,
    get_database_url,
)
from queryshield.knowledge.runtime import (
    CATALOG_SEARCH_ONLY,
    DEFAULT_KNOWLEDGE_ROOT,
    DEMO_KNOWLEDGE_ROOT,
    _cache_key,
    build_retrieval_runtime,
    reset_retrieval_cache,
)

TEST_URL = "postgresql://queryshield_ro:s3cret-pw@127.0.0.1:5433/queryshield_test"
DEMO_URL = "postgresql://queryshield_ro:s3cret-pw@127.0.0.1:5433/queryshield_demo"

# Baseline of the default knowledge snapshot on main 8ed932b.  These three do not depend on the
# Python version.  The snapshot id after the Fake embedding index is built does (floating point
# in the hashed-feature vectors: 97d2061e0b7c34b6 on the project venv, Python 3.14.3), so the tests
# compare it with the legacy construction instead of a literal.
BASELINE_INGEST_SNAPSHOT_ID = "knowledge-v1-9f580dd7f887ed0a"
BASELINE_INGEST_INDEX_HASH = "a897e0d46be5855a3dce7b6c70100556ac5a6ae95965519d6e5e546b4487c4aa"
BASELINE_REGISTRY_SHA256 = "e2743863f7eada2431fce32567e968cbf673cbbfd2f2c53fad69fa4b7d855460"


def legacy_default_snapshot():
    """The default Fake snapshot built exactly as before the demo setting (no version arguments)."""

    from queryshield.knowledge.index import build_embedding_index
    from queryshield.knowledge.ingest import import_knowledge
    from queryshield.knowledge.runtime import KNOWLEDGE_CATALOG_VERSION, HashFeatureEmbedding

    snapshot = import_knowledge(
        DEFAULT_KNOWLEDGE_ROOT, DEFAULT_KNOWLEDGE_ROOT / "source_registry.json", catalog_version=KNOWLEDGE_CATALOG_VERSION
    )
    assert (snapshot.snapshot_id, snapshot.index_hash) == (BASELINE_INGEST_SNAPSHOT_ID, BASELINE_INGEST_INDEX_HASH)
    return build_embedding_index(snapshot, HashFeatureEmbedding(), ingest_job_id="w05-retrieval-fake").snapshot


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ("QUERYSHIELD_DEMO_DATASET", "QUERYSHIELD_RETRIEVAL", "PGDATABASE"):
        monkeypatch.delenv(name, raising=False)
    reset_retrieval_cache()
    yield
    reset_retrieval_cache()


def _refusal(exc: BaseException) -> str:
    text = str(exc)
    # Never the URL, the password or the host.
    assert "s3cret-pw" not in text and "postgresql://" not in text and "127.0.0.1" not in text
    return getattr(exc, "code", "")


# --- database names -------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "names"),
    [
        (TEST_URL, ("queryshield_test",)),
        (DEMO_URL, ("queryshield_demo",)),
        ("postgresql://u:p@h/queryshield%5Fdemo?sslmode=disable", ("queryshield_demo",)),
        ("postgresql://u:p@h:5433/queryshield_demo?connect_timeout=5&application_name=x", ("queryshield_demo",)),
        ("postgresql://h/?dbname=other_demo", ("other_demo",)),
        ("host=h user=u dbname=kw_demo", ("kw_demo",)),
        ("host=h user=u dbname='quoted_demo'", ("quoted_demo",)),
        ("", ()),
        (None, ()),
    ],
)
def test_database_name_is_read_from_the_path_query_or_keyword_form(url, names) -> None:
    assert database_names_from_url(url) == names


def test_pgdatabase_is_the_name_only_when_the_url_has_none(monkeypatch) -> None:
    monkeypatch.setenv("PGDATABASE", "from_env_demo")
    assert database_names_from_url("postgresql://u:p@h") == ("from_env_demo",)
    assert database_names_from_url(TEST_URL) == ("queryshield_test",)


# --- pairing ---------------------------------------------------------------------


def test_setting_off_and_a_test_database_changes_nothing(monkeypatch) -> None:
    assert demo_dataset_enabled() is False
    check_demo_pairing(TEST_URL)
    check_demo_pairing(None)  # fixture-database runs have no URL at all
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", TEST_URL)
    assert get_database_url() == TEST_URL


@pytest.mark.parametrize("url", [TEST_URL, "postgresql://u:s3cret-pw@h/prod", None, "", "postgresql://u:s3cret-pw@h"])
def test_setting_on_refuses_anything_but_a_demo_database(monkeypatch, url) -> None:
    monkeypatch.setenv("QUERYSHIELD_DEMO_DATASET", "commerce-demo-v1")
    with pytest.raises(DatabaseConfigurationError) as caught:
        check_demo_pairing(url)
    assert _refusal(caught.value) == "demo_dataset_database_mismatch"


def test_setting_on_with_a_demo_database_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_DEMO_DATASET", "commerce-demo-v1")
    check_demo_pairing(DEMO_URL)
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", DEMO_URL)
    assert get_database_url() == DEMO_URL


def test_setting_on_refuses_when_the_query_string_names_another_database(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_DEMO_DATASET", "commerce-demo-v1")
    with pytest.raises(DatabaseConfigurationError) as caught:
        check_demo_pairing("postgresql://u:s3cret-pw@h/queryshield_demo?dbname=queryshield_test")
    assert _refusal(caught.value) == "demo_dataset_database_mismatch"


@pytest.mark.parametrize(
    "url",
    [DEMO_URL, "postgresql://u:s3cret-pw@h/queryshield%5Fdemo", "postgresql://u:s3cret-pw@h/?dbname=x_demo", "host=h dbname=x_demo"],
)
def test_a_demo_database_without_the_setting_is_refused(monkeypatch, url) -> None:
    with pytest.raises(DatabaseConfigurationError) as caught:
        check_demo_pairing(url)
    assert _refusal(caught.value) == "demo_database_without_demo_dataset"
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", url)
    with pytest.raises(DatabaseConfigurationError):
        get_database_url()


def test_get_database_url_refuses_before_any_connection_is_made(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", DEMO_URL)

    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("connected despite a refused pairing")

    monkeypatch.setattr(readonly.psycopg, "connect", explode)
    with pytest.raises(DatabaseConfigurationError):
        readonly.connect_readonly()


def test_an_unknown_setting_value_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_DEMO_DATASET", "yes")
    with pytest.raises(DatabaseConfigurationError) as caught:
        demo_dataset_enabled()
    assert getattr(caught.value, "code", "") == "invalid_demo_dataset"


# --- the default product path is untouched ------------------------------------------


def test_default_snapshot_index_hash_and_cache_key_are_the_baseline_ones() -> None:
    runtime = build_retrieval_runtime("fake")
    snapshot = runtime.snapshot
    legacy = legacy_default_snapshot()
    assert snapshot.snapshot_id == legacy.snapshot_id
    assert snapshot.index_hash == legacy.index_hash
    assert snapshot.manifest_sha256 == legacy.manifest_sha256
    assert snapshot.knowledge_version == "knowledge-v1"
    assert snapshot.catalog_version == "catalog-v2"
    assert (len(snapshot.chunk_records), len(snapshot.source_records)) == (27, 14)
    assert _cache_key("fake", DEFAULT_KNOWLEDGE_ROOT) == (
        "fake",
        str(DEFAULT_KNOWLEDGE_ROOT),
        BASELINE_REGISTRY_SHA256,
        "w05-hash-feature-fake-v1",
    )
    assert runtime.index_build.index.snapshot_id == legacy.snapshot_id


def test_product_retriever_with_the_setting_off_is_the_default_retriever(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", TEST_URL)
    legacy = legacy_default_snapshot()
    retriever = product_retriever("fake")
    assert retriever.snapshot.snapshot_id == legacy.snapshot_id
    monkeypatch.delenv("QUERYSHIELD_DATABASE_URL")
    assert product_retriever("fake").snapshot.snapshot_id == legacy.snapshot_id  # fixture-database runs


def test_catalog_and_disabled_retrieval_settings_still_work_with_the_setting_off(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", TEST_URL)
    monkeypatch.setenv("QUERYSHIELD_RETRIEVAL", "catalog")
    assert product_retriever("fake") is CATALOG_SEARCH_ONLY
    monkeypatch.setenv("QUERYSHIELD_RETRIEVAL", "disabled")
    assert product_retriever("fake") is None


# --- the demo setting selects the demo knowledge base --------------------------------


def test_setting_on_selects_the_demo_knowledge_base(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_DEMO_DATASET", "commerce-demo-v1")
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", DEMO_URL)
    snapshot = product_retriever("fake").snapshot
    assert snapshot.snapshot_id.startswith("knowledge-demo-v1-")
    assert snapshot.knowledge_version == "knowledge-demo-v1"
    assert snapshot.catalog_version == "catalog-v4"
    assert snapshot.snapshot_id != legacy_default_snapshot().snapshot_id
    assert {record.source_id for record in snapshot.source_records}.isdisjoint(
        {"semantic-metric-net", "tenant-a-orders-overview"}
    )
    assert _cache_key("fake", DEMO_KNOWLEDGE_ROOT) != _cache_key("fake", DEFAULT_KNOWLEDGE_ROOT)
    # The same process can still build the default one, unchanged.
    assert build_retrieval_runtime("fake").snapshot.snapshot_id == legacy_default_snapshot().snapshot_id


def test_setting_on_with_a_test_database_is_refused_by_the_product_retriever(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_DEMO_DATASET", "commerce-demo-v1")
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", TEST_URL)
    with pytest.raises(RuntimeConfigurationError) as caught:
        product_retriever("fake")
    assert _refusal(caught.value) == "demo_dataset_database_mismatch"
    monkeypatch.delenv("QUERYSHIELD_DATABASE_URL")
    with pytest.raises(RuntimeConfigurationError) as caught:
        product_retriever("fake")
    assert _refusal(caught.value) == "demo_dataset_database_mismatch"


def test_a_demo_database_without_the_setting_is_refused_by_the_product_retriever(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", DEMO_URL)
    with pytest.raises(RuntimeConfigurationError) as caught:
        product_retriever("fake")
    assert _refusal(caught.value) == "demo_database_without_demo_dataset"


# --- HTTP and the evaluation entry points ----------------------------------------------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYSHIELD_STATE_STORE_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.setenv("QUERYSHIELD_FAKE_DB", "1")
    monkeypatch.setenv("QUERYSHIELD_PROVIDER_MODE", "fake")
    monkeypatch.delenv("QUERYSHIELD_AGENT_PROFILE", raising=False)
    monkeypatch.setenv("QUERYSHIELD_TOKEN_A_REQUESTER", "demo-a-requester")
    reset_shared_state_stores()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    reset_shared_state_stores()


def _ask(client: TestClient):
    return client.post(
        "/queries",
        headers={"Authorization": "Bearer demo-a-requester"},
        json={"question": "2026年9月已支付订单总额"},
    )


def test_http_refuses_the_setting_on_a_test_database_with_a_fixed_code(client, monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_DEMO_DATASET", "commerce-demo-v1")
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", TEST_URL)
    response = _ask(client)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "demo_dataset_database_mismatch"
    assert "s3cret-pw" not in response.text and "queryshield_test" not in response.text


def test_http_refuses_a_demo_database_without_the_setting(client, monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", DEMO_URL)
    response = _ask(client)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "demo_database_without_demo_dataset"
    assert "s3cret-pw" not in response.text and "queryshield_demo" not in response.text


def test_http_with_the_setting_off_and_a_test_database_still_answers(client, monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", TEST_URL)
    response = _ask(client)
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCEEDED"


def test_the_evaluation_start_async_path_is_refused_when_the_setting_is_on(client, monkeypatch) -> None:
    """check_state.py and check_eval.py call start_async without dependencies: it reaches product_retriever."""

    monkeypatch.setenv("QUERYSHIELD_DEMO_DATASET", "commerce-demo-v1")
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", TEST_URL)
    service = shared_run_service()
    identity = {"tenant_id": "A", "principal_id": "a-requester", "role": "requester"}
    with pytest.raises(RuntimeConfigurationError) as caught:
        service.start_async(identity=identity, question="2026年9月已支付订单总额")
    assert caught.value.code == "demo_dataset_database_mismatch"
    with pytest.raises(RuntimeConfigurationError) as caught:
        service.run_sync(identity=identity, question="2026年9月已支付订单总额")
    assert caught.value.code == "demo_dataset_database_mismatch"


def test_the_stateful_evaluation_override_never_reaches_the_product_retriever(client, monkeypatch, tmp_path) -> None:
    """evaluation/stateful_product.py injects its own run service and replaces
    get_retriever_source with catalog-only search.

    The evaluation's own service (like stateful_product.py) is not the
    product service, so it neither publishes knowledge nor reads the demo
    setting.  The product service itself checks the demo setting before a run,
    like product_retriever, and refuses the mismatch with a fixed code.
    """

    from queryshield.api.main import get_run_service
    from queryshield.approval.service import RunService
    from queryshield.db.state_store import StateStore

    monkeypatch.setenv("QUERYSHIELD_DEMO_DATASET", "commerce-demo-v1")
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", TEST_URL)
    app.dependency_overrides[get_retriever_source] = lambda: (lambda: CATALOG_SEARCH_ONLY)
    refused = _ask(client)
    assert refused.status_code == 503 and refused.json()["error"]["code"] == "demo_dataset_database_mismatch"
    store = StateStore(tmp_path / "evaluation.sqlite3")
    app.dependency_overrides[get_run_service] = lambda: RunService(store=store, mode="fake")
    response = _ask(client)
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCEEDED"
    assert response.json()["facts"]["facts"][0]["catalog_version"] == "catalog-v4"
    assert store.get_snapshot() is None
    store.close()


def test_direct_default_retrieval_builds_used_by_check_eval_ignore_the_setting(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_DEMO_DATASET", "commerce-demo-v1")
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", TEST_URL)
    assert build_retrieval_runtime("fake").snapshot.snapshot_id == legacy_default_snapshot().snapshot_id
