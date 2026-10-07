"""The Real adapters pointed at the fake upstream behave exactly like the in-process Fake.

Both sides run the same agent assembly over the same fixture executor and the product
knowledge base; the only change is the model and the embedding: ``FakeModel`` and
``HashFeatureEmbedding`` in process, or ``OpenAICompatibleModel`` and
``OpenAICompatibleEmbedding`` sending HTTP to the fake upstream (FastAPI's TestClient
as the httpx client, so no port is opened).  Every step's decision, every event and
the result must be equal once the listed provider fields are set aside; those fields
are checked on their own.
"""

from __future__ import annotations

from dataclasses import replace
import json
import re

from fastapi.testclient import TestClient
import pytest

from agent_core_scenarios import FIXED_TIME, StepClock, canonical
import test_clarification as clarification_cases
from queryshield.agent import BoundedAgent, GraphLimits, ModelCallStore
from queryshield.agent.config import NATIVE_VERSIONS, RunConfig
from queryshield.agent.proposals import ExecutionContext
from queryshield.agent.runtime import B1_MODEL_CALL_LIMIT, B1_PROFILE, B1_TOOL_CALL_LIMIT, B1_WALL_CLOCK_SECONDS
from queryshield.catalog import load_default_catalog
from queryshield.knowledge.index import build_embedding_index
from queryshield.knowledge.retrieval import HybridRetriever
from queryshield.knowledge.runtime import HashFeatureEmbedding, product_knowledge
from queryshield.providers.contracts import usage_is_consistent
from queryshield.providers.embedding import EmbeddingConfig, OpenAICompatibleEmbedding
from queryshield.providers.fake_model import FakeModel
from queryshield.providers.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleModel
from queryshield.tools.semantic import ControlledTools
from scripts.fake_upstream import FAKE_UPSTREAM_MODEL, app

BASE_URL = "http://fake-upstream/v1"
PLACEHOLDER_KEY = "not-a-real-key"
SEPTEMBER = clarification_cases.SEPTEMBER
# (name, question, request window, resume answer)
PATHS = [
    ("verified_answer", "2026年9月已支付订单总额是多少？", SEPTEMBER, None),
    ("two_metrics", "2026年9月已支付订单数和支付金额", SEPTEMBER, None),
    ("net_plan", "2026年9月退款后净额", SEPTEMBER, None),
    ("ambiguous_metric_then_resume", "2026年9月销售额是多少？", SEPTEMBER, "按支付金额统计"),
    ("missing_time_then_resume", "支付金额是多少？", None, "2026年9月"),
    ("no_data", "你好，你能做什么？", None, None),
    ("knowledge", "退款后净额是怎么算的？", None, None),
    ("sensitive_field_waits_for_approval", "查询本租户所有客户的姓名", None, None),
]
PROTOCOLS = ["json", "native"]

# Allowed differences D2-D6 (a model_call event has no mode field, so D1 does not appear here).
CALL_FIELDS = ("provider", "model", "provider_call_id", "provider_request_id", "usage_status", "usage")
USAGE_FIELDS = (
    "status", "known_call_count", "unknown_call_count",
    "known_prompt_tokens", "known_completion_tokens", "known_total_tokens",
    "prompt_tokens", "completion_tokens", "total_tokens",
)
# D7: the embedded snapshot id (the embedding model name and revision are in the index hash).
SNAPSHOT_FIELDS = ("knowledge_snapshot_id",)
# The adapter passes its timeout to the injected client; TestClient ignores it and warns.
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")
_UUID_OR_HEX = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|result-[0-9a-f]{32}")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


class _Capturing:
    """Keeps the text of every decision the wrapped model returned."""

    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.mode = delegate.mode
        self.decisions: list[object] = []

    def complete(self, messages, **options):
        result = self.delegate.complete(messages, **options)
        calls = [(call.name, call.arguments) for call in result.tool_calls] if result.tool_calls is not None else None
        self.decisions.append({"content": result.content, "tool_calls": calls, "finish_reason": result.finish_reason})
        return result


def _upstream_model(client: TestClient) -> OpenAICompatibleModel:
    config = OpenAICompatibleConfig(base_url=BASE_URL, api_key=PLACEHOLDER_KEY, model=FAKE_UPSTREAM_MODEL)
    return OpenAICompatibleModel(config, client=client)


def _upstream_embedding(client: TestClient) -> OpenAICompatibleEmbedding:
    config = EmbeddingConfig(
        base_url=BASE_URL, api_key=PLACEHOLDER_KEY, model=FAKE_UPSTREAM_MODEL, model_revision=FAKE_UPSTREAM_MODEL, dimensions=128
    )
    return OpenAICompatibleEmbedding(config, client=client)


def _retriever(embedder, *, demo: bool = False) -> HybridRetriever:
    """The product hybrid retriever (``build_retrieval_runtime``), with the embedder passed in."""

    build = build_embedding_index(product_knowledge(demo=demo).snapshot, embedder, ingest_job_id="equivalence")
    return HybridRetriever(catalog=load_default_catalog(), snapshot=build.snapshot, index=build.index, embedder=embedder)




@pytest.fixture(scope="module", params=[False, True], ids=["default_knowledge", "demo_knowledge"])
def retrievers(request, client) -> dict[str, HybridRetriever]:
    demo = request.param
    return {"fake": _retriever(HashFeatureEmbedding(), demo=demo), "upstream": _retriever(_upstream_embedding(client), demo=demo)}


def _run(model, retriever: HybridRetriever, protocol: str, question: str, window, resume_answer) -> dict[str, object]:
    catalog = load_default_catalog()
    config = RunConfig(profile=B1_PROFILE, catalog_version=catalog.catalog_version, knowledge_snapshot_id=retriever.snapshot.snapshot_id)
    if protocol == "native":
        config = replace(config, **NATIVE_VERSIONS)
    executor = clarification_cases._Recording(clock=lambda: FIXED_TIME)
    capturing = _Capturing(model)
    agent = BoundedAgent(
        capturing,
        tools=ControlledTools(catalog=catalog, executor=executor, retriever=retriever),
        call_store=ModelCallStore(),
        limits=GraphLimits(max_model_calls=B1_MODEL_CALL_LIMIT, max_tool_calls=B1_TOOL_CALL_LIMIT, max_wall_clock_seconds=B1_WALL_CLOCK_SECONDS),
        clock=StepClock(),
        run_config=config,
    )
    context = ExecutionContext(run_id="run-equivalence", tenant_id="A", principal_id="principal-A", role="requester")
    record: dict[str, object] = {"result": agent.run(context, question, request_time_window=window).as_dict()}
    if resume_answer is not None:
        assert record["result"]["status"] == "waiting_user"
        record["checkpoint"] = agent.export_waiting_checkpoint(context.run_id)
        record["resumed"] = agent.resume(context, resume_answer).as_dict()
    record["executed_sql"] = list(executor.executed)
    record["decisions"] = capturing.decisions
    return record


def _set_aside(value: object, key: str | None = None) -> object:
    """The record with only the allowed-difference fields replaced by a marker."""

    if isinstance(value, dict):
        out = {}
        for name, item in value.items():
            allowed = (
                (value.get("kind") == "model_call" and name in CALL_FIELDS)
                or (key == "usage_summary" and name in USAGE_FIELDS)
                or name in SNAPSHOT_FIELDS
            )
            out[name] = "<allowed difference>" if allowed else _set_aside(item, name)
        return out
    if isinstance(value, (list, tuple)):
        return [_set_aside(item, key) for item in value]
    return value


def _comparable(record: dict[str, object]) -> str:
    # canonical() numbers random ids; result ids inside model-written text are numbered here too.
    text = canonical(_set_aside(record))
    return _UUID_OR_HEX.sub(lambda match: "<id>", text)


def _search(retriever: HybridRetriever, query: str, context: ExecutionContext) -> dict[str, object]:
    result = retriever.search(query, context=context, top_k=5)
    return {"items": [dict(item) for item in result.items], "evidence": result.evidence.as_dict()}


def _model_call_events(record: dict[str, object]) -> list[dict]:
    results = [record["result"]] + ([record["resumed"]] if "resumed" in record else [])
    return [event for result in results for event in result["events"] if event.get("kind") == "model_call"]


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize(("name", "question", "window", "resume_answer"), PATHS, ids=[path[0] for path in PATHS])
def test_each_step_and_the_result_match_the_in_process_fake(client, retrievers, protocol, name, question, window, resume_answer) -> None:
    fake = _run(FakeModel(), retrievers["fake"], protocol, question, window, resume_answer)
    upstream = _run(_upstream_model(client), retrievers["upstream"], protocol, question, window, resume_answer)

    assert _comparable(upstream) == _comparable(fake)
    assert fake["decisions"], "the path called the model"

    # The set-aside fields hold exactly the documented values.
    for event in _model_call_events(fake):
        assert (event["provider"], event["model"], event["usage_status"], event["usage"]) == ("fake", "fake-model", "unknown", None)
        assert event["provider_call_id"] is None and event["provider_request_id"] is None
    upstream_events = _model_call_events(upstream)
    assert len(upstream_events) == len(_model_call_events(fake))
    for event in upstream_events:
        assert (event["provider"], event["model"], event["usage_status"]) == ("openai_compatible", FAKE_UPSTREAM_MODEL, "known")
        assert re.fullmatch(r"fake-chatcmpl-[0-9a-f]{32}", event["provider_call_id"])
        assert re.fullmatch(r"fake-req-[0-9a-f]{32}", event["provider_request_id"])
        usage = event["usage"]
        assert usage_is_consistent(usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) and usage["completion_tokens"] > 0
    last = upstream.get("resumed") or upstream["result"]
    assert last["usage_summary"]["status"] == "known"
    assert fake["result"]["run_config"]["knowledge_snapshot_id"] == retrievers["fake"].snapshot.snapshot_id
    assert upstream["result"]["run_config"]["knowledge_snapshot_id"] == retrievers["upstream"].snapshot.snapshot_id


def test_the_paths_reach_the_outcomes_they_are_named_for(client, retrievers) -> None:
    outcomes = {}
    for name, question, window, resume_answer in PATHS:
        record = _run(_upstream_model(client), retrievers["upstream"], "json", question, window, resume_answer)
        last = record.get("resumed") or record["result"]
        outcomes[name] = (last["status"], last["answer_status"])
    assert outcomes["verified_answer"] == ("succeeded", "verified")
    assert outcomes["ambiguous_metric_then_resume"] == ("succeeded", "verified")
    assert outcomes["missing_time_then_resume"] == ("succeeded", "verified")
    assert outcomes["no_data"][0] == "succeeded"
    assert outcomes["knowledge"][0] == "succeeded"
    assert outcomes["sensitive_field_waits_for_approval"][0] == "waiting_approval"


# Search evidence fields set aside: new per search or wall clock (they differ between two
# in-process searches too), and the embedding provider fields of D4, D5 and D7.
SEARCH_SET_ASIDE = (
    "retrieval_id", "embedding_call_id", "elapsed_ms",
    "snapshot_id", "embedding_provider_call_id", "embedding_provider_request_id",
)
SEARCH_QUERIES = ("退款后净额", "支付金额", "已支付订单数", "客户姓名", "refund net amount", "paid order count")


def _search_comparable(search: dict[str, object]) -> dict[str, object]:
    evidence = {key: value for key, value in search["evidence"].items() if key not in SEARCH_SET_ASIDE}
    evidence["embedding_actual_return"] = {key: value for key, value in evidence["embedding_actual_return"].items() if key != "usage"}
    return {"items": search["items"], "evidence": evidence}


@pytest.mark.parametrize("demo", [False, True], ids=["default_knowledge", "demo_knowledge"])
def test_the_index_and_every_search_match_the_in_process_ones(client, demo) -> None:
    fake, upstream = _retriever(HashFeatureEmbedding(), demo=demo), _retriever(_upstream_embedding(client), demo=demo)
    assert [chunk.vector for chunk in upstream.index.chunks] == [chunk.vector for chunk in fake.index.chunks]
    assert upstream.snapshot.snapshot_id != fake.snapshot.snapshot_id  # D7: the embedding names are in the index hash
    context = ExecutionContext(run_id="run-search", tenant_id="A", principal_id="principal-A", role="requester")
    vector_hits = 0
    for query in SEARCH_QUERIES:
        expected, actual = _search(fake, query, context), _search(upstream, query, context)
        assert _search_comparable(actual) == _search_comparable(expected), query
        assert actual["evidence"]["embedding_actual_return"]["usage"]["model"] == FAKE_UPSTREAM_MODEL
        vector_hits += len(expected["evidence"]["vector_candidate_ids"])
    if not demo:
        assert vector_hits > 0, "the vector route took part, so equal rankings are not vacuous"
