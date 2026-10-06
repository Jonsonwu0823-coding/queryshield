"""One product runtime shared by HTTP entrypoints and the evaluation."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re

import pytest

from queryshield.agent.context import build_context
from queryshield.agent.graph import BoundedAgent
from queryshield.agent.proposals import ExecutionContext
from queryshield.agent.runtime import (
    B1_PROFILE,
    RUN_OUTCOMES,
    b1_result_payload,
    build_b1_agent,
    outcome_for,
)
from queryshield.approval.service import FixtureQueryExecutor
from queryshield.catalog import load_default_catalog
from queryshield.evaluation.state_cases import load_state_cases
from queryshield.evaluation.stateful_product import StateCaseFakeModel
from queryshield.evaluation.profile_runner import normalize_profile_observation, run_b1_bounded_agent as evaluation_run_b1
from queryshield.knowledge.runtime import (
    FAKE_EMBEDDING_MODEL,
    HashFeatureEmbedding,
    feature_vector,
    shared_retrieval_runtime,
)
from queryshield.providers.contracts import ModelCallResult
from queryshield.tools.semantic import ControlledTools


_CONTEXT = ExecutionContext(run_id="run-golden", tenant_id="A", principal_id="principal-A", role="requester")
_WINDOW = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z", "timezone": "UTC"}
# sha256 of the rendered messages, context-v16.  Every deliberate change to the
# context text re-pins it: baseline (37c91f2, context-v12); context-v13 (catalog-v3
# phrase table, ask_user clarification_id, non-metric example); context-v14
# (catalog-v4, the time-range ask rule, clarification_id wording, five duplicate
# instructions removed); context-v15 (final_answer basis rule, the no-query sentence
# of fact_refs_rule, prompt v25 and action schema v4); the no_data example, two
# shortened basis clauses, prompt v26 and action schema v5; and last only the prompt
# version string (v27; the contract text is unchanged).
_BASELINE_CONTEXT_SHA256 = {
    (False, False): "0e13a5705db5b069bb30fd0528487f3d3550b1100878e4f74e0ece80fbae221e",
    (False, True): "284a21eac7729f5b8d065559049d39cc7b4b85d29f7d3794b94c66fe4f1556d7",
    (True, False): "10737eaa4a4d28cdd080edb7211738f371fad8be0b3f2b641bba38df50fbdc96",
    (True, True): "dee973f59f9e5dd3a84c7c9d43587a400757768af4ba6a2612b844f3b1a320d5",
}


def _messages_sha256(result) -> str:
    data = json.dumps([dict(message) for message in result.messages], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(("has_window", "parallel"), sorted(_BASELINE_CONTEXT_SHA256))
@pytest.mark.parametrize("explicit", [False, True])
def test_default_context_rendering_is_byte_identical_to_the_pinned_baseline(has_window, parallel, explicit) -> None:
    kwargs = {"retrieval_available": True} if explicit else {}
    result = build_context(
        _CONTEXT,
        "2026年9月已支付订单总额",
        request_time_window=_WINDOW if has_window else None,
        parallel_available=parallel,
        **kwargs,
    )
    assert result.context_version == "context-v16"
    assert _messages_sha256(result) == _BASELINE_CONTEXT_SHA256[(has_window, parallel)]


@pytest.mark.parametrize("has_window", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_context_without_a_retriever_never_mentions_search_catalog(has_window, parallel) -> None:
    result = build_context(
        _CONTEXT,
        "2026年9月已支付订单总额",
        request_time_window=_WINDOW if has_window else None,
        parallel_available=parallel,
        retrieval_available=False,
    )
    text = json.dumps([dict(message) for message in result.messages], ensure_ascii=False)
    assert "search_catalog" not in text
    assert "describe_tables and query_readonly are tool names" in text
    assert result.context_version == "context-v16"


class _Scripted:
    mode = "fake"
    provider = "scripted"
    model = "scripted-v1"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.messages = []

    def complete(self, messages, *, request_id=None, model_call_id=None):
        self.messages.append([dict(message) for message in messages])
        return ModelCallResult(
            mode="fake",
            provider=self.provider,
            model=self.model,
            request_id=request_id,
            model_call_id=model_call_id,
            provider_call_id=None,
            provider_request_id=None,
            content=self.outputs[len(self.messages) - 1],
            usage=None,
            usage_status="unknown",
        )


def test_agent_without_retriever_refuses_search_catalog_and_does_not_describe_it() -> None:
    model = _Scripted(['{"type":"tool_call","name":"search_catalog","arguments":{"query":"净额"}}'])
    agent = BoundedAgent(model, tools=ControlledTools(executor=FixtureQueryExecutor()), retrieval_available=False)
    result = agent.run(_CONTEXT, "2026年9月退款后净额")
    assert result.status == "failed"
    assert result.error_code == "retrieval_unavailable"
    assert "search_catalog" not in json.dumps(model.messages[0], ensure_ascii=False)


@pytest.mark.parametrize(
    ("status", "error_code", "expected"),
    [
        ("succeeded", None, ("SUCCEEDED", "succeeded", "SUCCEEDED", 200)),
        ("waiting_user", None, ("WAITING_USER", "waiting_user", "WAITING_USER", 202)),
        ("waiting_approval", None, ("WAITING_APPROVAL", "waiting_approval", "WAITING_APPROVAL", 202)),
        ("denied", "forbidden", ("DENIED", "denied", "DENIED", 403)),
        ("failed", "invalid_sql", ("FAILED", "failed", "FAILED", 502)),
        ("failed", "database_unavailable", ("FAILED", "failed", "FAILED", 503)),
        ("failed", "missing_model_configuration", ("FAILED", "failed", "FAILED", 503)),
        ("failed", "query_timeout", ("FAILED", "failed", "FAILED", 504)),
        ("failed", "upstream_timeout", ("FAILED", "failed", "FAILED", 504)),
        ("failed", "result_row_limit", ("FAILED", "failed", "FAILED", 422)),
        ("limit_reached", "model_call_limit", ("LIMIT_REACHED", "unknown", "UNKNOWN", 502)),
        ("timeout", "upstream_timeout", ("FAILED", "unknown", "UNKNOWN", 504)),
        ("mystery", None, ("FAILED", "unknown", "UNKNOWN", 502)),
    ],
)
def test_shared_outcome_table(status, error_code, expected) -> None:
    outcome = outcome_for(status, error_code)
    assert (outcome.run_status, outcome.public_status, outcome.terminal_state, outcome.http_status) == expected


def test_outcome_table_matches_frozen_eval_http_expectations() -> None:
    frozen = {}
    for case in load_state_cases():
        expected = case.case["expected"]
        if case.case["action"]["entrypoint"] == "/queries":
            frozen.setdefault(expected["terminal_state"], set()).add(expected["http_status"])
    for terminal, codes in frozen.items():
        status = next(name for name, outcome in RUN_OUTCOMES.items() if outcome.terminal_state == terminal)
        assert codes == {RUN_OUTCOMES[status].http_status}, terminal


def test_eval_normalizer_uses_the_shared_table() -> None:
    for status in ("succeeded", "waiting_user", "waiting_approval", "denied", "failed", "limit_reached"):
        observation = normalize_profile_observation(
            B1_PROFILE,
            {"status": status, "error_code": None, "events": [], "model_call_count": 0},
            case_id="table",
        )
        outcome = outcome_for(status)
        assert observation["http_status"] == outcome.http_status
        assert observation["terminal_state"] == outcome.terminal_state
        assert observation["status"] == outcome.public_status


_VOLATILE_KEYS = {
    "run_id", "model_call_id", "model_call_ids", "request_id", "result_id", "fact_id", "fact_ids",
    "elapsed_ms", "observed_at", "retrieval_id", "content_sha256", "query_sha256", "evidence_sha256",
}


def _stable(value, key=""):
    if isinstance(value, Mapping):
        return {k: _stable(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_stable(v, key) for v in value]
    if key in _VOLATILE_KEYS or key.endswith("sha256"):
        return "<volatile>"
    if isinstance(value, str):
        return re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<uuid>", value)
    return value


def product_run_b1(model, tools, context, question):
    """One B1 run through the product's own assembly and result shaping."""

    return b1_result_payload(build_b1_agent(model, tools).run(context, question), context, question)


@pytest.mark.parametrize("case_id", ["gross-total-fen", "paid-order-count", "join-aggregate-by-customer"])
def test_evaluation_wrapper_and_product_runtime_are_equivalent_without_initial_items(case_id) -> None:
    case = next(item for item in load_state_cases() if item.case_id == case_id)
    question = str(case.case["action"]["parameters"]["question"])
    retriever = shared_retrieval_runtime("fake").retriever
    catalog = load_default_catalog()
    results = []
    for runner in (evaluation_run_b1, product_run_b1):
        tools = ControlledTools(catalog=catalog, executor=FixtureQueryExecutor(), retriever=retriever)
        context = ExecutionContext(run_id=f"run-{case_id}", tenant_id="A", principal_id="principal-A", role="requester")
        results.append(runner(StateCaseFakeModel(case), tools, context, question))
    assert results[0]["status"] == "succeeded"
    assert _stable(results[0]) == _stable(results[1])


def test_fake_embedding_handles_any_text_with_the_frozen_feature_vector() -> None:
    embedder = HashFeatureEmbedding()
    text = "一个从未登记过的问题：上个季度的复购客户有多少"
    call = embedder.embed([text])
    assert tuple(call.vectors[0]) == pytest.approx(feature_vector(text))
    assert embedder.model == FAKE_EMBEDDING_MODEL


def test_product_fake_retriever_serves_unregistered_questions() -> None:
    runtime = shared_retrieval_runtime("fake")
    assert shared_retrieval_runtime("fake") is runtime
    result = runtime.retriever.search("完全没有登记过的新问法：退款以后还剩多少钱", context=_CONTEXT, top_k=3)
    assert result.items
