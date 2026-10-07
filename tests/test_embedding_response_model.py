"""The embedding adapter records the model name the service answered with, as the chat adapter does.

The configured name and the answered name differ in every case, so reading the wrong
one cannot pass.  The index and the embedded snapshot keep the configured name and
revision: they describe the configuration the index was built for.
"""

from __future__ import annotations

import json

import httpx
import pytest

from queryshield.knowledge.index import build_embedding_index
from queryshield.knowledge.runtime import product_knowledge
from queryshield.providers.embedding import EmbeddingConfig, OpenAICompatibleEmbedding

CONFIGURED = "configured-embedding"
ANSWERED = "answered-embedding"


def _adapter(model_field: object = ANSWERED, *, dimensions: int = 2) -> OpenAICompatibleEmbedding:
    def handler(request: httpx.Request) -> httpx.Response:
        count = len(json.loads(request.content)["input"])
        body: dict[str, object] = {
            "id": "emb-1",
            "data": [{"index": index, "embedding": [1.0] + [0.0] * (dimensions - 1)} for index in range(count)],
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
        if model_field is not None:
            body["model"] = model_field
        return httpx.Response(200, json=body)

    config = EmbeddingConfig(
        base_url="https://embedding.test/v1", api_key="k", model=CONFIGURED, model_revision="configured-r1", dimensions=dimensions
    )
    return OpenAICompatibleEmbedding(config, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_the_result_and_its_usage_name_the_model_that_answered() -> None:
    result = _adapter().embed(["退款后净额"])

    assert result.model == ANSWERED
    assert result.usage.model == ANSWERED
    assert result.usage.as_dict()["model"] == ANSWERED
    assert result.model_revision == "configured-r1"


@pytest.mark.parametrize("model_field", [None, "", "   ", 7, ["x"]])
def test_without_a_usable_model_in_the_response_the_configured_name_is_kept(model_field) -> None:
    result = _adapter(model_field).embed(["退款后净额"])

    assert result.model == CONFIGURED
    assert result.usage.model == CONFIGURED


def test_surrounding_spaces_in_the_answered_name_are_dropped() -> None:
    assert _adapter(f"  {ANSWERED} ").embed(["x"]).model == ANSWERED


def test_the_index_and_snapshot_keep_the_configured_name() -> None:
    embedder = _adapter(dimensions=128)
    snapshot = product_knowledge(demo=False).snapshot
    answered = build_embedding_index(snapshot, embedder, ingest_job_id="answered")
    configured = build_embedding_index(snapshot, _adapter(CONFIGURED, dimensions=128), ingest_job_id="configured")

    assert answered.index.model == CONFIGURED
    assert {usage.model for usage in answered.operation_usages} == {ANSWERED}
    # The answered name changes neither the index hash nor the embedded snapshot id.
    assert answered.index.index_hash == configured.index.index_hash
    assert answered.snapshot.snapshot_id == configured.snapshot.snapshot_id
