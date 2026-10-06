from __future__ import annotations

import json

import httpx
import pytest

from queryshield.providers.rerank import (
    FakeReranker,
    HttpRerankAdapter,
    RerankCandidate,
    RerankInputError,
    RerankResponseError,
    authorized_candidates,
    parse_rerank_response,
    rerank_authorized_candidates,
)


def _candidate(candidate_id: str, *, tenant: str = "A", text: str | None = None) -> RerankCandidate:
    return RerankCandidate(
        candidate_id=candidate_id,
        text=text or f"tenant={tenant} metric information for {candidate_id}",
        source_id=f"source-{candidate_id}",
        version="2026-09-21",
    )


def test_fake_rerank_is_stable_and_uses_index_for_score_ties() -> None:
    candidates = (_candidate("first"), _candidate("second"), _candidate("third"))
    result = FakeReranker({"first": 0.4, "second": 0.9, "third": 0.9}).rerank(
        "metric query", candidates, top_n=3
    )

    assert result.status == "succeeded"
    assert result.returned_candidate_ids == ("second", "third", "first")
    assert result.scores == (0.9, 0.9, 0.4)
    assert result.usage_status == "unknown"
    assert result.total_tokens is None


def test_acl_filter_happens_before_adapter_receives_candidate_text() -> None:
    candidates = (
        _candidate("visible", tenant="A"),
        _candidate("other-tenant-secret", tenant="B", text="private tenant B order data"),
    )
    visible, filtered = authorized_candidates(
        candidates,
        authorized_candidate_ids={"visible"},
    )
    assert [candidate.candidate_id for candidate in visible] == ["visible"]
    assert filtered == ("other-tenant-secret",)

    result, filtered = rerank_authorized_candidates(
        FakeReranker({"visible": 1.0}),
        "query",
        candidates,
        authorized_candidate_ids={"visible"},
    )
    assert result is not None
    assert result.input_candidate_ids == ("visible",)
    assert filtered == ("other-tenant-secret",)


@pytest.mark.parametrize(
    "results, message, count",
    [
        ([{"index": 0, "relevance_score": 0.4}, {"index": 0, "relevance_score": 0.2}], "unique", 2),
        ([{"index": 9, "relevance_score": 0.4}], "out of range", 1),
        ([{"index": -1, "relevance_score": 0.4}], "out of range", 1),
        ([{"index": 0, "relevance_score": float("nan")}], "finite", 1),
        ([{"index": 0, "relevance_score": float("inf")}], "finite", 1),
        ([{"index": True, "relevance_score": 0.4}], "out of range", 1),
        ([{"index": 0, "relevance_score": 0.4, "unexpected": "value"}], "unsupported", 1),
    ],
)
def test_strict_response_parser_rejects_malformed_indexes_and_scores(results, message, count) -> None:
    candidates = tuple(_candidate(f"doc-{index}") for index in range(count))
    with pytest.raises(RerankResponseError, match=message):
        parse_rerank_response({"results": results}, candidates, top_n=count)


def test_request_bounds_reject_large_or_ambiguous_candidate_sets() -> None:
    candidates = tuple(_candidate(f"doc-{index}") for index in range(11))
    with pytest.raises(RerankInputError, match="at most ten"):
        FakeReranker({candidate.candidate_id: 0.5 for candidate in candidates}).rerank(
            "query", candidates, top_n=3
        )
    with pytest.raises(RerankInputError, match="top_n"):
        FakeReranker({"one": 0.5}).rerank("query", (_candidate("one"),), top_n=2)
    with pytest.raises(RerankInputError, match="byte limit"):
        FakeReranker({"one": 0.5}).rerank("q" * 9_000, (_candidate("one"),), top_n=1)


def test_provider_response_records_call_and_known_or_unknown_usage() -> None:
    candidates = (_candidate("zero"), _candidate("one"))
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.3},
                    {"index": 0, "relevance_score": 0.9},
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        )

    adapter = HttpRerankAdapter(
        "https://rerank.example/v1/rerank",
        "test-secret-never-recorded",
        "rerank-small",
        transport=httpx.MockTransport(handler),
    )
    result = adapter.rerank("query", candidates, top_n=2)

    assert result.status == "succeeded"
    assert result.call_id
    assert result.input_candidate_ids == ("zero", "one")
    assert result.returned_candidate_ids == ("zero", "one")
    assert result.usage_status == "known"
    assert result.total_tokens == 5
    assert result.call_id in requests[0].headers["X-Client-Call-Id"]
    assert "test-secret-never-recorded" not in repr(result.as_dict())


def test_bailian_qwen3_documented_response_and_request_shape() -> None:
    """Exercise the documented HTTP envelope without making a real provider call."""
    candidates = (_candidate("first"), _candidate("second"))

    def handler(request: httpx.Request) -> httpx.Response:
        request_body = json.loads(request.content)
        assert request_body["model"] == "qwen3-rerank"
        assert request_body["top_n"] == 2
        assert "input" not in request_body
        assert "return_documents" not in request_body
        return httpx.Response(200, json={
            "object": "list",
            "model": "qwen3-rerank",
            "id": "synthetic-provider-response-id",
            "results": [
                {"index": 1, "relevance_score": 0.93},
                {"index": 0, "relevance_score": 0.34},
            ],
            "usage": {"total_tokens": 79},
        })

    adapter = HttpRerankAdapter(
        "https://rerank.example/compatible-api/v1/reranks",
        "local-test-key",
        "qwen3-rerank",
        transport=httpx.MockTransport(handler),
    )
    result = adapter.rerank("query", candidates, top_n=2)
    assert result.status == "succeeded"
    assert result.returned_candidate_ids == ("second", "first")
    assert result.usage_status == "known"
    assert result.total_tokens == 79


@pytest.mark.parametrize("object_value", [None, True, "other", {"type": "list"}])
def test_rerank_object_discriminator_must_be_list(object_value) -> None:
    with pytest.raises(RerankResponseError, match="object"):
        parse_rerank_response(
            {"object": object_value, "results": [{"index": 0, "relevance_score": 0.5}]},
            (_candidate("first"),), top_n=1,
        )


def test_provider_unknown_usage_is_null_and_timeout_stays_a_failed_call() -> None:
    candidates = (_candidate("one"),)
    adapter = HttpRerankAdapter(
        "https://rerank.example/v1/rerank",
        "local-only",
        "rerank-small",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.7}]})
        ),
    )
    result = adapter.rerank("query", candidates, top_n=1)
    assert result.usage_status == "unknown"
    assert result.total_tokens is None

    timeout_adapter = HttpRerankAdapter(
        "https://rerank.example/v1/rerank",
        "local-only",
        "rerank-small",
        transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("timeout"))),
    )
    timeout = timeout_adapter.rerank("query", candidates, top_n=1)
    assert timeout.status == "timeout"
    assert timeout.error_code == "timeout"
    assert timeout.usage_status == "unknown"
    assert timeout.total_tokens is None


def test_provider_malformed_response_is_recorded_as_failure_not_a_fake_success() -> None:
    adapter = HttpRerankAdapter(
        "https://rerank.example/v1/rerank",
        "local-only",
        "rerank-small",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"results": [{"index": 1, "relevance_score": 0.7}]},
            )
        ),
    )
    result = adapter.rerank("query", (_candidate("one"),), top_n=1)
    assert result.status == "failed"
    assert result.error_code is not None and "out of range" in result.error_code
    assert result.usage_status == "unknown"
    assert result.total_tokens is None
