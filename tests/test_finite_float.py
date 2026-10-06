"""A number too large for a float is "not finite" where a service's numbers are checked, never an OverflowError."""

from __future__ import annotations

import math

import pytest

from queryshield.knowledge.index import INDEX_VERSION, EmbeddingIndex, IndexChunk, IndexValidationError
from queryshield.providers.contracts import finite_float
from queryshield.providers.embedding import finite_vector
from queryshield.providers.rerank import FakeReranker, RerankCandidate, RerankInputError, parse_rerank_response, RerankResponseError

from test_provider_adapter_records import _rerank

HUGE = 10**400
LARGEST_FLOAT_INT = 2**1023  # the largest power of two a float holds


@pytest.mark.parametrize(
    "value, expected",
    [(0, 0.0), (3, 3.0), (-7, -7.0), (1.5, 1.5), (-0.0, -0.0), (1e308, 1e308), (LARGEST_FLOAT_INT, float(LARGEST_FLOAT_INT))],
)
def test_a_finite_int_or_float_is_returned_as_a_float(value, expected) -> None:
    result = finite_float(value)
    assert type(result) is float and result == expected


@pytest.mark.parametrize(
    "value",
    [HUGE, -HUGE, 2**1024, float("inf"), float("-inf"), float("nan"), True, False, None, "1", b"1", [1], (1,)],
    ids=["huge", "-huge", "2**1024", "inf", "-inf", "nan", "True", "False", "None", "str", "bytes", "list", "tuple"],
)
def test_anything_else_is_none(value) -> None:
    assert finite_float(value) is None


def test_a_float_subclass_is_not_a_plain_float() -> None:
    class Score(float):
        pass

    assert finite_float(Score(1.0)) is None


def test_embedding_vector_check_reports_a_huge_int_as_non_finite() -> None:
    with pytest.raises(ValueError, match="^embedding vector 0 contains a non-finite value$"):
        finite_vector([1, HUGE], dimensions=2, label="embedding vector 0", error=ValueError)
    assert finite_vector([1, 2.5], dimensions=2, label="v", error=ValueError) == (1.0, 2.5)
    for bad in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="non-finite"):
            finite_vector([1, bad], dimensions=2, label="v", error=ValueError)


def _index(vector: tuple[object, ...]) -> EmbeddingIndex:
    chunk = IndexChunk("chunk-1", "source-1", "v1", "text", vector)  # type: ignore[arg-type]
    return EmbeddingIndex(INDEX_VERSION, "snapshot-1", "k1", "c1", "model", "rev", 2, "hash", (chunk,))


def test_index_reports_a_huge_int_as_a_non_finite_vector_value() -> None:
    with pytest.raises(IndexValidationError, match="^index contains a non-finite vector value$"):
        _index((1, HUGE))
    for bad in (float("nan"), float("-inf"), True):
        with pytest.raises(IndexValidationError, match="^index contains a non-finite vector value$"):
            _index((1, bad))
    assert _index((1, 2.5)).chunks[0].vector == (1, 2.5)


_CANDIDATES = tuple(RerankCandidate(f"c{index}", f"text {index}", f"s{index}", "v1") for index in range(2))


def test_rerank_response_reports_a_huge_int_score_as_not_finite() -> None:
    with pytest.raises(RerankResponseError, match="^relevance_score must be finite numeric$"):
        parse_rerank_response({"results": [{"index": 0, "relevance_score": HUGE}]}, _CANDIDATES, top_n=2)
    for bad in (True, "1", None):
        with pytest.raises(RerankResponseError, match="^relevance_score must be finite numeric$"):
            parse_rerank_response({"results": [{"index": 0, "relevance_score": bad}]}, _CANDIDATES, top_n=2)
    ids, scores, _, _ = parse_rerank_response(
        {"results": [{"index": 1, "relevance_score": 2}, {"index": 0, "relevance_score": 0.5}]}, _CANDIDATES, top_n=2
    )
    assert (ids, scores) == (("c1", "c0"), (2.0, 0.5))


def test_rerank_response_accepts_a_score_of_zero() -> None:
    ids, scores, _, _ = parse_rerank_response({"results": [{"index": 0, "relevance_score": 0}]}, _CANDIDATES, top_n=2)
    assert (ids, scores) == (("c0",), (0.0,)) and type(scores[0]) is float
    ids, scores, _, _ = parse_rerank_response(
        {"results": [{"index": 0, "relevance_score": 0.0}, {"index": 1, "relevance_score": 0.5}]}, _CANDIDATES, top_n=2
    )
    assert (ids, scores) == (("c1", "c0"), (0.5, 0.0))


def test_fake_reranker_reports_a_huge_int_score_as_not_finite() -> None:
    for bad in (HUGE, float("nan"), True):
        with pytest.raises(RerankInputError, match="^fake score must be finite numeric for every candidate$"):
            FakeReranker({"c0": bad, "c1": 1.0}).rerank("q", _CANDIDATES, top_n=2)
    record = FakeReranker({"c0": 1, "c1": 2}).rerank("q", _CANDIDATES, top_n=2)
    assert record.returned_candidate_ids == ("c1", "c0") and all(type(score) is float for score in record.scores)
    assert math.isfinite(sum(record.scores))


def test_fake_reranker_accepts_a_score_of_zero() -> None:
    record = FakeReranker({"c0": 0, "c1": 0.0}).rerank("q", _CANDIDATES, top_n=2)
    assert record.returned_candidate_ids == ("c0", "c1") and record.scores == (0.0, 0.0)
    record = FakeReranker({"c0": 0, "c1": 1}).rerank("q", _CANDIDATES, top_n=2)
    assert record.returned_candidate_ids == ("c1", "c0") and record.scores == (1.0, 0.0)


def test_the_http_rerank_call_fails_with_the_fixed_error_code_for_a_huge_int_score() -> None:
    record, _ = _rerank((200, {"results": [{"index": 0, "relevance_score": HUGE}]}, {}))
    assert (record.status, record.error_code) == ("failed", "invalid_response:relevance_score must be finite numeric")
