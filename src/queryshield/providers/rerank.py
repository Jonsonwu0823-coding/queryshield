"""Strict, ACL-scoped rerank adapter with auditable usage records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from typing import Protocol
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from queryshield.providers.contracts import finite_float
from queryshield.providers.http import base_url_is_valid


MAX_RERANK_CANDIDATES = 10
MAX_RERANK_QUERY_BYTES = 8_192
MAX_RERANK_DOCUMENT_BYTES = 16_384


class RerankConfigurationError(ValueError):
    """A rerank endpoint cannot be configured safely."""


class RerankInputError(ValueError):
    """A rerank request is malformed or contains unauthorized candidates."""


class RerankResponseError(ValueError):
    """A provider response violates the strict rerank response contract."""


@dataclass(frozen=True)
class RerankCandidate:
    candidate_id: str
    text: str
    source_id: str
    version: str


@dataclass(frozen=True)
class RerankCallRecord:
    call_id: str
    status: str
    model: str
    input_candidate_ids: tuple[str, ...]
    returned_candidate_ids: tuple[str, ...]
    scores: tuple[float, ...]
    usage_status: str
    total_tokens: int | None
    error_code: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "status": self.status,
            "model": self.model,
            "input_candidate_ids": list(self.input_candidate_ids),
            "returned_candidate_ids": list(self.returned_candidate_ids),
            "scores": list(self.scores),
            "usage_status": self.usage_status,
            "total_tokens": self.total_tokens,
            "error_code": self.error_code,
        }


class RerankAdapter(Protocol):
    model: str

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_n: int,
    ) -> RerankCallRecord: ...


def _validate_request(
    query: str,
    candidates: Sequence[RerankCandidate],
    *,
    top_n: int,
) -> None:
    if type(query) is not str or not query.strip():
        raise RerankInputError("query must be a non-empty string")
    if len(query.encode("utf-8")) > MAX_RERANK_QUERY_BYTES:
        raise RerankInputError("query exceeds the configured byte limit")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise RerankInputError("candidates must be a sequence")
    if len(candidates) > MAX_RERANK_CANDIDATES:
        raise RerankInputError("at most ten authorized candidates may be reranked")
    if not candidates:
        raise RerankInputError("at least one authorized candidate is required")
    if type(top_n) is not int or not 1 <= top_n <= len(candidates):
        raise RerankInputError("top_n must be between one and candidate count")
    ids: list[str] = []
    total_document_bytes = 0
    for candidate in candidates:
        if not isinstance(candidate, RerankCandidate):
            raise RerankInputError("candidate must be a RerankCandidate")
        if any(
            type(value) is not str or not value.strip()
            for value in (candidate.candidate_id, candidate.text, candidate.source_id, candidate.version)
        ):
            raise RerankInputError("candidate identity and content fields must be non-empty strings")
        ids.append(candidate.candidate_id)
        total_document_bytes += len(candidate.text.encode("utf-8"))
    if len(ids) != len(set(ids)):
        raise RerankInputError("candidate IDs must be unique")
    if total_document_bytes > MAX_RERANK_DOCUMENT_BYTES:
        raise RerankInputError("candidate text exceeds the configured byte limit")


def authorized_candidates(
    candidates: Sequence[RerankCandidate],
    *,
    authorized_candidate_ids: set[str] | frozenset[str],
) -> tuple[tuple[RerankCandidate, ...], tuple[str, ...]]:
    """Filter ACL-invisible candidates before any provider request is built."""

    # A string here would turn the ACL membership test into substring matching.
    # The candidates that are kept are checked by the adapter's _validate_request.
    if not isinstance(authorized_candidate_ids, (set, frozenset)):
        raise RerankInputError("authorized_candidate_ids must be a set")
    visible = tuple(candidate for candidate in candidates if candidate.candidate_id in authorized_candidate_ids)
    filtered = tuple(
        candidate.candidate_id for candidate in candidates if candidate.candidate_id not in authorized_candidate_ids
    )
    if len(visible) > MAX_RERANK_CANDIDATES:
        raise RerankInputError("at most ten authorized candidates may be reranked")
    return visible, filtered


def _parse_total_tokens(raw_usage: object) -> tuple[str, int | None]:
    if not isinstance(raw_usage, Mapping):
        return "unknown", None
    total = raw_usage.get("total_tokens")
    if type(total) is int and total >= 0:
        return "known", total
    prompt = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens"))
    completion = raw_usage.get("completion_tokens", raw_usage.get("output_tokens", 0))
    if type(prompt) is int and prompt >= 0 and type(completion) is int and completion >= 0:
        return "known", prompt + completion
    return "unknown", None


def parse_rerank_response(
    body: object,
    candidates: Sequence[RerankCandidate],
    *,
    top_n: int,
) -> tuple[tuple[str, ...], tuple[float, ...], str, int | None]:
    """Validate returned indexes/scores and bind indexes back to this call's candidates."""

    _validate_request("bounded query", candidates, top_n=top_n)
    if not isinstance(body, Mapping) or set(body) - {"results", "usage", "id", "model", "object"}:
        raise RerankResponseError("response root has unsupported fields")
    if "object" in body and body["object"] != "list":
        raise RerankResponseError("response object must be list")
    raw_results = body.get("results")
    if type(raw_results) is not list or not 1 <= len(raw_results) <= top_n:
        raise RerankResponseError("results must contain between one and top_n entries")
    parsed: list[tuple[int, float]] = []
    seen: set[int] = set()
    for item in raw_results:
        if not isinstance(item, Mapping) or set(item) - {"index", "relevance_score", "document", "id"}:
            raise RerankResponseError("result entry has unsupported fields")
        index = item.get("index")
        score = item.get("relevance_score")
        if type(index) is not int or not 0 <= index < len(candidates):
            raise RerankResponseError("result index is out of range")
        if index in seen:
            raise RerankResponseError("result indexes must be unique")
        number = finite_float(score)
        if number is None:
            raise RerankResponseError("relevance_score must be finite numeric")
        seen.add(index)
        parsed.append((index, number))
    parsed.sort(key=lambda item: (-item[1], item[0]))
    ids = tuple(candidates[index].candidate_id for index, _ in parsed)
    scores = tuple(score for _, score in parsed)
    usage_status, total_tokens = _parse_total_tokens(body.get("usage"))
    return ids, scores, usage_status, total_tokens


@dataclass(frozen=True)
class FakeReranker:
    """Deterministic provider-shaped reranker for contract and regression checks."""

    scores_by_candidate_id: Mapping[str, float]
    model: str = "fake-reranker-v1"

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_n: int,
    ) -> RerankCallRecord:
        _validate_request(query, candidates, top_n=top_n)
        scored: list[tuple[float, int, str]] = []
        for index, candidate in enumerate(candidates):
            score = self.scores_by_candidate_id.get(candidate.candidate_id)
            number = finite_float(score)
            if number is None:
                raise RerankInputError("fake score must be finite numeric for every candidate")
            scored.append((number, index, candidate.candidate_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = scored[:top_n]
        return RerankCallRecord(
            call_id=f"rerank-{uuid4()}",
            status="succeeded",
            model=self.model,
            input_candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
            returned_candidate_ids=tuple(candidate_id for _, _, candidate_id in selected),
            scores=tuple(score for score, _, _ in selected),
            usage_status="unknown",
            total_tokens=None,
        )


def _is_https_or_loopback(endpoint: str) -> bool:
    """The Key is sent only over HTTPS, or over plain HTTP to this machine."""

    if not base_url_is_valid(endpoint):
        return False
    parsed = urlparse(endpoint)
    return parsed.scheme == "https" or parsed.hostname in {"127.0.0.1", "localhost", "::1"}


@dataclass
class HttpRerankAdapter:
    endpoint: str
    api_key: str
    model: str
    timeout_seconds: float = 15.0
    transport: httpx.BaseTransport | None = None

    @classmethod
    def from_env(cls, *, transport: httpx.BaseTransport | None = None) -> "HttpRerankAdapter":
        endpoint = (os.getenv("QUERYSHIELD_RERANK_URL") or "").strip()
        api_key = (os.getenv("QUERYSHIELD_RERANK_API_KEY") or "").strip()
        model = (os.getenv("QUERYSHIELD_RERANK_MODEL_NAME") or "").strip()
        missing = [
            name
            for name, value in (
                ("QUERYSHIELD_RERANK_URL", endpoint),
                ("QUERYSHIELD_RERANK_API_KEY", api_key),
                ("QUERYSHIELD_RERANK_MODEL_NAME", model),
            )
            if not value
        ]
        if missing:
            raise RerankConfigurationError("missing configuration names: " + ",".join(missing))
        if not _is_https_or_loopback(endpoint):
            raise RerankConfigurationError("rerank URL must use HTTPS or a loopback HTTP endpoint")
        return cls(endpoint, api_key, model, transport=transport)

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_n: int,
    ) -> RerankCallRecord:
        _validate_request(query, candidates, top_n=top_n)
        call_id = f"rerank-{uuid4()}"
        input_ids = tuple(candidate.candidate_id for candidate in candidates)

        def failed(status: str, error_code: str) -> RerankCallRecord:
            return RerankCallRecord(call_id, status, self.model, input_ids, (), (), "unknown", None, error_code)

        headers = {"Authorization": f"Bearer {self.api_key}", "X-Client-Call-Id": call_id}
        payload = {
            "model": self.model,
            "query": query,
            "documents": [candidate.text for candidate in candidates],
            "top_n": top_n,
        }
        # Bailian qwen3-rerank uses the flat compatible API and does not document
        # return_documents; its response may include object="list".
        if self.model != "qwen3-rerank":
            payload["return_documents"] = False
        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = client.post(self.endpoint, headers=headers, json=payload)
            if response.status_code < 200 or response.status_code >= 300:
                return failed("failed", f"http_{response.status_code}")
            try:
                body = response.json()
            except (ValueError, UnicodeError):
                raise RerankResponseError("response body is not valid JSON") from None
            ids, scores, usage_status, total_tokens = parse_rerank_response(body, candidates, top_n=top_n)
            return RerankCallRecord(call_id, "succeeded", self.model, input_ids, ids, scores, usage_status, total_tokens)
        except httpx.TimeoutException:
            return failed("timeout", "timeout")
        except httpx.HTTPError:
            return failed("failed", "transport_error")
        except RerankResponseError as exc:
            return failed("failed", "invalid_response:" + str(exc))


def rerank_authorized_candidates(
    adapter: RerankAdapter,
    query: str,
    candidates: Sequence[RerankCandidate],
    *,
    authorized_candidate_ids: set[str] | frozenset[str],
    top_n: int = 3,
) -> tuple[RerankCallRecord | None, tuple[str, ...]]:
    visible, filtered_ids = authorized_candidates(
        candidates,
        authorized_candidate_ids=authorized_candidate_ids,
    )
    if not visible:
        return None, filtered_ids
    return adapter.rerank(query, visible, top_n=min(top_n, len(visible))), filtered_ids


__all__ = [
    "MAX_RERANK_CANDIDATES",
    "MAX_RERANK_DOCUMENT_BYTES",
    "MAX_RERANK_QUERY_BYTES",
    "FakeReranker",
    "HttpRerankAdapter",
    "RerankAdapter",
    "RerankCallRecord",
    "RerankCandidate",
    "RerankConfigurationError",
    "RerankInputError",
    "RerankResponseError",
    "authorized_candidates",
    "parse_rerank_response",
    "rerank_authorized_candidates",
]
