"""One retry, only for a first attempt that certainly never reached an upstream model.

Both HTTP adapters (chat and embedding) go through providers.http.send_json.  A refused
connection (httpx.ConnectError) and the gateway's 503 ``ledger_unavailable`` /
``rate_limiter_unavailable`` (sent, by the gateway's contract, without calling the upstream) are
sent once more after the injected wait, with the same Idempotency-Key and the same bytes and a new
X-Client-Request-Id.  Anything else is the call's result after one request (docs/operations.md
gives the reason for each kind of failure).  A call in a run carries Idempotency-Key (its
model_call_id); a call outside a run carries none.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from queryshield.providers.contracts import ModelProviderError
from queryshield.providers.embedding import EmbeddingConfig, EmbeddingProviderError, OpenAICompatibleEmbedding
from queryshield.providers.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleModel

CHAT = OpenAICompatibleConfig(base_url="https://gateway.test/v1", api_key="k", model="chat-model", timeout_seconds=7.0)
EMBED = EmbeddingConfig(base_url="https://gateway.test/v1", api_key="k", model="embed-model", model_revision="r1", dimensions=2)
# Non-ASCII text, so a body serialized with ASCII escapes would differ from httpx's json= bytes.
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "九月的支付金额"}]
CALL_ID = "local-call-1"
RUN_ID = "run-retry"


def _error(status: int, code: str | None, request_id: str | None = None) -> httpx.Response:
    headers = {"x-request-id": request_id} if request_id is not None else {}
    if code is None:
        return httpx.Response(status, text="busy", headers=headers)
    return httpx.Response(status, json={"error": {"code": code, "message": "secret text"}}, headers=headers)


def _chat_ok(tokens: int) -> httpx.Response:
    body = {
        "id": "chat-1",
        "choices": [{"message": {"content": '{"type":"deny","reason":"r"}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": tokens - 1, "completion_tokens": 1, "total_tokens": tokens},
    }
    return httpx.Response(200, json=body, headers={"x-request-id": f"upstream-{tokens}"})


def _embed_ok(tokens: int) -> httpx.Response:
    body = {"id": "embed-1", "data": [{"index": 0, "embedding": [1, 0]}], "usage": {"total_tokens": tokens}}
    return httpx.Response(200, json=body, headers={"x-request-id": f"upstream-{tokens}"})


def _chat(client, sleep, run_id):
    return OpenAICompatibleModel(CHAT, client=client, sleep=sleep).complete(
        MESSAGES, request_id="request-1", model_call_id=CALL_ID, run_id=run_id
    )


def _embed(client, sleep, run_id):
    return OpenAICompatibleEmbedding(EMBED, client=client, sleep=sleep).embed(
        ["九月的支付金额"], request_id="request-1", model_call_id=CALL_ID, run_id=run_id
    )


def _chat_tokens(result) -> int:
    return result.usage.total_tokens


def _embed_tokens(result) -> int:
    return result.usage.total_tokens


# (call, successful response, provider error type, token count of a result)
ADAPTERS = [
    pytest.param((_chat, _chat_ok, ModelProviderError, _chat_tokens), id="chat"),
    pytest.param((_embed, _embed_ok, EmbeddingProviderError, _embed_tokens), id="embedding"),
]


class _Upstream:
    """Answers each request with the next scripted outcome: an exception to raise or a response."""

    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _send(adapter, outcomes, *, run_id=RUN_ID):
    """The call's result (or provider error), the requests sent and the waits asked for."""

    call, _ok, error_type, _tokens = adapter
    upstream = _Upstream(outcomes)
    waits: list[float] = []
    with httpx.Client(transport=httpx.MockTransport(upstream)) as client:
        try:
            result = call(client, waits.append, run_id)
        except error_type as exc:
            result = exc
    return result, upstream.requests, waits


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """The adapters must wait only through the function they were given."""

    def refuse(seconds: float) -> None:
        raise AssertionError("time.sleep called directly")

    monkeypatch.setattr(time, "sleep", refuse)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_only_a_call_in_a_run_carries_the_key(adapter) -> None:
    ok = adapter[1]
    _, (inside,), _ = _send(adapter, [ok(5)])
    _, (outside,), _ = _send(adapter, [ok(5)], run_id=None)
    assert inside.headers["idempotency-key"] == CALL_ID and inside.headers["x-run-id"] == RUN_ID
    assert "idempotency-key" not in outside.headers and "x-run-id" not in outside.headers


RETRIED_FIRSTS = [
    pytest.param(lambda: httpx.ConnectError("refused"), {"error_code": "upstream_request_error"}, id="connect_error"),
    pytest.param(
        lambda: _error(503, "ledger_unavailable"),
        {"error_code": "upstream_http_error", "http_status": 503, "provider_error_code": "ledger_unavailable"},
        id="503_ledger_unavailable",
    ),
    pytest.param(
        lambda: _error(503, "rate_limiter_unavailable"),
        {"error_code": "upstream_http_error", "http_status": 503, "provider_error_code": "rate_limiter_unavailable"},
        id="503_rate_limiter_unavailable",
    ),
]


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("first, first_record", RETRIED_FIRSTS)
def test_a_first_attempt_that_never_reached_the_upstream_is_sent_once_more(adapter, first, first_record) -> None:
    ok, tokens = adapter[1], adapter[3]
    result, requests, waits = _send(adapter, [first(), ok(9), ok(99)])

    assert len(requests) == 2 and waits == [1.0]
    one, two = requests
    assert one.headers["idempotency-key"] == two.headers["idempotency-key"] == CALL_ID
    assert one.headers["x-run-id"] == two.headers["x-run-id"] == RUN_ID
    assert one.content == two.content
    # Serialized once, exactly as httpx's json= would have sent it.
    assert one.content == httpx.Request("POST", one.url, json=json.loads(one.content)).content
    assert one.headers["x-client-request-id"] == "request-1" != two.headers["x-client-request-id"]
    # The second attempt's response is the call's result.
    assert tokens(result) == 9 and result.provider_request_id == "upstream-9"
    assert (result.request_id, result.model_call_id) == ("request-1", CALL_ID)
    assert list(result.attempts) == [
        {"request_id": "request-1", "status": "failed", **first_record},
        {"request_id": two.headers["x-client-request-id"], "status": "succeeded"},
    ]


NOT_RETRIED = [
    pytest.param(httpx.ReadTimeout("slow"), "upstream_timeout", id="read_timeout"),
    pytest.param(httpx.ConnectTimeout("slow"), "upstream_timeout", id="connect_timeout"),
    pytest.param(httpx.ReadError("reset"), "upstream_request_error", id="read_error"),
    pytest.param(httpx.RemoteProtocolError("closed"), "upstream_request_error", id="remote_protocol_error"),
    pytest.param(_error(429, "quota_exhausted"), "model_quota_exhausted", id="429_quota_exhausted"),
    pytest.param(_error(429, "rate_limited"), "model_rate_limited", id="429_rate_limited"),
    pytest.param(_error(503, "upstream_rate_limited"), "upstream_http_error", id="503_upstream_rate_limited"),
    pytest.param(_error(503, "rate_limited"), "upstream_http_error", id="503_rate_limited"),
    pytest.param(_error(503, None), "upstream_http_error", id="503_without_body"),
    # The two retryable codes count only under 503.
    pytest.param(_error(502, "ledger_unavailable"), "upstream_http_error", id="502_ledger_unavailable"),
    pytest.param(_error(500, "rate_limiter_unavailable"), "upstream_http_error", id="500_rate_limiter_unavailable"),
    pytest.param(_error(409, "ledger_unavailable"), "upstream_http_error", id="409_ledger_unavailable"),
    pytest.param(_error(504, "upstream_timeout"), "upstream_http_error", id="504"),
    pytest.param(_error(500, "server_busy"), "upstream_http_error", id="500"),
    pytest.param(_error(409, "idempotency_in_progress"), "upstream_http_error", id="409_in_progress"),
    pytest.param(_error(409, "idempotency_outcome_unknown"), "upstream_http_error", id="409_outcome_unknown"),
    pytest.param(_error(409, "idempotency_response_too_large"), "upstream_http_error", id="409_response_too_large"),
    pytest.param(_error(409, "idempotency_response_unavailable"), "upstream_http_error", id="409_response_unavailable"),
    pytest.param(_error(422, "idempotency_key_reused"), "upstream_http_error", id="422_key_reused"),
]


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("first, code", NOT_RETRIED)
def test_anything_else_is_the_result_of_one_request(adapter, first, code) -> None:
    error, requests, waits = _send(adapter, [first, adapter[1](9)])

    assert isinstance(error, adapter[2]) and error.code == code
    assert len(requests) == 1 and waits == []
    assert [attempt["request_id"] for attempt in error.record["attempts"]] == ["request-1"]
    assert error.record["attempts"][0]["error_code"] == code
    assert "secret text" not in repr(error.record)


# (outcomes, call error code, fields of the second attempt, the record's provider_request_id)
TWO_FAILURES = [
    pytest.param(
        [httpx.ConnectError("refused"), httpx.ConnectError("refused")],
        "upstream_request_error",
        {},
        None,
        id="connect_error_twice",
    ),
    pytest.param(
        [httpx.ConnectError("refused"), _error(503, "ledger_unavailable", "gateway-2")],
        "upstream_http_error",
        {"http_status": 503, "provider_error_code": "ledger_unavailable"},
        "gateway-2",
        id="connect_error_then_503",
    ),
    pytest.param(
        [_error(503, "rate_limiter_unavailable", "gateway-1"), httpx.ReadTimeout("slow")],
        "upstream_timeout",
        {},
        None,
        id="503_then_timeout",
    ),
    pytest.param(
        [_error(503, "ledger_unavailable", "gateway-1"), _error(503, "rate_limiter_unavailable", "gateway-2")],
        "upstream_http_error",
        {"http_status": 503, "provider_error_code": "rate_limiter_unavailable"},
        "gateway-2",
        id="503_twice",
    ),
]


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("outcomes, code, last_fields, provider_request_id", TWO_FAILURES)
def test_when_both_attempts_fail_the_second_is_the_result(adapter, outcomes, code, last_fields, provider_request_id) -> None:
    # A third outcome is ready, so a second retry would succeed and be noticed.
    error, requests, waits = _send(adapter, [*outcomes, adapter[1](9)])

    assert len(requests) == 2 and waits == [1.0]
    assert error.code == code and error.record["error_code"] == code
    assert {key: error.record[key] for key in last_fields} == last_fields
    assert error.record.get("http_status") == last_fields.get("http_status")
    # The second attempt's response, or none when it got no response.
    assert error.record["provider_request_id"] == provider_request_id
    second = requests[1].headers["x-client-request-id"]
    assert [attempt["request_id"] for attempt in error.record["attempts"]] == ["request-1", second]
    assert error.record["attempts"][1] == {"request_id": second, "status": "failed", "error_code": code, **last_fields}


def test_the_chat_timeout_defaults_to_30_seconds(monkeypatch) -> None:
    assert OpenAICompatibleConfig(base_url="https://gateway.test/v1", api_key="k", model="m").timeout_seconds == 30.0
    monkeypatch.setenv("QUERYSHIELD_MODEL_BASE_URL", "https://gateway.test/v1")
    monkeypatch.setenv("QUERYSHIELD_MODEL_API_KEY", "k")
    monkeypatch.setenv("QUERYSHIELD_MODEL_NAME", "m")
    monkeypatch.delenv("QUERYSHIELD_MODEL_TIMEOUT_SECONDS", raising=False)
    assert OpenAICompatibleConfig.from_env().timeout_seconds == 30.0
