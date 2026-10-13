"""HTTP plumbing shared by the OpenAI-compatible chat and embedding adapters.

These helpers build the request, send it (once more only when the first attempt
certainly never reached an upstream model) and read what both adapters read the
same way from a response; each adapter keeps its own records.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import re
import secrets
from typing import Any
from urllib.parse import urlparse

import httpx
from opentelemetry.trace import SpanKind

from queryshield import tracing
from queryshield.providers.contracts import new_request_id


def base_url_is_valid(url: str) -> bool:
    """An http(s) URL with a host and no embedded credentials."""

    try:
        parsed = urlparse(url)
        parsed.port  # urlparse checks the port only when it is read
        host = httpx.URL(url).host  # the host a request would use: httpx decodes it only when read
    except (ValueError, httpx.InvalidURL):
        return False
    return parsed.scheme in {"http", "https"} and bool(host) and not (parsed.username or parsed.password)


def endpoint_url(base_url: str, path: str) -> str:
    """``base_url`` with ``path`` appended unless it already ends with it."""

    base = base_url.rstrip("/")
    return base if base.endswith(path) else f"{base}{path}"


def run_trace_id(run_id: str) -> str:
    """The W3C trace-id of every call of a run.

    Derived from the run id rather than stored, so a resume, a restarted process, a sub-agent
    and an MCP subprocess all send the run's one trace.
    """

    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:32]


def json_headers(api_key: str, request_id: str, run_id: str | None, model_call_id: str, trace_id: str) -> dict[str, str]:
    """The request headers.

    Only a call made inside a run carries ``X-Run-Id`` (so a gateway can bill per run) and
    ``Idempotency-Key``: calls outside a run, such as building the knowledge index, reuse the
    same ids on every start, and a gateway would refuse a reused key for a changed body.
    Every request is a new span (the parent-id) of the call's trace.
    """

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Client-Request-Id": request_id,
        "traceparent": f"00-{trace_id}-{secrets.token_hex(8)}-01",
    }
    if run_id is not None:
        headers["X-Run-Id"] = run_id
        headers["Idempotency-Key"] = model_call_id
    return headers


RETRY_WAIT_SECONDS = 1.0
# The gateway's HTTP 503 codes that, by its contract, it sends without having called the upstream model.
_NOT_SENT_UPSTREAM = frozenset({"ledger_unavailable", "rate_limiter_unavailable"})


@dataclass(frozen=True)
class Sent:
    """What a logical call got: the last attempt's response or transport error, and every attempt's record."""

    response: httpx.Response | None
    error: httpx.RequestError | None
    attempts: tuple[dict[str, object], ...]
    retryable: bool = False


def send_json(
    client: httpx.Client | None,
    url: str,
    payload: Mapping[str, object],
    *,
    api_key: str,
    request_id: str,
    model_call_id: str,
    run_id: str | None,
    timeout: float,
    sleep: Callable[[float], None],
) -> Sent:
    """POST ``payload``; once more, after RETRY_WAIT_SECONDS, when the first attempt never reached an upstream.

    Both attempts send the same bytes, the same Idempotency-Key and trace-id, each its own
    X-Client-Request-Id and parent-id.
    Anything else is the call's result; docs/operations.md gives the reason for each kind of failure.
    """

    # Exactly what httpx's json= sends, serialized once so a retry cannot differ.
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    # A call outside a run is a trace of its own; its retry stays in it.
    trace_id = run_trace_id(run_id) if run_id is not None else secrets.token_hex(16)
    # On the header's trace even when this thread lost the run's span, so the gateway's spans hang under ours.
    with tracing.span("queryshield.model_call", trace_id=trace_id) as call:
        endpoint = "embeddings" if url.endswith("/embeddings") else "chat"
        tracing.annotate(call, {"run_id": run_id, "model_call_id": model_call_id, "endpoint": endpoint})
        sent = _attempt(client, url, body, json_headers(api_key, request_id, run_id, model_call_id, trace_id), timeout)
        if sent.retryable:
            sleep(RETRY_WAIT_SECONDS)
            second = _attempt(client, url, body, json_headers(api_key, new_request_id(), run_id, model_call_id, trace_id), timeout)
            sent = replace(second, attempts=sent.attempts + second.attempts)
        last = sent.attempts[-1]
        tracing.annotate(call, {"attempt_count": len(sent.attempts), "status": last["status"], "error_code": last.get("error_code")})
    return sent


def _attempt(client: httpx.Client | None, url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> Sent:
    """One POST and its record (never a message text), as a CLIENT span whose id is the request's parent-id."""

    record: dict[str, object] = {"request_id": headers["X-Client-Request-Id"], "span_id": headers["traceparent"].split("-")[2]}
    with tracing.span("queryshield.http_attempt", span_id=str(record["span_id"]), kind=SpanKind.CLIENT, attributes={"http.request.method": "POST"}) as span:
        sent = _outcome(client, url, body, headers, timeout, record)
        tracing.annotate(span, sent.attempts[0])
    return sent


def _outcome(client: httpx.Client | None, url: str, body: bytes, headers: Mapping[str, str], timeout: float, record: dict[str, object]) -> Sent:
    """The attempt's result and record; the one place that decides whether it may be retried."""

    try:
        response = _post(client, url, body, headers, timeout)
    except httpx.TimeoutException as exc:
        # A read or write timeout may follow a request the upstream executed.  A ConnectTimeout sent
        # nothing, but it already waited the full timeout: another try could take a call past 60 s.
        return Sent(None, exc, ({**record, "status": "failed", "error_code": "upstream_timeout"},))
    except httpx.RequestError as exc:
        # httpx raises ConnectError only before any byte of the request is written.
        failed = {**record, "status": "failed", "error_code": "upstream_request_error"}
        return Sent(None, exc, (failed,), retryable=isinstance(exc, httpx.ConnectError))
    if 200 <= response.status_code < 300:
        return Sent(response, None, ({**record, "status": "succeeded"},))
    code, provider_code = http_error_codes(response)
    failed = {**record, "status": "failed", "error_code": code, "http_status": response.status_code}
    if provider_code is not None:
        failed["provider_error_code"] = provider_code
    return Sent(response, None, (failed,), retryable=response.status_code == 503 and provider_code in _NOT_SENT_UPSTREAM)


def _post(client: httpx.Client | None, url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
    """One POST through the injected client, or through a client made for this call."""

    if client is not None:
        return client.post(url, headers=headers, content=body, timeout=timeout)
    with httpx.Client(timeout=timeout) as one_off:
        return one_off.post(url, headers=headers, content=body)


_PROVIDER_ERROR_CODE = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def provider_error_code(response: Any) -> str | None:
    """The provider's machine error code from a JSON error body, or None.

    Only ``error.code`` (OpenAI style) or a top-level ``code`` is read, and
    only when it is a short identifier.  The message and any other text are
    never recorded: they can echo prompts, keys or endpoints.
    """

    try:
        body = response.json()
    except Exception:  # non-JSON or unreadable body
        return None
    if not isinstance(body, Mapping):
        return None
    error = body.get("error")
    code = error.get("code") if isinstance(error, Mapping) else None
    if code is None:
        code = body.get("code")
    if type(code) is int:
        code = str(code)
    if type(code) is not str or not _PROVIDER_ERROR_CODE.fullmatch(code):
        return None
    return code


# A model gateway's own limits, sent as HTTP 429 with an OpenAI-style error body:
# the gateway's error code and the run error code it ends the run with.
GATEWAY_LIMIT_CODES = {"quota_exhausted": "model_quota_exhausted", "rate_limited": "model_rate_limited"}


def http_error_codes(response: httpx.Response) -> tuple[str, str | None]:
    """The run error code for a non-2xx response, and the provider's error code to record.

    Only a 429 with one of the gateway's two limit codes gets a code of its own: those codes
    mean something only by agreement with the gateway, and other providers' 429s differ.
    """

    code = provider_error_code(response)
    if response.status_code == 429 and code in GATEWAY_LIMIT_CODES:
        return GATEWAY_LIMIT_CODES[code], code
    return "upstream_http_error", code


def response_model(body: Mapping[str, object], fallback: str) -> str:
    """The model name the provider answered with, else the configured one.

    A gateway passes the upstream's name on, so a run records what actually answered.
    """

    model = body.get("model")
    return model.strip() if isinstance(model, str) and model.strip() else fallback
