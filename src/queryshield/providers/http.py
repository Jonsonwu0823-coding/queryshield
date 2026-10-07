"""HTTP plumbing shared by the OpenAI-compatible chat and embedding adapters.

These helpers build the request, send it once (no retry) and read what both
adapters read the same way from a response; each adapter keeps its own records.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any
from urllib.parse import urlparse

import httpx


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


def json_headers(api_key: str, request_id: str, run_id: str | None) -> dict[str, str]:
    """The request headers; ``X-Run-Id`` only for a call made inside a run, so a gateway can bill per run."""

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Client-Request-Id": request_id,
    }
    if run_id is not None:
        headers["X-Run-Id"] = run_id
    return headers


def post_json(
    client: httpx.Client | None,
    url: str,
    payload: Mapping[str, object],
    headers: Mapping[str, str],
    timeout: float,
) -> httpx.Response:
    """One POST through the injected client, or through a client made for this call."""

    if client is not None:
        return client.post(url, headers=headers, json=payload, timeout=timeout)
    with httpx.Client(timeout=timeout) as one_off:
        return one_off.post(url, headers=headers, json=payload)


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
