"""HTTP plumbing shared by the OpenAI-compatible chat and embedding adapters.

Each adapter still decides its own error codes and records; these helpers only
build the request and send it once (no retry).
"""

from __future__ import annotations

from collections.abc import Mapping
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


def json_headers(api_key: str, request_id: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Client-Request-Id": request_id,
    }


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
