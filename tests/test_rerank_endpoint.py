"""The rerank Key is sent only over HTTPS or to a loopback host, judged by the parsed host name."""

from __future__ import annotations

import pytest

from queryshield.providers.rerank import HttpRerankAdapter, RerankConfigurationError

MESSAGE = "rerank URL must use HTTPS or a loopback HTTP endpoint"


def _from_env(monkeypatch, url: str) -> HttpRerankAdapter:
    monkeypatch.setenv("QUERYSHIELD_RERANK_URL", url)
    monkeypatch.setenv("QUERYSHIELD_RERANK_API_KEY", "key-for-test")
    monkeypatch.setenv("QUERYSHIELD_RERANK_MODEL_NAME", "model-for-test")
    return HttpRerankAdapter.from_env()


@pytest.mark.parametrize(
    "url",
    [
        "https://rerank.example.com/v1/rerank",
        "https://203.0.113.9:8443/rerank",
        "http://127.0.0.1:8000/rerank",
        "http://127.0.0.1/rerank",
        "http://localhost:8080/x",
        "http://[::1]:8000/x",
    ],
)
def test_https_and_loopback_http_endpoints_are_accepted(monkeypatch, url) -> None:
    assert _from_env(monkeypatch, url).endpoint == url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1.example.com/rerank",
        "http://127.0.0.1@example.com/rerank",
        "http://localhost.example.com/rerank",
        "http://localhost@example.com/rerank",
        "http://evil-localhost/rerank",
        "http://example127.0.0.1/rerank",
        "http://0.0.0.0/rerank",
        "http://127.0.0.2/rerank",
        "http://example.com/rerank",
        "https://user:secret@rerank.example.com/rerank",
        "https://user@rerank.example.com/rerank",
        "https:///rerank",
        "http://[::1/rerank",
        "http://[bad]/rerank",
        "ftp://127.0.0.1/rerank",
    ],
)
def test_any_other_endpoint_is_a_configuration_error(monkeypatch, url) -> None:
    with pytest.raises(RerankConfigurationError) as caught:
        _from_env(monkeypatch, url)
    assert str(caught.value) == MESSAGE
