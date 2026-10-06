"""The model, embedding and rerank URLs: a real host is required, and a malformed URL is a configuration error."""

from __future__ import annotations

import pytest

from queryshield.providers.embedding import EmbeddingConfig, EmbeddingConfigurationError
from queryshield.providers.http import base_url_is_valid
from queryshield.providers.openai_compatible import OpenAICompatibleConfig
from queryshield.providers.contracts import ModelProviderError
from queryshield.providers.rerank import HttpRerankAdapter, RerankConfigurationError, _is_https_or_loopback

GOOD = [
    "https://api.example.com/v1",
    "https://203.0.113.9:8443/x",
    "http://127.0.0.1:8000/x",
    "http://localhost/x",
    "http://[::1]:8000/x",
    "http://gateway:8080/v1",
    "https://api.example.com:65535/v1",
]
NO_HOST = ["https://:443/x", "http://:80", "https:///x", "http://@/x", "https://user@:443/x"]
MALFORMED = ["http://[::1", "http://[bad]/x", "https://[/x"]
# urlparse and the host check accept these; httpx raises only when a request is built
NOT_CONNECTABLE = [
    "https://api.example.com:abc/v1",
    "https://api.example.com:99999/v1",
    "https://api.example.com:-1/v1",
    "https://api.example.com:８０/v1",
    "https://xn--/v1",
    "https://[v1.x]/v1",
    "https://\u200b.example.com/v1",
]
# urlparse drops leading spaces and httpx does not: it reads such a URL as a relative one with no host.
# The settings read from the environment strip them first, so from_env still accepts these.
LEADING_SPACE = [" https://api.example.com/v1", "   http://gateway:8080/v1"]
OTHER_BAD = ["ftp://example.com/x", "https://user:secret@example.com/x", "example.com/x", "https://"]


@pytest.mark.parametrize("url", GOOD)
def test_a_url_with_a_host_is_valid(url: str) -> None:
    assert base_url_is_valid(url) is True


@pytest.mark.parametrize("url", NO_HOST + MALFORMED + NOT_CONNECTABLE + LEADING_SPACE + OTHER_BAD)
def test_a_url_without_a_host_or_malformed_is_invalid_not_an_exception(url: str) -> None:
    assert base_url_is_valid(url) is False


def _model_error(url: str) -> str:
    with pytest.raises(ModelProviderError) as caught:
        OpenAICompatibleConfig(base_url=url, api_key="key-for-test", model="model-for-test")
    return caught.value.code


def _embedding_error(url: str) -> str:
    with pytest.raises(EmbeddingConfigurationError) as caught:
        EmbeddingConfig(base_url=url, api_key="key-for-test", model="m", model_revision="r", dimensions=8)
    return f"{caught.value.code}:{caught.value.field_name}"


def _rerank_error(monkeypatch, url: str) -> str:
    monkeypatch.setenv("QUERYSHIELD_RERANK_URL", url)
    monkeypatch.setenv("QUERYSHIELD_RERANK_API_KEY", "key-for-test")
    monkeypatch.setenv("QUERYSHIELD_RERANK_MODEL_NAME", "model-for-test")
    with pytest.raises(RerankConfigurationError) as caught:
        HttpRerankAdapter.from_env()
    return str(caught.value)


@pytest.mark.parametrize("url", NO_HOST + MALFORMED + NOT_CONNECTABLE)
def test_all_three_adapters_report_their_own_configuration_error(monkeypatch, url: str) -> None:
    assert _model_error(url) == "invalid_model_configuration"
    assert _embedding_error(url) == "invalid_embedding_configuration:base_url"
    assert _rerank_error(monkeypatch, url) == "rerank URL must use HTTPS or a loopback HTTP endpoint"


@pytest.mark.parametrize("url", ["https://api.example.com/v1", "http://gateway:8080/v1", "http://[::1]:8000/x"])
def test_the_model_and_embedding_adapters_still_accept_a_url_with_a_host(url: str) -> None:
    assert OpenAICompatibleConfig(base_url=url, api_key="k", model="m").base_url == url
    assert EmbeddingConfig(base_url=url, api_key="k", model="m", model_revision="r", dimensions=8).base_url == url


@pytest.mark.parametrize("url", ["https://rerank.example.com/r", "http://127.0.0.1:8000/r", "http://localhost/r", "http://[::1]:8000/r"])
def test_the_rerank_adapter_still_accepts_https_and_loopback_urls(monkeypatch, url: str) -> None:
    monkeypatch.setenv("QUERYSHIELD_RERANK_URL", url)
    monkeypatch.setenv("QUERYSHIELD_RERANK_API_KEY", "key-for-test")
    monkeypatch.setenv("QUERYSHIELD_RERANK_MODEL_NAME", "model-for-test")
    assert HttpRerankAdapter.from_env().endpoint == url


@pytest.mark.parametrize("url", NO_HOST + MALFORMED + NOT_CONNECTABLE)
def test_the_model_and_embedding_settings_read_from_the_environment_report_the_same_errors(monkeypatch, url: str) -> None:
    for name, value in {
        "QUERYSHIELD_MODEL_BASE_URL": url,
        "QUERYSHIELD_MODEL_API_KEY": "key-for-test",
        "QUERYSHIELD_MODEL_NAME": "model-for-test",
        "QUERYSHIELD_EMBEDDING_BASE_URL": url,
        "QUERYSHIELD_EMBEDDING_API_KEY": "key-for-test",
        "QUERYSHIELD_EMBEDDING_MODEL_NAME": "model-for-test",
        "QUERYSHIELD_EMBEDDING_MODEL_REVISION": "revision-for-test",
        "QUERYSHIELD_EMBEDDING_DIMENSIONS": "8",
    }.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ModelProviderError) as model_error:
        OpenAICompatibleConfig.from_env()
    assert model_error.value.code == "invalid_model_configuration"
    with pytest.raises(EmbeddingConfigurationError) as embedding_error:
        EmbeddingConfig.from_env()
    assert (embedding_error.value.code, embedding_error.value.field_name) == ("invalid_embedding_configuration", "base_url")


@pytest.mark.parametrize("url", LEADING_SPACE)
def test_the_settings_read_from_the_environment_strip_a_leading_space_and_accept_the_url(monkeypatch, url: str) -> None:
    for name, value in {
        "QUERYSHIELD_MODEL_BASE_URL": url,
        "QUERYSHIELD_MODEL_API_KEY": "key-for-test",
        "QUERYSHIELD_MODEL_NAME": "model-for-test",
        "QUERYSHIELD_EMBEDDING_BASE_URL": url,
        "QUERYSHIELD_EMBEDDING_API_KEY": "key-for-test",
        "QUERYSHIELD_EMBEDDING_MODEL_NAME": "model-for-test",
        "QUERYSHIELD_EMBEDDING_MODEL_REVISION": "revision-for-test",
        "QUERYSHIELD_EMBEDDING_DIMENSIONS": "8",
        "QUERYSHIELD_RERANK_URL": url.replace("http://gateway:8080", "http://localhost:8080"),
        "QUERYSHIELD_RERANK_API_KEY": "key-for-test",
        "QUERYSHIELD_RERANK_MODEL_NAME": "model-for-test",
    }.items():
        monkeypatch.setenv(name, value)
    assert OpenAICompatibleConfig.from_env().base_url == url.strip()
    assert EmbeddingConfig.from_env().base_url == url.strip()
    assert HttpRerankAdapter.from_env().endpoint == url.replace("http://gateway:8080", "http://localhost:8080").strip()


@pytest.mark.parametrize("url", LEADING_SPACE)
def test_a_url_with_a_leading_space_is_refused_when_given_to_the_configuration_as_it_is(url: str) -> None:
    assert _model_error(url) == "invalid_model_configuration"
    assert _embedding_error(url) == "invalid_embedding_configuration:base_url"
    assert _is_https_or_loopback(url) is False
