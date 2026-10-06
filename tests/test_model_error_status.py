"""The HTTP status of a model-provider failure, for every error code a provider can raise."""

from __future__ import annotations

import pytest

from queryshield.api.main import _model_error_status


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("upstream_timeout", 504),
        ("missing_model_configuration", 503),
        ("invalid_model_configuration", 503),
        ("invalid_provider_mode", 503),
        ("upstream_request_error", 502),
        ("upstream_http_error", 502),
        ("invalid_response", 502),
        ("invalid_usage", 502),
        ("a_code_nobody_has_seen", 502),
    ],
)
def test_model_error_codes_map_to_their_http_status(code: str, status: int) -> None:
    assert _model_error_status(code) == status
