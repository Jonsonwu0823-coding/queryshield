from __future__ import annotations

from scripts.w02_t06_probe import validate_offline_cases


def test_t06_bypass_proposals_are_checked_without_database() -> None:
    assert validate_offline_cases() == (
        "T06-R1-model-identity-parameter",
        "T06-R2-model-b-filter-under-server-a",
    )
