from scripts.proposal_fs_probe import validate_fs01_record, validate_fs02_record


def test_fs01_requires_two_distinct_result_records() -> None:
    assert validate_fs01_record(
        {
            "case_id": "PROPOSAL-FS01-same-question-two-runs",
            "passed": True,
        }
    )
    assert not validate_fs01_record(
        {
            "case_id": "PROPOSAL-FS01-same-question-two-runs",
            "passed": False,
        }
    )


def test_fs02_requires_all_forgery_and_server_scope_cases() -> None:
    assert validate_fs02_record(
        {
            "passed": True,
            "cases": [{}, {}, {}],
        }
    )
    assert not validate_fs02_record(
        {
            "passed": True,
            "cases": [{}, {}],
        }
    )
