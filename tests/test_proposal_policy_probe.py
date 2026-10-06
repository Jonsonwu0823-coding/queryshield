from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from proposal_real_chain_probe import answer_matches_expected_rows  # noqa: E402
from proposal_policy_probe import REJECTION_CASES, SAFE_CASES  # noqa: E402
from queryshield.policy.sql import SQLPolicyError, parse_readonly_select  # noqa: E402
from queryshield.agent import ExecutionContext  # noqa: E402
from queryshield.tools.semantic import ToolError, check_sensitive_access  # noqa: E402


def test_t05_has_five_distinct_safe_queries_and_all_parse() -> None:
    assert len(SAFE_CASES) == 5
    assert len({case.case_id for case in SAFE_CASES}) == 5
    assert all(parse_readonly_select(case.sql).referenced_tables for case in SAFE_CASES)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("查询完成：共返回2行。", True),
        ("查询完成：共返回 2 行。", True),
        ("", False),
        ("查询完成：共返回3行。", False),
    ],
)
def test_real_chain_answer_must_be_present_and_match_observed_row_count(
    answer: str, expected: bool
) -> None:
    assert answer_matches_expected_rows(answer, SAFE_CASES[0].expected_rows) is expected


@pytest.mark.parametrize(
    "case",
    REJECTION_CASES,
    ids=[case.case_id for case in REJECTION_CASES],
)
def test_t05_rejection_queries_match_expected_policy_code(case) -> None:
    with pytest.raises(SQLPolicyError) as error:
        parse_readonly_select(case.sql)

    assert error.value.code == case.expected_error


def _requires_approval_for_requester(sql: str) -> bool:
    context = ExecutionContext(run_id="run-1", tenant_id="A", principal_id="a-requester", role="requester")
    try:
        check_sensitive_access(context, parse_readonly_select(sql))
    except ToolError as error:
        assert error.code == "approval_required"
        return True
    return False


def test_only_the_customer_name_case_is_an_approver_case() -> None:
    needs_approval = [case for case in SAFE_CASES if _requires_approval_for_requester(case.sql)]

    assert [case.case_id for case in needs_approval] == ["T05-Q3-join-customer-orders"]
    assert needs_approval[0].identity == "approver"
    assert all(case.identity == "requester" for case in SAFE_CASES if case not in needs_approval)
