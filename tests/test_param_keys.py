"""Parameter-object keys that are not valid positions are rejected without raising."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from queryshield.agent.proposals import ExecutionContext, ProposalParseError, ToolCallAction
from queryshield.agent.tool_execution import call_tool
from queryshield.api.main import _proposal_params
from queryshield.approval.service import ApprovalConflict, build_pending_action
from queryshield.catalog import load_default_catalog
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.policy.params import ordered_param_values
from queryshield.tools import ControlledTools, ToolError
from queryshield.tools.semantic import _validated_select

HUGE_KEY = "9" * 5000


def test_a_digit_key_past_the_integer_limit_is_not_an_index() -> None:
    assert ordered_param_values({HUGE_KEY: "x"}) is None
    assert ordered_param_values({"0": "a", HUGE_KEY: "x"}) is None
    assert ordered_param_values({"0": "a", "1": "b"}) == ("a", "b")


def test_a_key_that_is_only_a_string_subclass_is_still_not_an_index() -> None:
    class Key(str):
        pass

    assert ordered_param_values({Key("0"): "a"}) is None


def test_a_proposed_query_with_a_huge_digit_key_is_a_parse_error() -> None:
    action = ToolCallAction("query_readonly", {"sql": "SELECT 1", "params": {HUGE_KEY: "A"}})
    with pytest.raises(ProposalParseError) as caught:
        _proposal_params(action)
    assert caught.value.code == "invalid_field"


def test_an_approval_with_a_huge_digit_key_is_rejected_as_malformed() -> None:
    call = {"tool": "query_readonly", "sql": "SELECT 1", "params": {HUGE_KEY: 1}, "metrics": [], "time_window": None}
    with pytest.raises(ApprovalConflict) as caught:
        build_pending_action(call, run_id="r", tenant_id="A", requester_principal_id="p")
    assert caught.value.code == "approval_action_invalid"


# The tool layer's own check of the same keys: a canonical digit string past the integer limit is a
# gap in the indexes; a long one with a leading zero is not a canonical index at all.
CONSECUTIVE = "params keys must be consecutive indexes"
WITHOUT_GAPS = "params keys must start at zero without gaps"
SQL = "SELECT name FROM customers WHERE tenant_id = %s"


def _tool_error(callable_, *args, **kwargs) -> ToolError:
    with pytest.raises(ToolError) as caught:
        callable_(*args, **kwargs)
    return caught.value


@pytest.mark.parametrize(
    "params, message",
    [
        ({HUGE_KEY: "A"}, WITHOUT_GAPS),
        ({"0": "A", HUGE_KEY: "A"}, WITHOUT_GAPS),
        ({"0" + "1" * 5000: "A"}, CONSECUTIVE),
        ({"1" * 5000 + "x": "A"}, CONSECUTIVE),
        ({"٠": "A"}, CONSECUTIVE),
        ({"01": "A"}, CONSECUTIVE),
        ({"2": "A"}, WITHOUT_GAPS),
    ],
    ids=["huge", "zero-and-huge", "huge-leading-zero", "huge-not-digits", "arabic-indic-zero", "leading-zero", "gap"],
)
def test_the_tool_check_reports_a_long_digit_key_with_a_fixed_message(params, message) -> None:
    error = _tool_error(_validated_select, SQL, params)
    assert (error.code, error.message) == ("invalid_argument", message)


def test_the_tool_check_still_reads_ordered_parameters() -> None:
    values, _ = _validated_select("SELECT name FROM customers WHERE tenant_id = %s AND name = %s", {"1": "b", "0": "A"})
    assert values == ("A", "b")


def _never_connect():
    raise AssertionError("a query with malformed parameters must not reach the database")


def test_a_model_proposed_query_with_a_huge_digit_key_gets_a_controlled_tool_error() -> None:
    executor = GuardedQueryExecutor(connect=_never_connect, clock=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc))
    tools = ControlledTools(catalog=load_default_catalog(), executor=executor)
    context = ExecutionContext(run_id="run-keys", tenant_id="A", principal_id="principal-A", role="requester")
    error = _tool_error(call_tool, tools, "query_readonly", {"sql": SQL, "params": {HUGE_KEY: "A"}}, context=context)
    assert (error.code, error.message) == ("invalid_argument", WITHOUT_GAPS)
