"""The lineage check pairs a proposed query with its execution by parameter position, without raising on odd keys."""

from __future__ import annotations

import pytest

from queryshield.evaluation.provenance import classify_source_lineage

from test_provenance import _observation

HUGE_KEY = "9" * 5000


def _with_earlier_proposal(params: dict[str, object]) -> dict[str, object]:
    """A first proposal with ``params`` for the same SQL that the executed query (``call-query``) later ran."""

    observation = _observation()
    executed = observation["model_call_records"][0]
    earlier = {**executed, "model_call_id": "call-earlier", "proposal": {**executed["proposal"], "params": params}}
    observation["model_call_records"] = [earlier, *observation["model_call_records"]]
    return observation


def _bound_call_ids(observation: dict[str, object]) -> list[str]:
    result = classify_source_lineage(observation)
    assert result["status"] == "pass", result
    return result["source_paths"]["direct_database_catalog"][0]["model_call_ids"]


@pytest.mark.parametrize(
    "params",
    [{"a": "paid"}, {"²": "paid"}, {"0": "paid", "x": "junk"}, {HUGE_KEY: "paid"}, {"0": "paid", HUGE_KEY: "junk"}],
    ids=["letter", "superscript", "extra-letter-key", "huge", "zero-and-huge"],
)
def test_an_earlier_proposal_with_a_bad_key_is_skipped_not_an_error(params: dict[str, object]) -> None:
    assert _bound_call_ids(_with_earlier_proposal(params)) == ["call-query"]


def test_an_earlier_proposal_with_the_same_ordered_params_is_still_bound() -> None:
    assert _bound_call_ids(_with_earlier_proposal({"0": "paid"})) == ["call-query", "call-earlier"]


@pytest.mark.parametrize("params", [{"1": "paid"}, {"01": "paid"}, {"٠": "paid"}], ids=["gap", "leading-zero", "arabic-indic-zero"])
def test_an_earlier_proposal_whose_keys_are_not_positions_is_not_bound(params: dict[str, object]) -> None:
    """The product runs only keys exactly '0'..'n-1', so such a proposal was never the executed query."""

    assert _bound_call_ids(_with_earlier_proposal(params)) == ["call-query"]


def test_an_earlier_proposal_is_bound_by_position_whatever_the_key_insertion_order() -> None:
    observation = _observation()
    observation["sql_records"][0]["params"] = ["paid", "second"]
    executed = observation["model_call_records"][0]
    executed["proposal"] = {**executed["proposal"], "params": {"0": "paid", "1": "second"}}
    earlier = {**executed, "model_call_id": "call-earlier", "proposal": {**executed["proposal"], "params": {"1": "second", "0": "paid"}}}
    observation["model_call_records"] = [earlier, *observation["model_call_records"]]
    assert _bound_call_ids(observation) == ["call-query", "call-earlier"]


def test_an_earlier_proposal_with_eleven_params_is_bound_in_numeric_not_string_order() -> None:
    values = [f"value-{index}" for index in range(11)]
    observation = _observation()
    observation["sql_records"][0]["params"] = values
    executed = observation["model_call_records"][0]
    executed["proposal"] = {**executed["proposal"], "params": {str(index): value for index, value in enumerate(values)}}
    earlier_params = {str(index): value for index, value in reversed(list(enumerate(values)))}
    earlier = {**executed, "model_call_id": "call-earlier", "proposal": {**executed["proposal"], "params": earlier_params}}
    observation["model_call_records"] = [earlier, *observation["model_call_records"]]
    assert _bound_call_ids(observation) == ["call-query", "call-earlier"]
