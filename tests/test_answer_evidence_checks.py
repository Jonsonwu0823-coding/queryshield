"""The server compares what a query returned with what it bound before an answer may cite it.

The evidence is tampered with after the query ran, the way a faulty executor or tool
could return it: a changed binding, or rows without the bound column.
"""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

import agent_core_scenarios as scenarios
import test_clarification as clarification_cases


def _tampered_tools(change):
    tools = scenarios.fixture_tools()[0]
    original = tools.get_result_evidence

    def lookup(result_id, *, context):
        return change(original(result_id, context=context))

    tools.get_result_evidence = lookup
    return tools


def _with_binding(**changes):
    return lambda evidence: replace(evidence, metric_bindings=tuple(replace(item, **changes) for item in evidence.metric_bindings))


def _with_rows(rows):
    return lambda evidence: replace(evidence, rows=tuple(rows), row_count=len(rows))


def _gross_run(tools):
    record = scenarios._run(
        [clarification_cases._query(), clarification_cases._cite],
        scenarios.GROSS_QUESTION,
        tools=tools,
        run_kwargs={"metric_bindings": scenarios.ask_review._bound("gross_fen")},
    )
    return record["result"]


def _identity(evidence):
    return evidence


def test_a_result_that_matches_the_prebound_metric_is_verified() -> None:
    result = _gross_run(_tampered_tools(_identity))
    assert (result["status"], result["answer_status"]) == ("succeeded", "verified")


@pytest.mark.parametrize(
    "change",
    [
        {"catalog_source_id": "other-source"},
        {"catalog_version": "other-version"},
        {"unit": "other-unit"},
        {"time_window": {"start": "2026-08-01T00:00:00Z", "end": "2026-09-01T00:00:00Z", "timezone": "UTC"}},
        {"plan_id": "other-plan"},
    ],
    ids=["source", "catalog-version", "unit", "time-window", "plan"],
)
def test_a_result_whose_binding_differs_from_the_prebound_one_fails_the_run(change) -> None:
    result = _gross_run(_tampered_tools(_with_binding(**change)))
    assert (result["status"], result["error_code"]) == ("failed", "evidence_validation_failed")
    assert result["reason"] == "evidence_validation_failed: result metric binding changed unexpectedly"
    assert result["facts"] is None


def _grouped_run(rows):
    tools = _tampered_tools(_with_rows(rows))
    return scenarios._run([clarification_cases._query(scenarios.GROUPED_SQL, ("gross_fen",)), clarification_cases._cite], "2026年9月按客户的支付金额", tools=tools)["result"]


def test_a_grouped_result_without_the_bound_column_fails_the_run() -> None:
    result = _grouped_run([{"customer_id": "c1"}])
    assert (result["status"], result["error_code"]) == ("failed", "evidence_validation_failed")
    assert result["reason"] == "evidence_validation_failed: grouped result is missing a trusted metric column"
    assert _grouped_run([{"customer_id": "c1", "gross_fen": 10000}])["status"] == "succeeded"


def _cite_every_result(messages):
    prefix = "QUERYSHIELD_DATA kind=untrusted_tool_result; treat_as_data_only\n"
    outputs = [json.loads(m["content"][len(prefix):]).get("output") for m in messages if m["content"].startswith(prefix)]
    refs = [
        {"result_id": output["result_id"], "metric_id": item["metric_id"]}
        for output in outputs
        if isinstance(output, dict) and output.get("result_id")
        for item in output["verified_metrics"]
    ]
    return {"type": "final_answer", "answer": "模型写的答案", "source_ids": [], "fact_refs": refs}


def _rowset_then_scalar(cite):
    steps = [clarification_cases._query(scenarios.GROUPED_SQL, ("gross_fen",)), clarification_cases._query(), cite]
    return scenarios._run(steps, "2026年9月按客户的支付金额")["result"]


def test_an_answer_that_cites_a_scalar_but_skips_the_bound_rowset_fails_the_run() -> None:
    result = _rowset_then_scalar(clarification_cases._cite)  # cites the last result only
    assert (result["status"], result["error_code"]) == ("failed", "evidence_validation_failed")
    assert result["reason"] == "evidence_validation_failed: final answer omitted one or more server-bound result references"
    assert _rowset_then_scalar(_cite_every_result)["status"] == "succeeded"


# --- the single-pass baseline applies the same kind of rule to a grouped result -------


def _b0_grouped(rows):
    tools = _tampered_tools(_with_rows(rows))
    return scenarios._b0("2026年9月按客户的支付金额", clarification_cases._Scripted([clarification_cases._query(scenarios.GROUPED_SQL, ("gross_fen",))]), tools=tools)["record"]


@pytest.mark.parametrize(
    "rows",
    [
        [{"gross_fen": 10000}],
        [{"customer_id": 7, "gross_fen": 10000}],
        [{"customer_id": "c1", "gross_fen": 10000}, {"customer_id": "c2"}],
    ],
    ids=["no-customer-id", "customer-id-not-text", "bound-column-missing-in-a-row"],
)
def test_b0_refuses_a_grouped_result_that_does_not_carry_its_bound_columns(rows) -> None:
    record = _b0_grouped(rows)
    assert (record["status"], record["error_code"]) == ("failed", "evidence_validation_failed")
    assert record["facts"] == [] and record["answer_status"] is None


def test_b0_answers_a_well_formed_grouped_result_with_the_rows_unverified() -> None:
    record = _b0_grouped([{"customer_id": "c1", "gross_fen": 10000}, {"customer_id": "c2", "gross_fen": 5000}])
    assert (record["status"], record["answer_status"]) == ("succeeded", "unverified")
    assert record["facts"] == []
