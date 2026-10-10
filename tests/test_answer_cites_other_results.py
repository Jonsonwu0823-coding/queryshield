"""An answer may cite a result of the same run that the tools hold but this agent state does not.

The facts are resolved from the server-held evidence, and the answer's source ids include the
catalog source of each such fact (otherwise only results in the state would add sources).
"""

from __future__ import annotations

import agent_core_scenarios as core
import test_clarification as clarification_cases
from queryshield.agent.tool_execution import call_tool

from multi_agent_support import SEP

# Rows, no metric: a successful query in the state that binds nothing, so it adds no source.
ROWS_ONLY = {
    "type": "tool_call",
    "name": "query_readonly",
    "arguments": {"sql": "SELECT COUNT(*) AS order_rows FROM orders AS o", "params": {}},
}


def test_a_cited_result_outside_the_state_adds_its_catalog_source() -> None:
    tools = core.fixture_tools()[0]
    context = clarification_cases._context("run-cites-elsewhere")
    elsewhere = call_tool(tools, "query_readonly", clarification_cases._query()["arguments"], context=context, request_time_window=SEP)
    cite = {
        "type": "final_answer",
        "answer": "模型写的答案",
        "source_ids": [],
        "fact_refs": [{"result_id": elsewhere["result_id"], "metric_id": "gross_fen"}],
    }
    result = core.agent_for(clarification_cases._Scripted([ROWS_ONLY, cite]), tools).run(context, "2026年9月支付金额是多少")

    assert (result.status, result.answer_status) == ("succeeded", "verified")
    assert [fact["result_id"] for fact in result.facts["facts"]] == [elsewhere["result_id"]]
    assert result.action["source_ids"] == ["commerce-v1"]
