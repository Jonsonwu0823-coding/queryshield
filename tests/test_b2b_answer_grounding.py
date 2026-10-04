"""B2b follow-up: a final answer with nothing grounding it fails as answer_not_grounded.

B3c-2: the first such answer is sent back once (answer_bounce, its own
budget); the second fails.  Retrieval sources ground only basis knowledge.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from queryshield.agent.runtime import outcome_for
from queryshield.api.main import app, get_model_provider
from queryshield.approval.service import shared_w04_service

from test_b2b_http_queries import Scripted, ask, env  # noqa: F401  (env is a fixture)


UNGROUNDED_TEXT = "2026年8月没有订单，因此已支付订单数为0，总额为0元。"
UNGROUNDED = json.dumps(
    {"type": "final_answer", "answer": UNGROUNDED_TEXT, "fact_refs": [], "source_ids": []},
    ensure_ascii=False,
)


def _stored_bytes() -> bytes:
    directory = Path(os.environ["QUERYSHIELD_STATE_STORE_PATH"]).parent
    return b"".join(path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file())


def test_outcome_table_maps_answer_not_grounded_to_502() -> None:
    outcome = outcome_for("failed", "answer_not_grounded")
    assert (outcome.run_status, outcome.terminal_state, outcome.http_status) == ("FAILED", "FAILED", 502)


def test_first_call_final_answer_without_grounding_fails_over_http(env) -> None:
    model = Scripted([UNGROUNDED])
    app.dependency_overrides[get_model_provider] = lambda: model

    response = ask(env, "2026年8月已支付订单数和总额是多少？")

    body = response.json()
    assert response.status_code == 502
    assert body["status"] == "FAILED"
    assert body["error"]["code"] == "answer_not_grounded"
    assert body.get("answer") is None and body.get("facts") is None
    assert body["sql_exec_count"] == 0
    # B3c-2: sent back once (the scripted model repeats itself), then terminal.
    assert model.calls == 2
    service = shared_w04_service()
    run = service.store.get_run(body["run_id"])
    events = list(service.store.events(body["run_id"]))
    assert run["status"] == "FAILED"
    assert any(
        event["payload"].get("kind") == "answer_validation" and event["payload"].get("error_code") == "answer_not_grounded"
        for event in events
    )
    assert [event["payload"].get("error_code") for event in events if event["payload"].get("kind") == "answer_bounce"] == [
        "answer_not_grounded"
    ]
    for text in (UNGROUNDED_TEXT, "没有订单"):
        assert text not in response.text
        assert text not in json.dumps([dict(run), events], ensure_ascii=False, default=str)
        assert text.encode("utf-8") not in _stored_bytes()


def test_async_ungrounded_answer_fails_the_same_way(env) -> None:
    app.dependency_overrides[get_model_provider] = lambda: Scripted([UNGROUNDED])
    accepted = ask(env, "2026年8月已支付订单数和总额是多少？", asynchronous=True)
    assert accepted.status_code == 202
    from test_b2b_http_queries import wait

    body = wait(env, accepted.json()["run_id"], {"FAILED", "SUCCEEDED"})
    assert body["status"] == "FAILED" and body["error_code"] == "answer_not_grounded"
    assert UNGROUNDED_TEXT not in json.dumps(body, ensure_ascii=False)


def test_knowledge_answer_with_this_runs_retrieval_source_needs_basis_knowledge(env) -> None:
    answer = "退款后净额按支付金额减去退款金额计算。"
    model = Scripted([
        '{"type":"tool_call","name":"search_catalog","arguments":{"query":"退款后净额的口径","top_k":3}}',
        json.dumps(
            {"type": "final_answer", "answer": answer, "fact_refs": [], "source_ids": [], "basis": "knowledge"},
            ensure_ascii=False,
        ),
    ])
    app.dependency_overrides[get_model_provider] = lambda: model

    response = ask(env, "退款后净额是怎么算的？")

    body = response.json()
    assert response.status_code == 200 and body["status"] == "SUCCEEDED"
    assert body["answer"] == answer
    events = list(shared_w04_service().store.events(body["run_id"]))
    search = next(event["payload"] for event in events if event["payload"].get("tool_name") == "search_catalog")
    assert search["source_ids"]  # the server observed a source in this run


def test_retrieval_alone_no_longer_grounds_a_default_answer(env) -> None:
    # K1 (B3c-2): the same answer without basis knowledge is a business-value
    # answer with no query: sent back once, then 502, and the text never returned.
    answer = "退款后净额按支付金额减去退款金额计算。"
    model = Scripted([
        '{"type":"tool_call","name":"search_catalog","arguments":{"query":"退款后净额的口径","top_k":3}}',
        json.dumps({"type": "final_answer", "answer": answer, "fact_refs": [], "source_ids": []}, ensure_ascii=False),
    ])
    app.dependency_overrides[get_model_provider] = lambda: model

    response = ask(env, "退款后净额是怎么算的？")

    body = response.json()
    assert response.status_code == 502 and body["status"] == "FAILED"
    assert body["error"]["code"] == "answer_not_grounded"
    assert model.calls == 3
    assert answer not in response.text


def _metric_helpers():
    import test_metric_intent as helpers

    return helpers


def test_answer_after_a_successful_query_is_unchanged() -> None:
    h = _metric_helpers()
    tools, connection = h._tools()
    model = h._DynamicModel([
        h._query_step(),
        lambda messages: {"type": "final_answer", "answer": "2 笔，150 元", "source_ids": [], "fact_refs": []},
    ])
    result = h._agent(model, tools).run(h._context("run-b2b-query-then-plain-answer"), "q")
    assert result.status == "succeeded"
    assert result.answer == "2 笔，150 元"
    assert len(connection.executed) == 1

    verified = h._DynamicModel([h._query_step(metrics=["paid_count", "gross_fen"], time_window=h.SEPTEMBER), h._cite_verified])
    result = h._agent(verified, h._tools()[0]).run(h._context("run-b2b-query-verified"), "q")
    assert result.status == "succeeded"
    assert [fact["value"] for fact in result.facts["facts"]] == [2, 15000]


def test_agent_level_rule_is_terminal_and_hides_the_text() -> None:
    h = _metric_helpers()
    model = h._DynamicModel([lambda messages: json.loads(UNGROUNDED), lambda messages: json.loads(UNGROUNDED)])
    result = h._agent(model, h._tools()[0]).run(h._context("run-b2b-ungrounded"), "q")
    assert result.status == "failed"
    assert result.error_code == "answer_not_grounded"
    # B3c-2: one send-back (not the SQL repair), then terminal.
    assert result.repair_count == 0 and result.model_call_count == 2
    assert result.answer is None and result.facts is None
    assert UNGROUNDED_TEXT not in json.dumps(result.as_dict(), ensure_ascii=False, default=str)


def test_b0_single_pass_ungrounded_answer_fails_and_hides_the_text() -> None:
    from queryshield.evaluation.w05_runner import run_b0_single_pass

    h = _metric_helpers()
    output = run_b0_single_pass(
        h._DynamicModel([lambda messages: json.loads(UNGROUNDED)]),
        h._tools()[0],
        h._context("run-b0-ungrounded"),
        "2026年8月已支付订单数和总额是多少？",
    )
    assert output["status"] == "failed"
    assert output["error_code"] == "answer_not_grounded"
    assert output["http_status"] == 502 and output["terminal_state"] == "FAILED"
    assert output["answer"] is None and output["facts"] == []
    assert UNGROUNDED_TEXT not in json.dumps(output, ensure_ascii=False, default=str)
