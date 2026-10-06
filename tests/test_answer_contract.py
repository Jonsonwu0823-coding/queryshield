"""Answer contract: basis declared by the model, checked by the server.

Covers the basis table (query / knowledge / no_data), the one answer send-back
per run, answer_status on every entrypoint, resume HTTP codes and the HTTP
smoke judges.  Model text is a sentinel string that must never come back.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from queryshield.agent import BoundedAgent, ModelCallStore
from queryshield.agent.config import RunConfig
from queryshield.agent.context import build_context
from queryshield.agent.graph import (
    AGENT_CHECKPOINT_VERSION,
    MAX_ANSWER_BOUNCES,
    GraphLimits,
    _V3_AGENT_CHECKPOINT_VERSION,
)
from queryshield.agent.metric_intent import answer_basis_conflict_hint, answer_not_grounded_hint, declarable_metric_ids
from queryshield.agent.proposals import ExecutionContext, ProposalParseError, parse_query_proposal, proposal_shape_summary
from queryshield.agent.runtime import outcome_for
from queryshield.api.main import _answer_fields, app, get_guarded_executor, get_model_provider
from queryshield.approval.service import _answer_envelope, _approved_answer, _succeeded_without_evidence, shared_run_service
from queryshield.catalog import load_default_catalog
from queryshield.facts.render import render_no_data_answer

import test_metric_intent as h
from test_http_queries import APPROVER, REQUESTER, Scripted, ask, auth, env  # noqa: F401  (env is a fixture)


MODEL_TEXT = "模型自己写的哨兵文字MODEL-SENTINEL"
CATALOG = load_default_catalog()
NO_DATA_REPLY = render_no_data_answer([CATALOG.metric_name(m) for m in declarable_metric_ids(CATALOG)])


def _final(*, basis=None, fact_refs=(), source_ids=("model-written-source",), answer=MODEL_TEXT):
    payload = {"type": "final_answer", "answer": answer, "source_ids": list(source_ids), "fact_refs": list(fact_refs)}
    if basis is not None:
        payload["basis"] = basis
    return lambda messages: payload


def _search(messages):
    return {"type": "tool_call", "name": "search_catalog", "arguments": {"query": "退款后净额", "top_k": 3}}


def _describe(messages):
    return {"type": "tool_call", "name": "describe_tables", "arguments": {"tables": ["orders"]}}


def _invalid_query(messages):
    # Repairable parser failure: the query never runs.
    return {"type": "tool_call", "name": "query_readonly", "arguments": {"sql": "SELECT FROM", "params": {}}}


def _cite_fake(messages):
    return {
        "type": "final_answer",
        "answer": MODEL_TEXT,
        "source_ids": [],
        "fact_refs": [{"result_id": "result-not-in-this-run", "metric_id": "gross_fen"}],
    }


SCALAR = h._query_step(metrics=["paid_count", "gross_fen"], time_window=h.SEPTEMBER)
NON_METRIC = h._query_step()  # no declaration: rows are not facts
METRIC_ITEM = next(item for item in CATALOG.search_items() if item["id"] == "metric.net_fen")
TABLE_ITEM = next(item for item in CATALOG.search_items() if item["id"] == "table.orders")


class _GroupedCursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def execute(self, sql, params):
        self.connection.executed.append((sql, tuple(params)))

    def fetchmany(self, size):
        return [{"customer_id": "c1", "gross_fen": 10000}, {"customer_id": "c2", "gross_fen": 5000}][:size]


class _GroupedConnection(h._Connection):
    def cursor(self, *, row_factory):
        return _GroupedCursor(self)


GROUPED_SQL = (
    "SELECT o.customer_id, SUM(o.amount_fen) AS gross_fen FROM orders AS o INNER JOIN customers AS c "
    "ON o.tenant_id = c.tenant_id AND o.customer_id = c.customer_id "
    "WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s GROUP BY o.customer_id"
)


def _run(steps, *, connection=None, question="q", run_id="run-b3c2", initial_items=(), limits=None):
    tools, conn = h._tools(connection)
    agent = BoundedAgent(h._DynamicModel(steps), tools=tools, call_store=ModelCallStore(), limits=limits)
    result = agent.run(h._context(run_id), question, _evaluation_initial_retrieval_items=initial_items)
    return result, conn


def _events(result, kind):
    return [event for event in result.events if event.get("kind") == kind]


def _assert_no_model_text(result):
    encoded = json.dumps(result.as_dict(), ensure_ascii=False, default=str)
    assert MODEL_TEXT not in encoded
    assert "model-written-source" not in encoded


# --- the basis table, accepted shapes --------------------------------------------


def test_scalar_facts_are_server_rendered_and_verified():
    result, _ = _run([SCALAR, h._cite_verified])
    assert result.status == "succeeded" and result.answer_status == "verified"
    assert result.answer.startswith("已核实：")
    assert _events(result, "answer")[0]["status"] == "verified"
    assert result.action["basis"] == "query"


def test_rowset_answer_keeps_the_model_summary_and_is_unverified():
    def cite_rowset(messages):
        output = h._last_tool_output(messages)
        return {
            "type": "final_answer",
            "answer": MODEL_TEXT,
            "source_ids": ["model-written-source"],
            "fact_refs": [{"result_id": output["result_id"], "metric_id": "gross_fen"}],
        }

    grouped = h._query_step(sql=GROUPED_SQL, metrics=["gross_fen"], time_window=h.SEPTEMBER)
    result, _ = _run([grouped, cite_rowset], connection=_GroupedConnection())
    assert result.status == "succeeded"
    assert result.answer == MODEL_TEXT and result.facts["facts"] == []
    # Before the answer contract the answer event said "verified" here (fact_refs present).
    assert result.answer_status == "unverified" and _events(result, "answer")[0]["status"] == "unverified"
    assert result.action["source_ids"] == ["commerce-v1"]  # the server's, not the model's


def test_non_metric_query_answer_is_unverified_with_server_source_ids():
    result, _ = _run([NON_METRIC, _final()])
    assert result.status == "succeeded" and result.answer == MODEL_TEXT
    assert result.answer_status == "unverified"
    assert result.action["source_ids"] == []


@pytest.mark.parametrize("ground", ["search", "prepared"])
def test_knowledge_with_a_retrieval_source_is_returned_unverified(ground):
    steps = [_search, _final(basis="knowledge")] if ground == "search" else [_final(basis="knowledge")]
    items = () if ground == "search" else (METRIC_ITEM,)
    result, connection = _run(steps, initial_items=items)
    assert result.status == "succeeded" and result.answer == MODEL_TEXT
    assert result.answer_status == "unverified"
    assert result.action["source_ids"] == ["commerce-v1"]
    assert result.facts["facts"] == [] and connection.executed == []


@pytest.mark.parametrize("before", [[], [_search], [_describe]], ids=["nothing", "search", "describe"])
def test_no_data_returns_the_fixed_reply_and_never_the_model_text(before):
    result, connection = _run([*before, _final(basis="no_data")])
    assert result.status == "succeeded" and result.answer_status == "no_data"
    assert result.answer == NO_DATA_REPLY
    assert result.action["answer"] == NO_DATA_REPLY and result.action["source_ids"] == []
    assert connection.executed == []
    for metric_id in declarable_metric_ids(CATALOG):
        assert CATALOG.metric_name(metric_id) in result.answer
    _assert_no_model_text(result)


# --- the basis table, one send-back then terminal ----------------------------------


UNGROUNDED_CASES = {
    # Table structure or retrieval alone no longer grounds a business-value answer.
    "describe_only": ([_describe], (), "answer_not_grounded"),
    "search_only": ([_search], (), "answer_not_grounded"),
    "prepared_only": ([], (METRIC_ITEM,), "answer_not_grounded"),
    "nothing": ([], (), "answer_not_grounded"),
    "failed_query_only": ([_invalid_query], (), "answer_not_grounded"),
}


@pytest.mark.parametrize("case", sorted(UNGROUNDED_CASES))
def test_default_answer_without_a_query_is_sent_back_once_then_fails(case):
    before, items, code = UNGROUNDED_CASES[case]
    result, connection = _run([*before, _final(), _final()], initial_items=items)
    assert result.status == "failed" and result.error_code == code
    assert result.answer is None and result.facts is None and result.answer_status is None
    assert [event["error_code"] for event in _events(result, "answer_bounce")] == [code]
    assert result.events[-1]["kind"] == "answer_validation"
    assert connection.executed == []
    _assert_no_model_text(result)


def test_the_send_back_recovers_when_the_model_then_queries():
    result, connection = _run([_final(), SCALAR, h._cite_verified])
    assert result.status == "succeeded" and result.answer_status == "verified"
    assert result.repair_count == 0 and len(connection.executed) == 1


KNOWLEDGE_AND_NO_DATA_REJECTIONS = {
    "knowledge_describe_only": ([_describe], (), _final(basis="knowledge"), "answer_not_grounded"),
    "knowledge_table_entry_only": ([], (TABLE_ITEM,), _final(basis="knowledge"), "answer_not_grounded"),
    "knowledge_nothing": ([], (), _final(basis="knowledge"), "answer_not_grounded"),
    "knowledge_with_fact_refs": ([SCALAR], (), _final(basis="knowledge", fact_refs=[{"result_id": "x", "metric_id": "gross_fen"}]), "answer_basis_conflict"),
    "no_data_after_query": ([NON_METRIC], (), _final(basis="no_data"), "answer_basis_conflict"),
    "no_data_after_failed_query": ([_invalid_query], (), _final(basis="no_data"), "answer_basis_conflict"),
    "no_data_with_fact_refs": ([], (), _final(basis="no_data", fact_refs=[{"result_id": "x", "metric_id": "gross_fen"}]), "answer_basis_conflict"),
}


@pytest.mark.parametrize("case", sorted(KNOWLEDGE_AND_NO_DATA_REJECTIONS))
def test_knowledge_and_no_data_rejections(case):
    before, items, answer, code = KNOWLEDGE_AND_NO_DATA_REJECTIONS[case]
    result, _ = _run([*before, answer, answer], initial_items=items)
    assert result.status == "failed" and result.error_code == code
    if case.startswith("knowledge_") and code == "answer_not_grounded":
        # The server searched once for the question ("q"); nothing usable
        # came back, so no send-back.
        server = [event for event in _events(result, "tool_call") if event.get("initiated_by") == "server"]
        assert [(event["status"], event["grounding_source_count"]) for event in server] == [("succeeded", 0)]
        assert _events(result, "answer_bounce") == []
    else:
        assert [event["error_code"] for event in _events(result, "answer_bounce")] == [code]
    assert result.answer is None and result.facts is None
    _assert_no_model_text(result)
    assert outcome_for("failed", code).http_status == 502


def test_any_second_basis_error_is_terminal_even_of_another_kind():
    result, _ = _run([_final(), _final(basis="no_data", fact_refs=[{"result_id": "x", "metric_id": "gross_fen"}])])
    assert result.status == "failed" and result.error_code == "answer_basis_conflict"
    assert len(_events(result, "answer_bounce")) == MAX_ANSWER_BOUNCES == 1


def test_citing_results_before_any_query_uses_the_answer_bounce_not_the_repair():
    result, _ = _run([_cite_fake, SCALAR, h._cite_verified])
    assert result.status == "succeeded"
    assert result.repair_count == 0
    assert [event["error_code"] for event in _events(result, "answer_bounce")] == ["answer_without_query_result"]


def test_bound_result_not_cited_stays_terminal_without_a_send_back():
    # Table row 5: unchanged by the answer contract, for basis query and knowledge alike.
    for answer in (_final(), _final(basis="knowledge")):
        result, _ = _run([_search, SCALAR, answer, answer])
        assert result.status == "failed" and result.error_code == "evidence_validation_failed"
        assert _events(result, "answer_bounce") == []


def test_undeclared_metric_still_uses_the_sql_repair():
    def cite_undeclared(messages):
        output = h._last_tool_output(messages)
        return {"type": "final_answer", "answer": "x", "source_ids": [], "fact_refs": [{"result_id": output["result_id"], "metric_id": "gross_fen"}]}

    result, _ = _run([NON_METRIC, cite_undeclared, SCALAR, h._cite_verified])
    assert result.status == "succeeded" and result.repair_count == 1
    assert _events(result, "answer_bounce") == []


# --- hints: fixed text, business values first ------------------------------------


def _hint_records(result_or_messages):
    return [record for record in h._tool_records(result_or_messages) if record.get("tool_name") == "final_answer"]


def test_send_back_hint_puts_the_query_first_and_echoes_no_model_text():
    model = h._DynamicModel([_search, _final(), SCALAR, h._cite_verified])
    tools, _ = h._tools()
    result = BoundedAgent(model, tools=tools, call_store=ModelCallStore()).run(
        h._context("run-b3c2-hint"), "q", request_time_window={**h.SEPTEMBER, "timezone": "UTC"}
    )
    assert result.status == "succeeded"
    hint = _hint_records(model.messages[2])[-1]
    assert hint["error_code"] == "answer_not_grounded" and hint["repairable"] is True
    keys = list(hint["repair_hint"])
    assert keys[0] == "action"
    assert "always need a query" in hint["repair_hint"]["action"] and "reported as 0" in hint["repair_hint"]["action"]
    assert hint["repair_hint"]["only_if_definition"].startswith("Only if")
    assert hint["repair_hint"]["only_if_no_data_needed"].startswith("Only if")
    assert hint["repair_hint"]["request_time_window"] == h.SEPTEMBER
    assert MODEL_TEXT not in json.dumps(model.messages[2], ensure_ascii=False).split("QUERYSHIELD_DATA kind=untrusted_tool_result")[-1]


def test_hint_without_a_retriever_does_not_offer_knowledge():
    hint = answer_not_grounded_hint(None, retrieval_available=False)
    assert "only_if_definition" not in hint and "search_catalog" not in json.dumps(hint)
    assert list(answer_basis_conflict_hint(None))[0] == "action"


# --- budgets ------------------------------------------------------------------------


def test_single_repair_budget_shape_keeps_its_one_sql_repair():
    fault = h._query_step(sql=h.GROSS_SQL, params={"0": "paid"}, metrics=["gross_fen"], time_window=h.SEPTEMBER)
    good = h._query_step(sql=h.GROSS_SQL, metrics=["gross_fen"], time_window=h.SEPTEMBER)
    result, _ = _run([_final(), fault, good, h._cite_verified], question="2026年9月支付金额")
    assert result.status == "succeeded" and result.answer_status == "verified"
    assert result.repair_count == 1  # the frozen case allows exactly one repair call
    assert result.model_call_count == 4
    assert len(_events(result, "answer_bounce")) == 1


def test_worst_chain_uses_exactly_six_model_calls():
    ask = lambda messages: {"type": "ask_user", "clarification_id": "clarify.metric_basis", "question": "按支付金额还是退款后净额？"}
    fault = h._query_step(sql=h.GROSS_SQL, params={"0": "paid"}, metrics=["gross_fen"], time_window=h.SEPTEMBER)
    good = h._query_step(sql=h.GROSS_SQL, metrics=["gross_fen"], time_window=h.SEPTEMBER)
    result, _ = _run([_search, ask, _final(), fault, good, h._cite_verified], question="2026年9月支付金额是多少")
    assert result.status == "succeeded", (result.error_code, [e["kind"] for e in result.events])
    assert result.model_call_count == 6 and result.repair_count == 1
    assert len(_events(result, "clarification_bounce")) == 1 and len(_events(result, "answer_bounce")) == 1


def test_no_send_back_when_no_model_call_is_left():
    limits = GraphLimits(max_model_calls=2)
    result, _ = _run([_search, _final()], limits=limits)
    # Fails with the answer's own code, not a model_call_limit stop.
    assert result.status == "failed" and result.error_code == "answer_not_grounded"
    assert _events(result, "answer_bounce") == []


# --- checkpoints ---------------------------------------------------------------------


def _ask_time(messages):
    return {"type": "ask_user", "question": "请问要看哪个月？"}


def test_bounce_count_is_checkpointed_and_not_reset_by_resume():
    tools, _ = h._tools()
    context = h._context("run-b3c2-checkpoint")
    runtime = BoundedAgent(h._DynamicModel([_final(), _ask_time]), tools=tools, call_store=ModelCallStore())
    assert runtime.run(context, "支付金额是多少").status == "waiting_user"
    checkpoint = runtime.export_waiting_checkpoint(context.run_id)
    assert checkpoint["checkpoint_version"] == AGENT_CHECKPOINT_VERSION == "qs-bounded-agent-checkpoint-v4"
    assert checkpoint["answer_bounce_count"] == 1

    resumed = BoundedAgent(h._DynamicModel([_final()]), tools=tools, call_store=ModelCallStore(), run_config=runtime.run_config)
    result = resumed.resume_from_checkpoint(context, "2026年9月", checkpoint)
    # The budget spent before the pause still counts: terminal, not sent back again.
    assert result.status == "failed" and result.error_code == "answer_not_grounded"
    assert len([e for e in result.events if e.get("kind") == "answer_bounce"]) == 1


def test_v3_checkpoints_restore_with_an_unspent_answer_bounce():
    tools, _ = h._tools()
    context = h._context("run-b3c2-v3")
    runtime = BoundedAgent(h._DynamicModel([_ask_time]), tools=tools, call_store=ModelCallStore())
    runtime.run(context, "支付金额是多少")
    checkpoint = runtime.export_waiting_checkpoint(context.run_id)
    legacy = {key: value for key, value in checkpoint.items() if key != "answer_bounce_count"}
    legacy["checkpoint_version"] = _V3_AGENT_CHECKPOINT_VERSION
    resumed = BoundedAgent(
        h._DynamicModel([_final(), h._query_step(sql=h.GROSS_SQL, metrics=["gross_fen"], time_window=h.SEPTEMBER), h._cite_verified]),
        tools=tools,
        call_store=ModelCallStore(),
        run_config=runtime.run_config,
    )
    result = resumed.resume_from_checkpoint(context, "2026年9月", legacy)
    assert result.status == "succeeded" and len(_events(result, "answer_bounce")) == 1


def test_eval_prepared_fixtures_stay_on_v3():
    checkpoint = BoundedAgent.prepared_waiting_user_checkpoint(
        h._context("run-b3c2-prepared"), "销售额是多少？", run_config=RunConfig()
    )
    assert checkpoint["checkpoint_version"] == _V3_AGENT_CHECKPOINT_VERSION
    assert "answer_bounce_count" not in checkpoint


def test_a_v4_checkpoint_must_carry_the_count():
    tools, _ = h._tools()
    context = h._context("run-b3c2-v4-missing")
    runtime = BoundedAgent(h._DynamicModel([_ask_time]), tools=tools, call_store=ModelCallStore())
    runtime.run(context, "支付金额是多少")
    checkpoint = dict(runtime.export_waiting_checkpoint(context.run_id))
    checkpoint.pop("answer_bounce_count")
    from queryshield.agent.graph import RunResumeError

    with pytest.raises(RunResumeError):
        BoundedAgent(h._DynamicModel([]), tools=tools, call_store=ModelCallStore()).resume_from_checkpoint(context, "x", checkpoint)


# --- parsing and diagnostics ----------------------------------------------------------


def _parse(payload):
    return parse_query_proposal(json.dumps(payload, ensure_ascii=False), context=h._context(), model_call_id="call-1")


@pytest.mark.parametrize(("basis", "expected"), [(None, "query"), ("query", "query"), ("knowledge", "knowledge"), ("no_data", "no_data")])
def test_basis_values_and_null(basis, expected):
    payload = {"type": "final_answer", "answer": "a", "source_ids": [], "fact_refs": [], "basis": basis}
    assert _parse(payload).action.basis == expected
    without = {key: value for key, value in payload.items() if key != "basis"}
    assert _parse(without).action.basis == "query"


@pytest.mark.parametrize("basis", ["data", "", 1, ["query"], {"x": 1}])
def test_invalid_basis_is_a_parse_error(basis):
    with pytest.raises(ProposalParseError) as caught:
        _parse({"type": "final_answer", "answer": "a", "source_ids": [], "fact_refs": [], "basis": basis})
    assert caught.value.code == "invalid_field"


def test_only_final_answer_takes_basis():
    with pytest.raises(ProposalParseError) as caught:
        _parse({"type": "ask_user", "question": "哪个月？", "basis": "query"})
    assert caught.value.code == "unknown_field"


def test_shape_summary_records_the_basis_type_but_never_its_value():
    raw = json.dumps({"type": "final_answer", "answer": "a", "source_ids": [], "fact_refs": [], "basis": MODEL_TEXT}, ensure_ascii=False)
    summary = proposal_shape_summary(raw)
    assert summary["basis"] == "string" and summary["basis_known"] is False
    assert MODEL_TEXT not in json.dumps(summary, ensure_ascii=False)
    assert proposal_shape_summary(json.dumps({"type": "final_answer", "basis": None}))["basis"] == "null"
    assert proposal_shape_summary(json.dumps({"type": "final_answer"}))["basis"] == "absent"


def test_basis_rule_follows_the_retriever():
    with_retrieval = build_context(h._context(), "q", retrieval_available=True).messages[0]["content"]
    without = build_context(h._context(), "q", retrieval_available=False).messages[0]["content"]
    rule = json.loads(with_retrieval.split("\n", 1)[1])["action_contract"]["actions"]["final_answer"]["basis"]
    assert "knowledge" in rule and "no_data" in rule and rule.startswith("Optional; default query")
    rule = json.loads(without.split("\n", 1)[1])["action_contract"]["actions"]["final_answer"]["basis"]
    assert "knowledge" not in rule and "no_data" in rule


# --- B0 ------------------------------------------------------------------------------


def _b0(payload):
    from queryshield.evaluation.profile_runner import run_b0_single_pass

    return run_b0_single_pass(h._DynamicModel([lambda messages: payload]), h._tools()[0], h._context("run-b3c2-b0"), "你好")


def test_b0_no_data_succeeds_with_the_fixed_reply():
    output = _b0({"type": "final_answer", "answer": MODEL_TEXT, "source_ids": [], "fact_refs": [], "basis": "no_data"})
    assert output["status"] == "succeeded" and output["http_status"] == 200
    assert output["answer"] == NO_DATA_REPLY and output["answer_status"] == "no_data"
    assert MODEL_TEXT not in json.dumps(output, ensure_ascii=False, default=str)


@pytest.mark.parametrize(
    ("basis", "fact_refs", "code"),
    [
        ("knowledge", [], "answer_not_grounded"),
        ("query", [], "answer_not_grounded"),
        ("no_data", [{"result_id": "x", "metric_id": "gross_fen"}], "answer_basis_conflict"),
        ("knowledge", [{"result_id": "x", "metric_id": "gross_fen"}], "answer_basis_conflict"),
    ],
)
def test_b0_rejections_hide_the_text(basis, fact_refs, code):
    output = _b0({"type": "final_answer", "answer": MODEL_TEXT, "source_ids": [], "fact_refs": fact_refs, "basis": basis})
    assert output["status"] == "failed" and output["error_code"] == code and output["http_status"] == 502
    assert output["answer"] is None and output["answer_status"] is None
    assert MODEL_TEXT not in json.dumps(output, ensure_ascii=False, default=str)


def test_b0_scalar_is_verified():
    from queryshield.evaluation.profile_runner import run_b0_single_pass

    step = h._query_step(sql=h.GROSS_SQL, metrics=["gross_fen"], time_window=h.SEPTEMBER)
    output = run_b0_single_pass(h._DynamicModel([step]), h._tools()[0], h._context("run-b3c2-b0-scalar"), "2026年9月支付金额")
    assert output["status"] == "succeeded" and output["answer_status"] == "verified"


# --- persistence and HTTP --------------------------------------------------------------


def test_envelope_fields_only_for_a_succeeded_answer():
    assert _answer_envelope({"answer": "x", "answer_status": "verified"}, succeeded=False) == {
        "answer_status": None,
        "answer_source_ids": None,
    }
    assert _answer_envelope({"answer": None, "answer_status": "verified"}, succeeded=True)["answer_status"] is None
    assert _answer_envelope({"answer": "x", "answer_status": "bogus"}, succeeded=True)["answer_status"] == "unverified"
    stored = _answer_envelope(
        {"answer": "x", "answer_status": "unverified", "action": {"type": "final_answer", "source_ids": ["s1"]}}, succeeded=True
    )
    assert stored == {"answer_status": "unverified", "answer_source_ids": ["s1"]}


def test_public_fields_for_old_runs_are_unverified_never_verified():
    old = {"status": "SUCCEEDED", "answer": "x", "checkpoint": {}}
    assert _answer_fields(old) == {"answer_status": "unverified", "source_ids": []}
    assert _answer_fields({"status": "FAILED", "answer": "note", "checkpoint": {"answer_status": "verified"}}) == {
        "answer_status": None,
        "source_ids": None,
    }


def test_result_without_evidence_is_only_open_for_a_succeeded_answer_without_facts():
    assert _succeeded_without_evidence({"status": "SUCCEEDED", "result": None, "facts": None})
    assert not _succeeded_without_evidence({"status": "SUCCEEDED", "result": None, "facts": {"facts": [{"x": 1}]}})
    assert not _succeeded_without_evidence({"status": "RUNNING", "result": None, "facts": None})
    assert not _succeeded_without_evidence({"status": "WAITING_USER", "result": None, "facts": None})


def test_approved_answer_without_a_fact_is_the_fixed_row_count_text():
    grouped = SimpleNamespace(result_id="r1", row_count=2, metric_bindings=("bound",))
    assert _approved_answer([], grouped) == "审批通过，已执行只读查询：返回 2 行，见 result.rows。"
    names = SimpleNamespace(result_id="r2", row_count=3, metric_bindings=())
    assert _approved_answer([], names) == "审批通过，已执行只读查询：返回 3 行，见 result.rows。"


def test_http_sync_answers_carry_answer_status(env):
    from queryshield.providers.fake_model import FakeModel

    app.dependency_overrides[get_model_provider] = lambda: FakeModel()
    body = ask(env, "2026年9月已支付订单总额").json()
    assert body["status"] == "SUCCEEDED" and body["answer_status"] == "verified" and body["source_ids"] == ["commerce-v1"]
    result = env.get(f"/runs/{body['run_id']}/result", headers=auth(REQUESTER)).json()
    assert result["answer_status"] == "verified"
    waiting = ask(env, "2026年9月销售额是多少？").json()
    assert waiting["status"] == "WAITING_USER" and waiting["answer_status"] is None


def test_http_no_data_and_its_result(env):
    app.dependency_overrides[get_model_provider] = lambda: Scripted(
        [json.dumps({"type": "final_answer", "answer": MODEL_TEXT, "source_ids": [], "fact_refs": [], "basis": "no_data"}, ensure_ascii=False)]
    )
    response = ask(env, "你好，你能做什么？")
    body = response.json()
    assert response.status_code == 200 and body["status"] == "SUCCEEDED"
    assert body["answer"] == NO_DATA_REPLY and body["answer_status"] == "no_data" and body["source_ids"] == []
    assert body["sql_exec_count"] == 0 and body["facts"] is None
    result = env.get(f"/runs/{body['run_id']}/result", headers=auth(REQUESTER))
    assert result.status_code == 200
    assert result.json()["result"] is None and result.json()["answer_status"] == "no_data"
    assert MODEL_TEXT not in response.text + result.text


def test_http_knowledge_answer_is_unverified_with_server_sources(env):
    app.dependency_overrides[get_model_provider] = lambda: Scripted([
        '{"type":"tool_call","name":"search_catalog","arguments":{"query":"退款后净额的口径","top_k":3}}',
        json.dumps({"type": "final_answer", "answer": "定义", "source_ids": ["model-written-source"], "fact_refs": [], "basis": "knowledge"}, ensure_ascii=False),
    ])
    body = ask(env, "退款后净额是怎么算的？").json()
    assert body["status"] == "SUCCEEDED" and body["answer_status"] == "unverified"
    assert body["source_ids"] and "model-written-source" not in body["source_ids"]
    assert env.get(f"/runs/{body['run_id']}/result", headers=auth(REQUESTER)).json()["answer_status"] == "unverified"


def test_http_approval_of_names_is_unverified(env):
    app.dependency_overrides[get_model_provider] = lambda: Scripted([
        json.dumps({"type": "tool_call", "name": "query_readonly", "arguments": {"sql": "SELECT c.customer_id, c.name FROM customers AS c", "params": {}}}),
    ])
    pending = ask(env, "查询本租户所有客户的姓名").json()
    assert pending["status"] == "WAITING_APPROVAL" and pending["answer_status"] is None
    approved = env.post(
        f"/runs/{pending['run_id']}/approval",
        headers=auth(APPROVER),
        json={"approval_id": pending["approval_id"], "decision": "approve"},
    ).json()
    assert approved["status"] == "SUCCEEDED" and approved["answer_status"] == "unverified" and approved["source_ids"] == []
    result = env.get(f"/runs/{pending['run_id']}/result", headers=auth(REQUESTER)).json()
    assert result["answer_status"] == "unverified" and result["facts"] is None


def test_http_empty_window_bounce_then_query_reports_zero(env):
    class _ZeroCursor(h._Cursor):
        def execute(self, sql, params):
            self.connection.executed.append((sql, tuple(params)))
            self.rows = [{"paid_count": 0, "gross_fen": 0}]

    class _ZeroConnection(h._Connection):
        def cursor(self, *, row_factory):
            return _ZeroCursor(self)

    from queryshield.db.guarded import GuardedQueryExecutor

    connection = _ZeroConnection()
    app.dependency_overrides[get_guarded_executor] = lambda: GuardedQueryExecutor(connect=lambda: connection)
    august = {"start": "2026-08-01T00:00:00Z", "end": "2026-09-01T00:00:00Z"}
    direct = json.dumps({"type": "final_answer", "answer": "8月没有订单，0笔。", "source_ids": [], "fact_refs": []}, ensure_ascii=False)
    query = json.dumps({
        "type": "tool_call",
        "name": "query_readonly",
        "arguments": {
            "sql": h.DUAL_SQL,
            "params": {"0": "paid", "1": august["start"], "2": august["end"]},
            "metrics": ["paid_count", "gross_fen"],
            "time_window": august,
        },
    })

    class _Model(Scripted):
        def complete(self, messages, **kwargs):
            if self.calls == 2:
                output = h._last_tool_output(messages)
                self.outputs.append(json.dumps({
                    "type": "final_answer", "answer": "x", "source_ids": [],
                    "fact_refs": [{"result_id": output["result_id"], "metric_id": m["metric_id"]} for m in output["verified_metrics"]],
                }))
            return super().complete(messages, **kwargs)

    model = _Model([direct, query])
    app.dependency_overrides[get_model_provider] = lambda: model
    response = ask(env, "2026年8月没有订单时已支付订单数和总额是多少", time_window={**august, "timezone": "UTC"})
    body = response.json()
    assert response.status_code == 200 and body["status"] == "SUCCEEDED", body
    assert sorted((f["metric_id"], f["value"]) for f in body["facts"]["facts"]) == [("gross_fen", 0), ("paid_count", 0)]
    assert body["answer_status"] == "verified" and model.calls == 3
    events = list(shared_run_service().store.events(body["run_id"]))
    assert [e["payload"].get("error_code") for e in events if e["payload"].get("kind") == "answer_bounce"] == ["answer_not_grounded"]
    assert "没有订单" not in json.dumps(events, ensure_ascii=False, default=str)


# --- resume HTTP codes -------------------------------------------------------------------


def _waiting_on_metric_basis(env, outputs):
    app.dependency_overrides[get_model_provider] = lambda: Scripted(outputs)
    body = ask(env, "2026年9月销售额是多少？").json()
    assert body["status"] == "WAITING_USER", body
    return body["run_id"]


ASK = json.dumps({"type": "ask_user", "clarification_id": "clarify.metric_basis", "question": "按支付金额还是退款后净额？"}, ensure_ascii=False)


def _resume(env, run_id, answer, outputs):
    app.dependency_overrides[get_model_provider] = lambda: Scripted(outputs)
    return env.post(f"/runs/{run_id}/resume", headers=auth(REQUESTER), json={"answer": answer})


def test_resume_success_stays_200(env):
    from queryshield.providers.fake_model import FakeModel

    run_id = _waiting_on_metric_basis(env, [ASK])
    app.dependency_overrides[get_model_provider] = lambda: FakeModel()
    response = env.post(f"/runs/{run_id}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额统计"})
    assert response.status_code == 200 and response.json()["status"] == "SUCCEEDED"
    assert response.json()["answer_status"] == "verified"


def test_resume_still_waiting_stays_200(env):
    # clarification-context-ambiguous: the answer chooses no value, the run keeps waiting.
    run_id = _waiting_on_metric_basis(env, [ASK])
    response = _resume(env, run_id, "2026年9月", [ASK])
    assert response.status_code == 200 and response.json()["status"] == "WAITING_USER"
    assert response.json()["answer_status"] is None and "error" not in response.json()


def test_resume_into_approval_stays_200(env):
    # A time ask binds no metric; after resume the model asks for customer names.
    app.dependency_overrides[get_model_provider] = lambda: Scripted(['{"type":"ask_user","question":"请问要查哪个月？"}'])
    waiting = ask(env, "查询客户姓名").json()
    assert waiting["status"] == "WAITING_USER", waiting
    names = json.dumps({"type": "tool_call", "name": "query_readonly", "arguments": {"sql": "SELECT c.customer_id, c.name FROM customers AS c", "params": {}}})
    response = _resume(env, waiting["run_id"], "2026年9月", [names])
    assert response.status_code == 200 and response.json()["status"] == "WAITING_APPROVAL"


def test_resume_unsupported_scope_is_422_with_the_catalog_note(env):
    app.dependency_overrides[get_model_provider] = lambda: Scripted([
        json.dumps({"type": "ask_user", "clarification_id": "clarify.order_status_scope", "question": "统计哪些订单状态？"}, ensure_ascii=False)
    ])
    body = ask(env, "2026年9月订单数是多少？").json()
    if body["status"] != "WAITING_USER":
        pytest.skip("the catalog rule does not wait on this wording")
    response = _resume(env, body["run_id"], "包含已取消", [ASK])
    payload = response.json()
    assert response.status_code == 422 and payload["status"] == "FAILED"
    assert payload["error"]["code"] == "clarification_value_unsupported" and payload["run_id"] == body["run_id"]
    assert payload["answer"] and payload["answer_status"] is None


def test_resume_model_failure_is_502(env):
    run_id = _waiting_on_metric_basis(env, [ASK])
    ungrounded = json.dumps({"type": "final_answer", "answer": MODEL_TEXT, "source_ids": [], "fact_refs": []}, ensure_ascii=False)
    response = _resume(env, run_id, "按支付金额", [ungrounded])
    payload = response.json()
    assert response.status_code == 502 and payload["status"] == "FAILED"
    assert payload["error"]["code"] == "answer_not_grounded" and payload["run_id"] == run_id
    assert MODEL_TEXT not in response.text


def test_resume_denied_is_403(env):
    run_id = _waiting_on_metric_basis(env, [ASK])
    response = _resume(env, run_id, "按支付金额", ['{"type":"deny","reason":"no"}'])
    assert response.status_code == 403 and response.json()["status"] == "DENIED"
    assert response.json()["run_id"] == run_id


def test_resume_limit_reached_is_502(env):
    run_id = _waiting_on_metric_basis(env, [ASK])
    search = '{"type":"tool_call","name":"search_catalog","arguments":{"query":"x","top_k":1}}'
    response = _resume(env, run_id, "按支付金额", [search])
    assert response.status_code == 502 and response.json()["status"] == "LIMIT_REACHED"


# --- HTTP smoke judges (pure functions) ----------------------------------------------------


def _smoke():
    import runpy
    from pathlib import Path

    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "http_smoke.py"))


SMOKE = _smoke()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((200, "SUCCEEDED", "no_data", True, 0, 0), ("pass", [], [])),
        ((200, "SUCCEEDED", "no_data", False, 0, 0), ("model_text", ["no_data_model_text"], [])),
        ((200, "SUCCEEDED", "verified", False, 0, 1), ("verified_without_facts", ["no_data_verified_without_facts"], [])),
        ((200, "SUCCEEDED", "unverified", False, 0, 1), ("not_declared", [], ["no_data_not_declared"])),
        ((202, "WAITING_USER", None, False, 0, 0), ("not_declared", [], ["no_data_not_declared"])),
        ((502, "FAILED", None, False, 0, 0), ("server_error", ["no_data_step"], [])),
        ((403, "DENIED", None, False, 0, 0), ("other", ["no_data_step"], [])),
    ],
    ids=["pass", "model-text", "verified-no-facts", "queried", "asked", "502", "denied"],
)
def test_no_data_step_judge(args, expected):
    assert SMOKE["judge_no_data_step"](*args) == expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((200, "SUCCEEDED", "unverified", 2, 0, 0), ("pass", [], [])),
        ((200, "SUCCEEDED", "unverified", 0, 0, 0), ("no_sources", ["knowledge_without_sources"], [])),
        ((200, "SUCCEEDED", "verified", 1, 0, 1), ("verified_without_facts", ["knowledge_verified_without_facts"], [])),
        ((200, "SUCCEEDED", "verified", 1, 1, 2), ("not_declared", [], ["knowledge_not_declared"])),
        ((200, "SUCCEEDED", "no_data", 0, 0, 0), ("not_declared", [], ["knowledge_not_declared"])),
        ((502, "FAILED", None, 0, 0, 0), ("server_error", ["knowledge_step"], [])),
    ],
    ids=["pass", "no-sources", "verified-no-facts", "queried", "declared-no-data", "502"],
)
def test_knowledge_step_judge(args, expected):
    assert SMOKE["judge_knowledge_step"](*args) == expected


def test_answer_status_failures_for_existing_steps():
    failures = SMOKE["answer_status_failures"]
    fact = {"facts": [{"metric_id": "gross_fen"}]}
    assert failures("sync_success", {"status": "SUCCEEDED", "answer_status": "verified", "facts": fact}, "verified") == []
    assert failures("sync_success", {"status": "SUCCEEDED", "answer_status": "unverified", "facts": fact}, "verified") == [
        "answer_status:sync_success"
    ]
    assert failures("approval_approved", {"status": "SUCCEEDED", "answer_status": "verified", "facts": None}, "unverified") == [
        "answer_status:approval_approved",
        "verified_without_facts:approval_approved",
    ]
    assert failures("clarify_request", {"status": "WAITING_USER", "answer_status": None}, "verified") == []


# --- Basis diagnostics, the copyable no_data action, send-back recovery -------------------

from queryshield.agent.context import NO_DATA_ACTION  # noqa: E402
from queryshield.agent.metric_intent import DEFINITION_ANSWER_SHAPE, DEFINITION_SEARCH_ACTION  # noqa: E402

R1_NOT_GROUNDED_ACTION = (
    "No query result in this run supports this answer. Business values (counts, amounts, totals) always "
    "need a query: first send a tool_call named query_readonly with arguments.metrics and time_window, "
    "then cite its result. A window with no data is still queried and reported as 0."
)


@pytest.mark.parametrize(
    ("extra", "basis", "basis_field"),
    [({}, "query", "absent"), ({"basis": None}, "query", "null"), ({"basis": "query"}, "query", "declared"),
     ({"basis": "knowledge"}, "knowledge", "declared"), ({"basis": "no_data"}, "no_data", "declared")],
    ids=["absent", "null", "query", "knowledge", "no_data"],
)
def test_r1_parser_records_how_basis_was_written(extra, basis, basis_field):
    action = _parse({"type": "final_answer", "answer": "a", "source_ids": [], "fact_refs": [], **extra}).action
    assert (action.basis, action.basis_field) == (basis, basis_field)
    # Diagnostics only: never in the public action, never part of equality.
    assert "basis_field" not in action.as_dict()
    assert action == type(action)(answer="a", source_ids=(), fact_refs=(), basis=basis)


def test_r1_blank_answer_is_accepted_only_for_no_data():
    blank = {"type": "final_answer", "answer": "", "source_ids": [], "fact_refs": []}
    assert _parse({**blank, "basis": "no_data"}).action.basis == "no_data"
    for extra in ({}, {"basis": None}, {"basis": "query"}, {"basis": "knowledge"}):
        with pytest.raises(ProposalParseError, match="answer must not be blank"):
            _parse({**blank, **extra})


def test_r1_answer_events_carry_the_basis_and_how_it_was_written():
    null_basis = lambda messages: {"type": "final_answer", "answer": MODEL_TEXT, "source_ids": [], "fact_refs": [], "basis": None}  # noqa: E731
    result, conn = _run([null_basis, _final(basis="no_data")], run_id="run-b3c2-r1-events")
    assert result.status == "succeeded" and result.answer_status == "no_data" and not conn.executed
    validation, bounce, answer = (_events(result, kind)[0] for kind in ("answer_validation", "answer_bounce", "answer"))
    assert (validation["basis"], validation["basis_field"], validation["error_code"]) == ("query", "null", "answer_not_grounded")
    assert (bounce["basis"], bounce["basis_field"]) == ("query", "null")
    assert (answer["basis"], answer["basis_field"]) == ("no_data", "declared")
    _assert_no_model_text(result)


@pytest.mark.parametrize("retrieval", [True, False])
def test_r1_context_shows_the_copyable_no_data_action(retrieval):
    server = json.loads(
        build_context(h._context(), "q", retrieval_available=retrieval).messages[0]["content"].split("\n", 1)[1]
    )
    final_answer = server["action_contract"]["actions"]["final_answer"]
    assert final_answer["valid_shape_examples"][-1] == NO_DATA_ACTION
    assert "basis" not in final_answer["required_fields"]
    assert final_answer["basis"].startswith("Optional; default query. ")
    assert final_answer["basis"].endswith(" no_data: only when no business data is needed (greeting, what you can do); no query.")
    action = parse_query_proposal(NO_DATA_ACTION, context=h._context(), model_call_id="call-r1").action
    assert (action.basis, action.basis_field, action.answer, action.fact_refs) == ("no_data", "declared", "", ())


def test_r1_send_back_hint_keeps_the_query_first_and_gives_copyable_actions():
    hint = answer_not_grounded_hint({**h.SEPTEMBER, "timezone": "UTC"}, retrieval_available=True)
    assert list(hint) == [
        "action", "only_if_definition", "definition_search_action", "definition_answer_shape",
        "only_if_no_data_needed", "no_data_action", "request_time_window",
    ]
    assert hint["action"] == R1_NOT_GROUNDED_ACTION
    assert hint["only_if_definition"].startswith("Only if the question asks how a metric is defined, not for any value")
    assert hint["only_if_no_data_needed"].startswith("Only if the question needs no business data at all")
    assert (hint["definition_search_action"], hint["definition_answer_shape"], hint["no_data_action"]) == (
        DEFINITION_SEARCH_ACTION, DEFINITION_ANSWER_SHAPE, NO_DATA_ACTION,
    )
    parsed = [
        parse_query_proposal(hint[key], context=h._context(), model_call_id="call-r1").action
        for key in ("definition_search_action", "definition_answer_shape", "no_data_action")
    ]
    assert (parsed[0].name, parsed[0].arguments["top_k"]) == ("search_catalog", 3)
    assert (parsed[1].basis, parsed[1].fact_refs) == ("knowledge", ())
    assert (parsed[2].basis, parsed[2].answer) == ("no_data", "")
    without = answer_not_grounded_hint(None, retrieval_available=False)
    assert list(without) == ["action", "only_if_no_data_needed", "no_data_action", "request_time_window"]
    assert without["action"] == R1_NOT_GROUNDED_ACTION


def _store_trace(run_id):
    events = shared_run_service().store.events(run_id)
    return SMOKE["action_trace"]([event.get("payload") or {} for event in events if event.get("type") == "agent_step"])


DIRECT_ANSWER = json.dumps({"type": "final_answer", "answer": MODEL_TEXT, "source_ids": [], "fact_refs": []}, ensure_ascii=False)


def test_r1_http_no_data_after_one_send_back(env):
    model = Scripted([DIRECT_ANSWER, NO_DATA_ACTION])
    app.dependency_overrides[get_model_provider] = lambda: model
    response = ask(env, "你好，你能做什么？")
    body = response.json()
    assert response.status_code == 200 and body["status"] == "SUCCEEDED"
    assert body["answer"] == NO_DATA_REPLY and body["answer_status"] == "no_data" and body["sql_exec_count"] == 0
    assert model.calls == 2 and MODEL_TEXT not in response.text
    assert _store_trace(body["run_id"]) == [
        "final_answer(query,absent,answer_not_grounded)",
        "answer_bounce(answer_not_grounded)",
        "final_answer(no_data,declared)",
    ]


def test_r1_http_knowledge_after_one_send_back(env):
    model = Scripted([
        DIRECT_ANSWER,
        DEFINITION_SEARCH_ACTION.replace("<指标名>", "退款后净额"),
        DEFINITION_ANSWER_SHAPE.replace('"..."', '"按检索到的口径说明"'),
    ])
    app.dependency_overrides[get_model_provider] = lambda: model
    body = ask(env, "退款后净额是怎么算的？").json()
    assert body["status"] == "SUCCEEDED" and body["answer_status"] == "unverified"
    assert body["source_ids"] and body["sql_exec_count"] == 0 and model.calls == 3
    assert _store_trace(body["run_id"]) == [
        "final_answer(query,absent,answer_not_grounded)",
        "answer_bounce(answer_not_grounded)",
        "tool_call(search_catalog)",
        "final_answer(knowledge,declared)",
    ]


def test_r1_action_trace_uses_fixed_identifiers_only():
    trace = SMOKE["action_trace"]([
        {"kind": "model_call", "status": "succeeded"},
        {"kind": "proposal_validation", "error_code": "invalid_field"},
        {"kind": "tool_call", "tool_name": "search_catalog", "status": "succeeded"},
        {"kind": "tool_call", "tool_name": "query_readonly", "status": "failed", "error_code": "invalid_sql"},
        {"kind": "query_repair", "error_code": "invalid_sql"},
        {"kind": "parallel_group", "status": "SUCCEEDED"},
        {"kind": "clarification_review", "decision": "not_needed"},
        {"kind": "clarification_bounce", "error_code": "clarification_not_needed"},
        {"kind": "answer_validation", "basis": "knowledge", "basis_field": "declared", "error_code": "answer_not_grounded"},
        {"kind": "answer_bounce", "error_code": "answer_not_grounded"},
        {"kind": "answer", "basis": "query", "basis_field": "absent"},
        {"kind": "limit", "error_code": "model_call_limit"},
        {"kind": "tool_call", "tool_name": MODEL_TEXT, "status": "failed", "error_code": "Bad Code"},
        {"kind": "answer", "basis": MODEL_TEXT, "basis_field": None},
    ])
    assert trace == [
        "parse_failure(invalid_field)",
        "tool_call(search_catalog)",
        "tool_call(query_readonly,failed:invalid_sql)",
        "query_repair(invalid_sql)",
        "parallel_readonly(succeeded)",
        "ask_user(not_needed)",
        "clarification_bounce(clarification_not_needed)",
        "final_answer(knowledge,declared,answer_not_grounded)",
        "answer_bounce(answer_not_grounded)",
        "final_answer(query,absent)",
        "limit(model_call_limit)",
        "tool_call(other,failed:other)",
        "final_answer(other,other)",
    ]
    assert SMOKE["_sent_back"](trace) and not SMOKE["_sent_back"](trace[:8])


@pytest.mark.parametrize(
    ("judge", "args", "expected"),
    [
        ("judge_no_data_step", (200, "SUCCEEDED", "no_data", True, 0, 0), ("pass_after_send_back", [], ["no_data_after_send_back"])),
        ("judge_no_data_step", (200, "SUCCEEDED", "no_data", False, 0, 0), ("model_text", ["no_data_model_text"], [])),
        ("judge_no_data_step", (200, "SUCCEEDED", "verified", False, 0, 1), ("verified_without_facts", ["no_data_verified_without_facts"], [])),
        ("judge_no_data_step", (502, "FAILED", None, False, 0, 0), ("server_error", ["no_data_step"], [])),
        ("judge_knowledge_step", (200, "SUCCEEDED", "unverified", 2, 0, 0), ("pass_after_send_back", [], ["knowledge_after_send_back"])),
        ("judge_knowledge_step", (200, "SUCCEEDED", "unverified", 0, 0, 0), ("no_sources", ["knowledge_without_sources"], [])),
        ("judge_knowledge_step", (200, "SUCCEEDED", "verified", 1, 0, 1), ("verified_without_facts", ["knowledge_verified_without_facts"], [])),
        ("judge_knowledge_step", (502, "FAILED", None, 0, 0, 0), ("server_error", ["knowledge_step"], [])),
    ],
    ids=["no-data-pass", "no-data-model-text", "no-data-verified", "no-data-502",
         "knowledge-pass", "knowledge-no-sources", "knowledge-verified", "knowledge-502"],
)
def test_r1_judges_after_a_send_back(judge, args, expected):
    # Right only after the one send-back: a known gap, never a hard failure;
    # 502, model text and verified-without-facts stay hard failures.
    assert SMOKE[judge](*args, sent_back=True) == expected


# --- The server searches once for a knowledge answer with no source -------------------------

from queryshield.agent.graph import SERVER_SEARCH_TOP_K  # noqa: E402
from queryshield.agent.metric_intent import knowledge_from_server_search_hint  # noqa: E402
from queryshield.tools.semantic import ToolError  # noqa: E402

DEFINITION_QUESTION = "退款后净额是怎么算的？"
QUESTION_SENTINEL = "问题哨兵QUESTION-SENTINEL"
KNOWLEDGE = _final(basis="knowledge", source_ids=())


def _knowledge_run(steps, *, question=DEFINITION_QUESTION, limits=None, retrieval_available=True, patch=None, run_id="run-b3c2-r2"):
    tools, conn = h._tools()
    if patch is not None:
        tools.search_catalog = patch
    model = h._DynamicModel(steps)
    agent = BoundedAgent(
        model, tools=tools, call_store=ModelCallStore(), limits=limits, retrieval_available=retrieval_available
    )
    return agent.run(h._context(run_id), question), model, conn


def _server_searches(result):
    return [event for event in _events(result, "tool_call") if event.get("initiated_by") == "server"]


def test_r2_server_search_grounds_the_second_knowledge_answer():
    result, model, conn = _knowledge_run([KNOWLEDGE, KNOWLEDGE], question=DEFINITION_QUESTION + QUESTION_SENTINEL)
    assert result.status == "succeeded" and result.answer_status == "unverified" and not conn.executed
    assert (result.tool_call_count, result.model_call_count, result.repair_count) == (1, 2, 0)
    [search] = _server_searches(result)
    assert search["status"] == "succeeded" and search["grounding_source_count"] > 0
    assert search["input_summary"] == {"argument_keys": ["query", "top_k"], "top_k": SERVER_SEARCH_TOP_K, "query_source": "run_question"}
    assert result.action["source_ids"] == search["source_ids"] and result.action["source_ids"]
    assert [event["kind"] for event in result.events if event["kind"] in {"answer_validation", "tool_call", "answer_bounce", "answer"}] == [
        "answer_validation", "tool_call", "answer_bounce", "answer",
    ]
    # The model saw the server's search result, then the knowledge-only hint.
    records = h._tool_records(model.messages[1])
    assert [(record.get("tool_name"), record.get("initiated_by")) for record in records] == [
        ("search_catalog", "server"), ("final_answer", None),
    ]
    assert records[1]["repair_hint"] == knowledge_from_server_search_hint()
    # Fixed identifiers only: the question never reaches events or the result.
    assert QUESTION_SENTINEL not in json.dumps([result.events, result.as_dict()], ensure_ascii=False, default=str)


def test_r2_hint_is_knowledge_only_and_its_shape_parses():
    hint = knowledge_from_server_search_hint()
    assert list(hint) == ["action", "answer_shape"]
    encoded = json.dumps(hint)
    for menu_word in ("query_readonly", "no_data", "request_time_window", "only_if"):
        assert menu_word not in encoded
    action = parse_query_proposal(hint["answer_shape"], context=h._context(), model_call_id="call-r2").action
    assert (action.basis, action.fact_refs) == ("knowledge", ())


@pytest.mark.parametrize(
    "output",
    [{"items": []}, {"items": [dict(TABLE_ITEM)]}],
    ids=["empty", "table-entry-only"],
)
def test_r2_no_usable_search_result_fails_without_a_send_back(output):
    result, model, _ = _knowledge_run([KNOWLEDGE, KNOWLEDGE], patch=lambda arguments, *, context: output)
    assert result.status == "failed" and result.error_code == "answer_not_grounded"
    assert len(model.messages) == 1 and _events(result, "answer_bounce") == [] and result.tool_call_count == 1
    [search] = _server_searches(result)
    assert (search["status"], search["grounding_source_count"]) == ("succeeded", 0)
    assert result.events[-1] is search or result.events[-1]["initiated_by"] == "server"
    _assert_no_model_text(result)


def test_r2_search_error_fails_with_the_answer_code():
    def broken(arguments, *, context):
        raise ToolError("retrieval_unavailable", "the retriever is down")

    result, model, _ = _knowledge_run([KNOWLEDGE, KNOWLEDGE], patch=broken)
    assert result.status == "failed" and result.error_code == "answer_not_grounded" and len(model.messages) == 1
    [search] = _server_searches(result)
    assert (search["status"], search["error_code"]) == ("failed", "retrieval_unavailable")


def test_r2_search_counts_as_a_tool_call_and_respects_the_tool_limit():
    result, model, _ = _knowledge_run([_describe, KNOWLEDGE, KNOWLEDGE], limits=GraphLimits(max_tool_calls=1))
    assert result.status == "failed" and result.error_code == "answer_not_grounded" and len(model.messages) == 2
    assert result.tool_call_count == 1 and _events(result, "answer_bounce") == []
    [search] = _server_searches(result)
    assert (search["status"], search["error_code"]) == ("skipped", "tool_call_limit")
    # With room left, it is one more tool call.
    result, _, _ = _knowledge_run([_describe, KNOWLEDGE, KNOWLEDGE], limits=GraphLimits(max_tool_calls=2))
    assert result.status == "succeeded" and result.tool_call_count == 2


def test_r2_no_search_when_the_send_back_or_the_model_calls_are_spent():
    spent, _, _ = _knowledge_run([_final(), KNOWLEDGE, KNOWLEDGE])
    assert spent.status == "failed" and spent.error_code == "answer_not_grounded"
    assert _server_searches(spent) == [] and len(_events(spent, "answer_bounce")) == 1
    last_call, model, _ = _knowledge_run([KNOWLEDGE, KNOWLEDGE], limits=GraphLimits(max_model_calls=1))
    assert last_call.status == "failed" and last_call.error_code == "answer_not_grounded"
    assert _server_searches(last_call) == [] and len(model.messages) == 1


def test_r2_without_a_retriever_nothing_changes():
    result, model, _ = _knowledge_run([KNOWLEDGE, KNOWLEDGE], retrieval_available=False)
    assert result.status == "failed" and result.error_code == "answer_not_grounded"
    assert _server_searches(result) == [] and result.tool_call_count == 0
    # The R1 send-back, whose hint never mentions search_catalog here.
    hint = _hint_records(model.messages[1])[-1]["repair_hint"]
    assert "search_catalog" not in json.dumps(hint) and list(hint)[0] == "action"


def test_r2_http_knowledge_after_the_server_search(env):
    answer = json.dumps(
        {"type": "final_answer", "answer": MODEL_TEXT, "source_ids": [], "fact_refs": [], "basis": "knowledge"},
        ensure_ascii=False,
    )
    model = Scripted([answer, answer])  # Scripted counts its calls
    app.dependency_overrides[get_model_provider] = lambda: model
    body = ask(env, DEFINITION_QUESTION).json()
    assert body["status"] == "SUCCEEDED" and body["answer_status"] == "unverified"
    assert body["source_ids"] and body["sql_exec_count"] == 0 and len(model.messages) == 2
    trace = _store_trace(body["run_id"])
    assert trace == [
        "final_answer(knowledge,declared,answer_not_grounded)",
        "server_search_catalog",
        "answer_bounce(answer_not_grounded)",
        "final_answer(knowledge,declared)",
    ]
    assert SMOKE["judge_knowledge_step"](200, "SUCCEEDED", "unverified", len(body["source_ids"]), 0, 0, sent_back=SMOKE["_sent_back"](trace)) == (
        "pass_after_send_back", [], ["knowledge_after_send_back"],
    )


def test_r2_action_trace_marks_the_server_search():
    server = {"kind": "tool_call", "tool_name": "search_catalog", "initiated_by": "server"}
    assert SMOKE["action_trace"]([
        {**server, "status": "succeeded", "grounding_source_count": 2},
        {**server, "status": "succeeded", "grounding_source_count": 0},
        {**server, "status": "skipped", "error_code": "tool_call_limit"},
        {**server, "status": "failed", "error_code": "retrieval_unavailable"},
        {"kind": "tool_call", "tool_name": "search_catalog", "status": "succeeded"},
    ]) == [
        "server_search_catalog",
        "server_search_catalog(empty)",
        "server_search_catalog(skipped:tool_call_limit)",
        "server_search_catalog(failed:retrieval_unavailable)",
        "tool_call(search_catalog)",
    ]


def test_frozen_eval_questions_never_reach_the_server_search():
    from queryshield.agent.runtime import b1_result_payload, build_b1_agent
    from queryshield.approval.service import FixtureQueryExecutor
    from queryshield.evaluation.state_cases import load_state_cases, load_supplement_cases
    from queryshield.evaluation.stateful_product import StateCaseFakeModel
    from queryshield.knowledge.runtime import shared_retrieval_runtime
    from queryshield.tools.semantic import ControlledTools

    retriever = shared_retrieval_runtime("fake").retriever
    checked = 0
    for case in (*load_state_cases(), *load_supplement_cases()):
        parameters = case.case["action"].get("parameters") or {}
        # Same reading as the stateful runner: question, else query.
        question = parameters.get("question") or parameters.get("query")
        if case.case["action"]["entrypoint"] != "/queries" or not isinstance(question, str):
            continue
        tools = ControlledTools(catalog=CATALOG, executor=FixtureQueryExecutor(), retriever=retriever)
        context = ExecutionContext(run_id=f"run-r2-{case.case_id}", tenant_id="A", principal_id="principal-A", role="requester")
        output = b1_result_payload(build_b1_agent(StateCaseFakeModel(case), tools).run(context, question), context, question)
        assert not any(event.get("initiated_by") == "server" for event in output["events"]), case.case_id
        checked += 1
    assert checked == 11  # every /queries case of the frozen and supplement sets
