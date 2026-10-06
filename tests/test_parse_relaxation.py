"""Null/[] query_readonly fields equal omission; parse failures are diagnosable."""

from __future__ import annotations

import json

import pytest

from queryshield.agent.graph import BoundedAgent
from queryshield.agent.proposals import ExecutionContext, ProposalParseError, parse_query_proposal
from queryshield.api.main import get_model_provider, app
from queryshield.approval.service import FixtureQueryExecutor, shared_run_service
from queryshield.tools.semantic import ControlledTools

from test_http_queries import APPROVER, REQUESTER, Scripted, ask, auth, env  # noqa: F401  (env is a fixture)


CONTEXT = ExecutionContext(run_id="run-parse", tenant_id="A", principal_id="principal-A", role="requester")
WINDOW = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}
NAME_SQL = "SELECT c.customer_id, c.name FROM customers AS c ORDER BY c.customer_id"
GROSS_SQL = (
    "SELECT COALESCE(SUM(o.amount_fen), 0) AS gross_fen FROM orders AS o "
    "WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s"
)
GROSS_PARAMS = {"0": "paid", "1": WINDOW["start"], "2": WINDOW["end"]}


def _call(arguments: dict) -> str:
    return json.dumps({"type": "tool_call", "name": "query_readonly", "arguments": arguments}, ensure_ascii=False)


def _parsed(arguments: dict) -> dict:
    return dict(parse_query_proposal(_call(arguments), context=CONTEXT, model_call_id="call-1").action.arguments)


@pytest.mark.parametrize(
    ("relaxed", "canonical"),
    [
        ({"sql": NAME_SQL, "params": {}, "metrics": None}, {"sql": NAME_SQL, "params": {}}),
        ({"sql": NAME_SQL, "params": {}, "time_window": None}, {"sql": NAME_SQL, "params": {}}),
        ({"sql": NAME_SQL, "params": None}, {"sql": NAME_SQL, "params": {}}),
        ({"sql": NAME_SQL, "params": []}, {"sql": NAME_SQL, "params": {}}),
        (
            {"sql": NAME_SQL, "params": [], "metrics": None, "time_window": None},
            {"sql": NAME_SQL, "params": {}},
        ),
        (
            {"sql": GROSS_SQL, "params": GROSS_PARAMS, "metrics": ["gross_fen"], "time_window": None},
            {"sql": GROSS_SQL, "params": GROSS_PARAMS, "metrics": ["gross_fen"]},
        ),
    ],
)
def test_null_or_empty_fields_parse_exactly_like_omission(relaxed, canonical) -> None:
    assert _parsed(relaxed) == _parsed(canonical)


@pytest.mark.parametrize("params", [["paid"], [None], "x", 1])
def test_other_params_shapes_are_still_rejected(params) -> None:
    with pytest.raises(ProposalParseError) as caught:
        _parsed({"sql": NAME_SQL, "params": params})
    assert caught.value.code == "invalid_field"


def _agent_run(first: dict, request_window=None):
    model = Scripted([_call(first), json.dumps({"type": "final_answer", "answer": "x", "source_ids": [], "fact_refs": []})])
    agent = BoundedAgent(model, tools=ControlledTools(executor=FixtureQueryExecutor()))
    return agent.run(CONTEXT, "q", request_time_window=request_window)


def _stable(result) -> dict:
    payload = result.as_dict()
    return {
        "status": payload["status"],
        "error_code": payload["error_code"],
        "action": payload["action"],
        "facts": [(item["metric_id"], item["value"]) for item in (payload["facts"] or {}).get("facts", [])],
        "tool_calls": payload["tool_call_count"],
    }


@pytest.mark.parametrize(
    ("relaxed", "canonical", "window"),
    [
        ({"sql": NAME_SQL, "params": [], "metrics": None, "time_window": None}, {"sql": NAME_SQL, "params": {}}, None),
        (
            {"sql": GROSS_SQL, "params": GROSS_PARAMS, "metrics": ["gross_fen"], "time_window": None},
            {"sql": GROSS_SQL, "params": GROSS_PARAMS, "metrics": ["gross_fen"]},
            WINDOW,
        ),
        (
            {"sql": GROSS_SQL, "params": GROSS_PARAMS, "metrics": None},
            {"sql": GROSS_SQL, "params": GROSS_PARAMS},
            None,
        ),
    ],
)
def test_relaxed_call_runs_exactly_like_the_omitted_call(relaxed, canonical, window) -> None:
    relaxed_result = _stable(_agent_run(relaxed, window))
    canonical_result = _stable(_agent_run(canonical, window))
    assert relaxed_result == canonical_result
    # An explicit null declaration never creates a fact.
    if relaxed.get("metrics", ...) is None:
        assert relaxed_result["facts"] == []


def test_customer_names_with_null_fields_reach_approval_over_http(env) -> None:
    model = Scripted([
        '{"type":"tool_call","name":"describe_tables","arguments":{"tables":["customers"]}}',
        _call({"sql": NAME_SQL, "params": [], "metrics": None, "time_window": None}),
    ])
    app.dependency_overrides[get_model_provider] = lambda: model

    response = ask(env, "查询本租户所有客户的姓名")

    body = response.json()
    assert response.status_code == 202
    assert body["status"] == "WAITING_APPROVAL"
    action = shared_run_service().store.get_run(body["run_id"])["action"]
    assert action["sql"] == NAME_SQL and action["params"] == [] and action["metrics"] == [] and action["time_window"] is None
    approved = env.post(
        f"/runs/{body['run_id']}/approval",
        headers=auth(APPROVER),
        json={"approval_id": body["approval_id"], "decision": "approve"},
    )
    assert approved.status_code == 200 and approved.json()["status"] == "SUCCEEDED"


def test_parse_failure_records_fixed_detail_and_value_free_shape(env) -> None:
    secret = "secret-model-text"
    model = Scripted([
        json.dumps({
            "type": "tool_call",
            "name": "query_readonly",
            "arguments": {"sql": f"SELECT '{secret}'", "params": [secret], "extra_" + secret: 1},
        }),
    ])
    app.dependency_overrides[get_model_provider] = lambda: model

    body = ask(env, "查询本租户所有客户的姓名").json()

    assert body["status"] == "FAILED"
    events = [
        event["payload"]
        for event in shared_run_service().store.events(body["run_id"])
        if event["type"] == "agent_step" and event["payload"].get("kind") == "proposal_validation"
    ]
    assert len(events) == 1
    assert events[0]["status"] == "failed"
    assert events[0]["error_code"] == "unknown_field"
    # The model-chosen field name is not echoed; only the fixed prefix is kept.
    assert events[0]["error_detail"] == "proposal contains unknown field(s)"
    assert events[0]["action_shape"] == {
        "json": "valid",
        "type": "tool_call",
        "name": "query_readonly",
        "top_level_extra_field_count": 0,
        "basis": "absent",
        "arguments": {"sql": "string", "params": "array", "metrics": "absent", "time_window": "absent"},
        "arguments_extra_field_count": 1,
    }
    stored = json.dumps(list(shared_run_service().store.events(body["run_id"])), ensure_ascii=False)
    assert secret not in stored


@pytest.mark.parametrize(
    ("arguments", "detail", "shape"),
    [
        ({"sql": NAME_SQL, "params": ["c1"]}, "params must be an object", {"params": "array"}),
        ({"sql": NAME_SQL, "params": {}, "time_window": "2026-09"}, "time_window must be an object", {"time_window": "string"}),
        ({"sql": NAME_SQL, "params": {}, "metrics": "gross_fen"}, "metrics must be an array", {"metrics": "string"}),
    ],
)
def test_parse_failure_detail_names_the_field(env, arguments, detail, shape) -> None:
    app.dependency_overrides[get_model_provider] = lambda: Scripted([_call(arguments)])
    body = ask(env, "查询本租户所有客户的姓名").json()
    assert body["status"] == "FAILED" and body["error"]["code"] == "invalid_field"
    event = next(
        event["payload"]
        for event in shared_run_service().store.events(body["run_id"])
        if event["type"] == "agent_step" and event["payload"].get("kind") == "proposal_validation"
    )
    assert event["error_detail"] == detail
    for field, json_type in shape.items():
        assert event["action_shape"]["arguments"][field] == json_type


# --- Second relaxation: "metrics": [] is the same as omitting metrics. ---


def _run_http_name_query(env, arguments: dict) -> tuple[dict, dict, dict]:
    model = Scripted([
        '{"type":"tool_call","name":"describe_tables","arguments":{"tables":["customers"]}}',
        _call(arguments),
    ])
    app.dependency_overrides[get_model_provider] = lambda: model
    response = ask(env, "查询本租户所有客户的姓名")
    body = response.json()
    assert response.status_code == 202 and body["status"] == "WAITING_APPROVAL"
    action = shared_run_service().store.get_run(body["run_id"])["action"]
    approved = env.post(
        f"/runs/{body['run_id']}/approval",
        headers=auth(APPROVER),
        json={"approval_id": body["approval_id"], "decision": "approve"},
    )
    assert approved.status_code == 200 and approved.json()["status"] == "SUCCEEDED"
    result = env.get(f"/runs/{body['run_id']}/result", headers=auth(REQUESTER)).json()
    return body, action, result


@pytest.mark.parametrize(
    ("relaxed", "canonical"),
    [
        ({"sql": NAME_SQL, "params": {}, "metrics": []}, {"sql": NAME_SQL, "params": {}}),
        ({"sql": NAME_SQL, "params": [], "metrics": [], "time_window": None}, {"sql": NAME_SQL, "params": {}}),
        # A window without metrics is still the omitted-metrics case (rejected at tool time).
        (
            {"sql": GROSS_SQL, "params": GROSS_PARAMS, "metrics": [], "time_window": WINDOW},
            {"sql": GROSS_SQL, "params": GROSS_PARAMS, "time_window": WINDOW},
        ),
    ],
)
def test_empty_metrics_parse_exactly_like_omission(relaxed, canonical) -> None:
    assert _parsed(relaxed) == _parsed(canonical)


@pytest.mark.parametrize("metrics", [["name"], ["gross_fen"], [1], ["gross_fen", "gross_fen"]])
def test_non_empty_metrics_still_reach_the_declaration_rules(metrics) -> None:
    assert _parsed({"sql": GROSS_SQL, "params": GROSS_PARAMS, "metrics": metrics})["metrics"] == metrics


def test_non_empty_metric_rules_are_unchanged() -> None:
    from queryshield.agent.metric_intent import MetricDeclarationError, resolve_query_declaration
    from queryshield.catalog import load_default_catalog

    catalog = load_default_catalog()
    for metrics, code in (
        (["name"], "unknown_metric"),
        ([1], "invalid_metric_declaration"),
        (["gross_fen", "gross_fen"], "invalid_metric_declaration"),
        (["paid_count", "gross_fen", "net_fen", "refund_fen", "x"], "invalid_metric_declaration"),
    ):
        with pytest.raises(MetricDeclarationError) as caught:
            resolve_query_declaration({"sql": GROSS_SQL, "params": GROSS_PARAMS, "metrics": metrics, "time_window": WINDOW}, catalog=catalog)
        assert caught.value.code == code


def test_customer_names_with_empty_metrics_reach_approval_over_http(env) -> None:
    body, action, result = _run_http_name_query(env, {"sql": NAME_SQL, "params": {}, "metrics": []})
    assert body["sql_exec_count"] == 0
    assert action["sql"] == NAME_SQL and action["metrics"] == [] and action["time_window"] is None
    assert result["facts"] is None
    assert {row["name"] for row in result["result"]["rows"]} == {"甲", "乙"}


def test_customer_names_with_empty_metrics_approve_like_omitted_metrics(env) -> None:
    def comparable(run):
        body, action, result = run
        return (
            {key: action[key] for key in ("kind", "sql", "params", "metrics", "time_window")},
            result["facts"],
            result["result"]["rows"],
            result["answer"],
        )

    relaxed = comparable(_run_http_name_query(env, {"sql": NAME_SQL, "params": {}, "metrics": []}))
    canonical = comparable(_run_http_name_query(env, {"sql": NAME_SQL, "params": {}}))
    assert relaxed == canonical


def _metric_intent_helpers():
    import test_metric_intent as helpers

    return helpers


def _metric_run_summary(result) -> dict:
    return {
        "status": result.status,
        "error_code": result.error_code,
        "repair_count": result.repair_count,
        "facts": [(fact["metric_id"], fact["value"]) for fact in (result.facts or {}).get("facts", [])],
        "validation": [
            (event.get("kind"), event.get("error_code"))
            for event in result.events
            if event.get("kind") in {"answer_validation", "query_repair", "proposal_validation"}
        ],
    }


@pytest.mark.parametrize("redeclare", [True, False])
def test_metric_question_with_empty_metrics_runs_like_omitted_metrics(redeclare) -> None:
    h = _metric_intent_helpers()

    def run(first_arguments: dict):
        tools, connection = h._tools()
        steps = [lambda messages: {"type": "tool_call", "name": "query_readonly", "arguments": first_arguments}]
        steps.append(h._cite(["gross_fen"]))
        if redeclare:
            steps.append(h._query_step(sql=h.GROSS_SQL, metrics=["gross_fen"]))
            steps.append(h._cite_verified)
        else:
            steps.append(h._query_step(sql=h.GROSS_SQL))
            steps.append(h._cite(["gross_fen"]))
        result = h._agent(h._DynamicModel(steps), tools).run(
            h._context("run-b2b-empty-metrics"), "2026年9月支付总额是多少？", request_time_window=h.SEPTEMBER_UTC
        )
        return _metric_run_summary(result), len(connection.executed)

    relaxed = run(h._query(h.GROSS_SQL, metrics=[]))
    omitted = run(h._query(h.GROSS_SQL))
    assert relaxed == omitted
    summary = relaxed[0]
    # Both go through the metric_not_declared path.
    assert ("answer_validation", "metric_not_declared") in summary["validation"]
    if redeclare:
        assert summary["status"] == "succeeded" and summary["facts"] == [("gross_fen", 15000)]
    else:
        assert summary["status"] == "failed" and summary["error_code"] == "metric_not_declared"


def test_b0_with_empty_metrics_runs_like_omitted_metrics() -> None:
    from queryshield.evaluation.profile_runner import run_b0_single_pass

    h = _metric_intent_helpers()

    def run(arguments: dict) -> dict:
        tools, _ = h._tools()
        model = h._DynamicModel([lambda messages: {"type": "tool_call", "name": "query_readonly", "arguments": arguments}])
        output = run_b0_single_pass(model, tools, h._context("run-b0-empty-metrics"), "2026年9月支付总额是多少？", time_window=h.SEPTEMBER_UTC)
        return {key: output[key] for key in ("status", "error_code", "facts", "rows", "readonly_queries")}

    relaxed = run(h._query(h.GROSS_SQL, metrics=[]))
    assert relaxed == run(h._query(h.GROSS_SQL))
    assert relaxed["status"] == "succeeded" and relaxed["facts"] == []
