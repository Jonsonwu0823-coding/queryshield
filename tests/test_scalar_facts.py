"""Only a result whose rows are exactly the metric's scope becomes a scalar fact.

A grouped query (GROUP BY, also with ORDER BY ... LIMIT 1) is a rowset of
per-group values even when one row comes back; a customers join whose ON clause
has more than the two key equalities is not a trusted join.  Every place that
turns results into facts applies the same rule: the graph (B1), B0, the
approval path (including results checkpointed before the pause), FactResolver
and the /result re-check of persisted facts.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re

import pytest

from queryshield.agent.proposals import ExecutionContext, FactRef, MetricBinding, ResultEvidence
from queryshield.agent.tool_execution import _bind_metric_result_positions, call_tool
from queryshield.api.main import app, get_guarded_executor, get_model_provider
from queryshield.approval.service import shared_run_service
from queryshield.catalog import load_default_catalog
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.facts import FactResolutionError, FactResolver
from queryshield.facts.facts import is_scalar_metric_result
from queryshield.facts.persisted import evidence_from_record, validate_persisted_run_result
from queryshield.providers.contracts import ModelCallResult
from queryshield.tools import ControlledTools

from test_http_queries import APPROVER, REQUESTER, auth, env  # noqa: F401  (env is a fixture)
from test_demo_expected_answers import READONLY_URL, needs_demo_database


CATALOG = load_default_catalog()
JULY = {"start": "2026-07-01T00:00:00Z", "end": "2026-08-01T00:00:00Z"}
JULY_UTC = {**JULY, "timezone": "UTC"}
PARAMS = {"0": "paid", "1": JULY["start"], "2": JULY["end"]}
JOIN = "FROM orders AS o INNER JOIN customers AS c ON o.tenant_id = c.tenant_id AND o.customer_id = c.customer_id"
WHERE_O = "WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s"
WHERE = "WHERE status = %s AND created_at >= %s AND created_at < %s"
AGG = {"gross_fen": "COALESCE(SUM({col}amount_fen), 0)", "paid_count": "COUNT(*)"}


def _agg(metric: str, qualifier: str = "") -> str:
    return AGG[metric].format(col=qualifier)


def _sql(kind: str, metric: str) -> tuple[str, dict[str, object]]:
    """Ticket 1.1 writings, plus the controller's created_at/status and two more ON conditions."""

    o = _agg(metric, "o.")
    plain = _agg(metric)
    order = f"ORDER BY {metric} DESC"
    table = {
        "total": (f"SELECT {plain} AS {metric} FROM orders {WHERE}", PARAMS),
        "join_total": (f"SELECT {o} AS {metric} {JOIN} {WHERE_O}", PARAMS),
        "join_all": (f"SELECT o.customer_id, {o} AS {metric} {JOIN} {WHERE_O} GROUP BY o.customer_id", PARAMS),
        "join_top1_id": (f"SELECT o.customer_id, {o} AS {metric} {JOIN} {WHERE_O} GROUP BY o.customer_id {order} LIMIT 1", PARAMS),
        "join_top1_name": (f"SELECT c.name, {o} AS {metric} {JOIN} {WHERE_O} GROUP BY o.customer_id, c.name {order} LIMIT 1", PARAMS),
        "by_name_top1": (f"SELECT c.name, {o} AS {metric} {JOIN} {WHERE_O} GROUP BY c.name {order} LIMIT 1", PARAMS),
        "join_top2": (f"SELECT o.customer_id, {o} AS {metric} {JOIN} {WHERE_O} GROUP BY o.customer_id {order} LIMIT 2", PARAMS),
        "created_at_top1": (f"SELECT created_at, {plain} AS {metric} FROM orders {WHERE} GROUP BY created_at {order} LIMIT 1", PARAMS),
        "status_top1": (f"SELECT status, {plain} AS {metric} FROM orders {WHERE} GROUP BY status {order} LIMIT 1", PARAMS),
        "on_extra_customer": (
            f"SELECT {o} AS {metric} {JOIN} AND c.customer_id = %s WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s",
            {"0": "c1", "1": "paid", "2": JULY["start"], "3": JULY["end"]},
        ),
        "on_extra_orders": (
            f"SELECT {o} AS {metric} {JOIN} AND o.amount_fen > %s WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s",
            {"0": 6000, "1": "paid", "2": JULY["start"], "3": JULY["end"]},
        ),
        "on_extra_grouped": (
            f"SELECT o.customer_id, {o} AS {metric} {JOIN} AND c.customer_id = %s "
            f"WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s GROUP BY o.customer_id",
            {"0": "c1", "1": "paid", "2": JULY["start"], "3": JULY["end"]},
        ),
        "on_repeated_equality": (
            f"SELECT {o} AS {metric} {JOIN} AND o.tenant_id = c.tenant_id {WHERE_O}",
            PARAMS,
        ),
        "nojoin_group_customer": (f"SELECT customer_id, {plain} AS {metric} FROM orders {WHERE} GROUP BY customer_id {order} LIMIT 1", PARAMS),
    }
    return table[kind]


SCALAR = ("total", "join_total")
GROUPED = ("join_all", "join_top1_id", "join_top1_name", "by_name_top1", "join_top2", "created_at_top1", "status_top1")
REJECTED = ("on_extra_customer", "on_extra_orders", "on_extra_grouped", "on_repeated_equality", "nojoin_group_customer")
METRICS = ("gross_fen", "paid_count")


def _binding(metric: str, **changes) -> MetricBinding:
    entry = CATALOG.metric(metric)
    return MetricBinding(
        metric_id=metric,
        result_position=metric,
        unit=str(entry.payload["unit"]),
        time_window=JULY_UTC,
        catalog_source_id=entry.source_id,
        catalog_version=CATALOG.catalog_version,
        **changes,
    )


def _context(run_id: str = "run-b3e") -> ExecutionContext:
    return ExecutionContext(run_id=run_id, tenant_id="A", principal_id="a-requester", role="requester")


# --- the binding decision (the one place that sees the parsed statement) -------------


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("kind", SCALAR + GROUPED + REJECTED)
def test_binding_marks_grouped_and_refuses_untrusted_joins(kind, metric) -> None:
    sql, params = _sql(kind, metric)
    bound = _bind_metric_result_positions({"sql": sql, "params": params}, (_binding(metric),), context=_context())
    if kind in REJECTED:
        assert bound is None
    else:
        assert bound is not None and bound[0].result_position == metric
        assert bound[0].grouped is (kind in GROUPED)


def test_ungrouped_binding_serializes_exactly_as_before_and_old_records_read() -> None:
    binding = _binding("gross_fen")
    assert list(binding.as_dict()) == [
        "metric_id", "result_position", "unit", "time_window", "catalog_source_id", "catalog_version", "plan_id",
    ]
    assert MetricBinding(**binding.as_dict()) == binding
    grouped = _binding("gross_fen", grouped=True)
    assert grouped.as_dict() == {**binding.as_dict(), "grouped": True}
    assert MetricBinding(**grouped.as_dict()).grouped is True
    with pytest.raises(ValueError):
        _binding("gross_fen", grouped=1)


# --- a synthetic tenant-A database that honours GROUP BY and LIMIT ---------------------


ORDERS = (("c1", "甲", "2026-07-02T00:00:00Z", 10000), ("c2", "乙", "2026-07-03T00:00:00Z", 5000))
TOTAL = {"gross_fen": 15000, "paid_count": 2}


class _Cursor:
    def __init__(self, connection) -> None:
        self.connection = connection
        self.rows: list[dict[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def execute(self, sql, params) -> None:
        sql = str(sql)
        self.connection.executed.append(sql)
        projection = sql.split(" FROM ", 1)[0]
        metrics = [name for name in ("gross_fen", "paid_count") if f'AS "{name}"' in projection]
        if " GROUP BY " not in sql:
            self.rows = [{name: TOTAL[name] for name in metrics}]
            return
        group = sql.split(" GROUP BY ", 1)[1]
        key_index = 1 if '"name"' in group and '"customer_id"' not in group else 2 if '"created_at"' in group else 0
        if '"status"' in group:
            groups = {"paid": list(ORDERS)}
        else:
            groups = {}
            for order in ORDERS:
                groups.setdefault(order[key_index], []).append(order)
        rows = []
        for members in groups.values():
            row: dict[str, object] = {}
            for column, index in (("customer_id", 0), ("name", 1), ("created_at", 2)):
                if f'"{column}"' in projection:
                    row[column] = members[0][index]
            if '"status"' in projection:
                row["status"] = "paid"
            for name in metrics:
                row[name] = sum(item[3] for item in members) if name == "gross_fen" else len(members)
            rows.append(row)
        rows.sort(key=lambda item: -int(item[metrics[0]]))
        limits = [int(value) for value in re.findall(r"LIMIT (\d+)", sql)]
        self.rows = rows[: min(limits)] if limits else rows

    def fetchmany(self, size):
        return self.rows[:size]


class _Connection:
    def __init__(self) -> None:
        self.executed: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def cursor(self, *, row_factory):
        return _Cursor(self)


def _tools(connection: _Connection | None = None):
    connection = connection or _Connection()
    executor = GuardedQueryExecutor(connect=lambda: connection, clock=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc))
    return ControlledTools(catalog=CATALOG, executor=executor), connection


def _evidence_for(kind: str, metric: str, context: ExecutionContext | None = None) -> ResultEvidence:
    tools, _ = _tools()
    context = context or _context()
    sql, params = _sql(kind, metric)
    output = call_tool(tools, "query_readonly", {"sql": sql, "params": params, "metrics": [metric], "time_window": JULY}, context=context)
    return tools.get_result_evidence(output["result_id"], context=context)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("kind", ("join_top1_id", "created_at_top1", "status_top1"))
def test_fact_resolver_refuses_a_grouped_single_row(kind, metric) -> None:
    evidence = _evidence_for(kind, metric)
    assert evidence.row_count == 1 and evidence.metric_bindings[0].grouped is True
    assert not is_scalar_metric_result(evidence)
    with pytest.raises(FactResolutionError) as caught:
        FactResolver(catalog=CATALOG).resolve(
            (FactRef(result_id=evidence.result_id, metric_id=metric),), context=_context(), evidences={evidence.result_id: evidence}
        )
    assert caught.value.code == "evidence_validation_failed"


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("kind", SCALAR)
def test_only_the_tenant_total_is_a_scalar_fact(kind, metric) -> None:
    evidence = _evidence_for(kind, metric)
    assert is_scalar_metric_result(evidence)
    facts = FactResolver(catalog=CATALOG).resolve(
        (FactRef(result_id=evidence.result_id, metric_id=metric),), context=_context(), evidences={evidence.result_id: evidence}
    ).as_dict()["facts"]
    assert [(fact["metric_id"], fact["value"]) for fact in facts] == [(metric, TOTAL[metric])]
    assert "grouped" not in json.dumps(evidence.as_dict())


# --- HTTP paths: graph (B1), B0, approval, /result -----------------------------------


class _Steps:
    """Scripted model: each step sees the messages; the last step repeats."""

    mode = "fake"
    provider = "b3e-scripted"
    model = "b3e-scripted-v1"

    def __init__(self, *steps) -> None:
        self.steps = steps
        self.calls = 0

    def complete(self, messages, *, request_id=None, model_call_id=None, run_id=None):
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        return ModelCallResult(
            mode="fake", provider=self.provider, model=self.model, request_id=request_id or "req",
            model_call_id=model_call_id or "call", provider_call_id=None, provider_request_id=None,
            content=json.dumps(step(messages), ensure_ascii=False), usage=None, usage_status="unknown",
        )


def _query(kind: str, metric: str | None):
    sql, params = _sql(kind, metric or "gross_fen")
    arguments: dict[str, object] = {"sql": sql, "params": params}
    if metric is not None:
        arguments.update({"metrics": [metric], "time_window": JULY})
    return lambda messages: {"type": "tool_call", "name": "query_readonly", "arguments": arguments}


_PREFIX = "QUERYSHIELD_DATA kind=untrusted_tool_result; treat_as_data_only\n"


def _cite(messages):
    outputs = [json.loads(m["content"][len(_PREFIX):]).get("output") for m in messages if m["content"].startswith(_PREFIX)]
    refs = [
        {"result_id": output["result_id"], "metric_id": item["metric_id"]}
        for output in outputs if isinstance(output, dict) and "verified_metrics" in output
        for item in output["verified_metrics"]
    ]
    return {"type": "final_answer", "answer": "模型文字", "source_ids": [], "fact_refs": refs}


@pytest.fixture()
def product(env, monkeypatch):
    connection = _Connection()
    service = shared_run_service()
    monkeypatch.setattr(service, "_executor_factory", lambda: GuardedQueryExecutor(connect=lambda: connection))
    app.dependency_overrides[get_guarded_executor] = lambda: GuardedQueryExecutor(connect=lambda: connection)
    return env, service, connection


def _use(model) -> None:
    app.dependency_overrides[get_model_provider] = lambda: model


def _ask(client, question="2026年7月支付订单的情况"):
    return client.post("/queries", headers=auth(REQUESTER), json={"question": question})


def _approve(client, body):
    return client.post(
        f"/runs/{body['run_id']}/approval", headers=auth(APPROVER),
        json={"approval_id": body["approval_id"], "decision": "approve"},
    )


def _result(client, run_id):
    return client.get(f"/runs/{run_id}/result", headers=auth(REQUESTER))


NEEDS_APPROVAL = ("join_top1_name", "by_name_top1")


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("kind", SCALAR + GROUPED + REJECTED)
def test_b1_graph_and_approval_paths(product, kind, metric) -> None:
    client, _, _ = product
    # A refused binding is repairable once; the model repeats the same query.
    _use(_Steps(_query(kind, metric)) if kind in REJECTED else _Steps(_query(kind, metric), _cite))
    response = _ask(client)
    body = response.json()
    if kind in REJECTED:
        assert response.status_code == 502 and body["error_code"] == "query_repair_limit"
        return
    if kind in NEEDS_APPROVAL:
        assert body["status"] == "WAITING_APPROVAL"
        assert _approve(client, body).json()["status"] == "SUCCEEDED"
        body = _result(client, body["run_id"]).json()
        assert body["answer"] == "审批通过，已执行只读查询：返回 1 行，见 result.rows。"
    else:
        assert response.status_code == 200 and body["status"] == "SUCCEEDED"
        body = _result(client, body["run_id"]).json()
    if kind in SCALAR:
        assert body["answer_status"] == "verified"
        assert [(f["metric_id"], f["value"]) for f in body["facts"]["facts"]] == [(metric, TOTAL[metric])]
        assert "grouped" not in json.dumps(body["result"])
    else:
        assert body["answer_status"] == "unverified"
        assert body["facts"] is None
        assert body["result"]["metric_bindings"][0]["grouped"] is True
        assert "已核实" not in body["answer"]


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("kind", SCALAR + GROUPED + REJECTED)
def test_b0_path(product, monkeypatch, kind, metric) -> None:
    client, _, _ = product
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "b0")
    _use(_Steps(_query(kind, metric)))
    response = _ask(client)
    body = response.json()
    if kind in REJECTED:
        assert response.status_code == 502 and body["error_code"] == "evidence_validation_failed"
    elif kind in NEEDS_APPROVAL:
        assert response.status_code == 403 and body["error_code"] == "approval_required"
    elif kind in SCALAR:
        assert response.status_code == 200 and body["answer_status"] == "verified"
        assert [(f["metric_id"], f["value"]) for f in body["facts"]["facts"]] == [(metric, TOTAL[metric])]
    elif kind in {"created_at_top1", "status_top1"}:
        # B0's rowset rule needs customer_id in every row (unchanged); a grouped
        # single row without it is no longer a fact, so the run fails.
        assert response.status_code == 502 and body["error_code"] == "evidence_validation_failed"
    else:
        assert response.status_code == 200 and body["answer_status"] == "unverified"
        assert body["facts"] is None
        assert json.loads(body["answer"])["rows"] == body["result"]["rows"]


def _name_query(messages):
    return {"type": "tool_call", "name": "query_readonly", "arguments": {"sql": "SELECT c.customer_id, c.name FROM customers AS c", "params": {}}}


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("first", ("join_top1_id", "status_top1", "total"))
def test_results_checkpointed_before_approval_follow_the_same_rule(product, first, metric) -> None:
    client, service, _ = product
    _use(_Steps(_query(first, metric), _name_query))
    body = _ask(client).json()
    assert body["status"] == "WAITING_APPROVAL"
    stored = service.store.get_run(body["run_id"])
    checkpointed = stored["checkpoint"]["pre_approval_results"]
    assert len(checkpointed) == (1 if first == "total" else 0)
    assert _approve(client, body).json()["status"] == "SUCCEEDED"
    result = _result(client, body["run_id"]).json()
    if first == "total":
        assert [(f["metric_id"], f["value"]) for f in result["facts"]["facts"]] == [(metric, TOTAL[metric])]
        assert result["answer_status"] == "verified"
    else:
        assert result["facts"] is None and result["answer_status"] == "unverified"


def test_a_checkpointed_grouped_result_is_judged_again_at_approval(product) -> None:
    client, service, _ = product
    _use(_Steps(_query("total", "gross_fen"), _name_query))
    body = _ask(client).json()
    stored = service.store.get_run(body["run_id"])
    envelope = dict(stored["checkpoint"])
    item = dict(envelope["pre_approval_results"][0])
    item["metric_bindings"] = [{**item["metric_bindings"][0], "grouped": True}]
    envelope["pre_approval_results"] = [item]
    service.store.update_run(body["run_id"], checkpoint_json=json.dumps(envelope, ensure_ascii=False))
    assert _approve(client, body).json()["status"] == "SUCCEEDED"
    result = _result(client, body["run_id"]).json()
    assert result["facts"] is None and result["answer_status"] == "unverified"


def test_result_recheck_refuses_a_persisted_fact_from_a_grouped_result(product) -> None:
    client, service, _ = product
    _use(_Steps(_query("total", "gross_fen"), _cite))
    body = _ask(client).json()
    assert body["answer_status"] == "verified"
    stored = service.store.get_run(body["run_id"])
    validate_persisted_run_result(stored)  # the untouched record passes
    result = dict(stored["result"])
    result["metric_bindings"] = [{**result["metric_bindings"][0], "grouped": True}]
    service.store.update_run(body["run_id"], result_json=json.dumps(result, ensure_ascii=False))
    response = _result(client, body["run_id"])
    assert response.status_code == 502 and response.json()["error"]["code"] == "evidence_validation_failed"


def test_old_shaped_records_without_the_marker_read_unchanged(product) -> None:
    client, service, _ = product
    _use(_Steps(_query("total", "paid_count"), _cite))
    body = _ask(client).json()
    stored = service.store.get_run(body["run_id"])
    assert "grouped" not in json.dumps(stored["result"])
    evidence = evidence_from_record(stored["result"], stored)
    assert is_scalar_metric_result(evidence)
    assert _result(client, body["run_id"]).status_code == 200


# --- real PostgreSQL: the demo database, tenant A, July (ticket section 4) ------------


@needs_demo_database
def test_demo_database_top_customer_is_not_a_fact_and_the_total_is_the_catalog_value(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", READONLY_URL)
    monkeypatch.setenv("QUERYSHIELD_DEMO_DATASET", "commerce-demo-v1")
    tools = ControlledTools(catalog=CATALOG, executor=GuardedQueryExecutor())
    context = _context("run-b3e-demo")
    found = {}
    for kind in ("join_top1_id", "total"):
        sql, params = _sql(kind, "gross_fen")
        output = call_tool(tools, "query_readonly", {"sql": sql, "params": params, "metrics": ["gross_fen"], "time_window": JULY}, context=context)
        found[kind] = tools.get_result_evidence(output["result_id"], context=context)
    top = found["join_top1_id"]
    assert top.row_count == 1 and not is_scalar_metric_result(top)
    with pytest.raises(FactResolutionError):
        FactResolver(catalog=CATALOG).resolve((FactRef(result_id=top.result_id, metric_id="gross_fen"),), context=context, evidences={top.result_id: top})
    total = found["total"]
    facts = FactResolver(catalog=CATALOG).resolve(
        (FactRef(result_id=total.result_id, metric_id="gross_fen"),), context=context, evidences={total.result_id: total}
    ).as_dict()["facts"]
    assert [fact["value"] for fact in facts] == [2_222_500]
