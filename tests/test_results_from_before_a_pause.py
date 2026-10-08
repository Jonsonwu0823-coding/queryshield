"""Query results verified before a pause are kept with the pause and put back on resume.

A resume is a new execution with new tools; the answer check still loads every
successful query result of the run.  The pause stores those results' evidence in
the run's server checkpoint (``paused_results``); the resume rebuilds each one,
checked against the stored run, and puts it back before the graph runs.  A stored
record that is malformed or not this run's refuses the resume (409
``checkpoint_invalid``) and leaves the run waiting.

The sources that could be mixed up take different values: the pause ran a count
(``paid_count``), the resume a sum (``gross_fen``).
"""

from __future__ import annotations

import json

import pytest

from queryshield.agent import ModelCallStore
from queryshield.agent.proposals import ExecutionContext
from queryshield.api.main import app, get_model_provider
from queryshield.approval.service import ApprovalConflict, shared_run_service
from queryshield.mcp_metadata.tools import McpMetadataTools
from queryshield.tools import ControlledTools, ToolError

from mcp_helpers import config
from test_agent_step_writes import _Metered
from test_approval_api_pins import APPROVER_A, _edit
from test_clarification import COUNT_SQL, FILTER, REQUESTER, _ask, _cite, _params, _query, service  # noqa: F401
from test_http_queries import REQUESTER as REQUESTER_TOKEN
from test_http_queries import auth, env  # noqa: F401  (env is a fixture)
from test_resumed_and_approved_execution import AMBIGUOUS, COUNT, _assert_one_terminal_last, _cite_all, _events

# A query with no metric: its result is not a metric value, yet the answer check loads it.
UNBOUND_COUNT = {"type": "tool_call", "name": "query_readonly", "arguments": {"sql": COUNT_SQL, "params": _params()}}
# Customer names grouped with the confirmed metric: after a resume it waits for approval.
NAMES_WITH_GROSS = _query(
    "SELECT o.tenant_id, o.customer_id, c.name, COALESCE(SUM(o.amount_fen), 0) AS gross_fen FROM orders AS o "
    f"JOIN customers AS c ON c.tenant_id = o.tenant_id AND c.customer_id = o.customer_id WHERE {FILTER} "
    "GROUP BY o.tenant_id, o.customer_id, c.name",
    ("gross_fen",),
)
ASK_AGAIN = _ask("请问是哪一个月？")


def _start(service, steps) -> dict:
    svc, _ = service
    deps = svc.default_dependencies()
    deps.model, deps.retriever = _Metered(steps), None
    run = svc.run_sync(identity=REQUESTER, question=AMBIGUOUS, time_window=None, deps=deps)
    assert run["status"] == "WAITING_USER", run
    return run


def _resume(service, run, steps, *, answer="按支付金额", model=None) -> dict:
    svc, executor = service
    return svc.resume_waiting_user(
        run_id=run["run_id"], answer=answer, identity=REQUESTER, model=model or _Metered(steps),
        call_store=ModelCallStore(), executor=executor,
    )


def _query_result_ids(svc, run_id: str) -> list[str]:
    return [
        e["result_id"] for e in svc.store.events(run_id, after_event_id=0, limit=1000)
        if e["type"] == "agent_step" and e["payload"].get("kind") == "tool_call" and e["result_id"]
    ]


def _checkpoint_result_ids(run: dict) -> set[str]:
    """The results the resumed answer check loads: successful tool records of the agent checkpoint."""

    records = run["checkpoint"]["agent_checkpoint"]["tool_results"]
    return {
        r["output"]["result_id"] for r in records
        if r.get("status") == "succeeded" and isinstance(r.get("output"), dict) and isinstance(r["output"].get("result_id"), str)
    }


def _paused_ids(run: dict) -> set[str]:
    return {item["result_id"] for item in run["checkpoint"].get("paused_results", [])}


def _facts_by_result(run: dict) -> dict[str, tuple[str, object]]:
    return {f["result_id"]: (f["metric_id"].removeprefix("metric."), f["value"]) for f in run["facts"]["facts"]}


def _rows_by_result(run: dict) -> dict[str, list]:
    results = [run["result"], *run["result"].get("supporting_results", [])]
    return {item["result_id"]: item["rows"] for item in results}


# --- a resume can cite what the run verified before the pause -------------------------------------------------


def test_a_resume_cites_the_result_from_before_the_pause_and_its_own(service) -> None:
    svc, _ = service
    run = _start(service, [COUNT, _query()])
    [before] = _query_result_ids(svc, run["run_id"])
    assert _paused_ids(svc.store.get_run(run["run_id"])) == {before}

    done = _resume(service, run, [_query(), _cite_all])

    assert (done["status"], done["error_code"]) == ("SUCCEEDED", None)
    [_, after] = _query_result_ids(svc, run["run_id"])
    facts, rows = _facts_by_result(done), _rows_by_result(done)
    assert set(facts) == {before, after}
    # Each fact carries the value of the result it cites.
    assert facts[before] == ("paid_count", rows[before][0]["paid_count"])
    assert facts[after] == ("gross_fen", rows[after][0]["gross_fen"])
    assert rows[before][0]["paid_count"] != rows[after][0]["gross_fen"]
    # One is the primary result, the other a supporting one.
    assert {done["result"]["result_id"], *(item["result_id"] for item in done["result"]["supporting_results"])} == {before, after}
    # Counts are the pause's plus the resume's.
    assert (done["model_call_count"], done["tool_call_count"], done["sql_exec_count"]) == (4, 3, 2)
    _assert_one_terminal_last(_events(svc, run["run_id"]))


def test_an_answer_that_leaves_out_the_earlier_result_fails_on_the_citation_rule(service) -> None:
    svc, _ = service
    run = _start(service, [COUNT, _query()])

    done = _resume(service, run, [_query(), _cite, _cite])

    assert done["status"] == "FAILED"
    # The earlier result was loaded; the answer fails because it does not cite every result.
    assert "result_not_found" not in json.dumps(done["checkpoint"], ensure_ascii=False)


def test_a_result_that_is_not_a_metric_value_is_kept_too(service) -> None:
    svc, _ = service
    run = _start(service, [UNBOUND_COUNT, _query()])
    [before] = _query_result_ids(svc, run["run_id"])
    assert _paused_ids(svc.store.get_run(run["run_id"])) == {before}

    done = _resume(service, run, [_query(), _cite_all])

    assert (done["status"], done["error_code"]) == ("SUCCEEDED", None)
    assert [metric for metric, _ in _facts_by_result(done).values()] == ["gross_fen"]


def test_two_clarifications_keep_every_earlier_result(service) -> None:
    svc, _ = service
    run = _start(service, [COUNT, _query()])
    first_pause = svc.store.get_run(run["run_id"])
    assert _paused_ids(first_pause) == _checkpoint_result_ids(first_pause) != set()

    again = _resume(service, run, [_query(), ASK_AGAIN])
    assert again["status"] == "WAITING_USER"
    assert _paused_ids(again) == _checkpoint_result_ids(again) == set(_query_result_ids(svc, run["run_id"]))
    assert len(_paused_ids(again)) == 2

    done = _resume(service, run, [_query(), _cite_all], answer="2026年9月")

    assert (done["status"], done["error_code"]) == ("SUCCEEDED", None)
    assert set(_facts_by_result(done)) == set(_query_result_ids(svc, run["run_id"]))
    assert len(_facts_by_result(done)) == 3
    assert (done["model_call_count"], done["tool_call_count"], done["sql_exec_count"]) == (6, 4, 3)


def test_a_resume_that_waits_for_approval_keeps_the_earlier_metric_result(service) -> None:
    svc, _ = service
    run = _start(service, [COUNT, _query()])
    [before] = _query_result_ids(svc, run["run_id"])

    pending = _resume(service, run, [_query(), NAMES_WITH_GROSS])

    assert pending["status"] == "WAITING_APPROVAL"
    kept = {item["result_id"] for item in pending["checkpoint"]["pre_approval_results"]}
    assert before in kept and len(kept) == 2
    done = svc.approve(run_id=run["run_id"], approval_id=pending["approval_id"], identity=APPROVER_A, decision="approve")
    assert done["status"] == "SUCCEEDED"
    assert before in _facts_by_result(done) and len(_facts_by_result(done)) == 2


def test_a_pause_without_a_query_result_stores_no_entry(service) -> None:
    svc, _ = service
    run = _start(service, [_query()])
    assert "paused_results" not in svc.store.get_run(run["run_id"])["checkpoint"]


def test_a_run_paused_without_the_entry_resumes_as_before(service) -> None:
    svc, _ = service
    run = _start(service, [_query()])
    _edit(svc, run, envelope=lambda envelope: envelope.pop("paused_results", None))

    done = _resume(service, run, [_query(), _cite_all])

    assert (done["status"], done["error_code"]) == ("SUCCEEDED", None)


# --- a stored record that is malformed or not this run's -------------------------------------------------------


def _first(field, value):
    def change(envelope):
        envelope["paused_results"][0][field] = value

    return change


def _drop(field):
    def change(envelope):
        del envelope["paused_results"][0][field]

    return change


def _extra_row(envelope):
    envelope["paused_results"][0]["rows"].append(dict(envelope["paused_results"][0]["rows"][0]))


def _extra_binding_field(envelope):
    envelope["paused_results"][0]["metric_bindings"][0]["unexpected"] = 1


def _not_a_list(envelope):
    envelope["paused_results"] = {"result": envelope["paused_results"][0]}


@pytest.mark.parametrize(
    "tamper",
    [
        _first("tenant_id", "B"),
        _first("principal_id", "principal-other"),
        _first("run_id", "run-other"),
        _extra_row,
        _not_a_list,
        _drop("rows"),
        _drop("row_count"),
        _drop("policy_version"),
        _drop("catalog_version"),
        _drop("metric_bindings"),
        _extra_binding_field,
        _first("query_sha256", "not-a-hash"),
        _first("observed_at", "yesterday"),
    ],
    ids=[
        "tenant", "principal", "run", "extra_row", "not_a_list", "no_rows", "no_row_count", "no_policy_version",
        "no_catalog_version", "no_metric_bindings", "binding_extra_field", "bad_hash", "bad_timestamp",
    ],
)
def test_a_stored_result_that_is_not_this_runs_refuses_the_resume_and_leaves_the_run_waiting(service, tamper) -> None:
    svc, _ = service
    run = _start(service, [COUNT, _query()])
    _edit(svc, run, envelope=tamper)
    stored = svc.store.get_run(run["run_id"])
    events = _events(svc, run["run_id"])
    model = _Metered([_query(), _cite_all])

    with pytest.raises(ApprovalConflict) as refused:
        _resume(service, run, [], model=model)

    assert refused.value.code == "checkpoint_invalid"
    assert model.messages == []
    assert svc.store.get_run(run["run_id"]) == stored
    assert _events(svc, run["run_id"]) == events


# --- the tools' restore method ---------------------------------------------------------------------------------


def _evidence(service):
    svc, _ = service
    run = _start(service, [COUNT, _query()])
    from queryshield.facts.persisted import evidence_from_record

    stored = svc.store.get_run(run["run_id"])
    return evidence_from_record(stored["checkpoint"]["paused_results"][0], stored)


def _context(evidence, **changes) -> ExecutionContext:
    fields = {"run_id": evidence.run_id, "tenant_id": evidence.tenant_id, "principal_id": evidence.principal_id, "role": "requester"}
    return ExecutionContext(**{**fields, **changes})


@pytest.mark.parametrize("make_tools", [ControlledTools, lambda: McpMetadataTools(metadata_config=config())], ids=["local", "mcp"])
def test_restored_evidence_is_visible_only_to_its_own_run_tenant_and_requester(service, make_tools) -> None:
    evidence = _evidence(service)
    for changes in ({"run_id": "run-other"}, {"tenant_id": "B"}, {"principal_id": "principal-other"}):
        tools = make_tools()
        with pytest.raises(ToolError) as refused:
            tools.restore_result_evidence(evidence, context=_context(evidence, **changes))
        assert refused.value.code == "result_not_found"
        with pytest.raises(ToolError):
            tools.get_result_evidence(evidence.result_id, context=_context(evidence))

    tools = make_tools()
    tools.restore_result_evidence(evidence, context=_context(evidence))
    assert tools.get_result_evidence(evidence.result_id, context=_context(evidence)) == evidence


# --- the MCP setting, over HTTP --------------------------------------------------------------------------------


def test_with_mcp_metadata_tools_a_resume_cites_the_result_from_before_the_pause(env, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setenv("QUERYSHIELD_METADATA_TOOLS", "mcp")
    model = _Metered([COUNT, _query()])
    app.dependency_overrides[get_model_provider] = lambda: model
    paused = env.post("/queries", headers=auth(REQUESTER_TOKEN), json={"question": AMBIGUOUS}).json()
    assert paused["status"] == "WAITING_USER", paused

    model.steps += [_query(), _cite_all]
    response = env.post(f"/runs/{paused['run_id']}/resume", headers=auth(REQUESTER_TOKEN), json={"answer": "按支付金额"})

    body = response.json()
    assert (response.status_code, body["status"]) == (200, "SUCCEEDED"), body
    assert sorted(f["metric_id"].removeprefix("metric.") for f in body["facts"]["facts"]) == ["gross_fen", "paid_count"]
    run = shared_run_service().store.get_run(paused["run_id"])
    assert len(run["result"]["supporting_results"]) == 1
