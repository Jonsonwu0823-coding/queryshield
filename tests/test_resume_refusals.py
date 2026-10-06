"""What a resume refuses, in what order, and what the HTTP caller gets back."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from queryshield.agent.runtime import FAILED_ERROR_HTTP, RUN_OUTCOMES
from queryshield.api.main import _resume_run_response
from queryshield.approval.service import ApprovalConflict, ObjectNotFound, shared_run_service

from test_http_queries import REQUESTER, ask, auth, env  # noqa: F401  (env is a fixture)
from test_clarification import service  # noqa: F401  (a fixture)
from test_approval_api_pins import OTHER_ROLE, REQUESTER_A, _edit, _NoModel, _resume_fails, _waiting_user


# --- the service checks who is asking before it looks at the run's state -----------


def test_a_wrong_role_learns_nothing_about_a_run_that_is_not_waiting(service) -> None:
    svc, executor = service
    run = _waiting_user(service)
    svc.cancel(run_id=run["run_id"], identity=REQUESTER_A)

    def resume_as(identity):
        return svc.resume_waiting_user(
            run_id=run["run_id"], answer="按支付金额", identity=identity, model=_NoModel(), call_store=None, executor=executor
        )

    with pytest.raises(ObjectNotFound):
        resume_as(OTHER_ROLE)
    with pytest.raises(ApprovalConflict) as owner:
        resume_as(REQUESTER_A)
    assert (owner.value.code, str(owner.value)) == ("invalid_run_state", "run is not waiting for user input")


# --- a checkpoint that belongs to someone else is "not found", not a conflict ------------


def test_a_checkpoint_of_another_identity_is_reported_as_not_found(service) -> None:
    svc, _ = service
    run = _waiting_user(service)
    _edit(svc, run, agent=lambda checkpoint: checkpoint["context"].update(principal_id="someone-else"))
    _resume_fails(service, run, ApprovalConflict, "not_found", "resume checkpoint could not be restored")


def test_the_same_refusal_over_http_is_a_409_that_says_not_found(env) -> None:
    body = ask(env, "2026年9月销售额是多少？").json()
    assert body["status"] == "WAITING_USER"
    svc = shared_run_service()
    stored = svc.store.get_run(body["run_id"])
    checkpoint = json.loads(json.dumps(stored["checkpoint"]))
    checkpoint["agent_checkpoint"]["context"]["principal_id"] = "someone-else"
    svc.store.update_run(body["run_id"], checkpoint_json=json.dumps(checkpoint))
    response = env.post(f"/runs/{body['run_id']}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_found"
    assert response.json()["error"]["message"] == "任务续跑状态或checkpoint无效"
    assert svc.store.get_run(body["run_id"])["status"] == "WAITING_USER"


# --- a checkpoint of an earlier format is refused, not upgraded --------------------------


@pytest.mark.parametrize("version", ["qs-bounded-agent-checkpoint-v1", "qs-bounded-agent-checkpoint-v2"])
def test_a_stored_checkpoint_of_version_one_or_two_is_invalid(service, version) -> None:
    svc, _ = service
    run = _waiting_user(service)
    _edit(svc, run, agent=lambda checkpoint: checkpoint.update(checkpoint_version=version))
    _resume_fails(service, run, ApprovalConflict, "checkpoint_invalid", "resume checkpoint could not be restored")


# --- the route answers a run that is not waiting before the service is asked -------------


def test_resuming_a_finished_run_says_the_task_is_not_waiting(env) -> None:
    done = ask(env, "2026年9月已支付订单总额")
    assert done.status_code == 200 and done.json()["status"] == "SUCCEEDED"
    response = env.post(f"/runs/{done.json()['run_id']}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额"})
    assert response.status_code == 409
    assert (response.json()["error"]["code"], response.json()["error"]["message"]) == ("invalid_run_state", "任务不在等待用户状态")


# --- every outcome of a resume is a 200 or carries the error body ------------------------

RUN = {
    "run_id": "run-1", "tenant_id": "A", "principal_id": "principal-A", "role": "requester", "question": "q",
    "created_at": "2026-09-21T00:00:00Z", "updated_at": "2026-09-21T00:00:00Z", "mode": "fake",
    "model_call_count": 1, "tool_call_count": 0, "sql_exec_count": 0,
}
STATUSES = sorted({outcome.run_status for outcome in RUN_OUTCOMES.values()} | {"CANCELLED"})
ERROR_CODES = [None, "run_failed", *sorted(FAILED_ERROR_HTTP)]


@pytest.mark.parametrize("status", STATUSES)
def test_a_resume_outcome_is_a_200_or_has_an_error_body(status: str) -> None:
    for error_code in ERROR_CODES:
        request = SimpleNamespace(state=SimpleNamespace(request_id="request-1"))
        response = _resume_run_response(request, {**RUN, "status": status, "error_code": error_code})
        body = json.loads(response.body)
        assert response.status_code == 200 or "error" in body, (status, error_code, response.status_code)
        if response.status_code != 200:
            assert response.status_code >= 400 and body["error"]["code"] == (error_code or "run_failed" if status != "CANCELLED" else "run_cancelled")
