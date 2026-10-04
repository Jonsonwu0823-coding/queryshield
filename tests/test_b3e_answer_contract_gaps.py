"""B3e part 3: two B3c-2 behaviours the B3c-2 acceptance found correct but unpinned (B1, B2).

B1: a run cancelled while its resume is running answers 409 run_cancelled
(mutation IM5 returned 200).  B2: a SUCCEEDED run stored before B3c-2 has no
answer_status; /result reads it as unverified even with facts (mutation IM4
read it as verified).
"""

from __future__ import annotations

import json

from queryshield.api.main import app, get_model_provider
from queryshield.approval.service import shared_w04_service
from queryshield.providers.fake_model import FakeModel

from test_b2b_http_queries import REQUESTER, ask, auth, env  # noqa: F401  (env is a fixture)


class CancelDuringResume(FakeModel):
    """The Fake model; once armed, the owner cancels the run inside the model call."""

    def __init__(self) -> None:
        super().__init__()
        self.cancel_run: dict | None = None

    def complete(self, messages, **kwargs):
        if self.cancel_run is not None:
            run = self.cancel_run
            shared_w04_service().cancel(
                run_id=run["run_id"],
                identity={"tenant_id": run["tenant_id"], "principal_id": run["principal_id"], "role": run["role"]},
            )
        return super().complete(messages, **kwargs)


def test_b1_a_run_cancelled_during_resume_answers_409_run_cancelled(env) -> None:
    model = CancelDuringResume()
    app.dependency_overrides[get_model_provider] = lambda: model
    waiting = ask(env, "2026年9月销售额是多少？").json()
    assert waiting["status"] == "WAITING_USER"
    model.cancel_run = shared_w04_service().store.get_run(waiting["run_id"])

    resumed = env.post(f"/runs/{waiting['run_id']}/resume", headers=auth(REQUESTER), json={"answer": "按支付金额"})
    body = resumed.json()
    assert resumed.status_code == 409
    assert body["error"]["code"] == "run_cancelled"
    assert body["run_id"] == waiting["run_id"]
    assert body["status"] == "CANCELLED"
    assert body["answer_status"] is None
    assert shared_w04_service().store.get_run(waiting["run_id"])["status"] == "CANCELLED"


def test_b2_a_run_stored_before_b3c2_reads_unverified_even_with_facts(env) -> None:
    done = ask(env, "2026年9月已支付订单有几笔？").json()
    assert done["status"] == "SUCCEEDED" and done["answer_status"] == "verified"
    store = shared_w04_service().store
    envelope = dict(store.get_run(done["run_id"])["checkpoint"])
    # The B3c-2 envelope fields did not exist before B3c-2.
    envelope.pop("answer_status")
    envelope.pop("answer_source_ids", None)
    store.update_run(done["run_id"], checkpoint_json=json.dumps(envelope, ensure_ascii=False))

    result = env.get(f"/runs/{done['run_id']}/result", headers=auth(REQUESTER))
    body = result.json()
    assert result.status_code == 200
    assert body["facts"]["facts"]  # the facts are there and pass the persisted re-check
    assert body["answer_status"] == "unverified"
    assert body["source_ids"] == []
