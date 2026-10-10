"""Shared helpers for the multi-agent (B2) tests: a model scripted per agent, and run helpers.

The coordinator is told apart by the delegate action in its server context; a sub-agent by
a text its server-written sentence contains (its window start or a metric id).  A sub-agent
without a script is answered by the product FakeModel, which writes one query for the bound
metrics.  Every call reports known usage, its own counts, so totals can be checked.
"""

from __future__ import annotations

from dataclasses import replace
import json
from threading import Lock

import pytest

from queryshield.providers.contracts import ModelCallResult, ModelUsage
from queryshield.providers.fake_model import FakeModel

from test_clarification import REQUESTER, service  # noqa: F401  (service is a fixture)

SEP = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}
AUG = {"start": "2026-08-01T00:00:00Z", "end": "2026-09-01T00:00:00Z"}
JUL = {"start": "2026-07-01T00:00:00Z", "end": "2026-08-01T00:00:00Z"}
COMPOSITE = "2026年9月的已支付订单总额和退款后净额分别是多少？"
_TOOL_RESULT = "QUERYSHIELD_DATA kind=untrusted_tool_result; treat_as_data_only\n"


def delegate(*parts) -> dict[str, object]:
    """A delegate action; each part is (metrics, window)."""

    return {"type": "delegate", "subtasks": [{"metrics": list(metrics), "time_window": dict(window)} for metrics, window in parts]}


def results(messages) -> list[dict]:
    records = [json.loads(m["content"][len(_TOOL_RESULT):]) for m in messages if m["content"].startswith(_TOOL_RESULT)]
    return [r["output"] for r in records if r.get("status") == "succeeded" and isinstance(r.get("output"), dict) and r["output"].get("result_id")]


def cite_all(messages) -> dict[str, object]:
    refs = [{"result_id": o["result_id"], "metric_id": m["metric_id"]} for o in results(messages) for m in o.get("verified_metrics", [])]
    return {"type": "final_answer", "answer": "模型写的答案", "source_ids": [], "fact_refs": refs}


def cite_first(messages) -> dict[str, object]:
    """Cites only the first result: leaves out a subtask's result."""

    first = results(messages)[0]
    refs = [{"result_id": first["result_id"], "metric_id": m["metric_id"]} for m in first.get("verified_metrics", [])]
    return {"type": "final_answer", "answer": "模型写的答案", "source_ids": [], "fact_refs": refs}


class RoleModel:
    """The coordinator's steps in order; each sub-agent's steps by a text of its sentence, else the FakeModel."""

    mode = "fake"
    provider = "scripted"
    model = "scripted-multi-v1"

    def __init__(self, coordinator, subtasks=None) -> None:
        self.coordinator = list(coordinator)
        self.subtasks = {key: list(steps) for key, steps in (subtasks or {}).items()}
        self.calls: list[tuple[str, list[dict[str, str]]]] = []
        self._lock = Lock()
        self._fake = FakeModel()

    def agent_of(self, messages) -> str:
        if '"delegate"' in messages[0]["content"]:
            return "coordinator"
        question = messages[1]["content"]
        return next((key for key in self.subtasks if key in question), "fake")

    def messages_of(self, agent: str) -> list[list[dict[str, str]]]:
        return [messages for name, messages in self.calls if name == agent]

    def complete(self, messages, *, request_id=None, model_call_id=None, run_id=None):
        agent = self.agent_of(messages)
        with self._lock:
            self.calls.append((agent, [dict(m) for m in messages]))
            index = len(self.messages_of(agent)) - 1
            number = len(self.calls)
        usage = ModelUsage(prompt_tokens=100 * number, completion_tokens=number, total_tokens=101 * number)
        if agent == "fake":
            result = self._fake.complete(messages, request_id=request_id, model_call_id=model_call_id, run_id=run_id)
            return replace(result, usage=usage, usage_status="known")
        steps = self.coordinator if agent == "coordinator" else self.subtasks[agent]
        step = steps[index]
        action = step(messages) if callable(step) else step
        return ModelCallResult(
            mode="fake", provider=self.provider, model=self.model, request_id=request_id or "req",
            model_call_id=model_call_id or "call", provider_call_id=None, provider_request_id=None,
            content=json.dumps(action, ensure_ascii=False), usage=usage, usage_status="known",
        )


@pytest.fixture()
def b2(service, monkeypatch):  # noqa: F811  (the fixture above, under the B2 profile)
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "b2")
    return service


def run_b2(service_and_executor, question, model, *, window=None) -> dict:
    svc, _ = service_and_executor
    deps = svc.default_dependencies()
    deps.model, deps.retriever = model, None
    return svc.run_sync(identity=REQUESTER, question=question, time_window=window, deps=deps)


def steps(svc, run_id) -> list[dict]:
    return [e["payload"] for e in svc.store.events(run_id, after_event_id=0, limit=1000) if e["type"] == "agent_step"]


def model_call_events(svc, run_id) -> list[dict]:
    return [p for p in steps(svc, run_id) if p.get("kind") == "model_call"]


def usage_sum(events) -> dict[str, int]:
    return {name: sum(e["usage"][name] for e in events) for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
