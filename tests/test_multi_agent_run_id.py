"""Every model call of a B2 run, the coordinator's and each sub-agent's, carries the root run's ``X-Run-Id``.

The Real adapters against the fake upstream, as in tests/test_run_id_header.py; the upstream's
calls are counted and compared with the run's model_call_count.
"""

from __future__ import annotations

import pytest

from multi_agent_support import COMPOSITE
from test_run_id_header import _ask, _assert_tagged, retriever, service, upstream  # noqa: F401  (fixtures)

pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")


def test_every_call_of_a_delegated_run_carries_the_root_run_id(service, upstream, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", "b2")
    body = _ask(service, COMPOSITE)
    assert body["status"] == "SUCCEEDED" and body["profile"] == "B2-multi-agent"
    chats = upstream.sent("chat/completions")
    _assert_tagged(chats, body["run_id"])
    assert len(chats) == body["model_call_count"] == 5  # search, delegate, two sub-agents, the answer
