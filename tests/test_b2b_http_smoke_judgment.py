"""B3c-1 R1: the HTTP smoke judges the undated-question step by where it ends."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest


SMOKE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "b2b_http_smoke.py"))
judge_time_request = SMOKE["judge_time_request"]

BOUNCE = {"decision": "not_needed", "error_code": "clarification_not_needed", "signal": "two_values"}
MODEL_TEXT = {"decision": "model_text", "error_code": None, "signal": "weak_only"}


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        # the model's own time ask waits: go on to time_resume
        ((202, "WAITING_USER", None, [MODEL_TEXT], False), ("waiting", [], [], True)),
        # sent back once, then the model asked for the time: by design, a known gap only
        (
            (202, "WAITING_USER", None, [BOUNCE, MODEL_TEXT], False),
            ("waiting", [], ["time_ask_bounced_then_recovered"], True),
        ),
        # the server replaced the model's time ask with a catalog question
        ((202, "WAITING_USER", None, [MODEL_TEXT], True), ("rewritten", ["time_ask_rewritten"], [], False)),
        # sent back twice: the run ends
        ((502, "FAILED", "clarification_not_needed", [BOUNCE, BOUNCE], False), ("bounced", ["time_ask_bounced"], [], False)),
        # the model guessed a window
        ((200, "SUCCEEDED", None, [], False), ("guessed", [], ["time_window_guessed_by_model"], False)),
        # anything else
        ((502, "FAILED", "answer_not_grounded", [], False), ("other", ["time_request"], [], False)),
        ((500, None, None, [], False), ("other", ["time_request"], [], False)),
    ],
    ids=["waiting", "bounced-then-recovered", "rewritten", "bounced-twice", "guessed", "other-failed", "server-error"],
)
def test_time_request_is_judged_by_its_final_state(args, expected) -> None:
    assert judge_time_request(*args) == expected
