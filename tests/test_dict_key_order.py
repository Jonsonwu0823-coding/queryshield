"""Dictionaries built from a set come out in one fixed key order in every process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
HASH_SEEDS = ("1", "2", "3", "4", "5")
WINDOW = {"interval": "month", "timezone": "UTC", "end": "2026-10-01", "start": "2026-09-01"}
B0_RESULT = {
    name: 1
    for name in ("elapsed_ms", "usage", "side_effects", "invariants", "facts", "http_status", "terminal_state", "status")
}

PROGRAMS = {
    "time_window": (
        "from queryshield.agent.context import _validate_time_window as f; "
        f"print(json.dumps(list(f({WINDOW!r}))))"
    ),
    "time_window_without_interval": (
        "from queryshield.agent.context import _validate_time_window as f; "
        f"print(json.dumps(list(f({ {k: v for k, v in WINDOW.items() if k != 'interval'} !r}))))"
    ),
    "b0_observation": (
        "from queryshield.evaluation.profile_runner import normalize_profile_observation as f; "
        f"print(json.dumps(list(f('B0-single-pass', {B0_RESULT!r}, case_id='case-1'))))"
    ),
}
ORDERS = {
    "time_window": ["start", "end", "timezone", "interval"],
    "time_window_without_interval": ["start", "end", "timezone"],
    "b0_observation": [
        "case_id", "status", "terminal_state", "http_status", "facts", "invariants", "side_effects", "usage", "elapsed_ms",
    ],
}


def _keys_under(program: str, hash_seed: str) -> list[str]:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(SRC)}
    done = subprocess.run(
        [sys.executable, "-c", "import json; " + program], env=env, capture_output=True, text=True, check=True
    )
    return json.loads(done.stdout)


@pytest.mark.parametrize("name", PROGRAMS)
def test_the_keys_have_a_fixed_order_whatever_the_hash_seed(name: str) -> None:
    assert [_keys_under(PROGRAMS[name], seed) for seed in HASH_SEEDS] == [ORDERS[name]] * len(HASH_SEEDS)
