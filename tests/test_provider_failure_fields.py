"""The provider fields kept on a failure event come out in one fixed order in every process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
ORDER = [
    "provider",
    "model",
    "provider_call_id",
    "provider_request_id",
    "usage_status",
    "usage",
    "http_status",
    "provider_error_code",
]
_PROGRAM = (
    "import json; from queryshield.agent.graph import _provider_failure_fields as f; "
    f"record = {{key: 1 for key in reversed({ORDER!r} + ['message', 'headers'])}}; "
    "print(json.dumps(list(f(record))))"
)


def _keys_under(hash_seed: str) -> list[str]:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(SRC)}
    done = subprocess.run([sys.executable, "-c", _PROGRAM], env=env, capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def test_the_kept_fields_have_a_fixed_order_whatever_the_hash_seed() -> None:
    assert _keys_under("1") == _keys_under("2") == _keys_under("3") == ORDER
