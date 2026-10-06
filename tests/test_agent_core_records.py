"""The scripted runs of the agent core keep producing byte-identical records."""

from __future__ import annotations

import json

import pytest

import agent_core_scenarios as scenarios


PINS = json.loads(scenarios.PINS_PATH.read_text(encoding="utf-8"))


def test_every_scenario_has_a_pin_and_every_pin_a_scenario() -> None:
    assert sorted(PINS) == sorted(scenarios.SCENARIOS)


@pytest.mark.parametrize("name", sorted(scenarios.SCENARIOS))
def test_scenario_output_is_unchanged(name: str) -> None:
    first = scenarios.pinned(name)
    assert first["outcome"] == PINS[name]["outcome"], "the scenario no longer reaches the path it was written for"
    assert first["sha256"] == PINS[name]["sha256"], f"{name}: run `python tests/agent_core_scenarios.py --show {name}` on both sides and diff"


def test_pinned_records_do_not_depend_on_random_ids_or_time() -> None:
    for name in ("repair_then_success", "parallel_group_succeeds", "fake_model_native_protocol"):
        assert scenarios.pinned(name) == scenarios.pinned(name)
