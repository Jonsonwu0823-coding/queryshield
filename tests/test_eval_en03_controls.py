"""EN03 negative controls tamper with the item that really carried the lineage.

Fixtures are synthetic (``test_provenance._observation``); no raw evidence
is read.  Va/Vb/Vc are the three shapes a review reproduced where the
old controls (always ``retrieval_return_records[0]``/``items[0]``) were
accepted although the product lineage was valid.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from queryshield.evaluation import provenance
from queryshield.evaluation.provenance import classify_source_lineage
from test_provenance import _observation  # noqa: E402 - pytest prepend import mode puts tests/ on sys.path


CHECK_EVAL = Path(__file__).resolve().parents[1] / "scripts" / "check_eval.py"
CONTROL_NAMES = {
    "missing_return_record",
    "cross_run_return",
    "candidate_not_returned",
    "wrong_version",
    "revoked_or_acl_invisible",
}


@pytest.fixture(scope="module")
def check_eval() -> dict[str, object]:
    return runpy.run_path(str(CHECK_EVAL))


def _other_item() -> dict[str, object]:
    return {
        "id": "semantic-metric-net@2026-09-21#0001",
        "source_id": "semantic-metric-net",
        "version": "2026-09-21",
        "text_sha256": "b" * 64,
        "source_kind": "knowledge_document",
        "visibility_check": {"passed": True},
    }


def _va() -> dict[str, object]:
    """The same run searched twice; both returns hold the item in the model context."""

    observation = _observation(retrieval=True)
    second = deepcopy(observation["retrieval_return_records"][0])
    second["retrieval_id"] = "retrieval-lineage-2"
    observation["retrieval_return_records"].append(second)
    record = deepcopy(observation["retrieval_records"][0])
    record["retrieval_id"] = "retrieval-lineage-2"
    observation["retrieval_records"].append(record)
    return observation


def _vb() -> dict[str, object]:
    """The first returned item never entered a model call."""

    observation = _observation(retrieval=True)
    observation["retrieval_return_records"][0]["items"].insert(0, _other_item())
    observation["retrieval_records"][0]["selected_ids"].insert(0, _other_item()["id"])
    return observation


def _vc() -> dict[str, object]:
    """The returned item is also in the prepared run context."""

    observation = _observation(retrieval=True)
    prepared = _observation(prepared=True)
    observation["initial_retrieval_records"] = deepcopy(prepared["initial_retrieval_records"])
    return observation


SHAPES = {"V0": lambda: _observation(retrieval=True), "Va": _va, "Vb": _vb, "Vc": _vc}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_controls_pass_on_every_valid_lineage_shape(check_eval, shape: str) -> None:
    observation = SHAPES[shape]()
    assert classify_source_lineage(observation)["status"] == "pass"

    result = check_eval["_check_source_lineage_negative_controls"](observation)

    assert result["status"] == "pass"
    assert set(result["controls"]) == CONTROL_NAMES
    assert all(value["status"] == "fail" for value in result["controls"].values())
    assert result["reduced_baseline"]["status"] == "pass"
    assert result["target_return_record_count"] == (2 if shape == "Va" else 1)
    assert result["reduced_baseline"]["prepared_context_matches_removed"] == (1 if shape == "Vc" else 0)


def test_the_controls_do_not_modify_the_observation(check_eval) -> None:
    observation = _vc()
    before = json.dumps(observation, sort_keys=True)
    check_eval["_check_source_lineage_negative_controls"](observation)
    assert json.dumps(observation, sort_keys=True) == before


def _patched_classifier(check_eval, monkeypatch, wrapper) -> None:
    function = check_eval["_check_source_lineage_negative_controls"]
    monkeypatch.setitem(function.__globals__, "classify_source_lineage", wrapper)


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_classifier_that_ignores_the_version_fails_the_controls(check_eval, monkeypatch, shape: str) -> None:
    def same_context_without_version(item, source) -> bool:
        return all(item.get(key) == source.get(key) for key in ("id", "source_id", "text_sha256"))

    monkeypatch.setattr(provenance, "_same_context", same_context_without_version)

    with pytest.raises(check_eval["LineageControlsAccepted"]) as caught:
        check_eval["_check_source_lineage_negative_controls"](SHAPES[shape]())
    assert caught.value.accepted == ["wrong_version"]


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_classifier_that_ignores_visibility_fails_the_controls(check_eval, monkeypatch, shape: str) -> None:
    def ignoring_visibility(observation):
        copy = deepcopy(observation)
        for record in copy.get("retrieval_return_records") or ():
            for item in record.get("items") or ():
                item["visibility_check"] = {"passed": True}
        return classify_source_lineage(copy)

    _patched_classifier(check_eval, monkeypatch, ignoring_visibility)

    with pytest.raises(check_eval["LineageControlsAccepted"]) as caught:
        check_eval["_check_source_lineage_negative_controls"](SHAPES[shape]())
    assert caught.value.accepted == ["revoked_or_acl_invisible"]


def test_a_classifier_that_accepts_prepared_items_without_a_return_fails_the_controls(check_eval, monkeypatch) -> None:
    """Missing return records must not be excused by the prepared context the controls removed."""

    def restoring_prepared_context(observation):
        copy = deepcopy(observation)
        copy["initial_retrieval_records"] = deepcopy(_vc()["initial_retrieval_records"])
        return classify_source_lineage(copy)

    _patched_classifier(check_eval, monkeypatch, restoring_prepared_context)

    with pytest.raises(check_eval["LineageControlsAccepted"]) as caught:
        check_eval["_check_source_lineage_negative_controls"](_vc())
    assert "missing_return_record" in caught.value.accepted


def _raw(tmp_path: Path, records: list[dict[str, object]]) -> dict[str, str]:
    b0 = _observation()
    b0.update({"evaluation_profile": "B0", "case_id": "direct", "action_input": {"entrypoint": "/queries"}})
    prepared = _observation(prepared=True)
    prepared.update({"evaluation_profile": "B1", "case_id": "prepared", "action_input": {"entrypoint": "/queries"}})
    raw_path = tmp_path / "stateful-fake-raw.json"
    raw_path.write_text(json.dumps({"raw_records": {"B0": [b0], "B1": [prepared, *records]}}), encoding="utf-8")
    return {"raw_evidence": str(raw_path)}


def _case(observation: dict[str, object], case_id: str) -> dict[str, object]:
    observation.update({"evaluation_profile": "B1", "case_id": case_id, "action_input": {"entrypoint": "/queries"}})
    return observation


def _failing_reduced_baseline(case_ids: set[str]):
    """A classifier that needs the prepared context of these cases (so the reduced baseline fails)."""

    def classify(observation):
        if observation.get("case_id") in case_ids and not observation.get("initial_retrieval_records"):
            return {**classify_source_lineage(observation), "status": "fail", "errors": ["synthetic"]}
        return classify_source_lineage(observation)

    return classify


def test_a_record_whose_reduced_baseline_fails_is_skipped_for_the_next_one(check_eval, monkeypatch, tmp_path) -> None:
    _patched_classifier(check_eval, monkeypatch, _failing_reduced_baseline({"premise"}))
    replay = _raw(tmp_path, [_case(_vc(), "premise"), _case(_va(), "usable")])

    result = check_eval["_check_stateful_source_paths"](replay, "fake", tmp_path)

    controls = result["negative_controls"]
    assert controls["status"] == "pass"
    assert controls["records_skipped_for_premise"] == 1
    assert controls["target_return_record_count"] == 2


def test_no_record_with_a_passing_reduced_baseline_is_not_run_never_a_pass(check_eval, monkeypatch, tmp_path) -> None:
    _patched_classifier(check_eval, monkeypatch, _failing_reduced_baseline({"premise"}))
    replay = _raw(tmp_path, [_case(_vc(), "premise")])

    real = check_eval["_check_stateful_source_paths"](replay, "real", tmp_path)
    assert real["negative_controls"] == {
        "status": "not_run",
        "reason": "no control record kept a passing reduced baseline",
        "premise_failures": ["reduced_baseline_not_pass"],
        "records_tried": 1,
    }
    with pytest.raises(AssertionError, match="positive/negative controls are incomplete"):
        check_eval["_check_stateful_source_paths"](replay, "fake", tmp_path)


def test_accepted_controls_are_named_in_the_failure_evidence(check_eval, monkeypatch, tmp_path) -> None:
    def same_context_without_version(item, source) -> bool:
        return all(item.get(key) == source.get(key) for key in ("id", "source_id", "text_sha256"))

    monkeypatch.setattr(provenance, "_same_context", same_context_without_version)
    replay = _raw(tmp_path, [_case(_va(), "retrieved")])

    with pytest.raises(AssertionError, match="wrong_version"):
        check_eval["_check_stateful_source_paths"](replay, "real", tmp_path)

    diagnostic = json.loads((tmp_path / "real-source-lineage-failure.json").read_text(encoding="utf-8"))
    assert diagnostic["failed_assertion"] == "negative_controls_accepted"
    assert diagnostic["accepted_controls"] == ["wrong_version"]
    assert diagnostic["dataset_split"] == "development_only"
