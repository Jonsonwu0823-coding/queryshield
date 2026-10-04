"""Validate persisted result/fact bindings before the HTTP result is exposed."""

from collections.abc import Mapping
from datetime import datetime
import re

from queryshield.agent.proposals import ExecutionContext, FactRef, MetricBinding, ResultEvidence
from queryshield.facts.facts import FactResolutionError, FactResolver


def evidence_from_record(raw: Mapping[str, object], run: Mapping[str, object]) -> ResultEvidence:
    """Rebuild one stored ResultEvidence and check it belongs to the stored run."""

    if not isinstance(raw, Mapping):
        raise ValueError("missing result")
    if any(raw.get(key) != run.get(key) for key in ("run_id", "tenant_id", "principal_id")):
        raise ValueError("result ownership mismatch")
    if not isinstance(raw.get("result_id"), str) or not raw["result_id"]:
        raise ValueError("missing result identity")
    rows = raw["rows"]
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("invalid result rows")
    if type(raw["row_count"]) is not int or raw["row_count"] != len(rows):
        raise ValueError("row count mismatch")
    if any(not isinstance(raw.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", raw[key])
           for key in ("query_sha256", "params_sha256")):
        raise ValueError("missing execution hashes")
    if not isinstance(raw.get("observed_at"), str):
        raise ValueError("missing observation timestamp")
    evidence = ResultEvidence(
        result_id=raw["result_id"], run_id=raw["run_id"], tenant_id=raw["tenant_id"],
        principal_id=raw["principal_id"], rows=tuple(dict(row) for row in rows),
        row_count=raw["row_count"], query_sha256=raw["query_sha256"],
        params_sha256=raw["params_sha256"],
        observed_at=datetime.fromisoformat(raw["observed_at"].replace("Z", "+00:00")),
        policy_version=raw["policy_version"], catalog_version=raw["catalog_version"],
        metric_bindings=tuple(MetricBinding(**dict(item)) for item in raw["metric_bindings"]),
        metric_plan_id=raw.get("metric_plan_id"),
    )
    if evidence.observed_at.tzinfo is None:
        raise ValueError("missing observation timezone")
    return evidence


def validate_persisted_run_result(run: Mapping[str, object]) -> None:
    """A visible run does not by itself authorize every result stored in it.

    ``result`` is the run's primary evidence.  A run that combined facts from
    earlier results of the same run (for example metric results verified
    before an approval paused it) lists those in ``result.supporting_results``;
    each passes the same ownership and integrity checks.
    """
    try:
        raw = run["result"]
        evidence = evidence_from_record(raw, run)
        evidences = {evidence.result_id: evidence}
        supporting = raw.get("supporting_results", [])
        if not isinstance(supporting, list):
            raise ValueError("invalid supporting results")
        for item in supporting:
            extra = evidence_from_record(item, run)
            if extra.result_id in evidences:
                raise ValueError("duplicate result evidence")
            evidences[extra.result_id] = extra
        facts = run.get("facts")
        if facts is None:
            return
        if isinstance(facts, Mapping):
            facts = facts.get("facts")
        if not isinstance(facts, list) or any(not isinstance(item, Mapping) for item in facts):
            raise ValueError("invalid facts")
        context = ExecutionContext(
            run_id=run["run_id"], tenant_id=run["tenant_id"],
            principal_id=run["principal_id"], role=run["role"],
        )
        resolved = FactResolver().resolve(
            tuple(FactRef(result_id=item["result_id"], metric_id=item["metric_id"]) for item in facts),
            context=context, evidences=evidences,
        ).as_dict()["facts"]
        for actual, verified in zip(facts, resolved, strict=True):
            if any(type(actual.get(key)) is not type(value) or actual.get(key) != value
                   for key, value in verified.items()):
                raise ValueError("fact differs from execution evidence")
    except (KeyError, TypeError, ValueError) as exc:
        raise FactResolutionError("evidence_validation_failed", "stored result evidence is invalid") from exc
