from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import hashlib

from queryshield.agent.proposals import ExecutionContext, FactRef, MetricBinding, ResultEvidence
from queryshield.agent.context import NET_FEN_PLAN_ID
from queryshield.catalog import SemanticCatalog, load_default_catalog


FACTS_SCHEMA_VERSION = "qs-facts-v1"
_METRIC_LABELS = {
    "paid_count": "已支付订单数",
    "gross_fen": "支付订单总额",
    "refund_fen": "退款金额",
    "net_fen": "退款后净额",
}


class FactResolutionError(ValueError):
    """A model reference cannot be turned into server-verified facts."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class Fact:
    fact_id: str
    metric_id: str
    label: str
    value: int
    unit: str
    display_value: str
    time_window: Mapping[str, str]
    result_id: str
    catalog_source_id: str
    catalog_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "metric_id": self.metric_id,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "display_value": self.display_value,
            "time_window": dict(self.time_window),
            "result_id": self.result_id,
            "catalog_source_id": self.catalog_source_id,
            "catalog_version": self.catalog_version,
        }


@dataclass(frozen=True)
class FactsEnvelope:
    schema_version: str
    facts: tuple[Fact, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "facts": [fact.as_dict() for fact in self.facts],
        }


def is_scalar_metric_result(evidence: ResultEvidence) -> bool:
    """The one rule for "this result is a tenant-wide metric value".

    Exactly one row, server-bound metric bindings, and no binding from a
    grouped query.  A grouped result is a rowset of per-group values even
    when it has one row (ORDER BY ... LIMIT 1), so it never becomes a fact.
    Every place that turns results into facts asks this, and FactResolver
    refuses a grouped binding outright.
    """

    return (
        isinstance(evidence, ResultEvidence)
        and bool(evidence.metric_bindings)
        and evidence.row_count == 1
        and len(evidence.rows) == 1
        and not any(binding.grouped for binding in evidence.metric_bindings)
    )


def _metric_name(metric_id: str) -> str:
    normalized = metric_id.removeprefix("metric.")
    if normalized not in _METRIC_LABELS:
        raise FactResolutionError("evidence_validation_failed", "metric is not in the trusted catalog")
    return normalized


def _fact_id(evidence: ResultEvidence, metric_id: str) -> str:
    source = "|".join(
        (
            evidence.run_id,
            evidence.result_id,
            metric_id,
            evidence.query_sha256,
            evidence.catalog_version,
        )
    )
    return f"fact-{hashlib.sha256(source.encode('utf-8')).hexdigest()[:24]}"


def _display_value(metric_id: str, value: int) -> str:
    if metric_id == "paid_count":
        return f"{value}笔"
    return f"{Decimal(value) / Decimal(100):.2f}元"


class FactResolver:
    """Resolve model-provided references against server-created ResultEvidence."""

    def __init__(self, catalog: SemanticCatalog | None = None) -> None:
        self._catalog = catalog or load_default_catalog()

    def resolve(
        self,
        fact_refs: Sequence[FactRef],
        *,
        context: ExecutionContext,
        evidences: Mapping[str, ResultEvidence],
    ) -> FactsEnvelope:
        if not isinstance(context, ExecutionContext):
            raise FactResolutionError("unauthorized", "facts require a server-created execution context")
        if not isinstance(fact_refs, Sequence):
            raise FactResolutionError("invalid_fact_refs", "fact_refs must be a sequence")
        if len(fact_refs) > 10:
            raise FactResolutionError("invalid_fact_refs", "fact_refs has too many items")

        seen: set[tuple[str, str]] = set()
        facts: list[Fact] = []
        for ref in fact_refs:
            if not isinstance(ref, FactRef):
                raise FactResolutionError("invalid_fact_refs", "fact_refs must be server-parsed references")
            metric_id = _metric_name(ref.metric_id)
            key = (ref.result_id, metric_id)
            if key in seen:
                raise FactResolutionError("invalid_fact_refs", "fact_refs must be deduplicated")
            seen.add(key)

            evidence = self._owned_evidence(ref, context, evidences)
            binding = self._trusted_binding(evidence, metric_id)
            value = _bound_value(evidence, binding, metric_id)
            facts.append(
                Fact(
                    fact_id=_fact_id(evidence, metric_id),
                    metric_id=metric_id,
                    label=_METRIC_LABELS[metric_id],
                    value=value,
                    unit=binding.unit,
                    display_value=_display_value(metric_id, value),
                    time_window=dict(binding.time_window),
                    result_id=evidence.result_id,
                    catalog_source_id=binding.catalog_source_id,
                    catalog_version=binding.catalog_version,
                )
            )
        return FactsEnvelope(schema_version=FACTS_SCHEMA_VERSION, facts=tuple(facts))

    def _owned_evidence(
        self,
        ref: FactRef,
        context: ExecutionContext,
        evidences: Mapping[str, ResultEvidence],
    ) -> ResultEvidence:
        """The referenced evidence, when it belongs to this subject and is one aggregate row."""

        evidence = evidences.get(ref.result_id)
        if not isinstance(evidence, ResultEvidence):
            raise FactResolutionError("evidence_validation_failed", "result evidence was not found")
        if evidence.result_id != ref.result_id:
            raise FactResolutionError("evidence_validation_failed", "result identity does not match")
        if (
            evidence.run_id != context.run_id
            or evidence.tenant_id != context.tenant_id
            or evidence.principal_id != context.principal_id
        ):
            raise FactResolutionError("evidence_validation_failed", "result evidence is outside the current subject")
        if evidence.catalog_version != self._catalog.catalog_version:
            raise FactResolutionError("evidence_validation_failed", "result uses an incompatible catalog version")
        if evidence.row_count != 1 or len(evidence.rows) != 1:
            raise FactResolutionError("evidence_validation_failed", "a metric fact needs one aggregate result row")
        return evidence

    def _trusted_binding(self, evidence: ResultEvidence, metric_id: str) -> MetricBinding:
        """The server binding for the metric, when it is ungrouped and matches the catalog."""

        entry = self._catalog.metric(metric_id)
        binding = next(
            (item for item in evidence.metric_bindings if item.metric_id.removeprefix("metric.") == metric_id),
            None,
        )
        if binding is None:
            raise FactResolutionError("evidence_validation_failed", "result has no trusted binding for the metric")
        if binding.grouped:
            raise FactResolutionError("evidence_validation_failed", "a grouped result is a rowset, not a metric fact")
        if (
            binding.catalog_source_id != entry.source_id
            or binding.catalog_version != evidence.catalog_version
            or binding.unit != entry.payload.get("unit")
            or not isinstance(binding.result_position, str)
        ):
            raise FactResolutionError("evidence_validation_failed", "metric binding does not match the catalog")
        if metric_id == "net_fen" and (
            binding.plan_id != NET_FEN_PLAN_ID
            or evidence.metric_plan_id != NET_FEN_PLAN_ID
            or binding.time_window.get("timezone") != "UTC"
        ):
            raise FactResolutionError(
                "evidence_validation_failed",
                "net_fen binding does not match the controlled metric plan",
            )
        return binding


def _bound_value(evidence: ResultEvidence, binding: MetricBinding, metric_id: str) -> int:
    """The value in the bound column: an integer count or fen, never negative where it is raw."""

    row = evidence.rows[0]
    if binding.result_position not in row:
        raise FactResolutionError("evidence_validation_failed", "bound result column is absent")
    value = row[binding.result_position]
    if type(value) is not int:
        raise FactResolutionError("evidence_validation_failed", "fact values must be integer count or fen")
    if metric_id == "paid_count" and value < 0:
        raise FactResolutionError("evidence_validation_failed", "count facts cannot be negative")
    if metric_id in {"gross_fen", "refund_fen"} and value < 0:
        raise FactResolutionError("evidence_validation_failed", "raw money facts cannot be negative")
    return value


def resolve_fact_refs(
    fact_refs: Sequence[FactRef],
    *,
    context: ExecutionContext,
    evidences: Mapping[str, ResultEvidence],
    catalog: SemanticCatalog | None = None,
) -> FactsEnvelope:
    return FactResolver(catalog).resolve(fact_refs, context=context, evidences=evidences)


__all__ = [
    "FACTS_SCHEMA_VERSION",
    "Fact",
    "FactResolutionError",
    "FactsEnvelope",
    "FactResolver",
    "is_scalar_metric_result",
    "resolve_fact_refs",
]
