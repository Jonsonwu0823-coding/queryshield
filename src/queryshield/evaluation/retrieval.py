"""Versioned development retrieval queries.

The cases are development material, not a frozen holdout set.  The current
T02 catalog tool is queried with a fixed top_k of three; later knowledge and
hybrid retrieval stages can reuse the same case schema without changing the
business intent families.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from queryshield.agent.proposals import ExecutionContext
from queryshield.tools.semantic import ControlledTools


RETRIEVAL_CASE_VERSION = "retrieval-case-v1"
_REQUIRED_FIELDS = frozenset(
    {"query", "relevant_source_ids", "family_id", "split", "catalog_version"}
)


class RetrievalCaseError(ValueError):
    """A development retrieval fixture is malformed."""


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_source_ids: tuple[str, ...]
    family_id: str
    split: str
    catalog_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "relevant_source_ids": list(self.relevant_source_ids),
            "family_id": self.family_id,
            "split": self.split,
            "catalog_version": self.catalog_version,
        }


def _case_from_mapping(raw: Mapping[str, object], *, index: int) -> RetrievalCase:
    if set(raw) != _REQUIRED_FIELDS:
        raise RetrievalCaseError(f"cases[{index}] must contain exactly the retrieval-case-v1 fields")
    query = raw.get("query")
    family_id = raw.get("family_id")
    split = raw.get("split")
    catalog_version = raw.get("catalog_version")
    if any(type(value) is not str or not value.strip() for value in (query, family_id, split, catalog_version)):
        raise RetrievalCaseError(f"cases[{index}] has an invalid string field")
    sources = raw.get("relevant_source_ids")
    if type(sources) is not list or any(type(value) is not str or not value.strip() for value in sources):
        raise RetrievalCaseError(f"cases[{index}].relevant_source_ids must be a list of strings")
    if split != "development" or catalog_version != "catalog-v1":
        raise RetrievalCaseError(f"cases[{index}] must be a catalog-v1 development case")
    return RetrievalCase(
        query=query,
        relevant_source_ids=tuple(sources),
        family_id=family_id,
        split=split,
        catalog_version=catalog_version,
    )


def load_development_cases(path: str | Path | None = None) -> tuple[RetrievalCase, ...]:
    fixture_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[3] / "fixtures" / "semantic" / "retrieval-case-v1.json"
    )
    try:
        document = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RetrievalCaseError(f"cannot load retrieval fixture: {fixture_path}") from exc
    if not isinstance(document, Mapping) or document.get("schema_version") != RETRIEVAL_CASE_VERSION:
        raise RetrievalCaseError("retrieval fixture has an unsupported schema_version")
    raw_cases = document.get("cases")
    if type(raw_cases) is not list or len(raw_cases) < 12:
        raise RetrievalCaseError("at least twelve development retrieval cases are required")
    cases = tuple(_case_from_mapping(raw, index=index) for index, raw in enumerate(raw_cases))
    family_ids = [case.family_id for case in cases]
    if len(set(family_ids)) != len(family_ids):
        raise RetrievalCaseError("development retrieval families must be unique in this fixture")
    return cases


def run_development_catalog_retrieval(
    tools: ControlledTools,
    *,
    context: ExecutionContext,
    cases: Sequence[RetrievalCase] | None = None,
) -> tuple[dict[str, object], ...]:
    """Run the T02 lexical/synonym baseline with fixed ``top_k=3``."""

    if not isinstance(tools, ControlledTools):
        raise TypeError("tools must be a ControlledTools instance")
    if not isinstance(context, ExecutionContext):
        raise TypeError("context must be an ExecutionContext")
    selected_cases = tuple(cases) if cases is not None else load_development_cases()
    records: list[dict[str, object]] = []
    for case in selected_cases:
        result = tools.search_catalog(
            {"query": case.query, "top_k": 3},
            context=context,
        )
        items = result["items"]
        assert isinstance(items, list)
        records.append(
            {
                "family_id": case.family_id,
                "query": case.query,
                "top_k": 3,
                "items": items,
                "empty_hit": not items,
                "returned_source_ids": sorted(
                    {item["source_id"] for item in items if isinstance(item, Mapping)}
                ),
            }
        )
    return tuple(records)


__all__ = [
    "RETRIEVAL_CASE_VERSION",
    "RetrievalCase",
    "RetrievalCaseError",
    "load_development_cases",
    "run_development_catalog_retrieval",
]
