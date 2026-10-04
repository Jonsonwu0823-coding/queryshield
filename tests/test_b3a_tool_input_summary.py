"""B3a / N1: tool-call events never keep model-written text.

Only catalog-declarable metric ids and allow-listed table names are recorded verbatim;
any other value is reduced to a count and a hash of its canonical JSON, and those
fields exist only when such a value is present.
"""

from __future__ import annotations

from hashlib import sha256
import json

import pytest

from queryshield.agent.graph import _tool_input_summary
from queryshield.agent.proposals import ToolCallAction
from test_metric_intent import (  # noqa: F401  (pytest prepend import mode puts tests/ on sys.path)
    SEPTEMBER,
    _agent,
    _context,
    _DynamicModel,
    _query_step,
    _tools,
)

DECLARABLE = ("paid_count", "gross_fen", "net_fen")
LONG = "X" * 5000


def _query(**arguments: object) -> ToolCallAction:
    return ToolCallAction(name="query_readonly", arguments={"sql": "SELECT 1", "params": {}, **arguments})


def _hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def test_long_undeclarable_metric_is_only_counted_and_hashed() -> None:
    summary = _tool_input_summary(_query(metrics=[LONG]), declarable_metrics=DECLARABLE)

    assert summary["declared_metrics"] == []
    assert summary["declared_metrics_other_count"] == 1
    assert summary["declared_metrics_other_sha256"] == _hash([LONG])
    assert "XXXXXXXX" not in json.dumps(summary)


def test_declarable_id_is_kept_and_the_unknown_one_is_hashed() -> None:
    summary = _tool_input_summary(_query(metrics=["gross_fen", "evil"]), declarable_metrics=DECLARABLE)

    assert summary["declared_metrics"] == ["gross_fen"]
    assert summary["declared_metrics_other_count"] == 1
    assert summary["declared_metrics_other_sha256"] == _hash(["evil"])
    assert "evil" not in json.dumps(summary)


def test_catalog_metric_the_server_cannot_verify_is_not_kept_either() -> None:
    summary = _tool_input_summary(_query(metrics=["refund_fen"]), declarable_metrics=DECLARABLE)
    assert summary["declared_metrics"] == []
    assert "refund_fen" not in json.dumps(summary)


def test_non_string_values_hash_stably_whatever_their_order() -> None:
    first = _tool_input_summary(_query(metrics=[5, {"b": 1, "a": [2]}, None, "gross_fen"]), declarable_metrics=DECLARABLE)
    second = _tool_input_summary(_query(metrics=[None, "gross_fen", {"a": [2], "b": 1}, 5]), declarable_metrics=DECLARABLE)

    assert first["declared_metrics"] == ["gross_fen"]
    assert first["declared_metrics_other_count"] == 3
    assert first == second
    assert first["declared_metrics_other_sha256"] == _hash(sorted([5, {"b": 1, "a": [2]}, None], key=lambda v: json.dumps(v, sort_keys=True, separators=(",", ":"))))


def test_no_catalog_means_nothing_is_declarable() -> None:
    summary = _tool_input_summary(_query(metrics=["gross_fen"]))
    assert summary["declared_metrics"] == []
    assert summary["declared_metrics_other_count"] == 1


def test_at_most_four_known_metrics_are_kept() -> None:
    summary = _tool_input_summary(_query(metrics=["gross_fen"] * 6), declarable_metrics=DECLARABLE)
    assert summary["declared_metrics"] == ["gross_fen"] * 4
    assert "declared_metrics_other_count" not in summary


def test_valid_input_produces_the_same_fields_as_before() -> None:
    query = _tool_input_summary(_query(metrics=["paid_count", "gross_fen"]), declarable_metrics=DECLARABLE)
    assert query == {
        "argument_keys": ["metrics", "params", "sql"],
        "sql_length": len("SELECT 1"),
        "sql_sha256": sha256(b"SELECT 1").hexdigest(),
        "params_count": 0,
        "declared_metrics": ["paid_count", "gross_fen"],
    }
    tables = _tool_input_summary(ToolCallAction(name="describe_tables", arguments={"tables": ["refunds", "orders"]}))
    assert tables == {"argument_keys": ["tables"], "table_count": 2, "table_names": ["orders", "refunds"]}
    search = _tool_input_summary(ToolCallAction(name="search_catalog", arguments={"query": "净额"}))
    assert search["top_k"] == 3 and "query_length" in search


@pytest.mark.parametrize("names", [["orders", LONG], [LONG, "orders", "evil"], [7, "orders"]])
def test_describe_tables_keeps_only_allowed_names(names: list[object]) -> None:
    summary = _tool_input_summary(ToolCallAction(name="describe_tables", arguments={"tables": names}))

    assert summary["table_names"] == ["orders"]
    assert summary["table_count"] == len(names)
    assert summary["table_names_other_count"] == len(names) - 1
    encoded = json.dumps(summary)
    assert "evil" not in encoded and "XXXXXXXX" not in encoded


def test_describe_tables_with_mixed_types_does_not_raise() -> None:
    summary = _tool_input_summary(ToolCallAction(name="describe_tables", arguments={"tables": [3, "orders", None]}))
    assert summary["table_names"] == ["orders"]


def test_agent_run_never_records_the_declared_text_in_its_events() -> None:
    tools, _ = _tools()
    model = _DynamicModel(
        [_query_step(metrics=["gross_fen", LONG, "evil-metric"], time_window=SEPTEMBER), lambda _messages: {"type": "ask_user", "question": "q?"}]
    )
    result = _agent(model, tools).run(_context("run-n1"), "2026年9月支付总额是多少？")

    tool_event = next(event for event in result.events if event.get("kind") == "tool_call")
    assert tool_event["input_summary"]["declared_metrics"] == ["gross_fen"]
    assert tool_event["input_summary"]["declared_metrics_other_count"] == 2
    persisted = json.dumps(result.events, ensure_ascii=False)
    assert LONG not in persisted and "evil-metric" not in persisted
