"""``build_context`` refuses every malformed input with its own code, in a fixed order.

The context is what the model sees, so each input is checked before anything is
built; the first failing check is the one reported.
"""

from __future__ import annotations

import pytest

from queryshield.agent.context import ContextBudgetError, ContextBuildError, build_context
from test_clarification import _context


CONTEXT = _context("run-context-inputs")
ITEM = {"id": "i", "text": "t", "source_id": "s", "version": "v"}
BINDING = {"metric_id": "gross_fen", "result_position": "gross_fen", "unit": "fen", "time_window": None}
LONG = "字" * 8000
INVALID = "invalid_context_input"
WINDOW = "time_window must contain start, end and timezone"
TOO_LONG = "message_too_long"


def _bindings(**changes):
    return [{**BINDING, **changes}]


# (id, positional arguments, keyword arguments, error code, message)
REFUSALS = [
    ("context-not-from-the-server", ("x", "q"), {}, "unauthorized", "context must be server-created"),
    ("run-config", (CONTEXT, "q"), {"run_config": "x"}, INVALID, "run_config must be server-created"),
    ("question-blank", (CONTEXT, " "), {}, INVALID, "question must be a non-empty string"),
    ("question-not-text", (CONTEXT, 5), {}, INVALID, "question must be a non-empty string"),
    ("question-too-long", (CONTEXT, "字" * 8001), {}, TOO_LONG, "question exceeds 8000 characters"),
    ("clarification-blank", (CONTEXT, "q"), {"clarifications": [" "]}, INVALID, "clarifications[0] must be a non-empty string"),
    ("clarification-too-long", (CONTEXT, "q"), {"clarifications": ["字" * 8001]}, TOO_LONG, "clarifications[0] exceeds 8000 characters"),
    ("window-missing-a-part", (CONTEXT, "q"), {"time_window": {"start": "a", "end": "b"}}, INVALID, WINDOW),
    ("window-without-start", (CONTEXT, "q"), {"time_window": {"end": "b", "timezone": "UTC"}}, INVALID, WINDOW),
    ("window-not-a-mapping", (CONTEXT, "q"), {"time_window": "x"}, INVALID, WINDOW),
    ("catalog", (CONTEXT, "q"), {"metric_catalog": "x"}, INVALID, "metric_catalog must be the server catalog"),
    ("request-window-part-blank", (CONTEXT, "q"), {"request_time_window": {"start": "a", "end": "", "timezone": "UTC"}}, INVALID, "time_window.end must be a non-empty string"),
    ("bindings-as-text", (CONTEXT, "q"), {"metric_bindings": "x"}, INVALID, "metric_bindings must be a sequence"),
    ("binding-not-a-mapping", (CONTEXT, "q"), {"metric_bindings": ["x"]}, INVALID, "metric_bindings[0] must be a server binding"),
    ("binding-metric-blank", (CONTEXT, "q"), {"metric_bindings": _bindings(metric_id="")}, INVALID, "metric_bindings[0].metric_id must be a non-empty string"),
    ("binding-position-not-text", (CONTEXT, "q"), {"metric_bindings": _bindings(result_position=3)}, INVALID, "metric_bindings[0].result_position must be a non-empty string"),
    ("binding-unit-missing", (CONTEXT, "q"), {"metric_bindings": [{k: v for k, v in BINDING.items() if k != "unit"}]}, INVALID, "metric_bindings[0].unit must be a non-empty string"),
    ("binding-window", (CONTEXT, "q"), {"metric_bindings": _bindings(time_window={"start": "a"})}, INVALID, WINDOW),
    ("binding-repeated", (CONTEXT, "q"), {"metric_bindings": [BINDING, BINDING]}, INVALID, "metric_bindings must not repeat metric IDs"),
    ("four-retrieval-items", (CONTEXT, "q"), {"retrieval_items": [ITEM] * 4}, INVALID, "at most three retrieval items are allowed"),
    ("item-missing-a-field", (CONTEXT, "q"), {"retrieval_items": [{"id": "i"}]}, INVALID, "retrieval_items[0] must have exactly id/text/source_id/version"),
    ("item-with-an-extra-field", (CONTEXT, "q"), {"retrieval_items": [{**ITEM, "extra": "x"}]}, INVALID, "retrieval_items[0] must have exactly id/text/source_id/version"),
    ("item-text-blank", (CONTEXT, "q"), {"retrieval_items": [{**ITEM, "text": ""}]}, INVALID, "retrieval_items[0].text must be a non-empty string"),
    ("item-text-too-long", (CONTEXT, "q"), {"retrieval_items": [{**ITEM, "text": "字" * 7501}]}, TOO_LONG, "retrieval_items[0].text exceeds 7500 characters"),
    ("summary-blank", (CONTEXT, "q"), {"optional_summaries": [" "]}, INVALID, "optional_summaries[0] must be a non-empty string"),
    ("tool-result-not-a-mapping", (CONTEXT, "q"), {"tool_results": ["x"]}, INVALID, "tool_results[0] must be an object"),
    ("tool-result-not-json", (CONTEXT, "q"), {"tool_results": [{"x": object()}]}, INVALID, "untrusted_tool_result is not JSON serializable"),
    ("confirmed-metric-blank", (CONTEXT, "q"), {"confirmed_metric": " "}, INVALID, "confirmed_metric must be a non-empty string"),
    ("hard-messages-alone-exceed-the-budget", (CONTEXT, LONG), {"clarifications": [LONG] * 3}, "context_budget_exceeded", "hard context exceeds the configured budget"),
    # The first failing check is the one reported.
    ("order-run-config-before-question", (CONTEXT, " "), {"run_config": "x"}, INVALID, "run_config must be server-created"),
    ("order-question-before-clarifications", (CONTEXT, " "), {"clarifications": [" "]}, INVALID, "question must be a non-empty string"),
    ("order-clarifications-before-window", (CONTEXT, "q"), {"clarifications": [" "], "time_window": "x"}, INVALID, "clarifications[0] must be a non-empty string"),
    ("order-window-before-catalog", (CONTEXT, "q"), {"time_window": "x", "metric_catalog": "x"}, INVALID, WINDOW),
    ("order-catalog-before-request-window", (CONTEXT, "q"), {"metric_catalog": "x", "request_time_window": "x"}, INVALID, "metric_catalog must be the server catalog"),
    ("order-request-window-before-bindings", (CONTEXT, "q"), {"request_time_window": "x", "metric_bindings": "x"}, INVALID, WINDOW),
    ("order-bindings-before-item-count", (CONTEXT, "q"), {"metric_bindings": "x", "retrieval_items": [ITEM] * 4}, INVALID, "metric_bindings must be a sequence"),
    ("order-item-count-before-confirmed-metric", (CONTEXT, "q"), {"retrieval_items": [ITEM] * 4, "confirmed_metric": " "}, INVALID, "at most three retrieval items are allowed"),
    ("order-confirmed-metric-before-item-fields", (CONTEXT, "q"), {"confirmed_metric": " ", "retrieval_items": [{"id": "i"}]}, INVALID, "confirmed_metric must be a non-empty string"),
    ("order-item-fields-before-summaries", (CONTEXT, "q"), {"retrieval_items": [{"id": "i"}], "optional_summaries": [" "]}, INVALID, "retrieval_items[0] must have exactly id/text/source_id/version"),
    ("order-summaries-before-tool-results", (CONTEXT, "q"), {"optional_summaries": [" "], "tool_results": ["x"]}, INVALID, "optional_summaries[0] must be a non-empty string"),
    ("order-tool-results-before-the-budget", (CONTEXT, LONG), {"clarifications": [LONG] * 3, "tool_results": ["x"]}, INVALID, "tool_results[0] must be an object"),
]


@pytest.mark.parametrize(("name", "args", "kwargs", "code", "message"), REFUSALS, ids=[row[0] for row in REFUSALS])
def test_build_context_refuses_the_first_bad_input(name, args, kwargs, code, message) -> None:
    with pytest.raises(ContextBuildError) as caught:
        build_context(*args, **kwargs)
    assert (caught.value.code, str(caught.value)) == (code, f"{code}: {message}")
    assert isinstance(caught.value, ContextBudgetError) == (code == "context_budget_exceeded")


def test_the_budget_drops_only_optional_messages_oldest_first() -> None:
    results = [{"tool_name": "search_catalog", "status": "succeeded", "output": {"text": "字" * 3000 + str(index)}} for index in range(12)]
    built = build_context(CONTEXT, "2026年9月支付金额", tool_results=results)
    assert built.dropped_optional_ids == tuple(f"tool-result-{index}" for index in range(len(built.dropped_optional_ids)))
    assert built.dropped_optional_ids and built.serialized_bytes <= 24_000
    assert set(built.hard_message_ids) == {"server-context", "user-question", "retrieval-empty"}
    assert built.included_optional_ids == tuple(f"tool-result-{index}" for index in range(len(built.dropped_optional_ids), 12))
