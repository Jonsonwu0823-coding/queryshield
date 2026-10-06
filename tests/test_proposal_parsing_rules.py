"""How the proposal parser treats model output: every case pinned to its result.

The inputs are below; the expected outcome of each (the action it parses to,
or the error class, code, message and diagnostic detail) is in
``proposal_parsing_pins.json``, recorded before the parser was restructured.
Native cases go through native_action_text first, as the graph does.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import get_args

import pytest

from queryshield.agent import proposals
from queryshield.agent.call_store import DurableModelCallStore
from queryshield.agent.proposals import (
    CallIdentityError,
    ExecutionContext,
    ModelCallStore,
    ParallelReadonlyAction,
    ProposalParseError,
    ToolCallAction,
    native_action_text,
    parse_error_detail,
    parse_query_proposal,
    proposal_shape_summary,
)
from queryshield.providers.contracts import NativeToolCall


PINS = json.loads(Path(__file__).with_name("proposal_parsing_pins.json").read_text(encoding="utf-8"))
CONTEXT = ExecutionContext(run_id="run", tenant_id="A", principal_id="p", role="requester")
Q = {"sql": "SELECT COUNT(*) AS paid_count FROM orders AS o WHERE o.status = %s", "params": {"0": "paid"}}


def j(payload):
    return json.dumps(payload, ensure_ascii=False)


def tc(name, arguments, **extra):
    return j({"type": "tool_call", "name": name, "arguments": arguments, **extra})


FA = {"type": "final_answer", "answer": "a", "source_ids": [], "fact_refs": []}

# (label, model content) parsed in json mode
JSON_CASES = [
    ("empty", ""),
    ("blank", "  \n"),
    ("not_object", "[1]"),
    ("number", "1"),
    ("trailing", '{"type":"deny","reason":"x"} x'),
    ("duplicate_key", '{"type":"deny","type":"deny","reason":"x"}'),
    ("nested_duplicate_key", '{"type":"tool_call","name":"search_catalog","arguments":{"query":"a","query":"b"}}'),
    ("nan", '{"type":"deny","reason":NaN}'),
    ("infinity", '{"type":"deny","reason":-Infinity}'),
    ("no_type", "{}"),
    ("type_not_string", j({"type": 1})),
    ("type_blank", j({"type": " "})),
    ("top_unknown_field", j({"type": "deny", "reason": "x", "zeta": 1, "alpha": 2})),
    ("unknown_action", j({"type": "explode"})),
    ("tool_as_type_search", j({"type": "search_catalog", "query": "x"})),
    ("tool_as_type_query", j({"type": "query_readonly", "name": "query_readonly"})),
    ("unknown_field_before_tool_as_type", j({"type": "query_readonly", "sql": "x"})),
    ("tool_call_missing_arguments", j({"type": "tool_call", "name": "search_catalog"})),
    ("tool_call_extra_top_field", tc("search_catalog", {"query": "x"}, question="q")),
    ("tool_call_name_not_string", tc(3, {})),
    ("tool_call_name_unknown", tc("drop_table", {})),
    ("tool_call_name_is_an_action", tc("deny", {"reason": "x"})),
    ("tool_call_arguments_list", tc("search_catalog", [])),
    ("tool_call_arguments_string", tc("search_catalog", "{}")),
    ("tool_call_arguments_nonempty_list", tc("search_catalog", ["x"])),
    ("tool_call_with_reason", tc("search_catalog", {"query": "x"}, reason="r")),
    ("search_default_top_k", tc("search_catalog", {"query": "净额"})),
    ("search_top_k", tc("search_catalog", {"query": "q", "top_k": 5})),
    ("search_top_k_zero", tc("search_catalog", {"query": "q", "top_k": 0})),
    ("search_top_k_six", tc("search_catalog", {"query": "q", "top_k": 6})),
    ("search_top_k_bool", tc("search_catalog", {"query": "q", "top_k": True})),
    ("search_top_k_float", tc("search_catalog", {"query": "q", "top_k": 2.0})),
    ("search_missing_query", tc("search_catalog", {"top_k": 2})),
    ("search_extra", tc("search_catalog", {"query": "q", "tables": []})),
    ("search_extra_k", tc("search_catalog", {"query": "q", "k": 1})),
    ("search_query_blank", tc("search_catalog", {"query": "  "})),
    ("search_query_long", tc("search_catalog", {"query": "x" * 201})),
    ("search_query_max", tc("search_catalog", {"query": "x" * 200})),
    ("describe_ok", tc("describe_tables", {"tables": ["orders", "customers"]})),
    ("describe_not_list", tc("describe_tables", {"tables": "orders"})),
    ("describe_empty", tc("describe_tables", {"tables": []})),
    ("describe_four", tc("describe_tables", {"tables": ["orders", "customers", "refunds", "orders"]})),
    ("describe_duplicate", tc("describe_tables", {"tables": ["orders", "orders"]})),
    ("describe_unknown_table", tc("describe_tables", {"tables": ["users"]})),
    ("describe_blank_item", tc("describe_tables", {"tables": [" "]})),
    ("describe_item_not_string", tc("describe_tables", {"tables": [1]})),
    ("describe_extra", tc("describe_tables", {"tables": ["orders"], "x": 1})),
    ("query_ok", tc("query_readonly", Q)),
    ("query_full", tc("query_readonly", {**Q, "metrics": ["paid_count"], "time_window": {"start": "a", "end": "b"}})),
    ("query_missing_params", tc("query_readonly", {"sql": "SELECT 1"})),
    ("query_missing_sql", tc("query_readonly", {"params": {}})),
    ("query_extra", tc("query_readonly", {**Q, "limit": 1})),
    ("query_sql_blank", tc("query_readonly", {"sql": " ", "params": {}})),
    ("query_sql_long", tc("query_readonly", {"sql": "S" * 4001, "params": {}})),
    ("query_sql_not_string", tc("query_readonly", {"sql": 1, "params": {}})),
    ("query_params_null", tc("query_readonly", {"sql": "SELECT 1", "params": None})),
    ("query_params_empty_list", tc("query_readonly", {"sql": "SELECT 1", "params": []})),
    ("query_params_list", tc("query_readonly", {"sql": "SELECT 1", "params": ["paid"]})),
    ("query_params_string", tc("query_readonly", {"sql": "SELECT 1", "params": "paid"})),
    ("query_params_reserved_tenant", tc("query_readonly", {"sql": "SELECT 1", "params": {"tenant_id": "B"}})),
    ("query_params_reserved_token", tc("query_readonly", {"sql": "SELECT 1", "params": {"0": "x", "token": "t"}})),
    ("query_params_reserved_after_bad_value", tc("query_readonly", {"sql": "SELECT 1", "params": {"0": [1], "role": "x"}})),
    ("query_params_value_list", tc("query_readonly", {"sql": "SELECT 1", "params": {"0": [1]}})),
    ("query_params_reserved_with_bad_value", tc("query_readonly", {"sql": "SELECT 1", "params": {"tenant_id": [1]}})),
    ("query_params_value_object", tc("query_readonly", {"sql": "SELECT 1", "params": {"0": {}}})),
    ("query_params_scalars", tc("query_readonly", {"sql": "SELECT 1", "params": {"0": None, "1": 1, "2": 1.5, "3": True, "4": "s"}})),
    ("query_metrics_null", tc("query_readonly", {**Q, "metrics": None})),
    ("query_metrics_empty", tc("query_readonly", {**Q, "metrics": []})),
    ("query_metrics_string", tc("query_readonly", {**Q, "metrics": "paid_count"})),
    ("query_metrics_any_items", tc("query_readonly", {**Q, "metrics": [1, "x"]})),
    ("query_window_null", tc("query_readonly", {**Q, "time_window": None})),
    ("query_window_string", tc("query_readonly", {**Q, "time_window": "2026-09"})),
    ("query_window_any_object", tc("query_readonly", {**Q, "time_window": {"x": 1}})),
    ("parallel_ok", j({"type": "parallel_readonly", "metric_ids": ["paid_count", "net_fen", "gross_fen"]})),
    ("parallel_one", j({"type": "parallel_readonly", "metric_ids": ["paid_count"]})),
    ("parallel_four", j({"type": "parallel_readonly", "metric_ids": ["paid_count", "net_fen", "gross_fen", "x"]})),
    ("parallel_duplicate", j({"type": "parallel_readonly", "metric_ids": ["paid_count", "paid_count"]})),
    ("parallel_unknown_metric", j({"type": "parallel_readonly", "metric_ids": ["paid_count", "refund_fen"]})),
    ("parallel_blank", j({"type": "parallel_readonly", "metric_ids": ["paid_count", " "]})),
    ("parallel_not_list", j({"type": "parallel_readonly", "metric_ids": "paid_count,net_fen"})),
    ("parallel_extra", j({"type": "parallel_readonly", "metric_ids": ["paid_count", "net_fen"], "name": "x"})),
    ("parallel_missing", j({"type": "parallel_readonly"})),
    ("ask_ok", j({"type": "ask_user", "question": "哪个月？"})),
    ("ask_clarification", j({"type": "ask_user", "question": "q", "clarification_id": "clarify.metric_basis"})),
    ("ask_clarification_null", j({"type": "ask_user", "question": "q", "clarification_id": None})),
    ("ask_clarification_blank", j({"type": "ask_user", "question": "q", "clarification_id": " "})),
    ("ask_clarification_long", j({"type": "ask_user", "question": "q", "clarification_id": "c" * 101})),
    ("ask_missing_question", j({"type": "ask_user"})),
    ("ask_question_long", j({"type": "ask_user", "question": "q" * 1001})),
    ("ask_question_not_string", j({"type": "ask_user", "question": ["q"]})),
    ("ask_extra", j({"type": "ask_user", "question": "q", "answer": "a"})),
    ("final_ok", j(FA)),
    ("final_refs", j({**FA, "source_ids": ["s1", "s2"], "fact_refs": [{"result_id": "r", "metric_id": "m"}, {"result_id": "r", "metric_id": "n"}]})),
    ("final_basis_null", j({**FA, "basis": None})),
    ("final_basis_knowledge", j({**FA, "basis": "knowledge"})),
    ("final_basis_unknown", j({**FA, "basis": "guess"})),
    ("final_basis_not_string", j({**FA, "basis": 1})),
    ("final_no_data_blank_answer", j({**FA, "answer": " ", "basis": "no_data"})),
    ("final_blank_answer", j({**FA, "answer": " "})),
    ("final_long_answer", j({**FA, "answer": "a" * 4001})),
    ("final_missing_fact_refs", j({"type": "final_answer", "answer": "a", "source_ids": []})),
    ("final_missing_two", j({"type": "final_answer", "answer": "a"})),
    ("final_extra", j({**FA, "reason": "x"})),
    ("final_source_ids_not_list", j({**FA, "source_ids": "s"})),
    ("final_source_ids_duplicate", j({**FA, "source_ids": ["s", "s"]})),
    ("final_source_ids_too_many", j({**FA, "source_ids": [f"s{i}" for i in range(33)]})),
    ("final_fact_refs_not_list", j({**FA, "fact_refs": {}})),
    ("final_fact_refs_too_many", j({**FA, "fact_refs": [{"result_id": "r", "metric_id": f"m{i}"} for i in range(11)]})),
    ("final_fact_ref_not_object", j({**FA, "fact_refs": ["r"]})),
    ("final_fact_ref_missing", j({**FA, "fact_refs": [{"result_id": "r"}]})),
    ("final_fact_ref_extra", j({**FA, "fact_refs": [{"result_id": "r", "metric_id": "m", "value": 1}]})),
    ("final_fact_ref_extra_v", j({**FA, "fact_refs": [{"result_id": "r", "metric_id": "m", "v": 1}]})),
    ("final_fact_ref_blank", j({**FA, "fact_refs": [{"result_id": " ", "metric_id": "m"}]})),
    ("final_fact_ref_duplicate", j({**FA, "fact_refs": [{"result_id": "r", "metric_id": "m"}, {"result_id": "r", "metric_id": "m"}]})),
    ("final_bad_refs_and_bad_basis", j({**FA, "fact_refs": "x", "basis": "guess"})),
    ("deny_ok", j({"type": "deny", "reason": "不能查"})),
    ("deny_missing", j({"type": "deny"})),
    ("deny_blank", j({"type": "deny", "reason": ""})),
    ("deny_long", j({"type": "deny", "reason": "r" * 1001})),
    ("deny_extra", j({"type": "deny", "reason": "r", "question": "q"})),
    ("lone_surrogate", '{"type":"deny","reason":"\\ud800"}'),
]

# (label, function name, arguments text) converted by native_action_text
NATIVE_CASES = [
    ("native_search", "search_catalog", '{"query":"净额","top_k":2}'),
    ("native_query", "query_readonly", j(Q)),
    ("native_deny", "deny", '{"reason":"x"}'),
    ("native_ask", "ask_user", '{"question":"哪个月？","clarification_id":"c"}'),
    ("native_final", "final_answer", j({"answer": "a", "source_ids": [], "fact_refs": [], "basis": "no_data"})),
    ("native_unknown_function", "parallel_readonly", '{"metric_ids":["paid_count","net_fen"]}'),
    ("native_type_in_arguments", "deny", '{"type":"deny","reason":"x"}'),
    ("native_arguments_not_json", "deny", '{"reason":'),
    ("native_arguments_not_object", "search_catalog", "[]"),
    ("native_arguments_duplicate", "deny", '{"reason":"a","reason":"b"}'),
    ("native_arguments_empty", "deny", ""),
    ("native_bad_tool_arguments", "describe_tables", '{"tables":["users"]}'),
    ("native_lone_surrogate", "deny", '{"reason":"\\ud800"}'),
]

SHAPE_INPUTS = [
    None,
    "",
    "not json",
    "[1,2]",
    "3",
    j({"type": "tool_call", "name": "query_readonly", "arguments": {**Q, "metrics": [], "zz": 1}}),
    j({"type": "query_readonly", "name": "secret_tool", "arguments": "x", "extra1": 1, "extra2": 2}),
    j({"type": 3, "name": None, "basis": "guess"}),
    j({"type": "final_answer", "basis": "knowledge"}),
    j({"type": "final_answer", "basis": None}),
    j({"type": "deny", "basis": 2}),
    j({"type": "parallel_readonly", "arguments": [1]}),
    j({"type": "ask_user", "arguments": None}),
    '{"type":"deny","reason":NaN}',
]


def _outcome(run) -> list[object]:
    try:
        value = run()
    except ProposalParseError as exc:
        return ["error", type(exc).__name__, exc.code, str(exc), parse_error_detail(exc)]
    return ["ok", value]


def _json_outcome(text: str) -> list[object]:
    return _outcome(lambda: parse_query_proposal(text, context=CONTEXT, model_call_id="m").as_dict())


def _native_outcome(name: str, arguments: str) -> list[object]:
    def run() -> dict[str, object]:
        text = native_action_text((NativeToolCall("c", name, arguments),))
        return {"text": text, "proposal": parse_query_proposal(text, context=CONTEXT, model_call_id="m").as_dict()}

    return _outcome(run)


def test_every_case_has_a_pin() -> None:
    labels = [label for label, _ in JSON_CASES] + [label for label, _, _ in NATIVE_CASES]
    assert len(labels) == len(set(labels))
    assert set(labels) == set(PINS["cases"])


@pytest.mark.parametrize("label, text", JSON_CASES, ids=[label for label, _ in JSON_CASES])
def test_json_proposal_parses_to_its_pinned_result(label: str, text: str) -> None:
    assert _json_outcome(text) == PINS["cases"][label]


@pytest.mark.parametrize("label, name, arguments", NATIVE_CASES, ids=[label for label, _, _ in NATIVE_CASES])
def test_native_call_converts_and_parses_to_its_pinned_result(label: str, name: str, arguments: str) -> None:
    assert _native_outcome(name, arguments) == PINS["cases"][label]


def test_proposal_shape_summaries_are_pinned() -> None:
    assert [proposal_shape_summary(value) for value in SHAPE_INPUTS] == PINS["shapes"]


def test_a_tool_name_used_as_the_type_is_its_own_error_class() -> None:
    with pytest.raises(proposals.ToolNameAsActionTypeError) as caught:
        parse_query_proposal(j({"type": "describe_tables", "arguments": {}}), context=CONTEXT, model_call_id="m")
    assert caught.value.tool_name == "describe_tables"
    with pytest.raises(ValueError, match="tool_name must be a server tool name"):
        proposals.ToolNameAsActionTypeError("deny")


def test_native_conversion_refuses_zero_or_several_calls() -> None:
    with pytest.raises(ProposalParseError) as none:
        native_action_text(())
    with pytest.raises(ProposalParseError) as several:
        native_action_text((NativeToolCall("a", "deny", "{}"), NativeToolCall("b", "deny", "{}")))
    assert (none.value.code, several.value.code) == ("native_tool_call_missing", "native_multiple_tool_calls")


def test_parse_rejects_a_context_that_is_not_server_built() -> None:
    with pytest.raises(TypeError, match="context must be an ExecutionContext"):
        parse_query_proposal("{}", context={"tenant_id": "A"}, model_call_id="m")
    with pytest.raises(ValueError, match="model_call_id must be a non-empty string"):
        parse_query_proposal("{}", context=CONTEXT, model_call_id=" ")
    with pytest.raises(ProposalParseError) as caught:
        parse_query_proposal(b"{}", context=CONTEXT, model_call_id="m")
    assert caught.value.code == "invalid_content"


def test_content_digest_is_of_the_exact_model_text() -> None:
    proposal = parse_query_proposal(' {"type":"deny","reason":"x"} ', context=CONTEXT, model_call_id="m")
    assert proposal.content_sha256 == "051c4e505879bb4142618d10c8be78dc98ba88f8ae738f14d98445877dd976fe"


def test_tool_names_and_tables_are_server_lists() -> None:
    assert proposals.TOOL_NAMES == ("search_catalog", "describe_tables", "query_readonly")
    assert get_args(proposals.ToolName) == proposals.TOOL_NAMES
    assert proposals.NATIVE_FUNCTION_NAMES == (
        "search_catalog", "describe_tables", "query_readonly", "ask_user", "final_answer", "deny"
    )
    assert proposals.ALLOWED_TABLES == frozenset({"customers", "orders", "refunds"})
    assert type(proposals.ALLOWED_TABLES) is frozenset
    assert proposals.PARALLEL_METRICS == frozenset({"paid_count", "gross_fen", "net_fen"})
    assert proposals.RESERVED_IDENTITY_PARAMS == frozenset({"tenant_id", "principal_id", "role", "authorization", "token"})


def test_tool_call_action_accepts_only_server_tool_names() -> None:
    for name in proposals.TOOL_NAMES:
        assert ToolCallAction(name, {}).as_dict() == {"type": "tool_call", "name": name, "arguments": {}}
    with pytest.raises(ValueError, match="unsupported tool name"):
        ToolCallAction("deny", {})


@pytest.mark.parametrize(
    "metric_ids, message",
    [
        (["paid_count", "net_fen"], "parallel_readonly requires two or three metrics"),
        (("paid_count",), "parallel_readonly requires two or three metrics"),
        (("paid_count", 1), "parallel metric IDs must be non-empty strings"),
        (("paid_count", "paid_count"), "parallel metric IDs must be unique"),
        (("paid_count", "refund_fen"), "parallel metric ID is not allowed"),
    ],
)
def test_parallel_action_rule(metric_ids, message) -> None:
    with pytest.raises(ValueError) as caught:
        ParallelReadonlyAction(metric_ids)
    assert str(caught.value) == message


_LOCAL_ID = re.compile(r"local-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")


@pytest.fixture(params=["memory", "durable"])
def call_store(request):
    if request.param == "memory":
        yield ModelCallStore()
    else:
        with DurableModelCallStore(":memory:") as store:
            yield store


def test_call_store_new_call_identity(call_store) -> None:
    given = call_store.new_call("run-1", request_id="request-1")
    generated = call_store.new_call("run-1")
    assert (given.run_id, given.request_id, given.attempt_kind, given.retry_of_model_call_id) == ("run-1", "request-1", "new", None)
    assert _LOCAL_ID.fullmatch(given.model_call_id) and _LOCAL_ID.fullmatch(generated.model_call_id)
    assert _UUID.fullmatch(generated.request_id)
    assert given.model_call_id != generated.model_call_id
    assert call_store.get("run-1", given.model_call_id) == given


def test_call_store_transport_retry_keeps_the_logical_call(call_store) -> None:
    first = call_store.new_call("run-1", request_id="request-1")
    retry = call_store.transport_retry(first, request_id="request-2")
    generated = call_store.transport_retry(first)
    assert (retry.model_call_id, retry.request_id, retry.attempt_kind, retry.retry_of_model_call_id) == (
        first.model_call_id, "request-2", "transport_retry", first.model_call_id
    )
    assert _UUID.fullmatch(generated.request_id)
    assert call_store.get("run-1", first.model_call_id) == first
    assert call_store.attempts("run-1", first.model_call_id) == (first, retry, generated)


@pytest.mark.parametrize("run_id", ["", "  ", None])
def test_call_store_refuses_a_blank_run_without_recording_it(call_store, run_id) -> None:
    with pytest.raises(ValueError) as caught:
        call_store.new_call(run_id)
    assert type(caught.value) is ValueError and str(caught.value) == "run_id must be a non-empty string"


def test_call_store_unknown_call_is_a_lookup_error(call_store) -> None:
    with pytest.raises(CallIdentityError, match="model call identity was not found"):
        call_store.get("run-1", "local-missing")
    with pytest.raises(CallIdentityError):
        call_store.attempts("run-1", "local-missing")
    stranger = proposals.ModelCallIdentity("run-1", "local-x", "r", "new")
    with pytest.raises(CallIdentityError):
        call_store.transport_retry(stranger)
