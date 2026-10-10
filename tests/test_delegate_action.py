"""The delegate action of a B2 coordinator: its parsing, the checks of its subtasks, the budget split,
the outcome rule, the sentence a sub-agent sees, and that every other agent refuses it as before.

No database: the parser, the catalog checks and the bounded agent over the fixture tools.
"""

from __future__ import annotations

import itertools
import json

import pytest

from queryshield.agent.config import RunConfig
from queryshield.agent.context import MAX_SERVER_CONTEXT_CHARS, build_context
from queryshield.agent.delegation import (
    COORDINATOR,
    DelegationBudgetError,
    Subtask,
    allocate,
    delegation_outcome,
    resolve_subtasks,
    subtask_question,
    subtask_role,
)
from queryshield.agent.graph import RunResumeError
from queryshield.agent.metric_intent import build_metric_binding, declarable_metric_ids
from queryshield.agent.proposals import DelegateAction, ExecutionContext, ProposalParseError, parse_error_detail, parse_query_proposal, proposal_shape_summary
from queryshield.agent.runtime import B1_PROFILE, B2_PROFILE, build_b1_agent, build_b2_agent, product_run_config
from queryshield.agent.tenant_scope import has_explicit_foreign_tenant
from queryshield.agent.tool_execution import ClarificationRequiredError, ClarificationValueUnsupportedError, MetricContradictsQuestionError
from queryshield.auth.identity import IDENTITY_CONFIG
from queryshield.catalog import load_default_catalog
from queryshield.catalog.phrases import check_declaration, read_clarifications
from queryshield.tools.semantic import ToolError

import agent_core_scenarios as core
import test_clarification as clarification_cases
from multi_agent_support import AUG, SEP, delegate

CATALOG = load_default_catalog()
CONTEXT = ExecutionContext(run_id="run-delegate", tenant_id="A", principal_id="principal-A", role="requester")
SUBTASKS = [{"metrics": ["gross_fen"], "time_window": SEP}, {"metrics": ["net_fen"], "time_window": SEP}]


def _parse(payload, *, delegate=True):
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    return parse_query_proposal(raw, context=CONTEXT, model_call_id="mc-delegate", delegate=delegate)


def _resolve(subtasks, *, question="2026年9月的已支付订单总额和退款后净额分别是多少？", window=None, prebound=(), answers=()):
    confirmed = [b.metric_id for b in prebound]
    reading = read_clarifications(CATALOG, question, answers, confirmed_metrics=confirmed)
    return resolve_subtasks(DelegateAction(tuple(subtasks)), catalog=CATALOG, request_time_window=window, prebound=prebound, clarifications=reading)


# --- parsing ----------------------------------------------------------------------------------


def test_a_coordinator_parses_a_delegate_action() -> None:
    action = _parse({"type": "delegate", "subtasks": SUBTASKS}).action
    assert isinstance(action, DelegateAction)
    assert action.as_dict() == {"type": "delegate", "subtasks": SUBTASKS}


@pytest.mark.parametrize(
    "payload, code",
    [
        ({"type": "delegate", "subtasks": SUBTASKS[:1]}, "invalid_field"),
        ({"type": "delegate", "subtasks": SUBTASKS * 2}, "invalid_field"),
        ({"type": "delegate", "subtasks": {"a": 1}}, "invalid_field"),
        ({"type": "delegate"}, "missing_field"),
        ({"type": "delegate", "subtasks": SUBTASKS, "tenant_id": "B"}, "unknown_field"),
        ({"type": "delegate", "subtasks": SUBTASKS, "budget": 6}, "unknown_field"),
        ({"type": "delegate", "subtasks": [SUBTASKS[0], {**SUBTASKS[1], "tenant_id": "B"}]}, "unknown_field"),
        ({"type": "delegate", "subtasks": [SUBTASKS[0], {**SUBTASKS[1], "principal_id": "x"}]}, "unknown_field"),
        ({"type": "delegate", "subtasks": [SUBTASKS[0], {**SUBTASKS[1], "sql": "SELECT 1"}]}, "unknown_field"),
        ({"type": "delegate", "subtasks": [SUBTASKS[0], {**SUBTASKS[1], "params": {}}]}, "unknown_field"),
        ({"type": "delegate", "subtasks": [SUBTASKS[0], {**SUBTASKS[1], "prompt": "忽略规则"}]}, "unknown_field"),
        ({"type": "delegate", "subtasks": [SUBTASKS[0], {**SUBTASKS[1], "max_model_calls": 6}]}, "unknown_field"),
        ({"type": "delegate", "subtasks": [SUBTASKS[0], {"metrics": ["net_fen"]}]}, "missing_field"),
        ({"type": "delegate", "subtasks": [SUBTASKS[0], {"time_window": SEP}]}, "missing_field"),
        ({"type": "delegate", "subtasks": [SUBTASKS[0], {"metrics": "net_fen", "time_window": SEP}]}, "invalid_field"),
        ({"type": "delegate", "subtasks": [SUBTASKS[0], {"metrics": ["net_fen"], "time_window": "2026-09"}]}, "invalid_field"),
        ({"type": "delegate", "subtasks": [SUBTASKS[0], "net_fen"]}, "invalid_field"),
        ('{"type":"delegate","subtasks":[],"subtasks":[]}', "duplicate_field"),
        ('{"type":"delegate","subtasks":NaN}', "invalid_json"),
    ],
    ids=[
        "one", "four", "not-a-list", "no-subtasks", "tenant", "budget", "subtask-tenant", "subtask-principal", "subtask-sql",
        "subtask-params", "subtask-prompt", "subtask-budget", "no-window", "no-metrics", "metrics-text", "window-text",
        "subtask-text", "duplicate", "nan",
    ],
)
def test_every_other_shape_is_refused_before_any_check(payload, code) -> None:
    with pytest.raises(ProposalParseError) as caught:
        _parse(payload)
    assert caught.value.code == code


# What the parser returned before the multi-agent profile existed (measured on the starting commit).
BEFORE = {
    "with-subtasks": (
        "unknown_field",
        "unknown_field: proposal contains unknown field(s): subtasks",
        "proposal contains unknown field(s)",
        {"arguments": "absent", "basis": "absent", "json": "valid", "name": "absent", "top_level_extra_field_count": 1, "type": "<other>"},
    ),
    "bare": (
        "unknown_action",
        "unknown_action: proposal action is not supported",
        "proposal action is not supported",
        {"arguments": "absent", "basis": "absent", "json": "valid", "name": "absent", "top_level_extra_field_count": 0, "type": "<other>"},
    ),
}
RAW = {"with-subtasks": json.dumps({"type": "delegate", "subtasks": SUBTASKS}), "bare": '{"type":"delegate"}'}


@pytest.mark.parametrize("case", sorted(RAW))
def test_everyone_but_a_coordinator_refuses_delegate_exactly_as_before(case) -> None:
    with pytest.raises(ProposalParseError) as caught:
        _parse(RAW[case], delegate=False)
    exc = caught.value
    assert (exc.code, str(exc), parse_error_detail(exc), proposal_shape_summary(RAW[case])) == BEFORE[case]


@pytest.mark.parametrize("role", [None, subtask_role(1)], ids=["b1", "sub-agent"])
def test_a_b1_agent_and_a_sub_agent_end_on_delegate_as_before(role) -> None:
    tools = core.fixture_tools()[0]
    model = clarification_cases._Scripted([{"type": "delegate", "subtasks": SUBTASKS}])
    agent = core.agent_for(model, tools, role=role)
    result = agent.run(CONTEXT, "2026年9月的已支付订单总额和退款后净额分别是多少？")
    assert (result.status, result.error_code) == ("failed", "unknown_field")
    validation = [e for e in result.events if e["kind"] == "proposal_validation"]
    assert [(e["error_code"], e["error_detail"], e["action_shape"]) for e in validation] == [
        ("unknown_field", BEFORE["with-subtasks"][2], BEFORE["with-subtasks"][3])
    ]
    assert ("agent" in validation[0]) is (role is not None)


# --- the checks of the subtasks -----------------------------------------------------------------


def test_subtasks_get_catalog_built_bindings() -> None:
    subtasks = _resolve(SUBTASKS)
    assert [(s.index, s.metric_ids, s.time_window) for s in subtasks] == [
        (1, ["gross_fen"], {**SEP, "timezone": "UTC"}),
        (2, ["net_fen"], {**SEP, "timezone": "UTC"}),
    ]
    assert subtasks[0].bindings == (build_metric_binding(CATALOG, "gross_fen", SEP),)


@pytest.mark.parametrize(
    "second, window, code",
    [
        ({"metrics": ["no_such_metric"], "time_window": SEP}, None, "unknown_metric"),
        ({"metrics": ["net_fen", "gross_fen"], "time_window": SEP}, None, "invalid_metric_declaration"),
        ({"metrics": [], "time_window": SEP}, None, "invalid_metric_declaration"),
        ({"metrics": ["net_fen"], "time_window": {"start": "2026-09-01", "end": "2026-10-01"}}, None, "invalid_time_window"),
        ({"metrics": ["net_fen"], "time_window": AUG}, SEP, "time_window_mismatch"),
        ({"metrics": ["gross_fen"], "time_window": SEP}, None, "invalid_metric_declaration"),
    ],
    ids=["unknown", "net-with-other", "empty", "bad-window", "not-the-request-window", "repeated-metric-and-window"],
)
def test_a_declaration_error_is_a_tool_error_with_the_query_code(second, window, code) -> None:
    with pytest.raises(ToolError) as caught:
        _resolve([SUBTASKS[0], second], window=window)
    assert caught.value.code == code


def test_the_same_metric_in_two_windows_is_two_subtasks() -> None:
    subtasks = _resolve([{"metrics": ["paid_count"], "time_window": AUG}, {"metrics": ["paid_count"], "time_window": SEP}], question="2026年8月和2026年9月的已支付订单数分别是多少？")
    assert [s.time_window["start"] for s in subtasks] == [AUG["start"], SEP["start"]]


def test_the_phrase_table_reads_the_users_question() -> None:
    with pytest.raises(ClarificationRequiredError):
        _resolve([{"metrics": ["gross_fen"], "time_window": SEP}, {"metrics": ["paid_count"], "time_window": SEP}], question="2026年9月的销售额和已支付订单数分别是多少？")
    with pytest.raises(MetricContradictsQuestionError):
        _resolve([{"metrics": ["gross_fen"], "time_window": SEP}, {"metrics": ["paid_count"], "time_window": SEP}], question="2026年9月的退款后净额和已支付订单数分别是多少？")
    with pytest.raises(ClarificationValueUnsupportedError):
        _resolve([{"metrics": ["paid_count"], "time_window": AUG}, {"metrics": ["paid_count"], "time_window": SEP}], question="2026年8月和2026年9月的已取消订单数分别是多少？")


def test_after_a_clarification_the_subtasks_together_keep_the_users_choice() -> None:
    confirmed = (build_metric_binding(CATALOG, "gross_fen", SEP),)
    question, answers = "2026年9月的销售额和已支付订单数分别是多少？", ("按支付金额统计",)
    kept = _resolve([{"metrics": ["gross_fen"], "time_window": SEP}, {"metrics": ["paid_count"], "time_window": SEP}], question=question, prebound=confirmed, answers=answers)
    assert kept[0].bindings == confirmed  # the confirmed binding itself
    for subtasks in (
        [{"metrics": ["paid_count"], "time_window": SEP}, {"metrics": ["net_fen"], "time_window": SEP}],  # another option
        [{"metrics": ["paid_count"], "time_window": SEP}, {"metrics": ["paid_count"], "time_window": AUG}],  # dropped
    ):
        with pytest.raises(ToolError) as caught:
            _resolve(subtasks, question=question, prebound=confirmed, answers=answers)
        assert caught.value.code == "metric_declaration_mismatch"
    with pytest.raises(ToolError) as caught:  # the confirmed metric in another window
        _resolve([{"metrics": ["gross_fen"], "time_window": AUG}, {"metrics": ["paid_count"], "time_window": SEP}], question=question, prebound=confirmed, answers=answers)
    assert caught.value.code == "time_window_mismatch"


# --- the budget split ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_used, tool_used, count, expected",
    [(1, 0, 2, (2, 4)), (2, 1, 2, (1, 3)), (1, 0, 3, (1, 2)), (2, 1, 3, (1, 2)), (3, 2, 2, (1, 3)), (2, 5, 3, (1, 1)), (2, 6, 3, None), (4, 0, 2, None), (3, 0, 3, None)],
)
def test_each_sub_agent_gets_an_even_share_and_the_coordinator_keeps_one_call(model_used, tool_used, count, expected) -> None:
    arguments = dict(max_model_calls=6, max_tool_calls=8, max_seconds=60.0, model_used=model_used, tool_used=tool_used, elapsed=10.0, count=count)
    if expected is None:
        with pytest.raises(DelegationBudgetError) as caught:
            allocate(**arguments)
        assert caught.value.code == "delegation_budget_insufficient"
    else:
        assert allocate(**arguments) == (*expected, 50.0)


def test_the_run_never_exceeds_its_limits_whatever_was_used() -> None:
    for model_used, tool_used, count in itertools.product(range(1, 7), range(0, 9), (2, 3)):
        try:
            model, tool, seconds = allocate(max_model_calls=6, max_tool_calls=8, max_seconds=60.0, model_used=model_used, tool_used=tool_used, elapsed=1.0, count=count)
        except DelegationBudgetError:
            continue
        assert model >= 1 and tool >= 1 and seconds <= 60
        assert model_used + model * count + 1 <= 6 and tool_used + tool * count <= 8


def test_no_time_left_is_the_wall_clock_limit() -> None:
    with pytest.raises(DelegationBudgetError) as caught:
        allocate(max_model_calls=6, max_tool_calls=8, max_seconds=60.0, model_used=1, tool_used=0, elapsed=60.0, count=2)
    assert caught.value.code == "wall_clock_limit"


# --- the outcome rule -----------------------------------------------------------------------------

DONE = {"status": "succeeded", "final_action": {"type": "subtask_complete"}}


@pytest.mark.parametrize(
    "states, expected",
    [
        ([DONE, DONE], None),
        ([DONE, {"status": "succeeded", "final_action": {"type": "final_answer"}}], ("failed", "subtask_incomplete")),
        ([DONE, {"status": "waiting_user"}], ("failed", "subtask_incomplete")),
        ([DONE, {"status": "waiting_approval"}], ("failed", "subtask_incomplete")),
        ([DONE, {"status": "denied", "error_code": None}], ("failed", "subtask_incomplete")),
        ([{"status": "failed", "error_code": "upstream_timeout"}, {"status": "denied", "error_code": "forbidden"}], ("denied", "forbidden")),
        ([{"status": "limit_reached", "error_code": "tool_call_limit"}, {"status": "denied", "error_code": "forbidden"}], ("denied", "forbidden")),
        ([{"status": "failed", "error_code": "upstream_timeout"}, {"status": "limit_reached", "error_code": "tool_call_limit"}], ("limit_reached", "tool_call_limit")),
        ([{"status": "limit_reached", "error_code": "wall_clock_limit"}, {"status": "limit_reached", "error_code": "tool_call_limit"}], ("limit_reached", "wall_clock_limit")),
        ([{"status": "waiting_user"}, {"status": "failed", "error_code": "upstream_timeout"}], ("failed", "subtask_incomplete")),
        ([DONE, {"status": "failed", "error_code": "upstream_timeout"}, {"status": "failed", "error_code": "model_rate_limited"}], ("failed", "upstream_timeout")),
    ],
)
def test_the_run_outcome_rule(states, expected) -> None:
    assert delegation_outcome(states) == expected


# --- the sentence a sub-agent sees ---------------------------------------------------------------


@pytest.mark.parametrize(
    "metrics", [[m] for m in declarable_metric_ids(CATALOG)] + [["paid_count", "gross_fen"]], ids=lambda m: "+".join(m)
)
def test_the_subtask_sentence_names_only_its_own_subtask_and_passes_the_phrase_table(metrics) -> None:
    subtask = Subtask(1, tuple(build_metric_binding(CATALOG, m, AUG) for m in metrics))
    sentence = subtask_question(CATALOG, subtask)
    assert AUG["start"] in sentence and AUG["end"] in sentence and all(m in sentence for m in metrics)
    assert not has_explicit_foreign_tenant(sentence, "A") and not has_explicit_foreign_tenant(sentence, "B")
    assert check_declaration(read_clarifications(CATALOG, sentence, (), confirmed_metrics=metrics), metrics) is None


# --- contexts -----------------------------------------------------------------------------------


def _http_identity() -> ExecutionContext:
    values = IDENTITY_CONFIG.values()
    return ExecutionContext(
        run_id="run-" + "a" * 8 + "-" + "a" * 4 + "-" + "a" * 4 + "-" + "a" * 4 + "-" + "a" * 12,
        tenant_id=max((v["tenant_id"] for v in values), key=len),
        principal_id=max((v["principal_id"] for v in values), key=len),
        role=max((v["role"] for v in values), key=len),
    )


def test_the_coordinators_worst_server_message_stays_within_the_policy_limit() -> None:
    """B2 runs only behind HTTP (uuid run ids); never parallel; every other length parameter at its worst."""

    config = product_run_config(B2_PROFILE, catalog=CATALOG, retriever=None)
    december = {"start": "2026-12-01T00:00:00Z", "end": "2027-01-01T00:00:00Z", "timezone": "UTC"}
    sizes = []
    for retrieval, window, count, confirmed in itertools.product((True, False), (None, {**SEP, "timezone": "UTC"}, december), range(4), (None, "paid_count", "gross_fen", "net_fen")):
        bindings = [{"metric_id": m, "result_position": m, "unit": "CNY_fen", "time_window": window or {**SEP, "timezone": "UTC"}} for m in ("paid_count", "gross_fen", "net_fen")[:count]]
        built = build_context(
            _http_identity(), "q", request_time_window=window, metric_bindings=bindings, confirmed_metric=confirmed, time_window=window,
            parallel_available=False, retrieval_available=retrieval, delegate_available=True, run_config=config,
        )
        sizes.append(len(built.messages[0]["content"]))
    assert max(sizes) <= MAX_SERVER_CONTEXT_CHARS - 500 == 11_500, max(sizes)


def test_only_a_coordinator_is_told_about_delegate() -> None:
    def system(**options) -> str:
        return build_context(CONTEXT, "q", parallel_available=False, **options).messages[0]["content"]

    assert '"delegate"' in system(delegate_available=True)
    assert '"delegate"' not in system()
    assert system() == build_context(CONTEXT, "q", parallel_available=False, delegate_available=False).messages[0]["content"]


def test_b2_has_its_own_versions_and_runs_json_only(monkeypatch) -> None:
    b1 = product_run_config(B1_PROFILE, catalog=CATALOG, retriever=None)
    b2 = product_run_config(B2_PROFILE, catalog=CATALOG, retriever=None)
    assert b2.profile == B2_PROFILE and (b2.prompt_version, b2.action_schema_version) != (b1.prompt_version, b1.action_schema_version)
    assert b2.model_protocol == "json"
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", "native")
    from queryshield.agent.runtime import RuntimeConfigurationError

    with pytest.raises(RuntimeConfigurationError) as caught:
        product_run_config(B2_PROFILE, catalog=CATALOG, retriever=None)
    assert caught.value.code == "invalid_model_protocol"


# --- resuming under another profile ---------------------------------------------------------------


def _waiting_checkpoint(builder, config: RunConfig) -> dict:
    tools = core.fixture_tools()[0]
    agent = builder(clarification_cases._Scripted([clarification_cases._ask()]), tools, run_config=config)
    result = agent.run(CONTEXT, "2026年9月的已支付订单数是多少？")
    assert result.status == "waiting_user"
    return agent.export_waiting_checkpoint(CONTEXT.run_id)


@pytest.mark.parametrize("waits_under, resumes_under", [(B1_PROFILE, B2_PROFILE), (B2_PROFILE, B1_PROFILE)])
def test_a_checkpoint_never_resumes_under_the_other_profile(waits_under, resumes_under) -> None:
    builders = {B1_PROFILE: build_b1_agent, B2_PROFILE: build_b2_agent}
    configs = {profile: product_run_config(profile, catalog=CATALOG, retriever=None) for profile in builders}
    checkpoint = _waiting_checkpoint(builders[waits_under], configs[waits_under])
    agent = builders[resumes_under](clarification_cases._Scripted([]), core.fixture_tools()[0], run_config=configs[resumes_under])
    with pytest.raises(RunResumeError) as caught:
        agent.resume_from_checkpoint(CONTEXT, "2026年9月", checkpoint)
    assert caught.value.code == "resume_profile_mismatch"


def test_only_the_coordinator_role_delegates() -> None:
    assert COORDINATOR.can_delegate and not COORDINATOR.ends_when_bound
    assert subtask_role(2).label == "subtask-2" and subtask_role(2).ends_when_bound and not subtask_role(2).can_delegate
