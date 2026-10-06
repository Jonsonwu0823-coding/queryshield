"""Scripted runs of the agent core whose full output is pinned byte for byte.

A refactor of ``agent/graph.py``, ``agent/runtime.py``, ``agent/context.py`` or
``agent/config.py`` must not change any field, key order, value or hash these
scenarios produce.  Time is a fake clock and every random id is replaced by a
numbered placeholder, so equal runs give equal text.  ``SCENARIOS`` maps a name
to a function that returns the object to pin; ``agent_core_pins.json`` holds
the SHA-256 of each canonical text and the outcome the scenario is meant to
reach.  Run ``python tests/agent_core_scenarios.py --write`` after an
intentional change of agent behaviour.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from queryshield.agent import BoundedAgent, GraphLimits, ModelCallStore  # noqa: E402
from queryshield.agent.config import DEFAULT_RUN_CONFIG, NATIVE_VERSIONS, RunConfig  # noqa: E402
from queryshield.agent.context import NON_METRIC_QUERY_EXAMPLE, build_context  # noqa: E402
from queryshield.agent.parallel import BranchExecution, ParallelPlan, ParallelScheduler  # noqa: E402
from queryshield.agent.proposals import ExecutionContext  # noqa: E402
from queryshield.agent import runtime  # noqa: E402
from queryshield.agent.runtime import b1_result_payload, run_b0_single_pass  # noqa: E402
from queryshield.catalog import load_default_catalog  # noqa: E402
from queryshield.providers.contracts import ModelCallResult, ModelProviderError, ModelUsage  # noqa: E402
from queryshield.providers.fake_model import FakeModel  # noqa: E402
from queryshield.tools.semantic import ControlledTools, ToolError  # noqa: E402

import test_clarification as clarification_cases  # noqa: E402
import test_ask_review as ask_review  # noqa: E402

PINS_PATH = Path(__file__).with_name("agent_core_pins.json")
SEPTEMBER = clarification_cases.SEPTEMBER
NATIVE = replace(DEFAULT_RUN_CONFIG, **NATIVE_VERSIONS)
FIXED_TIME = datetime(2026, 9, 21, tzinfo=timezone.utc)

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_FACT = re.compile(r"fact-[0-9a-f]{24}")
# Hashes of model output that embeds a random result id.
_HASH_KEYS = {"content_sha256", "draft_sha256", "arguments_sha256"}


class StepClock:
    """Every reading advances by ``step`` seconds, so elapsed values depend on how often the code reads it."""

    def __init__(self, step: float = 0.25) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def _scrub(value: object) -> object:
    if isinstance(value, dict):
        scrubbed = {key: "<hash>" if key in _HASH_KEYS else _scrub(item) for key, item in value.items()}
        if value.get("kind") == "model_call" and value.get("status") == "failed":
            # The provider fields come from iterating a set, so their order changes between processes.
            return dict(sorted(scrubbed.items()))
        return scrubbed
    if isinstance(value, (list, tuple)):
        return [_scrub(item) for item in value]
    return value


def canonical(value: object) -> str:
    """JSON text in the object's own key order with random ids numbered by first appearance."""

    text = json.dumps(_scrub(value), ensure_ascii=False, default=str)
    for pattern, label in ((_UUID, "id"), (_FACT, "fact")):
        seen: dict[str, str] = {}
        text = pattern.sub(lambda match: seen.setdefault(match.group(0), f"<{label}{len(seen) + 1}>"), text)
    return text


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


# --- building blocks ----------------------------------------------------------------


def fixture_tools() -> tuple[ControlledTools, clarification_cases._Recording]:
    executor = clarification_cases._Recording(clock=lambda: FIXED_TIME)
    return ControlledTools(catalog=load_default_catalog(), executor=executor), executor


def agent_for(model, tools, *, config=None, step=0.25, **options) -> BoundedAgent:
    return BoundedAgent(model, tools=tools, call_store=ModelCallStore(), clock=StepClock(step), run_config=config, **options)


class ErrorModel:
    """Fails like a provider that timed out."""

    mode = "fake"

    def complete(self, messages, *, request_id=None, model_call_id=None):
        raise ModelProviderError(
            "upstream_timeout",
            {"status": "failed", "provider": "scripted", "usage": None, "usage_status": "unknown", "http_status": 504},
        )


class UsageModel:
    """A scripted model that reports token usage, optionally malformed."""

    mode = "fake"

    def __init__(self, steps, usages) -> None:
        self.steps = list(steps)
        self.usages = list(usages)
        self.calls = 0

    def complete(self, messages, *, request_id=None, model_call_id=None):
        index = self.calls
        self.calls += 1
        step = self.steps[index]
        action = step(messages) if callable(step) else step
        usage, status = self.usages[index]
        return ModelCallResult(
            mode="fake",
            provider="usage-scripted",
            model="usage-scripted-v1",
            request_id=request_id,
            model_call_id=model_call_id,
            provider_call_id=f"call-{index + 1}",
            provider_request_id=f"request-{index + 1}",
            content=json.dumps(action, ensure_ascii=False),
            usage=usage,
            usage_status=status,
        )


def run_record(agent: BoundedAgent, context: ExecutionContext, question: str, **kwargs) -> dict[str, object]:
    """The result, and the checkpoint when the run waits for the user."""

    result = agent.run(context, question, **kwargs)
    record: dict[str, object] = {"result": result.as_dict()}
    if result.status == "waiting_user":
        record["checkpoint"] = agent.export_waiting_checkpoint(context.run_id)
    return record


def outcome(value: object) -> str:
    """``status/error_code`` of a pinned record, so a scenario that stops reaching its path is noticed."""

    if isinstance(value, dict):
        for key in ("result", "record"):
            if key in value:
                return outcome(value[key])
        status = value.get("status")
        if status is not None:
            return f"{status}/{value.get('error_code')}"
        if "checkpoint_version" in value:
            return f"checkpoint/{value['checkpoint_version']}"
        nested = [outcome(item) for item in value.values() if isinstance(item, dict)]
        if nested and all(item != "-" for item in nested):
            return ",".join(nested)
    return "-"


SCENARIOS: dict[str, Callable[[], object]] = {}


def scenario(function: Callable[[], object]) -> Callable[[], object]:
    SCENARIOS[function.__name__] = function
    return function


def _run(steps, question, *, run_id="run-core", tools=None, window=SEPTEMBER, model=None, run_kwargs=None, **options):
    tools = tools or fixture_tools()[0]
    agent = agent_for(model or clarification_cases._Scripted(steps), tools, **options)
    kwargs = dict(run_kwargs or {})
    if window is not None:
        kwargs["request_time_window"] = window
    return run_record(agent, clarification_cases._context(run_id), question, **kwargs)


GROSS_QUESTION = "2026年9月支付金额"
GROUPED_SQL = (
    "SELECT o.customer_id, SUM(o.amount_fen) AS gross_fen FROM orders AS o INNER JOIN customers AS c "
    "ON o.tenant_id = c.tenant_id AND o.customer_id = c.customer_id "
    f"WHERE {clarification_cases.FILTER} GROUP BY o.customer_id"
)
NO_METRIC_QUERY = {"type": "tool_call", "name": "query_readonly", "arguments": {"sql": "SELECT c.customer_id FROM customers AS c", "params": {}}}
SEARCH = {"type": "tool_call", "name": "search_catalog", "arguments": {"query": "退款后净额", "top_k": 3}}
DESCRIBE = {"type": "tool_call", "name": "describe_tables", "arguments": {"tables": ["orders"]}}
NO_DATA = {"type": "final_answer", "answer": "", "source_ids": [], "fact_refs": [], "basis": "no_data"}
KNOWLEDGE = {"type": "final_answer", "answer": "模型写的定义", "source_ids": ["model-source"], "fact_refs": [], "basis": "knowledge"}
PLAIN_ANSWER = {"type": "final_answer", "answer": "模型写的答案", "source_ids": [], "fact_refs": []}
FAKE_CITATION = {
    "type": "final_answer",
    "answer": "模型写的答案",
    "source_ids": [],
    "fact_refs": [{"result_id": "result-not-in-this-run", "metric_id": "gross_fen"}],
}
DEFINITION_QUESTION = "退款后净额是怎么算的？"


def _cite_metrics(*extra_metrics):
    def step(messages):
        action = clarification_cases._cite(messages)
        output_id = action["fact_refs"][0]["result_id"]
        action["fact_refs"] += [{"result_id": output_id, "metric_id": metric} for metric in extra_metrics]
        return action

    return step


def _cite_nothing(messages):
    return dict(PLAIN_ANSWER)


# --- queries and answers --------------------------------------------------------------


@scenario
def scalar_verified():
    return _run([clarification_cases._query(), clarification_cases._cite], GROSS_QUESTION)


@scenario
def two_scalar_metrics():
    return _run([clarification_cases._query(clarification_cases.DUAL_SQL, ("paid_count", "gross_fen")), clarification_cases._cite], "2026年9月已支付订单数和支付金额")


@scenario
def net_plan_verified():
    return _run([clarification_cases._query(clarification_cases.GROSS_SQL, ("net_fen",)), clarification_cases._cite], "2026年9月退款后净额")


@scenario
def rowset_answer_unverified():
    return _run([clarification_cases._query(GROUPED_SQL, ("gross_fen",)), clarification_cases._cite], "2026年9月按客户的支付金额")


@scenario
def non_metric_rows_answer():
    return _run([NO_METRIC_QUERY, PLAIN_ANSWER], "列出客户编号")


@scenario
def repair_then_success():
    return _run([clarification_cases._query(clarification_cases.BAD_SQL), clarification_cases._query(), clarification_cases._cite], GROSS_QUESTION)


@scenario
def repair_budget_exhausted():
    return _run([clarification_cases._query(clarification_cases.BAD_SQL), clarification_cases._query(clarification_cases.BAD_SQL)], GROSS_QUESTION)


@scenario
def policy_denied_table():
    step = {"type": "tool_call", "name": "query_readonly", "arguments": {"sql": "SELECT x FROM secrets", "params": {}}}
    return _run([step], "列出机密")


@scenario
def approval_required():
    return _run([json.loads(NON_METRIC_QUERY_EXAMPLE)], "列出客户姓名")


@scenario
def tool_name_written_as_type_then_repaired():
    wrong = {"type": "query_readonly", "name": "query_readonly", "arguments": clarification_cases._query()["arguments"]}
    return _run([wrong, clarification_cases._query(), clarification_cases._cite], GROSS_QUESTION)


@scenario
def tool_name_written_as_type_twice():
    wrong = {"type": "query_readonly", "name": "query_readonly", "arguments": clarification_cases._query()["arguments"]}
    return _run([wrong, wrong], GROSS_QUESTION)


@scenario
def unparseable_model_output():
    return _run(["not an action"], GROSS_QUESTION)


@scenario
def deny_action():
    return _run([{"type": "deny", "reason": "越权请求"}], "删除所有订单")


@scenario
def provider_failure():
    return _run([], GROSS_QUESTION, model=ErrorModel())


@scenario
def explicit_foreign_tenant_is_denied():
    return _run([], "tenant-B 的支付金额", run_id="run-foreign")


@scenario
def search_without_retriever():
    return _run([SEARCH], "退款后净额", retrieval_available=False)


@scenario
def model_call_limit():
    return _run([SEARCH, SEARCH], "退款后净额", limits=GraphLimits(max_model_calls=1))


@scenario
def tool_call_limit():
    return _run([SEARCH, SEARCH, clarification_cases._cite], "退款后净额", limits=GraphLimits(max_tool_calls=1))


@scenario
def wall_clock_limit():
    return _run([SEARCH, SEARCH], "退款后净额", step=40.0)


@scenario
def initial_retrieval_items():
    item = next(entry for entry in load_default_catalog().search_items() if entry["id"] == "metric.net_fen")
    tools = fixture_tools()[0]
    agent = agent_for(clarification_cases._Scripted([KNOWLEDGE]), tools)
    return run_record(agent, clarification_cases._context("run-initial"), DEFINITION_QUESTION, _evaluation_initial_retrieval_items=[item])


# --- asking and the phrase table -------------------------------------------------------


@scenario
def catalog_gate_waits_on_order_scope():
    return _run([clarification_cases._query(clarification_cases.COUNT_SQL, ("paid_count",))], "2026年9月订单数是多少")


@scenario
def ask_not_needed_then_plain_ask():
    return _run([clarification_cases._ask(), clarification_cases._ask("请问是哪一个月？")], "支付金额是多少", window=None)


@scenario
def ask_not_needed_twice():
    return _run([clarification_cases._ask(), clarification_cases._ask()], "支付金额是多少", window=None)


@scenario
def ask_names_the_catalog_rule():
    return _run([clarification_cases._ask(clarification_cases.MODEL_ASK, ask_review.MB)], "2026年9月销售额")


@scenario
def plain_ask_after_a_repair():
    return _run([clarification_cases._query(clarification_cases.BAD_SQL), clarification_cases._ask("请问是哪一个月？")], "支付金额是多少", window=None)


@scenario
def contradiction_then_corrected_answer():
    return _run([ask_review.NET_AS_GROSS_SQL, clarification_cases._query(), clarification_cases._cite], GROSS_QUESTION)


@scenario
def contradiction_twice():
    return _run([ask_review.NET_AS_GROSS_SQL, ask_review.NET_AS_GROSS_SQL], GROSS_QUESTION)


@scenario
def contradiction_then_repair_then_answer():
    return _run([ask_review.NET_AS_GROSS_SQL, clarification_cases._query(clarification_cases.BAD_SQL), clarification_cases._query(), clarification_cases._cite], GROSS_QUESTION)


@scenario
def prebound_metric_makes_ask_unneeded():
    return _run([clarification_cases._ask(), clarification_cases._ask()], "2026年9月销售额", run_kwargs={"metric_bindings": ask_review._bound("net_fen")})


# --- the answer contract --------------------------------------------------------------


@scenario
def no_data_reply():
    return _run([NO_DATA], "你好", window=None)


@scenario
def no_data_after_a_query_is_a_conflict():
    return _run([clarification_cases._query(), NO_DATA, NO_DATA], GROSS_QUESTION)


@scenario
def ungrounded_answer_is_sent_back_then_queries():
    return _run([PLAIN_ANSWER, clarification_cases._query(), clarification_cases._cite], GROSS_QUESTION)


@scenario
def ungrounded_answer_twice():
    return _run([PLAIN_ANSWER, PLAIN_ANSWER], GROSS_QUESTION)


@scenario
def undeclared_metric_reference_is_repaired():
    return _run([clarification_cases._query(), _cite_metrics("paid_count"), clarification_cases._cite], GROSS_QUESTION)


@scenario
def citation_before_any_query():
    return _run([FAKE_CITATION, FAKE_CITATION], GROSS_QUESTION)


@scenario
def bound_result_not_cited_is_terminal():
    return _run([clarification_cases._query(), _cite_nothing], GROSS_QUESTION)


@scenario
def knowledge_answer_citing_results_is_a_conflict():
    cited = {**KNOWLEDGE, "fact_refs": [{"result_id": "result-x", "metric_id": "gross_fen"}]}
    return _run([clarification_cases._query(), cited, cited], GROSS_QUESTION)


def _broken_evidence(tools, message_code="result_not_found"):
    def lookup(result_id, *, context):
        raise ToolError(message_code, "gone")

    tools.get_result_evidence = lookup
    return tools


@scenario
def evidence_of_a_successful_query_cannot_be_loaded():
    return _run([clarification_cases._query(), clarification_cases._cite], GROSS_QUESTION, tools=_broken_evidence(fixture_tools()[0]))


@scenario
def citing_an_unknown_result_after_a_query():
    return _run([NO_METRIC_QUERY, FAKE_CITATION], "列出客户编号")


@scenario
def tool_inputs_are_summarized_without_model_text():
    steps = [
        {"type": "tool_call", "name": "describe_tables", "arguments": {"tables": ["orders", "customers"]}},
        {"type": "tool_call", "name": "search_catalog", "arguments": {"query": "退款后净额"}},
        clarification_cases._query(clarification_cases.GROSS_SQL, ("gross_fen", "made_up_metric")),
        clarification_cases._query(),
        clarification_cases._cite,
    ]
    return _run(steps, GROSS_QUESTION)


@scenario
def knowledge_from_a_search():
    return _run([SEARCH, KNOWLEDGE], DEFINITION_QUESTION, window=None)


@scenario
def knowledge_without_source_triggers_the_server_search():
    return _run([KNOWLEDGE, KNOWLEDGE], DEFINITION_QUESTION, window=None)


@scenario
def server_search_without_retriever():
    return _run([KNOWLEDGE, KNOWLEDGE], DEFINITION_QUESTION, window=None, retrieval_available=False)


def _patched_search(replacement):
    tools = fixture_tools()[0]
    tools.search_catalog = replacement
    return tools


@scenario
def server_search_finds_nothing():
    tools = _patched_search(lambda arguments, *, context: {"items": []})
    return _run([KNOWLEDGE, KNOWLEDGE], DEFINITION_QUESTION, window=None, tools=tools)


@scenario
def server_search_error():
    def broken(arguments, *, context):
        raise ToolError("retrieval_unavailable", "the retriever is down")

    return _run([KNOWLEDGE, KNOWLEDGE], DEFINITION_QUESTION, window=None, tools=_patched_search(broken))


@scenario
def server_search_with_no_tool_call_left():
    return _run([DESCRIBE, KNOWLEDGE, KNOWLEDGE], DEFINITION_QUESTION, window=None, limits=GraphLimits(max_tool_calls=1))


# --- parallel reads -------------------------------------------------------------------


PARALLEL_METRICS = ("gross_fen", "paid_count")


def _parallel_plan(context, metrics=PARALLEL_METRICS):
    return ParallelPlan.from_context(context, metrics, time_window={**SEPTEMBER, "timezone": "UTC", "interval": "[start,end)"})


def _parallel_scheduler():
    def runner(context, metric_id, branch_id) -> BranchExecution:
        return BranchExecution(metric_id, f"result-{metric_id}", ({metric_id: 1},), "2026-09-21T00:00:00Z")

    return ParallelScheduler(runner)


def _parallel_agent(steps, *, with_scheduler=True, limits=None, metrics=PARALLEL_METRICS):
    tools = fixture_tools()[0]
    context = clarification_cases._context("run-parallel")

    scheduler = _parallel_scheduler() if with_scheduler else None
    agent = agent_for(clarification_cases._Scripted(steps), tools, parallel_scheduler=scheduler, limits=limits)
    return run_record(
        agent, context, "2026年9月支付金额和已支付订单数", parallel_plan=_parallel_plan(context, metrics), request_time_window=SEPTEMBER
    )


PARALLEL = {"type": "parallel_readonly", "metric_ids": ["gross_fen", "paid_count"]}


@scenario
def parallel_group_succeeds():
    return _parallel_agent([PARALLEL, PLAIN_ANSWER])


@scenario
def parallel_without_scheduler_is_repaired():
    return _parallel_agent([PARALLEL, clarification_cases._query(), clarification_cases._cite], with_scheduler=False)


@scenario
def parallel_without_scheduler_twice():
    return _parallel_agent([PARALLEL, PARALLEL], with_scheduler=False)


@scenario
def parallel_over_the_tool_budget():
    return _parallel_agent([PARALLEL], limits=GraphLimits(max_tool_calls=1))


@scenario
def parallel_contradiction_twice():
    steps = [{"type": "parallel_readonly", "metric_ids": ["net_fen", "paid_count"]}] * 2
    return _parallel_agent(steps, metrics=("net_fen", "paid_count"))


# --- resume ---------------------------------------------------------------------------


def _waiting_agent(question="支付金额是多少", step=None, config=None, tools=None):
    tools = tools or fixture_tools()[0]
    agent = agent_for(clarification_cases._Scripted(step or [clarification_cases._ask("请问是哪一个月？")]), tools, config=config)
    context = clarification_cases._context("run-resume")
    first = agent.run(context, question)
    assert first.status == "waiting_user"
    return agent, context, tools, agent.export_waiting_checkpoint(context.run_id)


def _resumer(steps, tools, config=None):
    return agent_for(clarification_cases._Scripted(steps), tools, config=config)


@scenario
def resume_from_checkpoint_answers():
    agent, context, tools, checkpoint = _waiting_agent()
    second = _resumer([clarification_cases._query(), clarification_cases._cite], tools)
    return {"checkpoint": checkpoint, "result": second.resume_from_checkpoint(context, "2026年9月", checkpoint).as_dict()}


@scenario
def resume_in_memory_answers():
    agent, context, tools, checkpoint = _waiting_agent(step=[clarification_cases._ask("请问是哪一个月？"), clarification_cases._query(), clarification_cases._cite])
    return {"result": agent.resume(context, "2026年9月").as_dict()}


@scenario
def resume_from_checkpoint_waits_again():
    agent, context, tools, checkpoint = _waiting_agent()
    second = _resumer([clarification_cases._ask("请问是哪一个月？")], tools)
    result = second.resume_from_checkpoint(context, "不知道", checkpoint)
    return {"result": result.as_dict(), "checkpoint": second.export_waiting_checkpoint(context.run_id)}


@scenario
def resume_after_catalog_question():
    agent, context, tools, checkpoint = _waiting_agent("2026年9月订单数是多少", [clarification_cases._query(clarification_cases.COUNT_SQL, ("paid_count",))])
    second = _resumer([clarification_cases._query(clarification_cases.COUNT_SQL, ("paid_count",)), clarification_cases._cite], tools)
    return {"checkpoint": checkpoint, "result": second.resume_from_checkpoint(context, "只统计已支付", checkpoint).as_dict()}


@scenario
def continue_waiting_for_clarification():
    agent, context, tools, checkpoint = _waiting_agent()
    second = _resumer([], tools)
    result = second.continue_waiting_for_clarification(context, " 还没想好 ", checkpoint)
    return {"result": result.as_dict(), "checkpoint": second.export_waiting_checkpoint(context.run_id)}


@scenario
def fail_unsupported_clarification():
    agent, context, tools, checkpoint = _waiting_agent()
    second = _resumer([], tools)
    return {"result": second.fail_unsupported_clarification(context, "包含已取消", checkpoint, note="目录说明").as_dict()}


@scenario
def checkpoint_after_a_repair():
    agent, context, tools, checkpoint = _waiting_agent(step=[clarification_cases._query(clarification_cases.BAD_SQL), clarification_cases._ask("请问是哪一个月？")])
    return {"checkpoint": checkpoint}


@scenario
def prepared_checkpoint_is_v3():
    context = clarification_cases._context("run-prepared")
    checkpoint = BoundedAgent.prepared_waiting_user_checkpoint(
        context, "支付金额是多少", run_config=DEFAULT_RUN_CONFIG, waiting_question="请问是哪一个月？", request_time_window=SEPTEMBER
    )
    tools = fixture_tools()[0]
    result = _resumer([clarification_cases._query(), clarification_cases._cite], tools).resume_from_checkpoint(context, "2026年9月", checkpoint)
    return {"checkpoint": checkpoint, "result": result.as_dict()}


@scenario
def resume_keeps_bindings_window_and_parallel_plan():
    tools = fixture_tools()[0]
    context = clarification_cases._context("run-resume-state")
    agent = agent_for(clarification_cases._Scripted([clarification_cases._ask("请问是哪一个月？")]), tools, parallel_scheduler=_parallel_scheduler())
    first = agent.run(
        context,
        "2026年9月支付金额",
        metric_bindings=ask_review._bound("gross_fen"),
        parallel_plan=_parallel_plan(context),
        request_time_window=SEPTEMBER,
    )
    checkpoint = agent.export_waiting_checkpoint(context.run_id)
    second = agent_for(clarification_cases._Scripted([clarification_cases._query(), clarification_cases._cite]), tools, parallel_scheduler=_parallel_scheduler())
    result = second.resume_from_checkpoint(context, "2026年9月", checkpoint)
    return {"first": first.as_dict(), "checkpoint": checkpoint, "result": result.as_dict()}


@scenario
def question_too_long_for_the_context():
    return _run([], "字" * 9000)


@scenario
def server_search_mcp_failure():
    def broken(arguments, *, context):
        raise ToolError("mcp_unavailable", "the metadata server is down")

    return _run([KNOWLEDGE, KNOWLEDGE], DEFINITION_QUESTION, window=None, tools=_patched_search(broken))


@scenario
def native_checkpoint_resumes_under_native():
    tools = fixture_tools()[0]
    context = clarification_cases._context("run-native-resume")
    first = agent_for(FakeModel(), tools, config=NATIVE)
    waiting = first.run(context, "支付金额是多少？")
    checkpoint = first.export_waiting_checkpoint(context.run_id)
    second = agent_for(FakeModel(), tools, config=NATIVE)
    result = second.resume_from_checkpoint(context, "2026年9月，按支付金额", checkpoint)
    return {"first": waiting.as_dict(), "checkpoint": checkpoint, "result": result.as_dict()}


# --- the default Fake model, both protocols -------------------------------------------

FAKE_QUESTIONS = (
    "2026年9月已支付订单总额",
    "2026年9月支付金额",
    "2026年9月退款后净额",
    "支付金额是多少？",
    "列出客户姓名",
    "2026年9月按客户的支付金额",
    "你好",
)


def _fake_runs(config):
    records = {}
    for index, question in enumerate(FAKE_QUESTIONS):
        tools = fixture_tools()[0]
        agent = agent_for(FakeModel(), tools, config=config)
        records[question] = run_record(agent, clarification_cases._context(f"run-fake-{index}"), question)
    return records


@scenario
def fake_model_json_protocol():
    return _fake_runs(DEFAULT_RUN_CONFIG)


@scenario
def fake_model_native_protocol():
    return _fake_runs(NATIVE)


# --- results as the product shapes them -----------------------------------------------


@scenario
def b1_payload_binds_facts_to_the_context():
    tools = fixture_tools()[0]
    context = clarification_cases._context("run-payload")
    result = agent_for(clarification_cases._Scripted([clarification_cases._query(), clarification_cases._cite]), tools).run(context, GROSS_QUESTION, request_time_window=SEPTEMBER)
    return b1_result_payload(result, context, GROSS_QUESTION)


@scenario
def b1_payload_marks_a_pre_model_rejection():
    tools = fixture_tools()[0]
    context = clarification_cases._context("run-payload-foreign")
    result = agent_for(clarification_cases._Scripted([]), tools).run(context, "租户B的支付金额")
    return b1_result_payload(result, context, "租户B的支付金额")


# --- usage ----------------------------------------------------------------------------

KNOWN = (ModelUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12), "known")
UNKNOWN = (None, "unknown")
INCONSISTENT = (ModelUsage(prompt_tokens=10, completion_tokens=2, total_tokens=13), "known")
NEGATIVE = (ModelUsage(prompt_tokens=-1, completion_tokens=2, total_tokens=1), "known")
KNOWN_BUT_NO_USAGE = (None, "known")
UNKNOWN_WITH_USAGE = (ModelUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2), "unknown")


def _usage_run(usages):
    steps = [SEARCH] * (len(usages) - 1) + [NO_DATA]
    model = UsageModel(steps, usages)
    return run_record(agent_for(model, fixture_tools()[0]), clarification_cases._context("run-usage"), "退款后净额")


@scenario
def usage_all_known():
    return _usage_run([KNOWN, KNOWN, KNOWN])


@scenario
def usage_with_an_unknown_call():
    return _usage_run([KNOWN, UNKNOWN, KNOWN])


@scenario
def usage_inconsistent_total():
    return _usage_run([KNOWN, INCONSISTENT])


@scenario
def usage_negative_tokens():
    return _usage_run([NEGATIVE])


@scenario
def usage_known_status_without_usage():
    return _usage_run([KNOWN_BUT_NO_USAGE, KNOWN])


@scenario
def usage_unknown_status_with_numbers():
    return _usage_run([UNKNOWN_WITH_USAGE, KNOWN])


# --- the single-pass baseline ---------------------------------------------------------

B0_WINDOW = {**SEPTEMBER, "timezone": "UTC"}


def _b0(question, model, *, tools=None, config=None):
    tools = tools or fixture_tools()[0]
    context = clarification_cases._context("run-b0")
    kwargs = {} if config is None else {"run_config": config}
    real_counter = runtime.perf_counter
    runtime.perf_counter = StepClock(0.25)
    try:
        record = run_b0_single_pass(model, tools, context, question, time_window=B0_WINDOW, **kwargs)
    finally:
        runtime.perf_counter = real_counter
    return {"record": record}


def _b0_steps(question, action):
    return _b0(question, clarification_cases._Scripted([action]))


@scenario
def b0_scalar_verified():
    return _b0_steps(GROSS_QUESTION, clarification_cases._query())


@scenario
def b0_net_plan():
    return _b0_steps("2026年9月退款后净额", clarification_cases._query(clarification_cases.GROSS_SQL, ("net_fen",)))


@scenario
def b0_rowset():
    return _b0_steps("2026年9月按客户的支付金额", clarification_cases._query(GROUPED_SQL, ("gross_fen",)))


@scenario
def b0_foreign_tenant():
    return _b0("tenant-B 的支付金额", clarification_cases._Scripted([]))


@scenario
def b0_provider_timeout():
    return _b0(GROSS_QUESTION, ErrorModel())


@scenario
def b0_unparseable_output():
    return _b0_steps(GROSS_QUESTION, "not an action")


@scenario
def b0_search_is_not_allowed():
    return _b0_steps("退款后净额", SEARCH)


@scenario
def b0_clarification_required():
    return _b0_steps("2026年9月销售额是多少？", clarification_cases._query())


@scenario
def b0_unsupported_scope():
    return _b0_steps("2026年9月已取消订单数", clarification_cases._query(clarification_cases.COUNT_SQL, ("paid_count",)))


@scenario
def b0_contradiction():
    return _b0_steps(GROSS_QUESTION, ask_review.NET_AS_GROSS_SQL)


@scenario
def b0_policy_denied():
    step = {"type": "tool_call", "name": "query_readonly", "arguments": {"sql": "SELECT x FROM secrets", "params": {}}}
    return _b0_steps("列出机密", step)


@scenario
def b0_bad_sql_is_failed_not_denied():
    return _b0_steps(GROSS_QUESTION, clarification_cases._query(clarification_cases.BAD_SQL))


@scenario
def b0_approval_required():
    return _b0_steps("列出客户姓名", json.loads(NON_METRIC_QUERY_EXAMPLE))


@scenario
def b0_ask_not_needed():
    return _b0_steps(GROSS_QUESTION, clarification_cases._ask())


@scenario
def b0_plain_ask():
    return _b0_steps("支付金额是多少", clarification_cases._ask("请问是哪一个月？"))


@scenario
def b0_deny():
    return _b0_steps("删除所有订单", {"type": "deny", "reason": "越权请求"})


@scenario
def b0_no_data():
    return _b0_steps("你好", NO_DATA)


@scenario
def b0_basis_conflict():
    return _b0_steps("你好", {**KNOWLEDGE, "fact_refs": [{"result_id": "r", "metric_id": "gross_fen"}]})


@scenario
def b0_ungrounded_answer():
    return _b0_steps(GROSS_QUESTION, PLAIN_ANSWER)


@scenario
def b0_answer_citing_a_missing_result():
    return _b0_steps(GROSS_QUESTION, FAKE_CITATION)


@scenario
def b0_known_usage():
    model = UsageModel([clarification_cases._query()], [KNOWN])
    return _b0(GROSS_QUESTION, model)


@scenario
def b0_inconsistent_usage():
    model = UsageModel([clarification_cases._query()], [INCONSISTENT])
    return _b0(GROSS_QUESTION, model)


# --- context bytes --------------------------------------------------------------------

CONTEXT = clarification_cases._context("run-context")
BINDING = ask_review._bound("gross_fen")[0].as_dict()
ITEMS = [
    {"id": "metric.net_fen", "text": "退款后净额", "source_id": "commerce-v1", "version": "v1"},
    {"id": "metric.gross_fen", "text": "支付订单总额", "source_id": "commerce-v1", "version": "v1"},
]
RESULTS = [
    {"tool_name": "search_catalog", "status": "succeeded", "output": {"items": ITEMS}},
    {"tool_name": "query_readonly", "status": "failed", "error_code": "invalid_sql", "repairable": True, "repair_hint": {"action": "fix"}},
]


def _built(**kwargs):
    result = build_context(CONTEXT, "2026年9月支付金额", **kwargs)
    return {"as_dict": result.as_dict()}


@scenario
def context_default_json():
    return _built()


@scenario
def context_native_protocol():
    return _built(run_config=NATIVE, tool_results=RESULTS)


@scenario
def context_without_retrieval_or_parallel():
    return _built(retrieval_available=False, parallel_available=False)


@scenario
def context_with_bindings_window_and_clarifications():
    return _built(
        clarifications=["只统计已支付"],
        confirmed_metric="gross_fen",
        time_window={**SEPTEMBER, "timezone": "UTC"},
        metric_bindings=[BINDING],
        request_time_window={**SEPTEMBER, "timezone": "UTC"},
        retrieval_items=ITEMS,
        tool_results=RESULTS,
        optional_summaries=["旧摘要"],
    )


@scenario
def context_trimmed_to_the_budget():
    big = [{"tool_name": "search_catalog", "status": "succeeded", "output": {"text": "字" * 3000 + str(i)}} for i in range(12)]
    return _built(tool_results=big)


# --- the run configuration ------------------------------------------------------------


@scenario
def run_config_serialization():
    custom = RunConfig(profile="p", skill_versions=("a", "b"), knowledge_snapshot_id="k", model_version="m", adapter_version="x")
    return {"default": DEFAULT_RUN_CONFIG.as_dict(), "native": NATIVE.as_dict(), "custom": custom.as_dict()}


def pinned(name: str) -> dict[str, str]:
    value = SCENARIOS[name]()
    return {"sha256": digest(value), "outcome": outcome(value)}


def pinned_values() -> dict[str, dict[str, str]]:
    return {name: pinned(name) for name in SCENARIOS}


if __name__ == "__main__":
    if sys.argv[1:] == ["--write"]:
        PINS_PATH.write_text(json.dumps(pinned_values(), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {len(SCENARIOS)} pins")
    elif len(sys.argv) == 3 and sys.argv[1] == "--show":
        print(canonical(SCENARIOS[sys.argv[2]]()))
    else:
        for name, pin in pinned_values().items():
            print(f"{name:60s} {pin['outcome']}")
