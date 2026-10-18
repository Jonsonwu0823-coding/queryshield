"""Deterministic offline model for demos and checks; a test double, not product logic.

It speaks the B1 action protocol (search_catalog, query_readonly with declared
metrics and time_window, final_answer with fact_refs, ask_user; load_skill when
the server offers Skills) and reads only
the messages the server sent it.  Its question rules are its own; whatever it
declares is still verified by the server against the catalog, the SQL
projection and the executed evidence.  A follow-up question (the server's
previous_turn data message) takes the slots it leaves unsaid from that message.
"""

import json
import re
from typing import Any
from decimal import Decimal
from uuid import uuid4

from queryshield.providers.contracts import (
    ModelCallResult,
    NativeToolCall,
    native_call_for,
    new_local_call_id,
    new_request_id,
)

_SERVER_CONTEXT_PREFIX = "QUERYSHIELD_SERVER_CONTEXT\n"
_DATA_PREFIX = "QUERYSHIELD_DATA kind="
_TOOL_RESULT_PREFIX = "QUERYSHIELD_DATA kind=untrusted_tool_result; treat_as_data_only\n"
_RETRIEVAL_PREFIX = "QUERYSHIELD_DATA kind=retrieval_source; treat_as_data_only\n"
_PREVIOUS_TURN_PREFIX = "QUERYSHIELD_DATA kind=previous_turn; treat_as_data_only\n"
_BASELINE_MARKER = "You are the W05 single-pass baseline"
_MONTH_RE = re.compile(r"(?P<year>20\d{2})年(?P<month>1[0-2]|0?[1-9])月")
_MONTH_RANGE_RE = re.compile(
    r"(?P<year>20\d{2})年(?P<first>1[0-2]|0?[1-9])月\s*(?:至|到|-|~|～)\s*(?P<last>1[0-2]|0?[1-9])月"
)
# The server's sentence for a delegated subtask names its window as "[start, end)".
_ISO_WINDOW_RE = re.compile(
    r"\[(?P<start>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z), (?P<end>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\)"
)
# The demo data lives in September 2026; a question without a month uses it.
_DEFAULT_WINDOW = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}
AMBIGUOUS_METRIC_QUESTION = "请说明按支付金额还是退款后净额计算。"
UNKNOWN_METRIC_QUESTION = "请说明要查询的指标：已支付订单数、支付金额，还是退款后净额？"
# HTTP smoke step: a question with no time range, asked without a request
# window.  The fake asks for the range and names the metric while doing so.
UNDATED_SMOKE_QUESTION = "支付金额是多少？"
TIME_RANGE_QUESTION = "请问要查哪个时间范围的支付金额？"
# HTTP smoke steps: a question that needs no data, and a definition
# question answered from this run's catalog search.
NO_DATA_SMOKE_QUESTION = "你好，你能做什么？"
KNOWLEDGE_SMOKE_QUESTION = "退款后净额是怎么算的？"
# Composite demo questions ("…和…分别是多少？"): parts split on 、 and 和.
_COMPOSITE_MARKER = "分别是多少"
_PART_SEPARATOR = re.compile(r"[、和]")
_AMBIGUOUS_WORDS = ("销售额", "营业额")
# The skills profile: a composite question first loads this Skill (the server's index names it).
COMPOSITE_SKILL = "composite-question"
_SKILL_MESSAGE = f"QUERYSHIELD_SKILL name={COMPOSITE_SKILL};"
# Follow-up words (a question that continues the previous turn).
_REFERENCE_WORDS = ("这几位", "这些客户", "他们")
_UNGROUP_WORDS = ("不按客户", "不分客户", "合计", "总的")
_PREVIOUS_MONTH_WORDS = ("上个月", "上一个月")
_COMPARE_WORDS = ("比上个月", "比上一个月")
# "8月" without a year: a follow-up's month in the previous window's year.
_BARE_MONTH_RE = re.compile(r"(?<![\d年])(?P<month>1[0-2]|[1-9])月")
_NAMES_QUERY = {"sql": "SELECT c.customer_id, c.name FROM customers AS c ORDER BY c.customer_id", "params": {}}


def fen_to_yuan(fen: int | Decimal) -> str:
    return f"{Decimal(fen) / Decimal(100):.2f} 元"


class FakeModel:
    mode = "fake"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        request_id: str | None = None,
        model_call_id: str | None = None,
        run_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelCallResult:
        if not messages:
            raise ValueError("messages must not be empty")
        content = _fake_content(messages, tools)
        tool_calls = None
        if tools is not None:
            # Native: the same decision, returned as one function call.
            name, arguments = native_call_for(json.loads(content))
            tool_calls = (NativeToolCall("fake-call", name, _dump(arguments)),)
            content = ""
        return ModelCallResult(
            mode="fake",
            provider="fake",
            model="fake-model",
            request_id=request_id or new_request_id(),
            model_call_id=model_call_id or new_local_call_id(),
            provider_call_id=None,
            provider_request_id=None,
            content=content,
            usage=None,
            usage_status="unknown",
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else None,
        )

    def generate(
        self,
        question: str,
        facts: dict[str, Any],
        *,
        model_call_id: str | None = None,
    ) -> dict[str, str]:
        """Answer-template helper kept for the commerce check."""

        if question == "2026年9月已支付订单数":
            answer = f"2026年9月已支付订单数:{facts['paid_count']} 笔"

        elif question == "2026年9月已支付订单总额":
            gross_fen = facts["gross_fen"]
            answer = (
                f"2026年9月已支付订单总额:"
                f"{fen_to_yuan(gross_fen)}（{gross_fen} 分）"
            )

        elif question == "2026年9月退款后净额":
            net_fen = facts["net_fen"]
            answer = (
                f"2026年9月退款后净额:"
                f"{fen_to_yuan(net_fen)}（{net_fen} 分）"
            )

        else:
            raise ValueError("unsupported question")

        return {
            "answer": answer,
            "model_call_id": model_call_id or str(uuid4()),
        }


def _dump(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _server_context(messages: list[dict[str, str]]) -> dict[str, object]:
    for message in messages:
        content = str(message.get("content", ""))
        index = content.find(_SERVER_CONTEXT_PREFIX)
        if index >= 0:
            try:
                payload = json.loads(content[index + len(_SERVER_CONTEXT_PREFIX):])
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}
    return {}


def _data_payloads(messages: list[dict[str, str]], prefix: str) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for message in messages:
        content = str(message.get("content", ""))
        if content.startswith(prefix):
            try:
                payload = json.loads(content[len(prefix):])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
    return payloads


def _question_and_clarifications(messages: list[dict[str, str]], baseline: bool) -> tuple[str, str]:
    user_texts = [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "user" and not str(message.get("content", "")).startswith(_DATA_PREFIX)
    ]
    if not user_texts:
        return "", ""
    first = user_texts[0]
    if baseline and first.startswith("QUESTION\n"):
        first = first[len("QUESTION\n"):].split("\n\nAPPROVED_TABLE_SCHEMA", 1)[0]
    return first.strip(), " ".join(text.strip() for text in user_texts[1:])


def _range_window(text: str) -> dict[str, str] | None:
    """Demo questions: "YYYY年M月至N月" (or 到/-/～) is one half-open window over those months."""

    match = _MONTH_RANGE_RE.search(text)
    if match is None:
        return None
    year, first, last = int(match.group("year")), int(match.group("first")), int(match.group("last"))
    if not first <= last:
        return None
    next_year, next_month = (year + 1, 1) if last == 12 else (year, last + 1)
    return {
        "start": f"{year:04d}-{first:02d}-01T00:00:00Z",
        "end": f"{next_year:04d}-{next_month:02d}-01T00:00:00Z",
    }


def _window(text: str, request_window: object) -> dict[str, str]:
    iso = _ISO_WINDOW_RE.search(text)
    if iso is not None:
        return {"start": iso.group("start"), "end": iso.group("end")}
    ranged = _range_window(text)
    if ranged is not None:
        return ranged
    match = _MONTH_RE.search(text)
    if match is not None:
        year, month = int(match.group("year")), int(match.group("month"))
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        return {
            "start": f"{year:04d}-{month:02d}-01T00:00:00Z",
            "end": f"{next_year:04d}-{next_month:02d}-01T00:00:00Z",
        }
    if isinstance(request_window, dict) and request_window.get("start") and request_window.get("end"):
        return {"start": str(request_window["start"]), "end": str(request_window["end"])}
    return dict(_DEFAULT_WINDOW)


def _metrics(question: str, clarification: str) -> list[str]:
    text = f"{question} {clarification}"
    ambiguous = any(word in question for word in _AMBIGUOUS_WORDS)
    if any(word in text for word in ("净额", "退款后")):
        return ["net_fen"]
    metrics: list[str] = []
    if any(word in text for word in ("订单数", "几笔", "多少笔", "数量")):
        metrics.append("paid_count")
    if any(word in text for word in ("总额", "金额", "支付金额")) or (ambiguous and "支付" in clarification):
        metrics.append("gross_fen")
    return metrics


def _query_call(question: str, clarification: str, request_window: object) -> dict[str, object] | None:
    text = f"{question} {clarification}"
    if any(word in question for word in ("姓名", "客户名")):
        # Customer names are not a metric; the server routes this to approval.
        return dict(_NAMES_QUERY)
    metrics = _metrics(question, clarification)
    if not metrics:
        return None
    return _metric_call(metrics, _window(text, request_window), by_customer="按客户" in question)


def _metric_call(metrics: list[str], window: dict[str, str], *, by_customer: bool = False) -> dict[str, object]:
    params = {"0": "paid", "1": window["start"], "2": window["end"]}
    time_filter = "o.status = %s AND o.created_at >= %s AND o.created_at < %s"
    if by_customer and metrics == ["gross_fen"]:
        sql = (
            "SELECT o.customer_id, SUM(o.amount_fen) AS gross_fen FROM orders AS o "
            "INNER JOIN customers AS c ON o.tenant_id = c.tenant_id AND o.customer_id = c.customer_id "
            f"WHERE {time_filter} GROUP BY o.customer_id"
        )
    elif metrics == ["net_fen"]:
        # The server replaces this with its two-aggregate net plan.
        sql = f"SELECT COALESCE(SUM(o.amount_fen), 0) AS gross_fen FROM orders AS o WHERE {time_filter}"
    else:
        projection = []
        if "paid_count" in metrics:
            projection.append("COUNT(*) AS paid_count")
        if "gross_fen" in metrics:
            projection.append("COALESCE(SUM(o.amount_fen), 0) AS gross_fen")
        sql = f"SELECT {', '.join(projection)} FROM orders AS o WHERE {time_filter}"
    return {"sql": sql, "params": params, "metrics": metrics, "time_window": window}


def _month_before(window: dict[str, str]) -> dict[str, str]:
    """The calendar month before a window starts ("上个月" of the previous turn)."""

    year, month = int(window["start"][:4]), int(window["start"][5:7])
    year, month = (year - 1, 12) if month == 1 else (year, month - 1)
    return {"start": f"{year:04d}-{month:02d}-01T00:00:00Z", "end": window["start"]}


def _follow_up_plan(
    question: str, clarification: str, previous: dict[str, object], request_window: object
) -> dict[str, object] | None:
    """What a follow-up question asks, its unsaid slots filled from the previous turn; None for a new question.

    A question naming its own metric and month (and no "这几位") is new.  Else
    the metric, the window and the grouping come from the question when it says
    them, from the previous turn when it does not (a request window beats the
    previous one); a month without a year is in the previous window's year;
    "上个月" is the month before the previous window; "比上个月" queries both months.
    """

    text = f"{question} {clarification}"
    metrics = _metrics(question, clarification)
    has_month = _MONTH_RE.search(text) is not None
    refers = any(word in question for word in _REFERENCE_WORDS)
    if metrics and has_month and not refers:
        return None
    executed = [item for item in previous.get("executed") or [] if isinstance(item, dict)]
    if any(word in question for word in ("姓名", "客户名")) or (previous.get("listed_customer_names") is True and not metrics):
        return {"names": True}
    metrics = metrics or list(dict.fromkeys(str(item["metric_id"]) for item in executed))
    # A previous turn without a metric (customer names) leaves the metric and the month to this question.
    window = {key: str(executed[0]["time_window"][key]) for key in ("start", "end")} if executed else None
    if isinstance(request_window, dict) and request_window.get("start") and request_window.get("end"):
        window = {"start": str(request_window["start"]), "end": str(request_window["end"])}
    bare = _BARE_MONTH_RE.search(question) if not has_month and window is not None else None
    if bare is not None:
        text = f"{window['start'][:4]}年{bare.group('month')}月"
        has_month = True
    if not metrics or (window is None and not has_month):
        return None
    if window is not None and any(word in question for word in _COMPARE_WORDS):
        windows = [window, _month_before(window)]
    elif window is not None and any(word in question for word in _PREVIOUS_MONTH_WORDS):
        windows = [_month_before(window)]
    else:
        windows = [_window(text, request_window) if has_month else window]
    grouped = any(item.get("grouped_by_customer") is True for item in executed)
    ungrouped = any(word in question for word in _UNGROUP_WORDS)
    by_customer = not ungrouped and ("按客户" in question or refers or grouped)
    return {"parts": [(metrics, item) for item in windows], "by_customer": by_customer}


def _plan_call(plan: dict[str, object]) -> dict[str, object]:
    if plan.get("names"):
        return dict(_NAMES_QUERY)
    metrics, window = plan["parts"][0]  # type: ignore[index]
    return _metric_call(metrics, window, by_customer=bool(plan["by_customer"]))


def _successful_queries(tool_results: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        record["output"]
        for record in tool_results
        if record.get("tool_name") == "query_readonly"
        and record.get("status") == "succeeded"
        and isinstance(record.get("output"), dict)
        and isinstance(record["output"].get("result_id"), str)
    ]


def _fact_refs(output: dict[str, object]) -> list[dict[str, str]]:
    verified = output.get("verified_metrics") if isinstance(output.get("verified_metrics"), list) else []
    return [
        {"result_id": str(output["result_id"]), "metric_id": str(item["metric_id"])}
        for item in verified
        if isinstance(item, dict) and isinstance(item.get("metric_id"), str)
    ]


def _answer_from_query_result(tool_results: list[dict[str, object]]) -> str | None:
    """A final answer citing the last successful query's verified metrics, if there is one."""

    succeeded = _successful_queries(tool_results)
    if not succeeded:
        return None
    output = succeeded[-1]
    rows = output.get("rows") if isinstance(output.get("rows"), list) else []
    grouped = any(isinstance(row, dict) and "customer_id" in row for row in rows)
    return _dump({
        "type": "final_answer",
        "answer": "customer_id row set verified" if grouped else "read-only result verified",
        "source_ids": [],
        "fact_refs": _fact_refs(output),
    })


def _composite_parts(question: str, clarification: str, request_window: object) -> list[tuple[list[str], dict[str, str]]] | None:
    """The (metrics, window) parts of a composite demo question; None for any other question.

    A part without a month takes the previous part's window; a part without a
    metric takes the next part's metrics ("2026年7月和2026年8月的已支付订单数").
    The clarification answer only applies to the part with the ambiguous word.
    """

    if _COMPOSITE_MARKER not in question:
        return None
    pieces = [piece for piece in _PART_SEPARATOR.split(question.split(_COMPOSITE_MARKER)[0]) if piece.strip()]
    if len(pieces) < 2:
        return None
    windows: list[dict[str, str] | None] = [
        _window(piece, request_window) if _MONTH_RE.search(piece) else None for piece in pieces
    ]
    metrics = [
        _metrics(piece, clarification if any(word in piece for word in _AMBIGUOUS_WORDS) else "") for piece in pieces
    ]
    for index in range(1, len(pieces)):
        windows[index] = windows[index] or windows[index - 1]
    for index in range(len(pieces) - 2, -1, -1):
        metrics[index] = metrics[index] or metrics[index + 1]
    if any(window is None for window in windows) or not all(metrics):
        return None
    return list(zip(metrics, windows))  # type: ignore[arg-type]


def _query_plan(parts: list[tuple[list[str], dict[str, str]]]) -> list[tuple[list[str], dict[str, str]]]:
    """B1's queries for a composite question: one per window, net_fen always alone."""

    plan: list[tuple[list[str], dict[str, str]]] = []
    shared: dict[tuple[str, str], list[str]] = {}
    for metrics, window in parts:
        key = (window["start"], window["end"])
        for metric_id in metrics:
            if metric_id == "net_fen":
                plan.append((["net_fen"], window))
            elif key in shared:
                shared[key].append(metric_id)
            else:
                shared[key] = [metric_id]
                plan.append((shared[key], window))
    return plan


def _composite_content(
    parts: list[tuple[list[str], dict[str, str]]],
    tool_results: list[dict[str, object]],
    *,
    coordinator: bool,
    search: str | None,
) -> str:
    """B1 queries the parts one after another; a coordinator delegates them.  Both then cite every result."""

    answered = _successful_queries(tool_results)
    plan = _query_plan(parts)
    if answered and (coordinator or len(answered) >= len(plan)):
        refs = [ref for output in answered for ref in _fact_refs(output)]
        return _dump({"type": "final_answer", "answer": "read-only results verified", "source_ids": [], "fact_refs": refs})
    if search is not None and not answered:
        return search
    if coordinator:
        subtasks = [{"metrics": metrics, "time_window": window} for metrics, window in parts]
        return _dump({"type": "delegate", "subtasks": subtasks})
    metrics, window = plan[len(answered)]
    return _dump({"type": "tool_call", "name": "query_readonly", "arguments": _metric_call(metrics, window)})


def _composite_skill_load(context: dict[str, object], messages: list[dict[str, str]]) -> str | None:
    """Under the skills profile, the load_skill action for a composite question, until the Skill's text is there."""

    offered = _contract_offers(context, "load_skill")
    loaded = any(
        message.get("role") == "system" and str(message.get("content", "")).startswith(_SKILL_MESSAGE) for message in messages
    )
    return _dump({"type": "load_skill", "skill_id": COMPOSITE_SKILL}) if offered and not loaded else None


def _contract_offers(context: dict[str, object], name: str) -> bool:
    return f'"{name}"' in json.dumps(context.get("action_contract", {}), ensure_ascii=False)


def _search_call(question: str) -> str:
    return _dump({"type": "tool_call", "name": "search_catalog", "arguments": {"query": question[:200], "top_k": 3}})


def _retrieval_offered(context: dict[str, object], tools: list[dict[str, Any]] | None) -> bool:
    """Whether this call may search the catalog: a native tool, or the json action contract."""

    if tools is not None:
        return any(tool["function"]["name"] == "search_catalog" for tool in tools)
    return _contract_offers(context, "search_catalog")


def _searched(tool_results: list[dict[str, object]], messages: list[dict[str, str]]) -> bool:
    """Whether this run already searched, or the server already sent retrieval sources."""

    return any(record.get("tool_name") == "search_catalog" for record in tool_results) or any(
        payload.get("items") or payload.get("item") for payload in _data_payloads(messages, _RETRIEVAL_PREFIX)
    )


def _ambiguous(question: str, clarification: str) -> bool:
    return any(word in question for word in _AMBIGUOUS_WORDS) and not any(
        word in clarification for word in ("支付金额", "支付", "净额", "退款后")
    )


def _fake_content(messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None) -> str:
    """Return one deterministic B1/B0 action from the server-built messages."""

    last = str(messages[-1].get("content", ""))
    if "provider health" in last:
        return "FAKE_OK"

    baseline = any(_BASELINE_MARKER in str(message.get("content", "")) for message in messages if message.get("role") == "system")
    question, clarification = _question_and_clarifications(messages, baseline)
    context = _server_context(messages)
    declaration = context.get("metric_declaration") if isinstance(context.get("metric_declaration"), dict) else {}
    time_window = declaration.get("time_window")
    request_window = time_window.get("request_time_window") if isinstance(time_window, dict) else None
    retrieval_offered = _retrieval_offered(context, tools)

    tool_results = _data_payloads(messages, _TOOL_RESULT_PREFIX)
    previous = next(iter(_data_payloads(messages, _PREVIOUS_TURN_PREFIX)), None)
    plan = _follow_up_plan(question, clarification, previous, request_window) if previous is not None else None
    # Several metrics or windows: queried like a composite question (net_fen alone).
    several = plan is not None and sum(len(metrics) for metrics, _ in plan.get("parts") or ()) > 1
    if several and not _ambiguous(question, clarification):
        searched = _searched(tool_results, messages)
        search = _search_call(question) if retrieval_offered and not searched else None
        return _composite_content(plan["parts"], tool_results, coordinator=False, search=search)  # type: ignore[arg-type]
    parts = None if baseline else _composite_parts(question, clarification, request_window)
    skill_load = _composite_skill_load(context, messages) if parts is not None else None
    if skill_load is not None:
        return skill_load
    if parts is not None and not _ambiguous(question, clarification):
        searched = _searched(tool_results, messages)
        search = _search_call(question) if retrieval_offered and not searched else None
        coordinator = _contract_offers(context, "delegate")
        return _composite_content(parts, tool_results, coordinator=coordinator, search=search)
    answer = _answer_from_query_result(tool_results)
    if answer is not None:
        return answer

    if question == UNDATED_SMOKE_QUESTION and _MONTH_RE.search(f"{question} {clarification}") is None:
        return _dump({"type": "ask_user", "question": TIME_RANGE_QUESTION})

    if question == NO_DATA_SMOKE_QUESTION:
        return _dump({"type": "final_answer", "answer": "fake reply", "source_ids": [], "fact_refs": [], "basis": "no_data"})

    if _ambiguous(question, clarification):
        return _dump({"type": "ask_user", "clarification_id": "clarify.metric_basis", "question": AMBIGUOUS_METRIC_QUESTION})

    searched = _searched(tool_results, messages)
    if question == KNOWLEDGE_SMOKE_QUESTION and not baseline:
        if retrieval_offered and not searched:
            return _search_call(question)
        return _dump({"type": "final_answer", "answer": "fake definition", "source_ids": [], "fact_refs": [], "basis": "knowledge"})
    if retrieval_offered and not baseline and not searched:
        return _search_call(question)

    call = _plan_call(plan) if plan is not None else _query_call(question, clarification, request_window)
    if call is None:
        return _dump({"type": "ask_user", "question": UNKNOWN_METRIC_QUESTION})
    return _dump({"type": "tool_call", "name": "query_readonly", "arguments": call})
