from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import math
from typing import Any, Literal, TypeAlias, get_args

from queryshield.policy.argument_limits import (
    ANSWER_MAX_CHARS,
    CLARIFICATION_ID_MAX_CHARS,
    DESCRIBE_TABLES_RANGE,
    LIST_ITEM_MAX_CHARS,
    MAX_FACT_REFS,
    MAX_SOURCE_IDS,
    QUESTION_MAX_CHARS,
    REASON_MAX_CHARS,
    SEARCH_QUERY_MAX_CHARS,
    SQL_MAX_CHARS,
    TOP_K_DEFAULT,
    TOP_K_RANGE,
)
from queryshield.policy.sql import ALLOWED_TABLES
from queryshield.providers.contracts import NativeToolCall, new_local_call_id, new_request_id


ToolName = Literal["search_catalog", "describe_tables", "query_readonly"]
CallAttemptKind = Literal["new", "transport_retry"]
PARALLEL_METRICS = frozenset({"paid_count", "gross_fen", "net_fen"})
# What a final_answer says it rests on; the server checks each claim.
# query: business values from this run's query; knowledge: a definition from
# this run's retrieval sources; no_data: needs no data (the server writes it).
ANSWER_BASES = ("query", "knowledge", "no_data")
# How the model wrote basis, for diagnostics: a fixed identifier, never its text.
BASIS_FIELD_STATES = ("declared", "null", "absent")

RESERVED_IDENTITY_PARAMS = frozenset(
    {"tenant_id", "principal_id", "role", "authorization", "token"}
)


class ProposalParseError(ValueError):
    """A model proposal failed before any tool or database code was called."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


TOOL_NAMES: tuple[str, ...] = get_args(ToolName)
# The functions offered under native function calling (no parallel_readonly).
NATIVE_FUNCTION_NAMES = TOOL_NAMES + ("ask_user", "final_answer", "deny")
_ACTION_TYPES = ("tool_call", "parallel_readonly", "ask_user", "final_answer", "deny")
# Every field any action may carry; each action type then allows its own subset.
_TOP_LEVEL_FIELDS = frozenset(
    {"type", "name", "arguments", "question", "clarification_id", "answer", "source_ids", "fact_refs", "reason", "metric_ids", "basis"}
)
_QUERY_ARGUMENT_FIELDS = ("sql", "params", "metrics", "time_window")


class ToolNameAsActionTypeError(ProposalParseError):
    """The model put a tool name in ``type`` instead of ``type: tool_call``.

    ``tool_name`` is always one of the server's own TOOL_NAMES, never model text.
    """

    def __init__(self, tool_name: str) -> None:
        if tool_name not in TOOL_NAMES:
            raise ValueError("tool_name must be a server tool name")
        self.tool_name = tool_name
        super().__init__(
            "tool_name_as_action_type",
            "a tool name was used as the action type; use type tool_call and put the tool in name",
        )


class CallIdentityError(LookupError):
    """A call identity was not present in the current run context."""


class _DuplicateKeyError(ValueError):
    pass


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(value)


def _parse_finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):  # an overflowing literal such as 1e400
        raise ValueError(value)
    return number


def _load_json(raw_content: str) -> dict[str, object]:
    if type(raw_content) is not str:
        raise ProposalParseError("invalid_content", "model content must be a string")
    if not raw_content.strip():
        raise ProposalParseError("invalid_json", "model content is empty")

    try:
        decoded = json.loads(
            raw_content,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_float,
        )
    except _DuplicateKeyError as exc:
        raise ProposalParseError("duplicate_field", "duplicate JSON field") from exc
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProposalParseError("invalid_json", "model content is not strict JSON") from exc

    if type(decoded) is not dict:
        raise ProposalParseError("invalid_shape", "proposal must be a JSON object")
    return decoded


def _require_fields(
    payload: Mapping[str, object],
    *,
    required: set[str],
    allowed: set[str] | frozenset[str],
) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise ProposalParseError(
            "missing_field",
            f"required proposal field is missing: {', '.join(missing)}",
        )
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ProposalParseError(
            "unknown_field",
            f"proposal contains unknown field(s): {', '.join(unknown)}",
        )


def _string(
    value: object,
    *,
    field: str,
    nonempty: bool = True,
    max_length: int | None = None,
) -> str:
    if type(value) is not str:
        raise ProposalParseError("invalid_field", f"{field} must be a string")
    if nonempty and not value.strip():
        raise ProposalParseError("invalid_field", f"{field} must not be blank")
    if max_length is not None and len(value) > max_length:
        raise ProposalParseError("invalid_field", f"{field} is too long")
    return value


def _strict_int(value: object, *, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ProposalParseError("invalid_field", f"{field} must be an integer in range")
    return value


def _string_list(
    value: object,
    *,
    field: str,
    minimum: int,
    maximum: int | None = None,
    unique: bool = False,
) -> tuple[str, ...]:
    if type(value) is not list:
        raise ProposalParseError("invalid_field", f"{field} must be a list")
    if not minimum <= len(value) or (maximum is not None and len(value) > maximum):
        raise ProposalParseError("invalid_field", f"{field} has an invalid length")

    items = tuple(
        _string(item, field=f"{field}[]", max_length=LIST_ITEM_MAX_CHARS) for item in value
    )
    if unique and len(set(items)) != len(items):
        raise ProposalParseError("invalid_field", f"{field} must not contain duplicates")
    return items


@dataclass(frozen=True)
class FactRef:
    result_id: str
    metric_id: str

    def __post_init__(self) -> None:
        _check_server_id(self.result_id, "result_id")
        _check_server_id(self.metric_id, "metric_id")

    def as_dict(self) -> dict[str, str]:
        return {"result_id": self.result_id, "metric_id": self.metric_id}


@dataclass(frozen=True)
class ToolCallAction:
    name: ToolName
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.name not in TOOL_NAMES:
            raise ValueError("unsupported tool name")
        object.__setattr__(self, "arguments", dict(self.arguments))

    def as_dict(self) -> dict[str, object]:
        return {"type": "tool_call", "name": self.name, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class ParallelReadonlyAction:
    """Strict read-only parallel action; plan details stay server-owned."""

    metric_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.metric_ids) is not tuple or not 2 <= len(self.metric_ids) <= 3:
            raise ValueError("parallel_readonly requires two or three metrics")
        if any(type(metric_id) is not str or not metric_id.strip() for metric_id in self.metric_ids):
            raise ValueError("parallel metric IDs must be non-empty strings")
        if len(set(self.metric_ids)) != len(self.metric_ids):
            raise ValueError("parallel metric IDs must be unique")
        if any(metric_id not in PARALLEL_METRICS for metric_id in self.metric_ids):
            raise ValueError("parallel metric ID is not allowed")

    def as_dict(self) -> dict[str, object]:
        return {"type": "parallel_readonly", "metric_ids": list(self.metric_ids)}


# A coordinator's delegate action splits a question into this many subtasks.
MIN_SUBTASKS = 2
MAX_SUBTASKS = 3
_SUBTASK_FIELDS = frozenset({"metrics", "time_window"})


@dataclass(frozen=True)
class DelegateAction:
    """A coordinator's split of the question: each subtask is only a metric declaration.

    Each item holds exactly ``metrics`` and ``time_window``; their content is
    checked like a query_readonly declaration before any subtask runs.
    """

    subtasks: tuple[Mapping[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return {"type": "delegate", "subtasks": [dict(item) for item in self.subtasks]}


@dataclass(frozen=True)
class AskUserAction:
    question: str
    # Optional catalog clarification rule id; the server still identifies the
    # rule from the question text, so omitting it does not skip the check.
    clarification_id: str | None = None

    def as_dict(self) -> dict[str, object]:
        action: dict[str, object] = {"type": "ask_user", "question": self.question}
        if self.clarification_id is not None:
            action["clarification_id"] = self.clarification_id
        return action


@dataclass(frozen=True)
class FinalAnswerAction:
    answer: str
    source_ids: tuple[str, ...]
    fact_refs: tuple[FactRef, ...]
    basis: str = "query"
    # Set by the parser only; diagnostics, not part of the public action.
    basis_field: str = field(default="absent", compare=False)

    def __post_init__(self) -> None:
        if self.basis not in ANSWER_BASES:
            raise ValueError("unsupported answer basis")
        if self.basis_field not in BASIS_FIELD_STATES:
            raise ValueError("unsupported basis field state")

    def as_dict(self) -> dict[str, object]:
        return {
            "type": "final_answer",
            "answer": self.answer,
            "source_ids": list(self.source_ids),
            "fact_refs": [fact_ref.as_dict() for fact_ref in self.fact_refs],
            "basis": self.basis,
        }


@dataclass(frozen=True)
class DenyAction:
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {"type": "deny", "reason": self.reason}


ProposalAction: TypeAlias = (
    ToolCallAction | ParallelReadonlyAction | AskUserAction | FinalAnswerAction | DenyAction | DelegateAction
)


@dataclass(frozen=True)
class ExecutionContext:
    """Identity and authorization context constructed by the server."""

    run_id: str
    tenant_id: str
    principal_id: str
    role: str

    def __post_init__(self) -> None:
        _check_server_id(self.run_id, "run_id")
        _check_server_id(self.tenant_id, "tenant_id")
        _check_server_id(self.principal_id, "principal_id")
        _check_server_id(self.role, "role")


@dataclass(frozen=True)
class QueryProposal:
    """Validated action plus server-owned call context; raw model text is not retained."""

    context: ExecutionContext
    model_call_id: str
    action: ProposalAction
    content_sha256: str

    def __post_init__(self) -> None:
        _check_server_id(self.model_call_id, "model_call_id")
        if len(self.content_sha256) != 64:
            raise ValueError("content_sha256 must be a SHA-256 hex digest")

    @property
    def run_id(self) -> str:
        return self.context.run_id

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.context.run_id,
            "tenant_id": self.context.tenant_id,
            "principal_id": self.context.principal_id,
            "role": self.context.role,
            "model_call_id": self.model_call_id,
            "action": self.action.as_dict(),
            "content_sha256": self.content_sha256,
        }


def parse_query_proposal(
    raw_content: str,
    *,
    context: ExecutionContext,
    model_call_id: str,
    delegate: bool = False,
) -> QueryProposal:
    """Parse one provider response without executing any model-selected action.

    ``delegate`` is set only for a coordinator; everyone else parses exactly as before.
    """
    if not isinstance(context, ExecutionContext):
        raise TypeError("context must be an ExecutionContext")
    _check_server_id(model_call_id, "model_call_id")
    payload = _load_json(raw_content)
    action = _parse_delegate(payload) if delegate and payload.get("type") == "delegate" else _parse_action(payload)
    return QueryProposal(
        context=context,
        model_call_id=model_call_id,
        action=action,
        content_sha256=sha256(raw_content.encode("utf-8")).hexdigest(),
    )


def native_action_text(tool_calls: Sequence[NativeToolCall]) -> str:
    """The json action that one native function call stands for.

    The text goes to parse_query_proposal, so both protocols share one
    validator.  The function name alone decides the action type.
    """
    if not tool_calls:
        raise ProposalParseError("native_tool_call_missing", "the model returned no function call")
    if len(tool_calls) > 1:
        raise ProposalParseError("native_multiple_tool_calls", "the model returned more than one function call")
    call = tool_calls[0]
    if call.name not in NATIVE_FUNCTION_NAMES:
        raise ProposalParseError("unknown_action", "the function is not offered")
    arguments = _load_json(call.arguments)
    if call.name in TOOL_NAMES:
        payload = {"type": "tool_call", "name": call.name, "arguments": arguments}
    elif "type" in arguments:
        raise ProposalParseError("unknown_field", "function arguments cannot set the action type")
    else:
        payload = {"type": call.name, **arguments}
    # ASCII escapes keep the text encodable even for a lone surrogate the model escaped,
    # which json mode accepts too.
    return json.dumps(payload, separators=(",", ":"))


def native_call_summary(tool_calls: Sequence[NativeToolCall]) -> list[dict[str, object]]:
    """Each call's offered function name (else <other>) and its arguments' size and digest, never the text."""
    return [
        {
            "name": _known_or_other(call.name, NATIVE_FUNCTION_NAMES),
            "arguments_length": len(call.arguments.encode("utf-8")),
            "arguments_sha256": sha256(call.arguments.encode("utf-8")).hexdigest(),
        }
        for call in tool_calls
    ]


def _parse_action(payload: Mapping[str, object]) -> ProposalAction:
    _require_fields(payload, required={"type"}, allowed=_TOP_LEVEL_FIELDS)
    action_type = _string(payload["type"], field="type")
    if action_type == "tool_call":
        return _parse_tool_call(payload)
    if action_type == "parallel_readonly":
        return _parse_parallel_readonly(payload)
    if action_type == "ask_user":
        return _parse_ask_user(payload)
    if action_type == "final_answer":
        return _parse_final_answer(payload)
    if action_type == "deny":
        return _parse_deny(payload)
    if action_type in TOOL_NAMES:
        raise ToolNameAsActionTypeError(action_type)
    raise ProposalParseError("unknown_action", "proposal action is not supported")


def _parse_ask_user(payload: Mapping[str, object]) -> AskUserAction:
    _require_fields(payload, required={"type", "question"}, allowed={"type", "question", "clarification_id"})
    clarification_id = payload.get("clarification_id")
    return AskUserAction(
        _string(payload["question"], field="question", max_length=QUESTION_MAX_CHARS),
        None if clarification_id is None else _string(clarification_id, field="clarification_id", max_length=CLARIFICATION_ID_MAX_CHARS),
    )


def _parse_final_answer(payload: Mapping[str, object]) -> FinalAnswerAction:
    _require_fields(
        payload,
        required={"type", "answer", "source_ids", "fact_refs"},
        allowed={"type", "answer", "source_ids", "fact_refs", "basis"},
    )
    fact_refs = _parse_fact_refs(payload["fact_refs"])
    # Accepted relaxation: an explicit null basis is the default (query).
    basis = payload.get("basis")
    basis_field = "declared" if basis is not None else "null" if "basis" in payload else "absent"
    if basis is None:
        basis = "query"
    elif type(basis) is not str or basis not in ANSWER_BASES:
        raise ProposalParseError("invalid_field", "basis must be query, knowledge or no_data")
    return FinalAnswerAction(
        # A no_data answer may be blank: the server writes that answer itself.
        answer=_string(payload["answer"], field="answer", nonempty=basis != "no_data", max_length=ANSWER_MAX_CHARS),
        source_ids=_string_list(
            payload["source_ids"],
            field="source_ids",
            minimum=0,
            maximum=MAX_SOURCE_IDS,
            unique=True,
        ),
        fact_refs=fact_refs,
        basis=basis,
        basis_field=basis_field,
    )


def _parse_deny(payload: Mapping[str, object]) -> DenyAction:
    _require_fields(payload, required={"type", "reason"}, allowed={"type", "reason"})
    return DenyAction(_string(payload["reason"], field="reason", max_length=REASON_MAX_CHARS))


def _parse_tool_call(payload: Mapping[str, object]) -> ToolCallAction:
    _require_fields(
        payload,
        required={"type", "name", "arguments"},
        allowed={"type", "name", "arguments"},
    )
    name = _string(payload["name"], field="name")
    if name not in TOOL_NAMES:
        raise ProposalParseError("unknown_action", "tool name is not supported")
    arguments = payload["arguments"]
    if type(arguments) is not dict:
        raise ProposalParseError("invalid_field", "arguments must be an object")
    if name == "search_catalog":
        normalized = _search_catalog_arguments(arguments)
    elif name == "describe_tables":
        normalized = _describe_tables_arguments(arguments)
    else:
        normalized = _query_readonly_arguments(arguments)
    return ToolCallAction(name=name, arguments=normalized)  # type: ignore[arg-type]


def _search_catalog_arguments(arguments: dict[str, object]) -> dict[str, object]:
    _require_fields(
        arguments,
        required={"query"},
        allowed={"query", "top_k"},
    )
    top_k = (
        _strict_int(arguments["top_k"], field="top_k", minimum=TOP_K_RANGE[0], maximum=TOP_K_RANGE[1])
        if "top_k" in arguments
        else TOP_K_DEFAULT
    )
    return {
        "query": _string(arguments["query"], field="query", max_length=SEARCH_QUERY_MAX_CHARS),
        "top_k": top_k,
    }


def _describe_tables_arguments(arguments: dict[str, object]) -> dict[str, object]:
    _require_fields(arguments, required={"tables"}, allowed={"tables"})
    tables = _string_list(
        arguments["tables"], field="tables", minimum=DESCRIBE_TABLES_RANGE[0], maximum=DESCRIBE_TABLES_RANGE[1], unique=True
    )
    if any(table not in ALLOWED_TABLES for table in tables):
        raise ProposalParseError("invalid_field", "tables must be in the server allowlist")
    return {"tables": list(tables)}


def _query_readonly_arguments(arguments: dict[str, object]) -> dict[str, object]:
    _require_fields(
        arguments,
        required={"sql", "params"},
        allowed={"sql", "params", "metrics", "time_window"},
    )
    sql = _string(arguments["sql"], field="sql", max_length=SQL_MAX_CHARS)
    params = arguments["params"]
    # Accepted relaxation: null or an empty array means "no parameters",
    # exactly like {}.  A non-empty array is still rejected.
    if params is None or (type(params) is list and not params):
        params = {}
    if type(params) is not dict:
        raise ProposalParseError("invalid_field", "params must be an object")
    for key, value in params.items():
        if type(key) is not str:
            raise ProposalParseError("invalid_field", "params keys must be strings")
        if key in RESERVED_IDENTITY_PARAMS:
            raise ProposalParseError("reserved_parameter", "identity parameters are server-owned")
        if value is not None and type(value) not in {str, int, float, bool}:
            raise ProposalParseError("invalid_field", "params values must be scalar")
    normalized: dict[str, object] = {"sql": sql, "params": dict(params)}
    # Only the JSON shape is checked here.  The server validates the
    # declared metric ids and window against the catalog at tool time, so
    # a wrong declaration stays inside the bounded repair budget.
    # Accepted relaxations: an explicit null or empty metrics declaration
    # is the same as omitting it (no metrics are declared, so no fact can
    # come from it).  A non-empty list is still checked at tool time.
    metrics = arguments.get("metrics")
    if metrics is not None and not (type(metrics) is list and not metrics):
        if type(metrics) is not list:
            raise ProposalParseError("invalid_field", "metrics must be an array")
        normalized["metrics"] = list(metrics)
    if arguments.get("time_window") is not None:
        if type(arguments["time_window"]) is not dict:
            raise ProposalParseError("invalid_field", "time_window must be an object")
        normalized["time_window"] = dict(arguments["time_window"])
    return normalized


def _json_type(value: object) -> str:
    return {
        dict: "object",
        list: "array",
        str: "string",
        bool: "boolean",
        int: "number",
        float: "number",
        type(None): "null",
    }.get(type(value), "other")


def _known_or_other(value: object, known: tuple[str, ...]) -> str:
    if value is ...:
        return "absent"
    if type(value) is str:
        return value if value in known else "<other>"
    return _json_type(value)


def proposal_shape_summary(raw_content: object) -> dict[str, object]:
    """Structure of a rejected proposal for diagnosis, without any model value.

    Only server-known type/name values are echoed (anything else is
    ``<other>``); argument fields are reported as present/absent and JSON
    type; unexpected fields are counted, never named.
    """

    try:
        payload = json.loads(raw_content) if type(raw_content) is str else None
    except (TypeError, ValueError):
        return {"json": "invalid"}
    if type(payload) is not dict:
        return {"json": "valid", "top_level": _json_type(payload)}
    summary: dict[str, object] = {
        "json": "valid",
        "type": _known_or_other(payload.get("type", ...), _ACTION_TYPES + TOOL_NAMES),
        "name": _known_or_other(payload.get("name", ...), TOOL_NAMES),
        "top_level_extra_field_count": sum(1 for key in payload if key not in _TOP_LEVEL_FIELDS),
        # The basis field's JSON type and whether it is a server-known
        # value; the value itself is never echoed.
        "basis": _json_type(payload["basis"]) if "basis" in payload else "absent",
    }
    if type(payload.get("basis")) is str:
        summary["basis_known"] = payload["basis"] in ANSWER_BASES
    arguments = payload.get("arguments", ...)
    if type(arguments) is dict:
        summary["arguments"] = {
            field: _json_type(arguments[field]) if field in arguments else "absent"
            for field in _QUERY_ARGUMENT_FIELDS
        }
        summary["arguments_extra_field_count"] = sum(1 for key in arguments if key not in _QUERY_ARGUMENT_FIELDS)
    else:
        summary["arguments"] = "absent" if arguments is ... else _json_type(arguments)
    return summary


def parse_error_detail(error: ProposalParseError) -> str:
    """The server's own fixed explanation of a parse failure.

    Messages that would list model-chosen field names are reduced to their
    fixed prefix.
    """

    if error.code == "unknown_field":
        return "proposal contains unknown field(s)"
    text = str(error)
    prefix = f"{error.code}: "
    return text[len(prefix):] if text.startswith(prefix) else error.code


def _parse_parallel_readonly(payload: Mapping[str, object]) -> ParallelReadonlyAction:
    _require_fields(
        payload,
        required={"type", "metric_ids"},
        allowed={"type", "metric_ids"},
    )
    metric_ids = _string_list(
        payload["metric_ids"],
        field="metric_ids",
        minimum=2,
        maximum=3,
        unique=True,
    )
    if any(metric_id not in PARALLEL_METRICS for metric_id in metric_ids):
        raise ProposalParseError("invalid_field", "metric_ids contains an unsupported metric")
    return ParallelReadonlyAction(metric_ids=metric_ids)


def _parse_delegate(payload: Mapping[str, object]) -> DelegateAction:
    _require_fields(payload, required={"type", "subtasks"}, allowed={"type", "subtasks"})
    subtasks = payload["subtasks"]
    if type(subtasks) is not list or not MIN_SUBTASKS <= len(subtasks) <= MAX_SUBTASKS:
        raise ProposalParseError("invalid_field", f"subtasks must list {MIN_SUBTASKS} to {MAX_SUBTASKS} items")
    items: list[dict[str, object]] = []
    for item in subtasks:
        if type(item) is not dict:
            raise ProposalParseError("invalid_field", "each subtask must be an object")
        _require_fields(item, required=set(_SUBTASK_FIELDS), allowed=_SUBTASK_FIELDS)
        if type(item["metrics"]) is not list:
            raise ProposalParseError("invalid_field", "subtask metrics must be an array")
        if type(item["time_window"]) is not dict:
            raise ProposalParseError("invalid_field", "subtask time_window must be an object")
        items.append({"metrics": list(item["metrics"]), "time_window": dict(item["time_window"])})
    return DelegateAction(tuple(items))


def _parse_fact_refs(value: object) -> tuple[FactRef, ...]:
    if type(value) is not list:
        raise ProposalParseError("invalid_field", "fact_refs must be a list")
    if len(value) > MAX_FACT_REFS:
        raise ProposalParseError("invalid_field", "fact_refs has too many items")

    parsed: list[FactRef] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        if type(item) is not dict:
            raise ProposalParseError("invalid_field", "fact_refs items must be objects")
        _require_fields(item, required={"result_id", "metric_id"}, allowed={"result_id", "metric_id"})
        result_id = _string(item["result_id"], field="fact_refs.result_id")
        metric_id = _string(item["metric_id"], field="fact_refs.metric_id")
        key = (result_id, metric_id)
        if key in seen:
            raise ProposalParseError("invalid_field", "fact_refs must be deduplicated")
        seen.add(key)
        parsed.append(FactRef(result_id=result_id, metric_id=metric_id))
    return tuple(parsed)


@dataclass(frozen=True)
class ModelCallIdentity:
    run_id: str
    model_call_id: str
    request_id: str
    attempt_kind: CallAttemptKind
    retry_of_model_call_id: str | None = None

    def __post_init__(self) -> None:
        _check_server_id(self.run_id, "run_id")
        _check_server_id(self.model_call_id, "model_call_id")
        _check_server_id(self.request_id, "request_id")
        if self.attempt_kind == "new" and self.retry_of_model_call_id is not None:
            raise ValueError("new calls cannot point to a retry parent")
        if self.attempt_kind == "transport_retry":
            if self.retry_of_model_call_id != self.model_call_id:
                raise ValueError("transport retries must keep the logical call id")


def new_call_identity(run_id: str, request_id: str | None) -> ModelCallIdentity:
    """The identity of a new logical model call; its id is server-made, never the provider's."""

    return ModelCallIdentity(
        run_id=run_id,
        model_call_id=new_local_call_id(),
        request_id=request_id or new_request_id(),
        attempt_kind="new",
    )


def retry_identity(canonical: ModelCallIdentity, request_id: str | None) -> ModelCallIdentity:
    """A transport retry of ``canonical``: same logical call, new request id."""

    return ModelCallIdentity(
        run_id=canonical.run_id,
        model_call_id=canonical.model_call_id,
        request_id=request_id or new_request_id(),
        attempt_kind="transport_retry",
        retry_of_model_call_id=canonical.model_call_id,
    )


class ModelCallStore:
    """Run-scoped in-memory identity store; agent.call_store.DurableModelCallStore persists the same interface."""

    def __init__(self) -> None:
        self._calls: dict[tuple[str, str], ModelCallIdentity] = {}
        self._attempts: dict[tuple[str, str], list[ModelCallIdentity]] = {}

    def new_call(self, run_id: str, *, request_id: str | None = None) -> ModelCallIdentity:
        identity = new_call_identity(run_id, request_id)
        key = (identity.run_id, identity.model_call_id)
        self._calls[key] = identity
        self._attempts[key] = [identity]
        return identity

    def transport_retry(
        self,
        identity: ModelCallIdentity,
        *,
        request_id: str | None = None,
    ) -> ModelCallIdentity:
        canonical = self.get(identity.run_id, identity.model_call_id)
        retry = retry_identity(canonical, request_id)
        self._attempts[(canonical.run_id, canonical.model_call_id)].append(retry)
        return retry

    def get(self, run_id: str, model_call_id: str) -> ModelCallIdentity:
        identity = self._calls.get((run_id, model_call_id))
        if identity is None:
            raise CallIdentityError("model call identity was not found")
        return identity

    def attempts(self, run_id: str, model_call_id: str) -> tuple[ModelCallIdentity, ...]:
        self.get(run_id, model_call_id)
        return tuple(self._attempts[(run_id, model_call_id)])


@dataclass(frozen=True)
class MetricBinding:
    metric_id: str
    result_position: str | int
    unit: str
    time_window: Mapping[str, str]
    catalog_source_id: str
    catalog_version: str
    plan_id: str | None = None
    # Set by the server when the bound query has a GROUP BY.  Its rows are
    # per-group values (a rowset), never the tenant-wide metric, even when
    # exactly one row comes back (e.g. ORDER BY ... LIMIT 1).
    grouped: bool = False

    def __post_init__(self) -> None:
        _check_server_id(self.metric_id, "metric_id")
        if type(self.grouped) is not bool:
            raise ValueError("grouped must be a boolean")
        if type(self.result_position) not in {str, int}:
            raise ValueError("result_position must be a string or integer")
        _check_server_id(self.unit, "unit")
        _check_server_id(self.catalog_source_id, "catalog_source_id")
        _check_server_id(self.catalog_version, "catalog_version")
        if self.plan_id is not None:
            _check_server_id(self.plan_id, "plan_id")
        if set(self.time_window) != {"start", "end", "timezone"}:
            raise ValueError("time_window must contain start, end and timezone")
        if any(type(value) is not str or not value.strip() for value in self.time_window.values()):
            raise ValueError("time_window values must be non-empty strings")
        object.__setattr__(self, "time_window", dict(self.time_window))

    def as_dict(self) -> dict[str, object]:
        record: dict[str, object] = {
            "metric_id": self.metric_id,
            "result_position": self.result_position,
            "unit": self.unit,
            "time_window": dict(self.time_window),
            "catalog_source_id": self.catalog_source_id,
            "catalog_version": self.catalog_version,
            "plan_id": self.plan_id,
        }
        # Only grouped bindings carry the key, so stored records written
        # without it (every ungrouped binding) read unchanged.
        if self.grouped:
            record["grouped"] = True
        return record


@dataclass(frozen=True)
class ResultEvidence:
    """Server-created evidence binding actual results to one authenticated run."""

    result_id: str
    run_id: str
    tenant_id: str
    principal_id: str
    rows: tuple[Mapping[str, object], ...]
    row_count: int
    query_sha256: str
    params_sha256: str
    observed_at: datetime
    policy_version: str
    catalog_version: str
    metric_bindings: tuple[MetricBinding, ...]
    metric_plan_id: str | None = None

    @classmethod
    def from_server_execution(
        cls,
        context: ExecutionContext,
        *,
        result_id: str,
        rows: Sequence[Mapping[str, object]],
        normalized_query: str,
        params: object,
        observed_at: datetime,
        policy_version: str,
        catalog_version: str,
        metric_bindings: Sequence[MetricBinding],
        metric_plan_id: str | None = None,
    ) -> ResultEvidence:
        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")
        _check_server_id(result_id, "result_id")
        _check_server_id(normalized_query, "normalized_query")
        _check_server_id(policy_version, "policy_version")
        _check_server_id(catalog_version, "catalog_version")
        if not isinstance(observed_at, datetime) or observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError("rows must be a sequence of mappings")

        copied_rows: list[Mapping[str, object]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("each row must be a mapping")
            copied_rows.append(dict(row))

        copied_bindings = tuple(metric_bindings)
        if any(not isinstance(binding, MetricBinding) for binding in copied_bindings):
            raise ValueError("metric_bindings must contain MetricBinding values")
        if metric_plan_id is not None:
            _check_server_id(metric_plan_id, "metric_plan_id")

        return cls(
            result_id=result_id,
            run_id=context.run_id,
            tenant_id=context.tenant_id,
            principal_id=context.principal_id,
            rows=tuple(copied_rows),
            row_count=len(copied_rows),
            query_sha256=sha256(normalized_query.encode("utf-8")).hexdigest(),
            params_sha256=_sha256_json(params),
            observed_at=observed_at.astimezone(timezone.utc),
            policy_version=policy_version,
            catalog_version=catalog_version,
            metric_bindings=copied_bindings,
            metric_plan_id=metric_plan_id,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "result_id": self.result_id,
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "principal_id": self.principal_id,
            "rows": [dict(row) for row in self.rows],
            "row_count": self.row_count,
            "query_sha256": self.query_sha256,
            "params_sha256": self.params_sha256,
            "observed_at": self.observed_at.isoformat().replace("+00:00", "Z"),
            "policy_version": self.policy_version,
            "catalog_version": self.catalog_version,
            "metric_bindings": [binding.as_dict() for binding in self.metric_bindings],
            "metric_plan_id": self.metric_plan_id,
        }


def _sha256_json(value: object) -> str:
    try:
        encoded = json.dumps(
            _canonicalize_for_digest(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value cannot be serialized for a digest") from exc
    return sha256(encoded).hexdigest()


def _canonicalize_for_digest(value: object) -> object:
    if isinstance(value, datetime):
        normalized = value.astimezone(timezone.utc) if value.tzinfo is not None else value
        return normalized.isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("digest mappings require string keys")
        return {key: _canonicalize_for_digest(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_for_digest(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise ValueError("value cannot be canonicalized for a digest")


def _check_server_id(value: object, field: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
