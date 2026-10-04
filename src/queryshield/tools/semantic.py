from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import re
from typing import Any
from uuid import uuid4

from queryshield.agent.context import (
    NET_FEN_GROSS_QUERY,
    NET_FEN_PLAN_ID,
    NET_FEN_REFUND_QUERY,
)
from queryshield.agent.proposals import ExecutionContext, MetricBinding, ResultEvidence
from queryshield.catalog import SemanticCatalog, load_default_catalog
from queryshield.catalog.catalog import ALLOWED_TABLE_COLUMNS
from queryshield.db.guarded import GuardedQueryError, GuardedQueryExecutor, render_scoped_select
from queryshield.policy.sql import (
    BinaryExpression,
    BooleanExpression,
    ColumnRef,
    Comparison,
    Condition,
    FunctionCall,
    NotExpression,
    ParameterRef,
    SQLPolicyError,
    SelectStatement,
    Star,
    UnaryExpression,
    parse_readonly_select,
)


_ALLOWED_ROLES = frozenset({"requester", "approver"})
_RESERVED_PARAMS = frozenset(
    {"tenant_id", "principal_id", "role", "authorization", "token"}
)
_SCALAR_TYPES = (str, int, float, bool)
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+")
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "已支付订单数": ("paid_count", "paid"),
    "订单数量": ("paid_count",),
    "营业额": ("gross_fen",),
    "支付订单总额": ("gross_fen",),
    "销售额": ("gross_fen", "net_fen"),
    "退款": ("refund_fen",),
    "退款金额": ("refund_fen",),
    "净额": ("net_fen",),
    "退款后": ("net_fen",),
}


class ToolError(ValueError):
    """A controlled tool rejected its input or the current authorization."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _controlled_error_detail(error: ValueError, code: str) -> str:
    """Expose only the fixed server error detail, never SQL or parameter values."""

    prefix = f"{code}: "
    text = str(error)
    return text[len(prefix) :] if text.startswith(prefix) else text


def _require_context(context: ExecutionContext) -> None:
    if not isinstance(context, ExecutionContext):
        raise ToolError("unauthorized", "tool calls require a server-created execution context")
    if context.role not in _ALLOWED_ROLES:
        raise ToolError("forbidden", "the current role cannot use semantic tools")
    if not context.tenant_id.strip() or not context.principal_id.strip():
        raise ToolError("unauthorized", "execution context is incomplete")


def _arguments_object(arguments: Mapping[str, object], *, required: set[str], optional: set[str]) -> None:
    if not isinstance(arguments, Mapping):
        raise ToolError("invalid_arguments", "tool arguments must be an object")
    keys = set(arguments)
    if not required <= keys:
        raise ToolError("missing_argument", "a required tool argument is missing")
    if keys - required - optional:
        raise ToolError("unknown_argument", "tool arguments contain an unknown field")


def _string(value: object, *, field: str, minimum: int = 1, maximum: int = 4000) -> str:
    if type(value) is not str:
        raise ToolError("invalid_argument", f"{field} must be a string")
    value = value.strip()
    if not minimum <= len(value) <= maximum:
        raise ToolError("invalid_argument", f"{field} length is outside the allowed range")
    return value


def _top_k(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 5:
        raise ToolError("invalid_argument", "top_k must be an integer from 1 to 5")
    return value


def _table_names(value: object) -> tuple[str, ...]:
    if type(value) is not list or not 1 <= len(value) <= 3:
        raise ToolError("invalid_argument", "tables must contain one to three names")
    if any(type(item) is not str or not item.strip() for item in value):
        raise ToolError("invalid_argument", "tables must contain non-empty strings")
    names = tuple(item.strip() for item in value)
    if len(set(names)) != len(names):
        raise ToolError("invalid_argument", "tables must not contain duplicates")
    if any(name not in ALLOWED_TABLE_COLUMNS for name in names):
        raise ToolError("table_not_allowed", "the requested table is not in the server allowlist")
    return names


def _search_catalog_arguments(arguments: Mapping[str, object]) -> tuple[str, int]:
    """search_catalog's argument checks, shared with the MCP host's pre-check."""

    _arguments_object(arguments, required={"query"}, optional={"top_k"})
    query = _string(arguments["query"], field="query", maximum=200)
    top_k = _top_k(arguments.get("top_k", 3))
    return query, top_k


def _describe_tables_arguments(arguments: Mapping[str, object]) -> tuple[str, ...]:
    """describe_tables' argument checks, shared with the MCP host's pre-check."""

    _arguments_object(arguments, required={"tables"}, optional=set())
    return _table_names(arguments["tables"])


def _params(value: object) -> tuple[object, ...]:
    if not isinstance(value, Mapping):
        raise ToolError("invalid_argument", "params must be a JSON object")
    keys = list(value)
    if any(type(key) is not str for key in keys):
        raise ToolError("invalid_argument", "params keys must be strings")
    if any(key in _RESERVED_PARAMS for key in keys):
        raise ToolError("reserved_parameter", "identity parameters are server-owned")
    if any(not key.isdecimal() or str(int(key)) != key for key in keys):
        raise ToolError("invalid_argument", "params keys must be consecutive indexes")
    indexes = sorted(int(key) for key in keys)
    if indexes != list(range(len(indexes))):
        raise ToolError("invalid_argument", "params keys must start at zero without gaps")
    values = tuple(value[str(index)] for index in indexes)
    if any(item is not None and type(item) not in _SCALAR_TYPES for item in values):
        raise ToolError("invalid_argument", "params values must be scalar")
    return values


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower()))


def _expanded_query_terms(query: str) -> set[str]:
    terms = _tokens(query)
    for phrase, replacements in _SYNONYMS.items():
        if phrase in query:
            for replacement in replacements:
                terms.update(_tokens(replacement))
    return terms


def _walk_expression(expression: object) -> tuple[ColumnRef, ...]:
    if isinstance(expression, ColumnRef):
        return (expression,)
    if isinstance(expression, FunctionCall):
        return tuple(item for arg in expression.arguments for item in _walk_expression(arg))
    if isinstance(expression, UnaryExpression):
        return _walk_expression(expression.operand)
    if isinstance(expression, BinaryExpression):
        return _walk_expression(expression.left) + _walk_expression(expression.right)
    return ()


def _walk_condition(condition: Condition | None) -> tuple[ColumnRef, ...]:
    if condition is None:
        return ()
    if isinstance(condition, Comparison):
        return _walk_expression(condition.left) + _walk_expression(condition.right)
    if isinstance(condition, BooleanExpression):
        return _walk_condition(condition.left) + _walk_condition(condition.right)
    if isinstance(condition, NotExpression):
        return _walk_condition(condition.operand)
    return ()


def _statement_columns(statement: SelectStatement) -> tuple[ColumnRef, ...]:
    columns: list[ColumnRef] = []
    for item in statement.projection:
        columns.extend(_walk_expression(item.expression))
    for expression in statement.group_by:
        columns.extend(_walk_expression(expression))
    for item in statement.order_by:
        columns.extend(_walk_expression(item.expression))
    columns.extend(_walk_condition(statement.where))
    for join in statement.joins:
        columns.extend(_walk_condition(join.condition))
    return tuple(columns)


def _has_customer_star(statement: SelectStatement) -> bool:
    if "customers" not in statement.referenced_tables:
        return False
    return any(isinstance(item.expression, Star) for item in statement.projection)


def check_sensitive_access(context: ExecutionContext, statement: SelectStatement) -> None:
    if context.role == "approver":
        return
    aliases = {
        statement.from_table.alias or statement.from_table.name: statement.from_table.name
    }
    aliases.update({join.table.alias or join.table.name: join.table.name for join in statement.joins})
    if _has_customer_star(statement):
        raise ToolError("approval_required", "customers.name values require the approval path")
    for column in _statement_columns(statement):
        table = aliases.get(column.qualifier) if column.qualifier is not None else None
        if column.name == "name" and (table == "customers" or (table is None and "customers" in aliases.values())):
            raise ToolError("approval_required", "customers.name values require the approval path")


@dataclass
class ControlledTools:
    """Small, server-bound semantic tool facade used by the W03 runtime."""

    catalog: SemanticCatalog | None = None
    executor: GuardedQueryExecutor | None = None
    retriever: Any | None = None

    def __post_init__(self) -> None:
        if self.catalog is None:
            self.catalog = load_default_catalog()
        if self.executor is None:
            self.executor = GuardedQueryExecutor(catalog_version=self.catalog.catalog_version)
        self._evidence: dict[tuple[str, str], ResultEvidence] = {}
        self._retrieval_evidence: dict[tuple[str, str], Any] = {}
        # W05 internal sidecar: retain the exact returned candidate metadata so
        # the evaluator can prove that this run's selected items reached a later
        # model request. The public tool result remains unchanged.
        self._retrieval_return_records: dict[tuple[str, str], dict[str, object]] = {}

    def call(
        self,
        name: str,
        arguments: Mapping[str, object],
        *,
        context: ExecutionContext,
        metric_bindings: Sequence[MetricBinding] = (),
    ) -> dict[str, object]:
        _require_context(context)
        if name == "search_catalog":
            return self.search_catalog(arguments, context=context)
        if name == "describe_tables":
            return self.describe_tables(arguments, context=context)
        if name == "query_readonly":
            return self.query_readonly(arguments, context=context, metric_bindings=metric_bindings)
        raise ToolError("unknown_tool", "the requested tool is not registered")

    def search_catalog(
        self,
        arguments: Mapping[str, object],
        *,
        context: ExecutionContext,
    ) -> dict[str, object]:
        _require_context(context)
        query, top_k = _search_catalog_arguments(arguments)
        assert self.catalog is not None

        if self.retriever is not None:
            search = getattr(self.retriever, "search", None)
            if not callable(search):
                raise ToolError("retrieval_unavailable", "the configured retrieval strategy is invalid")
            result = search(query, context=context, top_k=top_k)
            retrieval_evidence = getattr(result, "evidence", None)
            items = getattr(result, "items", None)
            if retrieval_evidence is None or not isinstance(items, Sequence):
                raise ToolError("retrieval_unavailable", "the retrieval strategy returned an invalid result")
            key = (context.run_id, retrieval_evidence.retrieval_id)
            returned_items = [dict(item) for item in items]
            self._retrieval_evidence[key] = retrieval_evidence
            index = getattr(self.retriever, "index", None)
            chunk_by_id = {
                str(getattr(chunk, "chunk_id", "")): chunk
                for chunk in getattr(index, "chunks", ())
                if getattr(chunk, "chunk_id", None)
            }
            source_by_id = {
                str(getattr(source, "source_id", "")): source
                for source in getattr(getattr(self.retriever, "snapshot", None), "source_records", ())
                if getattr(source, "source_id", None)
            }
            catalog_by_id = {
                str(getattr(entry, "id", "")): entry
                for entry in getattr(self.catalog, "entries", ())
                if getattr(entry, "id", None)
            }
            self._retrieval_return_records[key] = {
                "run_id": context.run_id,
                "tenant_id": context.tenant_id,
                "principal_id": context.principal_id,
                "role": context.role,
                "retrieval_id": retrieval_evidence.retrieval_id,
                "snapshot_id": retrieval_evidence.snapshot_id,
                "query_sha256": retrieval_evidence.query_sha256,
                "strategy_version": retrieval_evidence.strategy_version,
                "selected_ids": list(retrieval_evidence.selected_ids),
                "items": [
                    {
                        "id": item.get("id"),
                        "source_id": item.get("source_id"),
                        "version": item.get("version"),
                        "text_sha256": hashlib.sha256(str(item.get("text", "")).encode("utf-8")).hexdigest(),
                        "source_kind": (
                            "knowledge_document"
                            if item.get("id") in chunk_by_id
                            else (
                                "catalog_retrieval_candidate"
                                if item.get("id") in catalog_by_id
                                else "retriever_candidate"
                            )
                        ),
                        "visibility_check": self._retrieval_item_visibility(
                            item,
                            context=context,
                            chunk=chunk_by_id.get(str(item.get("id"))),
                            source=source_by_id.get(str(item.get("source_id"))),
                            catalog_entry=catalog_by_id.get(str(item.get("id"))),
                        ),
                    }
                    for item in returned_items
                ],
                "acl_basis": "returned by HybridRetriever after server identity, role, tenant, source-status, and version filtering",
            }
            return {"items": returned_items}

        query_terms = _expanded_query_terms(query)
        ranked: list[tuple[int, str, dict[str, str]]] = []
        for entry in self.catalog.entries:
            if entry.requires_approval and context.role != "approver":
                continue
            item = entry.as_search_item()
            item_text = f"{entry.id} {entry.text}".lower()
            entry_id = entry.id.lower()
            score = sum(3 if term in entry_id else 1 for term in query_terms if term in item_text)
            if query.lower() in entry.text.lower():
                score += 5
            if score:
                ranked.append((score, entry.id, item))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return {"items": [item for _, _, item in ranked[:top_k]]}

    @staticmethod
    def _retrieval_item_visibility(item, *, context, chunk, source, catalog_entry) -> dict[str, object]:
        if chunk is not None:
            from queryshield.knowledge.retrieval import _tenant_matches

            # The retriever's own tenant rule, not a second copy of it.
            tenant_scope = getattr(source, "tenant_scope", None)
            tenant_visible = isinstance(tenant_scope, str) and _tenant_matches(tenant_scope, context.tenant_id)
            version_matches = (
                getattr(chunk, "source_id", None) == item.get("source_id")
                and getattr(chunk, "source_version", None) == item.get("version")
                and getattr(source, "version", None) == item.get("version")
            )
            active = getattr(source, "status", None) == "active"
            role_visible = context.role in getattr(source, "allowed_roles", ())
            return {
                "candidate_type": "knowledge_document",
                "source_active": active,
                "version_matches_snapshot": version_matches,
                "tenant_visible": bool(tenant_visible),
                "role_visible": bool(role_visible),
                "passed": bool(active and version_matches and tenant_visible and role_visible),
            }
        if catalog_entry is not None:
            version_matches = (
                getattr(catalog_entry, "source_id", None) == item.get("source_id")
                and getattr(catalog_entry, "version", None) == item.get("version")
            )
            role_visible = not bool(getattr(catalog_entry, "requires_approval", False)) or context.role == "approver"
            return {
                "candidate_type": "catalog_retrieval_candidate",
                "source_active": True,
                "version_matches_snapshot": version_matches,
                "tenant_visible": True,
                "role_visible": role_visible,
                "passed": bool(version_matches and role_visible),
            }
        return {
            "candidate_type": "unknown",
            "source_active": False,
            "version_matches_snapshot": False,
            "tenant_visible": False,
            "role_visible": False,
            "passed": False,
        }

    def describe_tables(
        self,
        arguments: Mapping[str, object],
        *,
        context: ExecutionContext,
    ) -> dict[str, object]:
        _require_context(context)
        requested = _describe_tables_arguments(arguments)
        assert self.catalog is not None
        tables: list[dict[str, object]] = []
        for table_name in sorted(requested):
            table_entry = next(
                (entry for entry in self.catalog.entries if entry.kind == "table" and entry.table == table_name),
                None,
            )
            if table_entry is None:
                raise ToolError("table_not_allowed", "the requested table is not described by the catalog")
            columns = sorted(ALLOWED_TABLE_COLUMNS[table_name])
            tables.append(
                {
                    "name": table_name,
                    "columns": columns,
                    "source_id": table_entry.source_id,
                    "version": table_entry.version,
                }
            )
        return {"tables": tables}

    def query_readonly(
        self,
        arguments: Mapping[str, object],
        *,
        context: ExecutionContext,
        metric_bindings: Sequence[MetricBinding] = (),
    ) -> dict[str, object]:
        _require_context(context)
        _arguments_object(arguments, required={"sql", "params"}, optional=set())
        sql = _string(arguments["sql"], field="sql", maximum=4000)
        params = _params(arguments["params"])
        try:
            statement = parse_readonly_select(sql)
        except SQLPolicyError as exc:
            raise ToolError(exc.code, _controlled_error_detail(exc, exc.code)) from exc
        check_sensitive_access(context, statement)
        if any(table not in ALLOWED_TABLE_COLUMNS for table in statement.referenced_tables):
            raise ToolError("table_not_allowed", "query references a table outside the server allowlist")
        if any(not isinstance(binding, MetricBinding) for binding in metric_bindings):
            raise ToolError("invalid_binding", "metric bindings are server-owned")
        net_bindings = tuple(
            binding
            for binding in metric_bindings
            if binding.metric_id.removeprefix("metric.") == "net_fen"
        )
        if len(net_bindings) > 1:
            raise ToolError("invalid_binding", "net_fen must have one metric binding")
        net_binding = net_bindings[0] if net_bindings else None
        controlled_net_binding = (
            net_binding if net_binding is not None and self._is_valid_net_binding(net_binding) else None
        )
        assert self.executor is not None
        try:
            # Validate the model-selected SQL with the same AST/tenant renderer
            # before routing to the server-owned plan.  The MetricBinding, not
            # a model-selected output alias, identifies the trusted metric.
            render_scoped_select(statement, tenant_id=context.tenant_id, input_params=params)
            if controlled_net_binding is not None:
                if any(table not in {"orders", "refunds"} for table in statement.referenced_tables):
                    raise ToolError(
                        "invalid_binding",
                        "net_fen controlled plan accepts only orders and refunds requests",
                    )
                evidence = self._execute_net_fen_plan(context=context, binding=controlled_net_binding)
            else:
                result = self.executor.execute(
                    sql,
                    context=context,
                    params=params,
                    metric_bindings=metric_bindings,
                )
                evidence = result.evidence
        except SQLPolicyError as exc:
            raise ToolError(exc.code, _controlled_error_detail(exc, exc.code)) from exc
        except GuardedQueryError as exc:
            raise ToolError(exc.code, _controlled_error_detail(exc, exc.code)) from exc

        self._evidence[(context.run_id, evidence.result_id)] = evidence
        response = {
            "rows": [dict(row) for row in evidence.rows],
            "row_count": evidence.row_count,
            "result_id": evidence.result_id,
            "policy_version": evidence.policy_version,
        }
        if controlled_net_binding is not None and evidence.metric_bindings == (controlled_net_binding,):
            response.update({"metric_plan_id": NET_FEN_PLAN_ID, "plan_query_count": 2})
        return response

    def _is_valid_net_binding(self, binding: MetricBinding) -> bool:
        assert self.catalog is not None
        entry = self.catalog.metric("net_fen")
        return (
            binding.plan_id == NET_FEN_PLAN_ID
            and binding.result_position == "net_fen"
            and binding.unit == entry.payload.get("unit")
            and binding.catalog_source_id == entry.source_id
            and binding.catalog_version == self.catalog.catalog_version
            and binding.time_window.get("timezone") == "UTC"
        )

    def _execute_net_fen_plan(
        self,
        *,
        context: ExecutionContext,
        binding: MetricBinding,
    ) -> ResultEvidence:
        assert self.executor is not None
        window = dict(binding.time_window)
        gross_result = self.executor.execute(
            NET_FEN_GROSS_QUERY,
            context=context,
            params=("paid", window["start"], window["end"]),
            metric_bindings=(),
        )
        refund_result = self.executor.execute(
            NET_FEN_REFUND_QUERY,
            context=context,
            params=("paid", window["start"], window["end"], window["start"], window["end"]),
            metric_bindings=(),
        )
        gross_fen = self._plan_value(gross_result.evidence, "gross_fen")
        refund_fen = self._plan_value(refund_result.evidence, "refund_fen")
        if gross_fen < 0 or refund_fen < 0:
            raise ToolError("metric_plan_result_invalid", "controlled raw money result is negative")
        if (
            gross_result.evidence.policy_version != refund_result.evidence.policy_version
            or gross_result.evidence.catalog_version != refund_result.evidence.catalog_version
        ):
            raise ToolError("metric_plan_result_invalid", "controlled query results have incompatible versions")

        # The combined evidence hash commits to both actual guarded SQL
        # executions, while the public result remains one aggregate row.
        normalized_query = (
            f"{NET_FEN_PLAN_ID}\n"
            f"gross_query_sha256={gross_result.evidence.query_sha256}\n"
            f"refund_query_sha256={refund_result.evidence.query_sha256}"
        )
        normalized_params = {
            "plan_id": NET_FEN_PLAN_ID,
            "time_window": dict(window),
            "gross_params_sha256": gross_result.evidence.params_sha256,
            "refund_params_sha256": refund_result.evidence.params_sha256,
        }
        return ResultEvidence.from_server_execution(
            context,
            result_id=f"result-{uuid4()}",
            rows=({"net_fen": gross_fen - refund_fen},),
            normalized_query=normalized_query,
            params=normalized_params,
            observed_at=max(gross_result.evidence.observed_at, refund_result.evidence.observed_at),
            policy_version=gross_result.evidence.policy_version,
            catalog_version=gross_result.evidence.catalog_version,
            metric_bindings=(binding,),
            metric_plan_id=NET_FEN_PLAN_ID,
        )

    @staticmethod
    def _plan_value(evidence: ResultEvidence, column: str) -> int:
        if evidence.row_count != 1 or len(evidence.rows) != 1 or column not in evidence.rows[0]:
            raise ToolError("metric_plan_result_invalid", "controlled aggregate did not return one bound value")
        value = evidence.rows[0][column]
        if type(value) is not int:
            raise ToolError("metric_plan_result_invalid", "controlled aggregate value is not an integer")
        return value

    def get_result_evidence(
        self,
        result_id: str,
        *,
        context: ExecutionContext,
    ) -> ResultEvidence:
        _require_context(context)
        evidence = self._evidence.get((context.run_id, result_id))
        if evidence is None or evidence.tenant_id != context.tenant_id or evidence.principal_id != context.principal_id:
            raise ToolError("result_not_found", "the result is not visible in this run")
        return evidence

    def get_retrieval_evidence(
        self,
        retrieval_id: str,
        *,
        context: ExecutionContext,
    ) -> Any:
        _require_context(context)
        evidence = self._retrieval_evidence.get((context.run_id, retrieval_id))
        if evidence is None or evidence.run_id != context.run_id:
            raise ToolError("retrieval_not_found", "the retrieval is not visible in this run")
        return evidence


__all__ = ["ControlledTools", "ToolError"]
