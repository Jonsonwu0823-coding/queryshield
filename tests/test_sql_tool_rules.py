"""Behaviour of the SQL, tool, fact, database and parallel rules, pinned before they are tidied.

Nothing here needs a database server: queries run against a fake connection.
Each group pins what a consumer can see (error codes and texts, rebound
parameters, rendered SQL, persisted hashes), not how the code is arranged, so a
tidy-up that keeps behaviour keeps these green.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import psycopg
import pytest

from queryshield.agent.context import NET_FEN_PLAN_ID
from queryshield.agent.metric_intent import build_metric_binding
from queryshield.agent.parallel import ParallelPlan
from queryshield.agent.parallel_durable import DurableParallelError, DurableParallelScheduler, _plan
from queryshield.agent.proposals import ExecutionContext, FactRef, MetricBinding, ResultEvidence
from queryshield.agent.tool_execution import (
    ApprovalRequiredError,
    _resolve_call,
    call_tool,
    execute_approved_query,
    prepare_pending_call,
)
from queryshield.approval import versions
from queryshield.catalog import load_default_catalog
from queryshield.catalog.catalog import ALLOWED_ROLES, ALLOWED_TABLE_COLUMNS
from queryshield.db import guarded
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.db.state_store import StateStore
from queryshield.facts import FactResolutionError, FactResolver
from queryshield.knowledge.retrieval import KeywordSynonymRetriever
from queryshield.knowledge.ingest import load_snapshot
from queryshield.policy import sql as sql_policy
from queryshield.policy.sql import SQLPolicyError, parse_readonly_select
from queryshield.tools import ControlledTools, ToolError
from queryshield.tools.semantic import check_sensitive_access


SEPTEMBER = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}
SEPTEMBER_UTC = {**SEPTEMBER, "timezone": "UTC"}
CLOCK = datetime(2026, 9, 21, tzinfo=timezone.utc)


class _Cursor:
    def __init__(self, connection: "_Connection") -> None:
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def execute(self, sql, params) -> None:
        self.connection.executed.append((sql, tuple(params)))

    def fetchmany(self, size):
        return [{"gross_fen": 15000, "refund_fen": 3000}][:size]


class _Connection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.statements: list[tuple[str, tuple[object, ...]]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def cursor(self, *, row_factory):
        return _Cursor(self)

    def execute(self, sql, params=()):
        self.statements.append((sql, tuple(params)))


def _tools() -> tuple[ControlledTools, _Connection]:
    connection = _Connection()
    executor = GuardedQueryExecutor(connect=lambda: connection, clock=lambda: CLOCK)
    return ControlledTools(catalog=load_default_catalog(), executor=executor), connection


def _context(*, role: str = "requester", tenant_id: str = "A") -> ExecutionContext:
    return ExecutionContext(run_id="run-rules", tenant_id=tenant_id, principal_id="principal-A", role=role)


def _error(callable_, *args, **kwargs) -> ToolError:
    with pytest.raises(ToolError) as caught:
        callable_(*args, **kwargs)
    return caught.value


# --------------------------------------------------------------------------
# The state store creates exactly this schema.
# --------------------------------------------------------------------------

_EXPECTED_SCHEMA = {
    ('index', 'sqlite_autoindex_approvals_1', 'approvals'): None,
    ('index', 'sqlite_autoindex_events_1', 'events'): None,
    ('index', 'sqlite_autoindex_knowledge_acl_1', 'knowledge_acl'): None,
    ('index', 'sqlite_autoindex_knowledge_snapshots_1', 'knowledge_snapshots'): None,
    ('index', 'sqlite_autoindex_parallel_branches_1', 'parallel_branches'): None,
    ('index', 'sqlite_autoindex_parallel_branches_2', 'parallel_branches'): None,
    ('index', 'sqlite_autoindex_parallel_groups_1', 'parallel_groups'): None,
    ('index', 'sqlite_autoindex_parallel_groups_2', 'parallel_groups'): None,
    ('index', 'sqlite_autoindex_preferences_1', 'preferences'): None,
    ('index', 'sqlite_autoindex_runs_1', 'runs'): None,
    ('index', 'sqlite_autoindex_state_meta_1', 'state_meta'): None,
    ('table', 'approvals', 'approvals'): 'CREATE TABLE approvals ( approval_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), tenant_id TEXT NOT NULL, requester_principal_id TEXT NOT NULL, action_hash TEXT NOT NULL, action_json TEXT NOT NULL, policy_version TEXT NOT NULL, catalog_version TEXT NOT NULL, knowledge_snapshot_id TEXT NOT NULL, status TEXT NOT NULL, expires_at TEXT NOT NULL, approver_principal_id TEXT, decision_at TEXT, decision_json TEXT )',
    ('table', 'events', 'events'): 'CREATE TABLE events ( run_id TEXT NOT NULL REFERENCES runs(run_id), event_id INTEGER NOT NULL, type TEXT NOT NULL, status TEXT NOT NULL, occurred_at TEXT NOT NULL, result_id TEXT, payload_json TEXT NOT NULL, PRIMARY KEY (run_id, event_id) )',
    ('table', 'knowledge_acl', 'knowledge_acl'): 'CREATE TABLE knowledge_acl ( source_id TEXT PRIMARY KEY, tenant_scope TEXT NOT NULL, allowed_roles_json TEXT NOT NULL, status TEXT NOT NULL, acl_version INTEGER NOT NULL DEFAULT 1 )',
    ('table', 'knowledge_snapshots', 'knowledge_snapshots'): 'CREATE TABLE knowledge_snapshots ( snapshot_id TEXT PRIMARY KEY, catalog_version TEXT NOT NULL, manifest_json TEXT NOT NULL, published_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1 )',
    ('table', 'parallel_branches', 'parallel_branches'): 'CREATE TABLE parallel_branches ( group_id TEXT NOT NULL REFERENCES parallel_groups(group_id), branch_id TEXT NOT NULL, metric_id TEXT NOT NULL, status TEXT NOT NULL, result_json TEXT, error_code TEXT, updated_at TEXT NOT NULL, PRIMARY KEY (group_id, branch_id), UNIQUE (group_id, metric_id) )',
    ('table', 'parallel_groups', 'parallel_groups'): 'CREATE TABLE parallel_groups ( group_id TEXT PRIMARY KEY, run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id), plan_hash TEXT NOT NULL, plan_json TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, summary_json TEXT )',
    ('table', 'preferences', 'preferences'): 'CREATE TABLE preferences ( tenant_id TEXT NOT NULL, principal_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, version INTEGER NOT NULL, confirmed_at TEXT NOT NULL, PRIMARY KEY (tenant_id, principal_id, key) )',
    ('table', 'runs', 'runs'): 'CREATE TABLE runs ( run_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, principal_id TEXT NOT NULL, role TEXT NOT NULL, question TEXT NOT NULL, status TEXT NOT NULL, mode TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, action_json TEXT, approval_id TEXT, result_json TEXT, facts_json TEXT, answer TEXT, usage_json TEXT, checkpoint_json TEXT, run_config_json TEXT, error_code TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0, sql_exec_count INTEGER NOT NULL DEFAULT 0, model_call_count INTEGER NOT NULL DEFAULT 0, tool_call_count INTEGER NOT NULL DEFAULT 0 )',
    ('table', 'state_meta', 'state_meta'): 'CREATE TABLE state_meta ( key TEXT PRIMARY KEY, value TEXT NOT NULL )',
}


def test_state_store_creates_the_expected_schema_and_version_row(tmp_path) -> None:
    with StateStore(":memory:") as store:
        rows = store._connection.execute("SELECT type, name, tbl_name, sql FROM sqlite_master").fetchall()
        actual = {
            (row["type"], row["name"], row["tbl_name"]): " ".join(row["sql"].split()) if row["sql"] else None
            for row in rows
        }
        assert actual == _EXPECTED_SCHEMA
        meta = store._connection.execute("SELECT key, value FROM state_meta").fetchall()
        assert [tuple(row) for row in meta] == [("schema_version", "qs-state-v1")]
    # Opening an existing file again creates nothing new and keeps the version row.
    path = tmp_path / "state.sqlite"
    StateStore(path).close()
    with StateStore(path) as again:
        assert again._connection.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] == len(_EXPECTED_SCHEMA)
        assert tuple(again._connection.execute("SELECT value FROM state_meta").fetchone()) == ("qs-state-v1",)


# --------------------------------------------------------------------------
# One SQL policy version, one table list, one role set, one reserved-name set.
# --------------------------------------------------------------------------


def test_the_sql_policy_version_is_qs_sql_v1_everywhere() -> None:
    assert versions.BOUND_POLICY_VERSION == "qs-sql-v1"
    assert versions.BOUND_CATALOG_VERSION == "catalog-v4"
    connection = _Connection()
    executor = GuardedQueryExecutor(connect=lambda: connection, clock=lambda: CLOCK)
    result = executor.execute("SELECT COUNT(*) AS paid_count FROM orders", context=_context())
    assert result.evidence.policy_version == "qs-sql-v1"
    plan = ParallelPlan.from_context(_context(), ["net_fen", "gross_fen"], time_window=SEPTEMBER_UTC)
    assert plan.policy_version == "qs-sql-v1"
    durable_plan, _ = _plan(_context(), ["net_fen", "gross_fen"], SEPTEMBER_UTC)
    assert durable_plan["policy_version"] == "qs-sql-v1"


def test_persisted_parallel_plan_hashes_do_not_change() -> None:
    window = {"start": "2026-07-01T00:00:00Z", "end": "2026-08-01T00:00:00Z", "timezone": "UTC"}
    context = ExecutionContext(run_id="run-1", tenant_id="A", principal_id="p-A", role="requester")
    _, durable_hash = _plan(context, ["net_fen", "gross_fen", "paid_count"], window)
    assert durable_hash == "e28b87cd3dc6b8ff6e6196144bdba10ed987c34df7485b32f12b409b02dd3970"
    memory = ParallelPlan.from_context(context, ["net_fen", "gross_fen"], time_window=window)
    assert memory.plan_hash == "e2cf3b44d0d9ecbe12661743d8b5f2e8c93912f27755508bd89862c187785f7f"


def test_the_sql_table_allowlist_is_the_catalog_table_list() -> None:
    assert sql_policy.ALLOWED_TABLES == frozenset({"customers", "orders", "refunds"})
    assert sql_policy.ALLOWED_TABLES == frozenset(ALLOWED_TABLE_COLUMNS)
    for table in ("customers", "orders", "refunds"):
        assert parse_readonly_select(f"SELECT COUNT(*) AS n FROM {table}").from_table.name == table
    with pytest.raises(SQLPolicyError) as caught:
        parse_readonly_select("SELECT COUNT(*) AS n FROM invoices")
    assert caught.value.code == "table_not_allowed"
    tools, _ = _tools()
    assert _error(tools.describe_tables, {"tables": ["invoices"]}, context=_context()).code == "table_not_allowed"


def test_tools_accept_exactly_the_catalog_roles() -> None:
    assert ALLOWED_ROLES == frozenset({"requester", "approver"})
    tools, _ = _tools()
    for role in sorted(ALLOWED_ROLES):
        assert tools.call("search_catalog", {"query": "paid_count"}, context=_context(role=role))["items"]
    forbidden = _error(tools.call, "search_catalog", {"query": "paid_count"}, context=_context(role="auditor"))
    assert (forbidden.code, forbidden.message) == ("forbidden", "the current role cannot use semantic tools")
    unauthorized = _error(tools.call, "search_catalog", {"query": "paid_count"}, context="not a context")
    assert (unauthorized.code, unauthorized.message) == (
        "unauthorized",
        "tool calls require a server-created execution context",
    )


@pytest.mark.parametrize("name", ["tenant_id", "principal_id", "role", "authorization", "token"])
def test_identity_parameters_are_reserved_in_the_tool_layer(name) -> None:
    tools, _ = _tools()
    error = _error(tools.query_readonly, {"sql": "SELECT COUNT(*) AS n FROM orders", "params": {name: "x"}}, context=_context())
    assert (error.code, error.message) == ("reserved_parameter", "identity parameters are server-owned")


# --------------------------------------------------------------------------
# query_readonly: argument and parameter checks, in order, with their texts.
# --------------------------------------------------------------------------

_COUNT = "SELECT COUNT(*) AS paid_count FROM orders WHERE status = %s"


@pytest.mark.parametrize(
    "arguments, code, message",
    [
        ("not an object", "invalid_arguments", "tool arguments must be an object"),
        ({"params": {}}, "missing_argument", "a required tool argument is missing"),
        ({"sql": _COUNT, "params": {"0": "paid"}, "extra": 1}, "unknown_argument", "tool arguments contain an unknown field"),
        ({"sql": 5, "params": {}}, "invalid_argument", "sql must be a string"),
        ({"sql": "   ", "params": {}}, "invalid_argument", "sql length is outside the allowed range"),
        ({"sql": "S" * 4001, "params": {}}, "invalid_argument", "sql length is outside the allowed range"),
        ({"sql": _COUNT, "params": ["paid"]}, "invalid_argument", "params must be a JSON object"),
        ({"sql": _COUNT, "params": {1: "paid"}}, "invalid_argument", "params keys must be strings"),
        ({"sql": _COUNT, "params": {"01": "paid"}}, "invalid_argument", "params keys must be consecutive indexes"),
        ({"sql": _COUNT, "params": {"٠": "paid"}}, "invalid_argument", "params keys must be consecutive indexes"),
        ({"sql": _COUNT, "params": {"1": "paid"}}, "invalid_argument", "params keys must start at zero without gaps"),
        ({"sql": _COUNT, "params": {"0": ["paid"]}}, "invalid_argument", "params values must be scalar"),
        ({"sql": "SELECT FROM", "params": {}}, "invalid_sql", "expected column name"),
        ({"sql": "SELECT * FROM invoices", "params": {}}, "table_not_allowed", "table is not in the allowlist"),
        ({"sql": "DELETE FROM orders", "params": {}}, "statement_not_allowed", "only SELECT is allowed"),
    ],
)
def test_query_readonly_argument_checks_keep_their_codes_and_texts(arguments, code, message) -> None:
    tools, connection = _tools()
    error = _error(tools.query_readonly, arguments, context=_context())
    assert (error.code, error.message) == (code, message)
    assert connection.executed == []


def test_query_readonly_checks_the_sql_before_the_sensitive_gate_and_the_gate_before_bindings() -> None:
    tools, connection = _tools()
    sensitive = {"sql": "SELECT name FROM customers", "params": {}}
    assert _error(
        tools.query_readonly,
        {"sql": "SELECT name FROM nowhere", "params": {}},
        context=_context(),
        metric_bindings=("not a binding",),
    ).code == "table_not_allowed"
    gate = _error(tools.query_readonly, sensitive, context=_context(), metric_bindings=("not a binding",))
    assert (gate.code, gate.message) == ("approval_required", "customers.name values require the approval path")
    approver = _error(tools.query_readonly, sensitive, context=_context(role="approver"), metric_bindings=("not a binding",))
    assert (approver.code, approver.message) == ("invalid_binding", "metric bindings are server-owned")
    assert connection.executed == []


def test_query_readonly_renders_the_scoped_query_and_runs_it() -> None:
    tools, connection = _tools()
    response = tools.query_readonly({"sql": _COUNT, "params": {"0": "paid"}}, context=_context())
    assert response["row_count"] == 1 and response["policy_version"] == "qs-sql-v1"
    assert connection.executed == [
        (
            'SELECT COUNT(*) AS "paid_count" FROM (SELECT * FROM "orders" WHERE "tenant_id" = %s) AS "orders" '
            'WHERE "status" = %s LIMIT 101',
            ("A", "paid"),
        )
    ]


# --------------------------------------------------------------------------
# Sensitive access: the alias map decides which table a column belongs to.
# --------------------------------------------------------------------------

_JOIN = "INNER JOIN customers AS c ON o.tenant_id = c.tenant_id AND o.customer_id = c.customer_id"


@pytest.mark.parametrize(
    "sql, sensitive",
    [
        ("SELECT name FROM customers", True),
        ("SELECT c.name FROM customers AS c", True),
        ("SELECT customers.name FROM customers", True),
        ("SELECT * FROM customers", True),
        ("SELECT * FROM orders", False),
        ("SELECT COUNT(*) AS n FROM customers", False),
        ("SELECT customers.customer_id FROM customers", False),
        (f"SELECT o.order_id FROM orders AS o {_JOIN}", False),
        (f"SELECT c.name FROM orders AS o {_JOIN}", True),
        (f"SELECT name FROM orders AS o {_JOIN}", True),
        (f"SELECT o.name FROM orders AS o {_JOIN}", False),
        (f"SELECT x.name FROM orders AS o {_JOIN}", True),
        ("SELECT x.name FROM orders AS o", False),
        (f"SELECT o.order_id FROM orders AS o {_JOIN} WHERE c.name = %s", True),
        (f"SELECT o.order_id FROM orders AS o {_JOIN} ORDER BY c.name", True),
        (f"SELECT COUNT(*) AS n FROM orders AS o {_JOIN} GROUP BY c.name", True),
        (
            "SELECT o.order_id FROM orders AS o INNER JOIN customers AS c "
            "ON o.customer_id = c.customer_id AND c.name = %s",
            True,
        ),
        ("SELECT COALESCE(c.name, 0) AS n FROM customers AS c", True),
        ("SELECT (c.name + 1) AS n FROM customers AS c", True),
        ("SELECT o.order_id FROM orders AS o WHERE NOT (o.status = %s)", False),
        ("SELECT c.customer_id FROM customers AS c WHERE NOT (c.name = %s)", True),
    ],
)
def test_check_sensitive_access_follows_the_alias_to_its_table(sql, sensitive) -> None:
    statement = parse_readonly_select(sql)
    if sensitive:
        error = _error(check_sensitive_access, _context(), statement)
        assert (error.code, error.message) == ("approval_required", "customers.name values require the approval path")
    else:
        assert check_sensitive_access(_context(), statement) is None
    assert check_sensitive_access(_context(role="approver"), statement) is None


def test_a_sensitive_query_that_is_otherwise_broken_keeps_its_own_code() -> None:
    tools, connection = _tools()
    wrong_qualifier = _error(
        call_tool,
        tools,
        "query_readonly",
        {"sql": "SELECT x.name FROM customers AS c", "params": {}},
        context=_context(),
    )
    assert (wrong_qualifier.code, wrong_qualifier.message) == (
        "unknown_qualifier",
        "column qualifier is not a known table alias",
    )
    missing_parameter = _error(
        call_tool,
        tools,
        "query_readonly",
        {"sql": "SELECT c.name FROM customers AS c WHERE c.customer_id = %s", "params": {}},
        context=_context(),
    )
    assert missing_parameter.code == "parameter_mismatch"
    assert connection.executed == []


def test_a_clean_sensitive_query_parks_for_approval_with_its_canonical_call() -> None:
    tools, connection = _tools()
    sql = "SELECT c.name FROM customers AS c WHERE c.customer_id = %s"
    error = _error(call_tool, tools, "query_readonly", {"sql": sql, "params": {"0": "c1"}}, context=_context())
    assert isinstance(error, ApprovalRequiredError)
    assert error.pending_call == {
        "tool": "query_readonly",
        "sql": sql,
        "params": {"0": "c1"},
        "metrics": [],
        "time_window": None,
    }
    assert prepare_pending_call(tools, {"sql": sql, "params": {"0": "c1"}}, context=_context()) == error.pending_call
    assert connection.executed == []


# --------------------------------------------------------------------------
# The parameter-index rule at each place that reads parameters by position.
# --------------------------------------------------------------------------

_TENANT_SQL = "SELECT COUNT(*) AS n FROM orders WHERE tenant_id = %s AND status = %s"
_ELEVEN_SQL = "SELECT COUNT(*) AS n FROM orders WHERE " + " OR ".join(["status = %s"] * 10 + ["tenant_id = %s"])
_BAD_KEYS = [
    {"01": "B", "1": "paid"},
    {"0": "B", "2": "paid"},
    {"1": "B", "2": "paid"},
    {"٠": "B", "1": "paid"},
    {"0": "B", 1: "paid"},
    {"-1": "B", "0": "paid"},
]


def _resolve(tools, arguments, *, bindings=()):
    return _resolve_call(tools, "query_readonly", arguments, context=_context(), metric_bindings=bindings)


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"0": "B", "1": "paid"}, {"0": "A", "1": "paid"}),
        ({"1": "paid", "0": "B"}, {"0": "A", "1": "paid"}),
    ],
)
def test_tenant_equality_parameters_are_rebound_to_the_authenticated_tenant(params, expected) -> None:
    tools, _ = _tools()
    arguments, bindings, declared = _resolve(tools, {"sql": _TENANT_SQL, "params": params})
    assert dict(arguments["params"]) == expected and bindings == () and declared == ()


def test_tenant_equality_rebinding_reads_eleven_parameters_in_index_order() -> None:
    tools, _ = _tools()
    params = {str(index): "paid" for index in range(10)}
    params["10"] = "B"
    arguments, _, _ = _resolve(tools, {"sql": _ELEVEN_SQL, "params": params})
    assert dict(arguments["params"]) == {**{str(index): "paid" for index in range(10)}, "10": "A"}


@pytest.mark.parametrize("params", _BAD_KEYS)
def test_tenant_equality_rebinding_leaves_malformed_parameter_keys_alone(params) -> None:
    tools, _ = _tools()
    arguments, _, _ = _resolve(tools, {"sql": _TENANT_SQL, "params": params})
    assert dict(arguments["params"]) == params


def test_tenant_equality_rebinding_leaves_non_scalar_values_and_missing_parameters_alone() -> None:
    tools, _ = _tools()
    sql = "SELECT COUNT(*) AS n FROM orders WHERE status = %s AND tenant_id = %s"
    for params in ({"0": "paid", "1": ["B"]}, {"0": "paid"}):
        arguments, _, _ = _resolve(tools, {"sql": sql, "params": params})
        assert dict(arguments["params"]) == params


_NET_SQL = "SELECT COALESCE(SUM(amount_fen), 0) AS net_fen FROM orders WHERE status = %s AND created_at >= %s AND created_at < %s"
_NET_ERROR = (
    "evidence_validation_failed",
    "a declared net_fen query must read only orders/refunds and filter orders.created_at by the declared window",
)
_GROSS_SQL = "SELECT COALESCE(SUM(amount_fen), 0) AS gross_fen FROM orders WHERE status = %s AND created_at >= %s AND created_at < %s"
_PROJECTION_ERROR = (
    "evidence_validation_failed",
    "query projection or filters do not match the server-bound metric semantics; customer_id grouped results "
    "require a composite tenant_id and customer_id join whose ON clause has only those two equalities",
)


def _declared(sql: str, params: dict, metric: str) -> dict:
    return {"sql": sql, "params": params, "metrics": [metric], "time_window": dict(SEPTEMBER)}


_WINDOW_PARAMS = {"0": "paid", "1": SEPTEMBER["start"], "2": SEPTEMBER["end"]}


def test_a_declared_net_fen_query_with_ordered_parameters_is_accepted() -> None:
    tools, _ = _tools()
    _, bindings, declared = _resolve(tools, _declared(_NET_SQL, _WINDOW_PARAMS, "net_fen"))
    assert declared == ("net_fen",) and bindings[0].plan_id == NET_FEN_PLAN_ID


@pytest.mark.parametrize(
    "params",
    [
        {"0": "paid", "1": SEPTEMBER["start"], "3": SEPTEMBER["end"]},
        {"0": "paid", "01": SEPTEMBER["start"], "2": SEPTEMBER["end"]},
        {"1": "paid", "2": SEPTEMBER["start"], "3": SEPTEMBER["end"]},
        {"0": "paid", "1": SEPTEMBER["start"], 2: SEPTEMBER["end"]},
    ],
)
def test_a_declared_net_fen_query_with_malformed_parameter_keys_is_refused(params) -> None:
    tools, _ = _tools()
    error = _error(_resolve, tools, _declared(_NET_SQL, params, "net_fen"))
    assert (error.code, error.message) == _NET_ERROR


def test_a_declared_gross_query_with_ordered_parameters_is_bound_to_its_column() -> None:
    tools, _ = _tools()
    _, bindings, _ = _resolve(tools, _declared(_GROSS_SQL, _WINDOW_PARAMS, "gross_fen"))
    assert [(item.metric_id, item.result_position, item.grouped) for item in bindings] == [("gross_fen", "gross_fen", False)]


@pytest.mark.parametrize(
    "params",
    [
        {"0": "paid", "1": SEPTEMBER["start"], "3": SEPTEMBER["end"]},
        {"0": "paid", "01": SEPTEMBER["start"], "2": SEPTEMBER["end"]},
        {"1": "paid", "2": SEPTEMBER["start"], "3": SEPTEMBER["end"]},
        {"0": "paid", "1": SEPTEMBER["start"], 2: SEPTEMBER["end"]},
    ],
)
def test_a_declared_gross_query_with_malformed_parameter_keys_is_refused(params) -> None:
    tools, _ = _tools()
    error = _error(_resolve, tools, _declared(_GROSS_SQL, params, "gross_fen"))
    assert (error.code, error.message) == _PROJECTION_ERROR


# --------------------------------------------------------------------------
# Metric binding: the filters, the operator direction and the customer join.
# --------------------------------------------------------------------------


def _gross(where: str) -> str:
    return f"SELECT COALESCE(SUM(amount_fen), 0) AS gross_fen FROM orders WHERE {where}"


_PAID_WINDOW = ("paid", SEPTEMBER["start"], SEPTEMBER["end"])


@pytest.mark.parametrize(
    "where, values, accepted",
    [
        ("status = %s AND created_at >= %s AND created_at < %s", _PAID_WINDOW, True),
        ("%s = status AND %s <= created_at AND %s > created_at", _PAID_WINDOW, True),
        ("status = %s AND created_at >= %s AND %s > created_at", _PAID_WINDOW, True),
        ("status = %s AND %s <= created_at AND created_at < %s", _PAID_WINDOW, True),
        ("status = %s AND %s >= created_at AND %s > created_at", _PAID_WINDOW, False),
        ("status = %s AND created_at <= %s AND created_at < %s", _PAID_WINDOW, False),
        ("status = %s AND created_at >= %s AND created_at <= %s", _PAID_WINDOW, False),
        ("status = %s AND created_at >= %s AND created_at > %s", _PAID_WINDOW, False),
        ("status = %s AND created_at <> %s AND created_at < %s", _PAID_WINDOW, False),
        ("status <> %s AND created_at >= %s AND created_at < %s", _PAID_WINDOW, False),
        ("status = %s AND created_at >= %s", _PAID_WINDOW[:2], False),
        ("status = %s OR created_at >= %s AND created_at < %s", _PAID_WINDOW, False),
        ("status = %s AND created_at >= %s AND created_at < %s AND tenant_id = %s", (*_PAID_WINDOW, "A"), True),
        ("status = %s AND created_at >= %s AND created_at < %s AND tenant_id = %s", (*_PAID_WINDOW, "B"), True),
    ],
)
def test_the_gross_binding_filters_accept_only_the_bound_window_in_either_direction(where, values, accepted) -> None:
    tools, _ = _tools()
    params = {str(index): value for index, value in enumerate(values)}
    arguments = _declared(_gross(where), params, "gross_fen")
    if accepted:
        _, bindings, _ = _resolve(tools, arguments)
        assert bindings[0].result_position == "gross_fen"
    else:
        assert _error(_resolve, tools, arguments).code == "evidence_validation_failed"


def test_the_net_fen_window_check_accepts_either_operand_order() -> None:
    tools, _ = _tools()
    flipped = "SELECT COALESCE(SUM(amount_fen), 0) AS net_fen FROM orders WHERE status = %s AND %s <= created_at AND %s > created_at"
    _, bindings, _ = _resolve(tools, _declared(flipped, _WINDOW_PARAMS, "net_fen"))
    assert bindings[0].plan_id == NET_FEN_PLAN_ID
    wrong = "SELECT COALESCE(SUM(amount_fen), 0) AS net_fen FROM orders WHERE status = %s AND %s >= created_at AND %s > created_at"
    assert (_error(_resolve, tools, _declared(wrong, _WINDOW_PARAMS, "net_fen")).message) == _NET_ERROR[1]
    other_table = (
        "SELECT COALESCE(SUM(amount_fen), 0) AS net_fen FROM customers WHERE status = %s AND created_at >= %s AND created_at < %s"
    )
    assert (_error(_resolve, tools, _declared(other_table, _WINDOW_PARAMS, "net_fen")).message) == _NET_ERROR[1]


def _by_customer(from_clause: str, qualifier: str, order: str) -> str:
    return (
        f"SELECT {qualifier}.customer_id, COALESCE(SUM({qualifier}.amount_fen), 0) AS gross_fen FROM {from_clause} "
        f"WHERE {qualifier}.status = %s AND {qualifier}.created_at >= %s AND {qualifier}.created_at < %s "
        f"GROUP BY {qualifier}.customer_id ORDER BY gross_fen DESC LIMIT 1"
    )


_KEYS = "{o}.tenant_id = {c}.tenant_id AND {o}.customer_id = {c}.customer_id"


@pytest.mark.parametrize(
    "from_clause, qualifier, trusted",
    [
        (f"orders AS o INNER JOIN customers AS c ON {_KEYS.format(o='o', c='c')}", "o", True),
        (f"orders AS o INNER JOIN customers AS c ON {_KEYS.format(o='c', c='o')}", "o", True),
        (f"orders AS c INNER JOIN customers AS o ON {_KEYS.format(o='c', c='o')}", "c", True),
        (f"orders INNER JOIN customers ON {_KEYS.format(o='orders', c='customers')}", "orders", True),
        (f"orders AS o INNER JOIN customers AS c ON o.customer_id = c.customer_id", "o", False),
        (
            f"orders AS o INNER JOIN customers AS c ON {_KEYS.format(o='o', c='c')} AND c.name = %s",
            "o",
            False,
        ),
        (f"customers AS c INNER JOIN orders AS o ON {_KEYS.format(o='o', c='c')}", "o", False),
        (f"orders AS o INNER JOIN refunds AS r ON o.order_id = r.order_id", "o", False),
    ],
)
def test_grouped_customer_results_need_exactly_the_two_key_equalities(from_clause, qualifier, trusted) -> None:
    tools, _ = _tools()
    params = dict(_WINDOW_PARAMS)
    if "c.name = %s" in from_clause:
        params = {"0": "x", **{str(int(key) + 1): value for key, value in _WINDOW_PARAMS.items()}}
    arguments = _declared(_by_customer(from_clause, qualifier, ""), params, "gross_fen")
    if trusted:
        _, bindings, _ = _resolve(tools, arguments)
        assert [(item.result_position, item.grouped) for item in bindings] == [("gross_fen", True)]
    else:
        assert _error(_resolve, tools, arguments).code == "evidence_validation_failed"


def test_a_gross_amount_taken_from_another_table_is_refused() -> None:
    tools, _ = _tools()
    sql = (
        "SELECT COALESCE(SUM(r.amount_fen), 0) AS gross_fen FROM orders AS o INNER JOIN refunds AS r "
        "ON o.order_id = r.order_id WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s"
    )
    assert _error(_resolve, tools, _declared(sql, _WINDOW_PARAMS, "gross_fen")).code == "evidence_validation_failed"


# --------------------------------------------------------------------------
# A query the parser rejects is left to query_readonly, which reports it.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("metric, sql", [("gross_fen", "SELECT FROM"), ("net_fen", "SELECT FROM"), ("paid_count", "DELETE FROM orders")])
def test_a_declared_metric_with_unparsable_sql_gets_the_parser_error(metric, sql) -> None:
    tools, connection = _tools()
    arguments = _declared(sql, {}, metric)
    resolved = _resolve(tools, arguments)
    assert resolved[2] == (metric,)
    error = _error(call_tool, tools, "query_readonly", arguments, context=_context())
    assert error.code in {"invalid_sql", "statement_not_allowed"}
    assert connection.executed == []


def test_unparsable_sql_with_a_tenant_equality_text_is_left_unchanged() -> None:
    tools, _ = _tools()
    arguments, _, _ = _resolve(tools, {"sql": "SELECT FROM orders WHERE tenant_id = %s", "params": {"0": "B"}})
    assert dict(arguments["params"]) == {"0": "B"}


# --------------------------------------------------------------------------
# Approved execution re-checks what was stored.
# --------------------------------------------------------------------------


def _pending(**changes):
    call = {
        "tool": "query_readonly",
        "sql": "SELECT c.name FROM customers AS c WHERE c.customer_id = %s",
        "params": {"0": "c1"},
        "metrics": [],
        "time_window": None,
    }
    call.update(changes)
    return call


def test_an_approved_query_runs_through_the_scoped_renderer() -> None:
    tools, connection = _tools()
    evidence = execute_approved_query(tools, _pending(), context=_context())
    assert evidence.row_count == 1
    assert connection.executed[0][1] == ("A", "c1")


@pytest.mark.parametrize(
    "changes, code, message",
    [
        ({"tool": "search_catalog"}, "approval_action_invalid", "the approved action is not a query_readonly call"),
        ({"sql": 5}, "invalid_argument", "sql length is outside the allowed range"),
        ({"sql": "  "}, "invalid_argument", "sql length is outside the allowed range"),
        ({"sql": "S" * 4001}, "invalid_argument", "sql length is outside the allowed range"),
        ({"params": None}, "invalid_argument", "params must be a JSON object"),
        ({"params": {"1": "c1"}}, "invalid_argument", "params keys must start at zero without gaps"),
        ({"params": {"tenant_id": "B"}}, "reserved_parameter", "identity parameters are server-owned"),
        ({"sql": "SELECT FROM"}, "invalid_sql", "expected column name"),
        ({"sql": "SELECT * FROM invoices"}, "table_not_allowed", "table is not in the allowlist"),
        ({"params": {}}, "parameter_mismatch", "query parameter is missing"),
        ({"sql": "SELECT x.name FROM customers AS c"}, "unknown_qualifier", "column qualifier is not a known table alias"),
    ],
)
def test_a_tampered_approved_call_is_refused_with_its_own_code(changes, code, message) -> None:
    tools, connection = _tools()
    error = _error(execute_approved_query, tools, _pending(**changes), context=_context())
    assert (error.code, error.message) == (code, message)
    assert connection.executed == []


def test_an_approved_call_that_is_not_a_mapping_or_has_no_server_context_is_refused() -> None:
    tools, _ = _tools()
    assert _error(execute_approved_query, tools, "nope", context=_context()).code == "approval_action_invalid"
    unauthorized = _error(execute_approved_query, tools, _pending(), context="nope")
    assert (unauthorized.code, unauthorized.message) == (
        "unauthorized",
        "approved execution requires a server-created context",
    )


def test_an_approved_net_fen_call_is_refused_and_a_declared_gross_call_is_rebound() -> None:
    tools, connection = _tools()
    net = _pending(
        sql=_NET_SQL,
        params=_WINDOW_PARAMS,
        metrics=["net_fen"],
        time_window=dict(SEPTEMBER),
    )
    error = _error(execute_approved_query, tools, net, context=_context())
    assert (error.code, error.message) == (
        "invalid_binding",
        "net_fen controlled plan accepts only orders and refunds requests",
    )
    gross = _pending(sql=_GROSS_SQL, params=_WINDOW_PARAMS, metrics=["gross_fen"], time_window=dict(SEPTEMBER))
    evidence = execute_approved_query(tools, gross, context=_context())
    assert [(item.metric_id, item.result_position) for item in evidence.metric_bindings] == [("gross_fen", "gross_fen")]
    mismatch = _pending(
        sql=_GROSS_SQL.replace("status = %s", "status <> %s"),
        params=_WINDOW_PARAMS,
        metrics=["gross_fen"],
        time_window=dict(SEPTEMBER),
    )
    error = _error(execute_approved_query, tools, mismatch, context=_context())
    assert (error.code, error.message) == ("evidence_validation_failed", "approved query no longer matches its metric semantics")
    assert len(connection.executed) == 1


# --------------------------------------------------------------------------
# A controlled net_fen plan, and errors from the database driver.
# --------------------------------------------------------------------------


def test_a_declared_net_fen_query_runs_the_server_plan_with_two_executions() -> None:
    tools, connection = _tools()
    output = call_tool(tools, "query_readonly", _declared(_NET_SQL, _WINDOW_PARAMS, "net_fen"), context=_context())
    assert output["rows"] == [{"net_fen": 12000}]
    assert output["metric_plan_id"] == NET_FEN_PLAN_ID and output["plan_query_count"] == 2
    assert output["verified_metrics"] == [{"metric_id": "net_fen", "result_position": "net_fen"}]
    assert len(connection.executed) == 2
    assert all("A" in params for _, params in connection.executed)
    assert '"refunds"' in connection.executed[1][0] and '"refunds"' not in connection.executed[0][0]


def test_a_net_fen_plan_over_another_table_is_refused_before_it_runs() -> None:
    tools, connection = _tools()
    binding = build_metric_binding(load_default_catalog(), "net_fen", SEPTEMBER)
    error = _error(
        tools.query_readonly,
        {"sql": "SELECT COUNT(*) AS n FROM customers", "params": {}},
        context=_context(),
        metric_bindings=(binding,),
    )
    assert (error.code, error.message) == (
        "invalid_binding",
        "net_fen controlled plan accepts only orders and refunds requests",
    )
    two = _error(
        tools.query_readonly,
        {"sql": "SELECT COUNT(*) AS n FROM orders", "params": {}},
        context=_context(),
        metric_bindings=(binding, binding),
    )
    assert (two.code, two.message) == ("invalid_binding", "net_fen must have one metric binding")
    assert connection.executed == []


class _FailingConnection(_Connection):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def cursor(self, *, row_factory):
        raise self.error


@pytest.mark.parametrize(
    "error, code",
    [
        (psycopg.errors.SyntaxError("secret"), "invalid_sql"),
        (psycopg.errors.InsufficientPrivilege("secret"), "forbidden"),
        (psycopg.errors.QueryCanceled("secret"), "query_timeout"),
        (psycopg.OperationalError("secret"), "database_unavailable"),
        (psycopg.errors.DataException("secret"), "database_error"),
    ],
)
def test_driver_errors_become_fixed_codes_without_the_driver_text(error, code) -> None:
    connection = _FailingConnection(error)
    tools = ControlledTools(
        catalog=load_default_catalog(),
        executor=GuardedQueryExecutor(connect=lambda: connection, clock=lambda: CLOCK),
    )
    raised = _error(call_tool, tools, "query_readonly", {"sql": _COUNT, "params": {"0": "paid"}}, context=_context())
    assert raised.code == code and "secret" not in raised.message


def test_a_result_over_the_row_limit_has_its_own_code() -> None:
    class _Many(_Cursor):
        def fetchmany(self, size):
            return [{"paid_count": index} for index in range(size)]

    class _ManyConnection(_Connection):
        def cursor(self, *, row_factory):
            return _Many(self)

    connection = _ManyConnection()
    tools = ControlledTools(
        catalog=load_default_catalog(),
        executor=GuardedQueryExecutor(connect=lambda: connection, clock=lambda: CLOCK),
    )
    raised = _error(call_tool, tools, "query_readonly", {"sql": "SELECT order_id FROM orders", "params": {}}, context=_context())
    assert (raised.code, raised.message) == ("result_row_limit", "query returned more than the 100 row limit")
    approved = _error(execute_approved_query, tools, _pending(), context=_context())
    assert (approved.code, approved.message) == ("result_row_limit", "query returned more than the 100 row limit")


# --------------------------------------------------------------------------
# The default executor connects read-only and binds the tenant in the transaction.
# --------------------------------------------------------------------------


def test_the_default_executor_binds_the_tenant_and_a_custom_factory_does_not(monkeypatch) -> None:
    monkeypatch.setenv("QUERYSHIELD_DATABASE_URL", "postgresql://reader@localhost/queryshield_test")
    monkeypatch.delenv("QUERYSHIELD_DEMO_DATASET", raising=False)
    connection = _Connection()
    seen: list[dict[str, object]] = []

    def connect(url, **kwargs):
        seen.append(kwargs)
        return connection

    monkeypatch.setattr(psycopg, "connect", connect)
    GuardedQueryExecutor(clock=lambda: CLOCK).execute(_COUNT, context=_context(tenant_id="B"), params=("paid",))
    assert connection.statements == [("SELECT set_config('queryshield.tenant_id', %s, true)", ("B",))]
    assert seen[0]["options"] == "-c default_transaction_read_only=on -c statement_timeout=2000"
    assert connection.executed[0][1] == ("B", "paid")

    custom = _Connection()
    GuardedQueryExecutor(connect=lambda: custom, clock=lambda: CLOCK).execute(
        _COUNT, context=_context(tenant_id="B"), params=("paid",)
    )
    assert custom.statements == []
    assert guarded.connect_readonly.__name__ == "connect_readonly"


# --------------------------------------------------------------------------
# The retrieval word list: both search routes expand the same synonyms.
# --------------------------------------------------------------------------

_SEARCHES = ["销售额", "退款后净额", "订单数量", "营业额", "已支付订单数", "退款金额", "支付订单总额", "paid_count", "客户姓名", "__none__"]


def _ids(items) -> list[str]:
    return [item["id"] for item in items]


def test_the_catalog_fallback_search_expands_the_synonym_table() -> None:
    tools, _ = _tools()
    requester = {query: _ids(tools.search_catalog({"query": query, "top_k": 5}, context=_context())["items"]) for query in _SEARCHES}
    approver = {
        query: _ids(tools.search_catalog({"query": query, "top_k": 5}, context=_context(role="approver"))["items"])
        for query in _SEARCHES
    }
    assert requester == _FALLBACK_REQUESTER
    assert approver == _FALLBACK_APPROVER


def test_the_keyword_retriever_expands_the_same_synonym_table() -> None:
    snapshot = load_snapshot("fixtures/knowledge/snapshots/knowledge-v1-9f580dd7f887ed0a.json")
    retriever = KeywordSynonymRetriever(catalog=load_default_catalog(), snapshot=snapshot)
    ranked = {query: _ids(retriever.search(query, context=_context(), top_k=5).items) for query in _SEARCHES}
    assert ranked == _KEYWORD_RANKED


_FALLBACK_REQUESTER: dict[str, list[str]] = {
    '销售额': ['metric.net_fen', 'metric.gross_fen'],
    '退款后净额': ['metric.net_fen', 'metric.refund_fen', 'field.refunds.created_at'],
    '订单数量': ['metric.paid_count'],
    '营业额': ['metric.gross_fen', 'metric.net_fen'],
    '已支付订单数': ['metric.paid_count', 'field.orders.status', 'metric.gross_fen', 'metric.refund_fen'],
    '退款金额': ['metric.refund_fen', 'field.refunds.amount_fen', 'field.refunds.created_at', 'metric.net_fen'],
    '支付订单总额': ['metric.gross_fen', 'metric.net_fen'],
    'paid_count': ['metric.paid_count'],
    '客户姓名': [],
    '__none__': [],
}
_FALLBACK_APPROVER: dict[str, list[str]] = {
    '销售额': ['metric.net_fen', 'metric.gross_fen'],
    '退款后净额': ['metric.net_fen', 'metric.refund_fen', 'field.refunds.created_at'],
    '订单数量': ['metric.paid_count'],
    '营业额': ['metric.gross_fen', 'metric.net_fen'],
    '已支付订单数': ['metric.paid_count', 'field.orders.status', 'metric.gross_fen', 'metric.refund_fen'],
    '退款金额': ['metric.refund_fen', 'field.refunds.amount_fen', 'field.refunds.created_at', 'metric.net_fen'],
    '支付订单总额': ['metric.gross_fen', 'metric.net_fen'],
    'paid_count': ['metric.paid_count'],
    '客户姓名': ['field.customers.name'],
    '__none__': [],
}
_KEYWORD_RANKED: dict[str, list[str]] = {
    '销售额': ['metric.net_fen', 'semantic-ambiguity-sales@2026-09-21#0001', 'semantic-metric-net@2026-09-21#0001', 'semantic-metric-gross@2026-09-21#0001'],
    '退款后净额': ['metric.net_fen', 'semantic-metric-net@2026-09-21#0001', 'semantic-ambiguity-sales@2026-09-21#0001', 'semantic-metric-refund@2026-09-21#0001', 'semantic-refund-policy-v2@2026-09-20#0001'],
    '订单数量': ['metric.paid_count', 'semantic-metric-paid-count@2026-09-21#0001'],
    '营业额': ['metric.gross_fen', 'semantic-ambiguity-sales@2026-09-21#0001', 'semantic-metric-gross@2026-09-21#0001', 'semantic-metric-net@2026-09-21#0001'],
    '已支付订单数': ['metric.paid_count', 'semantic-metric-paid-count@2026-09-21#0001', 'semantic-metric-paid-count@2026-09-21#0000', 'semantic-metric-refund@2026-09-21#0001', 'tenant-a-orders-overview@2026-09-21#0001'],
    '退款金额': ['metric.refund_fen', 'semantic-metric-net@2026-09-21#0001', 'semantic-metric-refund@2026-09-21#0001', 'semantic-refund-policy-v2@2026-09-20#0001'],
    '支付订单总额': ['metric.gross_fen', 'semantic-ambiguity-sales@2026-09-21#0001', 'semantic-metric-gross@2026-09-21#0001', 'semantic-metric-net@2026-09-21#0001'],
    'paid_count': ['metric.paid_count', 'semantic-metric-paid-count@2026-09-21#0001'],
    '客户姓名': [],
    '__none__': [],
}


# --------------------------------------------------------------------------
# FactResolver: each check, in order, with its text.
# --------------------------------------------------------------------------


def _binding(metric: str = "gross_fen", **changes) -> MetricBinding:
    values = dict(
        metric_id=metric,
        result_position=metric,
        unit="count" if metric == "paid_count" else "CNY_fen",
        time_window=dict(SEPTEMBER_UTC),
        catalog_source_id="commerce-v1",
        catalog_version="catalog-v4",
        plan_id=NET_FEN_PLAN_ID if metric == "net_fen" else None,
    )
    values.update(changes)
    return MetricBinding(**values)


def _evidence(metric: str = "gross_fen", *, rows=None, binding=None, **changes) -> ResultEvidence:
    context = _context()
    evidence = ResultEvidence.from_server_execution(
        context,
        result_id="result-1",
        rows=rows if rows is not None else ({metric: 1200},),
        normalized_query="SELECT 1",
        params={},
        observed_at=CLOCK,
        policy_version="qs-sql-v1",
        catalog_version="catalog-v4",
        metric_bindings=(binding if binding is not None else _binding(metric),),
        metric_plan_id=NET_FEN_PLAN_ID if metric == "net_fen" else None,
    )
    return replace(evidence, **changes) if changes else evidence


def _resolve_facts(refs, evidences, *, context=None):
    return FactResolver().resolve(refs, context=context or _context(), evidences=evidences)


def _fact_error(refs, evidences, **kwargs) -> tuple[str, str]:
    with pytest.raises(FactResolutionError) as caught:
        _resolve_facts(refs, evidences, **kwargs)
    return caught.value.code, str(caught.value)


REF = FactRef("result-1", "gross_fen")


def test_facts_come_from_the_bound_column_with_a_catalog_label_and_display_value() -> None:
    envelope = _resolve_facts((REF, FactRef("result-2", "paid_count")), {
        "result-1": _evidence("gross_fen", rows=({"gross_fen": 1250},)),
        "result-2": replace(_evidence("paid_count", rows=({"paid_count": 7},)), result_id="result-2"),
    })
    assert [(fact.metric_id, fact.label, fact.value, fact.unit, fact.display_value) for fact in envelope.facts] == [
        ("gross_fen", "支付订单总额", 1250, "CNY_fen", "12.50元"),
        ("paid_count", "已支付订单数", 7, "count", "7笔"),
    ]
    assert all(fact.fact_id.startswith("fact-") and len(fact.fact_id) == 29 for fact in envelope.facts)
    refund = _evidence("refund_fen", rows=({"refund_fen": 300},))
    assert _resolve_facts((FactRef("result-1", "metric.refund_fen"),), {"result-1": refund}).facts[0].display_value == "3.00元"
    net = _evidence("net_fen", rows=({"net_fen": -250},))
    assert _resolve_facts((FactRef("result-1", "net_fen"),), {"result-1": net}).facts[0].display_value == "-2.50元"


@pytest.mark.parametrize(
    "refs, evidences, expected",
    [
        (5, {}, ("invalid_fact_refs", "invalid_fact_refs: fact_refs must be a sequence")),
        (tuple(FactRef(f"result-{i}", "gross_fen") for i in range(11)), {}, ("invalid_fact_refs", "invalid_fact_refs: fact_refs has too many items")),
        (("ref",), {}, ("invalid_fact_refs", "invalid_fact_refs: fact_refs must be server-parsed references")),
        ((FactRef("result-1", "unknown_metric"),), {}, ("evidence_validation_failed", "evidence_validation_failed: metric is not in the trusted catalog")),
        ((REF, REF), {"result-1": _evidence()}, ("invalid_fact_refs", "invalid_fact_refs: fact_refs must be deduplicated")),
        ((REF,), {}, ("evidence_validation_failed", "evidence_validation_failed: result evidence was not found")),
        ((REF,), {"result-1": "not evidence"}, ("evidence_validation_failed", "evidence_validation_failed: result evidence was not found")),
        ((REF,), {"result-1": _evidence(result_id="result-9")}, ("evidence_validation_failed", "evidence_validation_failed: result identity does not match")),
        ((REF,), {"result-1": _evidence(run_id="run-other")}, ("evidence_validation_failed", "evidence_validation_failed: result evidence is outside the current subject")),
        ((REF,), {"result-1": _evidence(tenant_id="B")}, ("evidence_validation_failed", "evidence_validation_failed: result evidence is outside the current subject")),
        ((REF,), {"result-1": _evidence(principal_id="other")}, ("evidence_validation_failed", "evidence_validation_failed: result evidence is outside the current subject")),
        ((REF,), {"result-1": _evidence(catalog_version="catalog-v3")}, ("evidence_validation_failed", "evidence_validation_failed: result uses an incompatible catalog version")),
        ((REF,), {"result-1": _evidence(row_count=2)}, ("evidence_validation_failed", "evidence_validation_failed: a metric fact needs one aggregate result row")),
        ((REF,), {"result-1": _evidence(rows=({"gross_fen": 1}, {"gross_fen": 2}))}, ("evidence_validation_failed", "evidence_validation_failed: a metric fact needs one aggregate result row")),
        ((REF,), {"result-1": _evidence(metric_bindings=())}, ("evidence_validation_failed", "evidence_validation_failed: result has no trusted binding for the metric")),
        ((REF,), {"result-1": _evidence(binding=_binding("paid_count"))}, ("evidence_validation_failed", "evidence_validation_failed: result has no trusted binding for the metric")),
        ((REF,), {"result-1": _evidence(binding=_binding(grouped=True))}, ("evidence_validation_failed", "evidence_validation_failed: a grouped result is a rowset, not a metric fact")),
        ((REF,), {"result-1": _evidence(binding=_binding(catalog_source_id="other-v1"))}, ("evidence_validation_failed", "evidence_validation_failed: metric binding does not match the catalog")),
        ((REF,), {"result-1": _evidence(binding=_binding(catalog_version="catalog-v3"))}, ("evidence_validation_failed", "evidence_validation_failed: metric binding does not match the catalog")),
        ((REF,), {"result-1": _evidence(binding=_binding(unit="count"))}, ("evidence_validation_failed", "evidence_validation_failed: metric binding does not match the catalog")),
        ((REF,), {"result-1": _evidence(binding=_binding(result_position=3))}, ("evidence_validation_failed", "evidence_validation_failed: metric binding does not match the catalog")),
        ((REF,), {"result-1": _evidence(rows=({"other": 5},))}, ("evidence_validation_failed", "evidence_validation_failed: bound result column is absent")),
        ((REF,), {"result-1": _evidence(rows=({"gross_fen": "5"},))}, ("evidence_validation_failed", "evidence_validation_failed: fact values must be integer count or fen")),
        ((REF,), {"result-1": _evidence(rows=({"gross_fen": True},))}, ("evidence_validation_failed", "evidence_validation_failed: fact values must be integer count or fen")),
        ((REF,), {"result-1": _evidence(rows=({"gross_fen": 5.0},))}, ("evidence_validation_failed", "evidence_validation_failed: fact values must be integer count or fen")),
        ((REF,), {"result-1": _evidence(rows=({"gross_fen": -1},))}, ("evidence_validation_failed", "evidence_validation_failed: raw money facts cannot be negative")),
        ((FactRef("result-1", "refund_fen"),), {"result-1": _evidence("refund_fen", rows=({"refund_fen": -1},))}, ("evidence_validation_failed", "evidence_validation_failed: raw money facts cannot be negative")),
        ((FactRef("result-1", "paid_count"),), {"result-1": _evidence("paid_count", rows=({"paid_count": -1},))}, ("evidence_validation_failed", "evidence_validation_failed: count facts cannot be negative")),
        ((FactRef("result-1", "net_fen"),), {"result-1": _evidence("net_fen", binding=_binding("net_fen", plan_id="other-plan"))}, ("evidence_validation_failed", "evidence_validation_failed: net_fen binding does not match the controlled metric plan")),
        ((FactRef("result-1", "net_fen"),), {"result-1": replace(_evidence("net_fen"), metric_plan_id=None)}, ("evidence_validation_failed", "evidence_validation_failed: net_fen binding does not match the controlled metric plan")),
    ],
)
def test_fact_resolution_refuses_each_untrusted_reference_with_its_text(refs, evidences, expected) -> None:
    assert _fact_error(refs, evidences) == expected


def test_fact_resolution_needs_a_server_created_context() -> None:
    assert _fact_error((REF,), {"result-1": _evidence()}, context="not a context") == (
        "unauthorized",
        "unauthorized: facts require a server-created execution context",
    )


def test_fact_resolution_checks_run_ownership_before_the_shape_and_the_shape_before_the_binding() -> None:
    both = _evidence(run_id="run-other", row_count=2, metric_bindings=())
    assert _fact_error((REF,), {"result-1": both})[1].endswith("outside the current subject")
    shape = _evidence(row_count=2, metric_bindings=())
    assert _fact_error((REF,), {"result-1": shape})[1].endswith("needs one aggregate result row")
    version = _evidence(catalog_version="catalog-v3", row_count=2)
    assert _fact_error((REF,), {"result-1": version})[1].endswith("incompatible catalog version")
    grouped_and_wrong_unit = _evidence(binding=_binding(grouped=True, unit="count"))
    assert _fact_error((REF,), {"result-1": grouped_and_wrong_unit})[1].endswith("not a metric fact")


# --------------------------------------------------------------------------
# The durable parallel scheduler: what each branch runs, and how a group recovers.
# --------------------------------------------------------------------------


class _Recording:
    """An executor that records each call and runs it against the fake connection."""

    calls: list[tuple[str, tuple[object, ...], list[dict[str, object]]]] = []

    def __init__(self) -> None:
        self._inner = GuardedQueryExecutor(connect=lambda: _Connection(), clock=lambda: CLOCK)

    def execute(self, sql, *, context, params=(), metric_bindings=()):
        self.calls.append((sql, tuple(params), [binding.as_dict() for binding in metric_bindings]))
        return self._inner.execute(sql, context=context, params=params, metric_bindings=metric_bindings)


def _run_row(state: StateStore) -> ExecutionContext:
    context = _context()
    state.create_run(
        run_id=context.run_id,
        tenant_id=context.tenant_id,
        principal_id=context.principal_id,
        role=context.role,
        question="q",
        mode="fake",
    )
    return context


_GROSS_BRANCH_SQL = (
    "SELECT COALESCE(SUM(o.amount_fen), 0) AS gross_fen FROM orders AS o "
    "WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s"
)
_PAID_PARAMS = ("paid", SEPTEMBER["start"], SEPTEMBER["end"])


def _expected_binding(metric: str, unit: str) -> dict[str, object]:
    return {
        "metric_id": metric,
        "result_position": metric,
        "unit": unit,
        "time_window": dict(SEPTEMBER_UTC),
        "catalog_source_id": "commerce-v1",
        "catalog_version": "catalog-v4",
        "plan_id": None,
    }


def test_each_durable_branch_runs_its_fixed_query_with_its_own_binding() -> None:
    _Recording.calls = []
    with StateStore(":memory:") as state:
        context = _run_row(state)
        scheduler = DurableParallelScheduler(state=state, executor_factory=_Recording)
        result = scheduler.run(context, ("net_fen", "gross_fen", "paid_count"), time_window=SEPTEMBER_UTC)
        assert (result.status, result.sql_exec_count, result.new_branch_count, result.reused) == ("SUCCEEDED", 4, 3, False)
        assert result.plan_hash == "139c9d78dedfc06a3e9f795c3e2343b9349520c4e2a7336b5ad3a961a1c50ea2"
        by_sql = {sql: (params, bindings) for sql, params, bindings in _Recording.calls if bindings}
        assert by_sql == {
            _GROSS_BRANCH_SQL: (_PAID_PARAMS, [_expected_binding("gross_fen", "CNY_fen")]),
            "SELECT COUNT(*) AS paid_count FROM orders AS o WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s": (
                _PAID_PARAMS,
                [_expected_binding("paid_count", "count")],
            ),
        }
        plan_calls = sorted(sql for sql, _, bindings in _Recording.calls if not bindings)
        assert len(plan_calls) == 2 and "refund_fen" in plan_calls[1] and "gross_fen" in plan_calls[0]
        group = state.get_parallel_group(context.run_id)
        assert group["status"] == "SUCCEEDED"
        assert [branch["metric_id"] for branch in group["branches"]] == ["gross_fen", "net_fen", "paid_count"]
        assert sorted(group["summary"]) == ["branches", "sql_exec_count"]
        assert [branch["metric_id"] for branch in group["summary"]["branches"]] == ["gross_fen", "net_fen", "paid_count"]
        assert group["summary"]["sql_exec_count"] == 4
        replay = scheduler.run(context, ("paid_count", "gross_fen", "net_fen"), time_window=SEPTEMBER_UTC)
        assert (replay.status, replay.reused, replay.new_branch_count, replay.sql_exec_count) == ("SUCCEEDED", True, 0, 0)
        conflict = pytest.raises(DurableParallelError, scheduler.run, context, ("paid_count", "gross_fen"), time_window=SEPTEMBER_UTC)
        assert conflict.value.code == "parallel_plan_conflict"


def test_a_failed_durable_branch_fails_the_group_with_its_error_code() -> None:
    class _Failing:
        def execute(self, sql, **kwargs):
            raise ToolError("branch_boom", "boom")

    with StateStore(":memory:") as state:
        context = _run_row(state)
        result = DurableParallelScheduler(state=state, executor_factory=_Failing).run(
            context, ("gross_fen", "paid_count"), time_window=SEPTEMBER_UTC
        )
        assert result.status == "FAILED" and result.new_branch_count == 0 and result.sql_exec_count == 0
        group = state.get_parallel_group(context.run_id)
        assert group["summary"] == {"errors": ["branch_boom", "branch_boom"]}
        assert [(branch["status"], branch["error_code"]) for branch in group["branches"]] == [("FAILED", "branch_boom")] * 2


@pytest.mark.parametrize(
    "metrics, window, code",
    [
        (("gross_fen",), SEPTEMBER_UTC, "invalid_parallel_action"),
        (("gross_fen", "gross_fen"), SEPTEMBER_UTC, "invalid_parallel_action"),
        (("gross_fen", "refund_fen"), SEPTEMBER_UTC, "invalid_parallel_action"),
        (("gross_fen", "net_fen"), SEPTEMBER, "invalid_parallel_plan"),
    ],
)
def test_the_durable_plan_rejects_a_bad_metric_set_or_window(metrics, window, code) -> None:
    with StateStore(":memory:") as state:
        error = pytest.raises(
            DurableParallelError,
            DurableParallelScheduler(state=state, executor_factory=_Recording).run,
            _context(),
            metrics,
            time_window=window,
        )
        assert error.value.code == code
        assert pytest.raises(DurableParallelError, DurableParallelScheduler(state=state).run, "not a context", metrics).value.code == "unauthorized"


def _persisted_group(state: StateStore, statuses: dict[str, str]) -> ExecutionContext:
    context = _run_row(state)
    plan, plan_hash = _plan(context, list(statuses), SEPTEMBER_UTC)
    state.create_parallel_group(
        group_id="parallel-1", run_id=context.run_id, plan_hash=plan_hash, plan=plan, metric_ids=sorted(statuses)
    )
    group = state.get_parallel_group(context.run_id)
    for branch in group["branches"]:
        status = statuses[branch["metric_id"]]
        result = {"metric_id": branch["metric_id"], "rows": [{branch["metric_id"]: 1}]} if status == "SUCCEEDED" else None
        state.update_parallel_branch("parallel-1", branch["branch_id"], status=status, result=result)
    return context


def test_startup_recovery_reuses_a_fully_persisted_group_and_fails_an_uncertain_one() -> None:
    with StateStore(":memory:") as state:
        context = _persisted_group(state, {"gross_fen": "SUCCEEDED", "paid_count": "SUCCEEDED"})
        report = DurableParallelScheduler(state=state).recover_on_startup()
        assert report == {
            "recovered_submitted": [context.run_id],
            "failed_uncertain": [],
            "reconciled_terminal": [],
            "executor_calls": 0,
        }
        group = state.get_parallel_group(context.run_id)
        assert group["status"] == "SUCCEEDED" and group["summary"]["recovered"] is True
        run = state.get_run(context.run_id)
        assert run["status"] == "SUCCEEDED" and run["error_code"] is None
        assert [event["type"] for event in state.events(context.run_id)] == ["accepted", "terminal"]

    with StateStore(":memory:") as state:
        context = _persisted_group(state, {"gross_fen": "SUCCEEDED", "paid_count": "RUNNING"})
        report = DurableParallelScheduler(state=state).recover_on_startup()
        assert report["failed_uncertain"] == [context.run_id] and report["recovered_submitted"] == []
        group = state.get_parallel_group(context.run_id)
        assert group["status"] == "FAILED"
        assert group["summary"] == {"error_code": "recovery_required", "recovered": False}
        assert {branch["metric_id"]: (branch["status"], branch["error_code"]) for branch in group["branches"]} == {
            "gross_fen": ("SUCCEEDED", None),
            "paid_count": ("FAILED", "recovery_required"),
        }
        run = state.get_run(context.run_id)
        assert run["status"] == "FAILED" and run["error_code"] == "recovery_required"
        again = DurableParallelScheduler(state=state).recover_on_startup()
        assert again["reconciled_terminal"] == [context.run_id] and again["failed_uncertain"] == []


def test_running_again_after_a_crash_reports_recovery_required() -> None:
    with StateStore(":memory:") as state:
        context = _persisted_group(state, {"gross_fen": "SUCCEEDED", "paid_count": "RUNNING"})
        scheduler = DurableParallelScheduler(state=state, executor_factory=_Recording)
        error = pytest.raises(
            DurableParallelError, scheduler.run, context, ("gross_fen", "paid_count"), time_window=SEPTEMBER_UTC
        )
        assert error.value.code == "recovery_required"
        assert state.get_parallel_group(context.run_id)["status"] == "FAILED"
        assert state.get_run(context.run_id)["status"] == "FAILED"


# --------------------------------------------------------------------------
# The evaluation sidecar record of a retrieval: its keys, their order and the visibility verdicts.
# --------------------------------------------------------------------------

_SNAPSHOT_PATH = "fixtures/knowledge/snapshots/knowledge-v1-9f580dd7f887ed0a.json"
_RECORD_KEYS = [
    "run_id",
    "tenant_id",
    "principal_id",
    "role",
    "retrieval_id",
    "snapshot_id",
    "query_sha256",
    "strategy_version",
    "selected_ids",
    "items",
    "acl_basis",
]
_ITEM_KEYS = ["id", "source_id", "version", "text_sha256", "source_kind", "visibility_check"]
_VISIBLE = {
    "source_active": True,
    "version_matches_snapshot": True,
    "tenant_visible": True,
    "role_visible": True,
    "passed": True,
}


def _hybrid_retriever():
    from queryshield.knowledge.index import build_embedding_index
    from queryshield.knowledge.retrieval import HybridRetriever
    from queryshield.providers.embedding import FixedEmbedding

    snapshot = load_snapshot(_SNAPSHOT_PATH)
    vectors = {chunk.text: (1.0, 0.0, 0.0) if "gross" in chunk.source_id else (0.0, 1.0, 0.0) for chunk in snapshot.chunk_records}
    vectors["营业额"] = (1.0, 0.0, 0.0)
    embedder = FixedEmbedding(vectors, model_revision="fixed-rules-v1", dimensions=3)
    build = build_embedding_index(snapshot, embedder, ingest_job_id="ingest-rules-test")
    return HybridRetriever(catalog=load_default_catalog(), snapshot=build.snapshot, index=build.index, embedder=embedder)


def test_the_retrieval_sidecar_record_keeps_its_keys_order_and_verdicts() -> None:
    import hashlib

    tools = ControlledTools(retriever=_hybrid_retriever())
    context = _context()
    output = tools.search_catalog({"query": "营业额", "top_k": 5}, context=context)
    (record,) = tools._retrieval_return_records.values()
    assert list(record) == _RECORD_KEYS
    assert (record["run_id"], record["tenant_id"], record["principal_id"], record["role"]) == (
        "run-rules",
        "A",
        "principal-A",
        "requester",
    )
    assert record["strategy_version"] == "hybrid-v1"
    assert record["acl_basis"] == (
        "returned by HybridRetriever after server identity, role, tenant, source-status, and version filtering"
    )
    assert record["selected_ids"] == [item["id"] for item in output["items"]]
    assert [item["id"] for item in record["items"]] == [item["id"] for item in output["items"]]
    kinds = {item["id"]: item["source_kind"] for item in record["items"]}
    assert kinds["metric.gross_fen"] == "catalog_retrieval_candidate"
    assert {kind for id_, kind in kinds.items() if id_ != "metric.gross_fen"} == {"knowledge_document"}
    for item, returned in zip(record["items"], output["items"]):
        assert list(item) == _ITEM_KEYS
        assert item["text_sha256"] == hashlib.sha256(returned["text"].encode("utf-8")).hexdigest()
        assert (item["source_id"], item["version"]) == (returned["source_id"], returned["version"])
        assert list(item["visibility_check"]) == [
            "candidate_type",
            "source_active",
            "version_matches_snapshot",
            "tenant_visible",
            "role_visible",
            "passed",
        ]
        assert item["visibility_check"] == {"candidate_type": item["source_kind"], **_VISIBLE}


def test_candidates_the_retriever_cannot_account_for_are_recorded_as_unknown() -> None:
    snapshot = load_snapshot(_SNAPSHOT_PATH)
    retriever = KeywordSynonymRetriever(catalog=load_default_catalog(), snapshot=snapshot)
    tools = ControlledTools(retriever=retriever)
    tools.search_catalog({"query": "已支付订单数", "top_k": 5}, context=_context())
    (record,) = tools._retrieval_return_records.values()
    verdicts = {item["id"]: (item["source_kind"], item["visibility_check"]) for item in record["items"]}
    assert verdicts["metric.paid_count"] == ("catalog_retrieval_candidate", {"candidate_type": "catalog_retrieval_candidate", **_VISIBLE})
    unknown = {
        "candidate_type": "unknown",
        "source_active": False,
        "version_matches_snapshot": False,
        "tenant_visible": False,
        "role_visible": False,
        "passed": False,
    }
    assert [verdict for id_, verdict in verdicts.items() if id_ != "metric.paid_count"] == [("retriever_candidate", unknown)] * 4


# --------------------------------------------------------------------------
# Gaps the first tidy-up pass found: parser aliases, SUM(*), the net_fen binding, tenant mentions, failed groups.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql, code, message",
    [
        (
            "SELECT COUNT(*) AS n FROM orders INNER JOIN refunds AS orders ON orders.order_id = refunds.order_id",
            "invalid_sql",
            "table aliases must be unique",
        ),
        (
            "SELECT COUNT(*) AS n FROM orders AS x INNER JOIN refunds AS x ON x.order_id = x.order_id",
            "invalid_sql",
            "table aliases must be unique",
        ),
        (
            "SELECT COUNT(*) AS n FROM orders INNER JOIN orders ON orders.order_id = orders.order_id",
            "invalid_sql",
            "table aliases must be unique",
        ),
        ("SELECT SUM(*) AS n FROM orders", "invalid_sql", "SUM does not accept wildcard input"),
        ("SELECT COUNT(amount_fen, 1) AS n FROM orders", "invalid_sql", "COUNT received the wrong number of arguments"),
        ("SELECT COALESCE(amount_fen) AS n FROM orders", "invalid_sql", "COALESCE received the wrong number of arguments"),
        ("SELECT AVG(amount_fen) AS n FROM orders", "function_not_allowed", "function is not in the allowlist"),
    ],
)
def test_the_parser_rejects_duplicate_names_and_malformed_aggregates(sql, code, message) -> None:
    with pytest.raises(SQLPolicyError) as caught:
        parse_readonly_select(sql)
    assert (caught.value.code, str(caught.value)) == (code, f"{code}: {message}")


def test_an_invalid_net_fen_binding_does_not_trigger_the_server_plan() -> None:
    catalog = load_default_catalog()
    valid = build_metric_binding(catalog, "net_fen", SEPTEMBER)
    invalid = [
        replace(valid, plan_id="other-plan"),
        replace(valid, result_position="other"),
        replace(valid, unit="count"),
        replace(valid, catalog_source_id="other-v1"),
        replace(valid, catalog_version="catalog-v3"),
        replace(valid, time_window={**SEPTEMBER, "timezone": "CST"}),
    ]
    for binding in invalid:
        tools, connection = _tools()
        response = tools.query_readonly(
            {"sql": _COUNT, "params": {"0": "paid"}}, context=_context(), metric_bindings=(binding,)
        )
        assert "metric_plan_id" not in response and len(connection.executed) == 1
    tools, connection = _tools()
    response = tools.query_readonly({"sql": _NET_SQL, "params": _WINDOW_PARAMS}, context=_context(), metric_bindings=(valid,))
    assert response["metric_plan_id"] == NET_FEN_PLAN_ID and len(connection.executed) == 2


def test_a_net_fen_fact_needs_a_utc_window_in_its_binding() -> None:
    evidence = _evidence("net_fen", binding=_binding("net_fen", time_window={**SEPTEMBER_UTC, "timezone": "CST"}))
    assert _fact_error((FactRef("result-1", "net_fen"),), {"result-1": evidence}) == (
        "evidence_validation_failed",
        "evidence_validation_failed: net_fen binding does not match the controlled metric plan",
    )


@pytest.mark.parametrize(
    "question, tenant, foreign",
    [
        ("tenant-B 的订单", "A", ["B"]),
        ("tenant B orders", "A", ["B"]),
        ("租户B的订单", "A", ["B"]),
        ("tenant-B 的订单", "B", []),
        ("tenant-B 的订单", "tenant-B", []),
        ("租户B的订单", "tenant-B", []),
        ("租户 b 的订单", "B", []),
        ("tenant-C and 租户D", "tenant-A", ["C", "D"]),
        ("tenant-C tenant-c", "A", ["C"]),
        ("没有提到租户", "A", []),
        ("tenant-B", "", []),
    ],
)
def test_a_question_naming_another_tenant_is_recognized(question, tenant, foreign) -> None:
    from queryshield.agent.tenant_scope import explicit_foreign_tenant_mentions, has_explicit_foreign_tenant

    assert list(explicit_foreign_tenant_mentions(question, tenant)) == foreign
    assert has_explicit_foreign_tenant(question, tenant) is bool(foreign)
    assert explicit_foreign_tenant_mentions(5, tenant) == () and explicit_foreign_tenant_mentions(question, None) == ()


def test_the_in_memory_scheduler_reuses_a_group_that_has_a_failed_branch() -> None:
    from queryshield.agent import BranchExecution, ParallelReadonlyAction, ParallelScheduler

    calls: list[str] = []

    def runner(context, metric_id, branch_id):
        calls.append(metric_id)
        if metric_id == "net_fen":
            raise RuntimeError("boom")
        return BranchExecution(metric_id=metric_id, result_id=f"result-{metric_id}", rows=({metric_id: 1},), observed_at="2026-09-21T00:00:00Z")

    context = _context()
    plan = ParallelPlan.from_context(context, ("gross_fen", "net_fen"), time_window=SEPTEMBER_UTC)
    scheduler = ParallelScheduler(runner)
    action = ParallelReadonlyAction(("gross_fen", "net_fen"))
    first = scheduler.run(context, action, plan=plan)
    assert first.status == "FAILED" and first.new_branch_count == 2 and first.reused is False
    assert [(branch.metric_id, branch.status, branch.error_code) for branch in first.branches] == [
        ("gross_fen", "SUCCEEDED", None),
        ("net_fen", "FAILED", "branch_failed"),
    ]
    second = scheduler.run(context, action, plan=plan)
    assert (second.status, second.reused, second.new_branch_count) == ("FAILED", True, 0)
    assert [branch.branch_id for branch in second.branches] == [branch.branch_id for branch in first.branches]
    assert sorted(calls) == ["gross_fen", "net_fen"]


# --------------------------------------------------------------------------
# Window end written with the value on the left, parameters beyond ten, and the order of the argument checks.
# --------------------------------------------------------------------------

_COUNT_PAID = "SELECT COUNT(*) AS paid_count FROM orders WHERE {where}"
_NET_WHERE_SQL = "SELECT COALESCE(SUM(amount_fen), 0) AS net_fen FROM orders WHERE {where}"
_ELEVEN_WHERE = " AND ".join(["status = %s"] * 9 + ["created_at >= %s", "created_at < %s"])


def _eleven_params(*, swap_second_and_tenth: bool = False) -> dict[str, object]:
    values = ["paid"] * 9 + [SEPTEMBER["start"], SEPTEMBER["end"]]
    if swap_second_and_tenth:
        values[1], values[9] = values[9], values[1]
    return {str(index): value for index, value in enumerate(values)}


@pytest.mark.parametrize(
    "template, metric",
    [(_gross("{where}"), "gross_fen"), (_COUNT_PAID, "paid_count")],
)
def test_a_value_on_the_left_of_less_than_is_not_the_window_end(template, metric) -> None:
    tools, _ = _tools()
    # %s < created_at means created_at > end, which is not the half-open window's end.
    where = "status = %s AND created_at >= %s AND %s < created_at"
    arguments = _declared(template.format(where=where), dict(_WINDOW_PARAMS), metric)
    error = _error(_resolve, tools, arguments)
    assert (error.code, error.message) == _PROJECTION_ERROR
    accepted = _declared(template.format(where="status = %s AND created_at >= %s AND %s > created_at"), dict(_WINDOW_PARAMS), metric)
    _, bindings, _ = _resolve(tools, accepted)
    assert bindings[0].result_position == metric


def test_a_declared_net_fen_query_does_not_take_a_value_on_the_left_of_less_than_as_the_window_end() -> None:
    tools, _ = _tools()
    where = "status = %s AND created_at >= %s AND %s < created_at"
    arguments = _declared(_NET_WHERE_SQL.format(where=where), dict(_WINDOW_PARAMS), "net_fen")
    error = _error(_resolve, tools, arguments)
    assert (error.code, error.message) == _NET_ERROR


@pytest.mark.parametrize(
    "template, metric",
    [(_gross("{where}"), "gross_fen"), (_COUNT_PAID, "paid_count")],
)
def test_eleven_parameters_are_read_by_numeric_index_when_binding_a_metric(template, metric) -> None:
    tools, _ = _tools()
    sql = template.format(where=_ELEVEN_WHERE)
    _, bindings, _ = _resolve(tools, _declared(sql, _eleven_params(), metric))
    assert [(item.metric_id, item.result_position) for item in bindings] == [(metric, metric)]
    swapped = _error(_resolve, tools, _declared(sql, _eleven_params(swap_second_and_tenth=True), metric))
    assert (swapped.code, swapped.message) == _PROJECTION_ERROR


def test_eleven_parameters_are_read_by_numeric_index_when_checking_a_declared_net_fen_query() -> None:
    tools, _ = _tools()
    sql = _NET_WHERE_SQL.format(where=_ELEVEN_WHERE)
    _, bindings, _ = _resolve(tools, _declared(sql, _eleven_params(), "net_fen"))
    assert bindings[0].plan_id == NET_FEN_PLAN_ID
    swapped = _error(_resolve, tools, _declared(sql, _eleven_params(swap_second_and_tenth=True), "net_fen"))
    assert (swapped.code, swapped.message) == _NET_ERROR


def test_the_parameter_object_is_checked_before_the_sql_is_parsed() -> None:
    tools, connection = _tools()
    arguments = {"sql": "SELEC 1", "params": {"tenant_id": "B"}}
    error = _error(tools.query_readonly, arguments, context=_context())
    assert (error.code, error.message) == ("reserved_parameter", "identity parameters are server-owned")
    approved = _error(execute_approved_query, tools, _pending(**arguments), context=_context())
    assert (approved.code, approved.message) == ("reserved_parameter", "identity parameters are server-owned")
    assert connection.executed == []
