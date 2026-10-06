import psycopg
from queryshield.db.readonly import bind_transaction_tenant, connect_readonly


TABLES = ("customers", "orders", "refunds")
COUNT_SQL = """
    SELECT
        (SELECT COUNT(*) FROM customers),
        (SELECT COUNT(*) FROM orders),
        (SELECT COUNT(*) FROM refunds)
"""
# commerce-v1 rows per tenant (customers, orders, refunds); their sum is the fixture (3, 4, 3).
TENANT_FIXTURE_COUNTS = {"A": (2, 3, 2), "B": (1, 1, 1)}


def tenant_bound_counts(conn) -> dict[str, tuple[int, int, int]]:
    """Row counts per tenant, read with the server-bound tenant context.

    The tables force row-level security keyed on the transaction's
    tenant, so an unbound read sees no rows; each count binds its tenant first,
    the way the product does.  Each read runs in its own transaction.
    """

    counts: dict[str, tuple[int, int, int]] = {}
    for tenant_id in TENANT_FIXTURE_COUNTS:
        conn.rollback()
        bind_transaction_tenant(conn, tenant_id)
        counts[tenant_id] = tuple(conn.execute(COUNT_SQL).fetchone())
    conn.rollback()
    return counts


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def on_off(value: bool) -> str:
    return "on" if value else "off"


def reject_write_detail(conn, statement: str, params: tuple[object, ...], *, prepare=None) -> dict[str, str] | None:
    """Run one write probe and expose its PostgreSQL class and SQLSTATE."""

    conn.rollback()
    if prepare is not None:
        prepare(conn)
    try:
        with conn.transaction():
            conn.execute(statement, params)
    except Exception as exc:
        return {"exception": type(exc).__name__, "sqlstate": str(getattr(exc, "sqlstate", ""))}
    return None


def reject_write(conn, statement: str, params: tuple[object, ...]) -> str | None:
    """Run one write probe and return the observed database error class."""

    detail = reject_write_detail(conn, statement, params)
    return detail["exception"] if detail is not None else None


def main() -> int:
    try:
        with connect_readonly() as conn:
            row = conn.execute(
                """
                SELECT
                    current_database(),
                    current_user,
                    current_setting('server_version'),
                    current_setting('default_transaction_read_only'),
                    current_setting('transaction_read_only'),
                    pg_backend_pid(),
                    r.rolcanlogin,
                    r.rolsuper,
                    r.rolcreatedb,
                    r.rolcreaterole,
                    r.rolbypassrls,
                    has_table_privilege(current_user, 'public.customers', 'SELECT'),
                    has_table_privilege(current_user, 'public.orders', 'SELECT'),
                    has_table_privilege(current_user, 'public.refunds', 'SELECT'),
                    has_table_privilege(current_user, 'public.customers', 'UPDATE'),
                    has_table_privilege(current_user, 'public.orders', 'UPDATE'),
                    has_table_privilege(current_user, 'public.refunds', 'UPDATE'),
                    has_table_privilege(current_user, 'public.customers', 'INSERT'),
                    has_table_privilege(current_user, 'public.orders', 'INSERT'),
                    has_table_privilege(current_user, 'public.refunds', 'INSERT')
                FROM pg_roles AS r
                WHERE r.rolname = current_user
                """
            ).fetchone()

            require(row is not None, "current role was not found")
            (
                database_name,
                user_name,
                server_version,
                default_read_only,
                transaction_read_only,
                backend_pid,
                can_login,
                is_superuser,
                can_create_db,
                can_create_role,
                bypass_rls,
                customer_select,
                order_select,
                refund_select,
                customer_update,
                order_update,
                refund_update,
                customer_insert,
                order_insert,
                refund_insert,
            ) = row

            privileges = {
                "customers": {
                    "select": customer_select,
                    "update": customer_update,
                    "insert": customer_insert,
                },
                "orders": {
                    "select": order_select,
                    "update": order_update,
                    "insert": order_insert,
                },
                "refunds": {
                    "select": refund_select,
                    "update": refund_update,
                    "insert": refund_insert,
                },
            }
            for table in TABLES:
                table_privileges = privileges[table]
                print(
                    "table="
                    f"{table} select={on_off(table_privileges['select'])} "
                    f"update={on_off(table_privileges['update'])} "
                    f"insert={on_off(table_privileges['insert'])}"
                )

            require(
                str(database_name).endswith("_test"),
                "database is not a test database",
            )
            require(user_name == "queryshield_ro", "unexpected application role")
            require(can_login, "application role cannot login")
            require(not is_superuser, "application role is superuser")
            require(not can_create_db, "application role can create databases")
            require(not can_create_role, "application role can create roles")
            require(not bypass_rls, "application role bypasses RLS")
            require(default_read_only == "on", "default transaction is not read-only")
            require(transaction_read_only == "on", "transaction is not read-only")
            for table in TABLES:
                table_privileges = privileges[table]
                require(
                    table_privileges["select"],
                    f"{table} SELECT privilege missing",
                )
                require(
                    not table_privileges["update"],
                    f"{table} UPDATE privilege present",
                )
                require(
                    not table_privileges["insert"],
                    f"{table} INSERT privilege present",
                )

            conn.rollback()
            no_context = conn.execute(COUNT_SQL).fetchone()
            require(no_context == (0, 0, 0), f"missing tenant context exposed rows={no_context}")
            tenant_counts = tenant_bound_counts(conn)
            require(tenant_counts == TENANT_FIXTURE_COUNTS, f"unexpected tenant fixture counts: {tenant_counts}")
            counts = tuple(sum(values[index] for values in tenant_counts.values()) for index in range(3))
            require(counts == (3, 4, 3), f"unexpected fixture counts: {counts}")

            write_probes = (
                (
                    "customers:UPDATE",
                    """
                    UPDATE customers
                    SET name = name
                    WHERE tenant_id = %s AND customer_id = %s
                    """,
                    ("A", "c1"),
                ),
                (
                    "orders:UPDATE",
                    """
                    UPDATE orders
                    SET amount_fen = amount_fen
                    WHERE tenant_id = %s AND order_id = %s
                    """,
                    ("A", "o1"),
                ),
                (
                    "refunds:UPDATE",
                    """
                    UPDATE refunds
                    SET amount_fen = amount_fen
                    WHERE tenant_id = %s AND refund_id = %s
                    """,
                    ("A", "r1"),
                ),
                (
                    "customers:INSERT",
                    """
                    INSERT INTO customers (tenant_id, customer_id, name)
                    VALUES (%s, %s, %s)
                    """,
                    ("A", "__w01_permission_probe_customer__", "probe"),
                ),
                (
                    "orders:INSERT",
                    """
                    INSERT INTO orders (
                        tenant_id, order_id, customer_id, status, amount_fen, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        "A",
                        "__w01_permission_probe_order__",
                        "c1",
                        "paid",
                        1,
                        "2026-09-17T00:00:00Z",
                    ),
                ),
                (
                    "refunds:INSERT",
                    """
                    INSERT INTO refunds (
                        tenant_id, refund_id, order_id, amount_fen, created_at
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        "A",
                        "__w01_permission_probe_refund__",
                        "o1",
                        1,
                        "2026-09-17T00:00:00Z",
                    ),
                ),
            )
            rejected_writes = {}
            for label, statement, params in write_probes:
                error_name = reject_write(conn, statement, params)
                require(
                    error_name == "ReadOnlySqlTransaction",
                    f"{label} was not rejected as read-only: {error_name}",
                )
                rejected_writes[label] = error_name

            unchanged_counts = tenant_bound_counts(conn)
            require(
                unchanged_counts == tenant_counts,
                f"write probes changed fixture counts: {unchanged_counts}",
            )

            reused = conn.execute("SELECT 1").fetchone()[0]
            require(reused == 1, "connection could not be reused after rejected write")
            conn.rollback()

    except (RuntimeError, psycopg.OperationalError, psycopg.InterfaceError) as exc:
        print(f"db_check_blocked={type(exc).__name__}")
        return 2
    except AssertionError as exc:
        print(f"db_check_failed={exc}")
        return 1
    except Exception as exc:
        print(f"db_check_failed={type(exc).__name__}")
        return 1

    print("database_mode=real_postgresql")
    print(f"database={database_name}")
    print(f"user={user_name}")
    print(f"server_version={server_version}")
    print(f"read_only={default_read_only}")
    print(f"transaction_read_only={transaction_read_only}")
    print("fixture_counts=customers:3 orders:4 refunds:3")
    print("select_privileges=customers:on orders:on refunds:on")
    print("write_privileges=all_tables:update:off insert:off")
    print(
        "write_rejected="
        + ",".join(
            f"{label}:{error_name}"
            for label, error_name in rejected_writes.items()
        )
    )
    print("write_probe_cleanup=fixture_counts_unchanged")
    print("connection_reuse=ok")
    print(f"backend_pid={backend_pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
