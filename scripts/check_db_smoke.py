import psycopg

from queryshield.db.readonly import bind_transaction_tenant, connect_readonly


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    try:
        with connect_readonly() as conn:
            database_name, user_name, server_version, read_only, backend_pid = conn.execute(
                """
                SELECT
                    current_database(),
                    current_user,
                    current_setting('server_version'),
                    current_setting('transaction_read_only'),
                    pg_backend_pid()
                """
            ).fetchone()

            tables = conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN ('customers', 'orders', 'refunds')
                ORDER BY table_name
                """
            ).fetchall()
            # The tables force row-level security keyed on the transaction's
            # tenant, so the fixture is counted per tenant with the server-bound context.
            tenant_counts = []
            for tenant_id in ("A", "B"):
                conn.rollback()
                bind_transaction_tenant(conn, tenant_id)
                tenant_counts.append(
                    conn.execute(
                        """
                        SELECT
                            (SELECT COUNT(*) FROM customers),
                            (SELECT COUNT(*) FROM orders),
                            (SELECT COUNT(*) FROM refunds)
                        """
                    ).fetchone()
                )
            conn.rollback()
            counts = tuple(sum(values[index] for values in tenant_counts) for index in range(3))

        require(str(database_name).endswith("_test"), "database is not a test database")
        require(user_name == "queryshield_ro", "unexpected smoke-test role")
        require(read_only == "on", "transaction is not read-only")
        require(
            [row[0] for row in tables] == ["customers", "orders", "refunds"],
            f"commerce tables are incomplete: {tables}",
        )
        require(counts == (3, 4, 3), f"unexpected fixture counts: {counts}")

    except (RuntimeError, psycopg.OperationalError, psycopg.InterfaceError) as exc:
        print(f"db_smoke_blocked={type(exc).__name__}")
        return 2
    except AssertionError as exc:
        print(f"db_smoke_failed={exc}")
        return 1
    except Exception as exc:
        print(f"db_smoke_failed={type(exc).__name__}")
        return 1

    print(f"database={database_name}")
    print(f"user={user_name}")
    print(f"server_version={server_version}")
    print(f"read_only={read_only}")
    print("migration_version=001_commerce_v1.sql")
    print("fixture_version=commerce-v1")
    print("fixture_counts=customers:3 orders:4 refunds:3")
    print(f"backend_pid={backend_pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
