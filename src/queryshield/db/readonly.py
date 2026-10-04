import os
import re
from urllib.parse import parse_qsl, unquote, urlsplit

import psycopg


DEFAULT_CONNECT_TIMEOUT_SECONDS = 5
CONNECT_TIMEOUT_RANGE = (1, 30)


class DatabaseConfigurationError(RuntimeError):
    """A database setting is invalid; it is refused, never silently replaced."""

    code = "invalid_database_configuration"


def connect_timeout_seconds() -> int:
    """Seconds libpq may spend establishing a connection (default 5).

    Without it an unreachable database can hold a request for the OS TCP
    timeout (about 130 seconds on Windows) before failing.
    """

    raw = os.getenv("QUERYSHIELD_DB_CONNECT_TIMEOUT")
    if raw is None or not raw.strip():
        return DEFAULT_CONNECT_TIMEOUT_SECONDS
    value = raw.strip()
    low, high = CONNECT_TIMEOUT_RANGE
    if not re.fullmatch(r"[0-9]{1,2}", value) or not low <= int(value) <= high:
        raise DatabaseConfigurationError(
            f"QUERYSHIELD_DB_CONNECT_TIMEOUT must be an integer from {low} to {high}"
        )
    return int(value)


# B3d: the demo database and the demo knowledge base are one setting, chosen by
# the server operator (never by a client).  The demo data lives only in a database
# whose name ends with _demo; the evaluation and history checks use _test.  Both
# directions are refused so neither can read the other's data.  Only the database
# name is parsed from the URL; the URL itself is never printed or put in a message.
DEMO_DATASET_ENV = "QUERYSHIELD_DEMO_DATASET"
DEMO_DATASET_VERSION = "commerce-demo-v1"
DEMO_DATABASE_SUFFIX = "_demo"


class DemoDatasetConfigurationError(DatabaseConfigurationError):
    """The demo setting and the database do not match; refused with a fixed code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def demo_dataset_enabled() -> bool:
    """True when the server is set to the demo dataset; an unknown value is refused."""

    raw = os.getenv(DEMO_DATASET_ENV)
    if raw is None or not raw.strip():
        return False
    if raw.strip() != DEMO_DATASET_VERSION:
        raise DemoDatasetConfigurationError(
            "invalid_demo_dataset", f"{DEMO_DATASET_ENV} must be {DEMO_DATASET_VERSION} or unset"
        )
    return True


def database_names_from_url(database_url: str | None) -> tuple[str, ...]:
    """Every database name the URL could connect to (path, ?dbname=, keyword form, PGDATABASE).

    Only the path and query-parameter names are read; percent-encoding is decoded.
    """

    names: list[str] = []
    text = (database_url or "").strip()
    if "://" in text:
        try:
            parts = urlsplit(text)
            names += [unquote(segment) for segment in parts.path.split("/") if segment]
            names += [value for key, value in parse_qsl(parts.query) if key == "dbname"]
        except ValueError:
            names.append("")
    elif text:
        names += [
            (quoted or bare)
            for quoted, bare in re.findall(r"\bdbname\s*=\s*(?:'([^']*)'|(\S+))", text)
        ]
    if not names:
        fallback = os.getenv("PGDATABASE", "").strip()
        if fallback:
            names.append(fallback)
    return tuple(names)


def check_demo_pairing(database_url: str | None) -> None:
    """Refuse a demo setting with a non-demo database, and a demo database without the setting.

    Setting off and a database name that does not end with _demo: nothing happens,
    exactly as before the demo existed.
    """

    names = database_names_from_url(database_url)
    if demo_dataset_enabled():
        if not names or not all(name.endswith(DEMO_DATABASE_SUFFIX) for name in names):
            raise DemoDatasetConfigurationError(
                "demo_dataset_database_mismatch",
                f"{DEMO_DATASET_ENV} requires a database whose name ends with {DEMO_DATABASE_SUFFIX}",
            )
    elif any(name.endswith(DEMO_DATABASE_SUFFIX) for name in names):
        raise DemoDatasetConfigurationError(
            "demo_database_without_demo_dataset",
            f"a {DEMO_DATABASE_SUFFIX} database needs {DEMO_DATASET_ENV}={DEMO_DATASET_VERSION}",
        )


def get_database_url() -> str:
    database_url = os.getenv("QUERYSHIELD_DATABASE_URL")

    if not database_url:
        raise RuntimeError("QUERYSHIELD_DATABASE_URL is not configured")

    check_demo_pairing(database_url)

    return database_url

def connect_readonly(*, tenant_id: str | None = None) -> psycopg.Connection:
    if tenant_id is not None and (type(tenant_id) is not str or not tenant_id.strip()):
        raise ValueError("tenant_id must be a non-empty string")
    return psycopg.connect(
        get_database_url(),
        connect_timeout=connect_timeout_seconds(),
        options= (
            "-c default_transaction_read_only=on "
            "-c statement_timeout=2000"
        ),
    )


def bind_transaction_tenant(connection: psycopg.Connection, tenant_id: str) -> None:
    """Bind the trusted application tenant inside the current read-only transaction.

    The model cannot issue this call: it is performed by the server before the
    guarded SQL cursor is exposed.  RLS is defense in depth; the guarded
    renderer still injects a parameterized tenant predicate for every table.
    """

    if type(tenant_id) is not str or not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    connection.execute(
        "SELECT set_config('queryshield.tenant_id', %s, true)",
        (tenant_id,),
    )
