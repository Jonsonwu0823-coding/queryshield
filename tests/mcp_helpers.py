"""Shared helpers for the MCP metadata tests (identities, launchers, raw SDK sessions)."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile

from queryshield.agent.proposals import ExecutionContext
from queryshield.mcp_metadata.launch import LaunchSpec, McpMetadataConfig, product_launch
from queryshield.mcp_metadata.session import McpMetadataSession


FIXTURE_SERVER = Path(__file__).resolve().parent / "mcp_fixtures" / "fixture_server.py"


def context(tenant: str = "A", role: str = "requester", run_id: str = "run-mcp-test") -> ExecutionContext:
    return ExecutionContext(run_id=run_id, tenant_id=tenant, principal_id=f"{tenant.lower()}-{role}", role=role)


def fixture_launcher(fixture: str):
    """A launcher that starts the fixture server (the real server plus one fault)."""

    def launch(ctx, retriever, mode, cwd) -> LaunchSpec:
        spec = product_launch(ctx, retriever, mode, cwd)
        assert spec.args[:2] == ("-m", "queryshield.mcp_metadata.server")
        return LaunchSpec(
            command=spec.command,
            args=(str(FIXTURE_SERVER), f"--fixture={fixture}", *spec.args[2:]),
            env=spec.env,
            cwd=spec.cwd,
            retrieval=spec.retrieval,
            knowledge_snapshot_id=spec.knowledge_snapshot_id,
        )

    return launch


def config(*, fixture: str | None = None, call_timeout: float = 2.0, launcher=None) -> McpMetadataConfig:
    return McpMetadataConfig(
        mode="fake",
        call_timeout=call_timeout,
        launcher=launcher or (fixture_launcher(fixture) if fixture else None),
    )


class _OwnedDirSession(McpMetadataSession):
    """A session whose working directory belongs to the test helper: it goes with the session."""

    def close(self) -> dict[str, object]:
        try:
            return super().close()
        finally:
            shutil.rmtree(self.spec.cwd, ignore_errors=True)


def started_session(
    ctx: ExecutionContext, retriever=None, *, call_timeout: float = 2.0, args_edit=None, cwd=None
) -> McpMetadataSession:
    """A started session over the product server.

    Without ``cwd`` the helper makes the working directory and removes it when the
    session is closed or fails to start; a ``cwd`` the caller passes (for example
    pytest's ``tmp_path``) is the caller's to remove.
    """

    owned = cwd is None
    directory = tempfile.mkdtemp(prefix="queryshield-mcp-test-") if owned else str(cwd)
    spec = product_launch(ctx, retriever, "fake", directory)
    if args_edit is not None:
        spec = LaunchSpec(spec.command, tuple(args_edit(list(spec.args))), spec.env, spec.cwd, spec.retrieval, spec.knowledge_snapshot_id)
    session = (_OwnedDirSession if owned else McpMetadataSession)(spec, call_timeout=call_timeout, start_timeout=20.0)
    try:
        session.start()
    except BaseException:
        if owned:
            shutil.rmtree(directory, ignore_errors=True)
        raise
    return session


async def raw_requests(spec: LaunchSpec, requests):
    """Run SDK-level requests against a server; each request is ``(callable(client)) -> value``."""

    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=spec.command, args=list(spec.args), env=dict(spec.env), cwd=spec.cwd)
    results = []
    with open(Path(spec.cwd) / "raw-stderr.txt", "w+", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as client:
                await client.initialize()
                for request in requests:
                    try:
                        results.append(await request(client))
                    except Exception as exc:  # noqa: BLE001 - the test inspects it
                        results.append(exc)
    return results


def python() -> str:
    return sys.executable
