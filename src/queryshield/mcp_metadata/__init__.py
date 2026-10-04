"""The two read-only metadata tools over a real MCP stdio session.

``search_catalog`` and ``describe_tables`` can run in an MCP server process
that the product host starts per run with the run's identity.  The host checks
every argument before sending and every result against its own trusted catalog
and knowledge index before the result reaches the run.  ``query_readonly``
never leaves the host.  The package is named ``mcp_metadata`` so that it never
shadows the SDK's ``mcp`` package.
"""
