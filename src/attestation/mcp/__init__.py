"""The MCP tool surface, split by domain.

`mcp_server.py` reached 1454 lines holding 37 tools written twice each -- an
`_impl` function plus a `@mcp.tool()` wrapper that was a docstring and one
delegating call -- with the same ritual repeated in every body: 26 copies of
`with open_db()`, 28 broad `except` blocks, 10 hand-written unknown-user
checks.

The split is by domain rather than by layer, and the namespaces are the point:
a calling agent chooses a domain first (4 ways) and then a tool within it,
instead of picking from 34 flat names where `runs_compare`, `kg_path`,
`sym_integrate` and `digest` all look like peers.

`_tool.tool` owns the ritual so each tool body computes only its own answer.
"""

from attestation.mcp import feed, knowledge, provenance, symbolic

DOMAINS = (feed, knowledge, provenance, symbolic)


def register_all(mcp) -> None:
    """Attach every domain's tools to the server."""
    for domain in DOMAINS:
        domain.register(mcp)
