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

import os

from attestation.mcp import ask, feed, knowledge, provenance, symbolic

DOMAINS = (ask, feed, knowledge, provenance, symbolic)

# Which namespaces (or individual tools) each agent may see.
#
# One agent choosing among 37 tools picked correctly 8 times in 15, measured on
# gemma4:e2b over realistic turns, three runs, no variance. Restricting the
# surface is the half of the fix that needs no routing: a tool outside an
# agent's remit is ABSENT rather than undocumented, because a model that can
# see a tool will eventually call it.
#
# Selection is per SESSION -- a person launches the agent they want. A measured
# arm where one model picked the namespace and a second picked within it scored
# 7.3/15 at twice the latency: a second call is a second chance to be wrong,
# and a namespace miss cannot be recovered.
#
# Two boundaries are deliberate and were argued in the design spec:
#
#   `feed.search` is duplicated into `knowledge` because "how does X connect to
#   Y, and what did I read about it" is one question, and a knowledge session
#   with no way to reach the items is a dead end.
#
#   Claims live with `runs`, not with the graph. `runs.claims_check` verifies
#   numbers in Markdown AGAINST recorded runs; separating them would put a
#   claim checker in a session that cannot see what it checks against.
AGENT_SURFACES: dict[str, frozenset[str]] = {
    "feed": frozenset({"feed"}),
    "provenance": frozenset({"runs"}),
    "knowledge": frozenset({"kg", "feed.search"}),
    "symbolic": frozenset({"sym"}),
}


def _allowed(name: str, surface: frozenset[str]) -> bool:
    return name in surface or name.split(".", 1)[0] in surface


class _FilteringServer:
    """Passes `mcp.tool()` through, dropping registrations outside the surface.

    A wrapper rather than a post-registration filter: FastMCP has no public
    deregister, and dropping at the decorator keeps the tool from ever being
    constructed.
    """

    def __init__(self, inner, surface: frozenset[str]):
        self._inner = inner
        self._surface = surface

    def tool(self, *args, name: str | None = None, **kwargs):
        if name is not None and not _allowed(name, self._surface):
            return lambda fn: fn  # registered nowhere; the tool simply is not
        return self._inner.tool(*args, name=name, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._inner, attr)


def register_all(mcp) -> None:
    """Attach the tools this agent is allowed to serve.

    `ATTEST_TOOLS` names one of AGENT_SURFACES. Unset serves everything, so a
    single-agent setup keeps working unchanged.
    """
    requested = os.environ.get("ATTEST_TOOLS", "").strip()
    if requested:
        if requested not in AGENT_SURFACES:
            # Loud rather than permissive: a typo that silently served
            # everything would be a restriction that quietly stopped
            # restricting, which is the failure worth preventing.
            raise ValueError(
                f"unknown ATTEST_TOOLS surface: {requested!r}."
                f" Expected one of {', '.join(sorted(AGENT_SURFACES))}"
            )
        mcp = _FilteringServer(mcp, AGENT_SURFACES[requested])
    for domain in DOMAINS:
        domain.register(mcp)
