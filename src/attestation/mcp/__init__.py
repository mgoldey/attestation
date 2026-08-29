"""The MCP tool surface, split by domain.

`mcp_server.py` reached 1454 lines holding 37 tools written twice each -- an
`_impl` function plus a `@mcp.tool()` wrapper that was a docstring and one
delegating call -- with the same ritual repeated in every body: 26 copies of
`with open_db()`, 28 broad `except` blocks, 10 hand-written unknown-user
checks.

The split is by domain rather than by layer, and the namespaces are the point:
a calling agent chooses a domain first (4 ways) and then a tool within it,
instead of picking from 34 flat names where runs_compare, kg_path,
sym_integrate and digest all looked like peers. (Those names are quoted here as
history -- none is served any more, and backticking them would tell a reader
they were callable.)

`_tool.tool` owns the ritual so each tool body computes only its own answer.
"""

import os
from dataclasses import dataclass

from attestation.mcp import (
    ask,
    citation,
    feed,
    knowledge,
    provenance,
    subscriptions,
    symbolic,
)

DOMAINS = (ask, citation, feed, knowledge, provenance, subscriptions, symbolic)

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
#   claim checker in a session that cannot see what it checks against. By the
#   same rule `cite.check` -- a claim checker for citation keys -- is in
#   `provenance` as well as `knowledge`: a session that can lint uncited claims
#   but cannot see the runs the other claims are checked against would report
#   half a document's problems and look complete.
#
# `summary` and `rationale` are data rather than comments because
# `attestation.emit` generates agent configs from this table, and prose that
# explains a config should live where the config is generated from -- otherwise
# it rots separately from it.


@dataclass(frozen=True)
class Surface:
    """One agent's remit: what it may see, and why it is its own session."""

    prefixes: frozenset[str]
    summary: str
    rationale: str
    # What the agent is FOR, in the words a user would use, plus the questions
    # it should recognise as its own. A definition that lists capabilities and
    # omits purpose is how a live agent with 3,106 arXiv papers answered a
    # request for papers with "I have no tool for searching academic
    # repositories" -- every tool was registered and working, and nothing said
    # what they were for.
    goal: str = ""


AGENT_SURFACES: dict[str, Surface] = {
    "feed": Surface(
        prefixes=frozenset({"feed"}),
        summary="Read and search the science feed, and manage what it subscribes to.",
        rationale="Conversational; a wrong guess costs a retry.",
        goal=(
            "Help this reader keep up with the literature. The corpus is their"
            " own subscribed feeds -- mostly arXiv, plus Nature, Scientific"
            ' Reports, Hugging Face and HN -- so questions like "find me'
            ' recent papers on X", "what should I read this week" and'
            ' "what\'s new in Y" are YOURS to answer, locally, without any'
            " network call. Reach for feed.ask when the question is in plain"
            " words. Reading an item is itself feedback: it trains the ranker,"
            " so open what looks relevant rather than only listing titles."
        ),
    ),
    "provenance": Surface(
        prefixes=frozenset({"runs", "cite.check"}),
        summary="Scan experiment runs, compare arms, and check claims against them.",
        rationale=(
            "Verification: a wrong answer reaches a manuscript, and the caveats are the product."
        ),
        goal=(
            'Answer "which arm actually won, and on what evidence" and "is'
            ' what my draft says still true". You read experiment artifacts'
            " already on disk -- no instrumentation, no re-running anything."
            " The caveats you return are the product, not a disclaimer: relay"
            " them verbatim. When a comparison refuses, the refusal is the"
            " right answer and it names what to do next; do not work around it"
            " by picking a metric yourself."
        ),
    ),
    "knowledge": Surface(
        prefixes=frozenset({"kg", "feed.search", "cite"}),
        summary=(
            "Explore the reading knowledge graph, the items behind it, and the"
            " references they cite."
        ),
        rationale="Exploratory, read-only.",
        goal=(
            'Answer "what have I been reading about", "how do these two'
            ' topics connect" and "what are my main research areas", from'
            " the concept graph built out of this reader's own items. Concept"
            " names are lowercase and hyphenated; turn a user's phrasing into"
            " one with kg.concepts before looking it up."
        ),
    ),
    "symbolic": Surface(
        prefixes=frozenset({"sym"}),
        summary="Symbolic algebra and calculus, in a sandboxed subprocess.",
        rationale="Sandboxed subprocess, touches no database.",
        goal=(
            "Do the algebra and calculus exactly, so a derivation in a paper"
            " can be checked rather than eyeballed. `^` and `2x` both parse."
            " sym.verify reports unproven as unproven -- it is never a"
            " disproof."
        ),
    ),
}


def _allowed(name: str, surface: frozenset[str]) -> bool:
    return name in surface or name.split(".", 1)[0] in surface


class _FilteringServer:
    """Passes `mcp.tool()` through, dropping registrations outside the surface.

    A wrapper rather than a post-registration filter: FastMCP has no public
    deregister, and dropping at the decorator keeps the tool from ever being
    constructed.

    `expand` controls progressive disclosure. Measured on gemma4:e2b over 26
    turns across the four agents: with the specific tools visible the model
    picked the `ask` router 1 time in 26; with only `ask` visible, 26 in 26.
    Marking the specifics "advanced; prefer .ask" in their descriptions moved
    one agent from 8/10 to 7/10 -- inside noise, and no help. A visible tool
    gets called, so "ask first, specifics on request" has to mean the
    specifics are genuinely absent until requested.
    """

    def __init__(self, inner, surface: frozenset[str], expand: bool):
        self._inner = inner
        self._surface = surface
        self._expand = expand

    def tool(self, *args, name: str | None = None, **kwargs):
        """`FastMCP.tool`, but a no-op decorator (never registering the
        function) for anything outside `self._surface`, or -- unless
        `expand` -- for anything that is not an entry point. See the class
        docstring for why filtering happens here, at the decorator, rather
        than after registration."""
        if name is not None:
            if not _allowed(name, self._surface):
                return lambda fn: fn  # registered nowhere; the tool simply is not
            if not self._expand and not _is_entry_point(name):
                return lambda fn: fn
        return self._inner.tool(*args, name=name, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._inner, attr)


def _is_entry_point(name: str) -> bool:
    """The two tools an agent sees before it asks for more."""
    return name.endswith(".ask") or name.endswith(".tools")


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
        # ATTEST_EXPAND=1 reveals this agent's specific tools from the start,
        # for a caller that already knows what it wants. It never crosses the
        # surface boundary -- disclosure widens what you can see, never what
        # you are allowed to touch.
        expand = os.environ.get("ATTEST_EXPAND", "").strip() not in ("", "0", "false")
        mcp = _FilteringServer(mcp, AGENT_SURFACES[requested].prefixes, expand)
    for domain in DOMAINS:
        domain.register(mcp)
