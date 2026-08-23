"""The `<surface>.tools` disclosure tool.

Split out of ask.py, which hit its size cap. Small, but it is one coherent
concern -- what a restricted agent tells a caller about the tools it is not
showing -- and it kept four descriptions, one answer body and a registration
condition tangled with the routers.

Registered ONLY when ATTEST_TOOLS restricts the surface. With all 50 tools
served the specifics are visible, and four byte-identical tools claiming
otherwise is a falsehood in the commonest configuration, for 676 chars.
"""

SURFACE_CONCERNS = {
    "feed": "ranked reading, keyword search, personas and feed subscriptions",
    "runs": "the experiment ledger and the claim checker over your drafts",
    "kg": "the concept graph built from what you have read",
    "sym": "symbolic algebra, calculus and unit-aware evaluation",
}


def _tools_doc(surface: str, expanded: bool) -> str:
    """The `<surface>.tools` description, which depends on what is registered.

    All four carried one byte-identical string saying the specifics were
    hidden. True under ATTEST_TOOLS, false in the default 50-tool surface where
    this tool sat beside every tool it called concealed -- and a model choosing
    between four identical descriptions has nothing to choose on.
    """
    body = _tools_body(surface, expanded)
    return f"What this agent can do: {SURFACE_CONCERNS[surface]}.\n\n{body}"


def _tools_body(surface: str, expanded: bool) -> str:
    """One sentence on where this agent's specific tools are. Shared by the
    description and the tool's own answer, so the two cannot disagree."""
    if expanded:
        return (
            f"The specific {surface}.* tools are already listed alongside this"
            f" one; call them directly, or {surface}.ask to route a question."
        )
    return (
        f"The specific {surface}.* tools are hidden so the router is chosen"
        f" rather than guessed at -- measured on gemma4:e2b, a model picked the"
        f" router 1 time in 26 when the specifics were listed beside it, and 26"
        f" in 26 when they were not. Set ATTEST_EXPAND=1 to see them, or call"
        f" {surface}.ask with a plain question."
    )


def _tools_listing(surface: str, expanded: bool) -> dict:
    """The answer `<surface>.tools` returns."""
    return {
        "ok": True,
        "answer": _tools_body(surface, expanded),
        "refs": [],
        "caveat": None,
        "options": [f"{surface}.ask"],
        "tool_used": f"{surface}.tools",
    }


def register_disclosure(mcp, restricted: str, expanded: bool) -> None:
    """Attach `<surface>.tools`, but only when the specifics are hidden."""
    if not restricted:
        return
    from attestation.mcp.ask import Answer

    for _surface in ("feed", "runs", "kg", "sym"):

        def _make(surface=_surface):
            def _list_tools() -> Answer:
                return Answer(**_tools_listing(surface, expanded))

            # Set BEFORE @mcp.tool sees it: FastMCP reads __doc__ at decoration
            # time, so assigning afterwards left the description empty.
            _list_tools.__doc__ = _tools_doc(surface, expanded)
            return mcp.tool(name=f"{surface}.tools")(_list_tools)

        _make()
