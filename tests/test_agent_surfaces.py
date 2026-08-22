"""Each agent sees only the tools its job needs.

One agent choosing among 37 tools picked correctly 8 times out of 15, measured
on gemma4:e2b over realistic turns, three runs, no variance. Restricting the
surface per agent is the half of the fix that does not depend on routing: a
tool outside an agent's remit should be ABSENT, not merely undocumented, since
a model that can see a tool will eventually call it.

Selection is by session, not by a supervisor at runtime. A measured arm where
one model picked a namespace and a second picked within it scored 7.3/15 at
twice the latency -- a second call is a second chance to be wrong, and a
namespace miss cannot be recovered.
"""

import asyncio

import pytest

from attestation.mcp import AGENT_SURFACES, register_all


def _served(monkeypatch, tmp_path, surface=None, *, expand=True):
    """Tools this agent serves. `expand` defaults to True because most tests
    here are about the BOUNDARY -- which tools an agent may touch -- and the
    default surface hides the specifics behind disclosure."""
    from mcp.server.fastmcp import FastMCP

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    if surface is None:
        monkeypatch.delenv("ATTEST_TOOLS", raising=False)
    else:
        monkeypatch.setenv("ATTEST_TOOLS", surface)
    monkeypatch.setenv("ATTEST_EXPAND", "1" if expand else "0")
    mcp = FastMCP("test")
    register_all(mcp)
    return {t.name for t in asyncio.run(mcp.list_tools())}


def test_no_restriction_serves_everything(monkeypatch, tmp_path):
    """The single-agent setup must keep working untouched."""
    names = _served(monkeypatch, tmp_path)
    assert len(names) >= 37
    assert {"feed.list", "runs.compare", "kg.path", "sym.solve"} <= names


@pytest.mark.parametrize("surface", sorted(AGENT_SURFACES))
def test_each_surface_serves_only_its_own_namespaces(monkeypatch, tmp_path, surface):
    names = _served(monkeypatch, tmp_path, surface)
    allowed = AGENT_SURFACES[surface]
    for name in names:
        namespace = name.split(".", 1)[0]
        assert namespace in allowed or name in allowed, (
            f"the {surface} agent serves {name}, which is outside {sorted(allowed)}"
        )
    assert names, f"the {surface} agent serves nothing"


def test_a_provenance_tool_is_absent_from_the_feed_agent(monkeypatch, tmp_path):
    """Absent, not undocumented. A model that can see a tool will call it."""
    names = _served(monkeypatch, tmp_path, "feed")
    assert "runs.compare" not in names
    assert "sym.solve" not in names
    assert "feed.list" in names


def test_the_knowledge_agent_can_still_reach_items(monkeypatch, tmp_path):
    """ "How does X connect to Y, and what did I read about it" is one
    question. A knowledge session with no way to reach the items is a dead
    end, so feed.search is deliberately duplicated into it."""
    names = _served(monkeypatch, tmp_path, "knowledge")
    assert "kg.path" in names
    assert "feed.search" in names
    assert "feed.rate" not in names, "the knowledge agent is read-only"


def test_claims_live_with_runs_not_with_the_graph(monkeypatch, tmp_path):
    """claims_check verifies Markdown numbers AGAINST recorded runs. Putting
    it in the knowledge agent would give a claim checker no access to what it
    checks against."""
    assert "runs.claims_check" not in _served(monkeypatch, tmp_path, "knowledge")
    assert "runs.claims_check" in _served(monkeypatch, tmp_path, "provenance")


def test_an_unknown_surface_is_refused_loudly(monkeypatch, tmp_path):
    """A typo in ATTEST_TOOLS must not silently serve everything -- that is
    the failure mode where a restriction quietly stops restricting."""
    from mcp.server.fastmcp import FastMCP

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("ATTEST_TOOLS", "feeed")
    with pytest.raises(ValueError, match="feeed"):
        register_all(FastMCP("test"))


def test_the_surfaces_cover_every_tool(monkeypatch, tmp_path):
    """No tool may be unreachable from every agent -- that is a tool nobody
    can call, which is worse than one nobody should."""
    everything = _served(monkeypatch, tmp_path)
    reachable: set[str] = set()
    for surface in AGENT_SURFACES:
        reachable |= _served(monkeypatch, tmp_path, surface)
    assert everything - reachable == set(), f"unreachable: {sorted(everything - reachable)}"


def test_progressive_disclosure_hides_the_specifics_by_default(monkeypatch, tmp_path):
    """Measured: with the specific tools visible the model picks `ask` 1 time
    in 26. With only `ask` visible, 26 in 26.

    Marking the specifics "advanced; prefer .ask" in their descriptions moved
    the feed agent from 8/10 to 7/10 -- inside noise, and no help. A visible
    tool gets called. So the specifics are genuinely absent until a caller
    asks for them, which is what "ask first, specifics on request" has to mean
    to be worth anything.
    """
    names = _served(monkeypatch, tmp_path, "feed", expand=False)
    assert "feed.ask" in names
    assert "feed.tools" in names, "there must be a way to reach the specifics"
    assert "feed.list" not in names, "a specific tool is visible by default"
    assert len(names) == 2, f"the default surface should be ask + tools, got {sorted(names)}"


def test_the_full_surface_is_reachable_on_request(monkeypatch, tmp_path):
    """Hiding without an escape hatch is worse than not hiding: a question the
    router mis-routes would have nowhere to go."""
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("ATTEST_TOOLS", "feed")
    monkeypatch.setenv("ATTEST_EXPAND", "1")
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("test")
    register_all(mcp)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "feed.list" in names
    assert len(names) > 10


def test_expanding_never_crosses_the_agent_boundary(monkeypatch, tmp_path):
    """Disclosure reveals THIS agent's tools, never another's -- otherwise the
    restriction is advisory."""
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("ATTEST_TOOLS", "provenance")
    monkeypatch.setenv("ATTEST_EXPAND", "1")
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("test")
    register_all(mcp)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "runs.compare" in names
    assert not any(n.startswith("feed.") or n.startswith("sym.") for n in names)
