"""The MCP flow's scripted calls must cover every registered tool and only
registered tools. Model-free: the plan is data; the server is checked in
the flow itself."""

import asyncio
import importlib.util
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP

from attestation.mcp import register_all

FLOWS = Path(__file__).parents[1] / "examples" / "flows"


def _flow():
    spec = importlib.util.spec_from_file_location("flows_mcp", FLOWS / "mcp_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _served(monkeypatch, surface=None):
    monkeypatch.setenv("ATTEST_EXPAND", "1")
    if surface:
        monkeypatch.setenv("ATTEST_TOOLS", surface)
    else:
        monkeypatch.delenv("ATTEST_TOOLS", raising=False)
    m = FastMCP("plan")
    register_all(m)
    return {t.name for t in asyncio.run(m.list_tools())}


def test_every_served_tool_is_called_at_least_once(monkeypatch):
    called = {c[0] for c in _flow().CALLS}
    served = _served(monkeypatch)
    for surface in ("feed", "provenance", "knowledge", "symbolic"):
        served |= _served(monkeypatch, surface)
    assert served - called == set(), f"never called: {sorted(served - called)}"
    assert called - served == set(), f"called but not served: {sorted(called - served)}"


def test_every_ask_router_gets_a_question_that_must_disambiguate():
    calls = _flow().CALLS
    for router in ("feed.ask", "runs.ask", "kg.ask", "sym.ask"):
        assert any(c[0] == router and c[2] == "options" for c in calls), router


def test_destructive_tools_are_called_on_entities_the_flow_created():
    calls = _flow().CALLS
    names = [c[0] for c in calls]
    assert names.index("feed.persona_create") < names.index("feed.persona_delete")
    assert names.index("feed.source_add") < names.index("feed.source_remove")
    for tool in ("feed.persona_delete", "feed.persona_reset", "feed.source_remove", "runs.scan"):
        assert any(c[0] == tool and c[1].get("confirm") is True for c in calls), tool


def test_envelope_check_accepts_the_contract_and_rejects_shape_drift():
    m = _flow()
    assert m.check_envelope({"ok": True, "message": "", "items": []}, "ok") is None
    assert m.check_envelope({"ok": False, "message": "no", "items": []}, "refused") is None
    assert m.check_envelope({"ok": False, "message": "boom", "items": []}, "ok")
    assert m.check_envelope({"ok": True, "message": ""}, "refused")
    assert m.check_envelope({"message": "no ok key"}, "ok")
    # The .ask tools return an Answer, which has `answer` where the dict
    # envelopes have `message`. Both are envelopes; neither has both keys.
    assert (
        m.check_envelope(
            {
                "ok": True,
                "answer": "here is your feed",
                "refs": [],
                "caveat": None,
                "options": [],
                "tool_used": "feed.list",
            },
            "ok",
        )
        is None
    )


@pytest.mark.parametrize("surface", ("feed", "provenance", "knowledge", "symbolic"))
def test_surface_for_agrees_with_what_each_spawn_actually_registers(monkeypatch, surface):
    """Per surface, not only in the union.

    `surface_for` decides which spawn calls which tool, and a tool it maps to
    the wrong surface is silently absent from the matrix rather than failing.
    Matching a `.tools` tool on the first segment of every prefix did exactly
    that: knowledge's prefixes are {"kg", "feed.search", "cite"}, so
    "feed.search" made surface_for("feed.tools") claim knowledge, which serves
    only kg.tools. The union test could not see it, because the feed spawn
    covered feed.tools.
    """
    m = _flow()
    planned = {c[0] for c in m.CALLS if surface in m.surface_for(c[0])}
    served = _served(monkeypatch, surface)
    assert planned == served, (
        f"{surface}: planned-not-served {sorted(planned - served)},"
        f" served-not-planned {sorted(served - planned)}"
    )


def test_the_unrestricted_spawn_serves_no_disclosure_tool(monkeypatch):
    """`<ns>.tools` registers ONLY under ATTEST_TOOLS, so planning one for the
    full spawn would plan a call that can never be listed."""
    m = _flow()
    planned = {c[0] for c in m.CALLS if "full" in m.surface_for(c[0])}
    assert planned == _served(monkeypatch)
    assert not any(t.endswith(".tools") for t in planned)


def test_a_planned_tool_the_server_does_not_list_is_a_failed_row():
    """Skipping it silently is the drift the flow exists to catch."""
    m = _flow()
    rows = asyncio.run(m._call_all(None, "feed", set(), {}))
    missing = [r for r in rows if r["tool"] == "feed.list"]
    assert missing and missing[0]["problem"]
    assert "did not list it" in missing[0]["problem"]
