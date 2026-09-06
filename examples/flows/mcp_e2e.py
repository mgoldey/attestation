"""Every MCP tool, called over stdio, on every agent surface.

Spawns `attest-mcp` five times -- once per ATTEST_TOOLS surface with
ATTEST_EXPAND=1, once unrestricted -- with the `mcp` package's stdio client,
lists the tools, and calls each one with a scripted argument set in the
order a person would. This is the path every agent takes and the one
nothing in tests/ exercises: the entry point, the env the server reads at
import, the stdio framing, the schema FastMCP emits, the stale-process
problem.

Prints a matrix of surface x tool x ok/refused/FAILED. Exit 1 if any call
did not do what CALLS says it should.

    uv run python examples/flows/mcp_e2e.py --offline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common

# (tool, arguments, expectation). Expectation: "ok", "refused", or "options"
# (an .ask router that must return options rather than a default).
CALLS: list[tuple[str, dict, str]] = [
    # --- feed: personas first, since everything else needs one
    ("feed.persona_status", {}, "ok"),
    (
        "feed.persona_create",
        {"name": "flow-temp", "interests": "coral reef ecology and fish telemetry"},
        "ok",
    ),
    (
        "feed.persona_update",
        {"name": "flow-temp", "interests": "coral reef ecology, fisheries, marine protected areas"},
        "ok",
    ),
    ("feed.persona_suggest_interests", {"limit": 5}, "ok"),
    ("feed.persona_status", {"user": "bench-chemist"}, "ok"),
    # `since_days` is explicit because the corpus is a FIXTURE with fixed
    # publication dates: under the 14-day default it shrinks by one item a day
    # and had already decayed to one, which made $ITEM and $ITEM2 the same row
    # and rated it useful and not-useful in consecutive calls.
    ("feed.list", {"user": "bench-chemist", "limit": 4, "since_days": 3650}, "ok"),
    ("feed.list", {"user": "bench-chemist", "limit": 13, "since_days": 3650}, "ok"),
    ("feed.search", {"user": "ml-engineer", "query": "quantisation inference latency"}, "ok"),
    ("feed.read", {"user": "bench-chemist", "item_id": "$ITEM"}, "ok"),
    ("feed.explain", {"user": "bench-chemist", "item_id": "$ITEM"}, "ok"),
    ("feed.rate", {"user": "bench-chemist", "item_id": "$ITEM", "useful": True}, "ok"),
    ("feed.rate", {"user": "bench-chemist", "item_id": "$ITEM2", "useful": False}, "ok"),
    ("feed.harvest_engagement", {"user": "bench-chemist"}, "ok"),
    ("feed.simulate_ratings", {"user": "flow-temp", "limit": 3, "confirm": True}, "ok"),
    ("feed.digest", {"user": "ml-engineer", "days": 3650}, "ok"),
    # "what is new" is NOT a routing phrase -- the rule table has "what's new"
    # and "whats new", and the router declines rather than guessing at a third
    # spelling. "what should i read" is the phrase it does claim.
    ("feed.ask", {"user": "bench-chemist", "question": "what should i read this week?"}, "ok"),
    ("feed.ask", {"user": "bench-chemist", "question": "find papers on flow chemistry"}, "ok"),
    ("feed.ask", {"user": "bench-chemist", "question": "hmm"}, "options"),
    ("feed.persona_reset", {"name": "flow-temp", "confirm": True}, "ok"),
    ("feed.persona_delete", {"name": "flow-temp", "confirm": True}, "ok"),
    ("feed.persona_delete", {"name": "never-existed", "confirm": True}, "refused"),
    # --- feed subscriptions
    ("feed.sources", {}, "ok"),
    ("feed.source_preview", {"url": "$CORPUS_XML", "limit": 3}, "ok"),
    ("feed.source_add", {"url": "$CORPUS_XML", "title": "flows fixture again"}, "ok"),
    ("feed.source_suggest", {"user": "ml-engineer", "limit": 3}, "ok"),
    ("feed.source_remove", {"feed_id": "$FEED_ID", "confirm": True}, "ok"),
    # --- provenance
    ("runs.scan", {"root": "$WORKSPACE", "confirm": True}, "ok"),
    # Preview only (no confirm): writing into the committed workspace fixture
    # here would mutate examples/workspace/ on every flow run and refuse on
    # the second, since record.write() only ever creates new files.
    (
        "runs.record",
        {
            "family": "flow-preview",
            "arms": [{"name": "a", "metrics": {"wer": 0.1}}],
            "root": "$WORKSPACE",
            "project": "speech-distill",
        },
        "ok",
    ),
    ("runs.list", {"limit": 10}, "ok"),
    ("runs.list", {"project": "speech-distill", "family": "kdsweep"}, "ok"),
    ("runs.compare", {"family": "kdsweep", "metric": "wer"}, "ok"),
    ("runs.detail", {"project": "speech-distill", "name": "kdsweep_t4"}, "ok"),
    ("runs.claims_coverage", {"path": "$FINDINGS"}, "ok"),
    ("runs.claims_check", {"path": "$FINDINGS"}, "ok"),
    ("runs.ask", {"question": "which arm of kdsweep won?", "family": "kdsweep"}, "ok"),
    ("runs.ask", {"question": "check the claims in my draft", "path": "$FINDINGS"}, "ok"),
    ("runs.ask", {"question": "hmm"}, "options"),
    # --- knowledge
    ("kg.concepts", {"limit": 10}, "ok"),
    ("kg.central", {"metric": "degree", "limit": 5}, "ok"),
    ("kg.communities", {"min_size": 2}, "ok"),
    ("kg.neighbors", {"node": "$CONCEPT"}, "ok"),
    ("kg.path", {"source": "$CONCEPT", "target": "$CONCEPT2"}, "ok"),
    ("kg.ask", {"question": "what concepts is my reading centred on?"}, "ok"),
    ("kg.ask", {"question": "hmm"}, "options"),
    # --- citations (local sources only; the flow has no .bib, so lookups refuse cleanly)
    ("cite.sources", {}, "ok"),
    ("cite.lookup", {"key": "vaswani2017attention"}, "refused"),
    ("cite.search", {"query": "attention is all you need"}, "ok"),
    ("cite.check", {"path": "$FINDINGS"}, "ok"),
    # cite.sync over the feed only: the corpus items carry no DOI or arXiv id,
    # so the store stays empty and the report says so in structure (0 seen).
    ("cite.sync", {"sources": ["feed"]}, "ok"),
    # --- symbolic (no database)
    ("sym.simplify", {"expr": "(x**2 - 1)/(x - 1)"}, "ok"),
    ("sym.solve", {"expr": "x**2 - 4", "symbol": "x"}, "ok"),
    ("sym.solve", {"expr": "x*y - 1"}, "refused"),
    ("sym.differentiate", {"expr": "x**3", "symbol": "x"}, "ok"),
    ("sym.integrate", {"expr": "x**2", "symbol": "x", "bounds": [0, 1]}, "ok"),
    ("sym.derivation", {"expr": "x**2", "operation": "integrate"}, "ok"),
    ("sym.verify", {"lhs": "sin(x)**2 + cos(x)**2", "rhs": "1"}, "ok"),
    ("sym.evaluate", {"expr": "2*pi", "subs": None}, "ok"),
    ("sym.simplify", {"expr": "(x+1)**200000", "timeout": 3}, "refused"),
    ("sym.ask", {"expr": "x**2 - 9", "question": "solve"}, "ok"),
    ("sym.ask", {"expr": "x**2", "question": "hmm"}, "options"),
    # --- disclosure (only under ATTEST_TOOLS)
    ("feed.tools", {}, "ok"),
    ("runs.tools", {}, "ok"),
    ("kg.tools", {}, "ok"),
    ("sym.tools", {}, "ok"),
]

SURFACES = ("feed", "provenance", "knowledge", "symbolic", None)


def surface_for(tool: str) -> set[str]:
    """Which spawns list this tool.

    Mirrors `mcp._allowed`: the server matches a tool's FULL name against the
    prefix set, or its bare namespace. Matching a `.tools` tool on the first
    segment of every prefix instead was wrong in one direction that mattered
    -- knowledge's prefixes are {"kg", "feed.search", "cite"}, so
    "feed.search".split(".")[0] made surface_for("feed.tools") claim
    {"feed", "knowledge"} while the knowledge agent serves only kg.tools. The
    flow then skipped a tool it claimed to cover, which is the drift it
    exists to catch.
    """
    from attestation.mcp import AGENT_SURFACES

    # The four `<ns>.tools` register only under ATTEST_TOOLS, so they are never
    # part of the unrestricted spawn.
    out = {"full"} if not tool.endswith(".tools") else set()
    for name, surface in AGENT_SURFACES.items():
        if tool in surface.prefixes or tool.split(".", 1)[0] in surface.prefixes:
            out.add(name)
    return out


def check_envelope(payload: dict, expect: str) -> str | None:
    """Did this payload do what CALLS says it should?

    Two envelope shapes are legitimate. The `@tool`-decorated tools return
    `ok` + `message` + their declared empty fields; the four `.ask` routers
    and the `.tools` disclosure tools return an `Answer`, whose prose field
    is `answer` and which has no `message` at all. Requiring both keys would
    have reported every router call as shape drift.
    """
    if "ok" not in payload or not ({"message", "answer"} & set(payload)):
        return "not an envelope: missing ok, and message/answer"
    if expect == "options":
        if payload.get("options"):
            return None
        return f"router chose {payload.get('tool_used')!r} instead of asking"
    if expect == "ok" and not payload["ok"]:
        return f"refused: {_prose(payload)}"
    if expect == "refused" and payload["ok"]:
        return "succeeded but should have refused"
    return None


def _prose(payload: dict) -> str:
    return str(payload.get("message") or payload.get("answer") or "")


def _resolve(arguments: dict, ctx: dict) -> dict:
    def sub(v):
        if isinstance(v, str) and v.startswith("$"):
            return ctx[v[1:]]
        return v

    return {k: sub(v) for k, v in arguments.items()}


async def run_surface(surface: str | None, env: dict, ctx: dict) -> list[dict]:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    spawn_env = {**os.environ, **env, "ATTEST_EXPAND": "1"}
    if surface:
        spawn_env["ATTEST_TOOLS"] = surface
    else:
        spawn_env.pop("ATTEST_TOOLS", None)
    params = StdioServerParameters(
        command="uv", args=["run", "--project", str(_common.REPO_ROOT), "attest-mcp"], env=spawn_env
    )
    label = surface or "full"
    rows: list[dict] = []
    try:
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            listed = {t.name for t in (await session.list_tools()).tools}
            rows += await _call_all(session, label, listed, ctx)
    except Exception as exc:  # noqa: BLE001 -- a surface that will not start is one row saying so
        # The docstring promises a matrix, and render() needs at least one row
        # to size its columns: a server that fails to start used to traceback
        # out of main() instead of reporting which surface died and why.
        rows.append(_row(label, "<spawn>", False, "ok", f"server did not start: {exc}", 0.0))
    return rows


async def _call_all(session, label: str, listed: set[str], ctx: dict) -> list[dict]:
    """Every scripted call this surface is meant to serve, in order."""
    rows = []
    for tool, arguments, expect in CALLS:
        if label not in surface_for(tool):
            continue
        if tool not in listed:
            # NOT `continue`. Skipping a planned-but-unserved tool silently is
            # exactly the drift this flow exists to catch: surface_for said
            # this spawn lists it and the spawn did not, so one of the two is
            # wrong and the matrix has to say so.
            rows.append(
                _row(
                    label,
                    tool,
                    None,
                    expect,
                    f"planned for {label} but the server did not list it",
                    0.0,
                )
            )
            continue
        t0 = time.perf_counter()
        try:
            result = await session.call_tool(tool, _resolve(arguments, ctx))
            payload = json.loads(result.content[0].text) if result.content else {}
            if result.isError and "ok" not in payload:
                payload = {
                    "ok": False,
                    "message": result.content[0].text if result.content else "error",
                }
        except Exception as exc:  # noqa: BLE001 -- one bad call is one row, and the row says why
            payload = {"ok": False, "message": f"{type(exc).__name__}: {exc}"}
        rows.append(
            _row(
                label,
                tool,
                payload.get("ok"),
                expect,
                check_envelope(payload, expect),
                round(time.perf_counter() - t0, 2),
                _prose(payload)[:160],
            )
        )
        _learn(tool, payload, ctx)
    rows.append(_row(label, "<list_tools>", True, "ok", None, 0.0, f"{len(listed)} tools"))
    return rows


def _row(
    surface: str,
    tool: str,
    ok: bool | None,
    expect: str | None,
    problem: str | None,
    seconds: float,
    message: str = "",
) -> dict:
    return {
        "surface": surface,
        "tool": tool,
        "expect": expect,
        "ok": ok,
        "problem": problem,
        "message": message,
        "seconds": seconds,
    }


def _learn(tool: str, payload: dict, ctx: dict) -> None:
    """Ids the later calls need, taken from the earlier ones' answers."""
    if tool == "feed.list" and payload.get("items"):
        ids = [i["item_id"] for i in payload["items"]]
        ctx.setdefault("ITEM", ids[0])
        ctx.setdefault("ITEM2", ids[-1] if len(ids) > 1 else ids[0])
    if tool == "feed.source_add" and payload.get("ok"):
        ctx["FEED_ID"] = payload.get("feed_id")
    if tool == "kg.concepts" and payload.get("concepts"):
        names = [c["name"] if isinstance(c, dict) else c for c in payload["concepts"]]
        ctx["CONCEPT"] = names[0]
        ctx["CONCEPT2"] = names[1] if len(names) > 1 else names[0]
    # kg.path's target comes from kg.neighbors, not from the second concept in
    # the alphabetical list: on the real graph those two were 'astrophysics'
    # and 'catalysis', in different components, and "no path" is a CORRECT
    # answer that the flow was scripting as a failure. A neighbour is adjacent
    # by construction, so the pair is always connected.
    if tool == "kg.neighbors" and payload.get("neighbors"):
        ctx["CONCEPT2"] = payload["neighbors"][0]["name"]


def _tag_and_click(db_path: Path, base_url: str, chat_model: str, users: dict) -> None:
    """Tags for the graph, simulated clicks for the classifier: what a lived-in DB has."""
    from attestation.db import get_db
    from attestation.features import run_tagging
    from attestation.llm import ChatClient
    from attestation.simulate import simulate_feedback

    conn = get_db(db_path)
    chat = ChatClient(base_url=base_url, model=chat_model)
    run_tagging(conn, chat.chat_json, chat_model)
    items = conn.execute("SELECT * FROM items ORDER BY id LIMIT 20").fetchall()
    for name in users:
        simulate_feedback(conn, chat.chat_json, name, items)
    conn.close()


def render(rows: list[dict]) -> str:
    width = max(len(r["tool"]) for r in rows)
    out = []
    for r in rows:
        status = "FAILED" if r["problem"] else ("refused" if r["ok"] is False else "ok")
        line = f"{r['surface']:<10} {r['tool']:<{width}} {status:<8} {r['seconds']:>6.2f}s"
        if r["problem"]:
            line += f"  <- {r['problem']}"
        out.append(line)
    failed = sum(1 for r in rows if r["problem"])
    out.append(f"{len(rows)} calls, {failed} failed")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--surface", choices=[s for s in SURFACES if s] + ["full"])
    args = ap.parse_args(argv)

    from attestation.llm import base_url, chat_model, embed_model, load_env

    server = None
    if args.offline:
        import stub_openai

        server, url = stub_openai.start()
        chat, embed = stub_openai.MODEL, stub_openai.MODEL
    else:
        load_env()
        url, chat, embed = base_url(), chat_model(), embed_model()

    persona_eval = _common.load_script("persona_eval")
    tmp = tempfile.TemporaryDirectory()
    db_path = Path(tmp.name) / "flows.db"
    rows: list[dict] = []
    try:
        prepared = persona_eval.prepare_db(db_path, url, chat, embed)
        _tag_and_click(db_path, url, chat, prepared["users"])
        workspace = _common.REPO_ROOT / "examples" / "workspace"
        ctx = {
            "WORKSPACE": str(workspace),
            "FINDINGS": str(workspace / "speech-distill" / "FINDINGS.md"),
            "CORPUS_XML": str(_common.CORPUS_DIR / "labelled.xml"),
        }
        env = {
            "ATTEST_DB": str(db_path),
            "LLM_BASE_URL": url,
            "CHAT_MODEL": chat,
            "EMBED_MODEL": embed,
        }
        surfaces = [args.surface if args.surface != "full" else None] if args.surface else SURFACES
        for surface in surfaces:
            rows += asyncio.run(run_surface(surface, env, ctx))
    finally:
        if server:
            server.shutdown()
        tmp.cleanup()
    print(f"mcp e2e -- mode={'offline' if args.offline else 'live'} chat={chat} embed={embed}")
    print(render(rows))
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "flow": "mcp_e2e",
                    "mode": "offline" if args.offline else "live",
                    "chat_model": chat,
                    "rows": rows,
                },
                indent=2,
            )
        )
    return 1 if any(r["problem"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
