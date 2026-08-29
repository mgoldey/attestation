# examples/citations/check_citations.py
"""Drives cite.sources, cite.check, cite.lookup and cite.search over stdio.

`attest claims` (see `attestation.cli.cmd_claims`) now builds its own
resolver and runs the citation lint too, so `attest claims DRAFT.md` and
this script report the same `uncited` verdict. This script remains the
MCP-side demonstration: it is still the only path to `cite.lookup` and
`cite.search`, and to `cite.sources`' `offline: true` reporting.

Pattern copied from `examples/flows/mcp_e2e.py`'s `run_surface`: spawn
`attest-mcp` over stdio with `ATTEST_TOOLS=knowledge ATTEST_EXPAND=1` (the
`knowledge` surface is where `cite.*` lives, alongside `kg.*` and
`feed.search`), list tools, call each one, print what came back.

    uv run python check_citations.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _payload(result) -> dict:
    return json.loads(result.content[0].text) if result.content else {}


async def run() -> int:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = {**os.environ, "ATTEST_TOOLS": "knowledge", "ATTEST_EXPAND": "1"}
    params = StdioServerParameters(
        command="uv", args=["run", "--project", str(REPO_ROOT), "attest-mcp"], env=env
    )
    draft = str(HERE / "DRAFT.md")
    failed = False

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listed = {t.name for t in (await session.list_tools()).tools}
        for name in ("cite.sources", "cite.check", "cite.lookup", "cite.search"):
            if name not in listed:
                print(f"MISSING {name}: not served by the knowledge surface")
                failed = True

        sources = _payload(await session.call_tool("cite.sources", {}))
        print(f"cite.sources -> offline={sources.get('offline')} sources={sources.get('sources')}")

        checked = _payload(await session.call_tool("cite.check", {"path": draft}))
        print(f"cite.check -> {checked.get('message')}")
        for u in checked.get("uncited", []):
            print(f"  uncited key={u['key']!r} at {u['where']}")

        found = _payload(await session.call_tool("cite.lookup", {"key": "vaswani2017attention"}))
        ref = found.get("reference") or {}
        print(f"cite.lookup vaswani2017attention -> {ref.get('title')!r} ({ref.get('year')})")

        result = await session.call_tool("cite.lookup", {"key": "doe2099imaginary"})
        refusal = _payload(result)
        print(f"cite.lookup doe2099imaginary -> refused: {refusal.get('message')!r}")
        if refusal.get("ok") is not False:
            print("  expected a refusal for an unresolvable key; got none")
            failed = True

        hits = _payload(await session.call_tool("cite.search", {"query": "attention"}))
        print(f"cite.search 'attention' -> {hits.get('n_matches')} match(es)")

    n_uncited = len(checked.get("uncited", []))
    if n_uncited != 1:
        print(f"expected exactly 1 uncited claim, got {n_uncited}")
        failed = True

    return 1 if failed else 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
