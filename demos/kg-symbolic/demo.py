# demos/kg-symbolic/demo.py
"""Narrated walkthrough of kg.* and sym.* over stdio, for the terminal demo.

Pattern copied from examples/citations/check_citations.py: spawn attest-mcp
with ATTEST_TOOLS=knowledge,symbolic ATTEST_EXPAND=1, list tools, call a
handful with real arguments, print what came back. kg.* needs a database
with real tagged items -- run seed_kg_db.py first and pass its path via
ATTEST_DB. sym.* needs no data at all.

    uv run python seed_kg_db.py /path/to/demo.db   # once; needs a model server
    ATTEST_DB=/path/to/demo.db uv run python demo.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _payload(result) -> dict:
    return json.loads(result.content[0].text) if result.content else {}


def _heading(text: str) -> None:
    print(f"\n--- {text} ---")


async def run() -> int:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    # kg.* lives on the `knowledge` surface, sym.* on `symbolic` -- ATTEST_TOOLS
    # takes exactly one surface name, so this demo leaves it unset and gets
    # all 46 tools rather than running two sessions for two namespaces.
    env = {k: v for k, v in os.environ.items() if k not in ("ATTEST_TOOLS", "ATTEST_EXPAND")}
    params = StdioServerParameters(
        command="uv", args=["run", "--project", str(REPO_ROOT), "attest-mcp"], env=env
    )

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        _heading("kg.concepts — what's in this reader's graph")
        concepts = _payload(await session.call_tool("kg.concepts", {"limit": 10}))
        for c in concepts.get("concepts", []):
            print(f"  {c}")

        _heading("kg.central — what this reader's work centres on")
        central = _payload(await session.call_tool("kg.central", {"metric": "degree", "limit": 5}))
        for row in central.get("nodes", []):
            print(f"  {row['name']:<28} score={row['score']}")

        _heading("kg.neighbors('distributed-training') — what else to read")
        neighbors = _payload(
            await session.call_tool("kg.neighbors", {"node": "distributed-training", "limit": 5})
        )
        for row in neighbors.get("neighbors", []):
            print(f"  {row['name']:<28} weight={row['weight']}")

        _heading("kg.path — connected topics")
        path = _payload(
            await session.call_tool(
                "kg.path", {"source": "machine-learning", "target": "model-parallelism"}
            )
        )
        print(f"  machine-learning -> model-parallelism: {path.get('path')}")

        _heading("kg.path — disjoint topics (a real 'no path' answer, not an error)")
        no_path = _payload(
            await session.call_tool(
                "kg.path", {"source": "machine-learning", "target": "catalysis"}
            )
        )
        print(f"  machine-learning -> catalysis: ok={no_path.get('ok')} path={no_path.get('path')}")

        _heading("kg.communities — this reader's research areas")
        communities = _payload(await session.call_tool("kg.communities", {"min_size": 2}))
        for group in communities.get("communities", []):
            print(f"  {group['label']}: {', '.join(group['members'])}")

        _heading("sym.simplify — canonical form")
        simplified = _payload(
            await session.call_tool("sym.simplify", {"expr": "(x**2 - 1)/(x - 1)"})
        )
        print(f"  (x**2 - 1)/(x - 1) -> {simplified.get('result')}")

        _heading("sym.solve — roots")
        solved = _payload(await session.call_tool("sym.solve", {"expr": "x**2 - 4"}))
        print(f"  x**2 - 4 = 0 -> {solved.get('result')}")

        _heading("sym.differentiate")
        diff = _payload(
            await session.call_tool("sym.differentiate", {"expr": "x**3 + 2*x", "symbol": "x"})
        )
        print(f"  d/dx(x**3 + 2*x) -> {diff.get('result')}")

        _heading("sym.integrate — definite, with bounds")
        integral = _payload(
            await session.call_tool("sym.integrate", {"expr": "x**2", "bounds": [0, 1]})
        )
        print(f"  integral of x**2 from 0 to 1 -> {integral.get('result')}")

        _heading("sym.derivation — the steps, not just the answer")
        derivation = _payload(
            await session.call_tool("sym.derivation", {"expr": "x**2", "operation": "integrate"})
        )
        for step in derivation.get("steps") or []:
            rule = step.get("rule", "?")
            integrand = step.get("integrand", "")
            print(f"  [{step.get('depth')}] {rule}: {integrand}")

        _heading("sym.verify — is this identity true?")
        verified = _payload(
            await session.call_tool("sym.verify", {"lhs": "(x+1)**2", "rhs": "x**2 + 2*x + 1"})
        )
        print(f"  (x+1)**2 == x**2 + 2*x + 1 -> {verified.get('verdict')}")

        _heading("sym.evaluate — units conversion")
        evaluated = _payload(
            await session.call_tool(
                "sym.evaluate", {"expr": "5", "units": "meter/second -> kilometer/hour"}
            )
        )
        print(f"  5 m/s -> km/h: {evaluated.get('numeric')}")

    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
