<!-- checked by tests/test_golden_paths.py -->

# Example agents

## What you get

The path that replaces reading 300 README lines to get an agent connected:
what the doctor reports about a fresh checkout, the per-surface configs
`attest emit` generates, and one of the four `AGENT_SURFACES` (`provenance`)
driven over stdio the way hermes-agent actually drives it — spawn
`attest-mcp`, list tools, call each one, read the envelope back.

## Prerequisites

`none — pure local computation`

The MCP surface itself needs no model: `runs.*` reads files off disk and
`cite.check` is a lint. `mcp_e2e.py --offline` points the *rest* of the
in-process stub server at `stub_openai.py` so the process starts the same
way in CI as it does here — no Ollama, no network.

## Run it

```bash
uv run attest install --check || true
uv run attest emit
ATTEST_TOOLS=provenance ATTEST_EXPAND=1 uv run python ../flows/mcp_e2e.py --surface provenance --offline
```

Relative to the repo root (`run.sh` does `cd "$(dirname "$0")/../.."` first).

## What it prints

```
13 calls, 0 failed
```

Abridged — `attest install --check` prints one aligned `[ok]`/`[BROKEN]`/
`[skipped]` line per step (every model-dependent step reports `[BROKEN]`
with no server reachable; the run ledger and claim checker steps need none
of them); `attest emit` reports `N config problem(s)` or `mcp surfaces: all
4 present and current` depending on whether a `hermes` binary is on `PATH`,
then always reports the four `.claude/agents/attestation-*.md` files as
present and current (this checkout already has them — `attest emit` never
overwrites, so re-running never changes that line); `mcp_e2e.py` prints one
row per tool call on the `provenance` surface — `provenance runs.claims_coverage
ok` and eleven more like it — ending in the pinned line above.

## What it demonstrates

**The four surfaces.** `ATTEST_TOOLS=provenance` restricts the spawned
server to the `runs.*` + `cite.check` prefixes in `AGENT_SURFACES["provenance"]`
— `feed.list` or `kg.path` are absent from `list_tools()`, not merely
undocumented, because a model that can see a tool will eventually call it.
The other three surfaces (`feed`, `knowledge`, `symbolic`) are the same
mechanism with a different prefix set; `mcp_e2e.py` (no `--surface`) drives
all four plus the unrestricted server in one pass.

**Progressive disclosure.** Without `ATTEST_EXPAND=1` a restricted surface
serves two tools: the surface's own `.ask` router plus one companion. The
specific tools underneath (`runs.scan`, `runs.compare`, …) exist on the
server the whole time; `ATTEST_EXPAND=1` is what reveals them. This flow
sets it because the script calls the specific tools directly, the way an
agent that already knows what it wants would.

**Why `attest emit` never overwrites.** It reports a difference between
the generated per-surface config and what is on disk — `missing`, `stale`,
or `orphaned` — and leaves the file alone. `--write` (not used here) creates
what is missing and refuses to touch a `.claude/agents/*.md` that already
differs from the generated body, because the realistic case is not running
it fresh, it is adding a fifth surface after someone has already hand-edited
one of the four.

**`attest reload` and the stale-server problem.** `attest-mcp` is spawned
once per session and holds that code until it dies; editing `src/` and
calling a tool again talks to the old process. `attest reload` SIGTERMs
every live `attest-mcp`; the respawn is lazy — nothing happens for several
seconds, then the next tool call gets the new code. `hermes mcp test` does
not catch this: it spawns a fresh process to probe the connection, so it
always reports the code on disk, never the code a long-running session is
actually running.

## When it goes wrong

- `ATTEST_TOOLS` set to anything other than `feed`, `provenance`,
  `knowledge` or `symbolic` raises at server startup rather than silently
  serving everything — a typo is loud, not a fallback to the full surface.
- `attest install --check` exits nonzero the moment any step reports
  `[BROKEN]`, which is every model-dependent step when no server is
  reachable at `LLM_BASE_URL`; `run.sh` carries `|| true` on that line so
  the rest of the script still runs. Each `[BROKEN]` line names the command
  that still works without a model (`attest runs scan` / `attest runs
  compare`), which is the honest report the doctor exists to give, not a
  bug.
- A server already running stale code (see *What it demonstrates* above)
  answers tool calls that silently don't match `src/` — the fix is `attest
  reload`, not restarting the agent, since the agent's own process was
  never the one holding the code.

## Next

See `docs/guides/agents.md` (the agents guide) for the manual, step-by-step
registration this path automates, and the catalogue at `examples/README.md`
for the other golden paths.
