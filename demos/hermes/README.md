# Hermes agent demo recording

## What you get

An asciinema recording of a real Hermes Agent session (`hermes chat`)
calling `runs.ask` over MCP on the `attestation-provenance` skill, against
the run-ledger fixture in `examples/workspace/`. This is the one demo that
drives a real *agent*, not a script calling the tools directly -- everything
else under `demos/` shows the tools themselves.

Two real bugs surfaced building this (both fixed on `main` before this
recording): `runs.ask` silently compared arms by whichever metric most of
them shared rather than the one named in the question, and it never named
the winner in its own summary even though the tool it called knew one. A
follow-up finding, also fixed: text-extracting the metric from the model's
own paraphrase of the question is not reliable (a real session normalised
"using the wer metric, compare..." down to "which arm won?" three runs
straight) -- `runs.ask` now takes an explicit `metric` parameter, and the
skill tells the agent to pass it.

A separate finding on the feed surface (attestation-feed, same session):
with no toolset restriction, hermes carries ~16k prompt tokens of its OWN
built-in tools (filesystem, terminal, browser, ...) alongside attestation's
2, and gemma4:e2b sometimes talked itself out of calling a tool it had
already correctly identified in its own reasoning trace ("I need to know
which feeds they subscribe to first" -- invented, not asked for), or
reasoned to the right call and then printed it as literal text instead of
invoking it. `-t <mcp-server-name>` (below) restricts hermes to just that
server's tools and made both failures reliably go away in repeated live
testing.

## Prerequisites

- A running Ollama server with `gemma4:e2b-it-q4_K_M` pulled.
- `hermes` (Hermes Agent) installed and configured, with an `attestation`
  or `attestation-provenance` MCP server entry in `~/.hermes/config.yaml`.

## Run it

```bash
uv tool install asciinema         # once
cargo install --locked --git https://github.com/asciinema/agg   # once

# Point the attestation MCP server your ~/.hermes/config.yaml already has
# at a scratch database seeded from the workspace fixture:
ATTEST_DB=/tmp/attest-hermes-demo.db uv run attest runs scan --root ../../examples/workspace
# then add `env: {ATTEST_DB: /tmp/attest-hermes-demo.db}` to that server's
# entry in ~/.hermes/config.yaml and `attest reload` (see record.sh's own
# comments for the exact block).

./record.sh
```

`record.sh` passes `-t attestation-provenance` to `hermes chat` -- the MCP
server name from `~/.hermes/config.yaml`, used as an exclusive toolset
allowlist. Reuse the same flag (with that server's own name) for any other
skill: `-t attestation-feed`, `-t attestation-knowledge`, etc.

Writes `../../demo/hermes.cast` and `.gif` (gitignored, not committed).
