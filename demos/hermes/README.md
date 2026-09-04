# Hermes agent demo recordings

## What you get

Two asciinema recordings of a real Hermes Agent session (`hermes chat`)
calling a router tool over MCP: `runs.ask` on the `attestation-provenance`
skill against the run-ledger fixture in `examples/workspace/`
(`record-provenance.sh`), and `feed.ask` on the `attestation-feed` skill
against a tagged demo corpus (`record-feed.sh`). This is the one pair of
demos that drives a real *agent*, not a script calling the tools directly
-- everything else under `demos/` shows the tools themselves.

Two real bugs surfaced building the provenance recording (both fixed on
`main` before it): `runs.ask` silently compared arms by whichever metric
most of them shared rather than the one named in the question, and it
never named the winner in its own summary even though the tool it called
knew one. A follow-up finding, also fixed: text-extracting the metric
from the model's own paraphrase of the question is not reliable (a real
session normalised "using the wer metric, compare..." down to "which arm
won?" three runs straight) -- `runs.ask` now takes an explicit `metric`
parameter, and the skill tells the agent to pass it.

A separate finding on the feed recording: with no toolset restriction,
hermes carries ~16k prompt tokens of its OWN built-in tools (filesystem,
terminal, browser, ...) alongside attestation's 2, and gemma4:e2b
sometimes talked itself out of calling a tool it had already correctly
identified in its own reasoning trace ("I need to know which feeds they
subscribe to first" -- invented, not asked for), or reasoned to the right
call and then printed it as literal text instead of invoking it. `-t
<mcp-server-name>` (below) restricts hermes to just that server's tools
and made both failures reliably go away in repeated live testing.

## Prerequisites

- A running Ollama server with `gemma4:e2b-it-q4_K_M` pulled.
- `hermes` (Hermes Agent) installed and configured, with `attestation`,
  `attestation-provenance` and `attestation-feed` MCP server entries in
  `~/.hermes/config.yaml`.

## Run it

```bash
uv tool install asciinema         # once
cargo install --locked --git https://github.com/asciinema/agg   # once
```

Both scripts need `~/.hermes/config.yaml`'s `attestation` (unrestricted)
entry disabled and the skill-specific entry enabled with `ATTEST_DB`
pointed at a scratch database, or hermes may connect to the wrong server
or the wrong data. Edit `mcp_servers` there, then `attest reload`.

### `record-provenance.sh`

```bash
ATTEST_DB=/tmp/attest-hermes-demo.db uv run attest runs scan --root ../../examples/workspace
# add `env: {ATTEST_TOOLS: provenance, ATTEST_DB: /tmp/attest-hermes-demo.db}`
# to attestation-provenance's entry, enable it, disable attestation's, reload
./record-provenance.sh
```

### `record-feed.sh`

```bash
ollama serve   # or whatever LLM_BASE_URL points at
uv run python ../feed/seed_feed_db.py /tmp/attest-hermes-feed-demo.db     # ~2 min, tags 40 items for real
# add `env: {ATTEST_TOOLS: feed, ATTEST_DB: /tmp/attest-hermes-feed-demo.db}`
# to attestation-feed's entry, enable it, disable attestation's, reload
./record-feed.sh
```

`narrate-feed.sh` is the script under recording for `record-feed.sh` --
split out because the query's own apostrophe ("I'm demo-reader") does not
survive `asciinema rec --command "bash -c '...'"`'s nested quoting.

Both scripts pass `-t <mcp-server-name>` to `hermes chat` -- the MCP
server name from `~/.hermes/config.yaml`, used as an exclusive toolset
allowlist. Reuse the same flag (with that server's own name) for any
other skill: `-t attestation-knowledge`, etc.

Writes `../../demo/hermes-provenance.cast`/`.gif` and
`../../demo/hermes-feed.cast`/`.gif` (gitignored, not committed).
