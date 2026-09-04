# Install

How do I set it up, with or without a model server? One command
(`attest install`) handles both tiers, and everything below is reference for
what it automates.

## Prerequisites

Two tiers, because half this tool needs no model at all:

**For the run ledger and claim checker** — Python 3.12+ and
[`uv`](https://docs.astral.sh/uv/). That is the whole list. `ledger.py` and
`claims.py` import no LLM or embedding module, and the quickstart above is
verified against an unreachable backend.

**Additionally, for the feed, tagging, and knowledge graph** —
[Ollama](https://ollama.com) running locally. `attest install` pulls the
required models (`embeddinggemma` for embeddings, `gemma4:e2b` for
explanations and tagging by default) — no manual `ollama pull` needed.
gemma4:e2b needs ollama >= 0.32.9. Budget for it: a first `ingest` of ~1000
items takes about 6 minutes, and `attest tag` runs at roughly 2.3s/item, so
tagging that same 1000 items is a ~40-minute unattended job.

## One-liner

`attest install` is an idempotent setup command: it creates `.env`, pulls
missing Ollama models, runs the first ingest, and — if a local
[hermes-agent](https://github.com/NousResearch/hermes-agent) install is
found — wires up the MCP server, the skill copy, the reasoning override,
and the refresh cron job. Re-running it repairs whatever's missing; nothing
it does is destructive.

From a local clone:

```bash
git clone https://github.com/mgoldey/attestation ~/attestation
cd ~/attestation
uv sync
uv run attest install
```

Or with no checkout at all:

```bash
uvx --from git+https://github.com/mgoldey/attestation attest install
```

Add `--check` to see what's missing without changing anything (exits 1 on
gaps — useful in scripts), and `--yes` to skip the confirmation prompt for
non-interactive runs. `setup.sh` (see the [agents guide](agents.md)) wraps
this same command.

```bash
uv run attest install --check   # diagnose only
uv run attest install --yes     # non-interactive repair
```

Once installed:

```bash
uv run attest serve             # http://127.0.0.1:8899
```

To talk to it from Discord instead of the browser, see the README's
"Chat with it from Discord" and section 8 of the [agents guide](agents.md).

The first screen asks who is reading and what about -- ranking starts from
that interests text alone, and a new database has no personas until someone
answers. To compare per-identity ranking before you have clicked anything,
`attest bootstrap-persona bench-chemist` (or `ml-engineer`, `researcher`)
creates that demo persona and gives it pseudo-clicks.
Click ✓/✗ on items; the feed retrains and re-ranks on every click. Switch users
in the nav to see the same feed ranked per-identity.

`feeds.toml` seeds the feed list when the database is first created. After
that the **database is the source of truth**: use the `feed.source_add` /
`feed.source_remove` MCP tools (or edit the database directly) to change which feeds
are tracked, then run `uv run attest ingest` to fetch from any newly added
feed. Editing `feeds.toml` after the first ingest has no effect.

## What `attest install` does (manual-setup reference)

The steps below are what the installer automates. You normally don't need
to do any of this by hand — it's here as reference for what's happening
under the hood, or if you'd rather configure a piece yourself.

<details>
<summary>Manual setup steps</summary>

#### Models

```bash
ollama pull embeddinggemma        # 256-dim embeddings (required)
ollama pull gemma4:e2b-it-q4_K_M  # chat model for explanations + tagging
```

The default chat model is `gemma4:e2b-it-q4_K_M`; set `CHAT_MODEL` to
override. Measured on 2x GTX 1080 (8 GB each), ollama 0.32.9: 2.2 GB resident,
100% GPU, ~2.2s per tagging call. `gemma4:12b` partially CPU-offloads on
8 GB-class cards (~60-90s/call), and `hermes3:3b` is faster still but emitted a
malformed tag on 40% of items.

```bash
export OLLAMA_MAX_LOADED_MODELS=2   # keep chat + embed models co-resident
uv run attest warmup                # pin both models in VRAM for 30 min (a cold load is ~30s)
uv run attest ingest                # fetch feeds.toml -> hermes.db
```

#### Configuration (.env)

```bash
cp .env.sample .env    # then edit — gemma4:e2b is pre-selected as the chat model
```

The `attest` CLI and the MCP server load `.env` at startup (real environment
variables always win), so your shell, cron, and hermes-agent-spawned
processes all see the same configuration. All LLM traffic speaks the
OpenAI-compatible API (`LLM_BASE_URL`, default Ollama's
`http://localhost:11434/v1`) — point it at vLLM, llama.cpp server, or
OpenRouter (set `LLM_API_KEY`) to swap backends. See `.env.sample`
for every variable, including the Ollama daemon settings
(`OLLAMA_KEEP_ALIVE=30m`, `OLLAMA_CONTEXT_LENGTH=32768`) that replace the
per-request pinning the native API used to provide.

For a machine that chats through hermes-agent all day, 30 minutes is too
short: every message after a quiet spell paid the cold load again. The
[agents guide](agents.md#8-chat-from-discord-or-telegram) has the permanent
pin (`keep_alive: -1` via a user timer, no sudo) and the per-platform tool
allowlist that together took a Discord turn from 54-249s to 23-28s.

#### No-checkout alternative (uvx-from-git)

The engine runs without cloning:

```bash
uvx --from git+https://github.com/mgoldey/attestation attest ingest
uvx --from git+https://github.com/mgoldey/attestation attest serve
```

Note the package is `attestation` but its console script is `attest`
(`[project.scripts] attest = "attestation.cli:main"`) — with `uvx`, `--from`
takes the *package*, the trailing word is the *executable*, so
`uvx --from attestation attest ...`. The script was deliberately not named
`hermes`: that shadowed hermes-agent's own binary inside the venv, which made
`_find_agent_binary()` need a `sys.prefix` guard to avoid calling itself.

</details>
