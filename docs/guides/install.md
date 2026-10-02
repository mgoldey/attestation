# Install

How do I set it up, with or without a model server? One command
(`attest install`) handles both tiers, and everything below is reference for
what it automates.

## Prerequisites

Two tiers, because half this tool needs no model at all:

**For the run ledger and claim checker** — Python 3.12+ and
[`uv`](https://docs.astral.sh/uv/). That is the whole list. `ledger.py` and
`claims.py` import no LLM or embedding module, and the quickstart above is
verified against an unreachable backend. If that's all you want, stop here —
nothing below this tier is required, and nothing gets downloaded.

**Additionally, for the feed, tagging, and knowledge graph** —
[Ollama](https://ollama.com/download) installed and running locally
(`ollama serve`, or the desktop app). This is a separate download from
attestation itself and is not something `attest install` can do for you.

Once Ollama is running, `attest install` pulls the required models — no
manual `ollama pull` needed — but budget for the download before you run it:

| Model | Size | Role |
|-------|------|------|
| `gemma4:e2b-it-q4_K_M` | 7.2 GB | chat: explanations + tagging |
| `embeddinggemma` | 621 MB | embeddings |
| **Total** | **~7.8 GB** | |

While warm, the two models together hold roughly 5.4 GB resident in
RAM/VRAM for `OLLAMA_KEEP_ALIVE` (default 30 minutes) — see the `.env.sample`
warning about `keep_alive=-1` before changing that default; a permanent pin
OOM-killed a 23 GB box in this project's own history. Check free disk space
and RAM/VRAM against these numbers before you start; on a metered or
capped connection, budget for a 7.8 GB download.

`gemma4:e2b` is documented as needing `ollama >= 0.32.9` (earlier builds
abort with a `GGML_ASSERT` in the graph scheduler). Nothing in this repo
checks the daemon's version — `attest install`/`--check` verify model
*presence*, not the Ollama version — so treat this as a manual requirement:
run `ollama --version` yourself and upgrade if it's older.

Budget for the rest of setup too: first `ingest` takes several minutes
(depends on feed count and size), and `attest tag` runs at roughly
2.3s/item, so tagging is the slower, backgroundable step. Tagging is also
deferrable — ranking only joins items/item_vectors/feeds, so untagged items
still rank fine; tags just aren't there yet for tag chips, digest
clustering, and source suggestions.

## One-liner

From PyPI, with nothing cloned (the package ships a console script under its
own name, so `uvx attestation ...` is the whole command):

```bash
uvx attestation install          # idempotent setup, see below
uvx attestation install --check  # diagnose only
uvx attestation serve            # http://127.0.0.1:8899
```

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

## Hosted models instead of Ollama

The feed tier's model calls speak the OpenAI-compatible API, so any hosted
endpoint replaces Ollama: no GPU, no 7.8 GB download. Set four variables
(`.env.sample` has the block) and run the check:

```bash
LLM_BASE_URL=https://integrate.api.nvidia.com/v1   # NVIDIA NIM, for example
LLM_API_KEY=nvapi-...
CHAT_MODEL=<a chat model your account can call>
EMBED_MODEL=nvidia/nemotron-3-embed-1b
uvx attestation install --check
```

With a non-local `LLM_BASE_URL` the Ollama steps are skipped and one step,
`hosted_models`, makes two tiny real requests -- one embedding and a
one-token chat completion -- and reports the server's own reason when either
fails. That is deliberate: a hosted catalogue lists models an account cannot
call. Measured on 2026-09-11 against NIM, 82 models were listed;
`nvidia/nemotron-3-embed-1b` answered (2048 dims, truncated client-side to
`EMBED_DIMS`), and every chat model tried returned `410 Gone` (end of life)
or `404` (not enabled for the account). The check prints exactly that:

```
[BROKEN] hosted_models: meta/llama-3.1-8b-instruct: HTTP 410 Gone -- The model
'meta/llama-3.1-8b-instruct' has reached its end of life on 2026-08-26 ...
```

Three things change when you leave Ollama. Titles, abstracts and a
persona's interests text are sent to the endpoint -- the run ledger and
claim checker never touch a model and stay local either way. The stored
embedding model is pinned per database, so switching means a fresh database.
And the tagging, explanation and reaction prompts were measured on
`gemma4:e2b`; `evals/` re-measures them on a different model.

## Chat on the model Hermes is connected to

On a machine provisioned for an agent (a hosted Research Desk), the customer
connects a model to Hermes -- NVIDIA NIM, ChatGPT sign-in, an OpenAI or
Anthropic key, OpenRouter, an Ollama or other OpenAI-compatible endpoint --
while Attestation embeds on a small loopback server of its own. Attestation
never learns what model Hermes runs unless the host says so. Without it, chat
falls through to the built-in Ollama default and asks the embedder-only server
for `gemma4:e2b-it-q4_K_M`; the server answers `404 model ... not found`. Set

```bash
ATTEST_LLM_FROM_HERMES=1
```

and the chat endpoint, model and key come from the model Hermes is connected to,
whichever provider that is. The default is off, so nobody else's behaviour
changes. Embeddings are never part of it: they stay on `EMBED_BASE_URL`, and a
chat provider with no embeddings API is never asked for one.

**Resolution order for chat**, per field, first match wins, read on every call
(so "Change model" in AgentMarkit is followed with no restart):

1. `LLM_BASE_URL` / `CHAT_MODEL` / `LLM_API_KEY`, when set to something of your
   own. A `LLM_BASE_URL` other than the built-in default is a decision: Hermes
   is not consulted at all. (The two lines `.env.sample` ships uncommented --
   the built-in URL and model -- count as unset, because `attest install`
   copies them into `.env` on every machine.)
2. With `ATTEST_LLM_FROM_HERMES=1`: the model Hermes is connected to.
3. The built-in Ollama default.

**Where Hermes keeps it, and who makes the call.** Files under `HERMES_HOME`
(default `~/.hermes`) are what is read, because the hourly refresh runs under
cron outside the agent: `config.yaml` (`model.default`, `model.provider`,
`model.base_url`, `model.api_key`, `model.api_mode`), `.env` (the provider's key)
and `auth.json` (OAuth credentials, read only by Hermes' own code). The process
environment is consulted in exactly one case: a key variable (`NVIDIA_API_KEY`,
`${CUSTOM_API_KEY}`...) that `.env` does not set. `.env` wins.

**Which key goes where** follows Hermes' own routing, because a key sent to the
wrong host is a leak: an explicit `model.api_key` is used, and a `${VAR}` or
`${env:VAR}` reference to a variable that is not set is an error naming it, never
a fall back to some other variable; otherwise the provider's own variable.
`OPENAI_API_KEY` is used for a `custom` endpoint only when its host is
`openai.com` or `openai.azure.com`. `LLM_API_KEY` is not paired with Hermes'
URL (it was set for an `LLM_BASE_URL`, and a set `LLM_BASE_URL` skips Hermes
altogether). A key is never sent over plain `http` to a host that is not this
machine (the Ollama placeholder key `ollama` is not a secret and is exempt), and
a key with a space, newline or non-ASCII character is refused by name rather
than failing every request. Keys and OAuth tokens are redacted from every
message Attestation prints or logs, on both paths, and a 401 or 403 is reported
as the status alone.

| What Hermes is connected to | Who calls it | Key comes from |
|---|---|---|
| NVIDIA NIM (`nvidia`), OpenRouter, OpenAI (`openai-api`), DeepSeek, xAI, AI Gateway, Hugging Face, GMI, Arcee | Attestation, OpenAI chat-completions | `<HERMES_HOME>/.env` (`NVIDIA_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, ...), else the process environment |
| A custom OpenAI-compatible endpoint (`custom`), Ollama or llama.cpp as Hermes runs them, LM Studio | Attestation, OpenAI chat-completions | `model.api_key`, usually `${CUSTOM_API_KEY}` resolved in `.env`; none needed for a local server |
| ChatGPT sign-in (`openai-codex`, OAuth), Anthropic's native API, Bedrock, Vertex, any provider not in the rows above, any `model.api_mode` other than `chat_completions` | **Hermes**, through its own runtime | held by Hermes; Attestation never reads or refreshes it |

The third row exists because those providers do not speak chat-completions with
a static key. A ChatGPT sign-in is OAuth against the Responses API, and OAuth
refresh tokens are single-use, so a second process refreshing the same
credential would sign the first one out. Attestation therefore starts a small
helper (with a minimal environment: no `LLM_API_KEY`, `EMBED_API_KEY` or other
secrets, so Hermes' fallback chain cannot silently chat on another provider)
in Hermes' own Python (`hermes_helper.py`; found through the `hermes`
launcher on `PATH` or in `~/.local/bin`, else `<HERMES_HOME>/hermes-agent/venv`,
else `ATTEST_HERMES_PYTHON`) which calls Hermes' own `agent.auxiliary_client.
call_llm` -- the function Hermes uses for every non-agent model call. The
credential, the wire format and the token refresh stay in Hermes' process. It
costs a subprocess per run (about 3-7 s to the first reply, then milliseconds),
and the reply schema goes in the prompt instead of `response_format`. If a
Hermes release changes `call_llm`, the helper says so with Hermes' version and
the three variables to set instead.

**What it says when something is wrong**, each naming where the URL and model
came from (`env LLM_BASE_URL`, `Hermes (nvidia, <path>)`, `built-in Ollama
default`) and how to change it, and never a key:

- `Hermes has no model connected; connect one in AgentMarkit (...)` -- no
  `config.yaml`, no `model.default`, or Hermes signed out;
- `Hermes has a model (..., provider nvidia) but no API key: set NVIDIA_API_KEY
  in .../.env, or connect one in AgentMarkit`;
- `... HTTP 404: model 'x' not found` / `HTTP 401` / `HTTP 410` -- the server
  *was* reached and refused. This stops the run once instead of retrying every
  item; before, only a dead socket did, and a 404 printed nothing but
  `failed: N`;
- `cannot connect (ConnectError)` -- the URL named is where it tried.

A 404 or 410 stops the run even when it is a proxy's passing hiccup rather than a
missing model. Nothing is lost: progress is saved per item and the next hourly
refresh picks up where it stopped. A 429 or 502/503/504 is retried twice with a
short, capped backoff (`Retry-After` honoured up to 5 s) before it counts.
A call to Hermes' runtime has a hard deadline (the request timeout plus 5 s) after
which the helper process is killed, and a helper cannot outlive Attestation: it
exits when its parent does.

`attest install --check` prints one `backends` line: the resolved chat backend
(host, model, and where each came from) and the embedding backend -- hosts and
names only, never a key -- and its `hosted_models` step sends one real request
to each. It is BROKEN when the host opted in and Hermes has nothing usable.

**Not verified against the real service:** NVIDIA NIM, OpenAI, OpenRouter and
the ChatGPT sign-in were exercised against stubs shaped like their wires (and,
for the two Hermes-served providers, against Hermes' real code with those
stubs behind it -- `tests/test_hermes_real.py`, opt-in), not against the
live services. Bedrock, Vertex and Copilot are delegated to Hermes by the same
mechanism but have no test of their own.

## What `attest install` does (manual-setup reference)

The steps below are what the installer automates. You normally don't need
to do any of this by hand — it's here as reference for what's happening
under the hood, or if you'd rather configure a piece yourself.

<details>
<summary>Manual setup steps</summary>

#### Models

```bash
ollama pull embeddinggemma        # 621 MB, 256-dim embeddings (required)
ollama pull gemma4:e2b-it-q4_K_M  # 7.2 GB, chat model for explanations + tagging
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

## Publishing a release (maintainers)

`.github/workflows/release.yml` builds and publishes to PyPI on a `v*` tag
through trusted publishing, so no token lives in the repo. One-time setup on
pypi.org: project `attestation`, Publishing, add a GitHub publisher with
owner `mgoldey`, repository `attestation`, workflow `release.yml`,
environment `pypi`. Then bump `version` in `pyproject.toml`, move the
changelog's Unreleased entries under the version, and:

```bash
git tag v0.2.0 && git push origin v0.2.0
```

The build job refuses a tag that does not match the pyproject version.
