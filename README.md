# attestation

**Auditable research provenance, fully local.** Every number this tool reports
traces back to the file that produced it, and it refuses to state more than the
evidence supports.

Research generates artifacts — config files, eval dumps, benchmark tables — and
the numbers that end up in a README get transcribed from them by hand, where
nothing checks them again. `attestation` closes that loop, and adds the
surrounding tooling a working scientist needs:

| Capability | What it answers |
|---|---|
| **Run ledger** | "Which arm of that sweep actually won, and on what evidence?" |
| **Claim checker** | "Is what my README says still true?" |
| **Knowledge graph** | "What connects to what in my reading?" |
| **Symbolic math** | "Is this derivation right?" |
| **Feed ranking** | "What should I read next?" |

Design constraints throughout: **fully local** (Ollama by default, any
OpenAI-compatible backend), a single SQLite file, no new services, and a strong
bias against inventing structure the data does not have. A tool that reports
success for work it did not do is worse than no tool.

There is exactly one documented exception, and it is off unless you turn it on:
setting `ATTEST_CITATION_WEB` adds a citation reader that queries
api.crossref.org. It is checked when the resolver is built rather than when it
is called, so a disabled reader cannot be coaxed into a request, and
`cite.sources()` reports which readers can reach the network.

The ledger reads artifacts that already exist — no instrumentation, no
`log_metric()` calls, no change to how anything runs. That is deliberate:
adoption cost is the constraint that decides whether a tool gets used at all.

## Feed ranking

The original core, still here. An agent orchestrator for personalized feed
recommendations coordinating three layers with distinct reliability contracts:
deterministic ingest (fetch → dedup → embed), a per-user learnable ranking core
(profile embedding + click-trained classifier), and a LangGraph explain agent
that says *why* items rank where they do — lazily, cached, never blocking the
feed. The LLM is a swappable OpenAI-compatible backend, not the point.

It runs two ways, and they share one database:

1. **Standalone** — a web UI at `http://127.0.0.1:8899` with ✓/✗ feedback
   buttons that retrain the ranker on every click.
2. **Alongside [hermes-agent](https://github.com/NousResearch/hermes-agent)** —
   as an MCP stdio server exposing the tools below as native tool calls, plus an
   optional `research-provenance` skill for setup automation and fallback.
## Installation

### Prerequisites

- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally. `attest install` pulls the
  required models (`embeddinggemma` for embeddings, `gemma4:e2b` for
  explanations and tagging by default) for you — no manual `ollama pull`
  needed. gemma4:e2b needs ollama >= 0.32.9.

### One-liner

`attest install` is an idempotent setup command: it creates `.env`, pulls
missing Ollama models, runs the first ingest, and — if a local
[hermes-agent](https://github.com/NousResearch/hermes-agent) install is
found — wires up the MCP server, the skill copy, the reasoning override,
and the refresh cron job. Re-running it repairs whatever's missing; nothing
it does is destructive.

No checkout, straight from git — replace `REPO_URL` below with your actual
remote (see "Once the repo has a reachable remote" further down for how the
URL gets set for hermes-agent too):

```bash
REPO_URL=https://github.com/you/attestation
uvx --from "git+$REPO_URL" attest install
```

From a local clone (replace `<your-attestation-remote-url>` with your remote):

```bash
git clone <your-attestation-remote-url> ~/attestation
cd ~/attestation
uv sync
uv run attest install
```

Add `--check` to see what's missing without changing anything (exits 1 on
gaps — useful in scripts), and `--yes` to skip the confirmation prompt for
non-interactive runs. `setup.sh` (see "Launching alongside hermes-agent"
below) wraps this same command.

```bash
uv run attest install --check   # diagnose only
uv run attest install --yes     # non-interactive repair
```

Once installed:

```bash
uv run attest serve             # http://127.0.0.1:8899
```

Click ✓/✗ on items; the feed retrains and re-ranks on every click. Switch users
in the nav to see the same feed ranked per-identity.

`feeds.toml` seeds the feed list when the database is first created. After
that the **database is the source of truth**: use the `feed.source_add` /
`feed.source_remove` MCP tools (or edit the database directly) to change which feeds
are tracked, then run `uv run attest ingest` to fetch from any newly added
feed. Editing `feeds.toml` after the first ingest has no effect.

### What `attest install` does (manual-setup reference)

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
uv run attest warmup                # pin both models in VRAM (avoids 10-20s cold loads)
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
(`OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_CONTEXT_LENGTH=8192`) that replace the
per-request pinning the native API used to provide.

#### No-checkout alternative (uvx-from-git)

Once the repo has a reachable remote, the engine runs without cloning:

```bash
uvx --from git+<REPO_URL> attest ingest
uvx --from git+<REPO_URL> attest serve
```

Note the package is `attestation` but its console script is `attest`
(`[project.scripts] attest = "attestation.cli:main"`) — with `uvx`, `--from`
takes the *package*, the trailing word is the *executable*, so
`uvx --from attestation attest ...`. The script was deliberately not named
`hermes`: that shadowed hermes-agent's own binary inside the venv, which made
`_find_agent_binary()` need a `sys.prefix` guard to avoid calling itself.

</details>

## Launching alongside hermes-agent

attestation integrates with a locally installed
[hermes-agent](https://github.com/NousResearch/hermes-agent) (tested against
v0.20.0) through two lanes that complement each other:

- **MCP server (primary)** — native, typed tool-calling. No prose
  interpretation, no curl hallucination risk. The MCP process is stateless
  and opens the database directly, so the web server does **not** need to be
  running for the agent to use it.
- **Skill (optional)** — `src/attestation/skills/research-provenance/`
  documents workflow judgment ("when NOT to use this"), runs an idempotent
  setup/repair script, and provides an HTTP fallback path if MCP is
  disconnected. It lives inside the package so it ships in the wheel too —
  `uvx` installs get the skill without a checkout.

See `docs/hermes-agent-plugin-research.md` for why MCP was chosen over
hermes-agent's plugin system.

`attest install` (see "One-liner" above) automates steps 1, 3, and 6 below
when it finds a local hermes-agent binary — MCP registration, the skill
copy, and the refresh cron job. The steps are kept here as reference and
for anyone who wants to wire a piece up by hand.

### 1. Register the MCP server (one-time)

From anywhere (uses this checkout's path; adjust if you cloned elsewhere):

```bash
hermes mcp add attestation \
  --command uv \
  --args run --project /home/matt/attestation attest-mcp
```

This writes an `mcp_servers.attestation` block into `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  attestation:
    command: uv
    args: [run, --project, /home/matt/attestation, attest-mcp]
    enabled: true
```

Verify with `hermes mcp list` — you should see `attestation ... ✓ enabled`.

The server (`attest-mcp`, from `src/attestation/mcp_server.py`) exposes 50 tools.
These counts move: re-measure rather than quoting this paragraph.

```bash
uv run python -c "from mcp.server.fastmcp import FastMCP; \
from attestation.mcp import register_all; import asyncio; \
m=FastMCP('x'); register_all(m); print(len(asyncio.run(m.list_tools())))"
```


| Tool | What it does | Speed |
|---|---|---|
| `feed.personas()` | List reader personas + interest profiles | instant |
| `feed.list(user, limit)` | Ranked unread items, best first (capped at 50) | fast |
| `feed.search(user, query, tag, content_type, limit)` | Search the whole archive, ranked for this user; includes already-rated items | fast |
| `feed.read(user, item_id)` | Read ONE item in full — title, source, abstract | fast |
| `feed.rate(user, item_id, useful)` | Record a ✓/✗ click; retrains ranking | fast |
| `feed.explain(user, item_id)` | One-sentence "why did this rank here" | **slow** first call (local LLM), cached after |
| `feed.persona_create(name, interests)` | Create a reader persona | instant |
| `feed.persona_update(name, interests)` | Replace a persona's interests text | instant |
| `feed.persona_suggest_interests(limit)` | Most common tags, to help write an interests string | instant |
| `feed.persona_status(user)` | Click count, behavior-vs-text blend weight, top liked/disliked tags | instant |
| `feed.persona_delete(name, confirm)` | Delete a persona and its feedback (needs `confirm=true`) | instant |
| `feed.persona_reset(name, confirm)` | Clear a persona's clicks, keep the persona (needs `confirm=true`) | instant |
| `feed.harvest_engagement(user)` | Turn past "why is this here?" questions into weak positive feedback | fast |
| `feed.simulate_ratings(user, limit, confirm)` | Generate simulated reader reactions to train ranking (needs `confirm=true`) | **slow** (local LLM per item) |
| `feed.source_add(url, title)` | Subscribe to a feed (register-only; items arrive at the next ingest) | fast |
| `feed.sources()` | Subscribed feeds with item counts and last-fetched times | instant |
| `feed.source_preview(url, limit)` | Show a feed's recent entries without subscribing | fast |
| `feed.source_remove(feed_id, confirm)` | Unsubscribe; keeps existing items and feedback (needs `confirm=true`) | instant |
| `feed.source_suggest(user, limit)` | Suggest feeds from a curated list, scored against liked tags | instant |
| `sym.simplify(expr, timeout)` | Simplify an expression to canonical form | fast |
| `sym.solve(expr, symbol, timeout)` | Solve expr = 0 for a symbol | fast |
| `sym.differentiate(expr, symbol, order, timeout)` | Differentiate | fast |
| `sym.integrate(expr, symbol, bounds, timeout)` | Integrate, indefinitely or over bounds | fast |
| `sym.derivation(expr, operation, symbol, timeout)` | Step-by-step trace (genuine only for integrals) | fast |
| `sym.verify(lhs, rhs, timeout)` | Test symbolic equality; "unproven" is NOT a disproof | fast |
| `sym.evaluate(expr, subs, units, timeout)` | Numeric value, substitutions, unit conversion | fast |
| `kg.neighbors(node, limit)` | Concepts directly adjacent to this one in your reading graph | instant |
| `kg.path(source, target)` | Shortest chain of concepts linking two topics | instant |
| `kg.central(metric, limit)` | Most-connected or most-bridging concepts | instant |
| `kg.communities(min_size)` | Topic clusters, each labelled by its hub concept | instant |
| `kg.concepts(prefix, limit)` | List the concept names the other `kg_*` tools accept | instant |
| `runs.scan(root, project, confirm)` | Read experiment runs from artifacts on disk into the ledger | fast |
| `runs.list(project, family, limit)` | Recorded runs, and the families they group into | instant |
| `runs.compare(family, metric)` | Rank the arms of a sweep, with provenance and caveats | instant |
| `runs.detail(project, name)` | One run: config, every metric, source path, hypothesis header | instant |
| `runs.claims_check(path, verdict)` | Verify Markdown claims against recorded runs | instant |
| `runs.claims_coverage(path)` | Numbers asserted in prose that no claim covers | instant |
| `feed.digest(user, days, per_topic, limit)` | Ranked unread feed grouped by topic, with a ranking-quality caveat | fast |
| `cite.lookup(key)` | One bibliographic record, and which source it came from | fast |
| `cite.search(query, limit)` | Find references by title or author, from local sources only | fast |
| `cite.check(path)` | Claims whose citation key no configured source can resolve | fast |
| `cite.sources()` | Which citation sources are configured, and which can reach the network | instant |
| `feed.ask(user, question)` | Route a plain-language feed question to the right tool | fast |
| `kg.ask(question, source, target)` | Route a plain-language graph question to the right tool | instant |
| `runs.ask(question, family, path)` | Route a plain-language ledger question to the right tool | instant |
| `sym.ask(expr, question)` | Route a plain-language math question to the right tool | fast |
| `feed.tools()`, `kg.tools()`, `runs.tools()`, `sym.tools()` | List the tools this agent surface actually has | instant |

The knowledge graph is derived from the tagging pass, not from separate
content: concepts are tags used at least twice, and two concepts are linked
when they co-occur on at least two items. Spelling variants are merged via
`src/attestation/kg_aliases.toml` — without that, `machine-learning` and
`machinelearning` appear as two separate hubs and every centrality number is
wrong. Tags used only once (70% of them) are excluded: they connect to
nothing. Every `kg_*` read tool derives the graph fresh from `item_tags` on
each call via `build_graph`, which takes the tag assignments as a plain
argument rather than a connection.

There is no stored graph. `kg_nodes`, `kg_edges`, `kg_meta`, a `kg_rebuild`
tool and a `stale` flag existed until 2026-08-21; nothing ever read the
tables, and all eight `kg_*` answers were byte-identical with and without
them, so they were deleted rather than left as a cache that could only ever
be wrong.

Symbolic tools never `eval` your input: expressions are parsed against a
whitelist of mathematical names with builtins removed, and every computation
runs in a subprocess with a wall-clock timeout and a 2 GB memory cap. A
malicious expression is refused, and a runaway one is cancelled rather than
taking down the server — at the cost of roughly 0.3 s of process spawn per
call.

Every recorded click stores its provenance, and provenance decides what a row
may be used for:

| source | what it is |
|---|---|
| `ui` | you pressed a button on the web page |
| `agent` | an MCP `feed.rate` call, usually the agent reading your reply |
| `implicit` | you asked why an item ranked; engagement, counted as a weak positive |
| `simulated` | a local model reacting to the text as the persona would |
| `bootstrap` | synthetic persona seeding |

`bootstrap` labels are a linear threshold on the same embedding the ranker's
classifier consumes, so scoring against them is a tautology and
`evaluate_user` excludes them. The other four are trainable, and
`feed.persona_status` breaks the counts down by source so you can see how much of a
persona's history is yours.

Explicit feedback is scarce by nature — this project's own database held 68
web clicks and 2 agent clicks across 5,167 items before the two synthetic
channels were added, every one of them positive, which is why the click
classifier had never fired for a real account.

The MCP server reads `.env` from the checkout at startup, so `CHAT_MODEL` set there applies — no `--env` flag needed (though `--env` still works and wins over `.env`).

### 2. Required for local models: context window + eager tools

Two settings, both outside this repo, decide whether a local model can call
these tools through hermes-agent at all. Without them the model reports that
the tools do not exist — which looks like a broken MCP server and is not one.

**a) `OLLAMA_CONTEXT_LENGTH` >= 32768.** Ollama serves a **4096-token** window
by default, ignoring both the model's declared context (gemma4:e2b advertises
131072) and any `context_length` in hermes-agent's config. Verify what is
actually served — not what is configured — with:

```bash
curl -s http://localhost:11434/api/ps | python3 -m json.tool | grep context
```

hermes-agent sends ~13k of system context per turn (skill registry, persona,
memory headers, tool schemas), so at 4096 the tool schemas and the user's
question are truncated away before the model sees them. Fix it in the systemd
drop-in, each variable on **its own** `Environment=` line:

```bash
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'CONF'
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_CONTEXT_LENGTH=32768"
CONF
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

Two traps worth stating: putting this in the *vendor* unit
(`/etc/systemd/system/ollama.service`) gets overwritten on upgrade, and
appending it to an existing `Environment=` line makes systemd read the whole
string as one variable — the setting silently does nothing, and `PATH` is
clobbered as a side effect. Cost: a larger KV cache (gemma4:e2b goes 2.8 GB ->
4.0 GB resident).

**b) `tools.tool_search.enabled: off` in `~/.hermes/config.yaml`.** When the
deferrable surface is large, hermes-agent hides MCP tools behind a three-step
bridge — `tool_search` -> `tool_describe` -> `tool_call`. Every local model
tested failed to drive it, and *scaling up made it worse*:

| Model | Tool calls | Outcome |
|---|---|---|
| gemma4:e2b (7.2 GB) | 2 | got furthest — reached `tool_describe`, failed at `tool_call` |
| gemma4:e4b (9.6 GB) | 2 | listed tool categories, never called |
| hermes3:8b (4.7 GB) | 8 | looped; hallucinated "create github issue" |
| gemma4:12b (7.6 GB) | 16 | never escaped `tool_search`; gave up after 3 min |

With it off, all tools are presented eagerly and gemma4:e2b calls them
correctly. This costs ~14k tokens of schemas per turn, which is exactly why
(a) is a prerequisite — the two settings only work together.

Known upstream: [ollama#14958](https://github.com/ollama/ollama/issues/14958)
(tool calls silently drop with large system prompts),
[openclaw#4028](https://github.com/openclaw/openclaw/issues/4028) (`num_ctx`
not passed via the OpenAI-compatible API).

Frontier models drive the bridge fine; this section is about local ones.

### 3. Ollama chat-model gotcha (hermes3 only)

Not needed with the default `gemma4:e2b`, which accepts `think: true` and
returns a `thinking` field — verified against the live daemon. But if
hermes-agent itself chats through a **hermes3** model via Ollama, **disable
reasoning for that model** or every tool-calling turn fails with
`HTTP 400: "hermes3:8b" does not support thinking`. `attest install` applies
this automatically when `CHAT_MODEL` matches `hermes3*`, and skips it
otherwise. To persist it by hand:

```yaml
agent:
  reasoning_overrides:
    hermes3:8b: none
```

(Or pass `--reasoning none` per-invocation.)

### 4. Install the skill (optional but recommended)

```bash
cp -r src/attestation/skills/research-provenance ~/.hermes/skills/
```

Re-run that copy whenever the skill changes in this repo — the installed copy
does not track the checkout (`attest install` does this sync for you, see
"One-liner" above). The skill's `scripts/setup.sh` is a thin delegator to
`attest install --yes`: it resolves the checkout (or falls back to
`uvx --from git+<repo_url> attest install --yes` when no local checkout is
found) and lets the installer handle models, `.env`, first ingest, MCP/skill/
cron wiring. It reads `HERMES_RSS_PROJECT_DIR` (defaulting to the checkout the
script itself lives in) to find the checkout, and falls back to
`uvx --from git+<repo_url>` when a
`science_recommendations.repo_url` key is set in `~/.hermes/config.yaml`:

```yaml
science_recommendations:
  repo_url: https://github.com/<owner>/attestation
```

### 5. Shared database

When running alongside hermes-agent, the live database is co-located with
other skill state at `~/.hermes/skills/science-recommendations/data/hermes.db`
— **not** inside the checkout. Note the directory name: the *skill* was
renamed to `research-provenance`, but the *database* path deliberately kept
the old `science-recommendations` name, so as not to orphan every database
created before the rename. Every entry point (CLI, web server, MCP server)
resolves the DB the same way (`resolve_db_path()` in `src/attestation/db.py`):

1. explicit `--db <path>` flag
2. `RSS_DB` env var
3. `~/.hermes/skills/science-recommendations/data/hermes.db`, if it exists
4. `./hermes.db` (cwd-relative fallback for ad hoc/dev use)

So once the co-located DB exists, the web UI, the MCP tools, and cron ingest
all read and write the same feed and the same clicks with no flags needed.

### 6. Launch and verify

```bash
hermes            # start a hermes-agent chat session
```

Then try, in chat:

- *"What's in my science feed?"* → agent calls `feed.list`
- *"Mark the second one useful"* → `feed.rate`
- *"Why did that first paper rank so high?"* → `feed.explain` (slow first call)

The web UI can run at the same time (`uv run attest serve`) — clicks from
either side land in the same database. Concurrent use is safe: writers keep
transactions short and WAL mode + `busy_timeout` absorb overlap.

### 7. Keep the feed fresh (cron)

```
17 * * * * cd ~/attestation && uv run attest ingest && uv run attest tag
```

Concurrent ingest + serving is safe in practice: ingest keeps write
transactions short — entries are embedded first with no lock held, then
written in one quick transaction per feed.

## The experiment ledger

Research generates artifacts — config files, eval dumps, benchmark tables — and
the numbers that end up in a README get transcribed from them by hand, where
nothing checks them again. The ledger reads those artifacts so the numbers stay
derivable.

```bash
export RESEARCH_ROOT=~/projects
uv run attest runs scan                    # read runs from artifacts on disk
uv run attest runs list                    # what exists, and what groups into families
uv run attest runs compare <family>        # rank the arms of a sweep
uv run attest runs show <project> <name>   # one run, with its source path
```

Two conventions decide whether a project is read, and both are worth knowing
before the first scan:

```
~/projects/
  asr-ablation/            <- a project: any directory under RESEARCH_ROOT
    results/               <- results live IN a recognised directory, not beside the run
      asr_baseline.json    <- arms share a prefix; `asr` is the family
      asr_biglm.json          {"wer": 0.0433} , or a list of per-sample records
      asr_moredata.json       (a list also gives the comparison its sample size)
    configs/               <- optional: recorded as provenance, never as a metric
      asr_baseline.json
```

Recognised results directories are `results/`, `logs/`, `outputs/`, `metrics/`,
`eval/`, `evals/`, `benchmarks/` and `reports/`; recognised config directories
are `configs/`, `config/`, `conf/`, `experiments/` and `examples/`. A scan that
finds nothing says which of these it looked for and where your files actually
were, rather than reporting an empty success — and `runs compare` on something
that is not a family names the families that exist.

**This is deliberately not an experiment tracker.** MLflow, Sacred, W&B and DVC
all instrument runs at the moment they happen. That does not help a corpus of
runs which already finished, across many projects, in several languages, some
dormant for months. Adoption cost is the design constraint: a tool needing new
discipline gets used for a week, while one that reads what is already there
keeps working after you forget it exists.

That argument cuts the other way for trackers you already run, so a scan does
read existing `wandb/` and `mlruns/` trees (`TRACKER_DIRS` in
`src/attestation/ledger_adapters/generic.py`) — they are conventions in the
strict sense, since the tool picks the directory name, not you. Two honest
caveats: it records **final values, not curves**, because a ledger that
compares finished arms has no use for the whole series; and **neither reader
has been run against a real directory.** There was no `wandb/` or `mlruns/` on
the machine where they were written, so both are built from the published
layouts and tested against transcribed fixtures — plausible, not verified. If
you have a real one, point this at it.

It reads the conventions research repos already use — `results/`, `logs/`,
`configs/`, `outputs/`, `benchmarks/` holding JSON, JSONL, CSV, YAML or TOML —
and no project is registered in advance. On the author's machine this found
**849 runs across 16 projects** with zero instrumentation.

Two rules keep it honest:

- **Record what is unambiguous, refuse the rest.** A config file is a
  specification with no result attached, so it gets no metrics rather than an
  invented one. An unrecognised shape yields no run. A mapping of hundreds of
  numeric keys is a lookup table, not a metrics record — a real tokenizer
  `vocab.json` was briefly read as 50,258 metrics, one of them keyed `wer`.
- **Never rank a metric whose direction is undeclared.** `compare` raises on
  total energies, because "lower is better" is false across different systems.
  Guessing would order a sweep backwards with total confidence.

Comparisons carry provenance and caveats. Every arm shows its `source_path` and
sample size, and the result warns when all arms are small-n, when arms differ in
sample size, when the top two are within 5%, or when arms sit at different
training steps. A healthy comparison emits no caveats — a tool that always warns
trains you to ignore it.

## Verifiable claims

A README says "MAE 0.353 eV vs experiment". That number was transcribed by hand
and nothing checks it: re-run the benchmark and the document asserts 0.353
forever. A claim is an HTML comment beside the prose it describes, so the
document renders exactly as before:

```markdown
The cut leaves WER essentially unchanged (**0.053 vs 0.043** baseline).
<!-- claim: ablation/results/stack_4 metric=wer value=0.053 tol=0.001 -->
```

```bash
uv run attest claims ~/projects              # verify every claim
uv run attest claims ~/projects --coverage   # numbers no claim covers
```

Five verdicts, and the distinctions are the design. `supported`: a run agrees.
`contradicted`: a run disagrees — the document or the run is wrong.
`unsupported`: no run matches, so the claim may be true but nothing backs it.
`ambiguous`: a wildcard matched several runs, so which is meant is undecidable.
`stale`: the value matches but the artifact changed after `as_of`.

`unsupported` and `contradicted` never collapse together — one needs a run, the
other needs a correction. `ambiguous` exists because silently taking the first
of several matches is how a checker reports a confident wrong answer.
`attest claims` exits non-zero on a contradiction, so it can gate a commit.

`--coverage` is the inverse, and the more useful half for adoption: a document
with zero contradicted claims can still assert a dozen unverifiable numbers.
Only decimals count as measurements — on a real index, 212 numbers reduce to 30
decimals and the decimals are the results.

## Browsing the ledger

```bash
uv run attest browse            # read-only Datasette at :8898
uv run attest kg-report         # graph health + topic clusters
```

`browse` opens the database in [Datasette](https://datasette.io) with
`--immutable`, so a viewer can never write. The canned queries in
`datasette.yml` are the point: named SQL with shareable URLs, so a reviewer can
open the exact query behind a number, change one parameter, and watch the answer
move. `metric_over_time` is the time-series view; `unevaluated_configs` lists
sweeps that were specified but never run.

Datasette is a dev dependency and a separate process, never imported, so a
`uvx` install does not pay for it. It needs `--load-extension` for `sqlite-vec`
or it refuses to open the database at all — `attest browse` handles that.

## How ranking works

- 0 clicks: cosine similarity between item embeddings (embeddinggemma, 256-dim)
  and your `interests` profile text.
- With clicks, two click-driven terms join the blend (weight
  `w = n_clicks / (n_clicks + 5)`, averaged when both are active):
  - a per-user logistic regression over item embeddings (guard: this term
    only participates once your clicks contain both classes);
  - a feature-preference term — Laplace-smoothed like/dislike ratios per
    LLM-extracted topic tag, content type, and source feed (see
    `uv run attest tag`). This term works from click one, including for
    users who have only ever downvoted: two ✗ on items sharing a tag demote
    every item carrying that tag on the next render.
- Visible movement by click 3-4; tag-level demotion is visible immediately.

## Commands

    uv run attest ingest [--feeds feeds.toml] [--db hermes.db]
    uv run attest tag [--limit N]           # LLM-tag untagged items (topics + content type)
    uv run attest serve [--port 8899]
    uv run attest eval --user matt          # holdout AUC (noisy at small n)
    uv run attest kg-report [--min-size 3]  # graph health metrics + topic clusters
    uv run attest runs scan                 # read experiment runs from $RESEARCH_ROOT
    uv run attest runs compare <family>     # rank the arms of a sweep, with caveats
    uv run attest claims [path]             # verify Markdown claims against runs
    uv run attest browse                    # read-only Datasette UI over the ledger
    uv run attest bootstrap-persona bench-chemist   # optional persona pseudo-clicks
    uv run attest warmup
    uv run attest-mcp                       # MCP stdio server (normally launched by hermes-agent)

## Tests

    uv run pytest
    uv run ruff check .                     # lint (E, F, W, I, BLE; line length 100)
    uv run ty check                         # type check
    uv run radon cc -s -n C src/attestation      # complexity report (empty = nothing worse than B)

## Live smoke test notes

Verified against real feeds and a real local Ollama instance on 2026-08-04
(2x GTX 1080 8GB). `uv run attest ingest` against `feeds.toml`'s 7 feeds
succeeded with zero feed failures: `{'added': 889, 'skipped': 616,
'failed_feeds': 0}`.

The default chat model `gemma4:12b` partially CPU-offloads on this GPU class
and took ~94s for a single `/explanation` call — usable but slow. Setting
`CHAT_MODEL=hermes3:8b` brought that down to ~5.7s on a warm model;
this is the model used for the demo runbook (see `DEMO.md`). Both models
fit resident simultaneously with `OLLAMA_MAX_LOADED_MODELS=2`.

## Publishing the skill to a registry

These are account-mutating / network-mutating actions — run them yourself
when ready. Investigated on the installed `hermes-agent v0.20.0` via
`hermes skills publish --help` and reading
`~/.hermes/hermes-agent/hermes_cli/skills_hub.py` (read-only — no publish
was executed):

- `hermes skills publish <skill_path> --to github --repo <owner/repo>`
  requires either `GITHUB_TOKEN` set in `~/.hermes/.env`, or `gh auth
  login` having been run. It also runs a self-scan (`skills_guard.scan_skill`)
  and refuses to publish anything with a "dangerous" verdict.
- `--to clawhub` is **not yet supported** by this version — it just prints
  "Submit manually at https://clawhub.ai/submit".
- `SKILL.md` must have a non-empty `description` in its YAML frontmatter
  (ours does).

**Run these yourself, in order, once you're ready:**

```bash
# 1. Push this repo to a real remote (replace with your actual remote/URL).
git remote add origin <your-attestation-remote-url>
git push -u origin main   # or your default branch

# 2. Set science_recommendations.repo_url in ~/.hermes/config.yaml to that
#    remote's HTTPS URL so setup.sh's uvx fallback (see above) can find it.

# 3. Publish the skill directory to a GitHub-backed skill registry.
#    Requires GITHUB_TOKEN in ~/.hermes/.env, or `gh auth login` already done.
hermes skills publish src/attestation/skills/research-provenance --to github --repo <owner/skills-repo>
```

ClawHub has no CLI publish path yet in v0.20.0 — submit manually at
https://clawhub.ai/submit if you want it listed there.
