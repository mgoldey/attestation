# Use it from an agent

How does an agent use this? Through an MCP server exposing every tool as a
native, typed call — registered once with `hermes mcp add`, or generated for
Claude Code with `attest emit` — plus an optional skill for setup and
fallback judgment.

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
  `uvx` installs get the skill without a checkout. See
  `src/attestation/skills/research-provenance/SKILL.md` itself for what it
  tells the agent to do.

See `docs/hermes-agent-plugin-research.md` for why MCP was chosen over
hermes-agent's plugin system.

`attest install` (see the [install guide](install.md)) automates steps 1, 3,
and 6 below when it finds a local hermes-agent binary — MCP registration,
the skill copy, and the refresh cron job. The steps are kept here as
reference and for anyone who wants to wire a piece up by hand.

## 1. Register the MCP server (one-time)

From the checkout root (the `--project` path is what gets recorded):

```bash
hermes mcp add attestation \
  --command uv \
  --args run --project "$PWD" attest-mcp
```

This writes an `mcp_servers.attestation` block into `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  attestation:
    command: uv
    args: [run, --project, /path/to/attestation, attest-mcp]
    enabled: true
```

Verify with `hermes mcp list` — you should see `attestation ... ✓ enabled`.

The server (`attest-mcp`, from `src/attestation/mcp_server.py`) exposes 46 tools.
These counts move: re-measure rather than quoting this paragraph.

```bash
uv run python -c "from mcp.server.fastmcp import FastMCP; \
from attestation.mcp import register_all; import asyncio; \
m=FastMCP('x'); register_all(m); print(len(asyncio.run(m.list_tools())))"
```


| Tool | What it does | Speed |
|---|---|---|
| `feed.personas()` | List reader personas + interest profiles | instant |
| `feed.list(user, limit)` | Ranked unread items, best first (capped at 13) | fast |
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
| `kg.concepts(prefix, limit)` | List the concept names the other `kg.*` tools accept | instant |
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

## 2. Required for local models: context window + eager tools

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

## 3. Ollama chat-model gotcha (hermes3 only)

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

## 4. Install the skill (optional but recommended)

```bash
cp -r src/attestation/skills/research-provenance ~/.hermes/skills/
```

Re-run that copy whenever the skill changes in this repo — the installed copy
does not track the checkout (`attest install` does this sync for you, see
the [install guide](install.md)). The skill's `scripts/setup.sh` is a thin
delegator to `attest install --yes`: it resolves the checkout (or falls back
to `uvx --from git+https://github.com/mgoldey/attestation attest install --yes` when no local checkout is
found) and lets the installer handle models, `.env`, first ingest, MCP/skill/
cron wiring. It reads `HERMES_RSS_PROJECT_DIR` (defaulting to the checkout the
script itself lives in) to find the checkout, and otherwise falls back to
`uvx --from git+https://github.com/mgoldey/attestation`. A
`science_recommendations.repo_url` key in `~/.hermes/config.yaml` overrides
that URL, for a fork:

```yaml
science_recommendations:
  repo_url: https://github.com/<you>/attestation
```

## 5. Shared database

When running alongside hermes-agent, the live database is co-located with
other skill state at `~/.hermes/skills/science-recommendations/data/hermes.db`
— **not** inside the checkout. Note the directory name: the *skill* was
renamed to `research-provenance`, but the *database* path deliberately kept
the old `science-recommendations` name, so as not to orphan every database
created before the rename. Every entry point (CLI, web server, MCP server)
resolves the DB the same way (`resolve_db_path()` in `src/attestation/db.py`):

1. explicit `--db <path>` flag
2. `ATTEST_DB` env var (`RSS_DB`, its pre-rename name, is still honoured)
3. `~/.hermes/skills/science-recommendations/data/hermes.db`, if it exists
4. `./hermes.db` (cwd-relative fallback for ad hoc/dev use)

So once the co-located DB exists, the web UI, the MCP tools, and cron ingest
all read and write the same feed and the same clicks with no flags needed.

## 6. Launch and verify

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

## 7. Keep the feed fresh (cron)

```
17 * * * * cd ~/attestation && uv run attest ingest && uv run attest tag
```

Concurrent ingest + serving is safe in practice: ingest keeps write
transactions short — entries are embedded first with no lock held, then
written in one quick transaction per feed.

## Restricted surfaces and generated agent configs

A model that can see a tool will eventually call it wrong, so a Claude Code
session does not need all 46: `ATTEST_TOOLS=feed|provenance|knowledge|symbolic`
restricts what an `attest-mcp` process registers to one namespace (`feed.*`,
`runs.*` + `cite.check`, `kg.*` + `feed.search`, or `sym.*`), and
`ATTEST_EXPAND=1` reveals the specific tools underneath a surface's `.ask`
router and one companion — without it, each restricted surface serves just
those two, which is progressive disclosure, not a smaller surface. A typo in
`ATTEST_TOOLS` raises at server startup rather than silently serving
everything.

`attest emit` generates the `.claude/agents/attestation-*.md` file for each
of the four surfaces from this same `AGENT_SURFACES` mapping, and reports a
diff — `missing`, `stale`, or `orphaned` — against what's on disk without
touching it; pass `--write` to create what's missing (it refuses to
overwrite a file that already differs from the generated body, since the
realistic case is a hand-edited file, not a fresh run). See
`docs/superpowers/specs/2026-08-22-agent-surfaces-design.md` for why the
four-surface split beat both a flat 37-tool list and an LLM swarm on
measurement, and `examples/agents/` for a worked run: the install doctor,
`attest emit`'s configs, and one surface driven over stdio end to end.
