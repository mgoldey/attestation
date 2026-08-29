---
name: research-provenance
description: "Research provenance tools over a local SQLite ledger: verify claims written in Markdown against recorded experiment runs, compare the arms of a sweep, resolve citation keys against a local Zotero library and .bib files, query a knowledge graph of your reading, run symbolic derivations, and rank a personalized science/arXiv feed."
version: 1.0.0
author: attestation project
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [rss, recommendations, arxiv, research, feed, ranking, science, local-api]
    related_skills: []
---

# Research provenance

Use this skill when the user asks about their science/research feed, wants
today's recommended papers, or wants to give feedback on an item ("mark that
useful", "not interested in that one", "why did this rank first?").

**Also use it when they ask what you can do.** Answer by doing: call
`feed.ask(user, question="what should I read?")`, name two papers, and offer
the obvious next step. Do not recite the tool surface -- a list of
capabilities is a menu, and nobody asked for a menu. If no persona exists yet,
one is created on that first call, so there is nothing to set up first and
nothing to ask.

This skill talks to **attestation**, a local personalized RSS/arXiv ranking
engine running at `http://127.0.0.1:8899`. The engine is deterministic
(profile-embedding + click-trained classifier) with an LLM-generated
explanation layer that fills in lazily and never blocks the feed.

## When NOT to use this

- The user wants general web search or news outside the configured feeds —
  this skill only knows about items already ingested into `hermes.db`.
- The user wants to change what feeds are tracked, manage a persona, search
  the archive, check a manuscript, or resolve a citation key — use the MCP
  tools instead (no checkout needed; call them directly from the agent). This
  skill's HTTP endpoints below only cover the list/click/explain path;
  everything else is MCP-only, and the `.ask` routers are the entry point.
  See **MCP tools** below for the full set.

## Setup

Before the first call in a session, run the idempotent installer/doctor
(via this skill's wrapper script, which resolves the checkout for you):

```bash
bash ${HERMES_SKILL_DIR}/scripts/setup.sh
```

`setup.sh` is a thin delegator: it resolves a local checkout (or falls back
to `uvx --from git+https://github.com/mgoldey/attestation attest install --yes` — see below) and execs
`uv run attest install --yes`. You can also run the installer directly from
the project dir:

```bash
uv run attest install --check   # diagnose only, exit 1 on gaps, changes nothing
uv run attest install --yes     # non-interactive repair of any gaps found
```

`--check` prints one line per step (`ok` / `BROKEN` / `skipped`) and never
mutates anything — use it to see what's missing before repairing. `--yes`
fixes gaps without prompting (pulls missing Ollama models, creates `.env`
from `.env.sample`, runs initial ingest, wires the MCP server + skill copy +
reasoning override + refresh cron job). If `setup.sh` exits non-zero, read
its one-line reason and fix that before proceeding — do not retry blindly.

If the engine server needs to be started manually:

```bash
cd "${HERMES_RSS_PROJECT_DIR:-<checkout>}" && uv run attest serve &
```

This uses the default DB resolution (see below), which finds the
co-located database automatically — no `--db` flag needed.

### Data directory

The live database lives at
`~/.hermes/skills/research-provenance/data/hermes.db`, co-located with
other hermes-agent skill state rather than inside the project checkout. DB
path resolution order (see `resolve_db_path` in `src/attestation/db.py`):

1. explicit `--db <path>` flag
2. `ATTEST_DB` env var (`RSS_DB`, the pre-rename name, still works)
3. `~/.hermes/skills/research-provenance/data/hermes.db`, if that file
   already exists
4. `./hermes.db` (cwd-relative fallback for ad hoc/dev use)

### Running without a local checkout (uvx)

If `src/attestation/skills/research-provenance/scripts/setup.sh` doesn't find a local
project checkout at `HERMES_RSS_PROJECT_DIR` (default: the checkout the
script itself lives in), it runs the installer straight from git with no
clone step:

```bash
uvx --from git+https://github.com/mgoldey/attestation attest install --yes
```

A `science_recommendations.repo_url` key in `~/.hermes/config.yaml` overrides
that URL, for a fork:

```yaml
science_recommendations:
  repo_url: https://github.com/<you>/attestation
```

The package name (`attestation`) and its console-script name (`attest`)
differ — `uvx --from <package>` takes the *package*, and the trailing word is
the *executable*, so the invocation is `uvx --from git+https://github.com/mgoldey/attestation attest
...`, not `... attestation ...`. `uvx --from . attestation` (wrong) fails with
"An executable named `attestation` is not provided by package `attestation`".

### Configuration contract

Where each piece of attestation config lives, and who writes it:

| Setting | Store | Written by |
|---|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `CHAT_MODEL`, `EMBED_MODEL`, `EMBED_DIMS`, `ATTEST_DB` | `<checkout>/.env` (real env wins) | `attest install` step 3 / user edit |
| `mcp_servers.attestation` | `~/.hermes/config.yaml` | `hermes mcp add` (install step 6) |
| `agent.reasoning_overrides.<model>` | `~/.hermes/config.yaml` | `hermes config set` (install step 8) |
| live DB | `~/.hermes/skills/research-provenance/data/hermes.db` | engine (resolve_db_path default) |
| refresh schedule | `~/.hermes` cron store + `~/.hermes/scripts/attestation-refresh.sh` | install step 9 |
| `ATTEST_TOOLS`, `ATTEST_EXPAND`, `ATTEST_CITATION_WEB` | server environment (`<checkout>/.env` or the MCP entry's `env:`) | user edit |

### Agent surfaces (`ATTEST_TOOLS`)

A session sees one surface, not the whole toolset. `ATTEST_TOOLS` names it,
and it is read once when the server registers its tools:

| Surface | Sees | Why it is its own session |
|---|---|---|
| `feed` | `feed.*` | Conversational; a wrong guess costs a retry. |
| `provenance` | `runs.*`, `cite.check` | Verification: a wrong answer reaches a manuscript, and the caveats are the product. |
| `knowledge` | `kg.*`, `feed.search`, `cite.*` | Exploratory, read-only. |
| `symbolic` | `sym.*` | Sandboxed subprocess, touches no database. |

Unset serves all 46 tools (this count moves -- re-measure rather than
quoting it, per CLAUDE.md). An unknown value **raises** rather than falling
back — a restriction that quietly stopped restricting is the failure worth
preventing, so a typo takes the server down loudly.

Two boundaries look like duplication and are not. `feed.search` is in
`knowledge` because "how does X connect to Y, and what did I read about it"
is one question. `cite.check` is in `provenance` as well, because a session
that lints uncited claims but cannot see the runs the numeric claims check
against reports half a document's problems and looks complete.

**A person picks the surface at launch; you never pick one at runtime.** That
is measured, not stylistic: letting a model choose the namespace and then the
tool scored 7.3/15 against 13/15 for routing inside a fixed surface, at twice
the latency. If a question falls outside your surface, say which agent
answers it rather than trying.

`attest emit` generates the config for these — four `attestation-<surface>`
entries in `~/.hermes/config.yaml` (disabled by default) and matching
`.claude/agents/attestation-<surface>.md` files. It reports drift by default
and writes only with `--write`, and it never overwrites a file you edited.

## Setup notes

**Only applies to hermes3 models.** The default `gemma4:e2b` accepts
`think: true` and returns a `thinking` field, so nothing below is needed for
it; `attest install` applies the override only when `CHAT_MODEL` matches
`hermes3*`.

On hermes-agent v0.20.0 with `hermes3:8b` served via an Ollama custom
endpoint, **reasoning/thinking must be disabled for this model** or every
tool-calling turn fails outright with `HTTP 400: "hermes3:8b" does not
support thinking`. This is a real, reproduced failure, not a hypothetical:
hermes-agent's default `agent.reasoning_effort: medium` sends a
thinking/reasoning request parameter that Ollama's OpenAI-compatible
endpoint rejects for this model.

This is persisted in `~/.hermes/config.yaml` (not something you need to
pass at call time) via a per-model override that does not touch other
models/providers:

```yaml
agent:
  reasoning_overrides:
    hermes3:8b: none
```

If invoking `hermes` ad hoc against a config that hasn't been updated yet,
pass `--reasoning none` explicitly instead.

## Quick Reference

| Action | Call |
|---|---|
| List ranked feed for a user | `GET http://127.0.0.1:8899/list?user=<name>` → returns an HTML `<ol>` fragment; extract titles, URLs, source, and `data-item-id` per `<li>`. |
| Mark an item useful | `POST http://127.0.0.1:8899/clicks` with form fields `user=<name>`, `item_id=<id>`, `useful=1` |
| Mark an item not useful | `POST http://127.0.0.1:8899/clicks` with form fields `user=<name>`, `item_id=<id>`, `useful=0` |
| Get why an item ranked where it did | `GET http://127.0.0.1:8899/explanation?user=<name>&item_id=<id>` → plain text, one sentence. Can take several seconds (LLM call); it's fine to wait. |

**`useful` is an integer, not a boolean word.** Send `useful=1` or `useful=0`
as form data — sending `useful=true` or `useful=false` returns HTTP 422.

Known users in the demo database: `matt`, `bench-chemist`, `ml-engineer`. If
the user doesn't say which profile, default to `matt` and ask if unsure.

## MCP tools

When running alongside hermes-agent, the rest of the toolset is exposed as
native MCP tools (`src/attestation/mcp/`), not HTTP — call these
directly rather than reaching for `curl`.

**Ask the router first.** Four tools take a question in the reader's own
words and route it to the right specific tool by rule — `feed.ask(user,
question)`, `runs.ask(question, family, path)`, `kg.ask(question, source,
target)`, `sym.ask(expr, question)`. There is no extra model call: the routing
is a table of phrases in `mcp/ask.py`, so it costs nothing and a wrong route
is a bug someone can fix rather than a sampling accident.

Use them because they are measured better than choosing yourself. On
gemma4:e2b over 15 realistic turns, three runs, no variance: routing scored
13/15, picking from the flat tool surface scored 8/15. A second arm where one
model picked the namespace and another picked the tool scored 7.3/15 at twice
the latency — a second call is a second chance to be wrong.

```
feed.ask(user="matt", question="anything new on retrieval augmented generation?")
runs.ask(question="which arm won the kd sweep?", family="kdsweep")
kg.ask(question="how do protein folding and diffusion models connect?",
       source="protein-folding", target="diffusion-models")
sym.ask(expr="x**2 - 4", question="solve")
```

Every router returns the same shape: `answer` (relay it VERBATIM, do not
reformat), `refs` (item ids for your next call), `caveat` (ranking honesty and
comparison caveats, unabridged), `options`, and `tool_used`.

**A router never guesses.** When the rules do not claim a question
confidently it returns `ok=false` with a question in `answer` and the
alternatives in `options` — "Did you mean your current feed, or a search of
the whole archive?". Ask the reader that question; do not pick from `options`
yourself. There is deliberately no catch-all destination: an early `doctor`
tool absorbed three of the four remaining misses, so ambiguity now comes back
as a question instead of a default.

Fall through to a specific tool in two cases: the router declined and the
reader has now told you which, or the call needs an argument the question
does not carry (an `item_id` to rate, a URL to add, a path to check — the
router says so and names the tool).

**Your session may only see the routers.** Specific tools are hidden by
default, because a visible tool gets called: measured over 26 turns, a model
picked the `ask` router 1 time in 26 with the specifics listed beside it, and
26 in 26 with them absent. `<surface>.tools` explains this and says how to
reveal them (`ATTEST_EXPAND=1` on the server). If a tool named below is not
in your list, that is the reason — use the router.

**Personas — never make the reader do bookkeeping.**

You do not need a persona to exist before you use it. `feed.list`,
`feed.search`, `feed.digest` and `feed.read` create one on first sight and
answer the question in the same call. So:

- **Never ask "which persona?" or "what should I call your profile?"** The
  name is whatever you already have — the chat handle, the username, the name
  they gave you. A reader asked to invent a profile name has been handed
  admin work they did not come for.
- **Do not call `feed.persona_create` to get started.** It exists for
  deliberately building a *second* reader (a colleague's profile, a demo
  persona), not for the person in front of you. Reaching for it on an unknown
  name is what put a duplicate `Matthew Goldey` in this database days after
  that reader had been merged into `matt`.
- **Ask what they read about, once, after answering.** A new persona starts
  from the corpus's own common topics and says so. That is the moment to ask
  — and the only question worth asking, because the interests text IS the
  profile embedding. "What do you actually work on?" beats any question about
  names or profiles.
- Then `feed.persona_update(name, interests)` with what they said. Ranking
  re-steers immediately.

**Growing the feed is your job, not theirs.** A reader who says "I want more
on X" is asking you to widen their sources, not to hand them a URL. The path
is `feed.source_suggest(user)` -> `feed.source_preview(url)` -> confirm ->
`feed.source_add(url, title)`. Suggestions are scored against tags this
reader already liked and come from a curated list -- never web-searched,
never invented. Preview before subscribing: adding a feed sight-unseen is how
a bad one gets in, and items are permanent once ingested.

**Show, do not list.** When someone asks what you can do, do not recite the
tool surface. Run something: pull their feed, name two papers in it, and say
what you could do next with them. A list of capabilities is a menu; a real
answer is a demonstration. `feed.persona_status(user)` is the same move for
"how well is this trained?" — it reports click count and how much of the
order is behaviour-driven versus text-driven, which is an answer rather than
a claim.

**Search** — `feed.search(user, query, tag, content_type, limit)` searches
the *whole* archive (not just unread items) for a keyword, optionally
filtered by tag or content type, and flags items already rated. Use this
instead of the `/list` HTTP endpoint when the user wants to find something
specific rather than browse what's new.

**Digest** — `feed.digest(user, days, per_topic, limit)` is the weekly-review tool:
it returns the ranked unread feed already grouped by topic, so "what's worth
reading this week, and why?" is one call rather than manual assembly from
`feed.list` + `kg.communities` + `feed.explain`. Each item joins the cluster
its tags overlap most (ties break on label, so repeated calls agree); items
matching no cluster come back in `unclustered` rather than being dropped, and
that bucket being large is a real signal, not a bug. `per_topic` caps the items
shown per group while `n_total` reports how many the group actually had —
truncation is visible. It returns structure and never prose: no LLM runs
inside it, and a per-item `explanation` appears only when `feed.explain`
already cached one, so ask for explanations separately if they're missing.
**Read `ranking_quality` before trusting the order** — it reports whether the
click classifier is actually active, and with a single-class click history it
never fires, leaving the order as embedding similarity alone. `days` bounds
how far back the feed reaches (default 7, echoed back as `window_days`), so
widen it when a quiet week returns little.

**Feed management** — `feed.sources()` shows subscribed feeds with item counts
and last-fetch times; `feed.source_preview(url, limit)` shows a candidate feed's
recent entries without subscribing (use before `feed.source_add`); `feed.source_suggest(user, limit)`
recommends feeds from a curated candidate list, scored against tags the user
has marked useful.

**Destructive actions require `confirm=true`**: `feed.persona_delete(name, confirm)`
(irreversibly removes a persona and its feedback), `feed.persona_reset(name, confirm)`
(clears a persona's clicks but keeps the persona), and `feed.source_remove(feed_id, confirm)`
(unsubscribes but keeps existing items and feedback on them). Calling any of
these without `confirm=true` is safe — it returns a refusal message instead
of mutating anything, so use that as a dry-run to see what would happen.

`feed.source_add(url, title)` is **register-only**: it validates the URL parses as
a feed and subscribes, but does not fetch anything. New items only appear
after the next ingest (hourly cron, or `attest ingest`) — do not expect
`feed.list` to show items from a feed added moments ago.

**Knowledge graph** — derived from the tagging pass (concepts are tags used
at least twice, linked when they co-occur on at least two items):
`kg.neighbors(node, limit)` finds the concepts directly adjacent to one you
give it ("what else should I read about this"), strongest co-occurrence
first. It returns direct neighbours only — for anything spanning more than
one hop, use `kg.path(source, target)`, which finds the shortest chain of
concepts linking two topics, returning `ok=false` with `path=null` when they
never co-occur — a real answer, not an error. That answer is only ever given
about two concepts that really are in the graph: a name that is not a concept
is refused separately and says so, so a typo can never come back as "these
topics never co-occur". `kg.concepts(prefix, limit)` lists the valid names
(`prefix` is a case-insensitive substring match, and `n_concepts` reports how
many matched so a capped list is never mistaken for the whole vocabulary) —
call it when you are not certain a name exists;
`kg.central(metric, limit)` surfaces the most-connected (`metric="degree"`)
or most-bridging (`metric="betweenness"`) concepts; `kg.communities(min_size)`
clusters the graph into topic groups by modularity, each labelled by its hub
member (a dense hub cannot swallow the graph — concepts join a group only
when their links there beat chance, so even a tightly interconnected corpus
splits into real topics). Groups overlap in subject matter and each concept
belongs to exactly one, so a bridging concept lands where its links are
strongest.
Every `kg_*` read tool derives the graph fresh from `item_tags` on each call,
so there is nothing to rebuild and no staleness to report. A `kg_rebuild` tool
and a `stale` flag existed until 2026-08-21; the tables they maintained were
never read by anything, and all eight kg answers were byte-identical with and
without them, so both were deleted.

Concepts come from the tagging pass (`attest tag`). On an untagged database
every `kg_*` tool returns an empty graph, which is a setup gap rather than a
finding about the reading.

**Symbolic math** — `sym.simplify(expr, timeout)` simplifies an expression
to canonical form; `sym.solve(expr, symbol, timeout)` solves expr = 0 for a
given symbol (or auto-detects if the expression has exactly one); `sym.differentiate(expr, symbol, order, timeout)` and `sym.integrate(expr, symbol, bounds, timeout)` compute derivatives and integrals; `sym.derivation(expr, operation, symbol, timeout)` returns a step-by-step trace (genuine rule-by-rule tracing exists only for integrals; the differentiate branch returns the result with a note saying so); `sym.verify(lhs, rhs, timeout)` tests symbolic equality and returns `equal`, `unequal`, or `unproven` — **"unproven" is NOT a disproof**, since `simplify` is incomplete and can only mean "could not decide"; and `sym.evaluate(expr, subs, units, timeout)` computes a numeric value, optionally with variable substitutions and unit conversion (e.g. `units="meter/second -> kilometer/hour"`). `numeric` is `null` whenever any symbol is still free, and the message names which — an unsubstituted expression has no value to report, and reporting one anyway is how a wrong number reaches a paper.

**Experiment ledger** — records of the user's *own* runs, read from artifacts
already on disk (`results/`, `logs/`, `configs/`, `outputs/`, `benchmarks/`
holding JSON, JSONL, CSV, YAML or TOML). Nothing is instrumented and no project
is registered in advance: `runs.scan(root, project, confirm)` walks a workspace
(defaulting to `$RESEARCH_ROOT`), treats each subdirectory as a project, and
needs `confirm=true` since it replaces each scanned project's rows.
Directories with nothing recognisable are reported in `empty` rather than
omitted — "found nothing" must never look like "nothing was there".

`runs.scan` also reads experiment-tracker trees: `wandb/` (each
`run-<timestamp>-<id>/files/wandb-summary.json`) and `mlruns/` (each
`<experiment_id>/<run_id>/metrics/`). A run already found in `results/` is not
recorded twice. Two caveats, and say them rather than implying more coverage
than there is:

- **Neither reader has been run against a real directory.** Both are built
  from the tools' published layouts and tested against transcribed fixtures,
  so they are plausible, not verified. If a scan of a real `wandb/` or
  `mlruns/` returns nothing or looks wrong, that is the likely reason — report
  it as such rather than telling the reader their runs are not there.
- **They read final metric values, not curves.** MLflow logs one line per
  step; only the last is recorded, because one row per step would put 200 rows
  per metric into a 200-epoch run's ledger. A diverged run whose last value is
  `nan` records nothing rather than a mid-training number. A reader who wants
  training curves is not served by this ledger — say so.

`runs.list(project, family, limit)` shows what exists plus the *families* runs
group into; a family is the arms of a sweep or one run's checkpoints over
training. `runs.compare(family, metric)` ranks those arms — the question a
sweep exists to answer, which usually lives only in filenames. It **refuses to
rank a metric whose direction is undeclared** rather than guessing: ranking WER
as if higher were better would name the worst arm the winner. It also returns
`caveats` — small samples, arms evaluated on different sample sizes, a top two
within 5%, arms at different training steps — and every row carries its
`source_path` and `n`. A comparison with no caveats has earned that silence;
do not omit them when reporting. `runs.detail(project, name)` gives one run in
full, including any prose header comment from its config, which is often where
the hypothesis and the single changed variable are written down.

**Claim checking** — `runs.claims_check(path, verdict)` verifies numeric claims
written in Markdown against those runs. A claim is an HTML comment beside the
prose it describes, so it renders as nothing:

    <!-- claim: project/run metric=wer value=0.053 tol=0.001 as_of=2026-05-28 -->

Five verdicts, and the distinctions are the point. `supported`: a run agrees
within tolerance. `contradicted`: a run disagrees — the document or the run is
wrong. `unsupported`: no run matches, so the claim may still be true but
nothing backs it. `ambiguous`: a wildcard matched several runs, so which is
meant is undecidable. `stale`: the value matches but the artifact changed after
`as_of`. Never report `unsupported` as if it meant false — one needs a run, the
other needs a correction.

`runs.claims_coverage(path)` is the inverse: numbers asserted in prose that **no**
claim covers. A document with zero contradicted claims can still assert a dozen
unverifiable numbers, and this is what surfaces the difference. Only decimals
count as measurements; versions, dates, URLs, package pins and anything inside
an HTML comment are excluded.

Both are read-only. They report; they never edit a document.

**Citations** — references read from disk: a Zotero library at
`~/Zotero/zotero.sqlite` and any `.bib` files in the working directory.
`cite.lookup(key)` returns one record by citation key, DOI or arXiv id, and
every record carries `source` (which reader answered) and `fetched_at`.
`cite.search(query, limit)` finds references by words from a title or an
author's name.

`cite.check(path)` is the citation half of checking a manuscript. A claim
annotation may name a reference:

    <!-- claim: project/run metric=wer value=0.053 cite=vaswani2017 -->

and `cite.check` reports the keys **no configured source can resolve**. It is
a lint: it says the key is unknown here, never that the cited work fails to
support the claim — that needs a model and is not what this does. Claims with
no `cite` are skipped rather than flagged.

**A reader checking a manuscript wants both linters.** `cite.check` lints
citations, `runs.claims_check` lints numbers, and a document can pass either
one while failing the other. Run both and report them together:

```
runs.claims_check(path="paper.md")   # do the numbers match the runs?
runs.claims_coverage(path="paper.md") # which numbers nothing covers
cite.check(path="paper.md")           # which citation keys resolve to nothing
```

**Everything above is local.** The one exception is `ATTEST_CITATION_WEB`,
which is off by default and adds a CrossRef reader when set. It is read when
the resolver is *built*, so a disabled reader cannot be coaxed into one
request by an unusual code path, and `cite.search` never touches the network
even when it is enabled. `cite.sources()` answers "did anything leave my
machine" from the same surface that would have done the leaving: it lists
each configured reader with a `network` flag, and `offline: true` means every
one of them is on disk. Call it before telling a reader their bibliography
stayed local, rather than asserting it.

## Working patterns

Sequences that work, and the mistakes that look reasonable but do not. Every
one below was run against a live database before being written down.

### "What should I read?"

```
feed.ask(user="matt", question="what should I read today?")   # routes to feed.list
```

`feed.list(user="matt")` is the same answer if your session has the specific
tools; the router is the reliable path when the question is phrased in the
reader's own words rather than yours.

**Present each item as one line: a linked title, then source and topic.**

```
1. [LogicIF: Towards Complex Logic Instruction Following](https://arxiv.org/abs/2508.09125)
   arXiv cs.LG · language-models, reasoning
```

**Markdown link syntax is correct on every surface this agent ships to.**
Telegram is the measured one: the gateway runs a markdown->MarkdownV2
converter before sending (`plugins/platforms/telegram/adapter.py`,
`format_message`), and `[title](url)` comes through as a real link -- five of
five in a five-item list. Do NOT hand-write another surface's syntax to
"help": Slack's `<url|title>` contains angle brackets, which trips that same
sender's HTML auto-detect (`re.search(r'<[a-zA-Z/][^>]*>', message)`), so the
whole message is posted as `parse_mode=HTML` where `<url|title>` is not a tag.
Emitting Markdown and letting the gateway convert is the working path.

**List every item the tool returned.** "What should I read first?" is answered
by the ORDER, not by truncating to one. Measured on gemma4:e2b against a
five-item payload: with the presentation rule alone it rendered one item and
dropped four; told to list them all, five of five.

Nothing else. Do not restate `item_id`, `content_type` or `n_tags` in prose --
they are there for your next tool call, not for the reader. Do not reproduce
the JSON.

A reader on Telegram asked for their feed and replied "you didn't give links",
then "the links weren't clickable". Neither was a rendering bug: the model had
emitted no links at all, printing `ID: 2385` instead -- an argument for the
next tool call, useless to a human. The urls were in the payload the whole
time (5499 of 5499 items carry one).

**List every item the tool returned.** "What should I read first?" is answered
by the ORDER, not by truncating to one. Measured on gemma4:e2b against a
five-item payload: with the presentation rule alone it rendered one item and
dropped four; told to list them all, five of five.

**If a response is too long to render, say so in one sentence and show
fewer.** Do not apologise, do not re-render the same payload in another
format, and do not fall back to dumping raw JSON. A watched failure ran
exactly that loop -- bullet list, apology, JSON, apology -- and the reader
saw half of one item. `feed.list` defaults to 5 for this reason; ask for more
only when the reader does.

**Read `ranking_quality` before you present the order.** It is on every tool
that returns a ranking, and it is the difference between "here is what the
system learned about you" and "here is cosine similarity in a trenchcoat":

```
"ranking_quality": {"clicks": 67, "classifier_active": true, ...}
```

`classifier_active: false` means the click classifier never fired -- the
reader has feedback of only one kind -- and the `caveat` field says which
terms are actually contributing. Say so rather than implying the ranking is
personalised. A caveat is absent only when it has been earned.

### "Find me papers about X"

```
feed.ask(user="matt", question="anything on protein folding?")   # routes to feed.search
```

The router strips the asking from the question and searches for the topic, so
"is there anything new on retrieval augmented generation?" searches for the
topic and not the sentence.

If you are filtering by a tag rather than describing a topic, do NOT guess the
tag. Ask what vocabulary exists first:

```
kg.concepts(prefix="protein")     -> ["protein", "protein-engineering", ...]
feed.search(user="matt", query="", tag="protein-folding")
```

`feed.search` takes a semantic query, a tag filter, or both. An empty
`query` with a `tag` is a filter rather than a search, which is the right
call when the reader named a topic rather than described one. The search is
semantic: "LLM" finds papers titled "Large Language Models" with the acronym
nowhere in the text, so do not fall back to keyword guessing when a query
returns little.

Every result carries `match` (`semantic`, `literal`, `both`) and a
`relevance` score. `both` is the strongest signal. A short result list is
usually the relevance floor doing its job, not a failure.

### "Which arm won?"

```
runs.ask(question="which arm won?")                  # names the comparable families
runs.ask(question="which arm won?", family="kdsweep")
```

`runs.ask` without a `family` does not guess one: it lists what is comparable
and asks. Pass the name back and it compares. The specific sequence is the
same three calls:

```
runs.scan(confirm=true)        # only if runs.list says the ledger is empty
runs.list()                    # returns `families`, which is what compare takes
runs.compare(family="kdsweep", metric="wer")
```

**A family is a shared filename prefix, not a project.** `runs.compare` with
a project name is the commonest mistake, and the error says so and lists what
is comparable -- read it rather than guessing again.

**The ledger also reads W&B, MLflow, Sacred, DVC and Hydra directories, not
just `results/` JSON and CSV.** A project's sweep may live in `wandb/`,
`mlruns/`, `sacred_runs/`, a `dvc.yaml`-declared `metrics:` file, or a Hydra
`multirun/` tree, and `runs.scan` reads all of them the same way. `runs.compare`
carries a per-adapter caveat for whichever of these produced the arms being
ranked -- always "final values, not curves" in substance, though the wording
differs per tracker -- and relaying it is not optional.

**Report every caveat, verbatim.** They are the point of the tool:

```
winner: kdsweep_t4
caveat: the top two arms differ by 0.0017 (2.6%) -- too close to call
caveat: each arm is a single run; no seed replication ...
```

A comparison whose margin is smaller than its seed variance has not found
anything, and presenting the winner without the caveat misrepresents it.
Also: `runs.compare` refuses outright when a metric's direction is
undeclared, because guessing ranks an ablation backwards. That refusal is
the tool working.

### "Is this number in my draft right?"

```
runs.ask(question="are the numbers in my draft right?", path="paper.md")
```

All three linters when they are asking about the whole manuscript, because
each catches what the others cannot:

```
runs.claims_check(path="paper.md")
runs.claims_coverage(path="paper.md")
cite.check(path="paper.md")
```

Four verdicts, and they mean different things. `contradicted` is the
document disagreeing with the artifact. `unsupported` means no run backs it
-- possibly true, but nothing here says so. `stale` means the value matches
a run whose artifact changed afterwards. `malformed` is a broken annotation,
reported rather than skipped so a claim cannot vanish from review silently.

`claims_coverage` is the inverse and is the one people forget: numbers
asserted in prose that no claim annotation covers at all.

### Feedback: the part that gets skipped

`feed.rate(user, item_id, useful)` after the reader expresses an opinion --
**including the opinions they never label as feedback**. See the Procedure
section for the phrasings that carry a verdict.

Two tools exist because real feedback is scarce:

- `feed.harvest_engagement(user)` -- free, instant, no model. Turns past
  "why is this here?" questions into weak positives. Run it once for a
  reader who has asked for explanations; it is idempotent.
- `feed.simulate_ratings(user, confirm=true)` -- slow, one LLM call per item.
  Generates BOTH classes so the classifier can fire at all. Check the
  `caveat` it returns: if most of a persona's positives come from one feed,
  the classifier can separate them by publication rather than by topic and
  any evaluation score is meaningless.

### Mistakes that look reasonable

| Instead of | Do |
|---|---|
| Picking a specific tool from the whole surface | Ask the router: 13/15 against 8/15 |
| Picking one of the `options` a router returned | Ask the reader which; the router declined on purpose |
| Guessing a tag name | `kg.concepts(prefix=...)` first |
| `runs.compare(family="<project>")` | `runs.list()` and use a name from `families` |
| Presenting a winner alone | Include every caveat verbatim |
| Presenting a ranking alone | Check `ranking_quality.caveat` |
| Only recording what the reader liked | Record the negatives; the classifier needs both |
| Retrying a failed call with different arguments | Read the message -- it names the fix |
| Treating an empty search as broken | The relevance floor cuts weak matches on purpose |
| Checking a manuscript with `runs.claims_check` alone | Add `cite.check`; they lint different things |
| Reading `cite.check` as "the paper does not support this" | It says the key does not resolve, nothing more |
| Concluding a missing tool is broken | `attest reload` -- your server may be serving stale code |


## Procedure

1. **Fetch the feed**: `curl -s --max-time 10 "http://127.0.0.1:8899/list?user=<name>"`.
   The response is an HTML fragment, not JSON — parse out each `<li>`'s title
   (inside the `<a>` tag), URL (`href`), source, and `data-item-id`. Present
   the user a clean summarized list (title + source), not raw HTML. Items are
   already ranked best-first; the top ~10-15 are usually what's worth
   surfacing conversationally.

2. **Act on feedback — including feedback the user never labels as such.**
   Find the `item_id` from the most recently fetched list (match by title, or
   by position — "the second one" — against the order you just presented) and
   `POST /clicks` with `useful=1` or `useful=0`. The response is the re-ranked
   feed fragment; the engine retrains on every click.

   **Record the negatives.** This is the part that gets skipped, and skipping
   it is why the ranker cannot learn. The click classifier needs BOTH classes
   to fire at all: a reader whose history is all positive gets ranked by
   embedding similarity alone, forever, no matter how many items they approve.
   In this project's own database, real users had recorded 70 clicks across
   5,167 items and every one was positive.

   Ordinary conversation is full of verdicts that are never phrased as
   feedback. Treat these as `useful=0`:

   - "not really what I'm after" / "that's not my area"
   - "I've already read that one" / "old news"
   - "too applied" / "too theoretical" / "wrong subfield"
   - asking for something *instead of* an item — "anything on X rather than
     these?" rejects what was shown
   - skipping past items to ask about one further down the list

   And these as `useful=1`:

   - "that looks interesting", "send me that one", "good find"
   - asking a follow-up question **about the item's content** (not about why
     it ranked — see step 3, which is recorded separately as weak engagement)

   When you are unsure whether a remark is a verdict, ask — one short
   question is cheaper than a wrong label. But do not wait to be told
   explicitly: a user who never presses a button still has opinions, and an
   unrecorded opinion trains nothing.

3. **Explain a ranking** (optional, only if asked "why is this here" / "why
   did this rank first"): `GET /explanation?user=<name>&item_id=<id>`. This
   is an LLM call through a local Ollama model and can take a few seconds
   to tens of seconds depending on which chat model is loaded — don't treat
   a multi-second wait as a failure.

## Notes

- All calls are local HTTP — no auth, no API key.
- The feed and click endpoints are fast (SQLite-backed); only `/explanation`
  invokes an LLM and can be slow. Never let a slow `/explanation` call block
  presenting the feed or confirming a click — those two are independent of it.
- If any call fails to connect, the engine server is probably not running:
  rerun `scripts/setup.sh` or `cd "${HERMES_RSS_PROJECT_DIR:-<checkout>}" && uv run attest serve &`.

### When the tools change under you

**An MCP server never reloads.** One is spawned per session and holds that
code until it dies, so edits to the checkout are invisible to a session that
started before them. Both live servers here were once found running code five
commits stale — the Hermes gateway and a Claude Code session, against commits
made hours later.

```bash
uv run attest reload   # SIGTERMs every running attest-mcp
```

Respawn is **lazy**: measured against a live gateway, nothing restarted for at
least ten seconds and the new process appeared only when a tool was next
called. So the first call after a reload is slower, and a reload followed by
no call leaves the server down. That is fine; do not read it as a failure.

`hermes mcp test` does not catch staleness — it spawns a fresh process, so it
reports the code on disk rather than what your session is serving. If a tool
documented here is missing, or behaves as an older version of itself, reload
before concluding it is broken.
