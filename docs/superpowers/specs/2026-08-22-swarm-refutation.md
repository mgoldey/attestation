# Swarm — refuted, and what replaced it

**Date:** 2026-08-22
**Status:** closed. This spec exists to close a question, not to open work.
**Roadmap:** replaces spec 4 of `2026-08-21-architecture-roadmap.md`
**Supersedes:** the swarm section of that roadmap

## Why this document exists instead of the spec that was planned

The roadmap's spec 4 proposed per-domain LLM agents (ingest, rerank, knowledge,
provenance, citations) coordinated by a user-interaction supervisor, defined in
a `swarm.toml`. It was the largest of the five and the one the whole effort
started from — the opening request of this project was for exactly that
hierarchy.

That roadmap also wrote its own exit condition:

> **Measure before building.** Spec 1 alone may substantially resolve the
> confusion this spec exists to fix. […] Re-check whether the confusion
> persists after spec 1 lands before committing to this one — it is the
> expensive spec and the one that spends determinism.

The measurement was run. The swarm lost.

Writing the spec anyway would specify a mechanism this repo has already
disproved on its own hardware. Writing nothing would leave a roadmap entry that
reads as pending work, and the question would be reopened by the next person —
or the next model — who reads it. So the spec slot is spent on the refutation.

## The measurement

Three architectures, the same 15 realistic user turns, the same model
(`gemma4:e2b-it-q4_K_M`), three runs each.

| architecture | correct | latency |
|---|---|---|
| **Routed** — 4 intent tools, deterministic dispatch | **13/15** | 1.3s |
| Flat — one namespace (37 tools at the time of measurement) | 8/15 | 1.3s |
| **Swarm** — supervisor LLM + namespace subagent LLM | **7.3/15** | **2.8s** |

**The swarm performed worse than doing nothing, at twice the latency.**

The mechanism is not subtle. A second model call is a second chance to be
wrong, and a namespace miss is unrecoverable: once the supervisor routes "which
arm of my sweep won?" to the knowledge agent, no amount of competence inside
that agent recovers the turn. Errors compound rather than cancel. Two 80%-
accurate stages in series are a 64%-accurate pipeline, and neither stage here
was at 80%.

This is the determinism objection from the roadmap, now with numbers instead of
an opinion.

## What was built instead

Two things, both shipped:

**Deterministic routers** (`src/attestation/mcp/ask.py`). Four intent tools —
`feed.ask`, `runs.ask`, `kg.ask`, `sym.ask` — that map a question to a tool by
rule table, with no model call and no database read. They return a Pydantic
`Answer`, so MCP emits a real `outputSchema`. This is the 13/15 arm.

**Per-domain agent surfaces** (`src/attestation/mcp/__init__.py`). The
`AGENT_SURFACES` table plus `ATTEST_TOOLS` restricts tool registration to one
namespace. Measured 2026-08-22 with `ATTEST_EXPAND=1`: `feed` 22, `symbolic` 9,
`provenance` 8, `knowledge` 8 (`kg`'s 7 plus `feed.search`), against 46
unrestricted. Without expansion each surface shows 2 — its `ask` router and one
companion — which is the progressive-disclosure default, not the surface size.
Four entries exist in `~/.hermes/config.yaml` today.

(Note that a surface count is not a namespace count: `kg.*` is 7 tools, but the
`knowledge` surface serves 8 because it also carries `feed.search`. `CLAUDE.md`
said 37 and 19/7/7/8 when this was written -- correct once, silently wrong
later, and quoted into a draft of this spec before anyone measured. It has since
been corrected to 46 and is now pinned by
`test_architecture.py::test_claude_md_tool_counts_match_the_live_surface`, which
asserts the per-namespace split as well as the total: a total can stay right
while two namespaces drift in opposite directions. Re-measure rather than
quoting either number.)

Note what the second one is: **the roadmap's deliverable, by a different
mechanism.** Spec 4 wanted per-domain agents. There are four per-domain agents.
What was rejected is not the decomposition — it is having a *model* choose
between them at runtime.

That distinction is the whole finding, and it generalises:

> Separate agents help when a **person** chooses which to talk to. They hurt
> when a **model** chooses at runtime.

So the split is by session, selected at launch, and enforced at tool
registration — where it costs one environment variable and cannot be gotten
wrong mid-conversation.

## Two findings worth keeping

Both moved routing from 9.7 to 13, and both are encoded as tests in
`tests/test_ask_routing.py` rather than left as prose.

**No catch-all destination.** An early routed version had a `doctor` tool for
"diagnose the system". It became a magnet: three of the four remaining misses
went to it. An ambiguous question must return options and ask back, never pick
a default. A catch-all does not absorb the hard cases, it *attracts* the
ordinary ones.

**Descriptions must contain the words users actually say.** "which topics are
most central or most read about" catches a turn that "what is central" does
not. This is unglamorous and it is worth more than architecture.

## What survives of the swarm idea

One narrow case, deliberately left unbuilt.

The measurement tested **routing** — one question, one tool. It did not test
**orchestration** — a multi-step task where an agent decides which deterministic
function runs next, and the sequence itself is the work. The roadmap's
determinism constraint was written for exactly that case:

> Agents may **orchestrate only** — choose which service calls to make and in
> what order. Every service call is pure Python. The supervisor writes the full
> call trace to the ledger, so any run stays reconstructable.

That constraint is sound and the refutation above does not touch it. But the
case is currently hypothetical: **there is no multi-step workflow in this repo
that a user performs today and that a planner would improve.** `runs.scan` then
`runs.compare` is two calls, and a rule can sequence two calls.

The condition for reopening this, stated so it is checkable rather than a
matter of taste:

> Reopen when a real workflow exists that (a) requires three or more calls
> whose order depends on intermediate results, and (b) a rule table cannot
> express. Then measure the planner against a rule table on that workflow
> before building it.

Absent that, a planner is a second model call looking for a job, and this
document is the record of what that cost last time.

## What is closed

- No `swarm.toml`.
- No LLM subagents for routing.
- No inter-agent messaging. The four surfaces never talk to each other; a
  message bus between them would reintroduce the exact compounding-error
  mechanism the swarm arm demonstrated.
- No supervisor process.

## Consequences for the roadmap

Roadmap spec 4 is closed by this document. Roadmap spec 5 (agent-config
emitters) declared a dependency on spec 4 and on the `swarm.toml` it would have
produced. That dependency is void; spec 5 is rescoped in
`2026-08-22-config-emitters-design.md` to generate from `AGENT_SURFACES`, which
exists, instead of from a file that will not.
