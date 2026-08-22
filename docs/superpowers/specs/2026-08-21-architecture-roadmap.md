# Architecture roadmap — five specs

**Date:** 2026-08-21
**Status:** proposed
**Kind:** roadmap. Names the specs, their interfaces, and their order. Each
numbered section below becomes its own design doc; none is designed here.

## Problem

Agents get confused by this repo in three distinct ways, and the three have
been wearing one costume.

**Tool-choice confusion.** `mcp_server.py` exposes 36 tools in one flat
namespace. A calling agent sees `runs_compare`, `kg_path`, `sym_integrate` and
`digest` as peers with equal claim on any question. Nothing in the surface says
`runs_scan` precedes `runs_compare`, or that `sym_*` has nothing to do with the
feed. The agent is choosing from 36 undifferentiated options every turn.

**Coding-agent confusion.** `mcp_server.py` is 1454 lines: 36 tools, 36 `_impl`
functions, 18 raw SQL strings of its own, and 11 deferred
`from attestation import ...` statements inside function bodies. Deferred
imports are the module confessing to import cycles it is routing around. An
agent asked to change one tool must hold the whole file to find it.

**No data layer.** Eleven modules speak `sqlite3` directly; 94 raw SQL strings
are spread across the tree. `db.py` is connection management and migrations,
not a repository. That `mcp_server.py` — the outermost layer — writes its own
SQL is the clearest statement of the problem: presentation reaches to disk.

Secondary tangles found in the same survey, recorded so they are not
rediscovered later:

- `ledger.py` (637) + `ledger_adapters/generic.py` (554): scan, artifact-shape
  detection, parse, compare and caveat logic in two files.
- `install.py` (614): installer, doctor, cron scheduler, Ollama puller, MCP
  wiring and skill copier in one module.
- `cli.py`, `server.py`, `mcp_server.py` are three parallel presentations that
  each re-derive their calls. `server.py` imports `rank_items` directly;
  `cli.py` goes over HTTP to the same logic. No shared service layer, so the
  three drift.
- `rank.py` exports `_PROFILE_VEC_CACHE` and `_db_identity` to `mcp_server.py`.
  Private names crossing a module boundary is a missing interface.

## The shape being moved toward

```
┌─ presentation ────────────────────────────────┐
│  cli.py    server.py    mcp/<domain>.py       │  thin adapters, no SQL
├─ services ────────────────────────────────────┤
│  feed  knowledge  provenance  symbolic  cite  │  one facade per domain
├─ domain ──────────────────────────────────────┤
│  rank kg ledger claims corpus features embed  │  pure logic, no sqlite3
├─ ports ───────────────────────────────────────┤
│  repository protocols · chat/embed ports      │  interfaces only
└─ infrastructure ──────────────────────────────┘
   sqlite repos · Ollama clients · feedparser · bib/zotero readers
```

Dependencies point inward only. Domain logic names a protocol, never a
`sqlite3.Connection`.

## Sequence and rationale

Order is set by dependency, not by appetite. Specs 2–5 all call into the
service facades that spec 1 creates; building any of them first means building
it against the current tangle and redoing the work.

| # | Spec | Depends on | Size |
|---|------|-----------|------|
| 1 | Onion refactor | — | Large, stageable |
| 2 | Citations domain | 1 | Medium |
| 3 | Experiment-tracker adapters | — (adapter seam exists) | Small |
| 4 | Swarm + `swarm.toml` | 1 | Large |
| 5 | Agent-config emitters | 4 | Small |

Spec 3 is independent and may be pulled forward at any time; it is listed third
because it is small, not because it blocks anything.

---

## 1. Onion refactor

**Fixes:** tool-choice confusion, coding-agent confusion, no data layer.

**Scope.** Introduce ports as Protocol classes; implement SQLite repositories
behind them; move every raw SQL string out of domain and presentation modules
into infrastructure; add one service facade per domain; split `mcp_server.py`
into `mcp/feed.py`, `mcp/knowledge.py`, `mcp/provenance.py`, `mcp/symbolic.py`
with a thin `mcp/__init__.py` registering all four; namespace the tools as
`feed.*`, `kg.*`, `runs.*`, `sym.*`.

**Why namespacing addresses tool-choice confusion.** The same 36 tools, grouped
into 4 domains of 6–10, turns one 36-way choice into a 4-way choice followed by
a smaller one. The hierarchy the swarm was reached for is obtained here without
spawning a process.

**Staging.** Each stage lands green and independently:
1. Ports + repository protocols, no callers changed.
2. SQLite repositories implementing them; `db.py` keeps migrations only.
3. Domain modules migrated off `sqlite3` one at a time.
4. Service facades; `cli.py` and `server.py` retargeted onto them.
5. `mcp_server.py` split; tools namespaced. Old names alias for one release.

**Open questions for the spec.** Whether `resolve_db_path` precedence moves
into infrastructure or stays a module function; whether repositories are
per-table or per-aggregate; how `_PROFILE_VEC_CACHE` is expressed once `rank.py`
no longer owns a connection.

**Success criteria.** `grep -rn sqlite3 src/attestation --include=*.py` matches
only infrastructure. No module over ~400 lines. No deferred imports inside
function bodies. Full pre-commit gate green throughout.

## 2. Citations domain

**Fixes:** nothing broken; adds a domain. Doubles as the proof that spec 1's
boundaries hold — if adding a domain is hard, spec 1 was wrong.

**Scope.** Four readers behind one `CitationPort`:
- **Zotero** — local SQLite / local HTTP at `127.0.0.1:23119`. Offline. The
  natural pair for the reading knowledge graph.
- **BibTeX / BibLaTeX** — parse and write `.bib` on disk. Offline. Pairs with
  `claims.py`: a claim can cite a bib key.
- **arXiv / CrossRef / Semantic Scholar** — metadata lookup by DOI or arXiv ID.
  **Network.** See the exception below.
- **Pandoc / CSL** — render citations into formatted output. Offline; a
  formatting concern, kept at the presentation edge rather than in the domain.

**Offline-guarantee exception.** `CLAUDE.md` states "Local models via Ollama;
nothing leaves the machine." The third reader breaks that, so it is opt-in
behind a flag, **default off**, and `CLAUDE.md` gains an explicit note naming
which tools reach the network. The resolver records per-record whether metadata
came from disk or the wire, so a citation's own provenance is inspectable. A
stated guarantee that quietly stops holding is worse than one with a documented
exception.

**Open questions.** Whether citations become KG nodes or a sibling store;
whether `claims.py` gains a cite-key verdict; cache policy for network lookups.

## 3. Experiment-tracker adapters

**Fixes:** coverage. Small and independent.

**Scope.** New adapters reading **local artifacts already on disk** —
`wandb/run-*/files/` and `mlruns/` — through the existing
`ledger_adapters` seam. Prefer teaching `generic` these conventions over adding
named adapters, per that package's own docstring.

**Explicitly out of scope.** Calling W&B or MLflow servers, and writing runs
back into either. `ledger.py` opens by declaring itself "deliberately NOT an
experiment tracker… requires no change to how anything is run," with adoption
cost named as the design constraint. Reading local artifact directories honours
that; wrapping training or logging into a tracker inverts it. If bidirectional
sync is ever wanted, it needs its own spec arguing against this docstring, not
a quiet extension of this one.

**Open questions.** Whether `mlruns/` metric-per-file layout fits `generic`'s
conventions or genuinely defeats them; how tracker-declared metric directions
map onto the ledger's refusal to rank undeclared ones.

## 4. Swarm + `swarm.toml`

**Fixes:** internal orchestration.

**Scope.** Per-domain agents (ingest, rerank, knowledge, provenance, citations)
with a user-interaction agent as supervisor, defined in a `swarm.toml` that sits
alongside `feeds.toml` and is read by attestation itself. Repo-native and not
locked to any one agent runtime.

**The determinism constraint — load-bearing.** This project's value rests on
the ledger being auditable and ranking being deterministic. Agents messaging
each other produce nondeterministic call sequences. The constraint that keeps
both true:

> Agents may **orchestrate only** — choose which service calls to make and in
> what order. Every service call is pure Python. The supervisor writes the full
> call trace to the ledger, so any run stays reconstructable.

The swarm is a planner over an auditable substrate. No agent computes a rank, a
metric, or a verdict; agents decide which deterministic function runs next.

**Measure before building.** Spec 1 alone may substantially resolve the
confusion this spec exists to fix. Namespaced groups plus a split server address
tool-choice and coding-agent confusion directly. Re-check whether the confusion
persists after spec 1 lands before committing to this one — it is the expensive
spec and the one that spends determinism.

**Open questions.** Message contract shape; whether agents run in-process or as
subprocesses; failure and timeout semantics; how a partial swarm run is recorded
in the ledger.

## 5. Agent-config emitters

**Fixes:** portability.

**Scope.** Generate runtime-specific agent definitions from `swarm.toml` —
`.claude/agents/*.md` with frontmatter for Claude Code, and whatever other
runtime is wanted. One schema, many consumers, so the definitions cannot drift.
Emitters are generated artifacts, never hand-edited.

**Open questions.** Whether emission is a build step, a CLI subcommand, or part
of `install.py`; how to detect a hand-edited emitted file.

---

## Decisions recorded

- **Subagents: yes, deliberately.** Raised the determinism objection; the
  decision was made to build real subagents anyway. Spec 4 constrains them to
  orchestration so the audit trail survives.
- **Experiment tracking: local artifacts only**, per `ledger.py`'s stated
  non-goal.
- **Hermes configs: repo-native**, not locked to Claude Code; emitters in
  spec 5.
- **Citations: all four readers**, with the network one opt-in and the
  offline-guarantee exception documented rather than silent.

## What this roadmap does not do

It designs nothing. Each section names a problem, a scope, and its open
questions so that its own brainstorm starts from a written boundary instead of
a blank page. Spec 1 is the recommended next brainstorm; spec 3 is the cheapest
if a quick win is wanted first.
