# Onion refactor — design

**Date:** 2026-08-21
**Status:** proposed
**Roadmap:** spec 1 of 5, see `2026-08-21-architecture-roadmap.md`

## Problem

Three failures share one root: there is no layer boundary anywhere in this
codebase.

**Tool-choice confusion.** `mcp_server.py` exposes 36 tools in one flat
namespace. `runs_compare`, `kg_path`, `sym_integrate` and `digest` appear to a
calling agent as peers with equal claim on any question. Nothing in the surface
says `runs_scan` precedes `runs_compare`, or that `sym_*` has nothing to do with
the feed.

**Coding-agent confusion.** `mcp_server.py` is 1454 lines holding 36 tools, 36
`_impl` functions, and 18 raw SQL strings. `cli.py` (444) is tangled to a
similar degree.

**Correction, measured 2026-08-21.** An earlier draft of this spec claimed the
deferred `from attestation import ...` statements inside function bodies were
"the module confessing to import cycles it routes around," and proposed a test
banning them. That was wrong, and the test would have been actively harmful.

An AST probe found 58 such imports (not 11). Building the module graph from
top-level imports and testing each deferred import for a would-be cycle found
**exactly one real cycle** (`symbolic` <-> `symbolic_ops`). The other 29 edges
are deliberate lazy-loading. `sklearn` costs 929ms to import, `mcp` 526ms,
`fastapi` 280ms, `sympy` 252ms; `attest --help` currently returns in 0.22s.
Promoting `cli -> rank` to top level would put a second onto every CLI
invocation.

The deferred imports are a load-bearing performance optimization. They stay.
The only one worth removing is the genuine `symbolic` cycle.

**No data layer.** Eleven modules speak `sqlite3` directly; 94 raw SQL strings
are spread across the tree. `db.py` is connection management and migrations,
not a repository. That the outermost layer writes its own SQL is the clearest
statement of the problem.

`_list_feed_impl` shows all three at once. It clamps input, opens a connection,
orchestrates three domain calls, and hand-builds a response dict. Four jobs, one
function, no name for any of the seams.

## Layers

Dependencies point strictly inward.

```
presentation   cli.py · server.py · mcp/{feed,knowledge,provenance,symbolic}.py
      ↓        constructs repos, calls facades. No SQL. No domain imports.
services       FeedService · KnowledgeService · ProvenanceService · SymbolicService
      ↓        orchestrates domain calls, shapes responses, owns the error contract
domain         rank · kg · ledger · claims · corpus · features · embed · explain
      ↓        pure logic. Names protocols. Never imports sqlite3.
ports          FeedRepo · KnowledgeRepo · LedgerRepo · ChatPort · EmbedPort
      ↑        Protocol classes only. No implementations.
infrastructure sqlite/ · llm clients · feedparser · migrations
```

`ports` sits below domain but is imported by it: domain depends on the
abstraction, infrastructure implements it. That inversion is what makes this an
onion rather than a stack.

**The rule is mechanically checkable.** `import sqlite3` appears only under
`infrastructure/`. This is enforced by a test (see Testing), not by a convention
in a document nobody re-reads.

### Where the reliability contract moves

`explain.py` and `rank.py` catch broad exceptions deliberately — ranking must
never wait on explanations, and a cold Ollama must degrade to a cached vector
rather than a 500. That contract **moves up to the service layer**. Services own
"never raise into presentation"; domain code becomes free to raise honestly.

This is an improvement, not a relocation. Today the swallowing happens deep in
domain code where it also hides real bugs. Concentrated in services, the
degradation is deliberate and visible in one place per domain.

**Correction:** `CLAUDE.md` states `BLE001` is per-file-ignored in
`pyproject.toml`. It is not — `pyproject.toml` has no `per-file-ignores` section
at all. There are four inline `# noqa: BLE001` comments (`cli.py:244`,
`install.py:292`, `symbolic.py:238`, `rank.py:170`), each carrying its own
reason. That is the better arrangement: the suppression travels with the code
when it moves. `CLAUDE.md` gets fixed as part of stage 1.

## Repositories

Carved per-aggregate, one repository backing one service backing one tool group.

Reading the code revised the roadmap's guess of five repositories down to
**three**:

| Port | Tables owned | Backs |
|---|---|---|
| `FeedRepo` | `users`, `feeds`, `items`, `clicks`, `explanations`, `item_features`, `item_vectors` | `FeedService` → `feed.*` |
| `KnowledgeRepo` | `kg_nodes`, `kg_edges`, `kg_meta`, `item_tags` | `KnowledgeService` → `kg.*` |
| `LedgerRepo` | `runs`, `run_metrics`, `corpora`, `corpus_splits` | `ProvenanceService` → `runs.*` |

Two deliberate departures from the roadmap:

**No `ClaimsRepo`.** `claims.py` verifies prose against recorded runs. It owns no
tables and is a *consumer* of `LedgerRepo`, not a peer. Giving it a repository
would create an empty abstraction.

**No `CorpusRepo`.** `runs.corpus_id` links every run to its corpus, and
`compare()` guards arms that cross one. Splitting corpus out would force a join
across two repositories on every comparison — the aggregate boundary is wrong
there. `corpora` and `corpus_splits` belong to `LedgerRepo`.

**`item_tags` goes to `KnowledgeRepo`,** not `FeedRepo`. `kg.build_graph()`
derives the graph fresh from it on every read. It is knowledge-owned data that
ingestion happens to write.

`SymbolicService` has no repository: `symbolic.py` and `symbolic_ops.py` touch
no tables. Its port is the process-isolation boundary (`run_isolated`), not
storage.

### Connection lifetime

Repositories own it, constructed at the presentation edge:

```python
# mcp/feed.py
def list_feed(user: str, limit: int = 10, since_days: int | None = 14) -> dict:
    limit = min(max(int(limit), 1), MAX_LIST_LIMIT)
    svc = FeedService(SqliteFeedRepo(resolve_db_path(None)), embedder())
    return svc.list_feed(user, limit, since_days)
```

The repository opens and closes per method call. This preserves `open_db()`'s
existing one-connection-per-tool-call contract exactly — it is relocated behind
an interface, not changed. No pooling, no unit-of-work, no shared global
connection: WAL plus `check_same_thread=False` across FastAPI's threadpool is
where subtle bugs live, and this refactor is not the place to take that on.

`db.py` retains migrations, `SCHEMA`, `embed_dims()`, `resolve_db_path()` and
`seed_demo_users()`, and moves to `infrastructure/sqlite/`. Its
dimension-mismatch guard in `get_db()` stays exactly where it is.

## Services

Each facade owns what `_impl` was improvising. The split of `_list_feed_impl`:

- **presentation** clamps `limit`, resolves the db path, constructs the repo
- **service** orchestrates `get_user` → rank → `_ranking_quality`, shapes the
  response, catches and degrades
- **domain** computes ranks and quality from data handed to it

The `{"ok": bool, "message": str, ...}` envelope becomes a **declared type**
rather than a convention re-typed 36 times. That envelope is the actual contract
a calling agent reads, so it deserves a name and a test.

`_ranking_quality()`'s honesty reporting — `classifier_active` plus caveat —
stays in the service layer and stays mandatory in the response. A reader must
not be able to assume the ranker learned something it did not.

## Presentation

`mcp_server.py` (1454) splits into four domain modules plus a thin registrar:

```
mcp/__init__.py       FastMCP construction, registers all four
mcp/feed.py           feed.*   — 16 tools
mcp/knowledge.py      kg.*     —  5 tools
mcp/provenance.py     runs.*   —  7 tools (runs + claims)
mcp/symbolic.py       sym.*    —  7 tools
```

Namespacing turns one 36-way choice into a 4-way choice followed by a 5-to-16
way choice. The hierarchy the swarm was reached for is obtained here without
spawning a process. This is the change that must be measured before spec 4 is
committed to.

Old flat names alias to the namespaced ones for one release. Aliases do not warn
on use: a warning reaches the agent's transcript as noise it cannot act on,
since the tool name comes from a config it may not control. Aliases are listed
in the release notes and removed on schedule.

`cli.py` and `server.py` retarget onto the same facades. This fixes the
three-presentations drift directly: `server.py` stops importing `rank_items`,
`cli.py` stops making HTTP calls to reach logic it could call in-process.

## Testing

This repo's recurring failure mode is recorded in `CLAUDE.md`: *tests that pass
against the bug they were written to catch*. A refactor is where that failure is
most likely, because the tests move at the same time as the code. Three defenses.

### 1. The boundary is a test, not a doc

```python
def test_no_sqlite3_outside_infrastructure():
    """The onion's one mechanical rule. A doc nobody re-reads cannot enforce it."""
```

Walks `src/attestation/`, asserts `import sqlite3` appears only under
`infrastructure/`. Companion tests assert no deferred `from attestation import`
inside function bodies (AST-checked, catching the cycle-dodging that motivated
this work), and that no module under `presentation/` imports from `domain/`.

These fail loudly the first time someone reaches through a layer, including
future agents, which is the point.

### 2. Characterization tests before each move

Each of the nine domain modules migrating off `sqlite3` in stage 3 gets its
behavior pinned **before** it moves:

- Capture current output for a representative set of inputs against a real
  SQLite fixture.
- Move the module behind its repository.
- Assert byte-identical output.

A refactor that changes behavior has failed, and the characterization test is
the only thing that can say so. Where current behavior is *wrong*, it is pinned
as-is and fixed in a separate commit that says so — never silently corrected
mid-move, because a refactor commit that also changes behavior is unreviewable.

### 3. Fake repositories, kept honest

Each port gets an in-memory fake alongside its SQLite implementation. The
precedent exists: `conftest.py`'s `FakeEmbedder` is already a hand-rolled port
implementation that was never named as one.

The risk of a second implementation is that it drifts and tests pass against a
fake that no longer resembles the real thing. Defense: **one contract test suite
runs against both.** Any behavior a fake claims must be demonstrated by the
SQLite implementation under the same assertions.

```python
@pytest.fixture(params=["sqlite", "fake"])
def feed_repo(request, tmp_path): ...

# every test in TestFeedRepoContract runs twice
```

Service and domain tests then use fakes and stay fast; the contract suite and
characterization tests hold the real boundary.

### Migrating the 83 `_impl` call sites

Tests retarget onto service facades — which is what `_impl` was approximating —
incrementally, per domain, alongside stage 3 and 4. No single PR carries 83
rewrites. `_impl` names disappear entirely by stage 5 rather than surviving as
aliases: leaving the private seam in place invites new code to reach for it,
which is how it became a seam in the first place.

### Gate

Every stage lands green under the full `pre-commit run --all-files` — ruff
format, ruff check, ty, uv.lock sync, full pytest. The ~70s pytest hook is not
bypassed at any stage. CI runs the same five gates on Linux and macOS across
Python 3.12 and 3.13.

## Staging

Each stage lands independently green.

| # | Stage | Risk | Notes |
|---|---|---|---|
| 1 | Ports + protocols + boundary tests | None | New files; nothing imports them |
| 2 | SQLite repos + fakes + contract suite | Low | `db.py` → `infrastructure/sqlite/` |
| 3 | Domain off `sqlite3`, one module at a time | Medium | 9 PRs, each characterization-pinned |
| 4 | Service facades; `cli.py` + `server.py` retarget | Medium | Fixes three-presentation drift |
| 5 | Split `mcp_server.py`; namespace tools | Low | Aliases for one release |

Stage 3's nine modules: `rank`, `kg`, `ledger`, `claims`, `corpus`, `features`,
`feeds`, `ingest`, `explain`.

## Success criteria

- `import sqlite3` outside `infrastructure/` — zero occurrences, enforced by test
- Module import graph acyclic at module scope — enforced by test, currently
  passing; the test is verified against an injected cycle rather than trusted
- `attest --help` under 0.5s — enforced by test, protecting the lazy imports
- No module over ~400 lines (from 1454, 637, 614)
- All 36 tools reachable under both namespaced and legacy names
- Full gate green at every stage; no behavior change attributable to any
  refactor commit

## Open questions

**`_PROFILE_VEC_CACHE`.** Keyed on `_db_identity`, currently exported from
`rank.py` to `mcp_server.py` — a private name crossing a module boundary, which
is the missing interface this refactor exists to supply. Recommendation:
repo-owned, since cache validity is a storage-identity concern and `_db_identity`
is already a storage question. To be settled in stage 3's `rank` PR.

**`explanations` table ownership.** Assigned to `FeedRepo` above, but `explain.py`
is arguably its own domain. Left in `FeedRepo` because explanations are read only
in the context of a ranked feed. Revisit if `explain` grows independent callers.

**`install.py` (614 lines).** Doing five jobs — installer, doctor, cron
scheduler, Ollama puller, MCP wiring. Out of scope here: it sits outside the
onion entirely, touching the filesystem and other processes rather than this
system's layers. It needs its own spec.

## What this spec does not do

No behavior changes. No new features. No LLM anywhere it is not already. The
ranker stays deterministic, the ledger stays artifact-read, and
`digest`/`runs_compare` keep returning structure rather than prose.
