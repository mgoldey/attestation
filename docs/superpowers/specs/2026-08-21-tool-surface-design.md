# Tool surface — design

**Date:** 2026-08-21
**Status:** implemented 2026-08-21/22, with two deviations recorded below
**Supersedes:** `2026-08-21-onion-refactor-design.md`
**Roadmap:** replaces spec 1 of 5 in `2026-08-21-architecture-roadmap.md`

## Problem

Unchanged from the superseded spec, and still real:

- **Tool-choice confusion.** 36 tools in one flat namespace. `runs_compare`,
  `kg_path`, `sym_integrate` and `digest` present to a calling agent as peers
  with equal claim on any question.
- **Coding-agent confusion.** `mcp_server.py` is 1454 lines.

What changed is the diagnosis of the second one. The file is not long because it
lacks layers. It is long because **every tool is written twice and each copy
repeats the same ritual**: 26 copies of `with open_db()`, 28 broad `except`
blocks, 10 hand-written unknown-user checks, and 36 `_impl`/tool pairs where the
public half is a docstring plus one delegating call.

`_propose_interests_impl` is the whole problem in 25 lines: one `GROUP BY`,
wrapped in a connection, a try, a success envelope typed by hand, and a failure
envelope typed by hand again.

## Why the onion was dropped

Two reviews, run independently against the superseded spec — one on
simple-vs-easy, one on module depth — converged:

- **Stages 1-4 relocate the tangle.** The `_impl` split moves
  clamp/orchestrate/shape/catch from one function into two files. Same four
  jobs, one more boundary.
- **A 34-method `FeedRepo` is shallow.** ~1.4 SQL statements per method is a
  one-method-per-callsite mapping. `item_exists(id)` is `SELECT 1 FROM items
  WHERE id = ?` with the SQL removed and nothing put in its place.
- **Five layers to run one `SELECT`.** Today it is two.
- **`KnowledgeRepo` would be built around a dead write.** The superseded spec
  proved by grep that nothing in `src/` reads `kg_nodes` or `kg_edges`, then
  deferred the existence question to a stage that would inherit a Protocol, an
  implementation, a fake, and a contract suite to justify removing.

The deciding argument: an abstraction's existence cannot be deferred by building
and testing it first.

Four things survived both reviews. They are this spec.

## 1. Delete the write-only knowledge graph

`kg_nodes` and `kg_edges` are **write-only**. Verified: every statement touching
them in `src/` is one of the two `DELETE`s in `kg.rebuild()` (`kg.py:88-89`) or
the two `INSERT`s that follow. `README.md:262` says so outright.

`kg_meta` stores one fingerprint, read only by `is_stale()`. Follow what
`is_stale()` does: `kg_neighbors` computes its answer from `build_graph()` —
fresh from `item_tags` — and then reports `stale`, a flag describing whether a
materialization that took no part in the answer is out of date.
`CLAUDE.md` already says it: *"stored kg_nodes/kg_edges are advisory; is_stale()
never changes a read tool's answer."*

Delete: the three tables, `rebuild()`, `is_stale()`, `fingerprint()`, the
`kg_rebuild` MCP tool, the `stale` field on four kg responses, and the
`features.py:214` hook that keeps the dead tables current inside a swallowed
exception.

**Tools 36 -> 35**, which also helps problem one. Three test files lose tests
that only ever asserted the materialization against itself.

This is a deletion, not a feature removal: nothing observable changes, because
nothing observes it.

## 2. `build_graph()` becomes a pure function

```python
def build_graph(assignments: Iterable[tuple[int, str]]) -> tuple[Adjacency, Edges]:
```

instead of taking a connection. This was the superseded spec's best idea and it
needs none of its architecture.

Consequences, all good:

- `test_aliases_merge_before_filtering` — which guards the load-bearing
  alias -> frequency-filter -> co-occurrence ordering — becomes a unit test with
  no database.
- `kg.health()` stops deriving the graph twice. It calls `build_graph()` at
  `kg.py:302` and then `communities()` at `:307`, which derives it again at
  `:245`. Passing the graph down fixes a double scan that a repository boundary
  would only have made more expensive.
- The knowledge domain ends up with no storage dependency at all, which is why
  `KnowledgeRepo` was vestigial.

One caller reads `item_tags` and passes the result in.

## 3. One envelope, owned in one place

The `{"ok": bool, "message": str, ...}` shape is the contract a calling agent
reads, and it is currently maintained by hand in 36 places. It has already
drifted: `_profile_status_impl` hand-writes eight keys in its error branch to
match its success branch; `_runs_compare_impl` has three exits each re-listing
`family`, `metric`, `arms`, `winner`; `_explain_item_impl` writes
`"explanation": None` three times.

A caller cannot rely on "the failure shape matches the success shape," because
that property is maintained by hand thirty-six times.

Fix it with a declared type **and** a decorator that owns the ritual:

```python
@tool(empty={"items": [], "ranking_quality": {}}, needs_user=True)
def list_feed(conn, user_row, limit: int = 10, since_days: int | None = 14):
    items = _ranked_items(conn, user_row, limit, since_days)
    return {"items": [...], "ranking_quality": _ranking_quality(conn, user_row["id"])}
```

The decorator opens and closes the connection, resolves the user or returns the
unknown-user envelope, wraps the return in `ok: True`, and catches broadly to
return `ok: False` with `empty` filled in. Roughly 20 lines replacing 26
connection blocks, 28 excepts, and 10 user checks.

**Scope limit, and it matters.** This consolidates the *presentation* envelope
only. The four inline `# noqa: BLE001` sites stay exactly where they are.
`rank.py:198` is not a generic swallow — it implements a specific documented
policy: embedder down with a warm cache serves a stale vector; a cold cache
raises. A decorator does not know a cache exists. The superseded spec proposed
hoisting these to a service layer, which would have been the behavior regression
it forbade elsewhere in the same document.

## 4. Namespace the tools and split the file

Both reviews called this the change that justifies the work on its own.

```
mcp/feed.py        feed.*   — 16 tools
mcp/knowledge.py   kg.*     —  4 tools  (kg_rebuild deleted in §1)
mcp/provenance.py  runs.*   —  7 tools
mcp/symbolic.py    sym.*    —  7 tools
```

A 36-way choice becomes a 4-way choice followed by a 4-to-16-way one. Old flat
names alias for one release, without deprecation warnings: the tool name comes
from a config the calling agent may not control, so a warning is noise it cannot
act on.

The `_impl` doubling goes with it. It exists for one stated reason —
`CLAUDE.md`: *"each tool pairs with `_<name>_impl()` kept FastMCP-free so tests
import it directly"* — which is a testing workaround braided into the production
surface. `mcp.tool()` applies as a plain call rather than a decorator, so one
definition serves both. That removes roughly 500 lines.

With §3's decorator and the doubling gone, each module lands near 200 lines **as
a consequence**, not as a target. Line count is dropped as a criterion: it would
pass a `FeedRepo` with 34 shallow methods and fail `ledger.py`, whose 637 lines
are one coherent argument about when a comparison should not be trusted.

## What is deliberately NOT in this spec

**No ports, no repositories, no service facades, no fakes, no contract suite.**
Both reviews argued these are cost without depth at this codebase's size, and
neither found a defect attributable to SQL living near its callers.

**No `queries.py` consolidation yet.** It is defensible and remains available,
but after §1-§4 the motivation may evaporate. Revisit with evidence.

**`_PROFILE_VEC_CACHE` stays module-level.** The superseded spec proposed
repo-owned; both reviews called that complecting memoization with storage. There
is a real latent bug — `_update_persona_impl` changes `users.interests` without
evicting, safe today only because the cache value is keyed on
`sha256(interests)`, so a changed string misses and recomputes. Correct by
accident, and the accident is load-bearing. **Fix the eviction; leave the
ownership.** One line, own commit, own test.

**`install.py` (614 lines) stays untouched.** It needs its own spec.

## The `record_scan()` transaction fix survives

The superseded spec's best piece of engineering, kept independent of the
architecture that produced it. `ledger.scan()` interleaves `corpus.upsert()` and
`_replace_project()` across every project and commits once at `ledger.py:253`,
with `runs.corpus_id` a foreign key to a row created mid-transaction.

No repository is introduced here, so nothing threatens that transaction — but
the **atomicity test is still worth writing**, because nothing currently proves
it: fail a multi-project scan partway through, assert no orphaned corpora and no
runs with a dangling `corpus_id`.

## Testing

**Characterization before deletion.** §1 removes code. Pin the four kg read
tools' output first, delete, and assert byte-identical output minus the `stale`
key. If any answer changes, the premise was wrong and the deletion stops.

**§2 converts a DB test to a unit test.** `test_aliases_merge_before_filtering`
must keep asserting the same ordering; only its fixture changes.

**§3 needs a test the old code could not have.** One parameterized test asserting
every tool's failure envelope has the same keys as its success envelope. That
property is what drifted, and it was never checkable while the shape was
hand-written 36 times.

**§4 is mechanical.** Assert all 35 tools resolve under both namespaced and
legacy names.

Full `pre-commit run --all-files` at every stage. The architecture tests in
`tests/test_architecture.py` stay: the sqlite3 confinement test loses its
purpose without repositories and is **deleted with §1**, but the acyclic-import
and CLI-startup tests are independently valuable and remain.

## Order

1. Delete the write-only graph (§1) — pure deletion, biggest single reduction
2. `build_graph()` purity (§2) — one signature, enables the kg unit test
3. Envelope decorator (§3) — removes the ritual, several hundred lines
4. Namespace + split + kill `_impl` (§4) — the measurable payoff
5. `_PROFILE_VEC_CACHE` eviction fix — one line, unrelated to the rest
6. `record_scan()` atomicity test — proves an invariant nothing proves today

Each is independently revertible and independently valuable. §4 is where the
stated problem gets measured: after it lands, check whether tool-choice
confusion actually improved before committing to the swarm in roadmap spec 4.

## Outcome

Delivered, with two departures from what was written here.

**Tool count is 37, not 35.** The count was wrong in the spec to begin with
(the repo had 35, not 36, and `CLAUDE.md` claimed both), then `kg_rebuild` was
deleted and three tools were added during implementation: `kg.concepts`,
`feed.harvest_engagement`, `feed.simulate_ratings`.

**No aliases.** The spec proposed keeping the flat names for one release. That
would have doubled the listing to ~74 entries, making the surface worse in
exactly the way the rename exists to fix, so the old names were removed
outright and the skill documentation was updated in the same commit.

**Namespacing landed a day late.** The file split shipped first and the rename
did not, which a later architecture review caught: tools had moved between
files, which a calling agent cannot see, so the primary stated problem was
untouched while the spec's success criteria read as met. Names now carry the
namespace and two tests in `test_architecture.py` enforce it.

**The `_impl` doubling survives**, contrary to what is written below. Each tool
is still a thin `@mcp.tool()` wrapper over a module-level implementation. That
turned out to be load-bearing rather than incidental: the wrapper holds the
docstring an agent reads, the implementation is what tests call directly, and
`mcp.tool()` applied as a plain call would have merged the two only by giving
up one of those. Not delivered, and no longer wanted.

## Success criteria

- 37 tools, namespaced `feed.*` / `kg.*` / `runs.*` / `sym.*` — DONE
- ~~No `_impl` doubling; one definition per tool~~ — NOT DONE, see above
- `mcp_server.py` -> four modules, each near 200 lines as a consequence
- Zero `kg_nodes`/`kg_edges`/`kg_meta` references outside migrations
- Every tool's failure envelope structurally matches its success envelope,
  enforced by test
- The four `# noqa: BLE001` domain degradations unchanged and unmoved
- Full gate green at every step; no behavior change outside the deletions in §1
