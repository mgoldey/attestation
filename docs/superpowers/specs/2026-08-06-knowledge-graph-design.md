# Knowledge graph — design

**Date:** 2026-08-06
**Status:** approved (brainstorming dialogue; all sections approved)

## Problem

hermes-rss knows a great deal about what you read — 1619 tagged items across
2347 tags, plus every click — but it can only answer one question at a time
("what should I read next"). It cannot answer questions *about the shape* of
what you read: what dominates it, what connects to what, which topics bridge
otherwise separate clusters.

A knowledge graph over the existing data answers those. Nothing here needs
new content: the tagging pass already extracted the concepts.

## What the data actually supports (measured, not assumed)

These numbers drove every decision below.

| Measurement | Value |
|---|---|
| Tagged items | 1619 |
| Distinct tags | 2347 |
| **Tags used exactly once** | **2020 (86%)** |
| Tags used 2+ times | 327 |
| Concept nodes after filtering (≥2 uses, edge weight ≥2) | 176 |
| **Largest connected component** | **170 of 176** |
| Full graph build from SQL | **0.226s** |

Two findings shape the design:

**The singleton tail is noise.** 86% of tags appear once and connect to
nothing. A graph including them would be mostly isolated points. Filtering at
≥2 uses is what turns this data into a graph.

**Variant tags split hubs.** `machine-learning` (degree 90) and
`machinelearning` (degree 36) are one concept counted twice; `ai`,
`large-language-models`, and `deep-neural-networks` overlap heavily. Until
merged, every centrality number is wrong in a way that looks plausible.

**`CITES` edges are not buildable.** RSS provides title, summary, and URL —
no reference lists. The originating notes' headline example ("papers that use
method X and were cited by Y") cannot be answered from this data, and the
design does not pretend otherwise.

## Decisions

### Storage: `kg_nodes` / `kg_edges` in `hermes.db`

Materialized tables, not derive-on-demand.

This is a deliberate choice against the simpler option, made with the
trade-off visible: the graph is *fully derived*, and a 0.226s rebuild would
fit inside any tool call, so tables buy little today and introduce a cache
that can drift. They are chosen anyway because they give hand-authored nodes
(ideas, experiments) somewhere to live later without a further migration, and
because repeat queries get cheaper.

The drift risk is real and is handled explicitly, below — not left latent.

### Staleness: `hermes tag` auto-rebuilds

`features.tag_one_item` is the single mutation point for `item_tags`, so the
graph has exactly one thing that can invalidate it.

`run_tagging` calls `kg_rebuild` **once, after the loop finishes** — not per
item. Measured cost: 0.226s against a ~571s tagging run for the current
408-item backlog, i.e. **0.04% overhead**. Per-item rebuilding would be ~92s
of pure waste.

`kg_meta` still stores a source fingerprint (`max(item_tags.rowid)` plus
`COUNT(*)`), and read tools still report `stale: true` when it does not match.
Auto-rebuild closes the ordinary drift window; the fingerprint catches the
cases it cannot — a database edited by hand, a restored backup, or a tagging
run that died partway.

### Node and edge types

| Node type | Source |
|---|---|
| `concept` | A tag used ≥2 times, after alias merging |
| `paper` | An item in `items` |
| `source` | A feed in `feeds` |

| Edge type | Direction | Weight |
|---|---|---|
| `MENTIONS` | paper → concept | 1 |
| `CO_OCCURS` | concept ↔ concept | count of shared items (≥2) |
| `PUBLISHED_IN` | paper → source | 1 |

No `CITES`. No `USES_METHOD`, `EXTENDS`, or `CONTRADICTS` — those need
relationship extraction the data does not support, and inventing them from
co-occurrence would misrepresent correlation as a typed claim.

### Alias table

A committed `src/hermes/kg_aliases.toml` maps variants to canonical form:

```toml
"machinelearning" = "machine-learning"        # 141 uses -> merges into 532
"llm" = "large-language-models"               # 6
"llms" = "large-language-models"              # 4
"language-models" = "large-language-models"   # 103
"nlp" = "natural-language-processing"         # 68 -> merges into 101
"transformer" = "transformers"                # 2 -> merges into 109
```

Every entry above was verified present in the live database with the use
count shown; the seed list contains no invented tags. Note what is
deliberately *absent*: `deep-neural-networks` (104) and `neural-networks`
(52) are near-synonyms, and `deep-learning` (103) overlaps
`machine-learning`, but merging genuinely distinct-if-related concepts would
destroy the structure the graph exists to reveal. The table merges spelling
and abbreviation variants only, never conceptual hierarchies.

Hand-written, deterministic, reviewable, and version-controlled. It is a
maintenance burden the user has accepted; the alternative (LLM tag
consolidation) would be a large batch of calls on a machine that
OOM-crashed on 2026-08-06, and would need review anyway.

Aliasing is applied when building nodes, so `machinelearning`'s 36 edges
merge into `machine-learning` rather than competing with it.

## MCP tools

Five new tools, bringing the served surface from 23 to **28**. Each wraps an
`_impl` following the existing `mcp_server.py` pattern, with the structured
`{"ok": ..., "message": ...}` contract and success-path keys preserved on
error paths.

| Tool | Behavior |
|---|---|
| `kg_neighbors(node, limit=20)` | Direct neighbours, ranked by co-occurrence weight. The "what else should I read" query. (A `depth` parameter was specified here and shipped, then removed — multi-ring traversal produced four defects; `kg_path` answers multi-hop questions exactly.) |
| `kg_path(source, target)` | Shortest path between two concepts; clean "no path" when they are in different components. |
| `kg_central(metric="degree", limit=10)` | Most connected or most bridging concepts. `metric` is `degree` or `betweenness`. |
| `kg_communities(min_size=3)` | Topic clusters via label propagation, each with its highest-degree member as a label. |
| `kg_rebuild(confirm)` | Regenerate both tables. Idempotent. Requires `confirm=true` since it replaces table contents. |

**No `networkx` dependency.** Centrality (degree, plus betweenness by
Brandes' algorithm) and label-propagation communities are implemented in
pure Python over an adjacency dict. This is not asceticism: `networkx`
silently disappeared from this environment once it was no longer a
transitive dependency, and the graph is small enough (176 nodes) that BFS
over a dict is both fast and obvious. One fewer dependency to break.

## Module layout

New `src/hermes/kg.py`, importing only `db` and the standard library:

```
build_graph(conn) -> dict[str, set[str]]      # adjacency, alias-applied, filtered
rebuild(conn) -> dict                          # writes kg_nodes/kg_edges, returns counts
fingerprint(conn) -> str                       # cheap staleness hash
is_stale(conn) -> bool
neighbors(conn, node, limit) -> list[dict]    # direct neighbours only
shortest_path(conn, source, target) -> list[str] | None
central(conn, metric, limit) -> list[dict]
communities(conn, min_size) -> list[dict]
```

`mcp_server.py` gains five thin `_impl` + `@mcp.tool()` pairs. `features.py`
gains one call to `kg.rebuild` at the end of `run_tagging`.

## Testing

- **Graph construction from a fixture**: known items and tags produce the
  expected nodes, edges, and `CO_OCCURS` weights.
- **The filter**: a tag used once is absent; a tag used twice is present.
- **Aliasing**: `machinelearning` and `machine-learning` merge into one node
  whose degree is the union, not either alone.
- **Staleness**: the fingerprint changes after an `item_tags` write, and
  `is_stale` flips accordingly.
- **Auto-rebuild**: `run_tagging` leaves the graph fresh, and calls
  `kg.rebuild` exactly **once** regardless of how many items it tagged.
- **Idempotency**: two consecutive `rebuild` calls yield identical tables.
- **`kg_path`**: finds a real path inside a connected fixture and returns
  `None` (not an error) for genuinely disconnected nodes.
- **`kg_rebuild(confirm=False)`** mutates nothing.
- **Algorithms against hand-computed answers** on a small fixture, so a
  wrong betweenness implementation cannot pass by agreeing with itself.
- All existing tests pass untouched.

## Out of scope (YAGNI)

`CITES` and other typed relationships requiring extraction the data cannot
support; LLM triple extraction from summaries; hand-authored `idea` and
`experiment` nodes (the tables are shaped to accept them later, but nothing
creates them now); Obsidian vault sync; graph visualization; export to
GraphML/Neo4j; incremental graph updates (a full rebuild is 0.226s);
`networkx`.

## Sequencing note

The implementation plan should keep three concerns separable: (1) `kg.py`
with construction, aliasing, and the fingerprint; (2) the algorithms
(neighbors, path, centrality, communities) with their hand-computed tests;
(3) the MCP tools plus the `run_tagging` hook and documentation. Task 1
delivers a queryable graph on its own.
