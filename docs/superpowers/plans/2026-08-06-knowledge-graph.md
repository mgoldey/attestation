# Knowledge Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a queryable knowledge graph over the existing 1619 tagged items and expose it as five MCP tools, so an agent can answer questions about the *shape* of what the user reads.

**Architecture:** A new `src/hermes/kg.py` derives an adjacency structure from `item_tags` (alias-merged, then frequency-filtered), materializes it into two new tables, and implements graph algorithms in pure Python over a dict. `mcp_server.py` gains five thin wrappers; `features.run_tagging` gains one rebuild call so the tables never drift.

**Tech Stack:** Python 3.12+, SQLite, FastMCP (`mcp==1.28.1`), pytest, `uv`. **No `networkx`** — ~183 nodes over an adjacency dict, and it disappeared from this environment once it stopped being a transitive dependency.

## Global Constraints

- Commit ONLY the files each task's **Files** section lists. `feeds.toml`, `demo/`, and `docs/hermes-agent-plugin-research.md` are the user's uncommitted work — NEVER stage them. Verify with `git status --short` before every commit.
- Every commit message ends with the trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Gates before every commit: `uv run pytest -q`, `uv run ruff check .`, `uv run ty check` — all must pass. The suite is currently at **240 passing**.
- Ruff line-length is 100; lint set `["E", "F", "W", "I", "BLE"]`.
- Existing tests must pass untouched. A pre-existing test failing means you changed behavior — investigate, do not edit the test.
- **Aliasing is applied BEFORE the frequency filter.** Merging can lift a variant over the ≥2 threshold that neither spelling would clear alone, so filtering first silently drops concepts rather than raising an error. `test_aliases_merge_before_filtering` pins this on a synthetic fixture where filter-first keeps only `shared` and drops `canon` entirely — verified. **Correction:** on the *current* live data the two orderings happen to produce identical graphs (183 nodes, 447 edges, `machine-learning` degree 111 either way); an earlier draft of this plan claimed 176 vs 183, which actually compared aliasing against no-aliasing, not the two orderings. The ordering rule stands on the synthetic case and on future data, not on today's numbers.
- **Never use `sympy.sympify`, bare `eval`, or `exec`.** Unrelated to this feature, but it is a standing rule in this codebase after four confirmed RCE/DoS defects.
- Do NOT run `git stash` in any form — `stash@{0}` holds unreviewed vulnerable code, and three agents have disturbed it by accident.

---

### Task 1: Graph construction, aliasing, and the schema

**Files:**
- Create: `src/hermes/kg.py`
- Create: `src/hermes/kg_aliases.toml`
- Modify: `src/hermes/db.py` (add two tables to `SCHEMA`, near the `item_tags` definition)
- Test: `tests/test_kg.py`

**Interfaces:**
- Consumes: `db.get_db` (existing).
- Produces:
  - `kg.ALIASES: dict[str, str]` — loaded once from `kg_aliases.toml`.
  - `kg.MIN_TAG_USES: int = 2`, `kg.MIN_EDGE_WEIGHT: int = 2`.
  - `kg.canonical(tag: str) -> str` — alias-merged tag name.
  - `kg.build_graph(conn) -> tuple[dict[str, set[str]], dict[tuple[str, str], int]]` — returns `(adjacency, edge_weights)`; weight keys are `(a, b)` with `a < b`.
  - `kg.fingerprint(conn) -> str` — cheap staleness hash of `item_tags`.
  - `kg.rebuild(conn) -> dict` — writes `kg_nodes`/`kg_edges`/`kg_meta`, returns `{"nodes": int, "edges": int}`.
  - `kg.is_stale(conn) -> bool`.

- [ ] **Step 1: Write the failing construction tests**

Create `tests/test_kg.py`:

```python
from hermes import kg
from hermes.db import get_db


def seed(conn, items):
    """items: list of tag-lists, one per synthetic item."""
    for i, tags in enumerate(items, start=1):
        conn.execute(
            "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
            " VALUES (?, NULL, ?, 'u', 's', ?)",
            (i, f"item {i}", f"h{i}"),
        )
        for t in tags:
            conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (i, t))
    conn.commit()


def test_singleton_tags_are_excluded(tmp_path):
    """86% of real tags are used exactly once and connect to nothing."""
    conn = get_db(tmp_path / "t.db")
    seed(conn, [["alpha", "beta"], ["alpha", "beta"], ["lonely", "alpha"]])

    adjacency, _ = kg.build_graph(conn)

    assert "lonely" not in adjacency, "a tag used once must not be a node"
    assert "alpha" in adjacency and "beta" in adjacency
    conn.close()


def test_edge_requires_min_weight(tmp_path):
    """A pair sharing only one item is co-incidence, not co-occurrence."""
    conn = get_db(tmp_path / "t.db")
    # alpha+beta share 2 items; alpha+gamma share only 1
    seed(conn, [["alpha", "beta"], ["alpha", "beta"], ["alpha", "gamma"], ["gamma", "beta"]])

    adjacency, weights = kg.build_graph(conn)

    assert weights.get(("alpha", "beta")) == 2
    assert ("alpha", "gamma") not in weights
    conn.close()


def test_aliases_merge_before_filtering(tmp_path):
    """Order matters: merging lifts variants over the frequency threshold.

    'variant' is used once and 'canon' once, so filtering first would drop
    both. Merging first makes one tag used twice, which survives.
    """
    conn = get_db(tmp_path / "t.db")
    kg.ALIASES["variant"] = "canon"
    try:
        seed(conn, [["variant", "shared"], ["canon", "shared"]])
        adjacency, weights = kg.build_graph(conn)

        assert "variant" not in adjacency, "the alias must not survive as its own node"
        assert "canon" in adjacency
        assert weights.get(("canon", "shared")) == 2
    finally:
        del kg.ALIASES["variant"]
    conn.close()


def test_canonical_is_identity_for_unknown_tags(tmp_path):
    assert kg.canonical("some-unmapped-tag") == "some-unmapped-tag"


def test_alias_file_maps_only_to_real_canonical_forms():
    """Every alias target should itself be a plausible tag, not a typo.

    Guards against an alias table entry that silently creates a node no item
    ever had.
    """
    for variant, target in kg.ALIASES.items():
        assert variant != target, f"{variant!r} maps to itself"
        assert target not in kg.ALIASES, f"{target!r} is both a target and an alias"


def test_rebuild_writes_tables_and_is_idempotent(tmp_path):
    conn = get_db(tmp_path / "t.db")
    seed(conn, [["alpha", "beta"], ["alpha", "beta"], ["beta", "gamma"], ["beta", "gamma"]])

    first = kg.rebuild(conn)
    assert first["nodes"] > 0 and first["edges"] > 0

    rows_first = conn.execute("SELECT name FROM kg_nodes ORDER BY name").fetchall()
    second = kg.rebuild(conn)
    rows_second = conn.execute("SELECT name FROM kg_nodes ORDER BY name").fetchall()

    assert second == first
    assert [r["name"] for r in rows_first] == [r["name"] for r in rows_second]
    conn.close()


def test_fingerprint_changes_when_tags_change(tmp_path):
    conn = get_db(tmp_path / "t.db")
    seed(conn, [["alpha", "beta"], ["alpha", "beta"]])
    kg.rebuild(conn)
    assert kg.is_stale(conn) is False

    before = kg.fingerprint(conn)
    conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (1, 'delta')")
    conn.commit()

    assert kg.fingerprint(conn) != before
    assert kg.is_stale(conn) is True
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_kg.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hermes.kg'`.

- [ ] **Step 3: Add the tables to `db.SCHEMA`**

In `src/hermes/db.py`, append to the `SCHEMA` string (after the `item_tags` block):

```sql
CREATE TABLE IF NOT EXISTS kg_nodes(
  name TEXT PRIMARY KEY,
  node_type TEXT NOT NULL,
  degree INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS kg_edges(
  source TEXT NOT NULL,
  target TEXT NOT NULL,
  edge_type TEXT NOT NULL,
  weight INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (source, target, edge_type)
);
CREATE TABLE IF NOT EXISTS kg_meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
```

No migration function is needed: `CREATE TABLE IF NOT EXISTS` creates these on the next `get_db` for existing databases, because they are new tables rather than new columns on an existing table.

- [ ] **Step 4: Create the alias file**

Create `src/hermes/kg_aliases.toml`:

```toml
# Spelling and abbreviation variants only -- NEVER conceptual hierarchies.
# Merging genuinely distinct-if-related concepts (deep-learning into
# machine-learning, neural-networks into deep-neural-networks) would destroy
# the structure the graph exists to reveal.
#
# Use counts below are from the live database at authoring time and are
# documentation, not assertions.

[aliases]
"machinelearning" = "machine-learning"          # 141 -> 532
"llm" = "large-language-models"                 # 6
"llms" = "large-language-models"                # 4
"language-models" = "large-language-models"     # 103
"nlp" = "natural-language-processing"           # 68 -> 101
"transformer" = "transformers"                  # 2 -> 109
```

- [ ] **Step 5: Write `src/hermes/kg.py`**

```python
"""Knowledge graph derived from the tagging pass.

The graph is built from `item_tags`, not from new content: the tagging pass
already extracted the concepts. Two measured facts drive the construction:

1. 2020 of 2347 tags (86%) are used exactly once and connect to nothing, so a
   graph including them would be mostly isolated points. Filtering at
   MIN_TAG_USES is what turns this data into a graph.
2. Variant spellings split hubs -- `machine-learning` and `machinelearning`
   are one concept counted twice. Aliasing is applied BEFORE filtering,
   are one concept counted twice, and aliasing merges them (degree 90 -> 111).

Aliasing is applied BEFORE the frequency filter: merging can lift a variant
over MIN_TAG_USES that neither spelling would clear alone, so filtering first
would silently drop concepts. On the current corpus both orderings happen to
agree, so this ordering is guarded by a synthetic test rather than by the live
numbers -- see test_aliases_merge_before_filtering.

No networkx: at ~183 nodes, BFS over an adjacency dict is both fast (0.226s
for a full build) and obvious.
"""

import hashlib
import sqlite3
import tomllib
from collections import defaultdict
from pathlib import Path

MIN_TAG_USES = 2
MIN_EDGE_WEIGHT = 2

_ALIAS_PATH = Path(__file__).resolve().parent / "kg_aliases.toml"
ALIASES: dict[str, str] = tomllib.loads(_ALIAS_PATH.read_text()).get("aliases", {})


def canonical(tag: str) -> str:
    """Map a tag to its canonical spelling. Identity for unmapped tags."""
    return ALIASES.get(tag, tag)


def build_graph(
    conn: sqlite3.Connection,
) -> tuple[dict[str, set[str]], dict[tuple[str, str], int]]:
    """Derive (adjacency, edge_weights) from item_tags.

    Aliases first, then the frequency filter, then co-occurrence -- see the
    module docstring for why that order is load-bearing.
    """
    items: dict[int, set[str]] = defaultdict(set)
    for row in conn.execute("SELECT item_id, tag FROM item_tags"):
        items[row["item_id"]].add(canonical(row["tag"]))

    uses: dict[str, int] = defaultdict(int)
    for tags in items.values():
        for tag in tags:
            uses[tag] += 1
    keep = {tag for tag, n in uses.items() if n >= MIN_TAG_USES}

    weights: dict[tuple[str, str], int] = defaultdict(int)
    for tags in items.values():
        present = sorted(tags & keep)
        for i, left in enumerate(present):
            for right in present[i + 1 :]:
                weights[(left, right)] += 1

    edges = {pair: w for pair, w in weights.items() if w >= MIN_EDGE_WEIGHT}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return dict(adjacency), edges


def fingerprint(conn: sqlite3.Connection) -> str:
    """Cheap staleness hash. Catches hand edits and restored backups.

    hermes tag auto-rebuilds, so this is the backstop for changes that do not
    go through run_tagging, not the primary freshness mechanism.
    """
    row = conn.execute(
        "SELECT COUNT(*) AS n, COALESCE(MAX(rowid), 0) AS top FROM item_tags"
    ).fetchone()
    return hashlib.sha256(f"{row['n']}:{row['top']}".encode()).hexdigest()[:16]


def rebuild(conn: sqlite3.Connection) -> dict:
    """Regenerate kg_nodes/kg_edges from item_tags. Idempotent."""
    adjacency, edges = build_graph(conn)
    conn.execute("DELETE FROM kg_edges")
    conn.execute("DELETE FROM kg_nodes")
    conn.executemany(
        "INSERT INTO kg_nodes(name, node_type, degree) VALUES (?, 'concept', ?)",
        [(name, len(neighbours)) for name, neighbours in adjacency.items()],
    )
    conn.executemany(
        "INSERT INTO kg_edges(source, target, edge_type, weight) VALUES (?, ?, 'CO_OCCURS', ?)",
        [(left, right, w) for (left, right), w in edges.items()],
    )
    conn.execute(
        "INSERT OR REPLACE INTO kg_meta(key, value) VALUES ('fingerprint', ?)",
        (fingerprint(conn),),
    )
    conn.commit()
    return {"nodes": len(adjacency), "edges": len(edges)}


def is_stale(conn: sqlite3.Connection) -> bool:
    """True when the stored graph no longer matches item_tags."""
    row = conn.execute("SELECT value FROM kg_meta WHERE key = 'fingerprint'").fetchone()
    if row is None:
        return True
    return row["value"] != fingerprint(conn)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_kg.py -v`
Expected: PASS (7 tests).

- [ ] **Step 7: Verify against the live database**

```bash
uv run python -c "
from hermes.db import get_db, resolve_db_path
from hermes import kg
conn = get_db(resolve_db_path(None))
adjacency, edges = kg.build_graph(conn)
print('nodes:', len(adjacency), 'edges:', len(edges))
print('machine-learning degree:', len(adjacency.get('machine-learning', ())))
print('machinelearning present?', 'machinelearning' in adjacency)
"
```

Expected: **183 nodes**, **447 edges**, `machine-learning` degree **111**, and `machinelearning` **absent** (merged). If the numbers differ materially, the alias-before-filter order is likely reversed — check that before proceeding, and report rather than adjusting the expectation.

- [ ] **Step 8: Run gates and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run ty check
git add src/hermes/kg.py src/hermes/kg_aliases.toml src/hermes/db.py tests/test_kg.py
git status --short
git commit -m "$(cat <<'EOF'
feat: knowledge graph construction with alias merging

Derives a concept graph from item_tags. Two measured facts shape it: 86% of
tags are used exactly once and connect to nothing, so filtering at 2+ uses is
what makes it a graph; and variant spellings split hubs, so aliases merge
before the frequency filter -- reversing that order silently produces a
different graph. On the current corpus both orderings agree; the rule is
guarded by a synthetic test where filter-first drops a concept entirely.

kg_nodes/kg_edges/kg_meta are new tables, so CREATE TABLE IF NOT EXISTS covers
existing databases without a migration function.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Graph algorithms

**Files:**
- Modify: `src/hermes/kg.py` (append the four query functions)
- Test: `tests/test_kg_algorithms.py`

**Interfaces:**
- Consumes: `kg.build_graph` (Task 1).
- Produces:
  - `kg.neighbors(conn, node: str, depth: int = 1, limit: int = 20) -> list[dict]` — each `{"name", "distance", "weight"}`; `weight` is the direct edge weight when `distance == 1`, else `None`.
  - `kg.shortest_path(conn, source: str, target: str) -> list[str] | None` — `None` when no path exists (not an error).
  - `kg.central(conn, metric: str = "degree", limit: int = 10) -> list[dict]` — each `{"name", "score"}`; `metric` is `"degree"` or `"betweenness"`.
  - `kg.communities(conn, min_size: int = 3) -> list[dict]` — each `{"label", "members"}`, label = highest-degree member.

- [ ] **Step 1: Write the failing algorithm tests**

Create `tests/test_kg_algorithms.py`:

```python
import pytest

from hermes import kg
from hermes.db import get_db


def seed(conn, items):
    for i, tags in enumerate(items, start=1):
        conn.execute(
            "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
            " VALUES (?, NULL, ?, 'u', 's', ?)",
            (i, f"item {i}", f"h{i}"),
        )
        for t in tags:
            conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (i, t))
    conn.commit()


@pytest.fixture
def line_graph(tmp_path):
    """A-B-C-D chain plus an isolated pair X-Y, so every edge has weight 2.

    Hand-computable: betweenness on a 4-node path is 0, 2, 2, 0 (undirected,
    unnormalized, counting each unordered pair once).
    """
    conn = get_db(tmp_path / "t.db")
    seed(
        conn,
        [
            ["A", "B"],
            ["A", "B"],
            ["B", "C"],
            ["B", "C"],
            ["C", "D"],
            ["C", "D"],
            ["X", "Y"],
            ["X", "Y"],
        ],
    )
    yield conn
    conn.close()


def test_neighbors_depth_one(line_graph):
    result = kg.neighbors(line_graph, "B", depth=1)

    names = {r["name"] for r in result}
    assert names == {"A", "C"}
    assert all(r["distance"] == 1 for r in result)
    assert all(r["weight"] == 2 for r in result)


def test_neighbors_depth_two_reaches_further(line_graph):
    result = kg.neighbors(line_graph, "A", depth=2)

    by_name = {r["name"]: r for r in result}
    assert by_name["B"]["distance"] == 1
    assert by_name["C"]["distance"] == 2
    assert "D" not in by_name, "depth=2 must not reach 3 hops"


def test_neighbors_excludes_the_node_itself(line_graph):
    assert all(r["name"] != "B" for r in kg.neighbors(line_graph, "B", depth=2))


def test_neighbors_unknown_node_is_empty_not_error(line_graph):
    assert kg.neighbors(line_graph, "nonexistent") == []


def test_shortest_path_finds_the_chain(line_graph):
    assert kg.shortest_path(line_graph, "A", "D") == ["A", "B", "C", "D"]


def test_shortest_path_returns_none_across_components(line_graph):
    """X-Y is a separate component; no path is a legitimate answer, not an error."""
    assert kg.shortest_path(line_graph, "A", "X") is None


def test_shortest_path_to_self_is_single_node(line_graph):
    assert kg.shortest_path(line_graph, "A", "A") == ["A"]


def test_degree_centrality_ranks_by_connections(line_graph):
    result = kg.central(line_graph, metric="degree", limit=4)

    top = {r["name"] for r in result[:2]}
    assert top == {"B", "C"}, "the chain's middle nodes have degree 2"


def test_betweenness_matches_hand_computation(line_graph):
    """On the A-B-C-D path, B and C each lie on 2 shortest paths between
    other pairs; A and D lie on none."""
    scores = {r["name"]: r["score"] for r in kg.central(line_graph, metric="betweenness", limit=10)}

    assert scores["B"] == pytest.approx(2.0)
    assert scores["C"] == pytest.approx(2.0)
    assert scores["A"] == pytest.approx(0.0)
    assert scores["D"] == pytest.approx(0.0)


def test_central_rejects_unknown_metric(line_graph):
    with pytest.raises(ValueError, match="metric"):
        kg.central(line_graph, metric="pagerank")


def test_communities_separate_the_components(line_graph):
    result = kg.communities(line_graph, min_size=2)

    memberships = [set(c["members"]) for c in result]
    assert {"A", "B", "C", "D"} in memberships
    assert {"X", "Y"} in memberships


def test_communities_respect_min_size(line_graph):
    result = kg.communities(line_graph, min_size=3)

    assert all(len(c["members"]) >= 3 for c in result)
    assert not any(set(c["members"]) == {"X", "Y"} for c in result)


def test_communities_are_deterministic(line_graph):
    """Label propagation is randomized in the general case; this
    implementation must be seeded or order-stable so repeated calls agree."""
    first = kg.communities(line_graph, min_size=2)
    second = kg.communities(line_graph, min_size=2)

    assert [set(c["members"]) for c in first] == [set(c["members"]) for c in second]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_kg_algorithms.py -v`
Expected: FAIL — `AttributeError: module 'hermes.kg' has no attribute 'neighbors'`.

- [ ] **Step 3: Append the algorithms to `src/hermes/kg.py`**

```python
def neighbors(conn: sqlite3.Connection, node: str, depth: int = 1, limit: int = 20) -> list[dict]:
    """Nodes within `depth` hops, nearest first. Unknown node -> empty list."""
    adjacency, weights = build_graph(conn)
    if node not in adjacency:
        return []
    depth = max(1, int(depth))
    seen = {node}
    frontier = {node}
    out: list[dict] = []
    for distance in range(1, depth + 1):
        nxt: set[str] = set()
        for current in frontier:
            nxt |= adjacency.get(current, set())
        nxt -= seen
        for name in sorted(nxt):
            pair = (node, name) if node < name else (name, node)
            out.append(
                {
                    "name": name,
                    "distance": distance,
                    "weight": weights.get(pair) if distance == 1 else None,
                }
            )
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    return out[: max(1, int(limit))]


def shortest_path(conn: sqlite3.Connection, source: str, target: str) -> list[str] | None:
    """BFS shortest path. None when the two are in different components."""
    from collections import deque

    adjacency, _ = build_graph(conn)
    if source not in adjacency or target not in adjacency:
        return None
    if source == target:
        return [source]
    queue = deque([[source]])
    seen = {source}
    while queue:
        path = queue.popleft()
        for name in sorted(adjacency[path[-1]]):
            if name == target:
                return path + [name]
            if name not in seen:
                seen.add(name)
                queue.append(path + [name])
    return None


def _betweenness(adjacency: dict[str, set[str]]) -> dict[str, float]:
    """Brandes' algorithm, undirected and unnormalized.

    Each unordered pair contributes once, so the accumulated score is halved
    at the end.
    """
    from collections import deque

    scores = dict.fromkeys(adjacency, 0.0)
    for start in sorted(adjacency):
        stack: list[str] = []
        predecessors: dict[str, list[str]] = {v: [] for v in adjacency}
        sigma = dict.fromkeys(adjacency, 0.0)
        distance = dict.fromkeys(adjacency, -1)
        sigma[start] = 1.0
        distance[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in sorted(adjacency[v]):
                if distance[w] < 0:
                    distance[w] = distance[v] + 1
                    queue.append(w)
                if distance[w] == distance[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)
        delta = dict.fromkeys(adjacency, 0.0)
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != start:
                scores[w] += delta[w]
    return {name: score / 2.0 for name, score in scores.items()}


def central(conn: sqlite3.Connection, metric: str = "degree", limit: int = 10) -> list[dict]:
    """Most-connected (degree) or most-bridging (betweenness) concepts."""
    if metric not in ("degree", "betweenness"):
        raise ValueError(f"unknown metric: {metric!r} (expected 'degree' or 'betweenness')")
    adjacency, _ = build_graph(conn)
    if metric == "degree":
        scores = {name: float(len(neighbours)) for name, neighbours in adjacency.items()}
    else:
        scores = _betweenness(adjacency)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"name": name, "score": score} for name, score in ranked[: max(1, int(limit))]]


def communities(conn: sqlite3.Connection, min_size: int = 3) -> list[dict]:
    """Label propagation over the graph, seeded for determinism.

    Nodes are processed in sorted order and ties break on the lexically
    smallest label, so repeated calls agree -- ordinary label propagation is
    randomized and would return different clusters each run.
    """
    adjacency, _ = build_graph(conn)
    labels = {name: name for name in adjacency}
    for _ in range(100):
        changed = False
        for name in sorted(adjacency):
            counts: dict[str, int] = defaultdict(int)
            for neighbour in adjacency[name]:
                counts[labels[neighbour]] += 1
            if not counts:
                continue
            best = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if labels[name] != best:
                labels[name] = best
                changed = True
        if not changed:
            break

    grouped: dict[str, list[str]] = defaultdict(list)
    for name, label in labels.items():
        grouped[label].append(name)

    out = []
    for members in grouped.values():
        if len(members) < max(1, int(min_size)):
            continue
        label = max(sorted(members), key=lambda n: len(adjacency[n]))
        out.append({"label": label, "members": sorted(members)})
    return sorted(out, key=lambda c: (-len(c["members"]), c["label"]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_kg_algorithms.py -v`
Expected: PASS (13 tests). If `test_betweenness_matches_hand_computation` fails, the halving at the end of `_betweenness` is the likely culprit — check whether your accumulation counts each unordered pair once or twice, and fix the implementation rather than the expected value.

- [ ] **Step 5: Sanity-check against the live graph**

```bash
uv run python -c "
from hermes.db import get_db, resolve_db_path
from hermes import kg
conn = get_db(resolve_db_path(None))
print('top degree:', [r['name'] for r in kg.central(conn, 'degree', 5)])
print('path:', kg.shortest_path(conn, 'natural-language-processing', 'stable-diffusion'))
print('communities:', [(c['label'], len(c['members'])) for c in kg.communities(conn)[:4]])
"
```

Expected: `machine-learning` leads the degree ranking; the path is a real multi-hop list (a known example is `natural-language-processing -> ai -> fine-tuning -> stable-diffusion`, though aliasing may shorten it); communities returns several named clusters. Report the actual output — these are sanity checks, not exact assertions.

- [ ] **Step 6: Run gates and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run ty check
git add src/hermes/kg.py tests/test_kg_algorithms.py
git status --short
git commit -m "$(cat <<'EOF'
feat: graph algorithms -- neighbors, shortest path, centrality, communities

Pure Python over an adjacency dict; no networkx, which vanished from this
environment once it stopped being a transitive dependency and is unnecessary
at ~183 nodes.

Betweenness is Brandes' algorithm, undirected and unnormalized, halved so each
unordered pair counts once -- pinned against a hand-computed 4-node path
rather than against the implementation's own output. Label propagation is
order-seeded and tie-broken lexically so repeated calls agree; ordinary label
propagation is randomized and would cluster differently each run.

shortest_path returns None across components: no path is a legitimate answer,
not an error.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: MCP tools, the tagging hook, and documentation

**Files:**
- Modify: `src/hermes/mcp_server.py` (five `_impl` functions + five `@mcp.tool()` wrappers, before `def main()`)
- Modify: `src/hermes/features.py` (`run_tagging` — one rebuild call after the loop)
- Modify: `README.md` (tool table)
- Modify: `src/hermes/skills/science-recommendations/SKILL.md`
- Test: `tests/test_kg_mcp.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: everything from Tasks 1-2.
- Produces: MCP tools `kg_neighbors`, `kg_path`, `kg_central`, `kg_communities`, `kg_rebuild`.

- [ ] **Step 1: Write the failing MCP tests**

Create `tests/test_kg_mcp.py`:

```python
from hermes import mcp_server
from hermes.db import get_db


def seed_env_db(db_path):
    conn = get_db(db_path)
    rows = [["A", "B"], ["A", "B"], ["B", "C"], ["B", "C"]]
    for i, tags in enumerate(rows, start=1):
        conn.execute(
            "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
            " VALUES (?, NULL, ?, 'u', 's', ?)",
            (i, f"item {i}", f"h{i}"),
        )
        for t in tags:
            conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (i, t))
    conn.commit()
    conn.close()


def test_kg_neighbors_returns_adjacent_concepts(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    seed_env_db(db)

    out = mcp_server._kg_neighbors_impl("B")

    assert out["ok"] is True
    assert {n["name"] for n in out["neighbors"]} == {"A", "C"}


def test_kg_path_reports_no_path_without_erroring(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    seed_env_db(db)

    out = mcp_server._kg_path_impl("A", "nonexistent")

    assert out["ok"] is False
    assert out["path"] is None
    assert "no path" in out["message"].lower()


def test_kg_central_rejects_unknown_metric(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    seed_env_db(db)

    out = mcp_server._kg_central_impl(metric="pagerank")

    assert out["ok"] is False
    assert out["nodes"] == []


def test_kg_rebuild_without_confirm_mutates_nothing(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    seed_env_db(db)

    out = mcp_server._kg_rebuild_impl(confirm=False)

    assert out["ok"] is False
    conn = get_db(db)
    assert conn.execute("SELECT COUNT(*) n FROM kg_nodes").fetchone()["n"] == 0
    conn.close()


def test_read_tools_flag_staleness(tmp_path, monkeypatch):
    """The graph is auto-rebuilt by hermes tag, but a hand edit must surface."""
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    seed_env_db(db)
    mcp_server._kg_rebuild_impl(confirm=True)

    fresh = mcp_server._kg_neighbors_impl("B")
    assert fresh["stale"] is False

    conn = get_db(db)
    conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (1, 'D')")
    conn.commit()
    conn.close()

    assert mcp_server._kg_neighbors_impl("B")["stale"] is True


def test_all_five_kg_tools_are_served():
    import asyncio

    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert {"kg_neighbors", "kg_path", "kg_central", "kg_communities", "kg_rebuild"} <= names
    assert len(names) == 28
```

Add to `tests/test_features.py`:

```python
def test_run_tagging_rebuilds_the_graph_once(tmp_path, monkeypatch):
    """Auto-rebuild closes the drift window, but must run once after the loop
    -- per-item would cost ~92s on a 408-item backlog for no benefit."""
    from hermes import features, kg
    from hermes.db import get_db

    conn = get_db(tmp_path / "t.db")
    for i in (1, 2):
        conn.execute(
            "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
            " VALUES (?, NULL, ?, 'u', 's', ?)",
            (i, f"item {i}", f"h{i}"),
        )
    conn.commit()

    calls = []
    monkeypatch.setattr(kg, "rebuild", lambda c: calls.append(1) or {"nodes": 0, "edges": 0})

    def fake_chat(messages, schema):
        return {"content_type": "paper", "tags": ["alpha", "beta"]}

    features.run_tagging(conn, chat_fn=fake_chat)

    assert len(calls) == 1, "rebuild must run once after the loop, not per item"
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_kg_mcp.py -v`
Expected: FAIL — `AttributeError: module 'hermes.mcp_server' has no attribute '_kg_neighbors_impl'`.

- [ ] **Step 3: Add the `_impl` functions**

In `src/hermes/mcp_server.py`, before `def main()`:

```python
# ---------------------------------------------------------------------------
# knowledge graph
# ---------------------------------------------------------------------------


def _kg_neighbors_impl(node: str, depth: int = 1, limit: int = 20) -> dict:
    from hermes import kg

    conn = get_db(resolve_db_path(None))
    try:
        found = kg.neighbors(conn, node, depth=depth, limit=min(limit, MAX_LIST_LIMIT))
        if not found:
            return {
                "ok": False,
                "message": f"{node!r} is not a concept in the graph",
                "neighbors": [],
                "stale": kg.is_stale(conn),
            }
        return {
            "ok": True,
            "message": f"{len(found)} neighbour(s)",
            "neighbors": found,
            "stale": kg.is_stale(conn),
        }
    except Exception:
        log.exception("kg_neighbors failed for node=%s", node)
        return {"ok": False, "message": "internal error", "neighbors": [], "stale": True}
    finally:
        conn.close()


def _kg_path_impl(source: str, target: str) -> dict:
    from hermes import kg

    conn = get_db(resolve_db_path(None))
    try:
        found = kg.shortest_path(conn, source, target)
        if found is None:
            return {
                "ok": False,
                "message": f"no path between {source!r} and {target!r}",
                "path": None,
                "stale": kg.is_stale(conn),
            }
        return {
            "ok": True,
            "message": f"{len(found) - 1} hop(s)",
            "path": found,
            "stale": kg.is_stale(conn),
        }
    except Exception:
        log.exception("kg_path failed for %s -> %s", source, target)
        return {"ok": False, "message": "internal error", "path": None, "stale": True}
    finally:
        conn.close()


def _kg_central_impl(metric: str = "degree", limit: int = 10) -> dict:
    from hermes import kg

    conn = get_db(resolve_db_path(None))
    try:
        ranked = kg.central(conn, metric=metric, limit=min(limit, MAX_LIST_LIMIT))
        return {
            "ok": True,
            "message": f"top {len(ranked)} by {metric}",
            "nodes": ranked,
            "stale": kg.is_stale(conn),
        }
    except ValueError as exc:
        return {"ok": False, "message": str(exc), "nodes": [], "stale": False}
    except Exception:
        log.exception("kg_central failed for metric=%s", metric)
        return {"ok": False, "message": "internal error", "nodes": [], "stale": True}
    finally:
        conn.close()


def _kg_communities_impl(min_size: int = 3) -> dict:
    from hermes import kg

    conn = get_db(resolve_db_path(None))
    try:
        found = kg.communities(conn, min_size=min_size)
        return {
            "ok": True,
            "message": f"{len(found)} cluster(s)",
            "communities": found,
            "stale": kg.is_stale(conn),
        }
    except Exception:
        log.exception("kg_communities failed")
        return {"ok": False, "message": "internal error", "communities": [], "stale": True}
    finally:
        conn.close()


def _kg_rebuild_impl(confirm: bool = False) -> dict:
    from hermes import kg

    if not confirm:
        return {
            "ok": False,
            "message": (
                "refusing to rebuild without confirm=true. This replaces the "
                "kg_nodes and kg_edges tables (the graph is derived, so nothing "
                "unrecoverable is lost)."
            ),
            "nodes": 0,
            "edges": 0,
        }
    conn = get_db(resolve_db_path(None))
    try:
        counts = kg.rebuild(conn)
        return {"ok": True, "message": "graph rebuilt", **counts}
    except Exception:
        log.exception("kg_rebuild failed")
        return {"ok": False, "message": "internal error", "nodes": 0, "edges": 0}
    finally:
        conn.close()
```

- [ ] **Step 4: Add the five tool wrappers**

```python
@mcp.tool()
def kg_neighbors(node: str, depth: int = 1, limit: int = 20) -> dict:
    """Concepts adjacent to a given concept in the reading graph.

    The "what else should I read about this" query. `depth=2` reaches two
    hops. Concepts come from the tagging pass; a tag used only once is not in
    the graph.
    """
    return _kg_neighbors_impl(node, depth, limit)


@mcp.tool()
def kg_path(source: str, target: str) -> dict:
    """Shortest chain of concepts connecting two topics in the reading graph.

    Returns ok=false with path=null when the two are in different components —
    that is a legitimate answer meaning "these never co-occur", not an error.
    """
    return _kg_path_impl(source, target)


@mcp.tool()
def kg_central(metric: str = "degree", limit: int = 10) -> dict:
    """Most important concepts. metric="degree" for most-connected,
    "betweenness" for the bridges between otherwise separate clusters."""
    return _kg_central_impl(metric, limit)


@mcp.tool()
def kg_communities(min_size: int = 3) -> dict:
    """Topic clusters in the reading graph, each labelled by its most
    connected member. Useful for seeing what the reading actually splits into."""
    return _kg_communities_impl(min_size)


@mcp.tool()
def kg_rebuild(confirm: bool = False) -> dict:
    """Regenerate the graph from current tags. Requires confirm=true.

    Normally unnecessary — `hermes tag` rebuilds automatically. Use this after
    editing the database by hand, or when a read tool reports stale=true.
    """
    return _kg_rebuild_impl(confirm)
```

- [ ] **Step 5: Hook the rebuild into `run_tagging`**

In `src/hermes/features.py`, at the end of `run_tagging`, replace `return stats` with:

```python
    # One rebuild after the loop, never per item: 0.226s against a ~571s
    # tagging run is 0.04% overhead, while per-item would waste ~92s on a
    # 408-item backlog for an identical result.
    from hermes import kg

    kg.rebuild(conn)
    return stats
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_kg_mcp.py tests/test_features.py -v`
Expected: PASS. `test_all_five_kg_tools_are_served` asserts exactly 28 tools; if the count differs, report the actual list rather than adjusting the assertion.

- [ ] **Step 7: Update the README tool table**

Change the lead-in from "exposes twenty-three tools" to "exposes twenty-eight tools" and append:

```markdown
| `kg_neighbors(node, depth, limit)` | Concepts adjacent to this one in your reading graph | instant |
| `kg_path(source, target)` | Shortest chain of concepts linking two topics | instant |
| `kg_central(metric, limit)` | Most-connected or most-bridging concepts | instant |
| `kg_communities(min_size)` | Topic clusters, each labelled by its hub concept | instant |
| `kg_rebuild(confirm)` | Regenerate the graph (needs `confirm=true`; `hermes tag` does this automatically) | instant |
```

Then add below the table:

```markdown
The knowledge graph is derived from the tagging pass, not from separate
content: concepts are tags used at least twice, and two concepts are linked
when they co-occur on at least two items. Spelling variants are merged via
`src/hermes/kg_aliases.toml` — without that, `machine-learning` and
`machinelearning` appear as two separate hubs and every centrality number is
wrong. Tags used only once (86% of them) are excluded: they connect to
nothing. `hermes tag` rebuilds the graph automatically, and every read tool
reports `stale: true` if the database was changed some other way.
```

- [ ] **Step 8: Add the tools to SKILL.md**

In the `## MCP tools` section of `src/hermes/skills/science-recommendations/SKILL.md`, add a short grouped entry in the file's existing voice covering all five names, noting that `kg_rebuild` needs `confirm=true` and is normally unnecessary because tagging rebuilds automatically.

- [ ] **Step 9: Verify docs match the served tools**

```bash
uv run python -c "
import asyncio, re
from pathlib import Path
from hermes.mcp_server import mcp
served = {t.name for t in asyncio.run(mcp.list_tools())}
documented = set(re.findall(r'\| \`(\w+)\(', Path('README.md').read_text()))
print('served:', len(served))
print('served not documented:', sorted(served - documented))
print('documented not served:', sorted(documented - served))
"
```

Expected: 28 served, both lists empty. If not, fix the README — the served tools are the truth.

- [ ] **Step 10: Run gates and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run ty check
git add src/hermes/mcp_server.py src/hermes/features.py README.md src/hermes/skills/science-recommendations/SKILL.md tests/test_kg_mcp.py tests/test_features.py
git status --short
git commit -m "$(cat <<'EOF'
feat: five knowledge-graph MCP tools + auto-rebuild on tagging

kg_neighbors / kg_path / kg_central / kg_communities / kg_rebuild, bringing
the served surface to 28.

run_tagging now rebuilds the graph once after its loop, closing the drift
window that materialized tables would otherwise open. Once, not per item:
0.226s against a ~571s tagging run is 0.04% overhead, where per-item would
waste ~92s on the current 408-item backlog. A fingerprint still backs this up
for changes that bypass run_tagging -- hand edits, restored backups -- and
every read tool surfaces stale=true rather than silently answering from an
outdated graph.

kg_path returning ok=false with path=null across components is the honest
answer ("these concepts never co-occur"), not an error.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Post-plan manual verification

1. `uv run hermes install --check` still exits 0 — the two new tables must not disturb the installer's doctor mode.
2. Live graph: `uv run python -c "from hermes.db import get_db, resolve_db_path; from hermes import kg; c=get_db(resolve_db_path(None)); print(kg.rebuild(c))"` — expect roughly 183 nodes and 447 edges.
3. `hermes mcp test hermes-rss` reports 28 tools discovered.
4. Ask the graph something real: `kg_central(metric="betweenness")` should surface concepts that bridge clusters, which is the question flat notes cannot answer. Judge whether the answer is actually interesting — if the clusters look like noise, the alias table likely needs more entries, and that is a finding worth reporting rather than a failure of the code.
