"""Knowledge graph derived from the tagging pass.

The graph is built from `item_tags`, not from new content: the tagging pass
already extracted the concepts. Two measured facts drive the construction:

1. 2020 of 2347 tags (86%) are used exactly once and connect to nothing, so a
   graph including them would be mostly isolated points. Filtering at
   MIN_TAG_USES is what turns this data into a graph.
2. Variant spellings split hubs -- `machine-learning` and `machinelearning`
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


def neighbors(conn: sqlite3.Connection, node: str, limit: int = 20) -> list[dict]:
    """Direct neighbours of `node`, strongest edge first. Unknown -> empty list.

    Every returned row is adjacent to `node`, so `distance` is always 1 and
    `weight` is always the real co-occurrence weight of that edge. Ranking is
    by edge weight descending, name as tie-break, then truncated to `limit`.

    This deliberately does NOT walk further than one hop. A `depth` parameter
    existed and was removed: multi-ring traversal produced four successive
    defects, the last of which reported genuine direct neighbours as
    `distance=2, weight=None` whenever a small `limit` cut them from ring 1
    and the walk re-reached them through a survivor. Every honest fix needed
    a further rule about starvation, ranking proxies, or re-entry, and each
    rule bought a new edge case. `kg_path` answers multi-hop questions
    exactly and cheaply; a truncated breadth-first walk answered them
    approximately and, as shipped, sometimes wrongly.
    """
    adjacency, weights = build_graph(conn)
    if node not in adjacency:
        return []

    def _edge_weight(name: str) -> int:
        pair = (node, name) if node < name else (name, node)
        return weights.get(pair, 0)

    ranked = sorted(adjacency[node], key=lambda name: (-_edge_weight(name), name))
    return [
        {"name": name, "distance": 1, "weight": _edge_weight(name)}
        for name in ranked[: max(1, int(limit))]
    ]


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
    """Topic clusters by modularity (Louvain phase 1), seeded for determinism.

    Label propagation was used here first and produced ONE cluster holding 95%
    of the graph. That was not a corpus artefact: 331 of 347 nodes really are a
    single connected component (this reading is all machine learning, so every
    concept links to every other eventually), and label propagation has nothing
    to stop a dense hub's label spreading over an entire component. Weighting
    its votes by edge weight made it strictly worse -- the hub's edges are the
    heaviest, so weighting fed the hub. Raising MIN_EDGE_WEIGHT did not help
    either: at weight >= 3 the graph is still one component, just with 156 of
    347 nodes left.

    Modularity is what actually separates them, because its `- k_i * k_c / 2m`
    term penalises joining a community that is already large: a node only joins
    if its links there beat what random chance would predict. On the same graph
    this returns 13 communities with the largest at 32% of nodes, and they are
    coherent -- quantum-chemistry pulls together electronic-structure,
    solid-state-physics and statistical-mechanics; security groups cryptography
    with prompt-injection.

    Determinism (which the tools depend on) comes from iterating nodes in
    sorted order, breaking equal modularity gains on the lexically smaller
    community, and never using randomness. Only phase 1 of Louvain runs -- no
    graph coarsening -- since at this scale it already resolves topics and the
    result stays a direct node->community map.
    """
    adjacency, weights = build_graph(conn)

    def edge_weight(a: str, b: str) -> int:
        return weights.get((a, b) if a < b else (b, a), 0)

    two_m = sum(weights.values()) * 2.0
    if not adjacency or two_m == 0:
        return []

    strength = {n: sum(edge_weight(n, x) for x in adjacency[n]) for n in adjacency}
    labels = {n: n for n in adjacency}

    for _ in range(20):
        changed = False
        totals: dict[str, float] = defaultdict(float)
        for n in adjacency:
            totals[labels[n]] += strength[n]
        for name in sorted(adjacency):
            current = labels[name]
            totals[current] -= strength[name]
            links: dict[str, float] = defaultdict(float)
            for neighbour in adjacency[name]:
                links[labels[neighbour]] += edge_weight(name, neighbour)
            best = current
            best_gain = links.get(current, 0.0) - totals[current] * strength[name] / two_m
            for candidate, link in sorted(links.items()):
                gain = link - totals[candidate] * strength[name] / two_m
                if gain > best_gain + 1e-12 or (abs(gain - best_gain) < 1e-12 and candidate < best):
                    best, best_gain = candidate, gain
            totals[best] += strength[name]
            if best != current:
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


def health(conn: sqlite3.Connection) -> dict:
    """Metrics that say whether the graph is usable, not just how big it is.

    These are the numbers that caught real problems on 2026-08-11:
    `singleton_rate` showed one tagging model minting 85% one-off tags where
    another minted 74%, and `largest_community_pct` at 95% exposed the
    clustering bug that made kg_communities useless.
    """
    adjacency, edges = build_graph(conn)
    degrees = sorted((len(v) for v in adjacency.values()), reverse=True)
    tags = conn.execute("SELECT tag, COUNT(*) n FROM item_tags GROUP BY tag").fetchall()
    singles = sum(1 for r in tags if r["n"] == 1)
    nodes = len(adjacency)
    groups = communities(conn, min_size=3)
    largest = max((len(c["members"]) for c in groups), default=0)

    def pct(part: float, whole: float) -> float:
        return round(100 * part / whole, 1) if whole else 0.0

    return {
        "nodes": nodes,
        "edges": len(edges),
        "distinct_tags": len(tags),
        # tags used once never clear MIN_TAG_USES -- vocabulary the graph discards
        "singleton_rate": pct(singles, len(tags)),
        "degree_1_pct": pct(sum(1 for d in degrees if d == 1), nodes),
        "median_degree": degrees[len(degrees) // 2] if degrees else 0,
        "max_degree": degrees[0] if degrees else 0,
        # one concept's share of all edge endpoints; a high value means a hub
        # that no clustering can separate around
        "hub_dominance_pct": pct(degrees[0] if degrees else 0, sum(degrees)),
        "communities": len(groups),
        "largest_community_pct": pct(largest, nodes),
    }
