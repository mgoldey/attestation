"""Knowledge graph derived from the tagging pass.

The graph is built from `item_tags`, not from new content: the tagging pass
already extracted the concepts. Two measured facts drive the construction:

1. 2736 of 3879 canonical tags (70%) are used exactly once and connect to nothing, so a
   graph including them would be mostly isolated points. Filtering at
   MIN_TAG_USES is what turns this data into a graph.
2. Variant spellings split hubs -- `machine-learning` and `machinelearning`
   are one concept counted twice, and aliasing merges them (degree 90 -> 111).

Aliasing is applied BEFORE the frequency filter: merging can lift a variant
over MIN_TAG_USES that neither spelling would clear alone, so filtering first
would silently drop concepts. On the current corpus both orderings happen to
agree, so this ordering is guarded by a synthetic test rather than by the live
numbers -- see test_aliases_merge_before_filtering.

No networkx: at ~711 nodes, BFS over an adjacency dict is both fast (0.226s
for a full build) and obvious.
"""

import sqlite3
import tomllib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

MIN_TAG_USES = 2
MIN_EDGE_WEIGHT = 2

_ALIAS_PATH = Path(__file__).resolve().parent / "kg_aliases.toml"
_ALIAS_DOC = tomllib.loads(_ALIAS_PATH.read_text())
ALIASES: dict[str, str] = _ALIAS_DOC.get("aliases", {})

# Fold key -> the spelling to canonicalize to. Populated below from the alias
# table's targets plus [fold_canonical], so that folding never has to guess
# which of two spellings is the real one.
_FOLD_CANON: dict[str, str] = {}


# Suffixes where a trailing "s" is not a plural marker. Stripping it anyway
# is what makes an automatic rule dangerous, so each group is here for a
# reason observed on the live tag list:
#   ss/us/is/as/os/ys  -- mass, virus, analysis, gas, alloys: the "s" is stem
#   ics                -- physics, robotics, genomics, ethics, dynamics are
#                         field names, not plurals of physic/robotic/genomic
#   ews/ens            -- news, lens
#   ies                -- series and species are singular; folding -ies to -y
#                         also mints non-words (series -> sery)
_NOT_PLURAL = ("ss", "us", "is", "as", "os", "ys", "ics", "ews", "ens", "ies")


def _singular(word: str) -> str:
    """Strip one trailing plural "s". Conservative by construction."""
    if len(word) < 4 or not word.endswith("s") or word.endswith(_NOT_PLURAL):
        return word
    return word[:-1]


def _fold(tag: str) -> str:
    """The equivalence key: separators removed, last word singularized.

    Only these two axes fold. Both are spelling, never meaning: `rna` and
    `dna` differ by a letter that is not a separator or a plural, so no rule
    here can reach them. Stemming or edit-distance would, which is why
    neither is used.
    """
    return _singular(tag.replace("_", "-").replace("-", ""))


# Every alias target is authoritative for its own fold key, so `hugging-face`
# beats `huggingface` and `large-language-models` beats `languagemodels`
# without the corpus being consulted. [fold_canonical] in the TOML names a
# spelling for any remaining key the default would get wrong.
for _target in set(ALIASES.values()):
    _FOLD_CANON[_fold(_target)] = _target
for _key, _spelling in _ALIAS_DOC.get("fold_canonical", {}).items():
    _FOLD_CANON[_fold(_key)] = _spelling


def resolve_query(text: str, adjacency) -> str:
    """A user's phrasing turned into a node name the graph actually holds.

    Distinct from `canonical`, deliberately. canonical() maps a STORED tag to
    its canonical spelling and is a pure function used when building the
    graph; this maps what a person or a model TYPED to a node, and is used only
    at lookup. Merging them would let a query's spelling rules leak into the
    graph's node names.

    Measured: `kg.neighbors` refused 'reinforcement learning' and
    'Reinforcement Learning' while 'reinforcement-learning' returned 95
    neighbours. Both refusals are correct about the stored name and useless to
    the caller, and gemma4:e2b hit exactly this -- it called
    kg_path(source="transformers", target="reinforcement learning"), because
    spaces are how the question was asked.

    Falls back to the input unchanged when nothing matches, so an unknown
    concept is still refused rather than silently resolved to something near.
    """
    if text in adjacency:
        return text
    direct = canonical(text)
    if direct in adjacency:
        return direct
    # Spaces and case are how a person writes a concept; the graph stores
    # lowercase and hyphenated. Try that shape before giving up.
    spelled = canonical("-".join(text.lower().split()))
    if spelled in adjacency:
        return spelled
    # Last resort: match on the fold key, which already ignores separators and
    # trailing plurals, so "Large Language Model" reaches large-language-models.
    key = _fold("-".join(text.lower().split()))
    for node in adjacency:
        if _fold(node) == key:
            return node
    return text


def canonical(tag: str) -> str:
    """Map a tag to its canonical spelling. Identity for unmapped tags.

    Two layers, hand-curated first. `ALIASES` is consulted before and after
    folding, so the alias table always wins: it is the only place that can
    merge things folding cannot see (`nlp` -> `natural-language-processing`)
    and the only place that can pick a canonical spelling folding would get
    wrong (folding alone would elect `huggingface`, the more frequent
    spelling, over the `hugging-face` the table names).

    Folding then handles the open-ended cases no hand-list can enumerate:
    separator variants (`machinelearning`, `fine-tuning`/`finetuning`) and
    singular/plural pairs (`transformer`/`transformers`). On the live corpus
    this merges 79 variant pairs the table never listed.

    A tag only folds onto a spelling that is already NAMED -- an alias target
    or a [fold_canonical] entry. An unrecognised fold key is left alone rather
    than rewritten to a computed singular, because measurement showed that
    computing one is both useless and harmful: of 588 tags a computed fold
    touched on the live corpus, 506 had no merge partner at all. Those gained
    nothing and cost accuracy, renaming established tags
    (`agentic-workflows` x457, `neural-networks` x91) and minting non-words
    (`diabetes` -> `diabete`, `stochastic-processes` -> `stochastic-processe`).
    Worse, a plural is often the term of art -- `mixture-of-experts` and
    `scaling-laws` are not "one expert" or "one law".

    So folding is a merging rule, never a renaming rule: it fires only where
    a canonical spelling is already known, which keeps `canonical()` a pure
    function of the tag and leaves the graph's node names stable.
    """
    if tag in ALIASES:
        return ALIASES[tag]
    folded = _FOLD_CANON.get(_fold(tag))
    if folded is None:
        return tag
    return ALIASES.get(folded, folded)


def tag_assignments(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """Every (item_id, tag) pair. The only storage read the graph needs."""
    return [(r["item_id"], r["tag"]) for r in conn.execute("SELECT item_id, tag FROM item_tags")]


def build_graph(
    assignments: Iterable[tuple[int, str]],
) -> tuple[dict[str, set[str]], dict[tuple[str, str], int]]:
    """Derive (adjacency, edge_weights) from (item_id, tag) pairs.

    Aliases first, then the frequency filter, then co-occurrence -- see the
    module docstring for why that order is load-bearing.

    Takes assignments rather than a connection so that ordering can be tested
    without a database, and so a caller already holding a graph does not derive
    it twice -- health() did, via communities().
    """
    items: dict[int, set[str]] = defaultdict(set)
    for item_id, tag in assignments:
        items[item_id].add(canonical(tag))

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
    adjacency, weights = build_graph(tag_assignments(conn))
    # canonical() FIRST. build_graph canonicalises on write, so the graph holds
    # `large-language-models` and a caller asking for `llm` -- the most common
    # tag in the live corpus, and an alias in kg_aliases.toml -- was told the
    # concept does not exist while its canonical form has 163 neighbours. The
    # suggested recovery made it worse: kg.concepts(prefix="llm") returns
    # code-llm and llm-safety and NOT the hub, so the caller concludes their
    # reading is absent. Measured on gemma4:e2b, the model did exactly that.
    node = resolve_query(node, adjacency)
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

    adjacency, _ = build_graph(tag_assignments(conn))
    # Both ends canonicalised, for the same reason as neighbors(): the graph is
    # built from canonical names, so `llm` -> `large-language-models`. Without
    # this, a path between two real concepts reported None whenever either was
    # spelled the way people actually write it.
    source = resolve_query(source, adjacency)
    target = resolve_query(target, adjacency)
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
    adjacency, _ = build_graph(tag_assignments(conn))
    if metric == "degree":
        scores = {name: float(len(neighbours)) for name, neighbours in adjacency.items()}
    else:
        scores = _betweenness(adjacency)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"name": name, "score": score} for name, score in ranked[: max(1, int(limit))]]


def communities(
    conn: sqlite3.Connection,
    min_size: int = 3,
    graph: tuple[dict[str, set[str]], dict[tuple[str, str], int]] | None = None,
) -> list[dict]:
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

    (Those counts are from a 347-node graph; the corpus has since grown to
    721 nodes over 5,222 items and the largest community is 28%. The
    conclusion held on re-measurement -- the shape is the same -- but the
    numbers are a snapshot, not a constant.)

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
    adjacency, weights = graph if graph is not None else build_graph(tag_assignments(conn))

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

    `distinct_tags` and `singleton_rate` count CANONICAL tags over distinct
    items -- the same vocabulary build_graph filters. Counting raw rows
    instead (as this did) made the metric blind to the merging it exists to
    watch: every alias added left `singleton_rate` unmoved, because both
    spellings were still counted separately, and a tag on two copies of one
    item read as reused when the graph saw it once.
    """
    assignments = tag_assignments(conn)
    adjacency, edges = build_graph(assignments)
    degrees = sorted((len(v) for v in adjacency.values()), reverse=True)
    per_item: dict[int, set[str]] = defaultdict(set)
    for item_id, tag in assignments:
        per_item[item_id].add(canonical(tag))
    uses: dict[str, int] = defaultdict(int)
    for names in per_item.values():
        for name in names:
            uses[name] += 1
    tags = uses
    singles = sum(1 for n in uses.values() if n == 1)
    nodes = len(adjacency)
    groups = communities(conn, min_size=3, graph=(adjacency, edges))
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
