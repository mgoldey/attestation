import pytest

from attestation import kg
from attestation.db import get_db


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


def test_neighbors_direct_ring(line_graph):
    result = kg.neighbors(line_graph, "B")

    names = {r["name"] for r in result}
    assert names == {"A", "C"}
    assert all(r["distance"] == 1 for r in result)
    assert all(r["weight"] == 2 for r in result)


def test_neighbors_excludes_the_node_itself(line_graph):
    assert all(r["name"] != "B" for r in kg.neighbors(line_graph, "B"))


def test_neighbors_unknown_node_is_empty_not_error(line_graph):
    assert kg.neighbors(line_graph, "nonexistent") == []


@pytest.fixture
def hub_graph(tmp_path):
    """A hub with two direct neighbours of unequal weight: 'aardvark' is
    alphabetically first but weakly tied (weight 2), while 'zebra' is
    alphabetically last but strongly tied (weight 5).
    """
    conn = get_db(tmp_path / "hub.db")
    seed(
        conn,
        [
            ["hub", "aardvark"],
            ["hub", "aardvark"],
            ["hub", "zebra"],
            ["hub", "zebra"],
            ["hub", "zebra"],
            ["hub", "zebra"],
            ["hub", "zebra"],
        ],
    )
    yield conn
    conn.close()


def test_neighbors_ranks_by_edge_weight_not_alphabetically(hub_graph):
    result = kg.neighbors(hub_graph, "hub", limit=1)

    assert [r["name"] for r in result] == ["zebra"]


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


def test_communities_never_span_disconnected_components(line_graph):
    """A community must never contain nodes from two components -- there is no
    path between them, so no honest clustering can group them.

    This asserts the invariant rather than one exact partition: the A-B-C-D
    chain has no real community structure (every edge has weight 2), so a
    modularity-based clusterer may legitimately split it at the middle while
    label propagation returned it whole. What must never happen is X or Y
    landing with any of A-D.
    """
    result = kg.communities(line_graph, min_size=2)

    memberships = [set(c["members"]) for c in result]
    assert {"X", "Y"} in memberships
    chain = {"A", "B", "C", "D"}
    for members in memberships:
        assert members <= chain or members <= {"X", "Y"}, f"{members} spans components"
    assert chain == set().union(*(m for m in memberships if m <= chain))


def test_communities_respect_min_size(line_graph):
    result = kg.communities(line_graph, min_size=3)

    assert all(len(c["members"]) >= 3 for c in result)
    assert not any(set(c["members"]) == {"X", "Y"} for c in result)


def _hub_joined_topics(conn, n_topics: int):
    """`n_topics` internally-dense triangles, every node also touching one hub.

    The live shape: a hub whose reach spans everything, weak links to it
    (weight 2, the MIN_EDGE_WEIGHT floor -- 48% of the real graph's edges sit
    there), strong links inside each topic (weight 6).
    """
    rows: list[list[str]] = []
    names: list[str] = []
    for i in range(n_topics):
        a, b, c = f"t{i}a", f"t{i}b", f"t{i}c"
        names += [a, b, c]
        for x, y in ((a, b), (b, c), (a, c)):
            rows += [[x, y]] * 6
    for name in names:
        rows += [["hub", name]] * 2
    seed(conn, rows)
    return conn


@pytest.mark.parametrize("n_topics", [2, 4, 8])
def test_communities_split_topics_joined_by_a_hub(tmp_path, n_topics):
    """Regression: label propagation returned ONE community holding 95% of the
    live graph (331 of 347 nodes). A hub's label spreads across a whole
    connected component unopposed, because nothing in label propagation
    penalises joining an already-huge community.

    Verified: label propagation collapses this fixture to a single community at
    every size tested, including the smallest. Modularity's `- k_i * k_c / 2m`
    term is what resists it -- a node joins only if its links there beat chance.
    """
    conn = _hub_joined_topics(get_db(tmp_path / f"hub{n_topics}.db"), n_topics)

    result = kg.communities(conn, min_size=3)

    assert len(result) == n_topics, (
        f"expected {n_topics} topics, got {len(result)}: {[c['label'] for c in result]}"
    )
    for i in range(n_topics):
        topic = {f"t{i}a", f"t{i}b", f"t{i}c"}
        assert any(topic <= set(c["members"]) for c in result), f"topic {i} was not recovered"
    conn.close()


def test_communities_are_deterministic(line_graph):
    """Label propagation is randomized in the general case; this
    implementation must be seeded or order-stable so repeated calls agree."""
    first = kg.communities(line_graph, min_size=2)
    second = kg.communities(line_graph, min_size=2)

    assert [set(c["members"]) for c in first] == [set(c["members"]) for c in second]


@pytest.fixture
def direct_neighbour_cut_by_limit(tmp_path):
    """A hub whose weakest direct neighbours ('n4', 'n5') are cut by a small
    limit and are ALSO adjacent to the strongest one ('n1') -- the shape that
    made the old multi-ring walk re-reach them through 'n1' and relabel them
    distance=2/weight=None, mislabelling 100 rows on the live graph.
    """
    conn = get_db(tmp_path / "cut.db")
    rows: list[list[str]] = []
    for name, count in (("n1", 5), ("n4", 2), ("n5", 2)):
        rows += [["hub", name]] * count
    rows += [["n1", "n4"]] * 3
    rows += [["n1", "n5"]] * 3
    seed(conn, rows)
    yield conn
    conn.close()


@pytest.mark.parametrize("limit", [1, 2, 3, 10])
def test_neighbors_reports_only_direct_neighbours(direct_neighbour_cut_by_limit, limit):
    """Contract check, not a reproduction: with the ring walk gone, distance==1
    and a real weight hold by construction rather than by any branch, so this
    cannot fail without a redesign. It is here to pin the contract if one is
    ever attempted again -- the discriminating assertions are the two below,
    which fail against plausible wrong implementations.

    Truncation must drop the WEAKEST edges, keeping a prefix of the
    weight-ranked order: 'n1' (weight 5) outranks 'n4'/'n5' (weight 2), so it
    survives every limit. Truncating alphabetically or arbitrarily would cut
    'n1' first at limit=1.
    """
    conn = direct_neighbour_cut_by_limit
    result = kg.neighbors(conn, "hub", limit=limit)
    adjacency, weights = kg.build_graph(conn)

    assert {r["name"] for r in result} <= adjacency["hub"]
    for row in result:
        assert row["distance"] == 1
        pair = ("hub", row["name"]) if "hub" < row["name"] else (row["name"], "hub")
        assert row["weight"] == weights[pair]

    # Discriminating: the strongest edge survives every truncation, and the
    # result is exactly the weight-ranked prefix of the full neighbour set.
    assert result[0]["name"] == "n1"
    assert len(result) == min(limit, len(adjacency["hub"]))
    assert [r["weight"] for r in result] == sorted((r["weight"] for r in result), reverse=True)


def test_health_reports_usability_not_just_size(line_graph):
    """health() exists to answer "is this graph usable", so it must report the
    ratios that caught real problems -- a singleton-heavy vocabulary and a
    single community swallowing the graph -- not only node/edge counts."""
    report = kg.health(line_graph)

    assert report["nodes"] == 6  # A-D chain plus the X-Y pair
    assert report["edges"] == 4
    for key in (
        "singleton_rate",
        "degree_1_pct",
        "hub_dominance_pct",
        "communities",
        "largest_community_pct",
    ):
        assert key in report, f"health() must report {key}"
    assert 0 <= report["largest_community_pct"] <= 100


def test_health_on_an_empty_graph_does_not_divide_by_zero(tmp_path):
    """A fresh database has no tags; the report must still return."""
    conn = get_db(tmp_path / "empty.db")

    report = kg.health(conn)

    assert report["nodes"] == 0
    assert report["singleton_rate"] == 0.0
    assert report["largest_community_pct"] == 0.0
    conn.close()
