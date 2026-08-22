import asyncio

from attestation import mcp_server
from attestation.db import get_db
from attestation.mcp import _shared


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


def seed_deep_wide_graph(db_path):
    """A hub with 80 direct neighbours -- more than MAX_LIST_LIMIT (50) on its
    own, so the response cap has to do real work.
    """
    conn = get_db(db_path)
    item_id = 1
    pairs = []
    for i in range(80):
        r1 = f"r1-{i:02d}"
        pairs.append(["hub", r1])
        pairs.append(["hub", r1])
    for tags in pairs:
        conn.execute(
            "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
            " VALUES (?, NULL, ?, 'u', 's', ?)",
            (item_id, f"item {item_id}", f"h{item_id}"),
        )
        for t in tags:
            conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (item_id, t))
        item_id += 1
    conn.commit()
    conn.close()


def seed_two_components(db_path):
    """Two real concepts with no route between them.

    A--B on items 1-2, X--Y on items 3-4. Every tag is used twice, so all four
    clear MIN_TAG_USES and both edges clear MIN_EDGE_WEIGHT: these are genuine
    concepts in the graph that simply never co-occur.
    """
    conn = get_db(db_path)
    rows = [["A", "B"], ["A", "B"], ["X", "Y"], ["X", "Y"]]
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


def test_kg_neighbors_never_exceeds_max_list_limit_total_rows(tmp_path, monkeypatch):
    """A hub can have more direct neighbours than MAX_LIST_LIMIT, so the
    response must honour it as a true cap, matching every other list tool --
    including when the caller asks for more."""
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    seed_deep_wide_graph(db)

    for requested in (50, 200):
        out = mcp_server._kg_neighbors_impl("hub", limit=requested)
        assert len(out["neighbors"]) <= _shared.MAX_LIST_LIMIT


def test_kg_neighbors_returns_adjacent_concepts(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    seed_env_db(db)

    out = mcp_server._kg_neighbors_impl("B")

    assert out["ok"] is True
    assert {n["name"] for n in out["neighbors"]} == {"A", "C"}


def test_kg_path_reports_no_path_without_erroring(tmp_path, monkeypatch):
    """Two REAL concepts with no route between them: a finding, not an error.

    This asserted the same thing with target="nonexistent" until 2026-08-21,
    which passed for the wrong reason -- kg.shortest_path returns None for an
    absent node too, so the test was satisfied by the very confusion it now
    guards against. The seed gives A and X genuine edges in separate
    components, so only a real no-path answer can satisfy it.
    """
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    seed_two_components(db)

    out = mcp_server._kg_path_impl("A", "X")

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


def test_the_five_kg_tools_are_served(tmp_path, monkeypatch):
    """kg_rebuild is deliberately absent: it materialized kg_nodes/kg_edges,
    which nothing read. Deleting it took the tool count from 36 to 35;
    kg_concepts, added 2026-08-21 so concept names are discoverable at all,
    brings it back to 36."""
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert {"kg.neighbors", "kg.path", "kg.central", "kg.communities", "kg.concepts"} <= names
    assert "kg.rebuild" not in names


def test_kg_path_unknown_target_is_not_reported_as_no_path(tmp_path, monkeypatch):
    """A typo'd concept must not read as "these never co-occur".

    kg_path's docstring tells the caller that path=null means the two topics
    are in different components -- a legitimate finding. If a name that is not
    in the graph at all produced the same message, an agent that mistyped a
    concept would confidently report a fact about a concept that does not
    exist.
    """
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    seed_env_db(db)

    out = mcp_server._kg_path_impl("A", "nonexistent")

    assert out["ok"] is False
    assert out["path"] is None
    assert "no path" not in out["message"].lower()
    assert "nonexistent" in out["message"]
    assert "kg.concepts" in out["message"]


def test_kg_path_unknown_source_names_the_source(tmp_path, monkeypatch):
    """The message must name WHICH of the two is unknown, not just that one is."""
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    seed_env_db(db)

    out = mcp_server._kg_path_impl("nosuchsource", "A")

    assert out["ok"] is False
    assert out["path"] is None
    assert "nosuchsource" in out["message"]
    assert "no path" not in out["message"].lower()
    assert "kg.concepts" in out["message"]


def test_kg_neighbors_unknown_node_points_at_kg_concepts(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    seed_env_db(db)

    out = mcp_server._kg_neighbors_impl("nonexistent")

    assert out["ok"] is False
    assert "nonexistent" in out["message"]
    assert "kg.concepts" in out["message"]


def test_kg_concepts_lists_every_concept(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    seed_env_db(db)

    out = mcp_server._kg_concepts_impl()

    assert out["ok"] is True
    assert out["concepts"] == ["A", "B", "C"]
    assert out["n_concepts"] == 3


def test_kg_concepts_filters_by_prefix(tmp_path, monkeypatch):
    """The filter is a case-insensitive substring match, and the count that
    comes back is the count of MATCHES, not of the whole graph -- otherwise a
    caller cannot tell a narrow filter from a truncated one."""
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    seed_deep_wide_graph(db)

    out = mcp_server._kg_concepts_impl(prefix="R1-0")

    assert out["ok"] is True
    assert out["concepts"] == [f"r1-{i:02d}" for i in range(10)]
    assert out["n_concepts"] == 10
    assert "hub" not in out["concepts"]


def test_kg_concepts_reports_truncation(tmp_path, monkeypatch):
    """81 concepts, capped at MAX_LIST_LIMIT: the total must still be visible,
    so a caller knows the list it got is not the whole vocabulary."""
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    seed_deep_wide_graph(db)

    out = mcp_server._kg_concepts_impl(limit=200)

    assert out["ok"] is True
    assert len(out["concepts"]) == _shared.MAX_LIST_LIMIT
    assert out["n_concepts"] == 81
    assert "81" in out["message"]
