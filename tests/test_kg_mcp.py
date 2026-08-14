from attestation import mcp_server
from attestation.db import get_db


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


def test_kg_neighbors_never_exceeds_max_list_limit_total_rows(tmp_path, monkeypatch):
    """A hub can have more direct neighbours than MAX_LIST_LIMIT, so the
    response must honour it as a true cap, matching every other list tool --
    including when the caller asks for more."""
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    seed_deep_wide_graph(db)

    for requested in (50, 200):
        out = mcp_server._kg_neighbors_impl("hub", limit=requested)
        assert len(out["neighbors"]) <= mcp_server.MAX_LIST_LIMIT


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
    # No total-count assertion: it fails on every unrelated tool added, which
    # makes it a change-detector rather than a test of anything. Each tool group
    # asserts its own names instead.
    assert len(names) >= 5
