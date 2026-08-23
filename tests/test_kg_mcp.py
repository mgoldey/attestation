import asyncio
import json

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


def test_communities_returns_a_summary_not_a_dump(tmp_path, monkeypatch):
    """13 communities came back as 12,055 chars -- 6x the next biggest tool.

    One held 201 members listed alphabetically. That is the graph's contents,
    not a finding: an agent cannot read it, cannot quote it, and cannot decide
    anything from it. What a caller wants is which clusters exist, how big
    they are, and what each is about.

    Members are capped and the true size reported, so a large cluster is
    visible as large rather than being pasted in full.
    """
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    # One big cluster of co-occurring tags plus a small one.
    for item in range(1, 41):
        conn.execute(
            "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
            " VALUES (?, NULL, ?, 'u', 's', ?)",
            (item, f"t{item}", f"h{item}"),
        )
        group = [f"big{i}" for i in range(12)] if item <= 30 else ["small-a", "small-b", "small-c"]
        for tag in group:
            conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (item, tag))
    conn.commit()
    conn.close()

    out = mcp_server._kg_communities_impl(min_size=3)

    assert out["ok"] is True
    assert out["communities"], "no communities found in a seeded graph"
    for group in out["communities"]:
        assert len(group["members"]) <= 12, f"{len(group['members'])} members pasted in full"
        assert "n_members" in group, "the true size must survive truncation"
        assert group["n_members"] >= len(group["members"])

    # indent=2, which is what FastMCP emits. Measured compact this read 2500
    # while the model received 1.238x that; round 9 measured this tool at 4039
    # chars emitted against the live database.
    payload = len(json.dumps(out, indent=2))
    assert payload < 2500, f"kg.communities is {payload} chars; a caller cannot read it"


def test_communities_orders_by_size_so_the_big_ones_come_first(tmp_path, monkeypatch):
    """A caller reading only the first few must see the largest clusters."""
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    for item in range(1, 41):
        conn.execute(
            "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
            " VALUES (?, NULL, ?, 'u', 's', ?)",
            (item, f"t{item}", f"h{item}"),
        )
        group = [f"big{i}" for i in range(12)] if item <= 30 else ["small-a", "small-b", "small-c"]
        for tag in group:
            conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (item, tag))
    conn.commit()
    conn.close()

    groups = mcp_server._kg_communities_impl(min_size=3)["communities"]
    sizes = [g["n_members"] for g in groups]
    assert sizes == sorted(sizes, reverse=True), f"not ordered by size: {sizes}"


def test_communities_caps_the_number_of_groups_not_just_their_members(tmp_path, monkeypatch):
    """One axis capped, the other left open -- the digest bug, again.

    `_communities` caps members per group at MEMBERS_SHOWN but iterates every
    group `kg.communities` returns, and the tool exposes no limit. Measured
    against the live graph: 16 groups at the default min_size=3 emits 4039
    chars, and min_size=2 -- schema-allowed, ge=2 -- emits 6033, larger than
    the digest worst case that was bounded for the same reason.

    The guard above could not catch it: its fixture builds two communities and
    asserts under 2500, five times more headroom than it needs.
    """
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    # 30 disjoint 3-concept clusters: every one qualifies at min_size=3.
    item = 0
    for group in range(30):
        for pair in range(3):
            item += 1
            conn.execute(
                "INSERT INTO items(feed_id, title, url, summary, content_hash)"
                " VALUES (NULL, ?, 'u', 's', ?)",
                (f"i{item}", f"h{item}"),
            )
            for tag in (f"g{group}a", f"g{group}b", f"g{group}c"):
                conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (item, tag))
    conn.commit()
    conn.close()

    from attestation.mcp import knowledge as kn

    out = kn._communities()
    assert out["ok"] is True
    assert len(out["communities"]) <= kn.MAX_COMMUNITIES_SHOWN, (
        f"{len(out['communities'])} groups returned with no cap"
    )
    assert "n_communities" in out, "the true count must survive truncation"
    assert out["n_communities"] >= len(out["communities"])
    assert len(json.dumps(out, indent=2)) < 2500


def test_members_shown_cannot_ratchet_either(tmp_path, monkeypatch):
    """MAX_COMMUNITIES_SHOWN was guarded and MEMBERS_SHOWN, right beside it,
    was not.

    Raising it 8 -> 500 kept 38 tests green while the live kg.communities went
    from 2556 to 17360 emitted, past the 7000 ceiling. The existing guard
    asserts `len(members) <= 12` against a fixture whose biggest cluster has 12
    tags, so the cap never binds -- the cheap-fixture shape again.

    Anchored on a cluster far larger than any cap, so the assertion measures
    the cap rather than the fixture.
    """
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    tags = [f"concept{i}" for i in range(80)]
    for item in range(1, 6):
        conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'u', 's', ?)",
            (f"i{item}", f"h{item}"),
        )
        for tag in tags:
            conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (item, tag))
    conn.commit()
    conn.close()

    from attestation.mcp import knowledge as kn

    out = kn._communities()
    for group in out["communities"]:
        assert len(group["members"]) <= kn.MEMBERS_SHOWN, (
            f"{len(group['members'])} members returned; MEMBERS_SHOWN is {kn.MEMBERS_SHOWN}"
        )
    assert kn.MEMBERS_SHOWN <= 20, (
        f"MEMBERS_SHOWN is {kn.MEMBERS_SHOWN}; at 500 the live response reached 17360 chars"
    )
    assert len(json.dumps(out, indent=2)) < 2500
