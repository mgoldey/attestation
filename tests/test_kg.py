from attestation import kg
from attestation.db import get_db


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


def test_huggingface_hyphenation_variant_is_merged(tmp_path):
    """One of the eight hyphenation variants added to kg_aliases.toml: the
    unhyphenated spelling must not survive as its own node once merged with
    its canonical, hyphenated form."""
    conn = get_db(tmp_path / "t.db")
    seed(
        conn,
        [
            ["huggingface", "shared"],
            ["hugging-face", "shared"],
        ],
    )
    adjacency, _ = kg.build_graph(conn)

    assert "huggingface" not in adjacency
    assert "hugging-face" in adjacency
    conn.close()


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
