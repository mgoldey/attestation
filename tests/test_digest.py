"""Digest: the ranked feed grouped by topic."""

import pytest
from conftest import seeded_db

from attestation import mcp_server
from attestation.mcp import _shared
from attestation.mcp import feed as feed_mod
from attestation.rank import RankedItem


def seed(db_path, clicks=((1, 1),), n_items=6):
    """Items tagged so they fall into two distinct concept clusters."""
    conn = seeded_db(db_path)
    clusters = [["alpha", "beta", "gamma"], ["delta", "epsilon", "zeta"]]
    for i in range(1, n_items + 1):
        conn.execute(
            "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
            " VALUES (?, NULL, ?, ?, 's', ?)",
            (i, f"item {i}", f"http://x/{i}", f"h{i}"),
        )
        for t in clusters[i % 2]:
            conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (i, t))
        conn.execute(
            "INSERT INTO item_features(item_id, content_type, model) VALUES (?, 'paper', 'm')",
            (i,),
        )
    for item_id, useful in clicks:
        conn.execute(
            "INSERT INTO clicks(user_id, item_id, useful, source) VALUES (1, ?, ?, 'ui')",
            (item_id, useful),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(path))
    return path


def _item(item_id, tags, score=1.0, title=None):
    return RankedItem(
        item_id=item_id,
        title=title or f"t{item_id}",
        url="u",
        source="s",
        score=score,
        tags=tags,
        content_type="paper",
    )


def _fake_ranked_items(items):
    """Build a stand-in for ranked_items(conn, user_row, limit, since_days)."""

    def fake(conn, user_row, limit, since_days):
        return items

    return fake


def test_items_group_under_the_cluster_their_tags_match(db, monkeypatch):
    seed(db)
    monkeypatch.setattr(
        feed_mod,
        "ranked_items",
        _fake_ranked_items(
            [
                _item(1, ["alpha", "beta"]),
                _item(2, ["delta", "epsilon"], score=0.9),
            ]
        ),
    )

    out = mcp_server._digest_impl("researcher")

    assert out["ok"] is True
    labels = {t["label"] for t in out["topics"]}
    assert len(labels) == 2, f"two tag clusters should give two topics, got {labels}"


def test_an_item_matching_no_cluster_is_reported_not_dropped(db, monkeypatch):
    """Most tags are used once and never become concepts, so this bucket is
    expected to be non-empty -- its size is a real signal about the week."""
    seed(db)
    monkeypatch.setattr(
        feed_mod,
        "ranked_items",
        _fake_ranked_items([_item(9, ["nothing-matches-this"], title="orphan")]),
    )

    out = mcp_server._digest_impl("researcher")

    assert out["topics"] == []
    assert [i["item_id"] for i in out["unclustered"]] == [9]


def test_per_topic_truncates_but_n_total_stays_true(db, monkeypatch):
    """Silent truncation reads as 'that was everything'."""
    seed(db)
    items = [_item(i, ["alpha", "beta"]) for i in range(1, 6)]
    monkeypatch.setattr(feed_mod, "ranked_items", _fake_ranked_items(items))

    out = mcp_server._digest_impl("researcher", per_topic=2)

    assert len(out["topics"][0]["items"]) == 2
    assert out["topics"][0]["n_total"] == 5


def test_single_class_clicks_report_the_classifier_as_inactive(db, monkeypatch):
    """rank.py's single-class guard means the click classifier never fires, so
    the order blends embedding similarity with a feature-preference term
    learned from the clicks. A digest that hides this looks identical to one
    from a well-trained ranker."""
    seed(db, clicks=((1, 1), (2, 1), (3, 1)))
    monkeypatch.setattr(feed_mod, "ranked_items", _fake_ranked_items([_item(1, ["alpha"])]))

    quality = mcp_server._digest_impl("researcher")["ranking_quality"]

    assert quality["classifier_active"] is False
    assert quality["clicks"] == 3
    assert "classifier OFF" in quality["caveat"]


def test_both_classes_activate_the_classifier(db, monkeypatch):
    seed(db, clicks=((1, 1), (2, 0)))
    monkeypatch.setattr(feed_mod, "ranked_items", _fake_ranked_items([_item(1, ["alpha"])]))

    quality = mcp_server._digest_impl("researcher")["ranking_quality"]

    assert quality["classifier_active"] is True
    assert "WITHOUT" not in quality.get("caveat", "")


def test_empty_feed_preserves_success_path_keys(db, monkeypatch):
    seed(db)
    monkeypatch.setattr(feed_mod, "ranked_items", _fake_ranked_items([]))

    out = mcp_server._digest_impl("researcher")

    assert out["ok"] is False
    assert out["topics"] == [] and out["unclustered"] == []


def test_unknown_user_is_reported(tmp_path, monkeypatch):
    """An unknown reader is now CREATED, not refused.

    Refusing and listing the valid names taught agents to call
    persona_create with whatever string they had -- the live database
    grew a duplicate persona that way, days after that reader had been
    merged away. The refusal caused the duplicate it was meant to
    prevent, so read-side tools create on first sight and say so.
    """
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    out = mcp_server._list_feed_impl("nobody-yet", limit=2)

    assert out["ok"] is True, out["message"]
    assert "created" in out["message"].lower()
    assert "nobody-yet" in out["message"]
    # and it asks the one question only the reader can answer
    assert "monitor" in out["message"].lower() or "topics" in out["message"].lower()


def test_digest_is_deterministic(db, monkeypatch):
    seed(db)
    items = [_item(i, ["alpha", "beta"]) for i in range(1, 4)]
    monkeypatch.setattr(feed_mod, "ranked_items", _fake_ranked_items(items))

    first = mcp_server._digest_impl("researcher")
    second = mcp_server._digest_impl("researcher")

    assert [t["label"] for t in first["topics"]] == [t["label"] for t in second["topics"]]


def test_digest_is_served():
    import asyncio

    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert "feed.digest" in names


def test_days_reaches_the_ranker(db, monkeypatch):
    """`days` must filter, not just echo back as window_days.

    The bug this pins: _list_feed_impl took only (user, limit), so days never
    reached rank_items' since_days and digest(days=7) returned exactly what
    digest(days=90) did. _ranked_items is now the shared chokepoint both
    list_feed and digest funnel through, so this pins since_days at that layer.
    """
    seed(db)
    seen = {}

    def fake_rank(conn, embedder, user_id, since_days=14, **kw):
        seen["since_days"] = since_days
        return []

    monkeypatch.setattr(_shared, "rank_items", fake_rank)
    monkeypatch.setattr(_shared, "get_embedder", lambda: object())

    mcp_server._digest_impl("researcher", days=30)

    assert seen["since_days"] == 30


def test_list_feed_keeps_its_default_window(db, monkeypatch):
    """digest gained a window; list_feed's own default must not shift."""
    seed(db)
    seen = {}

    def fake_rank(conn, embedder, user_id, since_days=14, **kw):
        seen["since_days"] = since_days
        return []

    monkeypatch.setattr(_shared, "rank_items", fake_rank)
    monkeypatch.setattr(_shared, "get_embedder", lambda: object())

    mcp_server._list_feed_impl("researcher", limit=5)

    assert seen["since_days"] == 14


def test_the_digest_message_counts_what_it_actually_shipped(tmp_path, monkeypatch, fake_embedder):
    """The summary line described a payload the caller did not receive.

    `dropped` counted groups omitted by the item budget and unclustered items
    left over -- never items cut INSIDE a shown topic by `per_topic`. Measured
    on all five live personas: every one said "16 item(s)" while shipping 6-11.

    tests/test_digest.py already has a case titled "Silent truncation reads as
    'that was everything'". It asserts n_total stays true per topic and never
    asserts the MESSAGE reports the total, so this bug passed its own guard.
    """
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = seeded_db(tmp_path / "t.db")
    conn.execute("INSERT INTO users(name, interests) VALUES ('ana', 'machine learning')")
    for i in range(1, 21):
        conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'u', 's', ?)",
            (f"item {i}", f"h{i}"),
        )
        conn.execute(
            "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
            (i, fake_embedder.embed_document(f"item {i}", "s").tobytes()),
        )
        # One big topic, so per_topic=3 cuts far more than it shows.
        for tag in ("machine-learning", "evaluation", "ranking"):
            conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (i, tag))
    conn.commit()
    conn.close()

    from attestation.mcp import _shared
    from attestation.mcp import feed as feed_mod

    monkeypatch.setattr(_shared, "get_embedder", lambda: fake_embedder)
    out = feed_mod._digest("ana")

    shipped = sum(len(t["items"]) for t in out["topics"]) + len(out["unclustered"])
    considered = int(out["message"].split(" item(s)")[0])
    assert considered >= shipped
    if considered > shipped:
        assert "not shown" in out["message"], (
            f"message says {considered} item(s), payload has {shipped}: {out['message']!r}"
        )


def test_digest_budget_accounts_for_per_topic_truncation():
    """`_allocate_digest_budget` is the pure allocator `_digest_body` calls:
    given grouped topics, leftover unclustered items, a per-topic cap and a
    total budget, `shipped` must equal what actually ends up in the returned
    payload -- topics' shown items plus shown unclustered -- not a count that
    forgets per-topic truncation. That mismatch is the bug this seam existed
    to give a DB-free regression test for (see the digest-truncation test
    above): every live persona read "16 item(s)" while shipping 6 to 11.
    """
    from attestation.mcp.feed import _allocate_digest_budget

    item = {"item_id": 0}
    grouped = {
        "a": [dict(item, item_id=i) for i in range(8)],
        "b": [dict(item, item_id=i) for i in range(2)],
    }
    unclustered = [dict(item, item_id=i) for i in range(100, 105)]
    out = _allocate_digest_budget(grouped, unclustered, per_topic=3, budget=12)
    in_topics = sum(len(t["items"]) for t in out["topics"])
    assert out["shipped"] == in_topics + len(out["shown_unclustered"])
    assert [t["label"] for t in out["topics"]] == ["a", "b"]  # largest first
    assert out["topics"][0]["n_total"] == 8 and len(out["topics"][0]["items"]) == 3
    assert out["shipped"] == 3 + 2 + 5 and len(out["shown_unclustered"]) == 5
