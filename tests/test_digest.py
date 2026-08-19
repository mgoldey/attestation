"""Digest: the ranked feed grouped by topic."""

import pytest

from attestation import mcp_server
from attestation.db import get_db
from attestation.rank import RankedItem


def seed(db_path, clicks=((1, 1),), n_items=6):
    """Items tagged so they fall into two distinct concept clusters."""
    conn = get_db(db_path)
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
    """Build a stand-in for _ranked_items(conn, user_row, limit, since_days)."""

    def fake(conn, user_row, limit, since_days):
        return items

    return fake


def test_items_group_under_the_cluster_their_tags_match(db, monkeypatch):
    seed(db)
    monkeypatch.setattr(
        mcp_server,
        "_ranked_items",
        _fake_ranked_items(
            [
                _item(1, ["alpha", "beta"]),
                _item(2, ["delta", "epsilon"], score=0.9),
            ]
        ),
    )

    out = mcp_server._digest_impl("matt")

    assert out["ok"] is True
    labels = {t["label"] for t in out["topics"]}
    assert len(labels) == 2, f"two tag clusters should give two topics, got {labels}"


def test_an_item_matching_no_cluster_is_reported_not_dropped(db, monkeypatch):
    """Most tags are used once and never become concepts, so this bucket is
    expected to be non-empty -- its size is a real signal about the week."""
    seed(db)
    monkeypatch.setattr(
        mcp_server,
        "_ranked_items",
        _fake_ranked_items([_item(9, ["nothing-matches-this"], title="orphan")]),
    )

    out = mcp_server._digest_impl("matt")

    assert out["topics"] == []
    assert [i["item_id"] for i in out["unclustered"]] == [9]


def test_per_topic_truncates_but_n_total_stays_true(db, monkeypatch):
    """Silent truncation reads as 'that was everything'."""
    seed(db)
    items = [_item(i, ["alpha", "beta"]) for i in range(1, 6)]
    monkeypatch.setattr(mcp_server, "_ranked_items", _fake_ranked_items(items))

    out = mcp_server._digest_impl("matt", per_topic=2)

    assert len(out["topics"][0]["items"]) == 2
    assert out["topics"][0]["n_total"] == 5


def test_single_class_clicks_report_the_classifier_as_inactive(db, monkeypatch):
    """rank.py's single-class guard means the click classifier never fires, so
    the order blends embedding similarity with a feature-preference term
    learned from the clicks. A digest that hides this looks identical to one
    from a well-trained ranker."""
    seed(db, clicks=((1, 1), (2, 1), (3, 1)))
    monkeypatch.setattr(mcp_server, "_ranked_items", _fake_ranked_items([_item(1, ["alpha"])]))

    quality = mcp_server._digest_impl("matt")["ranking_quality"]

    assert quality["classifier_active"] is False
    assert quality["clicks"] == 3
    assert "WITHOUT its click classifier" in quality["caveat"]


def test_both_classes_activate_the_classifier(db, monkeypatch):
    seed(db, clicks=((1, 1), (2, 0)))
    monkeypatch.setattr(mcp_server, "_ranked_items", _fake_ranked_items([_item(1, ["alpha"])]))

    quality = mcp_server._digest_impl("matt")["ranking_quality"]

    assert quality["classifier_active"] is True
    assert "WITHOUT" not in quality.get("caveat", "")


def test_empty_feed_preserves_success_path_keys(db, monkeypatch):
    seed(db)
    monkeypatch.setattr(mcp_server, "_ranked_items", _fake_ranked_items([]))

    out = mcp_server._digest_impl("matt")

    assert out["ok"] is False
    assert out["topics"] == [] and out["unclustered"] == []


def test_unknown_user_is_reported(db):
    seed(db)

    out = mcp_server._digest_impl("nobody")

    assert out["ok"] is False
    assert "unknown user" in out["message"]


def test_digest_is_deterministic(db, monkeypatch):
    seed(db)
    items = [_item(i, ["alpha", "beta"]) for i in range(1, 4)]
    monkeypatch.setattr(mcp_server, "_ranked_items", _fake_ranked_items(items))

    first = mcp_server._digest_impl("matt")
    second = mcp_server._digest_impl("matt")

    assert [t["label"] for t in first["topics"]] == [t["label"] for t in second["topics"]]


def test_digest_is_served():
    import asyncio

    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert "digest" in names


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

    monkeypatch.setattr(mcp_server, "rank_items", fake_rank)
    monkeypatch.setattr(mcp_server, "_get_embedder", lambda: object())

    mcp_server._digest_impl("matt", days=30)

    assert seen["since_days"] == 30


def test_list_feed_keeps_its_default_window(db, monkeypatch):
    """digest gained a window; list_feed's own default must not shift."""
    seed(db)
    seen = {}

    def fake_rank(conn, embedder, user_id, since_days=14, **kw):
        seen["since_days"] = since_days
        return []

    monkeypatch.setattr(mcp_server, "rank_items", fake_rank)
    monkeypatch.setattr(mcp_server, "_get_embedder", lambda: object())

    mcp_server._list_feed_impl("matt", limit=5)

    assert seen["since_days"] == 14
