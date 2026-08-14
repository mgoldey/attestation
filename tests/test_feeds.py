import pytest

from attestation import feeds
from attestation.db import get_db


class FakeParsed:
    def __init__(self, title="Fake Feed", entries=None, bozo=0):
        self.feed = {"title": title}
        self.entries = entries if entries is not None else [{"title": "e1"}]
        self.bozo = bozo


def _parse_ok(url):
    return FakeParsed()


def _parse_bad(url):
    p = FakeParsed(entries=[], bozo=1)
    return p


@pytest.fixture
def conn(tmp_path):
    c = get_db(tmp_path / "t.db")
    yield c
    c.close()


def test_add_feed_registers_without_ingesting(conn):
    out = feeds.add_feed(conn, "http://example.com/rss", parse=_parse_ok)

    assert out["ok"] is True
    row = conn.execute("SELECT url, title FROM feeds WHERE id = ?", (out["feed_id"],)).fetchone()
    assert row["url"] == "http://example.com/rss"
    # register-only: no items were fetched
    assert conn.execute("SELECT COUNT(*) n FROM items").fetchone()["n"] == 0
    assert "ingest" in out["message"].lower()


def test_add_feed_rejects_unparseable_url_without_inserting(conn):
    out = feeds.add_feed(conn, "http://example.com/not-a-feed", parse=_parse_bad)

    assert out["ok"] is False
    assert conn.execute("SELECT COUNT(*) n FROM feeds").fetchone()["n"] == 0


def test_add_feed_is_idempotent(conn):
    first = feeds.add_feed(conn, "http://example.com/rss", parse=_parse_ok)
    second = feeds.add_feed(conn, "http://example.com/rss", parse=_parse_ok)

    assert second["feed_id"] == first["feed_id"]
    assert conn.execute("SELECT COUNT(*) n FROM feeds").fetchone()["n"] == 1


def test_list_feeds_reports_item_counts(conn):
    out = feeds.add_feed(conn, "http://example.com/rss", parse=_parse_ok)
    fid = out["feed_id"]
    conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (?, 't', 'u', 's', 'h1')",
        (fid,),
    )
    conn.commit()

    listed = feeds.list_feeds(conn)

    assert len(listed) == 1
    assert listed[0]["item_count"] == 1
    assert listed[0]["feed_id"] == fid


def test_remove_feed_orphans_items_and_preserves_clicks(conn):
    fid = feeds.add_feed(conn, "http://example.com/rss", parse=_parse_ok)["feed_id"]
    # get_db auto-seeds users 1-3 (matt, bench-chemist, ml-engineer); insert
    # without an explicit id to avoid colliding with the seeded rows.
    cur = conn.execute("INSERT INTO users(name, interests) VALUES ('u', 'x')")
    user_id = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (?, 't', 'u', 's', 'h1')",
        (fid,),
    )
    item_id = cur.lastrowid
    conn.execute(
        "INSERT INTO clicks(user_id, item_id, useful, source) VALUES (?, ?, 1, 'ui')",
        (user_id, item_id),
    )
    conn.commit()

    out = feeds.remove_feed(conn, fid)

    assert out["ok"] is True
    assert conn.execute("SELECT COUNT(*) n FROM feeds").fetchone()["n"] == 0
    # items and the click that trained the ranker both survive
    assert conn.execute("SELECT COUNT(*) n FROM items").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) n FROM clicks").fetchone()["n"] == 1


def test_remove_feed_unknown_id_is_not_ok(conn):
    out = feeds.remove_feed(conn, 999)

    assert out["ok"] is False


def test_preview_feed_does_not_subscribe(conn):
    out = feeds.preview_feed("http://example.com/rss", limit=1, parse=_parse_ok)

    assert out["ok"] is True
    assert len(out["entries"]) == 1
    assert conn.execute("SELECT COUNT(*) n FROM feeds").fetchone()["n"] == 0
