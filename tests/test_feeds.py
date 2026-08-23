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


def test_two_subscriptions_to_one_url_do_not_both_error(tmp_path):
    """Check-then-insert with a NETWORK FETCH in the window.

    add_feed reads `SELECT id FROM feeds WHERE url = ?`, then parses the feed
    over the network, then inserts -- so the gap between check and write is a
    whole HTTP round trip, the widest of the four sites sharing this shape.
    Measured: 8 concurrent calls gave 7 IntegrityErrors and 1 success, where
    the serial path returns an idempotent "already subscribed to ...".

    Losing the race means the subscription the caller asked for exists, which
    is the same outcome the serial path calls success.
    """
    import concurrent.futures
    from types import SimpleNamespace

    from attestation.db import get_db
    from attestation.feeds import add_feed

    db = tmp_path / "t.db"
    get_db(db).commit()

    def parse(_url):
        return SimpleNamespace(
            entries=[{"title": "t", "summary": "s", "id": "g1", "link": "u"}],
            feed={"title": "A Feed"},
        )

    def subscribe(_):
        try:
            return add_feed(get_db(db), "http://example.invalid/feed", parse=parse)
        except Exception as exc:  # noqa: BLE001 -- the point is what leaks out
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        results = list(pool.map(subscribe, range(8)))

    failed = [r["message"] for r in results if not r["ok"]]
    assert not failed, f"{len(failed)} of 8 concurrent subscribes failed: {failed[0]}"
    rows = get_db(db).execute("SELECT COUNT(*) FROM feeds").fetchone()[0]
    assert rows == 1, f"{rows} feed rows for one url"
