import pytest
from conftest import seeded_db

from attestation import feeds


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
    c = seeded_db(tmp_path / "t.db")
    yield c
    c.close()


def test_add_feed_registers_without_ingesting(conn):
    feed_id, message = feeds.add_source(conn, "http://example.com/rss", parse=_parse_ok)

    row = conn.execute("SELECT url, title FROM feeds WHERE id = ?", (feed_id,)).fetchone()
    assert row["url"] == "http://example.com/rss"
    # register-only: no items were fetched
    assert conn.execute("SELECT COUNT(*) n FROM items").fetchone()["n"] == 0
    assert "ingest" in message.lower()


def test_add_feed_rejects_unparseable_url_without_inserting(conn):
    with pytest.raises(feeds.FeedError):
        feeds.add_source(conn, "http://example.com/not-a-feed", parse=_parse_bad)

    assert conn.execute("SELECT COUNT(*) n FROM feeds").fetchone()["n"] == 0


def test_add_feed_is_idempotent(conn):
    first_id, _ = feeds.add_source(conn, "http://example.com/rss", parse=_parse_ok)
    second_id, _ = feeds.add_source(conn, "http://example.com/rss", parse=_parse_ok)

    assert second_id == first_id
    assert conn.execute("SELECT COUNT(*) n FROM feeds").fetchone()["n"] == 1


def test_list_feeds_reports_item_counts(conn):
    fid, _ = feeds.add_source(conn, "http://example.com/rss", parse=_parse_ok)
    conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (?, 't', 'u', 's', 'h1')",
        (fid,),
    )
    conn.commit()

    listed = feeds.list_sources(conn)

    assert len(listed) == 1
    assert listed[0]["item_count"] == 1
    assert listed[0]["feed_id"] == fid


def test_remove_feed_orphans_items_and_preserves_clicks(conn):
    fid, _ = feeds.add_source(conn, "http://example.com/rss", parse=_parse_ok)
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

    orphaned, message = feeds.remove_source(conn, fid)

    assert conn.execute("SELECT COUNT(*) n FROM feeds").fetchone()["n"] == 0
    # items and the click that trained the ranker both survive
    assert conn.execute("SELECT COUNT(*) n FROM items").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) n FROM clicks").fetchone()["n"] == 1
    assert orphaned == 1
    assert "unsubscribed" in message.lower()


def test_remove_feed_unknown_id_raises(conn):
    with pytest.raises(feeds.FeedError):
        feeds.remove_source(conn, 999)


def test_preview_feed_does_not_subscribe(conn):
    out = feeds.preview_source("http://example.com/rss", limit=1, parse=_parse_ok)

    assert "ok" not in out
    assert len(out["entries"]) == 1
    assert conn.execute("SELECT COUNT(*) n FROM feeds").fetchone()["n"] == 0


def test_two_subscriptions_to_one_url_do_not_both_error(tmp_path):
    """Check-then-insert with a NETWORK FETCH in the window.

    add_source reads `SELECT id FROM feeds WHERE url = ?`, then parses the feed
    over the network, then inserts -- so the gap between check and write is a
    whole HTTP round trip, the widest of the four sites sharing this shape.
    Measured: 8 concurrent calls gave 7 IntegrityErrors and 1 success, where
    the serial path returns an idempotent "already subscribed to ...".

    Losing the race means the subscription the caller asked for exists, which
    is the same outcome the serial path calls success.
    """
    import concurrent.futures
    from types import SimpleNamespace

    from attestation.feeds import add_source

    db = tmp_path / "t.db"
    seeded_db(db).commit()

    def parse(_url):
        return SimpleNamespace(
            entries=[{"title": "t", "summary": "s", "id": "g1", "link": "u"}],
            feed={"title": "A Feed"},
        )

    def subscribe(_):
        try:
            feed_id, message = add_source(seeded_db(db), "http://example.invalid/feed", parse=parse)
            return {"ok": True, "message": message}
        except Exception as exc:  # noqa: BLE001 -- the point is what leaks out
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        results = list(pool.map(subscribe, range(8)))

    failed = [r["message"] for r in results if not r["ok"]]
    assert not failed, f"{len(failed)} of 8 concurrent subscribes failed: {failed[0]}"
    rows = seeded_db(db).execute("SELECT COUNT(*) FROM feeds").fetchone()[0]
    assert rows == 1, f"{rows} feed rows for one url"


def test_suggest_feeds_ranks_by_tag_overlap_ties_by_title():
    """_score_candidates is pure: no DB, no TOML, just sets and a list -- the
    seam the loop inside suggest_feeds became once its queries were split out."""
    from attestation.feeds import _score_candidates

    cands = [
        {"url": "a", "title": "Z", "tags": ["nlp"]},
        {"url": "b", "title": "A", "tags": ["nlp"]},
        {"url": "c", "title": "M", "tags": ["nlp", "rl"]},
        {"url": "d", "title": "S", "tags": ["rl"]},
    ]
    out = _score_candidates({"nlp", "rl"}, {"d"}, cands, limit=5)
    assert [o["title"] for o in out] == ["M", "A", "Z"]
    assert out[0]["score"] == 2 and out[0]["matched_tags"] == ["nlp", "rl"]


def test_feeds_functions_raise_rather_than_return_ok(tmp_path):
    """add_source raises FeedError on an unparseable URL instead of returning
    {"ok": False, ...} -- the MCP layer maps FeedError to ToolError, so the
    envelope is built in exactly one place instead of twice."""
    from attestation.feeds import FeedError, add_source

    conn = seeded_db(tmp_path / "t.db")

    class NotAFeed:
        entries = []
        feed = {}
        bozo = 1

    with pytest.raises(FeedError, match="did not parse"):
        add_source(conn, "http://nope", parse=lambda url: NotAFeed())


def test_feeds_module_names_match_the_source_tool_vocabulary():
    """feeds.py's public functions back the feed.source_* / feed.sources tools
    one-for-one; none should still be named *_feed/*_feeds -- "feed" is
    ambiguous with the ranked-item product in mcp/feed.py, "source" is not."""
    public = {n for n in dir(feeds) if not n.startswith("_") and callable(getattr(feeds, n))}
    assert not [n for n in public if n.endswith(("feed", "feeds"))], public
