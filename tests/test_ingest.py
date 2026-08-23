from pathlib import Path
from types import SimpleNamespace

import feedparser
import pytest

from attestation.db import get_db
from attestation.ingest import content_hash, run_ingest, strip_boilerplate, sync_feeds

FIXTURES = Path(__file__).parent / "fixtures"


def fake_parse(url: str):
    name = "arxiv.xml" if "arxiv" in url else "blog.xml"
    return feedparser.parse((FIXTURES / name).read_text())


def write_feeds_toml(tmp_path, urls):
    lines = []
    for u in urls:
        lines += ["[[feeds]]", f'url = "{u}"', f'title = "{u}"', ""]
    p = tmp_path / "feeds.toml"
    p.write_text("\n".join(lines))
    return p


def test_strip_boilerplate_arxiv():
    raw = "arXiv:2608.00001v1 Announce Type: new\nAbstract: We study things."
    assert strip_boilerplate(raw) == "We study things."


def test_strip_boilerplate_html():
    assert strip_boilerplate("<p>Hello <b>world</b></p>") == "Hello world"


def test_ingest_adds_items_and_vectors(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    feeds = write_feeds_toml(tmp_path, ["https://arxiv.example/rss", "https://blog.example/rss"])
    stats = run_ingest(conn, fake_embedder, feeds, parse=fake_parse)
    assert stats == {"added": 3, "skipped": 0, "failed_feeds": 0}
    assert conn.execute("SELECT COUNT(*) c FROM items").fetchone()["c"] == 3
    assert conn.execute("SELECT COUNT(*) c FROM item_vectors").fetchone()["c"] == 3
    # boilerplate stripped before storage
    summary = conn.execute("SELECT summary FROM items WHERE title LIKE 'Paper One%'").fetchone()[
        "summary"
    ]
    assert "Announce Type" not in summary


def test_ingest_idempotent_by_guid(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    feeds = write_feeds_toml(tmp_path, ["https://arxiv.example/rss"])
    run_ingest(conn, fake_embedder, feeds, parse=fake_parse)
    stats = run_ingest(conn, fake_embedder, feeds, parse=fake_parse)
    assert stats["added"] == 0 and stats["skipped"] == 2


def test_ingest_dedup_by_hash_when_no_guid(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    feeds = write_feeds_toml(tmp_path, ["https://blog.example/rss"])
    run_ingest(conn, fake_embedder, feeds, parse=fake_parse)
    stats = run_ingest(conn, fake_embedder, feeds, parse=fake_parse)
    assert stats["added"] == 0 and stats["skipped"] == 1


def test_feed_failure_isolated(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    feeds = write_feeds_toml(tmp_path, ["https://boom.example/rss", "https://blog.example/rss"])

    def exploding_parse(url):
        if "boom" in url:
            raise RuntimeError("connection refused")
        return fake_parse(url)

    stats = run_ingest(conn, fake_embedder, feeds, parse=exploding_parse)
    assert stats["failed_feeds"] == 1 and stats["added"] == 1


def test_content_hash_stable():
    assert content_hash("a", "b") == content_hash("a", "b")
    assert content_hash("a", "b") != content_hash("a", "c")


def test_embed_calls_do_not_hold_write_lock(tmp_path, fake_embedder):
    """While embed_document runs (the slow HTTP call), a second connection
    must be able to write to the db without hitting SQLITE_BUSY -- proves no
    write transaction is held open across the embed calls."""
    conn = get_db(tmp_path / "t.db")
    feeds = write_feeds_toml(tmp_path, ["https://arxiv.example/rss"])

    other_conn = get_db(tmp_path / "t.db")

    class ProbingEmbedder:
        def embed_document(self, title, text):
            # A concurrent writer must be able to commit right now.
            other_conn.execute(
                "INSERT INTO feeds(url, title) VALUES (?, ?)",
                (f"https://probe.example/{title}", "probe"),
            )
            other_conn.commit()
            return fake_embedder.embed_document(title, text)

        def embed_query(self, text):
            return fake_embedder.embed_query(text)

    stats = run_ingest(conn, ProbingEmbedder(), feeds, parse=fake_parse)
    assert stats["failed_feeds"] == 0
    assert stats["added"] == 2
    probe_count = conn.execute(
        "SELECT COUNT(*) c FROM feeds WHERE url LIKE 'https://probe.example/%'"
    ).fetchone()["c"]
    assert probe_count == 2


def test_sync_feeds_missing_path_raises_clear_message(tmp_path):
    conn = get_db(tmp_path / "t.db")
    missing = tmp_path / "does-not-exist.toml"
    with pytest.raises(FileNotFoundError, match=r"does-not-exist\.toml"):
        sync_feeds(conn, missing)


def test_cli_default_feeds_resolves_packaged_copy_with_no_local_checkout(tmp_path, monkeypatch):
    """From an empty directory (no local feeds.toml -- e.g. after a wheel
    install with no checkout), the CLI's --feeds default must resolve to the
    packaged copy shipped inside src/attestation, not a bare relative string
    that dies with FileNotFoundError."""
    from attestation.cli import _default_feeds_path

    monkeypatch.chdir(tmp_path)
    resolved = Path(_default_feeds_path())
    assert resolved.exists()
    assert resolved.name == "feeds.toml"
    assert resolved.parent.name == "attestation"


def test_cli_default_feeds_prefers_local_checkout_copy(tmp_path, monkeypatch):
    """When a feeds.toml exists in cwd (dev checkout), it takes priority over
    the packaged copy -- preserves current dev workflow."""
    from attestation.cli import _default_feeds_path

    monkeypatch.chdir(tmp_path)
    local = tmp_path / "feeds.toml"
    local.write_text('[[feeds]]\nurl = "https://example.com/rss"\n')
    assert Path(_default_feeds_path()).resolve() == local.resolve()


class _Entries:
    def __init__(self, entries):
        self.entries = entries


def _entry(i, guid=None, summary=None):
    return {
        "title": f"Post {i}",
        "summary": summary or f"body {i}",
        "id": guid or f"g{i}",
        "link": f"http://x/{i}",
    }


def test_one_duplicate_guid_does_not_discard_the_whole_feed(tmp_path, fake_embedder):
    """A repeated GUID must cost one item, not the batch.

    _exists() checks the database, so it cannot see an earlier entry from the
    SAME batch. Both copies passed pass 1, the second hit the UNIQUE
    constraint in pass 3, and the broad except rolled back every good item
    inserted before it -- while stats["added"] had already counted them and
    was never rolled back. The caller was told 5 items were added when zero
    were stored, last_fetched stayed NULL, and the next run re-fetched and
    re-embedded the same feed to fail identically. Republished posts and
    misconfigured CMSes make duplicate GUIDs ordinary.
    """
    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO feeds(url, title) VALUES ('http://x', 'X')")
    conn.commit()
    feeds_toml = write_feeds_toml(tmp_path, [])

    entries = [_entry(i) for i in range(5)]
    entries.append(_entry(99, guid="g0", summary="a different body"))

    stats = run_ingest(conn, fake_embedder, feeds_toml, parse=lambda url: _Entries(entries))

    stored = conn.execute("SELECT COUNT(*) n FROM items").fetchone()["n"]
    assert stored == 5, "the five good items must survive one duplicate"
    assert stats["added"] == stored, f"stats claimed {stats['added']} added but {stored} rows exist"
    assert stats["failed_feeds"] == 0, "one duplicate entry is not a feed failure"
    assert conn.execute("SELECT last_fetched FROM feeds").fetchone()["last_fetched"] is not None
    conn.close()


def test_a_duplicate_content_hash_within_one_batch_is_skipped(tmp_path, fake_embedder):
    """Same defence for the other dedup key: two entries with different GUIDs
    but identical title+summary are one item, not two."""
    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO feeds(url, title) VALUES ('http://x', 'X')")
    conn.commit()
    feeds_toml = write_feeds_toml(tmp_path, [])

    entries = [_entry(1), _entry(2, guid="different-guid", summary="body 1")]
    entries[1]["title"] = "Post 1"

    stats = run_ingest(conn, fake_embedder, feeds_toml, parse=lambda url: _Entries(entries))

    stored = conn.execute("SELECT COUNT(*) n FROM items").fetchone()["n"]
    assert stored == 1
    assert stats["added"] == 1
    assert stats["skipped"] == 1


def test_stats_never_counts_an_item_that_was_rolled_back(tmp_path, fake_embedder):
    """A genuine mid-write failure must leave the counter honest too.

    Duplicate handling is fixed above, but any unexpected error in pass 3
    still rolls back. The count must reflect what committed, not what was
    attempted -- a caller that trusts `added` has no other way to find out.
    """
    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO feeds(url, title) VALUES ('http://x', 'X')")
    conn.commit()
    feeds_toml = write_feeds_toml(tmp_path, [])

    class ExplodingConn:
        """Proxies the real connection but fails the fourth item insert."""

        def __init__(self, inner):
            self._inner = inner
            self._inserts = 0

        def execute(self, sql, *args):
            if sql.startswith("INSERT INTO items"):
                self._inserts += 1
                if self._inserts > 3:
                    raise RuntimeError("disk full")
            return self._inner.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    stats = run_ingest(
        ExplodingConn(conn),
        fake_embedder,
        feeds_toml,
        parse=lambda url: _Entries([_entry(i) for i in range(5)]),
    )

    stored = conn.execute("SELECT COUNT(*) n FROM items").fetchone()["n"]
    assert stats["added"] == stored, (
        f"stats claimed {stats['added']} added but {stored} rows survived the rollback"
    )
    assert stats["failed_feeds"] == 1
    conn.close()


# ---------------------------------------------------------------------------
# Diagnosing the embedder, not blaming the feed
# ---------------------------------------------------------------------------


class _DeadEmbedder:
    """An embedder whose backend is not running -- the commonest first-run state."""

    dims = 256

    def embed_document(self, title, text):
        import httpx

        raise httpx.ConnectError("[Errno 111] Connection refused")

    def embed_query(self, text):
        return self.embed_document("", text)


def test_a_dead_embedder_is_named_once_not_blamed_on_every_feed(tmp_path, caplog):
    """Measured first-run failure: with Ollama down, ingest printed 22 full
    tracebacks -- one per feed -- each headed "feed failed: <url>", and exited 0.

    Three faults compounding. The stack dump is httpx internals for an expected,
    diagnosable condition. The message blames the FEED, sending a new user to
    debug their network or feeds.toml when the embedding model is what is down.
    And a zero exit means the documented cron line (`attest ingest && attest
    tag`) chains happily and produces nothing, forever, silently.

    `attest install --check` already diagnoses this correctly. Ingest just never
    asked.
    """
    import logging

    conn = get_db(tmp_path / "t.db")
    feeds = tmp_path / "feeds.toml"
    feeds.write_text(
        '[[feeds]]\nurl = "http://a.example/f"\ntitle = "A"\n'
        '[[feeds]]\nurl = "http://b.example/f"\ntitle = "B"\n'
        '[[feeds]]\nurl = "http://c.example/f"\ntitle = "C"\n'
    )

    def parse(url):
        return SimpleNamespace(
            entries=[{"title": "t", "summary": "s", "id": f"{url}#1", "link": f"{url}#1"}],
            feed=SimpleNamespace(title="T"),
        )

    with caplog.at_level(logging.WARNING):
        stats = run_ingest(conn, _DeadEmbedder(), feeds, parse=parse)

    text = caplog.text
    # caplog.text does not render exc_info, so check the records directly:
    # log.exception() sets exc_info, log.warning() does not, and it was the
    # former (22 times) that produced ~880 lines of httpx internals.
    assert not any(r.exc_info for r in caplog.records), (
        "an expected, diagnosable condition logged a stack trace"
    )
    assert text.lower().count("embedding") >= 1, "never named the embedder as the cause"
    # Said once, not once per feed.
    assert text.count("unreachable") <= 1, f"repeated the same diagnosis per feed:\n{text}"
    assert stats["failed_feeds"] >= 1
    assert stats.get("embedder_down") is True, "the caller cannot tell this apart from bad feeds"


def test_one_broken_feed_still_reports_per_feed(tmp_path, caplog, fake_embedder):
    """The embedder shortcut must not swallow ordinary per-feed failures --
    a 404 on one feed is still that feed's problem and the others must run."""
    import logging

    conn = get_db(tmp_path / "t.db")
    feeds = tmp_path / "feeds.toml"
    feeds.write_text('[[feeds]]\nurl = "http://bad.example/f"\ntitle = "Bad"\n')

    def parse(url):
        raise ValueError("malformed feed document")

    with caplog.at_level(logging.WARNING):
        stats = run_ingest(conn, fake_embedder, feeds, parse=parse)

    assert stats["failed_feeds"] == 1
    assert not stats.get("embedder_down")
    assert "bad.example" in caplog.text, "a genuine feed failure stopped naming the feed"
    # No stack trace even here. caplog.text omits exc_info, so inspect the
    # records: log.exception sets it, log.warning does not. A traceback for an
    # expected condition is what trained people to ignore this output.
    assert not any(r.exc_info for r in caplog.records), (
        "a malformed feed dumped a stack trace instead of naming the problem"
    )
    assert "malformed feed document" in caplog.text, "the reason was not reported"


def test_one_unencodable_entry_does_not_discard_the_whole_feed(tmp_path, fake_embedder, caplog):
    """The blast-radius bug this module already fixed once, in a second place.

    `_new_entries`' docstring records it: "two entries sharing a GUID both
    passed, the second hit the UNIQUE constraint during the write, and the
    rollback discarded every good item alongside it -- a whole feed lost to one
    republished post." The per-entry guard was added for duplicates and not for
    encoding failures.

    `content_hash` calls .encode() on title+summary, so a lone surrogate raises
    UnicodeEncodeError, unwinds past every good entry to the per-feed handler,
    and rolls the batch back. Reachable from ordinary feed data: a bare
    `&#xD800;` character reference is enough.
    """
    import logging

    conn = get_db(tmp_path / "t.db")
    feeds = tmp_path / "feeds.toml"
    feeds.write_text('[[feeds]]\nurl = "http://a.example/f"\ntitle = "A"\n')

    def parse(url):
        entries = [
            {"title": f"Post {i}", "summary": f"body {i}", "id": f"g{i}", "link": f"http://x/{i}"}
            for i in range(9)
        ]
        entries.insert(
            4,
            {
                "title": "bad \ud800 entry",
                "summary": "s",
                "id": "gbad",
                "link": "http://x/bad",
            },
        )
        return SimpleNamespace(entries=entries, feed=SimpleNamespace(title="A"))

    with caplog.at_level(logging.WARNING):
        stats = run_ingest(conn, fake_embedder, feeds, parse=parse)

    stored = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    assert stored == 9, f"one bad entry cost {9 - stored} good ones"
    assert stats["added"] == 9
    assert stats["failed_feeds"] == 0, "a per-entry problem was reported as a feed failure"
