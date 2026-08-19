from pathlib import Path

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
