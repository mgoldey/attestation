"""Migration 007: the library tables, the second vec table, items.doi/arxiv_id."""

import sqlite3
from pathlib import Path

from attestation import db as dbmod


def _v6_database(path):
    """A database as migration 006 left it: no library tables, items without ids."""
    conn = dbmod.get_db(path)
    conn.execute("INSERT INTO feeds(url, title) VALUES ('http://a', 'arXiv chem-ph')")
    conn.execute(
        "INSERT INTO items(feed_id, guid, title, url, summary, content_hash)"
        " VALUES (1, 'oai:arXiv.org:1003.0563v2', 't1',"
        " 'https://arxiv.org/abs/1003.0563', '', 'h1')"
    )
    conn.execute(
        "INSERT INTO items(feed_id, guid, title, url, summary, content_hash)"
        " VALUES (1, 'https://www.nature.com/articles/s41467-026-74391-4', 't2',"
        " 'https://www.nature.com/articles/s41467-026-74391-4', '', 'h2')"
    )
    conn.execute(
        "INSERT INTO items(feed_id, guid, title, url, summary, content_hash)"
        " VALUES (1, 'x', 't3', 'https://news.ycombinator.com/item?id=1', '', 'h3')"
    )
    conn.commit()
    # Rewind to v6 by dropping what 007 adds and resetting user_version.
    for stmt in (
        "DROP TRIGGER IF EXISTS trg_references_delete_vector",
        'DROP TABLE IF EXISTS "references"',
        "DROP TABLE IF EXISTS reference_sources",
        "DROP TABLE IF EXISTS reference_tags",
        "DROP TABLE IF EXISTS reference_cites",
        "DROP TABLE IF EXISTS reference_vectors",
        "ALTER TABLE items DROP COLUMN doi",
        "ALTER TABLE items DROP COLUMN arxiv_id",
        "PRAGMA user_version = 6",
    ):
        conn.execute(stmt)
    conn.commit()
    conn.close()
    raw = sqlite3.connect(path)
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 6
    assert "doi" not in {r[1] for r in raw.execute("PRAGMA table_info(items)")}
    raw.close()


def test_migration_007_adds_the_library_and_backfills_item_ids(tmp_path):
    path = tmp_path / "v6.db"
    _v6_database(path)
    conn = dbmod.get_db(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == dbmod.SCHEMA_VERSION
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "references",
        "reference_sources",
        "reference_tags",
        "reference_cites",
        "reference_vectors",
    } <= tables
    rows = conn.execute("SELECT title, doi, arxiv_id FROM items ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [
        ("t1", None, "1003.0563"),
        ("t2", "10.1038/s41467-026-74391-4", None),
        ("t3", None, None),
    ]
    # Reopening is a no-op: the backfill must not run twice.
    conn.close()
    conn = dbmod.get_db(path)
    n = conn.execute("SELECT count(*) FROM items WHERE arxiv_id = '1003.0563'").fetchone()[0]
    assert n == 1


def test_reference_vectors_follow_their_reference(tmp_path):
    conn = dbmod.get_db(tmp_path / "t.db")
    conn.execute(
        'INSERT INTO "references"(id, identity, title, first_seen, updated)'
        " VALUES (7, 'doi:x', 'T', '2026-09-05', '2026-09-05')"
    )
    conn.execute(
        "INSERT INTO reference_vectors(rowid, embedding) VALUES (7, ?)",
        (b"\x00" * 4 * dbmod.embed_dims(),),
    )
    conn.execute('DELETE FROM "references" WHERE id = 7')
    assert conn.execute("SELECT count(*) FROM reference_vectors").fetchone()[0] == 0


def test_a_fresh_database_has_the_counts_claude_md_states(tmp_path):
    """16 application tables; 27 rows in sqlite_master with the two vec0 tables,
    their four shadow tables each, and sqlite_sequence. CLAUDE.md's Storage
    line quotes these numbers and was found stale at 12/17 (review round 1)."""
    conn = dbmod.get_db(tmp_path / "fresh.db")
    names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    app = [n for n in names if not n.startswith(("item_vectors", "reference_vectors", "sqlite_"))]
    assert (len(names), len(app)) == (27, 16)
    claude_md = (Path(__file__).resolve().parents[1] / "CLAUDE.md").read_text()
    assert "16 APPLICATION tables" in claude_md and "a fresh file has 27" in claude_md


def test_a_fresh_database_has_the_library_tables(tmp_path):
    conn = dbmod.get_db(tmp_path / "fresh.db")
    cols = {r["name"] for r in conn.execute('PRAGMA table_info("references")')}
    assert {
        "identity",
        "doi",
        "arxiv_id",
        "title",
        "authors",
        "year",
        "venue",
        "abstract",
        "url",
        "bib_key",
        "first_seen",
        "updated",
    } <= cols
    assert {r["name"] for r in conn.execute("PRAGMA table_info(items)")} >= {"doi", "arxiv_id"}


def test_a_vec_dims_mismatch_is_refused_for_either_table(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    dbmod.get_db(path).close()
    monkeypatch.setenv("EMBED_DIMS", "128")
    try:
        dbmod.get_db(path)
    except RuntimeError as exc:
        assert "item_vectors" in str(exc)
    else:  # pragma: no cover - the assertion above is the test
        raise AssertionError("mismatched dims were accepted")
