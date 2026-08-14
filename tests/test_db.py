import sqlite3
from pathlib import Path

from attestation.db import get_db, resolve_db_path


def test_get_db_creates_schema(tmp_path):
    conn = get_db(tmp_path / "test.db")
    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    }
    assert {"users", "feeds", "items", "clicks", "explanations"} <= tables
    # vec0 virtual table accepts a 256-dim float32 blob keyed by rowid
    import numpy as np

    vec = np.zeros(256, dtype=np.float32)
    conn.execute("INSERT INTO items(feed_id, title, content_hash) VALUES (NULL, 't', 'h')")
    item_id = conn.execute("SELECT id FROM items").fetchone()["id"]
    conn.execute(
        "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)", (item_id, vec.tobytes())
    )
    row = conn.execute("SELECT rowid FROM item_vectors").fetchone()
    assert row["rowid"] == item_id


def test_get_db_pragmas(tmp_path):
    conn = get_db(tmp_path / "test.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_seed_users_idempotent(tmp_path):
    path = tmp_path / "test.db"
    get_db(path).close()
    conn = get_db(path)  # second open must not duplicate
    users = conn.execute("SELECT name, interests FROM users ORDER BY name").fetchall()
    assert [u["name"] for u in users] == ["bench-chemist", "matt", "ml-engineer"]
    assert all(u["interests"] for u in users)


def test_click_unique_per_user_item(tmp_path):
    conn = get_db(tmp_path / "test.db")
    conn.execute("INSERT INTO items(feed_id, title, content_hash) VALUES (NULL, 't', 'h')")
    conn.execute("INSERT INTO clicks(user_id, item_id, useful) VALUES (1, 1, 1)")
    import pytest

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO clicks(user_id, item_id, useful) VALUES (1, 1, 0)")


def test_resolve_db_path_explicit_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("RSS_DB", str(tmp_path / "env.db"))
    assert resolve_db_path(str(tmp_path / "explicit.db")) == Path(tmp_path / "explicit.db")


def test_resolve_db_path_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("RSS_DB", raising=False)
    monkeypatch.setenv("RSS_DB", str(tmp_path / "env.db"))
    assert resolve_db_path(None) == Path(tmp_path / "env.db")


def test_resolve_db_path_skill_data_dir_if_present(tmp_path, monkeypatch):
    monkeypatch.delenv("RSS_DB", raising=False)
    fake_home = tmp_path / "home"
    skill_data_dir = fake_home / ".hermes" / "skills" / "science-recommendations" / "data"
    skill_data_dir.mkdir(parents=True)
    skill_db = skill_data_dir / "hermes.db"
    skill_db.write_text("")  # must exist to be selected
    monkeypatch.setattr("attestation.db.SKILL_DATA_DB", skill_db)
    assert resolve_db_path(None) == skill_db


def test_resolve_db_path_falls_back_to_cwd_default(tmp_path, monkeypatch):
    monkeypatch.delenv("RSS_DB", raising=False)
    fake_home = tmp_path / "home"
    skill_db = fake_home / ".hermes" / "skills" / "science-recommendations" / "data" / "hermes.db"
    monkeypatch.setattr("attestation.db.SKILL_DATA_DB", skill_db)  # deliberately does not exist
    assert resolve_db_path(None) == Path("hermes.db")


def test_feature_tables_exist_and_accept_rows(tmp_path):
    conn = get_db(tmp_path / "t.db")
    cur = conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (NULL, 'a paper', 'http://x', 's', 'h1')"
    )
    item_id = cur.lastrowid
    conn.execute(
        "INSERT INTO item_features(item_id, content_type, model) VALUES (?, 'paper', 'testmodel')",
        (item_id,),
    )
    conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, 'quantum-chemistry')", (item_id,))
    # duplicate tag for same item must be a PK violation, not silently allowed
    import sqlite3 as _sq

    import pytest as _pt

    with _pt.raises(_sq.IntegrityError):
        conn.execute(
            "INSERT INTO item_tags(item_id, tag) VALUES (?, 'quantum-chemistry')", (item_id,)
        )
    row = conn.execute("SELECT content_type, model, tagged_at FROM item_features").fetchone()
    assert row["content_type"] == "paper"
    assert row["tagged_at"]  # default populated


def test_embed_dims_env(monkeypatch):
    from attestation.db import embed_dims

    monkeypatch.delenv("EMBED_DIMS", raising=False)
    assert embed_dims() == 256
    monkeypatch.setenv("EMBED_DIMS", "512")
    assert embed_dims() == 512


def test_vec_schema_follows_dims_env(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBED_DIMS", "128")
    conn = get_db(tmp_path / "small.db")
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'item_vectors'").fetchone()[
        "sql"
    ]
    assert "float[128]" in sql


def test_dims_mismatch_refuses_to_open(tmp_path, monkeypatch):
    import pytest

    monkeypatch.delenv("EMBED_DIMS", raising=False)
    get_db(tmp_path / "t.db").close()  # created at default 256
    monkeypatch.setenv("EMBED_DIMS", "512")
    with pytest.raises(RuntimeError, match=r"float\[256\].*EMBED_DIMS=512"):
        get_db(tmp_path / "t.db")


def test_resolve_db_path_reads_unprefixed_env(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_RSS_DB", raising=False)
    monkeypatch.setenv("RSS_DB", str(tmp_path / "x.db"))
    assert resolve_db_path(None) == tmp_path / "x.db"


def test_migration_adds_source_to_existing_clicks_db(tmp_path):
    """A pre-existing DB with the old clicks schema gains source='ui' without data loss."""
    import sqlite3

    from attestation.db import get_db

    path = tmp_path / "old.db"
    old = sqlite3.connect(str(path))
    old.executescript(
        """
        CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, interests TEXT);
        -- Mirrors the real items schema (not trimmed): get_db's SCHEMA indexes
        -- content_hash, and CREATE TABLE IF NOT EXISTS won't widen a table that
        -- already exists, so a trimmed items table breaks get_db before it ever
        -- reaches _migrate.
        CREATE TABLE items(
          id INTEGER PRIMARY KEY,
          feed_id INTEGER,
          guid TEXT,
          title TEXT,
          url TEXT,
          summary TEXT,
          published TEXT,
          content_hash TEXT NOT NULL
        );
        CREATE TABLE clicks(
          id INTEGER PRIMARY KEY,
          user_id INTEGER NOT NULL,
          item_id INTEGER NOT NULL,
          useful INTEGER NOT NULL,
          clicked_at TEXT NOT NULL DEFAULT (datetime('now')),
          UNIQUE(user_id, item_id)
        );
        INSERT INTO users(id, name) VALUES (1, 'matt');
        INSERT INTO items(id, title, content_hash) VALUES (1, 'a', 'h1'), (2, 'b', 'h2');
        INSERT INTO clicks(user_id, item_id, useful) VALUES (1, 1, 1), (1, 2, 0);
        """
    )
    old.commit()
    old.close()

    conn = get_db(path)

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(clicks)")}
    assert "source" in cols
    rows = conn.execute(
        "SELECT user_id, item_id, useful, source FROM clicks ORDER BY item_id"
    ).fetchall()
    assert len(rows) == 2, "migration lost rows"
    assert [r["source"] for r in rows] == ["ui", "ui"]
    assert [r["useful"] for r in rows] == [1, 0]
    conn.close()


def test_migration_is_idempotent(tmp_path):
    """Running get_db twice must not fail or duplicate the column."""
    from attestation.db import get_db

    path = tmp_path / "fresh.db"
    get_db(path).close()
    conn = get_db(path)

    cols = [r["name"] for r in conn.execute("PRAGMA table_info(clicks)")]
    assert cols.count("source") == 1
    conn.close()
