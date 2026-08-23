import sqlite3
from pathlib import Path

import pytest

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


def test_deleted_persona_stays_deleted_across_reopen(tmp_path):
    """A demo persona removed from the DB must not be resurrected by the next get_db().

    Regression test: get_db used to run INSERT OR IGNORE for SEED_USERS on
    every open, so a deleted seed persona (e.g. 'bench-chemist') came back on
    the very next connection -- which every MCP tool opens fresh.
    """
    path = tmp_path / "personas.db"
    conn = get_db(path)
    names_before = {r["name"] for r in conn.execute("SELECT name FROM users")}
    assert "bench-chemist" in names_before

    conn.execute("DELETE FROM users WHERE name = ?", ("bench-chemist",))
    conn.commit()
    conn.close()

    reopened = get_db(path)
    names_after = {r["name"] for r in reopened.execute("SELECT name FROM users")}
    assert "bench-chemist" not in names_after
    assert names_after == {"matt", "ml-engineer"}
    reopened.close()


def test_seed_demo_users_is_explicit_and_idempotent(tmp_path):
    """seed_demo_users() inserts SEED_USERS and is safe to call repeatedly."""
    from attestation.db import seed_demo_users

    conn = get_db(tmp_path / "explicit.db")
    conn.execute("DELETE FROM users WHERE name = ?", ("ml-engineer",))
    conn.commit()

    seed_demo_users(conn)
    names = {r["name"] for r in conn.execute("SELECT name FROM users")}
    assert names == {"matt", "bench-chemist", "ml-engineer"}

    # calling again must not duplicate or raise
    seed_demo_users(conn)
    rows = conn.execute("SELECT name, COUNT(*) AS n FROM users GROUP BY name").fetchall()
    assert all(r["n"] == 1 for r in rows)
    conn.close()


def test_schema_version_set_on_fresh_db(tmp_path):
    """A freshly created database records a nonzero PRAGMA user_version."""
    from attestation.db import SCHEMA_VERSION

    conn = get_db(tmp_path / "versioned.db")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION
    assert version > 0
    conn.close()


def test_schema_version_persists_across_reopen(tmp_path):
    path = tmp_path / "versioned.db"
    get_db(path).close()
    conn = get_db(path)
    from attestation.db import SCHEMA_VERSION

    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()


def test_old_db_without_user_version_migrates_and_stamps_version(tmp_path):
    """A pre-existing DB (created before versioning existed, user_version=0)
    still gets the clicks.source migration applied and ends up stamped at
    SCHEMA_VERSION -- the ladder must not require a special-cased first run.
    """
    from attestation.db import SCHEMA_VERSION

    path = tmp_path / "unversioned.db"
    old = sqlite3.connect(str(path))
    old.executescript(
        """
        CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, interests TEXT);
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
        """
    )
    old.commit()
    old.close()

    conn = get_db(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(clicks)")}
    assert "source" in cols
    conn.close()


def test_refuses_to_open_newer_schema_version(tmp_path, monkeypatch):
    """A database stamped with a user_version newer than this code supports
    must refuse to open rather than silently proceeding against an unknown
    schema shape.
    """
    import pytest

    import attestation.db as db_module

    path = tmp_path / "future.db"
    get_db(path).close()

    conn = sqlite3.connect(str(path))
    conn.execute(f"PRAGMA user_version = {db_module.SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="newer than this code supports"):
        get_db(path)


def test_corpora_tables_exist(tmp_path):
    """A corpus is its own row, not a column on runs.

    Twelve runs sharing WikiText-2 should point at one inspectable row that
    carries the corpus's own attributes -- source, tokenizer, fingerprint --
    rather than duplicating them per run or losing them.
    """
    conn = get_db(str(tmp_path / "h.db"))
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"corpora", "corpus_splits"} <= tables, sorted(tables)

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(corpora)")}
    assert {"name", "source", "tokenizer", "vocab_size", "seq_len"} <= cols, sorted(cols)
    # fingerprint_kind must exist alongside fingerprint: hashing a directory of
    # shards and hashing one .txt are different claims, and a size+mtime check
    # must never be reportable as a content hash.
    assert {"fingerprint", "fingerprint_kind"} <= cols, sorted(cols)


def test_runs_carry_a_nullable_corpus_id(tmp_path):
    """NULL means "the artifact did not say" -- never "no corpus", never
    "the default corpus". Most existing artifacts record nothing about data,
    so unknown is the common case and must be representable."""
    conn = get_db(str(tmp_path / "h.db"))
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)")}
    assert "corpus_id" in cols, sorted(cols)

    conn.execute("INSERT INTO runs(project, name, source_path) VALUES ('p', 'r', '/tmp/r.json')")
    row = conn.execute("SELECT corpus_id FROM runs WHERE name = 'r'").fetchone()
    assert row["corpus_id"] is None


def test_existing_db_migrates_to_corpora(tmp_path):
    """An old database gains the corpus tables without losing its runs."""
    path = tmp_path / "h.db"
    conn = get_db(str(path))
    conn.execute("INSERT INTO runs(project, name, source_path) VALUES ('p', 'old', '/tmp/o.json')")
    conn.commit()
    # Rewind to before the corpus migration and drop what it adds.
    conn.execute("DROP TABLE IF EXISTS corpus_splits")
    conn.execute("DROP TABLE IF EXISTS corpora")
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    conn = get_db(str(path))
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"corpora", "corpus_splits"} <= tables
    assert conn.execute("SELECT COUNT(*) c FROM runs").fetchone()["c"] == 1


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------


def test_backup_captures_writes_still_in_the_wal(tmp_path):
    """`cp hermes.db` is the obvious backup and it silently loses data.

    get_db sets journal_mode=WAL, so recent commits live in `hermes.db-wal`
    until a checkpoint. A copy of the main file alone opens fine, looks fine,
    and is missing the newest rows -- which is worse than no backup, because it
    looks trustworthy. Five such copies existed in the real data directory when
    this was written, made by hand in earlier sessions.

    VACUUM INTO is the fix: it writes one consistent file including the WAL,
    without stopping writers.
    """
    from attestation.db import backup_db

    src = tmp_path / "live.db"
    conn = get_db(src)
    for i in range(200):  # enough writes that the WAL is not trivially empty
        conn.execute("INSERT INTO users(name, interests) VALUES (?, 'analysis')", (f"u{i}",))
    conn.commit()

    naive = tmp_path / "naive.db"
    naive.write_bytes(src.read_bytes())  # what an operator would type
    dest = backup_db(conn, tmp_path / "backup.db")

    live_users = conn.execute("SELECT count(*) FROM users").fetchone()[0]
    backup_users = sqlite3.connect(dest).execute("SELECT count(*) FROM users").fetchone()[0]
    assert backup_users == live_users, "the backup lost rows the live database has"

    # The naive copy is worse than "missing recent rows": measured here, it has
    # no schema at all, because the CREATE TABLEs were still in the WAL too. It
    # is a valid, openable, empty SQLite file.
    try:
        naive_users = sqlite3.connect(naive).execute("SELECT count(*) FROM users").fetchone()[0]
    except sqlite3.OperationalError:
        naive_users = None
    assert naive_users != live_users, (
        "a plain file copy captured everything, so this test proves nothing --"
        " the WAL must be non-empty for the comparison to mean anything"
    )


def test_backup_refuses_to_overwrite(tmp_path):
    """A backup that silently replaces the previous one is one keystroke from
    having no backup at all."""
    from attestation.db import backup_db

    conn = get_db(tmp_path / "live.db")
    dest = backup_db(conn, tmp_path / "b.db")

    with pytest.raises(FileExistsError):
        backup_db(conn, dest)


def test_backup_result_is_a_readable_database(tmp_path):
    """VACUUM INTO produces a real database, not a byte copy -- verify it opens
    and carries the schema version, so a restore is not a surprise."""
    from attestation.db import SCHEMA_VERSION, backup_db

    conn = get_db(tmp_path / "live.db")
    dest = backup_db(conn, tmp_path / "b.db")

    restored = sqlite3.connect(dest)
    assert restored.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
