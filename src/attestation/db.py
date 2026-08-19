"""Single-file SQLite store: relational tables + sqlite-vec vectors."""

import os
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

import sqlite_vec

# Legacy path, kept deliberately: this is where existing databases live. The
# skill directory was renamed to research-provenance, but repointing this would
# orphan every database created before the rename for no benefit.
SKILL_DATA_DB = (
    Path.home() / ".hermes" / "skills" / "science-recommendations" / "data" / "hermes.db"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  interests TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS feeds(
  id INTEGER PRIMARY KEY,
  url TEXT UNIQUE NOT NULL,
  title TEXT,
  last_fetched TEXT
);
CREATE TABLE IF NOT EXISTS items(
  id INTEGER PRIMARY KEY,
  feed_id INTEGER REFERENCES feeds(id),
  guid TEXT,
  title TEXT NOT NULL,
  url TEXT,
  summary TEXT NOT NULL DEFAULT '',
  published TEXT NOT NULL DEFAULT (datetime('now')),
  content_hash TEXT NOT NULL,
  UNIQUE(feed_id, guid)
);
CREATE INDEX IF NOT EXISTS idx_items_hash ON items(content_hash);
CREATE TABLE IF NOT EXISTS clicks(
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id),
  item_id INTEGER NOT NULL REFERENCES items(id),
  useful INTEGER NOT NULL,
  clicked_at TEXT NOT NULL DEFAULT (datetime('now')),
  source TEXT NOT NULL DEFAULT 'ui',
  UNIQUE(user_id, item_id)
);
CREATE TABLE IF NOT EXISTS explanations(
  user_id INTEGER NOT NULL,
  item_id INTEGER NOT NULL,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (user_id, item_id)
);
CREATE TABLE IF NOT EXISTS item_features(
  item_id INTEGER PRIMARY KEY REFERENCES items(id),
  content_type TEXT NOT NULL,
  model TEXT NOT NULL,
  tagged_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS item_tags(
  item_id INTEGER NOT NULL REFERENCES items(id),
  tag TEXT NOT NULL,
  PRIMARY KEY (item_id, tag)
);
CREATE TABLE IF NOT EXISTS kg_nodes(
  name TEXT PRIMARY KEY,
  node_type TEXT NOT NULL,
  degree INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS kg_edges(
  source TEXT NOT NULL,
  target TEXT NOT NULL,
  edge_type TEXT NOT NULL,
  weight INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (source, target, edge_type)
);
CREATE TABLE IF NOT EXISTS kg_meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY,
  project TEXT NOT NULL,
  name TEXT NOT NULL,
  family TEXT,
  status TEXT NOT NULL DEFAULT 'unknown',
  started TEXT,
  source_path TEXT NOT NULL,
  config_json TEXT,
  notes TEXT,
  UNIQUE (project, name)
);
-- long format, not wide: projects report entirely different metrics, and a
-- wide table would need a migration per project.
CREATE TABLE IF NOT EXISTS run_metrics(
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  metric TEXT NOT NULL,
  value REAL NOT NULL,
  step INTEGER,
  split TEXT,
  PRIMARY KEY (run_id, metric, step, split)
);
CREATE INDEX IF NOT EXISTS idx_runs_family ON runs(project, family);
-- item_vectors (a vec0 virtual table, created separately below since its
-- dimensionality is only known at runtime) links to items by bare rowid
-- equality with no FK possible on a virtual table. items.id is a rowid alias
-- SQLite reuses after the highest-id row is deleted, so without this trigger
-- a later item can silently inherit an earlier, unrelated item's vector.
CREATE TRIGGER IF NOT EXISTS trg_items_delete_vector AFTER DELETE ON items BEGIN
  DELETE FROM item_vectors WHERE rowid = old.id;
END;
"""


def _migration_001_add_clicks_source(conn: sqlite3.Connection) -> None:
    """Pre-existing clicks backfill to 'ui': they came from the web UI and
    bootstrap_persona, which are not retroactively distinguishable, and 'ui'
    is right for the majority. Idempotent: only runs if the column is absent,
    so it is safe to re-apply to a database that already has it (e.g. one
    created fresh by SCHEMA, which already includes the column).
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(clicks)")}
    if "source" not in cols:
        conn.execute("ALTER TABLE clicks ADD COLUMN source TEXT NOT NULL DEFAULT 'ui'")


# Ordered ladder of (version, migration_fn). Each entry is applied, in order,
# exactly once per database: on open, every entry whose version is greater
# than the file's current `PRAGMA user_version` runs inside one transaction,
# then user_version is advanced to SCHEMA_VERSION. Fresh databases (created by
# SCHEMA above, which already reflects the latest shape) still run the ladder
# harmlessly -- every migration function must be a no-op against the current
# SCHEMA, same discipline as the old _migrate().
_MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, _migration_001_add_clicks_source),
]

SCHEMA_VERSION = _MIGRATIONS[-1][0]


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply pending migrations and advance PRAGMA user_version.

    Refuses to open a database whose user_version is NEWER than the code
    knows about -- that is the genuine silent-corruption case (older code
    against a newer schema), versus an old database opened by new code, which
    this ladder is designed to upgrade safely.
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema version {current} is newer than this code supports"
            f" (max {SCHEMA_VERSION}) — upgrade attestation before opening this database"
        )
    if current == SCHEMA_VERSION:
        return
    with conn:
        for version, fn in _MIGRATIONS:
            if version > current:
                fn(conn)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def embed_dims() -> int:
    """Stored embedding dimensionality (EMBED_DIMS, default 256).

    Read at call time so tests and .env loading are respected. Changing it
    invalidates every stored vector — get_db() refuses mismatched databases.
    """
    return int(os.environ.get("EMBED_DIMS", "256"))


def _vec_schema(dims: int) -> str:
    return f"CREATE VIRTUAL TABLE IF NOT EXISTS item_vectors USING vec0(embedding float[{dims}])"


SEED_USERS = {
    "matt": (
        "LLM systems, retrieval and ranking, ML infrastructure, quantum chemistry, "
        "open-weight models, evaluation methodology, Rust and Python engineering"
    ),
    "bench-chemist": (
        "organic synthesis, reaction mechanisms, catalysis, spectroscopy, "
        "medicinal chemistry, lab automation, total synthesis"
    ),
    "ml-engineer": (
        "deep learning architectures, model serving, GPU inference, MLOps, "
        "transformers, fine-tuning, distributed training"
    ),
}


def resolve_db_path(explicit: str | None) -> Path:
    """Resolve the hermes.db path with this precedence:

    1. `explicit` (the --db flag, when the caller actually passed one)
    2. `RSS_DB` env var, when set
    3. the co-located skill data dir (~/.hermes/skills/science-recommendations/data/hermes.db),
       but only if that file already exists
    4. ./hermes.db (cwd-relative default)
    """
    if explicit is not None:
        return Path(explicit)

    env_db = os.environ.get("RSS_DB")
    if env_db:
        return Path(env_db)

    if SKILL_DATA_DB.exists():
        return SKILL_DATA_DB

    return Path("hermes.db")


def seed_demo_users(conn: sqlite3.Connection) -> None:
    """Insert the three hardcoded demo personas (matt, bench-chemist, ml-engineer).

    INSERT OR IGNORE, so calling this against a database that already has
    these rows (or rows a researcher has since deleted and doesn't want back)
    is safe -- but it is still a write, and callers should call it only when
    they actually want demo data seeded, not on every connection open. A
    persona deleted via delete_persona must stay deleted; re-running this
    would silently resurrect it.
    """
    for name, interests in SEED_USERS.items():
        conn.execute(
            "INSERT OR IGNORE INTO users(name, interests) VALUES (?, ?)", (name, interests)
        )
    conn.commit()


def get_db(path: str | Path) -> sqlite3.Connection:
    """Open (creating if absent) the SQLite store at `path`.

    Demo personas (SEED_USERS) are seeded only the first time this database
    file is created, never on subsequent opens -- otherwise a persona removed
    via delete_persona would be resurrected by the very next tool call, since
    every MCP tool opens its own connection. Callers that want to guarantee
    seed data exists (installer, demo setup) should call seed_demo_users()
    explicitly.
    """
    is_new = not Path(path).exists()
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    dims = embed_dims()
    existing = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'item_vectors'").fetchone()
    if existing:
        m = re.search(r"float\[(\d+)\]", existing["sql"])
        stored = int(m.group(1)) if m else None
        if stored is not None and stored != dims:
            raise RuntimeError(
                f"database has float[{stored}] vectors but EMBED_DIMS={dims}"
                " — re-ingest into a fresh database or set matching dims"
            )
    else:
        conn.execute(_vec_schema(dims))
    conn.commit()
    if is_new:
        seed_demo_users(conn)
    return conn
