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
-- Things the reader DID, as opposed to things they judged. A read, a search
-- result they opened, an explanation they asked for: each is weak evidence of
-- interest, and implicit.harvest turns them into weak positive clicks tagged
-- `implicit` so provenance decides what they may be used for.
--
-- Separate from `clicks` because it is not a verdict, and separate from
-- `explanations` because that table holds generated TEXT and is deleted when a
-- persona's interests change -- engagement is a fact about the reader and
-- survives that. Measured before adding it: 11 human clicks across 19 days,
-- all from deliberately sitting down to rate. Feedback that requires a
-- gesture does not arrive.
CREATE TABLE IF NOT EXISTS engagement(
  user_id INTEGER NOT NULL REFERENCES users(id),
  item_id INTEGER NOT NULL REFERENCES items(id),
  kind TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (user_id, item_id, kind)
);
CREATE TABLE IF NOT EXISTS explanations(
  -- REFERENCES on both, unlike the first version of this table. It accepted a
  -- row pointing at a nonexistent user and item, and stayed clean only because
  -- delete_persona and reset_feedback clean it by hand -- an invariant upheld
  -- by application code that implicit.harvest then reads as weak positives.
  user_id INTEGER NOT NULL REFERENCES users(id),
  item_id INTEGER NOT NULL REFERENCES items(id),
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
  -- NULL means "the artifact did not say" -- never "no corpus" and never "the
  -- default corpus". Most artifacts record nothing about data, so unknown is
  -- the common case, and treating it as agreement is the bug this guards.
  corpus_id INTEGER REFERENCES corpora(id) ON DELETE SET NULL,
  -- Which reader produced this run: 'generic', 'wandb', 'mlflow', or a named
  -- adapter. NULL means "recorded before this column existed" -- never
  -- "generic", which is a guess the reader would have no way to challenge.
  -- The point is not bookkeeping: the tracker readers carry caveats the
  -- generic one does not (never run against a real directory; final metric
  -- values, not curves), and without this a wandb-derived run is
  -- indistinguishable in `runs.list` from a hand-written JSON one.
  adapter TEXT,
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
{corpus_schema}
CREATE TRIGGER IF NOT EXISTS trg_items_delete_vector AFTER DELETE ON items BEGIN
  DELETE FROM item_vectors WHERE rowid = old.id;
END;
"""


# The corpus tables, kept as their own string so migration 002 can create
# them in an existing database with exactly the DDL SCHEMA uses for a fresh
# one -- two divergent copies of a table definition is how schemas drift.
_CORPUS_SCHEMA = """
-- A corpus is its own entity, not a column on runs: it has attributes runs do
-- not (tokenizer, split sizes, fingerprint) and its own lifetime -- it can
-- change on disk while the runs citing it do not. Twelve runs sharing
-- WikiText-2 point at one inspectable row.
-- Every field but `name` is nullable: a partially-known corpus ("WikiText-2,
-- tokenizer unknown") is the normal case and is strictly more honest than
-- recording nothing, provided unknowns render as unknown.
CREATE TABLE IF NOT EXISTS corpora(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  source TEXT,
  config TEXT,
  tokenizer TEXT,
  vocab_size INTEGER,
  seq_len INTEGER,
  -- `fingerprint_kind` names what was hashed (file_sha256, dir_sha256,
  -- size_mtime, declared). Hashing a directory of shards and hashing one .txt
  -- are different claims; without this a cheap size+mtime check could be
  -- reported as a content hash, which is worse than no check at all.
  fingerprint TEXT,
  fingerprint_kind TEXT,
  measured_at TEXT,
  source_path TEXT,
  notes TEXT
);
-- Long format, as run_metrics is: projects name splits differently
-- (val/valid/validation/dev) and carry different counts.
CREATE TABLE IF NOT EXISTS corpus_splits(
  corpus_id INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
  split TEXT NOT NULL,
  n_tokens INTEGER,
  n_records INTEGER,
  n_bytes INTEGER,
  fingerprint TEXT,
  PRIMARY KEY (corpus_id, split)
);
-- item_vectors (a vec0 virtual table, created separately below since its
-- dimensionality is only known at runtime) links to items by bare rowid
-- equality with no FK possible on a virtual table. items.id is a rowid alias
-- SQLite reuses after the highest-id row is deleted, so without this trigger
-- a later item can silently inherit an earlier, unrelated item's vector.
"""

# Single-sourced: SCHEMA embeds the same DDL migration 002 applies, so a fresh
# database and a migrated one cannot drift apart.
SCHEMA = SCHEMA.format(corpus_schema=_CORPUS_SCHEMA.strip())


def _split_statements(script: str) -> list[str]:
    """SQL statements from a schema literal, with '--' comment lines removed.

    Comments are stripped BEFORE splitting on ';'. The other order splits
    inside a CREATE TABLE whose column list contains a '--' note, which these
    schemas do, and yields fragments sqlite rejects as "incomplete input".
    """
    clean = "\n".join(line for line in script.splitlines() if not line.strip().startswith("--"))
    return [statement.strip() for statement in clean.split(";") if statement.strip()]


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


def _migration_002_add_corpora(conn: sqlite3.Connection) -> None:
    """Add the corpus tables and runs.corpus_id.

    Purely additive and idempotent: existing runs get corpus_id NULL, which is
    the correct value for "the artifact did not say what data this saw".
    """
    # NOT executescript: it issues an implicit COMMIT before running, which
    # ends the transaction _migrate opened and leaves earlier migrations
    # committed while user_version still reads 0. Verified: a failure injected
    # into migration 003 left corpora and clicks.source in place at
    # user_version 0, so the next open re-ran 001 and 002 against a database
    # that already had them. Every migration is individually idempotent today,
    # so that re-run is a no-op -- but the ladder's atomicity is what makes
    # that discipline optional rather than load-bearing, and the first
    # migration to backfill data would be applied twice.
    # NOT executescript: it issues an implicit COMMIT before running, which
    # ends whatever _migrate opened. Verified by injecting a failure into
    # migration 003: corpora and clicks.source stayed committed while
    # user_version still read 0, so the next open re-ran 001 and 002 against a
    # database that already had them. A SAVEPOINT does not help -- executescript
    # destroys that too ("no such savepoint"). Nothing transactional survives
    # it, so the fix is to not call it.
    #
    # Order matters here: strip '--' lines FIRST, then split on ';'. Splitting
    # first lands mid-CREATE TABLE, because these schemas carry comments inside
    # their column lists.
    for statement in _split_statements(_CORPUS_SCHEMA):
        conn.execute(statement)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)")}
    if "corpus_id" not in cols:
        # No inline REFERENCES: SQLite's ALTER TABLE ADD COLUMN rejects a
        # column with a non-NULL default or a foreign key clause on some
        # versions. The FK is declared in SCHEMA for fresh databases; migrated
        # ones carry the column without it, which affects enforcement only.
        conn.execute("ALTER TABLE runs ADD COLUMN corpus_id INTEGER")


def _migration_003_add_runs_adapter(conn: sqlite3.Connection) -> None:
    """Add runs.adapter.

    Purely additive and idempotent. Existing runs get NULL, which is correct:
    they were recorded before anything tracked which reader produced them, and
    backfilling them to 'generic' would state as fact something nobody checked.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)")}
    if "adapter" not in cols:
        conn.execute("ALTER TABLE runs ADD COLUMN adapter TEXT")


def _migration_004_drop_dead_kg_tables(conn: sqlite3.Connection) -> None:
    """Drop kg_nodes / kg_edges / kg_meta.

    They were removed from SCHEMA on 2026-08-21 when the knowledge graph moved
    to being computed from item_tags -- nothing reads them, and the eight kg
    tool answers were byte-identical before and after. But removing a table
    from SCHEMA does not remove it from a database that already has it, so
    every DB created before that date still carries them: the live one holds
    701 + 2091 + 1 rows of state that nothing has updated since. Small on disk;
    the cost is a future reader mistaking them for live data.
    """
    for table in ("kg_nodes", "kg_edges", "kg_meta"):
        conn.execute(f"DROP TABLE IF EXISTS {table}")


def _migration_005_add_engagement(conn: sqlite3.Connection) -> None:
    """Add the engagement table.

    Purely additive. An existing database simply has no engagement recorded
    yet, which is the correct value for "we were not watching before now".
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS engagement("
        "  user_id INTEGER NOT NULL REFERENCES users(id),"
        "  item_id INTEGER NOT NULL REFERENCES items(id),"
        "  kind TEXT NOT NULL,"
        "  created_at TEXT NOT NULL DEFAULT (datetime('now')),"
        "  PRIMARY KEY (user_id, item_id, kind))"
    )


# Ordered ladder of (version, migration_fn). Each entry is applied, in order,
# exactly once per database: on open, every entry whose version is greater
# than the file's current `PRAGMA user_version` runs inside one transaction,
# then user_version is advanced to SCHEMA_VERSION. Fresh databases (created by
# SCHEMA above, which already reflects the latest shape) still run the ladder
# harmlessly -- every migration function must be a no-op against the current
# SCHEMA, same discipline as the old _migrate().
_MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, _migration_001_add_clicks_source),
    (2, _migration_002_add_corpora),
    (3, _migration_003_add_runs_adapter),
    (4, _migration_004_drop_dead_kg_tables),
    (5, _migration_005_add_engagement),
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
    # An explicit BEGIN, not `with conn:`. Two separate things defeated the
    # context manager, and fixing only one leaves the ladder just as partial:
    #
    #   1. Migration 002 called executescript, which issues an implicit COMMIT
    #      before running and ends whatever transaction is open. A SAVEPOINT
    #      does not survive it either ("no such savepoint").
    #   2. At sqlite3's default isolation_level, DDL does not open a
    #      transaction at all -- a CREATE TABLE inside a failed `with conn:`
    #      block stays committed. Every migration here is DDL.
    #
    # Verified against a real v0 database with a failure injected into
    # migration 003: before, corpora and clicks.source were both left behind at
    # user_version 0, so the next open re-ran 001 and 002 against a database
    # that already had them. After, the ladder rolls back whole.
    #
    # The migrations are each idempotent, so that re-run was a no-op today.
    # Atomicity is what keeps that discipline optional rather than
    # load-bearing: the first migration to backfill data would apply twice.
    previous_isolation = conn.isolation_level
    conn.isolation_level = None
    try:
        conn.execute("BEGIN")
        try:
            for version, fn in _MIGRATIONS:
                if version > current:
                    fn(conn)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    finally:
        conn.isolation_level = previous_isolation


def embed_dims() -> int:
    """Stored embedding dimensionality (EMBED_DIMS, default 256).

    Read at call time so tests and .env loading are respected. Changing it
    invalidates every stored vector — get_db() refuses mismatched databases.
    """
    return int(os.environ.get("EMBED_DIMS", "256"))


def _vec_schema(dims: int) -> str:
    return f"CREATE VIRTUAL TABLE IF NOT EXISTS item_vectors USING vec0(embedding float[{dims}])"


SEED_USERS = {
    "researcher": (
        "reproducibility, evaluation methodology, scientific computing, "
        "retrieval and ranking, open-weight models, research tooling"
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


def backup_db(conn: sqlite3.Connection, dest: str | Path) -> Path:
    """Write a consistent single-file copy of the database to `dest`.

    `cp hermes.db backup.db` is the obvious thing to type and it silently loses
    data: `get_db` sets journal_mode=WAL, so recent commits live in
    `hermes.db-wal` until a checkpoint. The copy opens, looks intact, and is
    missing the newest clicks and items -- worse than having no backup, because
    it looks trustworthy. Five such copies were sitting in the real data
    directory when this was written.

    VACUUM INTO reads through the WAL and writes one consistent, compacted file
    without stopping writers, which is exactly the operation an operator
    thought they were getting.

    Refuses an existing destination: a backup that silently replaces the
    previous one is one keystroke from being no backup at all.
    """
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; pick another name or remove it first")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # VACUUM INTO takes a literal, not a bound parameter, and rejects a path it
    # cannot create. Quote-escape rather than interpolate raw.
    conn.execute("VACUUM INTO ?", (str(dest),))
    return dest


def resolve_db_path(explicit: str | None) -> Path:
    """Resolve the hermes.db path with this precedence:

    1. `explicit` (the --db flag, when the caller actually passed one)
    2. `ATTEST_DB` env var, when set -- or `RSS_DB`, its pre-rename name, which
       stays honoured so no existing cron line or MCP entry breaks
    3. the co-located skill data dir (~/.hermes/skills/science-recommendations/data/hermes.db),
       but only if that file already exists
    4. ./hermes.db (cwd-relative default)
    """
    if explicit is not None:
        return Path(explicit)

    env_db = os.environ.get("ATTEST_DB") or os.environ.get("RSS_DB")
    if env_db:
        return Path(env_db)

    if SKILL_DATA_DB.exists():
        return SKILL_DATA_DB

    return Path("hermes.db")


def seed_demo_users(conn: sqlite3.Connection) -> None:
    """Insert the three hardcoded demo personas (researcher, bench-chemist, ml-engineer).

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


# 0600. The database holds every persona's interests text and the full click
# history behind their ranking -- a reading log, in other words. SQLite creates
# it with the process umask, which on an ordinary box is 0644, so on a shared
# machine any local account can read all of it.
#
# Applied at CREATION ONLY. Someone who deliberately widens the mode to share a
# database with a labmate's account must not have that quietly undone by the
# next tool call: this is a safe default, not a policy enforced on every open.
DB_FILE_MODE = 0o600


def _restrict(path: Path) -> None:
    """chmod DB_FILE_MODE, tolerating a filesystem that cannot express it.

    Windows and most network filesystems ignore or reject POSIX modes, and a
    failure to tighten permissions must not stop the database from opening --
    the caller loses a hardening measure, not their data.
    """
    try:
        path.chmod(DB_FILE_MODE)
    except OSError:
        pass


def _restrict_db_files(path: Path) -> None:
    """The main file and its WAL sidecars.

    Tightening only `hermes.db` would protect the history and publish today's
    writes: journal_mode=WAL keeps recent commits in `hermes.db-wal` until a
    checkpoint, and `-shm` is the shared index into it. The sidecars may not
    exist yet at open time, which is why this runs again after the first
    commit rather than once.
    """
    for candidate in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        if candidate.exists():
            _restrict(candidate)


def get_db(path: str | Path) -> sqlite3.Connection:
    """Open (creating if absent) the SQLite store at `path`.

    Never seeds personas. A new database is EMPTY, so the web UI's first
    screen is the onboarding form and an agent's first `feed.list` creates
    the reader it names. Demo personas (SEED_USERS) exist only when asked
    for: `attest bootstrap-persona <demo name>` creates the one it is given,
    and seed_demo_users() plants all three. Seeding on creation put the
    author's own persona in every stranger's database; seeding on every open
    resurrected personas the reader had deleted.
    """
    path = Path(path)
    is_new = not path.exists()
    conn = sqlite3.connect(str(path), check_same_thread=False)
    if is_new:
        # Before any schema is written, so the file is never readable with
        # content in it -- not even for the width of executescript().
        _restrict_db_files(path)
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
        # Again after the first commit: WAL mode creates -wal and -shm lazily,
        # so they did not exist for the pass above.
        _restrict_db_files(path)
    return conn
