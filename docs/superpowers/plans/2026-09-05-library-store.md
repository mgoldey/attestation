# Library Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One deduplicated, persisted reference library fed by BibTeX, Zotero and the feed, enriched opt-in from arXiv, CrossRef and Semantic Scholar, searchable semantically, exposed as `cite.*` tools and `attest library`.

**Architecture:** A new domain module `library.py` owns identity, merge, sync and search over five new tables (migration 007); readers yield `ReferenceRecord`s; the three network readers only enrich rows that already exist. Embedding and tagging reuse `embed.Embedder` and `features.tag_messages`. The `cite.*` tools call the store first and fall back to today's disk readers.

**Tech Stack:** Python 3.12, sqlite3 + sqlite-vec, httpx, xml.etree (arXiv Atom), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-05-library-store-design.md`

## Global Constraints

- Line length 100; ruff selects `E,F,W,I,BLE,RUF100`. No new `# noqa: BLE001` without a reason comment; `test_architecture.py` counts them (7 today) — a new one must update that count and its docs line.
- `uv run --frozen pre-commit run --all-files` is the gate (ruff format, ruff check, ty, lock sync, complexity/xenon ratchets, bandit, full pytest ~70 s). Run it before every commit that touches `src/`.
- Domain modules (`library.py`, `citations.py`, `ingest.py`, `rank.py`) never import from `attestation.mcp`; `mcp/*` imports domain. `library.py` must not import `llm` (llm is a composition-root concern; pass `chat_fn`/`embedder` in).
- Tests are hermetic: `conftest._hermetic_env` strips env; use `tmp_path` databases via `get_db(tmp_path / "t.db")`; never touch `~/.hermes`.
- Nothing leaves the machine unless `ATTEST_CITATION_WEB` (arXiv + CrossRef) or `ATTEST_CITATION_SCHOLAR` (Semantic Scholar) was set when the readers were built. `search` never calls a network reader.
- Tool count moves 46 → 47 (`cite.sync`). Every doc quoting 46 (`CLAUDE.md` lines 5 and 50, `README.md:88`, `docs/guides/agents.md:64,324`) and the per-namespace line `cite.*(4)` → `cite.*(5)` must change in the same commit as the tool, or `test_architecture.py` fails.
- `docs/reference/cli.md` is generated: run `uv run python scripts/render_cli_reference.py` after any parser change and commit the result.
- Work happens in the worktree `/home/matt/attestation/.claude/worktrees/library-store` on branch `feat/library-store`. Never open the live database from this branch (migration 007 would lock the gateway's older code out); manual runs set `ATTEST_DB=/tmp/<scratch>.db`.
- Commit after each task with a message that says what was measured or decided, not only what changed.

---

## File map

| file | responsibility |
|---|---|
| `src/attestation/library.py` (new) | `identity`, `normalise_title`, `merge`, `ReferenceRecord`, `upsert`, `sync`, `SyncReport`, `embed_missing`, `search`, `SearchHit`, `to_reference`, `status` |
| `src/attestation/library_readers.py` (new) | `BibtexRecords`, `ZoteroRecords`, `FeedRecords`, `ArxivEnricher`, `CrossrefEnricher`, `S2Enricher`, `readers_from_env` |
| `src/attestation/citations.py` | `_parse_bib_entries(text)` factored out of `BibtexReader.all` (returns key + lowercased field dict) so both readers share one parser; `Resolver.from_env` gains `ATTEST_BIB_PATHS`/`ATTEST_ZOTERO_PATH`; `Resolver` gains an optional `store` (a connection factory) consulted first |
| `src/attestation/ingest.py` | `extract_ids(guid, url)`; the insert writes `doi`, `arxiv_id` |
| `src/attestation/db.py` | SCHEMA additions, `_migration_007_add_library`, `_ref_vec_schema`, dims check covers both vec tables |
| `src/attestation/rank.py` | `RELEVANCE_FLOOR`, `RELEVANCE_ANCHOR`, `apply_relevance_floor`, `vector_search(conn, embedder, query, k, table)` moved here from `mcp/feed.py`, which imports them |
| `src/attestation/mcp/citation.py` | `cite.sync` new; `cite.lookup/search/sources` store-first; `needs_db=True` for the store tools |
| `src/attestation/mcp/__init__.py` | `cite.sync` is under the `knowledge` surface via the existing `cite` prefix (no change needed; verify) |
| `src/attestation/cli.py` | `attest library sync/search/tag/embed/status`; HELP entries |
| `src/attestation/features.py` | `run_reference_tagging(conn, chat_fn, model, limit)` beside `run_tagging` |
| `.env.sample` | four variables |
| `docs/guides/claims-and-citations.md` | "The library" section |
| `docs/reference/cli.md` | regenerated |
| `tests/test_library.py` (new) | pure + DB tests for identity/merge/upsert/sync/search |
| `tests/test_library_readers.py` (new) | readers, enrichers with committed fixture responses, zero-request guard |
| `tests/fixtures/library/` (new) | `arxiv_query.xml`, `s2_paper.json`, `crossref_work.json`, `sample.bib` |
| `tests/test_db.py` | migration 007 on a v6 file with live guid/url formats |
| `tests/test_ingest.py` | `extract_ids` and the insert |
| `tests/test_citations.py` | store-first resolver, flag-at-construction for S2 |
| `tests/test_response_size.py` | `cite.search` at `limit=13` |
| `tests/test_architecture.py` | count assertions (edit the docs, not the test) |

---

### Task 1: Identity, merge, and feed-id extraction (pure)

**Files:**
- Create: `src/attestation/library.py`
- Modify: `src/attestation/ingest.py` (add `extract_ids`)
- Test: `tests/test_library.py`, `tests/test_ingest.py`

**Interfaces:**
- Produces: `library.identity(doi: str | None, arxiv_id: str | None, title: str | None, year: int | None) -> str`; `library.normalise_title(title: str) -> str`; `library.normalise_doi(doi: str | None) -> str | None`; `library.normalise_arxiv(arxiv_id: str | None) -> str | None`; `library.merge(existing: dict, incoming: dict) -> tuple[dict, dict]`; `ingest.extract_ids(guid: str | None, url: str | None) -> tuple[str | None, str | None]` returning `(doi, arxiv_id)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_library.py`:

```python
"""library.py: identity, merge, upsert, sync, search."""

import pytest

from attestation import library


@pytest.mark.parametrize(
    ("doi", "arxiv", "title", "year", "want"),
    [
        ("10.1038/S41467-026-74391-4", None, "x", 2026, "doi:10.1038/s41467-026-74391-4"),
        ("https://doi.org/10.1000/ABC", None, "x", 2020, "doi:10.1000/abc"),
        ("doi:10.1000/abc", None, "x", 2020, "doi:10.1000/abc"),
        # DOI beats arXiv: a published preprint keeps its arXiv id but gains a DOI.
        ("10.1000/abc", "2106.02347v3", "x", 2021, "doi:10.1000/abc"),
        (None, "2106.02347v3", "x", 2021, "arxiv:2106.02347"),
        (None, "arXiv:2106.02347", "x", 2021, "arxiv:2106.02347"),
        (None, "cond-mat/0301234", "x", 2003, "arxiv:cond-mat/0301234"),
        (None, None, "SchNet: A continuous-filter CNN", 2017, "title:schnet a continuous filter cnn:2017"),
        (None, None, "  Équivariant  Force-Fields! ", None, "title:equivariant force fields:-"),
    ],
)
def test_identity_prefers_doi_then_arxiv_then_title(doi, arxiv, title, year, want):
    assert library.identity(doi, arxiv, title, year) == want


def test_identity_needs_something():
    with pytest.raises(ValueError):
        library.identity(None, None, "", None)


def test_merge_fills_empty_keeps_first_and_records_conflicts():
    existing = {"title": "SchNet", "abstract": None, "year": 2017, "authors": ["Schütt, K."]}
    incoming = {"title": "SchNet: a CNN", "abstract": "We present...", "year": 2018,
                "authors": ["Schütt, K.", "Kindermans, P."]}
    merged, conflicts = library.merge(existing, incoming)
    assert merged["abstract"] == "We present..."          # filled
    assert merged["title"] == "SchNet"                     # kept
    assert merged["year"] == 2017                          # kept
    assert conflicts["title"] == {"kept": "SchNet", "offered": "SchNet: a CNN"}
    assert conflicts["year"] == {"kept": 2017, "offered": 2018}
    # A longer author list EXTENDS rather than conflicts (a .bib truncated with "and others").
    assert merged["authors"] == ["Schütt, K.", "Kindermans, P."]
    assert "authors" not in conflicts


def test_merge_author_disagreement_is_a_conflict():
    merged, conflicts = library.merge({"authors": ["A, B"]}, {"authors": ["C, D"]})
    assert merged["authors"] == ["A, B"]
    assert conflicts["authors"] == {"kept": ["A, B"], "offered": ["C, D"]}
```

`tests/test_ingest.py` (append):

```python
@pytest.mark.parametrize(
    ("guid", "url", "want"),
    [
        # Measured on the live database 2026-09-05.
        ("oai:arXiv.org:1003.0563v2", "https://arxiv.org/abs/1003.0563", (None, "1003.0563")),
        (None, "https://arxiv.org/abs/2106.02347v3", (None, "2106.02347")),
        ("https://www.nature.com/articles/s41467-026-74391-4",
         "https://www.nature.com/articles/s41467-026-74391-4",
         ("10.1038/s41467-026-74391-4", None)),
        (None, "https://doi.org/10.1021/ACS.JCTC.9B00181", ("10.1021/acs.jctc.9b00181", None)),
        ("https://hnrss.org/best#123", "https://news.ycombinator.com/item?id=1", (None, None)),
        (None, None, (None, None)),
    ],
)
def test_extract_ids_from_live_formats(guid, url, want):
    from attestation.ingest import extract_ids

    assert extract_ids(guid, url) == want
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_library.py tests/test_ingest.py -q -k "identity or merge or extract_ids"`
Expected: FAIL, `ModuleNotFoundError: attestation.library` and `ImportError: extract_ids`.

- [ ] **Step 3: Implement**

`src/attestation/library.py` (start of file; later tasks append):

```python
"""The reference library: one row per paper, fed by many sources.

Identity is a pure function (DOI, else versionless arXiv id, else normalised
title and year) so that Zotero, three `.bib` files and the feed can all fill
ONE row. Merge never overwrites: an empty field takes a value, a differing
value is recorded as a conflict on the contributing source row. See
docs/superpowers/specs/2026-09-05-library-store-design.md.
"""

from __future__ import annotations

import re
import unicodedata

_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)\s*", re.IGNORECASE)
_ARXIV_PREFIX = re.compile(r"^(?:arxiv:)\s*", re.IGNORECASE)
_ARXIV_VERSION = re.compile(r"v\d+$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise_doi(doi: str | None) -> str | None:
    """Lowercase, scheme and doi.org prefix stripped; None for empty."""
    if not doi:
        return None
    out = _DOI_PREFIX.sub("", doi.strip()).lower()
    return out or None


def normalise_arxiv(arxiv_id: str | None) -> str | None:
    """`arXiv:2106.02347v3` -> `2106.02347`; old-style ids kept whole."""
    if not arxiv_id:
        return None
    out = _ARXIV_PREFIX.sub("", arxiv_id.strip())
    out = _ARXIV_VERSION.sub("", out)
    return out or None


def normalise_title(title: str) -> str:
    """NFKD, combining marks dropped, lowercase, non-alphanumerics collapsed.

    Leading articles are kept on purpose: dropping them merges "A survey"
    with "Survey", which are different papers more often than not.
    """
    decomposed = unicodedata.normalize("NFKD", title)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _NON_ALNUM.sub(" ", ascii_only.lower()).strip()


def identity(doi: str | None, arxiv_id: str | None, title: str | None, year: int | None) -> str:
    """The one string two records must share to be the same paper.

    DOI beats arXiv because a preprint that is later published gains a DOI
    while keeping its arXiv id; a row carries both columns so a record that
    knows only the arXiv id still finds it (see `upsert`).
    """
    if d := normalise_doi(doi):
        return f"doi:{d}"
    if a := normalise_arxiv(arxiv_id):
        return f"arxiv:{a}"
    if title and (t := normalise_title(title)):
        return f"title:{t}:{year if year is not None else '-'}"
    raise ValueError("a reference needs a DOI, an arXiv id, or a title")


def _authors_extend(existing: list, incoming: list) -> bool:
    """True when `incoming` is `existing` plus more names (a truncated list filled in)."""
    if len(incoming) <= len(existing):
        return False
    norm = [normalise_title(a) for a in existing]
    return [normalise_title(a) for a in incoming[: len(existing)]] == norm


def merge(existing: dict, incoming: dict) -> tuple[dict, dict]:
    """Fill empty fields from `incoming`; keep non-empty ones; record conflicts.

    Deliberately dumb -- longer abstracts do not win, first does -- so that
    nothing is ever overwritten silently. `cite.lookup` shows every source
    row, so a reader can see a disagreement rather than lose it.
    """
    merged = dict(existing)
    conflicts: dict = {}
    for field, offered in incoming.items():
        if offered in (None, "", [], {}):
            continue
        kept = merged.get(field)
        if kept in (None, "", [], {}):
            merged[field] = offered
            continue
        if field == "authors":
            if _authors_extend(kept, offered):
                merged[field] = list(offered)
                continue
            if [normalise_title(a) for a in kept] == [normalise_title(a) for a in offered]:
                continue
        elif field == "title" and normalise_title(kept) == normalise_title(offered):
            continue
        elif kept == offered:
            continue
        conflicts[field] = {"kept": kept, "offered": offered}
    return merged, conflicts
```

`src/attestation/ingest.py` (add near `strip_boilerplate`):

```python
_ARXIV_ID = re.compile(r"(?:oai:arXiv\.org:|arxiv\.org/(?:abs|pdf)/)([a-z\-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_DOI = re.compile(r"(?:doi\.org/|doi:)?(10\.\d{4,9}/[^\s#?]+)", re.IGNORECASE)
_NATURE = re.compile(r"nature\.com/articles/([a-z0-9\-]+)", re.IGNORECASE)


def extract_ids(guid: str | None, url: str | None) -> tuple[str | None, str | None]:
    """(doi, arxiv_id) from an entry's guid and url, or Nones.

    Pure and network-free; used by migration 007's backfill and by every new
    item. Formats are the ones measured on the live database 2026-09-05:
    arXiv guids `oai:arXiv.org:1003.0563v2`, Nature URLs carrying the DOI
    suffix under the 10.1038 prefix. Anything else stays NULL.
    """
    doi = arxiv = None
    for text in (guid or "", url or ""):
        if arxiv is None and (m := _ARXIV_ID.search(text)):
            arxiv = m.group(1)
        if doi is None and (m := _DOI.search(text)):
            doi = m.group(1).lower().rstrip(".")
        if doi is None and (m := _NATURE.search(text)):
            doi = f"10.1038/{m.group(1).lower()}"
    return doi, arxiv
```

Check `import re` exists in `ingest.py` (it does: `ARXIV_RE`).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_library.py tests/test_ingest.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/attestation/library.py src/attestation/ingest.py tests/test_library.py tests/test_ingest.py
git commit -m "Library identity and merge as pure functions; feed-id extraction for the live guid/url formats"
```

---

### Task 2: Migration 007 and the schema

**Files:**
- Modify: `src/attestation/db.py` (SCHEMA, `_migration_007_add_library`, `_MIGRATIONS`, vec helpers, `get_db` dims check)
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `ingest.extract_ids`.
- Produces: tables `"references"`, `reference_sources`, `reference_tags`, `reference_cites`, `reference_vectors`; columns `items.doi`, `items.arxiv_id`; `db.LIBRARY_SCHEMA` string.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_db.py`:

```python
def _v6_database(path):
    """A database as migration 006 left it: no library tables, items without ids."""
    import sqlite3

    from attestation import db as dbmod

    conn = dbmod.get_db(path)
    conn.execute("INSERT INTO feeds(url, title) VALUES ('http://a', 'arXiv chem-ph')")
    conn.execute(
        "INSERT INTO items(feed_id, guid, title, url, summary, content_hash)"
        " VALUES (1, 'oai:arXiv.org:1003.0563v2', 't1', 'https://arxiv.org/abs/1003.0563', '', 'h1')"
    )
    conn.execute(
        "INSERT INTO items(feed_id, guid, title, url, summary, content_hash)"
        " VALUES (1, 'https://www.nature.com/articles/s41467-026-74391-4', 't2',"
        " 'https://www.nature.com/articles/s41467-026-74391-4', '', 'h2')"
    )
    conn.execute("INSERT INTO items(feed_id, guid, title, url, summary, content_hash)"
                 " VALUES (1, 'x', 't3', 'https://news.ycombinator.com/item?id=1', '', 'h3')")
    conn.commit()
    # Rewind to v6 by dropping what 007 adds and resetting user_version.
    for stmt in (
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
    from attestation import db as dbmod

    path = tmp_path / "v6.db"
    _v6_database(path)
    conn = dbmod.get_db(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == dbmod.SCHEMA_VERSION
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"references", "reference_sources", "reference_tags", "reference_cites",
            "reference_vectors"} <= tables
    rows = conn.execute("SELECT title, doi, arxiv_id FROM items ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [
        ("t1", None, "1003.0563"),
        ("t2", "10.1038/s41467-026-74391-4", None),
        ("t3", None, None),
    ]
    # Reopening is a no-op: the backfill must not run twice.
    conn.close()
    conn = dbmod.get_db(path)
    assert conn.execute("SELECT count(*) FROM items WHERE arxiv_id = '1003.0563'").fetchone()[0] == 1


def test_reference_vectors_follow_their_reference(tmp_path):
    from attestation import db as dbmod

    conn = dbmod.get_db(tmp_path / "t.db")
    conn.execute(
        'INSERT INTO "references"(id, identity, title, first_seen, updated)'
        " VALUES (7, 'doi:x', 'T', '2026-09-05', '2026-09-05')"
    )
    conn.execute("INSERT INTO reference_vectors(rowid, embedding) VALUES (7, ?)",
                 (b"\x00" * 4 * dbmod.embed_dims(),))
    conn.execute('DELETE FROM "references" WHERE id = 7')
    assert conn.execute("SELECT count(*) FROM reference_vectors").fetchone()[0] == 0


def test_a_fresh_database_has_the_library_tables(tmp_path):
    from attestation import db as dbmod

    conn = dbmod.get_db(tmp_path / "fresh.db")
    cols = {r["name"] for r in conn.execute('PRAGMA table_info("references")')}
    assert {"identity", "doi", "arxiv_id", "title", "authors", "year", "venue", "abstract",
            "url", "bib_key", "first_seen", "updated"} <= cols
    assert {r["name"] for r in conn.execute("PRAGMA table_info(items)")} >= {"doi", "arxiv_id"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_db.py -q -k "007 or reference_vectors or library_tables"`
Expected: FAIL (`no such table: references`).

- [ ] **Step 3: Implement**

In `db.py`, add a `_LIBRARY_SCHEMA` string (own constant, like `_CORPUS_SCHEMA`, so 007 and SCHEMA share one DDL) and splice it into SCHEMA before the items delete trigger. Also add `doi TEXT, arxiv_id TEXT` to the fresh `items` DDL.

```python
_LIBRARY_SCHEMA = """
-- One row per paper. `identity` is library.identity(): DOI, else versionless
-- arXiv id, else normalised title+year. A paper held by Zotero and two .bib
-- files under three keys is one row here with three reference_sources rows.
CREATE TABLE IF NOT EXISTS "references"(
  id INTEGER PRIMARY KEY,
  identity TEXT NOT NULL UNIQUE,
  doi TEXT,
  arxiv_id TEXT,
  title TEXT NOT NULL,
  authors TEXT NOT NULL DEFAULT '[]',
  year INTEGER,
  venue TEXT,
  abstract TEXT,
  url TEXT,
  bib_key TEXT,
  first_seen TEXT NOT NULL,
  updated TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_references_doi ON "references"(doi);
CREATE INDEX IF NOT EXISTS idx_references_arxiv ON "references"(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_references_bib_key ON "references"(bib_key);
-- Provenance per CONTRIBUTION, not per record: which source offered what,
-- when (NULL fetched_at = read from disk), and any conflict merge() refused.
CREATE TABLE IF NOT EXISTS reference_sources(
  reference_id INTEGER NOT NULL REFERENCES "references"(id) ON DELETE CASCADE,
  source TEXT NOT NULL,
  source_key TEXT NOT NULL,
  fetched_at TEXT,
  raw TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (reference_id, source, source_key)
);
CREATE TABLE IF NOT EXISTS reference_tags(
  reference_id INTEGER NOT NULL REFERENCES "references"(id) ON DELETE CASCADE,
  tag TEXT NOT NULL,
  PRIMARY KEY (reference_id, tag)
);
-- cited_identity is a string, not a foreign key: most of a paper's references
-- are not in the library. Spec 2 decides what a citation neighbourhood shows.
CREATE TABLE IF NOT EXISTS reference_cites(
  citing_id INTEGER NOT NULL REFERENCES "references"(id) ON DELETE CASCADE,
  cited_identity TEXT NOT NULL,
  cited_title TEXT,
  source TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  PRIMARY KEY (citing_id, cited_identity)
);
CREATE TRIGGER IF NOT EXISTS trg_references_delete_vector AFTER DELETE ON "references" BEGIN
  DELETE FROM reference_vectors WHERE rowid = old.id;
END;
"""
```

Because the trigger references `reference_vectors`, that virtual table must exist before the trigger is created. Order in `get_db`: `executescript(SCHEMA)` currently runs before `_vec_schema`. Handle it the same way `item_vectors` is handled: create BOTH vec tables before `executescript(SCHEMA)` when absent, i.e. move the existing dims check/creation block above `conn.executescript(SCHEMA)` and extend it:

```python
def _vec_schema(dims: int, table: str = "item_vectors") -> str:
    return f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(embedding float[{dims}])"


def _ensure_vec_tables(conn: sqlite3.Connection) -> None:
    """Create item_vectors and reference_vectors at EMBED_DIMS, refusing a mismatch.

    Runs before SCHEMA because the delete triggers reference these tables.
    """
    dims = embed_dims()
    for table in ("item_vectors", "reference_vectors"):
        existing = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (table,)
        ).fetchone()
        if existing:
            m = re.search(r"float\[(\d+)\]", existing["sql"])
            stored = int(m.group(1)) if m else None
            if stored is not None and stored != dims:
                raise RuntimeError(
                    f"database has float[{stored}] vectors in {table} but EMBED_DIMS={dims}"
                    " — re-ingest into a fresh database or set matching dims"
                )
        else:
            conn.execute(_vec_schema(dims, table))
```

In `get_db`, replace the block from `dims = embed_dims()` through `conn.execute(_vec_schema(dims))` with a call `_ensure_vec_tables(conn)` placed BEFORE `conn.executescript(SCHEMA)`. Keep `_migrate(conn)` after SCHEMA.

The migration:

```python
def _migration_007_add_library(conn: sqlite3.Connection) -> None:
    """Add the reference library and items.doi / items.arxiv_id, backfilled.

    The first migration that writes data: `extract_ids` fills the two new
    columns from guid and url for every existing item. The ladder's explicit
    BEGIN/COMMIT (see _migrate) is what makes running this exactly once a
    property of the code rather than a hope; a re-run against the current
    SCHEMA is a no-op because the columns then already exist.
    """
    from attestation.ingest import extract_ids

    for statement in _split_statements(_LIBRARY_SCHEMA):
        conn.execute(statement)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(items)")}
    if "doi" in cols and "arxiv_id" in cols:
        return
    if "doi" not in cols:
        conn.execute("ALTER TABLE items ADD COLUMN doi TEXT")
    if "arxiv_id" not in cols:
        conn.execute("ALTER TABLE items ADD COLUMN arxiv_id TEXT")
    rows = conn.execute("SELECT id, guid, url FROM items").fetchall()
    for row in rows:
        doi, arxiv = extract_ids(row["guid"], row["url"])
        if doi or arxiv:
            conn.execute("UPDATE items SET doi = ?, arxiv_id = ? WHERE id = ?", (doi, arxiv, row["id"]))
```

Register `(7, _migration_007_add_library)` in `_MIGRATIONS`. `ingest.py` must not import `db` at module level (check: it does not), so the local import avoids a cycle.

- [ ] **Step 4: Run the DB tests and the whole suite**

Run: `uv run pytest tests/test_db.py -q` then `uv run pytest -q -x`
Expected: pass. If `test_architecture.py` has a table-count assertion ("12 APPLICATION tables ... 17 with shadow tables"), update `CLAUDE.md`'s Storage line: 16 application tables; a fresh file now has `reference_vectors` plus its four shadow tables too (count them: `SELECT count(*) FROM sqlite_master WHERE type='table'`) and put the measured number in CLAUDE.md.

- [ ] **Step 5: Commit**

```bash
git add src/attestation/db.py tests/test_db.py CLAUDE.md
git commit -m "Migration 007: the reference library tables, a second vec table, and items.doi/arxiv_id backfilled from the live guid/url formats"
```

---

### Task 3: Ingest writes the ids for new items

**Files:**
- Modify: `src/attestation/ingest.py:179-190` (the INSERT)
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Failing test** (append to `tests/test_ingest.py`; mirror the existing run_ingest test's fake `parse` — look at how the file builds a parsed feed with `types.SimpleNamespace(entries=[...])` and copy that helper):

```python
def test_ingest_stores_doi_and_arxiv_id(tmp_path, fake_embedder):
    from attestation.db import get_db
    from attestation.ingest import run_ingest

    conn = get_db(tmp_path / "t.db")
    feeds = tmp_path / "feeds.toml"
    feeds.write_text('[[feeds]]\nurl = "http://a"\ntitle = "arXiv"\n')
    entries = [
        {"id": "oai:arXiv.org:2106.02347v2", "title": "NequIP", "summary": "E(3)-equivariant",
         "link": "https://arxiv.org/abs/2106.02347"},
        {"id": "https://www.nature.com/articles/s41557-026-02200-y", "title": "N",
         "summary": "s", "link": "https://www.nature.com/articles/s41557-026-02200-y"},
    ]
    run_ingest(conn, fake_embedder, feeds, parse=lambda url: _parsed(entries))
    rows = {r["title"]: (r["doi"], r["arxiv_id"]) for r in conn.execute("SELECT * FROM items")}
    assert rows["NequIP"] == (None, "2106.02347")
    assert rows["N"] == ("10.1038/s41557-026-02200-y", None)
```

(`_parsed` = whatever helper the file already uses to fake `feedparser.parse`; if none exists, define `def _parsed(entries): return types.SimpleNamespace(entries=[types.SimpleNamespace(**e, get=e.get) ...])` — check how existing tests construct entries and reuse exactly that.)

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_ingest.py -q -k stores_doi` → FAIL (`no such column` or Nones).

- [ ] **Step 3: Implement** — in `run_ingest`'s INSERT:

```python
                doi, arxiv_id = extract_ids(guid, entry.get("link"))
                cur = conn.execute(
                    "INSERT INTO items(feed_id, guid, title, url, summary, published,"
                    " content_hash, doi, arxiv_id)"
                    " VALUES (?, ?, ?, ?, ?, COALESCE(?, datetime('now')), ?, ?, ?)",
                    (feed["id"], guid, title, entry.get("link"), summary,
                     _published_iso(entry), chash, doi, arxiv_id),
                )
```

- [ ] **Step 4: Run** `uv run pytest tests/test_ingest.py -q` → pass.
- [ ] **Step 5: Commit** `git commit -am "Ingest records doi and arxiv_id for every new item"`

---

### Task 4: Records and the three offline readers

**Files:**
- Create: `src/attestation/library_readers.py`, `tests/test_library_readers.py`, `tests/fixtures/library/sample.bib`
- Modify: `src/attestation/citations.py` (factor `_parse_bib_entries`), `src/attestation/library.py` (add `ReferenceRecord`)

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class ReferenceRecord:
      source: str; source_key: str; title: str | None = None; authors: list[str] = ...
      year: int | None = None; doi: str | None = None; arxiv_id: str | None = None
      venue: str | None = None; abstract: str | None = None; url: str | None = None
      bib_key: str | None = None; fetched_at: str | None = None
      cites: list[tuple[str, str | None]] = ...   # (identity, title)
      def fields(self) -> dict  # the references-table columns that are not None/empty
  ```
  `library_readers.BibtexRecords(paths).records()`, `ZoteroRecords(path).records()`, `FeedRecords(conn).records()`; each has `name`, `network = False`. `citations._parse_bib_entries(text) -> Iterator[tuple[str, dict]]` (key, lowercased-field dict).

- [ ] **Step 1: Failing tests** — `tests/fixtures/library/sample.bib`:

```bibtex
@article{schutt2017schnet,
  title = {SchNet: A continuous-filter convolutional neural network for modeling quantum interactions},
  author = {Sch{\"u}tt, Kristof T. and Kindermans, Pieter-Jan and Sauceda, Huziel E. and others},
  journal = {Advances in Neural Information Processing Systems},
  year = {2017},
  eprint = {1706.08566},
  archiveprefix = {arXiv},
  abstract = {Deep learning has the potential to revolutionize quantum chemistry.},
}
@inproceedings{batzner2022nequip,
  title = {E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials},
  author = {Batzner, Simon and Musaelian, Albert},
  booktitle = {Nature Communications},
  year = {2022},
  doi = {10.1038/s41467-022-29939-5},
}
```

`tests/test_library_readers.py`:

```python
from pathlib import Path

from attestation import library_readers
from attestation.db import get_db

FIX = Path(__file__).parent / "fixtures" / "library"


def test_bibtex_records_carry_abstract_venue_and_arxiv_from_eprint():
    recs = {r.bib_key: r for r in library_readers.BibtexRecords([FIX / "sample.bib"]).records()}
    s = recs["schutt2017schnet"]
    assert s.arxiv_id == "1706.08566" and s.year == 2017
    assert s.venue == "Advances in Neural Information Processing Systems"
    assert s.abstract.startswith("Deep learning")
    assert s.source == f"bibtex:{FIX / 'sample.bib'}" and s.fetched_at is None
    n = recs["batzner2022nequip"]
    assert n.doi == "10.1038/s41467-022-29939-5" and n.venue == "Nature Communications"


def test_feed_records_only_for_items_with_an_id(tmp_path):
    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO feeds(url, title) VALUES ('http://a', 'arXiv')")
    conn.execute("INSERT INTO items(feed_id, guid, title, url, summary, content_hash, arxiv_id, published)"
                 " VALUES (1, 'g1', 'NequIP', 'https://arxiv.org/abs/2106.02347', 'abs', 'h1', '2106.02347', '2021-06-04')")
    conn.execute("INSERT INTO items(feed_id, guid, title, url, summary, content_hash)"
                 " VALUES (1, 'g2', 'HN post', 'http://x', 's', 'h2')")
    recs = list(library_readers.FeedRecords(conn).records())
    assert [(r.source, r.source_key, r.arxiv_id, r.abstract, r.year) for r in recs] == [
        ("feed", "1", "2106.02347", "abs", 2021)
    ]
```

Add a Zotero test that reuses `tests/test_citations.py::_zotero_db` and `_add_item` (import them) and asserts `ZoteroRecords(db).records()` yields the item with `source == "zotero"` and `bib_key == <zotero key>`. Extend `_add_item` to accept `abstract=None` and write `abstractNote` if given (look at how it writes `title` into `itemData`/`itemDataValues`/`fields` and add the field the same way).

- [ ] **Step 2: Run** → FAIL (no module).

- [ ] **Step 3: Implement**

In `citations.py`, factor the parser out of `BibtexReader.all` (behaviour unchanged):

```python
def _parse_bib_entries(text: str):
    """(key, {lowercased field: single-spaced value}) for each entry with a title."""
    for _kind, key, body in _ENTRY.findall(text):
        fields = {k.lower(): " ".join(v.split()) for k, v in _FIELD.findall(body)}
        if fields.get("title"):
            yield key, fields
```

and have `BibtexReader.all` iterate `_parse_bib_entries(path.read_text(errors="replace"))`.

`library.py` gains:

```python
from dataclasses import dataclass, field


@dataclass
class ReferenceRecord:
    """What one source says about one paper. The incoming shape for `upsert`."""

    source: str
    source_key: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    venue: str | None = None
    abstract: str | None = None
    url: str | None = None
    bib_key: str | None = None
    fetched_at: str | None = None
    cites: list[tuple[str, str | None]] = field(default_factory=list)

    def fields(self) -> dict:
        """The `references` columns this record can fill (empty ones omitted)."""
        out = {
            "doi": normalise_doi(self.doi),
            "arxiv_id": normalise_arxiv(self.arxiv_id),
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "abstract": self.abstract,
            "url": self.url,
            "bib_key": self.bib_key,
        }
        return {k: v for k, v in out.items() if v not in (None, "", [])}
```

`library_readers.py`:

```python
"""Sources of ReferenceRecords: three from disk and the feed, three enrichers
behind flags. Only the offline readers can INTRODUCE a reference; the network
ones fill fields on rows that already exist (spec §3.1)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

from attestation import citations
from attestation.library import ReferenceRecord


def _bib_authors(value: str) -> list[str]:
    return [a.strip() for a in value.split(" and ") if a.strip() and a.strip() != "others"]


class BibtexRecords:
    network = False

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]

    @property
    def name(self) -> str:
        return "bibtex"

    def records(self) -> Iterator[ReferenceRecord]:
        for path in self.paths:
            if not path.is_file():
                continue
            for key, f in citations._parse_bib_entries(path.read_text(errors="replace")):
                arxiv = f.get("eprint") if f.get("archiveprefix", "").lower() == "arxiv" else None
                yield ReferenceRecord(
                    source=f"bibtex:{path}",
                    source_key=key,
                    title=f["title"],
                    authors=_bib_authors(f.get("author", "")),
                    year=citations._year(f.get("year") or f.get("date")),
                    doi=f.get("doi"),
                    arxiv_id=arxiv or f.get("arxivid"),
                    venue=f.get("journal") or f.get("booktitle"),
                    abstract=f.get("abstract"),
                    url=f.get("url"),
                    bib_key=key,
                )


class ZoteroRecords:
    name = "zotero"
    network = False

    def __init__(self, path: Path | None = None):
        self.reader = citations.ZoteroReader(path)

    def records(self) -> Iterator[ReferenceRecord]:
        for key, data, authors in self.reader.raw_items():
            extra = data.get("extra") or ""
            arxiv = None
            for line in extra.splitlines():
                if line.lower().startswith("arxiv:"):
                    arxiv = line.split(":", 1)[1].strip()
            yield ReferenceRecord(
                source="zotero", source_key=key, title=data["title"], authors=authors,
                year=citations._year(data.get("date")), doi=data.get("DOI"), arxiv_id=arxiv,
                venue=data.get("publicationTitle"), abstract=data.get("abstractNote"),
                url=data.get("url"), bib_key=key,
            )


class FeedRecords:
    name = "feed"
    network = False

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def records(self) -> Iterator[ReferenceRecord]:
        rows = self.conn.execute(
            "SELECT id, title, url, summary, published, doi, arxiv_id FROM items"
            " WHERE doi IS NOT NULL OR arxiv_id IS NOT NULL ORDER BY id"
        )
        for r in rows:
            yield ReferenceRecord(
                source="feed", source_key=str(r["id"]), title=r["title"],
                year=citations._year(r["published"]), doi=r["doi"], arxiv_id=r["arxiv_id"],
                abstract=r["summary"] or None, url=r["url"],
            )
```

`ZoteroReader` gains `raw_items()` yielding `(key, data_dict, authors_list)` — the loop body of `all()` before it constructs `Reference`; `all()` then iterates `raw_items()`. Same tolerance (returns nothing on `sqlite3.Error`).

- [ ] **Step 4: Run** `uv run pytest tests/test_library_readers.py tests/test_citations.py -q` → pass.
- [ ] **Step 5: Commit** `git add -A src tests && git commit -m "ReferenceRecord and the three offline readers; one BibTeX parser shared with citations"`

---

### Task 5: Upsert, sync, and the report (offline)

**Files:**
- Modify: `src/attestation/library.py`
- Test: `tests/test_library.py`

**Interfaces:**
- Produces: `library.upsert(conn, rec: ReferenceRecord) -> tuple[int, str]` returning `(reference_id, "added"|"merged"|"unchanged")`; `library.sync(conn, readers, *, embedder=None, limit=None) -> SyncReport`; `SyncReport.to_dict()`; `library.status(conn) -> dict`.

- [ ] **Step 1: Failing tests** (append):

```python
def _rec(**kw):
    from attestation.library import ReferenceRecord
    kw.setdefault("source", "bibtex:/a.bib")
    kw.setdefault("source_key", kw.get("bib_key", "k"))
    return ReferenceRecord(**kw)


def test_upsert_merges_zotero_and_bib_under_one_row(tmp_path):
    from attestation.db import get_db
    conn = get_db(tmp_path / "t.db")
    rid1, how1 = library.upsert(conn, _rec(bib_key="schnet", title="SchNet", year=2017,
                                          doi="10.5555/schnet", authors=["Schütt, K."]))
    rid2, how2 = library.upsert(conn, _rec(source="zotero", source_key="ABCD1234", bib_key="ABCD1234",
                                          title="SchNet: a CNN", doi="10.5555/SCHNET",
                                          abstract="We present", authors=["Schütt, K.", "Kindermans, P."]))
    assert rid1 == rid2 and (how1, how2) == ("added", "merged")
    row = conn.execute('SELECT * FROM "references"').fetchone()
    assert row["identity"] == "doi:10.5555/schnet"
    assert row["title"] == "SchNet" and row["abstract"] == "We present"
    assert row["bib_key"] == "schnet"   # first key seen
    sources = conn.execute("SELECT source, source_key, raw FROM reference_sources ORDER BY source").fetchall()
    assert [(s["source"], s["source_key"]) for s in sources] == [("bibtex:/a.bib", "schnet"), ("zotero", "ABCD1234")]
    import json
    assert json.loads(sources[1]["raw"])["conflicts"]["title"]["offered"] == "SchNet: a CNN"


def test_upsert_finds_a_row_by_arxiv_id_and_upgrades_its_identity(tmp_path):
    from attestation.db import get_db
    conn = get_db(tmp_path / "t.db")
    rid, _ = library.upsert(conn, _rec(source="feed", source_key="9", title="NequIP", arxiv_id="2106.02347v2"))
    assert conn.execute('SELECT identity FROM "references"').fetchone()[0] == "arxiv:2106.02347"
    rid2, how = library.upsert(conn, _rec(bib_key="nequip", title="NequIP", arxiv_id="2106.02347",
                                         doi="10.1038/s41467-022-29939-5"))
    assert rid2 == rid and how == "merged"
    assert conn.execute('SELECT identity FROM "references"').fetchone()[0] == "doi:10.1038/s41467-022-29939-5"


def test_sync_is_idempotent(tmp_path):
    from attestation.db import get_db
    from attestation.library_readers import BibtexRecords
    conn = get_db(tmp_path / "t.db")
    readers = [BibtexRecords([FIX / "sample.bib"])]
    first = library.sync(conn, readers).to_dict()
    second = library.sync(conn, readers).to_dict()
    assert first["sources"]["bibtex"] == {"seen": 2, "added": 2, "merged": 0, "unchanged": 0, "enriched": 0, "failed": 0}
    assert second["sources"]["bibtex"]["unchanged"] == 2 and second["sources"]["bibtex"]["added"] == 0
    assert library.status(conn)["references"] == 2
```

(`FIX` as in the readers test; import it or redefine.)

- [ ] **Step 2: Run** → FAIL (`upsert` missing).

- [ ] **Step 3: Implement** (append to `library.py`):

```python
import json
import sqlite3
from datetime import UTC, datetime


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _find(conn, fields: dict, ident: str):
    for sql, val in (
        ('SELECT * FROM "references" WHERE identity = ?', ident),
        ('SELECT * FROM "references" WHERE doi = ?', fields.get("doi")),
        ('SELECT * FROM "references" WHERE arxiv_id = ?', fields.get("arxiv_id")),
    ):
        if val and (row := conn.execute(sql, (val,)).fetchone()):
            return row
    return None


_COLUMNS = ("doi", "arxiv_id", "title", "authors", "year", "venue", "abstract", "url", "bib_key")


def upsert(conn: sqlite3.Connection, rec: ReferenceRecord) -> tuple[int, str]:
    """Merge one record into the store. Returns (id, added|merged|unchanged).

    Lookup order: identity, DOI, arXiv id -- so a record that knows only the
    arXiv id still finds the row a DOI-bearing record created, and a row
    created from an arXiv id is upgraded to the DOI identity when one arrives.
    """
    fields = rec.fields()
    ident = identity(fields.get("doi"), fields.get("arxiv_id"), fields.get("title"), fields.get("year"))
    now = _now()
    row = _find(conn, fields, ident)
    if row is None:
        cur = conn.execute(
            'INSERT INTO "references"(identity, doi, arxiv_id, title, authors, year, venue,'
            " abstract, url, bib_key, first_seen, updated)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ident, fields.get("doi"), fields.get("arxiv_id"), fields["title"],
             json.dumps(fields.get("authors", [])), fields.get("year"), fields.get("venue"),
             fields.get("abstract"), fields.get("url"), fields.get("bib_key"), now, now),
        )
        rid, how, conflicts = cur.lastrowid, "added", {}
    else:
        existing = {c: row[c] for c in _COLUMNS}
        existing["authors"] = json.loads(row["authors"])
        merged, conflicts = merge(existing, fields)
        changed = {c: merged[c] for c in _COLUMNS if merged.get(c) != existing.get(c)}
        new_ident = identity(merged.get("doi"), merged.get("arxiv_id"), merged.get("title"), merged.get("year"))
        if new_ident != row["identity"]:
            changed["identity"] = new_ident
        rid = row["id"]
        if changed:
            sets = ", ".join(f"{c} = ?" for c in changed) + ", updated = ?"
            vals = [json.dumps(v) if c == "authors" else v for c, v in changed.items()] + [now]
            conn.execute(f'UPDATE "references" SET {sets} WHERE id = ?', (*vals, rid))
        how = "merged" if changed else "unchanged"
    seen = conn.execute(
        "SELECT 1 FROM reference_sources WHERE reference_id = ? AND source = ? AND source_key = ?",
        (rid, rec.source, rec.source_key),
    ).fetchone()
    if seen is None:
        raw = {"fields": {k: v for k, v in fields.items() if k != "authors"}, "conflicts": conflicts}
        conn.execute(
            "INSERT INTO reference_sources(reference_id, source, source_key, fetched_at, raw)"
            " VALUES (?, ?, ?, ?, ?)",
            (rid, rec.source, rec.source_key, rec.fetched_at, json.dumps(raw)),
        )
    elif how == "unchanged":
        pass
    for cited_identity, cited_title in rec.cites:
        conn.execute(
            "INSERT OR IGNORE INTO reference_cites(citing_id, cited_identity, cited_title, source, fetched_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (rid, cited_identity, cited_title, rec.source, rec.fetched_at or now),
        )
    return rid, how


@dataclass
class SyncReport:
    """Structure, not prose: the caller is a model or a CLI printer."""

    sources: dict = field(default_factory=dict)
    embedded: int = 0
    unembedded: int = 0
    embed_error: str | None = None
    conflicts: int = 0
    conflict_samples: list = field(default_factory=list)

    def bucket(self, name: str) -> dict:
        return self.sources.setdefault(
            name, {"seen": 0, "added": 0, "merged": 0, "unchanged": 0, "enriched": 0, "failed": 0}
        )

    def to_dict(self) -> dict:
        return {
            "sources": self.sources, "embedded": self.embedded, "unembedded": self.unembedded,
            "embed_error": self.embed_error, "conflicts": self.conflicts,
            "conflict_samples": self.conflict_samples[:5],
        }


def sync(conn, readers, *, embedder=None, limit: int | None = None) -> SyncReport:
    """Run every reader in order, then embed rows without a vector.

    Offline readers introduce rows; enrichers (network=True) only fill rows
    that exist -- see `Enricher.records(conn, limit)` in library_readers. Each
    reader is its own short transaction, the ingest discipline, so `attest
    serve` keeps working alongside.
    """
    report = SyncReport()
    for reader in readers:
        bucket = report.bucket(reader.name)
        records = reader.records(conn, limit) if reader.network else reader.records()
        for rec in records:
            bucket["seen"] += 1
            if rec.title is None and not (rec.doi or rec.arxiv_id):
                bucket["failed"] += 1
                continue
            if reader.network and rec.title is None:
                # An enricher that found nothing still marks the row as tried.
                bucket["failed"] += 1
                continue
            _, how = upsert(conn, rec)
            bucket["enriched" if reader.network else how] += 1
            if how == "merged" or reader.network:
                raw = conn.execute(
                    "SELECT raw FROM reference_sources WHERE source = ? AND source_key = ?",
                    (rec.source, rec.source_key),
                ).fetchone()
                conflicts = json.loads(raw["raw"]).get("conflicts", {}) if raw else {}
                report.conflicts += len(conflicts)
                report.conflict_samples.extend((rec.source_key, f) for f in conflicts)
        conn.commit()
    if embedder is not None:
        report.embedded, report.unembedded, report.embed_error = embed_missing(conn, embedder, limit)
    else:
        report.unembedded = conn.execute(
            'SELECT count(*) FROM "references" r WHERE NOT EXISTS'
            " (SELECT 1 FROM reference_vectors v WHERE v.rowid = r.id)"
        ).fetchone()[0]
    return report


def status(conn) -> dict:
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    return {
        "references": q('SELECT count(*) FROM "references"'),
        "with_vectors": q("SELECT count(*) FROM reference_vectors"),
        "with_tags": q("SELECT count(DISTINCT reference_id) FROM reference_tags"),
        "with_cites": q("SELECT count(DISTINCT citing_id) FROM reference_cites"),
        "sources": {
            r["source"]: r["n"]
            for r in conn.execute("SELECT source, count(*) n FROM reference_sources GROUP BY source")
        },
    }
```

`embed_missing` is Task 7; for this task stub it as `def embed_missing(conn, embedder, limit): return 0, 0, None` and mark with a comment "filled in Task 7". Replace the lambda with a small `def _count(conn, sql)` if ruff E731 complains (it will — write the def).

- [ ] **Step 4: Run** `uv run pytest tests/test_library.py -q` → pass.
- [ ] **Step 5: Commit** `git commit -am "Library upsert and sync: identity-first lookup, first-wins merge with recorded conflicts, idempotent re-runs"`

---

### Task 6: The three enrichers, cached, behind flags

**Files:**
- Modify: `src/attestation/library_readers.py` (add `ArxivEnricher`, `CrossrefEnricher`, `S2Enricher`, `readers_from_env`), `src/attestation/citations.py` (`Resolver.from_env` reads `ATTEST_BIB_PATHS`, `ATTEST_ZOTERO_PATH`; a shared `_cached_get(cache_dir, url, params) -> tuple[dict|str, fetched_at]`), `.env.sample`
- Create: `tests/fixtures/library/arxiv_query.xml`, `tests/fixtures/library/crossref_work.json`, `tests/fixtures/library/s2_paper.json`
- Test: `tests/test_library_readers.py`, `tests/test_citations.py`

**Interfaces:**
- Produces: enrichers with `name`, `network = True`, `records(conn, limit) -> Iterator[ReferenceRecord]` (records whose `source_key` is the existing row's identity; `title=None` when the wire had nothing); `library_readers.readers_from_env(conn, *, bib_paths=None, zotero_path=None, cache_dir=None, sources=None) -> list` where `sources` filters by name; `citations.web_enabled() -> bool`, `citations.s2_enabled() -> bool` (read env once, at construction of the reader list).

- [ ] **Step 1: Fixtures.** Write minimal but real-shaped responses:

`arxiv_query.xml` — an Atom feed with one `<entry>` for `1706.08566`: `<id>http://arxiv.org/abs/1706.08566v5</id>`, `<title>`, `<summary>` (the SchNet abstract, first sentence is enough), two `<author><name>`, `<arxiv:doi>10.5555/schnet</arxiv:doi>`, `<published>2017-06-26T...`. Use the real arXiv namespace `http://arxiv.org/schemas/atom`.

`crossref_work.json` — `{"message": {"DOI": "10.1038/s41467-022-29939-5", "title": ["E(3)-equivariant graph neural networks..."], "container-title": ["Nature Communications"], "author": [{"family": "Batzner", "given": "Simon"}], "issued": {"date-parts": [[2022, 5, 4]]}, "URL": "https://doi.org/10.1038/s41467-022-29939-5", "abstract": "<jats:p>This work...</jats:p>"}}`.

`s2_paper.json` — `{"paperId": "abc", "title": "E(3)-equivariant ...", "externalIds": {"DOI": "10.1038/s41467-022-29939-5", "ArXiv": "2101.03164"}, "references": [{"title": "SchNet ...", "externalIds": {"ArXiv": "1706.08566", "DOI": "10.5555/schnet"}}, {"title": "Untraceable ref", "externalIds": {}}, {"title": null, "externalIds": null}]}`.

- [ ] **Step 2: Failing tests**

```python
import json

import httpx
import pytest


def _fake_transport(responses: dict[str, tuple[int, bytes, dict]]):
    """URL-prefix -> (status, body, headers). Records every request."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        for prefix, (status, body, headers) in responses.items():
            if str(request.url).startswith(prefix):
                return httpx.Response(status, content=body, headers=headers, request=request)
        return httpx.Response(404, request=request)

    return httpx.MockTransport(handler), calls


def _store_with(conn, **kw):
    from attestation.library import ReferenceRecord, upsert
    kw.setdefault("source", "bibtex:/a.bib"); kw.setdefault("source_key", "k")
    return upsert(conn, ReferenceRecord(**kw))[0]


def test_arxiv_enricher_fills_abstract_authors_and_doi(tmp_path, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, title="SchNet", arxiv_id="1706.08566")
    transport, calls = _fake_transport({
        "http://export.arxiv.org/api/query": (200, (FIX / "arxiv_query.xml").read_bytes(), {})})
    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=transport))
    recs = list(library_readers.ArxivEnricher(cache_dir=tmp_path / "cache").records(conn, None))
    assert len(recs) == 1 and recs[0].doi == "10.5555/schnet" and recs[0].abstract
    assert recs[0].fetched_at is not None and recs[0].source == "arxiv"
    # Cached: a second pass makes no request and keeps the ORIGINAL fetched_at.
    first = recs[0].fetched_at
    recs2 = list(library_readers.ArxivEnricher(cache_dir=tmp_path / "cache").records(conn, None))
    assert len(calls) == 1 and recs2[0].fetched_at == first


def test_s2_enricher_yields_cites_and_backs_off_on_429(tmp_path, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, title="NequIP", doi="10.1038/s41467-022-29939-5")
    body = (FIX / "s2_paper.json").read_bytes()
    seq = iter([(429, b"", {"Retry-After": "0"}), (200, body, {})])

    def handler(request):
        status, content, headers = next(seq)
        return httpx.Response(status, content=content, headers=headers, request=request)

    monkeypatch.setattr(library_readers, "_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(library_readers, "_sleep", lambda s: None)
    recs = list(library_readers.S2Enricher(cache_dir=tmp_path / "c").records(conn, None))
    assert recs[0].arxiv_id == "2101.03164"
    # Two traceable references; the one with no ids and no title is dropped.
    assert recs[0].cites == [("doi:10.5555/schnet", "SchNet ..."), ("title:untraceable ref:-", "Untraceable ref")]


def test_no_request_is_made_with_the_flags_unset(tmp_path, monkeypatch):
    from attestation.library import sync
    conn = get_db(tmp_path / "t.db")
    def explode(*a, **k): raise AssertionError("network touched")
    monkeypatch.setattr(httpx, "Client", explode); monkeypatch.setattr(httpx, "get", explode)
    monkeypatch.delenv("ATTEST_CITATION_WEB", raising=False); monkeypatch.delenv("ATTEST_CITATION_SCHOLAR", raising=False)
    readers = library_readers.readers_from_env(conn, bib_paths=[FIX / "sample.bib"], zotero_path=tmp_path / "none.sqlite")
    assert [r.name for r in readers] == ["bibtex", "feed"]
    sync(conn, readers)


def test_flags_arm_the_enrichers_at_construction(tmp_path, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    monkeypatch.setenv("ATTEST_CITATION_WEB", "1"); monkeypatch.setenv("ATTEST_CITATION_SCHOLAR", "1")
    names = [r.name for r in library_readers.readers_from_env(conn, bib_paths=[], zotero_path=tmp_path / "n")]
    assert names == ["feed", "arxiv", "crossref", "s2"]
```

Add to `tests/test_citations.py`: `Resolver.from_env` honours `ATTEST_BIB_PATHS` (two paths joined by `os.pathsep`) and `ATTEST_ZOTERO_PATH`.

- [ ] **Step 3: Run** → FAIL.

- [ ] **Step 4: Implement.** In `citations.py`:

```python
def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip() not in ("", "0", "false")


def web_enabled() -> bool:
    """ATTEST_CITATION_WEB: CrossRef and the arXiv API. Read by the caller that
    BUILDS readers, never by a reader at call time."""
    return _flag("ATTEST_CITATION_WEB")


def s2_enabled() -> bool:
    """ATTEST_CITATION_SCHOLAR: Semantic Scholar reference lists. A second flag
    because reference lists are a larger, rate-limited surface."""
    return _flag("ATTEST_CITATION_SCHOLAR")
```

and in `from_env`: `paths = list(bib_paths) if bib_paths else _bib_paths_from_env()` where `_bib_paths_from_env()` returns `[Path(p) for p in os.environ["ATTEST_BIB_PATHS"].split(os.pathsep) if p]` when set else `sorted(Path.cwd().glob("*.bib"))`; `zotero_path = zotero_path or os.environ.get("ATTEST_ZOTERO_PATH") or None`; use `web_enabled()` in place of the inline check.

In `library_readers.py`:

```python
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import httpx

from attestation.citations import _chmod, s2_enabled, web_enabled
from attestation.library import identity

DEFAULT_CACHE = Path.home() / ".hermes" / "citation-cache"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def _client() -> httpx.Client:
    """One place to build the HTTP client, so tests can swap the transport."""
    return httpx.Client(timeout=15.0, headers={"User-Agent": "attestation/library"})


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _cache_path(cache_dir: Path, url: str) -> Path:
    import hashlib
    return cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.json"


def _cached_get(cache_dir: Path, url: str) -> tuple[bytes | None, str | None, str | None]:
    """(body, fetched_at, error). A cache hit keeps the ORIGINAL fetched_at.

    Three attempts on 429/5xx, sleeping Retry-After or 2 s. Every other
    failure is returned as `error`, never raised: a dead network is an absent
    source, and the sync must finish.
    """
    path = _cache_path(cache_dir, url)
    if path.is_file():
        rec = json.loads(path.read_text())
        return rec["body"].encode(), rec["fetched_at"], None
    error = None
    for _attempt in range(3):
        try:
            with _client() as client:
                resp = client.get(url)
        except httpx.HTTPError as exc:
            return None, None, f"{type(exc).__name__}: {exc}"
        if resp.status_code == 429 or resp.status_code >= 500:
            error = f"HTTP {resp.status_code}"
            _sleep(float(resp.headers.get("Retry-After", "2") or 2))
            continue
        if resp.status_code != 200:
            return None, None, f"HTTP {resp.status_code}"
        fetched = datetime.now(UTC).date().isoformat()
        existed = cache_dir.is_dir()
        cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not existed:
            _chmod(cache_dir, 0o700)
        path.write_text(json.dumps({"url": url, "fetched_at": fetched, "body": resp.text}))
        _chmod(path, 0o600)
        return resp.content, fetched, None
    return None, None, error
```

Enrichers select their rows themselves:

```python
class _Enricher:
    network = True
    name = ""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE

    def _todo(self, conn, limit, where: str):
        sql = (
            'SELECT * FROM "references" r WHERE ' + where +
            " AND NOT EXISTS (SELECT 1 FROM reference_sources s WHERE s.reference_id = r.id AND s.source = ?)"
            " ORDER BY r.updated"
        )
        params: tuple = (self.name,)
        if limit is not None:
            sql += " LIMIT ?"
            params += (limit,)
        return conn.execute(sql, params).fetchall()


class ArxivEnricher(_Enricher):
    name = "arxiv"

    def records(self, conn, limit):
        rows = self._todo(conn, limit, "r.arxiv_id IS NOT NULL")
        for i in range(0, len(rows), 50):
            batch = rows[i : i + 50]
            ids = ",".join(r["arxiv_id"] for r in batch)
            url = f"http://export.arxiv.org/api/query?id_list={ids}&max_results={len(batch)}"
            body, fetched, error = _cached_get(self.cache_dir, url)
            found: dict[str, ReferenceRecord] = {}
            if body:
                for entry in ET.fromstring(body).iter(f"{_ATOM}entry"):
                    aid = (entry.findtext(f"{_ATOM}id") or "").rsplit("/", 1)[-1]
                    aid = re.sub(r"v\d+$", "", aid)
                    doi_el = entry.find(f"{_ARXIV_NS}doi")
                    found[aid] = ReferenceRecord(
                        source=self.name, source_key=aid,
                        title=" ".join((entry.findtext(f"{_ATOM}title") or "").split()) or None,
                        authors=[a.findtext(f"{_ATOM}name") or "" for a in entry.iter(f"{_ATOM}author")],
                        year=int((entry.findtext(f"{_ATOM}published") or "0000")[:4]) or None,
                        doi=doi_el.text if doi_el is not None else None,
                        arxiv_id=aid,
                        abstract=" ".join((entry.findtext(f"{_ATOM}summary") or "").split()) or None,
                        url=f"https://arxiv.org/abs/{aid}", fetched_at=fetched,
                    )
            for r in batch:
                yield found.get(r["arxiv_id"], ReferenceRecord(source=self.name, source_key=r["arxiv_id"], arxiv_id=r["arxiv_id"], fetched_at=fetched))
```

`CrossrefEnricher` (`name = "crossref"`, where `r.doi IS NOT NULL`): one GET per DOI to `https://api.crossref.org/works/{doi}`; map `title[0]`, `container-title[0]` → venue, `author[].family, given`, `issued.date-parts[0][0]` → year, `URL`, `abstract` with JATS tags stripped by `re.sub(r"<[^>]+>", "", ...)`. Yield an empty record (title None) on error.

`S2Enricher` (`name = "s2"`, where `r.doi IS NOT NULL OR r.arxiv_id IS NOT NULL`): per row, id = `DOI:<doi>` else `arXiv:<arxiv_id>`; GET `https://api.semanticscholar.org/graph/v1/paper/{id}?fields=title,externalIds,references.title,references.externalIds`; sleep 1.0 between UNCACHED requests (`_sleep(1.0)` only after a real fetch — detect via cache path existence before the call); `cites` = for each reference with a title or an id: `identity(ext.get("DOI"), ext.get("ArXiv"), title, None)` guarded by try/except `ValueError` → skip; record `doi`/`arxiv_id` from the paper's own `externalIds`.

Note `s2` records set `source_key` to the row identity and carry `doi`/`arxiv_id` so `upsert` finds the same row.

```python
def readers_from_env(conn, *, bib_paths=None, zotero_path=None, cache_dir=None, sources=None):
    """The reader list, flags read HERE. `sources` filters by name."""
    from attestation.citations import DEFAULT_ZOTERO, _bib_paths_from_env
    zp = Path(zotero_path) if zotero_path else Path(os.environ.get("ATTEST_ZOTERO_PATH") or DEFAULT_ZOTERO)
    readers: list = []
    paths = [Path(p) for p in bib_paths] if bib_paths is not None else _bib_paths_from_env()
    if paths:
        readers.append(BibtexRecords(paths))
    if zp.is_file():
        readers.append(ZoteroRecords(zp))
    readers.append(FeedRecords(conn))
    if web_enabled():
        readers += [ArxivEnricher(cache_dir), CrossrefEnricher(cache_dir)]
    if s2_enabled():
        readers.append(S2Enricher(cache_dir))
    if sources:
        readers = [r for r in readers if r.name in set(sources)]
    return readers
```

`.env.sample`: add `#ATTEST_BIB_PATHS=`, `#ATTEST_ZOTERO_PATH=`, `#ATTEST_CITATION_SCHOLAR=1` beside the existing web flag with one comment line each (S2: "reference lists, one request per second, cached forever").

- [ ] **Step 5: Run** `uv run pytest tests/test_library_readers.py tests/test_citations.py tests/test_library.py -q` → pass.
- [ ] **Step 6: Commit** `git add -A src tests .env.sample && git commit -m "arXiv, CrossRef and Semantic Scholar enrichers: fill-only, cached content-addressed, armed by flags read at construction"`

---

### Task 7: Embeddings and search

**Files:**
- Modify: `src/attestation/rank.py` (move floor + vector search), `src/attestation/mcp/feed.py` (import them), `src/attestation/library.py` (`embed_missing`, `search`, `SearchHit`, `to_reference`)
- Test: `tests/test_library.py`; run `tests/test_search.py`, `tests/test_rank_relevance.py` unchanged

**Interfaces:**
- Produces: `rank.RELEVANCE_FLOOR`, `rank.RELEVANCE_ANCHOR`, `rank.apply_relevance_floor(sims) -> dict`, `rank.vector_search(conn, embedder, query, k, table="item_vectors") -> dict[int, float]`; `library.embed_missing(conn, embedder, limit) -> tuple[int, int, str | None]`; `library.search(conn, query, *, embedder=None, author=None, year=None, year_from=None, year_to=None, tag=None, source=None, limit=10) -> SearchResult` with `SearchResult(hits: list[SearchHit], semantic: bool, caveat: str | None, n_matches: int)`; `SearchHit.to_row()`; `library.to_reference(row) -> citations.Reference`.

- [ ] **Step 1: Failing tests**

```python
def test_search_is_semantic_with_an_embedder_and_says_so(tmp_path, fake_embedder):
    from attestation.db import get_db
    conn = get_db(tmp_path / "t.db")
    a = _store_with(conn, bib_key="a", title="E(3)-equivariant interatomic potentials", abstract="force fields", year=2022)
    b = _store_with(conn, bib_key="b", title="Sourdough starter maintenance", abstract="bread", year=2019)
    embedded, missing, err = library.embed_missing(conn, fake_embedder, None)
    assert (embedded, missing, err) == (2, 0, None)
    # FakeEmbedder is hash-based, so plant the query vector on `a` to make the
    # ranking deterministic: what is tested is the plumbing (KNN + floor +
    # envelope), not embeddinggemma.
    q = fake_embedder.embed_query("equivariant force fields")
    conn.execute("UPDATE reference_vectors SET embedding = ? WHERE rowid = ?", (q.tobytes(), a))
    res = library.search(conn, "equivariant force fields", embedder=fake_embedder, limit=5)
    assert res.semantic is True and res.hits[0].id == a and res.hits[0].similarity > 0.99
    assert res.hits[0].to_row()["sources"] == ["bibtex:/a.bib"]


def test_search_without_an_embedder_is_fielded_and_says_so(tmp_path):
    from attestation.db import get_db
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, bib_key="a", title="SchNet", year=2017, authors=["Schütt, Kristof"])
    _store_with(conn, bib_key="b", title="NequIP", year=2022, authors=["Batzner, Simon"])
    res = library.search(conn, "schnet")
    assert res.semantic is False and "no embedder" in res.caveat
    assert [h.bib_key for h in res.hits] == ["a"]
    assert [h.bib_key for h in library.search(conn, "", author="batzner").hits] == ["b"]
    assert [h.bib_key for h in library.search(conn, "", year_from=2020).hits] == ["b"]


def test_an_identifier_query_is_a_direct_lookup(tmp_path):
    from attestation.db import get_db
    conn = get_db(tmp_path / "t.db")
    _store_with(conn, bib_key="n", title="NequIP", doi="10.1038/s41467-022-29939-5", arxiv_id="2101.03164")
    for q in ("10.1038/S41467-022-29939-5", "arXiv:2101.03164v1", "n"):
        res = library.search(conn, q)
        assert [h.bib_key for h in res.hits] == ["n"], q
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement.** Move `RELEVANCE_FLOOR`, `RELEVANCE_ANCHOR`, `_vector_search` (renamed `vector_search`, with a `table` parameter), `_apply_relevance_floor` (renamed `apply_relevance_floor`) and their comments from `mcp/feed.py` into `rank.py`. In `mcp/feed.py` replace the definitions with:

```python
from attestation.rank import (  # noqa: F401 -- re-exported for tests that patch them here
    RELEVANCE_ANCHOR,
    RELEVANCE_FLOOR,
    apply_relevance_floor as _apply_relevance_floor,
    vector_search as _vector_search,
)
```

(check `grep -rn "_apply_relevance_floor\|_vector_search\|RELEVANCE_" tests/` and keep whatever names tests import; ruff `I` wants the import sorted with the others).

In `library.py`:

```python
import numpy as np


def embed_missing(conn, embedder, limit: int | None) -> tuple[int, int, str | None]:
    """Embed rows with no vector: (embedded, still_missing, error).

    One embed call per row outside any transaction, then one short write --
    the ingest discipline. An embedder that raises stops the pass and is
    reported once; the rows stay unembedded and `search` degrades to fielded.
    """
    sql = ('SELECT r.id, r.title, r.abstract FROM "references" r WHERE NOT EXISTS'
           " (SELECT 1 FROM reference_vectors v WHERE v.rowid = r.id) ORDER BY r.id")
    rows = conn.execute(sql + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    done, error = 0, None
    for row in rows:
        try:
            vec = embedder.embed_document(row["title"], row["abstract"] or "")
        except Exception as exc:  # noqa: BLE001 -- the embedder is a network client; any
            # failure (connection refused, timeout, a 500) means the same
            # thing here: stop this pass, keep the rows, report once.
            error = f"{type(exc).__name__}: {exc}"
            break
        conn.execute("INSERT INTO reference_vectors(rowid, embedding) VALUES (?, ?)",
                     (row["id"], np.asarray(vec, dtype=np.float32).tobytes()))
        done += 1
    conn.commit()
    missing = conn.execute('SELECT count(*) FROM "references" r WHERE NOT EXISTS'
                           " (SELECT 1 FROM reference_vectors v WHERE v.rowid = r.id)").fetchone()[0]
    return done, missing, error
```

That `noqa: BLE001` is the 8th; update `test_architecture.py`'s count AND the CLAUDE.md reliability line ("8 inline sites ... library") in the same commit.

```python
@dataclass
class SearchHit:
    id: int
    identity: str
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    url: str | None
    bib_key: str | None
    venue: str | None
    sources: list[str]
    tags: list[str]
    n_tags: int
    similarity: float | None = None

    def to_row(self) -> dict:
        return {
            "id": self.id, "key": self.bib_key or self.identity, "title": self.title[:223],
            "authors": self.authors[:6], "n_authors": len(self.authors), "year": self.year,
            "doi": self.doi, "url": self.url, "venue": self.venue, "sources": self.sources,
            "tags": self.tags[:3], "n_tags": self.n_tags, "similarity": self.similarity,
        }


@dataclass
class SearchResult:
    hits: list[SearchHit]
    semantic: bool
    caveat: str | None
    n_matches: int


_ID_QUERY = re.compile(r"^(10\.\d{4,9}/\S+|(?:arxiv:)?\d{4}\.\d{4,5}(?:v\d+)?|(?:arxiv:)?[a-z\-]+/\d{7})$", re.IGNORECASE)


def _hit(conn, row, similarity=None) -> SearchHit:
    sources = [r["source"] for r in conn.execute(
        "SELECT source FROM reference_sources WHERE reference_id = ? ORDER BY source", (row["id"],))]
    tags = [r["tag"] for r in conn.execute(
        "SELECT tag FROM reference_tags WHERE reference_id = ? ORDER BY tag", (row["id"],))]
    return SearchHit(id=row["id"], identity=row["identity"], title=row["title"],
                     authors=json.loads(row["authors"]), year=row["year"], doi=row["doi"], url=row["url"],
                     bib_key=row["bib_key"], venue=row["venue"], sources=sources, tags=tags,
                     n_tags=len(tags), similarity=similarity)


def _fielded_where(author, year, year_from, year_to, tag, source):
    where, params = [], []
    if author:
        where.append("EXISTS (SELECT 1 FROM json_each(r.authors) a WHERE lower(a.value) LIKE ?)")
        params.append(f"%{author.lower()}%")
    if year is not None:
        where.append("r.year = ?"); params.append(year)
    if year_from is not None:
        where.append("r.year >= ?"); params.append(year_from)
    if year_to is not None:
        where.append("r.year <= ?"); params.append(year_to)
    if tag:
        where.append("EXISTS (SELECT 1 FROM reference_tags t WHERE t.reference_id = r.id AND t.tag = ?)")
        params.append(tag)
    if source:
        where.append("EXISTS (SELECT 1 FROM reference_sources s WHERE s.reference_id = r.id AND s.source LIKE ?)")
        params.append(f"{source}%")
    return (" AND ".join(where) or "1=1"), params


def lookup_row(conn, key: str):
    """A row by identity, DOI, arXiv id, or bib key -- the direct forms."""
    k = key.strip()
    for sql, val in (
        ('SELECT * FROM "references" WHERE identity = ?', k.lower()),
        ('SELECT * FROM "references" WHERE doi = ?', normalise_doi(k)),
        ('SELECT * FROM "references" WHERE arxiv_id = ?', normalise_arxiv(k)),
        ('SELECT * FROM "references" WHERE bib_key = ? COLLATE NOCASE', k),
    ):
        if val and (row := conn.execute(sql, (val,)).fetchone()):
            return row
    return None


def search(conn, query: str, *, embedder=None, author=None, year=None, year_from=None,
           year_to=None, tag=None, source=None, limit: int = 10) -> SearchResult:
    """Semantic when it can be, fielded when it must be, and it says which.

    Mirrors the feed's search: KNN over reference_vectors for 4x limit
    candidates, the relative relevance floor, fielded filters, a literal
    boost that never excludes. Falls back to substring over title, abstract,
    authors and key when there is no embedder or no vectors, with a caveat
    a caller cannot mistake for a semantic answer.
    """
    from attestation.rank import apply_relevance_floor, vector_search

    q = (query or "").strip()
    where, params = _fielded_where(author, year, year_from, year_to, tag, source)
    if q and _ID_QUERY.match(q) and (row := lookup_row(conn, q)):
        return SearchResult([_hit(conn, row)], semantic=False, caveat=None, n_matches=1)
    if q and not _ID_QUERY.match(q) and (row := lookup_row(conn, q)):
        return SearchResult([_hit(conn, row)], semantic=False, caveat=None, n_matches=1)
    n_vectors = conn.execute("SELECT count(*) FROM reference_vectors").fetchone()[0]
    if q and embedder is not None and n_vectors:
        sims = apply_relevance_floor(vector_search(conn, embedder, q, k=4 * limit, table="reference_vectors"))
        if sims:
            rows = conn.execute(
                f'SELECT * FROM "references" r WHERE r.id IN ({",".join("?" * len(sims))}) AND {where}',
                (*sims.keys(), *params),
            ).fetchall()
            words = [w for w in normalise_title(q).split() if len(w) > 2]

            def score(r):
                text = normalise_title(f"{r['title']} {r['abstract'] or ''}")
                boost = 0.02 * sum(w in text for w in words)
                return sims[r["id"]] + boost

            rows.sort(key=score, reverse=True)
            hits = [_hit(conn, r, round(sims[r["id"]], 4)) for r in rows[:limit]]
            caveat = None if n_vectors == conn.execute('SELECT count(*) FROM "references"').fetchone()[0] \
                else f"{n_vectors} of the references are embedded; run `attest library embed`"
            return SearchResult(hits, semantic=True, caveat=caveat, n_matches=len(rows))
    like = f"%{q.lower()}%" if q else "%"
    rows = conn.execute(
        f'SELECT * FROM "references" r WHERE {where} AND ('
        " lower(r.title) LIKE ? OR lower(coalesce(r.abstract, '')) LIKE ?"
        " OR lower(r.authors) LIKE ? OR lower(coalesce(r.bib_key, '')) LIKE ?)"
        " ORDER BY r.year DESC, r.title",
        (*params, like, like, like, like),
    ).fetchall()
    reason = "no embedder" if embedder is None else ("no vectors yet" if not n_vectors else "no semantic hits")
    caveat = f"substring search only ({reason}); run `attest library embed` for semantic search" \
        if reason != "no semantic hits" else "no semantic hit cleared the relevance floor; substring results"
    return SearchResult([_hit(conn, r) for r in rows[:limit]], semantic=False, caveat=caveat, n_matches=len(rows))


def to_reference(conn, row):
    """A store row as the `Reference` the cite.* tools already emit."""
    from attestation.citations import Reference
    sources = conn.execute(
        "SELECT source, fetched_at FROM reference_sources WHERE reference_id = ? ORDER BY fetched_at IS NOT NULL, source",
        (row["id"],)).fetchall()
    first = sources[0] if sources else None
    return Reference(key=row["bib_key"] or row["identity"], title=row["title"], authors=json.loads(row["authors"]),
                     year=row["year"], doi=row["doi"], arxiv_id=row["arxiv_id"], url=row["url"],
                     source="library:" + (first["source"] if first else "?"), fetched_at=first["fetched_at"] if first else None)
```

(Split `search` into helpers if the complexity ratchet complains: `_semantic_search(...)` and `_substring_search(...)`.)

- [ ] **Step 4: Run** `uv run pytest tests/test_library.py tests/test_search.py tests/test_rank_relevance.py tests/test_architecture.py -q` → pass (after the noqa count and CLAUDE.md line).
- [ ] **Step 5: Commit** `git add -A && git commit -m "Library search: semantic over reference_vectors with the feed's relative floor, fielded filters, honest fallback; floor policy moves to rank.py"`

---

### Task 8: Tagging references

**Files:**
- Modify: `src/attestation/features.py` (`run_reference_tagging`), `src/attestation/library.py` nothing
- Test: `tests/test_features.py`

- [ ] **Step 1: Failing test** — find how `tests/test_features.py` fakes `chat_fn` for `run_tagging` (a callable returning a JSON string) and copy it:

```python
def test_run_reference_tagging_writes_reference_tags(tmp_path):
    from attestation.db import get_db
    from attestation.features import run_reference_tagging
    from attestation.library import ReferenceRecord, upsert
    conn = get_db(tmp_path / "t.db")
    rid = upsert(conn, ReferenceRecord(source="bibtex:/a", source_key="k", bib_key="k",
                                       title="SchNet", abstract="quantum chemistry"))[0]
    calls = []
    def chat_fn(messages, model):
        calls.append(messages)
        return '{"content_type": "paper", "tags": ["quantum-chemistry", "graph-neural-networks"]}'
    stats = run_reference_tagging(conn, chat_fn, "m", limit=None)
    assert stats["tagged"] == 1
    assert {r["tag"] for r in conn.execute("SELECT tag FROM reference_tags WHERE reference_id = ?", (rid,))} == {"quantum-chemistry", "graph-neural-networks"}
    assert "SchNet" in calls[0][-1]["content"]           # rendered by tag_messages
    assert run_reference_tagging(conn, chat_fn, "m")["tagged"] == 0   # already tagged
```

(Match `chat_fn`'s real signature from `run_tagging`/`tag_one_item` — read them first; adjust the fake accordingly.)

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** in `features.py`, next to `run_tagging`, reusing `tag_messages`, `ItemTags` validation and the same retry-once policy `tag_one_item` uses (read it and mirror; do not duplicate the parsing — if `tag_one_item`'s parse step is a separate helper, call it; if not, extract `_parse_tag_reply(text) -> ItemTags | None` and use it from both):

```python
def run_reference_tagging(conn, chat_fn, model: str, limit: int | None = None) -> dict:
    """LLM-tag references with no tags yet, through the ONE tagging renderer."""
    vocab = tag_vocabulary(conn)
    sql = ('SELECT r.id, r.title, coalesce(r.abstract, "") AS summary FROM "references" r'
           " WHERE NOT EXISTS (SELECT 1 FROM reference_tags t WHERE t.reference_id = r.id) ORDER BY r.id")
    rows = conn.execute(sql + (" LIMIT ?" if limit is not None else ""), (limit,) if limit is not None else ()).fetchall()
    stats = {"tagged": 0, "failed": 0, "chat_down": False}
    for row in rows:
        parsed = _tag_with_retry(chat_fn, model, tag_messages(row["title"], row["summary"], vocab))
        if parsed is None:
            stats["failed"] += 1
            continue
        conn.executemany("INSERT OR IGNORE INTO reference_tags(reference_id, tag) VALUES (?, ?)",
                         [(row["id"], t) for t in parsed.tags])
        conn.commit()
        stats["tagged"] += 1
    return stats
```

where `_tag_with_retry` is whatever `tag_one_item` already does for one item (extract it if it is inline; keep `tag_one_item` behaviour byte-identical — `tests/test_tag_prompt.py` and `test_features.py` pin it).

- [ ] **Step 4: Run** `uv run pytest tests/test_features.py tests/test_tag_prompt.py -q` → pass.
- [ ] **Step 5: Commit** `git commit -am "Tag references through the one tagging renderer"`

---

### Task 9: The tools

**Files:**
- Modify: `src/attestation/mcp/citation.py`, `src/attestation/citations.py` (`Resolver` store-first), `src/attestation/claims.py` (nothing if `check_citations` goes through `Resolver.lookup`), `CLAUDE.md`, `README.md`, `docs/guides/agents.md`, `src/attestation/skills/attestation-knowledge/SKILL.md`, `src/attestation/skills/attestation-setup/SKILL.md` (only if it quotes 46)
- Test: `tests/test_citations.py`, `tests/test_response_size.py`, `tests/test_architecture.py` (run), `tests/test_skill_files.py` (run)

**Interfaces:**
- `Resolver(readers, store=None)` where `store` is a zero-arg callable returning a connection, or None; `lookup` consults the store first via `library.lookup_row` + `to_reference`, then readers. `from_env(..., store=None)`.
- Tools: `cite.sync(sources: list[str] | None = None, limit: int | None = None)`; `cite.search(query, limit=5, author=None, year=None, tag=None)`; `cite.lookup(key)`; `cite.sources()`.

- [ ] **Step 1: Failing tests**

`tests/test_citations.py`:

```python
def test_resolver_consults_the_store_before_the_readers(tmp_path):
    from attestation.db import get_db
    from attestation.library import ReferenceRecord, upsert
    db = tmp_path / "t.db"
    conn = get_db(db)
    upsert(conn, ReferenceRecord(source="zotero", source_key="Z1", bib_key="Z1", title="Stored", doi="10.1/stored"))
    conn.commit(); conn.close()
    (tmp_path / "refs.bib").write_text("@article{disk, title={On disk}, year={2020}}\n")
    resolver = citations.Resolver.from_env(bib_paths=[tmp_path / "refs.bib"], zotero_path=tmp_path / "no", store=lambda: get_db(db))
    assert resolver.lookup("10.1/stored").source == "library:zotero"
    assert resolver.lookup("disk").source == "bibtex"
    assert resolver.lookup("Z1").title == "Stored"
```

`tests/test_response_size.py` (append; use its existing `stocked`-style pattern but for references):

```python
def test_cite_search_is_bounded_at_the_cap(tmp_path, monkeypatch, fake_embedder):
    import json
    from attestation.db import get_db
    from attestation.library import ReferenceRecord, upsert
    from attestation.mcp import citation
    db = tmp_path / "t.db"; monkeypatch.setenv("RSS_DB", str(db))
    conn = get_db(db)
    long_title = "Retraction Note: " + "Photocatalytic Degradation of Organic Pollutants " * 4
    for i in range(30):
        upsert(conn, ReferenceRecord(source="bibtex:/a.bib", source_key=f"k{i}", bib_key=f"k{i}",
                                     title=f"{long_title} {i}", year=2020,
                                     authors=[f"Author{j}, Name" for j in range(12)],
                                     abstract="x" * 2000, doi=f"10.1/{i}"))
        conn.executemany("INSERT INTO reference_tags VALUES (?, ?)", [(i + 1, f"tag-{j}") for j in range(6)])
    conn.commit(); conn.close()
    out = citation._search("pollutants", 13)
    assert len(json.dumps(out)) <= HARD_RESPONSE_CEILING, len(json.dumps(out))
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

`citations.Resolver`:

```python
    def __init__(self, readers, store=None):
        self.readers = list(readers)
        self.store = store   # zero-arg callable -> sqlite3.Connection, or None

    def lookup(self, key):
        if self.store is not None:
            from attestation import library
            conn = self.store()
            try:
                row = library.lookup_row(conn, key)
                if row is not None:
                    return library.to_reference(conn, row)
            finally:
                conn.close()
        for reader in self.readers: ...
```

`from_env(cls, *, zotero_path=None, bib_paths=None, cache_dir=None, store=None)` passes it through. `search` stays reader-only here (the tool calls `library.search` directly).

`mcp/citation.py` — `_resolver()` becomes `_resolver(conn=None)`; when the tool has a connection, pass `store=lambda: conn`-style factory. Because `@tool` closes the connection after the body, build the resolver inside the body with a factory that returns the SAME `conn` and does not close it: give `Resolver` a `store_keepalive` flag, or simpler: `library.lookup_row`/`to_reference` are called directly in `_lookup` before falling back to `_resolver().lookup(key)`. Do the simpler thing:

```python
@tool(empty={"reference": None}, label="cite_lookup")
def _lookup(conn, key: str) -> dict:
    from attestation import library
    row = library.lookup_row(conn, key)
    if row is not None:
        ref = library.to_reference(conn, row)
        sources = [dict(r) for r in conn.execute(
            "SELECT source, source_key, fetched_at, raw FROM reference_sources WHERE reference_id = ?", (row["id"],))]
        conflicts = {s["source"]: json.loads(s.pop("raw")).get("conflicts", {}) for s in sources}
        return {"reference": ref.to_row(), "sources": sources,
                "conflicts": {k: v for k, v in conflicts.items() if v}}
    resolver = _resolver()
    found = resolver.lookup(key)
    if found is None:
        configured = ", ".join(s["name"] for s in resolver.sources()) or "none"
        raise ToolError(f"no source has {key!r} (configured: {configured}; library store: "
                        f"{library.status(conn)['references']} references)")
    return {"reference": found.to_row(), "sources": [], "conflicts": {}}


@tool(empty={"references": [], "n_matches": 0, "semantic": False, "caveat": None}, label="cite_search")
def _search(conn, query: str, limit: int = 5, author: str | None = None, year: int | None = None,
            tag: str | None = None) -> dict:
    from attestation import library
    from attestation.mcp._shared import clamp_limit
    limit = clamp_limit(limit)
    if library.status(conn)["references"] == 0:
        matches = _resolver().search(query)
        return {"references": [r.to_row() for r in matches[:limit]], "n_matches": len(matches),
                "semantic": False, "caveat": "library store is empty; substring search over the disk readers -- run attest library sync"}
    res = library.search(conn, query, embedder=_embedder(), author=author, year=year, tag=tag, limit=limit)
    return {"references": [h.to_row() for h in res.hits], "n_matches": res.n_matches,
            "semantic": res.semantic, "caveat": res.caveat}


def _embedder():
    """The embedder, or None when the model server is unreachable -- search
    degrades to fielded rather than failing (rank.py's policy)."""
    from attestation.embed import Embedder
    try:
        return Embedder()
    except Exception:  # noqa: BLE001 -- constructing the client can fail on config alone;
        # a missing embedder is the documented fielded-search path, not an error.
        return None
```

Hmm — that is a 9th BLE001. Check whether `Embedder()` construction can actually raise (it reads env only). If it cannot, drop the try: return `Embedder()` and let `library.search` catch embed failures — but `vector_search` calls `embedder.embed_query` which raises on a dead server. So catch in `library.search`: wrap the `vector_search` call in `try/except httpx.HTTPError` → treat as no semantic hits with caveat "embedding server unreachable". That is a typed except, no noqa. Do that instead and keep `_embedder()` trivial.

```python
@tool(empty={"sources": [], "store": {}}, label="cite_sources")
def _sources(conn) -> dict:
    from attestation import citations, library
    sources = _resolver().sources()
    return {"sources": sources, "store": library.status(conn),
            "offline": not (any(s["network"] for s in sources) or citations.s2_enabled())}


@tool(empty=lambda kw: {"sources": {}, "embedded": 0, "unembedded": 0, "conflicts": 0}, label="cite_sync")
def _sync(conn, sources: list[str] | None = None, limit: int | None = None) -> dict:
    from attestation import library, library_readers
    readers = library_readers.readers_from_env(conn, sources=sources)
    report = library.sync(conn, readers, embedder=_embedder(), limit=limit)
    out = report.to_dict()
    out["message"] = ", ".join(f"{k}: +{v['added']} added, {v['merged']} merged, {v['unchanged']} unchanged"
                               + (f", {v['enriched']} enriched" if v["enriched"] else "")
                               + (f", {v['failed']} failed" if v["failed"] else "")
                               for k, v in out["sources"].items()) or "no sources configured"
    return out
```

(`embed_missing` must also catch `httpx.HTTPError` typed rather than BLE001 — revise Task 7 accordingly: `except (httpx.HTTPError, OSError) as exc`. Then no new noqa sites at all; leave the count at 7.)

Register in `register(mcp)`:

```python
    @mcp.tool(name="cite.sync")
    def cite_sync(
        sources: Annotated[list[str] | None, Field(description="subset of bibtex, zotero, feed, arxiv, crossref, s2; default all configured")] = None,
        limit: Annotated[int | None, Field(description="max rows per enricher / embed pass")] = None,
    ) -> dict:
        """Read every configured library into the one deduplicated store.

        BibTeX files, Zotero and the feed's own items with a DOI or arXiv id
        become rows; the same paper from three sources is one row with three
        source entries. arXiv/CrossRef (ATTEST_CITATION_WEB) and Semantic
        Scholar reference lists (ATTEST_CITATION_SCHOLAR) only run if the operator
        set the flag before the server started -- this tool cannot arm them.
        Idempotent: re-running with unchanged sources changes nothing.
        """
        return _sync(sources, limit)
```

Update `cite.search`'s signature/docstring ("semantic when the library is embedded; `semantic` in the reply says which") and `cite.sources`' docstring (S2 named as the second possible network reader).

Docs in the same commit: `CLAUDE.md` line 5 "47 MCP tools", line 50 "MCP surface: 47 tools by default (2026-09-05, cite.sync added) NAMESPACED as feed.*(20) sym.*(8) runs.*(8) kg.*(6) cite.*(5)"; `README.md:88` 47; `docs/guides/agents.md:64` "exposes 47 tools" and add a `cite.sync` row to its tool table, `:324` "all 47"; `attestation-knowledge/SKILL.md`: one line naming `cite.sync` and `cite.search`'s `semantic` flag under its cite section (keep the "for you to read, not a literal call string" phrase intact; run `tests/test_skill_files.py`). Also the CLAUDE.md `Agent surfaces` line: knowledge count 12 → 13 with `ATTEST_EXPAND=1` (re-measure with the documented one-liner and paste the real number).

- [ ] **Step 4: Run** `uv run pytest tests/test_citations.py tests/test_response_size.py tests/test_architecture.py tests/test_skill_files.py tests/test_mcp_server.py tests/test_tool_envelope.py -q` → pass.
- [ ] **Step 5: Commit** `git add -A && git commit -m "cite.sync and a store-first cite.lookup/search/sources; the surface is 47 tools and every doc that counts them says so"`

---

### Task 10: CLI, guide, and the measurements

**Files:**
- Modify: `src/attestation/cli.py`, `docs/guides/claims-and-citations.md`, `docs/reference/cli.md` (regenerated), `docs/superpowers/specs/2026-09-05-library-store-design.md` (§8 numbers)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Failing test** (append to `tests/test_cli.py`, following its existing `main([...])`/capsys pattern):

```python
def test_library_sync_search_and_status(tmp_path, monkeypatch, capsys):
    from attestation.cli import main
    db = tmp_path / "t.db"; monkeypatch.setenv("ATTEST_DB", str(db))
    bib = tmp_path / "refs.bib"
    bib.write_text("@article{schnet, title={SchNet}, author={Schütt, K.}, year={2017}, doi={10.5555/schnet}}\n")
    monkeypatch.setenv("ATTEST_BIB_PATHS", str(bib))
    assert main(["library", "sync", "--sources", "bibtex"]) == 0
    assert "bibtex: +1 added" in capsys.readouterr().out
    assert main(["library", "search", "schnet"]) == 0
    out = capsys.readouterr().out
    assert "SchNet" in out and "substring" in out          # no embedder in tests -> fielded, said aloud
    assert main(["library", "status"]) == 0
    assert '"references": 1' in capsys.readouterr().out
```

- [ ] **Step 2: Run** → FAIL (argparse error).

- [ ] **Step 3: Implement** — HELP entries: `"library": "the deduplicated reference library (BibTeX, Zotero, feed, opt-in web)"`, `"library.sync": "read every configured source into the store"`, `"library.search": "search the library (semantic when embedded)"`, `"library.tag": "LLM-tag untagged references"`, `"library.embed": "embed references that have no vector"`, `"library.status": "counts per source, vectors, tags, citation edges"`. Parser, after `runs`:

```python
    sp = sub.add_parser("library", help=HELP["library"])
    add_db(sp)
    lib_sub = sp.add_subparsers(dest="library_command", required=True)
    lp = lib_sub.add_parser("sync", help=HELP["library.sync"])
    lp.add_argument("--sources", help="comma-separated subset: bibtex,zotero,feed,arxiv,crossref,s2")
    lp.add_argument("--limit", type=int, help="max rows per enricher and per embed pass")
    lp.set_defaults(func=cmd_library_sync)
    lp = lib_sub.add_parser("search", help=HELP["library.search"])
    lp.add_argument("query", nargs="?", default="")
    lp.add_argument("--author"); lp.add_argument("--year", type=int); lp.add_argument("--tag")
    lp.add_argument("--limit", type=int, default=10)
    lp.set_defaults(func=cmd_library_search)
    lp = lib_sub.add_parser("tag", help=HELP["library.tag"]); lp.add_argument("--limit", type=int); lp.set_defaults(func=cmd_library_tag)
    lp = lib_sub.add_parser("embed", help=HELP["library.embed"]); lp.add_argument("--limit", type=int); lp.set_defaults(func=cmd_library_embed)
    lp = lib_sub.add_parser("status", help=HELP["library.status"]); lp.set_defaults(func=cmd_library_status)
```

Commands (each `@_documented("library.<x>")`, using `open_db(args.db)` like `cmd_tag`):

```python
def _embedder_or_none():
    """An Embedder when the model server answers, else None (fielded search)."""
    from attestation.embed import Embedder
    from attestation.llm import base_url
    try:
        httpx.get(f"{base_url().rstrip('/')}/models", timeout=2.0)
    except httpx.HTTPError:
        return None
    return Embedder()


@_documented("library.sync")
def cmd_library_sync(args):
    from attestation import library, library_readers
    sources = args.sources.split(",") if args.sources else None
    with open_db(args.db) as conn:
        readers = library_readers.readers_from_env(conn, sources=sources)
        report = library.sync(conn, readers, embedder=_embedder_or_none(), limit=args.limit)
    for name, b in report.sources.items():
        line = f"{name}: +{b['added']} added, {b['merged']} merged, {b['unchanged']} unchanged"
        if b["enriched"]: line += f", {b['enriched']} enriched"
        if b["failed"]: line += f", {b['failed']} failed"
        print(line)
    print(f"embedded {report.embedded}, {report.unembedded} without a vector" + (f" ({report.embed_error})" if report.embed_error else ""))
    if report.conflicts: print(f"{report.conflicts} field conflict(s) recorded; first: {report.conflict_samples[:5]}")
    return 0


@_documented("library.search")
def cmd_library_search(args):
    from attestation import library
    with open_db(args.db) as conn:
        res = library.search(conn, args.query, embedder=_embedder_or_none(), author=args.author, year=args.year, tag=args.tag, limit=args.limit)
    for h in res.hits:
        sim = f" {h.similarity:.3f}" if h.similarity is not None else ""
        print(f"{h.year or '----'}{sim}  {h.title[:90]}  [{h.bib_key or h.identity}]")
    print(f"{res.n_matches} match(es); " + ("semantic" if res.semantic else res.caveat or "fielded"))
    return 0
```

`cmd_library_tag` mirrors `cmd_tag` with `run_reference_tagging`; `cmd_library_embed` calls `library.embed_missing(conn, Embedder(), args.limit)` and prints the triple; `cmd_library_status` prints `json.dumps(library.status(conn), indent=2)`.

Regenerate: `uv run python scripts/render_cli_reference.py`.

Guide: add `## The library` to `docs/guides/claims-and-citations.md` after the citations section: what sync reads, the identity rule in one sentence, the two flags, `attest library search` with the `semantic`/caveat contract, and that `cite.check` now resolves through the store. Keep the guide's opening paragraph untouched (the docs test requires the answer-first shape).

- [ ] **Step 4: Measurements (spec §8).** Against a scratch COPY of the live database:

```bash
cp ~/.hermes/skills/science-recommendations/data/hermes.db /tmp/lib-measure.db
ATTEST_DB=/tmp/lib-measure.db uv run attest library sync --sources feed
ATTEST_DB=/tmp/lib-measure.db uv run attest library status
```

Record in the spec's §8: items with an id / 9,401, per feed. Then `ATTEST_DB=/tmp/lib-measure.db uv run attest library embed --limit 200` and time it (the model is pinned), and note the rate. Delete `/tmp/lib-measure.db` afterwards. The S2 wall-time measurement waits for spec 2's example library.

- [ ] **Step 5: Run** `uv run pytest tests/test_cli.py tests/test_docs_site.py -q` → pass.
- [ ] **Step 6: Commit** `git add -A && git commit -m "attest library sync/search/tag/embed/status, the guide section, and the measured feed-id yield"`

---

### Task 11: Gate and PR

- [ ] **Step 1:** `uv run --frozen pre-commit run --all-files` → every hook Passed. Fix anything it raises (complexity ratchet is the likely one: split `search` and `upsert` into helpers rather than raising the ratchet).
- [ ] **Step 2:** `git push -u origin feat/library-store`
- [ ] **Step 3:** `gh pr create` with: the spec link, the identity rule, the two flags, the 46→47 change, the measured feed-id yield, and a test plan listing the offline-guarantee tests by name. End the body with the generated-with line.
- [ ] **Step 4:** Report back with the PR URL, the measurements, and anything the spec said that turned out wrong.

---

## Self-review

**Spec coverage.** §1 tables → Task 2; identity/merge → Task 1; backfill → Task 2; §3.1 readers → Tasks 4, 6; §3.2 config → Task 6; §3.3 sync/report → Task 5; §3.4 feed ids → Tasks 1, 3; §3.5 tagging → Task 8; §4 search → Task 7; §5 tools → Task 9, CLI → Task 10; §6 errors → Tasks 6, 7 (typed excepts, no new noqa), 9; §7 tests → each task; §8 measurements → Task 10 step 4 (S2 timing deferred to spec 2, stated).

**Placeholders.** None: every step has code. Two "read the existing helper and mirror it" instructions (Task 3's `_parsed`, Task 8's `_tag_with_retry`) point at named functions in named files.

**Type consistency.** `ReferenceRecord.fields()` used by `upsert` (Task 5) matches Task 4's definition; `readers_from_env(conn, sources=)` used by Tasks 9 and 10 matches Task 6; `library.search` returns `SearchResult` with `hits/semantic/caveat/n_matches` in Tasks 7, 9, 10; `embed_missing` returns `(done, missing, error)` in Tasks 5 (stub), 7, 10. Task 7's `noqa` was revised to a typed `except (httpx.HTTPError, OSError)` in Task 9's note — apply that in Task 7 when implementing so the BLE001 count stays at 7.
