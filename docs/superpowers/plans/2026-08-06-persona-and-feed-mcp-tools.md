# Persona + Feed MCP Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add twelve MCP tools for persona and feed curation (sixteen served in total), plus a `source` column on `clicks` recording whether feedback came from the web UI, an agent, or persona bootstrap.

**Architecture:** Three layers, bottom-up. A schema migration in `db.get_db` adds `clicks.source` to existing databases; a new `rank.record_click` helper becomes the single write path enforcing the enum; then MCP tools wrap `_impl` functions following the existing `mcp_server.py` pattern. Feed tools treat the database as the source of truth so they work without a checkout.

**Tech Stack:** Python 3.12+, SQLite (+ sqlite-vec), FastMCP (`mcp==1.28.1`), feedparser, numpy, pytest, `uv`.

## Global Constraints

- Commit ONLY the files each task's **Files** section lists. `feeds.toml`, `demo/`, and `docs/hermes-agent-plugin-research.md` are the user's uncommitted work — never stage them.
- Every commit message ends with the trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Gates before every commit: `uv run pytest -q`, `uv run ruff check .`, `uv run ty check` — all must pass.
- Ruff line-length is 100 (`[tool.ruff] line-length = 100`), lint set `["E", "F", "W", "I", "BLE"]`.
- Env vars are unprefixed and have NO legacy fallback: `RSS_DB`, `CHAT_MODEL`, `EMBED_MODEL`, `EMBED_DIMS`, `LLM_BASE_URL`, `LLM_API_KEY`.
- Valid `clicks.source` values are exactly `"ui"`, `"agent"`, `"bootstrap"`.
- Destructive tools take a required `confirm: bool`; with `confirm=False` they mutate nothing.
- Existing tests must pass untouched unless a task explicitly says otherwise. That is the evidence behavior was preserved.

---

### Task 1: `clicks.source` migration + single write path

**Files:**
- Modify: `src/hermes/db.py` (SCHEMA `clicks` table ~line 37; `get_db` after `executescript(SCHEMA)` ~line 125)
- Modify: `src/hermes/rank.py` (add `record_click`; `bootstrap_persona` INSERT at line 213)
- Modify: `src/hermes/mcp_server.py` (INSERT at line 85)
- Modify: `src/hermes/server.py` (INSERT at line 102)
- Test: `tests/test_db.py`, `tests/test_rank.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `db._migrate(conn) -> None` — idempotent; adds `clicks.source` when absent.
  - `rank.record_click(conn, user_id: int, item_id: int, useful: bool, source: str = "ui") -> None` — the single click write path. Raises `ValueError` on a source outside `{"ui","agent","bootstrap"}`. Uses `INSERT OR REPLACE`.
  - `rank.CLICK_SOURCES: tuple[str, ...]` = `("ui", "agent", "bootstrap")`.

- [ ] **Step 1: Write the failing migration test**

Add to `tests/test_db.py`:

```python
def test_migration_adds_source_to_existing_clicks_db(tmp_path):
    """A pre-existing DB with the old clicks schema gains source='ui' without data loss."""
    import sqlite3

    from hermes.db import get_db

    path = tmp_path / "old.db"
    old = sqlite3.connect(str(path))
    old.executescript(
        """
        CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, interests TEXT);
        CREATE TABLE items(id INTEGER PRIMARY KEY, title TEXT);
        CREATE TABLE clicks(
          id INTEGER PRIMARY KEY,
          user_id INTEGER NOT NULL,
          item_id INTEGER NOT NULL,
          useful INTEGER NOT NULL,
          clicked_at TEXT NOT NULL DEFAULT (datetime('now')),
          UNIQUE(user_id, item_id)
        );
        INSERT INTO users(id, name) VALUES (1, 'matt');
        INSERT INTO items(id, title) VALUES (1, 'a'), (2, 'b');
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
    from hermes.db import get_db

    path = tmp_path / "fresh.db"
    get_db(path).close()
    conn = get_db(path)

    cols = [r["name"] for r in conn.execute("PRAGMA table_info(clicks)")]
    assert cols.count("source") == 1
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db.py -k migration -v`
Expected: FAIL — `assert "source" in cols` (the column does not exist yet).

- [ ] **Step 3: Add `source` to SCHEMA and write the migration**

In `src/hermes/db.py`, change the `clicks` table in `SCHEMA` to include the column (this covers freshly-created databases):

```python
CREATE TABLE IF NOT EXISTS clicks(
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id),
  item_id INTEGER NOT NULL REFERENCES items(id),
  useful INTEGER NOT NULL,
  clicked_at TEXT NOT NULL DEFAULT (datetime('now')),
  source TEXT NOT NULL DEFAULT 'ui',
  UNIQUE(user_id, item_id)
);
```

Then add the migration function near `get_db`:

```python
def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent schema migrations for databases created before a column existed.

    SCHEMA is CREATE TABLE IF NOT EXISTS only, so it never alters a table that
    already exists -- live databases need this path. Pre-existing clicks
    backfill to 'ui': they came from the web UI and bootstrap_persona, which
    are not retroactively distinguishable, and 'ui' is right for the majority.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(clicks)")}
    if "source" not in cols:
        conn.execute("ALTER TABLE clicks ADD COLUMN source TEXT NOT NULL DEFAULT 'ui'")
        conn.commit()
```

Call it in `get_db` immediately after `conn.executescript(SCHEMA)`:

```python
    conn.executescript(SCHEMA)
    _migrate(conn)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py -k migration -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Write the failing `record_click` test**

Add to `tests/test_rank.py`:

```python
def test_record_click_writes_source_and_rejects_invalid(tmp_path):
    from hermes.db import get_db
    from hermes.rank import record_click

    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO users(id, name, interests) VALUES (1, 'u', 'x')")
    conn.execute(
        "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
        " VALUES (1, NULL, 't', 'u', 's', 'h')"
    )
    conn.commit()

    record_click(conn, 1, 1, True, source="agent")
    row = conn.execute("SELECT useful, source FROM clicks WHERE item_id = 1").fetchone()
    assert row["source"] == "agent"
    assert row["useful"] == 1

    # re-recording the same (user, item) replaces rather than duplicating
    record_click(conn, 1, 1, False, source="ui")
    rows = conn.execute("SELECT useful, source FROM clicks WHERE item_id = 1").fetchall()
    assert len(rows) == 1
    assert rows[0]["source"] == "ui"
    assert rows[0]["useful"] == 0

    with pytest.raises(ValueError, match="invalid click source"):
        record_click(conn, 1, 1, True, source="telepathy")
    conn.close()


def test_record_click_defaults_to_ui(tmp_path):
    from hermes.db import get_db
    from hermes.rank import record_click

    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO users(id, name, interests) VALUES (1, 'u', 'x')")
    conn.execute(
        "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
        " VALUES (1, NULL, 't', 'u', 's', 'h')"
    )
    conn.commit()

    record_click(conn, 1, 1, True)

    assert conn.execute("SELECT source FROM clicks").fetchone()["source"] == "ui"
    conn.close()
```

Ensure `import pytest` is present at the top of `tests/test_rank.py`.

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/test_rank.py -k record_click -v`
Expected: FAIL — `ImportError: cannot import name 'record_click'`.

- [ ] **Step 7: Implement `record_click`**

Add to `src/hermes/rank.py` (near `get_user`, above `blend_weight`):

```python
CLICK_SOURCES = ("ui", "agent", "bootstrap")


def record_click(conn, user_id: int, item_id: int, useful: bool, source: str = "ui") -> None:
    """The single click write path. `source` records provenance (see CLICK_SOURCES).

    SQLite cannot express a CHECK constraint added via ALTER TABLE, so the enum
    is enforced here rather than in the schema.
    """
    if source not in CLICK_SOURCES:
        raise ValueError(f"invalid click source: {source!r} (expected one of {CLICK_SOURCES})")
    conn.execute(
        "INSERT OR REPLACE INTO clicks(user_id, item_id, useful, source) VALUES (?, ?, ?, ?)",
        (user_id, item_id, int(useful), source),
    )
    conn.commit()
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_rank.py -k record_click -v`
Expected: PASS (2 tests).

- [ ] **Step 9: Route all three existing writers through `record_click`**

In `src/hermes/mcp_server.py`, replace the INSERT + `conn.commit()` (around line 85) with:

```python
        record_click(conn, row["id"], item_id, useful, source="agent")
```

and add `record_click` to the existing `from hermes.rank import ...` line.

In `src/hermes/server.py`, replace the INSERT + `conn.commit()` (around line 102) with:

```python
        record_click(conn, u["id"], item_id, bool(useful), source="ui")
```

and import `record_click` from `hermes.rank`.

In `src/hermes/rank.py`, `bootstrap_persona` currently uses `INSERT OR IGNORE`
inside an `executemany`. Replace that statement so it records provenance,
keeping `OR IGNORE` semantics (bootstrap must not overwrite real clicks):

```python
conn.executemany(
    "INSERT OR IGNORE INTO clicks(user_id, item_id, useful, source) VALUES (?, ?, ?, 'bootstrap')",
    rows_to_write,
)
```

Keep the existing variable name for the executemany argument — do not rename it.

- [ ] **Step 10: Add provenance tests for the three writers**

Add to `tests/test_mcp_server.py`:

```python
def test_record_feedback_records_agent_source(seeded_conn):
    from hermes.db import get_db, resolve_db_path

    out = mcp_server._record_feedback_impl("matt", 1, True)
    assert out["ok"] is True

    conn = get_db(resolve_db_path(None))
    assert (
        conn.execute("SELECT source FROM clicks WHERE item_id = 1").fetchone()["source"] == "agent"
    )
    conn.close()
```

If `seeded_conn` does not already create a `matt` user, use whatever user name that fixture seeds — read the fixture before writing this test and adjust the argument.

- [ ] **Step 11: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS, no regressions. `bootstrap_persona`, server, and MCP tests all still green.

- [ ] **Step 12: Verify the live database migrates cleanly**

```bash
cp ~/.hermes/skills/science-recommendations/data/hermes.db /tmp/hermes-backup.db
uv run python -c "
from hermes.db import get_db, resolve_db_path
c = get_db(resolve_db_path(None))
print('clicks:', c.execute('SELECT COUNT(*) n FROM clicks').fetchone()['n'])
print('by source:', [dict(r) for r in c.execute('SELECT source, COUNT(*) n FROM clicks GROUP BY source')])
"
```

Expected: `clicks: 68` and `by source: [{'source': 'ui', 'n': 68}]`. If the count is not 68, STOP — restore from `/tmp/hermes-backup.db` and report.

- [ ] **Step 13: Commit**

```bash
git add src/hermes/db.py src/hermes/rank.py src/hermes/mcp_server.py src/hermes/server.py tests/test_db.py tests/test_rank.py tests/test_mcp_server.py
git commit -m "$(cat <<'EOF'
feat: clicks.source provenance + first schema migration

Adds source (ui|agent|bootstrap) to clicks so agent-inferred feedback is
distinguishable from real UI clicks. Nothing consumes it yet -- ranking still
treats all clicks equally; this is instrumentation for later weighting.

SCHEMA is CREATE TABLE IF NOT EXISTS only, so live databases need a real
migration path: get_db now runs an idempotent _migrate that ALTERs the column
in when absent. Pre-existing clicks backfill to 'ui'.

The three duplicated INSERT sites (mcp_server, server, bootstrap_persona) now
route through one record_click helper, which enforces the enum -- SQLite
cannot express a CHECK constraint added via ALTER TABLE.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Feed tools

**Files:**
- Create: `src/hermes/feeds.py`
- Create: `src/hermes/feed_candidates.toml`
- Modify: `src/hermes/mcp_server.py` (add four `_impl` functions + four `@mcp.tool()` wrappers)
- Modify: `src/hermes/ingest.py` (`sync_feeds` docstring only — no behavior change)
- Test: `tests/test_feeds.py`, `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces (in `src/hermes/feeds.py`):
  - `add_feed(conn, url: str, title: str | None = None, parse=feedparser.parse) -> dict` — `{"ok": bool, "message": str, "feed_id": int | None}`.
  - `list_feeds(conn) -> list[dict]` — each `{"feed_id", "title", "url", "item_count", "last_fetched"}`.
  - `remove_feed(conn, feed_id: int) -> dict` — `{"ok", "message", "orphaned_items": int}`.
  - `preview_feed(url: str, limit: int = 5, parse=feedparser.parse) -> dict` — `{"ok", "message", "title", "entries": list[dict]}`.
  - `suggest_feeds(conn, user_id: int, limit: int = 5) -> list[dict]`.

- [ ] **Step 1: Write the failing feeds tests**

Create `tests/test_feeds.py`:

```python
import pytest

from hermes import feeds
from hermes.db import get_db


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
    conn.execute("INSERT INTO users(id, name, interests) VALUES (1, 'u', 'x')")
    cur = conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (?, 't', 'u', 's', 'h1')",
        (fid,),
    )
    item_id = cur.lastrowid
    conn.execute(
        "INSERT INTO clicks(user_id, item_id, useful, source) VALUES (1, ?, 1, 'ui')",
        (item_id,),
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_feeds.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hermes.feeds'`.

- [ ] **Step 3: Create the candidate list**

Create `src/hermes/feed_candidates.toml`:

```toml
# Curated feed suggestions for suggest_feeds. Scored against a persona's
# liked tags -- never web-searched, never model-invented.

[[candidates]]
url = "https://rss.arxiv.org/rss/cs.CL"
title = "arXiv cs.CL (computation and language)"
tags = ["nlp", "language-models", "machine-learning"]

[[candidates]]
url = "https://rss.arxiv.org/rss/cs.CV"
title = "arXiv cs.CV (computer vision)"
tags = ["computer-vision", "machine-learning", "deep-learning"]

[[candidates]]
url = "https://rss.arxiv.org/rss/q-bio.BM"
title = "arXiv q-bio.BM (biomolecules)"
tags = ["structural-biology", "proteins", "biochemistry"]

[[candidates]]
url = "https://rss.arxiv.org/rss/cond-mat.mtrl-sci"
title = "arXiv cond-mat.mtrl-sci (materials science)"
tags = ["materials", "crystallography", "condensed-matter"]

[[candidates]]
url = "https://www.nature.com/nature.rss"
title = "Nature"
tags = ["science", "research", "general-science"]

[[candidates]]
url = "https://phys.org/rss-feed/"
title = "Phys.org"
tags = ["physics", "science", "general-science"]

[[candidates]]
url = "https://openai.com/blog/rss.xml"
title = "OpenAI blog"
tags = ["machine-learning", "language-models", "industry"]

[[candidates]]
url = "https://www.deepmind.com/blog/rss.xml"
title = "Google DeepMind blog"
tags = ["machine-learning", "reinforcement-learning", "industry"]
```

- [ ] **Step 4: Implement `src/hermes/feeds.py`**

```python
"""Feed curation: register, list, remove, preview, and suggest feeds.

The database is the source of truth for the feed set. `feeds.toml` seeds a
fresh database on first ingest (ingest.sync_feeds uses INSERT OR IGNORE, so
it is a no-op afterwards); these functions are the supported way to change
which feeds are tracked, and they work with no checkout present.

add_feed is register-only by design: it validates the URL parses and inserts
the row, leaving the fetch to the next `hermes ingest`. Ingesting inline
would mean network I/O plus one embedding per item -- minutes for a busy
feed, inside a tool call an agent may time out on.
"""

import sqlite3
import tomllib
from pathlib import Path

import feedparser

CANDIDATES_PATH = Path(__file__).resolve().parent / "feed_candidates.toml"


def _looks_like_feed(parsed) -> bool:
    """A usable feed has entries, or at least a title we can show."""
    if getattr(parsed, "entries", None):
        return True
    feed_meta = getattr(parsed, "feed", None) or {}
    return bool(feed_meta.get("title")) and not getattr(parsed, "bozo", 0)


def add_feed(
    conn: sqlite3.Connection,
    url: str,
    title: str | None = None,
    parse=feedparser.parse,
) -> dict:
    """Register a feed after checking it parses. Does NOT ingest its items."""
    existing = conn.execute("SELECT id FROM feeds WHERE url = ?", (url,)).fetchone()
    if existing is not None:
        return {
            "ok": True,
            "feed_id": existing["id"],
            "message": f"already subscribed to {url}",
        }

    parsed = parse(url)
    if not _looks_like_feed(parsed):
        return {
            "ok": False,
            "feed_id": None,
            "message": f"{url} did not parse as an RSS/Atom feed; nothing was added",
        }

    resolved_title = title or (getattr(parsed, "feed", None) or {}).get("title") or url
    cur = conn.execute("INSERT INTO feeds(url, title) VALUES (?, ?)", (url, resolved_title))
    conn.commit()
    return {
        "ok": True,
        "feed_id": cur.lastrowid,
        "message": (
            f"subscribed to {resolved_title!r}. Items appear after the next ingest "
            "(run `hermes ingest`, or wait for the hourly refresh)."
        ),
    }


def list_feeds(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT f.id, f.title, f.url, f.last_fetched, COUNT(i.id) AS item_count"
        " FROM feeds f LEFT JOIN items i ON i.feed_id = f.id"
        " GROUP BY f.id ORDER BY f.title"
    ).fetchall()
    return [
        {
            "feed_id": r["id"],
            "title": r["title"],
            "url": r["url"],
            "item_count": r["item_count"],
            "last_fetched": r["last_fetched"],
        }
        for r in rows
    ]


def remove_feed(conn: sqlite3.Connection, feed_id: int) -> dict:
    """Unsubscribe. Items are ORPHANED, never deleted -- their clicks trained
    the ranker, and cascading would destroy that feedback."""
    row = conn.execute("SELECT title FROM feeds WHERE id = ?", (feed_id,)).fetchone()
    if row is None:
        return {"ok": False, "message": f"unknown feed_id: {feed_id}", "orphaned_items": 0}

    orphaned = conn.execute(
        "SELECT COUNT(*) AS n FROM items WHERE feed_id = ?", (feed_id,)
    ).fetchone()["n"]
    conn.execute("DELETE FROM feeds WHERE id = ?", (feed_id,))
    conn.commit()
    return {
        "ok": True,
        "message": (
            f"unsubscribed from {row['title']!r}; {orphaned} existing item(s) and all "
            "feedback on them were kept"
        ),
        "orphaned_items": orphaned,
    }


def preview_feed(url: str, limit: int = 5, parse=feedparser.parse) -> dict:
    """Fetch and show recent entries WITHOUT subscribing."""
    parsed = parse(url)
    if not _looks_like_feed(parsed):
        return {
            "ok": False,
            "message": f"{url} did not parse as an RSS/Atom feed",
            "title": None,
            "entries": [],
        }
    feed_meta = getattr(parsed, "feed", None) or {}
    entries = [
        {"title": e.get("title"), "url": e.get("link")} for e in list(parsed.entries)[:limit]
    ]
    return {
        "ok": True,
        "message": f"{len(entries)} recent entrie(s); not subscribed",
        "title": feed_meta.get("title") or url,
        "entries": entries,
    }


def _load_candidates() -> list[dict]:
    return tomllib.loads(CANDIDATES_PATH.read_text()).get("candidates", [])


def suggest_feeds(conn: sqlite3.Connection, user_id: int, limit: int = 5) -> list[dict]:
    """Score the curated candidate list against tags this user marked useful."""
    liked = {
        r["tag"]
        for r in conn.execute(
            "SELECT DISTINCT t.tag FROM clicks c JOIN item_tags t ON t.item_id = c.item_id"
            " WHERE c.user_id = ? AND c.useful = 1",
            (user_id,),
        )
    }
    subscribed = {r["url"] for r in conn.execute("SELECT url FROM feeds")}

    scored = []
    for cand in _load_candidates():
        if cand["url"] in subscribed:
            continue
        overlap = liked & set(cand.get("tags", []))
        scored.append(
            {
                "url": cand["url"],
                "title": cand["title"],
                "score": len(overlap),
                "matched_tags": sorted(overlap),
            }
        )
    scored.sort(key=lambda c: (-c["score"], c["title"]))
    return scored[:limit]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_feeds.py -v`
Expected: PASS (7 tests).

- [ ] **Step 6: Update the `sync_feeds` docstring**

In `src/hermes/ingest.py`, add a docstring to `sync_feeds` (behavior unchanged):

```python
def sync_feeds(conn: sqlite3.Connection, feeds_path: str | Path) -> None:
    """Seed the feeds table from feeds.toml.

    INSERT OR IGNORE, so this is a no-op for feeds already present: the
    database -- not the TOML file -- is the source of truth once seeded.
    Use hermes.feeds.add_feed / remove_feed to change the feed set.
    """
```

- [ ] **Step 7: Add the four MCP tools**

In `src/hermes/mcp_server.py`, add `_impl` functions following the existing pattern (open a connection via `resolve_db_path(None)`, `try/except/finally`, return a dict):

```python
def _add_feed_impl(url: str, title: str | None = None) -> dict:
    from hermes import feeds as feeds_mod

    conn = get_db(resolve_db_path(None))
    try:
        return feeds_mod.add_feed(conn, url, title)
    except Exception:
        log.exception("add_feed failed for url=%s", url)
        return {"ok": False, "feed_id": None, "message": "internal error adding feed"}
    finally:
        conn.close()


def _list_feeds_impl() -> dict:
    from hermes import feeds as feeds_mod

    conn = get_db(resolve_db_path(None))
    try:
        return {"feeds": feeds_mod.list_feeds(conn)}
    finally:
        conn.close()


def _remove_feed_impl(feed_id: int, confirm: bool = False) -> dict:
    from hermes import feeds as feeds_mod

    if not confirm:
        return {
            "ok": False,
            "message": (
                f"refusing to remove feed {feed_id} without confirm=true. This "
                "unsubscribes the feed; its existing items and your feedback on "
                "them are kept."
            ),
            "orphaned_items": 0,
        }
    conn = get_db(resolve_db_path(None))
    try:
        return feeds_mod.remove_feed(conn, feed_id)
    finally:
        conn.close()


def _preview_feed_impl(url: str, limit: int = 5) -> dict:
    from hermes import feeds as feeds_mod

    try:
        return feeds_mod.preview_feed(url, limit=min(limit, MAX_LIST_LIMIT))
    except Exception:
        log.exception("preview_feed failed for url=%s", url)
        return {
            "ok": False,
            "message": "internal error previewing feed",
            "title": None,
            "entries": [],
        }


def _suggest_feeds_impl(user: str, limit: int = 5) -> dict:
    from hermes import feeds as feeds_mod

    conn = get_db(resolve_db_path(None))
    try:
        row = get_user(conn, user)
        if row is None:
            return {"ok": False, "message": _unknown_user_message(conn, user), "suggestions": []}
        return {
            "ok": True,
            "message": "scored against tags you marked useful",
            "suggestions": feeds_mod.suggest_feeds(conn, row["id"], limit=limit),
        }
    finally:
        conn.close()
```

Then the tool wrappers:

```python
@mcp.tool()
def add_feed(url: str, title: str | None = None) -> dict:
    """Subscribe to an RSS/Atom feed.

    Validates that the URL parses as a feed, then registers it. Does NOT fetch
    its articles: items appear after the next ingest (hourly cron, or
    `hermes ingest`). Use preview_feed first to check a feed's content.
    """
    return _add_feed_impl(url, title)


@mcp.tool()
def list_feeds() -> dict:
    """List subscribed feeds with item counts and when each was last fetched."""
    return _list_feeds_impl()


@mcp.tool()
def remove_feed(feed_id: int, confirm: bool = False) -> dict:
    """Unsubscribe from a feed. Requires confirm=true.

    Existing items and all feedback on them are KEPT -- only the subscription
    is removed, so no ranking history is lost.
    """
    return _remove_feed_impl(feed_id, confirm)


@mcp.tool()
def preview_feed(url: str, limit: int = 5) -> dict:
    """Show recent entries from a feed WITHOUT subscribing to it."""
    return _preview_feed_impl(url, limit)


@mcp.tool()
def suggest_feeds(user: str, limit: int = 5) -> dict:
    """Suggest feeds from a curated list, scored against tags this user liked."""
    return _suggest_feeds_impl(user, limit)
```

- [ ] **Step 8: Write the MCP-level guardrail test**

Add to `tests/test_mcp_server.py`:

```python
def test_remove_feed_without_confirm_mutates_nothing(seeded_conn):
    from hermes.db import get_db, resolve_db_path

    conn = get_db(resolve_db_path(None))
    conn.execute("INSERT INTO feeds(id, url, title) VALUES (77, 'http://x/rss', 'X')")
    conn.commit()
    conn.close()

    out = mcp_server._remove_feed_impl(77, confirm=False)

    assert out["ok"] is False
    assert "confirm" in out["message"]
    conn = get_db(resolve_db_path(None))
    assert conn.execute("SELECT COUNT(*) n FROM feeds WHERE id = 77").fetchone()["n"] == 1
    conn.close()
```

- [ ] **Step 9: Run the full suite**

Run: `uv run pytest -q && uv run ruff check . && uv run ty check`
Expected: all PASS.

- [ ] **Step 10: Verify the tools are actually served**

```bash
uv run python -c "
import asyncio
from hermes.mcp_server import mcp
tools = asyncio.run(mcp.list_tools())
print(sorted(t.name for t in tools))
"
```

Expected: includes `add_feed`, `list_feeds`, `preview_feed`, `remove_feed`, `suggest_feeds` alongside the original four.

- [ ] **Step 11: Commit**

```bash
git add src/hermes/feeds.py src/hermes/feed_candidates.toml src/hermes/mcp_server.py src/hermes/ingest.py tests/test_feeds.py tests/test_mcp_server.py
git commit -m "$(cat <<'EOF'
feat: feed curation MCP tools (add/list/remove/preview/suggest)

The database is now the source of truth for the feed set; feeds.toml seeds a
fresh database and is a no-op afterwards (sync_feeds already used INSERT OR
IGNORE, so only its docstring changed). This is what lets feed tools work
with no checkout present, as in uvx installs.

add_feed is register-only: it validates the URL parses, then leaves fetching
to the next ingest. Ingesting inline would mean network I/O plus an embedding
per item inside a tool call the agent may time out on.

remove_feed orphans items instead of cascading -- their clicks trained the
ranker, and deleting them would destroy that feedback. suggest_feeds scores a
curated in-repo candidate list, never a web search.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Persona tools + `search_feed`

**Files:**
- Modify: `src/hermes/rank.py` (add `create_user`; add keyword-only params to `_candidate_items` ~line 96 and `rank_items` ~line 137)
- Modify: `src/hermes/mcp_server.py` (six `_impl` functions + six `@mcp.tool()` wrappers)
- Test: `tests/test_rank.py`, `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `rank.record_click`, `rank.CLICK_SOURCES` (Task 1); `features._key_stats` (existing).
- Produces:
  - `rank.create_user(conn, name: str, interests: str) -> int` — returns the new user id; raises `ValueError` if the name already exists.
  - `rank._candidate_items(conn, user_id, since_days, *, exclude_clicked: bool = True)` — `since_days: int | None`, where `None` means no window.
  - `rank.rank_items(conn, embedder, user_id, since_days=14, *, exclude_clicked: bool = True)`.

- [ ] **Step 1: Write the failing `create_user` and candidate-filter tests**

Add to `tests/test_rank.py`:

```python
def test_create_user_returns_id_and_rejects_duplicates(tmp_path):
    from hermes.db import get_db
    from hermes.rank import create_user, get_user

    conn = get_db(tmp_path / "t.db")

    uid = create_user(conn, "newbie", "protein folding, cryo-EM")

    assert get_user(conn, "newbie")["id"] == uid
    assert get_user(conn, "newbie")["interests"] == "protein folding, cryo-EM"
    with pytest.raises(ValueError, match="already exists"):
        create_user(conn, "newbie", "something else")
    conn.close()


def test_candidate_items_can_include_clicked_and_drop_window(tmp_path, fake_embedder):
    """search_feed needs both: default behavior must be unchanged."""
    from hermes.db import get_db
    from hermes.rank import _candidate_items, record_click

    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO users(id, name, interests) VALUES (1, 'u', 'x')")
    # one recent item, one far outside the default 14-day window
    for i, published in ((1, "datetime('now')"), (2, "datetime('now', '-400 days')")):
        conn.execute(
            f"INSERT INTO items(id, feed_id, title, url, summary, published, content_hash)"
            f" VALUES ({i}, NULL, 't{i}', 'u', 's', {published}, 'h{i}')"
        )
        vec = fake_embedder.embed_document(f"t{i}", "s")
        conn.execute("INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)", (i, vec.tobytes()))
    conn.commit()
    record_click(conn, 1, 1, True)

    default_ids = {r["id"] for r in _candidate_items(conn, 1, 14)}
    assert default_ids == set(), "clicked item and out-of-window item both excluded by default"

    search_ids = {r["id"] for r in _candidate_items(conn, 1, None, exclude_clicked=False)}
    assert search_ids == {1, 2}
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rank.py -k "create_user or candidate_items_can_include" -v`
Expected: FAIL — `ImportError: cannot import name 'create_user'`.

- [ ] **Step 3: Implement `create_user` and the candidate filters**

Add to `src/hermes/rank.py` next to `get_user`:

```python
def create_user(conn, name: str, interests: str) -> int:
    """Insert a persona. Ranking starts from the interests embedding alone."""
    if get_user(conn, name) is not None:
        raise ValueError(f"user already exists: {name!r}")
    cur = conn.execute("INSERT INTO users(name, interests) VALUES (?, ?)", (name, interests))
    conn.commit()
    return cur.lastrowid
```

Replace `_candidate_items` with:

```python
def _candidate_items(conn, user_id: int, since_days: int | None, *, exclude_clicked: bool = True):
    """Rankable items. Defaults reproduce feed behavior: recent and unclicked.

    search_feed passes since_days=None / exclude_clicked=False, since finding
    an older or already-rated item is a legitimate search result.
    """
    sql = (
        "SELECT i.id, i.title, i.url, f.title AS source, v.embedding"
        " FROM items i JOIN item_vectors v ON v.rowid = i.id"
        " LEFT JOIN feeds f ON f.id = i.feed_id"
        " WHERE 1=1"
    )
    params: list = []
    if since_days is not None:
        sql += " AND i.published >= datetime('now', ?)"
        params.append(f"-{since_days} days")
    if exclude_clicked:
        sql += " AND i.id NOT IN (SELECT item_id FROM clicks WHERE user_id = ?)"
        params.append(user_id)
    return conn.execute(sql, params).fetchall()
```

Update `rank_items`'s signature and its single call site:

```python
def rank_items(
    conn, embedder, user_id: int, since_days: int | None = 14, *, exclude_clicked: bool = True
) -> list[RankedItem]:
    rows = _candidate_items(conn, user_id, since_days, exclude_clicked=exclude_clicked)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rank.py -v`
Expected: PASS — including every pre-existing rank test, unchanged. That is the proof the defaults preserved behavior.

- [ ] **Step 5: Add the persona `_impl` functions**

In `src/hermes/mcp_server.py`:

```python
def _create_persona_impl(name: str, interests: str) -> dict:
    from hermes.rank import create_user

    conn = get_db(resolve_db_path(None))
    try:
        uid = create_user(conn, name, interests)
        return {
            "ok": True,
            "user_id": uid,
            "message": (
                f"created persona {name!r}. Ranking starts from its interests text; "
                "record_feedback calls will personalize it from the first click."
            ),
        }
    except ValueError as exc:
        return {"ok": False, "user_id": None, "message": str(exc)}
    finally:
        conn.close()


def _update_persona_impl(name: str, interests: str) -> dict:
    conn = get_db(resolve_db_path(None))
    try:
        row = get_user(conn, name)
        if row is None:
            return {"ok": False, "message": _unknown_user_message(conn, name)}
        conn.execute("UPDATE users SET interests = ? WHERE id = ?", (interests, row["id"]))
        conn.commit()
        return {
            "ok": True,
            "message": f"updated interests for {name!r}; ranking re-embeds on next use",
        }
    finally:
        conn.close()


def _propose_interests_impl(limit: int = 12) -> dict:
    conn = get_db(resolve_db_path(None))
    try:
        tags = [
            r["tag"]
            for r in conn.execute(
                "SELECT tag FROM item_tags GROUP BY tag ORDER BY COUNT(*) DESC, tag LIMIT ?",
                (limit,),
            )
        ]
        return {
            "ok": True,
            "prevalent_tags": tags,
            "message": (
                "most common tags in the current feed; combine the relevant ones into "
                "an interests string and pass it to create_persona"
            ),
        }
    finally:
        conn.close()


def _profile_status_impl(user: str) -> dict:
    from hermes.features import _key_stats, _score
    from hermes.rank import blend_weight

    conn = get_db(resolve_db_path(None))
    try:
        row = get_user(conn, user)
        if row is None:
            return {"ok": False, "message": _unknown_user_message(conn, user)}
        n_clicks = conn.execute(
            "SELECT COUNT(*) c FROM clicks WHERE user_id = ?", (row["id"],)
        ).fetchone()["c"]
        stats = _key_stats(conn, row["id"])
        scored = sorted(((k, _score(stats, k)) for k in stats), key=lambda kv: kv[1], reverse=True)
        by_source = {
            r["source"]: r["n"]
            for r in conn.execute(
                "SELECT source, COUNT(*) AS n FROM clicks WHERE user_id = ? GROUP BY source",
                (row["id"],),
            )
        }
        return {
            "ok": True,
            "user": user,
            "interests": row["interests"],
            "clicks": n_clicks,
            "clicks_by_source": by_source,
            "blend_weight": round(blend_weight(n_clicks), 3),
            "top_liked": [k for k, v in scored[:5] if v > 0.5],
            "top_disliked": [k for k, v in reversed(scored[-5:]) if v < 0.5],
            "message": (
                f"{n_clicks} click(s); ranking is {round(blend_weight(n_clicks) * 100)}% "
                "driven by observed behavior and the rest by the interests text"
            ),
        }
    finally:
        conn.close()


def _search_feed_impl(
    user: str,
    query: str,
    tag: str | None = None,
    content_type: str | None = None,
    limit: int = 10,
) -> dict:
    from hermes.rank import rank_items

    conn = get_db(resolve_db_path(None))
    try:
        row = get_user(conn, user)
        if row is None:
            return {"ok": False, "message": _unknown_user_message(conn, user), "items": []}

        ranked = rank_items(
            conn, _get_embedder(), row["id"], since_days=None, exclude_clicked=False
        )
        clicked = {
            r["item_id"]
            for r in conn.execute("SELECT item_id FROM clicks WHERE user_id = ?", (row["id"],))
        }
        needle = query.lower()
        matches = []
        for item in ranked:
            if needle and needle not in (item.title or "").lower():
                continue
            if tag and tag not in (item.tags or []):
                continue
            if content_type and item.content_type != content_type:
                continue
            matches.append(
                {
                    "item_id": item.item_id,
                    "title": item.title,
                    "url": item.url,
                    "source": item.source,
                    "tags": item.tags,
                    "content_type": item.content_type,
                    "already_rated": item.item_id in clicked,
                }
            )
            if len(matches) >= min(limit, MAX_LIST_LIMIT):
                break
        return {"ok": True, "message": f"{len(matches)} match(es), best first", "items": matches}
    finally:
        conn.close()


def _delete_persona_impl(name: str, confirm: bool = False) -> dict:
    conn = get_db(resolve_db_path(None))
    try:
        row = get_user(conn, name)
        if row is None:
            return {"ok": False, "message": _unknown_user_message(conn, name)}
        n = conn.execute(
            "SELECT COUNT(*) c FROM clicks WHERE user_id = ?", (row["id"],)
        ).fetchone()["c"]
        if not confirm:
            return {
                "ok": False,
                "message": (
                    f"refusing to delete {name!r} without confirm=true. This would "
                    f"permanently remove the persona and its {n} click(s) of training data."
                ),
            }
        conn.execute("DELETE FROM clicks WHERE user_id = ?", (row["id"],))
        conn.execute("DELETE FROM users WHERE id = ?", (row["id"],))
        conn.commit()
        return {"ok": True, "message": f"deleted persona {name!r} and its {n} click(s)"}
    finally:
        conn.close()


def _reset_feedback_impl(name: str, confirm: bool = False) -> dict:
    conn = get_db(resolve_db_path(None))
    try:
        row = get_user(conn, name)
        if row is None:
            return {"ok": False, "message": _unknown_user_message(conn, name)}
        n = conn.execute(
            "SELECT COUNT(*) c FROM clicks WHERE user_id = ?", (row["id"],)
        ).fetchone()["c"]
        if not confirm:
            return {
                "ok": False,
                "message": (
                    f"refusing to reset {name!r} without confirm=true. This would erase "
                    f"{n} click(s); the persona and its interests text would be kept."
                ),
            }
        conn.execute("DELETE FROM clicks WHERE user_id = ?", (row["id"],))
        conn.commit()
        return {"ok": True, "message": f"cleared {n} click(s) for {name!r}"}
    finally:
        conn.close()
```

- [ ] **Step 6: Add the six tool wrappers**

```python
@mcp.tool()
def create_persona(name: str, interests: str) -> dict:
    """Create a reader persona from a name and an interests description.

    Ranking starts from the interests text and personalizes from the first
    record_feedback call. Use propose_interests first if you want suggestions
    drawn from what is actually in the feed.
    """
    return _create_persona_impl(name, interests)


@mcp.tool()
def update_persona(name: str, interests: str) -> dict:
    """Replace a persona's interests text; re-steers ranking immediately."""
    return _update_persona_impl(name, interests)


@mcp.tool()
def propose_interests(limit: int = 12) -> dict:
    """List the most common tags in the feed, to help write an interests string."""
    return _propose_interests_impl(limit)


@mcp.tool()
def profile_status(user: str) -> dict:
    """Show how well-trained a persona is: click count, how much ranking is
    driven by behavior vs the written interests, and top liked/disliked tags."""
    return _profile_status_impl(user)


@mcp.tool()
def search_feed(
    user: str,
    query: str,
    tag: str | None = None,
    content_type: str | None = None,
    limit: int = 10,
) -> dict:
    """Search items by keyword (and optional tag/content_type), ranked for this user.

    Unlike list_feed this searches the whole archive and includes items already
    rated, flagging each with already_rated.
    """
    return _search_feed_impl(user, query, tag, content_type, limit)


@mcp.tool()
def delete_persona(name: str, confirm: bool = False) -> dict:
    """Delete a persona AND all its feedback. Requires confirm=true. Irreversible."""
    return _delete_persona_impl(name, confirm)


@mcp.tool()
def reset_feedback(name: str, confirm: bool = False) -> dict:
    """Erase a persona's clicks but keep the persona. Requires confirm=true."""
    return _reset_feedback_impl(name, confirm)
```

- [ ] **Step 7: Write the MCP persona tests**

Add to `tests/test_mcp_server.py`:

```python
def test_create_and_update_persona(seeded_conn):
    created = mcp_server._create_persona_impl("chemist", "catalysis, spectroscopy")
    assert created["ok"] is True

    dup = mcp_server._create_persona_impl("chemist", "anything")
    assert dup["ok"] is False

    updated = mcp_server._update_persona_impl("chemist", "electrochemistry")
    assert updated["ok"] is True

    status = mcp_server._profile_status_impl("chemist")
    assert status["interests"] == "electrochemistry"
    assert status["clicks"] == 0
    assert status["blend_weight"] == 0.0


def test_destructive_tools_refuse_without_confirm(seeded_conn):
    from hermes.db import get_db, resolve_db_path

    mcp_server._create_persona_impl("victim", "x")
    mcp_server._record_feedback_impl("victim", 1, True)

    assert mcp_server._delete_persona_impl("victim", confirm=False)["ok"] is False
    assert mcp_server._reset_feedback_impl("victim", confirm=False)["ok"] is False

    conn = get_db(resolve_db_path(None))
    assert conn.execute("SELECT COUNT(*) n FROM users WHERE name = 'victim'").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) n FROM clicks").fetchone()["n"] == 1
    conn.close()

    assert mcp_server._reset_feedback_impl("victim", confirm=True)["ok"] is True
    conn = get_db(resolve_db_path(None))
    assert conn.execute("SELECT COUNT(*) n FROM clicks").fetchone()["n"] == 0
    # persona itself survives a reset
    assert conn.execute("SELECT COUNT(*) n FROM users WHERE name = 'victim'").fetchone()["n"] == 1
    conn.close()


def test_search_feed_finds_already_rated_items(seeded_conn):
    mcp_server._create_persona_impl("searcher", "items")
    mcp_server._record_feedback_impl("searcher", 1, True)

    out = mcp_server._search_feed_impl("searcher", "item")

    assert out["ok"] is True
    assert out["items"], "search must reach items list_feed would exclude"
    assert any(i["already_rated"] for i in out["items"])
```

- [ ] **Step 8: Run the full suite and all gates**

Run: `uv run pytest -q && uv run ruff check . && uv run ty check`
Expected: all PASS.

- [ ] **Step 9: Verify all sixteen tools are served**

```bash
uv run python -c "
import asyncio
from hermes.mcp_server import mcp
names = sorted(t.name for t in asyncio.run(mcp.list_tools()))
print(len(names), names)
"
```

Expected: **16** tools — the original four (`list_users`, `list_feed`, `record_feedback`, `explain_item`) plus the twelve added in Tasks 2 and 3: `add_feed`, `list_feeds`, `preview_feed`, `remove_feed`, `suggest_feeds`, `create_persona`, `update_persona`, `propose_interests`, `profile_status`, `search_feed`, `delete_persona`, `reset_feedback`.

If the count differs, report the actual list rather than adjusting expectations to match it.

- [ ] **Step 10: Commit**

```bash
git add src/hermes/rank.py src/hermes/mcp_server.py tests/test_rank.py tests/test_mcp_server.py
git commit -m "$(cat <<'EOF'
feat: persona MCP tools + search_feed

create/update/delete persona, reset_feedback, propose_interests,
profile_status, and search_feed. rank.create_user is new -- the codebase had
get_user but no way to make one outside hand-editing the database.

search_feed required loosening rank._candidate_items, which hardcoded both a
recency window and NOT IN (clicks): as written, search could reach neither
older nor already-rated items, which is most of the archive. Both are now
keyword-only params defaulting to today's behavior, so list_feed is unchanged
and the existing rank tests pass untouched.

delete_persona and reset_feedback require confirm=true and name exactly what
would be lost when refused. profile_status surfaces clicks_by_source, making
the new provenance column visible.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Documentation

**Files:**
- Modify: `README.md` (MCP tool table ~line 184; feeds.toml references)
- Modify: `src/hermes/skills/science-recommendations/SKILL.md`

**Interfaces:**
- Consumes: the tool names and signatures from Tasks 2 and 3.
- Produces: nothing code-level.

- [ ] **Step 1: Update the README tool table**

Replace the four-row table under "The server (`hermes-mcp`...) exposes four tools:" with all sixteen, grouped, and change the lead-in to "exposes sixteen tools:".

```markdown
| Tool | What it does | Speed |
|---|---|---|
| `list_users()` | List reader personas + interest profiles | instant |
| `list_feed(user, limit)` | Ranked unread items, best first (capped at 50) | fast |
| `search_feed(user, query, tag, content_type, limit)` | Search the whole archive, ranked for this user; includes already-rated items | fast |
| `record_feedback(user, item_id, useful)` | Record a ✓/✗ click; retrains ranking | fast |
| `explain_item(user, item_id)` | One-sentence "why did this rank here" | **slow** first call (local LLM), cached after |
| `create_persona(name, interests)` | Create a reader persona | instant |
| `update_persona(name, interests)` | Replace a persona's interests text | instant |
| `propose_interests(limit)` | Most common tags, to help write an interests string | instant |
| `profile_status(user)` | Click count, behavior-vs-text blend weight, top liked/disliked tags | instant |
| `delete_persona(name, confirm)` | Delete a persona and its feedback (needs `confirm=true`) | instant |
| `reset_feedback(name, confirm)` | Clear a persona's clicks, keep the persona (needs `confirm=true`) | instant |
| `add_feed(url, title)` | Subscribe to a feed (register-only; items arrive at the next ingest) | fast |
| `list_feeds()` | Subscribed feeds with item counts and last-fetched times | instant |
| `preview_feed(url, limit)` | Show a feed's recent entries without subscribing | fast |
| `remove_feed(feed_id, confirm)` | Unsubscribe; keeps existing items and feedback (needs `confirm=true`) | instant |
| `suggest_feeds(user, limit)` | Suggest feeds from a curated list, scored against liked tags | instant |
```

- [ ] **Step 2: Correct the feeds.toml description**

Find the line reading `Edit `feeds.toml` to change which feeds are tracked, then re-run` and replace that paragraph with:

```markdown
`feeds.toml` seeds the feed list when the database is first created. After
that the **database is the source of truth**: use the `add_feed` /
`remove_feed` MCP tools (or edit the database directly) to change which feeds
are tracked, then run `uv run hermes ingest` to fetch from any newly added
feed. Editing `feeds.toml` after the first ingest has no effect.
```

- [ ] **Step 3: Add a feedback-provenance note to the README**

Directly after the tool table, add:

```markdown
Every recorded click stores its provenance — `ui` for the web UI, `agent` for
MCP `record_feedback` calls, `bootstrap` for synthetic persona seeding.
Ranking currently treats all three identically; `profile_status` breaks the
counts down by source so you can see how much of a persona's training came
from the agent rather than from you.
```

- [ ] **Step 4: Update SKILL.md**

In `src/hermes/skills/science-recommendations/SKILL.md`, find the section stating that changing feeds means editing `feeds.toml` in the project directory (around line 30) and replace it with a note that `add_feed` / `remove_feed` are the supported path and work without a checkout.

- [ ] **Step 5: Verify the docs match reality**

```bash
uv run python -c "
import asyncio, re
from pathlib import Path
from hermes.mcp_server import mcp
served = {t.name for t in asyncio.run(mcp.list_tools())}
documented = set(re.findall(r'\| \`(\w+)\(', Path('README.md').read_text()))
print('served not documented:', sorted(served - documented))
print('documented not served:', sorted(documented - served))
"
```

Expected: both lists empty. If not, fix the README — the served tools are the truth.

- [ ] **Step 6: Commit**

```bash
git add README.md src/hermes/skills/science-recommendations/SKILL.md
git commit -m "$(cat <<'EOF'
docs: document the sixteen MCP tools and the feeds.toml inversion

The README still told users to edit feeds.toml to change tracked feeds; that
stopped being true when the database became the source of truth. Also
documents click provenance and adds a check that every served tool is
documented and vice versa.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Post-plan manual verification

After all tasks land, confirm the feature works against the real system:

1. `uv run hermes install --check` still exits cleanly (the migration must not break the installer's doctor mode).
2. Live database: `uv run python -c "from hermes.db import get_db, resolve_db_path; c=get_db(resolve_db_path(None)); print([dict(r) for r in c.execute('SELECT source, COUNT(*) n FROM clicks GROUP BY source')])"` — expect 68 rows of `ui` plus any new activity.
3. `hermes mcp test hermes-rss` reports 16 tools discovered.
4. Round-trip through the agent: `hermes -z "Use profile_status on matt"` — note that local models currently struggle to emit tool calls, so a text-only reply is a known model limitation, not a regression in this work.
