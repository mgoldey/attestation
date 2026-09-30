# Reading Desk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A private AgentMarkit page that shows a researcher today's ranked papers with Useful / Not my area buttons, whose verdicts come back into attestation's `clicks`; plus saved research briefs.

**Architecture:** attestation gains `desk.py`: it reads the page's `state.sqlite` read-only, imports verdicts through `rank.record_click(source='ui')` before any ranking a person sees, and renders one self-contained HTML page. `attest desk refresh` (configured from the checkout `.env`) builds and publishes it hourly; AgentMarkit registers it as a private page, relaxes the page-link rule to any plain https URL, and wires the distribution.

**Tech Stack:** Python 3.12, SQLite, argparse, pytest (attestation); vanilla JS inside the page; Node `node:test` and Python `unittest` (agentmarkit).

**Spec:** `docs/superpowers/specs/2026-09-29-reading-desk-design.md` — read it first; this plan argues from it.

## Global Constraints

- The page makes no network request of its own: no `<script src`, no `<link`, no web fonts, no `fetch`. The host CSP is `default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; font-src data:; img-src data:; connect-src 'none'; form-action 'none'; base-uri 'none'; frame-src 'none'; object-src 'none'; worker-src 'none'`.
- Page file ≤ 8 MB (`private_pages.py` `MAX_HTML`). Page state ≤ 32,000 chars of JSON; the page prunes at 28,000.
- State format is exactly `{"v": 1, "verdicts": {"<item_id>": {"useful": <bool>, "at": "<ISO-8601 UTC>"}}}`.
- Verdicts are written only through `rank.record_click(conn, user_id, item_id, useful, source="ui")`.
- attestation never names AgentMarkit. Config is `ATTEST_DESK_STATE`, `ATTEST_DESK_USER`, `ATTEST_DESK_PUBLISH`, read from the environment (the checkout `.env` via `llm.load_env()` at entry points).
- Import never raises into a caller: `sqlite3.Error` / `ValueError` are caught in `desk.import_pending`, logged, and it returns `None`. No new `# noqa: BLE001` site.
- `feed.py` and `ask.py` are at their size caps (`tests/test_architecture.py::test_mcp_domain_modules_stay_small`); do not add code to them.
- Line length 100; ruff `E,F,W,I,BLE,RUF100`; `ty` clean.
- Gate before every commit that finishes a task: the task's own tests. Gate before the attestation PR: `uv run --frozen pre-commit run --all-files` in the FOREGROUND (never backgrounded; see memory `sdd-dispatch-lessons`).
- Commit locally; do not push before ~18:00 unless Matt asks (memory `no-daytime-pushes`). Check the PR is still open before pushing to its branch.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Tests never touch the live `~/.hermes` DB or `~/.hermes/private-pages`; use `tmp_path`.

## Review Focus

1. **A `.env` value with spaces and quotes** (`ATTEST_DESK_PUBLISH=python3 /x/private_pages.py register --id reading-desk --title "Reading desk" --html /x/desk.html`) must reach `subprocess.run` as the right argv after python-dotenv parsing and `shlex.split` — pinned in Task 3.
2. **The desk persona does not exist yet** (the hourly refresh fires before the first conversation creates `owner`): `attest desk refresh` must exit 0 with a clear line, not traceback or autocreate — pinned in Task 3.
3. **A hostile or huge feed title/summary** (10 MB title; `</script>` inside a title) must not break out of the embedded JSON or blow the 8 MB cap — pinned in Task 2.
4. **The page's state file is locked by the Worker mid-save** (`BEGIN IMMEDIATE`): the read-only import must not wait long or fail the ranking — pinned in Task 1 (held write lock → returns promptly, no raise).
5. **A non-https item URL** (`http://`, `javascript:`, None) must render as plain text, never as a link the host would drop silently — pinned in Task 2.

---

## Part A — attestation (`/home/matt/attestation`, branch `feat/reading-desk`, already created)

### Task 1: Import verdicts from the page state

**Files:**
- Create: `src/attestation/desk.py`
- Create: `tests/test_desk.py`
- Modify: `CLAUDE.md` (docs index: add `desk.py` to the `src/attestation:{...}` list and `test_desk.py` to the `tests:{...}` list — `test_the_docs_index_lists_every_source_and_test_file` fails otherwise)

**Interfaces:**
- Consumes: `attestation.rank.get_user(conn, name) -> Row | None`, `attestation.rank.record_click(conn, user_id, item_id, useful, source="ui") -> None`
- Produces:
  - `desk.STATE_VERSION: int = 1`
  - `desk.Imported` — frozen dataclass `(recorded: int = 0, unchanged: int = 0, skipped: int = 0)`
  - `desk.read_state(path: str | Path) -> dict`
  - `desk.import_verdicts(conn, user_id: int, state: dict) -> Imported`
  - `desk.import_pending(conn) -> Imported | None`
  - `desk.desk_config() -> tuple[Path | None, str | None]` — `(ATTEST_DESK_STATE expanded, ATTEST_DESK_USER)`, each `None` when unset/blank

- [ ] **Step 1: Write the failing tests**

`tests/test_desk.py`:

```python
"""The Reading desk: page state in, clicks out. See
docs/superpowers/specs/2026-09-29-reading-desk-design.md."""

import json
import sqlite3
import time

from conftest import seeded_db

from attestation import desk
from attestation.rank import get_user


def add_item(conn, title="a paper", url="https://example.org/a", summary="about it"):
    cur = conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash) VALUES (NULL, ?, ?, ?, ?)",
        (title, url, summary, f"hash-{title}"),
    )
    conn.commit()
    return cur.lastrowid


def page_state_file(path, value, revision=1):
    """The exact table private_pages.py's page_state() creates
    (hermes-starters/shared-skills/agentmarkit-links/scripts/private_pages.py,
    page_state): `state(id INTEGER PRIMARY KEY, revision INTEGER, value TEXT)`,
    one row at id=1 holding the JSON blob. Copied, not invented, so this test
    reads the format the Worker really writes."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, revision INTEGER, value TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, payload TEXT, result TEXT)"
    )
    conn.execute("INSERT OR REPLACE INTO state VALUES (1,?,?)", (revision, json.dumps(value)))
    conn.commit()
    conn.close()
    return path


def verdict(useful, at="2026-09-29T14:00:00Z"):
    return {"useful": useful, "at": at}


def clicks(conn, user_id):
    return {
        r["item_id"]: (r["useful"], r["source"])
        for r in conn.execute("SELECT item_id, useful, source FROM clicks WHERE user_id = ?", (user_id,))
    }


def test_import_records_each_verdict_as_a_ui_click(tmp_path):
    conn = seeded_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a, b = add_item(conn, "a"), add_item(conn, "b")
    state = {"v": 1, "verdicts": {str(a): verdict(True), str(b): verdict(False)}}

    got = desk.import_verdicts(conn, uid, state)

    assert got == desk.Imported(recorded=2, unchanged=0, skipped=0)
    assert clicks(conn, uid) == {a: (1, "ui"), b: (0, "ui")}


def test_import_is_idempotent(tmp_path):
    conn = seeded_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    state = {"v": 1, "verdicts": {str(a): verdict(True)}}

    desk.import_verdicts(conn, uid, state)
    again = desk.import_verdicts(conn, uid, state)

    assert again == desk.Imported(recorded=0, unchanged=1, skipped=0)
    assert conn.execute("SELECT COUNT(*) FROM clicks").fetchone()[0] == 1


def test_a_changed_mind_overwrites(tmp_path):
    conn = seeded_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    desk.import_verdicts(conn, uid, {"v": 1, "verdicts": {str(a): verdict(True)}})

    got = desk.import_verdicts(conn, uid, {"v": 1, "verdicts": {str(a): verdict(False)}})

    assert got.recorded == 1
    assert clicks(conn, uid) == {a: (0, "ui")}


def test_malformed_and_unknown_entries_are_skipped_not_raised(tmp_path):
    conn = seeded_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    state = {
        "v": 1,
        "verdicts": {
            "999999": verdict(True),  # no such item
            "abc": verdict(True),  # not an id
            str(a): {"useful": "yes", "at": "x"},  # not a bool
            "-3": verdict(True),
        },
    }

    got = desk.import_verdicts(conn, uid, state)

    assert got == desk.Imported(recorded=0, unchanged=0, skipped=4)
    assert clicks(conn, uid) == {}


def test_no_verdicts_key_imports_nothing(tmp_path):
    conn = seeded_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    assert desk.import_verdicts(conn, uid, {}) == desk.Imported()
    assert desk.import_verdicts(conn, uid, {"v": 1, "verdicts": []}) == desk.Imported()


def test_read_state_reads_the_worker_format(tmp_path):
    value = {"v": 1, "verdicts": {"7": verdict(True)}}
    path = page_state_file(tmp_path / "state.sqlite", value)
    assert desk.read_state(path) == value


def test_read_state_missing_file_or_wrong_version_is_empty(tmp_path):
    assert desk.read_state(tmp_path / "nope.sqlite") == {}
    path = page_state_file(tmp_path / "s.sqlite", {"v": 2, "verdicts": {}})
    assert desk.read_state(path) == {}
    garbage = tmp_path / "g.sqlite"
    garbage.write_bytes(b"not a database at all" * 100)
    assert desk.read_state(garbage) == {}


def test_read_state_never_writes(tmp_path):
    path = page_state_file(tmp_path / "state.sqlite", {"v": 1, "verdicts": {}})
    before = path.read_bytes()
    desk.read_state(path)
    assert path.read_bytes() == before
    assert not (tmp_path / "state.sqlite-journal").exists()


def test_read_state_returns_promptly_while_the_worker_holds_a_write_lock(tmp_path):
    """private_pages.py saves inside BEGIN IMMEDIATE. A reader in rollback-journal
    mode can still read during a RESERVED lock; this pins that the import does
    not hang or raise while a save is in flight."""
    value = {"v": 1, "verdicts": {"7": verdict(True)}}
    path = page_state_file(tmp_path / "state.sqlite", value)
    writer = sqlite3.connect(path)
    writer.execute("BEGIN IMMEDIATE")
    try:
        start = time.monotonic()
        got = desk.read_state(path)
        assert time.monotonic() - start < 6
        assert got in (value, {})
    finally:
        writer.rollback()
        writer.close()


def test_import_pending_is_none_when_unconfigured(tmp_path, monkeypatch):
    conn = seeded_db(tmp_path / "t.db")
    monkeypatch.delenv("ATTEST_DESK_STATE", raising=False)
    monkeypatch.delenv("ATTEST_DESK_USER", raising=False)
    assert desk.import_pending(conn) is None
    monkeypatch.setenv("ATTEST_DESK_STATE", str(tmp_path / "s.sqlite"))
    assert desk.import_pending(conn) is None  # user still unset


def test_import_pending_imports_for_the_desk_persona(tmp_path, monkeypatch):
    conn = seeded_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    a = add_item(conn)
    path = page_state_file(tmp_path / "s.sqlite", {"v": 1, "verdicts": {str(a): verdict(True)}})
    monkeypatch.setenv("ATTEST_DESK_STATE", str(path))
    monkeypatch.setenv("ATTEST_DESK_USER", "Researcher")  # case folds, like get_user

    got = desk.import_pending(conn)

    assert got == desk.Imported(recorded=1)
    assert clicks(conn, uid) == {a: (1, "ui")}


def test_import_pending_unknown_persona_imports_nothing(tmp_path, monkeypatch):
    conn = seeded_db(tmp_path / "t.db")
    a = add_item(conn)
    path = page_state_file(tmp_path / "s.sqlite", {"v": 1, "verdicts": {str(a): verdict(True)}})
    monkeypatch.setenv("ATTEST_DESK_STATE", str(path))
    monkeypatch.setenv("ATTEST_DESK_USER", "nobody")

    assert desk.import_pending(conn) is None
    assert conn.execute("SELECT COUNT(*) FROM clicks").fetchone()[0] == 0
    assert get_user(conn, "nobody") is None, "import must never autocreate a persona"


def test_import_pending_swallows_a_database_error(tmp_path, monkeypatch):
    conn = seeded_db(tmp_path / "t.db")
    path = page_state_file(tmp_path / "s.sqlite", {"v": 1, "verdicts": {"1": verdict(True)}})
    monkeypatch.setenv("ATTEST_DESK_STATE", str(path))
    monkeypatch.setenv("ATTEST_DESK_USER", "researcher")

    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(desk, "import_verdicts", boom)
    assert desk.import_pending(conn) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_desk.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'attestation.desk'`.

- [ ] **Step 3: Implement `src/attestation/desk.py` (import half)**

```python
"""The Reading desk: a private page of today's ranked papers whose verdicts
come back as clicks.

The page (rendered by `render_desk`) keeps its verdicts in one JSON blob that
the hosting platform stores in a small SQLite file. This module reads that
file READ-ONLY and imports the verdicts through `rank.record_click` before any
ranking a person sees -- a verdict changes nothing until a ranking is
computed, so importing then loses nothing. See
docs/superpowers/specs/2026-09-29-reading-desk-design.md.

Configured from the environment (the checkout `.env`, loaded at the entry
points): ATTEST_DESK_STATE (the state file), ATTEST_DESK_USER (the persona the
verdicts belong to), ATTEST_DESK_PUBLISH (a command run after a build).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from attestation.rank import get_user, record_click

log = logging.getLogger(__name__)

STATE_VERSION = 1
# Seconds a read waits on a lock the page host holds mid-save. The host's own
# write is a single-row replace, so anything longer means something is wrong
# and ranking should go ahead on the clicks it already has.
READ_TIMEOUT_S = 5


@dataclass(frozen=True)
class Imported:
    """What one import did. `unchanged` is a verdict already recorded the same
    way; `skipped` is an entry naming no known item or carrying no bool."""

    recorded: int = 0
    unchanged: int = 0
    skipped: int = 0


def desk_config() -> tuple[Path | None, str | None]:
    """(state file, persona name) from the environment; each None when blank."""
    raw_state = (os.environ.get("ATTEST_DESK_STATE") or "").strip()
    raw_user = (os.environ.get("ATTEST_DESK_USER") or "").strip()
    return (Path(raw_state).expanduser() if raw_state else None, raw_user or None)


def read_state(path: str | Path) -> dict:
    """The page's state blob, or {} when it is missing, unreadable, or not v1.

    Opened with `mode=ro` so this can never write, create, or hold a write
    lock on a file another program owns.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        conn = sqlite3.connect(
            path.absolute().as_uri() + "?mode=ro", uri=True, timeout=READ_TIMEOUT_S
        )
        try:
            row = conn.execute("SELECT value FROM state WHERE id = 1").fetchone()
        finally:
            conn.close()
        value = json.loads(row[0]) if row else {}
    except (sqlite3.Error, ValueError) as exc:
        log.warning("desk: cannot read page state %s: %s", path, exc)
        return {}
    if not isinstance(value, dict) or value.get("v") != STATE_VERSION:
        if value:
            log.warning("desk: page state %s is not version %d; ignored", path, STATE_VERSION)
        return {}
    return value


def import_verdicts(conn, user_id: int, state: dict) -> Imported:
    """Record each `verdicts[<item_id>] = {"useful": bool, ...}` as a `ui` click.

    Idempotent: a verdict already recorded the same way is left alone, so the
    hourly import and every ranking call can run it without rewriting rows.
    """
    verdicts = state.get("verdicts")
    if not isinstance(verdicts, dict):
        return Imported()
    recorded = unchanged = skipped = 0
    for key, entry in verdicts.items():
        useful = entry.get("useful") if isinstance(entry, dict) else None
        if not (isinstance(key, str) and key.isdecimal()) or not isinstance(useful, bool):
            skipped += 1
            continue
        item_id = int(key)
        if conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone() is None:
            skipped += 1
            continue
        row = conn.execute(
            "SELECT useful FROM clicks WHERE user_id = ? AND item_id = ?", (user_id, item_id)
        ).fetchone()
        if row is not None and bool(row[0]) == useful:
            unchanged += 1
            continue
        record_click(conn, user_id, item_id, useful, source="ui")
        recorded += 1
    return Imported(recorded, unchanged, skipped)


def import_pending(conn) -> Imported | None:
    """Import the configured page's verdicts; None when unconfigured or failed.

    Never raises: this runs in front of every ranking, and a broken state file
    must degrade to ranking on the clicks already recorded, not to an error.
    Never creates a persona either -- verdicts for a name that does not exist
    yet wait in the state file until it does.
    """
    state_path, name = desk_config()
    if state_path is None or name is None:
        return None
    user = get_user(conn, name)
    if user is None:
        log.warning("desk: ATTEST_DESK_USER=%r is not a persona yet; nothing imported", name)
        return None
    try:
        return import_verdicts(conn, user["id"], read_state(state_path))
    except (sqlite3.Error, ValueError) as exc:
        log.warning("desk: import failed, ranking on existing clicks: %s", exc)
        return None
```

Also add `desk.py` and `test_desk.py` to CLAUDE.md's docs index (alphabetical position is not enforced; append after `ports.py` in `src/attestation:{...}` and after `test_db.py` in `tests:{...}`).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_desk.py tests/test_architecture.py::test_the_docs_index_lists_every_source_and_test_file tests/test_architecture.py::test_import_graph_is_acyclic -q`
Expected: all pass. If `read_state_returns_promptly...` shows `{}` instead of the value, that is allowed by the assertion — it pins "prompt, no raise", not the WAL-vs-journal detail.

- [ ] **Step 5: Mutation check**

Temporarily change `record_click(conn, user_id, item_id, useful, source="ui")` to `record_click(conn, user_id, item_id, not useful, source="ui")`; run `uv run pytest tests/test_desk.py -q`; expect ≥ 2 failures. Revert.

- [ ] **Step 6: Commit**

```bash
git add src/attestation/desk.py tests/test_desk.py CLAUDE.md
git commit -m "desk: import page verdicts as ui clicks, read-only

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Render the page

**Files:**
- Create: `src/attestation/render.py` (move `MAX_RENDERED_CHARS`, `safe_href`, `clip` out of `server.py` verbatim, docstrings and comments included)
- Modify: `src/attestation/server.py` (delete the three definitions; `from attestation.render import MAX_RENDERED_CHARS, clip, safe_href` so every existing name still resolves)
- Modify: `src/attestation/desk.py` (add rendering)
- Modify: `CLAUDE.md` docs index (add `render.py`)
- Test: `tests/test_desk.py`

Why the move: `desk.py` needs `clip`, and importing `server.py` would drag FastAPI and Jinja into the CLI and the MCP server. The spec says reuse, not copy.

**Interfaces:**
- Consumes: `rank.rank_items(conn, embedder, user_id, since_days=14) -> list[RankedItem]` (fields `item_id, title, url, source, tags, summary`), `rank.ranking_quality(conn, user_id) -> dict` (`classifier_active: bool`, optional `caveat: str`, optional `real_clicks: int`), `render.clip(value, limit) -> str`
- Produces:
  - `desk.DEFAULT_DESK_LIMIT = 20`, `desk.SUMMARY_CHARS = 600`
  - `desk.desk_payload(conn, embedder, user_id: int, limit: int = DEFAULT_DESK_LIMIT) -> dict` — `{"items": [{"id","title","url","source","tags","summary"}], "caveat": str | None, "rated": int}`
  - `desk.render_desk(conn, embedder, user_id: int, limit: int = DEFAULT_DESK_LIMIT) -> str`
  - `desk.DESK_LOGIC_JS: str` — defines `prune(verdicts, cap)` and `merge(mine, theirs)` as plain functions (tested under node)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_desk.py`)

```python
import re
import shutil
import subprocess

import pytest

from attestation.rank import record_click

HOST_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "font-src data:; img-src data:; connect-src 'none'; form-action 'none'; "
    "base-uri 'none'; frame-src 'none'; object-src 'none'; worker-src 'none'"
)


def ranked_db(tmp_path, embedder, n=3, **item):
    """A seeded DB with n embedded items, ranked for 'researcher'."""
    conn = seeded_db(tmp_path / "t.db")
    ids = []
    for i in range(n):
        title = item.get("title", f"paper {i}")
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash) VALUES (NULL,?,?,?,?)",
            (title, item.get("url", f"https://example.org/{i}"), item.get("summary", "abstract"),
             f"h{i}"),
        )
        vec = embedder.embed_document(title, "abstract")
        conn.execute(
            "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)", (cur.lastrowid, vec.tobytes())
        )
        ids.append(cur.lastrowid)
    conn.commit()
    return conn, get_user(conn, "researcher")["id"], ids


def test_page_is_self_contained(tmp_path, fake_embedder):
    conn, uid, _ = ranked_db(tmp_path, fake_embedder)
    html = desk.render_desk(conn, fake_embedder, uid)

    assert "<script src" not in html.lower()
    assert "<link" not in html.lower()
    assert "http:" not in html
    assert "fetch(" not in html
    item_urls = {f"https://example.org/{i}" for i in range(3)}
    assert set(re.findall(r"https://[^\"'\s<\\]+", html)) <= item_urls
    assert len(html.encode()) < 8 * 1024 * 1024


def test_page_has_both_tabs_and_every_item(tmp_path, fake_embedder):
    conn, uid, ids = ranked_db(tmp_path, fake_embedder)
    payload = desk.desk_payload(conn, fake_embedder, uid)
    html = desk.render_desk(conn, fake_embedder, uid)

    assert 'data-tab="triage"' in html and 'data-tab="read"' in html
    assert sorted(i["id"] for i in payload["items"]) == sorted(ids)
    assert all(i["summary"] == "abstract" for i in payload["items"])


def test_empty_ranking_renders_the_waiting_state(tmp_path, fake_embedder):
    conn = seeded_db(tmp_path / "t.db")
    uid = get_user(conn, "researcher")["id"]
    html = desk.render_desk(conn, fake_embedder, uid)
    assert "No papers yet" in html


def test_caveat_shown_while_the_classifier_is_off(tmp_path, fake_embedder):
    conn, uid, ids = ranked_db(tmp_path, fake_embedder)
    payload = desk.desk_payload(conn, fake_embedder, uid)
    assert payload["caveat"]  # zero clicks: classifier inactive
    assert payload["rated"] == 0


def test_rated_counts_human_verdicts_only(tmp_path, fake_embedder):
    conn, uid, ids = ranked_db(tmp_path, fake_embedder, n=4)
    record_click(conn, uid, ids[0], True, source="ui")
    record_click(conn, uid, ids[1], False, source="agent")
    record_click(conn, uid, ids[2], True, source="simulated")
    assert desk.desk_payload(conn, fake_embedder, uid)["rated"] == 2


def test_hostile_title_cannot_close_the_data_script(tmp_path, fake_embedder):
    title = "</script><script>alert(1)</script>" + "x" * 10_000_000
    conn, uid, _ = ranked_db(tmp_path, fake_embedder, n=1, title=title)
    html = desk.render_desk(conn, fake_embedder, uid)

    assert html.count("</script>") == html.count("<script")  # only our own tags close
    assert len(html.encode()) < 200_000


@pytest.mark.parametrize("url", ["http://example.org/x", "javascript:alert(1)", None, ""])
def test_a_non_https_url_is_not_a_link(tmp_path, fake_embedder, url):
    conn, uid, _ = ranked_db(tmp_path, fake_embedder, n=1, url=url)
    payload = desk.desk_payload(conn, fake_embedder, uid)
    assert payload["items"][0]["url"] is None


node = shutil.which("node")


@pytest.mark.skipif(node is None, reason="node not installed")
def test_page_logic_prunes_oldest_first_and_merges_newest_wins():
    script = desk.DESK_LOGIC_JS + """
const v = {};
for (let i = 0; i < 1000; i++) v[String(i)] = {useful: true, at: new Date(2026, 0, 1, 0, 0, i).toISOString()};
const kept = prune(v, 28000);
const size = JSON.stringify({v: 1, verdicts: kept}).length;
const keys = Object.keys(kept).map(Number);
const merged = merge({"1": {useful: true, at: "2026-09-29T10:00:00Z"}, "2": {useful: false, at: "2026-09-29T09:00:00Z"}},
                     {"2": {useful: true, at: "2026-09-29T11:00:00Z"}, "3": {useful: true, at: "2026-09-29T08:00:00Z"}});
console.log(JSON.stringify({size, min: Math.min(...keys), max: Math.max(...keys), merged}));
"""
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert got["size"] <= 28000
    assert got["max"] == 999 and got["min"] > 0  # the OLDEST went, the newest stayed
    assert got["merged"] == {
        "1": {"useful": True, "at": "2026-09-29T10:00:00Z"},
        "2": {"useful": True, "at": "2026-09-29T11:00:00Z"},
        "3": {"useful": True, "at": "2026-09-29T08:00:00Z"},
    }
```

Also in `tests/test_server.py` nothing changes; the existing server tests prove the move kept `clip`/`safe_href` working.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_desk.py -q`
Expected: new tests fail with `AttributeError: module 'attestation.desk' has no attribute 'render_desk'` (and `desk_payload`, `DESK_LOGIC_JS`).

- [ ] **Step 3: Create `src/attestation/render.py`**

Move, verbatim, from `server.py`: the `MAX_RENDERED_CHARS` constant with its comment block, `safe_href`, and `clip`. Module docstring:

```python
"""Bounds and guards for third-party strings at a render boundary.

Shared by the web UI (server.py) and the Reading desk page (desk.py), so a
fix to either reaches both. No FastAPI, no Jinja: importing this must stay
cheap for the CLI and the MCP server.
"""
```

In `server.py`, replace the three definitions with
`from attestation.render import MAX_RENDERED_CHARS, clip, safe_href`.

- [ ] **Step 4: Add rendering to `desk.py`**

Add imports: `from attestation.rank import HUMAN_CLICK_SOURCES, get_user, rank_items, ranking_quality, record_click` and `from attestation.render import clip`.

```python
DEFAULT_DESK_LIMIT = 20
SUMMARY_CHARS = 600
TITLE_CHARS = 300
SOURCE_CHARS = 80
# The page's pruning threshold: under the platform's 32,000-char state cap
# with room for the envelope, so a save never fails on size.
STATE_PRUNE_CHARS = 28_000


def _https_or_none(url: str | None) -> str | None:
    """Only https survives: the page host opens https links behind a confirm
    dialog and drops everything else silently, so any other scheme would be
    a link that does nothing. Plain text is the honest rendering."""
    return url if isinstance(url, str) and url.startswith("https://") else None


def desk_payload(conn, embedder, user_id: int, limit: int = DEFAULT_DESK_LIMIT) -> dict:
    """What the page shows, as data: the ranked items, the ranking caveat
    while the classifier is off, and how many papers a person has rated."""
    items = rank_items(conn, embedder, user_id)[:limit]
    quality = ranking_quality(conn, user_id)
    rated = conn.execute(
        "SELECT COUNT(*) FROM clicks WHERE user_id = ? AND source IN ({})".format(
            ",".join("?" * len(HUMAN_CLICK_SOURCES))
        ),
        (user_id, *sorted(HUMAN_CLICK_SOURCES)),
    ).fetchone()[0]
    return {
        "items": [
            {
                "id": it.item_id,
                "title": clip(it.title, TITLE_CHARS),
                "url": _https_or_none(it.url),
                "source": clip(it.source or "", SOURCE_CHARS),
                "tags": [clip(t, 40) for t in it.tags[:3]],
                "summary": clip((it.summary or "").strip(), SUMMARY_CHARS),
            }
            for it in items
        ],
        "caveat": None if quality.get("classifier_active") else quality.get("caveat"),
        "rated": rated,
    }


def _embed_json(value: dict) -> str:
    """JSON safe inside a <script> element: `<` never appears raw, so no
    string in the data can close the element it lives in."""
    return json.dumps(value).replace("<", "\\u003c")
```

`DESK_LOGIC_JS` (pure functions, no DOM — the node test runs exactly this string):

```python
DESK_LOGIC_JS = """
function prune(verdicts, cap) {
  const entries = Object.entries(verdicts).sort((a, b) => (a[1].at < b[1].at ? -1 : 1));
  const size = () => JSON.stringify({v: 1, verdicts: Object.fromEntries(entries)}).length;
  while (entries.length && size() > cap) entries.shift();
  return Object.fromEntries(entries);
}
function merge(mine, theirs) {
  const out = Object.assign({}, theirs);
  for (const [k, e] of Object.entries(mine)) if (!out[k] || out[k].at < e.at) out[k] = e;
  return out;
}
"""
```

Note `size()` inside the loop is O(n²) on 1000 entries — fine (the test runs it; ~1000 × 55 KB stringify ≈ milliseconds). Do not optimise.

`DESK_UI_JS` (DOM + bridge; uses only `textContent`/`createElement`, never `innerHTML`):

```python
DESK_UI_JS = """
(() => {
  const data = JSON.parse(document.getElementById('desk-data').textContent);
  const am = window.agentmarkit;
  let revision = 0, verdicts = {};
  const $ = s => document.querySelector(s);
  const say = t => { $('#status').textContent = t; };
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text) n.textContent = text;
    return n;
  }
  function buttons(id) {
    const box = el('div', 'verdict');
    for (const [useful, label] of [[true, 'Useful'], [false, 'Not my area']]) {
      const b = el('button', useful ? 'yes' : 'no', label);
      b.type = 'button';
      b.setAttribute('aria-pressed', String(verdicts[id] ? verdicts[id].useful === useful : false));
      b.addEventListener('click', () => judge(id, useful));
      box.append(b);
    }
    return box;
  }
  function row(item, full) {
    const li = el('li', 'paper');
    const title = item.url ? el('a', 'title', item.title) : el('span', 'title', item.title);
    if (item.url) { title.href = item.url; title.rel = 'noreferrer'; }
    li.append(title, el('div', 'meta', [item.source, ...item.tags].filter(Boolean).join(' · ')));
    if (full && item.summary) li.append(el('p', 'summary', item.summary));
    if (am) li.append(buttons(item.id));
    return li;
  }
  function render() {
    for (const [sel, full] of [['#triage', false], ['#read', true]]) {
      const list = $(sel);
      list.replaceChildren(...data.items.map(i => row(i, full)));
    }
    const mine = Object.keys(verdicts).length;
    $('#rated').textContent = `${data.rated} rated so far` + (mine ? `, ${mine} on this page` : '');
  }
  async function save(retry) {
    const r = await am.save({v: 1, verdicts: prune(verdicts, %(cap)d)}, revision);
    if (r && r.ok) {
      revision = r.state.revision;
      verdicts = (r.state.data && r.state.data.verdicts) || {};
      am.dirty(false); say('Saved'); render(); return;
    }
    if (r && r.conflict && retry) {
      revision = r.state.revision;
      verdicts = merge(verdicts, (r.state.data && r.state.data.verdicts) || {});
      return save(false);
    }
    say('Not saved yet. Tap again to retry.');
  }
  function judge(id, useful) {
    verdicts[String(id)] = {useful, at: new Date().toISOString()};
    am.dirty(true); say('Saving…'); render();
    save(true).catch(e => say(e.message));
  }
  async function load() {
    try {
      const r = await am.state();
      if (r && r.ok) {
        revision = r.state.revision;
        const d = r.state.data || {};
        verdicts = d.v === 1 && d.verdicts ? d.verdicts : {};
      }
    } catch (e) { say(e.message); }
    render();
  }
  for (const b of document.querySelectorAll('[data-tab]')) {
    b.addEventListener('click', () => { document.body.dataset.show = b.dataset.tab; });
  }
  if (am) { load(); addEventListener('agentmarkit-resume', load); } else { render(); }
})();
""" % {"cap": STATE_PRUNE_CHARS}
```

Careful: `%` formatting means any literal `%` in the JS must be `%%`. There is none above; the node test and the render test catch a mistake.

The page shell and `render_desk`:

```python
_DESK_CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--accent:#2f5d50}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--muted:#a3a19a;
--line:#2c2b27;--accent:#8cc5b2}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
font:16px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:720px;margin:0 auto}
h1{font-size:1.3rem;margin:0 0 .25rem}
.sub,.meta,#rated,#status{color:var(--muted);font-size:.9rem}
.caveat{border-left:3px solid var(--accent);padding:.25rem .75rem;margin:.75rem 0}
nav{display:flex;gap:.5rem;margin:1rem 0}
nav button{flex:1;padding:.5rem;border:1px solid var(--line);background:none;color:var(--fg);
border-radius:6px;font:inherit}
body[data-show=triage] nav [data-tab=triage],body[data-show=read] nav [data-tab=read]{
border-color:var(--accent);color:var(--accent)}
body[data-show=triage] #read,body[data-show=read] #triage{display:none}
ol{list-style:none;padding:0;margin:0}
.paper{padding:.75rem 0;border-bottom:1px solid var(--line)}
.title{font-weight:600;color:var(--fg)}
a.title{color:var(--accent)}
.summary{margin:.4rem 0}
.verdict{display:flex;gap:.5rem;margin-top:.5rem}
.verdict button{padding:.35rem .75rem;border:1px solid var(--line);border-radius:999px;
background:none;color:var(--fg);font:inherit;font-size:.9rem}
.verdict button[aria-pressed=true]{background:var(--accent);border-color:var(--accent);
color:var(--bg)}
.empty{padding:2rem 0;color:var(--muted)}
"""


def render_desk(conn, embedder, user_id: int, limit: int = DEFAULT_DESK_LIMIT) -> str:
    """One self-contained HTML page: no request of its own, safe under the
    page host's CSP (see the spec's "What the platform allows")."""
    payload = desk_payload(conn, embedder, user_id, limit)
    if payload["items"]:
        lists = '<ol id="triage"></ol><ol id="read"></ol>'
    else:
        lists = (
            '<p class="empty">No papers yet: the refresh fetches new ones hourly.</p>'
            '<ol id="triage"></ol><ol id="read"></ol>'
        )
    caveat = (
        f'<p class="caveat">{html.escape(payload["caveat"])}</p>' if payload["caveat"] else ""
    )
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>Reading desk</title><style>{_DESK_CSS}</style></head>"
        '<body data-show="triage"><main>'
        '<h1>Reading desk</h1><p class="sub">Today\'s papers, best first. '
        "Mark what helps and what is not your area; the ranking learns from both.</p>"
        f'{caveat}<p id="rated"></p><p id="status" role="status"></p>'
        '<nav><button type="button" data-tab="triage">Triage</button>'
        '<button type="button" data-tab="read">Read</button></nav>'
        f"{lists}</main>"
        f'<script type="application/json" id="desk-data">{_embed_json(payload)}</script>'
        f"<script>{DESK_LOGIC_JS}{DESK_UI_JS}</script></body></html>"
    )
```

Add `import html` to the imports.

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_desk.py tests/test_server.py tests/test_architecture.py -q`
Expected: all pass. If `test_page_is_self_contained` finds an `https://` outside item URLs, the offender is in the CSS/JS/HTML strings — remove it.

- [ ] **Step 6: Look at it**

Run: `uv run python -c "from attestation import desk; print('ok')"` then render against a COPY of the live DB and open it in a browser (the Playwright MCP tools work): `cp ~/.hermes/skills/science-recommendations/data/hermes.db /tmp/claude-1000/desk.db && ATTEST_DB=/tmp/claude-1000/desk.db uv run attest desk build --user matt --out /tmp/claude-1000/desk.html` — this needs Task 3; if doing Task 2 alone, skip to Step 7 and do this look after Task 3. Check: both tabs switch, links present, no buttons (no bridge locally), dark mode readable.

- [ ] **Step 7: Mutation check**

Replace `.replace("<", "\\u003c")` in `_embed_json` with nothing (`json.dumps(value)` only); run `uv run pytest tests/test_desk.py -k hostile -q`; expect FAIL. Revert.

- [ ] **Step 8: Commit**

```bash
git add src/attestation/render.py src/attestation/server.py src/attestation/desk.py tests/test_desk.py CLAUDE.md
git commit -m "desk: render the Reading desk page; clip/safe_href move to render.py

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `attest desk build | import | refresh`

**Files:**
- Modify: `src/attestation/cli.py` (HELP entries, parser, three `cmd_desk_*` functions; imports stay lazy inside the functions — `test_cli_help_stays_fast`)
- Modify: `src/attestation/desk.py` (add `desk_output_path()` and `publish(command)`)
- Regenerate: `docs/reference/cli.md` via `uv run python scripts/render_cli_reference.py`
- Test: `tests/test_desk.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 1 `desk_config`, `import_pending`, `import_verdicts`, `read_state`; Task 2 `render_desk`; `attestation.paths.hermes_home() -> Path`; `attestation.cli.open_db`, `add_db`, `fail`
- Produces:
  - `desk.desk_output_path() -> Path` = `paths.hermes_home() / "workspace" / "research-desk" / "desk.html"`
  - `desk.publish_argv(command: str) -> list[str]` — `shlex.split` then `os.path.expanduser` per arg
  - `desk.publish(command: str) -> subprocess.CompletedProcess` — `subprocess.run(argv, capture_output=True, text=True, timeout=PUBLISH_TIMEOUT_S, check=False)`, `PUBLISH_TIMEOUT_S = 60`
  - CLI: `attest desk build --user NAME --out PATH [--limit N] [--db DB]`, `attest desk import --user NAME [--db DB]`, `attest desk refresh [--db DB]`
  - Exit codes: build/import 0 ok, 1 unknown persona; refresh 0 when done OR not configured OR persona missing (prints why), 1 when build or publish failed

- [ ] **Step 1: Write the failing tests**

In `tests/test_desk.py`:

```python
from dotenv import dotenv_values


def test_publish_command_survives_dotenv_and_shlex(tmp_path, monkeypatch):
    """Review focus 1: the value AgentMarkit writes into .env must reach
    subprocess as the argv it means, quotes and all."""
    env = tmp_path / ".env"
    env.write_text(
        "ATTEST_DESK_PUBLISH=python3 ~/.hermes/skills/x/private_pages.py register"
        ' --id reading-desk --title "Reading desk" --html ~/.hermes/workspace/d.html\n'
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    argv = desk.publish_argv(dotenv_values(env)["ATTEST_DESK_PUBLISH"])
    assert argv == [
        "python3", f"{tmp_path}/.hermes/skills/x/private_pages.py", "register",
        "--id", "reading-desk", "--title", "Reading desk",
        "--html", f"{tmp_path}/.hermes/workspace/d.html",
    ]


def test_output_path_follows_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    assert desk.desk_output_path() == tmp_path / "hh" / "workspace" / "research-desk" / "desk.html"
```

In `tests/test_cli.py` (follow that file's existing way of invoking `cli.main([...])` and capturing output — read its first 60 lines before writing; the shape below assumes `main(argv) -> int` plus `capsys`; adapt to what is there):

```python
def _desk_db(tmp_path, fake_embedder, monkeypatch):
    from conftest import seeded_db
    db = tmp_path / "t.db"
    conn = seeded_db(db)
    cur = conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (NULL, 'p', 'https://example.org/p', 's', 'h')"
    )
    conn.execute(
        "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
        (cur.lastrowid, fake_embedder.embed_document("p", "s").tobytes()),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("attestation.embed.Embedder", lambda *a, **k: fake_embedder)
    monkeypatch.setenv("ATTEST_DB", str(db))
    return db, cur.lastrowid


def test_desk_build_writes_the_page(tmp_path, fake_embedder, monkeypatch, capsys):
    from attestation import cli
    _desk_db(tmp_path, fake_embedder, monkeypatch)
    out = tmp_path / "desk.html"
    assert cli.main(["desk", "build", "--user", "researcher", "--out", str(out)]) == 0
    assert "Reading desk" in out.read_text()


def test_desk_build_unknown_persona_fails(tmp_path, fake_embedder, monkeypatch, capsys):
    from attestation import cli
    _desk_db(tmp_path, fake_embedder, monkeypatch)
    assert cli.main(["desk", "build", "--user", "nobody", "--out", str(tmp_path / "d.html")]) == 1
    assert not (tmp_path / "d.html").exists()


def test_desk_refresh_unconfigured_is_a_quiet_success(tmp_path, monkeypatch, capsys):
    from attestation import cli
    monkeypatch.delenv("ATTEST_DESK_STATE", raising=False)
    monkeypatch.delenv("ATTEST_DESK_USER", raising=False)
    assert cli.main(["desk", "refresh"]) == 0
    assert "desk not configured" in capsys.readouterr().out


def test_desk_refresh_before_the_persona_exists(tmp_path, fake_embedder, monkeypatch, capsys):
    """Review focus 2: the hourly refresh can fire before the first
    conversation creates the persona. Exit 0, say so, create nothing."""
    from attestation import cli
    from attestation.db import get_db
    from attestation.rank import get_user
    db, _ = _desk_db(tmp_path, fake_embedder, monkeypatch)
    monkeypatch.setenv("ATTEST_DESK_STATE", str(tmp_path / "s.sqlite"))
    monkeypatch.setenv("ATTEST_DESK_USER", "owner")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    assert cli.main(["desk", "refresh"]) == 0
    assert "no persona 'owner' yet" in capsys.readouterr().out
    assert get_user(get_db(db), "owner") is None


def test_desk_refresh_imports_builds_and_publishes(tmp_path, fake_embedder, monkeypatch, capsys):
    import json
    import sqlite3
    import sys
    from attestation import cli
    from attestation.db import get_db
    db, item_id = _desk_db(tmp_path, fake_embedder, monkeypatch)
    state = tmp_path / "s.sqlite"
    c = sqlite3.connect(state)
    c.execute("CREATE TABLE state (id INTEGER PRIMARY KEY, revision INTEGER, value TEXT)")
    c.execute("INSERT INTO state VALUES (1, 1, ?)", (json.dumps(
        {"v": 1, "verdicts": {str(item_id): {"useful": True, "at": "2026-09-29T00:00:00Z"}}}),))
    c.commit()
    c.close()
    marker = tmp_path / "published"
    monkeypatch.setenv("ATTEST_DESK_STATE", str(state))
    monkeypatch.setenv("ATTEST_DESK_USER", "researcher")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    monkeypatch.setenv(
        "ATTEST_DESK_PUBLISH", f"{sys.executable} -c \"open('{marker}','w').write('x')\""
    )

    assert cli.main(["desk", "refresh"]) == 0

    assert (tmp_path / "hh" / "workspace" / "research-desk" / "desk.html").exists()
    assert marker.exists()
    row = get_db(db).execute("SELECT useful, source FROM clicks").fetchone()
    assert (row["useful"], row["source"]) == (1, "ui")
    assert "recorded 1" in capsys.readouterr().out


def test_desk_refresh_reports_a_failed_publish(tmp_path, fake_embedder, monkeypatch, capsys):
    import sys
    from attestation import cli
    _desk_db(tmp_path, fake_embedder, monkeypatch)
    monkeypatch.setenv("ATTEST_DESK_STATE", str(tmp_path / "s.sqlite"))
    monkeypatch.setenv("ATTEST_DESK_USER", "researcher")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    monkeypatch.setenv("ATTEST_DESK_PUBLISH", f"{sys.executable} -c 'raise SystemExit(4)'")
    assert cli.main(["desk", "refresh"]) == 1
    assert "publish FAILED (exit 4)" in capsys.readouterr().out
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_desk.py tests/test_cli.py -k desk -q`
Expected: FAIL — `publish_argv`/`desk_output_path` missing; `attest: invalid choice: 'desk'`.

- [ ] **Step 3: Implement in `desk.py`**

```python
import shlex
import subprocess

from attestation import paths

PUBLISH_TIMEOUT_S = 60


def desk_output_path() -> Path:
    """Where the configured refresh writes the page."""
    return paths.hermes_home() / "workspace" / "research-desk" / "desk.html"


def publish_argv(command: str) -> list[str]:
    """ATTEST_DESK_PUBLISH as argv: shell-style quoting, `~` expanded per
    argument, and no shell -- the value comes from a file, and nothing in it
    needs pipes or variables."""
    return [os.path.expanduser(arg) for arg in shlex.split(command)]


def publish(command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        publish_argv(command), capture_output=True, text=True,
        timeout=PUBLISH_TIMEOUT_S, check=False,
    )
```

- [ ] **Step 4: Implement the CLI**

In `cli.py` `HELP`:

```python
    "desk": "the Reading desk page: today's ranked papers, verdicts taken back",
    "desk.build": "render the page for a persona to a file (imports pending verdicts first)",
    "desk.import": "record the page's pending verdicts as clicks",
    "desk.refresh": "import, build and publish as configured in .env; no-op when unconfigured",
```

Parser (mirror the `runs` group's structure exactly — read lines ~271-300 first):

```python
    sp = sub.add_parser("desk", help=HELP["desk"])
    dsub = sp.add_subparsers(dest="desk_command", required=True)
    dp = dsub.add_parser("build", help=HELP["desk.build"])
    add_db(dp)
    dp.add_argument("--user", required=True)
    dp.add_argument("--out", required=True, type=Path)
    dp.add_argument("--limit", type=int, default=20)
    dp.set_defaults(func=cmd_desk_build)
    dp = dsub.add_parser("import", help=HELP["desk.import"])
    add_db(dp)
    dp.add_argument("--user", required=True)
    dp.set_defaults(func=cmd_desk_import)
    dp = dsub.add_parser("refresh", help=HELP["desk.refresh"])
    add_db(dp)
    dp.set_defaults(func=cmd_desk_refresh)
```

Commands:

```python
def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _desk_import_line(got) -> str:
    return (
        f"verdicts: recorded {got.recorded}, unchanged {got.unchanged}, skipped {got.skipped}"
        if got is not None
        else "verdicts: nothing to import"
    )


def cmd_desk_build(args: argparse.Namespace) -> int:
    """Render the Reading desk for one persona. Pending page verdicts are
    imported first so the page never shows a paper already judged."""
    from attestation import desk
    from attestation.embed import Embedder
    from attestation.rank import get_user

    with open_db(args.db, need_vectors=True) as conn:
        user = get_user(conn, args.user)
        if user is None:
            return fail(f"no persona {args.user!r}")
        print(_desk_import_line(desk.import_pending(conn)))
        _write_atomic(args.out, desk.render_desk(conn, Embedder(), user["id"], args.limit))
    print(f"wrote {args.out}")
    return 0


def cmd_desk_import(args: argparse.Namespace) -> int:
    from attestation import desk
    from attestation.rank import get_user

    state_path, _ = desk.desk_config()
    with open_db(args.db) as conn:
        user = get_user(conn, args.user)
        if user is None:
            return fail(f"no persona {args.user!r}")
        if state_path is None:
            print("desk not configured: ATTEST_DESK_STATE is unset")
            return 0
        got = desk.import_verdicts(conn, user["id"], desk.read_state(state_path))
    print(_desk_import_line(got))
    return 0


def cmd_desk_refresh(args: argparse.Namespace) -> int:
    """What the hourly refresh runs. Unconfigured, or configured for a persona
    that the first conversation has not created yet, is success: the refresh
    must not go red on a machine that simply has no desk yet."""
    from attestation import desk
    from attestation.embed import Embedder
    from attestation.rank import get_user

    state_path, name = desk.desk_config()
    if state_path is None or name is None:
        print("desk not configured: set ATTEST_DESK_STATE and ATTEST_DESK_USER")
        return 0
    with open_db(args.db, need_vectors=True) as conn:
        user = get_user(conn, name)
        if user is None:
            print(f"desk: no persona {name!r} yet; nothing built")
            return 0
        print(_desk_import_line(desk.import_pending(conn)))
        out = desk.desk_output_path()
        _write_atomic(out, desk.render_desk(conn, Embedder(), user["id"]))
    print(f"wrote {out}")
    command = (os.environ.get("ATTEST_DESK_PUBLISH") or "").strip()
    if not command:
        return 0
    try:
        done = desk.publish(command)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"desk: publish FAILED ({exc})")
        return 1
    if done.returncode != 0:
        print(f"desk: publish FAILED (exit {done.returncode}): {done.stderr.strip()[:300]}")
        return 1
    print("desk: published")
    return 0
```

`os` and `Path` are already imported at module scope in `cli.py`; add `import subprocess` there (stdlib, cheap). `fail()` prints to STDERR and returns 1, so tests on a failure read `capsys.readouterr().err`. `build` and `refresh` pass `need_vectors=True` to `open_db` because ranking reads `item_vectors`; `import` does not.

Note the "no persona" message: the test expects `no persona 'owner' yet` — `{name!r}` renders `'owner'`. Keep them in sync.

- [ ] **Step 5: Regenerate the CLI reference**

Run: `uv run python scripts/render_cli_reference.py`
Then: `uv run pytest tests/test_docs_site.py tests/test_desk.py tests/test_cli.py tests/test_architecture.py::test_cli_help_stays_fast -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/attestation/cli.py src/attestation/desk.py tests/test_desk.py tests/test_cli.py docs/reference/cli.md
git commit -m "attest desk build|import|refresh

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Import before every ranking a person sees

**Files:**
- Modify: `src/attestation/mcp/_shared.py:83-87` (`ranked_items`)
- Test: `tests/test_desk.py`

**Interfaces:**
- Consumes: `desk.import_pending(conn) -> Imported | None` (never raises)
- Produces: `ranked_items(conn, user_row, limit, since_days)` unchanged signature; now imports first

- [ ] **Step 1: Write the failing test — through the real registered tool**

```python
def test_feed_ask_imports_page_verdicts_before_ranking(tmp_path, fake_embedder, monkeypatch):
    """Drives feed.ask on a real FastMCP server: the 0.2.0 regression shipped
    because a test checked the dict before the tool, not the tool."""
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from attestation.db import get_db
    from attestation.mcp import _shared, register_all

    db = tmp_path / "t.db"
    conn = seeded_db(db)
    uid = get_user(conn, "researcher")["id"]
    ids = []
    for i in range(3):
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, ?, 's', ?)", (f"paper {i}", f"https://example.org/{i}", f"h{i}"),
        )
        conn.execute("INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
                     (cur.lastrowid, fake_embedder.embed_document(f"paper {i}", "s").tobytes()))
        ids.append(cur.lastrowid)
    conn.commit()
    conn.close()
    state = page_state_file(tmp_path / "s.sqlite",
                            {"v": 1, "verdicts": {str(ids[0]): verdict(False)}})
    monkeypatch.setenv("ATTEST_DB", str(db))
    monkeypatch.setenv("ATTEST_DESK_STATE", str(state))
    monkeypatch.setenv("ATTEST_DESK_USER", "researcher")
    monkeypatch.setattr(_shared, "get_embedder", lambda: fake_embedder)

    server = FastMCP("t")
    register_all(server)

    async def call():
        result = await server.call_tool(
            "feed.ask", {"user": "researcher", "question": "What should I read today?"}
        )
        return result[1] if isinstance(result, tuple) else result

    structured = asyncio.run(call())

    assert structured.get("ok") is True, structured
    row = get_db(db).execute(
        "SELECT useful, source FROM clicks WHERE user_id = ? AND item_id = ?", (uid, ids[0])
    ).fetchone()
    assert row is not None and (row["useful"], row["source"]) == (0, "ui")
    # clicked items are excluded from the unread list, so the judged paper is gone
    assert all(ref.get("item_id") != ids[0] for ref in structured.get("refs", [])), structured
```

If `feed.ask`'s reading route does not return `refs` with `item_id`, drop the last assertion and keep the click assertion — read `mcp/ask.py`'s `Answer` model first to confirm the field names.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_desk.py -k feed_ask -q`
Expected: FAIL — `row is None`.

- [ ] **Step 3: Implement**

`src/attestation/mcp/_shared.py`:

```python
from attestation import desk

def ranked_items(conn, user_row, limit: int, since_days: int | None) -> list:
    """Rank items for an already-resolved user row against a connection the
    caller owns. Shared by list_feed and digest so digest does not open a
    second connection to rank the same feed.

    Pending Reading desk verdicts are imported first (a no-op unless
    ATTEST_DESK_STATE is configured): feed.list, feed.digest and feed.ask's
    reading routes all rank through here, so a verdict given on the page is a
    click before any of them orders anything. import_pending never raises.
    """
    desk.import_pending(conn)
    return rank_items(conn, get_embedder(), user_row["id"], since_days)[:limit]
```

Put `from attestation import desk` with the other top-level imports. `test_no_mcp_module_imports_a_private_domain_name` is satisfied (public name). `test_mcp_layer_sql_only_ratchets_down` is unaffected (no SQL added in `mcp/`).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_desk.py tests/test_architecture.py tests/test_ask_routing.py tests/test_response_size.py -q`
Expected: all pass.

- [ ] **Step 5: Mutation check**

Comment out `desk.import_pending(conn)`; run `uv run pytest tests/test_desk.py -k feed_ask -q`; expect FAIL. Restore.

- [ ] **Step 6: Commit**

```bash
git add src/attestation/mcp/_shared.py tests/test_desk.py
git commit -m "feed: import Reading desk verdicts before ranking

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The hourly refresh calls `attest desk refresh`

**Files:**
- Modify: `src/attestation/install.py` `_refresh_script_content` (after the tag block, before `refresh done`)
- Test: `tests/test_install.py`

**Interfaces:**
- Consumes: Task 3 `attest desk refresh` (exit 0 when unconfigured)
- Produces: the refresh script's fourth `uv run` call is `run attest desk refresh`; its failure never changes the exit status

- [ ] **Step 1: Update and add tests**

`test_refresh_script_runs_scan_when_research_root_is_set_and_exists` asserts the exact list of calls; append the new call:

```python
    assert lines == [
        "run attest runs scan",
        "run attest ingest",
        "run attest tag --limit 782",
        "run attest desk refresh",
    ]
```

Search `tests/test_install.py` for every other assertion over the full call list (`grep -n '"run attest tag' tests/test_install.py`) and extend each the same way.

New tests (use `_refresh_harness`):

```python
def test_refresh_script_runs_the_desk_after_tagging(tmp_path, monkeypatch):
    import subprocess

    body = '#!/bin/sh\necho "$*" >> {marker}\nexit 0\n'
    script, _, env, marker = _refresh_harness(tmp_path, body, monkeypatch)
    proc = subprocess.run([str(script)], env=env, capture_output=True, text=True, timeout=30)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = marker.read_text().splitlines()
    assert calls[-1] == "run attest desk refresh", calls
    assert "desk ok" in proc.stdout


def test_refresh_script_a_desk_failure_is_degraded_not_fatal(tmp_path, monkeypatch):
    import subprocess

    body = (
        "#!/bin/sh\n"
        'echo "$*" >> {marker}\n'
        'case "$*" in\n  *desk*) exit 5 ;;\n  *) exit 0 ;;\nesac\n'
    )
    script, _, env, marker = _refresh_harness(tmp_path, body, monkeypatch)
    proc = subprocess.run([str(script)], env=env, capture_output=True, text=True, timeout=30)

    assert proc.returncode == 0, "a desk failure must not turn the refresh red"
    assert "desk FAILED (exit 5)" in proc.stdout


def test_refresh_script_exit_status_stays_ingests_when_desk_also_fails(tmp_path, monkeypatch):
    import subprocess

    body = (
        "#!/bin/sh\n"
        'echo "$*" >> {marker}\n'
        'case "$*" in\n  *ingest*) exit 3 ;;\n  *desk*) exit 5 ;;\n  *) exit 0 ;;\nesac\n'
    )
    script, _, env, marker = _refresh_harness(tmp_path, body, monkeypatch)
    proc = subprocess.run([str(script)], env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_install.py -k refresh -q`
Expected: the three new tests and the extended list assertion FAIL.

- [ ] **Step 3: Implement**

In `_refresh_script_content`, between the tag block's closing `"fi\n"` + `"\n"` and `f'echo "[$(date -Iseconds)] refresh done"\n'`, insert:

```python
        # The Reading desk (desk.py). `attest desk refresh` reads its own
        # configuration from the checkout .env and exits 0 when there is none,
        # so this line is inert on a machine without a desk. A failure is
        # degraded like tagging: the page is a view of the feed, and a stale
        # page must not turn a successful ingest red.
        f"if uv run {CLI_NAME} desk refresh >/dev/null; then\n"
        '  echo "[$(date -Iseconds)] desk ok"\n'
        "else\n"
        '  echo "[$(date -Iseconds)] desk FAILED (exit $?) -- will retry next run"\n'
        "fi\n"
        "\n"
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_install.py tests/test_install_e2e.py -q`
Expected: all pass. If `test_install_e2e.py` diffs the generated script against a golden copy, regenerate that golden as the test's own docstring instructs.

- [ ] **Step 5: Commit**

```bash
git add src/attestation/install.py tests/test_install.py
git commit -m "refresh: rebuild the Reading desk after tagging, degraded on failure

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Docs, full gate, PR

**Files:**
- Modify: `CLAUDE.md` (Key API Patterns: one new `|Desk: ...` segment)
- Modify: `CHANGELOG.md` (`## [Unreleased]` → `### Added`)
- Modify: `docs/guides/feed.md` (a short "The Reading desk" section)

- [ ] **Step 1: CLAUDE.md** — add after the `|Feedback: ...` segment:

```
|Desk: desk.py renders the Reading desk (one self-contained HTML page, no request of its own, host CSP-safe) and imports its verdicts READ-ONLY from the page host's state.sqlite (`{"v":1,"verdicts":{"<item_id>":{"useful":bool,"at":iso}}}`) through record_click(source='ui') -- in mcp/_shared.ranked_items (so feed.list/digest/ask), `attest desk build|import|refresh`, and the hourly refresh after tagging (degraded, never fatal)|config is ATTEST_DESK_STATE/ATTEST_DESK_USER/ATTEST_DESK_PUBLISH in the checkout .env; unset = inert|import never raises and never autocreates a persona|verdicts import at the next RENDER, not on save: a verdict changes nothing until a ranking is computed
```

- [ ] **Step 2: CHANGELOG.md** under `## [Unreleased]`:

```markdown
### Added

- **The Reading desk.** `attest desk build|import|refresh` renders today's
  ranked papers as one self-contained page with Useful / Not my area on each,
  and records the verdicts a reader gives there as clicks before the next
  ranking. The ranker had no new human signal since 2026-08-22; this is a
  gesture surface on the reader's phone. See
  `docs/superpowers/specs/2026-09-29-reading-desk-design.md`.
```

- [ ] **Step 3: docs/guides/feed.md** — append:

```markdown
## The Reading desk

`attest desk build --user NAME --out desk.html` writes today's ranked papers
as one page with two tabs (Triage, Read). Hosted where a page can save state
(AgentMarkit's private pages), each paper gets Useful / Not my area buttons;
point `ATTEST_DESK_STATE` at that page's state file and every ranking imports
the verdicts first. `attest desk refresh` is the configured form the hourly
refresh runs: it imports, builds, and runs `ATTEST_DESK_PUBLISH` if set.
```

- [ ] **Step 4: Full gate, foreground**

Run: `uv run --frozen pre-commit run --all-files`
Expected: all hooks pass. If `test_claude_md_...` count tests complain, re-read their failure message; the desk adds no tool, no skill and no BLE001 site, so a count mismatch means something else changed.

- [ ] **Step 5: Commit, then PR (after 18:00 or on Matt's say-so)**

```bash
git add CLAUDE.md CHANGELOG.md docs/guides/feed.md
git commit -m "docs: the Reading desk

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin feat/reading-desk
gh pr create --title "Reading desk: a page of ranked papers whose verdicts come back as clicks" --body "$(cat <<'EOF'
Implements docs/superpowers/specs/2026-09-29-reading-desk-design.md.

- desk.py: read-only import of page verdicts through record_click(source='ui'); one self-contained HTML page
- `attest desk build|import|refresh`; the hourly refresh runs `desk refresh` after tagging (degraded, never fatal)
- mcp/_shared.ranked_items imports pending verdicts, so feed.list/digest/ask see them
- clip/safe_href moved to render.py so desk.py does not import FastAPI

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Then release as the repo does (bump version, `v`-prefixed tag — `release.yml` fires on `v*` only). The agentmarkit pin in Task 9 needs the release commit.

---

## Part B — agentmarkit (`/home/matt/qc/agentmarkit`, new branch `feat/reading-desk` off `main`)

Before starting: `git -C /home/matt/qc/agentmarkit status` must be clean apart from `.claude/` and `.playwright-mcp/`; `git checkout -b feat/reading-desk`.

### Task 7: Page links open any plain https URL

**Files:**
- Create: `agentmarkit-site/src/private-pages/links.mjs`
- Modify: `agentmarkit-site/src/private-pages/host.mjs:154` (use `externalLink`)
- Modify: `agentmarkit-site/scripts/build.mjs:418` (copy `links.mjs` with the other assets)
- Test: `agentmarkit-site/tests/private-pages-links.test.mjs`

**Interfaces:**
- Produces: `export function externalLink(href: string, adapter: 'page' | 'observatory'): string | null` — the normalized URL to show in the confirm modal, or `null` to ignore

- [ ] **Step 1: Write the failing test**

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { externalLink } from '../src/private-pages/links.mjs';

test('a generic page may open any plain https link, behind the modal', () => {
  for (const href of ['https://www.nature.com/articles/s41586-026-0001', 'https://arxiv.org/abs/2609.03501',
                      'https://doi.org/10.1126/science.abc', 'https://pubmed.ncbi.nlm.nih.gov/12345/?x=1#y'])
    assert.equal(externalLink(href, 'page'), new URL(href).href);
});

test('a generic page may not open anything else', () => {
  for (const href of ['http://www.nature.com/x', 'https://user:pw@x.org/', 'https://x.org:8443/', 'javascript:alert(1)',
                      'data:text/html,hi', 'file:///etc/passwd', 'not a url', 'https://localhost/x', 'https://127.0.0.1/'])
    assert.equal(externalLink(href, 'page'), null, href);
});

test('the Observatory keeps its LinkedIn-only rule', () => {
  assert.equal(externalLink('https://www.linkedin.com/in/someone/', 'observatory'), 'https://www.linkedin.com/in/someone/');
  assert.equal(externalLink('https://www.nature.com/articles/x', 'observatory'), null);
  assert.equal(externalLink('https://www.linkedin.com/in/someone/?a=1', 'observatory'), null);
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd agentmarkit-site && node --test tests/private-pages-links.test.mjs`
Expected: FAIL — cannot find module `links.mjs`.

- [ ] **Step 3: Implement `links.mjs`**

```js
// Which external links a private page may ask the host to open. The host
// always shows the full address in its confirm dialog before leaving; this
// decides only whether the dialog appears at all.
//
// The Observatory keeps its original rule (LinkedIn profiles only). A
// generic page -- the Research Desk's Reading desk links to arbitrary
// journals -- may open any plain https URL: no credentials, no port, and not
// a loopback or bare-IP host, since those are either bearer access or a
// machine address that means nothing on the owner's phone.
const LINKEDIN = /^\/in\/[A-Za-z0-9_%.-]+\/?$/;

export function externalLink(href, adapter) {
  let u;
  try { u = new URL(href); } catch { return null; }
  if (u.protocol !== 'https:' || u.username || u.password || u.port) return null;
  if (adapter === 'observatory') {
    if (!['www.linkedin.com', 'linkedin.com'].includes(u.hostname) || !LINKEDIN.test(u.pathname) || u.search || u.hash) return null;
    return u.href;
  }
  if (adapter !== 'page') return null;
  if (u.hostname === 'localhost' || /^[\d.]+$/.test(u.hostname) || u.hostname.startsWith('[')) return null;
  return u.href;
}
```

- [ ] **Step 4: Use it in `host.mjs`**

Add `import { externalLink } from './links.mjs?v=dev';` beside the other imports (match `observatory.mjs?v=dev`). Replace line 154's `try { const u = new URL(m.href); if (...) return; ... } catch {} return;` with:

```js
    const target = externalLink(m.href, current.adapter); if (!target) return;
    $('pages-external-address').textContent = target; $('pages-external-link').href = target; if (!$('pages-external').open) $('pages-external').showModal();
    return;
```

In `scripts/build.mjs:418` add `"links.mjs"` to the copied asset list. Check `scripts/asset-version.mjs` for how `?v=dev` is rewritten at build time and whether a new module needs registering there too (read it; follow what `observatory.mjs` gets).

- [ ] **Step 5: Run the page tests**

Run: `cd agentmarkit-site && node --test tests/private-pages*.test.mjs tests/private-pages-links.test.mjs`
Expected: all pass (the existing resume/observatory tests prove the LinkedIn path unchanged).

- [ ] **Step 6: Mutation check**

Change `u.protocol !== 'https:'` to `!u.protocol.startsWith('http')`; run the links test; expect FAIL on `http://`. Revert.

- [ ] **Step 7: Commit**

```bash
git add agentmarkit-site/src/private-pages/links.mjs agentmarkit-site/src/private-pages/host.mjs agentmarkit-site/scripts/build.mjs agentmarkit-site/tests/private-pages-links.test.mjs
git commit -m "Private pages: a generic page may open any plain https link, behind the confirm dialog

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Wire the Research Desk distribution

**Files:**
- Modify: `hermes-starters/distributions/academic-research-assistant/skills/research-operator/scripts/install_attestation.py` (`_feed_env`)
- Modify: `hermes-starters/distributions/academic-research-assistant/SOUL.md` (steps 3-4)
- Modify: `hermes-starters/distributions/academic-research-assistant/activation.json`
- Modify: `hermes-starters/distributions/academic-research-assistant/skills/research-operator/SKILL.md` (brief saving)
- Modify: `hermes-starters/distributions/academic-research-assistant/README.md` (first useful path)
- Test: `hermes-starters/lab/tests/test_install_attestation_safety.py`, `hermes-starters/lab/tests/test_distributions.py`

**Interfaces:**
- Consumes: attestation's `ATTEST_DESK_STATE`, `ATTEST_DESK_USER`, `ATTEST_DESK_PUBLISH`; `private_pages.py register --id --title --html`
- Produces: `.env` carries the three keys with absolute paths

- [ ] **Step 1: Failing test** — extend `FeedFirstTests.test_install_points_embeddings_locally_and_ingests_once` after the `ATTEST_FEEDS` assertion:

```python
            home = adapter._hermes_home_parent()
            self.assertEqual(
                str(home / ".hermes/private-pages/reading-desk/state.sqlite"), env["ATTEST_DESK_STATE"])
            self.assertEqual("owner", env["ATTEST_DESK_USER"])
            publish = env["ATTEST_DESK_PUBLISH"]
            self.assertIn(str(home / ".hermes/skills/agentmarkit-links/scripts/private_pages.py"), publish)
            self.assertIn('--id reading-desk --title "Reading desk" --html', publish)
            self.assertIn(str(home / ".hermes/workspace/research-desk/desk.html"), publish)
            self.assertNotIn("~", publish, "paths are absolute; cron has no shell to expand ~")
```

Run: `cd hermes-starters && python3 -m pytest lab/tests/test_install_attestation_safety.py -k feed_first -q` (use whatever runner `lab/AGENTS.md` names). Expected: FAIL (KeyError).

- [ ] **Step 2: Implement** — `_feed_env`:

```python
def _feed_env() -> dict[str, str]:
    # ATTEST_FEEDS names the promoted copy even while this runs from the
    # install payload: the payload is an attempt directory that does not
    # outlive the install, and the first ingest below passes its file itself.
    #
    # The ATTEST_DESK_* keys configure attestation's Reading desk: the private
    # page's state file (verdicts the owner gives on their phone), the persona
    # those verdicts belong to, and the command that re-registers the page after
    # each hourly build. Absolute paths: the refresh runs under cron.
    home = _hermes_home_parent() / ".hermes"
    desk_html = home / "workspace/research-desk/desk.html"
    pages = home / "skills/agentmarkit-links/scripts/private_pages.py"
    return {
        "EMBED_BASE_URL": EMBED_BASE_URL,
        "EMBED_MODEL": EMBED_MODEL,
        "ATTEST_FEEDS": str(PROMOTED_FEEDS),
        "ATTEST_DESK_STATE": str(home / "private-pages/reading-desk/state.sqlite"),
        "ATTEST_DESK_USER": "owner",
        "ATTEST_DESK_PUBLISH": (
            f'python3 {pages} register --id reading-desk --title "Reading desk" --html {desk_html}'
        ),
    }
```

Note `step_feed_first` only writes MISSING keys, so an existing install gains the three keys on its next run without touching the others. Confirm by reading lines 404-446.

- [ ] **Step 3: SOUL.md** — replace steps 3 and 4 of the first-conversation list with:

```markdown
3. Ask the same tool "What should I read today?". Show every paper as one
   line: its linked title, then source and topics. Say the caveat if there is
   one. Then run in the terminal
   `cd ~/.agent37-pilots/sources/attestation && env -u HERMES_MANAGED uv run attest desk refresh`
   and send the `pages` link from `agentmarkit.json` as a Markdown link: "Your
   Reading desk: the same list, with Useful / Not my area on each paper."
4. Ask which look useful and which are not their area -- here or on the
   Reading desk; both teach the ranking. Pass each verdict given here with
   that paper's `item_id` from `refs`: `question="useful"` or `"not useful"`.
```

And add under "Your research tools":

```markdown
- A research brief: fill `templates/research-brief.md`, save it as
  `~/.hermes/workspace/research-desk/briefs/YYYY-MM-DD-<short-slug>.md`, and
  send the `files` link. Never paste a file path.
```

- [ ] **Step 4: activation.json** — set:

```json
    "artifact": "The Reading desk page: today's papers ranked for this researcher, each a linked title with source and topics and a Useful / Not my area choice",
```

and in `done_when` insert after "every paper links to its source":

```json
      "the Reading desk link was sent",
```

Change the last `done_when` entry to `"at least one paper the reader judged, in chat or on the Reading desk, was recorded as useful or not useful"`. Add `"saved research briefs"` to `customize_after` replacing `"research briefs with evidence tables"`.

- [ ] **Step 5: research-operator/SKILL.md** — in "First conversation", after "Produce a first answer using `templates/research-brief.md`", add:

```markdown
Save every brief you produce as
`~/.hermes/workspace/research-desk/briefs/YYYY-MM-DD-<short-slug>.md` (create
the folder if needed) and send the `files` link from `agentmarkit.json`. A
brief the researcher cannot find again is a brief they did not get.
```

- [ ] **Step 6: README.md** — replace "First useful path" with:

```markdown
## First useful path

1. Install the profile and Attestation at the pinned revision (the adapter
   points embeddings at the machine's CPU embedder, seeds the journals, runs a
   first ingest, and configures the Reading desk in the checkout `.env`).
2. Ask what the researcher works on; show today's ranked papers.
3. Build and send the Reading desk page; take verdicts in chat or on the page.
4. Only then offer the hourly refresh, which also rebuilds the page.
```

- [ ] **Step 7: Run the distribution tests**

Run: `cd hermes-starters && python3 -m pytest lab/tests/test_install_attestation_safety.py lab/tests/test_distributions.py -q`
Expected: pass. `test_distributions.py` validates `activation.json` shape and may assert SOUL.md contents; fix wording it pins rather than loosening it.

- [ ] **Step 8: Commit**

```bash
git add hermes-starters/distributions/academic-research-assistant hermes-starters/lab/tests
git commit -m "Research Desk: the Reading desk page is the first win; briefs are saved

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Pin attestation and rehearse the loop

**Files:**
- Modify: every file `646e065` touched to pin v0.2.4 (`source-lock.yaml`, `lab/presets/research-attestation/preset.yaml`, `agentmarkit-site/src/worker/addons.js`, `agentmarkit-site/src/worker/starters.generated.json`, `agentmarkit-site/tests/addons.test.mjs`, `agentmarkit-site/docs/onboarding-acceptance.md`, `hermes-starters/lab/docs/01-attestation-runtime-map.md`)
- Modify: `hermes-starters/lab/scripts/rehearse_starter.sh` (a desk loop check, no model needed)

- [ ] **Step 1: Pin** — after attestation's release (Task 6), `git -C /home/matt/attestation rev-parse v<new>`; replace the old commit `3e949529cdd937e91b5b1c8655f9a16a6f229f50` and version string `0.2.4`/`v0.2.4` in exactly the files `git show 646e065 --stat` lists (grep each; do not blind-replace across the repo). Regenerate `starters.generated.json` with whatever script produced it (see `scripts/build.mjs`). Run `cd agentmarkit-site && node --test tests/addons.test.mjs`.

- [ ] **Step 2: Rehearsal check** — in `rehearse_starter.sh`, inside the `MACHINE` heredoc, after `. /rehearsal/smoke.sh` and before `if [ -n "$WITH_MODEL" ]`:

```bash
echo "== the Reading desk loop (page verdict -> attestation click)"
desk_loop() {
  cd /home/node/.agent37-pilots/sources/attestation || return 1
  set -a; . ./.env; set +a
  env -u HERMES_MANAGED uv run python - <<'PY' || return 1
from attestation.db import get_db, resolve_db_path
from attestation.rank import create_user, get_user
conn = get_db(resolve_db_path(None))
if get_user(conn, "owner") is None:
    create_user(conn, "owner", "machine learning")
PY
  env -u HERMES_MANAGED uv run attest desk refresh || return 1
  item=$(env -u HERMES_MANAGED uv run python -c 'from attestation.db import get_db, resolve_db_path; print(get_db(resolve_db_path(None)).execute("SELECT id FROM items ORDER BY id DESC LIMIT 1").fetchone()[0])') || return 1
  req=$(mktemp)
  printf '{"action":"save","page":"reading-desk","revision":0,"event":"rehearsal-0001","data":{"v":1,"verdicts":{"%s":{"useful":true,"at":"2026-09-29T00:00:00Z"}}}}' "$item" >"$req"
  python3 /home/node/.hermes/skills/agentmarkit-links/scripts/private_pages.py --request-file "$req" | grep -q '"ok": true' || return 1
  env -u HERMES_MANAGED uv run attest desk import --user owner | grep -q "recorded 1"
}
step "a page verdict becomes a ui click" desk_loop
```

Caveats to check while doing this (read, don't assume): `private_pages.py register` requires `~/.hermes/agentmarkit.json` with a valid `pages_url`/`agent_id` — the rehearsal's `apply.sh` may not write one. If it does not, write a minimal one in `desk_loop` before `attest desk refresh`:
`{"agent_id":"rehearsal","pages_url":"https://agentmarkit.com/pages/?agent=rehearsal"}` (agent_id must match `[A-Za-z0-9_-]{4,64}`). Also confirm the `--request-file` payload keys against `handle()` — `page`, `action`, `revision`, `event`, `data`.

- [ ] **Step 3: Run the rehearsal**

Build the template image first if absent: `cd hermes-starters && docker build -t agentmarkit-template:rehearsal -f lab/template/Dockerfile .`
Run: `ATTESTATION_SRC=/home/matt/attestation lab/scripts/rehearse_starter.sh research-assistant`
Expected: `ok  a page verdict becomes a ui click`, `== 0 step(s) failed`. Say which attestation commit it ran.

- [ ] **Step 4: Commit and PR**

```bash
git add -A hermes-starters/lab/scripts/rehearse_starter.sh <pinned files>
git commit -m "Pin attestation v<new>; rehearse the Reading desk loop

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

PR against `mzumby/agentmarkit` `main` after 18:00 or on Matt's say-so, body ending with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
