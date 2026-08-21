# Feature Extraction for Ranking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LLM-tag every feed item (topic tags + content type) in a post-ingest pass, and blend per-(user, tag/type/source) click-preference scores into ranking so downvoted kinds of items sink from click one.

**Architecture:** A new `src/hermes/features.py` owns the tagging pass (one Ollama chat call per untagged item, pydantic-validated, vocabulary-controlled) and the preference-score math (Laplace-smoothed like/dislike ratio per feature key). `rank.py` gains one new click-driven rank term. `ingest.py` is not modified. Spec: `docs/superpowers/specs/2026-08-05-feature-extraction-design.md`.

**Tech Stack:** Python 3.12, sqlite3 (schema in `db.py`, idempotent `CREATE TABLE IF NOT EXISTS`), pydantic v2, numpy, Ollama via the existing `hermes.explain.ollama_chat(messages, schema)` helper, pytest with fake `chat_fn` / `FakeEmbedder` (see `tests/conftest.py`).

## Global Constraints

- **`ingest.py` must not be modified** — its "deterministic, no LLM" contract stays true.
- **No LLM call anywhere on the rank path** (`rank_items` and everything it calls).
- Content types are exactly: `paper`, `survey`, `announcement`, `release`, `blog`, `other`.
- Tags: 1–4 per item, matching `^[a-z0-9][a-z0-9-]{0,31}$`.
- Preference score: `(u + 1) / (u + n + 2)`; 0.5 is neutral.
- Blend: `final = (1-w)·profile_rank + w·mean(available click-driven ranks)`, `w = blend_weight(n_clicks)` unchanged.
- LLM output validation: one retry, then skip (item stays untagged; next run retries).
- Ruff line length 100 (`uv run ruff check .` must stay clean); tests run with `uv run pytest`.
- Commit after each task; commit messages in the repo's existing `feat:`/`test:`/`docs:` style.

---

### Task 1: Schema — `item_features` and `item_tags` tables

**Files:**
- Modify: `src/hermes/db.py` (append to the `SCHEMA` string, after the `explanations` table)
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: existing `get_db()` which runs `SCHEMA` idempotently on every connect.
- Produces: tables `item_features(item_id PK → items.id, content_type TEXT NOT NULL, model TEXT NOT NULL, tagged_at TEXT NOT NULL DEFAULT now)` and `item_tags(item_id, tag, PK(item_id, tag))`. All later tasks rely on these exact names and columns.

- [ ] **Step 1: Write the failing test** (append to `tests/test_db.py`):

```python
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
```

(If `tests/test_db.py` imports differ, match its existing imports — it already imports `get_db`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py::test_feature_tables_exist_and_accept_rows -v`
Expected: FAIL with `sqlite3.OperationalError: no such table: item_features`

- [ ] **Step 3: Implement** — append to the `SCHEMA` string in `src/hermes/db.py` (inside the triple-quoted string, after the `explanations` table):

```sql
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py -v`
Expected: PASS (all — existing db tests must not regress; the new tables are additive)

- [ ] **Step 5: Commit**

```bash
git add src/hermes/db.py tests/test_db.py
git commit -m "feat: item_features + item_tags tables for feature extraction"
```

---

### Task 2: `features.py` — validated LLM tagging of a single item

**Files:**
- Create: `src/hermes/features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Consumes: `hermes.explain.ollama_chat(messages: list[dict], schema: dict) -> dict` (called via injected `chat_fn`, same pattern as `explain.py`); `hermes.explain.CHAT_MODEL`; Task 1 tables.
- Produces:
  - `CONTENT_TYPES: tuple[str, ...]` — the six content types.
  - `ItemTags(BaseModel)` with `content_type: Literal[...]` and `tags: list[str]` (1–4, regex-validated).
  - `tag_vocabulary(conn, limit: int = 40) -> list[str]` — most-used tags, most frequent first.
  - `tag_one_item(conn, item: sqlite3.Row, chat_fn, vocab: list[str], model_name: str) -> bool` — True = tagged and committed; False = failed after one retry, nothing written.

- [ ] **Step 1: Write the failing tests** — create `tests/test_features.py`:

```python
import pytest
from pydantic import ValidationError

from hermes.db import get_db
from hermes.features import ItemTags, tag_one_item, tag_vocabulary


def add_item(conn, title, summary="s", days_ago=0):
    cur = conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, published, content_hash)"
        " VALUES (NULL, ?, 'http://x', ?, datetime('now', ?), ?)",
        (title, summary, f"-{days_ago} days", f"hash-{title}"),
    )
    return cur.lastrowid


def get_item_row(conn, item_id):
    return conn.execute("SELECT id, title, summary FROM items WHERE id = ?", (item_id,)).fetchone()


def good_chat_fn(messages, schema):
    return {"content_type": "paper", "tags": ["quantum-chemistry", "dft"]}


def test_itemtags_validates_good_output():
    parsed = ItemTags.model_validate({"content_type": "paper", "tags": ["a-tag", "b2"]})
    assert parsed.content_type == "paper"
    assert parsed.tags == ["a-tag", "b2"]


@pytest.mark.parametrize(
    "bad",
    [
        {"content_type": "poem", "tags": ["ok"]},  # bad enum
        {"content_type": "paper", "tags": []},  # too few tags
        {"content_type": "paper", "tags": ["a", "b", "c", "d", "e"]},  # too many
        {"content_type": "paper", "tags": ["Bad Tag!"]},  # bad charset
        {"content_type": "paper", "tags": ["x" * 40]},  # too long
    ],
)
def test_itemtags_rejects_bad_output(bad):
    with pytest.raises(ValidationError):
        ItemTags.model_validate(bad)


def test_tag_one_item_writes_features_and_tags(tmp_path):
    conn = get_db(tmp_path / "t.db")
    item_id = add_item(conn, "DFT paper")
    ok = tag_one_item(conn, get_item_row(conn, item_id), good_chat_fn, [], "testmodel")
    assert ok is True
    feat = conn.execute(
        "SELECT content_type, model FROM item_features WHERE item_id = ?", (item_id,)
    ).fetchone()
    assert feat["content_type"] == "paper"
    assert feat["model"] == "testmodel"
    tags = {
        r["tag"] for r in conn.execute("SELECT tag FROM item_tags WHERE item_id = ?", (item_id,))
    }
    assert tags == {"quantum-chemistry", "dft"}


def test_tag_one_item_retries_once_then_succeeds(tmp_path):
    conn = get_db(tmp_path / "t.db")
    item_id = add_item(conn, "flaky")
    calls = {"n": 0}

    def flaky_chat_fn(messages, schema):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("garbled JSON")
        return {"content_type": "blog", "tags": ["mlops"]}

    assert tag_one_item(conn, get_item_row(conn, item_id), flaky_chat_fn, [], "m") is True
    assert calls["n"] == 2


def test_tag_one_item_gives_up_after_retry_writes_nothing(tmp_path):
    conn = get_db(tmp_path / "t.db")
    item_id = add_item(conn, "hopeless")

    def bad_chat_fn(messages, schema):
        return {"content_type": "paper", "tags": ["INVALID TAG"]}

    assert tag_one_item(conn, get_item_row(conn, item_id), bad_chat_fn, [], "m") is False
    assert (
        conn.execute("SELECT 1 FROM item_features WHERE item_id = ?", (item_id,)).fetchone() is None
    )
    assert conn.execute("SELECT 1 FROM item_tags WHERE item_id = ?", (item_id,)).fetchone() is None


def test_vocab_appears_in_prompt(tmp_path):
    conn = get_db(tmp_path / "t.db")
    item_id = add_item(conn, "vocab check")
    seen = {}

    def spy_chat_fn(messages, schema):
        seen["prompt"] = "\n".join(m["content"] for m in messages)
        return {"content_type": "paper", "tags": ["dft"]}

    tag_one_item(conn, get_item_row(conn, item_id), spy_chat_fn, ["dft", "llm-eval"], "m")
    assert "dft" in seen["prompt"] and "llm-eval" in seen["prompt"]


def test_tag_vocabulary_orders_by_use(tmp_path):
    conn = get_db(tmp_path / "t.db")
    ids = [add_item(conn, f"i{n}") for n in range(3)]
    for i in ids:
        conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, 'common')", (i,))
    conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, 'rare')", (ids[0],))
    assert tag_vocabulary(conn) == ["common", "rare"]
    assert tag_vocabulary(conn, limit=1) == ["common"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hermes.features'`

- [ ] **Step 3: Implement** — create `src/hermes/features.py`:

```python
"""Feature extraction: LLM tagging pass + per-key click-preference scores.

Reliability contract mirrors explain.py: tagging is lazy relative to ingest,
validated, and skips (never blocks) on failure. Nothing in this module runs
on the rank path except pure-SQL/numpy preference scoring.
"""

import logging
import sqlite3
from typing import Literal

from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

CONTENT_TYPES = ("paper", "survey", "announcement", "release", "blog", "other")
TAG_PATTERN = r"^[a-z0-9][a-z0-9-]{0,31}$"


class ItemTags(BaseModel):
    content_type: Literal["paper", "survey", "announcement", "release", "blog", "other"]
    tags: list[str] = Field(min_length=1, max_length=4)

    @field_validator("tags")
    @classmethod
    def _tags_shape(cls, v: list[str]) -> list[str]:
        import re

        for t in v:
            if not re.match(TAG_PATTERN, t):
                raise ValueError(f"bad tag: {t!r}")
        return v


def tag_vocabulary(conn: sqlite3.Connection, limit: int = 40) -> list[str]:
    return [
        r["tag"]
        for r in conn.execute(
            "SELECT tag FROM item_tags GROUP BY tag ORDER BY COUNT(*) DESC, tag LIMIT ?",
            (limit,),
        )
    ]


def _tag_prompt(item: sqlite3.Row, vocab: list[str]) -> list[dict]:
    vocab_line = ", ".join(vocab) if vocab else "(none yet)"
    return [
        {
            "role": "system",
            "content": (
                "You label science-feed items. Reply with JSON: content_type"
                " (one of paper, survey, announcement, release, blog, other)"
                " and tags: 1-4 short lowercase-hyphenated topic tags."
                " Strongly prefer tags from the existing vocabulary; invent a"
                " new tag only if nothing in it fits."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Existing vocabulary: {vocab_line}\n\n"
                f"Title: {item['title']}\nSummary: {item['summary'][:1000]}"
            ),
        },
    ]


def tag_one_item(conn, item: sqlite3.Row, chat_fn, vocab: list[str], model_name: str) -> bool:
    """One LLM call (plus one retry) -> item_features + item_tags rows. False = skipped."""
    parsed = None
    for _ in range(2):  # one retry, per spec
        try:
            out = chat_fn(_tag_prompt(item, vocab), ItemTags.model_json_schema())
            parsed = ItemTags.model_validate(out)
            break
        except Exception:
            log.debug("tagging attempt failed for item %s", item["id"], exc_info=True)
    if parsed is None:
        return False
    conn.execute(
        "INSERT OR REPLACE INTO item_features(item_id, content_type, model) VALUES (?, ?, ?)",
        (item["id"], parsed.content_type, model_name),
    )
    conn.execute("DELETE FROM item_tags WHERE item_id = ?", (item["id"],))
    conn.executemany(
        "INSERT INTO item_tags(item_id, tag) VALUES (?, ?)",
        [(item["id"], t) for t in dict.fromkeys(parsed.tags)],
    )
    conn.commit()
    return True
```

Note `dict.fromkeys(parsed.tags)` — dedups while preserving order, so a model
that repeats a tag doesn't trip the `item_tags` primary key.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features.py -v && uv run ruff check .`
Expected: PASS, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/hermes/features.py tests/test_features.py
git commit -m "feat: validated LLM tagging of single items (features.py)"
```

---

### Task 3: `features.py` — the `run_tagging` pass

**Files:**
- Modify: `src/hermes/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: Task 2's `tag_one_item` / `tag_vocabulary`; `hermes.explain.ollama_chat` and `hermes.explain.CHAT_MODEL` (imported lazily inside the function so tests never touch Ollama).
- Produces: `run_tagging(conn, chat_fn=None, limit: int | None = None) -> dict` returning `{"tagged": int, "failed": int}`. Failed items stay untagged (retried next run). Untagged selection is newest-`published`-first.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_features.py`):

```python
from hermes.features import run_tagging


def test_run_tagging_tags_all_untagged_then_is_idempotent(tmp_path):
    conn = get_db(tmp_path / "t.db")
    for n in range(3):
        add_item(conn, f"item {n}")
    stats = run_tagging(conn, chat_fn=good_chat_fn)
    assert stats == {"tagged": 3, "failed": 0}
    assert run_tagging(conn, chat_fn=good_chat_fn) == {"tagged": 0, "failed": 0}


def test_run_tagging_newest_first_and_limit(tmp_path):
    conn = get_db(tmp_path / "t.db")
    old = add_item(conn, "old", days_ago=5)
    new = add_item(conn, "new", days_ago=0)
    stats = run_tagging(conn, chat_fn=good_chat_fn, limit=1)
    assert stats["tagged"] == 1
    assert conn.execute("SELECT 1 FROM item_features WHERE item_id = ?", (new,)).fetchone()
    assert conn.execute("SELECT 1 FROM item_features WHERE item_id = ?", (old,)).fetchone() is None


def test_run_tagging_counts_failures_and_continues(tmp_path):
    conn = get_db(tmp_path / "t.db")
    add_item(conn, "will-fail")
    add_item(conn, "will-succeed")

    def chat_fn(messages, schema):
        if "will-fail" in messages[1]["content"]:
            raise ValueError("ollama down for this one")
        return {"content_type": "paper", "tags": ["dft"]}

    stats = run_tagging(conn, chat_fn=chat_fn)
    assert stats == {"tagged": 1, "failed": 1}


def test_new_tags_enter_vocabulary_within_a_run(tmp_path):
    conn = get_db(tmp_path / "t.db")
    add_item(conn, "first", days_ago=0)
    add_item(conn, "second", days_ago=1)
    prompts = []

    def chat_fn(messages, schema):
        prompts.append(messages[1]["content"])
        return {"content_type": "paper", "tags": ["fresh-tag"]}

    run_tagging(conn, chat_fn=chat_fn)
    assert "fresh-tag" not in prompts[0]  # vocab empty on first item
    assert "fresh-tag" in prompts[1]  # second item sees the new tag
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features.py -v`
Expected: new tests FAIL with `ImportError: cannot import name 'run_tagging'`

- [ ] **Step 3: Implement** (append to `src/hermes/features.py`):

```python
def run_tagging(conn, chat_fn=None, limit: int | None = None) -> dict:
    """Tag every untagged item, newest first. Returns {"tagged": n, "failed": n}.

    Failed items are skipped (stay untagged) and retried on the next run.
    """
    if chat_fn is None:
        from hermes.explain import ollama_chat

        chat_fn = ollama_chat
    from hermes.explain import CHAT_MODEL

    sql = (
        "SELECT i.id, i.title, i.summary FROM items i"
        " LEFT JOIN item_features f ON f.item_id = i.id"
        " WHERE f.item_id IS NULL ORDER BY i.published DESC"
    )
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    rows = conn.execute(sql, params).fetchall()

    stats = {"tagged": 0, "failed": 0}
    for item in rows:
        vocab = tag_vocabulary(conn)  # re-read so tags minted this run join the vocabulary
        if tag_one_item(conn, item, chat_fn, vocab, CHAT_MODEL):
            stats["tagged"] += 1
        else:
            stats["failed"] += 1
    return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features.py -v && uv run ruff check .`
Expected: PASS, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/hermes/features.py tests/test_features.py
git commit -m "feat: run_tagging pass — untagged items newest-first, skip-on-failure"
```

---

### Task 4: CLI — `hermes tag` subcommand + README

**Files:**
- Modify: `src/hermes/cli.py` (parser in `build_parser`, dispatch in `main`)
- Modify: `README.md` (Commands section + cron line)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 3's `run_tagging(conn, chat_fn=None, limit=None) -> {"tagged", "failed"}`.
- Produces: `hermes tag [--db PATH] [--limit N]`. Exit 0 normally; exit 1 only when `tagged == 0 and failed > 0` (total failure — meaningful cron noise only).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_cli.py`):

```python
def test_parser_tag_subcommand():
    args = build_parser().parse_args(["tag", "--limit", "5"])
    assert args.command == "tag"
    assert args.limit == 5


def test_tag_command_prints_stats(tmp_path, capsys, monkeypatch):
    import hermes.features

    db = tmp_path / "t.db"
    get_db(db).close()
    monkeypatch.setattr(
        hermes.features, "run_tagging", lambda conn, limit=None: {"tagged": 0, "failed": 0}
    )
    rc = main(["tag", "--db", str(db)])
    assert rc == 0
    assert "tagged" in capsys.readouterr().out


def test_tag_command_exit_1_on_total_failure(tmp_path, monkeypatch):
    import hermes.features

    db = tmp_path / "t.db"
    get_db(db).close()
    monkeypatch.setattr(
        hermes.features, "run_tagging", lambda conn, limit=None: {"tagged": 0, "failed": 3}
    )
    assert main(["tag", "--db", str(db)]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: new tests FAIL (`tag` is not a valid subcommand → SystemExit)

- [ ] **Step 3: Implement.** In `build_parser()` (after the `ingest` parser block):

```python
    sp = sub.add_parser("tag", help="LLM-tag untagged items (topic tags + content type)")
    add_db(sp)
    sp.add_argument("--limit", type=int, default=None, help="max items to tag this run")
```

In `main()` (after the `ingest` dispatch block). Note the module-attribute
call style (`hermes.features.run_tagging`, not `from ... import run_tagging`)
— it is what lets the tests monkeypatch it:

```python
    if args.command == "tag":
        import hermes.features

        stats = hermes.features.run_tagging(get_db(resolve_db_path(args.db)), limit=args.limit)
        print(stats)
        return 1 if (stats["tagged"] == 0 and stats["failed"] > 0) else 0
```

Also update the module docstring on line 1 to `"""hermes CLI: ingest | tag | serve | eval | warmup | bootstrap-persona."""`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v && uv run ruff check .`
Expected: PASS, ruff clean

- [ ] **Step 5: Update README.** In the Commands section add after the ingest line:

```
    uv run hermes tag [--limit N]           # LLM-tag untagged items (topics + content type)
```

And change the cron line (in "Keep the feed fresh (cron)") to:

```
17 * * * * cd ~/hermes-rss && uv run hermes ingest && uv run hermes tag
```

- [ ] **Step 6: Commit**

```bash
git add src/hermes/cli.py tests/test_cli.py README.md
git commit -m "feat: hermes tag CLI subcommand; chain into cron after ingest"
```

---

### Task 5: `features.py` — preference scores

**Files:**
- Modify: `src/hermes/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: Task 1 tables; `clicks` and `items` tables.
- Produces: `pref_scores_for_items(conn, user_id: int, item_ids: list[int]) -> np.ndarray` — float array aligned with `item_ids`, each value in (0, 1), exactly 0.5 for items with no feature keys. Item keys are `tag:<tag>` for each tag, `type:<content_type>`, and `source:<feed_id>` (skipped when `feed_id` is NULL). Per-key score `(u+1)/(u+n+2)`; item score = mean over its keys (keys with no click data contribute 0.5).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_features.py`; note `add_item` in this file inserts `feed_id=NULL`, so tag/type keys are exercised without source keys, and one test adds a real feed row for the source key):

```python
import numpy as np

from hermes.features import pref_scores_for_items


def _tag(conn, item_id, content_type, tags):
    conn.execute(
        "INSERT OR REPLACE INTO item_features(item_id, content_type, model) VALUES (?, ?, 'm')",
        (item_id, content_type),
    )
    for t in tags:
        conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (item_id, t))


def _click(conn, user_id, item_id, useful):
    conn.execute(
        "INSERT INTO clicks(user_id, item_id, useful) VALUES (?, ?, ?)", (user_id, item_id, useful)
    )


def _matt(conn):
    return conn.execute("SELECT id FROM users WHERE name = 'matt'").fetchone()["id"]


def test_pref_neutral_with_no_data(tmp_path):
    conn = get_db(tmp_path / "t.db")
    a = add_item(conn, "a")  # untagged, no feed -> no keys at all
    b = add_item(conn, "b")
    _tag(conn, b, "paper", ["dft"])  # tagged but user has no clicks
    scores = pref_scores_for_items(conn, _matt(conn), [a, b])
    assert scores[0] == 0.5
    assert scores[1] == 0.5  # keys exist but all have (0,0) stats -> 0.5


def test_downvoted_tag_scores_below_neutral(tmp_path):
    conn = get_db(tmp_path / "t.db")
    uid = _matt(conn)
    bad1, bad2, candidate, control = (add_item(conn, t) for t in ("bad1", "bad2", "cand", "ctrl"))
    for i in (bad1, bad2, candidate):
        _tag(conn, i, "announcement", ["llm-benchmarks"])
    _tag(conn, control, "paper", ["dft"])
    _click(conn, uid, bad1, useful=0)
    _click(conn, uid, bad2, useful=0)
    scores = pref_scores_for_items(conn, uid, [candidate, control])
    # candidate shares tag AND content type with two downvotes: well below neutral
    assert scores[0] < 0.4
    assert scores[1] == 0.5  # control untouched by those clicks


def test_upvotes_score_above_neutral_and_mix_averages(tmp_path):
    conn = get_db(tmp_path / "t.db")
    uid = _matt(conn)
    liked, candidate = add_item(conn, "liked"), add_item(conn, "cand")
    _tag(conn, liked, "paper", ["dft"])
    _tag(conn, candidate, "paper", ["dft"])
    _click(conn, uid, liked, useful=1)
    assert pref_scores_for_items(conn, uid, [candidate])[0] > 0.5


def test_source_key_used_when_feed_present(tmp_path):
    conn = get_db(tmp_path / "t.db")
    uid = _matt(conn)
    conn.execute("INSERT INTO feeds(id, url, title) VALUES (7, 'http://f', 'Feed7')")

    def add_fed(title):
        return conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (7, ?, 'http://x', 's', ?)",
            (title, f"h-{title}"),
        ).lastrowid

    clicked, candidate = add_fed("clicked"), add_fed("cand")
    _click(conn, uid, clicked, useful=0)  # downvote something from feed 7
    # candidate is UNTAGGED but shares the source -> sinks below neutral anyway
    assert pref_scores_for_items(conn, uid, [candidate])[0] < 0.5


def test_scores_are_per_user(tmp_path):
    conn = get_db(tmp_path / "t.db")
    uid = _matt(conn)
    other = conn.execute("SELECT id FROM users WHERE name = 'ml-engineer'").fetchone()["id"]
    a, b = add_item(conn, "a"), add_item(conn, "b")
    _tag(conn, a, "paper", ["dft"])
    _tag(conn, b, "paper", ["dft"])
    _click(conn, uid, a, useful=0)
    assert pref_scores_for_items(conn, other, [b])[0] == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features.py -v`
Expected: new tests FAIL with `ImportError: cannot import name 'pref_scores_for_items'`

- [ ] **Step 3: Implement** (append to `src/hermes/features.py`; add `import numpy as np` to the module imports):

```python
def _key_stats(conn, user_id: int) -> dict[str, list[int]]:
    """Per-feature-key [useful_count, not_useful_count] over this user's clicks."""
    rows = conn.execute(
        "SELECT 'tag:' || t.tag AS key, c.useful FROM clicks c"
        " JOIN item_tags t ON t.item_id = c.item_id WHERE c.user_id = ?"
        " UNION ALL"
        " SELECT 'type:' || f.content_type AS key, c.useful FROM clicks c"
        " JOIN item_features f ON f.item_id = c.item_id WHERE c.user_id = ?"
        " UNION ALL"
        " SELECT 'source:' || i.feed_id AS key, c.useful FROM clicks c"
        " JOIN items i ON i.id = c.item_id WHERE c.user_id = ? AND i.feed_id IS NOT NULL",
        (user_id, user_id, user_id),
    ).fetchall()
    stats: dict[str, list[int]] = {}
    for r in rows:
        u_n = stats.setdefault(r["key"], [0, 0])
        u_n[0 if r["useful"] else 1] += 1
    return stats


def _score(stats: dict[str, list[int]], key: str) -> float:
    u, n = stats.get(key, (0, 0))
    return (u + 1) / (u + n + 2)  # Laplace-smoothed; 0.5 = neutral


def _item_keys(conn, item_ids: list[int]) -> dict[int, list[str]]:
    keys: dict[int, list[str]] = {i: [] for i in item_ids}
    qmarks = ",".join("?" * len(item_ids))
    for r in conn.execute(
        f"SELECT item_id, 'tag:' || tag AS key FROM item_tags WHERE item_id IN ({qmarks})",
        item_ids,
    ):
        keys[r["item_id"]].append(r["key"])
    for r in conn.execute(
        f"SELECT item_id, 'type:' || content_type AS key FROM item_features"
        f" WHERE item_id IN ({qmarks})",
        item_ids,
    ):
        keys[r["item_id"]].append(r["key"])
    for r in conn.execute(
        f"SELECT id, 'source:' || feed_id AS key FROM items"
        f" WHERE id IN ({qmarks}) AND feed_id IS NOT NULL",
        item_ids,
    ):
        keys[r["id"]].append(r["key"])
    return keys


def pref_scores_for_items(conn, user_id: int, item_ids: list[int]) -> np.ndarray:
    """Mean per-key preference score per item, aligned with item_ids. 0.5 = neutral."""
    out = np.full(len(item_ids), 0.5, dtype=np.float64)
    if not item_ids:
        return out
    stats = _key_stats(conn, user_id)
    keys = _item_keys(conn, item_ids)
    for idx, iid in enumerate(item_ids):
        ks = keys[iid]
        if ks:
            out[idx] = float(np.mean([_score(stats, k) for k in ks]))
    return out
```

(The `IN (...)` lists run up to ~1,500 candidate ids — fine under SQLite's
default 32,766 variable limit on the bundled 3.45+.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features.py -v && uv run ruff check .`
Expected: PASS, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/hermes/features.py tests/test_features.py
git commit -m "feat: per-key click-preference scores (tag/type/source, Laplace-smoothed)"
```

---

### Task 6: `rank.py` — blend `pref_rank` into `rank_items`

**Files:**
- Modify: `src/hermes/rank.py` (`rank_items` only)
- Test: `tests/test_rank.py`

**Interfaces:**
- Consumes: Task 5's `pref_scores_for_items(conn, user_id, item_ids) -> np.ndarray` (higher = preferred).
- Produces: `rank_items` signature and return type unchanged. New blend:
  `final = (1-w)·profile_rank + w·mean(click_ranks)` where `click_ranks` holds `ranks(probs)` when the classifier is active, and `ranks(pref)` when the user has ≥1 click **and** the pref array carries signal (`max > min`). An all-neutral pref array (no feature key has click data) is "no signal" per the spec and must be skipped — otherwise `ranks()` of a constant array would inject an arbitrary tie-break permutation into the blend, a regression for users with clicks but an untagged corpus. No clicks → pure profile rank, exactly as today.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_rank.py`; `add_item` and `seed_corpus` helpers already exist at the top of the file):

```python
def _tag_item(conn, item_id, content_type, tags):
    conn.execute(
        "INSERT OR REPLACE INTO item_features(item_id, content_type, model) VALUES (?, ?, 'm')",
        (item_id, content_type),
    )
    for t in tags:
        conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (item_id, t))


def test_downvoted_tag_sinks_similar_item_even_single_class(tmp_path, fake_embedder):
    """Covers two spec behaviors: downvoted-tag demotion, and only-downvotes users
    (single-class history disables the classifier but NOT the pref term)."""
    conn = get_db(tmp_path / "t.db")
    ids = seed_corpus(conn, fake_embedder, n=40)
    user = get_user(conn, "matt")
    for i in ids[:3]:  # three items share a tag+type; the third is the survivor
        _tag_item(conn, i, "announcement", ["junk"])
    downvoted = ids[:2] + ids[3:21]  # 20 downvotes, ALL useful=0 -> single class
    for i in downvoted:
        conn.execute(
            "INSERT INTO clicks(user_id, item_id, useful) VALUES (?, ?, 0)", (user["id"], i)
        )
    # classifier is off (single class) -- any influence below is the pref term
    assert classifier_probs(conn, user["id"], np.zeros((1, 256), dtype=np.float32)) is None
    result = rank_items(conn, fake_embedder, user["id"])
    assert len(result) == 20  # 40 seeded - 20 clicked
    pos = {r.item_id: i for i, r in enumerate(result)}
    # Deterministic bound, independent of tie-breaking and profile order:
    # w = blend_weight(20) = 0.8; survivor uniquely holds the worst pref rank (19),
    # so every item with pref rank <= 14 ranks above it for ANY profile order
    # (0.8*14 + 0.2*19 = 15.0 < 15.2 = 0.8*19 + 0.2*0). That pins the survivor
    # to the bottom quarter.
    assert pos[ids[2]] >= 15


def test_clicks_without_feature_data_leave_profile_order_intact(tmp_path, fake_embedder):
    """Pref term must not inject tie-break noise when no feature key has click data."""
    conn = get_db(tmp_path / "t.db")
    ids = seed_corpus(conn, fake_embedder)
    user = get_user(conn, "matt")
    conn.execute(
        "INSERT INTO clicks(user_id, item_id, useful) VALUES (?, ?, 0)", (user["id"], ids[0])
    )
    with_click = [r.item_id for r in rank_items(conn, fake_embedder, user["id"])]
    conn.execute("DELETE FROM clicks WHERE user_id = ?", (user["id"],))
    baseline = [
        r.item_id for r in rank_items(conn, fake_embedder, user["id"]) if r.item_id != ids[0]
    ]
    assert with_click == baseline  # identical order: pref term contributed nothing


def test_no_clicks_ranking_unchanged_by_tags(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    ids = seed_corpus(conn, fake_embedder)
    user = get_user(conn, "matt")
    baseline = [r.item_id for r in rank_items(conn, fake_embedder, user["id"])]
    _tag_item(conn, ids[0], "paper", ["dft"])
    assert [r.item_id for r in rank_items(conn, fake_embedder, user["id"])] == baseline
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rank.py -v`
Expected: the two new behavior tests FAIL (ranking ignores tags today); `test_no_clicks_ranking_unchanged_by_tags` may already pass — that's fine, it's a regression guard.

- [ ] **Step 3: Implement.** In `src/hermes/rank.py`, add the import at the top with the other imports:

```python
from hermes.features import pref_scores_for_items
```

Replace the blend block in `rank_items` (currently `probs = classifier_probs(...)` through `final = w * ranks(probs) + (1 - w) * profile_rank`) with:

```python
probs = classifier_probs(conn, user_id, X)
n_clicks = conn.execute("SELECT COUNT(*) c FROM clicks WHERE user_id = ?", (user_id,)).fetchone()[
    "c"
]

click_ranks = []
if probs is not None:
    click_ranks.append(ranks(probs))
if n_clicks > 0:
    pref = pref_scores_for_items(conn, user_id, [r["id"] for r in rows])
    if float(pref.max() - pref.min()) > 0:  # all-neutral = no signal; skip, don't add tie noise
        click_ranks.append(ranks(pref))

if not click_ranks:
    final = profile_rank.astype(np.float64)
else:
    w = blend_weight(n_clicks)
    final = w * np.mean(click_ranks, axis=0) + (1 - w) * profile_rank
```

Also update the module docstring on line 1 to:
`"""Per-user ranking: profile cosine + click-trained classifier + feature-preference term, blended by rank."""`

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest && uv run ruff check .`
Expected: ALL PASS (existing rank tests — cold start, single-class guard, clicked-excluded, recency — must not regress; note the single-class-guard test asserts `rank_items` doesn't crash, and now exercises the new pref path too), ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/hermes/rank.py tests/test_rank.py
git commit -m "feat: blend feature-preference rank term into rank_items"
```

---

### Task 7: Surface tags in web UI and MCP `list_feed`

**Files:**
- Modify: `src/hermes/rank.py` (`RankedItem` + attach features in `rank_items`)
- Modify: `src/hermes/server.py` (FRAGMENT template + CSS in PAGE)
- Modify: `src/hermes/mcp_server.py` (`_list_feed_impl` item dict + `list_feed` docstring)
- Test: `tests/test_rank.py`, `tests/test_server.py`, `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: Task 1 tables; Task 6's `rank_items`.
- Produces: `RankedItem` gains `tags: list[str]` (default `[]`) and `content_type: str | None` (default `None`), populated by `rank_items`. MCP `list_feed` items gain `"tags"` and `"content_type"` keys.

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_rank.py` (uses `_tag_item` from Task 6):

```python
def test_ranked_items_carry_tags_and_content_type(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    ids = seed_corpus(conn, fake_embedder)
    user = get_user(conn, "matt")
    _tag_item(conn, ids[0], "paper", ["dft", "catalysis"])
    result = rank_items(conn, fake_embedder, user["id"])
    by_id = {r.item_id: r for r in result}
    assert by_id[ids[0]].content_type == "paper"
    assert by_id[ids[0]].tags == ["catalysis", "dft"]  # alphabetical
    assert by_id[ids[1]].content_type is None
    assert by_id[ids[1]].tags == []
```

In `tests/test_mcp_server.py`, update the exact-keys assertion in
`TestListFeed.test_returns_items_with_required_keys` to:

```python
assert set(item.keys()) == {"item_id", "title", "url", "source", "score", "tags", "content_type"}
```

Append to `tests/test_server.py` (self-contained — the file's shared `client` fixture doesn't expose the DB path, so build the app inline the same way that fixture does):

```python
def test_list_renders_tag_badges(tmp_path, fake_embedder):
    db_path = tmp_path / "badge.db"
    conn = get_db(db_path)
    cur = conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (NULL, 'tagged item', 'http://x', 's', 'h-badge')"
    )
    item_id = cur.lastrowid
    vec = fake_embedder.embed_document("tagged item", "s")
    conn.execute(
        "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)", (item_id, vec.tobytes())
    )
    conn.execute(
        "INSERT INTO item_features(item_id, content_type, model) VALUES (?, 'paper', 'm')",
        (item_id,),
    )
    conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, 'dft')", (item_id,))
    conn.commit()
    conn.close()
    app = create_app(db_path, embedder=fake_embedder, chat_fn=lambda m, s: {"text": "why"})
    resp = TestClient(app).get("/list", params={"user": "matt"})
    assert resp.status_code == 200
    assert '<span class="tag type">paper</span>' in resp.text
    assert '<span class="tag">dft</span>' in resp.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rank.py tests/test_server.py tests/test_mcp_server.py -v`
Expected: new/updated tests FAIL (`RankedItem` has no `tags` field; MCP items lack the new keys; no badges in HTML)

- [ ] **Step 3: Implement.**

In `src/hermes/rank.py`, extend `RankedItem`:

```python
class RankedItem(BaseModel):
    item_id: int
    title: str
    url: str | None
    source: str | None
    score: float  # blended rank; lower = better
    explanation: str | None = None
    tags: list[str] = []
    content_type: str | None = None
```

In `rank_items`, before the `return`, fetch features for the candidates and pass them into the constructor:

```python
    ids = [r["id"] for r in rows]
    qmarks = ",".join("?" * len(ids))
    ctype = {
        r["item_id"]: r["content_type"]
        for r in conn.execute(
            f"SELECT item_id, content_type FROM item_features WHERE item_id IN ({qmarks})", ids
        )
    }
    tags_by: dict[int, list[str]] = {}
    for r in conn.execute(
        f"SELECT item_id, tag FROM item_tags WHERE item_id IN ({qmarks}) ORDER BY tag", ids
    ):
        tags_by.setdefault(r["item_id"], []).append(r["tag"])
```

and in the `RankedItem(...)` construction add:

```python
tags = (tags_by.get(rows[i]["id"], []),)
content_type = (ctype.get(rows[i]["id"]),)
```

In `src/hermes/server.py`, add to the `<style>` block in `PAGE`:

```css
 .tag { background: #eef; color: #446; font-size: .7rem; padding: 0 .35rem;
        border-radius: .5rem; margin-left: .25rem; }
 .tag.type { background: #efe; color: #464; }
```

and in `FRAGMENT`, after the `<span class="src">...</span>` line:

```html
  {% if it.content_type %}<span class="tag type">{{ it.content_type }}</span>{% endif %}
  {% for t in it.tags %}<span class="tag">{{ t }}</span>{% endfor %}
```

In `src/hermes/mcp_server.py`, extend the item dict in `_list_feed_impl`:

```python
                {
                    "item_id": it.item_id,
                    "title": it.title,
                    "url": it.url,
                    "source": it.source,
                    "score": it.score,
                    "tags": it.tags,
                    "content_type": it.content_type,
                }
```

and extend the `list_feed` tool docstring's first paragraph to mention the new
fields, e.g. append: `Each item also carries its LLM-extracted topic "tags"
and "content_type" (paper/survey/announcement/release/blog/other) when the
tagging pass has processed it; both are empty/null for not-yet-tagged items.`

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest && uv run ruff check .`
Expected: ALL PASS, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/hermes/rank.py src/hermes/server.py src/hermes/mcp_server.py \
        tests/test_rank.py tests/test_server.py tests/test_mcp_server.py
git commit -m "feat: surface tags + content type in web UI and MCP list_feed"
```

---

## Post-plan verification (manual, once all tasks land)

Not part of any task's test cycle — a live smoke test against real Ollama:

1. `uv run hermes tag --limit 5` against the live DB; confirm 5 rows in `item_features`, sane tags in `item_tags`.
2. Downvote two same-tag items in the web UI; confirm a third same-tag item visibly sinks on the next render.
3. `hermes mcp test hermes-rss` (hermes-agent side) and one chat turn calling `list_feed`; confirm `tags`/`content_type` fields appear.
4. Kick off the backfill (`uv run hermes tag`, no limit) and let it churn.
