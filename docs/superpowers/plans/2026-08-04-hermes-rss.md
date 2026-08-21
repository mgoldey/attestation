# hermes-rss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Personalized RSS ranking: ingest feeds, rank per-user via profile embedding + click-trained logistic regression, explain rankings with a local Hermes 3 LLM, serve in a one-page web UI.

**Architecture:** Three layers with distinct reliability contracts — deterministic ingest+ranking (no LLM, never blocks), a lazy cached LangGraph "explain" agent (swappable Ollama chat model, degrades to no-explanation), and a FastAPI/htmx page that renders instantly and streams explanations in. Single SQLite file (WAL) holds users, feeds, items, vectors (sqlite-vec), clicks, and cached explanations.

**Tech Stack:** Python 3.12, uv, sqlite-vec, feedparser, scikit-learn, numpy, LangGraph, pydantic v2, FastAPI + htmx, Ollama (embeddinggemma for 256-dim embeddings; chat model via HERMES_CHAT_MODEL, default gemma4:12b).

## Global Constraints

- Python `>=3.12`; dependency manager is `uv` (run tests with `uv run pytest`, add nothing outside `pyproject.toml`).
- Embeddings: embeddinggemma via Ollama `/api/embed`, truncated 768→256 (Matryoshka) and **re-normalized after truncation**. Document prompt: `"title: {title} | text: {text}"` (title falls back to `"none"`); query prompt: `"task: search result | query: {text}"`.
- **"Hermes" names the orchestrator** (the `hermes` CLI + LangGraph graphs), not the LLM. Chat model is configurable: `HERMES_CHAT_MODEL` env var, default `gemma4:12b`; called via Ollama `/api/chat`, `"options": {"num_ctx": 8192}`, `"keep_alive": -1`, `"stream": false`, structured output via `"format": <json schema>`.
- SQLite: `PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=5000`, `PRAGMA foreign_keys=ON`. One writer connection inside the web app.
- Ranking must never call the chat model. Explanations are lazy, top-20 only, cached in the `explanations` table, never invalidated by re-ranking alone.
- Classifier: `LogisticRegression(class_weight="balanced", C=0.1, max_iter=1000)`; **never fit when the user's click labels contain fewer than 2 classes** — fall back to profile similarity.
- Blend by **rank**, not raw score: `final = w * classifier_rank + (1 - w) * profile_rank`, `w = n_clicks / (n_clicks + 5)`, lower final = better.
- Ranked feed covers items published in the last 14 days, excluding items the user already clicked.
- No live-network tests; Ollama is always stubbed in tests.
- Commits: conventional prefixes (`feat:`, `test:`, `docs:`), each task ends committed.

---

### Task 1: Project scaffold + database module

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/hermes/__init__.py`, `src/hermes/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Produces: `hermes.db.get_db(path: str | Path) -> sqlite3.Connection` — opens/creates the DB with WAL + busy_timeout + foreign_keys, loads sqlite-vec, creates all tables, seeds three users. Rows come back as `sqlite3.Row`.
- Produces: seeded users named `matt`, `bench-chemist`, `ml-engineer` (each with non-empty `interests` text).
- Tables (used by all later tasks): `users(id, name, interests)`, `feeds(id, url, title, last_fetched)`, `items(id, feed_id, guid, title, url, summary, published, content_hash)` with `UNIQUE(feed_id, guid)`, `item_vectors` vec0 virtual table `embedding float[256]` keyed by item id as rowid, `clicks(id, user_id, item_id, useful, clicked_at)` with `UNIQUE(user_id, item_id)`, `explanations(user_id, item_id, text, created_at)` PK `(user_id, item_id)`.

- [ ] **Step 1: Write pyproject and gitignore**

```toml
# pyproject.toml
[project]
name = "hermes-rss"
version = "0.1.0"
description = "Personalized RSS ranking agent (local Hermes 3 + embeddinggemma)"
requires-python = ">=3.12"
dependencies = [
    "feedparser>=6.0",
    "sqlite-vec>=0.1.6",
    "httpx>=0.27",
    "scikit-learn>=1.5",
    "numpy>=1.26",
    "langgraph>=0.2",
    "pydantic>=2.7",
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "jinja2>=3.1",
]

[project.scripts]
hermes = "hermes.cli:main"

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/hermes"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

```gitignore
# .gitignore
.venv/
__pycache__/
*.db
*.db-wal
*.db-shm
.ruff_cache/
.pytest_cache/
```

Create `src/hermes/__init__.py` (empty file).

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_db.py
import sqlite3

from hermes.db import get_db


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hermes.db'` (after `uv sync` completes).

- [ ] **Step 4: Implement db.py**

```python
# src/hermes/db.py
"""Single-file SQLite store: relational tables + sqlite-vec vectors."""

import sqlite3
from pathlib import Path

import sqlite_vec

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
  UNIQUE(user_id, item_id)
);
CREATE TABLE IF NOT EXISTS explanations(
  user_id INTEGER NOT NULL,
  item_id INTEGER NOT NULL,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (user_id, item_id)
);
"""

VEC_SCHEMA = "CREATE VIRTUAL TABLE IF NOT EXISTS item_vectors USING vec0(embedding float[256])"

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


def get_db(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.execute(VEC_SCHEMA)
    for name, interests in SEED_USERS.items():
        conn.execute(
            "INSERT OR IGNORE INTO users(name, interests) VALUES (?, ?)", (name, interests)
        )
    conn.commit()
    return conn
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv sync && uv run pytest tests/test_db.py -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "feat: project scaffold and sqlite-vec database module"
```

---

### Task 2: Embedding client

**Files:**
- Create: `src/hermes/embed.py`
- Test: `tests/test_embed.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `hermes.embed.OllamaEmbedder(base_url="http://localhost:11434", model="embeddinggemma", dims=256, client: httpx.Client | None = None)` with methods `embed_document(title: str, text: str) -> np.ndarray` and `embed_query(text: str) -> np.ndarray`, both returning unit-norm `float32` arrays of shape `(256,)`.
- Produces: `hermes.embed.truncate_normalize(vec: np.ndarray, dims: int = 256) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_embed.py
import json

import httpx
import numpy as np

from hermes.embed import OllamaEmbedder, truncate_normalize


def make_embedder(captured):
    """Embedder wired to a mock transport returning a fixed 768-dim vector."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[1.0] * 768]})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    return OllamaEmbedder(client=client)


def test_truncate_normalize_renormalizes():
    vec = np.ones(768, dtype=np.float32)
    out = truncate_normalize(vec, dims=256)
    assert out.shape == (256,)
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_truncate_normalize_zero_vector_safe():
    out = truncate_normalize(np.zeros(768, dtype=np.float32))
    assert out.shape == (256,)
    assert not np.any(np.isnan(out))


def test_embed_document_prompt_format():
    captured = []
    emb = make_embedder(captured)
    vec = emb.embed_document("My Title", "body text")
    assert captured[0]["input"] == "title: My Title | text: body text"
    assert captured[0]["model"] == "embeddinggemma"
    assert vec.shape == (256,) and vec.dtype == np.float32


def test_embed_document_missing_title_uses_none():
    captured = []
    make_embedder(captured).embed_document("", "body")
    assert captured[0]["input"] == "title: none | text: body"


def test_embed_query_prompt_format():
    captured = []
    make_embedder(captured).embed_query("chemistry papers")
    assert captured[0]["input"] == "task: search result | query: chemistry papers"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hermes.embed'`

- [ ] **Step 3: Implement embed.py**

```python
# src/hermes/embed.py
"""Ollama embeddinggemma client. 768-dim output truncated to 256 (Matryoshka)."""

import httpx
import numpy as np

DOC_PROMPT = "title: {title} | text: {text}"
QUERY_PROMPT = "task: search result | query: {text}"


def truncate_normalize(vec: np.ndarray, dims: int = 256) -> np.ndarray:
    v = vec[:dims].astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "embeddinggemma",
        dims: int = 256,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.dims = dims
        self.client = client or httpx.Client(base_url=base_url, timeout=60)

    def _embed(self, prompt: str) -> np.ndarray:
        resp = self.client.post(
            "/api/embed",
            json={"model": self.model, "input": prompt, "keep_alive": -1},
        )
        resp.raise_for_status()
        vec = np.asarray(resp.json()["embeddings"][0], dtype=np.float32)
        return truncate_normalize(vec, self.dims)

    def embed_document(self, title: str, text: str) -> np.ndarray:
        return self._embed(DOC_PROMPT.format(title=title or "none", text=text))

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(QUERY_PROMPT.format(text=text))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embed.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/hermes/embed.py tests/test_embed.py
git commit -m "feat: embeddinggemma client with truncate+renormalize and task prefixes"
```

---

### Task 3: Ingest pipeline

**Files:**
- Create: `src/hermes/ingest.py`, `feeds.toml`, `tests/fixtures/arxiv.xml`, `tests/fixtures/blog.xml`
- Create: `tests/conftest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `get_db` (Task 1), `OllamaEmbedder` (Task 2 — tests use a `FakeEmbedder`).
- Produces: `hermes.ingest.run_ingest(conn, embedder, feeds_path: str | Path, parse=feedparser.parse) -> dict` returning `{"added": int, "skipped": int, "failed_feeds": int}`. `parse` is injectable for tests: called with each feed URL, must return a feedparser-style object with `.entries`.
- Produces: `hermes.ingest.strip_boilerplate(text: str) -> str`, `hermes.ingest.content_hash(title: str, summary: str) -> str`, `hermes.ingest.sync_feeds(conn, feeds_path) -> None`.
- Produces: `tests/conftest.py` fixture `fake_embedder` — deterministic embedder whose vector depends on input text (used by Tasks 4–6 tests too):

- [ ] **Step 1: Write conftest and fixtures**

```python
# tests/conftest.py
import hashlib

import numpy as np
import pytest


class FakeEmbedder:
    """Deterministic stand-in for OllamaEmbedder: text -> stable unit vector."""

    dims = 256

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(256).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_document(self, title: str, text: str) -> np.ndarray:
        return self._vec(f"doc:{title}:{text}")

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(f"query:{text}")


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
```

```xml
<!-- tests/fixtures/arxiv.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>cs.LG updates on arXiv.org</title>
  <item>
    <title>Paper One: Preference Learning at Scale</title>
    <link>https://arxiv.org/abs/2608.00001</link>
    <guid isPermaLink="false">oai:arXiv.org:2608.00001v1</guid>
    <pubDate>Mon, 03 Aug 2026 00:00:00 GMT</pubDate>
    <description>arXiv:2608.00001v1 Announce Type: new
Abstract: We study preference learning for ranking tasks.</description>
  </item>
  <item>
    <title>Paper Two: Vector Search Tricks</title>
    <link>https://arxiv.org/abs/2608.00002</link>
    <guid isPermaLink="false">oai:arXiv.org:2608.00002v1</guid>
    <pubDate>Mon, 03 Aug 2026 00:00:00 GMT</pubDate>
    <description>arXiv:2608.00002v1 Announce Type: new
Abstract: Approximate nearest neighbor methods reviewed.</description>
  </item>
</channel></rss>
```

```xml
<!-- tests/fixtures/blog.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Example ML Blog</title>
  <item>
    <title>Serving LLMs on old GPUs</title>
    <link>https://blog.example.com/old-gpus</link>
    <pubDate>Sun, 02 Aug 2026 12:00:00 GMT</pubDate>
    <description>&lt;p&gt;How to squeeze an 8B model onto a GTX 1080.&lt;/p&gt;</description>
  </item>
</channel></rss>
```

```toml
# feeds.toml — curated DIVERSE feeds (legible persona contrast per spec)
[[feeds]]
url = "https://rss.arxiv.org/rss/physics.chem-ph"
title = "arXiv chem-ph"

[[feeds]]
url = "https://rss.arxiv.org/rss/cs.LG"
title = "arXiv cs.LG"

[[feeds]]
url = "https://simonwillison.net/atom/everything/"
title = "Simon Willison"

[[feeds]]
url = "https://huggingface.co/blog/feed.xml"
title = "Hugging Face blog"

[[feeds]]
url = "https://www.quantamagazine.org/feed/"
title = "Quanta Magazine"

[[feeds]]
url = "https://feeds.arstechnica.com/arstechnica/science"
title = "Ars Technica science"

[[feeds]]
url = "https://hnrss.org/best"
title = "HN best"
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_ingest.py
from pathlib import Path

import feedparser

from hermes.db import get_db
from hermes.ingest import content_hash, run_ingest, strip_boilerplate, sync_feeds

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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hermes.ingest'`

- [ ] **Step 4: Implement ingest.py**

```python
# src/hermes/ingest.py
"""Deterministic feed ingest: fetch -> dedup -> clean -> embed -> store. No LLM."""

import hashlib
import logging
import re
import sqlite3
import time
import tomllib
from pathlib import Path

import feedparser

log = logging.getLogger(__name__)

ARXIV_RE = re.compile(r"arXiv:\S+\s+Announce Type:\s*\S+\s*Abstract:\s*", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")


def strip_boilerplate(text: str) -> str:
    text = TAG_RE.sub(" ", text or "")
    text = ARXIV_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def content_hash(title: str, summary: str) -> str:
    return hashlib.sha256(f"{title}\n{summary}".encode()).hexdigest()


def sync_feeds(conn: sqlite3.Connection, feeds_path: str | Path) -> None:
    cfg = tomllib.loads(Path(feeds_path).read_text())
    for feed in cfg.get("feeds", []):
        conn.execute(
            "INSERT OR IGNORE INTO feeds(url, title) VALUES (?, ?)",
            (feed["url"], feed.get("title")),
        )
    conn.commit()


def _published_iso(entry) -> str | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    return time.strftime("%Y-%m-%dT%H:%M:%S", parsed) if parsed else None


def _exists(conn, feed_id: int, guid: str | None, chash: str) -> bool:
    if guid is not None:
        row = conn.execute(
            "SELECT 1 FROM items WHERE feed_id = ? AND guid = ?", (feed_id, guid)
        ).fetchone()
        if row:
            return True
    return (
        conn.execute("SELECT 1 FROM items WHERE content_hash = ?", (chash,)).fetchone() is not None
    )


def run_ingest(conn, embedder, feeds_path: str | Path, parse=feedparser.parse) -> dict:
    sync_feeds(conn, feeds_path)
    stats = {"added": 0, "skipped": 0, "failed_feeds": 0}
    for feed in conn.execute("SELECT * FROM feeds").fetchall():
        try:
            parsed = parse(feed["url"])
            for entry in parsed.entries:
                title = (entry.get("title") or "").strip()
                summary = strip_boilerplate(entry.get("summary", ""))
                guid = entry.get("id")
                chash = content_hash(title, summary)
                if _exists(conn, feed["id"], guid, chash):
                    stats["skipped"] += 1
                    continue
                vec = embedder.embed_document(title, summary)
                cur = conn.execute(
                    "INSERT INTO items(feed_id, guid, title, url, summary, published, content_hash)"
                    " VALUES (?, ?, ?, ?, ?, COALESCE(?, datetime('now')), ?)",
                    (
                        feed["id"],
                        guid,
                        title,
                        entry.get("link"),
                        summary,
                        _published_iso(entry),
                        chash,
                    ),
                )
                conn.execute(
                    "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
                    (cur.lastrowid, vec.tobytes()),
                )
                stats["added"] += 1
            conn.execute(
                "UPDATE feeds SET last_fetched = datetime('now') WHERE id = ?", (feed["id"],)
            )
            conn.commit()
        except Exception:
            log.exception("feed failed: %s", feed["url"])
            conn.rollback()
            stats["failed_feeds"] += 1
    return stats
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 7 PASS

- [ ] **Step 6: Commit**

```bash
git add src/hermes/ingest.py feeds.toml tests/conftest.py tests/fixtures tests/test_ingest.py
git commit -m "feat: deterministic feed ingest with guid+hash dedup and arXiv cleanup"
```

---

### Task 4: Ranking core

**Files:**
- Create: `src/hermes/rank.py`
- Test: `tests/test_rank.py`

**Interfaces:**
- Consumes: DB schema (Task 1), `FakeEmbedder` fixture (Task 3), `embed_query` (Task 2 signature).
- Produces: `hermes.rank.RankedItem` (pydantic): `item_id: int`, `title: str`, `url: str | None`, `source: str | None`, `score: float` (blended rank; lower = better), `explanation: str | None = None`.
- Produces: `hermes.rank.rank_items(conn, embedder, user_id: int, since_days: int = 14) -> list[RankedItem]` — best first, excludes items the user already clicked, never calls a chat model.
- Produces: `hermes.rank.blend_weight(n_clicks: int) -> float` (= `n/(n+5)`), `hermes.rank.ranks(scores: np.ndarray) -> np.ndarray` (rank 0 = highest score), `hermes.rank.classifier_probs(conn, user_id, X) -> np.ndarray | None` (None when <2 classes), `hermes.rank.evaluate_user(conn, user_id, n_holdout: int = 5) -> float | None`, `hermes.rank.bootstrap_persona(conn, embedder, user_name: str, k: int = 30) -> int` (pseudo-clicks written; returns count).
- Helper used by Task 6: `hermes.rank.get_user(conn, name: str) -> sqlite3.Row | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rank.py
import numpy as np

from hermes.db import get_db
from hermes.rank import (
    RankedItem,
    blend_weight,
    bootstrap_persona,
    classifier_probs,
    evaluate_user,
    get_user,
    rank_items,
    ranks,
)


def add_item(conn, embedder, title, days_ago=0):
    cur = conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, published, content_hash)"
        " VALUES (NULL, ?, 'http://x', ?, datetime('now', ?), ?)",
        (title, f"summary of {title}", f"-{days_ago} days", f"hash-{title}"),
    )
    vec = embedder.embed_document(title, f"summary of {title}")
    conn.execute(
        "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
        (cur.lastrowid, vec.tobytes()),
    )
    return cur.lastrowid


def seed_corpus(conn, embedder, n=20):
    return [add_item(conn, embedder, f"item {i}") for i in range(n)]


def test_blend_weight_ramp():
    assert blend_weight(0) == 0.0
    assert np.isclose(blend_weight(5), 0.5)
    assert blend_weight(20) > 0.75


def test_ranks_lower_is_better():
    r = ranks(np.array([0.1, 0.9, 0.5]))
    assert list(r) == [2, 0, 1]


def test_cold_start_no_clicks_uses_profile(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    seed_corpus(conn, fake_embedder)
    user = get_user(conn, "matt")
    result = rank_items(conn, fake_embedder, user["id"])
    assert len(result) == 20
    assert isinstance(result[0], RankedItem)
    assert result[0].score <= result[-1].score  # best (lowest) first


def test_classifier_guard_single_class(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    ids = seed_corpus(conn, fake_embedder)
    user = get_user(conn, "matt")
    for i in ids[:4]:  # four clicks, ALL positive -> one class
        conn.execute(
            "INSERT INTO clicks(user_id, item_id, useful) VALUES (?, ?, 1)", (user["id"], i)
        )
    X = np.stack([fake_embedder.embed_document(f"item {i}", "") for i in range(3)])
    assert classifier_probs(conn, user["id"], X) is None
    # rank_items must not crash on single-class history
    assert rank_items(conn, fake_embedder, user["id"])


def test_clicked_items_excluded(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    ids = seed_corpus(conn, fake_embedder)
    user = get_user(conn, "matt")
    conn.execute(
        "INSERT INTO clicks(user_id, item_id, useful) VALUES (?, ?, 1)", (user["id"], ids[0])
    )
    result = rank_items(conn, fake_embedder, user["id"])
    assert ids[0] not in [r.item_id for r in result]


def test_recency_window(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    add_item(conn, fake_embedder, "fresh", days_ago=1)
    add_item(conn, fake_embedder, "stale", days_ago=40)
    result = rank_items(conn, fake_embedder, get_user(conn, "matt")["id"])
    assert [r.title for r in result] == ["fresh"]


def test_clicks_shift_ranking(tmp_path, fake_embedder):
    """After mixed clicks, classifier blends in and changes the order."""
    conn = get_db(tmp_path / "t.db")
    ids = seed_corpus(conn, fake_embedder, n=30)
    user = get_user(conn, "matt")
    before = [r.item_id for r in rank_items(conn, fake_embedder, user["id"])]
    for i in ids[:5]:
        conn.execute(
            "INSERT INTO clicks(user_id, item_id, useful) VALUES (?, ?, 1)", (user["id"], i)
        )
    for i in ids[5:10]:
        conn.execute(
            "INSERT INTO clicks(user_id, item_id, useful) VALUES (?, ?, 0)", (user["id"], i)
        )
    after = [r.item_id for r in rank_items(conn, fake_embedder, user["id"])]
    remaining = [i for i in before if i not in ids[:10]]
    assert after != remaining  # order changed among unclicked items


def test_persona_ordering_differs(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    seed_corpus(conn, fake_embedder, n=15)
    chem = rank_items(conn, fake_embedder, get_user(conn, "bench-chemist")["id"])
    ml = rank_items(conn, fake_embedder, get_user(conn, "ml-engineer")["id"])
    assert [r.item_id for r in chem] != [r.item_id for r in ml]


def test_bootstrap_persona_writes_clicks(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    seed_corpus(conn, fake_embedder, n=40)
    n = bootstrap_persona(conn, fake_embedder, "bench-chemist", k=30)
    assert n == 30
    rows = conn.execute(
        "SELECT useful, COUNT(*) c FROM clicks GROUP BY useful ORDER BY useful"
    ).fetchall()
    assert [r["c"] for r in rows] == [15, 15]


def test_evaluate_user_insufficient_data(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    seed_corpus(conn, fake_embedder, n=5)
    assert evaluate_user(conn, get_user(conn, "matt")["id"]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hermes.rank'`

- [ ] **Step 3: Implement rank.py**

```python
# src/hermes/rank.py
"""Per-user ranking: profile cosine + click-trained logistic regression, blended by rank."""

import sqlite3

import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class RankedItem(BaseModel):
    item_id: int
    title: str
    url: str | None
    source: str | None
    score: float  # blended rank; lower = better
    explanation: str | None = None


def get_user(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchone()


def blend_weight(n_clicks: int) -> float:
    return n_clicks / (n_clicks + 5)


def ranks(scores: np.ndarray) -> np.ndarray:
    """Rank 0 = highest score."""
    order = np.argsort(-scores)
    out = np.empty(len(scores), dtype=np.int64)
    out[order] = np.arange(len(scores))
    return out


def _click_training_data(conn, user_id: int):
    rows = conn.execute(
        "SELECT c.useful, v.embedding FROM clicks c"
        " JOIN item_vectors v ON v.rowid = c.item_id WHERE c.user_id = ?",
        (user_id,),
    ).fetchall()
    if not rows:
        return None, None
    y = np.array([r["useful"] for r in rows])
    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    return X, y


def classifier_probs(conn, user_id: int, X: np.ndarray) -> np.ndarray | None:
    X_train, y = _click_training_data(conn, user_id)
    if y is None or len(set(y.tolist())) < 2:
        return None  # single-class guard: never let sklearn see one class
    clf = LogisticRegression(class_weight="balanced", C=0.1, max_iter=1000)
    clf.fit(X_train, y)
    return clf.predict_proba(X)[:, 1]


def _candidate_items(conn, user_id: int, since_days: int):
    return conn.execute(
        "SELECT i.id, i.title, i.url, f.title AS source, v.embedding"
        " FROM items i JOIN item_vectors v ON v.rowid = i.id"
        " LEFT JOIN feeds f ON f.id = i.feed_id"
        " WHERE i.published >= datetime('now', ?)"
        " AND i.id NOT IN (SELECT item_id FROM clicks WHERE user_id = ?)",
        (f"-{since_days} days", user_id),
    ).fetchall()


def rank_items(conn, embedder, user_id: int, since_days: int = 14) -> list[RankedItem]:
    rows = _candidate_items(conn, user_id, since_days)
    if not rows:
        return []
    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    profile_vec = embedder.embed_query(user["interests"] or user["name"])
    profile_rank = ranks(X @ profile_vec)

    probs = classifier_probs(conn, user_id, X)
    if probs is None:
        final = profile_rank.astype(np.float64)
    else:
        n_clicks = conn.execute(
            "SELECT COUNT(*) c FROM clicks WHERE user_id = ?", (user_id,)
        ).fetchone()["c"]
        w = blend_weight(n_clicks)
        final = w * ranks(probs) + (1 - w) * profile_rank

    order = np.argsort(final)
    return [
        RankedItem(
            item_id=rows[i]["id"],
            title=rows[i]["title"],
            url=rows[i]["url"],
            source=rows[i]["source"],
            score=float(final[i]),
        )
        for i in order
    ]


def bootstrap_persona(conn, embedder, user_name: str, k: int = 30) -> int:
    """Pseudo-clicks for a synthetic persona: top-k/2 by profile similarity -> useful,
    bottom-k/2 -> not useful. Optional demo garnish; persona switch works without it."""
    user = get_user(conn, user_name)
    rows = conn.execute(
        "SELECT i.id, v.embedding FROM items i JOIN item_vectors v ON v.rowid = i.id"
    ).fetchall()
    if len(rows) < k:
        k = len(rows) - (len(rows) % 2)
    if k == 0:
        return 0
    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    sims = X @ embedder.embed_query(user["interests"])
    order = np.argsort(-sims)
    half = k // 2
    chosen = [(rows[i]["id"], 1) for i in order[:half]]
    chosen += [(rows[i]["id"], 0) for i in order[-half:]]
    for item_id, useful in chosen:
        conn.execute(
            "INSERT OR IGNORE INTO clicks(user_id, item_id, useful) VALUES (?, ?, ?)",
            (user["id"], item_id, useful),
        )
    conn.commit()
    return len(chosen)


def evaluate_user(conn, user_id: int, n_holdout: int = 5) -> float | None:
    """Leave-last-N-out AUC. Honest noise at small n -- never present as evidence."""
    rows = conn.execute(
        "SELECT c.useful, v.embedding FROM clicks c"
        " JOIN item_vectors v ON v.rowid = c.item_id"
        " WHERE c.user_id = ? ORDER BY c.clicked_at, c.id",
        (user_id,),
    ).fetchall()
    if len(rows) < n_holdout + 5:
        return None
    y = np.array([r["useful"] for r in rows])
    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    X_train, y_train = X[:-n_holdout], y[:-n_holdout]
    X_test, y_test = X[-n_holdout:], y[-n_holdout:]
    if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
        return None
    clf = LogisticRegression(class_weight="balanced", C=0.1, max_iter=1000)
    clf.fit(X_train, y_train)
    return float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rank.py -v`
Expected: 10 PASS

- [ ] **Step 5: Commit**

```bash
git add src/hermes/rank.py tests/test_rank.py
git commit -m "feat: ranking core with single-class guard, rank blend, personas, eval"
```

---

### Task 5: Explain agent (LangGraph + Hermes)

**Files:**
- Create: `src/hermes/explain.py`
- Test: `tests/test_explain.py`

**Interfaces:**
- Consumes: DB schema (Task 1); `explanations` cache table.
- Produces: `hermes.explain.ollama_chat(messages: list[dict], schema: dict, base_url="http://localhost:11434", model=None) -> dict` (model defaults to `CHAT_MODEL` = `HERMES_CHAT_MODEL` env or `gemma4:12b`) — Ollama structured-output chat call (`num_ctx=8192`, `keep_alive=-1`, `stream=False`), returns parsed JSON dict.
- Produces: `hermes.explain.explain(conn, user_id: int, item_id: int, chat_fn=ollama_chat) -> str | None` — cache-first; on miss runs the LangGraph graph (profile-synthesis node → explanation node), validates output against a pydantic schema with **one retry then None**, caches non-None results. Never raises.
- Produces: `hermes.explain.Explanation` (pydantic): `text: str` (1–300 chars).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_explain.py
from hermes.db import get_db
from hermes.explain import explain


def setup_db(tmp_path):
    conn = get_db(tmp_path / "t.db")
    conn.execute(
        "INSERT INTO items(feed_id, title, summary, content_hash)"
        " VALUES (NULL, 'Attention Is Enough', 'a paper', 'h1')"
    )
    conn.execute("INSERT INTO clicks(user_id, item_id, useful) VALUES (1, 1, 1)")
    return conn


def good_chat(messages, schema):
    return {"text": "Because you clicked similar ranking papers."}


def test_explain_returns_and_caches(tmp_path):
    conn = setup_db(tmp_path)
    calls = []

    def counting_chat(messages, schema):
        calls.append(1)
        return good_chat(messages, schema)

    text = explain(conn, user_id=1, item_id=1, chat_fn=counting_chat)
    assert text == "Because you clicked similar ranking papers."
    n_first = len(calls)
    # second call: served from cache, no new LLM calls
    assert explain(conn, user_id=1, item_id=1, chat_fn=counting_chat) == text
    assert len(calls) == n_first
    row = conn.execute("SELECT text FROM explanations WHERE user_id=1 AND item_id=1").fetchone()
    assert row["text"] == text


def test_explain_retries_once_then_none(tmp_path):
    conn = setup_db(tmp_path)
    attempts = []

    def bad_chat(messages, schema):
        attempts.append(1)
        return {"wrong_key": 42}  # fails pydantic validation every time

    assert explain(conn, user_id=1, item_id=1, chat_fn=bad_chat) is None
    # profile node (1 try, falls back to interests) + explain node (2 tries)
    assert len(attempts) == 3
    assert conn.execute("SELECT COUNT(*) c FROM explanations").fetchone()["c"] == 0


def test_explain_never_raises_on_chat_exception(tmp_path):
    conn = setup_db(tmp_path)

    def dead_chat(messages, schema):
        raise ConnectionError("ollama down")

    assert explain(conn, user_id=1, item_id=1, chat_fn=dead_chat) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_explain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hermes.explain'`

- [ ] **Step 3: Implement explain.py**

```python
# src/hermes/explain.py
"""LangGraph explain agent: click history -> profile -> one-sentence 'why ranked here'.

Hermes is the orchestrator; the chat model is a swappable Ollama backend
(HERMES_CHAT_MODEL, default gemma4:12b).
Reliability contract: lazy, cached, degrades to None. Ranking never waits on this.
"""

import json
import logging
import os
import sqlite3

import httpx
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)

CHAT_MODEL = os.environ.get("HERMES_CHAT_MODEL", "gemma4:12b")


class Explanation(BaseModel):
    text: str = Field(min_length=1, max_length=300)


class ExplainState(BaseModel):
    user_id: int
    item_id: int
    profile: str = ""
    explanation: str | None = None


def ollama_chat(
    messages: list[dict],
    schema: dict,
    base_url: str = "http://localhost:11434",
    model: str | None = None,
) -> dict:
    model = model or CHAT_MODEL
    resp = httpx.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "keep_alive": -1,
            "options": {"num_ctx": 8192},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["message"]["content"])


def _build_graph(conn: sqlite3.Connection, chat_fn):
    def synthesize_profile(state: ExplainState) -> dict:
        titles = [
            r["title"]
            for r in conn.execute(
                "SELECT i.title FROM clicks c JOIN items i ON i.id = c.item_id"
                " WHERE c.user_id = ? AND c.useful = 1"
                " ORDER BY c.clicked_at DESC, c.id DESC LIMIT 20",
                (state.user_id,),
            )
        ]
        interests = conn.execute(
            "SELECT interests FROM users WHERE id = ?", (state.user_id,)
        ).fetchone()["interests"]
        if not titles:
            return {"profile": interests}
        try:
            out = chat_fn(
                [
                    {"role": "system", "content": "Summarize this reader in one sentence."},
                    {
                        "role": "user",
                        "content": "Recently useful titles:\n- " + "\n- ".join(titles),
                    },
                ],
                Explanation.model_json_schema(),
            )
            return {"profile": Explanation.model_validate(out).text}
        except Exception:
            log.warning("profile synthesis failed; using interests text")
            return {"profile": interests}

    def generate_explanation(state: ExplainState) -> dict:
        item = conn.execute(
            "SELECT title, summary FROM items WHERE id = ?", (state.item_id,)
        ).fetchone()
        messages = [
            {
                "role": "system",
                "content": (
                    "You explain feed rankings. One sentence, second person,"
                    " grounded ONLY in the reader profile given. No hedging."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Reader profile: {state.profile}\n"
                    f"Item: {item['title']}\n{item['summary'][:500]}\n"
                    "Why is this ranked here for this reader?"
                ),
            },
        ]
        for _ in range(2):  # one retry per spec
            try:
                out = chat_fn(messages, Explanation.model_json_schema())
                return {"explanation": Explanation.model_validate(out).text}
            except (ValidationError, Exception):
                continue
        return {"explanation": None}

    graph = StateGraph(ExplainState)
    graph.add_node("profile", synthesize_profile)
    graph.add_node("explain", generate_explanation)
    graph.set_entry_point("profile")
    graph.add_edge("profile", "explain")
    graph.add_edge("explain", END)
    return graph.compile()


def explain(conn, user_id: int, item_id: int, chat_fn=ollama_chat) -> str | None:
    cached = conn.execute(
        "SELECT text FROM explanations WHERE user_id = ? AND item_id = ?",
        (user_id, item_id),
    ).fetchone()
    if cached:
        return cached["text"]
    try:
        result = _build_graph(conn, chat_fn).invoke(ExplainState(user_id=user_id, item_id=item_id))
    except Exception:
        log.exception("explain graph failed")
        return None
    text = result.get("explanation")
    if text:
        conn.execute(
            "INSERT OR IGNORE INTO explanations(user_id, item_id, text) VALUES (?, ?, ?)",
            (user_id, item_id, text),
        )
        conn.commit()
    return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_explain.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/hermes/explain.py tests/test_explain.py
git commit -m "feat: LangGraph explain agent with schema validation, retry, cache"
```

---

### Task 6: Web UI (FastAPI + htmx)

**Files:**
- Create: `src/hermes/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `get_db` (Task 1), `OllamaEmbedder` (Task 2), `rank_items`/`get_user`/`RankedItem` (Task 4), `explain` (Task 5).
- Produces: `hermes.server.create_app(db_path, embedder=None, chat_fn=None) -> FastAPI`. `embedder` defaults to a real `OllamaEmbedder()`; `chat_fn` defaults to `ollama_chat`. Routes:
  - `GET /` → full HTML page (user buttons + list for `?user=<name>`, default `matt`)
  - `GET /list?user=<name>` → HTML fragment `<ol id="feed">…` of top 50 ranked items; the **first 20** rows include a lazy htmx explanation slot
  - `POST /clicks` (form: `user`, `item_id`, `useful` ∈ {0,1}) → records click, returns the re-ranked `/list` fragment
  - `GET /explanation?user=<name>&item_id=<id>` → plain-text explanation or empty string

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_server.py
import pytest
from fastapi.testclient import TestClient

from hermes.db import get_db
from hermes.server import create_app


@pytest.fixture
def client(tmp_path, fake_embedder):
    db_path = tmp_path / "t.db"
    conn = get_db(db_path)
    for i in range(25):
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'http://x', 's', ?)",
            (f"item {i}", f"h{i}"),
        )
        vec = fake_embedder.embed_document(f"item {i}", "s")
        conn.execute(
            "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, vec.tobytes()),
        )
    conn.commit()
    conn.close()
    app = create_app(db_path, embedder=fake_embedder, chat_fn=lambda m, s: {"text": "why"})
    return TestClient(app)


def test_index_renders_users_and_feed(client):
    html = client.get("/").text
    assert "bench-chemist" in html and "ml-engineer" in html and "matt" in html
    assert "item 0" in html


def test_list_fragment_per_user_differs(client):
    a = client.get("/list", params={"user": "bench-chemist"}).text
    b = client.get("/list", params={"user": "ml-engineer"}).text
    assert a != b


def test_click_rerenders_without_clicked_item(client):
    html = client.get("/list", params={"user": "matt"}).text
    first_id = html.split('data-item-id="')[1].split('"')[0]
    after = client.post("/clicks", data={"user": "matt", "item_id": first_id, "useful": "1"}).text
    assert f'data-item-id="{first_id}"' not in after


def test_first_click_all_one_class_no_500(client):
    """Regression for the single-class crash blocker."""
    for _ in range(3):
        html = client.get("/list", params={"user": "matt"}).text
        item_id = html.split('data-item-id="')[1].split('"')[0]
        resp = client.post("/clicks", data={"user": "matt", "item_id": item_id, "useful": "1"})
        assert resp.status_code == 200


def test_explanation_endpoint(client):
    html = client.get("/list", params={"user": "matt"}).text
    item_id = html.split('data-item-id="')[1].split('"')[0]
    resp = client.get("/explanation", params={"user": "matt", "item_id": item_id})
    assert resp.status_code == 200
    assert resp.text == "why"


def test_lazy_explanations_limited_to_top_20(client):
    html = client.get("/list", params={"user": "matt"}).text
    assert html.count('hx-get="/explanation') == 20  # 25 items seeded, only top 20 lazy-load
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hermes.server'`

- [ ] **Step 3: Implement server.py**

```python
# src/hermes/server.py
"""One-page FastAPI + htmx UI. List renders instantly; explanations stream in lazily."""

from pathlib import Path

from fastapi import FastAPI, Form, Query
from fastapi.responses import HTMLResponse, PlainTextResponse
from jinja2 import Template

from hermes.db import get_db
from hermes.explain import explain, ollama_chat
from hermes.rank import get_user, rank_items

EXPLAIN_LIMIT = 20
LIST_LIMIT = 50

PAGE = Template("""<!doctype html>
<html><head><title>hermes-rss</title>
<script src="https://unpkg.com/htmx.org@2.0.4"></script>
<style>
 body { font-family: system-ui; max-width: 52rem; margin: 2rem auto; }
 .user-btn { margin-right: .5rem; }  .active { font-weight: bold; }
 li { margin-bottom: .8rem; }  .src { color: #888; font-size: .8rem; }
 .why { color: #567; font-size: .85rem; font-style: italic; }
 button.yn { margin-left: .4rem; }
</style></head>
<body>
<h1>hermes-rss</h1>
<nav>
{% for u in users %}
 <a class="user-btn {{ 'active' if u == user else '' }}" href="/?user={{ u }}">{{ u }}</a>
{% endfor %}
</nav>
<div id="feed-wrap" hx-get="/list?user={{ user }}" hx-trigger="load" hx-swap="innerHTML">
Loading…</div>
</body></html>""")

FRAGMENT = Template("""<ol id="feed">
{% for it in items %}
 <li data-item-id="{{ it.item_id }}">
  <a href="{{ it.url or '#' }}">{{ it.title }}</a>
  <span class="src">{{ it.source or '' }} · rank {{ '%.1f' % it.score }}</span>
  <button class="yn" hx-post="/clicks" hx-vals='{"user":"{{ user }}","item_id":{{ it.item_id }},"useful":1}'
      hx-target="#feed" hx-swap="outerHTML">✓</button>
  <button class="yn" hx-post="/clicks" hx-vals='{"user":"{{ user }}","item_id":{{ it.item_id }},"useful":0}'
      hx-target="#feed" hx-swap="outerHTML">✗</button>
  {% if loop.index <= explain_limit %}
  <div class="why" hx-get="/explanation?user={{ user }}&item_id={{ it.item_id }}"
       hx-trigger="load delay:{{ loop.index }}s" hx-swap="innerHTML"></div>
  {% endif %}
 </li>
{% endfor %}
</ol>""")


def create_app(db_path: str | Path, embedder=None, chat_fn=None) -> FastAPI:
    if embedder is None:
        from hermes.embed import OllamaEmbedder

        embedder = OllamaEmbedder()
    chat_fn = chat_fn or ollama_chat
    app = FastAPI()
    conn = get_db(db_path)  # single writer connection for the whole app

    def render_list(user_name: str) -> str:
        user = get_user(conn, user_name)
        items = rank_items(conn, embedder, user["id"])[:LIST_LIMIT]
        return FRAGMENT.render(items=items, user=user_name, explain_limit=EXPLAIN_LIMIT)

    @app.get("/", response_class=HTMLResponse)
    def index(user: str = Query("matt")):
        users = [r["name"] for r in conn.execute("SELECT name FROM users ORDER BY name")]
        return PAGE.render(users=users, user=user)

    @app.get("/list", response_class=HTMLResponse)
    def list_view(user: str = Query("matt")):
        return render_list(user)

    @app.post("/clicks", response_class=HTMLResponse)
    def click(user: str = Form(...), item_id: int = Form(...), useful: int = Form(...)):
        u = get_user(conn, user)
        conn.execute(
            "INSERT OR REPLACE INTO clicks(user_id, item_id, useful) VALUES (?, ?, ?)",
            (u["id"], item_id, useful),
        )
        conn.commit()
        return render_list(user)  # retrain + re-rank happens inside rank_items

    @app.get("/explanation", response_class=PlainTextResponse)
    def explanation(user: str = Query(...), item_id: int = Query(...)):
        u = get_user(conn, user)
        return explain(conn, u["id"], item_id, chat_fn=chat_fn) or ""

    return app
```

Note for the implementer: `TestClient` runs handlers in a worker thread; SQLite needs
`check_same_thread=False` for that. Update `get_db` in `src/hermes/db.py` to
`sqlite3.connect(str(path), check_same_thread=False)` as part of this task (the app
serializes writes through its single connection; WAL + busy_timeout cover cron ingest).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py tests/test_db.py -v`
Expected: all PASS (including unchanged db tests after the `check_same_thread` edit)

- [ ] **Step 5: Commit**

```bash
git add src/hermes/server.py src/hermes/db.py tests/test_server.py
git commit -m "feat: one-page htmx UI with live re-rank and lazy explanations"
```

---

### Task 7: CLI (ingest / serve / eval / warmup / bootstrap-persona)

**Files:**
- Create: `src/hermes/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: console script `hermes` → `hermes.cli.main(argv: list[str] | None = None) -> int`. Subcommands:
  - `hermes ingest [--db hermes.db] [--feeds feeds.toml]` — prints stats dict
  - `hermes serve [--db hermes.db] [--port 8899]` — uvicorn on 127.0.0.1
  - `hermes eval [--db hermes.db] --user <name>` — prints AUC or explains insufficient data
  - `hermes warmup` — loads the configured chat model + embeddinggemma with `keep_alive=-1`
  - `hermes bootstrap-persona <name> [--db hermes.db] [-k 30]` — prints count written

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
from hermes.cli import build_parser, main
from hermes.db import get_db


def test_parser_subcommands():
    parser = build_parser()
    for argv in (
        ["ingest"],
        ["serve", "--port", "9000"],
        ["eval", "--user", "matt"],
        ["warmup"],
        ["bootstrap-persona", "bench-chemist", "-k", "10"],
    ):
        assert parser.parse_args(argv).command == argv[0]


def test_eval_insufficient_data_message(tmp_path, capsys):
    db = tmp_path / "t.db"
    get_db(db).close()
    rc = main(["eval", "--db", str(db), "--user", "matt"])
    assert rc == 0
    assert "insufficient" in capsys.readouterr().out.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hermes.cli'`

- [ ] **Step 3: Implement cli.py**

```python
# src/hermes/cli.py
"""hermes CLI: ingest | serve | eval | warmup | bootstrap-persona."""

import argparse

import httpx

OLLAMA = "http://localhost:11434"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hermes")
    sub = p.add_subparsers(dest="command", required=True)

    def add_db(sp):
        sp.add_argument("--db", default="hermes.db")

    sp = sub.add_parser("ingest", help="fetch feeds, embed, store")
    add_db(sp)
    sp.add_argument("--feeds", default="feeds.toml")

    sp = sub.add_parser("serve", help="run the web UI")
    add_db(sp)
    sp.add_argument("--port", type=int, default=8899)

    sp = sub.add_parser("eval", help="leave-last-N-out AUC for a user")
    add_db(sp)
    sp.add_argument("--user", required=True)

    sub.add_parser("warmup", help="pin chat + embedding models in VRAM")

    sp = sub.add_parser("bootstrap-persona", help="write pseudo-clicks for a persona")
    add_db(sp)
    sp.add_argument("name")
    sp.add_argument("-k", type=int, default=30)
    return p


def warmup() -> None:
    from hermes.explain import CHAT_MODEL

    httpx.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "keep_alive": -1,
            "options": {"num_ctx": 8192},
        },
        timeout=300,
    ).raise_for_status()
    httpx.post(
        f"{OLLAMA}/api/embed",
        json={"model": "embeddinggemma", "input": "warmup", "keep_alive": -1},
        timeout=300,
    ).raise_for_status()
    print(f"models loaded and pinned (chat={CHAT_MODEL}, keep_alive=-1)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "warmup":
        warmup()
        return 0

    from hermes.db import get_db

    if args.command == "ingest":
        from hermes.embed import OllamaEmbedder
        from hermes.ingest import run_ingest

        stats = run_ingest(get_db(args.db), OllamaEmbedder(), args.feeds)
        print(stats)
        return 0

    if args.command == "serve":
        import uvicorn

        from hermes.server import create_app

        uvicorn.run(create_app(args.db), host="127.0.0.1", port=args.port)
        return 0

    if args.command == "eval":
        from hermes.rank import evaluate_user, get_user

        conn = get_db(args.db)
        user = get_user(conn, args.user)
        auc = evaluate_user(conn, user["id"]) if user else None
        if auc is None:
            print("insufficient click data for a meaningful holdout (need 10+ mixed clicks)")
        else:
            print(f"leave-last-5-out AUC: {auc:.3f}  (noise at small n -- not evidence)")
        return 0

    if args.command == "bootstrap-persona":
        from hermes.embed import OllamaEmbedder
        from hermes.rank import bootstrap_persona

        n = bootstrap_persona(get_db(args.db), OllamaEmbedder(), args.name, k=args.k)
        print(f"wrote {n} pseudo-clicks for {args.name}")
        return 0

    return 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: entire suite PASS

- [ ] **Step 5: Lint**

Run: `uv run ruff check .`
Expected: clean (fix anything it flags before committing)

- [ ] **Step 6: Commit**

```bash
git add src/hermes/cli.py tests/test_cli.py
git commit -m "feat: hermes CLI with ingest, serve, eval, warmup, bootstrap-persona"
```

---

### Task 8: README + DEMO runbook + live smoke test

**Files:**
- Create: `README.md`, `DEMO.md`

**Interfaces:**
- Consumes: everything; this task is docs + a manual end-to-end verification.

- [ ] **Step 1: Write README.md**

```markdown
# hermes-rss

**Hermes** is an agent orchestrator for personalized feed recommendations, fully
local. The orchestrator coordinates three layers with distinct reliability
contracts: deterministic ingest (fetch → dedup → embed), a per-user learnable
ranking core (profile embedding + click-trained classifier), and a LangGraph
explain agent that says *why* items rank where they do — lazily, cached, never
blocking the feed. The LLM is a swappable Ollama backend, not the point.

## Requirements

- Ollama with `embeddinggemma` and a chat model pulled (default `gemma4:12b`;
  set `HERMES_CHAT_MODEL` to use another, e.g. `hermes3:8b`)
- Python 3.12+, `uv`

## Quick start

    export OLLAMA_MAX_LOADED_MODELS=2   # keep chat + embed models co-resident
    uv sync
    uv run hermes warmup                # pin both models in VRAM (avoids 10-20s cold loads)
    uv run hermes ingest                # fetch feeds.toml -> hermes.db
    uv run hermes serve                 # http://127.0.0.1:8899

Click ✓/✗ on items; the feed retrains and re-ranks on every click. Switch users
in the nav to see the same feed ranked per-identity.

## How ranking works

- 0 clicks: cosine similarity between item embeddings (embeddinggemma, 256-dim)
  and your `interests` profile text.
- With clicks: per-user logistic regression over item embeddings, blended with the
  profile by rank: `w = n_clicks / (n_clicks + 5)`. Visible movement by click 3-4.
- Guard: classifier only participates once your clicks contain both classes.

## Commands

    uv run hermes ingest [--feeds feeds.toml] [--db hermes.db]
    uv run hermes serve [--port 8899]
    uv run hermes eval --user matt          # holdout AUC (noisy at small n)
    uv run hermes bootstrap-persona bench-chemist   # optional persona pseudo-clicks
    uv run hermes warmup

Cron ingest: `17 * * * * cd ~/hermes-rss && uv run hermes ingest`
(WAL mode makes concurrent ingest + serving safe.)

## Tests

    uv run pytest
    uv run ruff check .
```

- [ ] **Step 2: Write DEMO.md**

```markdown
# Demo runbook

Rehearse once end-to-end before the call.

## Before (10 min prior)

1. `export OLLAMA_MAX_LOADED_MODELS=2` (in the shell that starts ollama serve,
   or systemd override) — prevents the embedder evicting the chat model.
2. `uv run hermes warmup` — both models resident; re-run until it returns fast.
3. `uv run hermes ingest` — fresh items.
4. `uv run hermes serve` — open http://127.0.0.1:8899, confirm list renders.
5. Reset demo state if needed: delete matt's clicks
   `sqlite3 hermes.db "DELETE FROM clicks WHERE user_id = (SELECT id FROM users WHERE name='matt')"`

## The show (in this order)

0. **The front door — Hermes Agent** (Nous Research's open agent framework, with
   our engine packaged as a skill; see Task 9 / `skills/science-recommendations/`):
   in a `hermes` session, ask *"What's in my science recommendations feed today?"*
   — the agent discovers the skill, calls our local API via its terminal tool, and
   presents the ranked feed. Then: *"The second one looks good, mark it useful"* →
   it POSTs the click. That's the founder's product shape live: Hermes Agent
   orchestrating a domain recommendation engine.
   Pre-flight for this act: `OLLAMA_CONTEXT_LENGTH=32768` set when Ollama starts
   (the <24GB-VRAM default of 4,096 tokens silently truncates tool schemas —
   verify with `ollama ps`), and the agent driver model must pass a tool-calling
   smoke test (`gemma4:12b` first; `hermes3:8b` is on the docs' known-good list
   as fallback).
1. **Persona switch** (zero clicks, zero LLM): open the web UI as `bench-chemist`,
   then `ml-engineer`. Same feed, different order, driven by profile embeddings alone.
2. **Live learning**: switch to `matt`, click ✓ on 3-4 on-topic items and ✗ on
   3-4 off-topic ones on a rehearsed sequence. Reorder is visible by click 3-4
   (blend weight w = n/(n+5)).
3. **Explanations as garnish**: point at the italics filling in asynchronously —
   the LangGraph explain graph calling a local model on a GTX 1080, cached per
   (user, item), and the feed never waits on it. The orchestrator's reliability
   contract, visible.

## Talking points

- Deterministic core / LLM garnish split = reliability contracts per layer.
- Cold start: interests text -> profile embedding; ramps smoothly into the
  classifier as clicks arrive. No cliff.
- `hermes eval` exists and reports honest noise at n=15 — eval-first habit,
  not decorative metrics.
- What this grows into at scale: preference-optimization post-training
  (DPO/ORPO), learned rerankers, pgvector. Same shapes, bigger substrate.

## Failure modes

- Explanations blank → Ollama down or model evicted; feed still works. Say so:
  that's the reliability contract doing its job.
- Feed empty → recency window; run `uv run hermes ingest`.
```

- [ ] **Step 3: Live smoke test (manual, not CI)**

Run, in order, and verify by hand:

```bash
ollama pull gemma4:12b               # or set HERMES_CHAT_MODEL to a pulled model
export OLLAMA_MAX_LOADED_MODELS=2
uv run hermes warmup                 # expect: "models loaded and pinned"
uv run hermes ingest                 # expect: added > 0, failed_feeds ideally 0
uv run hermes serve                  # open http://127.0.0.1:8899
```

Verify in the browser: (a) persona switch reorders the list, (b) three ✓ clicks
don't error and visibly move the list by click 3-4, (c) explanations appear within
~10s on warm models. If any feed in feeds.toml 404s, note it in README and move on
— per-feed failure isolation means the rest still ingests.

- [ ] **Step 4: Commit**

```bash
git add README.md DEMO.md
git commit -m "docs: README and demo runbook"
```

---

## Self-Review (completed at plan-writing time)

- **Spec coverage:** schema+seeds (T1), embedding prefixes+truncate/renorm (T2), ingest/dedup/boilerplate/feeds.toml/failure-isolation (T3), guard/ramp/rank-blend/recency/exclusion/eval/bootstrap (T4), explain graph/retry/fallback/cache/lazy (T5), UI/user-switch/click-rerank/top-20-lazy (T6), CLI incl. warmup with keep_alive=-1 and num_ctx=8192 (T7), demo runbook incl. GPU env vars and rehearsal (T8). Enrich graph: intentionally absent (cut by review).
- **Known deviation:** htmx is loaded from unpkg CDN in the page template — acceptable for a local tool; vendor the file into `src/hermes/static/` later if offline demo matters.
- **Type consistency:** `run_ingest` stats dict keys, `RankedItem` fields, `explain(conn, user_id, item_id, chat_fn)` signature, and `create_app(db_path, embedder, chat_fn)` are used identically across tasks 3–8.
```

---

### Task 9: Package the engine as a Hermes Agent skill

**Files:**
- Create: `skills/science-recommendations/SKILL.md` (versioned in this repo — source of truth)
- Create (outside repo): `~/.hermes/skills/science-recommendations/SKILL.md` (installed copy)
- Modify (outside repo): `~/.hermes/config.yaml` (provider block for local Ollama)
- Requirements detail: `.superpowers/sdd/2026-08-04-hermes-rss/task-9-brief.md` (research brief — install commands, provider config, skill anatomy, demo flow, gotchas)

**Interfaces:**
- Consumes: the running engine API (GET /list?user=NAME → HTML fragment; POST /clicks form fields user/item_id/useful where useful is INTEGER 1 or 0 — "true"/"false" will 422; GET /explanation?user=NAME&item_id=ID → plain text, route confirmed against server.py) and the CLI (`uv run hermes ingest` from /home/matt/hermes-rss).
- Produces: an installed, auto-discovered hermes-agent skill invocable as /science-recommendations or via natural conversation.

**Steps (environment-interactive; verify each, no strict TDD):**

- [ ] **Step 1:** Write `skills/science-recommendations/SKILL.md` in-repo, starting from the research brief's draft with these corrections: (a) all `useful=<true|false>` examples become `useful=1` / `useful=0`; (b) remove the UNVERIFIED caveats about /explanation — the route and params in the Quick Reference are confirmed correct; (c) the "Start the server" row reads `cd /home/matt/hermes-rss && uv run hermes serve &` (drop the ${HERMES_SKILL_DIR} relative-path acrobatics).
- [ ] **Step 1b:** Add `skills/science-recommendations/scripts/setup.sh` — idempotent requirements check-and-repair the agent can run via its terminal tool: verify `uv` on PATH, verify Ollama responds at localhost:11434 and `embeddinggemma` + the chat model are in `ollama list`, verify the project dir from skill config exists (else print the git clone + `uv sync` instructions), verify `hermes.db` exists (else run `uv run hermes ingest`), then probe `GET /list?user=<default user>` and start `uv run hermes serve &` if down. Exit non-zero with a one-line reason on any unrepairable gap. Reference it from SKILL.md's Procedure step 1 (`bash ${HERMES_SKILL_DIR}/scripts/setup.sh`).
- [ ] **Step 2:** Install hermes-agent per the brief (`curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash`), reload PATH, record `hermes --version`. If config key names drifted from the brief (docs pinned v0.14.0), trust the live `hermes model` wizard.
- [ ] **Step 3:** Configure provider: Custom Endpoint `http://localhost:11434/v1`, no API key, model `gemma4:12b`, context_length 32768 (via `hermes model` wizard or `~/.hermes/config.yaml` per the brief). NOT "Ollama Cloud".
- [ ] **Step 4:** Verify Ollama server-side context: ensure `OLLAMA_CONTEXT_LENGTH=32768` is in effect (env or systemd override; check `ollama ps` CONTEXT column after a chat call). Document what was changed and where.
- [ ] **Step 5:** Copy the skill: `mkdir -p ~/.hermes/skills/science-recommendations && cp skills/science-recommendations/SKILL.md ~/.hermes/skills/science-recommendations/`.
- [ ] **Step 6:** Tool-calling smoke: non-interactive `hermes` query (e.g. `hermes chat -q "list your skills"` or the closest live equivalent) confirming the skill is discovered and the model emits real tool calls, not plain-text JSON. If gemma4:12b fails tool-calling, switch config default to `hermes3:8b` (known-good list) and record the outcome.
- [ ] **Step 7:** End-to-end: with the engine served (`uv run hermes serve`), ask the agent for the feed; verify a terminal tool call hits GET /list and the reply presents titles (not raw HTML). Record the transcript excerpt in the report.
- [ ] **Step 8:** Commit the in-repo skill file: `git add skills && git commit -m "feat: package engine as hermes-agent skill (science-recommendations)"`.

---

### Task 10: Distribution packaging (serve the skill to strangers)

**Files:**
- Modify: `skills/science-recommendations/SKILL.md` (setup notes: uvx flow, data-dir default), `skills/science-recommendations/scripts/setup.sh`, `README.md` (Serving/Distribution section)
- Modify: `src/hermes/cli.py` + `src/hermes/server.py` + `src/hermes/db.py` ONLY as needed for a `HERMES_RSS_DB` env var / `--db` default resolution honoring `~/.hermes/skills/science-recommendations/data/hermes.db` when it exists (small, keep tests green; add a unit test for the resolution order: explicit --db > HERMES_RSS_DB > skill data dir if present > ./hermes.db)
- Out-of-repo: move the live hermes.db to `~/.hermes/skills/science-recommendations/data/` and re-verify one API call

**Steps:**
- [ ] setup.sh: prefer `uvx --from git+<REPO_URL> hermes-rss serve` when the configured project_dir is absent (REPO_URL read from skill config key `science_recommendations.repo_url`, default empty → fall back to project_dir flow with a clear message); keep local-path flow primary while repo is unpushed.
- [ ] pyproject: verify `uv run --project . hermes` and `uvx --from . hermes --help` both work (uvx-from-local proves the uvx path without any publish).
- [ ] DB resolution env var + test as above; default the skill's server-start commands to the co-located data dir.
- [ ] Verify on v0.20.0 what `hermes skills publish` actually requires (auth? account?) WITHOUT publishing; write the exact user-gated commands (git push + publish) into README's Serving section, clearly marked "run these yourself".
- [ ] Run full test suite + ruff; commit "feat: distribution packaging — uvx flow, co-located state, publish runbook".
