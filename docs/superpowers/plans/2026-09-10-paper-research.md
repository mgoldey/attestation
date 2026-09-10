# Paper Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standing research topics as feeds, an ad hoc `feed.research` tool that stores into the library, full text in windows, deterministic BibTeX, one flag on by default.

**Architecture:** A topic is a `feeds` row with a `research:` URL; `run_ingest` dispatches on scheme to pure-over-payload search clients in a new `research.py`, and the three ingest passes run unchanged. Results are also upserted into the library as `ReferenceRecord`s with source `research:<client>`. Full text lives in `reference_fulltext` and BibTeX is rendered from a `references` row; both ride on `cite.lookup`.

**Tech Stack:** Python 3.12, sqlite + sqlite-vec, httpx, defusedxml, pypdf (new), feedparser-shaped entries, FastMCP tools via `@tool`.

**Spec:** `docs/superpowers/specs/2026-09-10-paper-research-design.md`

## Global Constraints

- Work in the worktree `/home/matt/attestation/.claude/worktrees/research-topics` on branch `feat/research-topics`. Never open the live database from it: tests use `tmp_path`, manual checks use `ATTEST_DB=/tmp/...`.
- Every public `def`/`class` needs a docstring (`tests/test_docstring_ratchet.py`, baseline 0).
- Cyclomatic complexity per function ≤ 10 (`scripts/check_complexity.py`; xenon `--max-modules B --max-average A`). Split helpers rather than growing `run_ingest` or `add_source`.
- Line length 100; ruff `E,F,W,I,BLE,RUF100`. A broad `except Exception` needs `# noqa: BLE001 -- <reason>` AND the count CLAUDE.md states must be bumped (Task 9).
- Every `os.environ.get("ATTEST_X")` literal must have a `#ATTEST_X=` line in `.env.sample` (`tests/test_llm.py::test_env_sample_documents_exactly_the_vars_the_code_reads`).
- Tool names never repeat their namespace; the new tool is `feed.research`. Every doc quoting the total tool count (48) or the feed surface count (21) moves with it, or `test_architecture.py`'s stale-count guard fails.
- Network readers are built once (construction time), never per call. Tests never touch the network: inject `fetch`.
- Commit after every task with a message that says WHY, ending in `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Run `uv run --frozen pytest <files> -q` per step; the full gate `uv run --frozen pre-commit run --all-files` at the end of Tasks 6, 9 and 11.
- Do NOT push. The user pushes.

---

## File map

| file | responsibility |
|---|---|
| `src/attestation/db.py` | migration 009 (`feeds.added_by`, `"references".pmcid`, `reference_fulltext`) + SCHEMA |
| `src/attestation/library.py` | `pmcid` on `ReferenceRecord`/columns; `bibtex()`, `bibtex_key()`, `export_bib()`, `select_rows()`, `fulltext_window()` |
| `src/attestation/research.py` (new) | `Topic`/`parse_topic`/`topic_url`, `Paper`, `as_entries`/`as_records`, the three search clients + pure parsers, `NullClient`, `clients_from_env`, `fetch_topic`, `fetch_fulltext`, `pdf_text`, `parse_pmc` |
| `src/attestation/feeds.py` | `add_source` accepts `research:` URLs (no network), `added_by`; `list_sources` adds `kind`, `added_by` |
| `src/attestation/ingest.py` | scheme dispatch, identifier dedup, post-commit library upsert, `research_disabled` outcome |
| `src/attestation/mcp/research.py` (new) | `feed.research` tool |
| `src/attestation/mcp/subscriptions.py` | `feed.source_add(user=)` |
| `src/attestation/mcp/citation.py` | `cite.lookup` bibtex + full-text window; `cite.sources` offline flag |
| `src/attestation/mcp/routing.py`, `mcp/ask.py` | research and track routes |
| `src/attestation/mcp/__init__.py` | register the new module; feed goal text |
| `src/attestation/cli.py` | `attest research`, `attest sources add`, `attest library fulltext`, `attest library export`, `attest ingest --fulltext-limit/--no-research` |
| `tests/test_research.py` (new), `tests/fixtures/research/*` | parsers, topic URLs, clients, fulltext |
| `tests/test_research_tools.py` (new) | `feed.research` envelope, size, routing dispatch |
| docs, skills, CLAUDE.md, README, CHANGELOG, .env.sample | counts, guides, skill sections |

---

### Task 1: Migration 009 and the `pmcid` column on records

**Files:**
- Modify: `src/attestation/db.py` (SCHEMA `feeds`, `_LIBRARY_SCHEMA`, new `_migration_009_add_research`, `_MIGRATIONS`)
- Modify: `src/attestation/library.py` (`ReferenceRecord.pmcid`, `fields()`, `_COLUMNS`, `_insert`)
- Modify: `tests/test_library_db.py:90-98` (counts 27/16 → 28/17)
- Modify: `CLAUDE.md` Storage line (16 → 17 application tables, 27 → 28, name the new table)
- Test: `tests/test_db.py`, `tests/test_library_db.py`

**Interfaces:**
- Produces: table `reference_fulltext(reference_id PK, text, source, fetched_at)`; column `feeds.added_by INTEGER REFERENCES users(id)`; column `"references".pmcid TEXT`; `ReferenceRecord(pmcid=...)` persisted by `upsert`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_library_db.py`:

```python
def test_migration_009_adds_fulltext_added_by_and_pmcid(tmp_path):
    """A v8 file gains the three research additions; a fresh file already has them;
    re-opening is a no-op."""
    db = tmp_path / "v8.db"
    conn = dbmod.get_db(db)
    conn.execute("DROP TABLE reference_fulltext")
    conn.execute('ALTER TABLE "references" DROP COLUMN pmcid')
    conn.execute("ALTER TABLE feeds DROP COLUMN added_by")
    conn.execute("PRAGMA user_version = 8")
    conn.commit()
    conn.close()
    conn = dbmod.get_db(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == dbmod.SCHEMA_VERSION
    assert "added_by" in {r["name"] for r in conn.execute("PRAGMA table_info(feeds)")}
    assert "pmcid" in {r["name"] for r in conn.execute('PRAGMA table_info("references")')}
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(reference_fulltext)")}
    assert cols == {"reference_id", "text", "source", "fetched_at"}
    conn.close()
    dbmod.get_db(db).close()  # idempotent


def test_upsert_persists_pmcid(tmp_path):
    conn = dbmod.get_db(tmp_path / "t.db")
    rid, how = upsert(
        conn,
        ReferenceRecord(source="research:pubmed", source_key="1", title="T", pmcid="PMC12"),
    )
    assert how == "added"
    assert conn.execute('SELECT pmcid FROM "references" WHERE id = ?', (rid,)).fetchone()[0] == "PMC12"
```

(`tests/test_library_db.py` already imports `dbmod`; add `from attestation.library import ReferenceRecord, upsert` at the top if absent.) Then change the existing count test: `(27, 16)` → `(28, 17)`, and the two CLAUDE.md phrases to `"17 APPLICATION tables"` and `"a fresh file has 28"`.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen pytest tests/test_library_db.py -q`
Expected: FAIL (no `reference_fulltext`, unknown column `pmcid`, counts 27/16).

- [ ] **Step 3: Implement**

In `src/attestation/db.py`:

1. In `SCHEMA`, the `feeds` table becomes:
```sql
CREATE TABLE IF NOT EXISTS feeds(
  id INTEGER PRIMARY KEY,
  url TEXT UNIQUE NOT NULL,
  title TEXT,
  last_fetched TEXT,
  -- Who registered it (migration 009). Provenance, not scoping: the candidate
  -- pool stays global and the ranker sorts per reader.
  added_by INTEGER REFERENCES users(id)
);
```
2. In `_LIBRARY_SCHEMA`, add `pmcid TEXT,` after `arxiv_id TEXT,` in `"references"`, and append before the closing `"""`:
```sql
-- Full text is a side table, never embedded and never returned whole:
-- cite.lookup serves it in windows. source: arxiv-pdf | pmc-xml | none
-- ('none' = tried, nothing to fetch, do not retry hourly).
CREATE TABLE IF NOT EXISTS reference_fulltext(
  reference_id INTEGER PRIMARY KEY REFERENCES "references"(id) ON DELETE CASCADE,
  text TEXT,
  source TEXT NOT NULL,
  fetched_at TEXT NOT NULL
);
```
3. Add after `_migration_008_add_title_key`:
```python
def _migration_009_add_research(conn: sqlite3.Connection) -> None:
    """Paper research (spec 2026-09-10): `feeds.added_by`, `references.pmcid`,
    and the `reference_fulltext` side table. All guarded, so a fresh SCHEMA
    file runs this as a no-op."""
    feed_cols = {r["name"] for r in conn.execute("PRAGMA table_info(feeds)")}
    if "added_by" not in feed_cols:
        conn.execute("ALTER TABLE feeds ADD COLUMN added_by INTEGER REFERENCES users(id)")
    ref_cols = {r["name"] for r in conn.execute('PRAGMA table_info("references")')}
    if "pmcid" not in ref_cols:
        conn.execute('ALTER TABLE "references" ADD COLUMN pmcid TEXT')
    conn.execute(
        "CREATE TABLE IF NOT EXISTS reference_fulltext("
        ' reference_id INTEGER PRIMARY KEY REFERENCES "references"(id) ON DELETE CASCADE,'
        " text TEXT, source TEXT NOT NULL, fetched_at TEXT NOT NULL)"
    )
```
and `(9, _migration_009_add_research),` to `_MIGRATIONS`.

In `src/attestation/library.py`: add `pmcid: str | None = None` to `ReferenceRecord` after `bib_key`; add `"pmcid": self.pmcid,` to the `out` dict in `fields()`; add `"pmcid"` to `_COLUMNS`; in `_insert` add `pmcid` to the column list and `fields.get("pmcid")` to the values (keep the placeholder count in step).

In `CLAUDE.md` Storage line: `plus "references",reference_sources,reference_tags,reference_cites,reference_fulltext (17 APPLICATION tables — ... migration 007 adds the four library tables, 009 adds reference_fulltext)` and `a fresh file has 28 in sqlite_master (MEASURED 2026-09-10, pinned by test_library_db.py)`.

- [ ] **Step 4: Run**

Run: `uv run --frozen pytest tests/test_db.py tests/test_library_db.py tests/test_library.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/attestation/db.py src/attestation/library.py tests/test_library_db.py CLAUDE.md
git commit -m "Migration 009: feeds.added_by, references.pmcid, reference_fulltext -- the storage the research spec needs, all guarded; ReferenceRecord carries pmcid so a PubMed hit keeps the handle full text needs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `research.py` core: topics, papers, conversions, the flag

**Files:**
- Create: `src/attestation/research.py`
- Create: `tests/test_research.py`
- Modify: `.env.sample` (document `ATTEST_RESEARCH_WEB`, `NCBI_API_KEY`)

**Interfaces:**
- Produces:
  - `CLIENT_NAMES = ("arxiv", "pubmed", "crossref")`, `SCHEME = "research:"`
  - `class TopicError(ValueError)`
  - `Topic(clients: tuple[str, ...], query: str, journal: str | None)` with `.url` and `.title`
  - `is_topic_url(url) -> bool`, `parse_topic(url) -> Topic`, `topic_url(clients, query, journal=None) -> str`
  - `Paper(client, external_id, title, abstract="", authors=(), published=None, doi=None, arxiv_id=None, pmcid=None, url=None, venue=None, pdf_url=None)` frozen
  - `as_entries(papers) -> list[dict]` (keys `title, summary, id, link, doi, arxiv_id, published_parsed`)
  - `as_records(papers, fetched_at: str) -> list[ReferenceRecord]`
  - `research_enabled() -> bool`, `class NullClient(name)` with `.offline = True`, `.search(...) -> []`
  - `clients_from_env(*, fetch=None) -> dict[str, client]`, `null_clients() -> dict`

- [ ] **Step 1: Write the failing tests**

`tests/test_research.py`:

```python
"""research.py: topic URLs, payload parsing, conversions, the flag. No network."""

import time

import pytest

from attestation import research


def test_parse_topic_two_forms():
    t = research.parse_topic("research:arxiv,pubmed?q=equivariant+force+fields")
    assert t.clients == ("arxiv", "pubmed") and t.query == "equivariant force fields"
    assert t.journal is None
    t2 = research.parse_topic("research:crossref?q=protein+language+models&journal=Nature+Methods")
    assert t2.clients == ("crossref",) and t2.journal == "Nature Methods"
    assert t2.url == "research:crossref?q=protein+language+models&journal=Nature+Methods"
    assert t2.title == "crossref: protein language models in Nature Methods"


@pytest.mark.parametrize(
    "url, needle",
    [
        ("research:scholar?q=x", "unknown research client"),
        ("research:arxiv?q=", "needs q="),
        ("research:arxiv?q=x&journal=Nature", "categories, not journals"),
        ("https://example.com/rss", "not a research: URL"),
    ],
)
def test_parse_topic_refuses(url, needle):
    with pytest.raises(research.TopicError, match=needle):
        research.parse_topic(url)


def test_topic_url_round_trips():
    url = research.topic_url(("arxiv",), "graph neural networks", None)
    assert research.parse_topic(url).query == "graph neural networks"
    assert research.is_topic_url(url) and not research.is_topic_url("http://x")


def _paper(**kw):
    base = dict(
        client="arxiv",
        external_id="2106.02347",
        title="E(3)-equivariant graph neural networks",
        abstract="We propose NequIP.",
        authors=("Batzner, Simon", "Musaelian, Albert"),
        published="2021-06-04",
        arxiv_id="2106.02347",
        url="https://arxiv.org/abs/2106.02347",
    )
    base.update(kw)
    return research.Paper(**base)


def test_as_entries_is_feedparser_shaped():
    [e] = research.as_entries([_paper()])
    assert e["id"] == "arxiv:2106.02347"
    assert e["title"].startswith("E(3)") and e["summary"] == "We propose NequIP."
    assert e["link"] == "https://arxiv.org/abs/2106.02347"
    assert e["arxiv_id"] == "2106.02347" and e["doi"] is None
    assert time.strftime("%Y-%m-%d", e["published_parsed"]) == "2021-06-04"


def test_as_records_carries_what_the_payload_had():
    [r] = research.as_records([_paper(doi="10.1/x", pmcid="PMC1", venue=None)], "2026-09-10")
    assert r.source == "research:arxiv" and r.source_key == "2106.02347"
    assert r.authors == ["Batzner, Simon", "Musaelian, Albert"] and r.year == 2021
    assert r.doi == "10.1/x" and r.pmcid == "PMC1" and r.fetched_at == "2026-09-10"


def test_flag_default_on_and_off(monkeypatch):
    monkeypatch.delenv("ATTEST_RESEARCH_WEB", raising=False)
    assert research.research_enabled()
    for off in ("0", "false"):
        monkeypatch.setenv("ATTEST_RESEARCH_WEB", off)
        assert not research.research_enabled()
        clients = research.clients_from_env()
        assert set(clients) == set(research.CLIENT_NAMES)
        assert all(c.offline for c in clients.values())
        assert clients["arxiv"].search("x") == []
    monkeypatch.setenv("ATTEST_RESEARCH_WEB", "1")
    assert not any(c.offline for c in research.clients_from_env().values())
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --frozen pytest tests/test_research.py -q`
Expected: FAIL with `ModuleNotFoundError: attestation.research`.

- [ ] **Step 3: Implement `research.py` (core half; clients come in Task 3)**

```python
"""Paper research: search clients over arXiv, PubMed and CrossRef, topics as feeds.

Spec: docs/superpowers/specs/2026-09-10-paper-research-design.md. A standing
topic is a `feeds` row whose URL is `research:<clients>?q=<query>[&journal=]`;
`run_ingest` dispatches on that scheme and the clients here return entries
shaped like feedparser's, so the three ingest passes run unchanged. Parsing
is pure over committed fixture payloads; the HTTP call is one injected
`fetch(url) -> bytes`. Nothing here ranks or summarises: the caller is the
model.

`ATTEST_RESEARCH_WEB` (unset or 1 = on) is read once, in `clients_from_env`,
never per call -- the offline guarantee's construction-time rule.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date
from urllib.parse import parse_qs, quote_plus, urlencode

from attestation.library import ReferenceRecord

CLIENT_NAMES = ("arxiv", "pubmed", "crossref")
SCHEME = "research:"
DEFAULT_SINCE_DAYS = 365


class TopicError(ValueError):
    """A research URL the caller can fix: unknown client, no query, journal on arXiv."""


@dataclass(frozen=True)
class Topic:
    """One standing query: which clients, what to ask, optionally which journal."""

    clients: tuple[str, ...]
    query: str
    journal: str | None = None

    @property
    def url(self) -> str:
        """The canonical `research:` URL, so two spellings of one topic are one row."""
        return topic_url(self.clients, self.query, self.journal)

    @property
    def title(self) -> str:
        """The default feed title: `<clients>: <query>[ in <journal>]`."""
        tail = f" in {self.journal}" if self.journal else ""
        return f"{','.join(self.clients)}: {self.query}{tail}"


def is_topic_url(url: str) -> bool:
    """True for the `research:` scheme this module owns."""
    return url.startswith(SCHEME)


def topic_url(clients, query: str, journal: str | None = None) -> str:
    """Render a topic URL; `parse_topic` reads it back."""
    params = {"q": " ".join(query.split())}
    if journal:
        params["journal"] = " ".join(journal.split())
    return f"{SCHEME}{','.join(clients)}?{urlencode(params, quote_via=quote_plus)}"


def parse_topic(url: str) -> Topic:
    """A `research:` URL as a Topic, or TopicError naming what to fix."""
    if not is_topic_url(url):
        raise TopicError(f"{url!r} is not a research: URL (research:<clients>?q=<query>)")
    path, _, qs = url[len(SCHEME) :].partition("?")
    clients = tuple(c.strip() for c in path.split(",") if c.strip())
    unknown = [c for c in clients if c not in CLIENT_NAMES]
    if not clients or unknown:
        raise TopicError(
            f"unknown research client(s) {', '.join(unknown) or '(none)'};"
            f" the clients are {', '.join(CLIENT_NAMES)}"
        )
    params = parse_qs(qs, keep_blank_values=True)
    query = " ".join(params.get("q", [""])[0].split())
    if not query:
        raise TopicError("a research topic needs q=<query>, e.g. research:arxiv?q=graph+neural+networks")
    journal = " ".join(params.get("journal", [""])[0].split()) or None
    if journal and "arxiv" in clients:
        raise TopicError("arXiv has categories, not journals: drop journal= or drop arxiv")
    return Topic(clients, query, journal)


@dataclass(frozen=True)
class Paper:
    """One search hit, whichever client found it. `external_id` is the client's own handle."""

    client: str
    external_id: str
    title: str
    abstract: str = ""
    authors: tuple[str, ...] = ()
    published: str | None = None  # ISO date, the paper's own
    doi: str | None = None
    arxiv_id: str | None = None
    pmcid: str | None = None
    url: str | None = None
    venue: str | None = None
    pdf_url: str | None = None


def as_entries(papers) -> list[dict]:
    """Papers as the entry dicts `ingest._new_entries` and the insert read.

    `id` is `<client>:<external id>` so UNIQUE(feed_id, guid) holds across the
    clients one topic names; `published_parsed` is the paper's date, never the
    fetch time, so a 2019 hit does not surface as new.
    """
    out = []
    for p in papers:
        entry = {
            "title": p.title,
            "summary": p.abstract,
            "id": f"{p.client}:{p.external_id}",
            "link": p.url,
            "doi": p.doi,
            "arxiv_id": p.arxiv_id,
        }
        if p.published:
            try:
                entry["published_parsed"] = time.strptime(p.published[:10], "%Y-%m-%d")
            except ValueError:
                pass
        out.append(entry)
    return out


def as_records(papers, fetched_at: str) -> list[ReferenceRecord]:
    """Papers as library records under source `research:<client>`, with the
    authors, venue and abstract the payload carried (the stated exception to
    the enricher rule -- the reader asked for these by query)."""
    return [
        ReferenceRecord(
            source=f"research:{p.client}",
            source_key=p.external_id,
            title=p.title,
            authors=list(p.authors),
            year=int(p.published[:4]) if p.published and p.published[:4].isdigit() else None,
            doi=p.doi,
            arxiv_id=p.arxiv_id,
            pmcid=p.pmcid,
            venue=p.venue,
            abstract=p.abstract or None,
            url=p.url,
            fetched_at=fetched_at,
        )
        for p in papers
    ]


def research_enabled() -> bool:
    """ATTEST_RESEARCH_WEB: on unless set to 0/false. Read by `clients_from_env` only."""
    return os.environ.get("ATTEST_RESEARCH_WEB", "1").strip().lower() not in ("0", "false")


class NullClient:
    """What every client is when the flag is off: answers nothing, says it is offline."""

    offline = True

    def __init__(self, name: str):
        self.name = name

    def search(self, query, *, journal=None, since: date | None = None, limit: int = 50):
        """No results; the caller reports `offline` rather than 'no papers'."""
        return []


def null_clients() -> dict:
    """One NullClient per name -- `attest ingest --no-research` and the flag off."""
    return {name: NullClient(name) for name in CLIENT_NAMES}


def clients_from_env(*, fetch=None) -> dict:
    """The three clients, or NullClients when ATTEST_RESEARCH_WEB is off.

    The flag is read HERE, once, when the ingest run or the MCP server builds
    its clients -- a disabled client cannot be coaxed into a request later.
    """
    if not research_enabled():
        return null_clients()
    return {
        "arxiv": ArxivSearch(fetch=fetch),
        "pubmed": PubmedSearch(fetch=fetch, api_key=os.environ.get("NCBI_API_KEY")),
        "crossref": CrossrefSearch(fetch=fetch),
    }
```

For this task only, add temporary stand-ins at the bottom so the module imports (Task 3 replaces them):

```python
class ArxivSearch(NullClient):  # replaced in Task 3
    offline = False

    def __init__(self, fetch=None):
        super().__init__("arxiv")


class PubmedSearch(NullClient):  # replaced in Task 3
    offline = False

    def __init__(self, fetch=None, api_key=None):
        super().__init__("pubmed")


class CrossrefSearch(NullClient):  # replaced in Task 3
    offline = False

    def __init__(self, fetch=None):
        super().__init__("crossref")
```

In `.env.sample`, after the `#ATTEST_CITATION_SCHOLAR=1` line add:

```
# Paper research (standing topics as research: feeds, the feed.research tool
# and full text) reaches arXiv, PubMed and CrossRef. ON by default: `attest
# ingest` already fetches RSS over the network and a topic query is the same
# class of call in the same command. Set to 0 to make every research client a
# no-op that reports `offline`. Read once when the clients are built.
#ATTEST_RESEARCH_WEB=1
# Optional NCBI key: PubMed allows 3 requests/s without one, 10 with.
#NCBI_API_KEY=
```

- [ ] **Step 4: Run**

Run: `uv run --frozen pytest tests/test_research.py tests/test_llm.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/attestation/research.py tests/test_research.py .env.sample
git commit -m "research.py core: topic URLs (research:<clients>?q=...), Paper, feedparser-shaped entries and library records, and ATTEST_RESEARCH_WEB read once at construction

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: The three search clients and their pure parsers

**Files:**
- Modify: `src/attestation/research.py` (replace the stand-ins)
- Create: `tests/fixtures/research/arxiv_search.xml`, `pubmed_esearch.xml`, `pubmed_efetch.xml`, `crossref_works.json`
- Test: `tests/test_research.py`

**Interfaces:**
- Produces:
  - `parse_arxiv(body: bytes) -> list[Paper]`, `parse_pubmed_ids(body) -> list[str]`, `parse_pubmed(body) -> list[Paper]`, `parse_crossref(body) -> list[Paper]`
  - `ArxivSearch(fetch=None)`, `PubmedSearch(fetch=None, api_key=None)`, `CrossrefSearch(fetch=None)`, each `.name`, `.offline = False`, `.search(query, *, journal=None, since=None, limit=50) -> list[Paper]`, `.requests: list[str]` (URLs asked, for tests)
  - `Fetched(entries, papers, errors, offline)` and `fetch_topic(topic, clients, *, since=None, limit=50) -> Fetched`
  - `_sleep(seconds)` module function tests monkeypatch

- [ ] **Step 1: Write the fixtures**

Fixtures are hand-trimmed copies of real responses (two entries each, real field shapes). `tests/fixtures/research/arxiv_search.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title type="html">ArXiv Query: search_query=all:equivariant AND all:force AND all:fields</title>
  <entry>
    <id>http://arxiv.org/abs/2101.03164v3</id>
    <published>2021-01-08T18:56:11Z</published>
    <title>E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate
  Interatomic Potentials</title>
    <summary>  This work presents Neural Equivariant Interatomic Potentials (NequIP).
</summary>
    <author><name>Simon Batzner</name></author>
    <author><name>Albert Musaelian</name></author>
    <arxiv:doi xmlns:arxiv="http://arxiv.org/schemas/atom">10.1038/s41467-022-29939-5</arxiv:doi>
    <link href="http://arxiv.org/abs/2101.03164v3" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2101.03164v3" rel="related" type="application/pdf"/>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="physics.comp-ph"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2206.07697v2</id>
    <published>2022-06-15T17:59:59Z</published>
    <title>MACE: Higher Order Equivariant Message Passing Neural Networks</title>
    <summary>  We introduce MACE.
</summary>
    <author><name>Ilyes Batatia</name></author>
    <link href="http://arxiv.org/abs/2206.07697v2" rel="alternate" type="text/html"/>
  </entry>
</feed>
```

`tests/fixtures/research/pubmed_esearch.xml`:

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<eSearchResult><Count>2</Count><RetMax>2</RetMax><RetStart>0</RetStart>
<IdList><Id>36702928</Id><Id>34961817</Id></IdList></eSearchResult>
```

`tests/fixtures/research/pubmed_efetch.xml`:

```xml
<?xml version="1.0" ?>
<PubmedArticleSet>
<PubmedArticle><MedlineCitation Status="MEDLINE"><PMID Version="1">36702928</PMID>
<Article PubModel="Print-Electronic"><Journal><Title>Nature methods</Title></Journal>
<ArticleTitle>Protein language models for structure prediction.</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">Language models learn.</AbstractText>
<AbstractText Label="RESULTS">They predict structure.</AbstractText></Abstract>
<AuthorList><Author><LastName>Lin</LastName><ForeName>Zeming</ForeName></Author>
<Author><LastName>Rives</LastName><ForeName>Alexander</ForeName></Author></AuthorList>
</Article></MedlineCitation>
<PubmedData><History><PubMedPubDate PubStatus="pubmed"><Year>2023</Year><Month>01</Month><Day>27</Day></PubMedPubDate></History>
<ArticleIdList><ArticleId IdType="pubmed">36702928</ArticleId>
<ArticleId IdType="doi">10.1038/s41592-022-01760-3</ArticleId>
<ArticleId IdType="pmc">PMC9912345</ArticleId></ArticleIdList></PubmedData></PubmedArticle>
<PubmedArticle><MedlineCitation Status="MEDLINE"><PMID Version="1">34961817</PMID>
<Article PubModel="Print"><Journal><Title>Science</Title></Journal>
<ArticleTitle>A second paper.</ArticleTitle>
<AuthorList><Author><CollectiveName>The Consortium</CollectiveName></Author></AuthorList>
</Article></MedlineCitation>
<PubmedData><History><PubMedPubDate PubStatus="pubmed"><Year>2021</Year><Month>Dec</Month></PubMedPubDate></History>
<ArticleIdList><ArticleId IdType="pubmed">34961817</ArticleId></ArticleIdList></PubmedData></PubmedArticle>
</PubmedArticleSet>
```

`tests/fixtures/research/crossref_works.json`:

```json
{"status":"ok","message-type":"work-list","message":{"items":[
 {"DOI":"10.1038/s41592-022-01760-3","title":["Protein language models for structure prediction"],
  "container-title":["Nature Methods"],"author":[{"given":"Zeming","family":"Lin"},{"given":"Alexander","family":"Rives"}],
  "issued":{"date-parts":[[2023,1,27]]},"URL":"https://doi.org/10.1038/s41592-022-01760-3",
  "abstract":"<jats:p>Language models learn.</jats:p>"},
 {"DOI":"10.5555/no-title","container-title":[],"author":[],"issued":{"date-parts":[[2022]]}}
]}}
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_research.py`:

```python
from datetime import date
from pathlib import Path

FIX = Path(__file__).parent / "fixtures" / "research"


def _fetcher(routes: dict[str, str]):
    """fetch(url) -> bytes from the first route whose key is a substring of the url."""
    asked: list[str] = []

    def fetch(url: str) -> bytes:
        asked.append(url)
        for needle, name in routes.items():
            if needle in url:
                return (FIX / name).read_bytes()
        raise AssertionError(f"unexpected url {url}")

    return fetch, asked


def test_parse_arxiv_search():
    papers = research.parse_arxiv((FIX / "arxiv_search.xml").read_bytes())
    assert [p.arxiv_id for p in papers] == ["2101.03164", "2206.07697"]
    p = papers[0]
    assert p.client == "arxiv" and p.external_id == "2101.03164"
    assert p.title.startswith("E(3)-Equivariant Graph Neural Networks for Data-Efficient")
    assert "  " not in p.title
    assert p.authors == ("Batzner, Simon", "Musaelian, Albert")
    assert p.published == "2021-01-08" and p.doi == "10.1038/s41467-022-29939-5"
    assert p.url == "https://arxiv.org/abs/2101.03164"
    assert p.pdf_url == "https://arxiv.org/pdf/2101.03164" and p.venue is None
    assert papers[1].doi is None


def test_parse_pubmed():
    assert research.parse_pubmed_ids((FIX / "pubmed_esearch.xml").read_bytes()) == [
        "36702928",
        "34961817",
    ]
    papers = research.parse_pubmed((FIX / "pubmed_efetch.xml").read_bytes())
    p = papers[0]
    assert p.client == "pubmed" and p.external_id == "36702928"
    assert p.abstract == "Language models learn. They predict structure."
    assert p.authors == ("Lin, Zeming", "Rives, Alexander") and p.venue == "Nature methods"
    assert p.published == "2023-01-27" and p.doi == "10.1038/s41592-022-01760-3"
    assert p.pmcid == "PMC9912345" and p.url == "https://pubmed.ncbi.nlm.nih.gov/36702928/"
    assert papers[1].authors == ("The Consortium",) and papers[1].published == "2021-12-01"


def test_parse_crossref_skips_titleless():
    [p] = research.parse_crossref((FIX / "crossref_works.json").read_bytes())
    assert p.client == "crossref" and p.external_id == p.doi == "10.1038/s41592-022-01760-3"
    assert p.venue == "Nature Methods" and p.authors == ("Lin, Zeming", "Rives, Alexander")
    assert p.published == "2023-01-27" and p.abstract == "Language models learn."
    assert p.url == "https://doi.org/10.1038/s41592-022-01760-3"


def test_arxiv_search_builds_the_query_and_paces(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(research, "_sleep", slept.append)
    fetch, asked = _fetcher({"export.arxiv.org": "arxiv_search.xml"})
    client = research.ArxivSearch(fetch=fetch)
    papers = client.search("equivariant force fields", since=date(2021, 1, 1), limit=20)
    assert len(papers) == 2
    assert "search_query=all%3Aequivariant+AND+all%3Aforce+AND+all%3Afields" in asked[0]
    assert "submittedDate%3A%5B202101010000+TO" in asked[0] and "max_results=20" in asked[0]
    client.search("again")
    assert slept and slept[0] == pytest.approx(3.0, abs=0.5)  # 3 s between requests, as arXiv asks


def test_pubmed_search_two_calls_journal_and_key(monkeypatch):
    monkeypatch.setattr(research, "_sleep", lambda s: None)
    fetch, asked = _fetcher({"esearch": "pubmed_esearch.xml", "efetch": "pubmed_efetch.xml"})
    client = research.PubmedSearch(fetch=fetch, api_key="K")
    papers = client.search("protein language models", journal="Nature Methods", since=date(2022, 1, 1))
    assert len(papers) == 2 and len(asked) == 2
    assert "%22Nature+Methods%22%5BJournal%5D" in asked[0] and "mindate=2022%2F01%2F01" in asked[0]
    assert "api_key=K" in asked[0] and "id=36702928%2C34961817" in asked[1]


def test_pubmed_search_with_no_ids_makes_one_call():
    fetch, asked = _fetcher({"esearch": "pubmed_esearch_empty.xml"})
    (FIX / "pubmed_esearch_empty.xml").write_text(
        '<?xml version="1.0"?><eSearchResult><Count>0</Count><IdList/></eSearchResult>'
    )
    assert research.PubmedSearch(fetch=fetch).search("nothing") == [] and len(asked) == 1


def test_crossref_search_filters():
    fetch, asked = _fetcher({"api.crossref.org": "crossref_works.json"})
    papers = research.CrossrefSearch(fetch=fetch).search("x", journal="Nature Methods", since=date(2023, 1, 1))
    assert len(papers) == 1
    assert "query.container-title=Nature+Methods" in asked[0]
    assert "filter=from-pub-date%3A2023-01-01" in asked[0]


def test_fetch_topic_collects_across_clients_and_records_failures():
    class Boom:
        name, offline = "pubmed", False

        def search(self, *a, **k):
            raise research.httpx.ConnectError("down")

    fetch, _ = _fetcher({"export.arxiv.org": "arxiv_search.xml"})
    clients = {"arxiv": research.ArxivSearch(fetch=fetch), "pubmed": Boom()}
    out = research.fetch_topic(research.Topic(("arxiv", "pubmed"), "x"), clients)
    assert len(out.papers) == 2 and len(out.entries) == 2 and not out.offline
    assert out.errors == ["pubmed: ConnectError: down"]
    null = research.fetch_topic(research.Topic(("arxiv",), "x"), research.null_clients())
    assert null.offline and null.papers == []
```

Commit the empty-esearch fixture as a real file too (`tests/fixtures/research/pubmed_esearch_empty.xml`, same content) and drop the `write_text` line once it exists.

- [ ] **Step 3: Run to verify they fail**

Run: `uv run --frozen pytest tests/test_research.py -q`
Expected: FAIL (`parse_arxiv` missing, stand-ins return `[]`).

- [ ] **Step 4: Implement the clients**

Replace the three stand-ins at the bottom of `research.py` with:

```python
# ---------------------------------------------------------------------------
# the clients: one injected fetch, pure parsers, paced
# ---------------------------------------------------------------------------

import httpx  # noqa: E402 -- grouped with the clients that use it
from defusedxml import ElementTree as SafeET  # noqa: E402

from attestation.library_readers import (  # noqa: E402
    _client,
    _crossref_abstract,
    _crossref_authors,
)

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"
_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1
)}
FETCH_ERRORS = (httpx.HTTPError, httpx.InvalidURL, ValueError, SafeET.ParseError)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _default_fetch(url: str) -> bytes:
    """GET a URL through the library's client (redirects followed); raises httpx.HTTPError."""
    with _client() as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def _collapse(text: str | None) -> str:
    return " ".join((text or "").split())


class _Client:
    """Shared: the injected fetch, the request log, and pacing between requests."""

    name = ""
    offline = False
    pace_seconds = 0.0

    def __init__(self, fetch=None):
        self.fetch = fetch or _default_fetch
        self.requests: list[str] = []
        self._last: float | None = None

    def _get(self, url: str) -> bytes:
        if self._last is not None and self.pace_seconds:
            wait = self.pace_seconds - (time.monotonic() - self._last)
            if wait > 0:
                _sleep(wait)
        self.requests.append(url)
        self._last = time.monotonic()
        return self.fetch(url)


def parse_arxiv(body: bytes) -> list[Paper]:
    """arXiv Atom search results as Papers; versionless ids; venue stays None."""
    from attestation.library import normalise_arxiv

    out = []
    for entry in SafeET.fromstring(body).iter(f"{_ATOM}entry"):
        aid = normalise_arxiv((entry.findtext(f"{_ATOM}id") or "").partition("/abs/")[2])
        if not aid:
            continue
        doi_el = entry.find(f"{_ARXIV_NS}doi")
        published = entry.findtext(f"{_ATOM}published") or ""
        out.append(
            Paper(
                client="arxiv",
                external_id=aid,
                title=_collapse(entry.findtext(f"{_ATOM}title")),
                abstract=_collapse(entry.findtext(f"{_ATOM}summary")),
                authors=tuple(
                    _family_first(a.findtext(f"{_ATOM}name") or "")
                    for a in entry.iter(f"{_ATOM}author")
                ),
                published=published[:10] or None,
                doi=doi_el.text.strip() if doi_el is not None and doi_el.text else None,
                arxiv_id=aid,
                url=f"https://arxiv.org/abs/{aid}",
                pdf_url=f"https://arxiv.org/pdf/{aid}",
            )
        )
    return out


def _family_first(name: str) -> str:
    """'Simon Batzner' -> 'Batzner, Simon'; a single token or 'Family, Given' stays."""
    name = _collapse(name)
    if "," in name or " " not in name:
        return name
    given, _, family = name.rpartition(" ")
    return f"{family}, {given}"


class ArxivSearch(_Client):
    """The arXiv export API; 3 s between requests, as arXiv asks."""

    name = "arxiv"
    pace_seconds = 3.0

    def search(self, query, *, journal=None, since: date | None = None, limit: int = 50):
        """Papers matching every word of `query`, newest first, submitted since `since`."""
        terms = " AND ".join(f"all:{w}" for w in query.split())
        if since:
            terms += f" AND submittedDate:[{since:%Y%m%d}0000 TO 209912312359]"
        params = {
            "search_query": terms,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": limit,
        }
        url = "https://export.arxiv.org/api/query?" + urlencode(params, quote_via=quote_plus)
        return parse_arxiv(self._get(url))


def parse_pubmed_ids(body: bytes) -> list[str]:
    """PMIDs from an esearch response."""
    return [e.text for e in SafeET.fromstring(body).iter("Id") if e.text]


def _pubmed_date(article) -> str | None:
    node = article.find(".//PubmedData/History/PubMedPubDate[@PubStatus='pubmed']")
    if node is None:
        node = article.find(".//Article/Journal/JournalIssue/PubDate")
    if node is None:
        return None
    year = node.findtext("Year")
    if not year or not year.isdigit():
        return None
    month = (node.findtext("Month") or "1").strip().lower()
    m = int(month) if month.isdigit() else _MONTHS.get(month[:3], 1)
    day = node.findtext("Day") or "1"
    return f"{int(year):04d}-{m:02d}-{int(day):02d}"


def _pubmed_authors(article) -> tuple[str, ...]:
    out = []
    for a in article.iter("Author"):
        family, given, group = a.findtext("LastName"), a.findtext("ForeName"), a.findtext("CollectiveName")
        if family:
            out.append(f"{family}, {given}" if given else family)
        elif group:
            out.append(group)
    return tuple(out)


def parse_pubmed(body: bytes) -> list[Paper]:
    """efetch (rettype=abstract, retmode=xml) records as Papers."""
    out = []
    for article in SafeET.fromstring(body).iter("PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = _collapse(article.findtext(".//ArticleTitle"))
        if not pmid or not title:
            continue
        ids = {a.get("IdType"): (a.text or "").strip() for a in article.iter("ArticleId")}
        out.append(
            Paper(
                client="pubmed",
                external_id=pmid,
                title=title.rstrip("."),
                abstract=" ".join(_collapse(t.text) for t in article.iter("AbstractText") if t.text),
                authors=_pubmed_authors(article),
                published=_pubmed_date(article),
                doi=ids.get("doi") or None,
                pmcid=ids.get("pmc") or None,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                venue=_collapse(article.findtext(".//Journal/Title")) or None,
            )
        )
    return out


class PubmedSearch(_Client):
    """NCBI E-utilities: esearch for ids, efetch for records. 3 req/s, 10 with a key."""

    name = "pubmed"
    _BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def __init__(self, fetch=None, api_key: str | None = None):
        super().__init__(fetch)
        self.api_key = api_key
        self.pace_seconds = 0.1 if api_key else 0.34

    def _params(self, **params) -> str:
        if self.api_key:
            params["api_key"] = self.api_key
        return urlencode(params, quote_via=quote_plus)

    def search(self, query, *, journal=None, since: date | None = None, limit: int = 50):
        """Papers for `query` (AND a `[Journal]` term), entered since `since`."""
        term = f"({query})" if journal else query
        if journal:
            term += f' AND "{journal}"[Journal]'
        params = {"db": "pubmed", "term": term, "retmax": limit, "sort": "date"}
        if since:
            params.update({"datetype": "edat", "mindate": f"{since:%Y/%m/%d}", "maxdate": "3000"})
        ids = parse_pubmed_ids(self._get(self._BASE + "esearch.fcgi?" + self._params(**params)))
        if not ids:
            return []
        fetch_params = {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml"}
        return parse_pubmed(self._get(self._BASE + "efetch.fcgi?" + self._params(**fetch_params)))


def _crossref_date(item: dict) -> str | None:
    parts = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
    if not parts or not parts[0]:
        return None
    y, m, d = (list(parts) + [1, 1])[:3]
    return f"{int(y):04d}-{int(m or 1):02d}-{int(d or 1):02d}"


def parse_crossref(body: bytes) -> list[Paper]:
    """`/works` items as Papers; an item with no title cannot name a paper and is skipped."""
    import json

    items = ((json.loads(body) or {}).get("message") or {}).get("items") or []
    out = []
    for item in items:
        doi = (item.get("DOI") or "").lower()
        title = _collapse((item.get("title") or [""])[0])
        if not doi or not title:
            continue
        venue = _collapse((item.get("container-title") or [""])[0]) or None
        out.append(
            Paper(
                client="crossref",
                external_id=doi,
                title=title,
                abstract=_crossref_abstract(item) or "",
                authors=tuple(_crossref_authors(item)),
                published=_crossref_date(item),
                doi=doi,
                url=item.get("URL") or f"https://doi.org/{doi}",
                venue=venue,
            )
        )
    return out


class CrossrefSearch(_Client):
    """CrossRef `/works`: the only way to search a journal on neither arXiv nor PubMed."""

    name = "crossref"

    def search(self, query, *, journal=None, since: date | None = None, limit: int = 50):
        """Papers matching `query`, optionally within `journal`, published since `since`."""
        params: dict = {"query": query, "rows": limit, "sort": "published", "order": "desc"}
        if journal:
            params["query.container-title"] = journal
        if since:
            params["filter"] = f"from-pub-date:{since.isoformat()}"
        url = "https://api.crossref.org/works?" + urlencode(params, quote_via=quote_plus)
        return parse_crossref(self._get(url))


@dataclass
class Fetched:
    """What one topic run produced: entries for ingest, papers for the library, and why not."""

    entries: list[dict]
    papers: list[Paper]
    errors: list[str]
    offline: bool


def fetch_topic(topic: Topic, clients: dict, *, since: date | None = None, limit: int = 50) -> Fetched:
    """Run every client a topic names. A client's failure is one error string,
    never an exception: the other clients and the other feeds still run."""
    papers: list[Paper] = []
    errors: list[str] = []
    for name in topic.clients:
        client = clients[name]
        try:
            papers.extend(client.search(topic.query, journal=topic.journal, since=since, limit=limit))
        except FETCH_ERRORS as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    offline = all(clients[n].offline for n in topic.clients)
    return Fetched(as_entries(papers), papers, errors, offline)
```

Move the `import httpx`, `SafeET` and `library_readers` imports to the top of the module (with the other imports, in ruff's order) and delete the `noqa: E402` markers; they were written inline above only to show what the clients need. Check `_crossref_abstract(item)` and `_crossref_authors(item)` signatures in `library_readers.py` (both take the CrossRef message dict) before relying on them.

Amend the spec table (`docs/superpowers/specs/2026-09-10-paper-research-design.md`, the `crossref` row): `from-index-date` → `from-pub-date`, with the reason: a topic asks for papers *published* since the last run; index date would resurface old papers CrossRef only recently indexed.

- [ ] **Step 5: Run**

Run: `uv run --frozen pytest tests/test_research.py -q && uv run --frozen ruff check src/attestation/research.py && uv run --frozen ruff format --check src/attestation/research.py`
Expected: PASS, clean.

- [ ] **Step 6: Commit**

```bash
git add src/attestation/research.py tests/test_research.py tests/fixtures/research docs/superpowers/specs/2026-09-10-paper-research-design.md
git commit -m "Search clients for arXiv, PubMed and CrossRef: pure parsers over committed fixture payloads, one injected fetch, pacing per client, and fetch_topic that turns a failing client into one error string rather than a lost run

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `add_source` accepts a topic; `added_by`; `list_sources` kinds

**Files:**
- Modify: `src/attestation/feeds.py` (`add_source`, new `_register`, `list_sources`)
- Modify: `src/attestation/mcp/subscriptions.py` (`feed.source_add(user=)`)
- Test: `tests/test_feeds.py`

**Interfaces:**
- Consumes: `research.is_topic_url`, `research.parse_topic`, `research.TopicError` (Task 2)
- Produces: `add_source(conn, url, title=None, parse=feedparser.parse, *, added_by: int | None = None) -> (feed_id, message)`; `list_sources` rows gain `"kind": "rss" | "research"` and `"added_by": str | None` (persona name); `_add_feed(conn, url, title=None, user=None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_feeds.py`:

```python
def _no_network(url):
    raise AssertionError(f"parse() must not be called for {url}")


def test_add_research_topic_registers_without_network_and_canonicalises(conn):
    feed_id, message = feeds.add_source(conn, "research:arxiv,pubmed?q=graph  neural+networks", parse=_no_network)
    row = conn.execute("SELECT url, title FROM feeds WHERE id = ?", (feed_id,)).fetchone()
    assert row["url"] == "research:arxiv,pubmed?q=graph+neural+networks"
    assert row["title"] == "arxiv,pubmed: graph neural networks"
    assert "next ingest" in message
    again, msg2 = feeds.add_source(conn, "research:arxiv,pubmed?q=graph+neural+networks", parse=_no_network)
    assert again == feed_id and "already" in msg2


def test_add_research_topic_refuses_bad_urls_as_feed_errors(conn):
    with pytest.raises(feeds.FeedError, match="unknown research client"):
        feeds.add_source(conn, "research:scholar?q=x", parse=_no_network)
    assert conn.execute("SELECT COUNT(*) n FROM feeds WHERE url LIKE 'research:%'").fetchone()["n"] == 0


def test_added_by_and_kind_are_listed(conn):
    uid = conn.execute("SELECT id FROM users ORDER BY id LIMIT 1").fetchone()["id"]
    name = conn.execute("SELECT name FROM users WHERE id = ?", (uid,)).fetchone()["name"]
    feeds.add_source(conn, "research:arxiv?q=x", parse=_no_network, added_by=uid)
    feeds.add_source(conn, "http://example.com/rss", parse=_parse_ok)
    by_url = {s["url"]: s for s in feeds.list_sources(conn)}
    assert by_url["research:arxiv?q=x"]["kind"] == "research"
    assert by_url["research:arxiv?q=x"]["added_by"] == name
    assert by_url["http://example.com/rss"]["kind"] == "rss"
    assert by_url["http://example.com/rss"]["added_by"] is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen pytest tests/test_feeds.py -q`
Expected: FAIL (`parse()` called; `added_by` unexpected kwarg; no `kind`).

- [ ] **Step 3: Implement**

In `src/attestation/feeds.py`, replace `add_source` with:

```python
def add_source(
    conn: sqlite3.Connection,
    url: str,
    title: str | None = None,
    parse=feedparser.parse,
    *,
    added_by: int | None = None,
) -> tuple[int, str]:
    """Register a feed after checking it parses. Does NOT ingest its items.

    A `research:` URL (a standing topic, see research.parse_topic) is
    validated by parsing the URL itself -- no network at registration -- and
    stored in its canonical spelling so two spellings are one row. `added_by`
    is the registering persona's id: provenance, not scoping.

    Returns (feed_id, message). Raises FeedError if the URL does not parse
    as a feed -- a caller-fixable refusal, not a bug.
    """
    from attestation import research

    if research.is_topic_url(url):
        try:
            topic = research.parse_topic(url)
        except research.TopicError as exc:
            raise FeedError(str(exc)) from exc
        feed_id, existed = _register(conn, topic.url, title or topic.title, added_by)
        if existed:
            return feed_id, f"already tracking {topic.url}"
        return feed_id, (
            f"tracking {title or topic.title!r}. Papers appear after the next ingest "
            "(run `attest ingest`, or wait for the hourly refresh)."
        )

    parsed = parse(url)
    if not _looks_like_feed(parsed):
        raise FeedError(f"{url} did not parse as an RSS/Atom feed; nothing was added")
    resolved_title = title or (getattr(parsed, "feed", None) or {}).get("title") or url
    feed_id, existed = _register(conn, url, resolved_title, added_by)
    if existed:
        return feed_id, f"already subscribed to {url}"
    return feed_id, (
        f"subscribed to {resolved_title!r}. Items appear after the next ingest "
        "(run `attest ingest`, or wait for the hourly refresh)."
    )


def _register(conn, url: str, title: str, added_by: int | None) -> tuple[int, bool]:
    """Insert the row; (feed_id, already_existed). Owns the race the old
    add_source handled inline: a subscriber between the check and the write
    is the outcome the caller asked for, not an error."""
    existing = conn.execute("SELECT id FROM feeds WHERE url = ?", (url,)).fetchone()
    if existing is not None:
        return existing["id"], True
    try:
        cur = conn.execute(
            "INSERT INTO feeds(url, title, added_by) VALUES (?, ?, ?)", (url, title, added_by)
        )
        conn.commit()
        feed_id = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.rollback()
        raced = conn.execute("SELECT id FROM feeds WHERE url = ?", (url,)).fetchone()
        if raced is None:
            raise
        return raced["id"], True
    if feed_id is None:
        raise FeedError(f"insert for {url} did not return a row id")
    return feed_id, False
```

(The original race comment block explains why; keep its substance in `_register`'s docstring.) In `list_sources`, change the SQL and row to:

```python
    rows = conn.execute(
        "SELECT f.id, f.title, f.url, f.last_fetched, u.name AS added_by,"
        " COUNT(i.id) AS item_count"
        " FROM feeds f LEFT JOIN items i ON i.feed_id = f.id"
        " LEFT JOIN users u ON u.id = f.added_by"
        " GROUP BY f.id ORDER BY f.title"
    ).fetchall()
    return [
        {
            "feed_id": r["id"],
            "title": r["title"],
            "url": r["url"],
            "kind": "research" if r["url"].startswith("research:") else "rss",
            "added_by": r["added_by"],
            "item_count": r["item_count"],
            "last_fetched": r["last_fetched"],
        }
        for r in rows
    ]
```

In `src/attestation/mcp/subscriptions.py`:

```python
    @mcp.tool(name="feed.source_add")
    def add_feed(url: str, title: str | None = None, user: str | None = None) -> dict:
        """Subscribe to an RSS/Atom feed, or register a standing research topic.

        An RSS URL is validated by parsing it, then registered. A topic is a
        `research:` URL -- `research:arxiv,pubmed?q=graph+neural+networks`, or
        `research:crossref?q=...&journal=Nature+Methods` -- validated without any
        network call; every hourly ingest then searches arXiv, PubMed and/or
        CrossRef for it and the hits enter the feed like any item. Does NOT fetch
        anything now: items appear after the next ingest. `user` records who
        asked (feed.sources shows it); the feed itself is shared by every reader.
        """
        return _add_feed(url, title, user)
```

and

```python
@tool(empty={"feed_id": None}, label="add_feed")
def _add_feed(conn, url: str, title: str | None = None, user: str | None = None) -> dict:
    from attestation import feeds as feeds_mod

    added_by = None
    if user:
        row = conn.execute("SELECT id FROM users WHERE name = ?", (user,)).fetchone()
        added_by = row["id"] if row else None
    try:
        feed_id, message = feeds_mod.add_source(conn, url, title, added_by=added_by)
    except feeds_mod.FeedError as exc:
        raise ToolError(str(exc)) from exc
    return {"message": message, "feed_id": feed_id}
```

Update `list_feeds`'s docstring: "List subscribed feeds and research topics (`kind`), with item counts, who added each, and when each was last fetched."

- [ ] **Step 4: Run**

Run: `uv run --frozen pytest tests/test_feeds.py tests/test_mcp_server.py tests/test_tool_envelope.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/attestation/feeds.py src/attestation/mcp/subscriptions.py tests/test_feeds.py
git commit -m "A standing topic is a feed: add_source registers research: URLs by parsing the URL (no network), canonicalised so two spellings are one row; added_by records who asked and list_sources says which kind each row is

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Ingest dispatches on scheme, dedups by identifier, fills the library

**Files:**
- Modify: `src/attestation/ingest.py` (`_exists`, `_new_entries`, `_ingest_outcome`, `run_ingest`, new `_fetch_feed`, `_entry_ids`, `_store_references`)
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `research.is_topic_url`, `parse_topic`, `fetch_topic`, `as_records`, `clients_from_env`, `null_clients` (Tasks 2-3); `library.upsert` (Task 1)
- Produces: `run_ingest(conn, embedder, feeds_path, parse=feedparser.parse, *, clients=None) -> dict`; stats gain `research_disabled: n` when any topic was skipped for the flag; entries may carry `doi`/`arxiv_id` keys that beat `extract_ids`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ingest.py`:

```python
from attestation import research


def _topic_clients(papers):
    class Fixed:
        name, offline = "arxiv", False

        def __init__(self):
            self.calls = []

        def search(self, query, *, journal=None, since=None, limit=50):
            self.calls.append((query, since))
            return list(papers)

    fixed = Fixed()
    return {"arxiv": fixed, "pubmed": research.NullClient("pubmed"), "crossref": research.NullClient("crossref")}, fixed


def _paper(aid="2101.03164", title="NequIP", doi=None):
    return research.Paper(
        client="arxiv", external_id=aid, title=title, abstract="Equivariant potentials.",
        authors=("Batzner, Simon",), published="2021-01-08", doi=doi, arxiv_id=aid,
        url=f"https://arxiv.org/abs/{aid}",
    )


def test_research_feed_dispatches_to_the_client_not_parse(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    feeds = write_feeds_toml(tmp_path, ["research:arxiv?q=equivariant+force+fields"])
    clients, fixed = _topic_clients([_paper()])

    def parse_never(url):
        raise AssertionError("feedparser must not see a research: URL")

    stats = run_ingest(conn, fake_embedder, feeds, parse=parse_never, clients=clients)
    assert stats == {"added": 1, "skipped": 0, "failed_feeds": 0}
    assert fixed.calls[0] == ("equivariant force fields", None)  # first run: no since
    row = conn.execute("SELECT guid, published, doi, arxiv_id, summary FROM items").fetchone()
    assert row["guid"] == "arxiv:2101.03164" and row["published"].startswith("2021-01-08")
    assert row["arxiv_id"] == "2101.03164" and row["summary"] == "Equivariant potentials."
    # the same run left one reference with a research source row carrying the authors
    ref = conn.execute('SELECT id, authors FROM "references"').fetchone()
    assert '"Batzner, Simon"' in ref["authors"]
    src = conn.execute("SELECT source, source_key FROM reference_sources").fetchone()
    assert (src["source"], src["source_key"]) == ("research:arxiv", "2101.03164")
    # hourly idempotency, and since = last_fetched on the second run
    stats2 = run_ingest(conn, fake_embedder, feeds, parse=parse_never, clients=clients)
    assert stats2 == {"added": 0, "skipped": 1, "failed_feeds": 0}
    assert fixed.calls[1][1] is not None
    assert conn.execute("SELECT COUNT(*) c FROM reference_sources").fetchone()["c"] == 1


def test_identifier_dedup_skips_a_cross_list_and_a_topic_twin(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    feeds = write_feeds_toml(
        tmp_path, ["https://arxiv.example/rss", "research:arxiv?q=x"]
    )
    # the RSS fixture's first item is oai:arXiv.org:2608.00001v1 -> arxiv_id 2608.00001
    clients, _ = _topic_clients([_paper(aid="2608.00001", title="Paper One (API abstract differs)")])
    stats = run_ingest(conn, fake_embedder, feeds, parse=fake_parse, clients=clients)
    n = conn.execute("SELECT COUNT(*) c FROM items WHERE arxiv_id = '2608.00001'").fetchone()["c"]
    assert n == 1 and stats["skipped"] >= 1


def test_disabled_research_is_counted_not_failed(tmp_path, fake_embedder, monkeypatch):
    conn = get_db(tmp_path / "t.db")
    feeds = write_feeds_toml(tmp_path, ["research:arxiv?q=x", "https://blog.example/rss"])
    monkeypatch.setenv("ATTEST_RESEARCH_WEB", "0")
    stats = run_ingest(conn, fake_embedder, feeds, parse=fake_parse)
    assert stats["failed_feeds"] == 0 and stats["research_disabled"] == 1 and stats["added"] >= 1


def test_a_failing_client_is_one_feed_error_and_rss_still_ingests(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    feeds = write_feeds_toml(tmp_path, ["research:arxiv?q=x", "https://blog.example/rss"])

    class Boom:
        name, offline = "arxiv", False

        def search(self, *a, **k):
            raise research.httpx.ConnectError("down")

    clients = {"arxiv": Boom(), "pubmed": research.NullClient("pubmed"), "crossref": research.NullClient("crossref")}
    stats = run_ingest(conn, fake_embedder, feeds, parse=fake_parse, clients=clients)
    assert stats["failed_feeds"] == 1 and stats["added"] >= 1
```

Check the RSS fixture's first guid: `grep -m1 '<guid\|<id>' tests/fixtures/arxiv.xml`. If it is not `oai:arXiv.org:2608.00001v1`, use the id it does carry in the second test.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen pytest tests/test_ingest.py -q`
Expected: FAIL (`clients` unexpected kwarg; `parse_never` raised).

- [ ] **Step 3: Implement**

In `src/attestation/ingest.py`:

```python
def _entry_ids(entry) -> tuple[str | None, str | None]:
    """(doi, arxiv_id): what the entry states (a research hit carries both keys),
    else what its guid/url imply."""
    doi, arxiv_id = extract_ids(entry.get("id"), entry.get("link"))
    return entry.get("doi") or doi, entry.get("arxiv_id") or arxiv_id


def _exists(conn, feed_id: int, guid: str | None, chash: str, doi=None, arxiv_id=None) -> bool:
    if guid is not None:
        row = conn.execute(
            "SELECT 1 FROM items WHERE feed_id = ? AND guid = ?", (feed_id, guid)
        ).fetchone()
        if row:
            return True
    if conn.execute("SELECT 1 FROM items WHERE content_hash = ?", (chash,)).fetchone():
        return True
    # Same paper under ANY feed: a cs.LG/chem-ph cross-list (207 of 7,787
    # identified items, measured 2026-09-05) or a topic hit the RSS already
    # carried with a slightly different abstract. The library merges these
    # into one reference; this stops feed.list showing one paper twice.
    for column, value in (("doi", doi), ("arxiv_id", arxiv_id)):
        if value and conn.execute(f"SELECT 1 FROM items WHERE {column} = ?", (value,)).fetchone():
            return True
    return False
```

In `_new_entries`, add `seen_ids: set[str] = set()` beside the other sets; compute `doi, arxiv_id = _entry_ids(entry)` after `guid = entry.get("id")`; extend `duplicate_in_batch` with `or any(v in seen_ids for v in (doi, arxiv_id) if v)`; pass `doi, arxiv_id` to `_exists`; after the guid/hash `add` calls, `seen_ids.update(v for v in (doi, arxiv_id) if v)`.

In the insert loop of `run_ingest`, replace `doi, arxiv_id = extract_ids(guid, entry.get("link"))` with `doi, arxiv_id = _entry_ids(entry)`.

Add the dispatch helpers:

```python
def _fetch_feed(feed, parse, clients):
    """One feed's parsed payload: feedparser for an RSS URL, the research
    clients for a `research:` one (entries feedparser-shaped, plus `.papers`
    for the library). Returns None when research is disabled for this run."""
    from datetime import date

    from attestation import research

    if not research.is_topic_url(feed["url"]):
        return parse(feed["url"])
    topic = research.parse_topic(feed["url"])
    if all(clients[n].offline for n in topic.clients):
        return None
    since = date.fromisoformat(feed["last_fetched"][:10]) if feed["last_fetched"] else None
    fetched = research.fetch_topic(topic, clients, since=since)
    if fetched.errors and not fetched.papers:
        raise RuntimeError("; ".join(fetched.errors))
    for err in fetched.errors:
        log.warning("research client failed: %s -- %s", feed["url"], err)
    return fetched


def _store_references(conn, parsed) -> None:
    """After a topic's items are committed, the same hits enter the library
    under `research:<client>` with the authors and venue the payload had --
    the spec's stated exception to 'the wire never introduces a reference'."""
    from datetime import UTC, datetime

    from attestation import library, research

    papers = getattr(parsed, "papers", None)
    if not papers:
        return
    today = datetime.now(UTC).date().isoformat()
    for rec in research.as_records(papers, today):
        library.upsert(conn, rec)
    conn.commit()
```

In `run_ingest`: signature becomes `def run_ingest(conn, embedder, feeds_path, parse=feedparser.parse, *, clients=None) -> dict:`; after `sync_feeds(...)` add:

```python
    if clients is None:
        from attestation import research

        clients = research.clients_from_env()  # the flag is read here, once per run
```

Inside the `try`, replace `parsed = parse(feed["url"])` with:

```python
            parsed = _fetch_feed(feed, parse, clients)
            if parsed is None:
                outcomes.append(
                    {"feed": feed["url"], "new": 0, "skipped": 0, "error": None,
                     "embedder_down": False, "research_disabled": True}
                )
                continue
```

After `conn.commit()` (the one before `outcomes.append({... "new": added_here ...})`) add `_store_references(conn, parsed)`.

In `_ingest_outcome`, after the `embedder_down` block:

```python
    disabled = sum(1 for o in outcomes if o.get("research_disabled"))
    if disabled:
        stats["research_disabled"] = disabled
```

and extend its docstring: "`research_disabled` counts topics skipped because `ATTEST_RESEARCH_WEB` is off (or `--no-research`); present only when nonzero, and never a failure."

Update `run_ingest`'s docstring with one sentence: "A `research:` feed is searched through `clients` (built once here from the flag) instead of parsed, and its hits also enter the reference library."

- [ ] **Step 4: Run**

Run: `uv run --frozen pytest tests/test_ingest.py tests/test_library.py tests/test_cli.py -q && uv run --frozen python scripts/check_complexity.py`
Expected: PASS; complexity under the file's threshold (if `run_ingest` trips it, move the outcome-append for the disabled case into a helper `_disabled_outcome(url) -> dict`).

- [ ] **Step 5: Commit**

```bash
git add src/attestation/ingest.py tests/test_ingest.py
git commit -m "Ingest dispatches on scheme: a research: feed is searched, not parsed, its hits go through the unchanged three passes as items and into the library as references with the payload's authors; items dedup by DOI/arXiv id across feeds (the 207 measured cross-lists, and a topic hit the RSS already carried)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: The `feed.research` tool, counts, response size

**Files:**
- Create: `src/attestation/mcp/research.py`
- Create: `tests/test_research_tools.py`
- Modify: `src/attestation/mcp/__init__.py` (register; feed `goal`)
- Modify: every doc quoting 48 / feed 21: `CLAUDE.md:5,45,50`, `README.md:88`, `docs/guides/agents.md:64,416` (+ a row in the feed tool table near line 89), `docs/concepts.md:71`, `docs/architecture/research-profile.md:17`, `docs/architecture/structure-and-integration-points.md:85`
- Test: `tests/test_architecture.py`, `tests/test_agent_surfaces.py`, `tests/test_response_size.py`

**Interfaces:**
- Produces: tool `feed.research(query, sources="arxiv,pubmed", journal=None, since_days=365, limit=5, store=True)`; impl `_research(query, sources=..., journal=None, since_days=365, limit=5, store=True) -> dict` with keys `papers, n_found, stored, offline, errors`; module globals `CLIENTS` (built in `register`) and `MAX_RESEARCH_LIMIT = 8`; `paper_row(paper, stored) -> dict`.

- [ ] **Step 1: Find how `register_all` wires modules**

Run: `grep -n "subscriptions\|register(" src/attestation/mcp/__init__.py | head`. The new module is registered the same way, right after `subscriptions`.

- [ ] **Step 2: Write the failing tests**

`tests/test_research_tools.py`:

```python
"""feed.research: envelope, storing to the library, offline honesty, payload size."""

import json

from test_response_size import HARD_RESPONSE_CEILING

from attestation import research
from attestation.db import get_db
from attestation.mcp import research as tool_mod


def _db(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("ATTEST_DB", str(db))
    get_db(db).close()
    return db


def _paper(i: int):
    return research.Paper(
        client="arxiv", external_id=f"2101.{i:05d}", title=f"Paper {i} " + "x" * 200,
        abstract="A" * 500, authors=tuple(f"Author{j}, Given" for j in range(12)),
        published="2021-01-08", arxiv_id=f"2101.{i:05d}", url=f"https://arxiv.org/abs/2101.{i:05d}",
        venue="V" * 80,
    )


class Fixed:
    name, offline = "arxiv", False

    def __init__(self, papers):
        self.papers = papers

    def search(self, query, *, journal=None, since=None, limit=50):
        return self.papers[:limit]


def _clients(papers):
    return {"arxiv": Fixed(papers), "pubmed": research.NullClient("pubmed"), "crossref": research.NullClient("crossref")}


def test_research_stores_to_the_library_not_the_feed(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    monkeypatch.setattr(tool_mod, "CLIENTS", _clients([_paper(1)]))
    out = tool_mod._research("equivariant", sources="arxiv")
    assert out["ok"] and out["n_found"] == 1 and out["stored"] == 1 and out["offline"] is False
    assert out["papers"][0]["stored"] is True and out["papers"][0]["arxiv_id"] == "2101.00001"
    conn = get_db(db)
    assert conn.execute('SELECT COUNT(*) FROM "references"').fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0
    again = tool_mod._research("equivariant", sources="arxiv")
    assert again["stored"] == 0  # unchanged on the second call


def test_research_preview_does_not_store(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    monkeypatch.setattr(tool_mod, "CLIENTS", _clients([_paper(1)]))
    out = tool_mod._research("x", sources="arxiv", store=False)
    assert out["stored"] == 0 and out["papers"][0]["stored"] is False
    assert get_db(db).execute('SELECT COUNT(*) FROM "references"').fetchone()[0] == 0


def test_research_offline_says_so(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch)
    monkeypatch.setattr(tool_mod, "CLIENTS", research.null_clients())
    out = tool_mod._research("x")
    assert out["ok"] and out["offline"] is True and out["papers"] == []
    assert "ATTEST_RESEARCH_WEB" in out["message"]


def test_research_refuses_unknown_source_and_journal_on_arxiv(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch)
    monkeypatch.setattr(tool_mod, "CLIENTS", _clients([]))
    out = tool_mod._research("x", sources="scholar")
    assert out["ok"] is False and "arxiv, pubmed, crossref" in out["message"]
    out = tool_mod._research("x", sources="arxiv", journal="Nature")
    assert out["ok"] is False and "categories" in out["message"]


def test_research_payload_fits_at_the_cap(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch)
    monkeypatch.setattr(tool_mod, "CLIENTS", _clients([_paper(i) for i in range(20)]))
    out = tool_mod._research("x", sources="arxiv", limit=16, store=False)
    assert len(out["papers"]) == tool_mod.MAX_RESEARCH_LIMIT
    assert len(json.dumps(out, indent=2)) < HARD_RESPONSE_CEILING
```

Also in `tests/test_agent_surfaces.py::test_a_provenance_tool_is_absent_from_the_feed_agent`, add `assert "feed.research" in names`; and in `test_the_knowledge_agent_can_still_reach_items` add `assert "feed.research" not in names, "research is a feed-surface action"`.

- [ ] **Step 3: Run to verify they fail**

Run: `uv run --frozen pytest tests/test_research_tools.py tests/test_agent_surfaces.py -q`
Expected: FAIL (`attestation.mcp.research` missing).

- [ ] **Step 4: Implement the tool module**

`src/attestation/mcp/research.py`:

```python
"""Ad hoc paper research: the `feed.research` tool.

Searches arXiv, PubMed and CrossRef by query and stores the hits in the
reference library -- NOT the feed. The feed is what arrives on a schedule
(a standing topic is a `research:` feed, registered with feed.source_add);
the library is what the reader keeps, and an ad hoc question is the second
kind. Lives in the feed namespace because the chat surfaces run with
ATTEST_TOOLS=feed and must be able to reach it.

The clients are built ONCE, when `register` runs, so ATTEST_RESEARCH_WEB is
read at construction like every other network flag here.
"""

from datetime import UTC, date, datetime, timedelta
from typing import Annotated

from pydantic import Field

from attestation.mcp._shared import Limit, clamp_limit
from attestation.mcp._tool import ToolError, tool

# 8, not feed.list's 16: a research row carries authors and a venue that a
# feed row does not. Measured in test_research_tools: 8 worst-case rows plus
# the envelope stay under the 7000-char ceiling a 2B model can render.
MAX_RESEARCH_LIMIT = 8
DEFAULT_RESEARCH_LIMIT = 5
CLIENTS: dict | None = None


def _clients() -> dict:
    """The module's clients; built lazily for callers that skip `register` (tests, ask)."""
    global CLIENTS
    if CLIENTS is None:
        from attestation import research

        CLIENTS = research.clients_from_env()
    return CLIENTS


def paper_row(paper, stored: bool) -> dict:
    """One hit's wire row: the feed's clipping rules, 3 authors with the true count."""
    from attestation.rank import MAX_SOURCE_CHARS, MAX_URL_CHARS, _clip_field, _clip_title

    return {
        "title": _clip_title(paper.title),
        "authors": list(paper.authors[:3]),
        "n_authors": len(paper.authors),
        "venue": _clip_field(paper.venue, MAX_SOURCE_CHARS) if paper.venue else None,
        "published": paper.published,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "url": _clip_field(paper.url, MAX_URL_CHARS) if paper.url else None,
        "stored": stored,
    }


@tool(
    empty={"papers": [], "n_found": 0, "stored": 0, "offline": False, "errors": []},
    label="research",
)
def _research(
    conn,
    query: str,
    sources: str = "arxiv,pubmed",
    journal: str | None = None,
    since_days: int | None = 365,
    limit: int = DEFAULT_RESEARCH_LIMIT,
    store: bool = True,
) -> dict:
    from attestation import library, research

    limit = min(clamp_limit(limit), MAX_RESEARCH_LIMIT)
    names = tuple(s.strip() for s in sources.split(",") if s.strip())
    try:
        topic = research.parse_topic(research.topic_url(names, query, journal))
    except research.TopicError as exc:
        raise ToolError(str(exc)) from exc
    since = date.today() - timedelta(days=since_days) if since_days else None
    fetched = research.fetch_topic(topic, _clients(), since=since, limit=limit)
    stored = 0
    if store and fetched.papers:
        today = datetime.now(UTC).date().isoformat()
        for rec in research.as_records(fetched.papers, today):
            _rid, how = library.upsert(conn, rec)
            stored += how != "unchanged"
        conn.commit()
    if fetched.offline:
        message = (
            "research clients are off (ATTEST_RESEARCH_WEB=0 when the server started):"
            " nothing was searched. This is not 'no results'."
        )
    else:
        message = (
            f"{len(fetched.papers)} paper(s) from {', '.join(topic.clients)}"
            + (f" in {journal}" if journal else "")
            + (f"; {stored} new in the library" if store else "; preview only, nothing stored")
            + (f"; {len(fetched.errors)} client(s) failed" if fetched.errors else "")
        )
    return {
        "message": message,
        "papers": [paper_row(p, store) for p in fetched.papers[:limit]],
        "n_found": len(fetched.papers),
        "stored": stored,
        "offline": fetched.offline,
        "errors": fetched.errors,
    }


def register(mcp) -> None:
    """Attach feed.research; build the clients now (flag read at construction)."""
    from attestation import research

    global CLIENTS
    CLIENTS = research.clients_from_env()

    @mcp.tool(name="feed.research")
    def research_papers(
        query: Annotated[str, Field(min_length=2, description="what the papers are about")],
        sources: Annotated[
            str, Field(description="comma-separated subset of arxiv, pubmed, crossref")
        ] = "arxiv,pubmed",
        journal: Annotated[
            str | None, Field(description="journal name; pubmed/crossref only", max_length=120)
        ] = None,
        since_days: Annotated[int | None, Field(ge=1, le=3650)] = 365,
        limit: Limit = DEFAULT_RESEARCH_LIMIT,
        store: bool = True,
    ) -> dict:
        """Search arXiv, PubMed and/or CrossRef for papers on a topic, and keep them.

        Goes to the network (unlike feed.search, which searches what already
        arrived). Hits are stored in the reference library -- cite.lookup and
        cite.search find them afterwards by DOI, arXiv id or topic -- and are
        NOT added to the feed; to follow a topic on the hourly schedule, call
        feed.source_add with `research:arxiv,pubmed?q=<topic>`. `store=false`
        previews. `limit` is capped at 8. **Show the reader each `url`.**
        `offline: true` means the clients are disabled, not that nothing matched.
        """
        return _research(query, sources, journal, since_days, limit, store)
```

Also in `src/attestation/mcp/citation.py::_sources`: `from attestation import research` beside the other imports, `network = network or research.research_enabled()`, and add `"research": research.research_enabled()` to the returned dict, so `cite.sources`' `offline` reflects the research flag as the spec says. Extend `cite.sources`'s docstring's flag list with `ATTEST_RESEARCH_WEB (feed.research, research: feeds, full text; on by default)`. Add to `tests/test_library_tools.py`:

```python
def test_cite_sources_offline_reflects_the_research_flag(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch)
    monkeypatch.setenv("ATTEST_RESEARCH_WEB", "0")
    assert citation._sources()["offline"] is True
    monkeypatch.delenv("ATTEST_RESEARCH_WEB")
    out = citation._sources()
    assert out["offline"] is False and out["research"] is True
```

Confirm `_clip_field` and `_clip_title` exist in `rank.py` with those names (`grep -n "def _clip" src/attestation/rank.py`); `SearchHit.to_row` in `library.py` already imports them the same way.

Register it in `src/attestation/mcp/__init__.py` next to `subscriptions` (same call shape). In the feed `Surface.goal`, replace `" network call. Reach for feed.ask when the question is in plain"` with `" network call. When they want papers the feed has NOT seen -- on arXiv, PubMed or a named journal -- feed.research goes and looks, and feed.source_add with a research: URL follows a topic every hour. Reach for feed.ask when the question is in plain"`.

- [ ] **Step 5: Move every count**

- `CLAUDE.md:5` `48 MCP tools` → `49 MCP tools`; `:45` `feed 21` → `feed 22` and `all 48` → `all 49`; `:50` `48 tools by default (2026-09-05, cite.sync and cite.related added) NAMESPACED as feed.*(20)` → `49 tools by default (2026-09-10, feed.research added) NAMESPACED as feed.*(21)`.
- `README.md:88` 48 → 49. `docs/concepts.md:71` 48 → 49. `docs/architecture/research-profile.md:17` 48 → 49. `docs/architecture/structure-and-integration-points.md:85` `48 tools` → `49 tools` and `` `feed.*` 20 `` → `` `feed.*` 21 ``.
- `docs/guides/agents.md:64` 48 → 49; `:416` `all 48` → `all 49`; add after the `feed.source_add` row: ``| `feed.research(query, sources, journal, since_days, limit, store)` | Search arXiv/PubMed/CrossRef and store hits in the library (network) | slow |`` and change the `feed.source_add` row's text to `Subscribe to a feed or register a research: topic (register-only; items arrive at the next ingest)`.
- Re-measure: `uv run python -c "from mcp.server.fastmcp import FastMCP; from attestation.mcp import register_all; import asyncio; m=FastMCP('x'); register_all(m); print(len(asyncio.run(m.list_tools())))"` must print 49, and with `ATTEST_TOOLS=feed ATTEST_EXPAND=1` must print 22.

- [ ] **Step 6: Run the guards**

Run: `uv run --frozen pytest tests/test_research_tools.py tests/test_agent_surfaces.py tests/test_architecture.py tests/test_response_size.py tests/test_tool_envelope.py tests/test_mcp_server.py tests/test_docs_site.py -q`
Expected: PASS. If the payload test fails on size, lower `MAX_RESEARCH_LIMIT` until it passes and record the measured number in its comment.

- [ ] **Step 7: Full gate, then commit**

Run: `uv run --frozen pre-commit run --all-files`
Expected: all Passed.

```bash
git add -A src/attestation/mcp tests/test_research_tools.py tests/test_agent_surfaces.py CLAUDE.md README.md docs
git commit -m "feed.research: search arXiv/PubMed/CrossRef by query and keep the hits in the library, not the feed; clients built once at register time; 49 tools, feed surface 22, every doc that counts them moved

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Routing: research and track, executed by `feed.ask`

**Files:**
- Modify: `src/attestation/mcp/routing.py` (`route_feed`, new phrase tables, `_research_query`)
- Modify: `src/attestation/mcp/ask.py` (`_feed_ask` dispatch)
- Test: `tests/test_ask_routing.py`, `tests/test_research_tools.py`

**Interfaces:**
- Consumes: `tool_mod._research(query, sources=...)` (Task 6), `subs._add_feed(url, title, user)` (Task 4)
- Produces: `Decision("feed.research", {"query": ..., "sources": ...})`; `Decision("feed.source_add", {"url": "research:arxiv,pubmed?q=..."})`

**Routing rule (a deliberate narrowing of the spec's "find papers on X" line, recorded in the docs task):** "papers on X" stays on `feed.search`. The feed surface's own goal text promises "find me recent papers on X" is answered locally, `test_ask_routing.py` pins "anything new on retrieval augmented generation?" to `feed.search`, and a 9,000-item local archive is the right first answer. `feed.research` routes only when the question names a place to look (arXiv, PubMed, CrossRef, a journal, "the literature", "published") or says "research"/"look up"; `feed.source_add` with a research URL routes on "track/follow/watch/monitor <topic>" when no URL is present.

- [ ] **Step 1: Write the failing tests**

Append to `FEED_CASES` in `tests/test_ask_routing.py`:

```python
    # Going and looking, not searching what arrived: a named source or venue.
    ("what has been published on protein language models on pubmed?", "feed.research"),
    ("search arxiv for equivariant force fields", "feed.research"),
    ("look up recent papers in Nature Methods on cryo-EM", "feed.research"),
    ("research diffusion models for molecules", "feed.research"),
    # A standing topic, registered as a feed.
    ("track equivariant interatomic potentials for me", "feed.source_add"),
    ("follow graph neural networks for chemistry", "feed.source_add"),
```

and a new test:

```python
def test_research_and_track_decisions_carry_their_arguments():
    d = route_feed("search arxiv for equivariant force fields")
    assert d.kwargs == {"query": "equivariant force fields", "sources": "arxiv"}
    d = route_feed("what has been published on protein language models on pubmed?")
    assert d.kwargs["query"] == "protein language models" and d.kwargs["sources"] == "pubmed"
    d = route_feed("research diffusion models for molecules")
    assert d.kwargs == {"query": "diffusion models for molecules", "sources": "arxiv,pubmed"}
    d = route_feed("track equivariant interatomic potentials for me")
    assert d.tool == "feed.source_add"
    assert d.kwargs == {"url": "research:arxiv,pubmed?q=equivariant+interatomic+potentials"}
    # a URL keeps the old behaviour: no research kwargs, the caller supplies the url
    assert route_feed("follow https://example.com/rss").kwargs == {}
    # "papers on X" is still the local archive
    assert route_feed("anything new on retrieval augmented generation?").tool == "feed.search"
```

Append to `tests/test_research_tools.py`:

```python
def test_feed_ask_executes_research_and_track(tmp_path, monkeypatch):
    from conftest import seeded_db

    from attestation.mcp import ask

    db = _db(tmp_path, monkeypatch)
    seeded_db(db).close()
    monkeypatch.setattr(tool_mod, "CLIENTS", _clients([_paper(1)]))
    out = ask._feed_ask("researcher", "search arxiv for equivariant force fields")
    assert out["ok"] and out["tool_used"] == "feed.research" and "1 paper" in out["answer"]
    out = ask._feed_ask("researcher", "track equivariant interatomic potentials for me")
    assert out["ok"] and out["tool_used"] == "feed.source_add"
    row = get_db(db).execute("SELECT url, added_by FROM feeds WHERE url LIKE 'research:%'").fetchone()
    assert row["url"] == "research:arxiv,pubmed?q=equivariant+interatomic+potentials"
    assert row["added_by"] is not None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen pytest tests/test_ask_routing.py tests/test_research_tools.py -q`
Expected: FAIL (routes to `feed.search` / declines; `_feed_ask` returns "Tell me which item").

- [ ] **Step 3: Implement the router**

In `src/attestation/mcp/routing.py`, add after `_SEARCH_PHRASES`:

```python
# Going and looking is a different act from searching what arrived. These
# name a PLACE to look or say so outright; "papers on X" alone stays local,
# because the feed surface promises that question is answered without a
# network call and the archive is the right first answer.
_RESEARCH_SOURCES = (
    ("pubmed", "pubmed"),
    ("arxiv", "arxiv"),
    ("crossref", "crossref"),
)
_RESEARCH_PHRASES = (
    "on arxiv",
    "on pubmed",
    "from arxiv",
    "from pubmed",
    "search arxiv",
    "search pubmed",
    "crossref",
    "in the literature",
    "published on",
    "published about",
    "published in",
    "been published",
    "research ",
    "look up",
    " journal",
    "outside my feed",
    "beyond my feed",
)
_TRACK_PHRASES = ("track ", "follow ", "watch ", "monitor ", "keep an eye on ", "add a topic")
_RESEARCH_NOISE = re.compile(
    r"\b(on|from|in|via|search|the)\s+(arxiv|pubmed|crossref|the literature)\b"
    r"|\b(recent|new)\s+papers?\b|\bpapers?\s+(on|about|in)\b|\bfor me\b|\bwhat has been\b"
    r"|\bpublished\s+(on|about|in)\b|\blook up\b|\bresearch\b|^(search|track|follow|watch|monitor)\b",
    re.IGNORECASE,
)


def _research_query(question: str) -> str:
    """The topic with the going-and-looking words removed."""
    q = _RESEARCH_NOISE.sub(" ", question.strip().rstrip("?"))
    q = re.sub(r"\b(in|on)\s+[A-Z][\w-]*(\s+[A-Z][\w-]*)*\s*$", " ", q)  # trailing venue name
    return " ".join(_strip_topic(q).split())


def _route_research(q: str, question: str) -> Decision | None:
    """feed.research for a named source/venue; feed.source_add(research:) for 'track X'."""
    if "http" in q:
        return None
    if _has(q, *_TRACK_PHRASES) and not _has(q, *_SUGGEST_PHRASES):
        topic = _research_query(question)
        if len(topic.split()) >= 2:
            url = f"research:arxiv,pubmed?q={quote_plus(topic)}"
            return Decision("feed.source_add", {"url": url})
    if _has(q, *_RESEARCH_PHRASES):
        topic = _research_query(question)
        if len(topic.split()) >= 2:
            sources = ",".join(s for word, s in _RESEARCH_SOURCES if word in q) or "arxiv,pubmed"
            return Decision("feed.research", {"query": topic, "sources": sources})
    return None
```

with `from urllib.parse import quote_plus` at the top. In `route_feed`, insert before the `for tool_name, phrases in _FEED_RULES` loop:

```python
    if (research := _route_research(q, question)) is not None:
        return research
```

Run the routing tests and adjust `_RESEARCH_NOISE` until the six new cases AND every existing case pass; the assertion on exact `kwargs` in the new test is the spec for the stripping. If a phrase collides with an existing case, narrow the phrase, never the existing case.

- [ ] **Step 4: Implement the dispatch**

In `src/attestation/mcp/ask.py::_feed_ask`, before the `elif decision.tool in {"feed.rate", "feed.explain", "feed.source_add", "feed.source_remove"}` branch add:

```python
    elif decision.tool == "feed.research":
        from attestation.mcp import research as research_mod

        out = research_mod._research(
            decision.kwargs["query"], sources=decision.kwargs.get("sources", "arxiv,pubmed")
        )
    elif decision.tool == "feed.source_add" and "url" in decision.kwargs:
        from attestation.mcp import subscriptions as subs

        out = subs._add_feed(decision.kwargs["url"], None, user)
```

Check `_compose(out, tool)` renders `out["message"]` as `answer` when there are no items (read the rest of `_compose`); if it does not, add `answer = out.get("message") or answer` there. Also add `feed.research` to any `_TOOLS`/options list `ask.py` keeps for the feed router's `options` (grep `"feed.digest"` in ask.py).

- [ ] **Step 5: Run**

Run: `uv run --frozen pytest tests/test_ask_routing.py tests/test_research_tools.py tests/test_skill_files.py -q`
Expected: PASS (`test_skill_files::test_bundled_skills_together_teach_every_ask_router` is unaffected; the skill text lands in Task 11).

- [ ] **Step 6: Commit**

```bash
git add src/attestation/mcp/routing.py src/attestation/mcp/ask.py tests/test_ask_routing.py tests/test_research_tools.py
git commit -m "Route going-and-looking questions: a named source, venue or 'research' goes to feed.research with the topic and sources extracted; 'track X' registers a research: feed through feed.source_add. 'papers on X' stays on the local archive, which the feed surface promises

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: BibTeX from a library row; `cite.lookup` returns it

**Files:**
- Modify: `src/attestation/library.py` (`bibtex_key`, `bibtex`, `select_rows`, `export_bib`)
- Modify: `src/attestation/mcp/citation.py` (`_lookup` + `cite.lookup` docstring/`empty`)
- Test: `tests/test_library.py`, `tests/test_library_tools.py`

**Interfaces:**
- Produces: `library.bibtex_key(row) -> str`; `library.bibtex(row, *, key=None) -> str`; `library.select_rows(conn, *, author=None, year=None, tag=None, source=None) -> list[sqlite3.Row]`; `library.export_bib(rows) -> str`; `cite.lookup` payload gains `bibtex: str | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_library.py` (it already imports `get_db`, `ReferenceRecord`, `upsert`; add `from attestation import library`):

```python
def _row(conn, rid):
    return conn.execute('SELECT * FROM "references" WHERE id = ?', (rid,)).fetchone()


def test_bibtex_article_for_a_journal_row(tmp_path):
    conn = get_db(tmp_path / "t.db")
    rid, _ = upsert(conn, ReferenceRecord(
        source="research:crossref", source_key="10.1038/s41592-022-01760-3",
        title="Protein language models & structure", authors=["Lin, Zeming", "Rives, Alexander"],
        year=2023, venue="Nature Methods", doi="10.1038/s41592-022-01760-3",
        url="https://doi.org/10.1038/s41592-022-01760-3",
    ))
    out = library.bibtex(_row(conn, rid))
    assert out.startswith("@article{lin2023protein,\n")
    assert "  author = {Lin, Zeming and Rives, Alexander},\n" in out
    assert "  title = {Protein language models \\& structure},\n" in out
    assert "  journal = {Nature Methods},\n" in out and "  year = {2023},\n" in out
    assert "  doi = {10.1038/s41592-022-01760-3},\n" in out and out.rstrip().endswith("}")
    assert library.bibtex(_row(conn, rid)) == out  # deterministic


def test_bibtex_misc_for_a_preprint_and_bib_key_wins(tmp_path):
    conn = get_db(tmp_path / "t.db")
    rid, _ = upsert(conn, ReferenceRecord(
        source="bibtex:/a.bib", source_key="nequip", bib_key="nequip",
        title="E(3)-equivariant graph neural networks", authors=["Batzner, Simon"],
        year=2021, arxiv_id="2101.03164",
    ))
    out = library.bibtex(_row(conn, rid))
    assert out.startswith("@misc{nequip,\n")
    assert "  eprint = {2101.03164},\n  archivePrefix = {arXiv},\n" in out and "journal" not in out


def test_bibtex_key_is_ascii_and_export_suffixes_collisions(tmp_path):
    conn = get_db(tmp_path / "t.db")
    for i, title in enumerate(("Über Graphen I", "Über Graphen II")):
        upsert(conn, ReferenceRecord(
            source="zotero", source_key=f"Z{i}", title=title, authors=["Müller, Anna"], year=2020,
            doi=f"10.5555/{i}",
        ))
    rows = library.select_rows(conn, author="müller")
    assert [library.bibtex_key(r) for r in rows] == ["muller2020uber", "muller2020uber"]
    bib = library.export_bib(rows)
    assert "@misc{muller2020uber,\n" in bib and "@misc{muller2020uberb,\n" in bib
    assert library.select_rows(conn, year=1999) == []
```

Append to `tests/test_library_tools.py::test_cite_lookup_shows_every_source_and_the_conflicts` a final line: `assert out["bibtex"].startswith("@")`, and a new test:

```python
def test_cite_lookup_from_disk_reader_has_no_bibtex_or_text(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch)
    (tmp_path / "refs.bib").write_text("@article{k1, title={T}, author={A B}, year={2020}}\n")
    monkeypatch.setenv("ATTEST_BIB_PATHS", str(tmp_path / "refs.bib"))
    out = citation._lookup("k1")
    assert out["ok"] and out["bibtex"] is None and out["full_text"] is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen pytest tests/test_library.py tests/test_library_tools.py -q`
Expected: FAIL (`library.bibtex` missing; `bibtex` key absent).

- [ ] **Step 3: Implement**

In `src/attestation/library.py`, after `to_reference`:

```python
# ---------------------------------------------------------------------------
# BibTeX: rendered from the row, deterministic, no network
# ---------------------------------------------------------------------------

_BIB_ESCAPE = str.maketrans({c: f"\\{c}" for c in "&%$#_"})


def _ascii_word(text: str) -> str:
    """Letters and digits of the NFKD-folded text, lowercase -- what a BibTeX key may hold."""
    return re.sub(r"[^a-z0-9]", "", normalise_title(text))


def bibtex_key(row) -> str:
    """`bib_key` when a .bib or Zotero supplied one, else <family><year><first title word>."""
    if row["bib_key"]:
        return row["bib_key"]
    authors = json.loads(row["authors"])
    family = _ascii_word(authors[0].split(",")[0]) if authors else "anon"
    words = [w for w in normalise_title(row["title"]).split() if w not in ("a", "an", "the")]
    first = _ascii_word(words[0]) if words else "untitled"
    return f"{family}{row['year'] or ''}{first}"


def bibtex(row, *, key: str | None = None) -> str:
    """One BibTeX entry for a library row: @article when the venue is a journal,
    @misc with arXiv eprint fields for a preprint. Pure; the same row always
    renders the same text, so an exported file is stable."""
    venue = row["venue"]
    preprint = not venue or bool(_PREPRINT_VENUE.match(venue))
    fields: list[tuple[str, str | None]] = [
        ("author", " and ".join(json.loads(row["authors"])) or None),
        ("title", row["title"]),
        ("journal", None if preprint else venue),
        ("year", str(row["year"]) if row["year"] else None),
        ("doi", row["doi"]),
        ("url", row["url"]),
        ("eprint", row["arxiv_id"] if preprint else None),
        ("archivePrefix", "arXiv" if preprint and row["arxiv_id"] else None),
    ]
    body = "".join(
        f"  {name} = {{{value.translate(_BIB_ESCAPE) if name in ('author', 'title', 'journal') else value}}},\n"
        for name, value in fields
        if value
    )
    return f"@{'misc' if preprint else 'article'}{{{key or bibtex_key(row)},\n{body}}}\n"


def select_rows(conn: sqlite3.Connection, *, author=None, year=None, tag=None, source=None) -> list:
    """Rows matching the fielded filters, id order -- the export's input."""
    where, params = _fielded_where(author, year, None, None, tag, source)
    return conn.execute(f'SELECT * FROM "references" r WHERE {where} ORDER BY r.id', params).fetchall()


def export_bib(rows) -> str:
    """Every row as BibTeX; a repeated key takes a letter suffix in row order (b, c, ...)."""
    seen: dict[str, int] = {}
    out = []
    for row in rows:
        base = bibtex_key(row)
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.append(bibtex(row, key=base if n == 0 else f"{base}{chr(ord('a') + n)}"))
    return "\n".join(out)
```

Check `_fielded_where`'s return uses `r.` aliases (it does for search; if it does not, drop the `r` alias in `select_rows`). If `bibtex` trips the complexity ratchet, move the `fields` list into `_bib_fields(row, preprint)`.

In `src/attestation/mcp/citation.py::_lookup`: `empty` gains `"bibtex": None, "full_text": None`; the store branch's return gains `"bibtex": library.bibtex(row), "full_text": None` (Task 9 fills `full_text`); the disk-reader branch returns `"bibtex": None, "full_text": None`. Add to `cite.lookup`'s docstring: "`bibtex` is rendered from the library row (null for a disk-reader answer)."

- [ ] **Step 4: Run**

Run: `uv run --frozen pytest tests/test_library.py tests/test_library_tools.py tests/test_tool_envelope.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/attestation/library.py src/attestation/mcp/citation.py tests/test_library.py tests/test_library_tools.py
git commit -m "BibTeX rendered from the library row: @article for a journal, @misc with eprint fields for a preprint, the .bib key when one exists else family-year-word in ASCII; export suffixes collisions in row order so the file is stable; cite.lookup returns it

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Full text: capped fetch pass, served in windows

**Files:**
- Modify: `pyproject.toml` (`pypdf`), `uv.lock` (via `uv add`)
- Modify: `src/attestation/research.py` (`pdf_text`, `parse_pmc`, `fetch_fulltext`)
- Modify: `src/attestation/library.py` (`fulltext_window`)
- Modify: `src/attestation/mcp/citation.py` (`cite.lookup(text_offset, text_chars)`)
- Modify: `CLAUDE.md` Reliability-contract line (BLE001 count), `tests/test_architecture.py` if it pins the number
- Test: `tests/test_research.py`, `tests/test_library_tools.py`, `tests/test_response_size.py`

**Interfaces:**
- Produces: `research.pdf_text(body: bytes) -> str | None`; `research.parse_pmc(body: bytes) -> str | None`; `research.fetch_fulltext(conn, *, limit=10, fetch=None, clients=None) -> dict(fetched, none, failed)`; `library.fulltext_window(conn, reference_id, offset=0, chars=3000) -> dict | None`; `cite.lookup(key, text_offset=0, text_chars=3000)`; `citation.MAX_TEXT_CHARS = 3000`.

- [ ] **Step 1: Add the dependency**

Run: `uv add pypdf` then `git diff --stat pyproject.toml uv.lock` (both change). `pypdf` is pure Python.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_research.py`:

```python
from attestation.db import get_db
from attestation.library import ReferenceRecord, upsert


def _pdf_bytes(text: str) -> bytes:
    """A one-page PDF carrying `text`, written by pypdf itself."""
    import io

    from pypdf import PdfWriter

    w = PdfWriter()
    page = w.add_blank_page(width=200, height=200)
    # pypdf cannot lay out text; a content stream with a text object is enough for extract_text.
    from pypdf.generic import DecodedStreamObject, NameObject

    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = w._add_object(stream)
    page[NameObject("/Resources")] = w._add_object(
        w._add_object.__self__.__class__.__mro__ and __import__("pypdf").generic.DictionaryObject(
            {NameObject("/Font"): __import__("pypdf").generic.DictionaryObject({
                NameObject("/F1"): __import__("pypdf").generic.DictionaryObject({
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                })
            })}
        )
    )
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()
```

That helper is fiddly; if `pypdf`'s API makes it fail, replace it with a committed 1-page fixture `tests/fixtures/research/one_page.pdf` generated once by `uv run python - <<'EOF' ... EOF` using `reportlab`-free means (e.g. `print` a minimal hand-written PDF: header, one page, a content stream with `(Hello full text) Tj`, xref table) and assert `"Hello full text" in research.pdf_text(bytes)`. Either way the test contract is:

```python
def test_pdf_text_extracts_and_tolerates_garbage():
    assert "Hello full text" in (research.pdf_text(_pdf_bytes("Hello full text")) or "")
    assert research.pdf_text(b"not a pdf") is None


def test_parse_pmc_joins_body_paragraphs_and_none_without_body():
    xml = (
        b'<pmc-articleset><article><front><article-meta/></front>'
        b"<body><sec><title>Intro</title><p>First para.</p><p>Second <italic>para</italic>.</p></sec></body>"
        b"</article></pmc-articleset>"
    )
    assert research.parse_pmc(xml) == "Intro\nFirst para.\nSecond para."
    assert research.parse_pmc(b"<pmc-articleset><article><front/></article></pmc-articleset>") is None


def test_fetch_fulltext_caps_marks_none_and_retries_only_transient(tmp_path):
    conn = get_db(tmp_path / "t.db")
    ids = []
    for i in range(3):
        rid, _ = upsert(conn, ReferenceRecord(
            source="research:arxiv", source_key=f"2101.0000{i}", title=f"P{i}", arxiv_id=f"2101.0000{i}",
        ))
        ids.append(rid)
    pm, _ = upsert(conn, ReferenceRecord(source="research:pubmed", source_key="9", title="PM", pmcid="PMC9"))
    none_id, _ = upsert(conn, ReferenceRecord(source="zotero", source_key="Z", title="no ids"))
    conn.commit()
    calls = []

    def fetch(url):
        calls.append(url)
        if "2101.00002" in url:
            raise research.httpx.ConnectError("down")
        if "2101.00001" in url:
            return b"garbage"
        if "pmc" in url:
            return b"<pmc-articleset><article><body><p>PMC body.</p></body></article></pmc-articleset>"
        return _pdf_bytes("Hello full text")

    out = research.fetch_fulltext(conn, limit=2, fetch=fetch)
    assert out == {"fetched": 1, "none": 0, "failed": 1} or out == {"fetched": 1, "none": 1, "failed": 0}
    out2 = research.fetch_fulltext(conn, limit=10, fetch=fetch)
    rows = {r["reference_id"]: r for r in conn.execute("SELECT * FROM reference_fulltext")}
    assert none_id not in rows  # nothing to fetch, never selected
    assert rows[pm]["source"] == "pmc-xml" and rows[pm]["text"] == "PMC body."
    assert rows[ids[1]]["source"] == "none" and rows[ids[1]]["text"] is None  # unreadable pdf: tried
    assert ids[2] not in rows  # transient failure: no row, retried next run
    assert any("2101.00002" in u for u in calls[-3:])
    assert out2["failed"] == 1
    # the flag off does nothing
    assert research.fetch_fulltext(conn, limit=10, fetch=fetch, clients=research.null_clients()) == {
        "fetched": 0, "none": 0, "failed": 0,
    }
```

Append to `tests/test_library_tools.py`:

```python
def test_cite_lookup_serves_full_text_in_windows_under_budget(tmp_path, monkeypatch):
    from test_response_size import MAX_READ_RESPONSE_CHARS

    db = _db(tmp_path, monkeypatch)
    conn = get_db(db)
    rid, _ = upsert(conn, ReferenceRecord(source="research:arxiv", source_key="1", title="T", arxiv_id="2101.00001"))
    conn.execute(
        "INSERT INTO reference_fulltext(reference_id, text, source, fetched_at) VALUES (?, ?, ?, ?)",
        (rid, "x" * 10000, "arxiv-pdf", "2026-09-10"),
    )
    conn.commit()
    conn.close()
    out = citation._lookup("2101.00001")
    ft = out["full_text"]
    assert ft["total"] == 10000 and ft["offset"] == 0 and ft["chars"] == 3000 and len(ft["text"]) == 3000
    assert ft["source"] == "arxiv-pdf"
    assert len(json.dumps(out, indent=2)) < MAX_READ_RESPONSE_CHARS
    out = citation._lookup("2101.00001", 9500, 3000)
    assert out["full_text"]["chars"] == 500 and out["full_text"]["offset"] == 9500
    out = citation._lookup("2101.00001", 0, 99999)
    assert out["full_text"]["chars"] == citation.MAX_TEXT_CHARS
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run --frozen pytest tests/test_research.py tests/test_library_tools.py -q`
Expected: FAIL (`pdf_text` missing; `full_text` None).

- [ ] **Step 4: Implement**

In `src/attestation/research.py` (imports at the top: `import io`, `import sqlite3`, `from datetime import UTC, datetime`):

```python
# ---------------------------------------------------------------------------
# full text: a capped pass, arXiv PDF or PMC open-access XML, never embedded
# ---------------------------------------------------------------------------

_PMC_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&retmode=xml&id="


def pdf_text(body: bytes) -> str | None:
    """The text of every page, or None when pypdf cannot read the bytes.

    pypdf raises a zoo of exception types on malformed input (its own
    PdfReadError, but also KeyError/TypeError/RecursionError from deep inside
    the parser), and one unreadable PDF must cost itself, not the pass.
    """
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(body))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:  # noqa: BLE001 -- see the docstring: pypdf's failure modes are open-ended
        return None
    return text.strip() or None


def parse_pmc(body: bytes) -> str | None:
    """Section titles and paragraphs of a PMC article's <body>, one per line; None without a body."""
    root = SafeET.fromstring(body)
    body_el = root.find(".//body")
    if body_el is None:
        return None
    lines = ["".join(el.itertext()).strip() for el in body_el.iter() if el.tag in ("title", "p")]
    text = "\n".join(_collapse(line) for line in lines if line.strip())
    return text or None


def _fulltext_todo(conn: sqlite3.Connection, limit: int):
    return conn.execute(
        'SELECT r.id, r.arxiv_id, r.pmcid FROM "references" r'
        " WHERE (r.arxiv_id IS NOT NULL OR r.pmcid IS NOT NULL)"
        " AND NOT EXISTS (SELECT 1 FROM reference_fulltext f WHERE f.reference_id = r.id)"
        " ORDER BY r.first_seen DESC, r.id DESC LIMIT ?",
        (limit,),
    ).fetchall()


def fetch_fulltext(conn: sqlite3.Connection, *, limit: int = 10, fetch=None, clients=None) -> dict:
    """Pull bodies for up to `limit` references that have none yet, newest first.

    arXiv PDF through pypdf, else PMC open-access XML by PMCID. A body that
    cannot be read writes `source='none'` so the row is not retried hourly;
    a transport failure writes NOTHING, so the next run tries again. With the
    research flag off (all clients offline) this does nothing.
    """
    clients = clients or clients_from_env()
    counts = {"fetched": 0, "none": 0, "failed": 0}
    if all(c.offline for c in clients.values()):
        return counts
    fetch = fetch or _default_fetch
    for row in _fulltext_todo(conn, limit):
        if row["arxiv_id"]:
            url, source, parse = f"https://arxiv.org/pdf/{row['arxiv_id']}", "arxiv-pdf", pdf_text
        else:
            url, source, parse = _PMC_BASE + row["pmcid"].removeprefix("PMC"), "pmc-xml", parse_pmc
        try:
            text = parse(fetch(url))
        except FETCH_ERRORS:
            counts["failed"] += 1
            continue
        conn.execute(
            "INSERT INTO reference_fulltext(reference_id, text, source, fetched_at) VALUES (?, ?, ?, ?)",
            (row["id"], text, source if text else "none", datetime.now(UTC).date().isoformat()),
        )
        conn.commit()
        counts["fetched" if text else "none"] += 1
    return counts
```

In `src/attestation/library.py`:

```python
def fulltext_window(conn: sqlite3.Connection, reference_id: int, offset: int = 0, chars: int = 3000):
    """A slice of a reference's stored body, with the total so a caller can page. None when absent."""
    row = conn.execute(
        "SELECT text, source FROM reference_fulltext WHERE reference_id = ?", (reference_id,)
    ).fetchone()
    if row is None or not row["text"]:
        return None
    text = row["text"]
    piece = text[offset : offset + chars]
    return {"text": piece, "offset": offset, "chars": len(piece), "total": len(text), "source": row["source"]}
```

In `src/attestation/mcp/citation.py`: `MAX_TEXT_CHARS = 3000` at module level; `_lookup(conn, key, text_offset: int = 0, text_chars: int = MAX_TEXT_CHARS)`; in the store branch, `"full_text": library.fulltext_window(conn, row["id"], max(text_offset, 0), min(max(text_chars, 1), MAX_TEXT_CHARS))`; the registered tool gains

```python
        text_offset: Annotated[int, Field(ge=0, description="full-text window start")] = 0,
        text_chars: Annotated[int, Field(ge=1, le=MAX_TEXT_CHARS)] = MAX_TEXT_CHARS,
```

passed through as `_lookup(key, text_offset, text_chars)`, and the docstring adds: "`full_text` is a window of the stored body (`attest library fulltext` fills it from arXiv PDFs and PMC): `text`, `offset`, `chars`, `total`; page with `text_offset`. Never the whole body."

BLE001 bookkeeping: run `grep -rn "noqa: BLE001" src/attestation | wc -l`; set CLAUDE.md's Reliability-contract line to that number and add `research` to its module list; if `tests/test_architecture.py` (around line 748) pins a literal, move it too.

- [ ] **Step 5: Run and gate**

Run: `uv run --frozen pytest tests/test_research.py tests/test_library_tools.py tests/test_response_size.py tests/test_architecture.py -q && uv run --frozen pre-commit run --all-files`
Expected: PASS, all hooks Passed (including `uv.lock matches pyproject.toml`).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/attestation/research.py src/attestation/library.py src/attestation/mcp/citation.py tests CLAUDE.md
git commit -m "Full text as a capped pass: arXiv PDF via pypdf or PMC open-access XML, 'none' marks a body that cannot be read so it is not retried hourly while a transport failure leaves no row; cite.lookup serves it in 3000-char windows with the total, never whole

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: CLI: `research`, `sources add`, `library fulltext`, `library export`, `ingest` flags

**Files:**
- Modify: `src/attestation/cli.py` (`HELP`, `build_parser`, `cmd_ingest`, new `cmd_research`, `cmd_sources_add`, `cmd_library_fulltext`, `cmd_library_export`)
- Regenerate: `docs/reference/cli.md` (`uv run --frozen python scripts/render_cli_reference.py`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `research.clients_from_env/null_clients/fetch_topic/Topic/TopicError/fetch_fulltext`, `feeds.add_source(added_by=)`, `library.select_rows/export_bib`, `run_ingest(clients=)`.
- Produces the commands in the spec's CLI block.

- [ ] **Step 1: Read how `tests/test_cli.py` invokes commands**

Run: `grep -n "def test_\|main(\[" tests/test_cli.py | head -20` — reuse its pattern (`main([...])` with `--db tmp`, capsys).

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_cli.py` (adapting the invocation helper the file uses):

```python
def test_sources_add_research_topic_and_ingest_no_research(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "t.db")
    from conftest import seeded_db

    seeded_db(tmp_path / "t.db").close()
    rc = main(["--db", db, "sources", "add", "research:arxiv?q=graph+neural+networks", "--user", "researcher"])
    assert rc == 0 and "tracking" in capsys.readouterr().out
    rc = main(["--db", db, "sources", "add", "research:nope?q=x"])
    assert rc == 1 and "unknown research client" in capsys.readouterr().err
    feeds = tmp_path / "feeds.toml"
    feeds.write_text("feeds = []\n")
    monkeypatch.setattr("attestation.cli.Embedder", FakeEmbedderForCli, raising=False)  # see the file's existing embedder stub
    rc = main(["--db", db, "ingest", "--feeds", str(feeds), "--no-research", "--fulltext-limit", "0"])
    out = capsys.readouterr().out
    assert rc == 0 and "'research_disabled': 1" in out


def test_research_command_prints_rows_and_no_store(tmp_path, capsys, monkeypatch):
    from attestation import research

    class Fixed:
        name, offline = "arxiv", False

        def search(self, *a, **k):
            return [research.Paper(client="arxiv", external_id="1", title="Hit", arxiv_id="2101.00001", url="u")]

    monkeypatch.setattr(research, "clients_from_env", lambda **k: {"arxiv": Fixed(), "pubmed": research.NullClient("pubmed"), "crossref": research.NullClient("crossref")})
    db = str(tmp_path / "t.db")
    rc = main(["--db", db, "research", "equivariant", "--sources", "arxiv", "--no-store"])
    out = capsys.readouterr().out
    assert rc == 0 and "Hit" in out and "not stored" in out
    from attestation.db import get_db

    assert get_db(tmp_path / "t.db").execute('SELECT COUNT(*) FROM "references"').fetchone()[0] == 0
    rc = main(["--db", db, "research", "equivariant", "--sources", "arxiv"])
    assert rc == 0
    assert get_db(tmp_path / "t.db").execute('SELECT COUNT(*) FROM "references"').fetchone()[0] == 1


def test_library_export_writes_new_files_only(tmp_path, capsys):
    from attestation.db import get_db
    from attestation.library import ReferenceRecord, upsert

    db = tmp_path / "t.db"
    conn = get_db(db)
    upsert(conn, ReferenceRecord(source="zotero", source_key="Z", title="T", authors=["A, B"], year=2020, doi="10.5555/t"))
    conn.commit()
    conn.close()
    target = tmp_path / "out.bib"
    assert main(["--db", str(db), "library", "export", "--bib", str(target)]) == 0
    assert target.read_text().startswith("@misc{a2020t,")
    assert main(["--db", str(db), "library", "export", "--bib", str(target)]) == 1
    assert "exists" in capsys.readouterr().err
    assert main(["--db", str(db), "library", "export", "--bib", str(target), "--force"]) == 0
    assert main(["--db", str(db), "library", "export", "--bib", str(tmp_path / "none.bib"), "--year", "1999"]) == 1


def test_library_fulltext_command_reports_counts(tmp_path, capsys, monkeypatch):
    from attestation import research

    monkeypatch.setattr(research, "fetch_fulltext", lambda conn, limit, **k: {"fetched": 0, "none": 0, "failed": 0})
    assert main(["--db", str(tmp_path / "t.db"), "library", "fulltext", "--limit", "3"]) == 0
    assert "fetched 0" in capsys.readouterr().out
```

If `test_cli.py` has no embedder stub for `ingest`, monkeypatch `attestation.cli.cmd_ingest`'s `Embedder` import the way its existing ingest test does (grep `ingest` in the file); the assertion that matters is `research_disabled: 1` in the printed stats.

- [ ] **Step 3: Run to verify they fail**

Run: `uv run --frozen pytest tests/test_cli.py -q -k "research or sources_add or export or fulltext"`
Expected: FAIL (argparse: invalid choice).

- [ ] **Step 4: Implement**

In `HELP` add:

```python
    "research": "search arXiv/PubMed/CrossRef for papers and keep them in the library",
    "sources": "manage feed sources (RSS URLs and research: topics)",
    "sources.add": "register an RSS feed or a research: topic (no fetch; next ingest)",
    "library.fulltext": "fetch bodies (arXiv PDF / PMC XML) for references that lack one",
    "library.export": "write a filtered set of references as one .bib (new files only)",
```

In `build_parser`:

```python
    sp = sub.add_parser("ingest", help=HELP["ingest"])
    add_db(sp)
    sp.add_argument("--feeds", default=_default_feeds_path())
    sp.add_argument("--no-research", action="store_true", help="skip research: feeds this run")
    sp.add_argument(
        "--fulltext-limit", type=int, default=10, help="bodies to fetch after ingest (0 = none)"
    )
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("research", help=HELP["research"])
    add_db(sp)
    sp.add_argument("query")
    sp.add_argument("--sources", default="arxiv,pubmed", help="comma-separated: arxiv,pubmed,crossref")
    sp.add_argument("--journal", help="pubmed/crossref only")
    sp.add_argument("--since-days", type=int, default=365)
    sp.add_argument("--limit", type=int, default=10)
    sp.add_argument("--no-store", action="store_true", help="preview; do not write the library")
    sp.set_defaults(func=cmd_research)

    sp = sub.add_parser("sources", help=HELP["sources"])
    add_db(sp)
    src_sub = sp.add_subparsers(dest="sources_command", required=True)
    ap = src_sub.add_parser("add", help=HELP["sources.add"])
    ap.add_argument("url", help="an RSS/Atom URL or research:<clients>?q=<query>[&journal=...]")
    ap.add_argument("--title")
    ap.add_argument("--user", help="persona to record as the one who added it")
    ap.set_defaults(func=cmd_sources_add)
```

and under the `library` subparsers:

```python
    lp = lib_sub.add_parser("fulltext", help=HELP["library.fulltext"])
    lp.add_argument("--limit", type=int, default=10)
    lp.set_defaults(func=cmd_library_fulltext)

    lp = lib_sub.add_parser("export", help=HELP["library.export"])
    lp.add_argument("--bib", required=True, help="output path; refuses to overwrite without --force")
    lp.add_argument("--author")
    lp.add_argument("--year", type=int)
    lp.add_argument("--tag")
    lp.add_argument("--source", help="reference_sources.source prefix, e.g. zotero, research:arxiv")
    lp.add_argument("--force", action="store_true")
    lp.set_defaults(func=cmd_library_export)
```

Handlers:

```python
@_documented("ingest")
def cmd_ingest(args: argparse.Namespace) -> int:
    from attestation import research
    from attestation.embed import Embedder
    from attestation.ingest import run_ingest

    clients = research.null_clients() if args.no_research else research.clients_from_env()
    with open_db(args.db) as conn:
        stats = run_ingest(conn, Embedder(), args.feeds, clients=clients)
        if args.fulltext_limit > 0:
            stats["fulltext"] = research.fetch_fulltext(conn, limit=args.fulltext_limit, clients=clients)
    print(stats)
    return 0


@_documented("research")
def cmd_research(args: argparse.Namespace) -> int:
    from datetime import UTC, date, datetime, timedelta

    from attestation import library, research

    names = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    try:
        topic = research.parse_topic(research.topic_url(names, args.query, args.journal))
    except research.TopicError as exc:
        return fail(str(exc))
    since = date.today() - timedelta(days=args.since_days) if args.since_days else None
    fetched = research.fetch_topic(topic, research.clients_from_env(), since=since, limit=args.limit)
    if fetched.offline:
        return fail("research clients are off (ATTEST_RESEARCH_WEB=0); nothing searched")
    for err in fetched.errors:
        print(f"warning: {err}", file=sys.stderr)
    for p in fetched.papers:
        ident = p.doi or p.arxiv_id or p.external_id
        print(f"{p.published or '----'}  {ident:32s}  {p.title[:70]}")
    if args.no_store:
        print(f"{len(fetched.papers)} paper(s); not stored")
        return 0
    stored = 0
    with open_db(args.db) as conn:
        today = datetime.now(UTC).date().isoformat()
        for rec in research.as_records(fetched.papers, today):
            _rid, how = library.upsert(conn, rec)
            stored += how != "unchanged"
        conn.commit()
    print(f"{len(fetched.papers)} paper(s); {stored} new in the library")
    return 0


@_documented("sources.add")
def cmd_sources_add(args: argparse.Namespace) -> int:
    from attestation import feeds

    with open_db(args.db) as conn:
        added_by = None
        if args.user:
            row = conn.execute("SELECT id FROM users WHERE name = ?", (args.user,)).fetchone()
            if row is None:
                return fail(f"unknown user: {args.user!r}")
            added_by = row["id"]
        try:
            _feed_id, message = feeds.add_source(conn, args.url, args.title, added_by=added_by)
        except feeds.FeedError as exc:
            return fail(str(exc))
    print(message)
    return 0


@_documented("library.fulltext")
def cmd_library_fulltext(args: argparse.Namespace) -> int:
    from attestation import research

    with open_db(args.db) as conn:
        counts = research.fetch_fulltext(conn, limit=args.limit)
    print(f"fetched {counts['fetched']}, none {counts['none']}, failed {counts['failed']}")
    return 0


@_documented("library.export")
def cmd_library_export(args: argparse.Namespace) -> int:
    from pathlib import Path

    from attestation import library

    target = Path(args.bib)
    if target.exists() and not args.force:
        return fail(f"{target} exists; pass --force to overwrite")
    with open_db(args.db) as conn:
        rows = library.select_rows(
            conn, author=args.author, year=args.year, tag=args.tag, source=args.source
        )
    if not rows:
        return fail("no references match; nothing written")
    target.write_text(library.export_bib(rows))
    print(f"wrote {len(rows)} entr{'y' if len(rows) == 1 else 'ies'} to {target}")
    return 0
```

Check `_documented`'s contract: every `cmd_*` needs a `HELP` key of the same name (the test in `test_cli.py` or `test_architecture.py` that enforces it will say). Then:

Run: `uv run --frozen python scripts/render_cli_reference.py`

- [ ] **Step 5: Run**

Run: `uv run --frozen pytest tests/test_cli.py tests/test_docs_site.py tests/test_architecture.py -q`
Expected: PASS (docs-site test checks `cli.md` is regenerated).

- [ ] **Step 6: Commit**

```bash
git add src/attestation/cli.py docs/reference/cli.md tests/test_cli.py
git commit -m "CLI: attest research (ad hoc, --no-store previews), attest sources add for RSS and research: URLs, attest library fulltext, attest library export --bib (new files only, --force to overwrite), and ingest --no-research/--fulltext-limit

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Skills, guides, CLAUDE.md, CHANGELOG, spec amendments

**Files:**
- Modify: `src/attestation/skills/attestation-feed/SKILL.md` (new section after "Find me papers about X"), `src/attestation/skills/attestation-knowledge/SKILL.md` (References section)
- Modify: `docs/guides/feed.md`, `docs/guides/claims-and-citations.md`, `docs/guides/agents.md` (flag paragraph)
- Modify: `CLAUDE.md` (Key API Patterns: `|Research:` entry; OFFLINE GUARANTEE line; Docs Index adds `research.py`, `mcp/research.py`, `test_research.py`, `test_research_tools.py`, `tests/fixtures/research`, the spec and plan)
- Modify: `CHANGELOG.md` (Unreleased → Added)
- Modify: `docs/superpowers/specs/2026-09-10-paper-research-design.md` (three amendments)
- Test: `tests/test_skill_files.py`, `tests/test_docs_site.py`, `tests/test_architecture.py`, `tests/test_llm.py`

- [ ] **Step 1: Feed skill**

Insert after the "Find me papers about X" section (before `## The digest`):

```markdown
## "Search arXiv for X" / "track X for me"

Two different acts, and the reader's words tell them apart. `feed.search`
looks through what has already arrived. When they name a place -- arXiv,
PubMed, CrossRef, a journal, "the literature", "what has been published" --
they want you to go and look:

```
feed.ask(user="<name>", question="search arxiv for equivariant force fields")
```

routes to `feed.research(query, sources, journal, since_days, limit, store)`,
which searches the network and stores the hits in the reference library, NOT
the feed: `cite.lookup`/`cite.search` find them afterwards, `feed.list` does
not. Show every `url`. `offline: true` means the research clients are
switched off (`ATTEST_RESEARCH_WEB=0`), not that nothing matched -- say so.
`store=false` previews. Ask for a journal only with pubmed or crossref; arXiv
has categories, not journals, and the tool refuses the pair.

"Track / follow / watch X" is a standing topic: `feed.ask` registers a
`research:` feed through `feed.source_add(url="research:arxiv,pubmed?q=<X>",
user=<name>)`, and every hourly ingest searches it. Its hits ARE feed items
-- ranked, digested, readable -- and `feed.sources` lists it with
`kind: research` and who added it. Nothing is fetched at registration; say
"papers appear after the next ingest", which is what the tool's message says.
```

Keep the existing "for you to read, not a literal call string" warning where it is (the skill already carries it once).

- [ ] **Step 2: Knowledge skill**

In the References section, after the `cite.related` paragraph, add:

```markdown
`cite.lookup` also returns `bibtex` (rendered from the library row: `@article`
for a journal, `@misc` with eprint fields for a preprint) and, when a body
has been fetched, `full_text` as a window (`text`, `offset`, `chars`,
`total`); page with `text_offset` rather than asking for the whole thing,
which is never returned. Papers found by the feed's research tools sit in
the same library under `research:<client>` sources.
```

- [ ] **Step 3: Guides**

`docs/guides/feed.md`, new section before "Click provenance":

```markdown
## Standing topics and research

A topic is a feed whose URL has the `research:` scheme:

```bash
uv run attest sources add "research:arxiv,pubmed?q=equivariant+force+fields" --user researcher
uv run attest sources add "research:crossref?q=protein+language+models&journal=Nature+Methods"
uv run attest research "diffusion models for molecules" --sources arxiv --no-store
```

Every `attest ingest` (the hourly refresh) searches each topic through arXiv,
PubMed and/or CrossRef for papers since its last run; hits enter the feed as
items -- ranked, digested, readable -- and the reference library under a
`research:<client>` source with the authors and venue the payload carried.
`attest research` (the `feed.research` tool) is the ad hoc form: it stores
hits in the library only. An item already present under any feed with the
same DOI or arXiv id is skipped, so a cross-listed arXiv paper or a topic hit
the RSS already carried is one item.

`ATTEST_RESEARCH_WEB` is on unless set to `0`: ingest already fetches RSS
over the network and a topic query is the same class of call in the same
command. Off, topics are counted as `research_disabled` in the ingest stats
and `feed.research` says `offline`. See
`docs/superpowers/specs/2026-09-10-paper-research-design.md`.
```

`docs/guides/claims-and-citations.md`, after "The library"'s last paragraph:

```markdown
`attest library fulltext` pulls bodies (arXiv PDF, or PMC open-access XML by
PMCID) for references that have none, ten per run by default and also at the
end of `attest ingest`; `cite.lookup` serves them in 3000-character windows,
never whole, and never embeds them. `attest library export --bib refs.bib
[--author A] [--year Y] [--tag T] [--source S]` writes a filtered set as
BibTeX rendered from the rows -- `@article` for a journal, `@misc` with
`eprint`/`archivePrefix` for a preprint, the `.bib` key when one supplied it,
else `<family><year><word>` -- refusing to overwrite without `--force`.
`cite.lookup` returns the same `bibtex` for one record.
```

`docs/guides/agents.md`: in the paragraph (or table) that lists the two citation flags, add `ATTEST_RESEARCH_WEB` (on by default; `feed.research`, `research:` feeds, full text).

- [ ] **Step 4: CLAUDE.md**

- Key API Patterns: add a line after `|Feeds:`:
  `|Research: a standing topic is a feeds row with a research:<clients>?q=... URL (research.parse_topic, pure); run_ingest dispatches on scheme (_fetch_feed) and the three passes run unchanged with guid <client>:<id> and published = the paper's date; hits ALSO enter the library under research:<client> with the payload's authors (_store_references) — the stated exception to "the wire never introduces a reference"|feed.research is the ad hoc form and stores to the LIBRARY, not the feed; capped at MAX_RESEARCH_LIMIT (8, measured against the 7000 ceiling) because a research row carries authors and venue|items dedup by DOI/arXiv id across feeds (207 measured cross-lists)|full text: reference_fulltext, fetch_fulltext capped per run, 'none' = tried, cite.lookup windows of 3000|bibtex() is pure over the row; export suffixes collisions in row order|ATTEST_RESEARCH_WEB is ON by default (ingest already fetches RSS), read once in research.clients_from_env`
- OFFLINE GUARANTEE line: rename to `OFFLINE GUARANTEE + ITS THREE FLAGS` and add `ATTEST_RESEARCH_WEB (research.clients_from_env: the arXiv/PubMed/CrossRef SEARCH clients and full text; ON unless 0, read once per ingest run / at MCP register)`, then `enrichers only FILL rows the offline readers introduced` → `enrichers only FILL rows the offline readers introduced, and the research clients are the one source that INTRODUCES rows from the wire, under a research:<client> source row that says so`.
- Docs Index: add the new files to their directory lines.

- [ ] **Step 5: CHANGELOG**

Under `## [Unreleased]` → `### Added`, first entry:

```markdown
- Paper research (`2026-09-10`, spec `2026-09-10-paper-research-design.md`):
  standing topics as `research:` feeds (`attest sources add`,
  `feed.source_add`), searched every ingest through arXiv, PubMed and
  CrossRef; the `feed.research` tool and `attest research` for ad hoc search
  into the library; `reference_fulltext` filled by `attest library fulltext`
  and served by `cite.lookup` in windows; BibTeX rendered from library rows
  (`cite.lookup`, `attest library export`); items dedup by DOI/arXiv id
  across feeds; migration 009; `ATTEST_RESEARCH_WEB` on by default. 49 tools.
  Supersedes the unmerged 2026-09-04 design.
```

- [ ] **Step 6: Spec amendments (record what implementation changed)**

In the spec: (a) `feed.research` "`limit` caps at 13" → "caps at 8 (MAX_RESEARCH_LIMIT), measured in `test_research_tools.py`: a research row carries authors and a venue a feed row does not"; (b) the routing paragraph: replace `"find/search papers on X"` with `"search arxiv/pubmed for X"` and add the sentence "‘papers on X’ alone stays on `feed.search`: the feed surface's goal promises that question is answered locally, and the archive is the right first answer (decided in the plan, Task 7)"; (c) the CLI block already lists `attest sources add`; confirm the `crossref` row says `from-pub-date` (Task 3).

- [ ] **Step 7: Run the guards and the full gate**

Run: `uv run --frozen python scripts/render_spec_index.py && uv run --frozen pytest tests/test_skill_files.py tests/test_docs_site.py tests/test_architecture.py tests/test_llm.py tests/test_install_skills.py -q && uv run --frozen pre-commit run --all-files`
Expected: PASS, all hooks Passed.

- [ ] **Step 8: Commit**

```bash
git add -A src/attestation/skills docs CLAUDE.md CHANGELOG.md
git commit -m "Teach the research surface: the feed skill separates 'search what arrived' from 'go and look' and 'track X'; the knowledge skill names bibtex and full-text windows; guides, CLAUDE.md's research and offline-guarantee lines, the changelog, and the spec amended with what the plan decided (limit 8, 'papers on X' stays local, from-pub-date)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Measurements before shipping (network; a scratch copy of the live database)

**Files:**
- Modify: `docs/superpowers/specs/2026-09-10-paper-research-design.md` (new `## Measured` section)

This task needs the network and Ollama and is run by the user or with their go-ahead; it never touches the live database.

- [ ] **Step 1: Scratch copy**

```bash
cp ~/.hermes/skills/science-recommendations/data/hermes.db /tmp/research-scratch.db
export ATTEST_DB=/tmp/research-scratch.db
```

- [ ] **Step 2: Overlap of a topic's first run with the RSS**

```bash
uv run attest sources add "research:arxiv?q=equivariant+interatomic+potentials"
uv run attest sources add "research:pubmed?q=protein+language+models"
time uv run attest ingest --fulltext-limit 0
```

Record: items added vs skipped for each research feed (the skipped count on a first run IS the RSS overlap), and the wall time against a prior RSS-only `time uv run attest ingest --no-research --fulltext-limit 0`.

- [ ] **Step 3: Full text cost**

```bash
time uv run attest library fulltext --limit 10
uv run python -c "import sqlite3,os; c=sqlite3.connect(os.environ['ATTEST_DB']); print(c.execute('select source,count(*),avg(length(text)) from reference_fulltext group by source').fetchall())"
```

Record seconds per PDF and mean characters; if a PDF averages over ~5 s, lower `--fulltext-limit`'s default and say why.

- [ ] **Step 4: Payload sizes**

```bash
uv run python -c "
import json, os
from attestation.mcp import research as r
out = r._research('equivariant interatomic potentials', sources='arxiv', limit=8, store=False)
print(len(json.dumps(out, indent=2)))"
```

Record against 7000, alongside the `cite.lookup` window size the test already pins.

- [ ] **Step 5: Write `## Measured` into the spec, commit**

```bash
git add docs/superpowers/specs/2026-09-10-paper-research-design.md
git commit -m "Paper research: measured on a scratch copy of the live database -- first-run RSS overlap per topic, hourly wall time with topics, seconds per PDF, payload sizes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review notes (plan author)

- Spec coverage: topics-as-feeds (T4/T5), dispatch (T5), library exception (T5), clients (T3), `feed.research` (T6), routing (T7), full text (T9), BibTeX (T8), identifier dedup (T5), flag (T2/T5/T6), migration (T1), tools/CLI (T6/T10), skills/docs (T11), errors (T3/T5/T6/T9), offline-guarantee test (T2 flag tests + T5 disabled test + T9 clients-off; the zero-HTTP assertion is implicit in every test injecting `fetch`, and `_default_fetch` is the only place a request can originate), measurements (T12). `feed.sources` `kind`/`added_by` (T4). `cite.sources` offline flag reflecting the research flag (T6 step 4).
- Deviations from the spec, all recorded in T11 step 6: cap 8 not 13; "papers on X" stays local; `from-pub-date`.
- Names used across tasks: `research.Topic/parse_topic/topic_url/is_topic_url/Paper/as_entries/as_records/fetch_topic/Fetched/NullClient/null_clients/clients_from_env/research_enabled/FETCH_ERRORS/fetch_fulltext/pdf_text/parse_pmc`; `library.bibtex/bibtex_key/select_rows/export_bib/fulltext_window`; `feeds.add_source(added_by=)`; `run_ingest(clients=)`; `mcp.research._research/CLIENTS/MAX_RESEARCH_LIMIT/paper_row`; `citation.MAX_TEXT_CHARS`. Checked consistent.
