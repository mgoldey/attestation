# Library store: one deduplicated reference library, searchable, with provenance

**Date:** 2026-09-05
**Status:** approved in brainstorm 2026-09-05 (approach A of three; the user chose
"both from the start", "semantic + knowledge graph", "also Semantic Scholar
reference lists", and a generated molecular-AI example). Spec 1 of 3; specs 2
(references in the graph + `examples/molecular-ai/`) and 3 (skill optimization
with agentopt) build on it.
**Depends on:** `2026-08-22-citations-domain-design.md` (readers, the offline
guarantee, `Reference`), `2026-08-06-knowledge-graph-design.md` (why the graph
is pure and why citations are a sibling store).

## Problem

The citations domain can answer "does key X resolve?" against a `.bib` file in
the current directory or a Zotero library, first answer wins, substring search,
nothing persisted. It cannot:

- hold the same paper once when Zotero and two `.bib` files all have it under
  different keys (`Resolver.search` dedups by key only);
- see the papers the reader actually read: feed items carry no DOI or arXiv
  id, so nothing joins them to a reference;
- find a paper by what it is about ("equivariant force fields") rather than
  by a word in its title;
- know what a paper cites. The knowledge-graph spec refused to invent `CITES`
  edges from co-occurrence, correctly. Real reference lists exist at Semantic
  Scholar and were out of scope.

The August roadmap left "whether citations become KG nodes or a sibling store"
open, and the citations spec answered "sibling store, a `references` table
with an optional `item_id` FK". That table was never built. This spec builds
it, as a store rather than a table, because the interesting part is the
identity rule that lets three sources fill one row.

## Decisions already made

- **Sibling store, not items.** References are not feed items: `items` has no
  room for provenance, `feed.list` would surface a 2019 paper as new, and
  "cited" is a stronger signal than "read" (citations spec, Problem). The
  store reuses the item *pipelines* (embedding, tagging) without sharing the
  table.
- **The concept graph stays co-occurrence only.** `kg.build_graph` takes
  `(id, tag)` pairs and does not care what an id names, so references join
  the graph in spec 2 by supplying their own ids. *Superseded 2026-09-05 by
  spec 2:* they join as NEGATIVE ids, `(-reference_id, tag)`, not `ref:<id>`
  strings -- the id type stays `int` and the sign cannot collide with an
  item id. Citation edges live in their own table and are never mixed into
  concept adjacency.
- **Identity is a pure function.** DOI, else arXiv id, else normalised title
  and year. Tested without a database.
- **Network stays opt-in and construction-time.** `ATTEST_CITATION_WEB`
  keeps its meaning (CrossRef, now also arXiv). Semantic Scholar is a second
  flag, `ATTEST_CITATION_SCHOLAR`, because reference lists are a larger surface
  and a rate-limited one. Both are read when the sync's readers are built,
  never per call. `search` never fans out to the network, as today.
- **No model in the store.** Tagging references calls the existing tagging
  prompt, which is a model call the reader opts into with `attest library
  tag`, exactly as `attest tag` does for items. Identity, merge, sync and
  fielded search are deterministic.

## Scope

In: the tables, the identity and merge rules, six readers behind the existing
`CitationPort` shape, an idempotent sync, DOI/arXiv extraction for feed items,
fielded and semantic search, the `cite.*` tools that expose it, the CLI, and
the docs and tests that pin the counts this repo asserts.

Out (spec 2): references as graph participants, a citation-neighbourhood tool,
`examples/molecular-ai/`. Out (spec 3): skills. Out entirely: writing back to
Zotero or `.bib`, CSL rendering, an LLM judging whether a paper supports a
claim.

## 1. Data model

Migration **007** (006 is `runs.scanned_at`). All additive; every statement is
`IF NOT EXISTS` or guarded, so the ladder stays idempotent.

```sql
CREATE TABLE IF NOT EXISTS "references" (
  id         INTEGER PRIMARY KEY,
  identity   TEXT NOT NULL UNIQUE,   -- see §2
  doi        TEXT,                    -- lowercase, no scheme
  arxiv_id   TEXT,                    -- versionless, e.g. 2106.02347
  title      TEXT NOT NULL,
  authors    TEXT NOT NULL DEFAULT '[]',  -- JSON list of "Family, Given"
  year       INTEGER,
  venue      TEXT,
  abstract   TEXT,
  url        TEXT,
  bib_key    TEXT,                    -- first .bib/Zotero key seen; for cite= in claims
  first_seen TEXT NOT NULL,
  updated    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reference_sources (
  reference_id INTEGER NOT NULL REFERENCES "references"(id) ON DELETE CASCADE,
  source       TEXT NOT NULL,         -- bibtex:<path> | zotero | feed | arxiv | crossref | s2
  source_key   TEXT NOT NULL,         -- bib key | zotero item id | items.id | external id
  fetched_at   TEXT,                  -- NULL = read from disk
  raw          TEXT NOT NULL,         -- JSON: the fields this source contributed, plus conflicts
  PRIMARY KEY (reference_id, source, source_key)
);
CREATE TABLE IF NOT EXISTS reference_tags (
  reference_id INTEGER NOT NULL REFERENCES "references"(id) ON DELETE CASCADE,
  tag          TEXT NOT NULL,
  PRIMARY KEY (reference_id, tag)
);
CREATE TABLE IF NOT EXISTS reference_cites (
  citing_id      INTEGER NOT NULL REFERENCES "references"(id) ON DELETE CASCADE,
  cited_identity TEXT NOT NULL,       -- an identity string; usually NOT a row here
  cited_title    TEXT,
  source         TEXT NOT NULL,       -- s2 today
  fetched_at     TEXT NOT NULL,
  PRIMARY KEY (citing_id, cited_identity)
);
CREATE VIRTUAL TABLE IF NOT EXISTS reference_vectors USING vec0(embedding float[<dims>]);
-- plus the same AFTER DELETE trigger item_vectors has, so a removed reference
-- takes its vector with it.
ALTER TABLE items ADD COLUMN doi TEXT;        -- guarded by PRAGMA table_info
ALTER TABLE items ADD COLUMN arxiv_id TEXT;
```

`references` is quoted because it is a SQL keyword. `db.SCHEMA` gains the
same DDL for fresh databases, single-sourced with the migration as 002 is.
`reference_vectors` is created by the same `_vec_schema`-style helper as
`item_vectors`, so `EMBED_DIMS` mismatches are refused for both tables by the
one existing check.

`reference_cites.cited_identity` is a string, not a foreign key, because most
of a paper's references are not in the library. Spec 2 decides what a
citation neighbourhood shows for a cited identity with no row; this spec only
stores it.

**Backfill.** Migration 007 also fills `items.doi` and `items.arxiv_id` from
`guid` and `url` for existing rows, using the same pure extractor ingest uses
from then on (§3.4). Measured on the live database 2026-09-05: arXiv guids are
`oai:arXiv.org:1003.0563v2`, Nature URLs are
`https://www.nature.com/articles/s41467-026-74391-4`, whose DOI is
`10.1038/s41467-026-74391-4`. This is the first migration that writes data,
which is why the ladder's atomicity note in `db.py` matters: it runs once.

## 2. Identity and merge

`library.identity(doi, arxiv_id, title, year) -> str`, pure:

1. `doi:<doi lowercased, scheme and `doi.org/` prefix stripped>` if a DOI is
   present;
2. else `arxiv:<id with version suffix removed>` (`2106.02347v3` →
   `2106.02347`; old-style `cond-mat/0301234` kept as is);
3. else `title:<normalised>:<year or ->`, where normalised = lowercase,
   Unicode NFKD with combining marks dropped, every run of non-alphanumerics
   collapsed to one space, leading articles kept (removing them merges
   "A survey" with "Survey" — measured on the tagging corpus, not worth it).

DOI beats arXiv because an arXiv preprint that is later published gets a DOI
while keeping its arXiv id; a record that knows both carries both columns, so
a later record with only the arXiv id still finds it via the `arxiv_id`
column lookup that `upsert` tries before falling back to identity equality.
Concretely, `upsert` looks for an existing row by, in order: identity, DOI,
arXiv id. The first hit is the row to merge into; the row's identity is
upgraded to the DOI form if it was arXiv- or title-based and the incoming
record has a DOI.

**Merge** (`library.merge(existing_fields, incoming_fields) -> (merged,
conflicts)`, pure): an empty existing field takes the incoming value; a
non-empty one keeps its value, and a differing incoming value is recorded in
`conflicts` as `{field: {"kept": ..., "offered": ...}}`, which the sync
writes into the source row's `raw`. Longer abstracts do not win; first wins.
This is deliberately dumb: the point is that nothing is overwritten silently,
and `cite.lookup` shows every source row so a reader can see the disagreement.

`authors` are compared after the same normalisation as titles; a source that
offers more authors than the row has (a `.bib` truncated with "and others")
is not a conflict, it fills.

## 3. Sources and sync

### 3.1 Readers

Six readers, each a class with `name`, `network: bool`, and
`records() -> Iterator[ReferenceRecord]`. `ReferenceRecord` is the incoming
shape: the `references` columns as optional fields plus `source`,
`source_key`, `fetched_at`, and `cites: list[tuple[identity, title]]`.
`citations.Reference` stays as the outgoing shape the `cite.*` tools already
emit; `library.to_reference(row)` converts.

| reader | network | what it yields | notes |
|---|---|---|---|
| `bibtex:<path>` | no | one record per `@entry` | reuses `citations._parse_bib`; adds `abstract`, `journal`/`booktitle` → venue, `eprint` with `archiveprefix = arXiv` → `arxiv_id` |
| `zotero` | no | one per non-deleted item with a title | adds `abstractNote`, `publicationTitle`, `extra` parsed for `arXiv:` lines |
| `feed` | no | one per item with `doi` or `arxiv_id` | `source_key = items.id`; title, summary → abstract, published → year, url |
| `arxiv` | yes | metadata for ids the store already has | `http://export.arxiv.org/api/query?id_list=` in batches of 50, Atom parsed with `defusedxml` (not `xml.etree` as first written: the body came off the wire, and an entity-expansion payload would otherwise be parsed with no limit); fills abstract, authors, title, DOI when arXiv knows it |
| `crossref` | yes | metadata for DOIs the store already has | the existing `WebReader` cache and endpoint; adds venue (`container-title`), authors |
| `s2` | yes | reference lists for rows with a DOI or arXiv id | `api.semanticscholar.org/graph/v1/paper/<DOI:|arXiv:>id?fields=externalIds,title,references.externalIds,references.title`; one request every three seconds (the pace measured 2026-09-05, spec 2 §7), unauthenticated, honouring `Retry-After` (parsed defensively, floor 10 s, cap 60 s); each reference with an id, or a title of at least four words, becomes a `reference_cites` row with identity computed from its external ids or title |

The three network readers are **enrichers**: they never introduce a
reference on their own. A row exists because a disk source or the feed had
it; the wire only fills fields and adds edges. That is what keeps `cite.sources`'
`offline: true/false` honest and the library the reader's, not the web's.

Every network response is cached through the existing content-addressed
cache (`~/.hermes/citation-cache/`, 0700/0600), keyed by URL, never expiring,
with `fetched_at` preserved on a hit (citations spec: the cache must not
launder a wire record into one that looks local). S2 adds a 429/5xx backoff:
sleep `Retry-After` or 2 s, three attempts, then record the failure in the
sync report and continue.

### 3.2 Configuration

`.env` gains:

```
ATTEST_BIB_PATHS=            # colon-separated .bib files; unset = *.bib in cwd (today's rule)
ATTEST_ZOTERO_PATH=          # unset = ~/Zotero/zotero.sqlite if it exists (today's rule)
ATTEST_CITATION_WEB=         # existing: CrossRef; now also the arXiv API
ATTEST_CITATION_SCHOLAR=          # new: Semantic Scholar reference lists
```

`Resolver.from_env` keeps its behaviour and grows the two path variables, so
`cite.check` and the CLI see the same libraries the sync does.

### 3.3 Sync

`attest library sync [--sources bibtex,zotero,feed,arxiv,crossref,s2] [--limit N]`
and the `cite.sync` tool run the same `library.sync(conn, readers,
embedder=None, limit=None) -> SyncReport`:

1. For each offline reader in order (bibtex paths in the order given, zotero,
   feed): `upsert` every record. Idempotent: a second run with unchanged
   sources changes no row and reports `unchanged`.
2. For each armed network reader: enrich rows that lack the fields it
   provides, oldest `updated` first, up to `--limit`. A row already enriched
   by that source (a `reference_sources` row exists) is skipped, so re-running
   costs nothing on the wire beyond cache hits.
3. If an embedder is given, embed rows without a vector: `embed_document(title,
   abstract or "")`, `truncate_normalize`, insert into `reference_vectors`.
   An embedder that raises leaves the row unembedded and is reported once;
   the sync completes.

`SyncReport` is structure, not prose: per source `{seen, added, merged,
unchanged, enriched, failed}`, plus `embedded`, `unembedded`, and
`conflicts` (count, with the first five `(identity, field)` pairs). The tool
returns it as its envelope's body; the CLI prints one line per source.

The sync runs inside short transactions per reader, the same discipline
`ingest` follows so `attest serve` can run alongside it.

### 3.4 Feed identifiers

`ingest.extract_ids(guid, url) -> (doi | None, arxiv_id | None)`, pure, used
by migration 007's backfill and by every new item:

- arXiv: `oai:arXiv.org:<id>v<n>` in guid, or `arxiv.org/abs/<id>` in url;
  version stripped.
- DOI: a `10.\d{4,9}/\S+` match in url or guid, after a `doi.org/` prefix if
  present; Nature's `/articles/<suffix>` form is mapped to `10.1038/<suffix>`
  because the live Nature feeds carry the suffix and not the DOI.

Anything else stays NULL. No network lookup at ingest time.

### 3.5 Tagging

`attest library tag [--limit N]` renders `features.tag_messages(title,
abstract, vocabulary)` for each untagged reference, the ONE renderer the
tagging eval pins, and writes `reference_tags`. Same validation
(`ItemTags`), same vocabulary steering (`tag_vocabulary(conn)`, which spec 2
extends to count reference tags too). Measured cost for items is ~2.3 s each
on gemma4:e2b; the CLI says so and takes `--limit`.

## 4. Search

`library.search(conn, query, *, embedder=None, author=None, year=None,
year_from=None, year_to=None, tag=None, source=None, limit=10) ->
list[SearchHit]`:

- **Fielded** filters are exact after normalisation: `author` matches any
  normalised author surname, `year`/`year_from`/`year_to` on the column,
  `tag` on `reference_tags`, `source` on `reference_sources.source` prefix
  (`bibtex`, `zotero`, `feed`). A query that is a DOI, an arXiv id, or a bib
  key is looked up directly and returned as the single hit.
- **Semantic** ranking when an embedder is available and `reference_vectors`
  has rows: `embed_query(query)`, KNN over `reference_vectors` for `4 ×
  limit` candidates, then the fielded filters, then the relative floor
  `search_feed` uses (`RELEVANCE_FLOOR` of the best hit — absolute cutoffs
  failed there because top similarity varies by query). Each hit carries
  `similarity`.
- **Literal boost, not floor**: a hit whose title or abstract contains the
  query words moves up but nothing is excluded for lacking them (the feed
  lesson: flooring on a literal made all 711 "llm" matches tie).
- **Fallback**: no embedder, or no vectors yet, means fielded and substring
  search over title, abstract, authors, bib key, ordered by year desc, and
  the result says `semantic: false` with the reason. A caller cannot mistake
  a substring scan for a semantic one.

`SearchHit` = `Reference.to_row()` fields plus `similarity | None`,
`sources: [names]`, `n_tags`, `tags[:3]`. Payload sizes follow
`test_response_size.py`'s ceiling; `limit` caps at 13 like `feed.list`.

## 5. Tools and CLI

All under the existing `cite.*` namespace; no new namespace, no tool repeats
its namespace. The tools that touch the store are `needs_db=True`; the
existing three keep working with an empty store because they fall back to
the disk readers.

| tool | change |
|---|---|
| `cite.lookup(key)` | store first (identity, DOI, arXiv id, bib key), then the disk readers as today. The row reports `sources: [{source, source_key, fetched_at}]` and `conflicts` when any. |
| `cite.search(query, limit=5, author=None, year=None, tag=None)` | `library.search`. `semantic: true/false` and `caveat` in the envelope. Falls back to today's substring scan of the readers when the store is empty, and says so. |
| `cite.sources()` | adds `store: {references, with_vectors, with_tags, with_cites}` and per-source counts; `offline` now reflects `ATTEST_CITATION_SCHOLAR` too. |
| `cite.sync(sources=None, limit=None)` | **new**. Runs `library.sync` with the readers `from_env` arms. Returns `SyncReport`. Network readers run only if their flag was set when the server started — the tool cannot arm one. |
| `cite.check(path)` | unchanged; `check_citations` resolves through the store as well, so a key in the store but in no `.bib` in cwd no longer lints as uncited. |

That is 47 tools (46 + `cite.sync`); `test_architecture.py`'s count
assertions and the docs that quote 46 (README, agents guide, CLAUDE.md,
attestation-setup and attestation-knowledge skills) move to 47 in the same
change. `cite.sync` belongs to the `knowledge` surface (spec: knowledge owns
`cite.lookup/search/sources`); `kg.ask` gains no route for it — a sync is an
action the reader asks for by name, not a question.

CLI:

```
attest library sync   [--sources ...] [--limit N]
attest library search QUERY [--author A] [--year Y] [--tag T] [--limit N]
attest library tag    [--limit N]
attest library embed  [--limit N]
attest library status
```

`attest ingest` extracts ids for new items as part of the same insert. The
CLI reference under `docs/reference/cli.md` is regenerated by
`scripts/render_cli_reference.py`.

## 6. Errors and the offline guarantee

- An absent `.bib` or Zotero file is empty, not an error (citations spec).
- With neither flag set, a full `sync` issues zero HTTP requests. Test:
  monkeypatched `httpx` that raises on any call, full sync over fixtures.
- `search` never calls a network reader, armed or not (existing test,
  extended to the store path).
- A network failure mid-sync is recorded per row in `SyncReport.failed` and
  the sync continues; the cache means a retry is cheap.
- Embedder down: rows are stored without vectors; `search` degrades to
  fielded with `semantic: false`; `attest library embed` catches up later.
  This mirrors `rank.py`'s policy (serve stale, never 500).
- The migration refuses nothing new; older code opening a migrated database
  is refused by the existing user_version check. **Operational note**: the
  Hermes gateway serves attest-mcp from the main checkout, so this branch is
  developed in a worktree and tested against scratch databases (`ATTEST_DB`)
  until merged. Opening the live database from this branch would lock the
  gateway's older code out of it.

## 7. Testing

Pure, DB-free: `identity()` (DOI beats arXiv beats title; version stripping;
old-style arXiv ids; NFKD title normalisation), `merge()` (fill, keep,
conflict recorded, author extension is not a conflict), `extract_ids()` on
the measured live formats, `library.to_reference()`.

Database, hermetic (`conftest`'s scratch path): migration 007 on a real v6
file with items whose guid/url are the live formats, asserting the backfill
and that a second open is a no-op; `upsert` merges a `.bib` and a Zotero
fixture that share a DOI under different keys into one row with two source
rows; the feed source links an item by arXiv id; sync idempotency (run twice,
second report all `unchanged`); `reference_vectors` delete trigger;
`search` with `fake_embedder` ranks the semantically closer title first and
reports `semantic: true`, and without an embedder reports `semantic: false`.

Network, offline-verified: arXiv Atom and S2 JSON parsed from committed
fixture responses; the cache hit preserves `fetched_at`; S2 429 backs off and
records the failure; zero requests with flags unset.

Tools: `cite.sync` envelope shape and `empty=`; `cite.search` payload under
the response-size ceiling at `limit=13`; `cite.lookup` shows conflicts;
`test_architecture.py` count and namespace rules; `test_docs_site` and the
stale-count guard after the 46 → 47 edits; `test_skill_files` after the
knowledge skill names `cite.sync`.

## 8. Measurements

Taken 2026-09-05 against a scratch copy of the live database (9,407 items,
13 feeds), models pinned in Ollama:

- **Feed-id yield: 7,787 of 9,407 items (83%) carry a DOI or arXiv id.**
  Every arXiv item (6,105 cs.LG + 340 chem-ph) and every Nature-family item
  (Nature 443, Scientific Reports 469, Nature Communications 321, Materials
  39, Chemistry 35, Machine Intelligence 29) yields one; Hugging Face, Simon
  Willison, Ars Technica and Quanta yield none, and Hacker News 6 of 559 --
  the posts that link straight to arXiv or a DOI. As predicted, with HN the
  only surprise.
- **Cross-list dedup is real on the feed alone: 7,787 source rows collapsed
  to 7,580 references.** The 207 merges are the same arXiv paper announced
  in two feeds (cs.LG and chem-ph cross-lists), which is exactly the case
  the identity rule exists for. 7,580 is an upper bound on distinct papers:
  a journal item (DOI only) and its preprint (arXiv id only) share no column
  and stay two rows until an enricher supplies the missing id -- and when
  it does, the rows are not folded (review round 1): the collision is
  recorded on the source row and both keep their provenance. Folding is a
  later spec.
- **Embedding: 100 references in 33.9 s wall through the CLI (≈0.32 s each,
  including process start), so the full feed-derived library is a ~40-minute
  one-off `attest library embed`.** `sync --limit N` bounds the embed pass;
  the default is unbounded because a partial index says `semantic: true`
  with a caveat, which is the honest state either way.

Taken, in spec 2 §6 and `examples/molecular-ai/README.md`: semantic vs
substring on ten molecular-AI queries against the example library, with an
expected paper written down per query; and Semantic Scholar wall time for a
48-paper library -- five resumed passes at three seconds per request under
the shared unauthenticated rate limit, not the minute first estimated.

## What this spec does not decide

How a cited identity with no row is shown (spec 2). Whether references should
rank in `feed.search` (no, for now: the feed is what arrived, the library is
what the reader keeps; `cite.search` is the library's search). Writing back
to `.bib`.
