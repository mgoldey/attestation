# Paper research: standing topics as feeds, ad hoc search into the library, full text, BibTeX

**Date:** 2026-09-10
**Status:** approved in brainstorm 2026-09-10 (approach A of three: topics
are feeds; ad hoc search stores to the library; cross-feed item dedup by
identifier accepted). Supersedes `2026-09-04-paper-research-design.md`, which
lived only on the `feat/paper-research` branch and was never merged: it
designed a `papers` side table, its own dedup and its own BibTeX reader, and
claimed migration 007. The library store (`2026-09-05-library-store-design.md`)
took 007 and 008 and built the identity, provenance and search that spec
would have duplicated. That branch and its worktree can be deleted once this
spec lands; its plan is void.
**Depends on:** the library store (identity, `ReferenceRecord`, `upsert`,
the feed reader, `cite.lookup`), the feed/ingest design
(`2026-08-04-hermes-rss-design.md`: three-pass ingest, register-only
`add_source`), the citations domain (`2026-08-22-citations-domain-design.md`:
the offline guarantee and its construction-time flags), the tool-surface
rules (`2026-08-21-tool-surface-design.md`), and the response-size budgets in
`tests/test_response_size.py`.

## Why

Three requests landed on 2026-09-04 and share one shape:

1. store and retrieve arXiv and PubMed papers, with full text;
2. research topics automatically on the existing hourly schedule, and on
   demand from an agent;
3. search journals ad hoc, and pull a BibTeX entry for every paper stored.

The library store since built two-thirds of the first: every feed item
carrying a DOI or arXiv id (83% of the live database, measured 2026-09-05)
becomes a `references` row with per-source provenance, semantic search, and
opt-in enrichment from arXiv, CrossRef and Semantic Scholar. What it cannot
do is go and look. The feed subscribes to arXiv *categories* by RSS and
nothing else: there is no way to ask "what has been published on X" against
arXiv, PubMed or CrossRef, no way to search a specific journal, no standing
query a persona can register, no body text, and no BibTeX out. This spec
adds exactly those four things on top of the library rather than beside it.

## Decisions

### Topics are feeds

A standing topic is a row in `feeds` whose URL has the `research:` scheme:

```
research:arxiv,pubmed?q=equivariant+force+fields
research:crossref?q=protein+language+models&journal=Nature+Methods
```

The path names one or more clients (comma-separated, from `arxiv`,
`pubmed`, `crossref`); the query string carries `q` (required) and `journal`
(optional). `research.parse_topic(url) -> Topic(clients, query, journal)` is
pure and raises `FeedError` on an unknown client, an empty query, or a
`journal` given to a client that cannot filter by one (arXiv), naming the
valid clients in the message.

Everything that exists for a feed then exists for a topic with no new tool:
`feed.source_add` registers it, `feed.sources` lists it beside the RSS feeds
with its item count, `feed.source_remove` drops it (items found are kept,
as with any feed), `feeds.toml` can seed one, and the hourly
`attestation-refresh` job that already runs `attest ingest` refreshes it.
`add_source` validates a `research:` URL with `parse_topic` instead of
`feedparser`: there is no network round trip at registration, matching the
register-only rule. The default title is `<clients>: <query>` (plus
`in <journal>` when given).

Why feeds and not a `topics` table: the point of a standing topic is that
the persona *sees* what it finds. Items from a topic enter the same
candidate pool, ranking, digest, `feed.list`, `feed.read`, clicks and
engagement as everything else, because they are items. A separate store
would have needed its own listing tool and would never reach the digest.
The alternative that kept topics out of the feed (B in the brainstorm) was
rejected for that reason; the original side-table design (C) would have
been a second paper store with a second identity rule.

**Per-persona is provenance, not scoping.** `feeds` gains a nullable
`added_by` user id (migration 009). The candidate pool stays global and the
ranker sorts per reader, exactly as RSS feeds work today: a topic one
persona adds may surface for another persona whose profile it fits, which
is how `feed.source_add` has always behaved. `feed.source_add` gains an
optional `user` so the column is filled when the caller names one;
`feed.sources` reports `added_by` when set. "My topics" is therefore a
filter on `feed.sources`, not a new tool.

### Ingest dispatches on scheme; the pipeline does not change

`run_ingest` checks the URL scheme once per feed. An RSS feed goes through
`parse(url)` as today. A `research:` feed calls
`research.fetch_topic(topic, clients, since=last_fetched or 365 days ago)`
which returns an object with `.entries` shaped as the entries feedparser
yields (`title`, `link`, `summary`, `id`, `published_parsed`, and whatever
else `_new_entries`, `_published_iso` and the insert read), so the existing
three passes run unchanged: dedup, embed outside any transaction, one short
write. `id` (the guid) is `<client>:<external id>`, so `UNIQUE(feed_id,
guid)` holds across the clients a topic names, and `published` is the
paper's own date, never the fetch time, so a topic hit from 2019 does not
surface as new in `feed.list`.

After that commit, the same results are upserted into the library as
`ReferenceRecord`s with `source="research:<client>"`, `source_key` = the
external id, `fetched_at` = now, and the authors, venue, abstract, DOI,
arXiv id and PMCID the search payload carried. The library's own feed
reader links the item to that row on the next `cite.sync` through the shared
identity, so one paper found by a topic ends as one item and one reference
with two source rows.

**This is a stated exception to the library rule that the wire never
introduces a reference.** The three enrichers fill fields on rows a disk
source or the feed already had; a searcher creates rows. It is allowed
because the reader asked for exactly these papers by query, the source row
says `research:<client>` so `cite.lookup` shows where the row came from,
and `cite.sources`' `offline` flag reflects `ATTEST_RESEARCH_WEB` alongside
the two citation flags. Discarding the payload's authors and venue only to
re-fetch them through the opt-in enrichers would have been the alternative,
and it is wasteful.

### Search clients are pure over payloads and injected

`research.py` holds `ArxivSearch`, `PubmedSearch`, `CrossrefSearch`, each
with one public method:

```
search(query, *, journal=None, since=None, limit=50) -> list[Paper]
```

`Paper` is a frozen dataclass: `client`, `external_id`, `title`,
`abstract`, `authors: list[str]` ("Family, Given"), `published` (ISO date),
`doi`, `arxiv_id`, `pmcid`, `url`, `venue`, `pdf_url`. Two pure
conversions: `as_entries(papers)` for the ingest loop and `as_records(papers,
client)` for the library. The HTTP call is one private method; parsing each
payload into `Paper`s is a module-level pure function (`parse_arxiv(atom)`,
`parse_pubmed(xml)`, `parse_crossref(message)`) over a committed fixture
captured from a real response under `tests/fixtures/research/`. Wire XML
goes through `defusedxml`, as the library's arXiv enricher already does.
The clients take a `fetch` callable the way `run_ingest` takes `parse`, so
tests never touch the network.

| client | endpoint | `journal` | `since` | pacing |
|---|---|---|---|---|
| `arxiv` | `export.arxiv.org/api/query?search_query=all:<q> AND submittedDate:[<since> TO *]&sortBy=submittedDate` | refused by `parse_topic` (arXiv has categories, not journals) | in the query | 3 s between requests, as arXiv asks |
| `pubmed` | `eutils …/esearch.fcgi?db=pubmed&term=<q>[ AND "<journal>"[Journal]]&datetype=edat&mindate=<since>` then `efetch.fcgi?db=pubmed&rettype=abstract&retmode=xml` | `[Journal]` field | `mindate` | 3 requests/s, 10 with `NCBI_API_KEY` set |
| `crossref` | `api.crossref.org/works?query=<q>&query.container-title=<journal>&filter=from-pub-date:<since>` | `query.container-title` | `from-pub-date` | the polite-pool header the CrossRef enricher already sends |

`from-pub-date`, not `from-index-date`: a topic asks for papers *published*
since the last run, and index date would resurface old papers CrossRef only
recently indexed.

Search results are **not** cached: they change, and one request per client
per topic per hour is the whole load. (The enrichers' content-addressed
cache is for records that do not change.)

Nothing here ranks or summarises. The caller is the model.

### Ad hoc search is one tool, and it stores to the library, not the feed

`feed.research(query, sources="arxiv,pubmed", journal=None, since_days=365,
limit=10, store=True)` runs the named clients and returns `papers` (rows:
title, `authors[:6]` + `n_authors`, venue, published, doi, arxiv_id, url,
`stored: bool`), `stored` count, `offline`, and per-client `errors`. With
`store=True` every hit is upserted into the library under
`research:<client>` exactly as a topic hit is; with `store=False` it
previews. Hits never become items: the feed is what arrives on a schedule,
the library is what the reader keeps, and an ad hoc question is the second
kind. `limit` caps at 13, the `feed.list` ceiling, because the caller is a
model.

It lives in the `feed` namespace, not `cite`, because the chat surfaces run
with `ATTEST_TOOLS=feed` and a feed-only session must be able to reach it.
The surface goes from 48 to 49 tools; every doc and skill that quotes the
total is re-measured in the same change, and `test_architecture.py`'s
stale-count guard is what catches a missed one (it caught PR #6's merge
on 2026-09-10). `feed.research` belongs to the `feed` surface;
`feed.tools` lists it.

`feed.ask` routes: "find/search papers on X", "what has been published on
X", "in <journal>", "research X" go to `feed.research`; "track/follow/watch/
monitor X", "add a topic" go to `feed.source_add` with
`url="research:arxiv,pubmed?q=<X>"`; "my topics / what am I tracking" go to
`feed.sources`. "Search my feed for X" and "search for X" stay on
`feed.search`: the router keys on going-and-looking verbs and the words
"papers", "published", "journal", "research", never on "search" alone, and
the existing `test_ask_routing.py` cases for `feed.search` must still pass.

### Full text is a capped, separate pass, served in windows

Migration 009 adds `"references".pmcid` and:

```sql
CREATE TABLE IF NOT EXISTS reference_fulltext(
  reference_id INTEGER PRIMARY KEY REFERENCES "references"(id) ON DELETE CASCADE,
  text TEXT,
  source TEXT NOT NULL,      -- arxiv-pdf | pmc-xml | none
  fetched_at TEXT NOT NULL
);
```

`research.fetch_fulltext(conn, *, limit=10, fetch=...)` selects references
with an `arxiv_id` or a `pmcid` and no `reference_fulltext` row, newest
`first_seen` first, and tries the arXiv PDF (`https://arxiv.org/pdf/<id>`,
extracted with `pypdf`, a new pure-Python dependency) or PMC open-access
XML (`efetch.fcgi?db=pmc&id=<pmcid>`, `defusedxml`, body paragraphs
joined). A paper with neither, or whose PDF `pypdf` cannot read, gets a
row with `source='none'` and `text NULL` so it is not retried every hour;
the reason goes to the log. The cap is per run: the hourly refresh cannot
pull hundreds of PDFs, and the backlog drains over successive runs. It runs
at the end of `attest ingest` (`--fulltext-limit N`, default 10; `0`
disables) and on demand as `attest library fulltext [--limit N]`.

Full text is never embedded (the vector is for ranking beside RSS items,
and a 30-page body would swamp the abstract's signal) and never returned
whole. `cite.lookup` gains `text_offset: int = 0` and `text_chars: int =
3000` and returns `full_text: {text, offset, chars, total, source}` or
`null`, so a model pages through a body. The window plus the record must
stay under the read budget `tests/test_response_size.py` defines.

### BibTeX is rendered deterministically from the row

`library.bibtex(row, *, key=None) -> str` is pure. `@article` when `venue`
is set and is not a preprint server (the `_PREPRINT_VENUE` pattern the
library already has); otherwise `@misc` with `eprint = {<arxiv_id>}`,
`archivePrefix = {arXiv}`, and `primaryClass` when known. Fields: author
(joined with ` and `), title, journal or `howpublished`, year, doi, url.
The key is `bib_key` when a `.bib` or Zotero supplied one, else
`<first author family><year><first title word>` through the library's
NFKD normaliser so it is ASCII; within one export, a collision takes a
letter suffix in row-id order, so the same rows always render the same
file. `cite.lookup` returns `bibtex` on every record it finds in the store.

`attest library export --bib PATH [--tag T] [--author A] [--year Y]
[--source S]` writes the filtered set as one `.bib`, NEW FILES ONLY: it
refuses to overwrite unless `--force`, the discipline `attest runs record`
established. No DOI content negotiation (the library spec ruled out CSL and
writing back to Zotero; rendering our own record offline is enough and is
what makes every stored paper citable), and never writing into a `.bib`
the reader already has.

### Cross-feed item dedup by identifier

`_new_entries` today skips an entry whose guid exists under the same feed
or whose `content_hash(title, summary)` exists anywhere. It gains a third
check: an entry whose extracted DOI or arXiv id already exists in `items`
under any feed is skipped. Measured on the live database 2026-09-05, 207 of
7,787 identified items were the same arXiv paper announced in two feeds
(cs.LG and chem-ph cross-lists); a topic hit for a paper the RSS already
carried would have been a third copy with a slightly different abstract and
therefore a different hash. The library already merges these into one
reference, so this only stops `feed.list` showing one paper twice. The
skipped count in the ingest outcome includes them.

### One flag, on by default, read at construction

`ATTEST_RESEARCH_WEB` (unset or `1` = on; `0`/`false` = off) is read once
in `research.clients_from_env()`, called when `run_ingest` starts and when
the MCP server registers `feed.research`. Off, every client is a
`NullClient` returning `[]` with `offline = True`: research feeds are
counted as skipped in the ingest outcome with `error="research disabled"`,
`feed.research` says `offline: true` in its message rather than returning
an empty list that looks like "no results", and `fetch_fulltext` does
nothing.

On by default, unlike `ATTEST_CITATION_WEB` and `ATTEST_CITATION_SCHOLAR`:
`attest ingest` already fetches RSS over the network, and a topic query is
the same class of call in the same command. The offline guarantee's
statement in CLAUDE.md becomes: everything is local except RSS ingest,
research queries and full text (on by default, one flag), and the CrossRef,
arXiv and Semantic Scholar enrichers (off by default, two flags); every
flag is read at construction, never per call, so a disabled client cannot
be coaxed into a request. `cite.search` and `feed.search` still never fan
out to the network.

## Data model (migration 009)

All additive and guarded, as 007 and 008 are:

```sql
ALTER TABLE feeds ADD COLUMN added_by INTEGER REFERENCES users(id);   -- guarded by PRAGMA table_info
ALTER TABLE "references" ADD COLUMN pmcid TEXT;                        -- guarded
CREATE TABLE IF NOT EXISTS reference_fulltext( ... );                  -- above
```

`db.SCHEMA` gains the same DDL for fresh databases, single-sourced with the
migration as the library schema is. Application tables go from 16 to 17; the
count CLAUDE.md asserts and `test_db.py` pins move with it. No synthetic
feed rows are created by the migration: a topic exists only when someone
registers one.

## Tools and CLI

| tool | change |
|---|---|
| `feed.research(query, sources, journal, since_days, limit, store)` | **new**; described above |
| `feed.source_add(url, title=None, user=None)` | accepts `research:` URLs, validated by `parse_topic`, no network; fills `added_by` when `user` names a persona |
| `feed.sources()` | rows carry `kind: rss|research` and `added_by` |
| `feed.source_remove` | unchanged; works on a topic like any feed |
| `cite.lookup(key, text_offset=0, text_chars=3000)` | adds `bibtex` and the `full_text` window |
| `cite.sources()` | `offline` also reflects `ATTEST_RESEARCH_WEB`; counts `research:*` source rows and `reference_fulltext` |
| `feed.ask` | the routes above |

CLI:

```
attest sources add URL [--user NAME]
attest research QUERY [--sources arxiv,pubmed] [--journal J] [--since-days N] [--limit N] [--no-store]
attest ingest [--fulltext-limit N] [--no-research]
attest library fulltext [--limit N]
attest library export --bib PATH [--tag T] [--author A] [--year Y] [--source S] [--force]
```

No CLI command wraps `add_source` today (`feed.source_add` is its only
caller), so `attest sources add URL [--user NAME]` is added: a general verb
that takes an RSS URL or a `research:` one, not a research-specific command.
`docs/reference/cli.md` is regenerated by `scripts/render_cli_reference.py`.

## Skills and docs

`attestation-feed/SKILL.md` gains a "Research a topic" section naming
`feed.research`, the `research:` URL form for `feed.source_add`, and the
search-versus-research distinction in one sentence each; it names only
tools the feed surface serves (`test_skill_files.py`). `attestation-knowledge/
SKILL.md` gains one line on `cite.lookup`'s `bibtex` and full-text window.
`docs/guides/feed.md` gets the same in prose plus the flag; `docs/guides/
claims-and-citations.md` gets the export command. CLAUDE.md's Key API
Patterns gets one `|Research:` entry and the counts it asserts are updated.

## Error handling

- A network or parse failure in any client is an absent source: the client
  returns `[]` and records the error string; `feed.research` reports it
  under `errors`; `run_ingest` counts it in that feed's outcome as it does
  an RSS failure. Nothing raises past the client boundary except a
  programming error, which the `@tool` envelope turns into `ok: False` as
  everywhere.
- The embedder being down is handled by `run_ingest`'s existing policy;
  topic items are items.
- A `research:` URL that does not parse is a `FeedError` from `add_source`,
  which `feed.source_add` already turns into a `ToolError` naming the fix.
- An unknown name in `feed.research`'s `sources` is a `ToolError` listing
  the three clients.
- A PDF `pypdf` cannot read, or a PMC id with no open-access body, marks
  `source='none'` and is not retried.

## Testing

- `tests/test_research.py`: each parser against its fixture
  (`tests/fixtures/research/{arxiv.atom,pubmed-esearch.xml,pubmed-efetch.xml,crossref.json,pmc.xml}`);
  `parse_topic` accepts the two URL forms above and refuses unknown clients,
  empty queries, and `journal` with arXiv; `as_entries` produces what
  `_new_entries` and `_published_iso` read; `as_records` carries authors,
  venue, abstract and identifiers; `bibtex()` is deterministic, ASCII-keyed,
  `@article` vs `@misc` by venue, collision suffixes stable; `NullClient`
  when the flag is `0`, read at construction; `fetch_fulltext` respects the
  cap, marks `none`, does not retry, and never runs with the flag off.
- `tests/test_ingest.py`: a `research:` feed dispatches to the injected
  client and never to `parse`; its items carry guid `<client>:<id>`,
  `published` = the paper's date, and `doi`/`arxiv_id`; the same run leaves
  one `references` row per paper with a `research:<client>` source row; a
  second run with the same payload adds nothing (hourly idempotency);
  `since` is `last_fetched`; cross-feed identifier dedup skips an arXiv
  cross-list and a topic hit the RSS already carried, and the skipped count
  says so; a failing client is one feed's error and the RSS feeds still
  ingest.
- `tests/test_feeds.py`: `add_source` on a `research:` URL makes no network
  call and stores the title and `added_by`; `list_sources` reports `kind`.
- `tests/test_library.py`: the feed reader links a topic item to the row the
  research source made (one row, two source rows); `cite.lookup` returns
  `bibtex` and a full-text window with the right `total`; the export writes
  a stable file and refuses to overwrite.
- `tests/test_ask_routing.py`: the new routes, and every existing
  `feed.search` case still routes to `feed.search`.
- `tests/test_db.py`: migration 009 on a real v8 file; application table
  count.
- `tests/test_architecture.py`: 49 tools, namespace rules, the stale-count
  guard over the docs.
- `tests/test_response_size.py`: `feed.research` at limit 13 and
  `cite.lookup` with a 3000-char window stay under their budgets.
- Offline guarantee: with `ATTEST_RESEARCH_WEB=0` and both citation flags
  unset, a full `attest ingest` over RSS fixtures plus a registered topic,
  a `feed.research` call, and a `cite.sync` issue zero HTTP requests
  (monkeypatched `httpx` that raises).

## Measurements to take before this ships

Recorded in this file's §Measured when taken, as the library spec did:

- how many of one topic's first-run hits the RSS already carried
  (the overlap the identifier dedup exists for), on a cs.LG-shaped query
  and a chemistry one;
- wall time of one hourly refresh with three topics registered, against
  the RSS-only baseline;
- `pypdf` seconds per arXiv PDF and characters extracted, over the first
  ten, so the default cap is a number not a guess;
- `feed.research` payload size at limit 13 and `cite.lookup` with a window,
  against the budgets.

## Out of scope

- Summarising or ranking papers with a model inside a tool (the caller is
  the model; digest already composes).
- Semantic Scholar search, Google Scholar, bioRxiv (bioRxiv has RSS; add it
  as a feed). Semantic Scholar stays an enricher.
- Alerting new papers to Discord/Telegram (the digest path exists).
- OCR for scanned PDFs; embedding full text; chunked retrieval over bodies.
- Folding a DOI-only row with its arXiv-only twin once an enricher links
  them (deferred by the library spec; still deferred).
- Writing into a `.bib` the reader already maintains; CSL rendering.
- Per-persona topic *scoping* (a topic only one persona's feed sees). If
  it is ever wanted, it is a ranker-side filter on `added_by`, not a
  second store.

## What this spec does not decide

Whether `kg.ask` gets a route to `feed.research` (the knowledge surface
cannot see it; a hand-off phrase in the knowledge skill may be enough).
Whether topic items should be tagged by `attest tag` differently from RSS
items (no reason yet). What `feed.digest` should say about a topic's first
run, which can add dozens of items at once with old `published` dates (the
14-day window already hides most of them; measure before designing).
