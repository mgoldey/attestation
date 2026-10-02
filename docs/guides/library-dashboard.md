# The library dashboard

How does a page show what the library holds? `attest library dashboard` writes
one JSON file, `library.json`, that a host page (AgentMarkit's Files tab) reads
whole: totals, a weekly ingest chart, the busiest sources, and one searchable row
per reference. It reads the database and nothing else, needs no model, works on
an empty database, and the hourly refresh keeps it fresh.

## Running it

```bash
attest library dashboard                       # to the default path
attest library dashboard --out ./library.json --persona owner --limit 5000
```

The path is `--out`, else `$ATTEST_LIBRARY_DASHBOARD`, else
`<parent of HERMES_HOME>/.hermes/workspace/research-desk/library/library.json`
(the folder is created). The write is atomic -- a temp file in the same folder,
fsynced, renamed over the target -- and the file is mode 0600, so a reader sees
the previous file or the new one, never half. Running it twice on the same
database gives the same bytes except `generated_at` (sorted keys, stable
orderings with an id tie-break).

The refresh script `attest install` writes runs it as its last step, **degraded**
like the desk and the bibliography: a failure prints `dashboard FAILED (exit N)`
and never fails the refresh or the ingest before it.

## The contract (schema 1)

AgentMarkit is built against this. Fields may be **added**; none is renamed,
removed or retyped without a new `schema` number.

```json
{
  "schema": 1,
  "generated_at": "2026-10-02T21:08:59Z",
  "persona": "owner",
  "totals": {"items": 48, "references": 48, "tagged": 96, "embedded": 96,
             "with_fulltext": 0, "saved": 0},
  "last_refresh": "2026-10-02T21:08:21Z",
  "ingested": [{"week": "2026-W39", "start": "2026-09-21", "items": 3, "references": 0}],
  "by_source": [{"source": "Molecular AI preprints", "kind": "feed", "items": 48}],
  "failures": {"fetch": null, "tag": null, "embed": null, "since": null},
  "references": [{"key": "batzner2022equivariant", "title": "...", "authors": ["..."],
                  "year": 2022, "venue": "Nature Communications", "doi": "10.1038/...",
                  "arxiv": "2101.03164", "url": "https://...", "tags": ["..."],
                  "sources": ["bibtex:references.bib"], "added": "2026-10-02T21:08:00Z",
                  "saved": false, "fulltext": false}],
  "references_truncated": false,
  "references_limit": 5000
}
```

What each number means -- decided in `src/attestation/dashboard.py`, stated here:

- **`items`** is feed items ingested (the raw material of the ranked reading
  list); **`references`** is rows in the library store (DOI- or arXiv-identified
  references from `.bib` files, Zotero, saved papers and feed items that carry
  an id -- see [Claims and citations](claims-and-citations.md#the-library)).
- **`tagged`** and **`embedded`** count both stores (an item and a reference are
  different rows even for the same paper); the split is in the added fields
  `items_tagged`, `references_tagged`, `items_embedded`, `references_embedded`.
  **`with_fulltext`** counts references whose stored body is non-empty (a
  `none` row -- "tried, nothing to fetch" -- does not count).
- **`saved`** is the named persona's live bibliography entries (a removed entry
  does not count). `--persona` defaults to `$ATTEST_DESK_USER`, else `owner`; an
  unknown persona has zero saved and is **not** created.
- **`ingested`** is the last 26 ISO weeks ending with the current one, oldest
  first, Monday starts, zero-filled. **Items are bucketed by `items.published`**:
  the table has no ingest-time column, and `published` is the feed's own date
  when the entry carried one and the ingest moment when it did not. That is right
  for an hourly refresh and wrong only for a backfill (a first ingest of an old
  feed lands items in the weeks they were published, not the week they arrived).
  **References are bucketed by `first_seen`**, which *is* an ingest time (when the
  library first stored the row). An item dated in the future or outside the window
  is not counted in any week, so the weeks need not sum to `totals.items`.
- **`by_source`** is the top 20 by count: each feed with its item count
  (`feed`), and each contributing source with its reference count -- `bib` for a
  `.bib` file, `zotero`, else `library` (arXiv/CrossRef/Semantic Scholar
  enrichers, `research:` topics).
- **`last_refresh`** is the latest `feeds.last_fetched`, the only ingest time
  Attestation records; `null` before the first ingest.
- **`failures`** is `null` throughout. Ingest and tagging print their failure
  counts per run and **record nothing**, so there is no number to report and none
  is invented. The added `backlog` object is the honest substitute --
  `items_untagged`, `references_untagged`, `references_unembedded`: work
  outstanding right now, which is *not* a failure count (a fresh sync has a
  backlog and no failures).
- **`references`** is newest `added` (`first_seen`) first, ties by id, capped at
  `--limit`; `references_truncated` is true when more exist.

**Untrusted text.** Titles, authors and venues come from feeds, `.bib` files and
web enrichers. Every string is stripped of control and format characters
(whitespace collapsed) and length-capped (title 300, venue 200, author 120, tag
60, key/source/id 200, url 500, cut ones end in an ellipsis); authors are at most
8, tags 12, sources 5. HTML is **not** escaped -- `<script>` stays the literal text
it is, and the consumer renders text only. `url` is the one field meant to become
a link, so it must be `http` or `https` with a host, no credentials, no
whitespace and at most 500 characters, else `null` (never truncated: a cut URL
points somewhere else).

**Size.** 3,000 synthetic references measure 1.4 MB, and 5,000 well under the
3 MB the test asserts; the file is compact JSON.

There is deliberately no MCP tool for it: the dashboard is a file for a page, not
a question for an agent, and a new tool would move every documented tool count
(`CLAUDE.md`, the README, `agents.md`) for no agent benefit.
