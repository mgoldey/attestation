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
attest library dashboard --style ieee            # precompute IEEE instead of the default
```

The path is `--out`, else `$ATTEST_LIBRARY_DASHBOARD`, else
`<parent of HERMES_HOME>/.hermes/workspace/research-desk/library/library.json`
(the folder is created). The default follows the host installer's convention, `<parent of HERMES_HOME>/.hermes/workspace/...`, without resolving symlinks, so a `HERMES_HOME` that is not named `.hermes` (say `/opt/hermes-data`) writes under `/opt/.hermes/` -- which the host page will not read unless that is its home. **When in doubt, set `ATTEST_LIBRARY_DASHBOARD` to the exact path the page reads**; it always wins. For a Hermes profile (`HERMES_HOME=<root>/profiles/<name>`) it is the *root's* workspace, where the host page reads. The write is atomic -- a temp file in the same folder,
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
                  "saved": false, "fulltext": false,
                  "bibtex": "@article{batzner2022,\n  author = {...},\n  ...\n}\n",
                  "formatted": {"apa": "Batzner, S., ... (2022). E(3)-equivariant ..."},
                  "abstract": "This work presents ...",
                  "cited_by": [{"path": "papers/draft/main.tex", "line": 42}]}],
  "references_truncated": false,
  "references_limit": 5000,
  "references_trimmed": 0,
  "citation_styles": [{"id": "apa", "name": "APA 7th edition"}, {"id": "ieee", "name": "IEEE"},
                      {"id": "chicago-author-date", "name": "Chicago (author-date, 18th edition)"},
                      {"id": "nature", "name": "Nature"}, {"id": "vancouver", "name": "Vancouver (NLM)"}],
  "default_style": "apa"
}
```

What each number means -- decided in `src/attestation/dashboard.py`, stated here:

- **`items`** is feed items ingested (the raw material of the ranked reading
  list); **`references`** is rows in the library store (DOI- or arXiv-identified
  references from `.bib` files, Zotero, saved papers and feed items that carry
  an id -- see [Claims and citations](claims-and-citations.md#the-library)).
- **`tagged`** and **`embedded`** count both stores (an item and a reference are
  different rows even for the same paper, so `tagged` can exceed `items`: divide
  `items_tagged` by `items` and `references_tagged` by `references`); the split is in the added fields
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

### The citation manager's fields (added 2026-10-19)

All additive; see [The citation manager](citation-manager.md).

- **`bibtex`** (string or `null`): the reference as one BibTeX entry, written
  by the same writer as the owner's `.bib`. Its key is the persona's stable cite
  key when the paper is in their bibliography, else `bib_key` or the generated
  `<family><year><word>`. Newlines are kept; other control characters become
  spaces. An entry over 1,500 characters (or 3,000 UTF-8 bytes) is `null`, never
  cut -- a truncated entry is not BibTeX.
- **`formatted`** (`{style id: text}`): the formatted citation in
  **`default_style` only**, from the cache `attest library dashboard` warms
  before it builds (`{}` for a row the cache does not hold, e.g. from a bare
  `dashboard.build`). Capped at 600 characters. Plain text.
- **`abstract`** (string or `null`): the stored abstract, cleaned like every
  string and cut at 300 characters with an ellipsis.
- **`cited_by`** (`[{path, line}]`): where the drafts in the workspace cite this
  reference -- every `\cite`-family key in a `.tex` (comments ignored) and every
  claim's `cite=` key in a `.tex` or `.md`. The workspace is `ATTEST_WORKSPACE`,
  else `<parent of HERMES_HOME>/.hermes/workspace` (the root's for a profile,
  the same rule as this file's default path): the folder the host's Files page
  shows. Paths are relative to it; hidden folders and `node_modules` are
  skipped; at most 2,000 files of at most 2 MB are read; at most 10 places per
  reference, sorted. A key matches a reference through its `.bib`/Zotero keys,
  any persona's cite key, a merged-away key, or its identity, ignoring case.
- **`citation_styles`** (`[{id, name}]`): the five bundled styles, in the order
  a style switcher lists them; **`default_style`**: the id `formatted` holds
  (`--style`, else `ATTEST_CITATION_STYLE`, else `apa`).
- **`references_trimmed`** (int): how many of the listed references lost
  detail to the size budget (below); 0 normally.

**Why one style.** Precomputing all five costs about 1 KB more per reference
(measured: ~980 bytes for the four other styles), which would put 5,000
references at 10.7 MiB -- over the consumer's 8 MB. One style is what a page
shows by default; the others are one call away (`attest bib format --style`,
`cite.format`), and BibTeX is in every row because "copy BibTeX" is the one
action every reader needs. Rendering is cached per row and style
(`reference_formatted`): a first 5,000-reference APA pass took 112 s on the
measurement machine, a second 0.5 s, and an edit re-renders only its row.

**Untrusted text.** Titles, authors and venues come from feeds, `.bib` files and
web enrichers. Every string is stripped of control and format characters
(whitespace collapsed) and length-capped (title 300, venue 200, author 120, tag
60, key/source/id 200, url 500, cut ones end in an ellipsis); authors are at most
8, tags 12, sources 5. HTML is **not** escaped -- `<script>` stays the literal text
it is, and the consumer renders text only. `url` is the one field meant to become
a link, so it must be `http` or `https` with a host, no credentials, no
whitespace and at most 500 characters, else `null` (never truncated: a cut URL
points somewhere else).

**Size.** The file is compact JSON and is **guaranteed at most 6 MiB** (the
consumer refuses more than 8 MB). Every string is capped in UTF-8 *bytes* as well
as characters (twice the character cap), and `--limit` is clamped to 5,000. If
the rendered file is still over 6 MiB, the **oldest** references first lose their
`abstract` (`null`), then their `formatted` (`{}`) and `bibtex` (`null`), until it
fits; `references_trimmed` counts them. Only if every row stripped that far still
does not fit are the oldest references dropped: `references_truncated` is set,
`references_limit` becomes the number kept, and the kept rows get their detail
back, newest first, as far as the budget allows. `totals` always stay the
database's. The caps are title 300, venue 200, author 120, tag 60, source 120,
arXiv id 40, DOI and key 200, URL 500, persona 40, abstract 300, formatted 600,
BibTeX 1,500.

Measured 2026-10-19 on synthetic references shaped like a chemistry/ML library
(titles ~110 characters, 2-9 authors, a long abstract, a DOI or an
arXiv id or both, three tags, two sources, volume, pages and a publisher):
1,000 references are 1.6 MiB and 3,000 are 4.7 MiB with every field; from about
3,900 the oldest start losing detail. 5,000 would be 7.8 MiB in full, so the
file sheds every abstract and the citation and BibTeX of the oldest 528 rows and
lands at 6,290,932 bytes (6.00 MiB) with all 5,000 references listed. (Before the citation fields, 3,000
realistic references measured 1.4 MB.)

There is deliberately no MCP tool for it: the dashboard is a file for a page, not
a question for an agent, and a new tool would move every documented tool count
(`CLAUDE.md`, the README, `agents.md`) for no agent benefit.
