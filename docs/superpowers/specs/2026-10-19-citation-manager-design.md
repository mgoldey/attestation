# The citation manager

**Date:** 2026-10-19
**Status:** built on `feat/citation-manager`; follows the Attestation/AgentMarkit
boundary note (Attestation is the engine, AgentMarkit the screens).
**Depends on:** the library store (`2026-09-05-library-store-design.md`), the
reader's bibliography (`2026-09-30-bibliography-design.md`), the library
dashboard contract (`docs/guides/library-dashboard.md`).

## What this is for

A researcher on a hosted machine has no Zotero desktop and should not need
Zotero's servers. They need what a reference manager does: format a citation
in the journal's style, copy BibTeX, fix a wrong field, merge the two records
the preprint and the publication became, and import a `.bib` someone sent.
Attestation already owns the deduplicated store; this adds the management on
top, as CLI (`attest bib`), one MCP tool (`cite.format`) and fields in
`library.json`, which AgentMarkit's bibliography view reads.

## Decisions

**Libraries.** citeproc-py 0.11.1 (BSD-2-Clause-Views; releases through
2026-09, wheels for 3.12/3.13, depends only on lxml) renders CSL;
bibtexparser 2.x (MIT; 2.1.0 released 2026-10-02, depends on pylatexenc)
reads and writes BibTeX. Both install with `uv` on 3.12 and 3.13, and
citeproc-py ships its locale files, so rendering was verified with the
network namespace removed (`unshare -n`). Five styles are vendored from
citation-style-language/styles at `0e5503b` (CC BY-SA 3.0, license file
beside them); Vancouver is the repository's `nlm-citation-sequence.csl`
since its rename (PR 7928).

**One `.bib` parser.** bibtexparser replaces the hand-rolled regex reader
everywhere (`citations._parse_bib_entries` delegates), not only in the new
code: two parsers would disagree on `year = 2022` (the regex dropped
unbraced values) and on `@string` macros, and the store would hold different
fields depending on which path read the file. `latex.bib_keys` keeps its own
key-only regex: it must count untitled entries, as BibTeX does, and is fast.

**Each reference is rendered alone.** Measured on 5,000 references: a joint
APA bibliography took 81 s (quadratic year-suffix disambiguation) against
~20 ms per reference alone, and a joint rendering would make one paper's
text depend on which others are in the library. Numeric labels are stripped
for the same reason. Renderings are cached per (row, style) under a digest
of the CSL-JSON and the style file.

**library.json precomputes one style.** All five cost ~1 KB more per
reference (10.7 MiB at 5,000, over the consumer's 8 MB); the default style and
BibTeX are in every row, the rest is one CLI or MCP call away. The build
stays read-only and only reads the cache; `attest library dashboard` warms
it first. Over 6 MiB, the oldest rows shed abstract, then citation and
BibTeX, before any row is dropped.

**Edits change the store, never an upload.** An edit updates the row, is
logged in `reference_edits`, and saves the reference to the owner's
bibliography, so the generated `<owner>.bib` carries it under its stable
key. Sync's fill-only merge keeps an edited value; a field the owner
cleared can be refilled by a source that has it (stated in the guide).
Keys and identities are not editable.

**Merging deletes from the store and keeps the names.** The dropped row's
children move to the kept row; its names become `reference_aliases`, which
`library._row_by_ids` consults after every column, so a `\cite` key, a lookup
and the next sync of the file it came from all land on the kept row. Keeping
a tombstone row instead would have meant excluding it from search, export,
the dashboard and the graph separately.

**Duplicates come from the store's own evidence.** Same DOI, arXiv id, or
title and year in two rows, plus every id a sync refused because another row
held it (the `identity`/`doi`/`arxiv_id` conflicts upsert already records).

**`cited_by` reads the workspace.** `ATTEST_WORKSPACE`, else
`<HERMES_HOME's root>/workspace` (the same root rule as the dashboard's
default path, now `paths.hermes_root`), scanned for `\cite`-family keys
(`latex.cited_keys`) and claims' `cite=` (`claims.parse_file`), bounded at
2,000 files of 2 MB, paths relative to the root.

**`repeated_keys`** is ported from AgentMarkit's latex-paper check: a key
two LOADED `.bib` files define. It is reported, not counted as unresolved,
because BibTeX builds with the first definition.
