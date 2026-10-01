# The reader's bibliography

**Date:** 2026-09-30
**Status:** approved in conversation (option 1, engagement + explicit saves),
built on `feat/bibliography`, held as a draft PR until the AgentMarkit
rehearsal shows an agent saving a paper and reading an uploaded `.bib`.
**Depends on:** the library store (`2026-09-05-library-store-design.md`),
implicit feedback (`implicit.py`), the Reading desk's last-verdict-wins rule
(`2026-09-29-reading-desk-design.md`).

## What this is for

A researcher's agent reads the feed with them, and what they engage with
evaporates: the library knows a paper exists, nothing knows the reader cared.
Two asks, one file:

1. **The feed lands in a `.bib`.** Papers the reader engaged with -- rated
   useful (desk, UI, or a verdict the agent extracted), read in full through
   `feed.read`, or asked about through `feed.explain` -- and papers they
   explicitly asked to keep ("save this", "I'll cite this") become entries in
   one generated BibTeX file per persona, ready for `\cite{}`.
2. **Their own `.bib` uploads are read.** On a hosted AgentMarkit machine the
   reader has no Zotero desktop; a `.bib` exported from Zotero, Mendeley,
   EndNote or Paperpile, dropped into a folder, is the integration.

Zotero's web API was considered and deferred: an upload covers every
reference manager, needs no credential, and the bibliography tools have no
measured use yet. It returns when a customer needs live sync (see the
conversation record of 2026-09-30).

## Decisions

**One table, rendered.** `bibliography(user_id, reference_id, cite_key,
reason, added_at, removed_at)` -- migration 013. The `.bib` is a pure render
of a persona's live rows. Rejected: deriving the file from `clicks` +
`engagement` + a saves table at render time, which spreads the rule across
three tables, cannot answer "why is this in my bib" without reconstruction,
and cannot express a removal without inventing a negative signal.

**`reason` is provenance**, one of `useful`, `read`, `explained`, `saved`.
It is what the agent answers "why is this here?" with. Among implicit
reasons the first recorded stands (fold visits `useful`, then `explained`,
then `read`, so the strongest wins on a first fold); an explicit save always
overwrites an implicit reason with `saved`, because the reader said so.

**Removal is a tombstone**, `removed_at` set, never a delete. An implicit
signal never revives a tombstoned entry (re-reading a paper you removed does
not put it back); an explicit save does. A later human "not useful" verdict
tombstones an entry whose reason is implicit, and stops engagement adding the
paper; it never touches a `saved` entry. This is the desk's
last-explicit-verdict-wins rule, and like the rest of the repo it infers no
negative from silence.

**Only human signal adds.** Clicks with `source` in `HUMAN_CLICK_SOURCES`
(`ui`, `agent`); `bootstrap`, `simulated` and `implicit` clicks never reach a
reader's bibliography -- a chat model reacting as the persona is not the
reader choosing what to cite.

**Engagement is folded at render, not hooked at write.** `fold(conn,
user_id)` is one idempotent pass (INSERT OR IGNORE) over human clicks and
`engagement`, run before every render. The four write sites (record_click,
two engagement inserts, desk import) are untouched, so the hot paths keep
their single responsibility and a failure here can never cost a click.

**Automatic adds are papers.** Engagement adds only items with a DOI or an
arXiv id -- the same rule `FeedRecords` uses, so an HN thread the reader
opened does not become `@misc`. An explicit save takes any item, or any
library reference by key, DOI or arXiv id.

**A cite key never changes.** `export_bib` suffixes a repeated key by row
order, so removing one entry would rename another and break every `\cite`
using it. The key is assigned once, at add time (`bibtex_key` of the row, or
the first free `b`, `c`, ... suffix for that persona) and stored;
`UNIQUE(user_id, cite_key)`. A tombstoned entry keeps its key, so a removed
and re-saved paper comes back under the same one.

**Where the file goes.** `ATTEST_BIB_OUT` names a directory; each persona
with live entries is written to `<dir>/<persona>.bib`, atomically, with a
header saying it is generated and edits are overwritten. Unset = inert: the
table still records, nothing is written -- the desk's convention.

**Reading uploads.** `ATTEST_BIB_PATHS` entries may now be directories; a
directory contributes every `*.bib` directly inside it, non-recursively. The
hosted layout puts uploads in `~/.hermes/workspace/bibliography/` and the
generated file in its `attestation/` subfolder, so the reader never reads
attestation's own output back as user input, and a newly uploaded file is
picked up without editing `.env` or reloading the MCP server.

## Surface

- **`cite.save(paper, user, remove=False)`** -- the one new tool, on the
  knowledge surface (cite.* is 7, the default server 50). `paper` is a feed
  item id or a library key/DOI/arXiv id. Folds engagement, saves or
  tombstones, re-renders the file when `ATTEST_BIB_OUT` is set, and returns
  the entry (`key`, `title`, `reason`, `bibtex`), the live entry count and
  the file NAME (never a path: a machine string in every reply). An unknown
  persona refuses (a write never autocreates); an unknown paper refuses.
- **`attest library bib [--user NAME]`** -- fold and write every persona's file (or
  one), printing counts; inert when `ATTEST_BIB_OUT` is unset. The hourly
  refresh script runs it after the desk, degraded and never fatal.
- **Persona lifecycle.** `persona_delete` (purge with delete_user) deletes the
  persona's entries, since users.id is a reused rowid. `persona_reset` keeps
  them: reset is about the ranker, the bibliography is the reader's. `merge`
  moves entries to the keeper where neither the reference nor the key is
  already taken there.

## Skills

The knowledge skill learns "save this / add it to my bib / I'll cite this"
-> `cite.save`, "take it out" -> `cite.save(remove=True)`, and "why is this
in my bib" -> the `reason`. The feed skill (which cannot name `cite.save`,
not on its surface) says that rating, reading and asking about a paper add it
to the reader's bibliography. AgentMarkit's SOUL.md names
`mcp__attestation_knowledge__cite_save` and the rehearsal asks one save
question and one uploaded-`.bib` question.

## Testing

DB-level tests for fold (human-only, papers-only, tombstones respected, a
not-useful verdict tombstoning implicit entries but never saved ones), key
stability across removal, render determinism, the directory form of
`ATTEST_BIB_PATHS`, persona delete/merge, and the tool envelope including
both refusals. The live check is the AgentMarkit rehearsal with
`ATTESTATION_SRC` pointing at this branch -- that, not this suite, decides
when the PR leaves draft.
