# The citation manager

How do I format, fix, de-duplicate and export my references without Zotero's
servers? Attestation keeps them in its own library store and manages them
locally: `attest bib` searches, shows, formats (APA, IEEE, Chicago
author-date, Nature, Vancouver), imports, edits, merges and exports them, and
the MCP tool `cite.format` gives an agent the same formatted text and BibTeX.
Nothing leaves the machine. Uploaded `.bib` files are only ever read; your
edits live in the store and reach the `.bib` Attestation generates for you.

## What it works on

The library store (`attest library sync`, or `cite.sync`) already holds one
row per paper from every `.bib` in `ATTEST_BIB_PATHS`, a local Zotero
database (read-only, optional), feed items with a DOI or arXiv id, and saved
research papers. See [Claims and citations](claims-and-citations.md#the-library).
The citation manager adds four things to that store (migration 014):

- the BibTeX **entry type**, **volume** and **pages**, and every other field a
  `.bib` entry carried (`publisher`, `number`, `note`, a custom field...) in
  `extra`, so an imported entry renders back with them. A `file` field is
  dropped: it is a path on the machine that exported the `.bib`;
- **notes** and an **edit log** (`reference_edits`): every field, tag and note
  change, and every merge, with the old and new value;
- **aliases**: the names a merged-away row answered to;
- a **formatted-citation cache**, so a style is rendered once per row.

`.bib` files are read and written with **bibtexparser v2** (MIT). It is the
one parser: the cite.* lookup path, the library sync and `attest bib import`
all read through it, so `year = 2022` (unbraced), `@string` macros and quoted
values parse everywhere. A truncated entry is reported and skipped; the
entries around it still load.

## Commands

```bash
attest bib search "equivariant potentials"      # semantic when embedded, keyword otherwise
attest bib show batzner2022                     # every field, sources, tags, note, BibTeX, citation
attest bib format batzner2022 vaswani2017 --style ieee
attest bib format batzner2022 --bibtex
attest bib import ~/Downloads/thesis.bib        # reads the file, never writes it
attest bib import --zotero                      # ~/Zotero/zotero.sqlite, read-only
attest bib edit vaswani2017 --set pages=5998--6008 --set publisher="Curran" \
    --note "the transformer paper" --tag attention
attest bib merge                                # list duplicate pairs; changes nothing
attest bib merge 10.5555/nequip nequip2021      # keep the first, fold the second into it
attest bib export --out refs.bib                # the whole library as BibTeX (new files only)
attest bib export batzner2022 --style apa       # formatted text to stdout
```

A key can be a citation key, a DOI, an arXiv id, a library identity, a key
from a second `.bib` that holds the same paper, the key a persona's generated
`.bib` gave it, or a key a merge folded away.

### Formatting

**citeproc-py** (BSD) renders five **CSL styles** shipped in
`src/attestation/csl_styles/` (CC BY-SA 3.0, copied unmodified from the
citation-style-language/styles repository; see its `LICENSE.md`):

| id | style |
|---|---|
| `apa` | APA 7th edition (the default) |
| `ieee` | IEEE |
| `chicago-author-date` | Chicago Manual of Style 18th edition, author-date |
| `nature` | Nature |
| `vancouver` | NLM/Vancouver citation-sequence |

`ATTEST_CITATION_STYLE` sets the default; a name outside the table is refused,
never replaced by another style. The `en-US` locale ships inside citeproc-py,
so formatting needs no network.

A reference is converted to CSL-JSON (article, inproceedings, book, preprint,
misc and the other BibTeX types; authors, year, venue, volume, pages, issue,
publisher, DOI, URL), with LaTeX folded to text (`Sch{\"u}tt` is Schütt). An
arXiv paper with no journal is a preprint: publisher arXiv, its id as the
number, the abs page as URL.

Each reference is rendered **alone**. A whole-library bibliography would
disambiguate across entries (APA's 2022a/2022b), so one paper's text would
change when an unrelated one was added, and that disambiguation is quadratic
(81 s for 5,000 APA entries, against ~20 ms each alone). A numeric style's
label (`[1]`, `1.`) is dropped for the same reason. Output is plain text and
deterministic.

### Editing

`attest bib edit KEY` changes the store row (fields with `--set`, the note with
`--note`, tags with `--tag`/`--untag`) in one transaction, logs it, and saves
the reference to the **owner's** bibliography, so
`<ATTEST_BIB_OUT>/<owner>.bib` carries the edited entry under its stable key.
The owner is `ATTEST_DESK_USER`, else `owner`, created if absent. With
`ATTEST_BIB_OUT` unset the entry is recorded and written once it is set.

- Any BibTeX field can be set; `title`, `authors` (`A and B`), `year`, `venue`,
  `doi`, `arxiv`, `url`, `abstract`, `type`, `volume` and `pages` are columns,
  the rest go to `extra`. An empty value clears a field; a title cannot be
  cleared.
- Keys and identities are not editable: a key, once given, is what every
  `\cite` uses.
- A DOI or arXiv id another row already holds is refused with "merge the two
  instead".
- A sync never overwrites an edited field (merging fills only empty fields), but
  a field you **cleared** is filled again by the next sync of a source that has
  it.

### Duplicates and merging

`attest bib merge` with no keys lists pairs that are one paper: the same DOI,
arXiv id, or normalised title and year in two rows, or an id a sync refused
because another row already held it (the store's own record of a duplicate).
The suggested row to keep has a DOI, else an arXiv id, else is the older row.

`attest bib merge KEEP DROP` (or `--all` for every listed pair) folds DROP into
KEEP in one transaction:

- KEEP fills its empty fields from DROP; nothing on KEEP is overwritten;
- DROP's sources, tags, citation edges, full text, note, edit history and each
  persona's bibliography entry move to KEEP (an entry keeps its cite key, so a
  `\cite` of it still builds);
- every name DROP answered to (identity, DOI and arXiv forms, `.bib` and
  Zotero keys, cite keys) becomes an alias of KEEP, so a lookup, a `\cite` key
  check and the next sync of the file DROP came from all land on KEEP;
- DROP is deleted from the store only. No `.bib` file is touched.

### Importing

`attest bib import FILE.bib` validates the file (exists, at most 50 MB, at
least one readable entry) and reads it into the store as `bibtex:<file name>`,
the same source name a sync of that file uses. It reports the count added,
merged and unchanged, the entries it could not read (with line numbers), the
untitled ones, keys defined twice in the file (the first is kept, as BibTeX
does), and **key clashes**: keys that already named a different paper in the
store. `--zotero [PATH]` reads a local `zotero.sqlite` the same way.

## For an agent: `cite.format`

`cite.format(keys, style)` returns, for up to 5 keys, the formatted `text`, the
`cite_key` and the `bibtex` of each, plus `missing` for keys the library does
not hold. It is served wherever `cite.*` is (the default surface and
`knowledge`). `cite.search` rows carry `arxiv` beside `doi`, so a page can link
both. Edits stay a CLI action (`attest bib edit`), which an agent runs on the
owner's word.

## In library.json

`attest library dashboard` adds, per reference, `bibtex`, `formatted`
(`{style id: text}` for the default style), a truncated `abstract`, and
`cited_by` (where the workspace's drafts cite it), plus top-level
`citation_styles` and `default_style`. See
[The library dashboard](library-dashboard.md#the-contract-schema-1) for the
shapes and the size budget. `cited_by` scans every `.tex` and `.md` under
`ATTEST_WORKSPACE`, else `<HERMES_HOME's root>/workspace` (the folder the
host's Files page shows) for `\cite`-family keys and claims' `cite=` keys.

## When it goes wrong

- **`unknown citation style`**: `ATTEST_CITATION_STYLE` or `--style` is not one
  of the five ids above.
- **`no library reference matches`**: run `attest library sync` (or `attest bib
  import`) first; `attest bib search` finds the key.
- **`... already names ...: merge the two instead`**: the DOI or arXiv id you
  set belongs to another row. `attest bib merge KEEP DROP`.
- **The edited entry is not in my `.bib`**: `ATTEST_BIB_OUT` is unset; the edit
  is recorded and the file appears once it is set (`attest library bib`).
- **`repeated` in `attest claims --citations`**: two `.bib` files the paper
  loads define the same key -- usually an uploaded file and the generated
  owner `.bib` after a save or an edit. Cite from one of them.
