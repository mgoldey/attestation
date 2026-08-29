# Example citations

## What you get

A four-entry BibTeX library (`references.bib`) written by real BibTeX
software rather than typed to look like one, and a draft (`DRAFT.md`) whose
claims cite three of those four keys plus a fourth, `doe2099imaginary`, that
resolves nowhere on purpose. The numeric claims all check out against the
`speech-distill` runs in `examples/workspace/` -- what varies here is the
citation lint, not the numbers.

## Prerequisites

`none — pure local computation`

## Run it

```bash
export RESEARCH_ROOT=$PWD/../workspace
uv run attest runs scan --root ../workspace
uv run attest claims DRAFT.md || true
uv run python check_citations.py
```

Relative to this directory (`run.sh` does `cd "$(dirname "$0")"` first). The
`.bib` file is read from the **current directory** --
`citations.Resolver.from_env()` globs `*.bib` in `Path.cwd()`, so `cite.*`
tools only see `references.bib` when the server's cwd is this directory,
which is why `check_citations.py` spawns `attest-mcp` with this directory as
`cwd` (inherited from the shell, same as `run.sh`).

## What it prints

```
  retrieval-ablation           5 run(s)
  speech-distill               4 run(s)
9 run(s) across 2 project(s)
```

Abridged -- `attest claims DRAFT.md` prints one line per claim, all four
`supported` (the citation lint does not run here; see *What it demonstrates*),
then `check_citations.py` prints `cite.sources`, `cite.check`, one successful
and one refused `cite.lookup`, and a `cite.search` hit count, ending:

```
cite.check -> 4 claim(s) scanned for CITATION KEYS only -- 1 claim(s) cite a key no source can resolve. This says nothing about whether the numbers are right: call runs.claims_check for that.
  uncited key='doe2099imaginary' at .../DRAFT.md:28
```

## What it demonstrates

**The lint is "no source has this key", never "the paper does not support
this."** `claims.check_citations()` (`src/attestation/claims.py`) looks up
each `cite=` key in a `Resolver` and reports `VerdictKind.UNCITED` when
nothing answers. It never reads the cited work's content and makes no claim
about whether `hinton2015distilling` actually supports the sentence beside
it -- that would need a model, and every verdict this module produces is
deterministic. `doe2099imaginary` fails the same way a typo would: the lint
cannot distinguish "this paper does not exist" from "you misspelled the key."

**`attest claims` has no citation command -- a finding, not a defect to fix
here.** `cmd_claims` in `src/attestation/cli.py` calls
`claims.check(conn, target)` with no `resolver=` argument, so the CLI's own
codepath through `check_citations()` never runs; all four claims above print
`supported` and the process exits 0, key `doe2099imaginary` and all. The MCP
tool with the same job, `runs.claims_check`
(`src/attestation/mcp/claims_tools.py`), builds a resolver with
`citations.Resolver.from_env()` and passes it in, so calling that tool (or
`cite.check` directly) over MCP is the only way this repository currently
surfaces an `uncited` verdict. `check_citations.py` drives `cite.sources`,
`cite.check`, `cite.lookup` and `cite.search` over stdio -- the pattern in
`examples/flows/mcp_e2e.py`'s `run_surface`, spawning `attest-mcp` with
`ATTEST_TOOLS=knowledge ATTEST_EXPAND=1` -- because that is the only path in
this repository that actually reports the lint.

**`cite.sources` reports `offline: true`.** With no Zotero library on this
machine and `ATTEST_CITATION_WEB` unset, the only configured reader is
`bibtex`, and `network` is `false` for it -- `offline` is `not any(network)`
across configured readers, so it answers "does this leave the machine" from
the same surface that would have done the leaving.

**`ATTEST_CITATION_WEB` is read at resolver construction, never at call
time.** `citations.Resolver.from_env()` checks the environment variable once,
when it builds the reader list; a `WebReader` that was never constructed
cannot be coaxed into a request later by an unusual code path. This path
never sets the variable, so `cite.sources` above never lists a `web` reader
and no lookup here can reach CrossRef -- run
`ATTEST_CITATION_WEB=1 uv run attest-mcp` (outside this script) to see a
`web` entry with `network: true` appear.

**`references.bib` is what real BibTeX software writes, not a hand-typed
file.** `generate.py` builds the four entries as bibtexparser v2
`Entry`/`Field` records and writes them with `bibtexparser.write_string()`;
the tab-indented, brace-quoted layout in the committed file is the library's
writer output. JabRef's save and Zotero's Better BibTeX export produce files
in this same shape -- any of the three would satisfy `BibtexReader`
(`src/attestation/citations.py`), which parses `@type{key, field = {value},}`
regardless of which tool wrote it.

**Keys are the contract between a draft and a library.** Nothing here checks
that `vaswani2017attention` is spelled the way *some other* draft would
spell it -- the key is whatever the `.bib` entry says it is, and a claim's
`cite=` field has to match it exactly. `attestation.citations.ZoteroReader`
reads `~/Zotero/zotero.sqlite` directly when present, using the same key
(Zotero's item key, not a title) -- documented in `citations.py` and covered
by shape-tolerance tests, but not staged here: a fake Zotero database would
be the author writing its own fixture and calling that a demonstration. If
you have a real Zotero library, point `ATTEST_CITATION_WEB` questions aside
and just run `attest-mcp` from a directory Zotero can also reach -- the
resolver picks it up automatically (`DEFAULT_ZOTERO = Path.home() / "Zotero"
/ "zotero.sqlite"`), no flag needed.

## When it goes wrong

- `attest claims DRAFT.md` exits 0 here (all four numeric claims resolve),
  unlike `examples/workspace/`'s deliberately-contradicted claim -- `run.sh`
  still carries `|| true` on that line because a claims check is not
  guaranteed to stay clean as this fixture changes, and the golden-paths
  convention is that a claims-check line always tolerates exit 1.
- Running `generate.py` with an unpinned `bibtexparser` gets the 1.x API
  (the latest stable release), which has no `Library` to import;
  `build_library()` raises `ImportError` rather than writing a file that
  looks right under the wrong library version. Use
  `uv run --with "bibtexparser>=2.0.0b9" --no-project python generate.py`.
- `check_citations.py` must run with this directory as the working
  directory (`run.sh` already `cd`s here) -- `Resolver.from_env()` globs
  `*.bib` in `Path.cwd()`, so from any other directory `cite.sources` lists
  no `bibtex` reader and every lookup here fails to resolve, including the
  three keys that should succeed.
- An `ATTEST_DB` left over from a previous run accumulates duplicate runs on
  re-scan; `run.sh` always points `ATTEST_DB` at a fresh temp file.

## Next

Regenerate the library from real BibTeX software (a diff after editing the
entries in `generate.py` is expected and should be reviewed like any other
fixture change):

```bash
uv run --with "bibtexparser>=2.0.0b9" --no-project python generate.py
```

See the catalogue at `examples/README.md` for the other golden paths, and
`src/attestation/citations.py`'s module docstring for the offline guarantee
and its one documented exception.
