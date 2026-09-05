# Verifiable claims and citations

Can it check my draft? Yes — a claim is an HTML comment beside the prose it
describes, checked against the run ledger for one of five verdicts, and a
citation key is linted against your configured bibliography the same way.

## Verifiable claims

A README says "MAE 0.353 eV vs experiment". That number was transcribed by hand
and nothing checks it: re-run the benchmark and the document asserts 0.353
forever. A claim is an HTML comment beside the prose it describes, so the
document renders exactly as before:

```markdown
The cut leaves WER essentially unchanged (**0.053 vs 0.043** baseline).
<!-- claim: ablation/stack_4 metric=wer value=0.053 tol=0.001 -->
```

```bash
uv run attest claims ~/projects              # verify every claim
uv run attest claims ~/projects --coverage   # numbers no claim covers
```

Five verdicts, and the distinctions are the design. `supported`: a run agrees.
`contradicted`: a run disagrees — the document or the run is wrong.
`unsupported`: no run matches, so the claim may be true but nothing backs it.
`ambiguous`: a wildcard matched several runs, so which is meant is undecidable.
`stale`: the value matches but the artifact changed after `as_of`.

`unsupported` and `contradicted` never collapse together — one needs a run, the
other needs a correction. `ambiguous` exists because silently taking the first
of several matches is how a checker reports a confident wrong answer.
`attest claims` exits non-zero on a contradiction, so it can gate a commit.

A claim can also carry `cite=<key>`, and that key is linted too — `uncited`
when no configured `.bib` or Zotero source has it — from the CLI (`attest
claims`) as well as the `cite.check` / `runs.claims_check` MCP tools. It is a
lint ("no source has this key"), never "the paper does not support this
claim". See `examples/citations/` for a worked run of both linters together.

`--coverage` is the inverse, and the more useful half for adoption: a document
with zero contradicted claims can still assert a dozen unverifiable numbers.
Only decimals count as measurements — on a real index, 212 numbers reduce to 30
decimals and the decimals are the results.

## The library

One deduplicated store of references, filled by `attest library sync` (or the
`cite.sync` tool) from every configured source: the `.bib` files named in
`ATTEST_BIB_PATHS` (else any `*.bib` in the working directory), a Zotero
library at `ATTEST_ZOTERO_PATH` (else Zotero's default), and the feed's own
items that carry a DOI or arXiv id. The same paper from three sources is one
row with three source entries, each recording what that source offered and
when. Identity is a pure rule: DOI, else versionless arXiv id, else the
normalised title and year. Merging fills empty fields and keeps the first value
of a disagreement, recording the conflict on the source that offered it, so
`cite.lookup` can show a disagreement rather than lose it.

Two flags reach off the machine, both off by default and read only when the
readers are built: `ATTEST_CITATION_WEB` (arXiv and CrossRef: abstracts,
authors, venues) and `ATTEST_CITATION_SCHOLAR` (Semantic Scholar reference lists,
one request per second, cached forever). They fill fields on references the
library already holds and never add a paper on their own; `cite.sources` says
which are live.

```bash
uv run attest library sync                  # read every source; embed what it can
uv run attest library search "equivariant force fields"
uv run attest library search "" --author batzner --year 2022
uv run attest library tag --limit 50        # a model call per reference, ~2.3s each
uv run attest library status
```

Search is semantic when the library is embedded (the model server was
reachable during `sync` or `library embed`) and substring otherwise, and the
output says which — the `cite.search` tool carries the same `semantic` flag
and `caveat`. `cite.check` and `attest claims` resolve `cite=<key>` through
the store first, so a key synced from Zotero resolves even when no `.bib` sits
in the working directory. See `docs/superpowers/specs/2026-09-05-library-store-design.md`.

References join the concept graph through their tags — from `attest library
tag`, or from a `.bib` `keywords` field — and `attest library related KEY`
(the `cite.related` tool) walks a paper's citation edges both ways, from
Semantic Scholar reference lists or a `.bib` `cites` field
(`identity|title; identity|title`); a cited paper not in the library is
listed, never fetched. `examples/molecular-ai/` shows both from a committed
`.bib` that real software generated from real papers. See
`docs/superpowers/specs/2026-09-05-library-graph-and-molecular-ai-design.md`.
