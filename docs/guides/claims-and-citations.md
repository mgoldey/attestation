# Verifiable claims and citations

Can it check my draft? Yes, in Markdown or LaTeX — a claim is a comment beside
the prose it describes, checked against the run ledger for one of five verdicts, and a
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

## Does the cited paper say it?

`attest support draft.md` (MCP: `cite.support`) is the check the lint is not.
For every citing sentence -- Pandoc `[@key]`, a claim's `cite=<key>`, or LaTeX
`\cite{key}` -- it finds the paper in the library, reads the best text held
for it (full text from `attest library fulltext`, else the abstract, else the
title; reported as `text_level`), ranks the source's sentences against the
citing one, and returns `supported`, `partially_supported`, `contradicted`,
`not_found` or `source_unavailable` with a verbatim quote and a reason.

The judge is the local chat model (`--judge auto`, the default), falling back
to a deterministic check when none answers (`--judge none` forces it). Either
way, numbers are checked by code: a sentence that keeps the paper's number but
attaches it to something else -- "outperforms invariant models by up to three
orders of magnitude" against "up to three orders of magnitude fewer training
data" -- is `contradicted` even when the model says it is supported. Every
quote is checked to be an exact substring of the source. `attest support`
exits 1 on any `contradicted` (`--fail-on` widens the gate), like `attest
claims`; `--sentence ... --cite KEY` checks one sentence.

`--coverage` is the inverse, and the more useful half for adoption: a document
with zero contradicted claims can still assert a dozen unverifiable numbers.
Only decimals count as measurements — on a real index, 212 numbers reduce to 30
decimals and the decimals are the results.

## LaTeX papers

A `.tex` source is a first-class document. A claim is a `%` comment with the
same body, on its own line or trailing the table row it backs; it compiles to
nothing and needs no package:

```latex
The best student reaches a WER of \num{0.0642}.
% claim: speech/kdsweep_t4 metric=wer value=0.0642
$T=2$ & 0.0701 & 0.023 \\ % claim: speech/kdsweep_t2 metric=wer value=0.0701
```

```bash
uv run attest claims paper/main.tex              # the five verdicts, exit 1 on a contradiction
uv run attest claims paper/main.tex --coverage   # numbers no claim covers
uv run attest claims paper/main.tex --citations  # \cite keys the bibliography lacks
```

Pass the root file: `\input`, `\include` and `\subfile` are followed (a
cycle is read once), the whole tree is one document, and every finding names
its own file. `\%` is a percent sign, never a comment.

Coverage reads prose and table cells, `\num{}`, `\SI{}{}` and `\qty{}{}`, a
number alone in inline math (`$0.021$`, `$12.2\%$`), and `1{,}234.5` as
1234.5. It skips comments, equations and display math, math with a symbol in
it (`$\alpha = 0.5$`), `\label`/`\ref`/`\cite` arguments, package options,
and lengths (`0.48\textwidth`, `2pt`).

`cite.check` (and `attest claims --citations`) reads every `*cite*` command —
`\cite`, `\citep[see][p.~3]{a,b}`, `\citet`, `\citealp`, `\parencite`,
`\textcite`, `\autocite`, `\footcite`, `\cites{a}{b}` — skipping natbib's
`\setcitestyle`/`\citestyle`. The candidate `.bib` files are those beside
the paper, `ATTEST_BIB_PATHS`, and the owner bibliography in `ATTEST_BIB_OUT`.
Once the paper names its files (`\bibliography{refs}`, `\addbibresource{refs.bib}`)
a key resolves only in those, as BibTeX would read it; a key some other file
or the library defines is reported in `missing_keys` with
`defined_in_unloaded`, and a named file that does not exist in
`unresolved_bib_resources`. With no named file, any candidate, the library
store or Zotero resolves a key. A key two LOADED files both define is
reported in `repeated_keys` (`{key, files, paper}`; `attest claims --citations`
prints it as `repeated` without failing, since BibTeX builds with the first) --
the usual cause is saving or editing a paper from an uploaded `.bib`, which
puts the same key in the generated owner `.bib`. These rules match AgentMarkit's `latex-paper`
skill. `tests/fixtures/latex/` is a three-file paper exercising all of it.

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

Three flags reach off the machine. `ATTEST_CITATION_WEB` (arXiv and CrossRef:
abstracts, authors, venues) and `ATTEST_CITATION_SCHOLAR` (Semantic Scholar
reference lists, one request every three seconds, cached forever) are both
off by default and read only when the readers are built; they fill fields on
references the library already holds and never add a paper on their own.
`ATTEST_RESEARCH_WEB` is the opposite default -- on unless set to `0` -- and
governs `feed.research`, `research:` topic feeds, and `attest library
fulltext`: the one path that introduces a reference from the wire rather
than filling one that is already there. `cite.sources` says which are live.

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

`attest library fulltext` pulls bodies (arXiv PDF, or PMC open-access XML by
PMCID) for references that have none, ten per run by default and also at the
end of `attest ingest`; `cite.lookup` serves them in 2000-character windows,
never whole, and never embeds them. `attest library export --bib refs.bib
[--author A] [--year Y] [--tag T] [--source S]` writes a filtered set as
BibTeX rendered from the rows -- `@article` for a journal, `@misc` with
`eprint`/`archivePrefix` for a preprint, the `.bib` key when one supplied it,
else `<family><year><word>` -- refusing to overwrite without `--force`.
`cite.lookup` returns the same `bibtex` for one record.

Formatting (APA, IEEE, Chicago, Nature, Vancouver), owner edits, notes, tags,
duplicate merging and `.bib` import are the citation manager's: `attest bib`
and `cite.format`, in [The citation manager](citation-manager.md).

References join the concept graph through their tags — from `attest library
tag`, or from a `.bib` `keywords` field — and `attest library related KEY`
(the `cite.related` tool) walks a paper's citation edges both ways, from
Semantic Scholar reference lists or a `.bib` `cites` field
(`identity|title; identity|title`); a cited paper not in the library is
listed, never fetched. `examples/molecular-ai/` shows both from a committed
`.bib` that real software generated from real papers. See
`docs/superpowers/specs/2026-09-05-library-graph-and-molecular-ai-design.md`.
