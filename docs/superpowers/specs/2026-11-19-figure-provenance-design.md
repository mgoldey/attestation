# Figure provenance — design

**Date:** 2026-11-19
**Status:** implemented, in `src/attestation/figures.py`.
**Depends on:** the claim checker (`2026-08-12-claim-checker-design.md`) and
the LaTeX parity work (`2026-08-30-latex-and-reviewer-audits-design.md`,
Option A). Read `docs/measurement-lessons.md` before trusting a guard here.

## Problem

The claim checker links a NUMBER in prose to the run that produced it. A
figure is the most-read part of a draft and has no such link: `loss.png` can
be regenerated from different runs, or not regenerated at all after the data
moved, and the draft keeps including it. `claims.coverage` already reads
LaTeX table cells, so tables are mostly covered; figures are the unlinked
half.

## Decision: an annotation beside the figure, same style as claims

```markdown
![training loss](figs/loss.png)
<!-- figure: figs/loss.png from=results/a.json,results/b.json sha256=<64 hex> -->
```

```latex
\includegraphics[width=\linewidth]{figs/loss}
% figure: figs/loss.png from=results/a.json,results/b.json sha256=<64 hex>
```

The same reasons as the claim comment: it renders as nothing, `grep` and
`git diff` read it, and it sits beside what it describes. The annotation
NAMES the figure it is about (its first token), so it is attached by path,
not by adjacency -- a figure inside a float, a subfigure, or a paragraph away
from its comment is linked all the same.

Fields: `from=` comma-separated files that produced the figure (result
artifacts or the plotting script; no spaces in a path), and an optional
`sha256=` pinning the figure's own bytes. At least one of the two is
required: an annotation that links nothing asserts nothing.

## Verdicts

| Verdict | Meaning |
|---|---|
| `fresh` | Included, annotated, every `from=` file exists and is no newer than the figure, and `sha256` (if given) matches. |
| `stale` | A `from=` file's mtime is newer than the figure's, or `sha256` no longer matches the figure's bytes. |
| `orphan` | Included, with no annotation. Nothing links it to a run. A lint. |
| `missing` | The figure file, or a `from=` file, is absent. |

Precedence when several apply: `missing`, then `stale`, then `fresh`. A
malformed annotation (unknown field, no path, nothing linked, bad hash,
duplicate for one figure) is reported in `malformed`, as `claims.parse_file`
does, never skipped -- silent skipping is how a link disappears from review.
A malformed annotation does not attach, so its figure reads `orphan` as well.

### What `fresh` does NOT mean

`fresh` says no declared source is newer than the figure. It does not say the
figure was drawn from those files, that the plotting code is right, or that
the picture shows what the caption claims; the tool never opens an image.
mtime is also weak evidence: a `git clone` or a copy resets every mtime, so
after one, staleness-by-time can only under-report. That is why `sha256=`
exists -- it survives a copy, and it catches an edit to the figure itself.
The mtime rule mirrors `claims._is_stale` (changed-after, not a hash stored
per claim); the difference is that a figure's own mtime is the reference
point, not a date the author typed.

## Finding figures

- Markdown: `![alt](path "title")` (also `<path with spaces>`) and
  `<img src="path">`. HTML comments and code (fenced blocks, inline code)
  are masked first: they render as nothing or as text, so they assert no
  figure -- the coverage rule. Annotations are read from the code-masked but
  comment-preserving text, so an example annotation inside a code fence in a
  guide is not mistaken for a real one.
- LaTeX: `\includegraphics[opts]{path}` after `latex.strip_comments`, so a
  `%`-commented one is not found. An extensionless path resolves against
  `.pdf .png .jpg .jpeg .eps` (first existing wins; none existing is
  `missing`). `\graphicspath{{a/}{b/}}` directories are searched. A paper is a
  tree: the walk is `latex.tree` (cycle-guarded, each file once).
- Relative paths resolve against the including file's folder; for LaTeX also
  the root file's folder, where LaTeX runs.
- Skipped, counted in `skipped`, never fetched: URLs (`http(s)://`, any
  `scheme:`, `//host`) and paths built from macros (`\figdir/x`, `#1`), which
  cannot be resolved without running TeX. Nothing here touches the network.
- One finding per distinct figure file: the same file included twice, or
  reached by two routes, is reported once at its first location.

For a folder target, each `.tex` paper root (a `\documentclass` file) is a
unit with its whole `\input` tree, so a chapter file is resolved from the
paper's folder rather than judged alone; remaining Markdown and `.tex`
files are units of their own.

## Surfaces

- `attest claims --figures [path]`, an "instead:" mode like `--coverage` and
  `--citations`. Prints one line per figure and a summary. Exit 1 on `stale`,
  `missing` or a malformed annotation; 0 for orphans (a lint, like
  coverage). (Counting malformed deviates from "stale or missing" alone: a
  malformed annotation leaves its figure an orphan, and exit 0 would turn the
  typo into a silent pass. `attest claims` already fails on malformed.)
- `runs.claims_check` gains a `figures` block:
  `{total, counts, findings, malformed, truncated, skipped, note}`. `findings`
  holds only non-fresh figures, worst first, capped at
  `MAX_FIGURE_FINDINGS` (8), `malformed` at `MAX_FIGURE_MALFORMED` (4), each
  message at `MAX_MESSAGE_CHARS`; `counts` and `total` are always complete.
  The caps are measured against the 7000-char ceiling with a 50-figure draft
  of long paths in `tests/test_response_size.py`. For a draft with no figures
  the block is the empty block and every other key of the envelope is
  unchanged, including the pinned `7 claim(s): 1 contradicted, 5 supported,
  1 unsupported`. When a figure is not fresh, `; figures: 1 stale` is
  appended to `message` so a model relaying only the message cannot miss it.
- No new tool, no migration, no dependency.

## Out of scope

Tables; regenerating, opening or comparing the content of figures; recording
figure hashes in signed run records (a follow-up in `run_record/adapters.py`);
a `VerdictKind` change in `claims.py`; SKILL.md edits (the attestation-annotate
skill gets the grammar later, with a routing re-measure -- skill text edits
are not free on a 2B model).

## Measurement

Fixtures prove the rules, not the yield (the ledger fixture once overstated
yield). The one real-directory run is recorded in the commit message and the
CHANGELOG entry. Guards were mutation-checked per measurement-lessons section
4: dropping the mtime comparison, the sha256 comparison, the comment masking
and the `\input` walk each fail a named test in `tests/test_figures.py`.
