# LaTeX integration and reviewer-persona audits

**Date:** 2026-08-30
**Status:** brainstorm — options with a recommendation, awaiting Matt's
review before any spec is finalised or a plan written. Nothing here is built.
**Depends on:** the claim checker (`2026-08-12-claim-checker-design.md`),
citations (`2026-08-22-citations-domain-design.md`), the reaction/explanation
prompt-as-data pattern (`2026-08-23-dspy-prompt-optimization-design.md`), and
the bundled-skills split (`docs/bundled-skills-research.md`).

## What this is for

Two asks, one direction. Today the manuscript-side tools read **Markdown**:
claims are `<!-- claim: ... -->` comments, `claims.coverage` counts decimals
in prose, `cite.check` lints `cite=<key>` fields. Papers are written in
**LaTeX**, and the numbers that matter live in `tabular` environments,
`\num{}` calls and `\newcommand` macros, cited with `\cite{key}` against a
`.bib` the citation resolver can already read. So the first ask is parity:
make `.tex` a first-class document. The second ask builds on it: a **paper
audit** that reads a draft the way a reviewer would — several reviewers,
with different concerns — and reports findings the author can act on.

The constraint that shapes both: this project's tools return structure and
caveats, never invented content, and every model output is data with a
corpus and a model-free scorer behind it. A reviewer persona is a model
speaking; the design has to say what part of its output is *checkable* and
what part is an opinion labelled as one.

Assumptions made in Matt's absence, to be confirmed:

- The target is a single author's `.tex` tree on disk (a `main.tex` with
  `\input`s and a `.bib`), not an Overleaf project reached over the network.
- Results tables are the primary carrier of numbers; prose numbers second.
- The reviewer audit is for the author's own draft before submission, not
  for reviewing others' papers.

---

## Part 1: LaTeX integration

### Where the numbers live in a `.tex` source

| Form | Example | Today | Note |
|---|---|---|---|
| Prose decimal | `WER fell to 0.053` | counted by coverage in `.md` only | same regex works |
| Table cell | `& 0.0642 & 0.0701 \\` | invisible | most results live here |
| siunitx | `\num{0.0642}`, `\SI{12.2}{\percent}` | invisible | argument is the number |
| Macro | `\newcommand{\bestwer}{0.0642}` then `\bestwer` | invisible | the number is defined once, used many times |
| Math | `$\alpha = 0.5$`, equation environments | would be counted | usually *not* a measurement |
| Structural | `\label{tab:1}`, `\ref`, `\cite{x2019}`, `[width=0.5\textwidth]` | would be counted | never a measurement |
| Included files | `\input{tables/results}` | invisible | a paper is a tree |

### Option A — parser parity (claims in `%` comments)

A `.tex` branch in `claims.parse_file` and `claims.coverage`:

- claim annotations are LaTeX comment lines beside the prose or the table
  row they describe — `% claim: proj/kdsweep_t4 metric=wer value=0.0642
  tol=0.001` — same grammar as the Markdown form, so `parse_file` shares one
  field parser and only the comment syntax differs;
- `_masked_prose` gains a LaTeX masker: strip `%` comments, math (`$..$`,
  `\[..\]`, `equation*`/`align*` bodies), `\label/\ref/\cite/\eqref`
  arguments, `\includegraphics[...]` options and lengths (`0.5\textwidth`,
  `2pt`), then unwrap `\num{x}` and `\SI{x}{u}` to their number so table
  cells and siunitx calls count as measurements;
- `\input{}`/`\include{}` are followed from the root file with a cycle
  guard, so `attest claims main.tex` covers the whole paper and each
  finding names the file it came from;
- `cite.check` learns `\cite{a,b}` (and `\citep/\citet/\parencite`): every
  key must resolve against the `.bib` beside the document or the
  configured sources. This is a lint LaTeX authors already want and
  BibTeX reports only at compile time.

Cost: small — a masker, a comment prefix, an include walker, ~4 tests per
form in the table above. It keeps the "annotate by hand" model: every
number still has to be claimed to be checked.

### Option B — derive, don't transcribe (macros and tables from the ledger)

The ledger already holds every final value. Instead of annotating numbers
the author typed, emit the numbers so there is nothing to transcribe:

```
attest tex macros  --family kdsweep --metric wer   > results/numbers.tex
attest tex table   --family kdsweep --metric wer   > results/kdsweep.tex
```

`numbers.tex` is a file of `\newcommand{\kdsweepTfourWer}{0.0642}` lines
(names derived from run and metric, deterministic, documented in the
file's header comment); the paper `\input`s it and writes `\kdsweepTfourWer`.
`kdsweep.tex` is a `tabular` with one row per arm, the winner marked, and
the comparison's caveats rendered as a `% caveat:` comment block above the
table (never as visible text the author did not choose) — the same caveats
`runs.compare` returns, so "each arm is a single run" travels with the
table into the source.

With this in place, `coverage` on a `.tex` becomes sharper: a literal
decimal in a results table or in prose that is **not** a macro reference is
the finding ("a number you typed rather than derived"), and the `% claim:`
path from Option A is the fallback for numbers that come from elsewhere
(a baseline quoted from another paper, which is exactly where `cite=` belongs).

Both generators are pure functions over ledger rows; the CLI writes to
stdout and never edits a document, matching the read-only rule. A
regenerated file that differs from the committed one is the `stale`
verdict made concrete: `attest tex macros --check` exits non-zero when
`results/numbers.tex` no longer matches the ledger.

Cost: moderate — name derivation, two renderers, `--check`, tests on a
synthetic ledger. Value: the highest of the three, because a number that
is derived cannot be mistyped, and the caveats reach the manuscript.

### Option C — the compile gate

`attest claims main.tex` (A) and `attest tex macros --check` (B) as a
`latexmk` pre-hook or a CI step that fails on `contradicted` or on a stale
macro file, with findings printed in the `file:line: message` shape editors
already parse from LaTeX logs. No new capability, just the wiring, and it
is what makes A and B habitual rather than optional.

### Recommendation

A → B → C, in that order, as separate small specs. A is the parity the
second ask needs (an audit has to read `.tex`); B is where the value is; C
is an afternoon once A and B exist. Deliberately **not** proposed: an
Overleaf integration (remote; the offline rule), a LaTeX package that
executes attestation at compile time (a `\write18` shell escape is the
wrong trust boundary), or PDF parsing (the source is on disk; parsing the
rendered output throws that away).

Measured acceptance for A: a golden path `examples/latex/` with a real
three-file paper (`main.tex`, `tables/results.tex`, `refs.bib`) produced
from `examples/workspace/speech-distill`'s ledger, exercising every row of
the table above, with the claim verdicts and the coverage count pinned the
way `examples/citations/` pins its five verdicts. For B: regenerating
`numbers.tex` from the fixture ledger is byte-identical to the committed
file, and `--check` goes red when one metric in the fixture changes.

---

## Part 2: paper audits with reviewer personas

### The shape of the problem

A review has two kinds of finding. Some are **checkable without a model**:
a number no run backs, a citation key that resolves nowhere, a results
table whose arms are single runs, a claim of improvement inside seed
variance, a figure never referenced, a related paper the author has read
and not cited. Others are **judgments**: the baseline is weak, the claim in
the abstract is stronger than the evidence, the method section cannot be
reproduced from what is written. The first kind this project already knows
how to produce; the second needs a model, and the design question is how
to let a model speak without letting it invent.

### Layer 1 — the mechanised checklist (no model)

What every careful reviewer checks first, produced deterministically from
the `.tex` tree (Part 1, A) and the ledger:

| Check | Source | Finding |
|---|---|---|
| numbers vs runs | `claims_check` | contradicted / unsupported / stale |
| unclaimed numbers | `coverage` | "typed, not derived" |
| citation keys | `cite.check` over `\cite` | key resolves nowhere |
| replication | `runs.compare` caveats mapped to the table that shows the family | "each arm is a single run"; "top two within seed variance" |
| corpus mismatch | `compare`'s corpus guard | arms trained on different corpora presented as one sweep |
| direction | `metric_direction` | a metric ranked whose direction the author never declared |
| structure | `.tex` walk | figures/tables defined but never `\ref`ed; `\ref` to a missing label; a results table with no caption |
| **related work the author has read** | `kg.*` + clicks + `\cite` | items the reader marked useful, tagged with the paper's own concepts, that the paper does not cite |

The last row is the one only attestation can do: it joins the reader's own
feedback and knowledge graph to the manuscript, and it needs no model —
the paper's concepts come from its `\keywords` and section titles resolved
through `kg.resolve_query`, the candidate set is the reader's positive
clicks on those concepts, and the check is set difference against the
`.bib`'s DOIs and arXiv ids. Precision will be imperfect (a read paper is
not always a relevant one); it is reported as "you read this and did not
cite it", which is true, not as "you should cite this", which is not
established.

Layer 1 alone is a useful `attest audit main.tex` and ships with no model
at all. Everything in it is a `Finding{file, line, kind, message,
evidence}` where `evidence` is a path or a run name.

### Layer 2 — reviewer personas (a model, grounded)

A persona reads a section and returns findings. The grounding rule, taken
from the installed `grounded-citations` skill and from `simulate.py`'s
"reasoning before verdict": **a finding must quote the manuscript
verbatim, and the quote must be found in the source or the finding is
dropped.** That is the anti-invention mechanism, and it is model-free to
enforce. Each finding is:

```
Finding(
  persona="statistician",
  anchor="a 12.2% relative reduction over the baseline",   # verbatim, verified
  file="main.tex", line=23,
  kind="overclaim" | "unsupported" | "missing-baseline" | "unreproducible" | "unclear",
  concern="one run per arm; the margin is inside the reported seed spread",
  severity=1..3,
)
```

Personas are **data**, like the tagging and reaction prompts: one renderer
(`audit.review_messages(persona, section_text)`), persona definitions in
`evals/personas/*.toml` (name, stance, what to look for, what NOT to flag),
and a corpus `evals/audit_cases.json` of labelled excerpts — real paragraphs
with a known defect and a `note` naming it, plus clean paragraphs that must
produce **no** finding (the refusal case, which the explanation eval showed
is where small models fail: precision 1.0, recall 0.4). The scorer is
model-free: anchor is verbatim, `kind` matches the label, no finding on a
clean case. The gate is `tagging_eval.gate()` — not worse on the primary
model, better on two others, no wider spread.

Candidate personas, each defined by what it refuses to flag as much as
what it looks for:

| Persona | Looks for | Refuses |
|---|---|---|
| statistician | single-run comparisons, missing CIs, margins inside variance, "significant" without a test | prose style |
| replicator | what is needed to rerun that the text omits: seeds, data version, hyperparameters, compute | novelty judgments |
| skeptic (domain) | abstract/conclusion claims stronger than the results section; ablations that do not isolate one variable | typos |
| related-work | comparisons to baselines the reader has read (from Layer 1's list) presented without them | anything Layer 1 already reported |
| area chair | one paragraph: would this be accepted, and the single change that most improves the odds | itemised findings |

Where the model lives: an explicit slow tool with `confirm=true`, like
`feed.simulate_ratings` — never inside a composition tool, and never on the
`.ask` router's path. CLI: `attest audit main.tex [--persona statistician
...]`; MCP: `runs.audit(path, personas, confirm)` on the provenance
surface, returning Layer 1 findings always and Layer 2 findings when asked.
Output is structure; a Markdown report is a renderer over it.

### What has to be measured before multi-persona is believed

The swarm refutation's lesson applies: more model stages are not more
correct. Two questions, pre-registered:

1. **Grounding rate.** What fraction of a persona's findings survive the
   verbatim-anchor check on gemma4:e2b? If it is low, the persona prompt
   is the problem, not the idea, and the corpus is what fixes it.
2. **Do personas add findings, or only words?** On the labelled corpus,
   compare (a) one generic reviewer prompt, (b) five personas unioned and
   deduplicated by anchor, (c) one prompt containing all five concerns.
   The claim "personas find more" is accepted only if (b) beats (c) on
   labelled recall at equal or better precision. If (c) wins, ship (c)
   with the personas as sections of one prompt.

Also measured, because the repo's record says it will surprise: the audit
on the `speech-distill` example must reproduce its known defects (one
contradicted claim, one unsupported, single-run arms) and produce no
finding on a clean paragraph.

### Options for how far to go

- **Option 1 — Layer 1 only.** Deterministic, ships with `.tex` parity,
  includes the read-but-uncited check. No model, no corpus, no eval.
  Recommended as the first spec regardless of what follows.
- **Option 2 — Layer 1 + one grounded reviewer.** One persona (the
  statistician, whose findings are closest to Layer 1's evidence), the
  renderer, the corpus, the scorer, the gate. Establishes the grounding
  rate and the eval loop.
- **Option 3 — the panel.** All five personas plus the area-chair summary,
  shipped only if measurement 2 above favours it.

### Recommendation

Option 1 as its own spec, immediately after LaTeX parity (Part 1, A).
Option 2 as the second spec, with the corpus built from the flows and
`speech-distill` fixtures plus hand-written cases naming known failure
modes (the overclaim in an abstract; the single-run "significant"). Option
3 is gated on the measurement, not on the plan.

### Open questions for Matt

1. Whose `.tex` sources form the eval corpus? The audit cases need real
   paragraphs with real defects; your own past drafts are the honest
   source, and they stay local.
2. Venue checklists (the NeurIPS paper checklist, ICML reproducibility
   items) as Layer 1 rows — worth encoding, or noise for a non-ML field?
   The README says the ledger is the one part that assumes computational
   experiments; the audit would be the second.
3. Report shape: a Markdown `audit.md` beside the paper, or `% reviewer:`
   comments written back into the `.tex` (the read-only rule says the
   former; an author might want the latter).
4. Does "paper audit" extend to auditing *others'* papers from the feed
   (a `feed.read` item's abstract through the skeptic persona)? Different
   evidence base — no ledger, no runs — so Layer 1 shrinks to citations
   and the KG; worth stating in or out of scope now.
