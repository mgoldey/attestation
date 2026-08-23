# `attest claims --suggest`: propose the annotations, never write them

**Date:** 2026-08-23
**Status:** proposed
**Kind:** design.

## Problem

The claim checker is the sharpest idea in this repo and the one nothing else
does — five verdicts that genuinely do not collapse, `exit 1` on a
contradiction so it gates a commit. Two independent value reviews reached the
same conclusion about why it is not used: **nothing writes claims for you.**

Measured on the author's own machine: **1 of 9,827 markdown files** carries a
`<!-- claim: -->` annotation, and the two other files on the whole machine that
contain the string are the tool's own skill docs. The checker's value is gated
behind a blank-page problem.

`--coverage` already does the hard half. On the example paper it reports:

```
uncovered  FINDINGS.md:13   41.3      41.3M parameters.
uncovered  FINDINGS.md:23   12.2      Temperature 4 is better still, at 0.0642 -- a 12.2% relative
5/9 number(s) covered by a claim across 1 file(s)
```

It finds the numbers, their lines, and their context. It stops one step short
of the thing that would make it actionable: saying which run each number
probably came from.

## Design

`attest claims --suggest <path>` prints annotations a human can paste. It
writes nothing.

### Why it must not write

An annotation asserts "this number came from that run." A tool that inserts
that assertion automatically is manufacturing provenance, which inverts the
product's entire premise. The output is a proposal for a human to accept,
reject, or correct — the same stance `runs compare` takes when it refuses to
guess a metric direction.

`--suggest` prints to stdout in paste-ready form, with the target line named:

```
FINDINGS.md:13
  <!-- claim: speech-distill/kdsweep_baseline metric=n_params value=41.3 tol=0.1 unit=M -->
  # 41.3 in prose; ledger has n_params=41287424 in 3 runs of speech-distill
  # AMBIGUOUS: baseline, t2 and t4 all record this value -- pick one
```

### Matching is the whole design, and it is harder than it looks

The naive version — find a run metric equal to the prose number — finds
**nothing** on the example paper. Verified: `41.3` in prose is `41287424` in
the ledger. The real cases:

1. **Unit-scaled.** `41.3M` vs `41287424`. Requires trying common scale factors
   (K, M, B, %, and their absent-suffix equivalents) and reporting which one
   was applied, since a wrong scale is a wrong claim.
2. **Percentage-of.** `12.2% relative improvement` is not a stored metric at
   all — it is derived from two of them. Out of scope for v1, and it must say
   so rather than mis-attributing: an unmatched number gets `# no run in the
   ledger records this value` and no annotation.
3. **Ambiguous.** `41.3` matches three runs. All are listed, none is chosen.
4. **Rounded.** `0.0642` vs a stored `0.06421`. This is what `tol=` exists for,
   and the suggested tolerance should be derived from the prose number's
   significant figures, not a constant.

Precision beats recall here by a wide margin. A suggestion a human accepts
without checking, that turns out wrong, is worse than no suggestion — it puts
a false provenance claim into a document under the tool's own imprimatur.
When in doubt, emit the comment lines and no annotation.

### Reuses what exists

- `coverage()` already finds uncovered numbers with line and context.
- `_masked_prose()` already knows which numbers are inside an existing claim.
- `_REF_RE` and the `Claim` dataclass already define the syntax to emit.

The new part is the matcher and the printer. No schema change.

## Refusal conditions

Abandon this if, on a real paper, precision is below roughly 90% — that is,
if more than one suggestion in ten would be wrong if accepted. At that rate a
careful user must verify every line by hand, which is the work `--suggest` was
supposed to remove, and a careless one poisons their own provenance.

## Open questions

- Whether derived quantities (the `12.2% relative` case) ever get support. They
  are common in papers and they need an expression language, which is a much
  larger spec.
- Whether `--suggest` should read the whole ledger or only runs whose project
  matches the document's directory. The latter is far more precise and needs a
  convention that does not exist yet.
- Whether this belongs behind `--coverage` as an extra column rather than as
  its own flag. Probably not: coverage is a lint over prose, suggestion is a
  query against the ledger, and one of them can be wrong in a way the other
  cannot.
