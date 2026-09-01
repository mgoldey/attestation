---
name: attestation-annotate
description: "Annotate prose that states a numeric result with a claim comment beside each decimal so attestation's ledger can check it against a recorded run, add a citation key only after it resolves, and run the coverage linter before handing a draft back. Never asserts a number is true on its own judgement."
version: 1.0.0
author: attestation project
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [claims, citations, verification, provenance, local-api]
    related_skills: [attestation-provenance, attestation-record]
---

# attestation: annotating claims as you write them

Use this while you are drafting or editing prose that states a result --
"WER dropped to 0.053" or "the ablated arm loses 2.1 points". Your job is to
leave a checkable trail beside the number, not to judge on your own whether
it's true: `runs.claims_check` does that against the ledger. If you are
instead the one *reading* a manuscript someone else wrote, or checking
whether the runs on disk still support it, use `attestation-provenance`
instead -- this skill is for the write side, while the ink is still wet.

## When NOT to use this

- Citation *style* -- numbered vs. author-year, BibTeX generation, reference
  list formatting: that's `research-paper-writing`'s job, not this one.
- Grounding prose in a web source with `[n]`-style citations and verbatim
  quotes: that's `grounded-citations`. This skill's `cite=` field is about
  Zotero/`.bib` keys resolving locally, not about web provenance.
- Recording a run you just finished so it enters the ledger in the first
  place: that's `attestation-record`.

## Put a claim beside every decimal

The grammar, exactly (copy it, don't approximate it):

```
<!-- claim: <project>/<run> metric=<m> value=<v> tol=<t> -->
```

An HTML comment so it renders as nothing and the prose reads unchanged.
`<project>/<run>` is a single token, project and run separated by `/`, and
must come first in the body. After it, `metric=` and `value=` are
**required**; a claim missing either is reported as malformed rather than
silently dropped. Optional fields, each `key=value` with no spaces around
`=` and no spaces inside a value: `tol=` (a rounding tolerance, defaults to
effectively exact -- add it whenever the prose number was rounded),
`as_of=` (a freshness date -- see "never invent `as_of`" below before
using it), `split=`, `step=`, and `cite=` (below). `run` may end in `*` as
a wildcard to match several runs by prefix -- expect `ambiguous` if it
matches more than one and that isn't what you meant.

```markdown
The distilled model reaches **0.053 WER** on the held-out set.
<!-- claim: speech-distill/kdsweep_t4 metric=wer value=0.053 tol=0.001 -->
```

**If the run records that metric on more than one split or step, name
which one.** `wer` on a `train` split and a `test` split, or `bleu` logged
at step 18000 and step 22000, are two different rows for the same metric --
with no `split=`/`step=`, the checker cannot tell which one your claim is
about and returns `ambiguous`, exactly the same as an under-specified
wildcard `run`. Name the field the same way a config would:

```markdown
BLEU reaches **34.1** on the dev split and **31.7** on test.
<!-- claim: mt-with-split/kdsweep_t4 metric=bleu value=34.1 split=dev -->
<!-- claim: mt-with-split/kdsweep_t4 metric=bleu value=31.7 split=test -->
```

One claim per decimal, placed next to the sentence it backs. A paragraph
with three numbers needs three claims, not one claim covering the
paragraph -- `runs.claims_coverage` finds decimals with nothing beside them,
and it counts per number, not per paragraph. Two decimals in one sentence
still get two separate comments:

```markdown
The ablated arm scores **accuracy 0.912** and **f1 0.887**, both within
tolerance of the baseline.
<!-- claim: cls-two-metrics/kdsweep_t4 metric=accuracy value=0.912 -->
<!-- claim: cls-two-metrics/kdsweep_t4 metric=f1 value=0.887 -->
```

**Every decimal gets its own claim comment. No exceptions** -- not "the
two numbers are close together", not "the second one is obviously the
same run as the first". A reader (or `runs.claims_coverage`) checks
decimal by decimal, and an unannotated one is indistinguishable from one
you forgot.

## Never invent `as_of` -- it is what makes a claim go stale

`as_of=<ISO date>` asserts "the artifact backing this had not changed as
of this date". `stale` fires when the *value still matches* but the run's
file has an mtime *after* `as_of` -- it means "re-verify, something moved
since you last looked", not "this number is old" and not "format the
date". **Omit `as_of` entirely unless you are deliberately asserting a
freshness date you know to be true.** Writing today's date, or any date,
"to be thorough" on every claim is how three claims for the same run all
came back `stale` in one draft: each carried an `as_of` earlier than the
artifact's real mtime, which is exactly the condition that triggers it.
If you are not asserting a specific date, leave the field off -- a claim
with no `as_of` is simply never stale.

## Only cite a key that already resolves

Add `cite=<key>` to a claim only after confirming the key resolves:

```
cite.lookup(key="vaswani2017")
```

If it comes back empty, do not add the field anyway and do not invent a
plausible-looking key by composing one from an author/year you inferred --
`runs.claims_check`'s `uncited` verdict exists precisely to catch a key that
resolves nowhere, and a key you made up will fail it the same way a typo
would. If the reference isn't in Zotero or a `.bib` file yet, leave `cite=`
off until it is, or hand the citation off to `research-paper-writing` /
`grounded-citations` to get it added properly first.

```markdown
This confirms the scaling trend reported by Vaswani et al.
<!-- claim: speech-distill/kdsweep_t4 metric=wer value=0.053 cite=vaswani2017 -->
```

## Before handing a draft back, run the linter

Always run coverage before calling the draft done -- it's the one people
forget, because a document can have zero contradicted claims and still be
full of unverifiable numbers:

```
runs.claims_coverage(path="paper.md")   # decimals with no claim beside them
runs.claims_check(path="paper.md")      # do the claims that exist check out
```

Report the uncovered decimals from `claims_coverage` back to the reader by
value and location -- don't silently annotate them yourself with a guessed
run, and don't drop them from the report because they're minor. Only
decimals count as measurements; versions, ISO dates, URLs and package pins
are excluded on purpose, so don't add claims to those.

## Six verdicts from `runs.claims_check`, and what to do for each

`supported` -- done, nothing to change. `unsupported` -- no run matches;
the claim may be true but nothing backs it yet, so say that rather than
implying it's wrong. `ambiguous` -- either a wildcard `run` matched more
than one run, or the metric exists on more than one split/step and the
claim didn't say which (see `split=`/`step=` above) -- disambiguate rather
than guessing. `stale` -- the value matches but the artifact changed after
`as_of` (see above); re-verify, and only re-add `as_of` if you're sure of
the new date. `uncited` -- the `cite=` key doesn't resolve; fix or drop it
per the section above.

**`contradicted` is the one that needs action, not just a report**: a
recorded run disagrees with the value in the prose. Treat it as "fix the
document or the run, and say which" -- never paper over it by loosening
`tol=` until it passes, and never leave it in a draft you're handing back
without flagging exactly which claim contradicted and by how much. The
reader has to decide whether the prose was wrong or the run was, and that
decision is theirs, not something to guess your way past.

## The worked example

`examples/citations/` is a complete run of this: a four-entry `.bib`
library, a draft (`DRAFT.md`) whose claims cite three real keys plus one
(`doe2099imaginary`) that resolves nowhere on purpose, and
`check_citations.py` running the real linters against it. Read it before
annotating your first real document -- it shows the grammar and the
uncited case side by side rather than in isolation.
