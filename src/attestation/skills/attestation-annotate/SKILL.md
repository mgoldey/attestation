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
`=` and no spaces inside a value: `tol=` (defaults to an effectively-exact
tolerance -- add it whenever the prose number was rounded), `as_of=` (an
ISO date; the claim goes `stale` if the run's artifact changed after it),
`split=`, `step=`, and `cite=` (below). `run` may end in `*` as a wildcard
to match several runs by prefix -- expect `ambiguous` if it matches more
than one and that isn't what you meant.

```markdown
The distilled model reaches **0.053 WER** on the held-out set.
<!-- claim: speech-distill/kdsweep_t4 metric=wer value=0.053 tol=0.001 as_of=2026-05-28 -->
```

One claim per decimal, placed next to the sentence it backs. A paragraph
with three numbers needs three claims, not one claim covering the
paragraph -- `runs.claims_coverage` finds decimals with nothing beside them,
and it counts per number, not per paragraph.

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
implying it's wrong. `ambiguous` -- a wildcard matched more than one run;
narrow the `run` field to the one meant. `stale` -- the value matches but
the artifact changed since `as_of`; re-verify and bump `as_of`. `uncited`
-- the `cite=` key doesn't resolve; fix or drop it per the section above.

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
