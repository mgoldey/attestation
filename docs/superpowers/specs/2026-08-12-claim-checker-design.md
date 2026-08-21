# Claim checker — design

**Date:** 2026-08-12
**Status:** proposed

## Problem

`INDEX.md` says:

> GW100 complete 100/100, MAE 0.353 eV vs experiment @ aDZ

That number was transcribed by hand from a benchmark artifact. Nothing checks
it. If the benchmark is re-run and the MAE moves, the README keeps asserting
0.353 indefinitely, and the only way to notice is to remember to look.

This is the gap no existing tool fills. MLflow tracks runs but has no notion of
a claim in a document. DVC versions data, not assertions. Obsidian links notes
to notes with no verification. The missing primitive is a checkable link from
*an assertion a human wrote* to *the evidence that supports it*.

## What the claims actually look like

Measured, not assumed. From the real corpus:

> leaves WER essentially unchanged (**0.053 vs. 0.043** baseline)

> detector ranks held-out fragility ρ +0.41/+0.68, downstream dipole ρ +0.72

Two facts drive the whole design:

1. **Claims are embedded in narrative prose**, not in structured blocks. They
   are written to be read by a person, and that is their primary job.
2. **They are sparse.** `INDEX.md` has 5 numeric lines across 8 projects.

A format that requires restructuring this prose into YAML front-matter or a
claims table would not be adopted, and an unadopted checker is worth nothing.

## Decisions

### Annotation: an HTML comment beside the prose

```markdown
The cut leaves WER essentially unchanged (**0.053 vs. 0.043** baseline).
<!-- claim: ablation/whisper-small-ablated metric=wer value=0.053 tol=0.001 -->
```

An HTML comment because it is the only annotation that is simultaneously:
invisible in every Markdown renderer (GitHub, Obsidian, VS Code, pandoc),
plain text so `grep` and `git diff` work on it, and adjacent to the prose it
describes rather than in a separate file that drifts out of sync.

The prose is never touched. A claim is *added beside* an assertion that already
exists, which means annotating is incremental: one claim is useful, and there
is no all-or-nothing migration.

### Verdicts, and what each one means

| Verdict | Meaning |
|---|---|
| `supported` | A run was found, and its value matches within tolerance. |
| `contradicted` | A run was found and its value **disagrees**. The document is wrong, or the run is. |
| `unsupported` | No run matches the reference. The claim may still be true — nothing backs it. |
| `ambiguous` | The reference matched more than one run. Which one is meant is undecidable. |
| `stale` | The artifact changed after the claim was written. |

`unsupported` and `contradicted` are deliberately different. Conflating them
would let "I never recorded this" masquerade as "this is false", and the
response to each is different: one needs a run, the other needs a correction.

`ambiguous` exists because silently picking the first of several matching runs
is how a checker reports a confident wrong answer.

### Tolerance is required, and defaults tight

`value=0.353` in a document is a rounded transcription of something like
0.35281. An exact-match checker would report every claim contradicted, and be
switched off within a day.

Default tolerance is `1e-9` relative — effectively exact — and the annotation
carries `tol=` when the number was rounded. Requiring the author to state the
tolerance keeps the judgement with the person who knows how the number was
produced.

### Staleness by artifact mtime, not content hash

A claim records nothing about the artifact's contents. Staleness is: the source
file's mtime is newer than the claim's recorded `as_of` date, if it has one.
This is weaker than hashing and is chosen anyway — hashing means storing a
hash per claim, which is state that must be regenerated and can itself go
stale. mtime answers the only question asked: *has the evidence moved since
someone last looked?*

A claim with no `as_of` is never stale, only supported/contradicted.

### Read-only, no state

The checker parses Markdown, queries the ledger, and reports. It never edits
documents and stores nothing. Re-running it is the only way to get a verdict,
so a verdict can never be stale in the way a cached badge can.

## Format

```
<!-- claim: <project>/<run-name> metric=<name> value=<number> [tol=<number>] [as_of=YYYY-MM-DD] -->
```

`<run-name>` may end in `*` to match a family prefix, which is what makes
`ambiguous` reachable and useful: `ablation/whisper-*` says "some run in this
sweep", and if several match, that is worth knowing.

## MCP tools

| Tool | Behavior |
|---|---|
| `claims_check(path=None)` | Verify every claim under a path (default: the workspace). Returns per-claim verdicts. |
| `claims_list(path=None, verdict=None)` | The claims found, filterable by verdict — "show me what is unsupported". |

Plus `hermes claims check [path]` for the terminal.

## Testing

- Each verdict reachable and asserted, on fixtures written here.
- `contradicted` and `unsupported` never collapse into each other.
- Tolerance: a value inside `tol` is supported, outside is contradicted.
- A wildcard matching two runs is `ambiguous`, never silently first-match.
- A malformed annotation is reported as malformed, not skipped — silent
  skipping is how a claim disappears from review without anyone deciding to
  remove it.
- The checker never writes to a document.

## Out of scope

Auto-inserting claims; rewriting prose to match a run; NLP extraction of
implicit claims from text (a claim is asserted deliberately or not at all);
badges or CI gating.
