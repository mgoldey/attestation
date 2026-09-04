---
name: attestation-provenance
description: "Verify a manuscript you are handed against the experiment runs on disk: rank the arms of a sweep with their caveats, check every numeric claim in an existing Markdown draft against the recorded run, list the numbers no claim covers, and lint citation keys that resolve to nothing. Reads artifacts and existing prose; never re-runs, edits, or writes a document."
version: 2.0.0
author: attestation project
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [experiments, claims, verification, reproducibility, local-api]
    related_skills: [attestation-setup, attestation-knowledge]
---

# attestation: runs and claims

Use this when the reader asks which arm of a sweep won, whether the numbers
in a draft are still true, or what a recorded run actually contained. You
read experiment artifacts already on disk -- no instrumentation, no
re-running anything. **The caveats you return are the product, not a
disclaimer: relay them verbatim.** When a comparison refuses, the refusal is
the right answer and it names what to do next; do not work around it by
picking a metric yourself.

## When NOT to use this

- Launching or monitoring a training run: nothing here runs anything.
- Whether a cited work *supports* a claim: `cite.check` says a key resolves
  or does not, never that the paper agrees. Judging support needs a model
  and is not what this does.
- Finding papers, or what the reader has been reading about: the feed and
  knowledge agents (`attestation-feed`, `attestation-knowledge`).

## Ask the router first

```
runs.ask(question="which arm won?")                        # names the comparable families
runs.ask(question="which arm won?", family="kdsweep")      # compares
runs.ask(question="which arm won?", family="kdsweep", metric="wer")  # by a named metric
runs.ask(question="are the numbers in my draft right?", path="paper.md")
```

If the question names a metric ("compare by wer"), pass it as `metric` too
-- do not rely on it surviving into your own paraphrase of `question`. A
comparison with no metric declared falls back to whichever one most arms
share, which can silently answer a different question than the one asked.

These dotted names are for you to read, not a literal call string: some MCP
clients rewrite `runs.ask` to something like `mcp__attestation__runs_ask`
before it ever reaches you. Call the tool by the exact name your own tool
list shows for the same arguments -- never retry a plausible-looking
variant of the dotted name, and never conclude a tool "does not exist"
without checking your tool list first.

`runs.ask` without a `family` does not guess one: it lists what is
comparable and asks. Pass the name back and it compares. Every router returns
`answer` (relay VERBATIM), `refs`, `caveat` (unabridged), `options` and
`tool_used`; `ok=false` with `options` means ask the reader, never pick for
them. Specific tools may be hidden from your session; `runs.tools` explains
why and how to reveal them.

## The ledger

Runs are read from artifacts a project already has -- `results/`, `logs/`,
`outputs/`, `metrics/`, `eval/`, `benchmarks/`, `reports/` holding JSON,
JSONL, CSV, YAML or TOML, with `configs/` recorded as provenance and never as
a metric. Nothing is registered in advance. `runs.scan(root, project,
confirm)` walks a workspace (default `$RESEARCH_ROOT`), treats each
subdirectory as a project, and needs `confirm=true` because it replaces each
scanned project's rows. Directories with nothing recognisable are reported
in `empty` rather than omitted: "found nothing" must never look like
"nothing was there".

**The scan also reads five tracker layouts** -- W&B (`wandb/*/files/
wandb-summary.json`; offline mode writes no summary until synced), MLflow
(`mlruns/<exp>/<run>/metrics/`, final line per metric), Sacred
(`FileStorageObserver` dirs, `run.json`'s own `result` read as a metric),
DVC (`dvc.yaml`'s declared `metrics:` files) and Hydra (`--multirun` sweep
dirs, which need `hydra.job.chdir=true` or every arm overwrites one file).
Each was verified against a directory the real tool produced. All read
**final values, not curves**: a diverged run whose last value is `nan`
records nothing rather than a mid-training number, and a reader who wants
training curves is not served here -- say so. `runs.compare` carries the
per-tracker caveat for whichever produced the arms; relaying it is not
optional.

`runs.list(project, family, limit)` shows what exists and the *families*
runs group into. **A family is a shared filename prefix, not a project.**
`runs.compare` with a project name is the commonest mistake, and the error
lists what is comparable -- read it rather than guessing again.

## "Which arm won?"

```
runs.scan(confirm=true)        # only if runs.list says the ledger is empty
runs.list()                    # returns `families`
runs.compare(family="kdsweep", metric="wer")
```

`runs.compare` **refuses to rank a metric whose direction is undeclared**
rather than guessing: ranking WER as if higher were better names the worst
arm the winner. The refusal names the metric; the fix is a
`[metric_direction]` entry in `~/.hermes/metric_direction.toml`, made by the
person who knows which way is better. It returns `caveats` -- small samples,
arms on different sample sizes, a top two within 5%, arms at different
training steps, arms whose corpora differ -- and every row carries its
`source_path` and `n`. A comparison with no caveats has earned that silence.

**Report every caveat, verbatim:**

```
winner: kdsweep_t4
caveat: the top two arms differ by 0.0017 (2.6%) -- too close to call
caveat: each arm is a single run; no seed replication ...
```

A comparison whose margin is smaller than its seed variance has not found
anything, and presenting the winner without the caveat misrepresents it.

`runs.detail(project, name)` gives one run in full -- config, every metric,
source path, and any prose header from its config, which is often where the
hypothesis and the single changed variable are written down.

## "Is this number in my draft right?"

A claim is an HTML comment beside the prose it describes, so the document
renders unchanged:

    <!-- claim: project/run metric=wer value=0.053 tol=0.001 as_of=2026-05-28 -->

All three linters when the reader asks about a whole manuscript, because
each catches what the others cannot:

```
runs.claims_check(path="paper.md")     # do the numbers match the runs?
runs.claims_coverage(path="paper.md")  # which numbers nothing covers
cite.check(path="paper.md")            # which citation keys resolve to nothing
```

Five verdicts from `runs.claims_check`, and the distinctions are the point.
`supported`: a run agrees within tolerance. `contradicted`: a run disagrees
-- the document or the run is wrong, and say which needs a correction.
`unsupported`: no run matches, so the claim may be true but nothing backs
it. `ambiguous`: a wildcard matched several runs. `stale`: the value
matches a run whose artifact changed after `as_of`. `malformed` annotations
are reported rather than skipped, so a claim cannot vanish from review.
**Never report `unsupported` as if it meant false.**

`runs.claims_coverage` is the inverse and the one people forget: decimals
asserted in prose that no claim covers. A document with zero contradicted
claims can still assert a dozen unverifiable numbers. Only decimals count;
versions, dates, URLs and package pins are excluded.

`cite.check` reads the `cite=<key>` field a claim may carry:

    <!-- claim: project/run metric=wer value=0.053 cite=vaswani2017 -->

and reports keys **no configured source can resolve** -- Zotero, `.bib`
files, and CrossRef only when `ATTEST_CITATION_WEB` is set. It is a lint:
the key is unknown here, nothing more. Claims without `cite` are skipped.

All three are read-only. They report; they never edit a document.

## Mistakes that look reasonable

| Instead of | Do |
|---|---|
| `runs.compare(family="<project>")` | `runs.list()` and use a name from `families` |
| Picking a metric direction yourself | Relay the refusal; the reader declares it |
| Presenting a winner alone | Include every caveat verbatim |
| Checking a manuscript with `runs.claims_check` alone | Add `runs.claims_coverage` and `cite.check`; they lint different things |
| Reading `unsupported` as "wrong" | One needs a run, the other needs a correction |
| Reading `cite.check` as "the paper does not support this" | It says the key does not resolve, nothing more |
| Retrying a failed call with different arguments | Read the message -- it names the fix |
| Concluding a missing tool is broken | It is hidden (`runs.tools`) or the server is stale (`attestation-setup`) |
