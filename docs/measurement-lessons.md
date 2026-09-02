# What measurement kept overturning

**Date:** 2026-08-24
**Status:** living. Append findings; do not rewrite history.
**Scope:** cross-cutting method. Individual findings live in their own specs —
this file records what they have in common.

This repo's design decisions have been reversed by measurement often enough
that the pattern is now the most valuable thing to write down. Four subsystems
were built or rebuilt on results that contradicted a confident prior:

- the swarm (`docs/superpowers/specs/2026-08-22-swarm-refutation.md`)
- the 0.964 AUC that turned out to classify provenance (commit `bc37a5b`)
- the ledger fixture that overstated yield (3 real comparisons in 1045 runs)
- the agent-side prompt (2026-08-24, below)

The through-line: **a number is about the artifact it was taken from, not
about the system.** Every failure below is one of confusing the two.

---

## 1. Architecture: a model choosing at runtime is a stage that can be wrong

Measured on `gemma4:e2b-it-q4_K_M`, 15 turns × 3 runs:

| architecture | correct | latency |
|---|---|---|
| Routed — deterministic dispatch | **13/15** | 1.3s |
| Flat — one namespace | 8/15 | 1.3s |
| Swarm — supervisor LLM + subagent LLM | 7.3/15 | 2.8s |

The swarm did worse than doing nothing, at twice the latency. Two 80%-accurate
stages in series make a 64%-accurate pipeline, and a namespace miss is
unrecoverable — competence inside the wrong agent cannot save the turn.

The decomposition was not the problem. Four per-domain agent surfaces shipped;
what was rejected is a *model* selecting between them at runtime.

> Separate agents help when a **person** chooses which to talk to.
> They hurt when a **model** chooses at runtime.

Full argument, including what survives for multi-step orchestration:
`docs/superpowers/specs/2026-08-22-swarm-refutation.md`.

---

## 2. Prompt tuning: length is zero-sum on a small model

Every intuition here was wrong at least once. Measured on `gemma4:e2b`,
temp 0, against the live 45-tool MCP surface.

**Longer descriptions make routing worse.** Lengthening the skill description
for `research-provenance` to assert priority dropped it 12/16 → 9/16. The
extra words added keyword surface for competitors to match, and it lost cases
the short version won.

**A worked example outweighs the rule beside it.** A paragraph explaining that
Slack needs `<url|title>`, placed directly above a Markdown example, produced
5/5 items and **0/5** correct links across three runs — the model copied the
example and ignored the prose. Only changing the example moved the result.

**Fixing one case breaks another.** Adding "call a listing tool to discover
arguments you lack" repaired `runs_compare` and broke two feed cases. Net zero.

**Instructions must name concrete values, not describe them.** With no persona
named, the model passed `user: "user"` on **9 of 9** feed calls — the literal
placeholder. Because `@tool(autocreate_user=True)` creates unknown names rather
than refusing, every such call risked an empty persona that ranks badly
forever. Naming the persona in the prompt: 9/9 correct.

**Restricting the tool surface is not automatically a win.** `ATTEST_TOOLS=feed`
(21 tools) scored *worse* than the full 46 tools (7/12 vs 9/12) before the system
prompt was fixed. The failures were the model declining to call anything, not
choosing wrongly — a framing problem, not a surface-size problem.

Net effect of the final agent-side prompt, read from the live file:

| | before | after |
|---|---|---|
| correct persona passed | 0/9 | **9/9** |
| tool selection | 12/24 | **24/24** |
| Telegram links clickable | 0/5 | **5/5** |

**Rule:** re-measure after every prompt edit, including edits that only add.
There is no monotonic "more guidance is better" on a 2B model.

---

## 3. The recurring bug: measuring the artifact instead of the system

Four measurements in a single session (2026-08-24) were confidently wrong.
Each was repeatable, deterministic, and about the wrong thing.

| what was measured | what it was taken to mean | actual |
|---|---|---|
| a transcript header | reader is on Slack | Telegram; the "fix" would have broken every link |
| a hand-built 8-skill index | collisions cost 21/24 → 8/24 | on the real 70-skill tree: 4/16 either way |
| skill-name selection | "routing is 25%" | the feed uses MCP tools directly and never selects a skill |
| grading `feed_list` as the only right answer | 12/24 | `feed_digest` is also correct; correct refusals were scored as failures → 21/24 |

Each was cheap to check against the real thing: `~/.hermes/config.yaml` names
the platform and the enabled MCP servers; the skill tree can be walked; the
Telegram formatter can be imported and called on a candidate string. **Modelling
was faster than looking, and wrong every time.**

**Before reporting a measurement, answer three questions:**

1. Is this the path the system actually takes? (An enabled MCP server means the
   model never looks for a skill.)
2. Is this index/fixture/dump the live one? (A prompt dump is evidence about
   that dump. One sampled here predated the MCP wiring entirely.)
3. Is a "failure" here actually correct behaviour? (Declining to call
   `runs_claims_check` when the question carries no file path is right.)

---

## 4. Guards: a passing test is evidence of nothing

Every regression guard written on 2026-08-24 passed on first run. Three
protected nothing:

- **Wrong scope.** A "tools returning a url must say to show it" test keyed off
  `outputSchema`; only the four `.ask` routers declare one, so it inspected
  four tools and skipped `feed.list` and `feed.search` — the two from the bug
  report. Deleting either instruction still passed.
- **Co-occurrence, not adjacency.** `"url" in text and "show" in text` passed
  with the whole instruction removed: every docstring already says "Returns …
  url" in passing.
- **Prose vs machinery.** Scanning the serialized schema matched a field
  literally named `url` (`"title": "Url"`) beside an unrelated verb in one
  unsplittable JSON blob.

A fourth was *inverted* — it required Slack syntax, which measurement showed
breaks Telegram. Both directions are now guarded.

**Rule:** delete or weaken the exact thing a guard protects and confirm it
FAILS, then restore. Mutate each protected site separately — a partial mutation
that leaves matching text behind is a bad mutant, not a good guard. Where a
rule has two failure directions, mutate both.

This is the repo's stated failure mode ("tests that pass against the bug they
were written to catch") reproduced three times in one day by the person who
wrote that line.

---

## 5. Where the bug usually is not

The reported symptom pointed at the wrong layer in every case this session:

- "You didn't give links" — not a data bug. All 5499 items carry a url, and the
  five in question had real arXiv addresses in the payload the model received.
  It rendered `item_id` instead.
- "The links weren't clickable" — not a rendering bug. The gateway converts
  Markdown to MarkdownV2 correctly; there were simply no links to convert.
- "Routing is 25%" — not a routing bug. The eval was wrong.

The one genuine live defect (a junk persona created on every feed query)
surfaced only while testing something else.

**Rule:** reproduce the symptom against the live payload before theorising
about the layer that produced it.

---

## 5. An optimizer memorizes; only a held-out model tells you it happened

DSPy GEPA on the tagging prompt (2026-08-27; full record in
`docs/superpowers/specs/2026-08-23-dspy-prompt-optimization-design.md`).

**The train number lied in the usual way.** 23 train cases, 300 metric calls:
0.790 → 0.902 inside the optimizer. Through the production client on the
28 held-out dev cases: 0.819 → 0.824. The instruction it wrote is 8× longer
and quotes tags from specific train items back as "rules".

**Transfer told the truth the primary model could not.** Scored on three
models, twice per case: the candidate *tied* the model it was tuned on
(0.807 / 0.807) and beat the two it never saw by +0.110 and +0.086. The
pre-registered gate failed it on the tie, and stays as written. The finding
underneath is the hypothesis the spec started from: the hand-written prompt
is fitted to gemma4:e2b's idiom, and a longer instruction e2b cannot exploit
(§2: length is zero-sum on a 2B model) is exactly what larger models can.

**A tie at 56 samples is not a result either way.** Re-running until the
primary wins is the same tautology as selecting demonstrations from the
scoring set.

**The pre-registered rule was wrong, and it was changed in the open.** Rule 1
said "beats the baseline on the primary"; on this evidence Matt's call was
that transferability is the stronger signal, which is what the spec argued
from the start. Rule 1 is now "not worse on the primary", the amendment is
dated and cites this run in `gate()`'s docstring and the spec, and the
candidate shipped as `DEFAULT_TAG_INSTRUCTION`. The distinction that matters:
loosening a bar silently to pass a number is the tautology; loosening it
with the reason written next to the result is a design decision.

**Rule:** decide the acceptance bar before the run, score on models the
optimizer never saw, and treat the optimizer's own number as the artifact it
is. If the bar turns out to be wrong, change it where the next reader will
see why.

---

## 6. 2026-08-28/29: what the golden paths and corpora overturned

Twelve golden paths and three task corpora landed together. Measurement
overturned something in nearly every one of them.

**A passing transfer gate is a sample, not a certificate.** The tagging
transfer gate recorded `PASS` on 2026-08-27 at `repeat=2`. Re-run at
`repeat=1` on 2026-08-28 with the identical prompt and cases, it recorded
`FAIL` — `hermes3:8b` landed at 0.798 against a 0.818 baseline, where the
committed run had it the other way (`evals/prompts/transfer-2026-08-28.md`).
Nothing about the candidate changed; the sample did. **Rule:** treat a gate
verdict as a sample from the run that produced it, not a property of the
candidate — the committed, dated artifact is the record, and a lower-repeat
re-run is a demonstration of the mechanism, not a re-certification.

**Offline W&B writes no summary or config files at all, and the docstring
that said otherwise was also wrong about which detail mattered.** The
reader's docstring assumed the run-directory name (`run-<timestamp>-<id>`)
was the risk; a real offline run (`examples/wandb/generate.py`, wandb
0.17.6) named its directory `offline-run-<timestamp>-<id>` instead, but the
reader never filtered on that prefix, so nothing broke. The real gap was
upstream of naming: `wandb-summary.json` and `config.yaml` do not exist in
`files/` until `wandb sync` uploads to a server — every logged value stays
inside the run's binary `.wandb` transaction log until then. The committed
fixture's summary/config files were materialised by decoding that binary
log directly (`wandb.sdk.internal.datastore`), the documented community
workaround, not a real synced run. See
`docs/superpowers/specs/2026-08-22-tracker-adapters-design.md`'s 2026-08-28
update. **Rule:** a guessed bug list and a fixture both built from the same
documentation share a blind spot — only running the real tool against a
real directory finds the gap that documentation does not mention.

**Hydra 1.3 does not chdir into each arm's directory by default.** The
golden-paths brief assumed Hydra changes the working directory per job —
true through Hydra 1.1, changed in 1.2. Without `hydra.job.chdir=True`
passed explicitly, a real `--multirun lr=0.01,0.1,1,10` sweep wrote a
single top-level `metrics.json`, overwritten by each of the four arms in
turn, not four separate files under `multirun/`. The Hydra golden path
(`examples/hydra/generate.sh`) now passes the override explicitly and the
reader documents why it is required, in
`docs/superpowers/specs/2026-08-22-tracker-adapters-design.md`'s Hydra
subsection. **Rule:** a tool's own version history can invalidate a
convention "everyone knows" — verify the default against the installed
version rather than against memory of an older one.

**A fixture with fixed dates empties a 14-day demo with no test going red.**
The flows corpus originally carried an `<updated>` element dated August
2026 on each entry; `feed.list`/`rank_items` default to a 14-day window, so
the demo would have shown fewer items every week and nothing by
mid-September, silently, since no test asserts freshness. The entry-level
dates were dropped — see
`docs/superpowers/specs/2026-08-28-example-flows-design.md`. **Rule:** a
fixture whose correctness depends on wall-clock time is a bug with a delay
timer attached — remove the date rather than trusting a future test run to
notice it went stale.

**The attribution guard matched CI's own ambient username, not a leak.**
`test_no_committed_example_carries_attribution_or_machine_paths` scans every
committed example for the machine's username as a whole word. GitHub Actions
sets `$USER=runner`, and "runner" is ordinary prose in Hydra's own README
("a GitHub Actions runner") — a real CI failure (run 33233059347), not a
leak the guard was meant to catch. Ruling: the ambient-username check is
skipped under `CI`, and for a short list of generic account names (`runner`,
`root`, `user`, `ubuntu`, `admin`, `ci`) even off CI; the guard's job is
catching a real leaked path or handle, not colliding with the environment it
runs in. See `docs/superpowers/specs/2026-08-28-golden-paths-design.md`.
**Rule:** a guard that reads the ambient environment can be defeated by that
same environment — a username check must exclude the generic account names
the environment itself is likely to hand it, on and off CI alike.

**`family_of` returned no family at all for a bare hyperparameter stem, and
joins on a separator its own worked example did not state.** `lr_0.001`
and `lr_0.01` have no shared prefix beyond the recognised split token
itself — stripping the token the way a sweep or series case does leaves
nothing. `family_of` now falls back to the token's own name (`lr`) as the
family when it consumed the entire stem, so `attest runs compare lr` groups
a four-arm learning-rate sweep that a real Keras/CSVLogger run produced
(`examples/tensorflow/`). Separately, the *sweep* case's own worked example
under-specified its output: `family_of('dit_small_rope_crossattn')` returns
`dit-small-rope` — hyphen-joined regardless of whether the input used `-`
or `_` — not the underscored `dit_small_rope` the docstring's prose implied
before this pass corrected it. See
`src/attestation/ledger_adapters/generic.py::family_of`. **Rule:** run a
function's own worked examples through the function before publishing them
— a docstring that only describes behavior in prose can drift from what the
code actually returns without any test catching it, since prose is not
executable.

**The explanation prompt's refusal clause under-refuses more than four
items could show.** `explain.py`'s refusal clause — fixed after it once
claimed a termite-feed paper shared "advanced topics like AI" — had never
been scored on more than the four items that found the bug. Measured
properly on the new 40-case corpus: refusal precision 1.0, recall 0.4 — it
misses 6 of 10 unrelated items, not the near-perfect guard four items
suggested (`examples/prompt-evals/README.md`). **Rule:** a fix verified
against the items that exposed the bug is verified against a sample of one
failure mode — score it against a corpus built to cover the space the bug
came from before trusting the fix generalizes.

**The reaction prompt's confidence field carries no signal a corpus can
detect.** `simulate.py`'s `confidence` field was assumed, from the incident
that renamed `strength` to `confidence`, to now carry real information;
scored on the 100-case reaction corpus it stayed near-inert, a histogram
concentrated at `{4: 7, 5: 92}` — confirming on a real corpus what four
items could only hint at (`examples/prompt-evals/README.md`). **Rule:**
"the rename fixed the bug" and "the field now carries signal" are different
claims — the first is a code fix, the second needs a distribution measured
across enough cases to show variance, or its absence.

**`attest claims` never ran the citation lint for its whole life, until a
golden path drove it.** `check_citations()` existed and was wired into the
MCP tools (`cite.check`, `runs.claims_check`), but the CLI's `cmd_claims`
had no resolver to pass it, so every `cite=<key>` annotation checked from
the terminal was silently skipped. Building `examples/citations/` — a
draft with one citation key that resolves nowhere on purpose — required
running `attest claims` against it and noticing the fifth verdict never
appeared. `cmd_claims` now builds a resolver with `_citation_resolver()`
the same way the MCP side does, and the CLI reports `uncited` alongside the
four numeric verdicts (commit `4fb6007`). **Rule:** two code paths doing the
"same" job (a CLI command and an MCP tool calling the same library
function) can silently diverge when one forgets to wire a parameter — a
golden path that exercises the CLI path for real is what caught it, not a
read of the source.

## 7. 2026-08-29: three seam tests passed against the mutants they were written to catch

The onion-seams plan (`docs/superpowers/specs/2026-08-29-onion-seams-design.md`)
named, for each seam, the DB-free test that would prove the cut and the
mutation that test should kill: `avg_ranks` → `ranks` in `rank_rows`
(the tie-averaged no-op), `collapse_to_last` keyed on `(metric, split)`
instead of the name (the live worst case that collapsed nothing), and
`RELEVANCE_ANCHOR` 3 → 1 (the anchor width three rounds of live tuning
settled on). All three tests were written, went RED before the seam and
GREEN after, passed task review, and passed the full suite. The final
whole-branch review ran the three mutants with `scripts/mutate.py` and
**all three tests stayed green.** Each test had been built from the
spec's *example* — a tidy input on which the mutant and the real code
agree — rather than from the *failure* the mutant reproduces.

The fix was mechanical once named (an input where tied preferences make
`ranks` and `avg_ranks` disagree; two rows sharing a metric name with
different `split`s; a fourth similarity that the anchor-of-three keeps and
the anchor-of-one drops), and each mutant now fails with a concrete
assertion. What is worth recording is the sequence: RED → GREEN → task
review → suite green is **not** evidence a test bites, because none of
those steps applies the mutation. Section 4's rule stands and gains a
corollary: when a spec names the mutant a test exists to kill, the task
that writes the test runs that mutant before commit, and the report
carries its red output — the same discipline `scripts/mutate.py` was
written for, applied at authoring time rather than at final review.
