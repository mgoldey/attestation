# DSPy prompt optimization, with transfer as the acceptance test

**Date:** 2026-08-23
**Status:** implemented 2026-08-27. The optimizer ran; its output is the
default tagging prompt, after rule 1 of the gate was amended the same day —
see "What happened" at the end.
Deviations from the design: instruction-only (the demonstration pool could
not be built honestly, exactly as the spec anticipated), and a `train`/`dev`
split inside `tagging_cases.json` in place of a separate held-out file.
**Kind:** design. Names the problem, the measurement, and the refusal conditions.

## Problem

Three prompts in this repo steer a local model: `features._tag_prompt`,
`explain`'s item explanation, and `simulate._prompt`. All three are
hand-written and were tuned by reading outputs on one model
(`gemma4:e2b-it-q4_K_M`). `evals/run_tagging_eval.py` already scores the
tagging prompt against 10 labelled cases, which is more discipline than most
projects have — but nothing optimizes, and nothing checks that a prompt tuned
on one model still works on another.

That second gap is the one that matters here. This project's whole premise is
that the backend is swappable: `LLM_BASE_URL` points anywhere, and CLAUDE.md
records that zero ollama references exist in library code. A prompt that only
works on gemma4:e2b quietly breaks that promise.

## What was measured first

The current hand-written tagging prompt, same 10 cases, three model families:

| Model | Overall | Median latency |
|---|---|---|
| gemma4:e2b | 0.883 | 2.00s |
| gemma4:e4b | 0.850 | 1.84s |
| hermes3:8b | 0.792 | 2.01s |

**Spread: 0.091.** This is the number the whole design hangs on. It says the
hand-written prompt already transfers reasonably — it degrades on an unrelated
family but does not collapse. Any optimized prompt that scores higher on
gemma4:e2b and *wider* than 0.091 across families is worse for this project,
not better, however good its headline number looks.

Note the direction: the biggest model scores worst. That is evidence the
prompt is fitted to Gemma's instruction-following idiom rather than to the
task, which is exactly the failure DSPy could either fix or entrench.

## Design

### The optimizer runs offline and ships data, not code

DSPy compiles a prompt by running a model many times. That must never happen
inside `attest tag`. The boundary:

- `evals/optimize_tagging.py` — a deliberate, offline command, like
  `run_tagging_eval.py` beside it. Never imported by the library.
- Output is a **prompt artifact on disk** (`evals/prompts/tagging-<date>.json`)
  holding the instruction text and few-shot demonstrations, with the scores
  that justified it recorded alongside.
- `features._tag_prompt` gains an optional artifact path. With none, it uses
  today's hand-written prompt verbatim. The default path stays hand-written
  until an artifact beats it on the transfer test below.

`dspy` becomes an optional dev dependency, never a runtime one. A user running
`attest tag` must not need it installed, and the offline guarantee is unchanged
because optimization talks to the same local `LLM_BASE_URL` everything else does.

### Transfer is the acceptance test, not the score

An optimized prompt ships only if it clears all three:

1. **Beats the baseline on the optimizer's own model** — otherwise there is
   nothing to discuss.
2. **Beats it on at least two other model families** — currently gemma4:e4b and
   hermes3:8b, with qwen3.6-35b available as a third.
3. **Spread across families is no wider than the hand-written prompt's 0.091.**

Rule 3 is the one that makes this design different from running DSPy and
keeping the winner. A prompt scoring 0.95/0.94/0.60 has a better headline than
0.883/0.850/0.792 and is disqualified: it has been fitted to one model, and
the next backend swap silently degrades tagging quality with no error.

The output is a **transfer matrix** — prompt variant × model → score — printed
by `evals/transfer_matrix.py` and committed alongside the artifact. Whoever
reads it later sees what held and what did not.

### Why few-shot demonstrations are the risky part

DSPy's leverage is mostly in selecting demonstrations from labelled data. With
10 cases, demonstrations selected from those cases and then scored on those
same cases is a tautology — the same shape as this repo's bootstrap-label
problem, where a linear threshold on the same embedding the classifier trains
on produced a meaningless AUC.

So: demonstrations are drawn from a **held-out pool** of items tagged from the
real corpus, never from `tagging_cases.json`. If that pool cannot be built
honestly, the optimizer runs instruction-only and demonstrations are out of
scope. Better a smaller win that is real.

## Refusal conditions

This design fails and should be abandoned if:

- No optimized prompt clears all three gates after a reasonable search. That is
  a real result: it means the hand-written prompt is at the achievable frontier
  for 10 cases, and the honest response is to write more cases, not to lower
  the bar.
- The optimizer needs more than ~30 minutes of local inference per run. This is
  a single-workstation tool.
- `dspy` cannot be kept out of the runtime import path.

## Open questions

- Whether 10 cases can support optimization at all, or whether the first work
  is expanding `tagging_cases.json`. Suspicion: the latter, and the spec should
  probably be reordered around it.
- Whether `explain` and `simulate` prompts get the same treatment. Both lack
  a labelled eval set entirely, so they cannot be optimized until they have one.
- **2026-08-28:** the corpora now exist — `evals/reaction_cases.json` (100
  cases) and `evals/explanation_cases.json` (40 cases), each with a
  model-free scorer (`evals/reaction_eval.py`, `evals/explanation_eval.py`)
  and a public renderer (`simulate.reaction_messages`,
  `explain.explanation_messages`). Optimisers for either prompt
  (`optimize_reaction.py`, `optimize_explanation.py`) remain future work —
  this only makes them possible; see
  `docs/superpowers/specs/2026-08-28-task-corpora-design.md`.
- Whether a hosted model (Claude, GPT) belongs in the transfer matrix. It would
  strengthen the transfer claim and it contradicts the offline guarantee for
  anyone running the optimizer — probably opt-in behind the same flag shape as
  `ATTEST_CITATION_WEB`.

## What happened (2026-08-27)

**First the cases.** 10 could not support optimization, so the file grew to
51 (23 train / 28 dev; the original ten stay in dev so the number above
stays comparable). The new cases were drawn from the real feeds and the
personas' interests, and the largest family targets what the live corpus
actually shows: generic top-of-vocabulary tags on off-vocabulary items — a
conformal-field-theory paper tagged `deep-learning, machine-learning,
optimization, representation-learning`. The eval's vocabulary is now the
live top-40, frozen, because that failure depends on the real vocabulary's
shape. Hand-written prompt on all 51: gemma4:e2b 0.830, gemma4:e4b 0.703,
hermes3:8b 0.711.

**Then the plumbing.** `features.tag_messages` became the one renderer;
`attest tag`, the eval, the optimizer's DSPy adapter and the transfer
matrix all call it, so every score is of the prompt that ships.
`ATTEST_TAG_PROMPT` loads an artifact. `dspy` sits in its own `optimize`
group and a test asserts nothing under `src/` names it.

**Then GEPA** (student gemma4:e2b, reflection gemma4:12b, 300 metric calls,
train split only). It took 2h35m, not 30 minutes: two unrelated jobs held 9
of the 16 GB of VRAM, so both models ran partly on CPU. Train score rose
0.790 → 0.902 inside DSPy; through the production client, train 0.808 →
0.891 and **dev 0.819 → 0.824**. The instruction it wrote is 4246
characters against the hand-written 502, and it memorizes train items —
`llm-cli`/`tool-use`/`command-line-tools`, `lab-management`/`research-
culture`/`mentorship`, `algorithmic-trading`, "Chrome extension →
`web-development`" are all lifted from specific train cases.

**Then the gate** (dev split, 2 runs per case):

| prompt | gemma4:e2b | gemma4:e4b | hermes3:8b | spread |
|---|---|---|---|---|
| hand-written | 0.807 | 0.750 | 0.741 | 0.065 |
| tagging-2026-08-27 | 0.807 | 0.860 | 0.827 | 0.054 |

**Verdict: FAIL**, on rule 1 alone — it does not *beat* the baseline on the
model it was optimized for; it ties it. Rules 2 and 3 pass, and pass
convincingly: +0.110 and +0.086 on the other two models, with a narrower
spread. The artifact is committed with `shipped: false` and the gate is
unchanged. Loosening `>` to `>=` after seeing the number is the move this
spec exists to prevent.

Two things this says that the design did not predict:

1. The hypothesis was right in a way that the hand-written prompt hid. The
   biggest gains were on the models the prompt was NOT tuned on. The
   hand-written prompt is fitted to gemma4:e2b's idiom; a longer,
   rule-listing instruction that e2b cannot exploit (measurement-lessons §2:
   length is zero-sum on a 2B model) is exactly what e4b and hermes3 can.
2. A tie on the primary at 56 samples is inside sampling noise (production
   samples at the default temperature). Re-running until it passes would be
   the tautology again. The honest next step is the one the spec named:
   more cases, and a bigger dev split, before another run.

**Amendment, same day.** Reviewing the matrix, Matt's call: "transferability
is a strong signal of improvement." That is the premise of this spec, and
rule 1's strict `>` contradicted it — a prompt not worse on the primary and
better everywhere else is the opposite of the fitted-to-one-model failure
the gate exists to refuse. Rule 1 is now "not worse than the baseline on the
primary" (`tagging_eval.gate`, docstring records this). Re-derived from the
committed scores, the verdict is **PASS**; the candidate is
`features.DEFAULT_TAG_INSTRUCTION` verbatim (pinned to the artifact by a
test), and the hand-written prompt lives on as
`evals/prompts/hand-written.json`, runnable via `--artifact` or
`ATTEST_TAG_PROMPT`. The next optimizer run gates against the new default.
