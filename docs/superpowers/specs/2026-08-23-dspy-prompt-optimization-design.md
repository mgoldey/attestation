# DSPy prompt optimization, with transfer as the acceptance test

**Date:** 2026-08-23
**Status:** proposed
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
- Whether a hosted model (Claude, GPT) belongs in the transfer matrix. It would
  strengthen the transfer claim and it contradicts the offline guarantee for
  anyone running the optimizer — probably opt-in behind the same flag shape as
  `ATTEST_CITATION_WEB`.
