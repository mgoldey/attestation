# Example prompt evals

## What you get

A score for the tagging prompt that ships (`attest tag`'s
`DEFAULT_TAG_INSTRUCTION`) against 28 labelled dev cases, and the transfer
gate that decides whether a candidate prompt may replace it — run across
three model families rather than the one the candidate was optimized for.
Alongside it, a score for the reaction prompt (`feed.simulate_ratings`,
63 dev cases, precision/recall/AUC and a confidence histogram) and the
explanation prompt (`feed.explain`, 12 dev cases, refusal precision/recall)
— the same model-free, production-renderer scoring, for the two tasks that
don't have an optimizer yet.

## Prerequisites

`a model server at LLM_BASE_URL`

## Run it

```bash
uv run python evals/run_tagging_eval.py --split dev
uv run python evals/transfer_matrix.py --artifact evals/prompts/tagging-2026-08-27.json
uv run python evals/run_reaction_eval.py --split dev
uv run python evals/run_explanation_eval.py --split dev
```

Relative to the repo root (`run.sh` does `cd "$(dirname "$0")/../.."`
first). All four scripts render their prompt through the production
function — `attestation.features.tag_messages` (`attest tag`),
`attestation.simulate.reaction_messages` (`feed.simulate_ratings`),
`attestation.explain.explanation_messages` (`feed.explain`) — so the number
here is the number in production, not a proxy for it. None needs
`uv run --group optimize`; that group is only for
`evals/optimize_tagging.py`, the GEPA optimizer that produced the tagging
artifact the first two scripts evaluate — reaction and explanation have no
optimizer yet, only the corpus and the scorer.

## What it prints

```
prompt=default  model=gemma4:e2b-it-q4_K_M  split=dev  cases=28  repeat=1
```

Abridged, from a real run against `gemma4:e2b-it-q4_K_M` on this machine —
`run_tagging_eval.py` prints one line per case (`ok`/`~`/`FAIL` by mean
score, the tags it produced, and any error strings):

```
  ok  paper-lm                     1.00  ['large-language-models', 'modeling', 'scaling-laws']
  ~   release-pytorch              0.67  ['pytorch-release', 'hardware', 'distributed-computing']
         - no expected topic in [...]; wanted any of ['pytorch', 'deep-learning-frameworks', ...]
  ...
  FAIL blog-willison-ai-writing     0.33  ['natural-language-processing', 'safety', 'ai-ethics']

  overall           0.827
  median latency    1.92s
  distinct tags     70
  singleton rate    84%
```

`transfer_matrix.py` runs the baseline and every `--artifact` candidate
against each of `gemma4:e2b`, `gemma4:e4b` and `hermes3:8b` in turn (Ollama
serves one model at a time, so wall-clock is roughly
prompts x models x cases x 2s, plus a cold model load per swap), prints one
row per model per prompt, then a `| prompt | model... | spread |` table and
a `## Gate` verdict per candidate, and writes both as Markdown next to the
artifact (`evals/prompts/transfer-<date>.md` + `.json`). A run captured on
this machine:

```
  gemma4:e2b     baseline                     0.812  (2.11s/case)
  gemma4:e2b     tagging-2026-08-27           0.824  (2.23s/case)
  gemma4:e4b     baseline                     0.634  (2.36s/case)
  gemma4:e4b     tagging-2026-08-27           0.899  (2.38s/case)
  hermes3:8b     baseline                     0.818  (2.75s/case)
  hermes3:8b     tagging-2026-08-27           0.798  (2.59s/case)

| prompt | gemma4:e2b | gemma4:e4b | hermes3:8b | spread |
|---|---|---|---|---|
| baseline | 0.812 | 0.634 | 0.818 | 0.185 |
| tagging-2026-08-27 | 0.824 | 0.899 | 0.798 | 0.101 |

Gate verdict:
- **tagging-2026-08-27**: FAIL
  - beats the baseline on 1 other model(s) ['gemma4:e4b']; needs 2
```

That `FAIL` is real and is the point of running this live rather than
transcribing a number: `repeat=1` against a live model is
non-deterministic — this run's `hermes3:8b` score (0.798) landed just
under the baseline's (0.818), where the run recorded in
`evals/prompts/transfer-2026-08-27.md` (checked into the repo, `--repeat`
not shown, gate `PASS`) had it the other way. Same prompt, same cases,
different sample. This is why the gate exists as a committed, dated
artifact rather than a claim in prose — and why a single dev-split run at
`repeat=1` is a demonstration of the mechanism, not a re-certification of
the shipped default.

## What it demonstrates

**Prompt quality here is not a matter of taste.** `evals/tagging_cases.json`
holds 51 cases (23 train / 28 dev — the optimizer never sees dev), each
with a `note` naming the failure it targets; the largest family is
`bait-*`, off-vocabulary items the live corpus tagged `optimization` or
`representation-learning` because the prompt said to prefer the vocabulary.

**Transfer, not score, is the bar.** This project's premise is that
`LLM_BASE_URL` points anywhere, so a prompt that wins on the model it was
tuned against and collapses on another breaks that promise silently.
`tagging_eval.gate()` requires: not worse than the baseline on the
optimizer's own model, better on at least two other model families, and no
wider a spread across models than the baseline has.

**The shipped prompt's own history is the worked example.** The candidate
in `evals/prompts/tagging-2026-08-27.json` tied the baseline on the model
it was optimized for (0.807 vs 0.807 on `gemma4:e2b`) and beat it on the
two it never saw (+0.110 on `gemma4:e4b`, +0.086 on `hermes3:8b`) with a
narrower spread in the run recorded in the repo — it passed the gate and
became `DEFAULT_TAG_INSTRUCTION` (`tests/test_tag_prompt.py` pins the two
together). The gate's first rule originally demanded a strict win on the
primary model and would have refused this candidate on the tie; it was
amended to "not worse" with that run recorded as the reason, in both the
design spec and `gate()`'s own docstring.

**`ATTEST_TAG_PROMPT`** loads any other artifact (e.g.
`evals/prompts/hand-written.json`, the original, unoptimized prompt) at
`attest tag` time; unset, `attest tag` uses `DEFAULT_TAG_INSTRUCTION`
verbatim.

## When it goes wrong

- Ollama down: `ChatClient` raises with a message naming the unreachable
  `LLM_BASE_URL` — `chat model unreachable` — before the first case runs.
- A model named on the command line (or in `transfer_matrix.py`'s
  `DEFAULT_MODELS`) that has not been pulled fails the same way, once
  Ollama reports the 404.
- `uv run --group optimize python evals/optimize_tagging.py` (not part of
  this path) fails with a missing-module error if the `optimize`
  dependency group is not installed — neither `uv sync` nor CI installs
  it, by design (`dspy` pulls ~30 packages and nothing under `src/` may
  import it).

**The other two tasks are scored the same way, without an optimizer yet.**
`run_reaction_eval.py` prints precision, recall, AUC over signed confidence
(`n/a` when confidence never varies — measured true on gemma4:e2b, which is
why the histogram prints beside the score) and a confidence histogram;
`run_explanation_eval.py` prints refusal precision/recall (refused vs.
should-refuse), because the explanation prompt's refusal clause is
load-bearing and nothing else in the repo guards it. The ten refusal cases
are two families: `refuse-other-*` (a different field entirely — marine
ecology, astronomy, macroeconomics, linguistics) and `refuse-bait-*`
(content-free AI-for-science prose that names no real topic). A missed
refusal in the second family means the clause is defeated by AI-adjacent
wording specifically — the failure `explain.py`'s refusal clause was
written against — so the two recalls are not interchangeable; the live
2026-08-28 run recorded refusal recall 0.400. `score_verdicts` is
defined once, in `evals/reaction_eval.py`, and `examples/flows/
persona_eval.py` imports it rather than keeping its own copy.

## Next

See README § "Prompt evals and the optimizer" for the optimizer itself
(`evals/optimize_tagging.py`, DSPy GEPA, `uv run --group optimize`), and the
catalogue at `examples/README.md` for the other golden paths.
