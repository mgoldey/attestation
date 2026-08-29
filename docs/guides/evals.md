# Prompt evals and the optimizer

How are the prompts measured? Every model-driven prompt is scored against
its own labelled corpus, not tuned by taste, and the gate for accepting a
new prompt is transfer across model families rather than a single score.

## Prompt evals and the optimizer

Every model-driven prompt is scored against its own labelled corpus, not
tuned by taste: tagging (`evals/tagging_cases.json`, 51 cases), reaction
(`evals/reaction_cases.json`, 100 cases) and explanation
(`evals/explanation_cases.json`, 40 cases). Only tagging has an optimizer
today (DSPy GEPA, `uv run --group optimize python
evals/optimize_tagging.py`) — reaction and explanation have the corpus and
the scorer but no optimizer yet. See `examples/prompt-evals/` for the
eval → transfer-gate golden path: the commands, a live run's output, and
why the gate's bar is transfer across model families rather than a single
score. That gate is genuinely sample-sensitive: a `repeat=2` run recorded
2026-08-27 passed it, and a `repeat=1` re-run on 2026-08-28
(`evals/prompts/transfer-2026-08-28.md`) failed it on the same prompt and
cases — sampling variance, not a regression, and the reason the gate's
record is a committed, dated artifact rather than a claim in prose.
