# Task corpora: a labelled set for every model-driven task

**Date:** 2026-08-28
**Status:** design; implementation follows in the plan of the same date.
**Depends on:** the DSPy prompt-optimisation design
(`2026-08-23-dspy-prompt-optimization-design.md`), the example flows
(`2026-08-28-example-flows-design.md`), `docs/measurement-lessons.md` §5.

## Problem

Three tasks in this repo ask a chat model for a schema-bound answer:

| task | module | schema | prompt builder | corpus |
|---|---|---|---|---|
| tagging | `features.py` | `ItemTags` | `tag_messages()` — public, the one renderer | `evals/tagging_cases.json`, 51 cases |
| reaction | `simulate.py` | `Reaction` | `_prompt()` — private | none |
| explanation | `explain.py` | `Explanation` | inline in a closure inside `_build_graph` | none |

The DSPy spec's open question — *"whether `explain` and `simulate` prompts
get the same treatment. Both lack a labelled eval set entirely, so they
cannot be optimized until they have one"* — is still open. The tagging
work showed the order that matters: **first the cases, then the plumbing,
then the optimizer**; ten cases could not support optimisation and the file
had to grow to fifty-one before anything else was worth doing.

Two measured facts make the missing corpora costly today:

- `explain.py`'s refusal clause is load-bearing (without it the model
  claimed a termite-feed paper matched "advanced topics like AI"), and the
  wording that fixed it was chosen over four items. Nothing guards it.
- `simulate.py`'s `confidence` was measured inert on gemma4:e2b (4 or 5
  every time) and its `strength`→`confidence` rename fixed a bug that
  discarded every negative. Nothing measures either.

## Design

### One shape, three corpora

Every corpus is a JSON list of cases with the fields the tagging corpus
already uses — `id` (unique), `split` (`train` | `dev`), `note` (the failure
the case targets, in prose) — plus task inputs and task expectations:

**`evals/reaction_cases.json`** — inputs `persona`, `interests`, `title`,
`summary`; expectation `verdict: bool`. About a hundred cases: the eighty
that `examples/flows/corpus/labels.json` already implies (forty items × two
personas, with the fixture's wording), plus twenty hand-written hard cases
in four families named in `note`: `adjacent-*` (a real field next door —
chemical engineering for the bench chemist, statistics for the ML
engineer — labelled false), `bait-*` (generic AI-for-science prose,
false), `terse-*` (a one-line summary that is nonetheless clearly on
topic, true), `crossover-*` (ML for chemistry, labelled per persona). The
`dev` split holds every `bait-*` and `adjacent-*` case so the optimiser
never sees the families it is most likely to entrench.

**`evals/explanation_cases.json`** — inputs `interests`, `title`,
`summary`; expectation `refuse: bool` and, when not refusing,
`must_mention_any: [str]` (topic words one of which a correct explanation
names). About forty cases: thirty with a shared topic and ten with none
(the termite paper's descendants), split so `dev` holds at least half the
refusals.

Reusing the flows fixture's items is deliberate: they are hand-written,
labelled, and already scrubbed; a case's `note` says which fixture item it
came from.

### Scoring is model-free and maps to real failures

`evals/reaction_eval.py` and `evals/explanation_eval.py` mirror
`tagging_eval.py`: `load_cases(path, split)`, `score_one(case, out)`
returning `{"id", "score", "errors", ...}` with prose errors (the optimiser
feeds them to its reflection model), `evaluate(chat_json, cases, *,
repeat, on_case)` returning an `EvalResult` with `overall` and
`median_latency`, and the same `gate()` as tagging — imported, not copied.

Reaction sub-checks: the parsed `Reaction` validates; `verdict` matches;
`reasoning` is non-empty and names the item (mentions a title word), so a
verdict without a reason scores lower; the run reports the confidence
histogram beside the score because an inert confidence is the known
failure. Explanation sub-checks: validates; when `refuse` the text equals
the mandated refusal (`Outside your stated interests.`) — the prompt
requires those exact words and the check is exact; when not, the text
mentions one of `must_mention_any`, is under fifteen words, addresses the
reader as `you`, and does not open with a preamble ("You will find",
"This item"). Every check is a wording the prompt already mandates; the
scorer invents no preference.

The run scripts print precision, recall and AUC for the reaction task
(verdict vs label; signed confidence vs label, `n/a` when confidence does
not vary) in addition to the mean score — the same numbers
`examples/flows/persona_eval.py` prints, computed by the same function
(`persona_eval.score_verdicts` is moved to `evals/reaction_eval.py` and
imported from there, so there is one definition).

### One renderer per task

The tagging eval is trustworthy because `tag_messages()` is called by
`attest tag`, the eval, the optimiser's adapter and the transfer matrix —
a score is always a score of the prompt that ships. The other two tasks
get the same property:

- `simulate.reaction_messages(persona, interests, title, summary)` — the
  current `_prompt`, made public; `react_to_item` calls it.
- `explain.explanation_messages(profile, title, summary)` — lifted out of
  the closure; `generate_explanation` calls it.

A test asserts each eval renders through the production function by
mutating the function's system prompt under `monkeypatch` and observing
the eval's messages change (`measurement-lessons.md` §4: a guard that
passes with the protected thing removed guards nothing).

`ATTEST_TAG_PROMPT` has no counterpart here yet — prompt artifacts for the
two new tasks are the optimiser's concern, not this spec's. What this spec
guarantees is that when an optimiser exists, its adapter has a renderer to
call and a corpus to score against.

### DSPy readiness without DSPy

Each eval module exposes `dspy_fields()` → `(input_names, output_names)`
and `to_dspy_example(case) -> dict` (inputs plus the case), so an optimiser
can build `dspy.Example(**to_dspy_example(c)).with_inputs(*inputs)` for any
task in one line. Nothing under `evals/` other than `optimize_tagging.py`
imports `dspy`, and nothing under `src/` may.

### Where the corpora live and how they are used

`evals/` beside `tagging_cases.json`. `examples/prompt-evals/` (the golden
path) documents all three: run each eval against the current prompt;
`README.md`'s prompt-evals section points there. `CLAUDE.md`'s tagging
paragraph becomes a three-task paragraph.

## Not in scope

- Optimisers for the reaction and explanation prompts (`optimize_*.py`) —
  the DSPy spec's gate applies unchanged when they come; this spec makes
  them possible.
- Profile synthesis (`explain.py`'s fallback when a persona has no
  interests text) — measured worse than the interests string and kept only
  as a fallback; not worth a corpus.
- Changing any prompt's wording. The corpora score the prompts that ship.

## Success criteria

- Three corpora in `evals/`, each ≥ 40 cases, unique ids, both splits,
  both classes per split, a `note` per case; a test pins these.
- `run_reaction_eval.py` and `run_explanation_eval.py` run against Ollama
  and print score, precision/recall/AUC (reaction), latency and the
  confidence histogram; their offline mode is the flows' stub.
- The renderer mutation test fails when an eval stops calling the
  production builder.
- `persona_eval.score_verdicts` has one definition, in `reaction_eval.py`.
- `src/` imports nothing new; `attest tag`, `feed.simulate_ratings` and
  `feed.explain` behave exactly as before (existing tests unchanged).
