# Task Corpora Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A labelled corpus, a model-free scorer and a run script for each of the three model-driven tasks (tagging exists; add reactions and explanations), each rendering through the production prompt builder, stored under `evals/`.

**Architecture:** `evals/reaction_eval.py` and `evals/explanation_eval.py` mirror `evals/tagging_eval.py` (`load_cases`, `score_one`, `evaluate`, `EvalResult`, `gate` imported from tagging_eval). `simulate.reaction_messages` and `explain.explanation_messages` become the public renderers their modules call. `persona_eval.score_verdicts` moves to `reaction_eval.py`. Corpora are JSON lists with `id`/`split`/`note` plus task fields.

**Tech Stack:** Python ≥3.12; scikit-learn (already a dependency) for AUC; the flows' `stub_openai.py` for offline runs. No dspy anywhere new.

**Spec:** `docs/superpowers/specs/2026-08-28-task-corpora-design.md`

## Global Constraints

- Nothing under `src/` imports dspy/mlflow/etc. (`tests/test_tag_prompt.py` guard); nothing new under `evals/` imports dspy except `optimize_tagging.py`.
- Prompt wording in `src/attestation/simulate.py` and `explain.py` is NOT changed — only lifted into a public function; existing tests must pass unchanged.
- Corpus files: JSON list; every case has unique `id`, `split` in `("train","dev")`, non-empty `note`; ≥40 cases; both classes present in each split; reaction `dev` holds every `bait-*` and `adjacent-*` case; explanation `dev` holds ≥ half the refusals.
- Scorer checks map only to wording the prompts already mandate (see spec); errors are prose.
- Line length 100; ruff `E,F,W,I,BLE,RUF100`; no `# noqa: BLE001` under `src/`; new test files in CLAUDE.md's docs index; `git add` before the gate; commit by pathspec.
- Run scripts print, for reaction: overall score, precision, recall, AUC (signed confidence; `n/a` when constant), confidence histogram, median latency; for explanation: overall score, refusal precision/recall (refused vs should-refuse), median latency. Every line naming a number names the model.

---

### Task 1: Reaction corpus, eval and renderer

**Files:**
- Create: `evals/reaction_cases.json`, `evals/reaction_eval.py`, `evals/run_reaction_eval.py`
- Modify: `src/attestation/simulate.py` (rename `_prompt` → public `reaction_messages`, keep `_prompt = reaction_messages` for one release), `examples/flows/persona_eval.py` (import `score_verdicts` from `evals/reaction_eval.py` via `_common`-style path loading — or, simpler and allowed: keep a one-line wrapper `score_verdicts = _reaction_eval().score_verdicts`), `CLAUDE.md`
- Test: `tests/test_reaction_eval.py`

**Interfaces:**
- Produces: `reaction_eval.CASES_PATH`, `load_cases(path=CASES_PATH, split=None)`, `score_one(case, out) -> {"id","score","errors","verdict","confidence"}`, `score_verdicts(reactions, labels) -> dict` (moved verbatim from `persona_eval.py`, same keys), `evaluate(chat_json, cases, *, repeat=1, on_case=None) -> tagging_eval.EvalResult`, `dspy_fields() -> (("persona","interests","title","summary"), ("reasoning","verdict","confidence"))`, `to_dspy_example(case) -> dict`.

- [ ] **Step 1: Failing tests** — `tests/test_reaction_eval.py`: corpus shape (the global-constraint rules, including the `dev`-holds-`bait-*`/`adjacent-*` rule); `score_one` on hand cases (correct verdict with reasoning naming a title word → 1.0; wrong verdict → verdict check fails; empty reasoning → reasoning check fails; invalid payload → 0 with a validation error in prose); `score_verdicts` on the same hand-computed matrix `tests/test_flows_scoring.py` uses (import the module by path); the renderer mutation test: `monkeypatch.setattr(simulate, "reaction_messages", lambda *a: [{"role":"user","content":"MUTATED"}])` then `evaluate(fake_chat, cases[:1])` where `fake_chat` records its messages → the recorded content is `MUTATED` (and `simulate.react_to_item` renders through the same function: call it with the mutation and assert `fake_chat` saw `MUTATED`).
- [ ] **Step 2: `simulate.reaction_messages`** — rename, docstring says it is the one renderer; `react_to_item` calls it; `_prompt = reaction_messages` alias with a comment.
- [ ] **Step 3: The corpus** — generate the 80 fixture-derived cases with a small throwaway script from `examples/flows/corpus/{labelled.xml,labels.json,personas.toml}` (ids `flows-<guid-tail>-<persona>`, note "from examples/flows fixture item <guid>"), then hand-write 20 hard cases per the spec's four families with their own notes. Assign splits: all `bait-*`/`adjacent-*` → dev; of the fixture cases, alternate train/dev by item so both personas' cases for one item share a split. Commit the JSON, not the script.
- [ ] **Step 4: `reaction_eval.py` and `run_reaction_eval.py`** — mirror `tagging_eval.py`/`run_tagging_eval.py` (read both first); `--offline` starts `examples/flows/stub_openai.py` by path; print the metrics listed in the constraints.
- [ ] **Step 5: `persona_eval.py` uses the moved `score_verdicts`** — one definition; `tests/test_flows_scoring.py` still passes.
- [ ] **Step 6:** run offline, then live once (`nvidia-smi` first; ~100 calls ≈ 5 min); paste both outputs in the report. Gates, commit.

---

### Task 2: Explanation corpus, eval and renderer

**Files:**
- Create: `evals/explanation_cases.json`, `evals/explanation_eval.py`, `evals/run_explanation_eval.py`
- Modify: `src/attestation/explain.py` (lift the messages list in `generate_explanation` into module-level `explanation_messages(profile, title, summary) -> list[dict]`; the closure calls it; wording unchanged byte for byte — the comments move with it), `CLAUDE.md`
- Test: `tests/test_explanation_eval.py`

**Interfaces:**
- Produces: `explanation_eval.{CASES_PATH, load_cases, score_one, evaluate, dspy_fields, to_dspy_example}`; `REFUSAL = "Outside your stated interests."` (must equal the string in `explain.py` — a test asserts it by reading `explanation_messages("x","y","z")[0]["content"]`).

- [ ] **Step 1: Failing tests** — corpus shape (rules incl. `dev` ≥ half the refusals); `score_one` hand cases: refusal case + exact refusal → 1.0; refusal case + a manufactured connection → refuse check fails ("manufactured a connection"); topic case mentioning a `must_mention_any` word, ≤15 words, contains "you", no preamble → 1.0; 26-word answer opening "You will find" → two checks fail; the renderer mutation test as in Task 1 against `explain.explanation_messages` (and that `explain.explain_item`/the graph renders through it — read `explain.py` for the public entry point and drive it with a fake `chat_fn` over a tmp DB, seeded via `tests/conftest.py`'s helpers).
- [ ] **Step 2: `explain.explanation_messages`** — lift; existing `tests/test_explain.py` unchanged and green.
- [ ] **Step 3: The corpus** — ~40 cases: 30 shared-topic (from the flows fixture items paired with the persona whose label is true; `must_mention_any` = 3–5 words from the item's title/summary that a one-line explanation would name), 10 refusals (fixture items labelled false for both personas paired with each persona's interests, plus two hand-written termite-style cases); notes name the source item and the family (`topic-*`, `refuse-*`, `bait-refuse-*`).
- [ ] **Step 4: eval + run script**, mirroring Task 1's; refusal precision/recall printed.
- [ ] **Step 5:** offline run, live run once (~40 calls ≈ 1 min), gates, commit.

---

### Task 3: Docs and the guard

**Files:** Modify `README.md` (the "Prompt evals and the optimizer" section: one paragraph that there are three corpora and three evals, commands for all three), `CLAUDE.md` (the tagging-prompt line becomes three-task: names the two new renderers, corpora sizes, and that `score_verdicts` lives in `reaction_eval.py`), `examples/prompt-evals/README.md` if it exists by then (add the two commands to *Run it* and `run.sh`), `tests/test_tag_prompt.py` (the "no dspy under src" guard also asserts no file under `evals/` except `optimize_tagging.py` imports dspy), the DSPy spec's open question (a dated line: corpora now exist; optimisers remain future work).

- [ ] Steps: read the current README section; edit; run the full gate; commit.

---

## Self-review

Spec coverage: corpora (T1 step 3, T2 step 3), scorers with prose errors (T1/T2 step 1/4), renderers public + mutation tests (T1 step 2, T2 step 2), `score_verdicts` single definition (T1 step 5), DSPy readiness helpers (interfaces), docs (T3), guard (T3). No placeholders: each step names the file, the function, the check, and the expected outcome; corpus content is specified by source, family, size and split rule.
