# Write-side skills: attestation-record and attestation-annotate

**Date:** 2026-09-01
**Status:** implemented 2026-09-01 (commits `aea5dd3..` this date); the
acceptance ran and its numbers are in `evals/prompts/write-side-2026-09-01.md`
and `docs/bundled-skills-research.md` "The write-side skills, measured" —
record 0.515 (one named failure: the direction-declaration step does not
transfer to a 2B/4B model, 0/15), annotate 0.833; seven-entry routing
accepted with the annotate/provenance entanglement recorded. Deviations
below. The design is `docs/bundled-skills-research.md` §B, written
2026-08-30 — this spec pins the implementation and the acceptance
interpretation.
**Depends on:** the bundled-skills split (accepted 2026-09-01 by the routing
measurement recorded in the research doc), the ledger conventions
(`2026-08-22-tracker-adapters-design.md`), the claim checker
(`2026-08-12-claim-checker-design.md`).

## What is being built

Two bundled skills that teach an agent to *produce* the inputs
attestation's read-only tools consume — the write-side complement to a
ledger that only reads what is already there. Their content contract is
the research doc's §B verbatim:

- **`attestation-record`** (description leads with *record*): when the
  agent runs an experiment or evaluation, it writes final values as one
  JSON per arm into a recognised results directory with a shared filename
  prefix so `family_of` groups them; the config beside it in `configs/`
  as provenance, never as a metric; a `[metric_direction]` entry for any
  metric the built-in table does not know, BEFORE the first
  `runs.compare`; a `corpora.toml` declaration when detection cannot see
  the corpus; `hydra.job.chdir=true` for Hydra sweeps. It ends with
  `runs.scan(confirm=true)` and `runs.compare` in the same session. It
  leaves files; it never instruments.
- **`attestation-annotate`** (leads with *annotate*): when the agent
  writes prose stating a result, it puts a `<!-- claim: ... -->` beside
  each decimal, adds `cite=<key>` only after `cite.lookup(key)` resolves,
  runs `runs.claims_coverage` before handing the draft back and reports
  the uncovered decimals, and treats `contradicted` as "fix the document
  or the run, and say which". Citation *style* and BibTeX are handed to
  `research-paper-writing`; web-sourced prose to `grounded-citations`.
  The description must not lead with "citation".

Both enrol in `install.SKILL_NAMES` (after the five; the ordering comment
stays true — setup first), so `attest install` syncs them with the same
disable-rename and profile semantics, unchanged.

## Contracts (existing tests, extended)

`tests/test_skill_files.py`'s rules apply to the seven: frontmatter name
matches the directory; descriptions open with DISTINCT verbs (*record*
and *annotate* join install/rank/verify/connect/derive); no undeclared
console scripts; no `hermes install`. Rules that are surface-specific
(each surface has a skill; a surface skill names only its surface's
tools; teaches its own router) stay scoped to the five — the two
write-side skills are not surface skills, and the tests must express
that by deriving the surface set from `AGENT_SURFACES`, not from
`SKILL_NAMES`. A new rule: a write-side skill may name tools from any
surface it hands off to (record names `runs.*`; annotate names `runs.*`
and `cite.*`), but every tool it names must exist on the live surface
(the existing no-phantom-tools rule already covers this if applied to
all seven). `tests/test_install_skills.py`'s counts follow (seven
synced).

## Acceptance (the research doc's, interpreted)

"On a fresh sandbox project, an agent given the skill and asked to run a
two-arm comparison (B1) or to write a results paragraph from
`runs.detail` output (B2) produces artifacts that `runs.scan` reads to
the right number of runs, and claims that `runs.claims_check` returns as
`supported` — scored over ≥10 trials."

Interpretation, honest about what is measured: the repo has no generic
tool-calling agent loop, and building one for this eval would measure the
loop, not the skill. What the skill actually contributes is the *content*
the model emits when the skill body is in context. So:

- `evals/run_record_eval.py --live`: per trial, the model
  (`gemma4:e2b-it-q4_K_M`, temp 0, `think: false`) receives the
  `attestation-record` SKILL.md body plus a scenario ("you just ran
  <family> with arms <a>,<b>; the final metrics are <m>=<v1>,<v2>; the
  corpus was <c>") and must answer with a JSON manifest of files to
  write (path → content). The harness writes them into a sandbox
  workspace, then runs the REAL `ledger.scan` and `compare` against it.
  Scored deterministically: scan finds exactly 2 runs, one family,
  `compare` names the right winner with no direction refusal, config
  present under `configs/` and not read as a metric. ≥10 trials from
  ≥10 distinct scenarios (metric names beyond the built-in direction
  table included, so the `[metric_direction]` step is exercised).
- `evals/run_annotate_eval.py --live`: per trial, the model receives the
  `attestation-annotate` SKILL.md body plus a real `runs.detail`-shaped
  payload and must produce the results paragraph. Scored by the REAL
  `claims.parse_file` + `check_claim` against a fixture ledger: every
  decimal in the paragraph covered by a claim, all claims `supported`,
  no invented `cite=` key. ≥10 scenarios.
- Both scripts follow `run_tagging_eval.py`'s shape: `--offline` runs
  the scorer against committed fixture answers (so CI exercises the
  scorer with no model); `--live` is the acceptance and writes a dated
  record under `evals/prompts/` — the committed artifact is the number.
- The scorers live in `evals/record_eval.py` / `evals/annotate_eval.py`
  with `EvalResult` imported from `tagging_eval`, not copied; scorer
  unit tests run DB-free where possible (the scan step needs a tmp dir).

After both skills exist: re-run the description-collision routing
measurement with all SEVEN entries in the real index (the 2026-09-01
harness), including ≥6 record/annotate questions and the same controls;
recorded beside the five-entry numbers in the research doc. Acceptance:
the seven-entry index is not worse than the five-entry one on the 56
original questions, zero control theft, and the new questions route to
the right skill better than to any neighbour.

## Not in scope

- A tool-calling agent loop (measures the loop, not the skill).
- Any new MCP tool or CLI command; both skills teach existing surfaces.
- The C/D items the research doc folds or refuses (onboarding inside
  the feed skill; measurement-discipline skill; trackers; paper-writing).

## Success criteria

- Seven skills sync; all `test_skill_files.py` and `test_install_skills.py`
  contracts green; docs/counts follow (`CLAUDE.md`, `docs/guides/agents.md`).
- Both eval scorers are exercised offline in CI; the live acceptance runs
  ≥10 trials each and its dated record is committed with the numbers as
  they came out.
- The seven-entry routing re-measurement is recorded in the research doc
  with the same honesty rules as the five-entry one.

## Deviations and findings

- **Three samples per scenario, not one.** Temperature 0 on the local
  Ollama server is not deterministic (the same prompt produced different
  manifests across samples), so the drivers gained `--repeat N` and the
  record reports k/N per scenario; "≥10 trials" is scenarios × samples.
- **Raw answers are kept.** The first live run could not be root-caused
  because no answer was saved; the drivers now write a sidecar
  (`write-side-<date>.answers.json`) beside the dated record. Reading it
  is what separated "the model never writes the direction file" from
  "the sandbox dropped it".
- **A sandbox sentence in the record prompt.** The skill correctly says
  `~/.hermes/metric_direction.toml`; the sandbox cannot accept an absolute
  path, so the prompt says to write `metric_direction.toml` at the project
  root and the scorer points `LEDGER_METRIC_DIRECTION_FILE` at it. Harness
  alignment, stated in the driver.
- **Two skill gaps found by the eval and fixed:** the exact-stem rule for
  config files (a mismatched stem is an unevaluated run, by ledger design)
  and "nothing else numeric in a results directory". The direction-step
  finding is recorded as a recommendation (a deterministic `attest record`
  command), not built here.
- **The annotate and provenance descriptions were sharpened** (writing vs
  a manuscript you are handed) after the seven-entry routing run showed
  4/7 annotate questions going to provenance; the edit moved one question
  and is kept for accuracy, not as a fix.
- **The acceptance bar.** The research doc asked for artifacts scan reads
  and claims that come back supported, over ≥10 trials; it set no pass
  fraction. The numbers are committed as they came out. Record's file-shape
  steps pass at ≥0.91; its declaration step at 0/15 on small models is the
  finding that matters, and it is a design finding about procedures versus
  tools, not a wording gap.
