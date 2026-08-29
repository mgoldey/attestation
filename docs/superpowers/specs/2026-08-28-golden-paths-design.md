# Golden paths: documented use cases on disk

**Date:** 2026-08-28
**Status:** design; implementation follows in the plan of the same date.
**Depends on:** the example flows (`2026-08-28-example-flows-design.md`), the
tracker adapters (`2026-08-22-tracker-adapters-design.md`), the citations
domain (`2026-08-22-citations-domain-design.md`), the config emitters
(`2026-08-22-config-emitters-design.md`).

## Problem

Someone arriving at this repo with a real question — "will this read my
runs?", "can it check my draft?", "how do I point it at my model server?" —
finds the answer spread across a 750-line README, `examples/README.md`, the
`evals` section, and a 700-line SKILL.md. Four things are true today:

- **Three golden paths exist and are pinned by tests**, which is this repo's
  best convention ("a documented path is a tested path"): the README
  quickstart over `examples/workspace`, the workspace walkthrough, and the
  prompt eval → optimize → transfer gate. A fourth, the example flows, landed
  today. None of them has the same shape, and there is no index that says
  which needs a model server and which runs in a second on a clean clone.
- **The install-and-connect-an-agent path is documented across 300 README
  lines** and verified only by stubs. It is the path every agent user takes.
- **No third-party integration has a worked example.** The ledger reads
  MLflow (verified today, against a real directory) and W&B (never run
  against one). Sacred, DVC and Hydra — the other layouts a researcher's
  runs actually live in — have no convention at all. The citation checker
  reads `.bib` files and Zotero, with no example of either. `LLM_BASE_URL`
  accepts any OpenAI-compatible server and the README only ever names
  Ollama.
- **Every fixture that exists was written by the code's own author** — the
  failure mode CLAUDE.md names. The one exception is `examples/flows/
  training/mlruns`, produced by the real library, and it is the only reader
  this repo can call verified.

## What a golden path is

A golden path is a directory under `examples/<name>/` that a newcomer can
run from a clean clone, with everything it needs on disk, and that the test
suite runs the same way the README says to. Concretely, every path has:

1. **`README.md` with fixed sections, in this order:** *What you get* (one
   paragraph, the outcome); *Prerequisites* (one of exactly three honest
   labels: `none — pure local computation`, `a model server at
   LLM_BASE_URL`, `network`); *Run it* (the commands, copy-pasteable, at most
   six, or one `./run.sh`); *What it prints* (the real output, abridged, from
   the last time it was run); *What it demonstrates* (which rule or caveat
   fires and why that is the point); *When it goes wrong* (the two or three
   failure modes a newcomer hits and the message each produces); *Next*
   (one link).
2. **`run.sh`**: `#!/usr/bin/env bash`, `set -euo pipefail`, a temporary
   `ATTEST_DB`, and the *same commands the README shows* — a test asserts
   that every fenced `uv run …` line in the README appears verbatim in
   `run.sh`, so the two cannot drift. Exit 0 is green.
3. **Inputs on disk**, either hand-written (and said to be) or produced by a
   committed script using the real third-party library, then scrubbed of
   attribution and machine paths (the rule `examples/flows/training` set:
   `test_the_committed_mlruns_carries_no_personal_or_machine_attribution`).
   Regeneration is a deliberate act and the script says so.
4. **A test** in `tests/test_golden_paths.py`: for a path whose prerequisite
   is `none`, run `run.sh` and pin one line of its output; for the others,
   the README/`run.sh` agreement check only (the suite never touches a
   model server). The README's *What it prints* block is the source of the
   pinned line, so the docs are what the test asserts.
5. **A row in the catalogue** — `examples/README.md` becomes an index table:
   path, one line of what it shows, prerequisite label, measured runtime —
   ordered by prerequisite, `none` first. The repo README's "Try it in 60
   seconds" stays as it is and gains one line pointing at the catalogue.

These are the best practices, stated once here and enforced by
`test_golden_paths.py` rather than remembered: fixed sections (a test checks
the headings), README ⇔ `run.sh` agreement, offline-first, scrubbed
artifacts, a catalogue row.

## Scope

### Retrofit the paths that exist (no renames)

Renaming `examples/workspace` would break the README quickstart, three tests
and SKILL.md; the shape is what matters, not the name.

- `examples/workspace/` — the ledger + claims path. README gets the fixed
  sections and a `run.sh`; its walkthrough prose is already right.
- `examples/flows/` — the demonstration suite. README gets the fixed
  sections; `run.sh` is `run_all.py --offline`.
- `examples/prompt-evals/` — a README + `run.sh` for the eval → optimize →
  transfer path, pointing at the scripts in `evals/` (which stay where they
  are: `test_tag_prompt.py` and the optimizer's group depend on the path).
  Prerequisite: a model server. No new fixture.
- `examples/agents/` — "connect an agent": `attest install --check`,
  `attest emit`, and one MCP surface driven over stdio via
  `examples/flows/mcp_e2e.py --surface provenance --offline`. Prerequisite:
  `none` (the stub). This is the path that replaces reading README §
  "Launching alongside hermes-agent" to get started; that section stays as
  the reference.

### New third-party integration paths

Each produces its artifacts with the real library (all four install
ephemerally in under five seconds: `uv run --with <lib>`), commits them
scrubbed, and turns the corresponding reader from plausible to verified —
or adds the reader, as a convention in `generic.py` per the adapter rule.

| path | library | reader | src change |
|---|---|---|---|
| `examples/wandb/` | `wandb` offline mode | exists, unverified | expected: offline runs live in `wandb/offline-run-*`, the reader globs `run-*` — a real-directory finding, fixed with a test if confirmed |
| `examples/sacred/` | `sacred` `FileStorageObserver` | none | new `_sacred_runs`: `<dir>/<id>/{config.json, metrics.json, run.json}`; `run.json` `status`/`experiment.name`; `metrics.json` final value + step |
| `examples/dvc/` | `dvc` | none | new `_dvc_runs`: `dvc.yaml`'s declared `metrics:` files plus `params.yaml`; one run per stage that declares metrics |
| `examples/hydra/` | `hydra-core` multirun | none | new `_hydra_runs`: `multirun/<date>/<time>/<n>/.hydra/config.yaml` + the metrics file the job wrote; the sweep directory is the family |
| `examples/citations/` | none (`.bib` on disk) | exists | none: a `.bib`, a draft with `cite=` claims, `cite.sources` / `cite.check` / `cite.lookup`; one key deliberately unresolvable |
| `examples/model-servers/` | none | n/a | none: "bring your own OpenAI-compatible server" — `LLM_BASE_URL` for vLLM, llama.cpp, LM Studio, documented; `run.sh` uses `examples/flows/stub_openai.py` as the runnable server |

Rules for the three new conventions, inherited from the tracker spec: final
values not curves; no metric-direction inference; no named adapters (`NAMED`
stays empty); each in its own function that `discover()` calls; a
shape-tolerance test (missing optional files degrade, never raise); the
reader's docstring names the library version that produced the committed
fixture.

### Not in scope

- TensorBoard event files (binary protobuf; not a convention `generic`
  can read without a dependency).
- Network trackers (W&B cloud, MLflow tracking servers) — the tracker spec
  forbids network reads.
- Changing any golden path's *behaviour* to make its test easier. A path
  that exposes a bug fixes the bug in its own commit with its own test.

## Execution

The framework and the retrofits are one sequence (they share
`tests/test_golden_paths.py` and `examples/README.md`). The six integration
paths are independent of one another except that three touch
`generic.py`'s `discover()`; those three run one after another, the other
three in parallel, each on its own subagent with its own brief.

## Success criteria

- `examples/README.md` lists ten paths with prerequisite labels; the
  `none` ones run green from `tests/test_golden_paths.py` on a clean clone
  in CI with no model server.
- Every path's README has the seven sections in order and its fenced
  commands appear in its `run.sh`, asserted by one test.
- `generic.py`'s W&B reader has been run against a real directory; Sacred,
  DVC and Hydra directories scan to runs with final metric values; `NAMED`
  is still empty; the complexity ratchet holds.
- No committed artifact contains a username, an absolute home path, or a
  repository URL; the flows' attribution test is generalised to every
  `examples/**` directory.
- The repo README's "Try it in 60 seconds" is unchanged except for one
  line pointing at the catalogue.
