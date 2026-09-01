# `attest runs record`: derive the artifacts, do not transcribe them

**Date:** 2026-09-01
**Status:** approved by instruction ("build the attest record command");
implementation follows immediately.
**Depends on:** the write-side skills and their measurement
(`2026-09-01-write-side-skills-design.md`), whose finding this answers; the
ledger conventions (`2026-08-22-tracker-adapters-design.md`).

## Why

The `attestation-record` skill teaches a five-step procedure — one JSON
per arm with a shared prefix, a config with the exact same stem, a
`[metric_direction]` entry for any metric the ledger does not know, a
`corpora.toml` when detection cannot see the corpus, then scan and
compare. Measured on 2026-09-01: small local models follow the file-shape
steps at ≥0.91 and the declaration step at **0/15**, because it is the
one step that needs an inference rather than a cue. A procedure a model
half-follows is the wrong tool; a command that writes the files
deterministically is the right one. This is the "derive, don't
transcribe" principle from the LaTeX brainstorm applied to run recording:
a number a command wrote cannot be mistyped, and a direction the command
required cannot be forgotten.

## The command

```
attest runs record FAMILY --arm NAME METRIC=VALUE [METRIC=VALUE ...] [--arm ...]
                   [--corpus NAME] [--direction METRIC=lower_is_better|higher_is_better]
                   [--config KEY=VALUE ...] [--root DIR] [--dry-run] [--force] [--scan]
```

It writes, under `--root` (default: the current directory):

- `results/<FAMILY>_<NAME>.json` per arm — `{METRIC: VALUE, ...}` — so
  `family_of` groups the arms under `FAMILY` and each file is one run;
- `configs/<FAMILY>_<NAME>.yaml` per arm — provenance only: `family`,
  `arm`, `corpus` (when given), `recorded_at` (ISO-8601 UTC), any
  `--config` pairs; never a metric value, so the ledger reads it as the
  run's config and not as a second run (exact-stem pairing);
- `corpora.toml` at the root when `--corpus` is given — `[corpus.NAME]`
  (with `source = NAME`) and `[assign.family]` `FAMILY = NAME`, merged
  into an existing file, never clobbering other entries;
- the direction file (`ledger._metric_direction_path()`, i.e.
  `LEDGER_METRIC_DIRECTION_FILE` or `~/.hermes/metric_direction.toml`) —
  every metric not in the built-in table must be given a `--direction`;
  the command **refuses** otherwise, with the same sentence
  `runs.compare` would print, and never guesses. Declared directions are
  merged into the file; an existing entry for the same metric is kept
  unless `--force`.

Rules that make writing into a workspace acceptable in a tool whose
ledger only reads:

- **New files only.** Any target that already exists is listed and the
  command refuses before writing anything, unless `--force`. Nothing else
  in the workspace is touched.
- **`--dry-run` prints the manifest** — `{"files": {relpath: content}}`,
  the same shape the record eval scores — and writes nothing. That makes
  the command its own acceptance test: `evals/run_record_eval.py
  --command` builds each scenario's argument list, runs the planner, and
  scores the manifest with the existing scorer; the acceptance is 11/11.
- **`--scan`** runs `ledger.scan(confirm)` on the root and prints
  `compare` for the family, so the run enters the ledger in the same
  command; without it, nothing is read.
- Values are numbers (`float` parse; a non-number is a refusal, not a
  string metric); metric names are validated against the same regex the
  claim grammar uses so a typo cannot create an unrankable metric.

## Shape

`src/attestation/record.py` (domain, no `sqlite3`, no `attestation.llm`):
`plan(family, arms, *, corpus=None, directions=None, config=None,
recorded_at=None, known_directions=None) -> dict[str, str]` is pure over
plain data and returns the manifest — the same function the `--dry-run`
and the eval call; `undeclared(arms, known_directions) -> list[str]` names
the metrics that need a `--direction`; `write(root, manifest, *, force)`
performs the only I/O and returns the paths written. `merge_toml_table`
for the two TOML files lives beside them with tests on literal text.
`cli.py` gains the subcommand under `runs` with a `HELP` entry and
`@_documented`; no MCP tool in this spec (the skill tells an agent to run
the command; a `runs.record` tool is a follow-up once the CLI has been
used).

## The skill follows the command

`attestation-record`'s body leads with the one call and keeps the manual
layout as the fallback for a harness that cannot run `attest` (the
exact-stem rule, the direction refusal and the corpus declaration all
stay, now as "what the command does for you"). Its size should fall. The
record eval keeps its `--live` mode for the fallback path and gains
`--command`; the dated record reports both.

## What is tested

- `tests/test_record.py`: the manifest for a two-arm scenario, byte-exact;
  `undeclared` names the unknown metric and nothing else; refusal on a
  non-numeric value and on a bad metric name; `write` refuses on an
  existing file and writes on `--force`; TOML merge keeps foreign entries
  and refuses to clobber a differing direction without force; the
  manifest scored by `record_eval.score_one` passes every check for every
  committed scenario (the acceptance, offline and deterministic).
- `tests/test_cli.py`: `attest runs record --dry-run` prints the manifest
  and creates nothing; the refusal exit code and message on an undeclared
  metric.
- `tests/test_skill_files.py`: the record skill names `attest runs record`
  (a declared console script's subcommand) and its size did not grow.
- `docs/reference/cli.md` is regenerated (the render test asserts it).

## The MCP tool (amendment, same day: "add the runs.record MCP tool too")

`runs.record` on the provenance surface, a thin wrapper over the same
`record.plan`/`undeclared`/`write`:

```
runs.record(family, arms, corpus=None, directions=None, config=None,
            root=None, project=None, confirm=False)
```

- `arms` is a list of `{"name": str, "metrics": {metric: number}}`;
  `directions` a `{metric: "lower_is_better"|"higher_is_better"}` map;
  `config` a flat `{key: value}` of provenance pairs; `root`/`project`
  resolve the way `runs.scan` resolves them (`ledger.workspace_root(root)`,
  the project directory beneath it).
- **Without `confirm=true` it writes nothing and returns the manifest** —
  the tool's own preview, the same shape as `--dry-run`; with `confirm`
  it writes (new files only; an existing target is a `ToolError` naming
  every collision, before any write) and then scans the project and
  returns `compare` for the family, so one call takes a run from numbers
  to a ranked ledger entry. `force` is deliberately NOT a tool argument:
  an agent overwriting a result file is the failure the ledger exists to
  catch; a human does that at the CLI.
- An undeclared metric is a `ToolError` carrying the same sentence
  `runs.compare` prints; the tool never guesses a direction.
- Envelope: `empty={"written": [], "manifest": {}, "compare": None}`; the
  response carries paths, not file contents, except in the preview.
- Surface counts move: 46 tools, provenance 10 (expanded); every doc that
  states a count follows, and the existing count guards enforce it. The
  runs router gains a rule ("record"/"write the results"/"leave files for
  the ledger" → `runs.record`) with routing-test cases; the record skill
  names `runs.record` as the agent path and `attest runs record` as the
  shell path. `tests/test_tool_envelope.py` and `test_response_size.py`
  cover it like every other tool.

## Not in scope

- Multi-step metric curves; the ledger records final values.
- Writing tracker layouts (W&B/MLflow/…): the command writes the bare
  convention only.
