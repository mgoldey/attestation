# Experiment-tracker adapters

**Date:** 2026-08-22
**Status:** implemented 2026-08-23 in `984cc20`. Deviations: none. The
MLflow open question resolved as written -- final values, not curves.
Neither reader has yet run against a real directory; see the module
docstring in `ledger_adapters/generic.py`.
**Roadmap:** spec 3 of `2026-08-21-architecture-roadmap.md`
**Depends on:** nothing. The adapter seam already exists.

## Problem

A researcher who already uses Weights & Biases or MLflow has runs on disk that
this ledger cannot see. `generic.discover(root)` walks two fixed lists —
`RESULT_DIRS` (`results`, `logs`, `outputs`, `metrics`, `eval`, `evals`,
`benchmarks`, `reports`) and `CONFIG_DIRS` — at the project root. `wandb/` and
`mlruns/` are in neither, so a project whose entire experimental record lives in
one of them scans to zero runs and gives no indication why.

The cost is exactly the adoption cost `ledger.py` names as its design
constraint. Someone with 200 W&B runs is told the ledger "requires no change to
how anything is run" and then gets nothing back.

## Scope

Teach `generic` two more conventions. **No named adapters.**

`ledger_adapters/__init__.py` states the rule: *"Prefer teaching `generic` a
new convention over adding one — a convention helps every project, a named
adapter helps exactly one."* `wandb/` and `mlruns/` are conventions in the
strict sense: every project using those tools produces that layout, and the
directory name is fixed by the tool, not chosen by the user. They belong in
`generic` on its own terms.

### W&B local run directories

```
wandb/
  run-20260814_101133-a1b2c3d4/
    files/
      wandb-summary.json     final value of every logged metric
      config.yaml            hyperparameters, with a `desc`/`value` wrapper
      wandb-metadata.json    program, args, git commit, start time
```

- `wandb-summary.json` is a flat object of scalars — already exactly what
  `metrics_from_payload` handles. The mapping is nearly free.
- `config.yaml` wraps each entry as `{value: ..., desc: ...}` and injects a
  `_wandb` key. Unwrap to `value`; drop keys starting with `_`.
- Run **name** comes from `wandb-metadata.json`'s program plus the run id, not
  from the directory name alone: `run-20260814_101133-a1b2c3d4` names a
  timestamp and a hash, and a ledger listing forty of those is unreadable.
- `started` comes from the metadata timestamp, which the current adapter has no
  other way to learn.

### MLflow run directories

```
mlruns/
  0/                          experiment id
    <run_id>/
      meta.yaml               name, status, start_time (epoch ms)
      params/<name>           one file, one line, the value
      metrics/<name>          one file, N lines: "<timestamp> <value> <step>"
      tags/<name>             one file, one line
```

**This is the layout the roadmap flagged as possibly defeating the
conventions. It does not, but it needs a real decision.** The metric-per-file
format is genuinely unlike anything `generic` reads: not JSON, not CSV, one
file per metric with one line per logged step.

The decision: **read the last line of each metric file, and record the step.**
`Metric` already carries `step`, and `run_metrics` already stores it. The last
line is the final value, which is what `wandb-summary.json` gives for W&B and
what every other artifact this adapter reads gives. Recording the whole history
would make MLflow runs structurally unlike every other run in the ledger and
would flood `run_metrics` — a 200-epoch run becomes 200 rows per metric.

The consequence is stated rather than hidden: **this adapter reads final
values, not curves.** A user who wants training curves is not served by this
ledger, and should be told so in the docstring rather than discovering it.

`meta.yaml`'s `lifecycle_stage: deleted` means the run was deleted in the
MLflow UI. Skip those: resurrecting a run the user deleted is worse than
missing it.

## What this does not do

**No network calls to W&B or MLflow servers.** No `wandb.Api()`, no MLflow
tracking-server HTTP. Local artifact directories only.

**No writing back.** `ledger.py` opens by declaring itself "deliberately NOT an
experiment tracker… requires no change to how anything is run." Reading local
directories honours that; writing runs into a tracker inverts it. If
bidirectional sync is ever wanted it needs its own spec arguing against that
docstring, not a quiet extension of this one.

**No metric-direction inference.** W&B has a `goal` field on some metrics and
MLflow has none. It is tempting to read W&B's and use it. Do not: `ledger.py`
line 21 states *"Never rank a metric whose direction is undeclared"*, and the
`METRIC_DIRECTION` table is the single place that decision lives. Two sources
of truth for direction is how an ablation gets ranked backwards. A tracker's
`goal` field may be *suggested* to the user in a message; it may not silently
populate the table.

## Verification, and its honest limitation

**There is no W&B or MLflow directory on this machine.** `find ~ -maxdepth 4
-type d \( -name wandb -o -name mlruns \)` returns nothing. This spec is
therefore written against documented layouts, and the adapter will be tested
against synthetic fixtures built from those documents.

That is a real weakness and it must be recorded in the code, not just here.
`CLAUDE.md` names this repo's recurring failure mode as "tests that pass
against the bug they were written to catch," and a fixture written by the same
author as the parser is that failure mode with extra steps. Mitigations:

1. **Fixtures are transcribed from real published examples**, cited by URL in
   the fixture file, not invented from the prose above.
2. **The reader's docstring states it has never been run against a real
   library**, so the first person who has one knows their report is valuable.
3. **A shape-tolerance test**: unknown keys, missing optional files, and an
   empty `metrics/` directory must all degrade to fewer metrics, never to an
   exception. The parser's job is to be un-surprised.

Until someone points it at a real directory, this adapter is *plausible*, not
*verified*, and saying so is cheaper than discovering it later.

## Success criteria

- A project containing only `wandb/` scans to one run per run-directory, with
  metrics, config, and a start time.
- A project containing only `mlruns/` scans to one run per non-deleted run, with
  final metric values and their steps.
- A project containing `results/` **and** `wandb/` scans both without
  double-counting: `seen` already dedupes by name, and the new readers must
  participate in it.
- No metric direction is inferred from tracker metadata.
- `NAMED` in `ledger_adapters/__init__.py` stays empty.
- Complexity ratchet holds. `generic.py` is 554 lines and `discover` is already
  long; the two new conventions go in their own functions
  (`_wandb_runs(root)`, `_mlflow_runs(root)`) that `discover` calls, rather
  than as two more loops inside it.
