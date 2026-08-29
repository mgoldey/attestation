# Experiment-tracker adapters

**Date:** 2026-08-22
**Status:** implemented 2026-08-23 in `984cc20`. Deviations: none. The
MLflow open question resolved as written -- final values, not curves.
Neither reader had run against a real directory until 2026-08-28, when
`examples/flows/training/train_mlflow.py` produced a real `mlruns/`
(mlflow-skinny 3.x) that the reader scanned successfully -- run_name landed
in meta.yaml as documented. The W&B reader met a real directory the same
day (`examples/wandb/wandb`, wandb 0.17.6 via `generate.py`): its run
directory is named `offline-run-<timestamp>-<id>`, not `run-<timestamp>-
<id>` as this spec assumed, but `_wandb_runs` never filtered on that prefix
so both already worked and no reader code changed. The larger finding was
upstream of naming -- see the "Verification" section below, rewritten in
light of it, and the module docstring in `ledger_adapters/generic.py`. A
third convention, Sacred, was added 2026-08-28 as part of the golden-paths
work (`2026-08-28-golden-paths-design.md`) -- see the "Sacred" subsection
below. A fourth, DVC, followed the same day -- see the "DVC" subsection. A
fifth, Hydra, followed the same day -- see the "Hydra" subsection.
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

### Sacred (added 2026-08-28)

```
sacred_runs/
  1/                          run number, starting at 1
    run.json                  experiment.name, status, start_time, stop_time, result
    config.json               the resolved config for this run
    metrics.json              {name: {"steps": [...], "values": [...]}}
    cout.txt                  captured stdout/stderr (not read; not committed)
  _sources/                   hashed copies of the driver script (not read; not committed)
```

Verified 2026-08-28 against a real directory (sacred 0.8.7,
`examples/sacred/generate.py`) rather than transcribed from documentation --
`_sacred_runs` in `ledger_adapters/generic.py` was written and tested
against the real layout from the start, not retrofitted the way the W&B
reader's docstring was. The layout matched what `FileStorageObserver`'s own
source describes with no surprises: one numbered directory per run, no
"offline" mode to opt into (everything stays on disk unconditionally, so
there was no W&B-shaped gap to find), and `experiment.name` written straight
to `run.json` rather than needing a filename-derived fallback the way W&B's
family did.

Two decisions, both extending rules already made for W&B and MLflow rather
than adding new ones:

- **`metrics.json`'s series collapse to a final value and its last step**,
  the same decision `_mlflow_metric` made for MLflow's metric-per-file log
  and for the same reason: `Metric` carries one value, and recording every
  logged point would flood `run_metrics`.
- **`run.json`'s own `result` field becomes a metric named `result`, read
  separately from `metrics.json`.** Sacred is the first of the three
  trackers where a run's headline number can live somewhere `_run.log_scalar`
  never touches: `@ex.main`'s return value is recorded as `result` on
  `run.json` regardless of whether the driver script logs anything at all.
  Skipping it would mean a driver that only returns a value and never calls
  `log_scalar` scans to zero metrics and is dropped as an unmeasured spec.
  Only a numeric `result` is recorded; Sacred allows any JSON-serialisable
  return value, and a dict or string is silently not a metric, the same
  refusal every other shape in this module gets.
- **Only `status == "COMPLETED"` runs are recorded.** Sacred writes
  `run.json` for a crashed run too (`FAILED`, `INTERRUPTED`), and reading it
  the same way as a finished run would misreport a crash as a measurement --
  the same rule MLflow's `lifecycle_stage: deleted` check serves, applied to
  a different failure.

`ledger.ADAPTER_CAVEATS` gained a `"sacred"` entry alongside `"wandb"` and
`"mlflow"`, stating the same final-value-not-curve limitation; without it
`runs.compare` would rank Sacred arms silently while carrying the caveat for
every other tracker.

### DVC (added 2026-08-28)

```
dvc.yaml         declares stages; a foreach stage expands over params.yaml
params.yaml      the values a foreach stage (and ordinary params:) reads
dvc.lock         written by `dvc repro`: cmd, recorded params, output hashes
metrics/*.json   the metric files a stage's `metrics:` list points at
```

Verified 2026-08-28 against a real `dvc repro` (dvc 3.67.1,
`examples/dvc/generate.sh`) rather than transcribed from documentation --
`_dvc_runs` in `ledger_adapters/generic.py` was written and tested against
the real layout from the start. Running `dvc repro` on a `foreach: ${lr}`
stage over `params.yaml`'s `lr: [0.01, 0.1, 1, 10]` produced four stage
instances named `train@0.01`, `train@0.1`, `train@1`, `train@10` in
`dvc.lock` -- exactly the documented `foreach` expansion, with one genuine
surprise found only by running it (below).

**No dependency on the `dvc` package, or on PyYAML.** `dvc.yaml`,
`params.yaml` and `dvc.lock` are all read by a small, hand-rolled
indentation parser (`_indented_lines` and the stage/params helpers built on
it), the same reasoning `_yaml_scalars` and `_config_shape` already state
for `meta.yaml` and TOML configs (see `generic.py`'s own comment on the
`networkx` lesson). `PyYAML` is on this repo's disk only as a transitive
dependency of dev tools (`pre-commit`, `bandit`) -- never a direct one --
and reaching for it here would be exactly that mistake repeated. DVC itself
is never imported or shelled out to by the reader; only `generate.sh`
(a dev-time fixture script, not shipped code) runs the real `dvc` CLI.

**The real finding: `dvc.lock` records the whole `foreach`-swept list for
its own key, not the one value each stage instance ran with.** Every
`train@<lr>` entry's `params: params.yaml:` block carries the *identical*
`lr: [0.01, 0.1, 1, 10]` list -- DVC is echoing the source value each
instance was generated from, not the item it actually ran with. Trusting
that block verbatim would give all four arms the same useless config.
`_dvc_runs` instead reads the `foreach` param's per-instance value from the
stage-instance name itself (`train@0.1` implies `lr=0.1`), the same
"the name already carries the answer" move `_wandb_runs` makes for its
program-derived family and `_sacred_runs` makes for `experiment.name`.
Other, non-`foreach` params declared in `params:` still come from
`dvc.lock`'s recorded value directly, since those genuinely are scalars
there -- only the swept key needed the override. This was found only by
running `dvc repro` for real and reading its output; a fixture transcribed
from DVC's documentation, which does not dwell on this echo, would not have
surfaced it.

**A second, smaller finding: DVC substitutes `${item}` using the literal
text `params.yaml` wrote, not `str(float(...))`.** `examples/dvc/train.py`
originally wrote its output as `metrics/{float(argv[1])}.json`, which
produced `metrics/1.0.json` for the arm `dvc.yaml` calls `train@1` --
`dvc repro` then failed outright with `output 'metrics/1.json' does not
exist`, because the declared output path uses `params.yaml`'s own token
(`1`), not Python's `float` repr. `train.py`'s fix (write the file using
the raw argv string, not the parsed float) is the same category of bug
`_sacred_runs`'s decisions were meant to prevent: a naming convention
transcribed from documentation that the tool's actual behaviour quietly
contradicts.

**`metrics/` collides with `RESULT_DIRS`, unlike every other tracker
directory.** `wandb/`, `mlruns/` and `sacred_runs/` are directory names of
their own, invisible to the generic reader's ordinary `results/`-style
scan. DVC routinely writes its metric files into `metrics/`, one of
`RESULT_DIRS` -- so without a guard, a DVC project was scanned twice: once
by `_dvc_runs` as `train@0.1`, and again by the ordinary `metrics/` walk as
a bare `0.1`, for the same file. `discover()` now computes the set of
metric file paths `dvc.yaml`'s stages claim before the `RESULT_DIRS` walk
runs, and skips them there -- the one piece of cross-talk between a tracker
reader and the generic scan any of the four conventions has needed.

**A stage declaring no `metrics:` is not a run**, the same refusal the
generic reader gives a config file with no result attached: a `prepare` or
`preprocess` stage with only `outs:` is excluded by `_dvc_stages` before
`_dvc_runs` ever looks at it. **A plain (non-`foreach`) stage gets no
`family`** -- there is no sibling to group it with, the same reason
`_mlflow_runs` leaves `family` unset for a run with no `run_name`.

`ledger.ADAPTER_CAVEATS` gained a `"dvc"` entry alongside the other three,
naming DVC's own limitation: each metric file is a snapshot overwritten on
every `dvc repro`, not a curve -- there is no history of a prior run's
value once a stage reruns, a different shape of "final value, not curve"
than the per-line logs W&B, MLflow and Sacred each read.

### Hydra (added 2026-08-28)

```
multirun/<date>/<time>/          one sweep, e.g. 2026-08-28/23-36-10
  <n>/                           one arm, numbered from 0
    .hydra/
      config.yaml                the resolved config this arm ran with
      hydra.yaml                 Hydra's own job/sweep/runtime metadata
      overrides.yaml             the command-line overrides for this arm
    metrics.json                 whatever the driver script itself writes
    train.log                    Hydra's own per-job log (not read; not committed)
  multirun.yaml                  one sweep-level summary (not read; not committed)
```

Verified 2026-08-28 against a real `--multirun` sweep (hydra-core 1.3.5,
`examples/hydra/generate.sh`) rather than transcribed from documentation --
`_hydra_runs` in `ledger_adapters/generic.py` was written and tested
against the real layout from the start. Running `python train.py
--multirun lr=0.01,0.1,1,10 hydra.job.chdir=True` produced exactly the
documented `multirun/<date>/<time>/<n>/` layout, with one genuine surprise
found only by running it (below).

**The real finding: `hydra.job.chdir` is not the default a Hydra <1.2 user
remembers, and the golden-paths brief's own premise ("Hydra changes cwd
per job") needed correcting against the real tool.** hydra-core 1.3.5 does
not change the working directory into each arm's own output directory
unless `hydra.job.chdir=True` is passed explicitly -- Hydra 1.1 and
earlier always changed directory; 1.2 introduced the setting and defaults
it to `null` (behaving as `False`) for configs that read relative paths
from the launch directory. Without the override, a real run of `--multirun
lr=0.01,0.1,1,10` wrote a single top-level `metrics.json`, overwritten by
each of the four arms in turn, not four separate ones under `multirun/`.
`generate.sh` passes `hydra.job.chdir=True` explicitly, and `_hydra_runs`
reads only the layout that override produces -- the same category of
"a naming convention transcribed from documentation that the tool's actual
behaviour quietly contradicts" `_sacred_runs`'s decisions and the DVC
`${item}` finding above were each meant to prevent, found the same way:
by running the real tool, not by reading its docs.

**Naming drops the date and time, the same trade `_wandb_runs` makes for
W&B's timestamp-and-hash run directory.** `<job.name>/<date>/<time>/<n>`
is unreadable in a ledger listing several arms, so a run is named
`<job.name>/<n>` instead, with `job.name` (Hydra's own `hydra.job.name`,
defaulting to the driver script's stem) as the family. A second sweep of
the same job name is not silently dropped: if `<job.name>/<n>` is already
taken, `_hydra_runs` re-qualifies with the time directory
(`<job.name>/<time>/<n>`) before giving up -- `seen` is shared with every
other reader in the module, the same dedup `_wandb_runs`/`_mlflow_runs`/
`_sacred_runs`/`_dvc_runs` already participate in.

**A nested YAML key needed a small generalisation of `_yaml_scalars`, not
a new parser.** `.hydra/hydra.yaml`'s `hydra.job.name`, `hydra.overrides.
task` and `hydra.sweep.dir` all sit several levels deep, unlike the flat
top-level keys `_yaml_scalars` already reads for `meta.yaml`/`config.yaml`.
`_yaml_path_index`/`_yaml_path_scalar`/`_yaml_path_list`, built on the
existing `_indented_lines`, walk a dotted path of nested keys instead --
the same reuse `_dvc_stages`/`_dvc_lock_params` already make of
`_indented_lines` for DVC's nested `stages:` shape, extended rather than
forked, per the golden-paths brief's own instruction not to write a second
parser. One shape difference from DVC's own YAML needed a real fix, found
by running a real sweep rather than guessing from `dvc.yaml`'s style:
Hydra's dumper writes a block list's `- item` lines at the *same* indent
as the key introducing them (`task:` and `- lr=0.01` are both at indent
4), not one level deeper as DVC's writer does, so `_yaml_path_list` scans
forward while a line is a list item rather than requiring a deeper indent.

**Metrics come from any JSON/CSV file in an arm's directory, not a
Hydra-specific format**, reusing `metrics_from_payload`/`_csv_rows` the
same way the ordinary `results/` scan does -- Hydra itself has no metrics
convention of its own; whatever the driver script writes into its own
`os.getcwd()` (this example's `metrics.json`) is the record. An arm with
no such file is skipped, the same refusal an MLflow run with no metric
files or a DVC stage instance with no metric file on disk gets. A missing
`.hydra/hydra.yaml` (an older Hydra version, or a directory edited by
hand) falls back to naming the family after the sweep directory itself
rather than raising.

`ledger.ADAPTER_CAVEATS` gained a `"hydra"` entry alongside the other
four, naming the same "final value, not curve" limitation every tracker
convention here carries, since `_hydra_runs` reads whatever a driver
script wrote as one snapshot rather than a logged series.

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

### 2026-08-28 update: both readers verified, and the real gap was not the one guessed

Both trackers were finally run for real (`examples/flows/training/
train_mlflow.py` for MLflow, `examples/wandb/generate.py` for W&B). MLflow
matched the documented layout exactly. W&B did not, but not in the way this
spec's "Verification" section above anticipated:

- The run directory is named `offline-run-<timestamp>-<id>`, not
  `run-<timestamp>-<id>`. This looked, before checking, like the kind of gap
  mitigation 2 above was written for. It was not one: `_wandb_runs` walks
  every child of `wandb/` with no name filter, so both names already worked.
  Only the docstring's claim was too narrow; no reader code changed.
- The real gap: **offline W&B does not write `wandb-summary.json` or
  `config.yaml` to `files/` at all.** Every logged value reaches disk, but
  only inside the run's binary `.wandb` transaction log; the plain files
  this reader was written to read exist only after `wandb sync` uploads to
  a real server. This is not specific to this repo's reader -- it is
  documented, known upstream behaviour (wandb's own issue tracker, #7227
  and #9646; a maintainer's answer on #1768 confirms there is no local API
  for it). `generate.py`'s module docstring has the full account and the
  local decode step (`wandb.sdk.internal.datastore`, the community's own
  workaround) that makes the committed fixture real data.
- A second, smaller gap: `_wandb_runs` groups arms by the training script's
  filename (`wandb-metadata.json`'s `program`), because that is the only
  run-identity field committed to any of the three files it reads. W&B's
  own `run.name` is never written to a local file in offline mode.

The lesson generalizes past this spec: a fixture "transcribed from
documentation" and a bug list guessed from that same documentation share a
blind spot, because both are one step removed from the tool's actual
behaviour. Mitigation 1 (transcribe from real examples) reduced the risk;
it did not remove it. Only running the library did.

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
