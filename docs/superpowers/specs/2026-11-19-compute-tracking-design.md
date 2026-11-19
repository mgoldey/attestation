# Compute tracking

**Date:** 2026-11-19
**Status:** implemented in `compute.py`, called from `ledger._compare`; adapters read run
duration; `RunRecorder` captures the environment. No new tool, no migration, no new table.
**Depends on:** the run ledger (`2026-08-12-run-ledger-design.md`), replicate-aware
comparison (`2026-11-19-replicate-statistics-design.md`), the run record
(`2026-10-17-run-record-design.md`).

## Wall time is not compute

Read this before reading a number this feature prints.

What the ledger can record is **wall-clock seconds between a run's own start and stop
timestamps**. That is a fact about one run on one machine on one day. It is not compute:

- the same job takes different wall time on a laptop, an A100 and a shared node under load;
- it includes data loading, waiting on a queue, a cold cache and a sibling process;
- two arms run on different hardware are not comparable by it at all, and the ledger does
  not record hardware per run (the one place it is captured, the run record's `environment`,
  describes the machine that BUILT that record, not necessarily the one that trained);
- it says nothing about FLOPs, energy, memory, tokens or dollars. None of those is measured
  on disk by any tracker this reader supports, so none is reported.

So the feature makes exactly one modest claim: "these arms did not take about the same
time, so they may not have had the same budget". It is a prompt to check, never a measure
of cost, and the caveat says so in those words. It never ranks, adjusts or discounts an
arm.

## Problem

`runs.compare` ranks arms by a metric. A 20-epoch arm beating a 2-epoch arm is a statement
about budget, not configuration, and the only trace of the difference is a duration that the
trackers already write down and the ledger throws away: sacred's `run.json` has
`start_time`/`stop_time`, MLflow's `meta.yaml` has `start_time`/`end_time`, W&B's summary has
`_runtime` (the reader deliberately drops it, as a bookkeeping value, so it is never taken for
a measurement).

## Decisions

### Storage: a metric row, not a column

A run's duration is stored as an ordinary `run_metrics` row named `wall_time_s` (seconds, no
step, no split). No migration, no table, no column: the 24 application tables and 35
`sqlite_master` objects stay pinned. It is the name a generic JSON/CSV result already gets
when it carries an explicit `wall_time_s` key, so there is ONE rule: a duration exists only
when an artifact says one.

`runtime_s` is read as a synonym (explicit key, same meaning), preferring `wall_time_s` when a
run has both. Nothing else is read: no `time`, `elapsed`, `duration` or `_timestamp`, whose
unit and meaning are guesses.

Trackers, read where the tracker records it and nowhere else:

| Tracker | Source | Rule |
|---|---|---|
| sacred | `run.json` `start_time`, `stop_time` (naive ISO) | `stop - start`; absent or negative: none |
| mlflow | `meta.yaml` `start_time`, `end_time` (ms since epoch) | `(end - start) / 1000`; `end_time` absent, 0 or before start: none |
| wandb | `wandb-summary.json` `_runtime` (s) | read BEFORE the underscore filter drops it; non-numeric or negative: none |
| hydra | no timing file exists in a multirun arm | none (read only if a timing file is ever added) |
| dvc | no per-run timing in `dvc.yaml`/`dvc.lock` | none |

A run whose duration is not recorded has no `wall_time_s` row. It is never estimated from file
mtimes, `scanned_at` or `started` (the latter is a start, not a duration).

If a run carries several `wall_time_s` rows (a training log with one per step) the LARGEST is
the run's wall time: the series is cumulative, and the total is its end.

### Per arm

`compute.py` is pure and DB-free. Over the arms `ledger._compare` already builds:

- a **single run** arm gets `wall_time_s` when its run has one;
- a **pooled** replicate arm gets `wall_time_s` = the SUM over ALL its runs (not the five named
  in `runs`) and `wall_time_mean_s` = sum / `n_runs`. The sum is taken from the pre-pooling
  arms, so a twenty-seed arm sums twenty runs;
- if only SOME runs of a pooled arm have a duration, the arm gets NO `wall_time_s`: a sum over
  part of the runs would understate it and read as the arm's total. It counts as "compute not
  recorded", naming the arm.
- a nested arm (several arms inside one file) gets none: the file's duration belongs to the
  file, and attributing it to each arm would multiply it.

Arms without a recorded duration carry no key at all, so a family with no timings produces a
payload byte-identical to today's. `examples/workspace` is such a family and is checked.

### Caveats

Comparing like with like uses time **per run**: `wall_time_mean_s` for a pooled arm,
`wall_time_s` for a single run (a 5-seed arm costs 5x a 1-seed arm by construction, which is
not the mismatch being looked for).

- **Ratio.** When the top two ranked arms both have a duration and the longer is at least
  `COMPUTE_RATIO_WARN` (2.0) times the shorter, one caveat names both per-run times and the
  ratio, then says wall time is not compute. A ratio below 2.0 prints nothing.
- **Floor.** Nothing is said unless the longer arm took at least `MIN_WALL_S` (1.0 s). The
  tracker fixtures committed here run for 24-155 milliseconds, where "twice as long" is
  scheduler jitter and the warning would be a false alarm.
- **Not recorded.** When at least one scored arm has a duration and another does not, one
  caveat names the arms without. When NO arm has one nothing is said, so every family that
  predates this feature renders exactly as before.

The top two only, like the existing "too close to call" rule; an arm further down the table is
not what the ranking claims.

### Environment capture in run records

`RunRecorder` adds `predicate.environment`, captured when the recorder is built:

| Field | Value |
|---|---|
| `python` | `sys.version.split()[0]` and implementation |
| `argv` | `sys.argv`, at most 20 entries of 200 characters |
| `uv_lock_sha256` | sha256 of `<root>/uv.lock`, else null |
| `gpus` | `nvidia-smi --query-gpu=name --format=csv,noheader` lines, else `[]` |

The GPU probe runs only when `nvidia-smi` is on PATH, with a 5-second timeout, and any failure
(missing, nonzero exit, timeout, unreadable output) yields `[]`. It never fails the record.
The record is in the same trust class as the rest of it: the signer's account, not an
independent measurement. It also describes the process that BUILT the record, which for a spec
handed in by another tool is not necessarily the process that did the work. The existing
`git_sha`/`git_dirty` pair is unchanged and stays in `predicate.tool`.

### What is not done

Token or cost accounting for agents, energy/CO2, FLOPs, per-run hardware, a migration, a new
MCP tool, SKILL.md changes, ranking or normalising by time.

## Output

`runs.compare` arms gain `wall_time_s` (and `wall_time_mean_s` when pooled), only when known.
The CLI table gains a `wall` column only when some arm has a duration. The docstring gains two
lines.

## Tests and mutation checks

- sacred and mlflow fixtures with known timings: `wall_time_s` equals stop minus start;
- a pooled arm of 7 runs sums all 7, not 5; one untimed run drops the arm's total;
- ratio 3x warns, 1.2x does not, exactly 2.0x warns, sub-second floor silences;
- no-timing families: no new key, no new caveat, `examples/workspace` byte-identical;
- the run record verifies and contains `environment`; no `nvidia-smi` and a failing
  `nvidia-smi` both still build the record;
- mutations (measurement-lessons section 4): delete the ratio call, flip `>=` to `>`, sum only
  the named runs, and make the "not recorded" caveat unconditional; each must fail a test.

## After the ledger began reading only core records (2026-11-19 merge)

Roadmap 11 landed first and moved every tracker reader behind the core run
record, with `tests/test_tracker_names.py` forbidding a tracker name in
shipped code outside `src/attestation/ledger_adapters/`. The table above is
unchanged in behaviour; only where it lives moved:

- The per-tracker duration parsers (`iso_duration`, `epoch_ms_duration`) and
  `add_wall` live in `ledger_adapters/timing.py`; `compute.py` keeps the
  tracker-agnostic half (`seconds`, `run_wall`, `attach`, the caveats).
- **The wall time travels as a core metric**: `wall_time_s`, `unit: "s"`,
  `direction: null`. Chosen over an `ext.ledger-import` field because the
  core metric already has exactly this shape (a named value with a unit),
  because a native producer that times its own runs then writes the same
  thing and gets the same treatment, and because it reaches `run_metrics`
  through `row_from_core` like every other value -- no second path. A null
  direction keeps it out of every ranking, as before. It is still a
  `run_metrics` row; this feature adds no column (migration 016 belongs to
  the replicate spec).
- A pooled arm's runs are found through its uncapped `members` list rather
  than by re-parsing seed tokens.

The 2x caveat, its floor, the `compute not recorded` rule and every number
pinned above are unchanged.
