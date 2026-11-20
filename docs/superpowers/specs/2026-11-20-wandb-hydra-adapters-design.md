# The W&B and Hydra adapters read the whole run, as MLflow's does

**Date:** 2026-11-20
**Status:** implemented 2026-11-20 on `next/wandb-hydra` (roadmap item 16).
**Depends on:** `2026-11-20-mlflow-adapter-design.md` (THE template: map the
whole run into a core record, a history reference via `write_history`, no
curves table, git → inferred code, files by sha256, caveats in
`ext.ledger-import`), `2026-11-19-ledger-reads-core-records-design.md` (the
adapter protocol), `2026-11-19-compute-tracking-design.md` (`wall_time_s`).
**Not this:** a curves table in the ledger, W&B Artifacts (they are staged in
the user's cache and uploaded on sync, never in the run directory), W&B
`requirements.txt` → `dependencies` (deferred, below), Hydra's single-run
`outputs/<date>/<time>/` layout, any wandb or hydra runtime dependency.

## Problem

W&B and Hydra still go through the legacy readers in `generic.py`
(`_wandb_runs`, `_hydra_runs`): final values only, `found_to_core` with
every field it cannot see left null. For W&B that loses almost everything the
run recorded, because the reader looks only at `files/` -- and **an offline
W&B run writes nothing useful to `files/`.** Measured 2026-11-20 against
wandb 0.17.6: `files/wandb-summary.json` is `{"_wandb": {"runtime": 0}}`,
there is no `config.yaml`, and `wandb-metadata.json` has no project, no exit
code and (once scrubbed) no git. The committed fixture's summary and config
were *materialised* by `generate.py` from the run's binary `.wandb` log and
the log then deleted. So the legacy reader's family was the script name
(`generate`), not the project; its status was unknowable; its curve was gone.

Everything W&B recorded is in that log: the run record (`run_id`, `project`,
`display_name`, `start_time`, `git`), config, every `wandb.log` row, the
summary, and the exit record (`exit_code`). Hydra is the opposite case:
everything it records is in plain YAML (`.hydra/{config,hydra,overrides}.yaml`),
and the legacy reader parsed only the flat top level of `config.yaml` with a
hand-rolled line reader and ignored `overrides.yaml` -- Hydra's own statement
of what varies between arms.

## Decision: one module per tracker, mapping the whole run, no runtime dependency

`ledger_adapters/wandb_store.py` and `ledger_adapters/hydra_store.py`, each
the shape of `mlflow_store.py`: a reader returning a `Found` subclass (so
`generic.discover` callers see the same legacy view: name, family, config,
finite final values, `wall_time_s`) carrying the tracker's own facts, and a
`to_core_record` that maps them. `trackers.WANDB`/`trackers.HYDRA` become
`trackers.StoreAdapter`s (the class MLflow's adapter now shares) calling those; their
names, markers, registry order and `run_id_suffix` are unchanged.
`generic._wandb_runs`, `_wandb_config`, `_hydra_runs`, `_hydra_job_name` and
`_hydra_config` are deleted.

### Reading the `.wandb` log without wandb

The log is a LevelDB log file (wandb's `sdk/internal/datastore.py`): a 7-byte
file header (`:W&B`, magic `0xBEE1`, version 0), 32 KiB blocks, each record a
7-byte header (CRC-32 of type byte + data, little-endian length, type
FULL/FIRST/MIDDLE/LAST) and its bytes; each assembled payload is one
protobuf `Record` (`wandb_internal.proto`). Both halves are small and
stdlib-only: the checksum is `zlib.crc32` seeded with the type byte's CRC,
exactly as wandb computes it, and the protobuf wire format needs varints and
length-delimited fields only. The adapter decodes the handful of fields it
maps, by number:

| message | fields read |
|---|---|
| `Record` | 2 history, 3 summary, 5 config, 17 run, 18 exit |
| `RunRecord` | 1 run_id, 3 project, 4 config, 5 summary, 8 display_name, 17 start_time, 21 git |
| `HistoryRecord` | 1 item, 2 step |
| `HistoryItem`/`SummaryItem`/`ConfigItem` | 1 key, 2 nested_key, 16 value_json |
| `SummaryRecord`/`ConfigRecord` | 1 update, 2 remove |
| `RunExitRecord` | 1 exit_code |
| `GitRepoRecord` | 1 remote_url, 2 commit |
| `Timestamp` | 1 seconds, 2 nanos |

Every other field and record type is skipped by wire type, which is what
protobuf itself does with an unknown field. **Why these numbers are stable
enough to depend on:** `wandb sync` must upload an offline directory written
by an older client, so the log format and these field numbers are bound by
wandb's own compatibility promise, not by an implementation detail. They are
pinned by a test against wandb's descriptors when wandb is importable
(skipped otherwise; it is not a dev dependency), and the decoder is
cross-checked against wandb's own `DataStore` + `Record` parser on the
committed logs in the same test (run 2026-11-20 with `uv run --with
wandb==0.17.6 pytest tests/test_wandb_hydra_adapters.py`: all pass).

**Damage, decided.** A record whose checksum does not match, a bad file
header, or a protobuf payload that does not parse is a corrupt log: the run
is quarantined (`RecordError`, listed by `attest runs scan`, exit 1), never
half-read -- the conformance rule. A log that simply *ends* mid-record (a
short header or short data at EOF) is a log still being written or a
process killed mid-write: everything before that point is read, and a run
caveat says the log ends in a partial record.

### W&B: decision table

| W&B | core record | note |
|---|---|---|
| log `run.project` | `family` | Was the program name. **Fallback** when there is no log (a synced directory, or one materialised like the old fixture): the program name from `wandb-metadata.json`, the legacy family, with a run caveat saying it is not the project. Neither → `null`. |
| log `run.run_id`, else the directory's id suffix | `run_id` | The ledger **name** is unchanged: `<program stem>/<run id>` (the directory name when there is no program), so `runs.detail NAME` and claims keep resolving. |
| log exit record `exit_code` 0 / nonzero | `status` `complete` / `failed` | Nonzero is `failed`, never `cancelled`: wandb's exit code does not distinguish an interrupt from a crash in any way the log states. No exit record → `null` + run caveat ("still in flight, or the process died before `wandb.finish`; W&B cannot say which"). No log → `null` + run caveat. **Not skipped** like MLflow's `RUNNING`: MLflow says "running"; a W&B log without an exit record says nothing, and skipping it would hide every crashed run forever. |
| log `run.start_time` (seconds + nanos) | `started_at` | Microsecond truncation. Fallback `wandb-metadata.json` `startedAt` (naive, written in UTC -- the legacy rule). |
| -- | `finished_at` `null` | W&B records a duration, not an end instant; `start + _runtime` would be a computed time, not a recorded one. |
| log `run.config` + config records (updates and removes, in order); else `files/config.yaml` (`{desc, value}` unwrapped, read with `yaml.safe_load`) | `config` | `_`-prefixed keys (`_wandb`, W&B's bookkeeping) dropped, the legacy rule. A non-finite value becomes its string with a run caveat (`found.json_safe`, the rule every importer shares). |
| log summary (`run.summary`, then summary records' updates/removes, in order); else `files/wandb-summary.json` | `metrics` | Every numeric leaf, nested keys joined with `.`, verbatim case (the legacy reader lowercased; W&B keys are case-sensitive and MLflow's adapter keeps them verbatim). `_`-prefixed keys dropped (`_step`, `_runtime`, `_timestamp`, `_wandb`). Media objects (a dict with `_type`), strings and booleans are not metrics. `step` `null`: a summary value is not tied to a step. A non-finite value is kept (`"NaN"`), as MLflow's: the ledger's native-record rule then stores nothing and caveats it. |
| log history rows | `history` | One series per numeric key (nested joined with `.`, `_`-keys excluded), points in log order; `step` = the row's `_step` (else the record's `step.num`, else the row index), `at` = `_timestamp` (epoch seconds). Written with `write_history` into a throwaway directory for the reference, as MLflow's adapter does. No history rows → `null`. No log → `null`: there is no other local copy of the curve (W&B 0.17 writes no `wandb-history.jsonl`; reading one would be a format this repo has never seen). |
| log `run.git.commit` (else metadata `git.commit`) | `code.git_sha`, `provenance: "inferred"` | HEAD at `wandb.init`; `dirty: null` (W&B saves a `diff.patch`, not read: its absence does not mean clean). `remote_url` → `code.repo`, credentials stripped. |
| files under `files/` other than W&B's own (`wandb-summary.json`, `wandb-metadata.json`, `config.yaml`, `requirements.txt`, `output.log`, `conda-environment.yaml`, `*.patch`, `wandb-*.jsonl`) and `code/` | `data`: `files/<path>`, sha256, role `output` | What `wandb.save` and media logging put in the run directory -- the analogue of MLflow's artifacts. Sorted by path. |
| `_runtime` from the summary | metric `wall_time_s` (unit `s`) | The compute convention; as before. The exit record's `runtime` is whole seconds (0 for every committed run) and is not used. |
| -- | `seeds`, `arm`, `dependencies` | `null`, except the ledger-wide `<config>_seed<N>` naming convention (`trackers.replicate_of`), as every adapter. `requirements.txt` is a `pip freeze` at init and a real candidate for `dependencies`; deferred until a fixture carries one (the committed one is scrubbed of it). |
| -- | `ext.ledger-import` | `adapter: "wandb"`, `status: "recorded"`, the reader caveat, run caveats. |

**Where the log and `files/` disagree, the log wins.** It is what the run
wrote; `files/` is a materialisation (by `wandb sync`, or by a script like
`generate.py`), and offline wandb 0.17.6 writes a `wandb-summary.json` that
is plainly wrong (`{"_wandb": {"runtime": 0}}`, measured).

### Hydra: decision table

| Hydra | core record | note |
|---|---|---|
| `.hydra/hydra.yaml` `hydra.job.name`, else the sweep's time directory | `family` | Unchanged (legacy). |
| path under `multirun/` (`<date>/<time>/<n>`) | `run_id` | `hydra.job.id` is the launcher's (`'0'` for the basic launcher, a SLURM id for submitit) and not unique across sweeps; the path is Hydra's own naming and is. The ledger **name** is unchanged: `<job.name>/<n>` (qualified with the time directory on a collision). |
| `.hydra/overrides.yaml` (else `hydra.overrides.task`) | `arm`, `seeds` | Hydra's own declaration of what varies. `arm` = the overrides, sorted, joined with `,` -- Hydra's `override_dirname` with its default separators -- minus a `seed=<int>` override (also `+seed`/`++seed`), which becomes `seeds` `{"seed": N}`. A sweep over the seed alone has `arm` = the job name (the base configuration). No overrides → `arm` `null`. Without seeds, `arm` changes nothing in a ranking (`replicates.identity` needs both); with them, seed replicates of one configuration now pool, which the name convention never could for Hydra (`<job>/<n>` carries no seed). |
| `.hydra/config.yaml` (whole document, `yaml.safe_load`) | `config` | Was the flat top level only. Interpolations (`${...}`) stay as written: Hydra stores the composed, unresolved config. Non-finite → string + run caveat. |
| the arm's JSON/JSONL/CSV files | `metrics` | The existing convention (`generic._hydra_arm_metrics` → `metrics_from_payload`), unchanged: Hydra has no metrics format of its own, so there is nothing more faithful to read. Its non-finite values are still skipped and reported as `LOSSY`. |
| -- | `history` `null` | Hydra records no curve. |
| -- | `status` `null` | Hydra records no outcome (a failed job in a basic-launcher sweep aborts the sweep; its directory has no metrics and is not read). Said once, in the adapter caveat, not on every run. |
| -- | `started_at`/`finished_at` `null`, no `wall_time_s` | The sweep directory's date/time is local time with no zone; nothing is inferred (compute spec). |
| -- | `code` `unknown`, `dependencies` `null` | Hydra records neither. |
| every file in the arm directory except `.hydra/` | `data`: `<path>`, sha256, role `output` | With `hydra.job.chdir=True` (which the convention requires) the arm directory *is* the job's output directory. |
| -- | `ext.ledger-import` | `adapter: "hydra"`, `status: "recorded"`, the reader caveat, run caveats. |

## Output changes (deliberate)

- **W&B family.** The committed sweep's family is its project, `flows`, not
  the script name `generate`. `examples/wandb/` (README, `run.sh`: `attest
  runs compare flows --metric auc`; the pinned header line is now `flows —
  ranked by auc (higher_is_better)`) and the tests that pinned `generate`
  change with it. Run *names* keep their shape (`generate/<id>`).
- **The committed W&B fixture was regenerated** (2026-11-20, wandb 0.17.6)
  so it keeps each run's `.wandb` log, scrubbed through wandb's own
  `DataStore` writer by `generate.py` (`_scrub_wandb_log`: only the
  run/config/history/summary/exit records and the framing; host, settings,
  telemetry and git remote cleared; the commit kept). The four runs' values
  are byte-identical (fixed `random_state`, fixed split); their run ids and
  timestamps are new.
- **W&B status, curve, code, start time** are now recorded where the log
  has them: the committed runs are `complete`, carry a 10-point
  `train_loss` history, and their commit as inferred code.
- **Metric names from W&B keep their case** and nested summary keys are one
  dotted name (`val.acc`), not `acc` with split `val`.
- **Hydra** records gain `arm`, `run_id` as a path, the whole config, and
  output files by sha256. Ledger names, family and every final value are
  unchanged, so the hydra golden path prints what it printed.
- **Reader caveats** now say what the adapters read:
  - W&B: "read by the wandb adapter: each metric is the run's final summary value
    (by default the last value logged), not the best step; the curve is in
    the run's core record (history), not in the ledger. Read from the run's
    .wandb log; a run without one is read from files/ and has no status or
    curve"
  - Hydra: "read by the hydra adapter: metrics come from whatever JSON/CSV
    file an arm's directory holds, read as a final value the same way as
    every other tracker here, not a curve or a best step; Hydra records no
    outcome and no timing, so status and wall time are not recorded"

## Rejected alternatives

- **Depend on wandb (or protobuf) to read the log.** wandb is ~100 MB with
  a Go binary; protobuf is not needed for eight message types. The
  compatibility argument above is what makes a hand decoder safe; the
  wandb-gated cross-check is what keeps it honest.
- **Keep reading only `files/`.** That is the legacy reader. Offline runs
  -- the only kind that exists locally without a sync -- have nothing there.
- **Decode the whole `wandb_internal.proto`.** Stats, console output and
  telemetry name the machine and the ledger has no field for them.
- **Skip a W&B run with no exit record (MLflow's `RUNNING` rule).** See the
  status row: W&B cannot tell in-flight from dead.
- **`hydra.job.override_dirname` as the arm.** It honours the project's
  `exclude_keys`, which is attractive (people exclude `seed` there), but then
  the arm depends on a config knob and still carries the seed by default.
  Computing it from `overrides.yaml` gives one rule, with the seed lifted
  into `seeds` where the schema has a field for it.
- **Arm from W&B's `display_name` or `group`.** `generate.py` passes one
  `name` for four arms (`lr_sweep`), as MLflow's committed sweep reuses one
  run name: a display name does not identify a configuration.
- **`finished_at` = start + `_runtime`.** A computed instant, not a
  recorded one; `wall_time_s` already carries the duration.

## Testing

`tests/test_wandb_hydra_adapters.py`:

- **Committed W&B fixture, field by field** (no wandb needed): family
  `flows`, run_id from the log, status `complete`, `started_at` from the run
  record, config without `_wandb`, the summary's metrics, a 10-point
  `train_loss` history whose reference equals `write_history` of the
  adapter's series, the commit as inferred code with no repo, `data` empty,
  `wall_time_s` = the summary's `_runtime`, `ext.ledger-import`. The values
  agree with `files/` (the materialised copy) wherever both exist.
- **Against wandb itself** (skipped unless wandb is importable): the
  decoder's records equal wandb's `DataStore` + `Record` parse of each
  committed log, and the field numbers above equal wandb's descriptors.
- **Committed Hydra fixture, field by field**: family `train`, run_id path,
  arm `lr=…` per arm, seeds null, the whole config, metrics, status null,
  no wall time, `data` = `metrics.json` by sha256, the adapter caveat.
- Synthetic logs (written with a small encoder in the test, checked against
  the committed real logs first so the encoder is not the only witness):
  nonzero exit → `failed`; no exit record → `null` + caveat; a truncated
  tail → read up to it + caveat; a bad checksum → quarantined; NaN in the
  summary → `"NaN"` kept + ledger caveat; nested summary keys; summary
  removes; media values skipped; a `files/` artifact by sha256; a directory
  with no log (the old fixture's shape) → files/ fallback + caveat.
- Synthetic Hydra sweeps: a seed override → seeds + arm without it, and two
  seeds of one configuration pooling in `ledger.compare`; seed-only sweep →
  arm = job name; a non-integer seed stays in the arm; missing
  `overrides.yaml` → `hydra.overrides.task`; nested config; non-finite
  config value.
- The existing suites (`test_tracker_adapters.py`, `test_examples.py`,
  `test_golden_paths.py`) are the behaviour-preservation evidence; the W&B
  family assertions change deliberately, as above.
