# The MLflow adapter reads the whole run, not the last line

**Date:** 2026-11-20
**Status:** implemented 2026-11-20 on `next/mlflow-adapter` (roadmap item 12).
**Depends on:** `2026-11-19-ledger-reads-core-records-design.md` (the adapter
protocol, `ext.ledger-import`, in-memory conversion),
`2026-11-19-core-run-record-design.md` (Producer findings 1: the MLflow
mapping, proven by `examples/mlflow-records/export_records.py`),
`2026-11-19-compute-tracking-design.md` (`wall_time_s`).
**Not this:** a curves table in the ledger, a runs.history tool, MLflow's
logged models (`outputs/`, `model_id` on a metric point), a tracking server or
a sqlite store (the ledger reads files a project already has), a cache of
imported records on disk.

## Problem

The ledger's MLflow reader (`generic._mlflow_runs`, ported as-is by roadmap
11) keeps the last parseable line of each metric file, the params and the run
name. Everything else MLflow recorded is dropped before the ledger sees it:
the experiment (the reader used the *run name* as the family), the status
(a `FAILED` run ranked like a finished one, a `RUNNING` one was imported), the
curve, the git identity, the artifacts and the dataset inputs. Meanwhile the
exporter (`examples/mlflow-records/export_records.py`) maps all of it to a
core record through `MlflowClient`, and its mapping was checked against
MLflow itself. Two producers of one mapping that disagree on the family of
the same run is the drift the core record exists to remove.

## Decision: the exporter's mapping, read from the file store with no mlflow

A new module, `ledger_adapters/mlflow_store.py`, reads `mlruns/` directly
and maps each run to a core record with the exporter's mapping, line for
line:

| MLflow (file store) | core record | note |
|---|---|---|
| `<exp>/meta.yaml` `name` | `family` | Was the run name. A missing experiment `meta.yaml` (a hand-made fixture) is `null`, not a guess. |
| (no MLflow concept) | `arm` `null` | As the exporter without `--arm-param`. The one exception is the ledger-wide seed convention every legacy reader shares: a run *named* `<config>_seed<N>` gets arm `<config>` and seeds `{"seed": N}` (`trackers.replicate_of`, roadmap 17), because that name is the person's declaration. |
| run directory name | `run_id` | The ledger's *name* for the run is unchanged (`<run_name>/<id8>`, or the id), so every claim and every `runs.detail NAME` keeps resolving. The name is a property of the import (`Imported.name`), the id a property of the run. |
| `status` `FINISHED`/`FAILED`/`KILLED` (int 3/4/5, or the name) | `complete`/`failed`/`cancelled` | `RUNNING`/`SCHEDULED`/absent (MLflow's default is `RUNNING`) get no record: terminal-only, as the core adapter skips a run in flight. `failed`/`cancelled` get the ledger's existing "did not complete" run caveat. |
| `start_time`/`end_time` (epoch ms) | `started_at`/`finished_at` | `format_timestamp`, the exporter's arithmetic. `end_time` 0 or absent is `null`. |
| `params/**` | `config` | Verbatim file contents (MLflow's `read_file`), never re-typed, never stripped. Keys with `/` are nested files, read recursively as MLflow does. |
| `metrics/**` every line | `history` | One series per (key, linked dataset), points sorted by timestamp (stable), as the exporter does. |
| MLflow's latest point | `metrics` | `max(step, timestamp, value)` -- `FileStore._get_metric_from_file`'s own key. Was the last *line* of the file, which differs when steps are logged out of order. `direction`/`unit` `null`, `labels` `{}`; linked dataset name → `split`. A non-finite final value is kept (`"NaN"`), see below. |
| `tags/mlflow.source.git.commit` | `code.git_sha`, `provenance: "inferred"` | `dirty: null`, `version: null`, the exporter's evidence text. No tag → `unknown`. |
| `tags/mlflow.source.git.repoURL` | `code.repo`, credentials stripped | MLflow keeps a hand-set tag's token verbatim; the schema refuses one. |
| `inputs/*/meta.yaml` → `<exp>/datasets/<id>/meta.yaml` | `data`: `datasets/<name>@<digest>`, `sha256: null`, role = `mlflow.data.context` or `input` | MLflow's digest is not a sha256 of any bytes; it disambiguates the name. Sorted by name. Model inputs are not mapped. |
| artifact files | `data`: `artifacts/<path>`, sha256 of the bytes, role `output` | After the datasets, sorted by path. Where they are: below. |
| (none recorded) | `dependencies: null`, `seeds: null` | As the exporter. |
| start/end | metric `wall_time_s` (unit `s`) | The compute convention every adapter follows; not the exporter's. |
| -- | `ext.ledger-import` | The importer's one namespace: `adapter: "mlflow"`, `status: "recorded"` (the ledger's word, unchanged), the reader caveat, run caveats. **Not `ext.mlflow`**: that namespace belongs to the exporter, and nothing in the ledger would read a second copy. |

### Reading the store without mlflow

mlflow is not a runtime dependency and does not become one. The file store
is plain files: YAML `meta.yaml`s (read with `yaml.safe_load`, which is what
MLflow's own `read_yaml` calls -- pyyaml is already a runtime dependency of
attestation, so this adds nothing), one text file per param and tag, and one
line per logged metric point. Where MLflow's semantics matter, they are
mirrored, with the source named in the code:

- **Latest point**: greatest step, then latest timestamp, then largest value.
- **Metric lines**: `<ts> <value> [<step> [<dataset name> <dataset digest>]]`,
  split on single spaces; 2, 3 or 5 fields, anything else malformed.
- **A run MLflow cannot read is refused, not guessed at.** A malformed or
  empty metric file, or a `meta.yaml` whose `run_id` is not its directory,
  makes `MlflowClient.get_run` raise for the whole run. The adapter yields
  the run with its reason and `to_core_record` raises a `RecordError`, so the
  import pipeline quarantines it (listed, `attest runs scan` exits 1) --
  the conformance rule, "never coerced". The old reader skipped bad lines
  and imported whatever was left.
- **Deleted** runs (`lifecycle_stage: deleted`) and deleted experiments are
  skipped, as before; `.trash/` and the model registry's `models/` are not
  experiments.
- **Run name**: `meta.yaml`'s `run_name`, else the `mlflow.runName` tag
  (`FileStore._get_run_from_info`).

**Two file-store quirks, decided:**

1. *Empty `artifacts/` directories dropped by git.* MLflow 3 refuses a run
   directory without `artifacts/`, so the exporter's golden path copies the
   store and recreates them. Reading files directly has no such rule: a
   missing `artifacts/` is a run with no artifacts, which an empty one is
   anyway.
2. *The dataset-step bug.* MLflow 3.15.2 writes a dataset-linked point as
   `<ts> <value> <step> <name> <digest>` but `_get_metric_from_line` reads
   the step only when there are exactly three fields, so the client returns
   `step=0` for every such point (the exporter inherits it). The adapter
   reads the step the file holds. **This is the one deliberate divergence
   from MlflowClient**: mirroring it would make the ledger report a value
   logged at step 5 as step 0, and pick the latest point among dataset-linked
   values by timestamp alone. For a dataset-linked metric the two producers
   therefore disagree on `step` (and on `latest` when the steps were logged
   out of order); no committed fixture has one.

### Where artifacts are

`meta.yaml`'s `artifact_uri` says where MLflow put them. When it is the
store's default location (it ends in `<exp_id>/<run_id>/artifacts`), the
adapter reads `<run dir>/artifacts/` -- the copy in the store being scanned,
wherever the store now lives (a committed or moved store's URI names a path
on another machine, or the relative `file:./mlruns/...` the training
fixture's scrub leaves). Any other local `file:` URI that exists is read
there. A URI the adapter cannot read (`s3://`, a local path that is gone) is
not guessed at: `data` lists no artifacts and a run caveat says the
artifacts at that URI were not read.

Every artifact is hashed on every scan (the ledger converts in memory and
keeps nothing between scans). That costs time in proportion to the bytes; the
exporter pays it once. Measured on the committed fixtures it is nothing (no
artifacts); a project with large checkpoints logged as artifacts will feel
it. If it bites, the fix is a cache keyed by (path, size, mtime) -- not done
now, because there is no measurement saying it is needed.

### Non-finite final values

The old reader dropped a metric whose last line was `nan`/`inf` and reported
it under `LOSSY` (a run whose only metric diverged vanished). The record now
says what MLflow says: `"NaN"`, as the exporter does (Producer findings 1.3:
dropping it "makes a diverged arm look like one with no loss"). The ledger's
existing rule for a native record applies: the value is not stored in
`run_metrics` and a run caveat names it. A diverged run is now *in* the
ledger, with that caveat, rather than absent. `generic.discover`'s legacy
`Found.metrics` lists only finite final values, so its callers still never
see a NaN.

### Curves: in the record, not in the ledger

The ledger has no curves table, and this does not add one.

- **Nothing would read it.** `compare`, `claims`, `detail`, replicates and
  compute all read final values. A table with no reader is a migration
  (017), two object-count pins (`test_db.py`, `test_library_db.py`), the
  count in CLAUDE.md, and rows -- a 200-step run is 200 rows per metric --
  for no answer anyone can get.
- **The record still says what was logged.** `history` is the reference the
  exporter writes (`{path, sha256, series, points}`): the adapter builds the
  `Series` from the store, writes them with the schema's own
  `write_history` into a temporary directory to get the reference (the one
  sanctioned writer; no hand-rolled lines, no new schema API), and keeps the
  `Series` on the found run (`MlflowFound.series`). The sidecar bytes are a
  pure function of the store, so the digest names bytes anyone can
  regenerate, and a later materialization of an import (the ledger spec's
  "cache it owns") writes exactly these bytes. A `null` history would claim
  no curve was logged, which is false.
- **The ledger ignores it**, as it ignores `data`, `code` and
  `dependencies`. The reader caveat says so: the curve is in the run's core
  record, not in the ledger.

A curves table, if a tool ever needs one, is its own spec with the reader
that needs it.

### Shape in the code

`mlflow_store.mlflow_runs(root, seen)` is the reader (it replaces
`generic._mlflow_runs`, which is deleted with `_mlflow_metric`). It returns
`MlflowFound`, a subclass of the legacy `Found`, so `generic.discover` and
its callers (the tests, `attest record`'s pairing check) see the same legacy
view as before -- name, family, config, finite final values, `wall_time_s`
-- while the store's facts ride along for the record. `trackers.MLFLOW` is a
`TrackerAdapter` subclass whose `to_core_record` calls
`mlflow_store.to_core_record` instead of `found_to_core`; its name, marker,
order in the registry and `run_id_suffix` are unchanged. Every tracker name
stays inside `ledger_adapters/` (`tests/test_tracker_names.py`).

## Output changes (deliberate)

- **Family.** The committed sweep's family is its experiment, `flows`, not
  its run name, `c_sweep`. `attest runs compare c_sweep` becomes `attest
  runs compare flows` in `examples/mlflow/` (README, `run.sh`; the pinned
  header line is now `flows — ranked by auc (higher_is_better)`) and in
  `tests/test_examples.py`. Run *names* are unchanged (`c_sweep/06f405c1`),
  so `FINDINGS.md`'s claims, the winner line and the auc table are byte for
  byte what they were. `train_mlflow.py` still names its runs `c_sweep`; its
  docstring no longer says the run name is the family. The committed
  `RESULTS.md` heading ("Training family `c_sweep`") is a live-run record and
  is left as written.
- **The reader caveat** now reads: "read by the mlflow adapter: each metric
  is MLflow's own latest value (greatest step, then latest timestamp), not
  the best step; the full curve is in the run's core record (history), not in
  the ledger".
- **Status.** `FAILED`/`KILLED` runs carry the "did not complete" caveat;
  `RUNNING`/`SCHEDULED` runs are no longer imported.
- **Malformed stores** are quarantined instead of half-read.
- **NaN finals** are a run caveat instead of a `LOSSY` note, and the run is
  kept.

## Testing

`tests/test_mlflow_adapter.py`:

- **Cross-check, committed sweep (no mlflow needed).** For each of the four
  runs, the adapter's record equals the exporter's committed record in
  `examples/mlflow-records/records/` on every field the exporter sets --
  `schema, run_id, family, status, started_at, finished_at, code, config,
  data, metrics, history (the full reference, sha256 included, so every
  point and timestamp agrees byte for byte), seeds, dependencies` and `arm` --
  after applying the exporter's own declarations (`--arm-param C
  --seed-param seed --direction auc=max`, through the exporter's own `_arm`
  and `_seeds`) to the adapter's record, since a declaration is what neither
  the store nor the adapter knows. Excluded, each for a stated reason:
  `producer` (each names itself), `ext` (different namespaces by design), the
  exporter's manifest entry in `data` (it describes the exporter's run
  directory; an import has none), the adapter's `wall_time_s` (the compute
  convention, checked against `end - start`). The adapter's `Series` are
  also compared with `read_history` of the committed sidecar.
- **Cross-check, a fresh real store (needs mlflow, as the exporter's own
  live tests do)** in `tests/test_mlflow_records.py`, on the store those
  tests already write: datasets, an artifact, git identity, a hand-set token
  in a repo URL, a killed run, a running run, `inf`/`NaN`, two values at one
  step. Same comparison, no declarations.
- Synthetic stores for what neither real store has: the dataset-step line,
  out-of-order steps (latest point vs last line), a failed run, a malformed
  line quarantined, an unreadable artifact URI, nested param keys, a missing
  experiment `meta.yaml`.
- The existing suites are the behaviour-preservation evidence;
  `test_tracker_adapters.py`'s
  `test_a_finite_value_before_a_non_finite_one_is_still_lost` changes
  deliberately (the run is now kept, its loss is not reported as 0.5 and is
  NaN in its record), as do the family assertions in `test_examples.py`.
