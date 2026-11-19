<!-- checked by tests/test_golden_paths.py -->

# Example mlflow-records

## What you get

The real MLflow sweep committed at `examples/flows/training/mlruns/` (four
arms of a logistic-regression C sweep) written out as four core run records,
`attestation.run/1`, one run directory each — by a producer that reads MLflow
through MLflow's own client and writes only through `attestation_schema`. The
export is then diffed against the four records committed in `records/`, which
proves it is deterministic, and one record is read back.

This is roadmap item 3: the first of three unlike producers the draft schema
must survive before it is frozen. MLflow went first so the core could not come
out chemistry-shaped.

## Prerequisites

`none — pure local computation`

The export needs the `examples` dependency group (`mlflow-skinny`); `uv run
--group examples` installs it. Reading a record back needs only
`attestation-schema`, which is standard library only.

## Run it

```bash
STORE="$(mktemp -d)"
cp -R ../flows/training/mlruns "$STORE/mlruns"
for run in "$STORE"/mlruns/*/*/; do mkdir -p "${run}artifacts"; done
uv run --group examples python export_records.py --tracking-uri "$STORE/mlruns" --out "$STORE/records" --arm-param C --seed-param seed --direction auc=max
diff -r records "$STORE/records" && echo "byte-identical to the committed records/"
uv run python -c "import attestation_schema as s; r = s.read('records/run-06f405c125804496876ebfa098ad4c94'); print(r.schema, r.family, r.arm, [(m.name, m.value, m.direction) for m in r.metrics if m.name == 'auc'], len(s.read_history('records/run-06f405c125804496876ebfa098ad4c94', r)[-1].points), 'train_loss points')"
```

Relative to this directory (`run.sh` does `cd "$(dirname "$0")"` first).
`generate.sh` is the same export aimed at `records/` instead of a temporary
directory.

## What it prints

```
4 attestation.run/1 record(s) written and read back
byte-identical to the committed records/
attestation.run/1 flows C=10.0 [('auc', 0.9960317460317459, 'max')] 10 train_loss points
```

Above those lines, one per run:
`run-06f405c1  complete  family=flows arm=C=10.0  5 metrics, 5 series, 14 points, 1 data`.

## What it demonstrates

**The mapping, with nothing special-cased.** `export_records.py`'s docstring
is the table: experiment name → `family`; params → `config` verbatim, as the
strings MLflow stored (`"C": "10.0"`, never re-typed); every logged point →
the history sidecar; MLflow's own "latest" point per metric → `metrics`; the
`mlflow.source.git.commit` tag → `code.git_sha` with `dirty: null`; artifacts
→ `data` by sha256; dataset inputs → `data` with `sha256: null`; everything
MLflow-specific (experiment id, run name, every tag, every dataset input
verbatim) → `ext.mlflow`.

**What MLflow does not know stays null.** No metric has a direction unless
the person exporting declares one (`--direction auc=max`); no run has an arm
unless they name the params that define one (`--arm-param C`); no run has
seeds unless they name the params that hold them (`--seed-param seed`). The
committed sweep was scrubbed of its git tags before it was committed, so its
records honestly carry `git_sha: null`. MLflow records no client or library
version for a run, so `dependencies` is `null`: the exporter could not
observe the run's environment (`[]` would claim it saw one with nothing in it).

**History is kept, beside the record.** `train_loss` was logged at ten steps;
all ten points, with their timestamps, go to `attestation/history.jsonl` (one
canonical JSON line per series declaration and per point), and `run.json`
carries only `history: {path, sha256, series: 5, points: 14}` -- so a record
stays the same size however long the run, `read` never opens the curves, and
`read_history` checks their sha256 before parsing. `metrics` carries the final
value at `step: 9`. The ledger's MLflow reader keeps only the last line; a
run directory keeps the curve.

**The run directory layout.** Each `records/run-<run_id>/` holds
`attestation/history.jsonl` (written first), `attestation/digests.json` and
`attestation/run.json` (written last, binding the sidecar by `history.sha256`
and listing the manifest's sha256 with `role: "manifest"`; the manifest does
not list the sidecar, so it is bound once).
MLflow's artifacts stay in MLflow's store and are bound by digest, not
copied, so these manifests list no files.

**What MLflow forced into the schema** is the "Producer findings" section of
`docs/superpowers/specs/2026-11-19-core-run-record-design.md`: metric history
as a core field (since moved to the sidecar), NaN/infinite metric values,
data named without a digest, and one spelling per timestamp (now fixed-width:
`22:33:46.024000Z`).

## When it goes wrong

- `MlflowException: Run '06f405c1...' not found` means the store was read
  directly from `../flows/training/mlruns`. git cannot commit an empty
  directory, every run there logged no artifacts, and MLflow 3's file store
  refuses a run directory without `artifacts/` — hence the copy and the
  `mkdir` loop, which give MLflow back the store it wrote.
- `diff` reporting a difference means the mapping changed (or the committed
  sweep was regenerated): rerun `./generate.sh` and review the diff as a
  schema or mapping change, not noise.
- `ValueError: ... --arm-param ['X'] not among its params` or `... is not an
  integer seed`: a declaration that does not hold for every run is refused,
  never defaulted.
- A metric logged as `NaN` is written as the string `"NaN"`, not dropped and
  not an error; a reader gets `float('nan')` back.

## Next

`examples/mlflow/` reads the same sweep through the run ledger's MLflow
adapter, which keeps final values only. `packages/attestation-schema/` is the
schema these records are written in.
