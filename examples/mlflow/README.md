<!-- checked by tests/test_golden_paths.py -->

# Example mlflow

## What you get

A standalone front door onto the real MLflow directory committed at
`examples/flows/training/mlruns/`: four arms of one sweep, read back by the
run ledger, ranked, and checked against a findings file with one claim
deliberately wrong. This path adds no fixture of its own — `flows/` already
runs this as one step of three; this one exists so the MLflow reader has its
own README, its own `run.sh`, and its own row in the catalogue, for someone
who wants exactly this and nothing else.

## Prerequisites

`none — pure local computation`

The retrain step needs the `examples` dependency group (`mlflow-skinny` is
in it — `uv sync --group examples` if `uv run --group examples` fails with
`ModuleNotFoundError: mlflow`), but needs no model server.

## Run it

```bash
uv run attest runs scan --root ../flows --project training
uv run attest runs compare c_sweep --metric auc
uv run attest claims ../flows/training/FINDINGS.md || true
uv run --group examples python ../flows/training/train_mlflow.py --out "$(mktemp -d)"
```

Relative to this directory (`run.sh` does `cd "$(dirname "$0")"` first); the
inputs live one level up, in `examples/flows/training/`.

## What it prints

```
c_sweep — ranked by auc (higher_is_better)
```

Abridged — `runs scan` reports `training  4 run(s)`; `runs compare` prints
the table below the header line above, then:

```
winner: c_sweep/06f405c1
  caveat: some arms report no sample size, so their weight is unknown
  caveat: the top two arms differ by 0.0003307 (0.0%) -- too close to call from these numbers alone
  caveat: each arm is a single run; no seed replication, so this ranking cannot separate configuration from run-to-run variance
  caveat: read by the mlflow adapter: each metric is the FINAL line of its metrics file, not the curve and not the best step
```

The `06f405c1` run id is stable only for the *committed* fixture — it
changes if `train_mlflow.py` is ever run with no `--out` and the directory
regenerated, which is why the pinned line above is the header, not this
one. `claims` then prints four `supported` verdicts and one `contradicted`,
ending `5 claim(s): 1 contradicted, 4 supported`; the retrain step logs
four arms and finishes with `4 runs logged to <tmpdir>/mlruns in 2.0s`.

## What it demonstrates

**The MLflow layout**: `mlruns/<experiment_id>/<run_id>/{meta.yaml,params/,
metrics/}`. `run_name` in `meta.yaml` names the family every arm shares
(`c_sweep`, here); a run with `lifecycle_stage: deleted` is skipped, because
resurrecting a run someone deleted in the MLflow UI into a ledger meant for
provenance is worse than missing it. Each file under `metrics/` is one
metric with one line per logged step (`<timestamp_ms> <value> <step>`); the
ledger reads the **last** line — the final value and the step it was logged
at — never the curve. `training/mlruns/.../metrics/train_loss` has ten
lines from a ten-step curve; `runs.detail` reports its final value at
`step=9`, not an average and not the minimum.

**Why final values, not curves**: recording every logged step would put one
row per step into `run_metrics` — a 200-epoch run becomes 200 rows per
metric — and make MLflow runs structurally unlike every other run in the
ledger. The reader (`_mlflow_runs` in
`src/attestation/ledger_adapters/generic.py`) says this in its own
docstring: a user who wants training curves is not served by this ledger.

**The scrub**: MLflow writes personal attribution and machine-specific
paths by default — a tag naming who ran it, the git remote URL, this
machine's home directory inside every `artifact_uri`. `train_mlflow.py`'s
`scrub()` (see its `_SCRUB_TAGS`) strips all of it after training, because
the reader never touches any of it — it reads only `lifecycle_stage`,
`run_name`, `metrics/*`, `params/*`.
Regenerating the committed directory is deliberate, not accidental: it
changes every run id, which is exactly why the pinned line above is the
table header rather than the `winner:` line.

**The table `runs compare` prints for `auc`**, from the committed fixture:

| C | auc |
|---|---|
| 10.0 | 0.9960 |
| 1.0 | 0.9957 |
| 0.1 | 0.9940 |
| 0.01 | 0.9864 |

## When it goes wrong

- `ModuleNotFoundError: mlflow` on the retrain step means the `examples`
  group isn't installed: `uv sync --group examples`.
- A newer mlflow may refuse a `file:` tracking URI outright, or refuse it
  without `MLFLOW_ALLOW_FILE_STORE=true` for a different reason than the one
  this repo hit — `train_mlflow.py` sets that variable before `import
  mlflow` runs, so the documented command needs no extra environment
  variable; if a future mlflow release changes the refusal's shape, the
  fix is in `train_mlflow.py`, not here.
- `attest runs compare c_sweep --metric train_loss` refuses to rank rather
  than guessing whether lower or higher is better — `--metric` needs a
  declared direction, and `train_loss` has none.
- `attest claims ../flows/training/FINDINGS.md` exits 1 because one claim
  is deliberately `contradicted` — that is the point of the fixture, not a
  failure of the tool; `run.sh` carries `|| true` on that line so the
  retrain step still runs.

## Next

`examples/flows/` runs this same fixture as one of three flows, alongside a
persona-scored corpus and a full MCP call sweep; `examples/wandb/` is the
sibling tracker-directory path.
