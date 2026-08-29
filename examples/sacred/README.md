# Example: Sacred

## What you get

A real, committed Sacred `FileStorageObserver` directory (`sacred_runs/`),
four arms of one sweep, and the run ledger reading it back for the first
time. `generate.py` trains `LogisticRegression` on scikit-learn's bundled
breast-cancer set at four learning rates, logs a ten-step `train_loss`
curve to each run with `_run.log_scalar`, and returns the held-out `auc`
from `@ex.main` -- Sacred's own headline number, recorded separately from
`_run.log_scalar` in `run.json`'s `result` field. `attest runs scan` then
reads that directory through `ledger_adapters/generic.py`'s new
`_sacred_runs`, the first convention this ledger has for Sacred.

## Prerequisites

`none — pure local computation`

Every run stays on disk under `sacred_runs/<n>/` with no separate "offline"
mode to opt into and no account -- unlike W&B, Sacred's `FileStorageObserver`
never talks to a server at all. `sacred_runs/` is already committed here, so
`run.sh` below needs nothing installed to read it. Regenerating the fixture
is a separate, explicit step -- see *Next*.

## Run it

```bash
uv run attest runs scan --root .
uv run attest runs list
uv run attest runs compare lr_sweep --metric auc
```

Relative to this directory.

## What it prints

```
winner: lr_sweep/2
```

Abridged -- the full run also prints the scan summary, the four runs from
`attest runs list`, and the full comparison table before that header line:

```
lr_sweep — ranked by auc (higher_is_better)

  arm                                                 auc      n      step  source
  -------------------------------------------- ---------- ------  --------  ------
  lr_sweep/2                                       0.9970      ?         0  examples/sacred/sacred_runs/2
  lr_sweep/3                                       0.9960      ?         0  examples/sacred/sacred_runs/3
  lr_sweep/4                                       0.9957      ?         0  examples/sacred/sacred_runs/4
  lr_sweep/1                                       0.9937      ?         0  examples/sacred/sacred_runs/1

winner: lr_sweep/2
  caveat: some arms report no sample size, so their weight is unknown
  caveat: the top two arms differ by 0.0009921 (0.1%) -- too close to call from these numbers alone
  caveat: each arm is a single run; no seed replication, so this ranking cannot separate configuration from run-to-run variance
  caveat: read by the sacred adapter: each metric is the LAST value in metrics.json's series, not the curve and not the best step
```

The run number in `winner:` is stable across a regeneration -- Sacred
numbers runs by how many already exist in `sacred_runs/` when it starts, and
`generate.py` always clears the directory first, so the four arms are
renumbered `1`-`4` in the same `lr` order every time. Only the `auc` values
themselves depend on nothing but a fixed `random_state` and a fixed
stratified split, so they too are deterministic.

The family is `lr_sweep` -- Sacred's `Experiment(name)`, written verbatim to
`run.json`'s `experiment.name` -- unlike the W&B example, where the family
had to fall back to the training script's filename because W&B's own
`run.name` is never written to a local file in offline mode. Sacred writes
its experiment name straight to disk; the ledger did not need a fallback.

## What it demonstrates

**`run.json`'s own `result` field is a second source of a run's headline
number, kept apart from `metrics.json`.** Sacred lets `@ex.main` return
anything JSON-serialisable, and whatever it returns lands in `run.json` as
`result` -- never inside `metrics.json`, which only holds what
`_run.log_scalar` explicitly logged. This run's `_run.log_scalar("auc",
auc)` call and its `return auc` both fire, so `auc` and `result` carry the
identical value here, but a driver script that only returns a result and
never calls `log_scalar` would previously have scanned to zero metrics and
been skipped as a spec with nothing measured. `_sacred_runs` reads both:
`metrics.json`'s series (final value, last step -- the same rule
`_mlflow_runs` already applies to MLflow's metric-per-file log) and, when
`result` is numeric, one more metric named `result`. A non-numeric `result`
(Sacred permits a dict, a list, a string) is silently not recorded as a
metric, the same refusal `metrics_from_payload` makes for every other shape
this ledger does not know how to rank.

**Only a `COMPLETED` run is recorded.** A crashed Sacred run leaves its
numbered directory, `run.json`, and whatever `metrics.json` it managed to
flush before dying -- `status` is `FAILED` or `INTERRUPTED`, not
`COMPLETED`. Recording it as a result the same way as a finished run would
misreport a crash as a measurement, so `_sacred_runs` checks `status`
before reading anything else from the directory.

## When it goes wrong

- `attest runs compare "lr_sweep sweep"` (a name with a space, or Sacred's
  full experiment description if one were set) fails the same way any
  unknown family does -- `_sacred_runs` groups strictly by
  `experiment.name`, so the family name is always exactly what
  `Experiment(...)` was constructed with.
- Regenerating with the plain `--with sacred` in the module docstring's
  install command under a Python other than 3.12 fails at import time with
  `ModuleNotFoundError: No module named 'pkg_resources'` -- see
  `generate.py`'s module docstring for why Python 3.12 is required.

## Next

Regenerating the fixture is deliberate and explicit -- it rewrites the
committed `sacred_runs/` in place:

```bash
uv run --python 3.12 --with sacred --with scikit-learn --no-project python generate.py
```

The catalogue at `examples/README.md` lists the other golden paths, and
`examples/wandb/` is the W&B equivalent of this same verification, done
first.
