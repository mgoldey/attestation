# Example: W&B

## What you get

A real, committed Weights & Biases offline run directory (`wandb/`), four
arms of one sweep, and the ledger reading it back. `generate.py` trains
`LogisticRegression` on scikit-learn's bundled breast-cancer set at four
learning rates, logs a ten-step `train_loss` curve to each run, and sets
`accuracy`/`auc` as the run's summary — the same split `examples/flows/
training/train_mlflow.py` uses between a curve-producing surrogate and the
metric that is actually reported. `attest runs scan` then reads that
directory through `ledger_adapters/generic.py`'s `_wandb_runs`, the first
time that reader has met a directory it did not have transcribed from
documentation.

## Prerequisites

`none — pure local computation`

`WANDB_MODE=offline` keeps every write on disk under `wandb/wandb/`; no
account, no network call. `wandb/` is already committed here, so `run.sh`
below needs nothing installed to read it. Regenerating the fixture is a
separate, explicit step — see *What it demonstrates*.

## Run it

```bash
uv run attest runs scan --root .
uv run attest runs list
uv run attest runs compare generate --metric auc
```

Relative to this directory.

## What it prints

```
generate — ranked by auc (higher_is_better)
```

Abridged — the full run also prints the scan summary, the four runs from
`attest runs list`, and the full comparison table before that header line:

```
generate — ranked by auc (higher_is_better)

  arm                                                 auc      n      step  source
  -------------------------------------------- ---------- ------  --------  ------
  generate/fsecp1k3                                0.9970      ?            wandb/offline-run-20260828_213303-fsecp1k3
  generate/h2ythad2                                0.9960      ?            wandb/offline-run-20260828_213305-h2ythad2
  generate/92ajmw2l                                0.9957      ?            wandb/offline-run-20260828_213308-92ajmw2l
  generate/d9o76utt                                0.9937      ?            wandb/offline-run-20260828_213302-d9o76utt

winner: generate/fsecp1k3
  caveat: some arms report no sample size, so their weight is unknown
  caveat: the top two arms differ by 0.0009921 (0.1%) -- too close to call from these numbers alone
  caveat: each arm is a single run; no seed replication, so this ranking cannot separate configuration from run-to-run variance
  caveat: read by the wandb adapter: values come from wandb-summary.json, which holds each metric's final logged value rather than its curve or its best step. Offline W&B does not write that file until a run is synced -- see generate.py for how the committed fixture's was materialised without a network call
```

The `winner:` line's run id is not pinned above — regeneration reassigns W&B's
random run ids even though the accuracy/auc values are deterministic (a
fixed `random_state` and a fixed stratified split), so only the family header
line is stable across a regeneration.

The family is `generate`, not the sweep name `lr_sweep` passed to
`wandb.init(name=...)`. See *What it demonstrates*.

## What it demonstrates

**Confirming the reader against a real directory found two things, not the
one this task went looking for.**

The expected finding — that `_wandb_runs` globs `run-<timestamp>-<id>/` while
real offline runs are named `offline-run-<timestamp>-<id>/` — turned out to
be **false**. Reading the code: `_wandb_runs` walks every child of `wandb/`
looking for `files/wandb-summary.json`, with no name filter at all. Both
names already worked; only the function's docstring, which described a
narrower pattern than the code implements, was wrong. No behaviour changed;
the docstring did.

The real finding was upstream of naming: **offline W&B does not write
`wandb-summary.json` or `config.yaml` to `files/` at all.** Every value
`run.log`/`run.summary` records reaches disk, but only inside the run's
binary `.wandb` transaction log — the plain files this reader reads exist
only after `wandb sync` uploads to a real server. Confirmed here against
wandb 0.17.6 through 0.29.0 (0.29.0 doesn't even write `wandb-metadata.json`
offline); independently corroborated upstream on wandb's own issue tracker
(issues #7227 and #9646, and a maintainer's own answer on issue #1768:
*"At this moment we do not have a python API of sorts to pull the values
from an offline `*.wandb` file"*). `generate.py` materialises
those two files locally by decoding the `.wandb` log with
`wandb.sdk.internal.datastore` — the community's own published workaround
for this exact gap — so the committed `wandb/` is what a real synced run's
`files/` looks like, built entirely from values wandb itself logged. Nothing
in it is invented.

A second, smaller surprise: `_wandb_runs` groups arms by the training
script's filename (from `wandb-metadata.json`'s `program`), because that is
the only run-identity field committed to any of the three files it reads.
W&B's own `run.name`/family concept (here, `"lr_sweep"`) is never written to
`config.yaml`, `wandb-summary.json`, or `wandb-metadata.json` — it exists
only inside the binary log or the server's database. Grouping by script name
instead of by intended family is a real limitation of a local-files-only
reader, not a bug introduced here; `attest runs compare` above takes
`generate`, not `lr_sweep`, for exactly this reason.

The reader needed no code change. This example's contribution is the
verification itself, `tests/test_tracker_adapters.py::
test_the_reader_scans_the_real_committed_wandb_fixture` pinning the real
fixture directly, and retiring the "never run against a real directory"
caveat this reader carried since it was written.

## When it goes wrong

- `attest runs compare lr_sweep` (the sweep's own name) fails with "no family
  'lr_sweep'. Available: generate" — see *What it demonstrates* for why the
  family is the script name, not the W&B run name.
- `generate.py` refuses to run (exit 1, clear message) under any wandb
  version but the one it was verified against — offline materialisation is
  not stable across releases, so a version mismatch would otherwise
  silently write a fixture with less in `files/` than this README
  describes. See `generate.py`'s module docstring for why.

## Next

Regenerating the fixture is deliberate and explicit — it rewrites the
committed `wandb/` in place with new run ids and timestamps:

```bash
WANDB_MODE=offline uv run --with wandb==0.17.6 --with scikit-learn --no-project python generate.py
```

The catalogue at `examples/README.md` lists the other golden paths, and
`examples/flows/training/train_mlflow.py` is the MLflow equivalent of this
same verification, done first.
