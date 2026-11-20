<!-- checked by tests/test_golden_paths.py -->

# Example: W&B

## What you get

A real, committed Weights & Biases offline run directory (`wandb/`), four
arms of one sweep, and the ledger reading it back. `generate.py` trains
`LogisticRegression` on scikit-learn's bundled breast-cancer set at four
learning rates, logs a ten-step `train_loss` curve to each run, and sets
`accuracy`/`auc` as the run's summary — the same split `examples/flows/
training/train_mlflow.py` uses between a curve-producing surrogate and the
metric that is actually reported. `attest runs scan` then reads that
directory through `ledger_adapters/wandb_store.py`, which decodes each run's
binary `.wandb` log with no wandb installed and maps the whole run --
project, exit code, config, summary, the `train_loss` curve, git commit --
into a core run record.

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
uv run attest runs compare flows --metric auc
```

Relative to this directory.

## What it prints

```
flows — ranked by auc (higher_is_better)
```

Abridged — the full run also prints the scan summary, the four runs from
`attest runs list`, and the full comparison table before that header line:

```
flows — ranked by auc (higher_is_better)

  arm                                                 auc      n      step    wall_s  source
  -------------------------------------------- ---------- ------  --------  --------  ------
  generate/poh8k6cb                                0.9970      ?             0.09344  wandb/offline-run-20261120_012708-poh8k6cb
  generate/ubscahtx                                0.9960      ?             0.09693  wandb/offline-run-20261120_012711-ubscahtx
  generate/z11zj8ir                                0.9957      ?              0.1012  wandb/offline-run-20261120_012714-z11zj8ir
  generate/akn221eo                                0.9937      ?               0.103  wandb/offline-run-20261120_012705-akn221eo

winner: generate/poh8k6cb
  caveat: some arms report no sample size, so their weight is unknown
  caveat: the top two arms differ by 0.0009921 (0.1%) -- too close to call from these numbers alone
  caveat: each arm is a single run; no seed replication, so this ranking cannot separate configuration from run-to-run variance
  caveat: read by the wandb adapter: each metric is the run's final summary value (by default the last value logged), not the best step; the curve is in the run's core record (history), not in the ledger. Read from the run's .wandb log; a run without one is read from files/ and has no status or curve
```

The `winner:` line's run id and the `wall_s` column are not pinned above —
regeneration reassigns W&B's random run ids and re-times each run, even
though the accuracy/auc values are deterministic (a fixed `random_state` and
a fixed stratified split), so only the family header line is stable across
a regeneration.

The family is `flows`, the W&B project passed to `wandb.init(project=...)`
and recorded in each run's log — not the sweep name `lr_sweep` (the run's
display name, the same for all four arms) and, since 2026-11-20, no longer
the script name `generate`. Run names keep their `<script>/<run id>` shape.

## What it demonstrates

**Confirming the reader against a real directory found two things, not the
one this task went looking for.**

The expected finding — that `_wandb_runs` (the reader before 2026-11-20) globs `run-<timestamp>-<id>/` while
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

A second finding came later (2026-11-20, roadmap 16): W&B's own family
concept, the **project**, is not in any of the three files either — nor is
the run's exit code, its git commit (once the metadata is scrubbed), or its
curve. All of them are in the binary `.wandb` log, which `generate.py` used
to delete after materialising. The log is a LevelDB log of protobuf records,
small enough to decode with the standard library (`ledger_adapters/
wandb_log.py`), so `generate.py` now keeps it, scrubbed through wandb's own
`DataStore` writer to the run/config/history/summary/exit records with the
host, settings, telemetry and git remote cleared. The ledger reads the log
first and `files/` only for a directory without one. Measured on the way:
offline wandb 0.17.6 *does* write a `files/wandb-summary.json` — containing
only `{"_wandb": {"runtime": 0}}`, which is why the log wins wherever the two
disagree.

`tests/test_wandb_hydra_adapters.py` pins every field of the committed runs'
core records; `tests/test_tracker_adapters.py::
test_the_reader_scans_the_real_committed_wandb_fixture` pins the legacy view.

## When it goes wrong

- `attest runs compare lr_sweep` (the runs' display name) or `attest runs
  compare generate` (the family before 2026-11-20) fails with "no family
  ...". Available: flows" — the family is the W&B project.
- A run directory without a `.wandb` log (one that was synced and had its
  log removed) is read from `files/`: its family falls back to the script
  name, its status is unknown and it has no curve, each said as a run
  caveat.
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
