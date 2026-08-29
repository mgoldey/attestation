# Example: Hydra

## What you get

A real, committed Hydra `--multirun` sweep (`multirun/<date>/<time>/<n>/`),
four arms of one sweep, and the run ledger reading it back for the first
time. `train.py` trains `LogisticRegression` on scikit-learn's bundled
breast-cancer set at four learning rates via `@hydra.main`, and Hydra's own
sweep runner writes each arm's resolved config, overrides, and Hydra's own
job metadata into `.hydra/` beside the `metrics.json` `train.py` itself
writes. `attest runs scan` then reads that directory through
`ledger_adapters/generic.py`'s new `_hydra_runs`, the first convention
this ledger has for Hydra.

## Prerequisites

`none — pure local computation`

`multirun/` is already committed here, so `run.sh` below needs nothing
installed to read it -- `_hydra_runs` parses `.hydra/config.yaml` and
`.hydra/hydra.yaml` directly and never imports or shells out to `hydra`.
Regenerating the fixture is a separate, explicit step -- see *Next*.

## Run it

```bash
uv run attest runs scan --root .
uv run attest runs list
uv run attest runs compare train --metric auc
```

Relative to this directory.

## What it prints

```
winner: train/0
```

Abridged -- the full run also prints the scan summary (including one
honest diagnostic, see below), the runs from `attest runs list`, and the
full comparison table before that header line:

```
train — ranked by auc (higher_is_better)

  arm                                                 auc      n      step  source
  -------------------------------------------- ---------- ------  --------  ------
  train/0                                          0.9970      ?            multirun/2026-08-28/23-36-10/0
  train/1                                          0.9960      ?            multirun/2026-08-28/23-36-10/1
  train/2                                          0.9957      ?            multirun/2026-08-28/23-36-10/2
  train/3                                          0.9940      ?            multirun/2026-08-28/23-36-10/3

winner: train/0
  caveat: some arms report no sample size, so their weight is unknown
  caveat: the top two arms differ by 0.0009921 (0.1%) -- too close to call from these numbers alone
  caveat: each arm is a single run; no seed replication, so this ranking cannot separate configuration from run-to-run variance
  caveat: read by the hydra adapter: metrics come from whatever JSON/CSV file an arm's directory holds, read as a final value the same way as every other tracker here, not a curve or a best step
```

`attest runs scan --root .` also prints one diagnostic line worth expecting
rather than being surprised by: `conf` -- this example's own Hydra config
directory (`conf/config.yaml`) -- is one of `generic.py`'s `CONFIG_DIRS`,
so it is read as an ordinary config spec named `config` alongside the four
sweep arms, the same "a spec with no result attached is recorded as a run
with no metrics" honesty every config file gets. It is not a bug: `conf/`
really is a config directory, Hydra's own, and the ledger has no way to
know it is a Hydra input rather than an ordinary one.

`winner: train/0` is stable across a regeneration -- `lr` in `{0.01, 0.1,
1, 10}` is fixed, and `train.py`'s `auc` values depend on nothing but a
fixed `random_state` and a fixed stratified split, so `lr=0.01` (arm `0`)
wins by the same margin every time. Only the date/time sweep directory
Hydra names changes, which is exactly why the run is named `train/0`
(`<job.name>/<n>`) rather than `train/<date>/<time>/0` -- see *What it
demonstrates*.

The family is `train` -- Hydra's own `hydra.job.name`, read from `.hydra/
hydra.yaml`, which defaults to the driver script's stem (`train.py` ->
`train`) the same way Sacred writes `experiment.name` straight to disk
with no fallback needed.

## What it demonstrates

**`<job.name>/<date>/<time>/<n>` is unreadable, so the reader drops the
date and time and names a run `<job.name>/<n>` instead.** Hydra sweeps
runs under a timestamped directory precisely so two sweeps never collide
on disk, but a ledger listing `train/2026-08-28/23-36-10/0` beside three
siblings is the same unreadable-name problem `_wandb_runs` solved by
dropping W&B's timestamp-and-hash run directory in favour of the program
name plus a short id. `_hydra_runs` makes the same trade: `<job.name>/<n>`
is what a human reads, and a second sweep of the same job is not silently
dropped -- if `<job.name>/<n>` is already taken, the run is re-qualified
with its time directory (`<job.name>/<time>/<n>`) rather than lost, the
same `seen`-based dedup every other tracker reader here participates in.

**`hydra.job.chdir` is not what Hydra <1.2's users remember, and finding
that took running the sweep, not reading Hydra's changelog first.**
hydra-core 1.3.5's `@hydra.main` does not change the working directory
into each arm's own output directory unless `hydra.job.chdir=True` is
passed on the command line -- a backward-compatibility default Hydra
introduced in 1.2 for configs that read relative paths from the original
launch directory. Without it, `train.py`'s `open("metrics.json", "w")`
(matching the module docstring's promise that Hydra "writes metrics.json
into `os.getcwd()`") writes into the directory `generate.sh` was run from,
and all four arms of the sweep overwrote the exact same file -- a real run
of `--multirun lr=0.01,0.1,1,10` with no override produced one
`metrics.json`, not four. `generate.sh` passes the override explicitly;
`_hydra_runs` reads only the layout that override produces, since that is
the only layout `--multirun` reliably writes today.

**A nested YAML key needed a small generalisation of `_yaml_scalars`, not
a new parser.** `.hydra/hydra.yaml`'s `hydra.job.name`, `hydra.overrides.
task` and `hydra.sweep.dir` all sit several levels deep, unlike
`meta.yaml`'s and `config.yaml`'s flat top-level keys `_yaml_scalars`
already reads. `_yaml_path_index`/`_yaml_path_scalar`/`_yaml_path_list`
walk `_indented_lines`'s own (indent, key, value) triples by a dotted
path instead -- the same reuse `_dvc_stages`/`_dvc_lock_params` already
make of `_indented_lines` for DVC's own nested `stages:` shape, extended
rather than forked. One real surprise surfaced only by running a real
sweep: Hydra's YAML dumper writes a block list's `- item` lines at the
*same* indent as the key introducing them (`task:` and its `- lr=0.01`
line are both at indent 4), not one level deeper the way DVC's own
`dvc.yaml` writer does -- `_yaml_path_list` scans forward while a line is
a list item rather than requiring a deeper indent.

## When it goes wrong

- `attest runs compare train --metric wer` (a metric name `train.py` never
  writes) prints every arm as `(none)` and `winner: None`, with a caveat
  naming all four arms as having no `wer` -- the same "arm was never
  evaluated on this metric" honesty every adapter's comparison gives.
- Regenerating with a bare `--multirun lr=...` (omitting
  `hydra.job.chdir=True`) does not fail loudly -- it silently trains four
  arms into one shared `metrics.json`, and `attest runs scan` then reads
  one run, not four, with no error at all. This is the finding *What it
  demonstrates* describes above, reachable by anyone who copies the
  command from Hydra's own older documentation or tutorials without the
  override.

## Next

Regenerating the fixture is deliberate and explicit -- it rewrites the
committed `multirun/` in place:

```bash
uv run --with hydra-core --with scikit-learn --no-project python \
    train.py --multirun lr=0.01,0.1,1,10 hydra.job.chdir=True
```

Or run `./generate.sh`, which does the same thing, then scrubs this
machine's absolute path out of every arm's `.hydra/hydra.yaml`
(`hydra.runtime.cwd`, `hydra.runtime.output_dir`, and the `conf/` entry in
`hydra.runtime.config_sources`) and deletes each arm's `train.log` and the
sweep-level `multirun.yaml`, none of which `_hydra_runs` reads.

The catalogue at `examples/README.md` lists the other golden paths, and
`examples/sacred/` and `examples/dvc/` are the closest siblings to this
one: each reads a real sweep tool's own directory convention with a small,
hand-rolled parser rather than a dependency on the tool's own package.
