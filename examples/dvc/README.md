# Example: DVC

## What you get

A real, committed DVC pipeline (`dvc.yaml`, `params.yaml`, `dvc.lock`, and
the `metrics/` it wrote) and the run ledger reading it back for the first
time. `train.py` trains `LogisticRegression` on scikit-learn's bundled
breast-cancer set at four learning rates, declared as one `foreach` stage
in `dvc.yaml` (`train@0.01`, `train@0.1`, `train@1`, `train@10`) rather than
four separate stages. `attest runs scan` then reads that layout through
`ledger_adapters/generic.py`'s new `_dvc_runs`, the first convention this
ledger has for DVC -- parsed by hand, with no dependency on the `dvc`
package itself.

## Prerequisites

`none — pure local computation`

`dvc.yaml`, `params.yaml`, `dvc.lock` and `metrics/*.json` are already
committed here, so `run.sh` below needs nothing installed to read them --
`_dvc_runs` parses these files directly and never shells out to `dvc`.
Regenerating the fixture is a separate, explicit step that does need `dvc`
installed -- see *Next*.

## Run it

```bash
uv run attest runs scan --root .
uv run attest runs list
uv run attest runs compare train --metric auc
```

Relative to this directory.

## What it prints

```
winner: train@0.01
```

Abridged -- the full run also prints the scan summary, the four runs from
`attest runs list`, and the full comparison table before that header line:

```
train — ranked by auc (higher_is_better)

  arm                                                 auc      n      step  source
  -------------------------------------------- ---------- ------  --------  ------
  train@0.01                                       0.9970      ?            metrics/0.01.json
  train@0.1                                        0.9960      ?            metrics/0.1.json
  train@1                                          0.9957      ?            metrics/1.json
  train@10                                         0.9940      ?            metrics/10.json

winner: train@0.01
  caveat: some arms report no sample size, so their weight is unknown
  caveat: the top two arms differ by 0.0009921 (0.1%) -- too close to call from these numbers alone
  caveat: each arm is a single run; no seed replication, so this ranking cannot separate configuration from run-to-run variance
  caveat: read by the dvc adapter: each metric file is a snapshot overwritten on every `dvc repro`, not a curve -- there is no history of a prior run's value once a stage reruns
```

`winner: train@0.01` is stable across a regeneration for two independent
reasons: DVC's `foreach: ${lr}` names each stage instance `<stage>@<item>`
directly from `params.yaml`'s `lr` list, so the four run names never
depend on a hash or a timestamp the way W&B's or MLflow's ids do; and
`train.py`'s `auc` values depend on nothing but a fixed `random_state` and
a fixed stratified split, so `lr=0.01` wins by the same margin every time.

The family is `train` -- the stage name `dvc.yaml` declares before its
`foreach` -- read the same way `_sacred_runs` splits `experiment.name`
from a run number, just on `@` instead of `/`.

## What it demonstrates

**`dvc.lock` records the whole swept list for its `foreach` key, not the
one value each stage instance actually ran with.** Every `train@<lr>`
entry in `dvc.lock` carries the *same* `params.yaml: lr: [0.01, 0.1, 1,
10]` list, because DVC is quoting the source value each instance was
generated from, not the item it was instantiated with. Trusting that
verbatim would give every arm the identical, useless config `lr: [0.01,
0.1, 1, 10]`. `_dvc_runs` instead takes the `lr` value for each run from
the stage-instance name itself (`train@0.1` implies `lr=0.1`) -- the one
place DVC's own naming convention already carries the answer `dvc.lock`'s
`params:` block does not.

**No dependency on the `dvc` package.** `dvc.yaml`, `params.yaml` and
`dvc.lock` are all machine-written YAML with a small, predictable shape --
the same reasoning that keeps `generic.py`'s `_yaml_scalars` and
`_config_shape` from reaching for a YAML parser dependency for
`meta.yaml`/TOML configs (see that module's own comment on the `networkx`
lesson). `_dvc_runs` reads them with the same kind of small, hand-rolled
parser rather than adding `dvc` -- or even `PyYAML`, which is already on
disk only as a transitive dependency of dev tools like `pre-commit` and
`bandit`, never a direct one -- to `src/`'s runtime dependencies.

**A DVC sweep would otherwise be scanned twice.** `metrics/` is one of the
generic reader's own `RESULT_DIRS`, unlike `wandb/`, `mlruns/` or
`sacred_runs/`, none of which collide with a plain-results convention.
Without a guard, the ordinary `metrics/` walk in `discover()` reads
`metrics/0.1.json` as a second, bare-named run (`0.1`) alongside
`_dvc_runs`'s own `train@0.1` for the same file. `discover()` now computes
which metric files `dvc.yaml`'s stages claim before the ordinary scan runs,
and skips them there.

## When it goes wrong

- `attest runs compare train --metric wer` (a metric name `train.py` never
  writes) prints every arm as `(none)` and `winner: None`, with a caveat
  naming all four arms as having no `wer` -- the same "arm was never
  evaluated on this metric" honesty every adapter's comparison gives,
  rather than a crash or a silently empty table.
- Regenerating with `dvc repro` before `dvc init` (skipping the command in
  *Next* verbatim) fails with DVC's own `ERROR: you are not inside of a DVC
  repository` -- `dvc init --no-scm -q` is required first because this
  directory is not a git repository DVC could otherwise detect.

## Next

Regenerating the fixture is deliberate and explicit -- it rewrites the
committed `dvc.lock` and `metrics/*.json` in place:

```bash
uv run --with dvc==3.67.1 --with scikit-learn --no-project bash -c \
    'dvc init --no-scm -q && dvc repro -q'
```

The pin matches `DVC_VERSION` in `generate.sh`, which exits 1 if the
installed `dvc --version` disagrees -- the same guard
`examples/wandb/generate.py` established first. Or run `./generate.sh`,
which does the same thing, checks that version, and then deletes
`.dvc/cache`, `.dvc/tmp` and `.dvcignore` before printing what it wrote.

The catalogue at `examples/README.md` lists the other golden paths, and
`examples/sacred/` is the closest sibling to this one: another tracker
example whose `foreach`-like run naming (there, `experiment.name/<n>`)
carries its own family split.
