<!-- checked by tests/test_golden_paths.py -->

# Example: TensorFlow

## What you get

A real, committed Keras training run -- `results/lr_<lr>.{csv,json}` for
four learning rates, plus one TensorBoard event pair -- and the run ledger
reading the CSV/JSON back through nothing more exotic than
`ledger_adapters/generic.py`'s existing `results/*.csv` and `results/*.json`
conventions. `generate.py` trains a two-dense-layer classifier on
scikit-learn's bundled breast-cancer set at `learning_rate` in `{1e-3, 3e-3,
1e-2, 3e-2}`, five epochs each, CPU only. Per arm, `tf.keras.callbacks.
CSVLogger` writes the epoch curve and a plain `json.dumps` call writes the
held-out `accuracy`/`precision`/`recall`/`auc` -- computed with
`sklearn.metrics`, not `tf.keras.metrics`, because Keras's streaming metric
objects report a running batch average rather than one clean number against
the held-out set, and the two disagree in the third decimal place. No new
reader was needed: `results/*.json` and `results/*.csv` are already the
generic adapter's oldest conventions.

## Prerequisites

`none — pure local computation`

`results/` and `tb/` are already committed here, so `run.sh` below needs
nothing installed to read them. Regenerating the fixture is a separate,
explicit step that needs TensorFlow -- see *Next*.

## Run it

```bash
uv run attest runs scan --root .
uv run attest runs list
uv run attest runs compare lr --metric auc
```

Relative to this directory.

## What it prints

```
winner: lr_0.03
```

Abridged -- the full run also prints the scan summary (including one
honest diagnostic, see below), the four runs from `attest runs list`, and
the full comparison table before that header line:

```
lr — ranked by auc (higher_is_better)

  arm                                                 auc      n      step  source
  -------------------------------------------- ---------- ------  --------  ------
  lr_0.03                                          0.9911      ?            results/lr_0.03.json
  lr_0.01                                          0.9897      ?            results/lr_0.01.json
  lr_0.003                                         0.9812      ?            results/lr_0.003.json
  lr_0.001                                         0.9444      ?            results/lr_0.001.json

winner: lr_0.03
  caveat: some arms report no sample size, so their weight is unknown
  caveat: the top two arms differ by 0.001323 (0.1%) -- too close to call from these numbers alone
  caveat: each arm is a single run; no seed replication, so this ranking cannot separate configuration from run-to-run variance
```

`attest runs scan --root .` also prints one diagnostic line worth expecting
rather than being surprised by: `no runs in tb: 2 file(s), none in a
readable format`. `scan` treats every subdirectory of `--root` as a
candidate project, and `tb/` -- this example's TensorBoard event
directory -- is one of them; its two `.v2` event files are not a format the
generic adapter reads (see *What it demonstrates*), so it is reported as an
empty project rather than silently skipped. Exit code is still 0.

`winner: lr_0.03` is pinned because it is what the *committed* results/
files say, not a live guarantee: training is seeded
(`tf.random.set_seed`, `np.random.seed`, `random.seed`, `TF_DETERMINISTIC_OPS
=1`) and forced to CPU, which reproduces the same numbers on this machine
with this TensorFlow build, but is not guaranteed bit-for-bit across
different hardware or a different TensorFlow/XLA version -- run.sh and this
README's pin read the checked-in results/, never a fresh run, so that
uncertainty never reaches the test.

The family is `lr` -- see *What it demonstrates* for why that needed a
ledger fix.

## What it demonstrates

**A bare `<token>_<value>` stem has no separate prefix to fall back to, and
`family_of` did not group it.** The four result files are named
`results/lr_0.001.json` .. `results/lr_0.03.json` -- `family_of`'s existing
`_SPLIT` regex already recognises `lr` as a variant-token prefix (the same
regex that turns `dit_small_rope_lr1e-4` into family `dit-small-rope`), but
for a stem that is *only* that token plus its value, stripping the
recognised token empties the whole stem, and the function returned `None`
for all four arms: `attest runs compare lr` failed outright with `no family
'lr', and no run has one`. This example is what surfaced the bug against a
real four-arm sweep, not a synthetic one -- `family_of` was fixed
(`src/attestation/ledger_adapters/generic.py`, tested in
`tests/test_ledger_adapters.py`) so that when stripping the recognised
token leaves nothing behind, the token's own name is the family: `lr_0.001`
groups as family `lr`, `seed_3` would group as family `seed`, and the
existing sweep/series shapes (`dit_small_rope_crossattn`, `eval_step_22000`)
are unchanged.

**CSVLogger's CSV is written for a human, not the ledger, and that is by
design.** `_label_of` requires a column naming each row's arm
(`config_name`, `config`, `name`, `run`, `variant`, `arm`, `label`, `id`)
before a CSV's rows become per-row runs -- built for a sweep table with one
row per arm. A Keras `CSVLogger` file has one row per *epoch* of a single
arm, with no such column, so every row is silently unlabelled and the file
contributes no runs -- the JSON beside it is the one number per arm the
ledger actually reads, consistent with the golden-paths rule that a
tracker's *final* value is a run, not its curve.

**The ledger does not read TensorBoard's event files, and is not meant
to.** `tb/train/` and `tb/validation/` each hold one
`events.out.tfevents.<ts>.v2` file -- binary protobuf, out of scope by the
golden-paths spec. They are committed here only to show the convention
exists and that CSVLogger, not the event file, is the way to make a Keras
run legible to this ledger; `attest runs scan` correctly reports `tb/` as
an empty project rather than erroring on a format it does not understand.

## When it goes wrong

- `attest runs compare lr --metric wer` (a metric name `generate.py` never
  writes) prints every arm as `(none)` and `winner: None`, with a caveat
  naming all four arms as having no `wer` -- the same "arm was never
  evaluated on this metric" honesty every adapter's comparison gives.
- Regenerating without `--with tensorboard` fails at the first
  `model.fit()` call with `tensorflow.python.summary.tb_summary.
  TBNotInstalledError: TensorBoard is not installed, missing implementation
  for tf.summary.scalar` -- `tf.keras.callbacks.TensorBoard` needs the
  separate `tensorboard` package installed alongside `tensorflow-cpu`,
  confirmed by running this for real rather than reading TensorFlow's docs.

## Next

Regenerating the fixture is deliberate and explicit -- it rewrites the
committed `results/` and `tb/` in place, deleting and retraining all four
arms:

```bash
uv run --with tensorflow-cpu==2.20.0 --with tensorboard --with scikit-learn --no-project python generate.py
```

Installing `tensorflow-cpu` and `tensorboard` together is roughly 500 MB
and can take a few minutes on a cold `uv` cache; the committed artifacts
mean nobody else needs to run this just to read the example.

The catalogue at `examples/README.md` lists the other golden paths, and
`examples/dvc/` and `examples/sacred/` are the closest siblings to this
one: each commits a real tool's output the ledger reads with no dependency
on that tool's own package (DVC) or with a hand-written reader for its
directory convention (Sacred) -- this path needed no new reader at all,
only the `family_of` fix above.
