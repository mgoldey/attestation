<!-- checked by tests/test_golden_paths.py -->

# Example model-registry

## What you get

Two model versions traced back, by sha256, to the runs that trained them and
on to their training data — and a claim about a model checked against that
chain. One is a real MLflow registered model (`iris-clf` version 1, alias
`champion`) in a local file store; the other is a tiny GPT-2 saved with
transformers' `trainer.save_model()`, trained with `attestation_schema`'s
`AttestationCallback`. Every file here was written by the real libraries
(`generate.py`); nothing contacts a registry or the Hub.

## Prerequisites

`none — pure local computation`

Regenerating the fixtures needs more (see `generate.py`'s docstring):
MLflow's sklearn flavor with pandas and skops for the `mlflow` part, and
transformers + torch + `attestation-schema[hf]` for the `hf` part. Reading
them needs neither.

## Run it

```bash
uv run attest model trace hf/model
uv run attest model trace models:/iris-clf@champion
uv run attest runs scan --root .. --project model-registry
uv run attest claims DRAFT.md
uv run attest model trace models:/iris-clf/2 || true
```

Relative to this directory (`run.sh` does `cd "$(dirname "$0")"` first).
`--search` defaults to the current directory.

## What it prints

```
model hf/model
<- produced by run toy-seq-gpt2/toy-seq-gpt2-0 (run:43cc10f97fe5)  [output: 4/4 files]
model trace: verified
```

The saved model's four files are the four files of the last checkpoint the
callback recorded as `output` data, byte for byte; that run read the
`toy-seq` dataset by content digest, and the dataset names the `prep` run
that wrote it by record digest. The MLflow version ends differently:

```
   alias champion -> version 1
model trace: unverified links
```

with one `UNVERIFIED [untraced-training-data]` line: MLflow records a
dataset input by name and MLflow's own digest, never a sha256 of its bytes,
so the run reaches no dataset record (`data/iris/1/` is there, unlinked).
The claim check then prints three `supported` verdicts, one `untraced` (the
third claim credits the GPT-2 with the iris run's accuracy), and two
`model` lines with each chain:

```
3 claim(s): 3 supported, 1 untraced
```

and the last command shows a version that is not there:

```
BROKEN [unresolved-model] model:models:/iris-clf/2 at ref: mlruns: 'iris-clf' has no version 2
```

The run ids and digests are stable only for the committed fixture;
`generate.py` makes new ones.

## What it demonstrates

**A model version is linked to its run by digest, never by name.**
`models:/iris-clf/1` resolves through the registry's own files —
`mlruns/models/iris-clf/version-1/meta.yaml` names `storage_location`
(the logged model's `artifacts/`) and `run_id` — but the run is found
because its core record (`mlflow-records/`, written by `attest mlflow sync`)
lists those five files, by sha256, as `output` data. MLflow 3 keeps a logged
model outside the run directory; the run's `outputs/<model_id>/meta.yaml`
is MLflow's statement that the run produced it, and the ledger's MLflow
reader now reads those files as the run's outputs. The registry's `run_id`
only diagnoses: if it disagreed with the digests the trace would say
`registry-run-mismatch`, broken.

**A saved directory is a version too.** `hf/model` has no registry
coordinates, so it is identified by content alone; `hf:<org>/<name>@<rev>`
resolves the same way in a local hub cache (`models--org--name/
snapshots/<commit>/`) when one is under `--search`. Downloading one needs
the network, which this example does not use.

**A claim names its model with `model=`.** `attest claims` and
`runs.claims_check` trace it: a broken chain, or a claim whose run neither
output nor read any file of the model, is `untraced` — a lint like
`uncited`, never a contradiction. Drafts without `model=` print exactly
what they did before.

**One quirk shown, not fixed here.** The `prep` run lists `sequences.csv`
as an output, and `resolve_lineage` matches a run's data entries whatever
their role, so the tree shows the dataset once more under `prep`
("shown above"). The spec records it for a later schema release.

## When it goes wrong

- `unresolved-model` with "no MLflow store (mlruns/) under the search
  paths": pass `--search` the directory holding `mlruns/` (or a workspace
  whose projects hold one).
- `no-producing-run`: no record lists the version's files as outputs. For
  MLflow, run `attest mlflow sync` (or keep the store under `--search`); for
  a Trainer, the callback records checkpoints, so save the final model at a
  checkpoint step, or the saved files will not match any.
- `partial-output`: some files match and some do not — usually a final
  `save_model` after training changed weights the last checkpoint did not
  hold.

## Next

`docs/superpowers/specs/2026-11-20-model-registry-linkage-design.md` for the
decisions; `examples/mlflow-records/` for the MLflow mapping; `attest runs
lineage` for the run → dataset half alone.
