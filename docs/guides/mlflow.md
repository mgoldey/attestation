# MLflow runs with provenance

How do I get MLflow runs into the ledger with what MLflow does not record --
whether the tree was dirty, which build of torch or scikit-learn ran? Install
`attestation-schema[mlflow]` where you train: every `mlflow.start_run` then
carries `attestation.*` tags with the code's git state and the identity of
each key package the run imported, observed in the run's own process. Then
`attest mlflow sync` writes one core record per ended run from your
`mlruns/` directory, with those identities in its `code` and `dependencies`,
and the ledger reads the records as native runs.

## Tag runs as they start

```bash
pip install "attestation-schema[mlflow]"   # in the training environment
python train.py                             # any script that calls mlflow.start_run
```

Nothing to import: MLflow finds the provider through its
`mlflow.run_context_provider` entry point. Each run gets:

| tag | what it says |
|---|---|
| `attestation.format` | `attestation.mlflow-tags/1`, the layout of these tags |
| `attestation.producer` | the attestation-schema version that wrote them |
| `attestation.code` | the script's checkout at run start: HEAD, **dirty**, origin (credentials stripped), as a core `code` object |
| `attestation.dependencies` | python and every already-imported key package (mlflow, numpy, pandas, scikit-learn, scipy, torch, tensorflow, keras, jax, transformers, lightning, xgboost, lightgbm, catboost) with the identity rule's provenance |
| `attestation.error` | only when something could not be observed, or did not fit MLflow's tag size limit |

Name more packages with `ATTESTATION_MLFLOW_PACKAGES="my-lib,other-dist:other_module"`.
Turn the provider off with `ATTESTATION_MLFLOW_TAGS=0`. It is on by default
wherever attestation-schema and mlflow are both installed, because an opt-in
switch is one nobody sets; what keeps it safe is what it does: it imports
nothing the run did not, makes no network request, bounds its git calls to
2 seconds, writes only `attestation.*` tags (a tag you pass to `start_run`
still wins), and never raises -- a probe that fails is reported in
`attestation.error` on the run instead.

The provider runs on MLflow's fluent API (`mlflow.start_run`, autologging).
A run created with `MlflowClient().create_run` asks no provider, so it gets
no tags, and its record says `dependencies: null` (not observed).

## Write the records

```bash
attest mlflow sync                          # ./mlruns -> ./mlflow-records/run-<run_id>/
attest mlflow sync --tracking-dir path/to/mlruns --out records
attest runs scan .                          # the ledger reads them as native records
```

`sync` reads the file store directly (mlflow need not be installed where
you run it) with the same mapping the ledger's MLflow adapter uses:
experiment = family, FINISHED/FAILED/KILLED = complete/failed/cancelled,
params = config verbatim, every logged point in `attestation/history.jsonl`,
MLflow's own latest value per metric, artifacts by sha256. Runs still
running are skipped (a record is written once, when a run ends).

What the tags become in the record:

- **code**: the provider's object (with `dirty`) when it and MLflow's own
  `mlflow.source.git.commit` name the same commit; `unknown`, both commits
  named, when they disagree; MLflow's alone (`inferred`, `dirty: null`) for
  a run started without the provider.
- **dependencies**: each with the provenance the identity rule gave it *in
  the run's process at run start* -- `verified` stays `verified` (the
  imported module agreed with an immutable release, or carried a build
  stamp), `inferred` stays `inferred`. `sync` transcribes; it did not
  observe, so it neither raises nor lowers a claim, and each `evidence` says
  which tag it came from. No tag: `null`.

**Re-running is safe.** A run whose record is already there byte for byte
is left alone (`unchanged`), and so is one that differs only in which
attestation version wrote it. A run that changed upstream -- a late metric
point, a tag edited, a run re-terminated -- is **refused** with the fields
that differ, and `sync` exits 1 without touching the old record; `--force`
replaces it. Every directory is checked against the conformance suite
before it is moved into place.

**Read once.** When the records sit inside the project the ledger scans
(the default `./mlflow-records` does), the core adapter imports each run and
the MLflow adapter skips the store's copy of it. The run's ledger name is
then its run id, not `<run name>/<id8>`: update a claim that cites the old
name, or write the records outside the project.

## When it goes wrong

- **`refused ... --force replaces it`**: the run in `mlruns/` no longer
  matches its record. Check what changed (the message lists the fields), then
  `--force` if the new state is the one you want recorded.
- **`unreadable`**: MLflow's own client could not read the run either (a
  malformed metric line, a `meta.yaml` naming another run). Nothing is
  written for it; the ledger quarantines it the same way.
- **A record's notes say `the run-context provider reported: ...`**: a probe
  failed at run start; the rest of the tags were still written.
- **No `attestation.*` tags on a run**: the provider is not installed in the
  training environment, `ATTESTATION_MLFLOW_TAGS=0` is set, or the run was
  created with `MlflowClient` rather than `mlflow.start_run`.

Design: `docs/superpowers/specs/2026-11-20-mlflow-plugin-design.md`; the
store mapping itself: `docs/superpowers/specs/2026-11-20-mlflow-adapter-design.md`.
