# attestation-schema

The core run record that experiment producers write and
[attestation](https://github.com/mgoldey/attestation) reads:
`attestation.run/1`, **frozen 2026-11-19**: a record valid today stays valid,
and any change -- an additive optional field included -- is a new version
(`attestation.run/2`) that readers dispatch on. Standard library only.

```python
from attestation_schema import (
    RunRecord, Producer, CodeIdentity, Metric, HistoryWriter, write, read, read_history,
)

run_dir = "runs/run-3f2a9c01b7de"
with HistoryWriter(run_dir) as hist:            # streams attestation/history.jsonl
    loss = hist.series("loss", split="eval")
    for step, value in [(0, 0.9), (500, 0.4), (1000, 0.25)]:
        hist.point(loss, step, value)           # at=format_timestamp(...) if known

record = RunRecord(
    run_id="run-3f2a9c01b7de", family="lr-sweep", arm="lr=1e-3", status="complete",
    started_at="2026-11-19T10:00:00Z", finished_at="2026-11-19T10:42:13Z",
    producer=Producer(name="train.py", version=None),
    code=CodeIdentity(git_sha=None, dirty=None, repo=None, version=None,
                      provenance="unknown", evidence=None),
    config={"lr": 0.001}, data=[],
    metrics=[Metric(name="loss", value=0.25, direction="min", step=1000, split="eval",
                    unit=None, labels={})],
    history=hist.ref,      # {path, sha256, series, points}; None = no curve recorded
    seeds={"python": 0},
    dependencies=None,     # None = could not observe the environment; [] = observed, none
    ext={},
)
write(record, run_dir)     # -> runs/run-3f2a9c01b7de/attestation/run.json
assert read(run_dir) == record          # never opens history.jsonl
curves = read_history(run_dir)          # verifies the sha256, then parses
```

- `dumps`/`write` are the only sanctioned writers: validate, then RFC 8785
  canonical JSON, so the same record is always the same bytes.
- `loads`/`read` refuse anything non-conforming with a `RecordError` naming
  the path (`metrics[2].direction`); nothing is coerced or defaulted. They
  dispatch on `schema`, so later versions can still read older records.
- **Curves never go in `run.json`.** `metrics` holds final values; logged
  curves go to the sidecar `attestation/history.jsonl` (`attestation.history/1`:
  JSON Lines, one canonical line per series declaration or point, streamable
  and deterministic), written with `HistoryWriter` (incrementally, finalized
  atomically on close) or `write_history(run_dir, series)`, and bound to the
  record by sha256. `read_history` raises `HistoryMissingError`,
  `HistoryDigestError` or `HistoryFormatError` (all `RecordError`). Write
  order: outputs and history -> `layout.write_manifest` -> `write`.
- `ext.<namespace>` carries producer data the core does not model, untouched.
- `attestation_schema.layout` fixes the run directory (`attestation/run.json`,
  `attestation/state.json`, `attestation/digests.json`,
  `attestation/history.jsonl`, `steps/NN-<name>/`, `logs/`);
  `attestation_schema.jobs` is the job-backend `Protocol`, and
  `attestation_schema.state` is what every backend shares: `state.json`
  (`attestation.state/1`: `read_state`, `init_state`, `transition`), its
  cross-process lock (`state_lock`, POSIX `flock`; refused on Windows), and
  `settle` -- end a job exactly once, an existing record deciding, writing
  `terminal_record` for a job that died without one. A job's record carries
  its `JobSpec`'s `run_id`, `family`, `arm`, `config` and `ext` verbatim.
- `attestation_schema/schemas/run-1.schema.json` (the record),
  `history-1.schema.json` (one sidecar line) and `state-1.schema.json`
  (`state.json`) are JSON Schemas for other languages: necessary, not
  sufficient — the Python reader is the reference. They are frozen and
  pinned by digest. `dataset-1.schema.json` is a draft (below).
- Package version 0.2.0 (`attestation_schema.__version__`): the formats are
  frozen; the Python API may still change in a 0.x minor release until the
  job-state API has a consumer. 0.2.0 is the first release with datasets,
  lineage, conformance, the pytest plugin, the training callbacks and the
  shared helpers below.

## Shared helpers (0.2.0)

One rule for every producer, so the same install gets the same record
whoever writes it:

```python
import sys
import attestation_schema as s

s.package_identity("ferric", "ferric")      # -> Dependency: provenance, source, build_sha
s.package_identity("torch", sys.modules["torch"], name="torch")
s.strip_credentials("git+https://tok@host/r.git")   # -> "git+https://host/r.git"
s.git_state("path/to/checkout")             # -> GitState(sha, dirty, remote); never raises
s.sha256_file("weights.bin")                # chunked
```

- `package_identity(dist, module=None, *, name=None, probe_git=True)`: a
  `__build__ == {"git_sha", "dirty"}` stamp on the module is `verified`; so
  is a VCS install with a commit, and an index install (`INSTALLER` pip/uv)
  whose imported `__version__` agrees with the metadata. Everything else
  that names a version or commit is `inferred` (conda/mamba/micromamba,
  archives, directory installs -- probed with git --, a missing `INSTALLER`,
  any other installer), else `unknown`. `source` is in the record's
  vocabulary and credential-free. A module NAME not yet imported is
  imported; pass the module object to avoid that. `dependency_identity(...)`
  is the same rule, pure.
- `strip_credentials(url)` strips exactly what the reader refuses: a
  password for any scheme (`ssh://u:p@`), any user name for a web scheme
  (`https://tok@`, `git+https://tok@`); `ssh://git@host` is kept. On a
  `url:` source it also drops the query string and keeps a `#sha256=`
  fragment.
- Every writer in the package (record, state, history, both manifests,
  dataset) fsyncs and renames atomically through one private path.

Design: `docs/superpowers/specs/2026-11-19-dependency-identity-design.md`.
 
## Conformance

The reader stops at the first problem in `run.json`; a run directory has more
rules than that. The conformance suite checks all of them and reports every
violation with the file and the JSON path it is in: the record reads under
the frozen reader; its bytes are the canonical ones (`write` round-trips them
byte for byte); `digests.json` matches the files and is bound by the record;
the history sidecar matches its digest and format; `state.json` is
well-formed with legal transitions and agrees with the record; a terminal
state has its record; the record matches its `JobSpec` (when you pass one);
every `dependencies[].source` is in the vocabulary; no credentials anywhere
in the schema-owned files; every `ext` key is a valid namespace.

Run it in your CI -- three lines:

```
pip install "attestation-schema[conformance]"
pytest -p attestation_schema.pytest_plugin --attestation-runs 'runs/*'
```

One test per run directory (marker `attestation_conformance`). A pattern
that matches no run directory **fails**: a typo never turns the suite into a
green no-op. Or configure it once:

```toml
[tool.pytest.ini_options]
addopts = "-p attestation_schema.pytest_plugin"
attestation_runs = ["runs"]          # used when pytest runs without explicit paths
```

A failure names the record and the path inside it:

```
runs/r1/attestation/run.json at $.config.tracking_uri: [credentials] a URL with credentials (user:password@); strip them
```

To check a record your test just produced (and, for a job, the spec it ran
with), use the fixture:

```python
def test_my_producer_writes_a_conforming_record(tmp_path, attestation_conformance):
    run_my_pipeline(out=tmp_path)
    attestation_conformance(tmp_path, spec)        # spec: the JobSpec, or omit it
```

Without pytest: `attestation_schema.conformance.check_run_dir(run_dir)`
returns a `Report` (`.ok`, `.violations`, `.format()`), and
`assert_conformant(run_dir)` raises `ConformanceError`. The plugin is
**opt-in** (`-p`, no `pytest11` entry point), so a suite that merely has this
package installed is unchanged, and pytest is only the `[conformance]`
extra's: the package itself still depends on nothing. attestation's ledger
runs the same checks on every scan and quarantines any record that fails.
Golden fixtures (a conforming record for each shape, and one record broken
per rule) are in `tests/data/conformance/`.

## Datasets and lineage (`attestation.dataset/1`, DRAFT)

A dataset has its own record kind, so `attestation.run/1` stays frozen. It is
a **draft**: not frozen until two producers have written it, and a draft
change may stop older draft records from reading (pin this package by commit
meanwhile). The directory:

```
datasets/<dataset_id>/<version>/      immutable once dataset.json exists
├── runs.parquet, split-scaffold.json, ...
└── attestation/
    ├── digests.json                  attestation.digests/1 over every file outside attestation/
    └── dataset.json                  the record, written last
```

The **content digest** is the sha256 of `attestation/digests.json`'s bytes
(`sha256sum` checks it), so a training run names the dataset in `data[]`
with run/1's own meaning of `sha256`:

```python
import attestation_schema as s

content = s.write_dataset_manifest(ds_dir)               # files first, then this
record = s.DatasetRecord(
    dataset_id="smeltery-benzoic-ddE", name=None, version="1",
    created_at=s.format_timestamp(now), producer=s.Producer("smeltery export", "0.4.0"),
    code=code, content_sha256=content, files=s.dataset_files(ds_dir),
    rows=4, columns=None,                                   # None = not described
    splits=[s.Split("scaffold", "split-scaffold.json", split_sha, "scaffold", 0)],
    source_runs=[s.SourceRun(r.run_id, r.family, s.record_digest(r)) for r in runs],
    license="CC-BY-4.0",                                    # SPDX; None = not stated
    ext={},
)
s.write_dataset(record, ds_dir)                          # refuses without the manifest
data = s.dataset_lineage_entries(record, split="scaffold")  # the training run's data[]

lineage = s.resolve_lineage(training_run, datasets, runs)   # pure, stdlib
lineage.verified, lineage.issues, lineage.dependencies("ferric")
```

`resolve_lineage` links only by digest (a `data[]` sha256 equal to a content
digest, a split file or a file; a `source_runs[].record_sha256` equal to a
run's `record_digest`) and reports every broken or unverified link
(`missing-source-run`, `dataset-digest-mismatch`, `undigested-source-run`,
`ambiguous`, ...). The conformance checker and the plugin accept dataset
directories wherever they accept run directories (`check_dir`,
`check_dataset_dir`). `attest runs lineage RUN --search DIR` prints the
chain. Design: `docs/superpowers/specs/2026-11-19-dataset-records-design.md`.

## Training callbacks (Hugging Face Trainer, PyTorch Lightning)

Optional, behind extras; `import attestation_schema` imports no framework.

```
pip install "attestation-schema[hf]"          # transformers, torch, accelerate
pip install "attestation-schema[lightning]"   # lightning
```

```python
from attestation_schema.integrations.huggingface import AttestationCallback

# `with`: HF calls no hook when train() raises, so the context manager is
# what writes a `failed` (Ctrl-C: `cancelled`) record on the way out.
with AttestationCallback(datasets=[("datasets/toy/1", "scaffold")], family="lr-sweep") as cb:
    Trainer(model, args, train_dataset=train, callbacks=[cb]).train()
print(cb.record_path)   # <output_dir>/attestation/run-<run_id>/attestation/run.json
```

```python
from attestation_schema.integrations.lightning import AttestationCallback

L.seed_everything(7)
cb = AttestationCallback(datasets=["datasets/toy/1"])
L.Trainer(callbacks=[cb]).fit(model, datamodule=dm)   # on_exception writes `failed`
```

At the end of training, on global rank 0 only, each writes a conforming
`attestation.run/1` record (curves -> `digests.json` -> `run.json`):

- **seeds** -- `TrainingArguments.seed` (applied by `Trainer.__init__`) and
  `data_seed`; Lightning's `seed_everything` seed, or `null` when it never ran
  (never `{}`: the script may have seeded torch itself);
- **data** -- each dataset you pass (a dataset directory, its `dataset.json`
  or a `DatasetRecord`, optionally `(dataset, split)`) by content and split
  digest, so `resolve_lineage` links the run to it. Nothing is auto-detected:
  with no `datasets=`, `data[]` names `train` (and `eval`) with `sha256: null`;
- **checkpoints** -- every file of every checkpoint this run saved, by
  sha256, role `output` (`trainer.save_model()` after `train()` runs after the
  record and is not bound: save checkpoints to bind the weights);
- **metrics** -- final values in `metrics`, every logged value in the history
  sidecar; a direction only where you declared one (`metric_for_best_model`,
  `ModelCheckpoint`/`EarlyStopping` `mode`, or `directions=`);
- **config** -- `TrainingArguments.to_dict()`, or the module's and
  datamodule's `hparams` plus the trainer's loop settings; anything the
  conformance credential rule would flag is replaced by `<redacted>`;
- **code** -- the training script's git HEAD and dirtiness (`inferred`), or
  `unknown`; **dependencies** -- python, torch, transformers/accelerate or
  lightning, each by `package_identity` (above);
- **ext** -- world size, global step, removed checkpoints, the error.

`run_dir=` overrides the default `<output_dir>/attestation/run-<run_id>/`
(Lightning: `trainer.log_dir`); a directory that already holds a record is
refused before training starts. Design:
`docs/superpowers/specs/2026-11-19-training-callbacks-design.md`.

## Releasing

Tag `schema-v<version>` (the version in this package's `pyproject.toml`, which
the workflow checks), independent of attestation's `v<version>` tags:

```
git tag schema-v0.2.0 && git push origin schema-v0.2.0
```

`.github/workflows/release.yml` builds the sdist and wheel, runs this
package's tests on CPython 3.11 against the wheel, and publishes to PyPI
through trusted publishing (environment `pypi`, no token). Release it before
an attestation release whose `attestation-schema>=X` floor is new.

Design: `docs/superpowers/specs/2026-11-19-core-run-record-design.md` in the
attestation repository.
