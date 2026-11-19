# Training callbacks: Hugging Face Trainer and PyTorch Lightning write core records

**Date:** 2026-11-19
**Status:** implemented 2026-11-19 (roadmap item 34).
**Depends on:** `2026-11-19-core-run-record-design.md` (the frozen
`attestation.run/1`, `attestation.history/1`, `attestation.digests/1` and the
run-directory layout), `2026-11-19-dataset-records-design.md` (the dataset
record, its content digest and `resolve_lineage`),
`2026-11-19-conformance-suite-design.md` (`check_run_dir`, which every record
written here must pass).
**Changes no format.** Nothing frozen moves and `tests/test_frozen.py` is
untouched: the callbacks are a *producer*, writing through `write_history`,
`layout.write_manifest` and `write` like any other. Everything here is new,
optional Python API in `attestation-schema` (a 0.x minor addition) behind two
new extras.

## Problem

The lineage the stack wants to verify ends `dataset -> training run -> claim`,
and the dataset spec's freeze criteria need "a real training run's record
that `resolve_lineage` links end to end". Today the only way to get one is to
build a `RunRecord` by hand at the end of a training script, and every
training script would then re-derive the same things -- seeds, the
checkpoint's digest, the curves, the framework's version and where it came
from -- and get some of them subtly wrong (a `"complete"` record for a run that
raised; a checkpoint digested before it was finished writing; a seed nobody
applied). Most training code runs inside one of two loops, the Hugging Face
`Trainer` and PyTorch Lightning's `Trainer`, and both take callbacks. A
callback is the one place that sees the run start, every logged value, every
checkpoint save and the end, without the training script doing anything but
passing it in.

## What a record holds, and where each field comes from

| Field | Hugging Face `Trainer` | Lightning `Trainer` |
|---|---|---|
| `run_id` | the `run_id=` argument, else a fresh `uuid4().hex[:12]` | same |
| `family`, `arm` | the arguments, default `null` -- never guessed from `run_name` or an experiment name | same |
| `status` | `complete` at `on_train_end`; `failed` / `cancelled` (below) | `complete` at `on_fit_end`; `failed`/`cancelled` from `on_exception` |
| `started_at`, `finished_at` | `on_train_begin`, the end; `format_timestamp` | `on_fit_start`, the end |
| `producer` | `attestation-schema hf-trainer`, the package's installed version | `attestation-schema lightning` |
| `code` | `code=` argument, else the training script's git checkout (below) | same |
| `config` | `TrainingArguments.to_dict()` (HF's own serialization: enums as values, `*_token` already replaced by `<NAME_TOKEN>`) | `{"hparams": module.hparams, "datamodule": datamodule.hparams, "trainer": {max_epochs, max_steps, ...}}` |
| `data` | the datasets given (below), checkpoint files (role `output`), the manifest | same |
| `metrics` | the last logged value of every key, at its `global_step` | `trainer.callback_metrics` at the end (Lightning's own "current" values) |
| `history` | every `on_log` value, streamed through `HistoryWriter` | every value Lightning sends to its loggers |
| `seeds` | `{"seed": args.seed}` (+ `"data_seed"` when set) | `{"seed_everything": int(PL_GLOBAL_SEED)}` when `seed_everything` ran, else `null` |
| `dependencies` | python, torch, transformers, accelerate | python, torch, lightning |
| `ext` | `hf-trainer`: world size, global step, epoch, ... | `lightning`: world size, global step, epoch, `seed_workers`, ... |

### Seeds: what was applied, not what was configured

`Trainer.__init__` calls `set_seed(args.seed)` (or `enable_full_determinism`)
unconditionally (transformers 5.23, `trainer.py`), so `args.seed` is a seed the
trainer *applied*, not a default it merely carries: it is recorded as
`{"seed": 42}` even when the user never wrote 42. `args.data_seed` is recorded
as `"data_seed"` only when set; `None` means the sampler used `seed`, which is
already listed. The names are TrainingArguments' own.

Lightning's `seed_everything(n)` exports `PL_GLOBAL_SEED`, so the callback
reads it at `on_fit_start`: present = `{"seed_everything": n}` (and
`PL_SEED_WORKERS` goes to `ext.lightning.seed_workers`); absent = `null`, **not
`{}`**. `{}` claims no seed was set; the script may have called
`torch.manual_seed` itself, which no callback can observe. `seeds=` overrides
both.

### Datasets: given, never guessed

`datasets=` takes dataset directories, `dataset.json` paths or
`DatasetRecord`s (optionally `(dataset, split_name)` pairs), and the record's
`data[]` gets exactly `dataset_lineage_entries(record, split=...)` for each:
the content digest under the `dataset_id` (role `train`, or `role=`), and the
split file by its digest (role `split`). That is what `resolve_lineage` links
by `content` and `split`. `data=` takes further `DataDigest` entries verbatim,
for a digest held without a record.

**No auto-detection.** A `Dataset` object or a `DataLoader` does not know the
bytes it was built from (an HF `datasets` cache fingerprint is a hash of a
transformation history, not of bytes on disk; a `TensorDataset` has no
origin), so any digest the callback computed would be of the wrong thing or
of nothing -- the "measuring the artifact, not the system" failure.

**When none is given**, `data[]` says what the callback did observe -- that
the run read training data (and, if an evaluation ran, evaluation data) --
with the bytes undigested: `{name: "train", sha256: null, role: "train"}`
(and `{name: "eval", sha256: null, role: "eval"}`). Empty would be false: the
MLflow producer found (core spec, Producer findings 1.2) that a record with no
data entry says the run read *no* data. A `null` sha256 is run/1's own
"named but not digested", and the lineage resolver treats it as "not linked",
raising `undigested-data` only if the name happens to be a known `dataset_id`.
Passing `datasets=` (even `[]`) replaces both placeholders: the caller has then
stated the data.

### Checkpoints: the files this run saved

- **HF:** `on_save` fires after `_save_checkpoint` has written
  `<output_dir>/checkpoint-<global_step>/` (model, optimizer, scheduler, RNG
  state, `trainer_state.json`) and rotated old ones away. The callback records
  each folder saved *during this run*; at the end, every regular file under
  each folder that still exists is one `data[]` entry, `name` =
  `checkpoint-<step>/<file>` (relative to `output_dir`), `sha256` of its
  bytes, role `output`. Folders rotated away by `save_total_limit` have no
  bytes to digest and are listed by name in `ext.hf-trainer.checkpoints_removed`
  rather than as undigested outputs (their absence is the trainer's own
  decision, not a gap). A checkpoint folder left in `output_dir` by an
  earlier run is never listed.
- **Lightning:** every `ModelCheckpoint` in `trainer.checkpoint_callbacks`:
  `best_k_models`, `best_model_path` and `last_model_path`, each file that
  exists, named relative to `trainer.log_dir`, role `output`.
- **Not bound:** `trainer.save_model()` after `train()` returns runs after
  the record is written (HF's `on_train_end` fires inside `train()`), so the
  final export is not in the record. Saving checkpoints (`save_strategy`) is
  how its bytes get bound; the README says so. The callback never saves a
  model itself -- a provenance hook that writes gigabytes is not thin.

The **manifest** (`digests.json`) is written before the record and bound with
role `manifest`, as the layout requires. Its run directory holds nothing but
`attestation/` (below), so it lists no files -- the MLflow producer's shape
(core spec, Producer findings 1: "the manifest is still written, so every run
directory has one shape"). Outputs are bound by digest in `data[]`.

### Metrics and curves

- **HF:** `Trainer.log` is the one funnel every value passes through, and
  `on_log` receives exactly that dict. Every numeric, non-boolean value is a
  history point: series `(key, split=null, labels={})`, `step` =
  `state.global_step`, `at` = now. Keys stay verbatim (`loss`, `eval_loss`,
  `train_runtime`, `epoch`): `eval_` is a configurable prefix
  (`metric_key_prefix`), not a statement of which split, and splitting it off
  would collide the final `train_loss` with the `loss` curve. `metrics` = each
  series' last point.
- **Lightning:** what Lightning *logged* is what it sends to its loggers, so
  the callback appends a small internal `Logger` to `trainer.loggers` at
  `setup` -- **only when the trainer already has a logger**, and always last.
  Lightning resolves `trainer.log_dir` and `ModelCheckpoint`'s directory from
  `loggers[0]`, so appending to an empty list would move the user's
  checkpoints; and with no logger Lightning logs nothing anywhere
  (`log_metrics` returns early), so `history: null` ("no curve was recorded")
  is then the truth. `metrics` = `trainer.callback_metrics` at the end; `step`
  is the step of the matching history point when that point holds the same
  value, else `null` (Lightning does not say which step an epoch aggregate
  belongs to).
- **Direction is never inferred, but is read where the user declared it:**
  HF's `metric_for_best_model` with `greater_is_better` (which HF itself
  fills from the name when unset -- `min` for a `*loss` -- and then uses to
  pick the best checkpoint, so it is the criterion the trainer acted on), and
  Lightning's
  `ModelCheckpoint(monitor=..., mode=...)` / `EarlyStopping(monitor=...,
  mode=...)`. Two declarations that disagree leave the direction `null`.
  `directions=` overrides.

### Code identity

`code=` takes a `CodeIdentity`. Otherwise the callback asks git about the
directory of the running script (`sys.argv[0]`, or the working directory in a
notebook): `git rev-parse HEAD`, `git status --porcelain
--untracked-files=no`, `git config --get remote.origin.url` (credentials
stripped). `provenance: "inferred"`: HEAD at train start names the checkout,
not the bytes Python loaded -- MLflow's `mlflow.source.git.commit` and the
same rung. `dirty` is what `status` said (unknown, `null`, if it could not
run). No checkout, no git: `provenance: "unknown"`, every field `null`,
`evidence` saying why. The evidence never names an absolute path (a path
names a machine).

### Dependencies: observed, with their provenance and source

The modules this run actually imported (in `sys.modules`; a list is
observed, not exhaustive, core spec 5.B) plus Python itself:

- `version` = the imported module's `__version__` -- what ran, not what the
  disk says now. **Measured:** PyPI's torch 2.15.0 wheel reports `2.15.0` in
  its metadata and `2.15.0+cu132` in `__version__`; a first version of the
  agreement rule called that a disagreement and marked every PyPI torch
  `inferred`. A PEP 440 local label only the module reports now agrees (the
  recorded version keeps the label); two different local labels do not.
- `source`, from the distribution's PEP 610 `direct_url.json`: `vcs_info` ->
  `<vcs>+<url>` with `build_sha` = `commit_id`; `archive_info` -> `url:<url>`
  (query string dropped; a `file:` URL is `file`); `dir_info` -> `file`. No
  `direct_url.json` and `INSTALLER` `pip`/`uv` -> `index` (an index install
  is the only pip install that writes no `direct_url.json`); `INSTALLER`
  `conda` -> `conda`; anything else -> `null`. Credentials are stripped from
  every URL before it is written.
- `provenance`: `verified` when the module's `__version__` equals the
  installed distribution's metadata version and the install is not a local
  directory (two independent observations agree, and the install is
  immutable), or for a VCS install with a `commit_id`; `inferred` for a
  directory install (editable or not: its bytes can change after install) or
  when the two versions disagree (both in `evidence`); `unknown` when no
  version can be read.
- Python: `version` = `platform.python_version()`, `verified`, evidence
  `sys.version_info`, `source: null` (a uv-managed, pyenv or system Python
  look the same from inside).

### Distributed training

Only global rank 0 records: HF `state.is_world_process_zero`, Lightning
`trainer.is_global_zero`. Other ranks do nothing at all -- no directory, no
temporary file -- so a shared filesystem sees one writer. World size goes to
`ext.<namespace>.world_size` (`args.world_size`, `trainer.world_size`). Logged
values are rank 0's (HF logs only there; Lightning's are what its own loggers
see, synced or not as the user's `self.log(sync_dist=...)` says).

### Failure

- **Lightning** has `on_exception`: a `KeyboardInterrupt` records
  `cancelled`, anything else `failed`, with the exception's type and message
  in `ext.lightning.error`; then Lightning re-raises (or exits). Curves logged
  so far are kept (they are what the run did).
- **HF has no exception hook** (transformers 5.23: `on_train_end` is not
  called when the loop raises). The callback is therefore also a context
  manager: `with AttestationCallback(...) as cb: Trainer(..., callbacks=[cb]).train()`
  writes `failed` (`cancelled` for `KeyboardInterrupt`) on the way out if
  training began and no record was written, and re-raises. Without the
  `with`, an exception leaves no record, and the history's hidden temporary
  file (`attestation/.history.jsonl.<pid>.<id>.tmp`) stays in the run
  directory -- no reader opens it, and a directory with neither `run.json`
  nor `state.json` is not a run (`layout.find_runs`). The README leads with
  the `with` form. Rejected: patching
  `Trainer.train` or `sys.excepthook` -- a callback that rewires its host is
  not thin, and an excepthook fires for errors that are not the run's.
- The record for a failed run carries everything observed so far:
  checkpoints saved, curves, the data, seeds. `metrics` holds the last values
  logged (a diverged run's last loss is a result).

### Run directory and layout

`run_dir=` names it; the default is **`<output_dir>/attestation/run-<run_id>/`**
(HF `args.output_dir`; Lightning `trainer.log_dir`), so the record is
`<output_dir>/attestation/run-<run_id>/attestation/run.json`. Considered and
rejected, `run_dir = output_dir` (record at `<output_dir>/attestation/run.json`):

1. the trainer keeps writing to `output_dir` after the record --
   `trainer.save_model()`, a tokenizer, `push_to_hub` -- so a manifest over
   `output_dir` is stale the moment a standard HF script finishes, and
   `check_run_dir` fails it ("the file exists but the manifest does not list
   it") in the user's own CI;
2. HF scripts reuse `output_dir` between runs (`overwrite_output_dir`), and
   `write` refuses an existing record: the second run would fail -- or the
   manifest would bind the first run's checkpoints as the second's outputs;
3. hashing every file under `output_dir` re-reads every checkpoint ever
   saved there.

`run-<run_id>` is the core spec's recommended name and a fresh `run_id` per
execution makes collisions impossible by default; a given `run_dir` that
already holds a record is refused at train start (`FileExistsError`) --
before any training, never after. `layout.find_runs(output_dir)` finds every
run. Write order is the layout's: `HistoryWriter` streams during the run and
closes, then `layout.write_manifest`, then `write`.

## Where the callbacks live

**`attestation_schema.integrations`**, a subpackage of the schema
distribution, behind two extras:

```
pip install "attestation-schema[hf]"          # transformers, torch, accelerate
pip install "attestation-schema[lightning]"   # lightning
```

- `integrations/_core.py` -- stdlib only: the recorder (history, checkpoints,
  manifest, record), the dependency/code/config helpers. Everything testable
  without a framework is here, and runs in the bare 3.11 job.
- `integrations/huggingface.py` -- `AttestationCallback(TrainerCallback)`;
  imports `transformers` at module top.
- `integrations/lightning.py` -- `AttestationCallback(Callback)`; imports
  `lightning.pytorch`.
- `import attestation_schema` imports none of them, and
  `attestation_schema.integrations` itself imports nothing but `_core`; a
  missing framework is an `ImportError` naming the extra to install.

*Why not a separate distribution* (`attestation-integrations`): a third
release line, tag scheme and version for ~600 lines whose only dependency
besides the framework is this package; a producer would install two
distributions to write one record; and `_core` is the same recorder an
MLflow- or Keras-shaped producer would want. The boundary a separate
distribution would give -- the core never imports a framework -- is enforced
here by `tests/test_boundaries.py` instead: the two framework modules may
import their framework (and torch) and nothing else outside the stdlib;
every other module stays stdlib-only. Python 3.11 is kept: nothing here is
3.12-only.

**Cost, measured:** `uv.lock` resolves every workspace member's extras, so
torch, transformers, lightning and their CUDA wheels enter the lock (+478
lines). `uv sync --all-extras` installs the *root* project's extras only, so
neither the dev venv nor the gates job installs them (checked with `uv sync
--all-extras --dev --dry-run`). One side effect on the main environment:
transformers needs `huggingface-hub>=1.31`, and litellm already pulled
`huggingface-hub` in at 1.28, so the lock bumps it 1.28.0 -> 1.31.0 (locked
with `-P huggingface-hub==1.31.0`; an unconstrained lock jumped to 2.4.0 and
moved `tokenizers` too).

## Testing and CI

- `packages/attestation-schema/tests/test_integrations_core.py` (stdlib,
  runs everywhere): dependency identity for each PEP 610 shape (VCS,
  archive, editable and plain directory, none + pip/conda/unknown installer,
  version mismatch), credentials stripped from every URL, git code identity
  in a real repository (clean, dirty, no repository), config made JSON-able,
  checkpoint naming. Each guard was mutated to confirm it fails.
- `test_integrations_hf.py` / `test_integrations_lightning.py`: a **real**
  tiny training run on CPU (a linear model, a few steps), reading its data
  from a dataset directory the test writes with `write_dataset`. Each record
  passes `conformance.check_run_dir`; `resolve_lineage(record, [dataset])`
  is `verified` with a `content` edge (and a `split` edge); the checkpoint
  digests equal the files on disk; seeds, curves, directions, dependencies;
  a run that raises writes a conforming `failed` record; a non-zero rank
  writes nothing. They skip (`pytest.importorskip`) without the extra --
  unless `ATTESTATION_REQUIRE_INTEGRATIONS=1`, which turns the skip into a
  failure, so the CI job that installs the extras cannot go green by
  skipping.
- **CI: a new job, `schema-integrations`** (CPython 3.11, the package's
  floor), installs the built wheel with `[hf,lightning,conformance]` and torch
  from PyTorch's CPU index, and runs the integration tests with
  `ATTESTATION_REQUIRE_INTEGRATIONS=1`. *Not added to `schema-py311`*: that
  job's evidence is that the package works with **nothing** installed, and
  torch in its venv would let a stray top-level import pass unnoticed; there
  the framework tests skip, which is itself the check that the core does not
  need them. Not in `gates` either: torch and its CUDA wheels are ~2.5 GB from
  PyPI on Linux, on each of four matrix cells.

## Open questions

1. **`trainer.save_model()` after `train()`.** Not bound (above). If users
   want the final export in the record, the answer is an explicit
   `cb.bind_outputs(path)` before the record is written, or writing at
   `on_save` of the final export -- deferred until a user asks.
2. **Ranks' own curves.** Only rank 0's view is recorded. A per-rank series
   would be `labels={"rank": "3"}`; deferred until a reader needs it.
3. **The dataset freeze.** These callbacks are the "real training run" side
   of the dataset spec's freeze criteria, but a test run is not a producer
   finding; the criteria still wait for a real consumer.
