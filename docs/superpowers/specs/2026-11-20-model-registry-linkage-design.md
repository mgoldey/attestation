# Model registry linkage: a claim about a model traces to its training data

**Date:** 2026-11-20
**Status:** implemented 2026-11-20 (roadmap item 36). No schema change: no new
record kind, no new `ext` namespace, nothing in `attestation-schema`.
**Depends on:** `2026-11-19-core-run-record-design.md` (`attestation.run/1`,
frozen: `data[]` with `role` and `sha256`), `2026-11-19-dataset-records-design.md`
(`resolve_lineage`, links only by digest), `2026-11-19-training-callbacks-design.md`
(checkpoint files recorded as `output` data), `2026-11-20-mlflow-adapter-design.md`
and `2026-11-20-mlflow-plugin-design.md` (the store reader, `attest mlflow sync`),
`2026-11-19-figure-provenance-design.md` (the pattern for a lint beside claims).
**Not this:** contacting a registry or the Hub (offline guarantee), a
`runs.model_trace` MCP tool, a base-model ("fine-tuned from") link, MLflow
tracking-server/sqlite registries, writing anything into a registry.

## Problem

A draft says "`iris-clf` v1 reaches 0.97 accuracy". The claim checker can
verify 0.97 against a run, but nothing connects the *model* the sentence is
about to that run, or the run to what it trained on. The stack already has
every arrow but one:

```
claim ──► model version ──► its files ──► training run ──► data[] ──► dataset ──► source runs
          (registry name)      ?             (run/1)        (lineage.resolve_lineage, by digest)
```

A registry (MLflow's model registry, the Hugging Face Hub) maps a NAME and a
version to a directory of files. A training run's core record lists files it
wrote as `data[]` entries with role `output` and a sha256 -- the HF/Lightning
callbacks record every checkpoint file that way. The missing arrow is
"version files → the run that output them".

## Decision 1: representation -- a resolver over what is on disk, no new record

Three options were weighed.

**(a) A new draft record kind, `attestation.model/1`** (like dataset/1: a model
directory with `digests.json` of its weight files and `model.json` naming
registry coordinates and the training run's record digest). Rejected for now:

- The registry already IS a record of the version: MLflow's
  `models/<name>/version-<n>/meta.yaml` names the version, its
  `storage_location` and its `run_id`; a Hub snapshot is named by its commit.
  A second record would restate it and could disagree with it.
- Nobody produces one. The dataset spec's own principle -- "a schema no
  producer has written to is a guess" -- applies doubly to a kind invented
  for this task; its freeze criterion (two producers) could not be met.
- It would be a schema-package addition, shipping in a later
  attestation-schema release than 0.2.0 (being tagged from current main),
  and attestation would need a floor bump to rely on it.

**(b) Registry coordinates under `ext.<namespace>` in the training run's
record.** Rejected: registering a version happens *after* the run's record is
written and bound by digests (the callback writes at train end; a version
is created later, often by someone else, often several times). The run
cannot know its version number, and rewriting its record would change its
digest -- breaking every dataset or downstream record that names it.

**(c) A pure resolver: read the registry's own files, link by digest.**
Chosen. `ledger_adapters/registries.py` turns a reference into a directory of
files using only local files; `model_trace.py` hashes those files and finds
the run whose record lists them as outputs; `resolve_lineage` does the rest.
Nothing is written, no format is added, and every link is a sha256 equality
between bytes on disk and a digest a record bound earlier. The registry's
run id is a NAME and only diagnoses (Decision 4).

If a producer later wants to publish a model version *outside* any registry
with its provenance attached, (a) is the right shape, and this resolver
would read it as one more registry. Recorded, not built.

## Decision 2: references

| Reference | Resolved in | Coordinates |
|---|---|---|
| `models:/<name>/<version>` | an MLflow file store: `<root>/mlruns`, `<root>/<project>/mlruns` (where the ledger's MLflow adapter looks), or a root that is one | `models/<name>/version-<n>/meta.yaml` → `storage_location` (mapped into the store being read when its tail is a path inside it, as `_artifact_root` does for runs; scrubbed `file:./mlruns/...` included), `run_id` |
| `models:/<name>@<alias>` | same | the registered model's `meta.yaml` `aliases` → version (noted in the output) |
| `hf:<org>/<name>[@<revision>]` | a local hub cache under a search path (`models--<org>--<name>/`, at the root, one project down, or under `hub/`) | `refs/<revision>` → commit, or a commit / unique ≥7-hex prefix; files of `snapshots/<commit>/` |
| anything else | a directory (or one file) of saved model files, relative to the cwd (CLI) or the draft's directory (claims) | none: content only |

MLflow's own URI syntax is used verbatim so a reference can be pasted from
MLflow. `models:/m-<id>` (a logged model, not a registered version) is
refused with that said; `hf:x@` and other malformed forms are refused
(`RefError`, CLI exit 2). A Hub cache stores an LFS file as a link to
`blobs/<sha256>`: when a snapshot file links to a 64-hex blob, its bytes are
checked against that name and a mismatch is `model-file-mismatch`, broken.
Hidden names are skipped in every directory (huggingface_hub writes a
`.cache/` beside a local download).

All MLflow-layout knowledge stays in `ledger_adapters/` (tests/test_tracker_names.py);
`model_trace.py`, `claims.py` and `cli_model.py` name no tracker.

## Decision 3: version → run, by digest

For each version, every run record under the search paths is read **through
the ledger's own adapters** (`import_project` on each search root and each
of its project subdirectories, deduplicated by `(source_path, record
digest)`), so an MLflow store with no synced records is read exactly as the
ledger reads it, and a synced record outranks the store's copy (the
adapter's `taken` rule).

- A run **produced** the version when its `data[]` entries with
  `role == "output"` and a sha256 include files of the version. Coverage is
  counted per file; the runs with the most matched files are kept.
- All files matched → the link holds. Fewer → `partial-output`, **broken**:
  bytes of the version that no run output are untraced, whether changed
  weights or a model card added later, and a shared `config.json` matching is
  no evidence about the weights beside it. The message names the unmatched
  files.
- Several runs with full coverage → `ambiguous-producer` (unverified), all
  kept, as lineage keeps every match.
- A run whose non-output data contains a version file **read** the model (an
  evaluation run); it is listed as a reader. Readers matter to claims
  (Decision 5).
- Each producer's upstream is `resolve_lineage(producer, datasets, runs)`,
  its issues carried as they are. A producer whose lineage reaches no dataset
  record gets `untraced-training-data` (unverified), naming its inputs.

### What the MLflow side needed

MLflow 3 logs a model **outside** the run directory --
`<exp>/models/<model_id>/artifacts/` -- and records the run's statement that
it produced it in `<run>/outputs/<model_id>/meta.yaml` (`destination_type:
MODEL_OUTPUT`); a registered version's `storage_location` points at those
artifacts. The ledger's MLflow reader (`mlflow_store.py`) read only
`<run>/artifacts/`, so a run's record listed none of its model's files. It
now reads every model the run declared as an output, as `data[]` entries
`models/<model_id>/<path>`, role `output`, sha256 of the bytes, after the
artifacts. A model the run did not declare is not read. `<exp>/models/` is
also excluded from run directories by name (it held no `meta.yaml` and was
skipped anyway). `attest mlflow sync` uses the same reader, so a synced
record lists them too. The exporter (`examples/mlflow-records/export_records.py`)
does not read logged models yet; no committed run has one, so the
byte-identity pins (tests/test_mlflow_records.py, test_mlflow_adapter.py)
are unaffected. MLflow 2 logged models into `<run>/artifacts/model/`, which
the reader already lists, and its registry's `storage_location` points there.

**How strong the MLflow link is.** When the store is read in memory, the run's
digests and the version's are computed from the *same* files at the same
moment, so the digest match adds nothing beyond MLflow's own
`outputs/` declaration and the registry's pointer. The digest becomes
evidence once a record binds it earlier: `attest mlflow sync` writes the
record with the sha256s of the day, and a later change to the version's
files breaks the link (`partial-output`; pinned by a test that appends a byte
to `model.skops`). The HF callback's record is always bound at train end.

## Decision 4: names only diagnose

The registry's run id (MLflow `run_id`) never establishes a link:

| Kind | Severity | When |
|---|---|---|
| `unresolved-model` | broken | no store under the search paths holds the version; the message says, per store, why not (no such model, version, alias, snapshot; storage not local) |
| `empty-model` | broken | the version's directory holds no files |
| `model-file-mismatch` | broken | a cached file's bytes are not the blob sha256 it links to |
| `no-producing-run` | broken | no record lists any file of the version as output; with a registry run id, says whether a record of that run exists ("lists none of these files") or none does |
| `partial-output` | broken | the best producer output some files of the version, not all |
| `registry-run-mismatch` | broken | the registry names run X, the digests found other run(s): the registry and the records disagree, and it is not resolved by picking one |
| `ambiguous-model` | unverified | several stores hold the reference; all kept |
| `ambiguous-producer` | unverified | several runs output every file; all kept |
| `untraced-training-data` | unverified | a producer's lineage reaches no dataset record |

plus every `resolve_lineage` issue upstream. `ok` = no broken issue;
`verified` = no issue at all, the end-to-end claim.

## Decision 5: claims -- `model=`, and `untraced` as a lint

A claim names its model with one optional field, parsed like `cite=`:

```markdown
<!-- claim: model-registry/toy-seq-gpt2-0 metric=train_loss value=2.1381 tol=0.00005 model=hf/model -->
```

`claims.check(conn, root, resolver=..., tracer=...)` takes the tracer by
injection, as it takes the citation resolver, so `claims.py` imports no
registry reader. `model_trace.tracer_for(target)` searches the draft's
directory and `RESEARCH_ROOT`, and is lazy: the index is built on the first
claim with `model=`, so a draft without one reads nothing (a test passes a
tracer that raises).

A claim with `model=` gets a new verdict, `untraced`, only when

1. the reference does not parse, or the chain is broken (Decision 4); or
2. the chain holds but the claim's own run (its unique ledger match) neither
   output nor read any file of the version -- the number is real, but it is
   not about that model (`source_path` compared with the producers' and
   readers').

`untraced` is a lint, like `uncited`: the numeric verdict stands beside it,
and the exit status ignores it -- a broken provenance chain is not a false
number. Unverified links are not linted; they appear in the chain.

Every distinct reference also gets a summary: `attest claims` prints
`  model         <ref> -> run <label> (n/n files) -> dataset <labels> [verdict]`
after the verdicts; `runs.claims_check` carries a `models` block
(`ref, ok, verified, chain, issues, n_issues`, ≤4 references, ≤3 issues each,
strings ≤200 chars) and a `model` key on every claim row (null when absent),
as `cite` is. A draft with no `model=` prints the same lines and the same
`N claim(s): ...` summary byte for byte; its MCP payload gains only
`"models": []` and `"model": null` per row (the envelope pin in
tests/test_figures.py moved deliberately).

## Decision 6: the command, and no MCP tool

`attest model trace REF [--search PATH ...] [--json]` (`cli_model.py`) reads
files only, like `attest runs lineage`: exits 0 when no link is broken
(unverified links may remain), 1 when one is (an unresolvable reference
included), 2 when a registry reference does not parse. Unreadable records
are listed as skipped.

No MCP tool: `runs.claims_check` already answers the agent-facing question
("does the claim about this model trace?") with a bounded block, and a
standalone tool would move the 55-tool count and its pins for a question the
dataset spec deferred for the same reason (`runs.lineage`).

## The fixture

`examples/model-registry/` (a golden path), generated by `generate.py` with
the real libraries and scrubbed of machine paths and attribution:

- `mlflow` part (mlflow-skinny 3.15.2 + pandas + skops): scikit-learn's
  bundled `iris.csv` as `data/iris/1/` (a dataset record); a real MLflow file
  store in which one run fits a LogisticRegression, logs accuracy, its
  dataset input and `mlflow.sklearn.log_model`, then `register_model` →
  `iris-clf` version 1 and alias `champion`; `attest mlflow sync` →
  `mlflow-records/`. The scrub runs before the sync, so the synced digests
  are of the committed bytes. Trace: verified version → run, then
  `untraced-training-data` -- MLflow names a dataset input with its own
  digest, never a sha256 of the bytes, so the run reaches no dataset record.
- `hf` part (transformers 5.23 + torch on CPU, no download: a 1-layer GPT-2
  from a config): a native `prep` run writes `sequences.csv`; `data/toy-seq/1/`
  is that file as a dataset naming `prep` by record digest; a Trainer with
  `AttestationCallback(datasets=[...])` trains, checkpointing at the last
  step; `trainer.save_model("hf/model")`. The saved model's four files equal
  the last checkpoint's byte for byte, so `hf/model` traces **verified**:
  version → training run → dataset (content digest) → `prep`.
- `DRAFT.md`: three claims, one of which credits the GPT-2 with the iris
  run's accuracy → `3 claim(s): 3 supported, 1 untraced`.

**No real Hub snapshot.** Producing one needs the network (this environment
has none: huggingface.co is unreachable) and a model we trained would need a
push. The cache reader is tested against a layout the test builds from the
saved model's files with a commit named `c0ffee000...` so nobody mistakes it
for real; once, in a scratch environment, huggingface_hub 2.4.0's own
`scan_cache_dir` and `try_to_load_from_cache` read that layout as repo
`org/tiny`, revision `main`, the four files.

## Findings recorded, not fixed here

1. **`resolve_lineage` matches a run's `output` entries as its upstream.** In
   the fixture the `prep` run lists `sequences.csv` as an output and the
   dataset holds that file, so the tree shows `dataset <- prep <- dataset
   (shown above)` -- data drawn as flowing into the run that wrote it. The
   source-run link is already `source_runs[]`'s job; matching only
   non-output roles upstream would remove the cycle. A schema-package
   behaviour change, so for a later attestation-schema release, with the
   conformance lineage fixtures re-checked.
2. **MLflow dataset inputs carry no byte digest**, so an MLflow-only chain
   ends `unverified` at the run. A training script that also writes a native
   core record (or a callback) closes it; reading the input's local `source`
   path at scan time would digest today's bytes, not the run's, and is not
   done.
3. **The exporter lags the adapter** on logged models (Decision 3).
4. **Base models.** A fine-tune's starting weights are not in the HF
   callback's `data[]`, so "fine-tuned from" is not a link yet.

## Testing

`tests/test_model_trace.py` (29 tests): both committed chains; the store read
without synced records; a changed weight file breaking the synced binding;
no producing run naming the registry's run; a registry naming another run;
each unresolvable reference saying why; malformed references refused; a
file no run output; the synthetic hub cache by branch, commit and prefix,
and a corrupted blob; `model=` parsing; the draft's verdicts and chains; a
draft without `model=` reading nothing; `attest claims` output; the MCP
row/block shape under the 7000-char budget; `attest model trace` exit codes.
The golden-path suite runs `run.sh` and pins its output lines.
