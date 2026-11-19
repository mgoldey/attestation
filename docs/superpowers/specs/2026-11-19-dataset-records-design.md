# Dataset records and end-to-end lineage

**Date:** 2026-11-19
**Status:** DRAFT format, implemented on `next/dataset` (roadmap item 32).
`attestation.dataset/1` ships with `DATASET_SCHEMA_STATUS = "draft"`: no
producer has written one yet, so it is not frozen (see "Draft, not frozen").
**Depends on:** `2026-11-19-core-run-record-design.md` (the frozen
`attestation.run/1`, `attestation.digests/1` and their compatibility
policy), `2026-11-19-conformance-suite-design.md` (the checker and plugin
this extends).
**Changes no frozen format.** `attestation.run/1`, `attestation.history/1`,
`attestation.state/1` and `attestation.digests/1` are untouched and
`tests/test_frozen.py`'s digests do not move. Everything here is a new record
kind, new Python API in `attestation-schema` (a 0.x minor addition) and one
new `attest runs lineage` command. The ledger (`ledger.py`,
`ledger_adapters/`) is not changed.

## Problem

The lineage the stack wants to verify is

```
ferric build ──► smeltery run ──► dataset ──► training run ──► claim
```

Every arrow but one already has a digest-checked carrier:

- **ferric build → smeltery run.** smeltery's core record lists ferric as a
  `dependencies[]` entry, `provenance: "verified"`, `build_sha` from
  `ferric.__build__` (core spec, v1 freeze).
- **training run → claim.** `runs.claims_check` checks a claim against the
  run's metrics in the ledger.
- **smeltery run → dataset → training run** has nothing. smeltery's Parquet
  export copies provenance *columns* into the table (`run_id`,
  `record_sha256`, `ferric_git_sha`, ...), but no file outside the table says
  which runs the table was made from, and a training run's
  `data[].sha256` is a digest of *some* bytes that nothing resolves. Worse,
  the export's `record_sha256` column is smeltery's legacy self-hash of
  `run-<digest12>.json` (sha256 of `json.dumps(sort_keys=True)` minus the key
  itself), not the digest of the core `attestation/run.json`, so even a
  reader that opened the Parquet file could not find the core record from it
  (see "smeltery: what it must add").

## Decision 1: a new record kind, not a run/1 change

`attestation.run/1` is frozen; adding `source_runs` to it would be
`attestation.run/2` and break every deployed reader. A dataset is not a run
anyway: it has no status, no metrics, no seeds of its own. So the dataset is
its **own kind**, `attestation.dataset/1`, with its own JSON Schema file
(`schemas/dataset-1.schema.json`), writer, strict reader, `DATASET_READERS`
dispatch and status constant, in `attestation_schema.dataset`. The run
record's `from_dict` already refuses a `schema` of another kind ("is not a
run record"), and the dataset reader refuses a run the same way, so a file
can never be read as the wrong kind.

## Decision 2: the layout

```
<dataset_dir>/                   immutable once dataset.json exists
├── <files>                      the dataset: tables, split artifacts, anything
└── attestation/
    ├── digests.json             attestation.digests/1 over every file outside attestation/
    └── dataset.json             the dataset record; written last
```

- `digests.json` is the **frozen `attestation.digests/1` format**, written by
  the same `layout.build_manifest` a run uses. Nothing new to learn, and no
  new format to freeze.
- It is **required** for a dataset (it is optional for a run): it *defines*
  the content digest.
- **Write order** is files → `digests.json` → `dataset.json`, exactly as for a
  run. `write_dataset_manifest` refuses once `dataset.json` exists;
  `write_dataset` refuses unless `digests.json` is on disk with the content
  digest the record states, and refuses to overwrite a record.
- **A directory holds one record kind.** `run.json` and `dataset.json` in one
  `attestation/` would share one `digests.json` with two meanings; the
  conformance checker reports it. A run that *produces* a dataset writes it to
  its own directory (a sibling, or a subdirectory of the run's outputs is
  also fine -- it is then a file of the run, covered by the run's manifest,
  and a dataset of its own).
- `layout.find_datasets(root)` finds every directory with
  `attestation/dataset.json`; `layout.find_runs` is unchanged, so **the
  ledger never sees a dataset** and its scan output does not move.
- **Recommended path:** `datasets/<dataset_id>/<version>/`. A dataset
  directory is immutable: a new split, a re-export, a corrected row is a new
  version in a new directory, because a run that trained on the old content
  digest must still resolve to the old record.

## Decision 3: the content digest, defined

> **`content_sha256` = sha256 of the bytes of `attestation/digests.json`**,
> which is the RFC 8785 serialization of
> `{"schema": "attestation.digests/1", "files": {<posix path>: <sha256 of that file's bytes>}}`
> over every regular file under the dataset directory except `attestation/`.

Consequences, each deliberate:

- **It is a digest of real bytes on disk.** `sha256sum
  <dir>/attestation/digests.json` prints it. A training run lists it as
  `data[] = {name: "<dataset_id>", sha256: <content_sha256>, role: "train"}`,
  and that is run/1's `data[].sha256` meaning, verbatim ("64 lowercase hex
  of the bytes") -- the bytes of the dataset's manifest, which bind every
  file. No run/1 change, no new meaning for an old field.
- **Paths are content.** Renaming `train.parquet` to `test.parquet` changes
  the digest. A partition's name is part of what a model trained on.
- **Metadata is not content.** `version`, `name`, `license`, `columns`,
  `created_at`, `source_runs` do not enter it: two records describing the
  same bytes (a corrected license) share a content digest, and a run that
  trained on those bytes resolves to both (reported as ambiguous, never
  silently one -- the run/1 `run_id` rule). The record's own identity is
  `dataset_digest(record)` = sha256 of `dataset.json`.
- **The reader recomputes it from the record alone.** `files[]` carries every
  path and sha256, so `content_digest(files)` is a pure function and the
  reader refuses a record whose `content_sha256` disagrees with its own
  `files[]`. The conformance checker then verifies `files[]` against the
  directory (sha256 and byte count of every file, no extra, none missing) and
  `digests.json` against both.
- **Bytes, not values.** A Parquet file rewritten by another pyarrow version
  is another dataset. smeltery's split files already take that stance for the
  table digest ("the split names the file it was made on"); the alternative,
  a digest over decoded values, needs a Parquet reader in a stdlib package and
  a canonical value encoding nobody else computes.

Rejected: a digest over the canonical `files[]` list with sizes and media
types. It would be checkable only by this package (nothing on disk has those
bytes), so a run's `data[].sha256` could not be "the sha256 of bytes" and
`sha256sum` could not check it.

## Decision 4: the record, `attestation.dataset/1`

Every key required, `null` spelled out, unknown keys refused outside `ext`
-- run/1's conventions throughout, and its validators where the field is the
same (`producer`, `code`, `ext`, timestamps, credential checks).

| Key | Type | Meaning |
|---|---|---|
| `schema` | `"attestation.dataset/1"` | Kind and version; the reader dispatches on it. |
| `dataset_id` | non-empty string | The producer's stable name for the dataset across versions (`smeltery-benzoic-ddE`). **Not unique**: versions share it. A run names the dataset by this in `data[].name` (convention), which is what lets the resolver say "you trained on another version" instead of "unknown data". |
| `name` | string, nullable | Human title. |
| `version` | string, nullable | The producer's version label (`2026-11-19`, `v3`). A label, not an identity: identity is the content digest. |
| `created_at` | timestamp, nullable | When the record was made, in run/1's fixed-width `...SS.ffffffZ`. Null = the producer could not say (an importer). |
| `producer` | `{name, version}` | What wrote the record (run/1's `Producer`). |
| `code` | run/1's `code` object | The code that built the dataset (the exporter), on the provenance ladder; mixed code is `unknown` with `evidence`, as in run/1. |
| `content_sha256` | 64 hex | Decision 3. Reader-checked against `files[]`. |
| `files` | non-empty list of `{path, sha256, bytes, media_type}` | Exactly the manifest's files, **sorted by path** (UTF-16 code units, JCS key order) so one dataset is one byte sequence. `path` relative POSIX, no `.`/`..` segment, not under `attestation/`; `bytes` int ≥ 0; `media_type` `type/subtype` (RFC 6838 shape, lowercase), nullable = not stated. Media types are the producer's word, never sniffed (`mimetypes` reads the host's tables, so a sniffed type would make the bytes host-dependent); `suggest_media_type` offers a fixed table (`.parquet` -> `application/vnd.apache.parquet`, ...). |
| `rows` | int ≥ 0, nullable | Total rows, when the dataset is tabular and the producer counted. |
| `columns` | list of `{name, type, unit}`, nullable | The schema summary. `type` the producer's type spelling (`float64`, `list<struct<...>>`), nullable; `unit` UCUM suggested, nullable (as `metrics[].unit`). Names unique. `null` = not described; `[]` = described and not tabular. |
| `splits` | list of `{name, path, sha256, kind, seed}` | Split artifacts *inside* the dataset. `path` must be one of `files[]` and `sha256` must equal that file's (reader-checked) -- stated twice so a resolver can match a run's split entry without a lookup, and the reader makes them agree. `kind` free non-empty string (`scaffold`, `target`, `random`, `temporal`), `seed` int or list of ints, nullable (= no seed / deterministic). Names unique. `[]` = none. |
| `source_runs` | list of `{run_id, family, record_sha256}`, nullable | The runs whose outputs the dataset was made from. `record_sha256` = sha256 of that run's `attestation/run.json` bytes (= `record_digest`), nullable = named but not digested (run/1's `data[].sha256` rule). `family` nullable as in run/1. `null` = the producer could not say; `[]` = it can, and there are none (a downloaded corpus). No duplicate entry; a non-null `record_sha256` appears once. |
| `license` | SPDX license expression, nullable | `null` = not stated -- **never** "no license". Syntax-checked (ids, `LicenseRef-`/`DocumentRef-`, `+`, `AND`/`OR`/`WITH`, parentheses), not checked against the SPDX list (a stdlib package cannot ship it current). `NOASSERTION` and `NONE` are refused: they are SPDX *document* values, and `NOASSERTION` would be a second spelling of `null`. |
| `ext` | `{namespace: object}` | run/1's rules and grammar, `attestation*` reserved. |

**Credentials.** The run/1 reader refuses credentials in `code.repo` and
`dependencies[].source` only (conformance spec, v2 candidate 3, says why it
cannot widen). A new kind can start where v1 should have: **the dataset
reader refuses a URL with credentials in any string of the record** (a
password, or any userinfo on http(s), `git+https://token@...` included --
run/1's `_no_credentials` applied to every URL-shaped substring). The
heuristic half (token shapes, secret-named keys, query parameters) stays in
the conformance `credentials` rule, as for runs: a heuristic in the reader
would make a legitimate record unwritable on a false positive.

**What is not in it,** each deliberately: no `status` (a dataset is not an
execution; the run that built it has one); no row-level provenance (it is in
the table's own columns, which a stdlib package cannot read); no per-file
role (a split is named in `splits`, everything else is "the data"); no
`dependencies` (the exporter's environment is the source run's business, and
the exporter's own identity is `code`).

## Decision 5: draft, not frozen

The roadmap's principle is "a schema no producer has written to is a guess",
and run/1 is the evidence: between the first draft and the freeze, MLflow
and smeltery changed it eight times, every change breaking committed
records. Freezing `attestation.dataset/1` before smeltery has written one
would freeze a guess, and every fix after that would cost a `/2`.

So `DATASET_SCHEMA_STATUS = "draft"`, and the rules are run/1's pre-freeze
rules:

- The identifier stays `attestation.dataset/1` through the draft. A draft
  change may stop older draft records from reading; committed fixtures are
  regenerated (`conformance_fixtures.py`) in the same commit, and a producer
  pins this package by commit, as smeltery did for run/1.
- `dataset-1.schema.json` is **not** in `test_frozen.py`'s pinned digests.
  `test_frozen.py` instead requires every `*.schema.json` to be either pinned
  frozen or listed as a draft, so a new schema file cannot appear without
  that decision, and a draft flipped to `"frozen"` fails until its file and a
  reference record's bytes are pinned.
- **Freeze criteria:** two producers have written dataset records the reader
  accepts (smeltery's export, and one not written by this repository's
  authors for this test -- an MLflow/HF dataset exporter, or a training-data
  pipeline), at least one of them consumed by a real training run's record
  that `resolve_lineage` links end to end with no issue, and their findings
  are decided, as at the run/1 freeze. Then the status flips, the file and a
  reference record's bytes are pinned, and the compatibility policy applies
  unchanged.

Rejected: shipping frozen because "the fields are obvious". The smeltery walk
below already found two things the export cannot supply today; the draft
exists to absorb exactly that.

## Decision 6: resolving lineage, pure and stdlib

`attestation_schema.lineage.resolve_lineage(run, datasets, runs) -> Lineage`
takes a run record, a collection of dataset records and a collection of run
records, reads no file and returns a graph plus every issue.

**Matching.** For each run reached (starting with the given one), each
`data[]` entry with a sha256 is matched against every dataset by, in order of
meaning:

1. `content` -- the entry's sha256 is a dataset's `content_sha256` (the run
   names the whole dataset);
2. `split` -- it is a `splits[].sha256` (the run names the split it used);
3. `file` -- it is a `files[].sha256` (the run digested the table it opened:
   what a training script naturally hashes, and what smeltery's split file
   records as `table.sha256`).

An entry that matches nothing is not an issue -- most of a run's data is not
a dataset (configs, receptors, its own manifest). For each dataset reached,
each `source_runs[]` entry is matched to run records by `record_digest`.
The walk continues upstream from every run reached (a smeltery run that read
an earlier dataset links to it too), with a visited set, so cycles and
diamonds terminate.

**Graph.** Nodes are runs (keyed `run:<record_digest>`) and dataset records
(keyed `dataset:<dataset_digest>`; two records with one content digest are
two nodes). Edges point the way data flowed -- `source run -> dataset ->
consuming run` -- each naming how it was established (`content`, `split`,
`file`, `source_run`) and the `data[]`/`source_runs[]` entry it came from.
`Lineage.dependencies(name)` lists every upstream run's dependency of that
name (`ferric`), which is how the first arrow -- ferric build -> smeltery run
-- appears in the result: the smeltery run's own `verified` identity.

**Issues** carry a severity: `broken` (a link that should hold does not) or
`unverified` (a link that holds by name only, or cannot be checked).

| Kind | Severity | When |
|---|---|---|
| `missing-source-run` | broken | no given run has the `record_sha256`, nor the `(family, run_id)` |
| `source-run-digest-mismatch` | broken | no run has the digest, but one with that `(family, run_id)` does: the record was rewritten, or the dataset names another execution |
| `source-run-identity-mismatch` | broken | the digest matches a run whose `run_id`/`family` differ from the entry's: the dataset record contradicts itself |
| `dataset-digest-mismatch` | broken | a `data[]` entry's sha256 matches no dataset, but its `name` is a known `dataset_id`: the run trained on a version that is not given (or was changed) |
| `undigested-source-run` | unverified | `record_sha256` is null; linked by `(family, run_id)` only |
| `undigested-data` | unverified | a `data[]` entry naming a known `dataset_id` has `sha256: null` |
| `unrecorded-source-runs` | unverified | a dataset reached has `source_runs: null` |
| `ambiguous` | unverified | one sha256 or `(family, run_id)` matched several records; every match is an edge, none is picked |

`Lineage.ok` is "no broken issue"; `Lineage.verified` is "no issue at all"
-- the end-to-end claim. Name matching (`dataset_id`, `(family, run_id)`)
only ever *diagnoses*; a link is only ever established by a digest, or
flagged `unverified` when no digest exists.

Rejected: resolving by reading directories inside the resolver. Callers
already hold records (the CLI reads them, the ledger has them, a test builds
them); a pure function is testable without a filesystem and cannot be coaxed
into reading the wrong tree.

## Decision 7: conformance for dataset directories

`conformance.check_dataset_dir(dir, *, verify_outputs=True) -> Report`
(`Report` gains a `dataset` field), and `check_dir` dispatches on what the
directory holds. The rule names are the run rules, reused -- no new rule:

| Rule | Checks for a dataset |
|---|---|
| `read` | `dataset.json` reads under the dataset reader (which checks the content digest, splits and credentials in URLs); the directory holds no `run.json`/`state.json` beside it. |
| `canonical` | `dataset.json`'s bytes are `dumps_dataset(read)`. |
| `manifest` | `digests.json` exists, strict, `{schema, files}`, canonical; its sha256 is `content_sha256`; its `files` equal the record's `files[]` path→sha256; with `verify_outputs`, every file on disk matches by sha256 **and byte count**, none extra, none missing, no symlink. |
| `credentials` | the heuristic scan over `dataset.json` and `digests.json`. |
| `ext-namespace` | every `ext` key, all reported. |

`find_targets` and the pytest plugin accept dataset directories wherever
they accept run directories (`--attestation-runs 'datasets/**'`, a
`dataset.json` path, or a directory holding both kinds), one item per
directory. The option keeps its name: renaming it would break every
producer's CI line. Lineage is **not** a conformance rule: a dataset
directory does not hold its source runs, and "the source run exists" is a
property of a collection, which is what `resolve_lineage` and `attest runs
lineage` check.

**Fixtures** (built by `conformance_fixtures.build`, rebuilt and compared
byte for byte by the existing test): `datasets/valid/minimal` (one file, no
source runs, license null), `datasets/broken/<rule>` for `read`,
`canonical`, `manifest`, `credentials`, `ext-namespace`, each breaking
exactly its rule, and `lineage/` -- the whole chain (below).

## Decision 8: the ledger side, kept minimal

`attest runs lineage RUN [--search PATH ...] [--json]` reads the run record
at `RUN`, finds every run and dataset record under the search paths
(default: the current directory) with `layout.find_runs`/`find_datasets`,
calls `resolve_lineage`, and prints the graph upstream from the run with
every issue; it exits 1 when any link is broken. Unreadable records under the
search paths are listed and skipped (the conformance plugin is where they
fail). It lives in `cli.py` beside the other `runs` commands, reads no
database and touches no ledger adapter; the ledger's tables do not store
`data[]`, so a DB-backed version would need a ledger schema change, which
is deferred until a reader needs lineage over the ledger (an MCP
`runs.lineage` tool is deferred with it: the tool count is pinned in several
places and nothing yet asks an agent this question).

## The lineage fixture, and the smeltery walk-through

### What a smeltery export holds today

`smeltery export parquet RUN... -o runs.parquet` writes one table, one row per
(ligand, target, tier), with provenance columns copied from each run's legacy
record: `run_id` (= `input_digest[:12]`), `input_digest`, `record_sha256`
(the legacy record's self-hash, nullable), `campaign`, `smeltery_version`,
`smeltery_commit`, `smeltery_dirty`, `ferric_version`, `ferric_git_sha`,
`ferric_dirty`, `ferric_provenance_level`, ...; file metadata
`smeltery.schema_version`, `smeltery.units`, `smeltery.runs`,
`smeltery.grain`, `smeltery.generator`. `smeltery export split
scaffold|target runs.parquet -o split.json --seed N` writes a
deterministic split file naming the table by `table.sha256` (file bytes).

### How it maps

| Dataset field | From smeltery |
|---|---|
| directory | a new `-o DIR/` form of the export: `DIR/runs.parquet`, splits beside it, then `DIR/attestation/` |
| `dataset_id`, `name`, `version` | the export's arguments (`--dataset-id`, default the campaign; `--version`) |
| `created_at` | `core_record.utc_now()` at export (`format_timestamp`) |
| `producer` | `smeltery export`, smeltery's version |
| `code` | smeltery's own identity at export time, the same function `core_record` uses for a run (mixed commits impossible: one process) |
| `files[]`, `content_sha256` | `dataset_files(DIR, media_types=...)` after the table and splits are written: `.parquet` `application/vnd.apache.parquet`, `.arrow` `application/vnd.apache.arrow.file`, split `.json` `application/json` |
| `rows` | `len(rows)` |
| `columns` | `arrow_schema()`: name, `str(pa type)`; `unit` from `smeltery.units` for `score`/`paired_ddE*` only when one unit holds across the runs, else null (the per-row `score_unit` column says) |
| `splits[]` | each split file written into DIR: `name` = kind (or `--name`), `kind`, `seed`, its path and sha256. `algorithm_version` and `parameters` go in `ext.smeltery.splits` |
| `source_runs[]` | for each exported run: **the core record's** `run_id` and `family` (= campaign) and sha256 of `OUT/attestation/run.json`; `record_sha256: null` for a run with no core record (lineage then says `undigested-source-run`) |
| `license` | `--license SPDX`, default null. **Not** smeltery's own `MIT OR Apache-2.0`: that licenses the code, and the data's terms (a receptor's, a ligand set's) are not smeltery's to state |
| `ext.smeltery` | `schema_version` (the column layout), `grain`, `units`, `smiles_form`, `generator`, split parameters |

The **first arrow** needs nothing new: each source run's core record already
carries ferric as a `verified` dependency with `build_sha`, and the table's
`ferric_git_sha` column was copied from the same legacy record. The test
asserts both agree for every row, which is the producer-side cross-check a
stdlib reader cannot make (it cannot read Parquet).

### What smeltery must add to emit dataset records

1. **The core record's digest per run.** The export's `record_sha256` column
   is the legacy self-hash, which no core reader can resolve. The export must
   read `OUT/attestation/run.json` (beside `run-<digest12>.json`, which is
   where `export_rows` already looks) and record its sha256 -- as a new column
   `core_record_sha256` (nullable; `smeltery.schema_version` 2) and in
   `source_runs[]`.
2. **The core record's `run_id` and `family`.** The `run_id` column is
   `input_digest[:12]`; a run submitted as a job carries the submitter's
   `run_id` (`--run-id`, core spec 5.C.1), so the column and the core record
   disagree for every job. `source_runs[].run_id` must come from `run.json`.
3. **A dataset directory.** Today `-o` names one file, and writing
   `attestation/` beside an arbitrary file would digest whatever else is in
   that directory. A `-o DIR/` mode (or `--dataset-dir`) that owns the
   directory, writes the table, then splits, then `write_dataset_manifest`,
   then `write_dataset`.
4. **Splits at export time, or as a new version.** A dataset directory is
   immutable once `dataset.json` exists. `smeltery export split` run later on
   an exported directory makes a *new* dataset version (copy or hard-link the
   table into `datasets/<id>/<version+1>/`, add the split, write a new
   record); it never edits the old record, because runs trained on the old
   content digest must keep resolving.
5. **The training side.** smeltery's PyTorch dataset (the `dataset` extra)
   is where a consuming run is born. Whatever writes that run's core record
   lists the dataset as `data[] = {name: <dataset_id>, sha256:
   <content_sha256>, role: "train"}` and the split file as `{name:
   "<dataset_id>/<split path>", sha256: <split sha256>, role: "split"}` --
   `dataset_lineage_entries(record, split=...)` builds both.
6. **`--license`**, defaulting to null, and the dataset_id/version flags.

### The fixture

`packages/attestation-schema/tests/data/conformance/lineage/` is the chain,
stdlib-built so it runs on the 3.11 job (CSV stands in for Parquet: the
schema hashes bytes and never parses the table):

- `ferric` is not a directory: it is the `verified` `ferric` dependency, with
  `build_sha`, in each smeltery run.
- `smeltery-a/`, `smeltery-b/` -- smeltery-shaped core runs: the legacy
  `run-<digest12>.json` (role output), `steps/00-dock/stage.json`,
  `digests.json` bound as the manifest, `ext.smeltery.input_digest`.
- `dataset/` -- `runs.csv` with the export's provenance columns (`run_id` =
  `input_digest[:12]`, `record_sha256` = a legacy self-hash that is **not**
  the core digest, `ferric_git_sha`, ...), `split-scaffold.json` in
  smeltery's split format naming the table by `table.sha256`, and
  `attestation/{digests.json,dataset.json}` whose `source_runs` carry the core
  digests.
- `training/` -- a run whose `data[]` names the dataset by content digest and
  the split file by its digest.

`tests/test_lineage.py` resolves it end to end (`verified`), checks the
ferric column against the run records, and runs one mutant per issue kind on
copies of the records.

## Testing

- `packages/attestation-schema/tests/test_dataset.py` -- round trip and byte
  determinism; every required key removed in turn names its path; unknown
  keys refused; `content_sha256` recomputed (a changed file digest without a
  new content digest is refused); `files[]` order, path rules and sizes; split
  path/digest agreement; SPDX syntax (valid expressions, `NOASSERTION`,
  `NONE`, garbage); credentials in any URL; null-vs-`[]` for `columns` and
  `source_runs`; version dispatch both ways (a run is not a dataset, a dataset
  is not a run, an unknown version names the known ones); the writer's
  refusals (no manifest, a different manifest, an existing record); the JSON
  Schema file agrees with the reader on every valid and invalid case where it
  can express the rule.
- `tests/test_lineage.py` -- the chain resolves `verified`; each issue kind
  from a single mutation of the chain, and an unmutated control that has none
  (measurement lessons, 4); cycles terminate; ambiguity is reported, never
  resolved by picking.
- `tests/test_conformance.py`/`test_conformance_plugin.py` -- the dataset
  fixtures: valid ones conform, each broken one breaks exactly its rule with
  its file and JSON path, through the API and through `pytester`; a directory
  with both a run and a dataset is a `read` violation; `find_targets` finds
  both kinds.
- `tests/test_frozen.py` -- the run/1 pins unchanged; every schema file is
  frozen-and-pinned or a listed draft; `DATASET_SCHEMA_STATUS == "draft"`.
- `tests/test_cli.py` -- `attest runs lineage` on the fixture chain prints the
  chain and exits 0; with a source run removed it prints the broken link and
  exits 1.
