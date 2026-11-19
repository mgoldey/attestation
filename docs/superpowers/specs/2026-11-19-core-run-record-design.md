# The core run record, the run directory, and the job-backend interface

**Date:** 2026-11-19
**Status:** v1-DRAFT, not frozen (roadmap items 1, 2 and 43).
**Draft means:** records written under `attestation.run/1` before the freeze
may stop reading if the draft changes, and nothing outside the three
cooperating repos (attestation, smeltery, ferric) should write one until the
Status line says "frozen". The freeze is a deliberate edit to this spec and to
`SCHEMA_STATUS` in `attestation_schema.record`, made once the three producers
in "Fit check" below exist and have written real records.
**Ships as:** a second distribution, `attestation-schema` (import name
`attestation_schema`), built from this repository at
`packages/attestation-schema/`.
**Not this:** `attestation.run_record` (`2026-10-17-run-record-design.md`) is a
*signed* DSSE/in-toto statement about one *checker* run (which papers were
read, which claims were judged). This spec is the *experiment's* own record:
what ran, on what, with which code, and what it measured. The two meet later
— a signed record can name a core record as a subject by its digest — but
neither depends on the other.

## Why a core record, and why a separate distribution

Today every producer invents its own result file. The ledger
(`ledger.py`) reverse-engineers those files per tracker (`ledger_adapters/`),
and its `RunRecord` dataclass carries project/name/family/status/config/
metrics but no schema version, no code identity, no data digests, no metric
direction and no seeds — because the artifacts it reads do not say them. The
`metric_direction.toml` side file and `_caveats()` exist to paper over the
missing direction. smeltery writes `run-<digest12>.json` with its own
`SCHEMA_VERSION = 1`; its `ferric` identity block (VERIFIED / INFERRED /
UNKNOWN) is the most careful code-identity logic in the stack, and nothing
else can read it.

A core record fixes the contract once: producers write it, attestation
reads it, and adapters become importers of legacy artifacts instead of the
only path.

**Separate distribution** because the producers must not depend on the
observer. smeltery installing `attestation` to write a record would pull
FastAPI, sklearn, sqlite-vec, langgraph and an MCP server into a chemistry
pipeline. `attestation-schema` depends on **the standard library only**, so a
producer's dependency on it is free. The direction of dependency is fixed:

```
smeltery ──► attestation-schema ◄── attestation (dev group today; see Open decisions)
ferric   (depends on nothing here; its identity is recorded BY smeltery)
```

`attestation_schema` imports nothing from `attestation`, and nothing under
`src/attestation` or the schema package imports `smeltery` or `ferric`.
`tests/test_boundaries.py` enforces both, with a self-test that proves each
detector fires (roadmap 43).

**No `jsonschema` dependency.** The reference validator is hand-written
stdlib Python (`record.py`) because several rules cannot be said in JSON
Schema at all (duplicate keys in the input, one direction per metric name,
`finished_at >= started_at`, integers past 2^53), and a dependency a producer
must install to *write* a file is a dependency that will be skipped. A JSON
Schema file (`attestation_schema/schemas/run-1.schema.json`, draft 2020-12)
ships alongside for other languages and editors: it is **necessary, not
sufficient** — anything it rejects the reader rejects, not the reverse. The
test suite checks both against the same valid and invalid cases using the
`jsonschema` that the dev environment already has (via `mcp`), so the two
cannot drift silently; the package never imports it.

## The record: `attestation.run/1`

One JSON object. **Every key below is required**; "nullable" means the key
must be present with `null`, never absent. Absent-vs-null is exactly the
ambiguity the ledger already pays for ("None means did not say — never no
corpus"); making the producer spell `null` puts that decision where the
knowledge is. Unknown keys are refused everywhere except inside `config` and
`ext.<namespace>`.

| Key | Type | Meaning |
|---|---|---|
| `schema` | `"attestation.run/1"` | Record kind and version. The reader dispatches on it. |
| `run_id` | non-empty string | The producer's name for the run. **Not required unique**: an MLflow run id names one execution, smeltery's `<digest12>` names an input set, so re-executions share it. Two records with the same `(family, run_id)` are both valid and distinct (told apart by `record_digest`); a reader keeps both, treats them as re-executions of one run, and never silently picks one. |
| `family` | string, nullable | The comparison group (an MLflow experiment, a smeltery campaign). `null` = the run belongs to no comparison. |
| `arm` | string, nullable | The variant within the family. |
| `status` | `"complete"` \| `"failed"` \| `"cancelled"` | Terminal states only: a record is written once, when the run ends. In-flight state lives in `state.json` (below). |
| `started_at`, `finished_at` | RFC 3339 UTC `YYYY-MM-DDTHH:MM:SS[.f]Z`, nullable | `Z` only, no offsets, and a fraction of 1-6 digits with **no trailing zero** (`.25`, never `.250000`; `:00Z`, never `:00.000Z`), so one instant has one spelling. `format_timestamp(datetime)` produces it. `finished_at >= started_at` when both are known. |
| `producer` | `{name, version}` | What wrote this file (`smeltery 0.4.0`, `attestation mlflow-import 0.6.0`). `version` nullable. Not the code that ran — that is `code`. |
| `code` | `{git_sha, dirty, repo, version, provenance, evidence}` | The experiment code's identity, on the same ladder as a dependency. `git_sha` 40 or 64 lowercase hex, nullable; `dirty` bool, nullable (unknown ≠ clean); `repo` nullable string, refused if it carries `user:password@` credentials; `version` the code's own version, nullable (a non-editable install may have only this; `producer.version` is not it when converting old runs); `provenance` `"verified"` \| `"inferred"` \| `"unknown"`, `verified` requiring `git_sha` or `version`; `evidence` nullable free text. |
| `config` | JSON object | The run's configuration, verbatim. Opaque to attestation. Numbers are JSON numbers: JCS writes `2.0` as `2` and the reader returns an int, so there is no int/float distinction in `config` or `ext`. A producer digest over these values is checkable from the record only if it was taken over their JCS bytes (`attestation_schema.canonical.canonicalize`); a digest over another serialization is the producer's claim, carried in `ext` and bound to its source file through `data`. |
| `data` | list of `{name, sha256, role}` | Data the run read or produced. `sha256` 64 lowercase hex of the bytes, nullable: `null` = the producer knows the run read or wrote this but could not digest its bytes (never a digest in another algorithm — that belongs in `ext`). `role` a free non-empty string (suggested: `input`, `train`, `eval`, `test`, `output`, `manifest`). Names unique. |
| `metrics` | list of `{name, value, direction, step, split, unit, labels}` | Final values, not curves (curves are `history`). `value` a finite JSON number, or one of the strings `"NaN"`, `"Infinity"`, `"-Infinity"` (read as the Python float; a JSON `true` or any other string is refused); `direction` `"min"` \| `"max"` \| `null` (null = the producer did not say; **never inferred**); `step` int ≥ 0, nullable; `split` string, nullable; `unit` nullable free string (UCUM suggested, e.g. `kcal/mol`); `labels` an object `{string: string}`, `{}` for none — what the value is *about* (an analogue, a model), so a reader groups without parsing names. `(name, step, split, labels)` unique; one `direction` and one `unit` per `name`. **Absence**: a metric that is not in the list was not reported; `"NaN"` means computed and undefined (the SEM of one sample). There is no null value and no boolean value (a flag is not a metric: it belongs in `ext`). |
| `history` | list of `{name, split, labels, points: [{step, value, at}]}` | Logged curves. One series per `(name, split, labels)`; `points` non-empty, in the order the producer logged them; `step` int ≥ 0 (steps may repeat and need not increase — it is a log, not a map); `value` as in `metrics`; `at` timestamp, nullable. `[]` = no curve carried. A series need not have a `metrics` entry or vice versa; direction is a property of the metric name, as is unit, so both live on `metrics` only. |
| `seeds` | object `{name: int \| [int, ...]}`, nullable | Named RNG seeds (`{"python": 0, "numpy": 0}`); a non-empty list is one stage run once per seed, in order (`{"docking": [7]}`). `null` = unknown; `{}` = the producer states no seed was set. |
| `dependencies` | list of `{name, version, build_sha, dirty, provenance, evidence, source}` | Identity of what the code ran on. `provenance` is `"verified"` \| `"inferred"` \| `"unknown"`; `verified` requires `version` or `build_sha`. `evidence` nullable free text naming how the identity was established (`ferric.__build__`, `direct_url.json`). `source` nullable: where it was installed from (`pypi`, `git+https://...`, `conda-forge`, `file`), credentials refused as for `code.repo`. Names unique: one identity per dependency; a run that mixed builds of one dependency records it `unknown`, its `evidence` saying so (deferred until a reader needs per-scope identities). |
| `ext` | object `{namespace: object}` | Producer extensions. Namespace `^[a-z][a-z0-9_-]*(\.[a-z0-9_-]+)*$`; `attestation` and `attestation.*` reserved for this schema. |

Lists keep the producer's order (the writer never sorts them: order can mean
something, and a stable producer gives stable bytes anyway). Objects are
key-sorted by canonicalization.

### Extensions round-trip untouched

`ext` is how a producer carries what the core does not model (smeltery's
tiers, MLflow's own dataset digests -- the datasets themselves are named in
`data`) without a schema change. The reader
keeps each namespace's parsed JSON as-is and the writer emits it through the
same canonicalization, so for any record that was itself written canonically,
`dumps(loads(b)) == b`, `ext` included, for namespaces this code has never
heard of. A namespace is never validated beyond "is a JSON object
canonicalization can serialize": attestation knows no chemistry.

### Writing: one sanctioned path, deterministic bytes

`attestation_schema.dumps(record) -> bytes` validates the record (the same
rules as reading — a record the reader would refuse cannot be written) and
serializes it as **RFC 8785 JCS**, the canonicalization the signed run record
already uses, for the same reason given in that spec (`json.dumps(sort_keys=
True)` disagrees with JCS on key order above U+FFFF and on number spelling, so
a verifier in another language would disagree on the bytes). No trailing
newline: the file's sha256 *is* the record's digest (`record_digest()`).

`write(record, run_dir)` writes `dumps(record)` to
`<run_dir>/attestation/run.json` via a temporary file and `os.replace`, and
refuses to overwrite an existing record unless `overwrite=True` (the ledger's
new-files-only rule). Nothing reads the clock: timestamps come in on the
record, so the same record always produces the same bytes.

JCS is **duplicated**, not imported, from `attestation.run_record.canonical`
into `attestation_schema.canonical` — the schema package may not import
`attestation`. A parity test runs both over the RFC 8785 vectors and a corpus
of records and asserts identical bytes, so the copies cannot drift. When
attestation takes a runtime dependency on the schema package (Open decisions),
`run_record.canonical` becomes a re-export and the copy disappears.

### Reading: loud, never coercing, dispatched on version

`loads(bytes | str) -> RunRecord` and `read(path) -> RunRecord` (a file, or a
run directory, meaning its `attestation/run.json`). Both:

1. parse JSON **strictly**: a duplicated object key, `NaN`/`Infinity`, an
   integer past ±2^53 or invalid UTF-8 is a `RecordError` (the standard
   library would silently keep the last duplicate and accept `NaN`);
2. read `schema`, split it into kind and version, and dispatch through
   `READERS = {"attestation.run/1": _read_v1}`. An unknown version raises
   `UnsupportedSchemaError` naming the versions this reader knows; a missing or
   malformed `schema` is a `MissingFieldError` / `InvalidFieldError`;
3. validate every field, raising the first failure as a `RecordError`
   subclass with a JSON-path-like `.path` (`metrics[2].direction`):
   `MissingFieldError`, `UnknownFieldError`, `InvalidFieldError`.

`RecordError` subclasses `ValueError`. Nothing is defaulted, truncated,
rounded or dropped.

**Older versions later.** When `attestation.run/2` exists, `READERS` gains a
`_read_v2`, and `_read_v1` gains an upgrade step returning the v2 model with
the v1 meaning preserved (a field v2 adds is `null`, meaning "a v1 producer
could not say"). `read()` always returns the newest model; the version a file
was written in stays visible as the record's `schema`. Writers only ever write
the newest version.

## The run directory (fixed now, because attestation will scan it)

```
<run_dir>/                       name is the producer's; run-<run_id> recommended
├── attestation/                 owned by this schema; nothing else writes here
│   ├── run.json                 the core record; written ONCE, at a terminal state, last
│   ├── state.json               in-flight job state; rewritten atomically by the backend
│   └── digests.json             manifest: sha256 of every file outside attestation/
├── steps/
│   ├── 00-<name>/               one directory per step / tier, in execution order
│   └── 01-<name>/               index is 2+ digits; name ^[a-z0-9][a-z0-9._-]*$
└── logs/
    ├── run.log                  whole-run log
    └── <NN>-<name>.log          one per step, named as its step directory
```

`attestation_schema.layout` owns these names (`RUN_JSON`, `STATE_JSON`,
`DIGESTS_JSON`, `step_dir(run_dir, index, name)`, `step_log(...)`,
`find_runs(root)`), so neither a producer nor the scanner spells a path by
hand. `find_runs(root)` yields every directory under `root` that has
`attestation/run.json` *or* `attestation/state.json` — a run in flight is
visible before it has a record.

**Only `attestation/` is required.** `steps/` and `logs/` are optional, and a
run's outputs may live anywhere under the run directory: smeltery's outputs
span stages and sit where its GUI contract fixes them, and an MLflow run keeps
its artifacts in MLflow's store. When a producer has steps or logs, these are
their names. The manifest covers every file wherever it is; an empty
directory has no bytes and is invisible to it.

**Write order** is what makes a half-written directory honest: steps and logs
→ `digests.json` → `run.json`. A directory with `run.json` is complete; one
with only `state.json` is in flight (or died — `state.json`'s `updated_at`
says how long ago); one with neither is not a run.

**`digests.json`** (`"schema": "attestation.digests/1"`):
`{"schema", "files": {"<posix relpath>": "<sha256>"}}` over every regular file
under the run directory except `attestation/` itself, JCS-serialized.
Symlinks are refused rather than followed (a link out of the run directory
would make the manifest describe someone else's file). `run.json` lists the
manifest in `data` with `role: "manifest"`, which binds every output to the
record through one digest without listing thousands of files in it.
`build_manifest(run_dir)` and `write_manifest(run_dir)` are provided.

**`state.json`** (`"schema": "attestation.state/1"`, backend-owned):
`{"schema", "run_id", "backend", "job_id", "state", "updated_at", "message",
"history": [{"state", "at"}]}`. `state` is a `JobState`; `history` is
append-only and every adjacent pair is an allowed transition. `state.json` is
kept after the run ends — the history is the only record of queueing time.

## The job backend: interface only

`attestation_schema.jobs` defines the contract a runner (local process,
Slurm, a cloud batch queue) implements; this spec ships **no**
implementation.

```
            ┌──────────► cancelled ◄──────────┐
            │                                 │
  submit ─► submitted ───────► running ───────┼──► complete
            │                     │           │
            └──────────► failed ◄─┘           │
                                              (terminal: no exits)
```

`JobState` = `submitted | running | complete | failed | cancelled`;
`TRANSITIONS` encodes the arrows above and `can_transition(a, b)` reads it.
`submitted → failed` is real (a job a scheduler rejects never runs);
`submitted → complete` is not (nothing completes without running).

```python
class JobBackend(Protocol):
    name: str
    def submit(self, spec: JobSpec) -> JobHandle: ...
    def status(self, handle: JobHandle) -> JobStatus: ...
    def cancel(self, handle: JobHandle) -> JobStatus: ...
    def result(self, handle: JobHandle) -> RunRecord: ...
```

- `JobSpec(run_id, family, arm, config, run_dir, ext)` — what to run and
  where; `config` and `ext` mean what they mean in the record. How a backend
  turns a spec into work is its own business (a smeltery backend reads
  `config`); the interface does not model commands.
- `submit` writes `state.json` (`submitted`) before returning and raises
  `JobExistsError` if `run_dir` already holds a `state.json` or `run.json` —
  resubmission is a new `run_id`, never an overwrite.
- `status` returns `JobStatus(state, updated_at, message)` from the backend's
  view, which `state.json` mirrors.
- `cancel` is idempotent: on a terminal job it returns the unchanged status
  rather than raising, because a cancel that races completion is normal.
- `result` returns the core record for a terminal job and raises
  `JobNotTerminalError` otherwise. **Every terminal job has a record**: a
  backend whose job died without writing one writes a `failed` record itself
  (it knows `run_id`, `family`, `arm`, `config` and the times), so the scanner
  never has to infer failure from absence.

## Fit check (walked through, not built — producers are roadmap 3-5)

**An MLflow sweep** — **built** (roadmap 3, 2026-11-19); "Producer
findings" below supersedes the walk-through that stood here. The real sweep
overturned three of its calls: `arm` is not the run name, history cannot be
dropped, and dataset inputs are not an `ext`-only matter.

**A smeltery funnel run.** `family` = campaign, `arm` = the configuration
variant, `config` = today's `inputs`, `code` = smeltery's own identity
(`smeltery_identity()`'s commit/dirty/remote — the code that ran), `data` =
the ligand library and receptor files with role `input` plus `digests.json`
with role `manifest`, metrics such as `hit_rate` (`direction: "max"`) and
per-tier survivor counts named `tier.<tier>.survivors` (a tier is not a data
`split` and not a training `step`, so it goes in the name), `dependencies` =
ferric (below) plus Python. Everything chemistry-shaped lives in
`ext.smeltery`: `{"tiers": [...], "input_digest": "...", "host": {...},
"legacy_schema_version": 1}`. Each tier's outputs land in `steps/NN-<tier>/` (superseded: the real
producer keeps its outputs where its GUI contract puts them and writes only a
`stage.json` per step; see Producer findings, 2).
smeltery's `record_sha256` is not carried: the record's digest is the file's
sha256. **Misfit found and resolved:** smeltery's provenance is a sentence
("VERIFIED from direct_url.json (git install)"); the core splits it into the
enum (`verified`) and `evidence` (`direct_url.json (git install)`), so a
reader can filter on the enum without parsing prose.

**A ferric build as a dependency identity.** With `ferric.__build__`:
`{"name": "ferric", "version": "0.1.0", "build_sha": "<40 hex>", "dirty":
false, "provenance": "verified", "evidence": "ferric.__build__"}`. `version`
alone is not an identity (every ferric build reports 0.1.0), which is why
`build_sha` exists. A dirty stamped build is `"dirty": true, "provenance":
"verified"` — verified that it was dirty. A local-directory install is
`inferred` with the checkout's HEAD; an unstamped wheel is `inferred`, `build_sha:
null`, its pip hash in `evidence`; nothing found is `unknown` with every
identity field `null`.

## Producer findings

Each finding is what a real producer needed, what changed in the schema
(`record.py`, `run-1.schema.json`, this spec and the package tests change
together), and why. The rule while the draft is open: **a field a producer can
only fill with a workaround is a schema bug**, fixed before the producer moves
on. Changes here break draft records written earlier; that is what draft
means.

### 1. MLflow (roadmap 3, 2026-11-19)

**Producer:** `examples/mlflow-records/export_records.py` (golden path
`examples/mlflow-records/`). It reads MLflow through `MlflowClient` — so a
file store, a sqlite store and a tracking server read the same way — and
writes only through `attestation_schema` (`layout.write_manifest`, then
`write`). It imports nothing from `attestation`, as smeltery does not. It is
an example, not a module under `src/attestation`, because `attestation` does
not depend on the schema package at runtime (Open decision 2) and the ledger's
MLflow adapter is not refactored until roadmap 11/12; when the ledger imports
core records, this mapping is the one it should share.

**Evidence:** the committed real sweep (`examples/flows/training/mlruns`, four
runs of `train_mlflow.py`, mlflow-skinny 3.15.2), exported to the committed
`examples/mlflow-records/records/`, byte-identical on re-export; and a fresh
real MLflow store written in the test suite (`tests/test_mlflow_records.py`)
from a script inside a git checkout, covering what the committed sweep does
not: a diverged loss (`inf`, `NaN`), two values at one step, two dataset
inputs MLflow names identically, an artifact, a killed run, a run still
`RUNNING`, and a repo URL set by hand with a token in it.

**Mapping:**

| MLflow | core record | why |
|---|---|---|
| experiment name | `family` | MLflow's own comparison group. |
| (nothing) | `arm`: `null`, or `--arm-param P...` → `"P=v,..."` | MLflow has no arm. The real sweep gives all four arms one run name (`c_sweep`), so `arm = run name` would declare four different configurations replicates of one; MLflow's default run names are unique per run, so it would equally make every replicate its own arm. The arm is a function of the params the person varied, which only they can name. |
| run id | `run_id`; run dir `run-<run_id>` | Globally unique in MLflow. |
| `FINISHED`/`FAILED`/`KILLED` | `complete`/`failed`/`cancelled` | `RUNNING`/`SCHEDULED` get no record (terminal-only); a deleted run gets none (the ledger's rule: the user removed it). |
| `start_time`/`end_time` (epoch ms) | `started_at`/`finished_at` | Via `format_timestamp`; see change 4. |
| params | `config`, verbatim strings | `"C": "10.0"`, never re-typed — MLflow stores strings and the producer must not guess. |
| every logged point | `history` | Change 1. One series per (metric key, linked dataset). |
| MLflow's latest point | `metrics` | MLflow's own key: greatest step, then latest timestamp, then largest value; tested equal to MLflow's own `run.data.metrics` value for every metric. A first version broke ties by logging order and disagreed with MLflow whenever two values at one step landed in the same millisecond -- caught only because the test compares against MLflow, not against a reimplementation. |
| metric direction | `null` unless `--direction NAME=min\|max` | MLflow records none; declared, never inferred. |
| linked dataset (`log_metric(dataset=...)`, MLflow 3) | `split` | MLflow's own "which data this value is on". |
| `mlflow.source.git.commit` | `code.git_sha`, `provenance: "inferred"` | MLflow reads HEAD of the source's checkout at run start, so the code that ran may differ (smeltery's change 2 below gave `code` the provenance to say so); `dirty: null` — MLflow never records it; unknown, not clean. `version: null`. With no tag, `provenance: "unknown"`. The committed sweep had its git tags scrubbed before commit, so its records say `null`/`unknown`, honestly. |
| `mlflow.source.git.repoURL` | `code.repo`, credentials stripped | MLflow 3.15 strips credentials from the remote it detects, but stores a hand-set tag verbatim; the producer strips both copies (core and `ext`). |
| artifacts | `data`, role `output`, sha256 of the stored bytes | Hashed, not copied: the bytes stay in MLflow's store, bound by digest. |
| dataset inputs | `data`, `datasets/<name>@<mlflow digest>`, `sha256: null`, role = `mlflow.data.context` (`input` if absent) | Change 2. MLflow's digest disambiguates the name (two inputs both named `data` are real) and stays verbatim in `ext.mlflow.datasets`. |
| params named by `--seed-param` | `seeds` | `null` unless declared; a declared param that is not an integer is refused. |
| (no environment recorded) | `dependencies`: `mlflow`, `provenance: "unknown"` | A run's store records no client or library version; open questions 2-3 below. |
| experiment id, run name, all tags, dataset inputs | `ext.mlflow` | MLflow's own identifiers and vocabulary, carried untouched. |

**Schema changes MLflow forced:**

1. **`history` (new required top-level key).** MLflow logs curves: the
   committed sweep's `train_loss` has ten points. The draft said "final
   values, not curves", which left two workarounds: drop the curve (lossy —
   it is what the ledger's MLflow adapter already does, and why its README
   says a user who wants curves is not served), or put it in `ext.mlflow`
   (but curves are not MLflow's: W&B, TensorBoard, Sacred and Keras all log
   them, so every tracker would invent its own). Curves stay out of `metrics`,
   whose entries mean *final value* — mixing ten `train_loss` entries in would
   make every consumer pick the last one by hand. Shape: series keyed by
   `(name, split)` with ordered points `{step, value, at}`. Steps may repeat
   because MLflow really does log two values at one step (both kept, in
   order); `at` keeps MLflow's per-point timestamp, nullable for producers
   with none. Cost: a long run makes a large record (100k points ≈ 6 MB);
   open question 1.
2. **`data[].sha256` nullable.** The walk-through called putting MLflow's
   dataset inputs only in `ext.mlflow.datasets` "the extension mechanism
   working". Re-examined against a real store, it is a workaround: with no
   core entry the record says the run read *no* data, which is false, and a
   consumer guarding comparisons by data (the ledger's corpus guard) cannot
   see that two arms trained on different datasets. MLflow knows the name, the
   role (`mlflow.data.context`: training/eval) and a source; it has no sha256
   of any bytes a reader could fetch (its digest is a truncated
   MLflow-specific hash, and an HTTP or in-memory source cannot be hashed by
   the producer). So the core says what is known — named, role, bytes not
   digested — with `null`, as `git_sha: null` and `dirty: null` do, and
   MLflow's digest stays in `ext.mlflow` where its algorithm is named. What
   was *right* in the walk-through survives: a non-sha256 digest is never put
   in `sha256`.
3. **Non-finite metric values.** MLflow stores and returns `NaN` and `inf`
   (a diverged loss); the draft's "finite number" left only dropping the
   metric — which makes a diverged arm look like one with no loss, so a
   comparison would silently rank it on its other metrics — or stringifying
   it in `ext`. JSON has no NaN and JCS refuses one, so on the wire the value
   is one of the strings `"NaN"`, `"Infinity"`, `"-Infinity"` (spellings
   Python's `float()` and JavaScript's `Number()` both read); in Python it is
   the float, `dumps` spells it and `loads` reads it back. Only in
   `metrics[].value` and `history[].points[].value`: `config` and `ext` still
   refuse non-finite numbers.
4. **One spelling per timestamp.** The draft claimed `Z`-only gave one
   instant one spelling; it did not — `.024`, `.024000` and `.0240` were all
   accepted. MLflow's millisecond clock made the producer choose, and two
   producers exporting the same run would have written different bytes for
   the same instant, so different record digests. The fraction is now 1-6
   digits with no trailing zero (`:46Z`, `:45.99Z`, `:46.024Z`), and
   `attestation_schema.format_timestamp(datetime)` produces exactly that
   (refusing a naive datetime rather than assuming UTC). Python's
   `isoformat()` always writes six digits, so a producer using it directly is
   refused whenever the microseconds end in zero — the helper is the fix.

**Not schema changes (and why):**

- *Run name, experiment id, all tags* → `ext.mlflow`. MLflow's own
  identifiers; no generic consumer needs them to compare runs.
- *Empty `artifacts/` directories.* git cannot commit an empty directory, and
  MLflow 3's file store refuses a run directory without `artifacts/` ("Run
  ... not found") — so `MlflowClient` cannot read the committed sweep as
  committed. The golden path copies the store and recreates those
  directories. A fixture-transport problem, not a schema or mapping one.
- *An MLflow file-store defect seen in passing:* a metric logged with
  `dataset=` comes back from `get_metric_history` with `step=0` whatever
  step was logged (3.15.2; the file on disk has the right step). The producer
  reads through the client, so it inherits this on the deprecated file
  store; not worked around.
- *The run directory holds only `attestation/`* for an MLflow run: artifacts
  are bound by digest in `data`, not copied, so `digests.json` lists no
  files. The manifest is still written, so every run directory has one shape.

**Open decisions — where MLflow evidence bears on them:**

- *Decision 3 (every key required).* Held up: every MLflow field mapped to a
  present key, and `null` separated "MLflow did not say" (`dirty`,
  `direction`, `seeds`, dataset `sha256`) from a value each time. One gap it
  exposed: `dependencies: []` cannot separate "ran on nothing" from "the
  producer cannot see the environment"; the producer lists `mlflow` as
  `unknown` instead (open question 3).
- *Decision 5 (terminal-only status).* Held up: a `RUNNING` run is skipped
  and reported; a killed run is `cancelled`.
- *Decision 1 (freeze criteria).* Unchanged; MLflow is one of three.

**Open questions from MLflow (not decided):**

1. *History size.* In-record history is simple and digest-bound, but a long
   run (100k steps × several metrics) makes a multi-megabyte record that the
   strict reader parses slowly. The alternative is a schema-owned sidecar
   under `attestation/` bound by digest; not built until a real producer
   needs it.
2. *Dependencies a run can report.* MLflow records versions only for a logged
   model (`MLmodel`'s `mlflow_version`, `requirements.txt`,
   `python_env.yaml`). mlflow-skinny without pandas cannot log a model, so no
   real one exists here to map from; reading them waits until one does,
   rather than being parsed from documentation.
3. *Nullable `dependencies`.* Should `null` mean "the producer cannot see the
   environment", as it does for `seeds`? Deferred to the smeltery and third
   producer evidence.

### 2. smeltery (roadmap 4, 2026-11-19)

**Producer:** smeltery commit `90efc3b` (separate repository), mapping and
findings in its `docs/core-record.md`, golden record
`tests/data/core_record/benzoic-rhf.run.json`; written against this package
at `93bb5ab`. smeltery left fields out rather than bend another one, and
reported thirteen findings. Each was checked against the MLflow evidence
above. **Every accepted change below breaks smeltery's golden record** (new
required keys), so it re-pins and regenerates.

**Accepted (schema changed):**

1. **`code` gets `provenance` and `evidence`** (smeltery 2). smeltery's own
   identity is usually inferred (an editable install reports the checkout's
   HEAD *now*). MLflow independently needs the same: its git tag is HEAD at
   run start, uncommitted changes unknown — so two of two producers had a
   code identity they could not qualify. Same enum and rule as a dependency:
   `verified` needs `git_sha` or `version`.
2. **`code.version`** (smeltery 3), nullable. A non-editable install has a
   version and no commit; converting an old run, `producer.version` is
   today's converter, not the code that ran. MLflow: `null`.
3. **`metrics[].unit`** (smeltery 4), nullable free string, UCUM suggested;
   one unit per name, like direction. ddE is kcal/mol and a reader cannot
   compare across producers without it. MLflow has no units: `null`.
4. **`metrics[].labels` and `history[].labels`** (smeltery 5), `{string:
   string}`, `{}` for none, part of the uniqueness key. smeltery's ddE is per
   analogue and went into the name (`paired_ddE.<analogue>.mean`) — the
   workaround the walk-through itself prescribed for tiers; a reader grouping
   by candidate then parses names. MLflow 3 attaches a `model_id` to metric
   points, which is the same shape (what a value is about); the MLflow
   producer writes `{}` today because no logged model exists in this
   environment to test against (MLflow open question 2).
5. **Seed lists** (smeltery 7): `seeds` values are `int | [int, ...]`
   (non-empty). Docking records `seeds: [7]`, one Vina run per seed; `{}`
   would claim no seed was set and `docking.0` would invent a name.
6. **`dependencies[].source`** (smeltery 9), nullable, credentials refused:
   where a package was installed from, so a reader filters on it without
   parsing `evidence`, the same argument that split `provenance` from prose.
7. **`steps/` and `logs/` optional** (smeltery 11); outputs may live anywhere
   under the run directory and the manifest still covers them. MLflow agrees:
   an MLflow run directory has neither.
8. **Python floor 3.11** (smeltery 12; then required by the user). The package
   used nothing 3.12-only; its full test suite passes on CPython 3.11.15
   against the installed wheel, and CI's `schema-py311` job runs it there.
   `attestation` itself stays at 3.12.

**Accepted as spec text only (no field):**

- *JCS and numbers* (smeltery 1). `config`/`ext` hold JSON numbers with no
  int/float distinction; a producer digest is checkable from the record only
  if taken over JCS bytes. **Rejected:** an optional `config_digest
  {algorithm, value}` — a digest over a serialization the record does not
  preserve cannot be checked from the record, so a core field would advertise
  a verification it cannot offer. smeltery's `input_digest` belongs in
  `ext.smeltery`, verifiable against the legacy record bound in `data`.
- *Absent, NaN, boolean* (smeltery 6). Absent = not reported; `"NaN"` =
  computed and undefined (MLflow change 3 already made it writable, so a
  one-pose SEM is now `"NaN"`, not omitted); no null or boolean value
  (smeltery's `unranked: true` is a flag, so `ext`).
- *`run_id` is not unique* (smeltery 8). Requiring uniqueness would push
  smeltery to suffix a time onto an input digest — inventing an execution id.
  Instead: same `(family, run_id)` = re-executions of one run, both kept,
  told apart by `record_digest`. MLflow ids are per execution and unaffected.

**Deferred, said so:** *one identity per dependency name* (smeltery 10). A
`--temporal` run's SCFs can come from several ferric builds; until a reader
needs per-scope identities, a run that mixed builds records that dependency
`unknown` with `evidence` naming the mix. *Publishing to an index* (smeltery
13) is Open decision 2.

## Open decisions (for the user)

1. **Freeze criteria.** Proposed: freeze `attestation.run/1` when the three
   producers above have each written real records that the reader accepts and
   the ledger has imported. Until then the identifier stays `attestation.run/1`
   (not `1-draft`), on the grounds that a renamed identifier would itself be a
   breaking change at freeze time; the cost is that pre-freeze records may
   become unreadable.
2. **attestation's runtime dependency on the schema package.** Today
   `attestation-schema` is a uv workspace member installed in the dev group
   only, so `attestation`'s wheel and `uvx --from git+...` installs are
   unchanged. Making it a runtime dependency (so the ledger imports core
   records) needs it published to an index first.
3. **Required-and-nullable vs optional keys.** Chosen: every key required.
   It makes hand-written records verbose; it makes absent-vs-null impossible.
4. **`producer`** was added beyond the roadmap's field list: without it a bad
   record cannot be traced to the code that wrote it.
5. **Terminal-only `status`.** A record is never written for a run in flight.

## Testing

`packages/attestation-schema/tests/test_attestation_schema.py` (on pytest's
`testpaths`): round trip; history order and repeated steps; non-finite metric
values spelled and read back; data named without a digest; timestamp
spelling and `format_timestamp`; an unknown `ext` namespace preserved byte for byte;
**every** required field, top-level and nested, removed in turn → the specific
error naming its path; unknown fields refused; strict parse (duplicate key,
`NaN`, huge int, `true` as a metric value); version dispatch (unknown version
refused with the known list, a registered reader is used); writer determinism
(same record twice, key order of the input irrelevant, JCS parity with
`attestation.run_record.canonical`); the JSON Schema file agrees with the
reader on every valid and invalid case; layout and manifest; job state
machine. The wheel builds with `uv build --package attestation-schema` in CI's
wheel smoke job, which installs it into a clean venv and reads a record.
`tests/test_boundaries.py`: the import rules, with detector self-tests.
`tests/test_mlflow_records.py`: the MLflow producer (Producer findings, 1) —
the committed records validate, round-trip byte for byte and preserve every
point of the raw MLflow metric files without mlflow installed; with mlflow,
re-export is byte-identical to the committed records and a fresh real store
maps without workarounds.
