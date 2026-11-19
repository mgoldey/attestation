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
| `run_id` | non-empty string | Producer-chosen, unique within a family. smeltery's `<digest12>` and an MLflow run id both fit. |
| `family` | string, nullable | The comparison group (an MLflow experiment, a smeltery campaign). `null` = the run belongs to no comparison. |
| `arm` | string, nullable | The variant within the family. |
| `status` | `"complete"` \| `"failed"` \| `"cancelled"` | Terminal states only: a record is written once, when the run ends. In-flight state lives in `state.json` (below). |
| `started_at`, `finished_at` | RFC 3339 UTC `YYYY-MM-DDTHH:MM:SS[.ffffff]Z`, nullable | `Z` only, no offsets, so one instant has one spelling. `finished_at >= started_at` when both are known. |
| `producer` | `{name, version}` | What wrote this file (`smeltery 0.4.0`, `attestation mlflow-import 0.6.0`). `version` nullable. Not the code that ran — that is `code`. |
| `code` | `{git_sha, dirty, repo}` | The experiment code's identity. `git_sha` 40 or 64 lowercase hex, nullable; `dirty` bool, nullable (unknown ≠ clean); `repo` nullable string, refused if it carries `user:password@` credentials. |
| `config` | JSON object | The run's configuration, verbatim. Opaque to attestation. |
| `data` | list of `{name, sha256, role}` | Data the run read or produced, by digest. `sha256` 64 lowercase hex; `role` a free non-empty string (suggested: `input`, `train`, `eval`, `test`, `output`, `manifest`). Names unique. |
| `metrics` | list of `{name, value, direction, step, split}` | Final values, not curves. `value` a finite number (a JSON `true` is refused, not read as 1); `direction` `"min"` \| `"max"` \| `null` (null = the producer did not say; **never inferred**); `step` int ≥ 0, nullable; `split` string, nullable. `(name, step, split)` unique; one `direction` per `name`. |
| `seeds` | object `{name: int}`, nullable | Named RNG seeds (`{"python": 0, "numpy": 0}`). `null` = unknown; `{}` = the producer states no seed was set. |
| `dependencies` | list of `{name, version, build_sha, dirty, provenance, evidence}` | Identity of what the code ran on. `provenance` is `"verified"` \| `"inferred"` \| `"unknown"`; `verified` requires `version` or `build_sha`. `evidence` nullable free text naming where the identity came from (`ferric.__build__`, `direct_url.json`). Names unique. |
| `ext` | object `{namespace: object}` | Producer extensions. Namespace `^[a-z][a-z0-9_-]*(\.[a-z0-9_-]+)*$`; `attestation` and `attestation.*` reserved for this schema. |

Lists keep the producer's order (the writer never sorts them: order can mean
something, and a stable producer gives stable bytes anyway). Objects are
key-sorted by canonicalization.

### Extensions round-trip untouched

`ext` is how a producer carries what the core does not model (smeltery's
tiers, MLflow's own dataset digests) without a schema change. The reader
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

**An MLflow sweep** (4 learning rates). One record per MLflow run.
`family` = experiment name, `arm` = run name, `config` = params *as MLflow
stored them* (strings — the producer must not guess types), `metrics` = the
final value of each metric with `step` = its last step and `direction: null`
(MLflow never records one; the ledger's no-inference rule holds), `seeds` =
`null` unless the training script logged one by a name the producer is told,
`code.git_sha` from the `mlflow.source.git.commit` tag with `dirty: null`
(MLflow does not record it — unknown, not clean), `status` FINISHED→complete,
FAILED→failed, KILLED→cancelled; a RUNNING run gets no record. **Misfit
found:** MLflow dataset inputs carry MLflow's own digest, not a sha256, so they
cannot go in `data`; they go in `ext.mlflow.datasets` verbatim. That is the
extension mechanism working, not a gap to close by loosening `sha256`.

**A smeltery funnel run.** `family` = campaign, `arm` = the configuration
variant, `config` = today's `inputs`, `code` = smeltery's own identity
(`smeltery_identity()`'s commit/dirty/remote — the code that ran), `data` =
the ligand library and receptor files with role `input` plus `digests.json`
with role `manifest`, metrics such as `hit_rate` (`direction: "max"`) and
per-tier survivor counts named `tier.<tier>.survivors` (a tier is not a data
`split` and not a training `step`, so it goes in the name), `dependencies` =
ferric (below) plus Python. Everything chemistry-shaped lives in
`ext.smeltery`: `{"tiers": [...], "input_digest": "...", "host": {...},
"legacy_schema_version": 1}`. Each tier's outputs land in `steps/NN-<tier>/`.
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
`testpaths`): round trip; an unknown `ext` namespace preserved byte for byte;
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
