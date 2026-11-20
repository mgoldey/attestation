# The core run record, the run directory, and the job-backend interface

**Date:** 2026-11-19
**Status:** v1 FROZEN 2026-11-19 (roadmap items 1, 2, 6 and 43).
**Frozen means:** `attestation.run/1`, `attestation.history/1`,
`attestation.state/1` and `attestation.digests/1` never change again. A record
the reader accepts today is accepted by every later reader, and anything that
would change what a v1 reader accepts or what the writer emits -- an additive
optional field included -- is a new version (see "v1 freeze" below for the
compatibility policy and what each producer changed before it).
`SCHEMA_STATUS` in `attestation_schema.record` is `"frozen"` and
`packages/attestation-schema/tests/test_frozen.py` pins the JSON Schema files
and the writer's bytes by digest.
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
smeltery ──► attestation-schema ◄── attestation (runtime dependency since roadmap 11)
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
| `status` | `"complete"` \| `"failed"` \| `"cancelled"`, nullable | Terminal states only: a record is written once, when the run ends. In-flight state lives in `state.json` (below). `null` = the producer could not observe how the run ended -- an importer of artifacts someone else wrote (a results file says a value exists, not that its run finished). A producer that ran the experiment, and a job backend, always know and never write `null` (Producer findings 5.E). |
| `started_at`, `finished_at` | RFC 3339 UTC `YYYY-MM-DDTHH:MM:SS.ffffffZ`, nullable | `Z` only, no offsets, and **always exactly six fraction digits** (`:37.500000Z`, `:37.000000Z`), so one instant has one spelling *and* timestamps sort as text in time order (smeltery finding 10). `format_timestamp(datetime)` produces it. The same spelling is used for history points' `at`. `finished_at >= started_at` when both are known. |
| `producer` | `{name, version}` | What wrote this file (`smeltery 0.4.0`, `attestation mlflow-import 0.6.0`). `version` nullable. Not the code that ran — that is `code`. |
| `code` | `{git_sha, dirty, repo, version, provenance, evidence}` | The experiment code's identity, on the same ladder as a dependency. `git_sha` 40 or 64 lowercase hex, nullable; `dirty` bool, nullable (unknown ≠ clean); `repo` nullable string, refused if it carries `user:password@` credentials; `version` the code's own version, nullable (a non-editable install may have only this; `producer.version` is not it when converting old runs); `provenance` `"verified"` \| `"inferred"` \| `"unknown"`, `verified` requiring `git_sha` or `version`, `unknown` requiring `git_sha: null` (a sha held but not vouched for is `inferred`); `evidence` nullable free text. **Mixed code** (results from more than one commit): `provenance: "unknown"`, `git_sha: null`, `dirty: null`, `version` only if every commit shares it, `evidence` naming each commit and how many result rows it produced (`mixed: <sha> (412 rows), <sha> (88 rows)`) — the dependency rule, applied to the code (smeltery finding 9). |
| `config` | JSON object | The run's configuration, verbatim. Opaque to attestation. Numbers are JSON numbers: JCS writes `2.0` as `2` and the reader returns an int, so there is no int/float distinction in `config` or `ext`. A producer digest over these values is checkable from the record only if it was taken over their JCS bytes (`attestation_schema.canonical.canonicalize`); a digest over another serialization is the producer's claim, carried in `ext` and bound to its source file through `data`. |
| `data` | list of `{name, sha256, role}` | Data the run read or produced. `sha256` 64 lowercase hex of the bytes, nullable: `null` = the producer knows the run read or wrote this but could not digest its bytes (never a digest in another algorithm — that belongs in `ext`). `role` a free non-empty string (suggested: `input`, `train`, `eval`, `test`, `output`, `manifest`). Names unique. |
| `metrics` | list of `{name, value, direction, step, split, unit, labels}` | Final values, not curves (curves are in the `history` sidecar). `value` a finite JSON number, or one of the strings `"NaN"`, `"Infinity"`, `"-Infinity"` (read as the Python float; a JSON `true` or any other string is refused); `direction` `"min"` \| `"max"` \| `null` (null = the producer did not say; **never inferred**); `step` int ≥ 0, nullable; `split` string, nullable; `unit` nullable free string (UCUM suggested, e.g. `kcal/mol`); `labels` an object `{string: string}`, `{}` for none — what the value is *about* (an analogue, a model), so a reader groups without parsing names. `(name, step, split, labels)` unique; one `direction` and one `unit` per `name`. **Absence**: a metric that is not in the list was not reported; `"NaN"` means computed and undefined (the SEM of one sample). There is no null value and no boolean value (a flag is not a metric: it belongs in `ext`). |
| `history` | `{path, sha256, series, points}`, nullable | A **reference** to the logged curves, never the curves: `path` is always `"attestation/history.jsonl"` (the sidecar, below), `sha256` 64 lowercase hex of its bytes, `series` int ≥ 1 and `points` int ≥ `series` its counts, so a reader can size it without opening it. `null` = no curve was recorded. Each series is `(name, split, labels)` with points `{step, value, at}` in logging order (steps may repeat and need not increase — it is a log, not a map; `value` as in `metrics`; `at` nullable). A series need not have a `metrics` entry or vice versa; direction and unit are properties of the metric name, so both live on `metrics` only. Producer findings 3. |
| `seeds` | object `{name: int \| [int, ...]}`, nullable | Named RNG seeds (`{"python": 0, "numpy": 0}`); a non-empty list is one stage run once per seed, in order (`{"docking": [7]}`). `null` = unknown; `{}` = the producer states no seed was set. |
| `dependencies` | list of `{name, version, build_sha, dirty, provenance, evidence, source}`, nullable | Identity of what the code ran on. **A list is observed, not exhaustive** (Producer findings 5.B): every entry is something the run did run on, and a name's absence says nothing -- never "the run did not use it". `null` = the producer made no observation of the environment; `[]` = it observed and has nothing to name (the `seeds` convention; Producer findings 3). A dependency the producer knows the run used but cannot identify is listed `unknown`. `provenance` is `"verified"` \| `"inferred"` \| `"unknown"`; `verified` requires `version` or `build_sha`. `evidence` nullable free text naming how the identity was established (`ferric.__build__`, `direct_url.json`). `source` nullable: where it was installed from, in a closed, prefixed vocabulary — `index` or `index:<name or url>` (`index:pypi`), `<vcs>+<url>` for `git`, `hg`, `svn`, `bzr` (pip's own spelling), `url:<url>` (a direct-URL archive, PEP 610 `archive_info`: `pip install https://…/x.whl`; never `url:file:`, and no query string), `file` (a local path or wheel; the path is machine-specific, so not recorded), `conda` or `conda:<channel>` (`conda:conda-forge`), `system` (the OS or interpreter provided it); any URL in it credential-free, as for `code.repo` (`git+https://token@...` included). `null` = not known (smeltery finding 12, Producer findings 5.A). Names unique: one identity per dependency; a run that mixed builds of one dependency records it `unknown` with `build_sha: null` (enforced: `unknown` claims no sha), its `evidence` naming each build (deferred until a reader needs per-scope identities). |
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

JCS lives in `attestation_schema.canonical` — the schema package may not
import `attestation`. It was first a copy of `attestation.run_record.canonical`;
since attestation took a runtime dependency on the schema package (roadmap 11),
`run_record.canonical` is a re-export and the copy is gone (the parity test
asserts the two names are one function).

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
│   ├── digests.json             manifest: sha256 of every file outside attestation/
│   └── history.jsonl            logged curves (optional); bound by run.json's history.sha256
├── steps/
│   ├── 00-<name>/               one directory per step / tier, in execution order
│   └── 01-<name>/               index is 2+ digits; name ^[a-z0-9][a-z0-9._-]*$
└── logs/
    ├── run.log                  whole-run log
    └── <NN>-<name>.log          one per step, named as its step directory
```

`attestation_schema.layout` owns these names (`RUN_JSON`, `STATE_JSON`,
`DIGESTS_JSON`, `HISTORY_JSONL`, `step_dir(run_dir, index, name)`, `step_log(...)`,
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

**Write order** is what makes a half-written directory honest: steps, logs and
`history.jsonl` (streamed during the run, finalized when its writer closes) →
`digests.json` → `run.json`. `write` refuses a record whose `history` names a
sidecar that is missing or has other bytes, and the history writer refuses to
start or finish once `run.json` exists. A directory with `run.json` is complete; one
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
`history.jsonl` sits under `attestation/`, so the manifest never lists it: it
is bound exactly once, by the record's `history.sha256`. Listing it in both
would let the two disagree (a sidecar rewritten between the manifest and the
record) and leave a reader to decide which binding wins.

### The history sidecar: `attestation/history.jsonl`

Owned by `attestation_schema.history`; `"attestation.history/1"`. JSON Lines:
each line is one object in RFC 8785 canonical form followed by `\n` — the
last line too, so a file cut short mid-line is detectable even before its
digest is compared.

```
{"schema":"attestation.history/1"}                                   line 1, always
{"labels":{},"name":"loss","series":0,"split":"train"}               declares series 0
{"at":"2026-11-19T10:00:01.5Z","series":0,"step":0,"value":0.9}      a point of series 0
{"labels":{},"name":"loss","series":1,"split":"eval"}                declares series 1
{"at":null,"series":1,"step":0,"value":"NaN"}
{"at":null,"series":0,"step":1,"value":0.8}                          series interleave
```

- **One point per line**, not a chunk of points per line. A chunk size is a
  free parameter: the same points chunked twice would be two files with two
  digests. With one point per line the bytes are a pure function of the
  sequence of writer calls, and a producer appends a line per logged value
  with nothing to buffer.
- **Series declared inline**, once, immediately before their first point,
  with ids `0, 1, 2...` in order; every point names its series by id. Series
  may interleave because training loops log `train/loss` and `eval/loss`
  alternately — requiring each series to be contiguous would make a
  long-running producer hold every curve in memory until the end, which is
  the cost the sidecar exists to remove. One series per `(name, split,
  labels)`, every declared series has at least one point.
- **Values** are spelled as in `metrics` (`"NaN"`, `"Infinity"`,
  `"-Infinity"` for non-finite), parsed with the record's strict parser
  (duplicate keys, bare `NaN`, integers past ±2^53 refused).
- **Writing.** `HistoryWriter(run_dir)` streams lines to a temporary file in
  `attestation/` (`series(name, split, labels) -> handle`, `point(handle,
  step, value, at)`, `flush()`), validating each value as the reader will;
  `close()` fsyncs it, links it into place without clobbering, and returns the
  `HistoryRef` for the record — or `None` when no point was logged (a series
  declared and never logged writes nothing), which is the record's `null`. An
  exception inside `with HistoryWriter(...)`, or `abort()`, leaves no
  sidecar. `write_history(run_dir, series)` writes whole curves at once
  (each series in order, its points in order) and gives the same bytes as
  the same calls made one at a time.
- **Reading.** `read`/`loads` never open the sidecar: a record is small
  however long the run. `read_history(run_dir, ref=None)` (ref a record or its
  `history`; omitted, `run.json` is read for it) returns `None` for a `null`
  reference, else checks the sha256 before parsing a line and raises
  `HistoryMissingError`, `HistoryDigestError` or `HistoryFormatError`
  (malformed or truncated line, out-of-order or undeclared series, series
  and point counts other than the reference's), all subclasses of
  `HistoryError(RecordError)`; a line error's `.path` is
  `attestation/history.jsonl:<line>[.<field>]`.
- **`metrics` and history are not cross-checked**, and were not before this
  change: a final value need not be a curve's last point (MLflow's final value
  is its greatest `(step, timestamp, value)`, which with two values at one step
  in one millisecond is not the last one logged), and smeltery reports metrics
  with no curve. Checking "agreement" would need one producer's notion of
  "final" in the schema.
- `schemas/history-1.schema.json` describes one line (header, series or
  point) for other languages; ordering, ids, the final newline and the digest
  are the Python reader's.

**`state.json`** (`"schema": "attestation.state/1"`, backend-owned; owned
by `attestation_schema.state`, JSON Schema `schemas/state-1.schema.json`):
`{"schema", "run_id", "backend", "job_id", "state", "updated_at", "message",
"history": [{"state", "at"}]}`, exactly these keys, JCS with no trailing
newline, replaced atomically (temporary file, fsync, `os.replace`). `state` is
a `JobState`; `message` a string or `null` (why the job ended as it did);
`history` starts at `submitted`, is append-only, every adjacent pair is an
allowed transition, its times never go backwards (a lagging clock is clamped
to the last entry) and its last entry is `state` at `updated_at`. `read_state`
refuses a file that breaks any of this (`StateFormatError`) rather than
guess. `state.json` is kept after the run ends — the history is the only
record of queueing time. The format is the one smeltery's backends wrote
before it moved here; two files smeltery wrote are the package's fixtures and
re-serialize byte for byte (Producer findings 5.D).

## The job backend: the interface, and what every backend shares

`attestation_schema.jobs` defines the contract a runner (local process,
Slurm, a cloud batch queue) implements; this spec ships **no** backend. What
every backend needs identically -- `state.json`, its lock, `settle` and the
minimal terminal record -- is `attestation_schema.state` (Producer findings
5.D), so the rules below are code, not prose each backend re-implements.

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
  where. How a backend turns a spec into work is its own business (a
  smeltery backend reads `config`); the interface does not model commands.
- **The spec reaches the record verbatim** (Producer findings 5.C). The
  submitter owns `run_id`: the handle, `state.json` and the job's `run.json`
  all carry `JobSpec.run_id`, whoever writes the record. The record carries
  the spec's `family`, `arm` and `config` as those fields **in every terminal
  state** (what the run was given -- the one configuration that exists
  whether or not it finished; what a producer derives from it goes in
  `ext.<producer>`), and every namespace of the spec's `ext` unchanged. A
  backend that cannot hand these to its producer refuses the spec at
  `submit`. `state.spec_conflicts(spec, record)` checks all of it.
- `submit` writes `state.json` (`submitted`) before returning and raises
  `JobExistsError` if `run_dir` already holds a `state.json` or `run.json` —
  resubmission is a new run directory, never an overwrite. It may reuse a
  `run_id` (re-executions share one; see the `run_id` row).
- `status` returns `JobStatus(state, updated_at, message)` from the backend's
  view, which `state.json` mirrors.
- `cancel` is idempotent: on a terminal job it returns the unchanged status
  rather than raising, because a cancel that races completion is normal.
- `result` returns the core record for a terminal job and raises
  `JobNotTerminalError` otherwise. **Every terminal job has a record**: a
  backend whose job died without writing one writes a `failed` (or
  `cancelled`) record itself (it knows `run_id`, `family`, `arm`, `config`
  and the times), so the scanner never has to infer failure from absence.
- **Settling** is `attestation_schema.state.settle`, under `state_lock`: end
  exactly once (a terminal job is returned unchanged); a record already in
  the run directory decides the terminal state (a cancel that races
  completion ends `complete`); a record that is not the job's, or cannot be
  read, or has `status: null`, ends the job `failed` with `message` saying
  why; no record: `complete` becomes `failed` and the backend writes
  `state.terminal_record` (no metrics; `history`, `seeds`, `dependencies`
  null; the backend's `producer`, `code` -- `state.UNKNOWN_CODE` by default --
  and `ext` namespaces), the record before the state.
- **`state.json` may lag; `run.json` is authoritative.** A backend that
  learns of the end only when it looks (a Temporal workflow writes no
  `state.json`) settles then: `status`, `cancel` and `result` settle a job
  found to have ended before returning. When the backend knows the real end
  time it passes `ended_at` (and `started_at`); otherwise the terminal entry's
  `at` and the record's `finished_at` are the time of the look.
- **The lock is POSIX `flock`** on `attestation/`. On Windows there is no
  `flock`; `state_lock` raises `StateLockUnavailableError` and no state is
  written (reading needs no lock). A Windows backend is out of scope for v1;
  adding one is a new lock implementation, not a format change.

## Fit check (walked through before the producers were built; superseded by "Producer findings")

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
together), and why. The rule while the draft was open: **a field a producer
can only fill with a workaround is a schema bug**, fixed before the producer
moves on. Changes here broke draft records written earlier; that is what draft
meant. Since the v1 freeze the same rule holds, but the fix is a new version
(see "v1 freeze").

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
| every logged point | `attestation/history.jsonl` via `write_history`; `history` = its reference | Change 1, moved out of the record by finding 3. One series per (metric key, linked dataset). |
| MLflow's latest point | `metrics` | MLflow's own key: greatest step, then latest timestamp, then largest value; tested equal to MLflow's own `run.data.metrics` value for every metric. A first version broke ties by logging order and disagreed with MLflow whenever two values at one step landed in the same millisecond -- caught only because the test compares against MLflow, not against a reimplementation. |
| metric direction | `null` unless `--direction NAME=min\|max` | MLflow records none; declared, never inferred. |
| linked dataset (`log_metric(dataset=...)`, MLflow 3) | `split` | MLflow's own "which data this value is on". |
| `mlflow.source.git.commit` | `code.git_sha`, `provenance: "inferred"` | MLflow reads HEAD of the source's checkout at run start, so the code that ran may differ (smeltery's change 2 below gave `code` the provenance to say so); `dirty: null` — MLflow never records it; unknown, not clean. `version: null`. With no tag, `provenance: "unknown"`. The committed sweep had its git tags scrubbed before commit, so its records say `null`/`unknown`, honestly. |
| `mlflow.source.git.repoURL` | `code.repo`, credentials stripped | MLflow 3.15 strips credentials from the remote it detects, but stores a hand-set tag verbatim; the producer strips both copies (core and `ext`). |
| artifacts | `data`, role `output`, sha256 of the stored bytes | Hashed, not copied: the bytes stay in MLflow's store, bound by digest. |
| dataset inputs | `data`, `datasets/<name>@<mlflow digest>`, `sha256: null`, role = `mlflow.data.context` (`input` if absent) | Change 2. MLflow's digest disambiguates the name (two inputs both named `data` are real) and stays verbatim in `ext.mlflow.datasets`. |
| params named by `--seed-param` | `seeds` | `null` unless declared; a declared param that is not an integer is refused. |
| (no environment recorded) | `dependencies`: `null` | A run's store records no client or library version. Was `[mlflow, unknown]` until finding 3 made `null` sayable; open question 2 below. |
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
   open question 1. **Superseded by finding 3**: the curves keep this shape
   but live in a sidecar the record references by digest.
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
   **Superseded by smeltery finding 10**: still one spelling per instant, but
   always six digits (`:46.024000Z`), so that text order is time order.

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
  producer cannot see the environment"; the producer listed `mlflow` as
  `unknown` instead, until finding 3 made `dependencies` nullable.
- *Decision 5 (terminal-only status).* Held up: a `RUNNING` run is skipped
  and reported; a killed run is `cancelled`.
- *Decision 1 (freeze criteria).* Unchanged; MLflow is one of three.

**Open questions from MLflow (not decided):**

1. *History size.* **Decided by the user (finding 3):** the sidecar.
2. *Dependencies a run can report.* MLflow records versions only for a logged
   model (`MLmodel`'s `mlflow_version`, `requirements.txt`,
   `python_env.yaml`). mlflow-skinny without pandas cannot log a model, so no
   real one exists here to map from; reading them waits until one does,
   rather than being parsed from documentation.
3. *Nullable `dependencies`.* **Decided by the user (finding 3):** yes.

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

**Second round** (smeltery `000a335d`, against this package at `314e4ff`;
smeltery then wrote `history: []`, `code.version/provenance/evidence`, metric
`unit` and `labels`, seed lists, `dependencies[].source` and `"NaN"`). All four
accepted; numbered on from the first round.

9. **Mixed experiment code → `code` gets the dependency rule.** A real run
   (danuglipron-halogen) has result rows from two smeltery commits, so `code`
   claimed one of them as `inferred` — a single-identity field filled with a
   workaround, which is a schema bug by this section's rule. The rule the
   spec already had for a dependency built from several builds now applies to
   the code: `provenance: "unknown"`, `git_sha: null`, `evidence` naming each
   commit and its row count. **And the reader now enforces the half that can
   be enforced**, for code and dependencies alike: `unknown` with a non-null
   `git_sha` / `build_sha` is refused (a sha is an identity claim; one you
   hold but cannot vouch for is `inferred`). Without it the danuglipron record
   would have stayed writable as "unknown, but here is one of the commits".
   `version` stays allowed under `unknown` (every ferric build reports 0.1.0:
   a shared version is not an identity claim). *Rejected for now:* a
   structured per-commit list in the core (`[{git_sha, rows}]`) — one
   producer has needed it once, the dependency case defers the same thing,
   and a producer can carry the breakdown in `ext.<namespace>` today; both are
   revisited together when a reader needs per-scope identities.
10. **Timestamps sort as text → fixed six-digit fraction.** MLflow finding 4
    made one spelling per instant by forbidding trailing zeros, which made
    `…:37.5Z` sort before `…:37Z` as a string — and records are sorted as
    strings constantly (filenames, `sort`, SQL `ORDER BY` on a TEXT column,
    the ledger). The alternative, "timestamps compare only after parsing", is
    a rule every consumer must remember and that fails silently when one does
    not. A fixed width keeps one spelling per instant (each instant has
    exactly one six-digit fraction) and makes text order time order for years
    0000-9999 at a cost of up to seven bytes per timestamp. Six, not three:
    microseconds are what Python's `datetime` holds, and narrowing would round
    a producer's value. `format_timestamp` now always writes them; a producer
    using `isoformat()` directly is refused when microseconds are zero (it
    drops the fraction) and for any offset other than `Z`, so the helper stays
    the way to write one.
11. **`from_dict` vs the model's NaN → a public wire spelling, `from_dict`
    unchanged.** The dataclass holds `float('nan')`; `from_dict` takes only the
    wire string. Making `from_dict` accept the float would make it accept a
    dict that is not JSON (`json.dumps` would write a bare `NaN`, which every
    strict parser, this one included, refuses), so the dict form would have
    two spellings of one value. Instead the spelling is public and applied
    where a dict is made: `RunRecord.to_dict()` now returns the wire form
    (`from_dict(r.to_dict()) == r`; before, `to_dict` returned the float and
    was not, as documented, "plain JSON values"), and
    `attestation_schema.spell_value(x)` is exported for producers that build
    dicts by hand. smeltery's duplicate of `_spell` can go.
12. **`dependencies[].source` gets a closed, prefixed vocabulary**:
    `index[:<name or url>]`, `git+<url>`, `file`, `conda[:<channel>]`,
    `system` (`SOURCE_KINDS`, reader-enforced, also a JSON Schema pattern).
    `source` was split from `evidence` (first round, 6) so a reader could
    filter on it without parsing prose; free text loses that again as soon as
    two producers write `pypi` and `PyPI` and `index`. smeltery's four cases
    map onto it (`index`, `git+<url>`, `file`, `conda`); `system` covers the
    interpreter and OS packages (Python itself, a system `xtb`). The prefixed
    forms keep the detail without letting it fragment the kind: a reader
    filters on the part before `:`/`+`. The credential check now also covers
    `git+https://token@...` and `index:<url>`, which the URL-scheme test
    missed before.

### 3. User-directed changes (2026-11-19)

Two changes the user decided, not a producer; both break draft records
(MLflow's committed records were regenerated with `generate.sh`, as they were
for finding 10's timestamps; smeltery, which pins this package by commit and
writes `history: []`, re-pins).

1. **Curves are isolated from the core record.** `history` was a required
   list embedding every point, so a long run made a multi-megabyte `run.json`
   that every reader — including one that wants only final values to compare
   arms — had to parse in full (MLflow open question 1). Now `run.json` never
   holds a point: `history` is `null` or `{path, sha256, series, points}`
   referencing `attestation/history.jsonl` ("The history sidecar" above).
   *Why a dedicated object and not a `data` entry with a reserved role:*
   `data` is what the run read or produced, digested for provenance; the
   history is part of the record's own content moved out for size, and has
   a schema-owned format a reader must be able to find without scanning
   `data` for a magic role. A reserved role would also make `role` — a free
   string today — partly closed, would let a record list two history entries
   that then need a rule, and could not carry the counts that let a reader
   size the curves without opening them. `path` is in the reference so a
   reader needs no layout knowledge, but is fixed to the one value so a
   reference can never point out of the run directory; a later version may
   widen it. *Why bound only by the record:* see `digests.json` above — one
   binding, no contradiction to arbitrate. Measured: 100k points make a
   ~6 MB sidecar and a `run.json` that is the same size as with ten
   (`test_a_long_run_keeps_run_json_small`).
2. **`dependencies` is nullable**, as `seeds` is: `null` = the producer could
   not observe the environment, `[]` = it observed one with nothing to name.
   *MLflow now writes `null`.* The training process did import mlflow, so
   mlflow is a real dependency of the run — but the exporter did not observe
   it: MLflow's store records no client version, and it records nothing about
   numpy, scikit-learn or Python either. A list says "these are what the run
   ran on"; `[mlflow (unknown)]` read as a list says the run ran on mlflow
   *and nothing else*, which is false and less true than `null`. That the run
   logged to MLflow is already said by `producer` and `ext.mlflow`. When a
   logged model gives real versions (open question 2), the exporter lists
   what it can read from it.

### 4. The ledger's importers (roadmap 11, 2026-11-19)

**Producer:** `src/attestation/ledger_adapters/` -- every legacy reader now
ends in a core record before the ledger sees it, converted in memory and
validated by `from_dict` (`2026-11-19-ledger-reads-core-records-design.md`).
No schema change was made. Two findings, both recorded rather than worked
into the schema:

1. **Terminal status is not observable from most legacy artifacts, and a
   spec never ran.** A `results/*.json` file, a W&B summary or a DVC metric
   file says a value exists, not that the run completed; the `generic`
   reader also records a *spec* (a config with no result). `status` has no
   value for either, so the importer writes `"complete"` and carries the
   ledger's own word (`recorded` / `spec`) in `ext.ledger-import.status` --
   a workaround by this section's rule, made acceptable for now only because
   these records are never written to disk. **Decided at the freeze**
   (5.E below): `status` is nullable, and the importer writes `null`.
2. **Reader caveats are carried in `ext`, not a core field.** Each importer
   states what its format cannot say (final value, not curve) in
   `ext.ledger-import.caveats`, and per-run problems (a DVC output whose md5
   no longer matches `dvc.lock`) in `run_caveats`. *Not a core field:* only
   the importer has needed it, a native producer that knows its own limits
   can already say so in its own namespace, and the ledger reads the one
   namespace its importer writes. Revisit if a second producer wants the
   ledger to surface its caveats.

The MLflow adapter ports the old metric-file reader as-is rather than reusing
`examples/mlflow-records/export_records.py`: the exporter reads through
`MlflowClient`, which would make mlflow a runtime dependency of the ledger,
and adopting its richer mapping (curves, datasets, code) is roadmap 12.

### 5. The last findings, decided at the freeze (roadmap 6, 2026-11-19)

Open against `f809522` (smeltery's `docs/core-record.md`, "Open findings"
and "Job-backend schema findings") and from the ledger (4.1 above). Each was
decided before the freeze, because after it every answer costs a version.

**A. `dependencies[].source` for a direct-URL archive or a non-git VCS --
schema change.** `pip install https://host/x-1.0-py3-none-any.whl` leaves a
PEP 610 `direct_url.json` with `archive_info`: not an index (`index:<url>`
would claim one), not `file` (a local path), not `git+`. smeltery wrote
`null`, and the URL was then recorded nowhere. Two additions to the closed
vocabulary, not one:
- `url:<url>`, as proposed, for a direct-URL archive. Credentials refused as
  for `index:`; also refused: `url:file:` (a local file is `file`, whose path
  names a machine) and **any query string**, because a signed archive URL
  (S3, GCS, a CI artifact link) carries its credential in the query, which
  the userinfo check cannot see. A `#sha256=` fragment is kept: it is pip's
  own hash of the archive.
- `hg+<url>`, `svn+<url>`, `bzr+<url>` beside `git+<url>`, rather than
  `url:hg+https://...` as the proposal suggested: PEP 610's `vcs_info.vcs` is
  exactly `git|hg|svn|bzr`, `<vcs>+<url>` is pip's own requirement spelling,
  and a reader filtering on "installed from version control" filters on the
  part before `+` -- folding VCS into `url:` would make a VCS install
  indistinguishable from an archive. smeltery's `record.package_identity`
  labels every VCS install git today; it now has a kind for each.
`SOURCE_KINDS` and the JSON Schema pattern change together. No record written
before is refused (the vocabulary only widens).

**B. A `dependencies` list cannot say it is partial -- spec text, no
field.** Chosen: a list is *observed, not exhaustive* -- every entry is a
dependency the run did run on, and a name's absence says nothing. Rejected:
`dependencies_complete: bool`. Completeness is unattainable for any real run
(transitive packages, native libraries under numpy, the interpreter's own
modules), so `true` would be false whenever written, unverifiable by a
reader, and `false` would be the constant every honest producer writes -- a
field whose only honest value carries no information. The question a reader
actually asks is "do I know which X this run used?", which a lower bound
answers for every listed X and correctly leaves unanswered for the rest. And
a required key would have broken every record at the freeze. What this
changes: `null` vs `[]` keeps its meaning (no observation vs observed and
nothing to name), and the MLflow argument for `null` still holds without
exhaustiveness -- the exporter observed nothing, so it has no entry to make
(an entry is an observation, which `[mlflow (unknown)]` was not). smeltery's
converted records (ferric, Python, tier-recorded versions; core imports it
knows of as `unknown`; optional imports left out) are correct as written.

**C. Job backend vs record -- interface text and `state.spec_conflicts`.**
1. *Who owns `run_id`.* The submitter: `JobSpec.run_id` names the run in the
   handle, `state.json` and `run.json`, whoever writes the record; `settle`
   fails a job whose record names another. Rejected: letting the producer's
   own id win (smeltery's input digest) -- a run directory would then carry
   two names, and a failed attempt (backend-written, the job's id) could not
   be grouped with its successful re-execution (producer-written, the
   digest). A producer that wants its own identity as the run_id makes it
   the default at submit (smeltery can build the plan at submit and default
   `run_id` to `input_digest[:12]`, which keeps re-executions sharing a
   run_id); otherwise it goes in `ext.<producer>`. Rejected too: carrying
   both ids as core fields -- the job's id is already in `state.json`, which
   is kept.
2. *What `config` means for a run that never finished.* The same as for one
   that did: `JobSpec.config`, what the run was given, in every terminal
   state. It is the only configuration that exists in every outcome, so a
   reader comparing `config` across a family compares like with like.
   smeltery's `inputs` (derived from the config once the plan is built) go in
   `ext.smeltery`; `input_digest` already does. Outside jobs the general rule
   is unchanged (`config` = the run's configuration, verbatim); a producer
   should use the same form with and without a job.
3. *`JobSpec.ext` into a producer-written record.* The contract carries it:
   every spec `ext` namespace appears unchanged in the job's record, and a
   backend that cannot hand it (and `family`/`arm`) to its producer refuses
   the spec at `submit` -- which smeltery's `check_spec` already does.
   `JobSpec.ext` stays: dropping it is a breaking API change, and a backend
   wrapping a producer it controls can honour it. How a backend hands the
   spec over is its own business (argv, environment, its job store); no file
   is added to `attestation/`.
4. *(smeltery's fourth: Temporal does not push settle.)* Interface text:
   `state.json` may lag, `run.json` is authoritative, every `status`/
   `cancel`/`result` settles a job found to have ended, and `settle` takes
   the scheduler's `started_at`/`ended_at` when it has them.

**D. `state.json`, its lock, `settle` and `terminal_record` move into
`attestation_schema.state`** (stdlib only, Python 3.11), with
`schemas/state-1.schema.json` and `tests/test_state.py`. Ported from
`smeltery.jobs` at smeltery `2e21918`; the format is unchanged, evidenced by
two `state.json` files written by smeltery's own code (`tests/data/
smeltery-*.state.json`) that read and re-serialize byte for byte. The lock is
the same `flock` on the `attestation/` directory, so a backend on smeltery's
copy and one on this package exclude each other during a migration (the test
holds the lock in one process and checks a second `flock` is refused).
Differences from smeltery's copy, each deliberate:
- `read_state` returns a frozen `StateFile` (with `.status`, `.started_at`,
  `.handle()`), not a dict, as `read` returns a `RunRecord`; it also checks
  what smeltery's writer guaranteed but its reader did not (times never go
  backwards, the last entry is `state` at `updated_at`) and parses strictly
  (duplicate keys refused) -- no file smeltery's code writes is refused.
- `transition` refuses a terminal state: ending a job is `settle`'s, which
  also guarantees the record.
- `settle` takes `producer`, `code` (default `UNKNOWN_CODE`) and `ext` for the
  terminal record instead of smeltery's `observer` flag (the schema cannot
  know a producer's code); `started` is `started_at`, and `ended_at` is new
  (C.4); the terminal entry and the record's `finished_at` share one instant.
  It also checks the record against the spec (C) and fails a record with
  `status: null`. A manifest refused for a symlink costs the manifest (`data:
  []`), and the reason is in `message` -- smeltery wrote it to
  `ext.smeltery.job.manifest_refused`, which the schema cannot own.
- The job's message stays in `state.json`, not in a schema-owned `ext`
  namespace: `ext.attestation.*` is refused by every v1 reader, and the kept
  `state.json` already holds it.
- **Windows:** no `flock`, so `state_lock` raises `StateLockUnavailableError`
  and nothing is written. Rejected: an `msvcrt.locking` lock file -- it would
  add a file to `attestation/`, and nothing here can run it on Windows; a
  lock nobody has run is worse than a refusal. Adding one later changes no
  format.

**E. Status for artifacts that cannot show how their run ended -- schema
change: `status` is nullable.** Rejected: an `unknown` string. Every other
"the producer could not say" in this record is `null` (`dirty`,
`direction`, `seeds`, `dependencies`, a data `sha256`), and `unknown` already
means something else here (a provenance rung: "identity not established").
Rejected: keeping such artifacts out of the core record -- that would drop
every W&B, DVC, Hydra, MLflow-file and results-file run from the ledger,
which reads only core records. **The ledger's `spec`** (a config with no
results) maps to `null` too, not to its own exclusion: "it never ran" is an
inference from the absence of results, which is the inference this schema
refuses everywhere; what the importer observed is a configured run whose
outcome it cannot see and which reported no values (`metrics: []`). Decision
5 still holds for producers: a record is written once, at the end, and a
producer that ran the experiment or a job backend always knows the outcome
and never writes `null` (`settle` fails a job whose record does). The ledger
importer writes `null` for every reader except Sacred's, which keeps only
runs Sacred marked `COMPLETED` and therefore writes `complete`; the ledger's
word (`recorded`/`spec`) stays in `ext.ledger-import.status` and its rows are
unchanged. A native record with `status: null` is stored as `unknown` with a
run caveat. The widening refuses no record written before.

## Shared helpers (0.2.0)

Python API, no format change (`test_frozen` untouched):
`attestation_schema.files._atomic_write` is every writer's one durable path
(fsync, then `os.replace` or an exclusive `os.link`, then the directory
fsynced) -- the two manifest writers did not fsync before, so a crash could
keep `run.json` and lose the `digests.json` it binds -- and the two manifest
writers share one body. `identity.package_identity` is the one dependency
rule for every producer, `strip_credentials` strips exactly what this reader
refuses (`record._carries_credentials`), and `sha256_file`/`git_state` are
the shared probes. Each decision:
`2026-11-19-dependency-identity-design.md`.

## v1 freeze (roadmap 6, 2026-11-19)

The criteria (Open decision 1) were met: MLflow (`examples/mlflow-records`),
smeltery (golden record `tests/data/core_record/benzoic-rhf.run.json`, pinned
at `f809522`, plus its two job backends) and the ferric build identity (a
`verified` dependency in that record, from `ferric.__build__`) have each
written real records the reader accepts, and the ledger imports them
(`tests/test_core_adapter.py`). The last open findings were decided (Producer
findings 5) and `SCHEMA_STATUS` became `"frozen"`.

### What each producer changed, and when

All on 2026-11-19; the commit is the one that carried the change.

| Producer | Commit | What it changed in the schema |
|---|---|---|
| (initial draft) | `d11b928` | The record, writer, reader, layout, job interface. |
| MLflow (roadmap 3) | `040c451` | `history` (curves; later a sidecar); non-finite metric values spelled `"NaN"`/`"Infinity"`/`"-Infinity"`; `data[].sha256` nullable; one spelling per timestamp. Also needed `code.provenance` (with smeltery 1). |
| smeltery round 1 (roadmap 4) | `040c451`, `314e4ff` | `code.provenance`/`evidence`/`version`; `metrics[].unit` and `labels` (and history labels); seed lists; `dependencies[].source`; `run_id` not unique (text); `steps/`/`logs/` optional (text); JCS number note (text); Python floor 3.11 (`314e4ff`). Rejected `config_digest`; deferred per-scope dependency identities. |
| smeltery round 2 | `f809522` | Mixed code is `unknown` with no `git_sha`, and `unknown` with any sha is refused; timestamps always six fraction digits; `to_dict` is the wire form and `spell_value` public; `source` a closed vocabulary. (Same commit, user-directed: `history` became a digest-bound sidecar reference; `dependencies` nullable.) |
| ledger (roadmap 11) | `5325344` | No schema change; filed finding 4.1 (status not observable) and 4.2 (caveats in `ext`, kept). |
| smeltery job backends + freeze (roadmap 6) | this commit | `source` gains `url:<url>` and `hg+`/`svn+`/`bzr+` (5.A); a dependency list is a lower bound (5.B, text); the job/record contract (5.C); `attestation.state/1` reader, writer, lock, `settle` and `terminal_record` in `attestation_schema.state` with `state-1.schema.json` (5.D); `status` nullable, the ledger importer writes `null` (5.E). |

### Compatibility policy (from now on)

- **Readers dispatch on `schema`.** `from_dict` splits `<kind>/<version>`
  and calls `READERS[schema]`; an unknown version is an
  `UnsupportedSchemaError` naming the versions the reader knows. The same
  holds for `attestation.history/N`, `attestation.state/N` and
  `attestation.digests/N`.
- **Nothing about a frozen version changes.** Not the fields, not what the
  reader accepts or refuses, not the bytes `dumps` writes for a given record
  (a record's digest is its identity). The JSON Schema files are pinned by
  sha256 and so is the writer's output for a reference record and state
  (`tests/test_frozen.py`, whose failure message states this policy). A
  digest there is never updated to make the test pass.
- **Additive optional fields still need a version bump.** A v1 reader refuses
  unknown keys (that is how absent-vs-null stays impossible), so a record
  carrying a new key -- however optional -- is unreadable by every deployed
  v1 reader. "Additive" is not "compatible" here. Relaxing a rule is also a
  change (a v2 record a v1 writer could not have written must say v2).
  `ext.<namespace>` is the only place new data goes without a version.
- **A v2 reader still reads v1.** `READERS` keeps `_read_v1`, which gains an
  upgrade step returning the v2 model with v1 meaning preserved: a field v2
  adds is `null` ("a v1 producer could not say"), never a guessed default.
  The record's `schema` stays the version it was written in. Every frozen
  version stays readable for as long as the package exists; dropping one is
  itself a breaking package release.
- **Writers write only the newest version**, so a package release that
  writes v2 is a breaking release for producers (major version bump of the
  package once it is past 0.x).
- **The package version is not the format version.** `attestation-schema`
  goes to **0.1.0** now, not 1.0.0: the formats are frozen and that promise
  is carried by the `schema` strings and the digest pins, but the Python API
  added here (`attestation_schema.state`) has no consumer yet, and a 1.0
  promise about an API nobody has called would be a claim measured on a
  fixture. 1.0.0 follows when smeltery runs its job backends on it. Until
  then a 0.x minor release may change Python API (never a frozen format).

## Open decisions (for the user)

1. **Freeze criteria.** Met and applied 2026-11-19 ("v1 freeze"). The
   identifier stayed `attestation.run/1` throughout, so drafts written before
   a draft change could stop reading; every committed record was regenerated
   at each change, and smeltery re-pins to the frozen commit.
2. **Publishing the schema package -- decided: publish to PyPI.** Release
   tags are `schema-v<X.Y.Z>` (attestation keeps `v<X.Y.Z>`), both handled by
   `.github/workflows/release.yml` through trusted publishing; the user
   registered the pending publisher (repo mgoldey/attestation, workflow
   `release.yml`, environment `pypi`). Since roadmap 11 (2026-11-19)
   `attestation-schema` is a runtime dependency of attestation and its wheel
   says `Requires-Dist: attestation-schema>=0.1.0` (`>=0.2.0` since the
   shared helpers), so the schema is released
   first and an attestation release fails fast while its floor is not on PyPI
   (`2026-11-19-ledger-reads-core-records-design.md`, "Dependency"). Nothing
   has been published yet.
3. **Required-and-nullable vs optional keys.** Chosen: every key required.
   It makes hand-written records verbose; it makes absent-vs-null impossible.
4. **`producer`** was added beyond the roadmap's field list: without it a bad
   record cannot be traced to the code that wrote it.
5. **Terminal-only `status`.** A record is never written for a run in flight.
   Amended at the freeze (5.E): `status` may be `null` when the producer
   could not observe the outcome (importers); never in flight, and never from
   a producer that ran the experiment or a job backend.

## Testing

`packages/attestation-schema/tests/test_attestation_schema.py` (on pytest's
`testpaths`): round trip; the history sidecar (round trip in logging order with
repeated steps; interleaved streaming writes finalized atomically; a failed
producer leaves no sidecar; written before the record, never after; `write`
refuses a missing or different sidecar; digest mismatch and missing file on
read; every malformed or truncated line kind → `HistoryFormatError` naming
the line, with an unmutated control; deterministic bytes, streaming = batch;
100k points leave `run.json` under 2 KB; `read` never opens the sidecar; the
manifest does not list it; the line JSON Schema agrees with the writer);
`dependencies` null vs `[]`; `unknown` claiming no sha and a mixed-code
record; timestamps sorting as text in time order; `to_dict` being the wire
form and `spell_value` public; the `source` vocabulary; non-finite metric
values spelled and read back; data named without a digest; timestamp
spelling and `format_timestamp`; an unknown `ext` namespace preserved byte for byte;
**every** required field, top-level and nested, removed in turn → the specific
error naming its path; unknown fields refused; strict parse (duplicate key,
`NaN`, huge int, `true` as a metric value); version dispatch (unknown version
refused with the known list, a registered reader is used); writer determinism
(same record twice, key order of the input irrelevant, JCS parity with
`attestation.run_record.canonical`); the JSON Schema file agrees with the
reader on every valid and invalid case; layout and manifest; job state
machine; `null` status; the `url:` and VCS `source` kinds and their refusals.
`tests/test_state.py`: `attestation.state/1` -- two `state.json` files written
by smeltery's own code read and re-serialize byte for byte; every malformed
case refused with its path, and the JSON Schema file agrees where it can
express the rule; illegal transitions write nothing; clock clamping; `settle`
exactly once, an existing record deciding, a record that is not the job's
(run_id, family, config, ext, null status) or unreadable ending the job
failed, the terminal record's contents, a symlink costing only the manifest;
the lock is `flock` on `attestation/` (a second process's `flock` is refused
while it is held); eight processes settling at once leave one terminal entry
and a record that agrees; no `flock` (Windows) refuses writes, reads work.
`tests/test_frozen.py`: the frozen JSON Schema files and the writer's bytes
for a reference record and state, pinned by sha256. The wheel builds with `uv build --package attestation-schema` in CI's
wheel smoke job, which installs it into a clean venv and reads a record.
`tests/test_boundaries.py`: the import rules, with detector self-tests.
`tests/test_mlflow_records.py`: the MLflow producer (Producer findings, 1) —
the committed records validate, round-trip byte for byte, bind their
history sidecars by digest and preserve every point of the raw MLflow metric
files without mlflow installed; with mlflow,
re-export is byte-identical to the committed records and a fresh real store
maps without workarounds.
