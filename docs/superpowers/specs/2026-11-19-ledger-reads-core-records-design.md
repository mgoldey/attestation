# The ledger reads core records; trackers become adapters

**Date:** 2026-11-19
**Status:** implemented 2026-11-19 (roadmap item 11). The runtime dependency
on `attestation-schema` (below, "Dependency") is a decision for the user.
**Depends on:** `2026-11-19-core-run-record-design.md` (the record, the run
directory), `2026-08-22-tracker-adapters-design.md` (the readers being
ported), `2026-08-12-run-ledger-design.md` (the tables they feed).
**Not this:** roadmap 12 (a richer MLflow mapping: curves, code identity,
datasets), 16 (W&B and Hydra as separate, richer adapters) or 17 (seed rules).
Every reader is ported as-is; no mapping gains a field it did not have.

## Problem

`ledger.py` and the claim checker read the SQLite `runs`/`run_metrics`
tables, and those tables were filled from a ledger-private dataclass that
five tracker readers built directly (`_wandb_runs`, `_mlflow_runs`,
`_sacred_runs`, `_dvc_runs`, `_hydra_runs` in `ledger_adapters/generic.py`).
The tracker names leaked out of the adapters: `ledger.ADAPTER_CAVEATS` keyed
caveat text by tracker, `ledger._dvc_hash_caveats` recognised one tracker's
note by substring, and `db.py`'s comments listed trackers. And a directory
that already holds a core record -- smeltery's runs, the MLflow exporter's
`records/` -- was invisible to the ledger, although that record is the
contract the whole stack is converging on.

## Decision: one path into the ledger

```
 project dir ──► adapter.detect ─► adapter.scan ─► adapter.to_core_record ──► attestation_schema.RunRecord
                                                                                        │ (validated)
                                                         ledger._row_from_core ◄─────────┘
                                                                  │
                                                     runs / run_metrics (SQLite) ◄── claims.py, compare, detail
```

The ledger turns **only** `attestation_schema.RunRecord` into rows. Every
importer -- including the convention-based `generic` reader -- ends in a core
record, and the claim checker keeps reading the tables, which now hold
nothing a core record did not say. `kg.py` was checked and touches no run
data (it builds from `item_tags`); it is unchanged.

### The protocol

```python
class Adapter(Protocol):
    name: str
    def detect(self, path: Path) -> bool: ...
    def scan(self, path: Path, taken: set[str] | None = None) -> Iterable[Any]: ...
    def to_core_record(self, found: Any) -> attestation_schema.RunRecord: ...
```

- `detect(path)` is cheap and answers "does this project hold anything this
  adapter reads" (a `wandb/` directory, a `dvc.yaml`, a run directory).
- `scan(path, taken)` yields what the adapter found, in its own shape
  (a `Found` for every importer of a legacy artifact; the record itself for
  the native adapter). **`taken` is the one addition to the shape the roadmap
  named**, and it is not new behaviour: the readers always shared a `seen`
  set of run names, which is how a `results/` measurement outranks the
  `configs/` spec of the same name, how a W&B run colliding with a results
  run is recorded once, and how Hydra qualifies a second sweep of the same
  job with its time directory instead of dropping it. Without it, those
  three rules -- each pinned by a test -- move out of the readers into a
  pipeline that would have to know them per tracker. Optional: `None` means
  a fresh set.
- `to_core_record(found)` returns a validated record (`attestation_schema.
  from_dict`, the reader's own rules: a record the reader would refuse is
  never imported). An adapter whose artifacts have curves would also write
  the history sidecar; none of the ported readers keeps a curve (that was
  their documented limit), so every importer's `history` is `null` today, and
  a native record keeps the reference it was written with.

`ledger_adapters.import_project(root)` runs the registry in a fixed order --
`core`, `generic`, `wandb`, `mlflow`, `sacred`, `dvc`, `hydra`, the old
`discover` order with native records first -- sharing one `taken` set, and
returns `Imported(record, name, source_path, adapter)` envelopes plus the
records it refused. `name` and `source_path` are the ledger's key and where
the run was found: properties of the import, not of the run, so they are not
record fields. `generic.discover(root)` stays, returning the legacy `Found`
list for the tests and the `attest record` pairing check that read it, and is
now built from the same registry, so the two orders cannot drift.

### Native records: the `core` adapter

A directory holding `attestation/run.json` (`attestation_schema.layout.
is_run_dir`) is read with `attestation_schema.read`, no conversion. Rules:

- **In flight is skipped.** A directory with only `state.json` has no
  terminal record; the ledger records runs that ended.
- **A refused record is reported, never coerced.** A `RecordError` lands in
  `scan`'s `diagnostics` for the project (`<path>: <error>`), and the run is
  not imported. Draft records written before a schema change are expected
  to hit this.
- **Name.** `run_id`. `run_id` is not unique (the core spec: re-executions
  share it), so when two records in one project share a `run_id`, or one
  collides with a name an earlier adapter took, **every** copy is named
  `<run_id>@<record_digest[:12]>`. Suffixing only the second would let
  `runs.detail`/a claim naming `run_id` silently resolve to whichever record
  sorted first -- the spec's "never silently picks one".
- **Status** is the record's (`complete`, `failed`, `cancelled`), and a run
  that did not complete carries a run caveat saying so, because `compare`
  ranks every arm in a family and a partial value must not win silently.
- **Metrics.** `labels` fold into `split` (`test,analogue=a2`; split first, so
  `_split_rank`'s prefix match still ranks a `test` value as test); `run_metrics`
  has no labels column and the uniqueness key would otherwise collide. A
  non-finite value (`"NaN"`, a diverged loss) is not stored -- `run_metrics.
  value` is `REAL NOT NULL` and every ranking here assumes a finite number --
  and a run caveat names it, so the omission is visible rather than looking
  like a metric the run never reported. `direction` is not yet used: compare
  reads directions from `metric_direction.toml`, and making a record's own
  direction outrank that table is a ranking change for its own spec.
- `data`, `code`, `seeds`, `dependencies` and `history` are not stored: the
  ledger's tables have no place for them, and adding one is a feature.
- Files inside a run directory are not also read by `generic` (a run
  directory under `results/` would otherwise have its outputs read twice).

### Legacy artifacts: how a `Found` becomes a core record

One converter (`ledger_adapters/trackers.py:found_to_core`), parameterised
per adapter:

| ledger field (was) | core record | why |
|---|---|---|
| `name` | `run_id` | Already unique within the project (`taken`). |
| `family` | `family` | Verbatim. `arm`: `null` -- no reader knows one (roadmap 12/16). |
| `status` `recorded`/`spec` | `status: null` (Sacred: `"complete"`) + `ext.ledger-import.status` | See "Status" below. |
| `started` | `started_at` via `format_timestamp` | Only W&B and Sacred have one; both write naive UTC (`startedAt`, `start_time`), which the adapter declares, so the naive value is read as UTC. Unparseable → `null`. |
| `config` | `config` | Verbatim; `None` ↔ `{}`. A non-finite float (a CSV cell `nan`) becomes its string, the rule `_coerce` already applies to YAML config. |
| `metrics` | `metrics` | `direction: null` (never inferred), `unit: null`, `labels: {}`. Duplicates on `(name, step, split)` keep the first -- what `_replace_project`'s insert loop always did. |
| `notes`, `corpus`, `adapter` | `ext.ledger-import` | The core has no field for them (below). |
| tracker caveats | `ext.ledger-import.caveats` / `.run_caveats` | Moved out of `ledger.py`, below. |
| -- | `producer` `{"attestation <adapter>-import", <attestation version>}` | What made the record. |
| -- | `code` unknown, `data: []`, `seeds`/`dependencies`/`history: null` | The readers never read these; null says "did not say". |

**One extension namespace, `ledger-import`, not `ext.<tracker>`.** The ledger
must find the importer's data without naming the tracker (the point of this
item), and `ext.mlflow` already has an owner with a different shape: the
MLflow exporter writes MLflow's own identifiers there. Two producers writing
one namespace in two shapes would make the namespace unreadable. The
importer is one producer (attestation), so it gets one namespace, whose
`adapter` key says which reader made the record. `attestation`/`attestation.*`
is reserved for the schema, so the name is not under it. A record carrying
`ext.ledger-import` is read the same way whichever adapter delivered it, so
a future cache of imports written to disk (through `attestation_schema.write`)
and read back by the `core` adapter yields identical rows.

**Caveats travel as data.** Each tracker adapter owns its caveat text
(moved verbatim from `ledger.ADAPTER_CAVEATS`) and emits it in every record
as `ext.ledger-import.caveats` (what the reader cannot do: final values, not
curves). The DVC adapter also emits `run_caveats` (what is wrong with this
one run: a metric file no longer matching `dvc.lock`'s md5), which replaces
`ledger._dvc_hash_caveats`'s substring match on `notes`. The ledger stores
both in a new `runs.caveats_json` (migration 015), `detail` shows the reader
caveats exactly as before, and `compare` emits run caveats in run order, then
reader caveats ordered by adapter name and deduplicated -- the order it
emitted before. A run scanned before migration 015 has `caveats_json` NULL
and shows no caveat until it is re-scanned (`runs.scan` replaces every row).

### Status: a schema finding, decided at the v1 freeze

**Decided 2026-11-19 (core spec, Producer findings 5.E):** `status` is
nullable, `null` meaning the producer could not observe how the run ended.
The importer now writes `null` for every reader except Sacred's (which keeps
only runs Sacred marked `COMPLETED`, so it observed completion and writes
`complete`), and for a `spec` too; the ledger's own word stays in
`ext.ledger-import.status` and its rows are unchanged. A native record with
`status: null` is stored as `unknown` with a run caveat. The text below is
the finding as filed.

The core `status` is terminal-only (`complete`/`failed`/`cancelled`, decision
5 of the core spec). An importer of legacy artifacts mostly cannot observe a
terminal state: a `results/*.json` file, a W&B summary or a DVC metric file
says a value exists, not that the run completed (the MLflow reader never
looked at MLflow's `status`; only Sacred's reader filters on `COMPLETED`).
And the `generic` reader records a **spec** -- a config with no result
attached -- which did not run at all, as far as anything on disk says.

The importer writes `status: "complete"` for both and carries the ledger's
own word (`recorded` / `spec`) in `ext.ledger-import.status`, which is what
the ledger stores (the `runs.status` column is unchanged). This is a
workaround by the core spec's own rule ("a field a producer can only fill
with a workaround is a schema bug"), so it is recorded as **Producer finding
4** in the core spec rather than hidden. The records are in memory only and
never written, so nothing on disk carries the claim; the choice for the user
is between a status value meaning "the producer cannot observe the outcome"
and keeping specs out of the core record entirely (which would drop them from
the ledger). Not decided here, because decision 5 is the user's.

### Pipeline: in memory, not a cache

Imports are converted in memory on every scan and never written. Chosen
over a cache or materialized directory because:

- the scan already replaces a project's rows wholesale from what is on disk;
  a cache is a second copy of that truth that can go stale, and needs an
  invalidation rule the readers do not have (W&B's summary is rewritten in
  place by `wandb sync`);
- the obvious place for a materialized record, the run's own `attestation/`
  directory, belongs to the producer (the core spec: "nothing else writes
  here"), and writing into a user's `mlruns/` or `wandb/` breaks the ledger's
  founding rule that it changes nothing about how anything is run;
- the "no producer hand-rolls JSON" rule is about bytes on disk, and none are
  written; the conversion still goes through `attestation_schema.from_dict`,
  the same validator `read` and `write` use, so an import is held to every
  rule a written record is.

The cost: an importer that keeps curves (roadmap 12) needs a sidecar file,
which in-memory conversion cannot give it. That adapter will write a run
directory through `attestation_schema` (`write_history`, `write`) under a
cache it owns and let the `core` adapter read it; the `ledger-import`
namespace already reads identically from there.

### What stays outside the adapters

`ledger.py` keeps everything about *ranking* (metric directions, split
ranks, nested arms, corpus agreement) and knows no tracker. The tracker
caveats, the `Found` dataclass (formerly `ledger.RunRecord`/`ledger.Metric`,
which were the readers' shape, not the ledger's) and the converter move under
`ledger_adapters/`. `tests/test_tracker_names.py` enforces the boundary:
no file under `src/attestation/` (outside `ledger_adapters/`) or the schema
package's `src/` names a tracker -- `wandb`, `W&B`, `mlflow`, `mlruns`,
`sacred`, `dvc`, `hydra`, `multirun`, `tensorboard`, `tensorflow`, matched
case-insensitively with `_` as a word boundary so `sacred_runs` is caught and
`hydrate` is not -- with a self-test proving the detector fires, and an
explicit allowlist that holds only documentation (`skills/**/SKILL.md`,
which teach an agent which trackers the ledger reads). `docs/` and
`examples/` are outside the scanned trees.

## Dependency: attestation now needs attestation-schema at runtime

The ledger imports `attestation_schema` on every scan, so it is a runtime
dependency, and `attestation-schema` is a workspace member that is not on an
index. Options:

1. **Publish first, then depend normally** -- recommended. `attestation-schema`
   goes to PyPI (it is stdlib-only, so this is cheap) and attestation declares
   `attestation-schema>=0.1.0` (`>=0.1.0.dev0` until the v1 freeze). Every install path works, including
   `pip install attestation` from an index.
2. **Vendor a copy into attestation's wheel** (hatch `force-include`). Rejected:
   two distributions would own the `attestation_schema` import name and
   overwrite each other's files, and a copy is exactly the second writer the
   core spec forbids ("the schema package is the only writer"). A symlink is
   the same copy with a worse failure mode.
3. **An optional extra with a fallback reader.** Rejected: the fallback is the
   legacy path this item deletes, so `attest runs scan` would behave
   differently depending on an extra -- the hardest kind of bug to see.

**Implemented now, the least surprising step that keeps CI green:** option 1
without the publish. `attestation-schema` moves from the dev group to
`[project] dependencies` and resolves through the existing `[tool.uv.sources]`
workspace entry, so `uv sync`, `uv run` and the pre-commit gates use the local
package. The attestation wheel's metadata now says `Requires-Dist:
attestation-schema`, so a bare `pip install attestation-*.whl` needs the
schema wheel beside it; CI's wheel smoke job builds both and installs them
together. `uvx --from git+<repo> attest` resolves the requirement through the
same workspace source. **Measured 2026-11-19** (uv, `--no-cache`, from a
`git+file://` URL of this commit): `uvx --from git+... attest --help` exits 0
and `attestation_schema` is installed beside attestation; a clean venv given
only the attestation wheel fails to resolve (`attestation-schema` is on no
index), and given both wheels imports the ledger with all seven adapters. **Until the schema is published, an
attestation wheel uploaded to an index would not install** -- which is why
publishing is the user's decision to make before the next attestation release.

Because the dependency is now real, `attestation.run_record.canonical` becomes
a re-export of `attestation_schema.canonical`, as the core spec said it would,
and the duplicated JCS implementation disappears; the parity test asserts the
two names are one function.

## Behaviour preservation

Every golden path, `test_tracker_adapters.py`, `test_ledger_adapters.py`,
`test_ledger.py` and `test_claims.py` pass unchanged. Deliberate changes:

- **`runs.started` is now the core spelling** (`2026-08-29T01:33:02.010570Z`)
  rather than the tracker's raw text (`2026-08-29T01:33:02.010570`), since it
  now comes from `started_at`. Only W&B and Sacred runs have a start time;
  `runs.detail` shows the new spelling. No CLI output pins it.
- **`runs.caveats_json`** is a new column (migration 015); `runs.detail`
  does not print it (it prints `caveats`, as before).
- **A refused record** (a native record the reader rejects, or a legacy run
  the converter cannot express -- a negative step, an integer config value
  past 2^53) is listed in that project's `diagnostics` instead of being
  imported. No committed fixture hits this.
- **A run directory holding a core record is now a run.** A project with one
  used to scan its loose JSON outputs (if under `results/`) or nothing.

## Testing

- `tests/test_core_adapter.py`: the native adapter -- the committed MLflow
  exporter records scan as runs with MLflow's final values; a `failed` record
  gets a run caveat; a `NaN` metric is skipped with a caveat; labels fold into
  `split`; re-executions sharing a `run_id` are both kept, both suffixed; a
  draft record that no longer validates is a diagnostic, not a crash; an
  in-flight directory is skipped; files inside a run directory are not read
  by `generic`; every legacy adapter's output is a valid core record; the
  protocol's registry order matches `generic.discover`.
- `tests/test_tracker_names.py`: the boundary, with its detector self-test.
- The existing suites, unchanged, are the behaviour-preservation evidence.
