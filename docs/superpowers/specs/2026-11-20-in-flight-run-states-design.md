# In-flight run states: what is happening now, read from `state.json`

**Date:** 2026-11-20
**Status:** implemented 2026-11-20 on `next/inflight` (roadmap item 22).
**Depends on:** `2026-11-19-core-run-record-design.md` (the run-directory
layout, `attestation.state/1`, "The job backend": `settle`,
`terminal_record`, "state.json may lag; run.json is authoritative"),
`2026-11-19-ledger-reads-core-records-design.md` (the core adapter, which
skips a directory in flight).
**Not this:** a job backend, a scheduler client, a heartbeat, a liveness
verdict, killing or settling anything, a table, a migration, in-flight runs
in `runs.compare` or any ranking.

## Problem

`attest runs` answers "which arm won" about runs that ended. It cannot answer
"what is running now": the core adapter skips every run directory that holds
`attestation/state.json` but no `attestation/run.json` (ledger-reads-core-
records, "In flight is skipped"), so a sweep of twenty jobs, twelve queued and
eight running, reads as an empty workspace until the first one finishes. The
information is already on disk. Every job backend writes `state.json` through
`attestation_schema.state` (`init_state` at submit, `transition` to running,
`settle` at the end), the layout fixes where it lives, and smeltery -- the real
producer -- writes it from both its local queue and its workflow backend. What
is missing is a reader.

The constraint is the roadmap's: attestation scans the artifact layout and
never learns which system scheduled a job. Everything comes from the files.

## Decision 1: read at query time, store nothing

`attestation.inflight.status(root, project=None, *, now, stale_after)` walks
the workspace the way `ledger.scan` does and returns rows; nothing is written,
there is no table and no migration.

Why not a table:

- **The data is a snapshot of something moving.** A stored in-flight row is
  wrong the moment its job transitions, and the ledger refreshes only on
  `runs scan`. A status that is as stale as the last scan answers "what was
  happening when I last scanned", which is not the question.
- **Ranking must never see it.** `runs`/`run_metrics` are what `compare`,
  replicate pooling, prereg and claims read. An in-flight run has no final
  metrics; putting it there means every one of those readers must learn to
  filter it out, and the one that forgets ranks a half-finished run. Keeping
  it out of the database makes "ranking never sees an in-flight run" true by
  construction rather than by a `WHERE` clause.
- **It is cheap.** Reading a `state.json` is one small file per run directory;
  the walk is the core adapter's own (`MAX_DEPTH` levels, runs do not nest,
  hidden directories skipped, symlinks not followed).

The walk is shared, not copied: `ledger_adapters/core.py` gained
`all_run_dirs(root)` (every run directory: record, state, or both) and
`run_dirs` became the filter `all_run_dirs` + "has a record". One rule for
both readers, so a run the status view lists is the run the ledger will
import once it ends. Projects are `ledger.scan`'s (every non-hidden
subdirectory of the root, or `--project`), so a row's `project` is the name
`runs list` will use for the same run.

## Decision 2: the schema's vocabulary, and `run.json` wins

States are `attestation_schema.jobs.JobState` values verbatim: `submitted`,
`running`, `complete`, `failed`, `cancelled`. Nothing here invents a state.
What the view adds is a **kind** -- which relation a directory's two files are
in -- and that follows the core spec's authority rule:

| `state.json` | `run.json` | kind | listed? |
|---|---|---|---|
| submitted / running | absent | `in_flight` | yes |
| terminal | present | (finished) | no -- it is the ledger's |
| absent | present | (finished; an exporter's) | no |
| terminal | absent | `unrecorded` | yes |
| submitted / running | present | `lagging` | yes, with the record's own status |
| unreadable | absent | `unreadable` | yes, with the reader's error |
| unreadable | present | (finished) | no -- `runs scan` quarantines it |

- **`unrecorded`.** `settle` writes the record before the state, so a terminal
  state with no record is a writer that died between the two, or a record
  removed afterwards. Conformance already calls this a violation (`terminal`
  rule); this view says it out loud even when no scan ran. It is the
  `terminal_record` case: the backend's next `settle` cannot fix it (the job
  is already terminal), so a person has to look.
- **`lagging`.** "state.json may lag; run.json is authoritative" (core spec):
  a backend that settles on observation leaves `running` beside a finished
  record until someone asks it. The run is NOT in flight -- it is in the
  ledger already, with the record's status -- and the row says both, so the
  disagreement is visible rather than resolved by whichever file was read.
- **`unreadable`** never crashes the call. `read_state` refuses rather than
  guesses (`StateFormatError`), and the row carries that sentence. Beside a
  record it is the scan's business: conformance checks `state.json` and the
  directory is quarantined there, loudly; listing it twice would be noise.

Row fields: `project`, `run_dir` (relative to the root), `kind`, `state`,
`run_id`, `backend`, `job_id`, `submitted_at`, `started_at`, `updated_at`,
`age_s` (since the last transition), `idle_s`, `stale`, `record_status`,
`message`. `family` is absent because `attestation.state/1` does not carry
it: the spec is the record's, and guessing a family from a path or a
producer's config file is exactly what the ledger refuses to do. `backend`
and `job_id` are strings from the file, carried through and never
interpreted: that is how the view names a job without knowing what a
backend is.

## Decision 3: `stale` is evidence, never a verdict

`state.json` has no heartbeat: it changes only on a transition. Flagging a
`running` job stale on `updated_at` alone would flag every job that runs
longer than the threshold. So `idle_s` is the time since the later of the
last transition and the newest mtime among the files the layout says a run
writes *while it runs*: `attestation/history.jsonl` (streamed), the entries
of `logs/` and of `steps/` (a new file in a step directory changes the
directory's mtime). The probe is shallow on purpose -- an output tree can
hold millions of files and a status call must not walk them.

`stale` = `running`, in flight, and `idle_s >= stale_after` (default
`STALE_AFTER_S`, two hours; `--stale-hours` / `stale_hours`). It is a
default, not a measurement. A `submitted` job is never stale: queueing time
is legitimate and the history is the only record of it.

What `stale` does NOT say, and the CLI help, the tool description and the
tool's message all repeat: that the job died. A step can compute for hours
without writing; a job can be alive on a machine whose clock disagrees with
this one's (mtimes are local, `updated_at` may be a scheduler's). Nothing in
this view writes, settles, or cancels -- a backend that knows the job is
gone settles it with `terminal_record`, which is its job and not the
reader's.

## Decision 4: one CLI command, one tool

- **`attest runs status [--root DIR] [--project P] [--stale-hours H] [--json]`.**
  Reads files only (no ledger database). Prints one group per kind (in
  flight first, stale before running before submitted, most recent change
  first), each row `state  age  idle  run_dir  run_id  backend:job_id` and the
  message under it, then one summary line (`4 in flight (3 running, 1
  submitted; 1 stale); 1 unrecorded; 1 lagging; 1 unreadable`). Exits 0
  whatever it finds: a status is a report, not a check (contrast `runs scan`,
  strict by default, which refuses records).
- **`runs.status(root, project, stale_hours)`**, a new tool rather than a
  flag on `runs.list`: `runs.list` reads the database and needs no root;
  this reads the disk and does. Folding them would make one tool's
  arguments mean different things by mode, the shape `feed.persona_status`
  only justifies when arity distinguishes breadth from depth. In its own
  module (`mcp/run_status.py`, as `run_record.py`) since it shares nothing
  with provenance.py but the namespace and `NO_ROOT`. The surface goes from
  54 to 55 tools, `runs.*` from 10 to 11, the provenance surface from 13 to
  14, and every pin of those counts moves with it.

Response size: rows capped at `MAX_STATUS_ROWS` (8) and every string field
at its own cap (`run_dir` keeps its tail, the part that names the run),
because every string came from a file someone else wrote and
`attestation.state/1` bounds none of them. MEASURED: 12 uncapped worst-case
rows emitted 8778 chars against the 7000 ceiling; 8 rows at every cap emit
6275 as sent (indent=2), pinned in `tests/test_inflight.py`. The three
timestamps are dropped from the tool's rows (`age_s`/`idle_s` carry the
same fact in the unit a reader acts on); the CLI's `--json` keeps them.
`counts` is always complete and the message says how many rows were not
shown.

## Tests

`tests/test_inflight.py` builds every fixture with `attestation_schema.
state`'s own writers -- `init_state`, `transition`, `settle` -- the functions
smeltery's backends call, so the reader is tested against the format
producers write:

- submitted; running; running and idle for 5h (stale); running for 5h with a
  log written 5 minutes ago (not stale: activity counts); settled complete
  with the run's own record (not listed); settled failed, where `settle`
  wrote `terminal_record` (not listed); terminal state with its record
  removed (`unrecorded`); a record with an unsettled `running` state
  (`lagging`, record status `complete`); a corrupt `state.json`
  (`unreadable`, no crash); an exporter's record with no `state.json` (not
  listed).
- counts, order and the summary line; the threshold as an argument; `status`
  writes nothing (every mtime unchanged).
- **Ranking is unaffected:** `ledger.scan` over the same workspace imports
  exactly the four recorded runs, quarantines nothing, and `ledger.compare`
  has no in-flight run among its arms.
- CLI groups, `--json`, `--stale-hours`, the missing-root sentence on stderr;
  the tool's envelope, its refusal shape, and its worst-case size.

## What this does not do

- No backend is named or imported; `test_tracker_names`'s rule extends in
  spirit: nothing under `src/` knows what scheduled a job.
- No ETA, no progress. `history.jsonl`'s points could say how far a run got,
  but reading them is `read_history`'s job on a finished record, and an
  in-flight sidecar has no digest to check yet.
- No `runs.ask` route. "What is running" is one tool call; teaching the
  router a new destination is a measured change (routing is pinned at 13/15)
  and was not measured here.
