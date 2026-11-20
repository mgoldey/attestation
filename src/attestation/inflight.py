"""What is happening now: runs in flight, read from `state.json` at query time.

The ledger records runs that ENDED -- a run directory enters `runs` only once
its `attestation/run.json` exists. A job between submission and its record
has only `attestation/state.json` (`attestation.state/1`, written by whatever
ran it), and this module reads those. Nothing here is stored: there is no
table and no migration, the files are re-read on every call, and ranking
(`ledger.compare`) never sees an in-flight run because it reads only the
`runs` table. Which backend ran a job is a string in the file (`backend`,
`job_id`), carried through verbatim and never interpreted.

Rules (spec: docs/superpowers/specs/2026-11-20-in-flight-run-states-design.md):

- **States are the schema's** (`attestation_schema.jobs.JobState`): submitted,
  running, complete, failed, cancelled. Nothing here invents a state.
- **In flight** = a non-terminal `state.json` and no `run.json`.
- **`run.json` is authoritative.** A terminal state with a record is the
  ledger's, and not listed. A NON-terminal state beside a record is a backend
  that has not settled yet (state.json may lag): listed as `lagging`, with the
  record's own status, never as in flight.
- **A terminal state with no record** is listed as `unrecorded`: `settle`
  writes the record before the state, so this is a writer that died between
  the two, or a record removed afterwards.
- **`stale` is evidence, never a verdict.** `state.json` has no heartbeat: it
  changes only on a transition. So a `running` job is flagged stale only when
  neither its last transition nor any file the layout says is written during
  a run (`attestation/history.jsonl`, `logs/`, `steps/`) has changed for
  `stale_after` seconds. The job may still be alive (a step that logs nothing
  for hours); nothing here says it is dead, and nothing is written.
- **An unreadable `state.json` is reported, never fatal**, unless a record
  sits beside it: then the run ended and `runs scan` quarantines the
  directory on its own (conformance checks `state.json`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import attestation_schema as schema
from attestation_schema import layout
from attestation_schema.jobs import TERMINAL, JobState
from attestation_schema.state import StateFile, read_state

from attestation.ledger_adapters.core import all_run_dirs

# Two hours without a transition or a write to history.jsonl/logs/steps.
# A default, not a measurement: a step that writes nothing for longer is
# possible, which is why the flag is evidence and the threshold an argument.
STALE_AFTER_S = 2 * 60 * 60

# Kinds, in the order a reader wants them: what is moving, then what is not.
IN_FLIGHT = "in_flight"
UNRECORDED = "unrecorded"
LAGGING = "lagging"
UNREADABLE = "unreadable"
KIND_ORDER = (IN_FLIGHT, UNRECORDED, LAGGING, UNREADABLE)


@dataclass(frozen=True)
class Status:
    """One run directory that is not simply a finished, recorded run."""

    project: str
    run_dir: str  # relative to the workspace root
    kind: str  # one of KIND_ORDER
    state: str | None  # JobState value; None when state.json is unreadable
    run_id: str | None
    backend: str | None  # verbatim from state.json; never interpreted
    job_id: str | None
    submitted_at: str | None
    started_at: str | None
    updated_at: str | None
    age_s: float | None  # since the last transition
    idle_s: float | None  # since the last transition OR write to a run file
    stale: bool
    record_status: str | None  # LAGGING: the record's status
    message: str | None  # state.json's message, or why it is unreadable

    def to_dict(self) -> dict:
        """JSON-ready, as `runs.status` returns it."""
        return asdict(self)


def _instant(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _activity_paths(run_dir: Path) -> Iterable[Path]:
    """Files the layout says a run writes WHILE it runs. Shallow on purpose:
    an output tree can hold millions of files and a status call must not walk
    them. A new file in a step directory changes that directory's mtime."""
    yield run_dir / layout.HISTORY_JSONL
    for sub in (layout.LOGS_DIR, layout.STEPS_DIR):
        d = run_dir / sub
        try:
            yield from d.iterdir()
        except OSError:
            continue


def last_activity(run_dir: Path) -> datetime | None:
    """The newest mtime among `_activity_paths`, or None if there are none."""
    newest: float | None = None
    for p in _activity_paths(run_dir):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        newest = m if newest is None or m > newest else newest
    return None if newest is None else datetime.fromtimestamp(newest, UTC)


def _record_status(run_dir: Path) -> str:
    """The record's own status, for a lagging state; never raises."""
    try:
        return schema.read(run_dir).status or "null"
    except (schema.RecordError, OSError, ValueError):
        return "unreadable"


def _kind(st: StateFile, has_record: bool) -> str | None:
    terminal = st.state in TERMINAL
    if has_record:
        return None if terminal else LAGGING
    return UNRECORDED if terminal else IN_FLIGHT


def _row(
    project: str,
    rel: str,
    kind: str,
    st: StateFile,
    now: datetime,
    run_dir: Path,
    *,
    stale_after: float,
    record_status: str | None,
) -> Status:
    updated = _instant(st.updated_at)
    age = (now - updated).total_seconds()
    idle = age
    if kind == IN_FLIGHT:
        seen = last_activity(run_dir)
        if seen is not None and seen > updated:
            idle = (now - seen).total_seconds()
    stale = kind == IN_FLIGHT and st.state is JobState.RUNNING and idle >= stale_after
    return Status(
        project=project,
        run_dir=rel,
        kind=kind,
        state=st.state.value,
        run_id=st.run_id,
        backend=st.backend,
        job_id=st.job_id,
        submitted_at=st.history[0].at,
        started_at=st.started_at,
        updated_at=st.updated_at,
        age_s=round(age, 1),
        idle_s=round(idle, 1),
        stale=stale,
        record_status=record_status,
        message=st.message,
    )


def _unreadable(project: str, rel: str, exc: Exception) -> Status:
    return Status(
        project=project,
        run_dir=rel,
        kind=UNREADABLE,
        state=None,
        run_id=None,
        backend=None,
        job_id=None,
        submitted_at=None,
        started_at=None,
        updated_at=None,
        age_s=None,
        idle_s=None,
        stale=False,
        record_status=None,
        message=f"{layout.STATE_JSON} unreadable: {exc}",
    )


def read_one(
    run_dir: Path, project: str, rel: str, now: datetime, stale_after: float
) -> Status | None:
    """The status of one run directory, or None when it is simply finished
    (a terminal state with its record, or a record with no state.json)."""
    has_record = schema.record_path(run_dir).is_file()
    try:
        st = read_state(run_dir)
    except (schema.RecordError, OSError, ValueError) as exc:  # StateFormatError is a RecordError
        return None if has_record else _unreadable(project, rel, exc)
    if st is None:
        return None
    kind = _kind(st, has_record)
    if kind is None:
        return None
    record_status = _record_status(run_dir) if kind == LAGGING else None
    return _row(
        project, rel, kind, st, now, run_dir, stale_after=stale_after, record_status=record_status
    )


def _projects(root: Path, project: str | None) -> list[Path]:
    """The project directories `ledger.scan` would read, so a status row's
    `project` is the name `runs list` uses for the same run once it ends."""
    if project:
        return [root / project]
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _sort_key(s: Status) -> tuple:
    # Stale first within in-flight (the thing a reader should look at), then
    # running before submitted, then most recently changed first.
    rank = {JobState.RUNNING.value: 0, JobState.SUBMITTED.value: 1}
    return (
        KIND_ORDER.index(s.kind),
        not s.stale,
        rank.get(s.state or "", 2),
        s.age_s if s.age_s is not None else 0.0,
        s.run_dir,
    )


def status(
    root: Path,
    project: str | None = None,
    *,
    now: datetime | None = None,
    stale_after: float = STALE_AFTER_S,
) -> dict:
    """Every run under the workspace `root` that is not simply finished.

    Returns `{"rows": [Status...], "counts": {...}, "message"?: str}`; rows are
    complete (callers cap them). `counts` has one entry per kind, one per
    in-flight state, and `stale`. Reads files only: no database, no writes."""
    root = Path(root).expanduser()
    if not root.is_dir():
        return {"rows": [], "counts": {}, "message": f"no such directory: {root}"}
    now = now or datetime.now(UTC)
    rows: list[Status] = []
    for proj in _projects(root, project):
        if not proj.is_dir():
            continue
        for run_dir in all_run_dirs(proj):
            got = read_one(run_dir, proj.name, _relative(run_dir, root), now, stale_after)
            if got is not None:
                rows.append(got)
    rows.sort(key=_sort_key)
    return {"rows": rows, "counts": count(rows)}


def count(rows: list[Status]) -> dict[str, int]:
    """Counts by kind, by in-flight state, and stale -- zero-filled for the
    kinds and the two in-flight states, so an absent key never reads as
    'not checked'."""
    kinds = Counter(r.kind for r in rows)
    states = Counter(r.state for r in rows if r.kind == IN_FLIGHT)
    out = {k: kinds.get(k, 0) for k in KIND_ORDER}
    out[JobState.SUBMITTED.value] = states.get(JobState.SUBMITTED.value, 0)
    out[JobState.RUNNING.value] = states.get(JobState.RUNNING.value, 0)
    out["stale"] = sum(1 for r in rows if r.stale)
    return out


def summary(counts: dict[str, int]) -> str:
    """One line: `2 in flight (1 running, 1 submitted; 1 stale); 1 unrecorded`."""
    n = counts.get(IN_FLIGHT, 0)
    head = f"{n} in flight"
    if n:
        parts = f"{counts.get('running', 0)} running, {counts.get('submitted', 0)} submitted"
        if counts.get("stale"):
            parts += f"; {counts['stale']} stale"
        head += f" ({parts})"
    tail = [f"{counts[k]} {k}" for k in KIND_ORDER[1:] if counts.get(k)]
    return "; ".join([head, *tail])


__all__ = [
    "IN_FLIGHT",
    "KIND_ORDER",
    "LAGGING",
    "STALE_AFTER_S",
    "UNREADABLE",
    "UNRECORDED",
    "Status",
    "count",
    "last_activity",
    "read_one",
    "status",
    "summary",
]
