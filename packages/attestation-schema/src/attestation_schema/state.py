"""`attestation/state.json` (`attestation.state/1`): a job's state, its lock,
and the one rule that ends a job.

Every job backend needs the same four things, so they live here rather than in
each backend (they were written in smeltery's `smeltery.jobs` first and moved
here unchanged in format):

- the file: `{"schema", "run_id", "backend", "job_id", "state", "updated_at",
  "message", "history": [{"state", "at"}]}`, exactly these keys, RFC 8785
  canonical JSON with no trailing newline, replaced atomically (temporary
  file, fsync, `os.replace`). `history` starts at `submitted`, is append-only,
  every adjacent pair is an allowed transition, its times never go backwards
  (a later time from a lagging clock is clamped to the last entry) and its
  last entry is `state` at `updated_at`. Kept after the run ends.
- `state_lock`: an exclusive `flock` on the `attestation/` directory itself,
  so any two writers of one run directory (a runner and a backend in another
  process) never interleave a read-modify-write, and no lock file is added to
  a directory the schema owns. POSIX only: where there is no `fcntl` (Windows)
  it raises `StateLockUnavailableError` rather than write unserialized.
- `settle`: end a job exactly once. A record already in the run directory
  decides the terminal state, so `run.json` and `state.json` never disagree;
  a record that is not this job's (another `run_id`, `family`, `arm`,
  `config`, or missing the spec's `ext`) ends the job `failed`, saying so.
  The record is written before the state, so a crash in between leaves a
  record the next settle adopts.
- `terminal_record`: the minimal `failed`/`cancelled` record a backend writes
  for a job that died without one.

Spec: docs/superpowers/specs/2026-11-19-core-run-record-design.md ("The job
backend" and "Producer findings", 5).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from attestation_schema import layout
from attestation_schema.canonical import canonicalize
from attestation_schema.files import _atomic_write
from attestation_schema.jobs import (
    TERMINAL,
    JobError,
    JobExistsError,
    JobHandle,
    JobNotFoundError,
    JobSpec,
    JobState,
    JobStatus,
    can_transition,
)
from attestation_schema.record import (
    _TIMESTAMP,
    SCHEMA,
    CodeIdentity,
    DataDigest,
    Producer,
    RecordError,
    RunRecord,
    format_timestamp,
    from_dict,
    parse_strict,
    read,
    record_path,
    write,
)

fcntl: ModuleType | None
try:  # POSIX; absent on Windows, where state_lock refuses (module docstring)
    import fcntl as _fcntl

    fcntl = _fcntl
except ImportError:  # pragma: no cover - exercised by monkeypatching on POSIX
    fcntl = None

STATE_SCHEMA = layout.STATE_SCHEMA  # "attestation.state/1"
_KEYS = ("schema", "run_id", "backend", "job_id", "state", "updated_at", "message", "history")
_ENTRY_KEYS = ("state", "at")

# `code` for a terminal record whose writer does not know the run's code.
UNKNOWN_CODE = CodeIdentity(
    git_sha=None,
    dirty=None,
    repo=None,
    version=None,
    provenance="unknown",
    evidence="written by the job backend, which did not observe the run's code",
)


class StateFormatError(RecordError):
    """A `state.json` that breaks the format. A backend must not guess what
    state a job is in, so nothing is defaulted. `path` locates the field."""


class IllegalTransitionError(JobError):
    """A state change the job state machine does not allow (out of a terminal
    state, or `submitted -> complete`). Nothing is written."""


class StateLockUnavailableError(JobError):
    """This platform has no `flock` (Windows): `state.json` cannot be written
    safely by two processes, so it is not written at all."""


@dataclass(frozen=True)
class Transition:
    """One `history` entry: the job entered `state` at `at`."""

    state: JobState
    at: str


@dataclass(frozen=True)
class StateFile:
    """A parsed, validated `state.json`."""

    run_id: str
    backend: str
    job_id: str
    state: JobState
    updated_at: str
    message: str | None
    history: tuple[Transition, ...]
    schema: str = STATE_SCHEMA

    @property
    def status(self) -> JobStatus:
        """The backend's view of the job, as `JobBackend.status` returns it."""
        return JobStatus(self.state, self.updated_at, self.message)

    @property
    def started_at(self) -> str | None:
        """When the job started running, per its history; None if it never ran."""
        return next((h.at for h in self.history if h.state is JobState.RUNNING), None)

    def handle(self, run_dir: str | os.PathLike[str]) -> JobHandle:
        """The handle of this job, which lives in `run_dir`."""
        return JobHandle(self.backend, self.job_id, self.run_id, str(run_dir))

    def to_dict(self) -> dict[str, Any]:
        """The wire form: what `dumps_state` canonicalizes."""
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "backend": self.backend,
            "job_id": self.job_id,
            "state": self.state.value,
            "updated_at": self.updated_at,
            "message": self.message,
            "history": [{"state": h.state.value, "at": h.at} for h in self.history],
        }


# ---- the file ---------------------------------------------------------------


def _err(path: str, message: str) -> StateFormatError:
    return StateFormatError(path, message)


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _err(path, f"expected a non-empty string, got {value!r}")
    return value


def _when(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        raise _err(path, f"expected a timestamp YYYY-MM-DDTHH:MM:SS.ffffffZ, got {value!r}")
    try:
        _instant(value)
    except ValueError as exc:
        raise _err(path, f"{value!r} is not a real instant") from exc
    return value


def _job_state(value: Any, path: str) -> JobState:
    try:
        return JobState(value)
    except ValueError:
        raise _err(path, f"expected one of {[s.value for s in JobState]}, got {value!r}") from None


def _exact_keys(obj: Any, path: str, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise _err(path, f"expected an object, got {type(obj).__name__}")
    missing = [k for k in keys if k not in obj]
    if missing:
        raise _err(f"{path}.{missing[0]}" if path else missing[0], "required field is missing")
    extra = sorted(str(k) for k in obj if k not in keys)
    if extra:
        raise _err(f"{path}.{extra[0]}" if path else extra[0], f"unknown field (allowed: {keys})")
    return obj


def state_from_dict(obj: Any) -> StateFile:
    """Validate parsed JSON as `attestation.state/1`; `StateFormatError` otherwise."""
    d = _exact_keys(obj, "", _KEYS)
    if d["schema"] != STATE_SCHEMA:
        raise _err("schema", f"expected {STATE_SCHEMA!r}, got {d['schema']!r}")
    if d["message"] is not None and not isinstance(d["message"], str):
        raise _err("message", f"expected a string or null, got {d['message']!r}")
    if not isinstance(d["history"], list) or not d["history"]:
        raise _err("history", "expected a non-empty list of {state, at}")
    history = []
    for i, entry in enumerate(d["history"]):
        e = _exact_keys(entry, f"history[{i}]", _ENTRY_KEYS)
        history.append(
            Transition(
                _job_state(e["state"], f"history[{i}].state"), _when(e["at"], f"history[{i}].at")
            )
        )
    if history[0].state is not JobState.SUBMITTED:
        raise _err("history[0].state", "history starts at submitted")
    for i, (a, b) in enumerate(zip(history, history[1:], strict=False), start=1):
        if not can_transition(a.state, b.state):
            raise _err(
                f"history[{i}].state",
                f"{a.state.value} -> {b.state.value} is not an allowed transition",
            )
        if b.at < a.at:  # fixed-width UTC: text order is time order
            raise _err(f"history[{i}].at", "history never goes backwards in time")
    st = StateFile(
        run_id=_text(d["run_id"], "run_id"),
        backend=_text(d["backend"], "backend"),
        job_id=_text(d["job_id"], "job_id"),
        state=_job_state(d["state"], "state"),
        updated_at=_when(d["updated_at"], "updated_at"),
        message=d["message"],
        history=tuple(history),
    )
    if (st.state, st.updated_at) != (history[-1].state, history[-1].at):
        raise _err("history", "the last entry is the current state, at updated_at")
    return st


def loads_state(data: bytes | str) -> StateFile:
    """Parse (strictly: duplicate keys, `NaN`, huge integers refused) and validate."""
    try:
        obj = parse_strict(data)
    except RecordError as exc:
        raise _err(exc.path, str(exc)) from exc
    return state_from_dict(obj)


def dumps_state(st: StateFile) -> bytes:
    """Validate and serialize as RFC 8785 canonical JSON (no trailing newline)."""
    obj = st.to_dict()
    state_from_dict(obj)
    return canonicalize(obj)


def state_path(run_dir: str | os.PathLike[str]) -> Path:
    """Where a run directory's `state.json` lives."""
    return Path(run_dir) / layout.STATE_JSON


def read_state(run_dir: str | os.PathLike[str]) -> StateFile | None:
    """The job's `state.json`, validated; None when there is none. Needs no
    lock: writers replace the file atomically, so a reader sees one version."""
    try:
        raw = state_path(run_dir).read_bytes()
    except FileNotFoundError:
        return None
    return loads_state(raw)


def _write_state(run_dir: Path, st: StateFile) -> None:
    _atomic_write(state_path(run_dir), dumps_state(st), exclusive=False)


# ---- the lock ---------------------------------------------------------------


@contextlib.contextmanager
def state_lock(run_dir: str | os.PathLike[str]) -> Iterator[None]:
    """Exclusive while held, across processes: `flock` on `<run_dir>/attestation/`
    (created if missing). Released when the holder exits, however it exits.
    Not re-entrant: a holder that takes it again from another descriptor waits
    for itself. Raises `StateLockUnavailableError` where there is no `flock`."""
    if fcntl is None:
        raise StateLockUnavailableError(
            "state.json needs flock (POSIX); this platform has none, so no job state is written"
        )
    d = Path(run_dir) / layout.ATTESTATION_DIR
    d.mkdir(parents=True, exist_ok=True)
    fd = os.open(d, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


# ---- writing states ---------------------------------------------------------


def _now() -> str:
    return format_timestamp(datetime.now(UTC))


def _instant(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _refuse_existing(run_dir: Path) -> None:
    for p in (state_path(run_dir), record_path(run_dir)):
        if p.exists():
            raise JobExistsError(
                f"{p} exists: {run_dir} already holds a run; submit into a new run directory"
            )


def init_state(
    run_dir: str | os.PathLike[str],
    *,
    run_id: str,
    backend: str,
    job_id: str,
    at: str | None = None,
) -> StateFile:
    """Write a fresh `state.json` (`submitted`). Refuses (`JobExistsError`) a run
    directory already holding `state.json` or `run.json`: resubmission is a new
    run directory, never an overwrite."""
    run_dir = Path(run_dir)
    with state_lock(run_dir):
        _refuse_existing(run_dir)
        at = at or _now()
        st = StateFile(
            run_id=run_id,
            backend=backend,
            job_id=job_id,
            state=JobState.SUBMITTED,
            updated_at=at,
            message=None,
            history=(Transition(JobState.SUBMITTED, at),),
        )
        _write_state(run_dir, st)
    return st


def _append(st: StateFile, new: JobState, message: str | None, at: str | None) -> StateFile:
    if not can_transition(st.state, new):
        raise IllegalTransitionError(
            f"job {st.job_id}: {st.state.value} -> {new.value} is not an allowed transition"
        )
    at = at or _now()
    last = st.history[-1].at
    if _instant(at) < _instant(last):  # another clock (a scheduler's) may lag ours
        at = last
    return StateFile(
        run_id=st.run_id,
        backend=st.backend,
        job_id=st.job_id,
        state=new,
        updated_at=at,
        message=message,
        history=(*st.history, Transition(new, at)),
    )


def _locked_state(run_dir: Path) -> StateFile:
    st = read_state(run_dir)
    if st is None:
        raise JobNotFoundError(f"{state_path(run_dir)} does not exist")
    return st


def transition(
    run_dir: str | os.PathLike[str],
    new: JobState,
    message: str | None = None,
    *,
    at: str | None = None,
) -> StateFile:
    """Move the job to `new` and return the new state. `IllegalTransitionError`
    if the machine forbids it, `JobNotFoundError` without a `state.json`.
    A terminal `new` belongs to `settle`, which also guarantees the record."""
    run_dir = Path(run_dir)
    if new in TERMINAL:
        raise IllegalTransitionError(
            f"{new.value} is terminal: end a job with settle(), which also writes its record"
        )
    with state_lock(run_dir):
        st = _append(_locked_state(run_dir), new, message, at)
        _write_state(run_dir, st)
    return st


# ---- terminal records and settle ---------------------------------------------


def spec_conflicts(spec: JobSpec, record: RunRecord) -> list[str]:
    """Where `record` contradicts the job it is supposed to be the record of:
    the record of a job carries the spec's `run_id`, `family`, `arm` and
    `config` verbatim (config compared as canonical JSON, so `2.0` equals `2`)
    and every namespace of the spec's `ext` unchanged. Empty when it agrees."""
    out = []
    for name in ("run_id", "family", "arm"):
        if getattr(record, name) != getattr(spec, name):
            out.append(f"{name} is {getattr(record, name)!r}, the job's is {getattr(spec, name)!r}")
    try:
        same_config = canonicalize(record.config) == canonicalize(spec.config)
    except (TypeError, ValueError):
        same_config = False
    if not same_config:
        out.append("config is not the config the job was given")
    for namespace, body in spec.ext.items():
        if record.ext.get(namespace) != body:
            out.append(f"ext.{namespace} is not the job's")
    return out


def terminal_record(
    spec: JobSpec,
    status: JobState | str,
    *,
    producer: Producer,
    code: CodeIdentity = UNKNOWN_CODE,
    started_at: str | None,
    finished_at: str | None,
    data: list[DataDigest],
    ext: Mapping[str, dict[str, Any]] | None = None,
) -> RunRecord:
    """The minimal record of a job that ended `failed` or `cancelled` without
    writing its own: the spec's identity, config and `ext` (plus the writer's
    `ext`, in its own namespaces), its times, `data` (the manifest of whatever
    the run left), no metrics, and `history`, `seeds` and `dependencies` null
    (the writer observed none of them). Pure; validated by the reader's rules."""
    status = JobState(status)
    if status not in (JobState.FAILED, JobState.CANCELLED):
        raise ValueError(f"a {status.value} record is the run's own; only failed/cancelled here")
    merged: dict[str, dict[str, Any]] = dict(spec.ext)
    for namespace, body in (ext or {}).items():
        if namespace in merged:
            raise ValueError(f"ext.{namespace} is the job spec's; the writer cannot replace it")
        merged[namespace] = body
    return from_dict(
        {
            "schema": SCHEMA,
            "run_id": spec.run_id,
            "family": spec.family,
            "arm": spec.arm,
            "status": status.value,
            "started_at": started_at,
            "finished_at": finished_at,
            "producer": {"name": producer.name, "version": producer.version},
            "code": {
                "git_sha": code.git_sha,
                "dirty": code.dirty,
                "repo": code.repo,
                "version": code.version,
                "provenance": code.provenance,
                "evidence": code.evidence,
            },
            "config": spec.config,
            "data": [{"name": d.name, "sha256": d.sha256, "role": d.role} for d in data],
            "metrics": [],
            "history": None,
            "seeds": None,
            "dependencies": None,
            "ext": merged,
        }
    )


def _adopt(run_dir: Path, spec: JobSpec) -> tuple[JobState, str | None]:
    """The terminal state an existing record decides, and why if not its own."""
    try:
        record = read(run_dir)
    except (RecordError, OSError) as exc:
        return JobState.FAILED, f"the run directory holds an unreadable core record: {exc}"
    conflicts = spec_conflicts(spec, record)
    if conflicts:
        return JobState.FAILED, "the run directory's core record is not this job's: " + "; ".join(
            conflicts
        )
    if record.status is None:
        return JobState.FAILED, "the run's core record does not say how it ended"
    return JobState(record.status), None


def settle(
    run_dir: str | os.PathLike[str],
    spec: JobSpec,
    outcome: JobState,
    message: str | None = None,
    *,
    producer: Producer,
    code: CodeIdentity = UNKNOWN_CODE,
    ext: Mapping[str, dict[str, Any]] | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> StateFile:
    """End the job at `outcome` (complete, failed or cancelled), exactly once,
    and return its state. Under `state_lock`:

    - already terminal: returned unchanged (the first settle won);
    - a record in the run directory decides the terminal state (its status),
      unless it is not this job's (`spec_conflicts`) or unreadable: `failed`,
      and `message` says why;
    - no record: `complete` becomes `failed` (a complete run writes its own
      record; its absence is a failure to say so), and a failed or cancelled
      job gets `terminal_record` (`producer`, `code` and `ext` describe the
      writer) -- written before the state, so a crash in between leaves a
      record the next settle adopts.

    `started_at`: when the job began running, if `state.json` does not say (a
    scheduler's history may). `ended_at`: when it ended, if the caller knows
    (a scheduler's close time); otherwise now -- the time of this settle,
    which for a backend that settles on observation is later than the end.
    """
    run_dir = Path(run_dir)
    outcome = JobState(outcome)
    if outcome not in TERMINAL:
        raise ValueError(f"settle ends a job: {outcome.value} is not terminal")
    with state_lock(run_dir):
        st = _locked_state(run_dir)
        if st.state in TERMINAL:
            return st
        if spec.run_id != st.run_id:
            raise ValueError(f"the spec's run_id {spec.run_id!r} is not the job's {st.run_id!r}")
        if started_at and st.started_at is None and st.state is JobState.SUBMITTED:
            st = _append(st, JobState.RUNNING, None, started_at)
        at = ended_at or _now()
        if _instant(at) < _instant(st.history[-1].at):
            at = st.history[-1].at
        final, why = outcome, message
        if record_path(run_dir).exists():
            final, why = _settle_on_record(run_dir, spec, outcome, message)
        else:
            if final is JobState.COMPLETE:
                final, why = JobState.FAILED, "the run ended without writing its core record"
            data: list[DataDigest] = []
            try:
                data = [layout.write_manifest(run_dir)]
            except layout.LayoutError as exc:  # a symlink in the run directory
                why = f"{why + '; ' if why else ''}no manifest: {exc}"
            record = terminal_record(
                spec,
                final,
                producer=producer,
                code=code,
                started_at=st.started_at,
                finished_at=at,
                data=data,
                ext=ext,
            )
            try:
                write(record, run_dir)
            except FileExistsError:  # a producer outside the lock wrote one meanwhile
                final, why = _settle_on_record(run_dir, spec, outcome, message)
        if final is JobState.COMPLETE and st.started_at is None:  # it ran
            st = _append(st, JobState.RUNNING, None, at)
        st = _append(st, final, why, at)
        _write_state(run_dir, st)
        return st


def _settle_on_record(
    run_dir: Path, spec: JobSpec, outcome: JobState, message: str | None
) -> tuple[JobState, str | None]:
    final, problem = _adopt(run_dir, spec)
    why = message
    if problem:
        why = f"{why + '; ' if why else ''}{problem}"
    elif final is not outcome:
        why = f"{why + '; ' if why else ''}its core record says {final.value}"
    return final, why


def handle_in(run_dir: str | os.PathLike[str]) -> JobHandle:
    """The handle of the job in `run_dir`, from its `state.json`."""
    st = read_state(run_dir)
    if st is None:
        raise JobNotFoundError(f"{run_dir} holds no job (no {layout.STATE_JSON})")
    return st.handle(Path(run_dir))
