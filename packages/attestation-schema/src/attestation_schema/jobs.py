"""The job-backend interface: what a runner (local process, a batch queue)
implements so attestation can submit, watch, cancel and collect a run.

Interface only -- no backend ships here. States and transitions:

    submitted -> running | failed | cancelled
    running   -> complete | failed | cancelled
    complete, failed, cancelled: terminal, no exits

Spec: docs/superpowers/specs/2026-11-19-core-run-record-design.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from attestation_schema.record import RunRecord


class JobState(StrEnum):
    """Where a job is. The value is what `state.json` stores."""

    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.SUBMITTED: frozenset({JobState.RUNNING, JobState.FAILED, JobState.CANCELLED}),
    JobState.RUNNING: frozenset({JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}),
    JobState.COMPLETE: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.CANCELLED: frozenset(),
}
TERMINAL: frozenset[JobState] = frozenset(s for s, nxt in TRANSITIONS.items() if not nxt)


def can_transition(current: JobState, new: JobState) -> bool:
    """Whether a job in `current` may move to `new`."""
    return new in TRANSITIONS[current]


class JobError(Exception):
    """Base for job-backend refusals."""


class JobExistsError(JobError):
    """`submit` found a state.json or run.json in the run directory already."""


class JobNotTerminalError(JobError):
    """`result` was asked for a job that has not finished."""


class JobNotFoundError(JobError):
    """The backend does not know the handle."""


@dataclass(frozen=True)
class JobSpec:
    """What to run and where. `config`/`ext` mean what they mean in the record."""

    run_id: str
    family: str | None
    arm: str | None
    config: dict[str, Any]
    run_dir: str | os.PathLike[str]
    ext: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class JobHandle:
    """A submitted job, as its backend names it."""

    backend: str
    job_id: str
    run_id: str
    run_dir: str | os.PathLike[str]


@dataclass(frozen=True)
class JobStatus:
    """A job's state as of `updated_at` (RFC 3339 UTC, `Z`)."""

    state: JobState
    updated_at: str
    message: str | None = None


@runtime_checkable
class JobBackend(Protocol):
    """The contract. Every terminal job has a core record at
    `<run_dir>/attestation/run.json`; a backend whose job died without one
    writes a `failed` record itself."""

    name: str

    def submit(self, spec: JobSpec) -> JobHandle:
        """Start a job. Writes state.json (`submitted`) before returning;
        raises JobExistsError if the run directory already holds a run."""
        ...

    def status(self, handle: JobHandle) -> JobStatus:
        """The job's current state; raises JobNotFoundError for an unknown handle."""
        ...

    def cancel(self, handle: JobHandle) -> JobStatus:
        """Request cancellation. Idempotent: a terminal job's status comes back unchanged."""
        ...

    def result(self, handle: JobHandle) -> RunRecord:
        """The core record of a terminal job; JobNotTerminalError before then."""
        ...
