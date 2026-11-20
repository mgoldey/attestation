"""Runs in flight over MCP: `runs.status`.

Under the `runs` namespace, so the provenance surface serves it. Its own
module rather than more lines in provenance.py: it reads files at query time
(`attestation.inflight`), never the ledger database, so it shares nothing
with the other runs.* tools but the workspace root.

Spec: docs/superpowers/specs/2026-11-20-in-flight-run-states-design.md.
"""

from typing import Annotated

from pydantic import Field

from attestation import inflight, ledger
from attestation.mcp._tool import ToolError, tool
from attestation.mcp.provenance import NO_ROOT

# Rows one call may return. MEASURED against the 7000-char ceiling in
# tests/test_inflight.py at the worst case every field cap below permits:
# 12 uncapped rows emitted 8778; 8 rows at every cap emit 6275 (~750 each at
# indent=2, the envelope included). `counts` is always
# complete; the message says how many rows were not shown.
MAX_STATUS_ROWS = 8

# Every string in a row comes from a file someone else wrote, so every one is
# capped: a state.json `message` can be a crash's traceback tail (a producer
# may cap its own at 400), and run_id/job_id/backend have no length limit in
# `attestation.state/1`. Clipped in the MCP payload only; the CLI prints them
# whole. run_dir keeps its TAIL -- the run's own directory name is the part
# that identifies it.
MAX_MESSAGE_CHARS = 120
_FIELD_CAPS = {
    "project": 40,
    "run_dir": 120,
    "run_id": 64,
    "backend": 40,
    "job_id": 64,
    "message": MAX_MESSAGE_CHARS,
}

# The row fields an agent needs. `submitted_at`/`started_at`/`updated_at` are
# dropped: `age_s`/`idle_s` already say how long ago in the unit a reader
# acts on, and three timestamps per row were a third of the payload.
_ROW_KEYS = (
    "project",
    "run_dir",
    "kind",
    "state",
    "run_id",
    "backend",
    "job_id",
    "age_s",
    "idle_s",
    "stale",
    "record_status",
    "message",
)

StaleHours = Annotated[float, Field(gt=0, le=24 * 30)]


def register(mcp) -> None:
    """Attach runs.status to the server."""

    @mcp.tool(name="runs.status")
    def runs_status(
        root: str | None = None,
        project: str | None = None,
        stale_hours: StaleHours = inflight.STALE_AFTER_S / 3600,
    ) -> dict:
        """What is running now: experiment runs in flight, read from disk.

        Lists run directories whose `attestation/state.json` says submitted or
        running and which have no `run.json` yet -- the runs `runs.list` and
        `runs.compare` cannot show, because the ledger records runs that ended.
        Also lists a finished state with no record (`unrecorded`), a record
        whose state was never settled (`lagging`; the record wins), and an
        unreadable state.json. `stale: true` means a running job made no
        transition and wrote nothing to its history or logs for `stale_hours`:
        evidence to check, NOT proof it died -- say so when you relay it.
        Reads files only; changes nothing. `root` defaults to RESEARCH_ROOT.
        """
        return _status(root, project, stale_hours)


def _clip(value: str | None, cap: int, *, tail: bool = False) -> str | None:
    if value is None or len(value) <= cap:
        return value
    return "..." + value[-(cap - 3) :] if tail else value[: cap - 3] + "..."


def _row(r: inflight.Status) -> dict:
    d = r.to_dict()
    out = {k: d[k] for k in _ROW_KEYS}
    for key, cap in _FIELD_CAPS.items():
        out[key] = _clip(out[key], cap, tail=key == "run_dir")
    return out


@tool(
    empty={"counts": {}, "rows": [], "n_rows": 0},
    needs_db=False,
    label="runs_status",
)
def _status(
    root: str | None = None,
    project: str | None = None,
    stale_hours: float = inflight.STALE_AFTER_S / 3600,
) -> dict:
    target = ledger.workspace_root(root)
    if target is None:
        raise ToolError(NO_ROOT)
    out = inflight.status(target, project, stale_after=stale_hours * 3600)
    if out.get("message"):
        raise ToolError(
            f"{out['message']} -- set RESEARCH_ROOT or pass root= to a directory that exists"
        )
    rows = out["rows"]
    message = inflight.summary(out["counts"])
    if out["counts"].get("stale"):
        message += " -- stale is evidence (no transition or write lately), not proof a job died"
    if len(rows) > MAX_STATUS_ROWS:
        message += f"; showing {MAX_STATUS_ROWS} of {len(rows)} -- pass project= to narrow"
    return {
        "message": message,
        "counts": out["counts"],
        "rows": [_row(r) for r in rows[:MAX_STATUS_ROWS]],
        "n_rows": len(rows),
    }
