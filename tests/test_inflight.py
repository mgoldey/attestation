"""Runs in flight: `attestation.inflight`, `attest runs status`, `runs.status`.

Every fixture is written by `attestation_schema.state`'s own writers
(`init_state`, `transition`, `settle`) -- the functions a real backend calls
-- so these tests read the format producers write, not a hand-made imitation
of it. Spec: docs/superpowers/specs/2026-11-20-in-flight-run-states-design.md.
"""

import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import attestation_schema as schema
import pytest
from attestation_schema.jobs import JobSpec, JobState
from attestation_schema.record import Producer, format_timestamp
from attestation_schema.state import init_state, settle, transition

from attestation import cli, inflight, ledger
from attestation.db import get_db
from attestation.mcp import run_status

# The real clock, to the second: the CLI and the tool read `datetime.now`, and
# files the fixture does not backdate carry real mtimes.
NOW = datetime.now(UTC).replace(microsecond=0)
PRODUCER = Producer(name="test-backend", version=None)


def _at(hours_ago: float) -> str:
    return format_timestamp(NOW - timedelta(hours=hours_ago))


def _spec(run_dir: Path, run_id: str, arm: str = "a") -> JobSpec:
    return JobSpec(run_id=run_id, family="sweep", arm=arm, config={"lr": 0.1}, run_dir=run_dir)


def _record(run_id: str, arm: str = "a", status: str = "complete", wer: float = 0.2):
    return schema.from_dict(
        {
            "schema": schema.SCHEMA,
            "run_id": run_id,
            "family": "sweep",
            "arm": arm,
            "status": status,
            "started_at": None,
            "finished_at": None,
            "producer": {"name": "test", "version": None},
            "code": {
                "git_sha": None,
                "dirty": None,
                "repo": None,
                "version": None,
                "provenance": "unknown",
                "evidence": None,
            },
            "config": {"lr": 0.1},
            "data": [],
            "metrics": [
                {
                    "name": "wer",
                    "value": wer,
                    "direction": None,
                    "step": None,
                    "split": "test",
                    "unit": None,
                    "labels": {},
                }
            ],
            "history": None,
            "seeds": None,
            "dependencies": None,
            "ext": {},
        }
    )


def _age_files(run_dir: Path, hours_ago: float) -> None:
    """Backdate every file and directory in the run, so `now` is what decides."""
    t = (NOW - timedelta(hours=hours_ago)).timestamp()
    for dirpath, dirnames, filenames in os.walk(run_dir):
        for name in [*dirnames, *filenames]:
            os.utime(Path(dirpath) / name, (t, t))


@pytest.fixture
def ws(tmp_path):
    """A workspace with one of each case, in project `p`."""
    root = tmp_path / "ws"
    runs = root / "p" / "runs"

    # submitted, never started
    init_state(runs / "queued", run_id="queued", backend="queue", job_id="j1", at=_at(0.5))

    # running, last transition 10 minutes ago
    d = runs / "busy"
    init_state(d, run_id="busy", backend="queue", job_id="j2", at=_at(1))
    transition(d, JobState.RUNNING, at=_at(10 / 60))

    # running, no transition and no write for 5 hours: stale evidence
    d = runs / "quiet"
    init_state(d, run_id="quiet", backend="queue", job_id="j3", at=_at(6))
    transition(d, JobState.RUNNING, at=_at(5))
    (d / "logs").mkdir()
    (d / "logs" / "run.log").write_text("epoch 1\n")
    _age_files(d, 5)

    # running, transition 5 hours ago, but its log was written 5 minutes ago
    d = runs / "chatty"
    init_state(d, run_id="chatty", backend="queue", job_id="j4", at=_at(6))
    transition(d, JobState.RUNNING, at=_at(5))
    (d / "logs").mkdir()
    (d / "logs" / "run.log").write_text("epoch 40\n")
    _age_files(d, 5)
    os.utime(d / "logs" / "run.log", ((NOW - timedelta(minutes=5)).timestamp(),) * 2)

    # settled complete with its own record: finished, the ledger's, not listed
    d = runs / "done"
    init_state(d, run_id="done", backend="queue", job_id="j5", at=_at(3))
    transition(d, JobState.RUNNING, at=_at(2))
    schema.write(_record("done", arm="b", wer=0.1), d)
    settle(d, _spec(d, "done", arm="b"), JobState.COMPLETE, producer=PRODUCER, ended_at=_at(1))

    # settled failed with no record: settle writes a terminal record -> not listed
    d = runs / "crashed"
    init_state(d, run_id="crashed", backend="queue", job_id="j6", at=_at(3))
    transition(d, JobState.RUNNING, at=_at(2))
    settle(d, _spec(d, "crashed"), JobState.FAILED, "boom", producer=PRODUCER, ended_at=_at(1))

    # terminal state, record gone (a writer that died between the two, or deleted)
    d = runs / "orphan"
    init_state(d, run_id="orphan", backend="queue", job_id="j7", at=_at(3))
    settle(d, _spec(d, "orphan"), JobState.CANCELLED, "user", producer=PRODUCER, ended_at=_at(2))
    schema.record_path(d).unlink()

    # the run wrote its record; the backend has not settled (state.json lags)
    d = runs / "lag"
    init_state(d, run_id="lag", backend="queue", job_id="j8", at=_at(2))
    transition(d, JobState.RUNNING, at=_at(1))
    schema.write(_record("lag", arm="c", wer=0.3), d)

    # a corrupt state.json, no record
    d = runs / "garbled" / "attestation"
    d.mkdir(parents=True)
    (d / "state.json").write_text('{"schema": "attestation.state/1", "state": "running"')

    # a plain finished run with no state.json at all (an exporter's)
    schema.write(_record("exported", arm="d", wer=0.4), runs / "exported")
    return root


def _by_dir(out):
    return {Path(r.run_dir).name: r for r in out["rows"]}


def test_each_case_is_reported_in_the_schemas_own_vocabulary(ws):
    out = inflight.status(ws, now=NOW)
    rows = _by_dir(out)
    assert set(rows) == {"queued", "busy", "quiet", "chatty", "orphan", "lag", "garbled"}
    assert {s.value for s in JobState} >= {r.state for r in out["rows"] if r.state}

    assert rows["queued"].kind == inflight.IN_FLIGHT and rows["queued"].state == "submitted"
    assert rows["queued"].started_at is None and rows["queued"].age_s == 1800.0
    assert rows["busy"].state == "running" and not rows["busy"].stale
    assert rows["busy"].started_at == _at(10 / 60)
    assert (rows["busy"].backend, rows["busy"].job_id, rows["busy"].run_id) == (
        "queue",
        "j2",
        "busy",
    )
    assert rows["busy"].project == "p" and rows["busy"].run_dir == "p/runs/busy"

    assert rows["orphan"].kind == inflight.UNRECORDED and rows["orphan"].state == "cancelled"
    assert rows["orphan"].message == "user"
    assert rows["lag"].kind == inflight.LAGGING and rows["lag"].state == "running"
    assert rows["lag"].record_status == "complete"
    assert rows["garbled"].kind == inflight.UNREADABLE and rows["garbled"].state is None
    assert "state.json unreadable" in rows["garbled"].message


def test_stale_is_quiet_running_only_and_activity_counts(ws):
    rows = _by_dir(inflight.status(ws, now=NOW))
    # 5h since the transition, and nothing written since: stale evidence
    assert rows["quiet"].stale and rows["quiet"].idle_s == 5 * 3600
    # the same 5h since the transition, but a log line 5 minutes ago: alive
    assert not rows["chatty"].stale
    assert rows["chatty"].age_s == 5 * 3600 and rows["chatty"].idle_s == 300
    # a long queue is not stale: submitted jobs wait legitimately
    assert not rows["queued"].stale
    # the threshold is an argument: at 6h nothing is stale
    assert not any(r.stale for r in inflight.status(ws, now=NOW, stale_after=6 * 3600)["rows"])


def test_counts_and_order(ws):
    out = inflight.status(ws, now=NOW)
    assert out["counts"] == {
        "in_flight": 4,
        "unrecorded": 1,
        "lagging": 1,
        "unreadable": 1,
        "submitted": 1,
        "running": 3,
        "stale": 1,
    }
    order = [Path(r.run_dir).name for r in out["rows"]]
    # stale first, then running (newest change first), then submitted, then the rest
    assert order == ["quiet", "busy", "chatty", "queued", "orphan", "lag", "garbled"]
    assert inflight.summary(out["counts"]) == (
        "4 in flight (3 running, 1 submitted; 1 stale); 1 unrecorded; 1 lagging; 1 unreadable"
    )


def test_status_writes_nothing(ws):
    before = {p: p.stat().st_mtime_ns for p in ws.rglob("*") if p.is_file()}
    inflight.status(ws, now=NOW)
    after = {p: p.stat().st_mtime_ns for p in ws.rglob("*") if p.is_file()}
    assert before == after


def test_in_flight_runs_never_reach_the_ledger_or_a_ranking(ws, tmp_path):
    """The ledger holds the three recorded runs and only those; compare ranks
    them, and an in-flight run cannot appear as an arm."""
    conn = get_db(tmp_path / "t.db")
    try:
        out = ledger.scan(conn, ws)
        # the garbled state.json has no record, so scan never sees it
        assert not out["quarantined"], out["quarantined"]
        names = {r["name"] for r in conn.execute("SELECT name FROM runs")}
        assert names == {"done", "crashed", "lag", "exported"}
        result = ledger.compare(conn, "sweep", metric="wer")
        arms = {a["name"] for a in result["arms"]}
        assert not arms & {"queued", "busy", "quiet", "chatty", "orphan", "garbled"}
    finally:
        conn.close()


def test_a_project_narrows_and_a_missing_root_says_so(ws, tmp_path):
    other = ws / "q" / "runs" / "x"
    init_state(other, run_id="x", backend="queue", job_id="j9", at=_at(1))
    assert {r.project for r in inflight.status(ws, now=NOW)["rows"]} == {"p", "q"}
    assert {r.project for r in inflight.status(ws, "q", now=NOW)["rows"]} == {"q"}
    missing = inflight.status(tmp_path / "nope", now=NOW)
    assert missing["rows"] == [] and "no such directory" in missing["message"]


def test_an_empty_workspace_is_zero_in_flight(tmp_path):
    (tmp_path / "p").mkdir()
    out = inflight.status(tmp_path, now=NOW)
    assert out["rows"] == [] and out["counts"]["in_flight"] == 0
    assert inflight.summary(out["counts"]) == "0 in flight"


# ---- CLI -------------------------------------------------------------------


def test_cli_prints_groups_and_a_summary(ws, capsys):
    rc = cli.main(["runs", "status", "--root", str(ws)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "IN FLIGHT" in out and "running STALE" in out and "p/runs/quiet" in out
    assert "ENDED WITHOUT A RECORD" in out and "(record: complete)" in out
    assert "UNREADABLE" in out
    assert out.rstrip().endswith("1 unreadable")
    assert "p/runs/done" not in out and "p/runs/exported" not in out


def test_cli_json_and_stale_hours(ws, capsys):
    rc = cli.main(["runs", "status", "--root", str(ws), "--json", "--stale-hours", "1000"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0 and payload["counts"]["stale"] == 0
    assert {r["run_id"] for r in payload["rows"] if r["kind"] == "in_flight"} == {
        "queued",
        "busy",
        "quiet",
        "chatty",
    }


def test_cli_without_a_root_is_a_sentence_on_stderr(monkeypatch, capsys):
    monkeypatch.delenv("RESEARCH_ROOT", raising=False)
    assert cli.main(["runs", "status"]) == 1
    assert "RESEARCH_ROOT" in capsys.readouterr().err


# ---- MCP -------------------------------------------------------------------


def test_the_tool_returns_rows_counts_and_the_stale_caveat(ws):
    out = run_status._status(root=str(ws), stale_hours=2)
    assert out["ok"] is True
    assert out["counts"]["in_flight"] == 4 and out["n_rows"] == 7
    assert "not proof a job died" in out["message"]
    row = next(r for r in out["rows"] if r["run_id"] == "quiet")
    assert row["stale"] is True and row["state"] == "running"
    assert "submitted_at" not in row  # ages, not three timestamps per row


def test_the_tool_refuses_without_a_root(monkeypatch):
    monkeypatch.delenv("RESEARCH_ROOT", raising=False)
    out = run_status._status()
    assert out["ok"] is False and "RESEARCH_ROOT" in out["message"]
    assert out["rows"] == [] and out["counts"] == {}


def test_the_tool_is_bounded_at_its_worst_case(tmp_path):
    """MAX_STATUS_ROWS rows at their longest -- deep run dirs, long ids, a
    clipped message -- stay under the 7000-char ceiling as emitted (indent=2),
    and the message says what was not shown."""
    from test_response_size import HARD_RESPONSE_CEILING, emitted

    long = "x" * 60
    for i in range(run_status.MAX_STATUS_ROWS + 5):
        d = tmp_path / f"project_{long}" / "campaign" / f"{long}_{i:03d}"
        init_state(d, run_id=f"{long * 3}{i}", backend="b" * 99, job_id="j" * 99, at=_at(9))
        transition(d, JobState.RUNNING, "m" * 1000, at=_at(8))
    out = run_status._status(root=str(tmp_path), stale_hours=1)
    assert len(out["rows"]) == run_status.MAX_STATUS_ROWS
    assert out["n_rows"] == run_status.MAX_STATUS_ROWS + 5
    assert f"showing {run_status.MAX_STATUS_ROWS} of" in out["message"]
    for row in out["rows"]:
        for key, cap in run_status._FIELD_CAPS.items():
            assert len(row[key]) <= cap, (key, row[key])
        assert re.search(r"x_\d{3}$", row["run_dir"]), row["run_dir"]  # the tail is kept
    assert emitted(out) <= HARD_RESPONSE_CEILING, emitted(out)
