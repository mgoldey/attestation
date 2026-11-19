"""`attestation.state/1`: state.json, its lock, settle and the terminal record.

Ported from smeltery's `smeltery.jobs` (spec, Producer findings 5.D); the two
committed fixtures under data/ were written by smeltery's own implementation
(`init_state`/`transition`/`settle` at smeltery 2e21918), so the format test
reads what a real backend wrote, not what this module would write.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
from attestation_schema import (
    SCHEMA,
    CodeIdentity,
    IllegalTransitionError,
    JobExistsError,
    JobNotFoundError,
    JobSpec,
    JobState,
    Producer,
    RunRecord,
    StateFormatError,
    StateLockUnavailableError,
    dumps_state,
    from_dict,
    handle_in,
    init_state,
    layout,
    loads_state,
    read,
    read_state,
    settle,
    spec_conflicts,
    state_path,
    terminal_record,
    transition,
    write,
)
from attestation_schema import state as state_module

DATA = pathlib.Path(__file__).resolve().parent / "data"
FIXTURES = sorted(DATA.glob("smeltery-*.state.json"))
STATE_SCHEMA_FILE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "attestation_schema"
    / "schemas"
    / "state-1.schema.json"
)
PRODUCER = Producer("example-backend", "0.1.0")
T0, T1, T2 = (
    "2026-11-19T10:00:00.000000Z",
    "2026-11-19T10:00:01.000000Z",
    "2026-11-19T10:00:02.000000Z",
)


def _spec(run_dir, **changes) -> JobSpec:
    base = dict(
        run_id="job-1",
        family="camp",
        arm=None,
        config={"campaign": "camp", "lr": 2.0},
        run_dir=str(run_dir),
        ext={"caller": {"ticket": "T-1"}},
    )
    base.update(changes)
    return JobSpec(**base)


def _own_record(spec: JobSpec, status: str = "complete", **changes) -> RunRecord:
    """The record a producer writes for the job, honouring the contract."""
    obj = {
        "schema": SCHEMA,
        "run_id": spec.run_id,
        "family": spec.family,
        "arm": spec.arm,
        "status": status,
        "started_at": None,
        "finished_at": None,
        "producer": {"name": "producer", "version": None},
        "code": {
            "git_sha": None,
            "dirty": None,
            "repo": None,
            "version": None,
            "provenance": "unknown",
            "evidence": None,
        },
        "config": {"campaign": "camp", "lr": 2},  # JCS has no 2.0: the same config
        "data": [],
        "metrics": [
            {
                "name": "hit_rate",
                "value": 0.5,
                "direction": "max",
                "step": None,
                "split": None,
                "unit": None,
                "labels": {},
            }
        ],
        "history": None,
        "seeds": None,
        "dependencies": None,
        "ext": {**spec.ext, "producer": {"inputs": {"expanded": True}}},
    }
    obj.update(changes)
    return from_dict(obj)


# ---- the file ---------------------------------------------------------------


def test_fixtures_are_real_smeltery_state_files():
    assert len(FIXTURES) == 2


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.name)
def test_reads_and_rewrites_what_smeltery_writes_byte_for_byte(fixture):
    raw = fixture.read_bytes()
    st = loads_state(raw)
    assert st.state in (JobState.FAILED, JobState.CANCELLED)
    assert dumps_state(st) == raw


def test_lifecycle_writes_the_format(tmp_path):
    st = init_state(tmp_path, run_id="job-1", backend="local", job_id="j1", at=T0)
    assert st.state is JobState.SUBMITTED and st.history[0].at == T0
    st = transition(tmp_path, JobState.RUNNING, at=T1)
    raw = state_path(tmp_path).read_bytes()
    assert not raw.endswith(b"\n")
    assert json.loads(raw) == {
        "schema": "attestation.state/1",
        "run_id": "job-1",
        "backend": "local",
        "job_id": "j1",
        "state": "running",
        "updated_at": T1,
        "message": None,
        "history": [{"state": "submitted", "at": T0}, {"state": "running", "at": T1}],
    }
    assert read_state(tmp_path) == st
    assert st.started_at == T1 and st.status.state is JobState.RUNNING
    assert handle_in(tmp_path).job_id == "j1"


def test_no_state_is_none_and_unknown_jobs_are_not_found(tmp_path):
    assert read_state(tmp_path) is None
    with pytest.raises(JobNotFoundError):
        transition(tmp_path, JobState.RUNNING)
    with pytest.raises(JobNotFoundError):
        handle_in(tmp_path)


def test_init_refuses_a_run_directory_that_holds_a_run(tmp_path):
    init_state(tmp_path, run_id="a", backend="local", job_id="j")
    with pytest.raises(JobExistsError):
        init_state(tmp_path, run_id="b", backend="local", job_id="k")
    other = tmp_path / "other"
    write(_own_record(_spec(other)), other)
    with pytest.raises(JobExistsError):
        init_state(other, run_id="b", backend="local", job_id="k")


def test_illegal_transitions_write_nothing(tmp_path):
    init_state(tmp_path, run_id="a", backend="local", job_id="j", at=T0)
    before = state_path(tmp_path).read_bytes()
    with pytest.raises(IllegalTransitionError, match="settle"):
        transition(tmp_path, JobState.COMPLETE)
    with pytest.raises(IllegalTransitionError):
        transition(tmp_path, JobState.SUBMITTED)
    assert state_path(tmp_path).read_bytes() == before


def test_a_lagging_clock_is_clamped_never_written_backwards(tmp_path):
    init_state(tmp_path, run_id="a", backend="local", job_id="j", at=T1)
    st = transition(tmp_path, JobState.RUNNING, at=T0)
    assert st.history[-1].at == T1 and st.updated_at == T1


def _valid() -> dict:
    return json.loads(FIXTURES[0].read_bytes())


def _mutate(fn):
    obj = _valid()
    fn(obj)
    return json.dumps(obj)


MALFORMED = [
    ("schema", lambda o: o.update(schema="attestation.state/2")),
    ("run_id", lambda o: o.pop("run_id")),
    ("surprise", lambda o: o.update(surprise=1)),
    ("run_id", lambda o: o.update(run_id="")),
    ("backend", lambda o: o.update(backend=None)),
    ("state", lambda o: o.update(state="done")),
    ("updated_at", lambda o: o.update(updated_at="2026-11-19T10:00:00Z")),
    ("message", lambda o: o.update(message=3)),
    ("history", lambda o: o.update(history=[])),
    ("history[0].state", lambda o: o["history"].pop(0)),
    ("history[1].at", lambda o: o["history"][1].pop("at")),
    ("history[1].state", lambda o: o["history"].insert(1, dict(o["history"][0]))),
    ("history[2].at", lambda o: o["history"][2].update(at="2026-01-01T00:00:00.000000Z")),
    ("history", lambda o: o.update(state="running")),
    ("history", lambda o: o.update(updated_at="2027-01-01T00:00:00.000000Z")),
]


def test_the_malformed_cases_start_from_a_valid_file():
    loads_state(json.dumps(_valid()))


@pytest.mark.parametrize("path, fn", MALFORMED)
def test_malformed_state_files_are_refused_never_guessed(path, fn):
    with pytest.raises(StateFormatError) as exc:
        loads_state(_mutate(fn))
    assert exc.value.path == path, exc.value


@pytest.mark.parametrize("text", ['{"schema": "a", "schema": "b"}', '{"x": NaN}', "[1, 2", b"\xff"])
def test_state_parse_is_strict(text):
    with pytest.raises(StateFormatError):
        loads_state(text)


def test_published_state_schema_agrees_with_the_reader():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(STATE_SCHEMA_FILE.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    for fixture in FIXTURES:
        assert validator.is_valid(json.loads(fixture.read_bytes())), fixture.name
    # Ordering rules are the reader's alone: transitions, time, the last entry.
    reader_only = {"history[1].state", "history[2].at", "history"}
    checked = 0
    for path, fn in MALFORMED:
        obj = _valid()
        fn(obj)
        if path in reader_only and validator.is_valid(obj):
            continue
        assert not validator.is_valid(obj), f"state schema file accepts the {path} case"
        checked += 1
    assert checked >= len(MALFORMED) - 5


# ---- settle -----------------------------------------------------------------


def _running(run_dir, spec=None) -> JobSpec:
    spec = spec or _spec(run_dir)
    init_state(run_dir, run_id=spec.run_id, backend="local", job_id="j1", at=T0)
    transition(run_dir, JobState.RUNNING, at=T1)
    return spec


def test_a_complete_run_with_its_own_record_ends_complete(tmp_path):
    spec = _running(tmp_path)
    write(_own_record(spec), tmp_path)
    st = settle(tmp_path, spec, JobState.COMPLETE, producer=PRODUCER, ended_at=T2)
    assert st.state is JobState.COMPLETE and st.message is None
    assert [h.state for h in st.history] == [
        JobState.SUBMITTED,
        JobState.RUNNING,
        JobState.COMPLETE,
    ]
    assert read(tmp_path).run_id == st.run_id == spec.run_id


def test_settle_happens_exactly_once(tmp_path):
    spec = _running(tmp_path)
    first = settle(tmp_path, spec, JobState.FAILED, "boom", producer=PRODUCER)
    record = (tmp_path / layout.RUN_JSON).read_bytes()
    again = settle(tmp_path, spec, JobState.CANCELLED, "late", producer=PRODUCER)
    assert again == first and (tmp_path / layout.RUN_JSON).read_bytes() == record


def test_complete_without_a_record_is_failed_with_one(tmp_path):
    spec = _running(tmp_path)
    st = settle(tmp_path, spec, JobState.COMPLETE, producer=PRODUCER, ended_at=T2)
    assert st.state is JobState.FAILED and "without writing its core record" in st.message
    record = read(tmp_path)
    assert record.status == "failed" and record.run_id == spec.run_id
    assert record.started_at == T1 and record.finished_at == T2
    assert record.config == spec.config and record.ext == spec.ext
    assert record.metrics == [] and record.dependencies is None and record.history is None
    assert [d.role for d in record.data] == ["manifest"]
    assert record.code == state_module.UNKNOWN_CODE


def test_a_job_cancelled_before_it_ran_gets_a_cancelled_record(tmp_path):
    spec = _spec(tmp_path)
    init_state(tmp_path, run_id=spec.run_id, backend="local", job_id="j", at=T0)
    code = CodeIdentity(None, None, None, "0.4.0", "inferred", "the backend's own install")
    st = settle(
        tmp_path,
        spec,
        JobState.CANCELLED,
        "cancelled",
        producer=PRODUCER,
        code=code,
        ext={"backend.job": {"job_id": "j"}},
    )
    assert [h.state for h in st.history] == [JobState.SUBMITTED, JobState.CANCELLED]
    record = read(tmp_path)
    assert record.status == "cancelled" and record.started_at is None and record.code == code
    assert record.ext == {**spec.ext, "backend.job": {"job_id": "j"}}


def test_an_existing_record_decides_the_terminal_state(tmp_path):
    """A cancel that races completion ends complete: run.json and state.json agree."""
    spec = _running(tmp_path)
    write(_own_record(spec), tmp_path)
    st = settle(tmp_path, spec, JobState.CANCELLED, "cancel requested", producer=PRODUCER)
    assert st.state is JobState.COMPLETE
    assert st.message == "cancel requested; its core record says complete"


def test_a_complete_record_of_a_job_that_never_reported_running_still_ran(tmp_path):
    spec = _spec(tmp_path)
    init_state(tmp_path, run_id=spec.run_id, backend="local", job_id="j", at=T0)
    write(_own_record(spec), tmp_path)
    st = settle(tmp_path, spec, JobState.COMPLETE, producer=PRODUCER, ended_at=T2)
    assert [h.state for h in st.history][-2:] == [JobState.RUNNING, JobState.COMPLETE]


@pytest.mark.parametrize(
    "changes, why",
    [
        ({"run_id": "5f7df3db8c7b"}, "run_id is '5f7df3db8c7b', the job's is 'job-1'"),
        ({"family": "other"}, "family"),
        ({"config": {"campaign": "camp", "lr": 3}}, "config is not the config"),
        ({"ext": {}}, "ext.caller"),
        ({"status": None}, "does not say how it ended"),
    ],
)
def test_a_record_that_is_not_the_jobs_ends_it_failed(tmp_path, changes, why):
    """Producer finding 5.C: the record of a job names the job's run_id and
    carries its family, arm, config and ext -- a smeltery record named by its
    input digest, holding its derived `inputs`, is not the job's record."""
    spec = _running(tmp_path)
    write(_own_record(spec, **changes), tmp_path)
    st = settle(tmp_path, spec, JobState.COMPLETE, producer=PRODUCER)
    assert st.state is JobState.FAILED and why in st.message, st.message


def test_spec_conflicts_compares_config_as_canonical_json(tmp_path):
    spec = _spec(tmp_path)
    assert spec_conflicts(spec, _own_record(spec)) == []


def test_an_unreadable_record_ends_the_job_failed(tmp_path):
    spec = _running(tmp_path)
    (tmp_path / layout.RUN_JSON).write_text("{}")
    st = settle(tmp_path, spec, JobState.COMPLETE, producer=PRODUCER)
    assert st.state is JobState.FAILED and "unreadable core record" in st.message


def test_settle_takes_the_schedulers_times_and_clamps_them(tmp_path):
    spec = _spec(tmp_path)
    init_state(tmp_path, run_id=spec.run_id, backend="temporal", job_id="wf", at=T1)
    st = settle(
        tmp_path,
        spec,
        JobState.FAILED,
        "the workflow failed",
        producer=PRODUCER,
        started_at=T2,
        ended_at=T0,  # a lagging clock: clamped to the last entry
    )
    assert [(h.state, h.at) for h in st.history] == [
        (JobState.SUBMITTED, T1),
        (JobState.RUNNING, T2),
        (JobState.FAILED, T2),
    ]
    assert read(tmp_path).finished_at == T2


def test_settle_refuses_another_jobs_spec_and_a_non_terminal_outcome(tmp_path):
    _running(tmp_path)
    with pytest.raises(ValueError, match="run_id"):
        settle(tmp_path, _spec(tmp_path, run_id="other"), JobState.FAILED, producer=PRODUCER)
    with pytest.raises(ValueError, match="terminal"):
        settle(tmp_path, _spec(tmp_path), JobState.RUNNING, producer=PRODUCER)


def test_a_symlink_in_the_run_directory_costs_the_manifest_not_the_settle(tmp_path):
    spec = _running(tmp_path)
    (tmp_path / "link").symlink_to(tmp_path / "nowhere")
    st = settle(tmp_path, spec, JobState.FAILED, "boom", producer=PRODUCER)
    assert st.state is JobState.FAILED and "no manifest" in st.message
    assert read(tmp_path).data == []


def test_terminal_record_is_only_for_jobs_that_did_not_complete(tmp_path):
    spec = _spec(tmp_path)
    kw = dict(producer=PRODUCER, started_at=None, finished_at=T0, data=[])
    with pytest.raises(ValueError, match="run's own"):
        terminal_record(spec, JobState.COMPLETE, **kw)
    with pytest.raises(ValueError, match="ext.caller"):
        terminal_record(spec, "failed", ext={"caller": {}}, **kw)
    assert terminal_record(spec, "cancelled", **kw).status == "cancelled"


# ---- the lock, across processes -----------------------------------------------

_HOLD = textwrap.dedent(
    """
    import sys
    from attestation_schema import state_lock
    with state_lock(sys.argv[1]):
        print("held", flush=True)
        sys.stdin.readline()
    """
)


@pytest.mark.skipif(state_module.fcntl is None, reason="flock is POSIX")
def test_the_lock_is_flock_on_the_attestation_directory(tmp_path):
    """Another process's flock on attestation/ -- the lock smeltery's own
    `state_lock` takes, so a backend on either implementation excludes the
    other -- is refused while state_lock is held, and granted after."""
    import fcntl

    child = subprocess.Popen(
        [sys.executable, "-c", _HOLD, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "held"
        fd = os.open(tmp_path / layout.ATTESTATION_DIR, os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            child.stdin.write("\n")
            child.stdin.flush()
            child.wait(timeout=30)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    finally:
        child.kill()
        child.wait()


_SETTLE = textwrap.dedent(
    """
    import sys
    from attestation_schema import JobSpec, JobState, Producer, settle
    run_dir, outcome = sys.argv[1], sys.argv[2]
    spec = JobSpec("job-1", "camp", None, {"campaign": "camp"}, run_dir)
    settle(run_dir, spec, JobState(outcome), outcome, producer=Producer("p", None))
    """
)


@pytest.mark.skipif(state_module.fcntl is None, reason="flock is POSIX")
def test_racing_settles_end_the_job_exactly_once(tmp_path):
    """Eight processes settle one job at once with different outcomes: one
    terminal entry, one record, and the two agree."""
    spec = _running(tmp_path, _spec(tmp_path, ext={}, config={"campaign": "camp"}))
    outcomes = ["failed", "cancelled", "complete"] * 3
    procs = [
        subprocess.Popen([sys.executable, "-c", _SETTLE, str(tmp_path), o]) for o in outcomes[:8]
    ]
    assert all(p.wait(timeout=60) == 0 for p in procs)
    st = read_state(tmp_path)
    terminal = [h for h in st.history if h.state in state_module.TERMINAL]
    assert len(terminal) == 1
    assert read(tmp_path).status == st.state.value
    assert read(tmp_path).run_id == spec.run_id


def test_no_flock_means_no_state_writes_but_reads_work(tmp_path, monkeypatch):
    """Windows has no flock; unserialized writers could lose a transition, so
    writing refuses there and reading (atomic replace needs no lock) works."""
    init_state(tmp_path, run_id="a", backend="local", job_id="j")
    monkeypatch.setattr(state_module, "fcntl", None)
    with pytest.raises(StateLockUnavailableError, match="flock"):
        transition(tmp_path, JobState.RUNNING)
    with pytest.raises(StateLockUnavailableError):
        init_state(tmp_path / "b", run_id="b", backend="local", job_id="k")
    assert read_state(tmp_path).state is JobState.SUBMITTED
