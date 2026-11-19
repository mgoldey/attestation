"""The v1 freeze, pinned: an accidental edit to a frozen format fails here.

`attestation.run/1`, `attestation.history/1` and `attestation.state/1` were
frozen on 2026-11-19 (spec: docs/superpowers/specs/2026-11-19-core-run-record-
design.md, "v1 freeze"). A frozen format never changes -- not even by an
additive optional field, because a v1 reader refuses unknown keys, so a record
carrying one would stop reading in every deployed reader. A change is a new
version: a new schema file (`run-2.schema.json`), a new `READERS` entry, an
upgrade step from v1, and a new row below. These digests are therefore never
updated to make this test pass.
"""

from __future__ import annotations

import hashlib
import pathlib

from attestation_schema import (
    HISTORY_PATH,
    SCHEMA_STATUS,
    CodeIdentity,
    DataDigest,
    Dependency,
    HistoryRef,
    JobState,
    Metric,
    Producer,
    RunRecord,
    StateFile,
    Transition,
    dumps,
    dumps_state,
)

SCHEMAS = pathlib.Path(__file__).resolve().parents[1] / "src" / "attestation_schema" / "schemas"

POLICY = (
    "is FROZEN (attestation.*/1, 2026-11-19). Do not edit it, and do not update this digest "
    "to make the test pass: any change -- an additive optional field included -- is a new "
    "version (a new <kind>-2.schema.json, a READERS entry, an upgrade step from v1). See the "
    "spec's 'v1 freeze' section, docs/superpowers/specs/2026-11-19-core-run-record-design.md."
)

FROZEN_SCHEMA_FILES = {
    "run-1.schema.json": "3da18fcbabab886135c1cd7d18b21e5714b27728401575225e491c3a4c814000",
    "history-1.schema.json": "79b96deed3777e929f0ce2e5b898b595144bf950c8a8aa3e37e7319eef4332f3",
    "state-1.schema.json": "ba654d8c6df01452a6cf611fc4ee42631c67d110c5493b2df64ff4444d45fbd5",
}


def test_frozen_json_schema_files_are_byte_identical():
    present = sorted(p.name for p in SCHEMAS.glob("*.schema.json"))
    assert set(FROZEN_SCHEMA_FILES) <= set(present), f"a frozen schema file is missing: {present}"
    for name, digest in FROZEN_SCHEMA_FILES.items():
        actual = hashlib.sha256((SCHEMAS / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name} changed (sha256 {actual}): it {POLICY}"


def test_status_is_frozen():
    assert SCHEMA_STATUS == "frozen", f"attestation.run/1 {POLICY}"


# The writer is part of the format: the same record must give the same bytes
# in every release, or a record's digest -- its identity -- would change under
# it. One record exercising every field, and one state file.
REFERENCE_RECORD = RunRecord(
    run_id="run-3f2a9c01b7de",
    family="lr-sweep",
    arm="lr=1e-3",
    status="complete",
    started_at="2026-11-19T10:00:00.000000Z",
    finished_at="2026-11-19T10:42:13.250000Z",
    producer=Producer(name="train.py", version="0.1.0"),
    code=CodeIdentity("0" * 40, False, "https://example.org/x.git", "0.4.0", "inferred", "HEAD"),
    config={"lr": 0.001, "layers": [64, 64], "big": 1e21, "small": 1e-7, "note": "π"},
    data=[DataDigest("inputs/train.csv", "1" * 64, "train"), DataDigest("ds@ab12", None, "eval")],
    metrics=[
        Metric("loss", 0.25, "min", 1000, "eval", None, {}),
        Metric("loss", float("nan"), "min", 1000, "train", None, {}),
        Metric("ddE", -1.2, "min", None, None, "kcal/mol", {"analogue": "4-F"}),
    ],
    history=HistoryRef(HISTORY_PATH, "2" * 64, series=2, points=20),
    seeds={"python": 0, "docking": [7, 8]},
    dependencies=[
        Dependency("ferric", "0.1.0", "a" * 40, False, "verified", "ferric.__build__", None),
        Dependency("python", "3.12.12", None, None, "verified", None, "system"),
        Dependency("pkg", "1.0", None, None, "inferred", None, "url:https://example.org/p.whl"),
    ],
    ext={"producer.ns": {"anything": [1, {"b": None, "a": True}]}},
)
REFERENCE_RECORD_SHA256 = "2ba4cf3a40bd1d1023e46608883d2bc754d6b0dd990c4a18098cf3f244be0200"

REFERENCE_STATE = StateFile(
    run_id="07693d07e8b9",
    backend="local",
    job_id="817549d48ac9",
    state=JobState.FAILED,
    updated_at="2026-11-19T15:37:20.619766Z",
    message="smeltery run exited 1",
    history=(
        Transition(JobState.SUBMITTED, "2026-11-19T15:37:19.777219Z"),
        Transition(JobState.RUNNING, "2026-11-19T15:37:20.124245Z"),
        Transition(JobState.FAILED, "2026-11-19T15:37:20.619766Z"),
    ),
)
REFERENCE_STATE_SHA256 = "7261769c981d0c494bc6f97f02d0f8febfa1fd7780f6b301f89dc952fe3e27c1"


def test_the_writer_gives_the_frozen_bytes():
    actual = hashlib.sha256(dumps(REFERENCE_RECORD)).hexdigest()
    assert actual == REFERENCE_RECORD_SHA256, f"dumps() changed ({actual}): the writer {POLICY}"
    actual = hashlib.sha256(dumps_state(REFERENCE_STATE)).hexdigest()
    assert actual == REFERENCE_STATE_SHA256, f"dumps_state() changed ({actual}): it {POLICY}"
