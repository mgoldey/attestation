"""attestation-schema: the core run record, its layout, and the job interface.

Spec: docs/superpowers/specs/2026-11-19-core-run-record-design.md. Every guard
here was mutated once to confirm it can fail (docs/measurement-lessons.md, 4).
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib

import pytest
from attestation_schema import (
    READERS,
    SCHEMA,
    SCHEMA_STATUS,
    CodeIdentity,
    DataDigest,
    Dependency,
    InvalidFieldError,
    JobBackend,
    JobState,
    Metric,
    MissingFieldError,
    Producer,
    RecordError,
    RunRecord,
    UnknownFieldError,
    UnsupportedSchemaError,
    can_transition,
    dumps,
    from_dict,
    layout,
    loads,
    read,
    record_digest,
    write,
)
from attestation_schema.canonical import canonicalize
from attestation_schema.jobs import TERMINAL

SCHEMA_FILE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "attestation_schema"
    / "schemas"
    / "run-1.schema.json"
)
FERRIC_SHA = "a" * 40


def _record(**changes) -> RunRecord:
    base = dict(
        run_id="run-3f2a9c01b7de",
        family="screen-campaign",
        arm="baseline",
        status="complete",
        started_at="2026-11-19T10:00:00Z",
        finished_at="2026-11-19T10:42:13.250000Z",
        producer=Producer(name="example-producer", version="0.1.0"),
        code=CodeIdentity(git_sha="0" * 40, dirty=False, repo="https://example.org/lab/code.git"),
        config={"lr": 0.001, "layers": [64, 64], "note": "π"},
        data=[DataDigest(name="inputs/library.sdf", sha256="1" * 64, role="input")],
        metrics=[
            Metric(name="loss", value=0.25, direction="min", step=1000, split="eval"),
            Metric(name="loss", value=0.31, direction="min", step=1000, split="train"),
            Metric(name="tier.dock.survivors", value=412, direction=None, step=None, split=None),
        ],
        seeds={"python": 0, "numpy": 7},
        dependencies=[
            Dependency(
                name="ferric",
                version="0.1.0",
                build_sha=FERRIC_SHA,
                dirty=False,
                provenance="verified",
                evidence="ferric.__build__",
            )
        ],
        ext={"smeltery": {"tiers": [{"name": "dock", "n_in": 5000, "n_out": 412}]}},
    )
    base.update(changes)
    return RunRecord(**base)


def _dict(**changes) -> dict:
    return json.loads(dumps(_record(**changes)))


# ---- round trip and determinism ---------------------------------------------


def test_round_trip_through_bytes_and_files(tmp_path):
    record = _record()
    assert loads(dumps(record)) == record
    path = write(record, tmp_path / "run-x")
    assert path == tmp_path / "run-x" / "attestation" / "run.json"
    assert read(tmp_path / "run-x") == record
    assert read(path) == record
    assert path.read_bytes() == dumps(record)
    assert record_digest(record) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_writer_is_deterministic_and_ignores_input_key_order():
    a = dumps(_record())
    assert a == dumps(_record())
    reordered = dict(reversed(list(_dict().items())))
    reordered["config"] = dict(reversed(list(reordered["config"].items())))
    assert canonicalize(reordered) == a
    assert dumps(from_dict(reordered)) == a
    assert not a.endswith(b"\n")


def test_writer_output_is_the_canonical_form_of_its_own_reading():
    b = dumps(_record())
    assert dumps(loads(b)) == b


def test_write_refuses_to_overwrite(tmp_path):
    write(_record(), tmp_path)
    with pytest.raises(FileExistsError):
        write(_record(arm="other"), tmp_path)
    assert read(tmp_path).arm == "baseline"
    write(_record(arm="other"), tmp_path, overwrite=True)
    assert read(tmp_path).arm == "other"
    assert sorted(p.name for p in (tmp_path / "attestation").iterdir()) == ["run.json"]


def test_writer_validates_before_writing(tmp_path):
    with pytest.raises(InvalidFieldError, match="status"):
        write(_record(status="running"), tmp_path)
    assert not (tmp_path / "attestation" / "run.json").exists()
    with pytest.raises(InvalidFieldError, match=r"config\.when"):
        dumps(_record(config={"when": {1, 2}}))
    with pytest.raises(InvalidFieldError, match="code"):
        dumps(_record(code=None))
    with pytest.raises(TypeError):
        dumps(_dict())  # type: ignore[arg-type]


def test_canonical_copy_matches_attestations_serializer():
    """The schema package carries a copy of attestation.run_record.canonical;
    the two must produce identical bytes or records stop comparing."""
    upstream = pytest.importorskip("attestation.run_record.canonical")

    vectors = [
        _dict(),
        {"\U0001f600": 1, "": 2, "a": [1e-7, 1.0, 1e16, -0.0, 123456789.125, 5e-324]},
        {"n": [0.1, 1e21, 1e-6, 2**53 - 1, -(2**53 - 1)], "s": '\u0000\u001f"\\'},
    ]
    for v in vectors:
        assert canonicalize(v) == upstream.canonicalize(v)


# ---- extensions -------------------------------------------------------------


def test_unknown_ext_namespace_round_trips_byte_for_byte():
    ext = {
        "never.heard-of_it": {
            "nested": {"z": [1, 2.5, None, True, {"\U0001f600": "x", "": "y"}]},
            "float": 1e-7,
            "big": 9007199254740991,
            "empty": {},
        },
        "smeltery": {"tiers": []},
    }
    original = canonicalize(_dict(ext=ext))
    record = loads(original)
    assert record.ext == ext
    assert dumps(record) == original
    assert canonicalize(json.loads(dumps(record))["ext"]) == canonicalize(ext)


@pytest.mark.parametrize(
    "ext, match",
    [
        ({"Bad": {}}, "namespace"),
        ({"attestation": {}}, "reserved"),
        ({"attestation.core": {}}, "reserved"),
        ({"ok": [1]}, "must be an object"),
        ({"ok": {"x": float("nan")}}, "finite"),
        ({"ok": {"x": 2**60}}, "2\\^53"),
    ],
)
def test_bad_extensions_are_refused(ext, match):
    with pytest.raises(InvalidFieldError, match=match):
        from_dict(_dict() | {"ext": ext})


# ---- required fields --------------------------------------------------------


def _paths(value, prefix=""):
    """Every key path the schema defines, from a complete record. `config` and
    `ext` are opaque, so their insides are not schema keys."""
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            yield path, (prefix, key)
            if key not in ("config", "ext", "seeds") or prefix:
                yield from _paths(item, path)
    elif isinstance(value, list) and value:
        yield from _paths(value[0], f"{prefix}[0]")


def _remove(obj, path):
    parts = path.replace("[0]", ".0").split(".")
    target = obj
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else target[part]
    del target[parts[-1]]


ALL_PATHS = [p for p, _ in _paths(_dict())]


def test_the_required_field_inventory_is_complete():
    """Guards the parametrization below against a vacuous record."""
    for nested in ("producer.version", "code.dirty", "data[0].role", "metrics[0].direction"):
        assert nested in ALL_PATHS
    assert "dependencies[0].evidence" in ALL_PATHS and "seeds" in ALL_PATHS
    assert len(ALL_PATHS) == 15 + 2 + 3 + 3 + 5 + 6


@pytest.mark.parametrize("path", ALL_PATHS)
def test_every_required_field_missing_is_a_loud_error(path):
    obj = _dict()
    _remove(obj, path)
    with pytest.raises(MissingFieldError) as exc:
        from_dict(obj)
    assert exc.value.path == path
    with pytest.raises(MissingFieldError):
        loads(json.dumps(obj))


def test_null_is_not_absent():
    obj = _dict()
    obj["family"] = None
    obj["code"]["dirty"] = None
    obj["seeds"] = None
    record = from_dict(obj)
    assert record.family is None and record.code.dirty is None and record.seeds is None


@pytest.mark.parametrize(
    "where",
    [
        lambda d: d,
        lambda d: d["producer"],
        lambda d: d["code"],
        lambda d: d["data"][0],
        lambda d: d["metrics"][0],
        lambda d: d["dependencies"][0],
    ],
)
def test_unknown_fields_are_refused(where):
    obj = _dict()
    where(obj)["surprise"] = 1
    with pytest.raises(UnknownFieldError, match="surprise"):
        from_dict(obj)


# ---- values -----------------------------------------------------------------

INVALID = [
    ("run_id", ""),
    ("run_id", 7),
    ("status", "running"),
    ("status", "COMPLETE"),
    ("started_at", "2026-11-19T10:00:00+00:00"),
    ("started_at", "2026-13-19T10:00:00Z"),
    ("finished_at", "2026-11-19T09:00:00Z"),
    ("code.git_sha", "ABC"),
    ("code.git_sha", "A" * 40),
    ("code.dirty", "no"),
    ("code.repo", "https://user:token@example.org/x.git"),
    ("code.repo", "https://token@example.org/x.git"),
    ("config", []),
    ("data", {}),
    ("data[0].sha256", "1" * 63),
    ("metrics[0].value", True),
    ("metrics[0].value", "0.25"),
    ("metrics[0].direction", "maximize"),
    ("metrics[0].step", -1),
    ("metrics[0].step", 1.0),
    ("metrics[0].split", ""),
    ("seeds", [1]),
    ("seeds", {"python": True}),
    ("dependencies[0].provenance", "VERIFIED"),
    ("dependencies[0].build_sha", "xyz"),
    ("producer", "smeltery"),
]


def _set(obj, path, value):
    parts = path.replace("[0]", ".0").split(".")
    target = obj
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else target[part]
    target[parts[-1]] = value


@pytest.mark.parametrize("path, value", INVALID)
def test_invalid_values_are_refused_never_coerced(path, value):
    obj = _dict()
    _set(obj, path, value)
    with pytest.raises(InvalidFieldError) as exc:
        from_dict(obj)
    assert exc.value.path.startswith(path.split(".")[0])


def test_cross_field_rules():
    with pytest.raises(InvalidFieldError, match="one direction per name"):
        dumps(
            _record(
                metrics=[
                    Metric("loss", 1, "min", 0, "eval"),
                    Metric("loss", 1, "max", 0, "train"),
                ]
            )
        )
    with pytest.raises(InvalidFieldError, match="duplicate metric"):
        dumps(_record(metrics=[Metric("loss", 1, None, 0, None), Metric("loss", 2, None, 0, None)]))
    with pytest.raises(InvalidFieldError, match="duplicate data name"):
        dumps(_record(data=[DataDigest("a", "1" * 64, "input"), DataDigest("a", "2" * 64, "eval")]))
    with pytest.raises(InvalidFieldError, match="verified"):
        dumps(_record(dependencies=[Dependency("x", None, None, None, "verified", None)]))
    # A dirty build can still be verified -- verified that it was dirty.
    dumps(_record(dependencies=[Dependency("ferric", None, FERRIC_SHA, True, "verified", None)]))
    dumps(_record(dependencies=[Dependency("ferric", None, None, None, "unknown", None)]))


@pytest.mark.parametrize(
    "text, match",
    [
        ('{"schema": "attestation.run/1", "schema": "attestation.run/1"}', "duplicate"),
        ('{"x": NaN}', "NaN"),
        ('{"x": Infinity}', "Infinity"),
        ('{"x": 1e400}', "overflows"),
        ('{"x": 9007199254740993}', "2\\^53"),
        ("[1, 2]", "JSON object"),
        ("{", "not JSON"),
    ],
)
def test_strict_parse(text, match):
    with pytest.raises(RecordError, match=match):
        loads(text)
    with pytest.raises(RecordError):
        loads(b"\xff\xfe")


# ---- version dispatch -------------------------------------------------------


def test_schema_identifier_and_draft_status():
    assert SCHEMA == "attestation.run/1"
    assert SCHEMA_STATUS == "draft"
    assert set(READERS) == {SCHEMA}


@pytest.mark.parametrize(
    "schema, error, match",
    [
        ("attestation.run/2", UnsupportedSchemaError, "attestation.run/1"),
        ("attestation.state/1", UnsupportedSchemaError, "not a run record"),
        ("attestation.run", InvalidFieldError, "kind"),
        ("attestation.run/01", InvalidFieldError, "kind"),
        (1, InvalidFieldError, "kind"),
    ],
)
def test_unknown_versions_are_refused(schema, error, match):
    obj = _dict()
    obj["schema"] = schema
    with pytest.raises(error, match=match):
        from_dict(obj)


def test_missing_schema_is_missing():
    obj = _dict()
    del obj["schema"]
    with pytest.raises(MissingFieldError, match="schema"):
        from_dict(obj)


def test_read_dispatches_through_the_reader_table(monkeypatch):
    """A registered version is read by its own reader -- the hook a future
    version (and the v1 upgrade step it brings) plugs into."""
    seen = []

    def read_v2(obj):
        seen.append(obj["schema"])
        legacy = dict(obj, schema=SCHEMA)
        legacy.update(seeds=None, dependencies=[])
        return READERS[SCHEMA](legacy)

    monkeypatch.setitem(READERS, "attestation.run/2", read_v2)
    obj = _dict()
    obj["schema"] = "attestation.run/2"
    record = loads(json.dumps(obj))
    assert seen == ["attestation.run/2"]
    assert record.schema == SCHEMA and record.seeds is None


# ---- JSON Schema file -------------------------------------------------------


def test_published_json_schema_agrees_with_the_reader():
    """The schema file is necessary, not sufficient: every valid record passes
    it, and every invalid value it CAN express is rejected by both."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_FILE.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    assert validator.is_valid(_dict())
    assert sorted(schema["required"]) == sorted(_dict())
    for path in ALL_PATHS:
        obj = _dict()
        _remove(obj, path)
        assert not validator.is_valid(obj), f"schema file accepts a record missing {path}"
    reader_only = {
        ("finished_at", "'2026-11-19T09:00:00Z'"),  # finished_at >= started_at
        ("code.repo", "'https://user:token@example.org/x.git'"),  # credentials
        ("code.repo", "'https://token@example.org/x.git'"),
        ("started_at", "'2026-13-19T10:00:00Z'"),  # well-formed, impossible date
        ("metrics[0].step", "1.0"),  # JSON Schema's integer admits 1.0
    }
    checked = 0
    for path, value in INVALID:
        if (path, repr(value)) in reader_only:
            continue
        obj = _dict()
        _set(obj, path, value)
        assert not validator.is_valid(obj), f"schema file accepts {path}={value!r}"
        checked += 1
    assert checked == len(INVALID) - len(reader_only)


# ---- layout -----------------------------------------------------------------


def test_layout_paths_are_fixed(tmp_path):
    assert layout.RUN_JSON == "attestation/run.json"
    assert layout.STATE_JSON == "attestation/state.json"
    assert layout.DIGESTS_JSON == "attestation/digests.json"
    assert layout.step_dir(tmp_path, 0, "dock") == tmp_path / "steps" / "00-dock"
    assert layout.step_log(tmp_path, 12, "xtb") == tmp_path / "logs" / "12-xtb.log"
    for bad in ("Dock", "", "-x", "a/b"):
        with pytest.raises(layout.LayoutError):
            layout.step_name(0, bad)
    with pytest.raises(layout.LayoutError):
        layout.step_name(-1, "x")


def test_find_runs_sees_finished_and_in_flight_runs(tmp_path):
    write(_record(), tmp_path / "a" / "run-1")
    (tmp_path / "b" / "run-2" / "attestation").mkdir(parents=True)
    (tmp_path / "b" / "run-2" / "attestation" / "state.json").write_text("{}")
    (tmp_path / "c" / "not-a-run" / "steps").mkdir(parents=True)
    (tmp_path / ".hidden" / "run-3" / "attestation").mkdir(parents=True)
    (tmp_path / ".hidden" / "run-3" / "attestation" / "run.json").write_text("{}")
    assert layout.find_runs(tmp_path) == [tmp_path / "a" / "run-1", tmp_path / "b" / "run-2"]


def test_manifest_binds_outputs_to_the_record(tmp_path):
    run = tmp_path / "run"
    layout.step_dir(run, 0, "dock").mkdir(parents=True)
    (layout.step_dir(run, 0, "dock") / "poses.sdf").write_text("x")
    layout.step_log(run, 0, "dock").parent.mkdir(parents=True)
    layout.step_log(run, 0, "dock").write_text("log")
    (run / "attestation").mkdir()
    (run / "attestation" / "state.json").write_text("{}")
    entry = layout.write_manifest(run)
    manifest = json.loads((run / layout.DIGESTS_JSON).read_bytes())
    assert manifest == {
        "schema": "attestation.digests/1",
        "files": {
            "logs/00-dock.log": hashlib.sha256(b"log").hexdigest(),
            "steps/00-dock/poses.sdf": hashlib.sha256(b"x").hexdigest(),
        },
    }
    assert entry.role == "manifest" and entry.name == layout.DIGESTS_JSON
    assert entry.sha256 == hashlib.sha256((run / layout.DIGESTS_JSON).read_bytes()).hexdigest()
    write(_record(data=[entry]), run)
    with pytest.raises(layout.LayoutError, match="before the record"):
        layout.write_manifest(run)


def test_manifest_refuses_symlinks(tmp_path):
    (tmp_path / "steps").mkdir()
    (tmp_path / "steps" / "link").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(layout.LayoutError, match="symlink"):
        layout.build_manifest(tmp_path)


# ---- job backend interface --------------------------------------------------


def test_state_machine():
    S = JobState
    assert TERMINAL == {S.COMPLETE, S.FAILED, S.CANCELLED}
    assert can_transition(S.SUBMITTED, S.RUNNING)
    assert can_transition(S.SUBMITTED, S.FAILED)
    assert can_transition(S.SUBMITTED, S.CANCELLED)
    assert not can_transition(S.SUBMITTED, S.COMPLETE)
    for end in (S.COMPLETE, S.FAILED, S.CANCELLED):
        assert can_transition(S.RUNNING, end)
        assert not any(can_transition(end, s) for s in S)
    assert not can_transition(S.RUNNING, S.SUBMITTED)
    # The record's terminal statuses are exactly the job's terminal states.
    from attestation_schema.record import STATUSES

    assert set(STATUSES) == {s.value for s in TERMINAL}


def test_job_backend_is_a_structural_protocol():
    class Fake:
        name = "fake"

        def submit(self, spec): ...
        def status(self, handle): ...
        def cancel(self, handle): ...
        def result(self, handle): ...

    class Partial:
        name = "partial"

        def submit(self, spec): ...

    assert isinstance(Fake(), JobBackend)
    assert not isinstance(Partial(), JobBackend)


def test_records_are_value_objects():
    a = _record()
    b = copy.deepcopy(a)
    assert a == b and a is not b


# ---- distribution -----------------------------------------------------------


def test_wheel_builds_and_carries_the_schema_file(tmp_path):
    """`uv build --package attestation-schema` -- the same command CI's wheel
    smoke job runs -- produces a dependency-free wheel with the JSON Schema
    and py.typed inside."""
    import shutil
    import subprocess
    import zipfile

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    repo = pathlib.Path(__file__).resolve().parents[3]
    subprocess.run(
        [uv, "build", "--package", "attestation-schema", "--wheel", "--out-dir", str(tmp_path)],
        cwd=repo,
        check=True,
        capture_output=True,
        timeout=300,
    )
    (wheel,) = tmp_path.glob("attestation_schema-*.whl")
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        metadata = next(zf.read(n) for n in names if n.endswith(".dist-info/METADATA"))
    assert "attestation_schema/schemas/run-1.schema.json" in names
    assert "attestation_schema/py.typed" in names
    assert not any(n.startswith("attestation/") for n in names)
    assert b"Requires-Dist" not in metadata
