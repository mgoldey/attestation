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
    HISTORY_PATH,
    READERS,
    SCHEMA,
    SCHEMA_STATUS,
    CodeIdentity,
    DataDigest,
    Dependency,
    HistoryDigestError,
    HistoryFormatError,
    HistoryMissingError,
    HistoryRef,
    HistoryWriter,
    InvalidFieldError,
    JobBackend,
    JobState,
    Metric,
    MissingFieldError,
    Point,
    Producer,
    RecordError,
    RunRecord,
    Series,
    UnknownFieldError,
    UnsupportedSchemaError,
    can_transition,
    dumps,
    format_timestamp,
    from_dict,
    layout,
    loads,
    read,
    read_history,
    record_digest,
    write,
    write_history,
)
from attestation_schema.canonical import canonicalize
from attestation_schema.jobs import TERMINAL

SCHEMAS = pathlib.Path(__file__).resolve().parents[1] / "src" / "attestation_schema" / "schemas"
SCHEMA_FILE = SCHEMAS / "run-1.schema.json"
HISTORY_SCHEMA_FILE = SCHEMAS / "history-1.schema.json"
FERRIC_SHA = "a" * 40


def _record(**changes) -> RunRecord:
    base = dict(
        run_id="run-3f2a9c01b7de",
        family="screen-campaign",
        arm="baseline",
        status="complete",
        started_at="2026-11-19T10:00:00.000000Z",
        finished_at="2026-11-19T10:42:13.250000Z",
        producer=Producer(name="example-producer", version="0.1.0"),
        code=CodeIdentity(
            git_sha="0" * 40,
            dirty=False,
            repo="https://example.org/lab/code.git",
            version="0.4.0",
            provenance="inferred",
            evidence="editable install: the checkout's HEAD",
        ),
        config={"lr": 0.001, "layers": [64, 64], "note": "π"},
        data=[DataDigest(name="inputs/library.sdf", sha256="1" * 64, role="input")],
        metrics=[
            Metric("loss", 0.25, "min", 1000, "eval", None, {}),
            Metric("loss", 0.31, "min", 1000, "train", None, {}),
            Metric("tier.dock.survivors", 412, None, None, None, None, {}),
            Metric("paired_ddE.mean", -1.2, "min", None, None, "kcal/mol", {"analogue": "4-F"}),
            Metric("paired_ddE.mean", -0.4, "min", None, None, "kcal/mol", {"analogue": "4-Cl"}),
        ],
        # A reference only; tests that write() a record either write this
        # sidecar first or pass history=None.
        history=HistoryRef(path=HISTORY_PATH, sha256="2" * 64, series=1, points=3),
        seeds={"python": 0, "numpy": 7, "docking": [7, 8]},
        dependencies=[
            Dependency(
                name="ferric",
                version="0.1.0",
                build_sha=FERRIC_SHA,
                dirty=False,
                provenance="verified",
                evidence="ferric.__build__",
                source="git+https://example.org/lab/ferric.git",
            )
        ],
        ext={"smeltery": {"tiers": [{"name": "dock", "n_in": 5000, "n_out": 412}]}},
    )
    base.update(changes)
    return RunRecord(**base)


def _dict(**changes) -> dict:
    return json.loads(dumps(_record(**changes)))


LOSS = Series(
    name="loss",
    split="train",
    labels={},
    points=[
        Point(step=0, value=0.9, at="2026-11-19T10:00:01.500000Z"),
        Point(step=0, value=0.8, at="2026-11-19T10:00:02.000000Z"),
        Point(step=1000, value=0.31, at=None),
    ],
)


# ---- round trip and determinism ---------------------------------------------


def test_round_trip_through_bytes_and_files(tmp_path):
    assert loads(dumps(_record())) == _record()
    ref = write_history(tmp_path / "run-x", [LOSS])
    record = _record(history=ref)
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
    write(_record(history=None), tmp_path)
    with pytest.raises(FileExistsError):
        write(_record(arm="other", history=None), tmp_path)
    assert read(tmp_path).arm == "baseline"
    write(_record(arm="other", history=None), tmp_path, overwrite=True)
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
    """attestation.run_record.canonical re-exports this package's serializer
    (it was a copy until attestation took a runtime dependency on this
    package); the two must stay one function, or records stop comparing.
    Skipped where attestation is not installed (the 3.11 schema-only job)."""
    upstream = pytest.importorskip("attestation.run_record.canonical")
    assert upstream.canonicalize is canonicalize

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
            opaque = (key in ("config", "ext", "seeds") and not prefix) or key == "labels"
            if not opaque:
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
    assert "history.sha256" in ALL_PATHS and "history.points" in ALL_PATHS
    assert "code.provenance" in ALL_PATHS and "metrics[0].labels" in ALL_PATHS
    assert "dependencies[0].source" in ALL_PATHS and "history.path" in ALL_PATHS
    assert len(ALL_PATHS) == 16 + 2 + 6 + 3 + 7 + 4 + 7


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
    obj["history"] = None
    obj["dependencies"] = None
    record = from_dict(obj)
    assert record.family is None and record.code.dirty is None and record.seeds is None
    assert record.history is None and record.dependencies is None


def test_dependencies_null_is_not_empty():
    """null = the producer could not observe the environment; [] = it did and
    the code ran on nothing worth naming (as seeds: null vs {})."""
    unobserved, none = _record(dependencies=None), _record(dependencies=[])
    assert loads(dumps(unobserved)).dependencies is None
    assert loads(dumps(none)).dependencies == []
    assert dumps(unobserved) != dumps(none)


@pytest.mark.parametrize(
    "where",
    [
        lambda d: d,
        lambda d: d["producer"],
        lambda d: d["code"],
        lambda d: d["data"][0],
        lambda d: d["metrics"][0],
        lambda d: d["history"],
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
    ("started_at", "2026-13-19T10:00:00.000000Z"),
    ("started_at", "2026-11-19T10:00:00Z"),
    ("started_at", "2026-11-19T10:00:00.5Z"),
    ("started_at", "2026-11-19T10:00:00.500Z"),
    ("started_at", "2026-11-19T10:00:00.0000000Z"),
    ("started_at", "2026-11-19T10:00:00.Z"),
    ("finished_at", "2026-11-19T09:00:00.000000Z"),
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
    ("metrics[0].value", "nan"),
    ("metrics[0].value", "Inf"),
    ("metrics[0].direction", "maximize"),
    ("metrics[0].step", -1),
    ("metrics[0].step", 1.0),
    ("metrics[0].split", ""),
    ("history", []),
    ("history", "attestation/history.jsonl"),
    ("history.path", "attestation/curves.jsonl"),
    ("history.path", "../elsewhere/history.jsonl"),
    ("history.sha256", "2" * 63),
    ("history.sha256", None),
    ("history.series", 0),
    ("history.series", True),
    ("history.points", 0),
    ("history.points", 1.5),
    ("dependencies", {}),
    ("seeds", [1]),
    ("seeds", {"docking": []}),
    ("seeds", {"docking": [7, "8"]}),
    ("code.provenance", "VERIFIED"),
    ("code.version", ""),
    ("metrics[0].unit", ""),
    ("metrics[0].labels", []),
    ("metrics[0].labels", {"analogue": 4}),
    ("metrics[0].labels", {"analogue": ""}),
    ("dependencies[0].source", ""),
    ("dependencies[0].source", "https://user:pw@example.org/x.git"),
    ("dependencies[0].source", "pypi"),
    ("dependencies[0].source", "git+https://user:pw@example.org/x.git"),
    ("dependencies[0].source", "git+https://token@example.org/x.git"),
    ("dependencies[0].source", "index:https://user:pw@pypi.example.org/simple"),
    ("dependencies[0].source", "conda:"),
    ("dependencies[0].source", "File"),
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
                    Metric("loss", 1, "min", 0, "eval", None, {}),
                    Metric("loss", 1, "max", 0, "train", None, {}),
                ]
            )
        )
    with pytest.raises(InvalidFieldError, match="duplicate metric"):
        dumps(
            _record(
                metrics=[
                    Metric("loss", 1, None, 0, None, None, {}),
                    Metric("loss", 2, None, 0, None, None, {}),
                ]
            )
        )
    with pytest.raises(InvalidFieldError, match="one unit per name"):
        dumps(
            _record(
                metrics=[
                    Metric("dE", 1, None, None, None, "kcal/mol", {"a": "x"}),
                    Metric("dE", 2, None, None, None, "kJ/mol", {"a": "y"}),
                ]
            )
        )
    with pytest.raises(InvalidFieldError, match="duplicate metric"):
        dumps(
            _record(
                metrics=[
                    Metric("dE", 1, None, None, None, None, {"a": "x", "b": "y"}),
                    Metric("dE", 2, None, None, None, None, {"b": "y", "a": "x"}),
                ]
            )
        )
    with pytest.raises(InvalidFieldError, match="verified"):
        dumps(_record(code=CodeIdentity(None, None, None, None, "verified", "a guess")))
    dumps(_record(code=CodeIdentity(None, None, None, "0.0.2", "verified", "index release")))
    dumps(_record(code=CodeIdentity(None, None, None, None, "unknown", None)))
    with pytest.raises(InvalidFieldError, match="duplicate data name"):
        dumps(_record(data=[DataDigest("a", "1" * 64, "input"), DataDigest("a", "2" * 64, "eval")]))
    with pytest.raises(InvalidFieldError, match="verified"):
        dumps(_record(dependencies=[Dependency("x", None, None, None, "verified", None, None)]))
    # A dirty build can still be verified -- verified that it was dirty.
    dumps(
        _record(dependencies=[Dependency("ferric", None, FERRIC_SHA, True, "verified", None, None)])
    )
    dumps(_record(dependencies=[Dependency("ferric", None, None, None, "unknown", None, None)]))
    # `unknown` claims no sha: a run whose rows came from two commits names
    # both in `evidence`, not one of them in `git_sha` (smeltery, finding 2.9).
    with pytest.raises(InvalidFieldError, match="'unknown' claims no git_sha"):
        dumps(_record(code=CodeIdentity("0" * 40, None, None, None, "unknown", "mixed")))
    with pytest.raises(InvalidFieldError, match="'unknown' claims no build_sha"):
        dumps(
            _record(
                dependencies=[Dependency("f", "0.1.0", FERRIC_SHA, None, "unknown", None, None)]
            )
        )
    mixed = CodeIdentity(
        None, None, None, "0.4.0", "unknown", f"mixed: {'1' * 40} (412 rows), {'2' * 40} (88 rows)"
    )
    assert loads(dumps(_record(code=mixed))).code == mixed
    with pytest.raises(InvalidFieldError, match="at least one point"):
        dumps(_record(history=HistoryRef(HISTORY_PATH, "2" * 64, series=3, points=2)))


def test_non_finite_metric_values_are_spelled_not_dropped():
    """A diverged loss is a result. In Python it is a float; on the wire it
    is a string, because JSON has no NaN and JCS refuses one."""
    import math

    record = _record(
        metrics=[
            Metric("loss", math.nan, "min", 3, None, None, {}),
            Metric("grad", math.inf, None, 3, None, None, {}),
            Metric("score", -math.inf, None, 3, None, None, {}),
        ],
    )
    b = dumps(record)
    wire = json.loads(b)
    assert [m["value"] for m in wire["metrics"]] == ["NaN", "Infinity", "-Infinity"]
    back = loads(b)
    assert math.isnan(back.metrics[0].value) and back.metrics[1].value == math.inf
    assert back.metrics[2].value == -math.inf
    assert dumps(back) == b
    # The JSON form is the string: a raw float NaN handed to from_dict is refused,
    # and non-finite numbers stay refused outside metric values.
    obj = _dict()
    obj["metrics"][0]["value"] = math.nan
    with pytest.raises(InvalidFieldError, match="finite"):
        from_dict(obj)
    with pytest.raises(InvalidFieldError, match="finite"):
        dumps(_record(config={"lr": math.nan}))


def test_to_dict_is_the_wire_form_and_spell_value_is_public():
    """The dataclass holds float('nan'); its dict, and only its dict, holds the
    wire string -- so from_dict(to_dict(r)) == r with no private helper, and
    from_dict still accepts only JSON (smeltery, finding 2.11)."""
    import math

    from attestation_schema import spell_value

    record = _record(metrics=[Metric("loss", math.nan, "min", 3, None, None, {})])
    obj = record.to_dict()
    assert obj["metrics"][0]["value"] == "NaN"
    back = from_dict(obj)
    assert math.isnan(back.metrics[0].value) and dumps(back) == dumps(record)
    assert [spell_value(v) for v in (math.nan, math.inf, -math.inf, 1.5, 2)] == [
        "NaN",
        "Infinity",
        "-Infinity",
        1.5,
        2,
    ]
    obj["metrics"][0]["value"] = math.nan
    with pytest.raises(InvalidFieldError, match="finite"):
        from_dict(obj)


@pytest.mark.parametrize(
    "source",
    [
        "index",
        "index:pypi",
        "index:https://pypi.example.org/simple",
        "git+https://example.org/lab/ferric.git@" + FERRIC_SHA,
        "git+ssh://git@example.org/lab/ferric.git",
        "file",
        "conda",
        "conda:conda-forge",
        "system",
        None,
    ],
)
def test_dependency_source_vocabulary(source):
    dep = Dependency("ferric", "0.1.0", None, None, "inferred", None, source)
    assert loads(dumps(_record(dependencies=[dep]))).dependencies == [dep]


def test_data_may_be_named_without_a_digest():
    """MLflow logs datasets the producer cannot hash (an HTTP source, an
    in-memory frame). The core still says the run read them; null says how
    much is known (spec, Producer findings)."""
    entry = DataDigest(name="datasets/toy@ed51c932", sha256=None, role="training")
    assert loads(dumps(_record(data=[entry]))).data == [entry]


def test_format_timestamp_gives_the_one_accepted_spelling():
    from datetime import UTC, datetime, timedelta, timezone

    whole = format_timestamp(datetime(2026, 11, 19, 10, 0, tzinfo=UTC))
    assert whole == "2026-11-19T10:00:00.000000Z"
    when = datetime(2026, 11, 19, 10, 0, 0, 250000, tzinfo=timezone(timedelta(hours=-5)))
    assert format_timestamp(when) == "2026-11-19T15:00:00.250000Z"
    assert format_timestamp(datetime(2026, 1, 1, 0, 0, 0, 1, tzinfo=UTC)).endswith(".000001Z")
    assert format_timestamp(datetime(999, 1, 1, tzinfo=UTC)).startswith("0999-01-01T")
    # Text order is time order (smeltery, finding 2.10): ":37.5" sorted before
    # ":37" when the fraction had no fixed width.
    instants = [
        datetime(2026, 11, 19, 10, 0, 37, us, tzinfo=UTC) for us in (500000, 0, 999999, 1, 50)
    ] + [datetime(2026, 11, 19, 9, 59, 59, 999999, tzinfo=UTC)]
    assert sorted(map(format_timestamp, instants)) == list(map(format_timestamp, sorted(instants)))
    with pytest.raises(ValueError, match="aware"):
        format_timestamp(datetime(2026, 1, 1))
    # Whatever it produces, the reader accepts.
    dumps(_record(started_at=format_timestamp(when), finished_at=format_timestamp(when)))


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


# ---- history sidecar ----------------------------------------------------------


def _ref_for(run_dir: pathlib.Path, payload: bytes, series: int = 1, points: int = 1) -> HistoryRef:
    """Put `payload` in place as the sidecar and bind it, as a (buggy)
    producer would: the digest matches, so only the format can fail."""
    path = run_dir / HISTORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return HistoryRef(HISTORY_PATH, hashlib.sha256(payload).hexdigest(), series, points)


def test_history_round_trips_through_the_sidecar_in_logging_order(tmp_path):
    """A curve is a log: MLflow really does log two values at one step, and
    both are kept, in logging order (spec, Producer findings)."""
    eval_loss = Series("loss", "eval", {}, [Point(10, 0.5, None)])
    labelled = Series("loss", "train", {"seed": "1"}, [Point(0, 1.0, None)])
    ref = write_history(tmp_path, [LOSS, eval_loss, labelled])
    assert ref == HistoryRef(HISTORY_PATH, ref.sha256, series=3, points=5)
    assert ref.sha256 == hashlib.sha256((tmp_path / HISTORY_PATH).read_bytes()).hexdigest()
    write(_record(history=ref), tmp_path)
    assert read_history(tmp_path) == [LOSS, eval_loss, labelled]
    assert read_history(tmp_path, ref) == read_history(tmp_path, read(tmp_path))
    (series, *_) = read_history(tmp_path)
    assert [(p.step, p.value) for p in series.points] == [(0, 0.9), (0, 0.8), (1000, 0.31)]
    with pytest.raises(InvalidFieldError, match="duplicate series"):
        write_history(tmp_path / "dup", [LOSS, LOSS])
    with pytest.raises(InvalidFieldError, match="at least one point"):
        write_history(tmp_path / "empty", [Series("loss", None, {}, [])])
    assert not (tmp_path / "dup" / HISTORY_PATH).exists()


def test_no_history_is_null_not_an_empty_file(tmp_path):
    assert write_history(tmp_path, []) is None
    with HistoryWriter(tmp_path) as hist:
        hist.series("declared-but-never-logged")
    assert hist.ref is None
    assert not (tmp_path / HISTORY_PATH).exists()
    write(_record(history=None), tmp_path)
    assert read_history(tmp_path) is None


def test_streaming_writer_interleaves_series_and_finalizes_atomically(tmp_path):
    with HistoryWriter(tmp_path) as hist:
        never = hist.series("never")
        train = hist.series("loss", "train")
        val = hist.series("loss", "eval", {"model": "a"})
        for step in range(3):
            hist.point(train, step, 1.0 / (step + 1), format_timestamp_at(step))
            hist.flush()
            assert not (tmp_path / HISTORY_PATH).exists()  # finalized only on close
            hist.point(val, step, float("nan") if step == 2 else 0.5, None)
        with pytest.raises(InvalidFieldError, match="value"):
            hist.point(train, 3, True)
        with pytest.raises(InvalidFieldError, match="at"):
            hist.point(train, 3, 1.0, "2026-11-19T10:00:00.100Z")
        with pytest.raises(InvalidFieldError, match="step"):
            hist.point(train, -1, 1.0)
        with pytest.raises(InvalidFieldError, match="series"):
            hist.point(99, 0, 1.0)
        with pytest.raises(InvalidFieldError, match="duplicate series"):
            hist.series("loss", "train")
    assert never == 0
    assert hist.ref is not None and (hist.ref.series, hist.ref.points) == (2, 6)
    lines = (tmp_path / HISTORY_PATH).read_bytes().splitlines()
    assert lines[0] == b'{"schema":"attestation.history/1"}'
    assert lines[1] == b'{"labels":{},"name":"loss","series":0,"split":"train"}'
    assert lines[3] == b'{"labels":{"model":"a"},"name":"loss","series":1,"split":"eval"}'
    assert lines[-1] == b'{"at":null,"series":1,"step":2,"value":"NaN"}'
    train_s, val_s = read_history(tmp_path, hist.ref)
    assert [p.step for p in train_s.points] == [0, 1, 2]
    import math

    assert math.isnan(val_s.points[-1].value) and val_s.labels == {"model": "a"}
    assert sorted(p.name for p in (tmp_path / "attestation").iterdir()) == ["history.jsonl"]


def format_timestamp_at(second: int) -> str:
    from datetime import UTC, datetime

    return format_timestamp(datetime(2026, 11, 19, 10, 0, second, 500000, tzinfo=UTC))


def test_a_failed_producer_leaves_no_sidecar(tmp_path):
    with pytest.raises(RuntimeError):
        with HistoryWriter(tmp_path) as hist:
            hist.point(hist.series("loss"), 0, 1.0)
            raise RuntimeError("the run died")
    assert list((tmp_path / "attestation").iterdir()) == []
    hist = HistoryWriter(tmp_path)
    hist.point(hist.series("loss"), 0, 1.0)
    hist.abort()
    assert list((tmp_path / "attestation").iterdir()) == []


def test_history_is_written_before_the_record_never_after(tmp_path):
    ref = write_history(tmp_path, [LOSS])
    with pytest.raises(FileExistsError):
        write_history(tmp_path, [LOSS])
    assert write_history(tmp_path, [LOSS], overwrite=True) == ref
    write(_record(history=ref), tmp_path)
    with pytest.raises(layout.LayoutError, match="before the record"):
        HistoryWriter(tmp_path, overwrite=True)
    hist_dir = tmp_path / "late"
    hist = HistoryWriter(hist_dir)
    hist.point(hist.series("loss"), 0, 1.0)
    write(_record(history=None), hist_dir)
    with pytest.raises(layout.LayoutError, match="before the record"):
        hist.close()
    assert not (hist_dir / HISTORY_PATH).exists()


def test_write_refuses_a_record_whose_sidecar_is_missing_or_different(tmp_path):
    ref = HistoryRef(HISTORY_PATH, "2" * 64, 1, 3)
    with pytest.raises(HistoryMissingError):
        write(_record(history=ref), tmp_path)
    real = write_history(tmp_path, [LOSS])
    with pytest.raises(HistoryDigestError):
        write(_record(history=ref), tmp_path)
    assert not (tmp_path / layout.RUN_JSON).exists()
    write(_record(history=real), tmp_path)


def test_read_history_refuses_a_changed_or_missing_sidecar(tmp_path):
    write(_record(history=write_history(tmp_path, [LOSS])), tmp_path)
    sidecar = tmp_path / HISTORY_PATH
    good = sidecar.read_bytes()
    sidecar.write_bytes(good.replace(b"0.9", b"0.7"))
    with pytest.raises(HistoryDigestError, match="changed"):
        read_history(tmp_path)
    sidecar.write_bytes(good[:-1])  # truncated: the digest catches it first
    with pytest.raises(HistoryDigestError):
        read_history(tmp_path)
    sidecar.unlink()
    with pytest.raises(HistoryMissingError) as exc:
        read_history(tmp_path)
    assert isinstance(exc.value, RecordError) and exc.value.path == HISTORY_PATH


def test_loading_a_record_never_opens_the_sidecar(tmp_path, monkeypatch):
    record = _record(history=write_history(tmp_path, [LOSS]))
    write(record, tmp_path)
    opened: list[str] = []
    real_open = pathlib.Path.open

    def spy(self, *args, **kwargs):
        opened.append(self.name)
        return real_open(self, *args, **kwargs)

    (tmp_path / HISTORY_PATH).write_bytes(b"not even JSON")
    monkeypatch.setattr(pathlib.Path, "open", spy)
    assert read(tmp_path) == record
    assert loads((tmp_path / layout.RUN_JSON).read_bytes()) == record
    (tmp_path / HISTORY_PATH).unlink()
    assert read(tmp_path) == record
    assert "history.jsonl" not in opened and "run.json" in opened  # the spy saw the read


HEADER = b'{"schema":"attestation.history/1"}\n'
SERIES0 = b'{"labels":{},"name":"loss","series":0,"split":null}\n'
POINT0 = b'{"at":null,"series":0,"step":0,"value":1}\n'


@pytest.mark.parametrize(
    "payload, line, match",
    [
        (b"", None, "empty"),
        (HEADER + SERIES0 + POINT0[:-1], 3, "cut short"),
        (HEADER + SERIES0 + POINT0[:20] + b"\n", 3, "not JSON"),
        (b'{"schema":"attestation.history/2"}\n' + SERIES0 + POINT0, 1, "first line"),
        (SERIES0 + POINT0, 1, "first line"),
        (HEADER + POINT0, 2, "not declared"),
        (HEADER + SERIES0.replace(b":0,", b":1,") + POINT0, 2, "in order"),
        (HEADER + SERIES0, None, "no point"),
        (HEADER + SERIES0 + SERIES0.replace(b":0,", b":1,") + POINT0, 3, "no point"),
        (HEADER + SERIES0 + POINT0 + SERIES0.replace(b":0,", b":1,") + POINT0, 4, "duplicate"),
        (HEADER + SERIES0 + POINT0.replace(b"1}", b"NaN}"), 3, "NaN"),
        (HEADER + SERIES0 + POINT0.replace(b"1}", b'"nan"}'), 3, "value"),
        (HEADER + SERIES0 + POINT0.replace(b'"at":null,', b""), 3, "at"),
        (HEADER + SERIES0 + POINT0.replace(b"}", b',"x":1}'), 3, "x"),
        (HEADER + SERIES0 + b"[1]\n", 3, "object"),
        (HEADER + SERIES0 + b"\n", 3, "not JSON"),
        (HEADER + SERIES0 + POINT0 + b"\xff\n", 4, "UTF-8"),
        (HEADER + SERIES0 + POINT0 + POINT0, None, "says 1 / 1"),
    ],
)
def test_malformed_or_truncated_sidecar_lines_are_refused(tmp_path, payload, line, match):
    ref = _ref_for(tmp_path, payload)
    with pytest.raises(HistoryFormatError, match=match) as exc:
        read_history(tmp_path, ref)
    where = HISTORY_PATH if line is None else f"{HISTORY_PATH}:{line}"
    assert exc.value.path.startswith(where)


def test_the_malformed_cases_start_from_a_valid_sidecar(tmp_path):
    """The control for the parametrization above: unmutated, it reads."""
    ref = _ref_for(tmp_path, HEADER + SERIES0 + POINT0)
    assert read_history(tmp_path, ref) == [Series("loss", None, {}, [Point(0, 1, None)])]


def test_history_bytes_are_deterministic(tmp_path):
    a = write_history(tmp_path / "a", [LOSS])
    b = write_history(tmp_path / "b", [LOSS])
    assert a == b
    assert (tmp_path / "a" / HISTORY_PATH).read_bytes() == (
        tmp_path / "b" / HISTORY_PATH
    ).read_bytes()
    for raw in (tmp_path / "a" / HISTORY_PATH).read_bytes().splitlines(keepends=True):
        assert raw == canonicalize(json.loads(raw)) + b"\n"
    # Same points through the streaming writer: same bytes.
    with HistoryWriter(tmp_path / "c") as hist:
        h = hist.series(LOSS.name, LOSS.split, LOSS.labels)
        for p in LOSS.points:
            hist.point(h, p.step, p.value, p.at)
    assert hist.ref == a


def test_a_long_run_keeps_run_json_small(tmp_path):
    """100k points: the record is the same size as with ten, the sidecar
    holds the curve, and every point reads back."""
    n = 100_000
    with HistoryWriter(tmp_path) as hist:
        loss = hist.series("loss", "train")
        for step in range(n):
            hist.point(loss, step, 1.0 / (step + 1), None)
    record = _record(history=hist.ref)
    write(record, tmp_path)
    run_json = (tmp_path / layout.RUN_JSON).stat().st_size
    assert run_json < 2_000, run_json
    # Independent of the curve's size but for the digits of the count.
    assert run_json == len(dumps(_record(history=HistoryRef(HISTORY_PATH, "2" * 64, 1, n))))
    assert (tmp_path / HISTORY_PATH).stat().st_size > 3_000_000
    (curve,) = read_history(tmp_path)
    assert len(curve.points) == n and curve.points[-1] == Point(n - 1, 1.0 / n, None)


def test_the_manifest_does_not_list_the_sidecar(tmp_path):
    """history.jsonl lives under attestation/, so it is bound once -- by the
    record's history.sha256 -- and never also by digests.json."""
    (tmp_path / "out.txt").write_text("x")
    ref = write_history(tmp_path, [LOSS])
    entry = layout.write_manifest(tmp_path)
    assert list(json.loads((tmp_path / layout.DIGESTS_JSON).read_bytes())["files"]) == ["out.txt"]
    write(_record(history=ref, data=[entry]), tmp_path)
    assert layout.HISTORY_JSONL == HISTORY_PATH


def test_published_history_line_schema_agrees_with_the_writer(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(HISTORY_SCHEMA_FILE.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    write_history(tmp_path, [LOSS, Series("x", None, {"a": "b"}, [Point(1, float("inf"), None)])])
    lines = (tmp_path / HISTORY_PATH).read_bytes().splitlines()
    assert len(lines) == 1 + 2 + 4
    for raw in lines:
        assert validator.is_valid(json.loads(raw)), raw
    for bad in (
        {"schema": "attestation.history/2"},
        {"series": 0, "step": 0, "value": "nan", "at": None},
        {"series": 0, "step": -1, "value": 1, "at": None},
        {"series": 0, "step": 0, "value": 1},
        {"series": 0, "name": "", "split": None, "labels": {}},
        {"series": 0, "name": "x", "split": None, "labels": {"a": 1}},
        {"series": -1, "name": "x", "split": None, "labels": {}},
        {"series": 0, "step": 0, "value": 1, "at": None, "x": 1},
    ):
        assert not validator.is_valid(bad), bad


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
        ("finished_at", "'2026-11-19T09:00:00.000000Z'"),  # finished_at >= started_at
        ("code.repo", "'https://user:token@example.org/x.git'"),  # credentials
        ("code.repo", "'https://token@example.org/x.git'"),
        ("dependencies[0].source", "'git+https://user:pw@example.org/x.git'"),
        ("dependencies[0].source", "'git+https://token@example.org/x.git'"),
        ("dependencies[0].source", "'index:https://user:pw@pypi.example.org/simple'"),
        ("started_at", "'2026-13-19T10:00:00.000000Z'"),  # well-formed, impossible date
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
    write(_record(history=None), tmp_path / "a" / "run-1")
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
    write(_record(data=[entry], history=None), run)
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
    assert "attestation_schema/schemas/history-1.schema.json" in names
    assert "attestation_schema/py.typed" in names
    assert not any(n.startswith("attestation/") for n in names)
    assert b"Requires-Dist" not in metadata
