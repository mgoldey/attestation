"""Roadmap 12: the ledger's MLflow adapter reads the whole run.

Two producers of one mapping must agree: the adapter (`ledger_adapters/
mlflow_store.py`, reading `mlruns/` with no mlflow installed) and the
exporter (`examples/mlflow-records/export_records.py`, reading through
`MlflowClient`). The first test holds the adapter to the exporter's
committed records on every field the exporter sets; the live fresh-store
check is in tests/test_mlflow_records.py beside the store it reuses. The
synthetic stores below cover what neither real store has.

Spec: docs/superpowers/specs/2026-11-20-mlflow-adapter-design.md.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path

import attestation_schema as schema
import pytest

from attestation import ledger
from attestation.db import get_db
from attestation.ledger_adapters import generic, import_project, mlflow_store, trackers

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "examples" / "flows" / "training"
RECORDS = ROOT / "examples" / "mlflow-records" / "records"
EXPORTER = ROOT / "examples" / "mlflow-records" / "export_records.py"

# Fields the exporter sets that the adapter must reproduce. Not compared,
# each for a reason (spec, Testing): `producer` (each names itself), `ext`
# (the importer's namespace is ledger-import; ext.mlflow is the exporter's).
AGREED = (
    "schema",
    "run_id",
    "family",
    "arm",
    "status",
    "started_at",
    "finished_at",
    "code",
    "config",
    "data",
    "metrics",
    "history",
    "seeds",
    "dependencies",
)


def _exporter():
    name = "_mlflow_records_export"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, EXPORTER)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def declare(record: schema.RunRecord, declared) -> schema.RunRecord:
    """Apply the exporter's declarations (what neither the store nor the
    adapter knows) to the adapter's record, through the exporter's own code."""
    exp = _exporter()
    return replace(
        record,
        arm=exp._arm(record.config, declared.arm_params, record.run_id),
        seeds=exp._seeds(record.config, declared.seed_params, record.run_id),
        metrics=[replace(m, direction=declared.directions.get(m.name)) for m in record.metrics],
    )


def agree(ours: schema.RunRecord, theirs: schema.RunRecord) -> None:
    """Equal on every AGREED field, after the documented exclusions: the
    exporter's manifest entry (an import has no run directory) and the
    adapter's wall_time_s (the compute convention, not the exporter's).
    Compared in the wire form (`to_dict`), so a NaN equals a NaN."""
    a, b = ours.to_dict(), theirs.to_dict()
    b["data"] = [d for d in b["data"] if d["role"] != schema.layout.MANIFEST_ROLE]
    a["metrics"] = [m for m in a["metrics"] if m["name"] != "wall_time_s"]
    for key in AGREED:
        assert a[key] == b[key], f"{ours.run_id} {key}: adapter={a[key]!r} exporter={b[key]!r}"


# ---- the cross-check -------------------------------------------------------------


def test_the_adapter_agrees_with_the_exporters_committed_records():
    exp = _exporter()
    declared = exp.Declared(arm_params=("C",), seed_params=("seed",), directions={"auc": "max"})
    committed = {schema.read(d).run_id: schema.read(d) for d in schema.layout.find_runs(RECORDS)}
    result = import_project(TRAINING)
    assert result.refused == [] and len(result.imported) == len(committed) == 4
    for imported in result.imported:
        assert imported.adapter == "mlflow"
        theirs = committed[imported.record.run_id]
        agree(declare(imported.record, declared), theirs)
        # history equal as a reference means every point and timestamp agree
        # byte for byte; the curves themselves too, against the sidecar.
        found = next(
            f for f in trackers.MLFLOW.scan(TRAINING) if f.run_id == imported.record.run_id
        )
        run_dir = RECORDS / f"run-{theirs.run_id}"
        assert found.series == schema.read_history(run_dir, theirs)
        wall = next(m for m in imported.record.metrics if m.name == "wall_time_s")
        assert wall.unit == "s" and wall.value == pytest.approx(
            (found.end_ms - found.start_ms) / 1000
        )


def test_the_cross_check_can_fail():
    """A guard that cannot fail protects nothing (measurement-lessons 4):
    one changed value in the adapter's record must break agreement."""
    exp = _exporter()
    declared = exp.Declared(arm_params=("C",), seed_params=("seed",), directions={"auc": "max"})
    imported = import_project(TRAINING).imported[0]
    theirs = schema.read(RECORDS / f"run-{imported.record.run_id}")
    ours = declare(imported.record, declared)
    for mutant in (
        replace(ours, family="c_sweep"),  # the old reader's family
        replace(ours, metrics=[replace(m, step=0) for m in ours.metrics]),
        replace(ours, history=replace(ours.history, sha256="0" * 64)),
        replace(ours, status=None),
    ):
        with pytest.raises(AssertionError):
            agree(mutant, theirs)


def test_the_history_reference_names_the_bytes_the_schema_writer_writes(tmp_path):
    (found, *_) = trackers.MLFLOW.scan(TRAINING)
    record = trackers.MLFLOW.to_core_record(found)
    ref = schema.write_history(tmp_path, found.series)
    assert record.history == ref
    committed = RECORDS / f"run-{found.run_id}" / schema.HISTORY_PATH
    assert (tmp_path / schema.HISTORY_PATH).read_bytes() == committed.read_bytes()


def test_the_committed_sweep_is_one_family_named_for_its_experiment(tmp_path):
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, TRAINING.parent, project="training")
    rows = conn.execute("SELECT name, family, status FROM runs ORDER BY name").fetchall()
    assert {r["family"] for r in rows} == {"flows"}
    # The ledger's names (what FINDINGS.md's claims cite) are unchanged.
    assert [r["name"] for r in rows] == [
        "c_sweep/06f405c1",
        "c_sweep/55f000a4",
        "c_sweep/80a3f280",
        "c_sweep/bbb3a390",
    ]
    assert {r["status"] for r in rows} == {"recorded"}


# ---- synthetic stores: what neither real store has -----------------------------------


def _store(
    root: Path,
    run_id: str = "a" * 32,
    *,
    experiment: str | None = "sweep",
    exp_id: str = "7",
    run_name: str | None = "arm",
    status: int | str = 3,
    metrics: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    tags: dict[str, str] | None = None,
    artifact_uri: str | None = None,
    start: int = 1_755_000_000_000,
    end: int = 1_755_000_090_000,
) -> Path:
    exp_dir = root / "mlruns" / exp_id
    run = exp_dir / run_id
    run.mkdir(parents=True)
    if experiment is not None:
        (exp_dir / "meta.yaml").write_text(
            f"experiment_id: '{exp_id}'\nlifecycle_stage: active\nname: {experiment}\n"
        )
    uri = artifact_uri or f"file:///elsewhere/mlruns/{exp_id}/{run_id}/artifacts"
    (run / "meta.yaml").write_text(
        f"artifact_uri: {uri}\nend_time: {end}\nexperiment_id: '{exp_id}'\n"
        f"lifecycle_stage: active\nrun_id: {run_id}\n"
        + (f"run_name: {run_name}\n" if run_name else "")
        + f"start_time: {start}\nstatus: {status}\n"
    )
    for folder, files in (
        ("metrics", metrics or {"auc": "1755000001000 0.5 0\n"}),
        ("params", params or {}),
        ("tags", tags or {}),
    ):
        for key, text in files.items():
            path = run / folder / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    return run


def _record(root: Path) -> schema.RunRecord:
    result = import_project(root)
    assert result.refused == [], result.refused
    (imported,) = result.imported
    return imported.record


def test_family_is_the_experiment_and_a_missing_experiment_says_nothing(tmp_path):
    _store(tmp_path / "a")
    assert _record(tmp_path / "a").family == "sweep"
    _store(tmp_path / "b", experiment=None)
    assert _record(tmp_path / "b").family is None


def test_the_final_value_is_mlflows_latest_point_not_the_last_line(tmp_path):
    """Step 5 logged before step 2: MLflow's final value is step 5's (the old
    reader took the last line, step 2's)."""
    _store(tmp_path, metrics={"loss": "1000 0.9 5\n2000 0.1 2\n"})
    (loss,) = [m for m in _record(tmp_path).metrics if m.name == "loss"]
    assert (loss.value, loss.step) == (0.9, 5)
    # Same step, same millisecond: the larger value (MLflow's tiebreak).
    _store(tmp_path / "tie", metrics={"x": "1000 0.2 1\n1000 0.7 1\n1000 0.4 1\n"})
    (x,) = [m for m in _record(tmp_path / "tie").metrics if m.name == "x"]
    assert x.value == 0.7


def test_a_dataset_linked_point_keeps_the_step_it_was_logged_at(tmp_path):
    """MLflow 3.15.2's client returns step 0 for a 5-field line; the file
    holds the right step, and the adapter reads it (spec, the one divergence)."""
    _store(tmp_path, metrics={"eval/acc": "1000 0.5 3 data 2feca68d\n2000 0.7 5 data 2feca68d\n"})
    rec = _record(tmp_path)
    (acc,) = [m for m in rec.metrics if m.name == "eval/acc"]
    assert (acc.value, acc.step, acc.split) == (0.7, 5, "data")
    (found,) = trackers.MLFLOW.scan(tmp_path)
    (series,) = [s for s in found.series if s.name == "eval/acc"]
    assert [p.step for p in series.points] == [3, 5]


def test_status_maps_and_a_run_in_flight_is_not_imported(tmp_path):
    _store(tmp_path, "f" * 32, run_name="failed", status=4)
    _store(tmp_path, "k" * 32, run_name="killed", status="KILLED")
    _store(tmp_path, "r" * 32, run_name="running", status=1)
    by_name = {i.name.split("/")[0]: i.record for i in import_project(tmp_path).imported}
    assert {k: r.status for k, r in by_name.items()} == {
        "failed": "failed",
        "killed": "cancelled",
    }
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path.parent, project=tmp_path.name)
    stored = conn.execute("SELECT caveats_json FROM runs WHERE name LIKE 'failed/%'").fetchone()
    assert "has status failed" in stored[0] and "did not complete" in stored[0]


@pytest.mark.parametrize(
    "metrics",
    [{"wer": "not a metric line\n"}, {"wer": ""}, {"wer": "1000 abc 0\n"}],
    ids=["garbage", "empty", "bad-value"],
)
def test_a_run_mlflow_cannot_read_is_quarantined_not_half_read(tmp_path, metrics):
    _store(tmp_path, metrics={"auc": "1000 0.5 0\n", **metrics})
    result = import_project(tmp_path)
    assert result.imported == []
    (q,) = result.quarantined
    assert q.adapter == "mlflow" and "metrics/wer" in q.reasons[0]
    generic.discover(tmp_path)  # the legacy view still never raises


def test_a_meta_yaml_naming_another_run_is_quarantined(tmp_path):
    run = _store(tmp_path)
    meta = run / "meta.yaml"
    meta.write_text(meta.read_text().replace(f"run_id: {run.name}", "run_id: other"))
    assert [q.adapter for q in import_project(tmp_path).quarantined] == ["mlflow"]


def test_params_are_verbatim_and_nested_keys_are_read(tmp_path):
    _store(tmp_path, params={"lr": "0.0003", "model/depth": "12", "note": " spaced \n"})
    assert _record(tmp_path).config == {"lr": "0.0003", "model/depth": "12", "note": " spaced \n"}


def test_git_tags_become_inferred_code_with_credentials_stripped(tmp_path):
    _store(
        tmp_path,
        tags={
            "mlflow.source.git.commit": "ab" * 20,
            "mlflow.source.git.repoURL": "https://ci:t0ken@example.org/lab/x.git",
        },
    )
    code = _record(tmp_path).code
    assert (code.git_sha, code.provenance, code.dirty) == ("ab" * 20, "inferred", None)
    assert code.repo == "https://example.org/lab/x.git" and "HEAD" in code.evidence
    _store(tmp_path / "none")
    assert _record(tmp_path / "none").code.provenance == "unknown"


def test_the_run_name_falls_back_to_its_tag(tmp_path):
    _store(tmp_path, run_name=None, tags={"mlflow.runName": "tagged"})
    assert import_project(tmp_path).imported[0].name == "tagged/" + "a" * 8


def test_artifacts_are_hashed_in_the_store_wherever_it_now_lives(tmp_path):
    """The URI names another machine's path; the store's own copy is read."""
    run = _store(tmp_path)
    (run / "artifacts" / "model").mkdir(parents=True)
    (run / "artifacts" / "model" / "w.bin").write_bytes(b"\x00\x01weights")
    (run / "artifacts" / "a.txt").write_bytes(b"hello")
    data = _record(tmp_path).data
    assert [(d.name, d.role) for d in data] == [
        ("artifacts/a.txt", "output"),
        ("artifacts/model/w.bin", "output"),
    ]
    assert data[1].sha256 == hashlib.sha256(b"\x00\x01weights").hexdigest()


def test_an_artifact_location_that_cannot_be_read_is_a_caveat_not_a_guess(tmp_path):
    _store(tmp_path, artifact_uri="s3://bucket/somewhere")
    rec = _record(tmp_path)
    assert rec.data == []
    (caveat,) = rec.ext["ledger-import"]["run_caveats"]
    assert "s3://bucket/somewhere" in caveat and "not read" in caveat


def test_dataset_inputs_are_named_with_their_context_and_no_digest(tmp_path):
    run = _store(tmp_path)
    exp = run.parent
    for ds_id, digest, context in (("d1", "9128167c", "training"), ("d2", "2feca68d", None)):
        (exp / "datasets" / ds_id).mkdir(parents=True)
        (exp / "datasets" / ds_id / "meta.yaml").write_text(
            f"digest: {digest}\nname: data\nprofile: null\nschema: null\n"
            'source: \'{"url": "https://example.org/x.csv"}\'\nsource_type: http\n'
        )
        link = run / "inputs" / f"in-{ds_id}"
        link.mkdir(parents=True)
        tags = f"tags:\n  mlflow.data.context: {context}\n" if context else "tags: {}\n"
        (link / "meta.yaml").write_text(
            f"destination_id: {ds_id}\ndestination_type: RUN\nsource_id: {ds_id}\n"
            f"source_type: DATASET\n{tags}"
        )
    data = _record(tmp_path).data
    assert [(d.name, d.sha256, d.role) for d in data] == [
        ("datasets/data@2feca68d", None, "input"),
        ("datasets/data@9128167c", None, "training"),
    ]


def test_a_diverged_final_value_is_kept_in_the_record_and_caveated_in_the_ledger(tmp_path):
    _store(tmp_path, metrics={"loss": "1000 0.5 0\n2000 nan 1\n", "auc": "1000 0.9 0\n"})
    (loss,) = [m for m in _record(tmp_path).metrics if m.name == "loss"]
    assert math.isnan(loss.value) and loss.step == 1
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path.parent, project=tmp_path.name)
    stored = {r["metric"] for r in conn.execute("SELECT metric FROM run_metrics")}
    assert "loss" not in stored and "auc" in stored
    (caveats,) = conn.execute("SELECT caveats_json FROM runs").fetchone()
    assert "loss is NaN" in caveats and "not stored" in caveats


def test_deleted_runs_and_trash_are_not_read(tmp_path):
    run = _store(tmp_path)
    meta = run / "meta.yaml"
    meta.write_text(meta.read_text().replace("lifecycle_stage: active", "lifecycle_stage: deleted"))
    _store(tmp_path, "b" * 32, exp_id=".trash")
    assert import_project(tmp_path).imported == []


def test_the_reader_names_no_tracker_outside_the_adapters_package():
    """The module lives where tracker names may (tests/test_tracker_names.py)."""
    assert Path(mlflow_store.__file__).parent.name == "ledger_adapters"


def test_curve_points_are_in_logging_order_by_timestamp(tmp_path):
    """The exporter orders a series by timestamp, stably (points logged in
    one millisecond keep file order); a file written out of time order
    (log_batch with explicit timestamps) is reordered, not copied."""
    _store(tmp_path, metrics={"loss": "3000 0.3 2\n1000 0.9 0\n1000 0.8 1\n"})
    (found,) = trackers.MLFLOW.scan(tmp_path)
    (loss,) = found.series
    assert [(p.step, p.value) for p in loss.points] == [(0, 0.9), (1, 0.8), (2, 0.3)]
