"""Roadmap 3: a real MLflow sweep written as core run records, no special casing.

The producer is examples/mlflow-records/export_records.py; the records it
wrote from the committed sweep (examples/flows/training/mlruns) are committed
in examples/mlflow-records/records/. The first group of tests reads those
records with attestation_schema alone and checks them against the raw MLflow
files, so it runs without mlflow installed. The second group needs mlflow:
it re-exports the committed sweep and asserts the bytes are the committed
ones, then drives a fresh, real MLflow store through everything the
committed sweep does not exercise (NaN, a repeated step, dataset inputs,
artifacts, git identity, failed/killed/running runs).

Spec: docs/superpowers/specs/2026-11-19-core-run-record-design.md, "Producer
findings".
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest
from attestation_schema import HISTORY_PATH, dumps, layout, loads, read, read_history
from attestation_schema.conformance import check_run_dir, find_targets

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "mlflow-records"
RECORDS = EXAMPLE / "records"
MLRUNS = ROOT / "examples" / "flows" / "training" / "mlruns"
DECLARED_ARGS = ["--arm-param", "C", "--seed-param", "seed", "--direction", "auc=max"]


def _producer():
    """The example script, loaded once and registered (its dataclasses need
    their module in sys.modules to resolve string annotations)."""
    name = "_mlflow_records_export"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, EXAMPLE / "export_records.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _mlflow_run_dirs() -> dict[str, Path]:
    return {p.name: p for p in MLRUNS.glob("*/*") if (p / "meta.yaml").is_file()}


def _meta(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition(": ")
        if sep and not line.startswith(" "):
            out[key] = value.strip("'")
    return out


def _logged(metric_file: Path) -> list[tuple[int, float, int]]:
    """MLflow's own lines: `<timestamp_ms> <value> <step>`, in file order."""
    rows = []
    for line in metric_file.read_text().splitlines():
        ts, value, step = line.split()[:3]
        rows.append((int(ts), float(value), int(step)))
    return rows


# ---- the committed records (no mlflow needed) -------------------------------


def test_one_record_per_committed_mlflow_run():
    runs = layout.find_runs(RECORDS)
    assert len(runs) == 4
    assert {read(r).run_id for r in runs} == set(_mlflow_run_dirs())
    assert all(r.name == f"run-{read(r).run_id}" for r in runs)


@pytest.mark.parametrize("run_dir", layout.find_runs(RECORDS), ids=lambda p: p.name[:12])
def test_committed_record_validates_and_round_trips_byte_for_byte(run_dir):
    raw = (run_dir / layout.RUN_JSON).read_bytes()
    record = loads(raw)
    assert dumps(record) == raw
    # The manifest is the layout's, bound to the record by its digest.
    manifest_bytes = (run_dir / layout.DIGESTS_JSON).read_bytes()
    (entry,) = [d for d in record.data if d.role == layout.MANIFEST_ROLE]
    assert entry.name == layout.DIGESTS_JSON
    assert entry.sha256 == hashlib.sha256(manifest_bytes).hexdigest()
    assert json.loads(manifest_bytes) == layout.build_manifest(run_dir)
    # The curves are in the sidecar, bound by digest; run.json holds no point.
    assert record.history is not None and record.history.path == HISTORY_PATH
    sidecar = (run_dir / HISTORY_PATH).read_bytes()
    assert record.history.sha256 == hashlib.sha256(sidecar).hexdigest()
    assert json.loads(raw)["history"] == {
        "path": HISTORY_PATH,
        "sha256": record.history.sha256,
        "series": 5,
        "points": 14,
    }
    assert sidecar.count(b"\n") == 1 + 5 + 14  # header, one line per series and per point


@pytest.mark.parametrize("run_dir", layout.find_runs(RECORDS), ids=lambda p: p.name[:12])
def test_committed_record_preserves_mlflow_history_and_says_only_what_mlflow_said(run_dir):
    record = read(run_dir)
    history = read_history(run_dir, record)
    assert history is not None
    mlflow_dir = _mlflow_run_dirs()[record.run_id]
    experiment = _meta(mlflow_dir.parent / "meta.yaml")
    meta = _meta(mlflow_dir / "meta.yaml")

    assert record.family == experiment["name"]
    assert record.ext["mlflow"]["experiment_id"] == experiment["experiment_id"]
    assert record.ext["mlflow"]["run_name"] == meta["run_name"]
    params = {p.name: p.read_text() for p in (mlflow_dir / "params").iterdir()}
    assert record.config == params  # strings, as MLflow stored them
    assert record.arm == f"C={params['C']}"  # declared, not inferred
    assert record.seeds == {"seed": int(params["seed"])}
    assert record.status == "complete" and meta["status"] == "3"  # RunStatus.FINISHED

    # Every logged point, in order, with its own step, value and timestamp.
    files = {f.name: f for f in (mlflow_dir / "metrics").iterdir()}
    assert {s.name for s in history} == set(files)
    for series in history:
        logged = _logged(files[series.name])
        assert [(p.step, p.value) for p in series.points] == [(s, v) for _, v, s in logged]
        assert [p.at for p in series.points] == [
            _producer().timestamp_ms(ts) for ts, _, _ in logged
        ]
        assert series.split is None
    assert len(next(s for s in history if s.name == "train_loss").points) == 10

    # Final values: the latest point; a direction only where one was declared.
    for metric in record.metrics:
        ts, value, step = max(_logged(files[metric.name]), key=lambda r: (r[2], r[0]))
        assert (metric.value, metric.step) == (value, step)
        assert metric.direction == ("max" if metric.name == "auc" else None)

    # The committed sweep's git tags were scrubbed before commit: unknown, not clean.
    assert (record.code.git_sha, record.code.dirty) == (None, None)
    assert (record.code.provenance, record.code.version) == ("unknown", None)
    assert all(m.unit is None and m.labels == {} for m in record.metrics)
    # MLflow recorded no environment, so the exporter says it could not observe one.
    assert record.dependencies is None


def test_history_is_what_the_ledger_tables_discard():
    """The ledger's tables keep one final value per metric; the core record
    keeps every point (and since roadmap 12 the ledger's MLflow adapter
    builds the same history reference, tests/test_mlflow_adapter.py). Pin
    the difference on the one multi-line file so a future importer cannot
    quietly drop the curve."""
    for run_dir in layout.find_runs(RECORDS):
        record = read(run_dir)
        loss = next(s for s in read_history(run_dir, record) if s.name == "train_loss")
        final = next(m for m in record.metrics if m.name == "train_loss")
        assert (final.step, final.value) == (loss.points[-1].step, loss.points[-1].value)
        assert len({p.value for p in loss.points}) == 10


# ---- re-export through the real MLflow client --------------------------------


def _store_copy(tmp_path: Path) -> Path:
    """The committed store, with the empty artifacts/ dirs git cannot hold."""
    store = tmp_path / "mlruns"
    shutil.copytree(MLRUNS, store)
    for run in store.glob("*/*"):
        if (run / "meta.yaml").is_file():
            (run / "artifacts").mkdir(exist_ok=True)
    return store


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_re_export_of_the_committed_sweep_is_byte_identical(tmp_path):
    pytest.importorskip("mlflow")
    producer = _producer()
    store = _store_copy(tmp_path)
    for out in ("a", "b"):
        assert (
            producer.main(
                ["--tracking-uri", str(store), "--out", str(tmp_path / out)] + DECLARED_ARGS
            )
            == 0
        )
    assert _tree(tmp_path / "a") == _tree(tmp_path / "b") == _tree(RECORDS)


def test_export_never_overwrites_an_existing_record(tmp_path):
    pytest.importorskip("mlflow")
    producer = _producer()
    store = _store_copy(tmp_path)
    producer.main(["--tracking-uri", str(store), "--out", str(tmp_path / "out")])
    with pytest.raises(layout.LayoutError, match="before the record"):
        producer.main(["--tracking-uri", str(store), "--out", str(tmp_path / "out")])


def test_undeclarable_declarations_are_refused(tmp_path):
    pytest.importorskip("mlflow")
    producer = _producer()
    store = _store_copy(tmp_path)
    with pytest.raises(ValueError, match="not among its params"):
        producer.main(
            ["--tracking-uri", str(store), "--out", str(tmp_path / "x"), "--arm-param", "lr"]
        )
    with pytest.raises(ValueError, match="not an integer seed"):
        producer.main(
            ["--tracking-uri", str(store), "--out", str(tmp_path / "y"), "--seed-param", "C"]
        )


_FRESH_RUNS = textwrap.dedent(
    """
    import os, sys
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    import mlflow
    from mlflow.data.http_dataset_source import HTTPDatasetSource
    from mlflow.data.meta_dataset import MetaDataset
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("fresh")
    train = MetaDataset(source=HTTPDatasetSource("https://example.org/train.csv"), name="data")
    test = MetaDataset(source=HTTPDatasetSource("https://example.org/test.csv"), name="data")
    with mlflow.start_run(run_name="diverged") as run:
        mlflow.log_params({"lr": 10.0, "seed": 7})
        mlflow.log_input(train, context="training")
        mlflow.log_input(test, context="eval")
        for step, loss in enumerate([1.0, 4.0, float("inf"), float("nan")]):
            mlflow.log_metric("loss", loss, step=step)
        mlflow.log_metric("lr_scale", 1.0, step=0)
        mlflow.log_metric("lr_scale", 0.5, step=0)  # a second value at the same step
        with open("weights.bin", "wb") as fh:
            fh.write(b"\\x00\\x01weights")
        mlflow.log_artifact("weights.bin", artifact_path="model")
    with mlflow.start_run(run_name="stopped"):
        mlflow.log_metric("loss", 0.5)
        # MLflow strips credentials from the remote it detects (3.x), but a
        # tag set by hand -- how CI systems pass the remote -- is stored as is.
        mlflow.set_tag("mlflow.source.git.repoURL", "https://ci:t0ken@example.org/lab/code.git")
    mlflow.MlflowClient().set_terminated(mlflow.last_active_run().info.run_id, "KILLED")
    client = MlflowClient()
    pending = client.create_run(mlflow.get_experiment_by_name("fresh").experiment_id)
    print(run.info.run_id, pending.info.run_id)
    """
)


@pytest.fixture(scope="module")
def fresh_store(tmp_path_factory):
    """A real MLflow store, written by MLflow from a script inside a git
    checkout whose remote URL carries credentials -- MLflow records both."""
    pytest.importorskip("mlflow")
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    work = tmp_path_factory.mktemp("fresh")
    git = ["git", "-c", "user.email=ci@example.org", "-c", "user.name=ci"]
    subprocess.run([*git, "init", "-q", "."], cwd=work, check=True)
    subprocess.run(
        [*git, "remote", "add", "origin", "https://someone:s3cret@example.org/lab/code.git"],
        cwd=work,
        check=True,
    )
    (work / "train.py").write_text(_FRESH_RUNS)
    subprocess.run([*git, "add", "train.py"], cwd=work, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "train"], cwd=work, check=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=work, check=True, capture_output=True, text=True
    ).stdout.strip()
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_")}
    # MLflow's own record of a run, as an MLflow user without attestation-schema
    # has it: attestation-schema's run-context provider (registered in this venv
    # by its entry point) is turned off. Its tags are tested with it on, in
    # packages/attestation-schema/tests/test_integrations_mlflow.py.
    env["ATTESTATION_MLFLOW_TAGS"] = "0"
    proc = subprocess.run(
        [sys.executable, "train.py"], cwd=work, env=env, capture_output=True, text=True, check=True
    )
    diverged, pending = proc.stdout.split()
    return work, sha, diverged, pending


def _export_fresh(tmp_path, fresh_store):
    work = fresh_store[0]
    producer = _producer()
    written, skipped = producer.export(producer._uri(str(work / "mlruns")), tmp_path / "out")
    return {read(d).ext["mlflow"]["run_name"]: read(d) for d in written}, skipped


def test_a_fresh_real_store_maps_without_workarounds(tmp_path, fresh_store):
    work, sha, diverged, pending = fresh_store
    records, skipped = _export_fresh(tmp_path, fresh_store)

    # A run that has not ended gets no record; a killed one is cancelled.
    assert skipped == [pending]
    assert set(records) == {"diverged", "stopped"}
    assert records["stopped"].status == "cancelled"
    rec = records["diverged"]
    assert rec.run_id == diverged and rec.status == "complete"

    # Code identity from MLflow's own git tags; credentials stripped; dirty unknown.
    assert rec.code.git_sha == sha and rec.code.dirty is None
    # MLflow read HEAD at run start; it cannot know the tree was clean.
    assert rec.code.provenance == "inferred" and "HEAD" in rec.code.evidence
    assert rec.code.repo == "https://example.org/lab/code.git"
    assert "s3cret" not in (tmp_path / "out" / f"run-{diverged}" / layout.RUN_JSON).read_text()
    # A hand-set tag keeps its token in MLflow; the record carries neither copy.
    stopped = records["stopped"]
    assert stopped.code.repo == "https://example.org/lab/code.git"
    assert stopped.ext["mlflow"]["tags"]["mlflow.source.git.repoURL"] == stopped.code.repo
    stopped_json = tmp_path / "out" / f"run-{stopped.run_id}" / layout.RUN_JSON
    assert "t0ken" not in stopped_json.read_text()

    # NaN and infinity are results, kept in history and as the final value.
    run_dir = tmp_path / "out" / f"run-{diverged}"
    curves = read_history(run_dir, rec)
    assert curves is not None
    loss = next(s for s in curves if s.name == "loss")
    assert [p.step for p in loss.points] == [0, 1, 2, 3]
    assert loss.points[2].value == math.inf and math.isnan(loss.points[3].value)
    final = next(m for m in rec.metrics if m.name == "loss")
    assert final.step == 3 and math.isnan(final.value)
    # Two values at one step: both kept, in order. The final one is MLflow's:
    # the later, unless both landed in one millisecond, when MLflow takes the
    # larger (test_latest_point_matches_mlflows_own_latest_value checks it
    # against MLflow itself).
    scale = next(s for s in curves if s.name == "lr_scale")
    assert [(p.step, p.value) for p in scale.points] == [(0, 1.0), (0, 0.5)]
    same_ms = scale.points[0].at == scale.points[1].at
    assert next(m for m in rec.metrics if m.name == "lr_scale").value == (1.0 if same_ms else 0.5)

    # Two datasets MLflow names identically, told apart by MLflow's digest;
    # named in the core, undigested, with MLflow's context as the role.
    datasets = [d for d in rec.data if d.name.startswith("datasets/")]
    assert {d.role for d in datasets} == {"training", "eval"}
    assert all(d.sha256 is None for d in datasets) and len({d.name for d in datasets}) == 2
    assert len(rec.ext["mlflow"]["datasets"]) == 2

    # Artifacts by the sha256 of the bytes MLflow stored.
    (artifact,) = [d for d in rec.data if d.name.startswith("artifacts/")]
    assert artifact.name == "artifacts/model/weights.bin" and artifact.role == "output"
    assert artifact.sha256 == hashlib.sha256(b"\x00\x01weights").hexdigest()

    # On the wire (the sidecar's point lines), the non-finite values are the
    # schema's spellings; the record itself carries no point and no mlflow
    # dependency it could not observe.
    lines = [json.loads(x) for x in (run_dir / HISTORY_PATH).read_bytes().splitlines()]
    loss_id = next(x["series"] for x in lines if x.get("name") == "loss")
    values = [x["value"] for x in lines if "step" in x and x["series"] == loss_id]
    assert values[2:] == ["Infinity", "NaN"]
    assert rec.dependencies is None and rec.history is not None


def test_a_fresh_real_store_exports_deterministically(tmp_path, fresh_store):
    work, *_ = fresh_store
    producer = _producer()
    for out in ("a", "b"):
        producer.export(producer._uri(str(work / "mlruns")), tmp_path / out)
    assert _tree(tmp_path / "a") == _tree(tmp_path / "b")


def test_latest_point_matches_mlflows_own_latest_value(tmp_path, fresh_store):
    """`metrics` is MLflow's notion of the final value, not a reimplementation
    that happens to agree on a ten-line fixture."""
    from mlflow.tracking import MlflowClient

    work, *_ = fresh_store
    records, _ = _export_fresh(tmp_path, fresh_store)
    client = MlflowClient(_producer()._uri(str(work / "mlruns")))
    for rec in records.values():
        theirs = client.get_run(rec.run_id).data.metrics
        ours = {m.name: m.value for m in rec.metrics}
        assert set(ours) == set(theirs)
        for name, value in theirs.items():
            assert (math.isnan(value) and math.isnan(ours[name])) or ours[name] == value


def test_the_ledger_adapter_agrees_with_the_exporter_on_a_fresh_real_store(tmp_path, fresh_store):
    """Roadmap 12: two producers of one mapping. The ledger's adapter reads
    the same store from its files, with no mlflow; on every field the
    exporter sets (no declarations here) the two records are equal -- the
    datasets, the artifact digest, git identity, a stripped token, a killed
    run, inf/NaN and two values at one step included. The comparison and its
    documented exclusions are tests/test_mlflow_adapter.py's `agree`."""
    from test_mlflow_adapter import agree

    from attestation.ledger_adapters import import_project

    work, _sha, diverged, pending = fresh_store
    records, _ = _export_fresh(tmp_path, fresh_store)
    result = import_project(work)
    assert result.refused == []
    ours = {i.record.run_id: i.record for i in result.imported if i.adapter == "mlflow"}
    assert set(ours) == {r.run_id for r in records.values()}
    assert pending not in ours  # running: no record from either producer
    for theirs in records.values():
        agree(ours[theirs.run_id], theirs)
    assert any(d.name.startswith("artifacts/") for d in ours[diverged].data)


# ---- pure helpers -----------------------------------------------------------


def test_timestamps_and_credentials():
    producer = _producer()
    assert producer.timestamp_ms(1787956426024) == "2026-08-28T22:33:46.024000Z"
    assert producer.timestamp_ms(1787956426000) == "2026-08-28T22:33:46.000000Z"
    assert producer.timestamp_ms(1787956425990) == "2026-08-28T22:33:45.990000Z"
    assert producer.timestamp_ms(None) is None
    strip = producer.strip_credentials
    assert strip("https://u:p@example.org:8443/x.git") == "https://example.org:8443/x.git"
    assert strip("https://token@example.org/x.git") == "https://example.org/x.git"
    assert strip("git@example.org:lab/x.git") == "git@example.org:lab/x.git"
    assert strip("") is None


# ---- the conformance suite runs over these records (roadmap 15, Decision 5) ----
# A deleted ini line or CI step would stop the suite with nothing going red,
# so the wiring itself is asserted. Spec:
# docs/superpowers/specs/2026-11-19-conformance-suite-design.md.
CONFORMANCE_RECORDS = "examples/mlflow-records/records"


def test_the_plugin_is_loaded_in_this_session(pytestconfig):
    assert pytestconfig.pluginmanager.has_plugin("attestation_schema.pytest_plugin")


def test_pyproject_loads_the_plugin_and_lists_the_records():
    ini = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    assert "-p attestation_schema.pytest_plugin" in ini["addopts"]
    assert ini["attestation_runs"] == [CONFORMANCE_RECORDS]


def test_the_listed_records_exist_and_conform():
    """The ini list is not vacuous: it names four records, all conforming."""
    dirs, empty = find_targets([CONFORMANCE_RECORDS], ROOT)
    assert empty == [] and len(dirs) == 4
    for d in dirs:
        report = check_run_dir(d)
        assert report.ok, report.format()


def test_ci_runs_the_plugin_from_the_installed_wheel_on_311():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    job = ci.split("schema-py311:", 1)[1].split("\n  flows:", 1)[0]
    assert "[conformance]" in job
    assert "-p attestation_schema.pytest_plugin" in job
    assert f"--attestation-runs '{CONFORMANCE_RECORDS}/*'" in job
