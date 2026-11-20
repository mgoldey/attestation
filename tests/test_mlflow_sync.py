"""`attest mlflow sync`: an MLflow file store written down as core records.

Against the committed real sweep (`examples/flows/training/mlruns`, four
runs MLflow wrote) with no mlflow needed, plus copies of it edited the way
a store changes: a late tag, a new metric point, a malformed line, a run
still in flight. One test writes a fresh store through mlflow itself (with
the run-context provider on) and skips without it. Spec:
docs/superpowers/specs/2026-11-20-mlflow-plugin-design.md.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import attestation_schema as schema
import pytest
from attestation_schema import conformance
from attestation_schema.integrations import mlflow_mapping as mapping

import attestation
from attestation.cli import main
from attestation.ledger import row_from_core
from attestation.ledger_adapters import import_project, mlflow_sync, trackers

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "examples" / "flows" / "training" / "mlruns"
EXPERIMENT = "260449967480221589"
RUN = "06f405c125804496876ebfa098ad4c94"


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project directory holding a copy of the committed store."""
    shutil.copytree(STORE, tmp_path / "proj" / "mlruns")
    return tmp_path / "proj"


def _run_dir(project: Path, run: str = RUN) -> Path:
    return project / "mlruns" / EXPERIMENT / run


def _sync(project: Path, *extra: str) -> int:
    return main(
        [
            "mlflow",
            "sync",
            "--tracking-dir",
            str(project / "mlruns"),
            "--out",
            str(project / "mlflow-records"),
            *extra,
        ]
    )


# ---- the committed sweep ----------------------------------------------------------


def test_every_ended_run_becomes_a_conforming_record(project, capsys):
    assert _sync(project) == 0
    out = capsys.readouterr().out
    assert "4 written; 0 skipped" in out
    dirs = schema.layout.find_runs(project / "mlflow-records")
    assert len(dirs) == 4
    adapter = {f.run_id: trackers.MLFLOW.to_core_record(f) for f in trackers.MLFLOW.scan(project)}
    for d in dirs:
        report = conformance.check_run_dir(d)
        assert report.ok, report.violations
        rec = schema.read(d)
        assert d.name == f"run-{rec.run_id}"
        assert rec.producer == schema.Producer("attestation mlflow-sync", attestation.__version__)
        assert set(rec.ext) == {"mlflow-sync"}
        assert rec.ext["mlflow-sync"] == {
            "experiment_id": EXPERIMENT,
            "run_name": "c_sweep",
            "notes": [],
        }
        # The adapter's record, field for field, but for who wrote it and the
        # run directory's own manifest.
        ours, theirs = rec.to_dict(), adapter[rec.run_id].to_dict()
        assert ours["data"][-1]["role"] == schema.layout.MANIFEST_ROLE
        ours["data"] = ours["data"][:-1]
        for key in set(ours) - {"producer", "ext"}:
            assert ours[key] == theirs[key], key
        # The curves are written, and are the store's.
        assert schema.read_history(d, rec) is not None
        assert rec.family == "flows" and rec.status == "complete"


def test_re_sync_is_byte_identical_and_writes_nothing(project, capsys):
    assert _sync(project) == 0
    before = _tree(project / "mlflow-records")
    capsys.readouterr()
    assert _sync(project) == 0
    assert "4 unchanged" in capsys.readouterr().out
    assert _tree(project / "mlflow-records") == before


def test_a_changed_run_is_refused_unless_forced(project, capsys):
    assert _sync(project) == 0
    target = project / "mlflow-records" / f"run-{RUN}"
    before = _tree(target)
    # A metric point logged after the sync: the curve and the final value move.
    with (_run_dir(project) / "metrics" / "auc").open("a") as fh:
        fh.write("1787956426999 0.5 7\n")
    capsys.readouterr()
    assert _sync(project) == 1
    out = capsys.readouterr().out
    assert f"refused       {RUN}" in out
    assert "history, metrics differ" in out and "--force replaces it" in out
    assert _tree(target) == before  # never silently overwritten
    assert not [p for p in (project / "mlflow-records").iterdir() if p.name.startswith(".")]

    assert _sync(project, "--force") == 0
    assert f"replaced      {RUN}" in capsys.readouterr().out
    rec = schema.read(target)
    assert next(m for m in rec.metrics if m.name == "auc").value == 0.5
    assert conformance.check_run_dir(target).ok
    assert not [p for p in (project / "mlflow-records").iterdir() if p.name.startswith(".")]


def test_a_newer_attestation_leaves_an_unchanged_run_alone(project, monkeypatch, capsys):
    monkeypatch.setattr(attestation, "__version__", "0.0.1")
    assert _sync(project) == 0
    before = _tree(project / "mlflow-records")
    monkeypatch.undo()
    capsys.readouterr()
    assert _sync(project) == 0
    out = capsys.readouterr().out
    assert "4 unchanged" in out and "written by another producer version" in out
    assert _tree(project / "mlflow-records") == before


def test_runs_in_flight_and_unreadable_runs(project, capsys):
    meta = _run_dir(project) / "meta.yaml"
    meta.write_text(meta.read_text().replace("status: 3", "status: 1"))  # RUNNING
    runs = (project / "mlruns" / EXPERIMENT).iterdir()
    bad = sorted(p for p in runs if p.is_dir() and p.name != RUN)[0]
    (bad / "metrics" / "auc").write_text("not a metric line at all\n")
    assert _sync(project) == 1
    out = capsys.readouterr().out
    assert "2 written, 1 unreadable; 1 skipped" in out
    assert f"unreadable    {bad.name}" in out and "malformed" in out
    written = {p.name for p in (project / "mlflow-records").iterdir()}
    assert f"run-{RUN}" not in written and f"run-{bad.name}" not in written


def test_a_run_that_logged_no_metric_still_gets_a_record(project):
    shutil.rmtree(_run_dir(project) / "metrics")
    result = mlflow_sync.sync(project / "mlruns", project / "out")
    assert result.ok
    rec = schema.read(project / "out" / f"run-{RUN}")
    assert rec.history is None and [m.name for m in rec.metrics] == ["wall_time_s"]
    # The ledger's rule is unchanged: nothing measured, nothing to rank.
    assert RUN not in {f.run_id for f in trackers.MLFLOW.scan(project)}


def test_a_missing_store_is_a_sentence(tmp_path, capsys):
    rc = main(["mlflow", "sync", "--tracking-dir", str(tmp_path / "nope")])
    assert rc == 1
    assert "is not a directory" in capsys.readouterr().out


# ---- the ledger reads a synced run once, as native --------------------------------


def test_the_ledger_reads_each_synced_run_once_as_a_native_record(project):
    before = import_project(project)
    assert {i.adapter for i in before.imported} == {"mlflow"}
    assert _sync(project) == 0
    result = import_project(project)
    assert result.refused == []
    assert len(result.imported) == 4  # not 8: the store's copy yields to the record
    assert {i.adapter for i in result.imported} == {"core"}
    assert {i.name for i in result.imported} == {i.record.run_id for i in before.imported}
    for imported in result.imported:
        row = row_from_core(project.name, imported)
        assert row.adapter == "core" and row.family == "flows"


# ---- the provider's tags reach the record ----------------------------------------


def _tag(run_dir: Path, key: str, value: str) -> None:
    (run_dir / "tags" / key).write_text(value)


def test_provider_tags_become_code_and_dependencies(project):
    run_dir = _run_dir(project)
    sha = "c" * 40
    code = schema.CodeIdentity(sha, True, None, None, "inferred", "git HEAD at run start")
    deps = [
        schema.Dependency("python", "3.12.1", None, None, "verified", "sys.version_info", None),
        schema.Dependency("scikit-learn", "1.7.0", None, None, "verified", "INSTALLER=uv", "index"),
    ]
    _tag(run_dir, mapping.FORMAT_TAG, mapping.FORMAT)
    _tag(run_dir, mapping.CODE_TAG, json.dumps(asdict(code)))
    _tag(run_dir, mapping.DEPENDENCIES_TAG, json.dumps([asdict(d) for d in deps]))
    _tag(run_dir, mapping.GIT_COMMIT, sha)
    result = mlflow_sync.sync(project / "mlruns", project / "out")
    assert result.ok
    rec = schema.read(project / "out" / f"run-{RUN}")
    assert (rec.code.git_sha, rec.code.dirty, rec.code.provenance) == (sha, True, "inferred")
    assert rec.dependencies is not None
    assert [(d.name, d.provenance) for d in rec.dependencies] == [
        ("python", "verified"),
        ("scikit-learn", "verified"),
    ]
    assert all(mapping.TAG_EVIDENCE in (d.evidence or "") for d in rec.dependencies)
    # The other runs carry no such tag: not observed, null.
    others = [schema.read(d) for d in schema.layout.find_runs(project / "out")]
    assert sum(r.dependencies is None for r in others) == 3


def test_a_bad_tag_is_a_note_not_a_failure(project):
    run_dir = _run_dir(project)
    _tag(run_dir, mapping.FORMAT_TAG, mapping.FORMAT)
    _tag(run_dir, mapping.DEPENDENCIES_TAG, "{oops")
    _tag(run_dir, mapping.ERROR_TAG, "code not observed: OSError: x")
    result = mlflow_sync.sync(project / "mlruns", project / "out")
    assert result.ok
    rec = schema.read(project / "out" / f"run-{RUN}")
    assert rec.dependencies is None
    notes = rec.ext["mlflow-sync"]["notes"]
    assert any("is not JSON" in n for n in notes)
    assert any("code not observed" in n for n in notes)


# ---- a fresh real store, with the provider on (needs mlflow) -----------------------


def test_sync_and_the_exporter_agree_on_a_tagged_real_run(tmp_path):
    """One mapping: a run MLflow wrote with the provider's tags reads the same
    through the exporter (MlflowClient) and through sync (the files)."""
    pytest.importorskip("mlflow")
    import subprocess
    import sys

    from test_mlflow_records import _producer

    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    work = tmp_path / "work"
    work.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@e.org"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(work), *args], check=True)
    (work / "train.py").write_text(
        "import os\nos.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'\nimport mlflow\n"
        "mlflow.set_tracking_uri('file:./mlruns')\nmlflow.set_experiment('tagged')\n"
        "with mlflow.start_run(run_name='one'):\n    mlflow.log_metric('loss', 0.25)\n"
    )
    subprocess.run(["git", "-C", str(work), "add", "train.py"], check=True)
    subprocess.run(["git", "-C", str(work), "commit", "-q", "-m", "t"], check=True)
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("MLFLOW_")}
    env.pop(mapping.OPT_OUT_ENV, None)
    subprocess.run([sys.executable, "train.py"], cwd=work, env=env, check=True)

    producer = _producer()  # loading it allows mlflow's file store, for every later test too
    written, _ = producer.export(producer._uri(str(work / "mlruns")), tmp_path / "exported")
    (theirs,) = [schema.read(d) for d in written]
    result = mlflow_sync.sync(work / "mlruns", tmp_path / "synced")
    assert result.ok
    ours = schema.read(tmp_path / "synced" / f"run-{theirs.run_id}")
    assert ours.dependencies is not None and theirs.dependencies is not None
    assert ours.code == theirs.code and ours.code.dirty is False
    assert ours.dependencies == theirs.dependencies
    assert {d.name for d in ours.dependencies} >= {"python", "mlflow"}
