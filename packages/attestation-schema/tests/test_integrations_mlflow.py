"""The MLflow run-context provider, on a real local MLflow file store.

Each run is started by `mlflow.start_run` in a fresh interpreter inside a git
checkout, so MLflow discovers the provider through its entry point exactly
as a user's script would. Skips without `attestation-schema[mlflow]`, unless
ATTESTATION_REQUIRE_INTEGRATIONS=1 (CI's schema-integrations job), where a
missing mlflow fails. Spec: docs/superpowers/specs/2026-11-20-mlflow-plugin-design.md.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from importlib import metadata
from pathlib import Path

import pytest
from attestation_schema.integrations import mlflow_mapping as m
from integration_fixtures import require

mlflow = require("mlflow")
from attestation_schema.integrations.mlflow import AttestationRunContext  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

_RUN = textwrap.dedent(
    """
    import os, sys
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    import mlflow
    if len(sys.argv) > 1 and sys.argv[1] == "boom":
        from attestation_schema.integrations import mlflow_mapping
        def boom():
            raise RuntimeError("probe failed on purpose")
        mlflow_mapping.observe_code = boom
    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run() as run:
        mlflow.log_metric("loss", 0.5)
    print(run.info.run_id)
    """
)


@pytest.fixture(autouse=True)
def _file_store(monkeypatch):
    """mlflow 3 refuses a file store in this process's client unless asked."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path) -> tuple[Path, str]:
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q")
    _git(work, "config", "user.email", "t@example.org")
    _git(work, "config", "user.name", "t")
    _git(work, "remote", "add", "origin", "https://someone:s3cret@example.org/lab/code.git")
    (work / "train.py").write_text(_RUN)
    (work / ".gitignore").write_text("mlruns/\n")
    _git(work, "add", "train.py", ".gitignore")
    _git(work, "commit", "-q", "-m", "train")
    return work, _git(work, "rev-parse", "HEAD")


def _run(work: Path, *args: str, **env: str) -> dict[str, str]:
    """Start one MLflow run; return its tags as MLflow stored them."""
    base = {
        k: v for k, v in os.environ.items() if not k.startswith(("MLFLOW_", "ATTESTATION_MLFLOW_"))
    }
    proc = subprocess.run(
        [sys.executable, "train.py", *args],
        cwd=work,
        env={**base, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    run_id = proc.stdout.split()[-1]
    from mlflow.tracking import MlflowClient

    client = MlflowClient((work / "mlruns").resolve().as_uri())
    return dict(client.get_run(run_id).data.tags)


def test_the_entry_point_is_declared_and_mlflow_registers_it():
    eps = metadata.entry_points(group="mlflow.run_context_provider")
    ours = [ep for ep in eps if ep.value.startswith("attestation_schema.")]
    assert [ep.value for ep in ours] == [
        "attestation_schema.integrations.mlflow:AttestationRunContext"
    ]
    from mlflow.tracking.context.registry import _run_context_provider_registry

    assert any(isinstance(p, AttestationRunContext) for p in _run_context_provider_registry)


def test_in_context_reads_one_variable_and_runs_nothing(monkeypatch):
    def no(*a, **k):
        raise AssertionError("in_context() must not start a process")

    monkeypatch.setattr(subprocess, "run", no)
    monkeypatch.setattr(subprocess, "Popen", no)
    monkeypatch.delenv(m.OPT_OUT_ENV, raising=False)
    assert AttestationRunContext().in_context() is True
    monkeypatch.setenv(m.OPT_OUT_ENV, "0")
    assert AttestationRunContext().in_context() is False


def test_every_run_carries_its_provenance(checkout):
    work, sha = checkout
    tags = _run(work)
    assert tags[m.FORMAT_TAG] == m.FORMAT
    assert m.ERROR_TAG not in tags
    code = json.loads(tags[m.CODE_TAG])
    assert (code["git_sha"], code["dirty"]) == (sha, False)
    assert code["repo"] == "https://example.org/lab/code.git"  # credentials stripped
    assert "s3cret" not in json.dumps(tags)
    # MLflow's own git provider ran too and agrees; nothing of MLflow's was overwritten.
    assert tags[m.GIT_COMMIT] == sha

    reading = m.read_tags(tags)
    assert reading.problems == []
    assert reading.code.git_sha == sha and reading.code.dirty is False
    deps = {d.name: d for d in reading.dependencies or []}
    assert deps["python"].provenance == "verified"
    # mlflow was imported by the run, so it is named, by the identity rule.
    installed = metadata.version("mlflow-skinny")
    assert deps["mlflow"].version == mlflow.__version__ == installed
    assert deps["mlflow"].evidence and m.TAG_EVIDENCE in deps["mlflow"].evidence


def test_a_dirty_tree_is_recorded_as_dirty(checkout):
    work, sha = checkout
    (work / "train.py").write_text(_RUN + "\n# edited\n")
    code = m.read_tags(_run(work)).code
    assert (code.git_sha, code.dirty) == (sha, True)


def test_the_opt_out_removes_every_tag(checkout):
    work, sha = checkout
    tags = _run(work, **{m.OPT_OUT_ENV: "0"})
    assert not any(k.startswith(m.PREFIX) for k in tags)
    assert tags[m.GIT_COMMIT] == sha  # MLflow's own provider is unaffected
    code = m.read_tags(tags).code
    assert (code.git_sha, code.dirty, code.provenance) == (sha, None, "inferred")


def test_a_failing_probe_never_fails_the_run(checkout):
    work, _ = checkout
    tags = _run(work, "boom")
    assert tags[m.ERROR_TAG] == "code not observed: RuntimeError: probe failed on purpose"
    assert m.DEPENDENCIES_TAG in tags and m.CODE_TAG not in tags
