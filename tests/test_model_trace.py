"""Model registry linkage: a model version -> the run that output its files ->
that run's data -> dataset records -> source runs, every link by sha256.

The fixtures are examples/model-registry/, written by the real MLflow and
transformers libraries (generate.py). Tamper tests copy them to tmp_path and
change one thing, so each guard is seen to fail.
Spec: docs/superpowers/specs/2026-11-20-model-registry-linkage-design.md.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import attestation_schema as schema
import pytest

from attestation import claims, model_trace
from attestation.cli import main
from attestation.ledger_adapters import mlflow_store, registries

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "model-registry"
MLFLOW_RUN = "82891d65b22e4e838d8a0cf49f773f2f"


def _trace(ref: str, root: Path = EXAMPLE, base: Path | None = None):
    return model_trace.trace(ref, model_trace.build_index([root]), [root], base or root)


def _kinds(found) -> list[str]:
    return [i.kind for i in found.issues]


@pytest.fixture
def copy(tmp_path) -> Path:
    dest = tmp_path / "model-registry"
    shutil.copytree(EXAMPLE, dest, symlinks=True)
    return dest


# ---- the committed chain -----------------------------------------------------------


def test_a_saved_model_traces_to_its_training_run_its_dataset_and_the_run_before():
    found = _trace("hf/model")
    assert found.verified, found.format()
    (producer,) = found.producers
    assert producer.run.label == "toy-seq-gpt2/toy-seq-gpt2-0"
    assert producer.complete and len(producer.matched) == 4
    kinds = [n.kind for n in producer.lineage.nodes]
    assert kinds == ["run", "dataset", "run"]
    assert [n.label for n in producer.lineage.nodes][1:] == ["toy-seq@1", "toy-seq-prep/prep-0"]
    assert found.chain() == (
        "hf/model -> run toy-seq-gpt2/toy-seq-gpt2-0 (4/4 files) -> dataset toy-seq@1 [verified]"
    )


@pytest.mark.parametrize("ref", ["models:/iris-clf/1", "models:/iris-clf@champion"])
def test_a_registered_mlflow_version_traces_to_its_run_by_digest(ref):
    found = _trace(ref)
    assert found.ok, found.format()
    (version,) = found.versions
    assert (version.registry, version.name, version.version) == ("mlflow", "iris-clf", "1")
    assert version.source_run_id == MLFLOW_RUN
    assert sorted(f.path for f in version.files) == [
        "MLmodel",
        "conda.yaml",
        "model.skops",
        "python_env.yaml",
        "requirements.txt",
    ]
    (producer,) = found.producers
    assert producer.run.record.run_id == MLFLOW_RUN and producer.complete
    # The synced core record, not the in-memory conversion of the store.
    assert Path(producer.run.source_path).parts[-2:] == ("mlflow-records", f"run-{MLFLOW_RUN}")
    # MLflow names a dataset input with its own digest, never a sha256 of bytes.
    assert _kinds(found) == ["untraced-training-data"]
    assert ("alias champion -> version 1" in version.notes) == ref.endswith("@champion")


def test_the_store_alone_traces_through_the_adapters_own_reading(copy):
    """Without `attest mlflow sync`, the run is the MLflow adapter's in-memory
    record, whose outputs are the logged model's files (MLflow 3 keeps them
    outside the run; `outputs/<id>/meta.yaml` says the run made them)."""
    shutil.rmtree(copy / "mlflow-records")
    found = _trace("models:/iris-clf/1", copy)
    (producer,) = found.producers
    assert producer.complete and Path(producer.run.source_path).name == MLFLOW_RUN
    names = [d.name for d in producer.run.record.data if d.role == "output"]
    assert names and all(n.startswith("models/m-") for n in names)


def test_the_mlflow_reader_reads_declared_logged_models_and_no_run_named_models():
    exp = next(p for p in (EXAMPLE / "mlruns").iterdir() if (p / "models").is_dir())
    assert [p.name for p in mlflow_store.run_dirs(exp)] == [MLFLOW_RUN]
    found = mlflow_store.read_run(EXAMPLE, exp, exp / MLFLOW_RUN, "iris")
    assert found is not None and len(found.models) == 5


# ---- what breaks ------------------------------------------------------------------


def test_a_changed_model_file_breaks_the_link_the_synced_record_bound(copy):
    weights = next((copy / "mlruns").glob("*/models/*/artifacts/model.skops"))
    weights.write_bytes(weights.read_bytes() + b"\0")
    found = _trace("models:/iris-clf/1", copy)
    assert not found.ok
    assert "partial-output" in _kinds(found)  # four of five files still match
    assert "4 of 5" in found.format() and "model.skops" in found.format()


def test_no_run_output_these_files_names_the_run_the_registry_claims(copy):
    shutil.rmtree(copy / "mlflow-records")
    shutil.rmtree(next((copy / "mlruns").glob(f"*/{MLFLOW_RUN}")))
    found = _trace("models:/iris-clf/1", copy)
    assert not found.ok and _kinds(found) == ["no-producing-run"]
    assert f"the registry names run {MLFLOW_RUN}; no record of it" in found.issues[0].message


def test_a_registry_naming_another_run_than_the_digests_is_broken(copy):
    meta = copy / "mlruns" / "models" / "iris-clf" / "version-1" / "meta.yaml"
    meta.write_text(meta.read_text().replace(MLFLOW_RUN, "deadbeef" * 4))
    found = _trace("models:/iris-clf/1", copy)
    assert not found.ok and "registry-run-mismatch" in _kinds(found)
    assert found.producers, "the digests still found the run: names only diagnose"


@pytest.mark.parametrize(
    ("ref", "why"),
    [
        ("models:/iris-clf/2", "'iris-clf' has no version 2"),
        ("models:/iris-clf@staging", "'iris-clf' has no alias 'staging'"),
        ("models:/other/1", "no registered model 'other'"),
        ("hf:org/tiny", "no local hub cache of org/tiny"),
        ("nowhere/model", "neither a registry reference"),
    ],
)
def test_an_unresolvable_reference_is_broken_and_says_why(ref, why):
    found = _trace(ref)
    assert not found.ok and _kinds(found) == ["unresolved-model"]
    assert why in found.issues[0].message


@pytest.mark.parametrize("ref", ["models:/iris-clf", "models:/m-abc123", "hf:a/b/c", "hf:x@"])
def test_a_malformed_registry_reference_is_refused(ref):
    with pytest.raises(registries.RefError):
        _trace(ref)


def test_a_version_with_a_file_no_run_output_is_broken(copy):
    """Even a model card added later: those bytes trace to nothing, and the
    message names the file so the fix is obvious."""
    (copy / "hf" / "model" / "README.md").write_text("model card added after training\n")
    found = _trace("hf/model", copy)
    assert not found.ok
    assert _kinds(found) == ["partial-output"] and "README.md" in found.issues[0].message


# ---- a local Hugging Face hub cache --------------------------------------------------

COMMIT = "c0ffee" + "0" * 34  # synthetic: no Hub commit exists for a model trained offline


def _hub_cache(root: Path, model: Path) -> Path:
    """The layout huggingface_hub downloads into (models--org--name/{refs,blobs,
    snapshots}), built from the saved model's files. Synthetic -- this repo
    has no network -- and labelled so in the commit sha."""
    cache = root / "hub" / "models--org--tiny"
    (cache / "refs").mkdir(parents=True)
    (cache / "refs" / "main").write_text(COMMIT)
    snap = cache / "snapshots" / COMMIT
    snap.mkdir(parents=True)
    (cache / "blobs").mkdir()
    for f in sorted(model.iterdir()):
        blob = cache / "blobs" / schema.sha256_file(f)
        shutil.copyfile(f, blob)
        os.symlink(os.path.relpath(blob, snap), snap / f.name)
    return cache


@pytest.mark.parametrize("ref", ["hf:org/tiny", "hf:org/tiny@main", f"hf:org/tiny@{COMMIT[:8]}"])
def test_a_hub_cache_snapshot_traces_like_its_directory(copy, ref):
    _hub_cache(copy, copy / "hf" / "model")
    found = _trace(ref, copy)
    assert found.verified, found.format()
    (version,) = found.versions
    assert (version.registry, version.name, version.version) == ("hf-cache", "org/tiny", COMMIT)
    assert found.producers[0].run.label == "toy-seq-gpt2/toy-seq-gpt2-0"


def test_a_cached_file_whose_bytes_are_not_its_blob_is_broken(copy):
    cache = _hub_cache(copy, copy / "hf" / "model")
    blob = (cache / "snapshots" / COMMIT / "config.json").resolve()
    blob.write_text(blob.read_text().replace("gpt2", "gpt3"))
    found = _trace("hf:org/tiny", copy)
    assert "model-file-mismatch" in _kinds(found) and not found.ok


# ---- claims ---------------------------------------------------------------------------


def _scanned_db(tmp_path):
    from attestation import ledger
    from attestation.db import get_db

    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, EXAMPLE.parent, project="model-registry")
    return conn


def test_model_is_an_optional_claim_field(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text("<!-- claim: p/r metric=acc value=0.5 model=models:/clf/3 -->\n")
    (claim,), problems = claims.parse_file(doc)
    assert not problems and claim.model == "models:/clf/3"
    doc.write_text("<!-- claim: p/r metric=acc value=0.5 -->\n")
    assert claims.parse_file(doc)[0][0].model is None


def test_the_draft_reports_each_chain_and_lints_the_claim_whose_run_did_not_make_it(tmp_path):
    conn = _scanned_db(tmp_path)
    out = claims.check(conn, EXAMPLE / "DRAFT.md", tracer=model_trace.tracer_for(EXAMPLE))
    assert dict(out["counts"]) == {"supported": 3, "untraced": 1}
    (lint,) = [v for v in out["verdicts"] if v.verdict == "untraced"]
    assert lint.claim.line == 13 and "neither output nor read" in lint.message
    chains = {m["ref"]: m for m in out["models"]}
    assert chains["hf/model"]["verified"]
    assert chains["models:/iris-clf@champion"]["ok"]
    assert not chains["models:/iris-clf@champion"]["verified"]


def test_a_draft_without_model_reads_nothing_and_keeps_its_output(tmp_path):
    def never(ref, base):
        raise AssertionError("traced a draft that names no model")

    conn = _scanned_db(tmp_path)
    doc = tmp_path / "plain.md"
    doc.write_text(f"<!-- claim: model-registry/{MLFLOW_RUN} metric=accuracy value=1.0 -->\n")
    with_tracer = claims.check(conn, doc, tracer=never)
    without = claims.check(conn, doc)
    assert with_tracer["models"] == without["models"] == []
    assert with_tracer["counts"] == without["counts"] == {"supported": 1}


def test_a_broken_model_reference_is_untraced_never_contradicted(tmp_path):
    conn = _scanned_db(tmp_path)
    doc = tmp_path / "d.md"
    doc.write_text(
        f"<!-- claim: model-registry/{MLFLOW_RUN} metric=accuracy value=1.0"
        " model=models:/iris-clf/9 -->\n"
        f"<!-- claim: model-registry/{MLFLOW_RUN} metric=accuracy value=1.0 model=models:/x -->\n"
    )
    out = claims.check(conn, doc, tracer=model_trace.Tracer([EXAMPLE]))
    assert dict(out["counts"]) == {"supported": 2, "untraced": 2}
    messages = [v.message for v in out["verdicts"] if v.verdict == "untraced"]
    assert "[unresolved-model]" in messages[0] and "has no version 9" in messages[0]
    assert "expected models:/<name>/<version>" in messages[1]


def test_attest_claims_prints_the_chains_and_keeps_the_summary_shape(tmp_path, capsys):
    db = tmp_path / "t.db"
    _scanned_db(tmp_path).close()
    rc = main(["claims", "--db", str(db), str(EXAMPLE / "DRAFT.md")])
    out = capsys.readouterr().out
    assert rc == 0, out  # a lint never fails the check
    assert "  model         hf/model -> run toy-seq-gpt2/toy-seq-gpt2-0 (4/4 files)" in out
    assert out.rstrip().endswith("3 claim(s): 3 supported, 1 untraced")


def test_runs_claims_check_carries_model_on_every_row_and_a_models_block(tmp_path, monkeypatch):
    from attestation.mcp.claims_tools import MAX_MODEL_TRACES, _check

    _scanned_db(tmp_path).close()
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    out = _check(path=str(EXAMPLE / "DRAFT.md"))
    assert out["ok"] and out["message"] == "3 claim(s): 3 supported, 1 untraced"
    assert [r["model"] for r in out["claims"]][:3] == [
        "models:/iris-clf@champion",
        "hf/model",
        "hf/model",
    ]
    assert len(out["models"]) == 2 <= MAX_MODEL_TRACES
    assert all(
        set(m) == {"ref", "ok", "verified", "chain", "issues", "n_issues"} for m in out["models"]
    )
    assert len(json.dumps(out)) < 7000


# ---- the command ---------------------------------------------------------------------


def test_attest_model_trace_exit_codes(capsys, copy, monkeypatch):
    monkeypatch.chdir(copy)
    assert main(["model", "trace", "hf/model"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("model trace: verified")
    assert main(["model", "trace", "models:/iris-clf/1", "--json"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["ok"] and not got["verified"] and got["skipped"] == []
    assert got["producers"][0]["complete"] and got["versions"][0]["version"] == "1"
    assert main(["model", "trace", "models:/iris-clf/2"]) == 1
    assert "lineage" not in capsys.readouterr().err
    assert main(["model", "trace", "models:/iris-clf"]) == 2
    assert "expected models:/" in capsys.readouterr().err
    monkeypatch.chdir(copy.parent)
    assert main(["model", "trace", "hf/model", "--search", str(copy)]) == 1  # path is cwd-relative
    assert main(["model", "trace", str(copy / "hf" / "model"), "--search", str(copy)]) == 0
