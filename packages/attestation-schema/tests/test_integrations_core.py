"""The callbacks' stdlib core: identities, config, outputs and the recorder.

Runs everywhere, the bare 3.11 job included (no framework needed). Spec:
docs/superpowers/specs/2026-11-19-training-callbacks-design.md.
"""

from __future__ import annotations

import ast
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import attestation_schema as s
import pytest
from attestation_schema import conformance
from attestation_schema.integrations import _core as core
from integration_fixtures import DATASET_ID, SPLIT, make_dataset

INTEGRATIONS = Path(core.__file__).parent

# ---- the import boundary ---------------------------------------------------------


def test_importing_the_package_imports_no_framework():
    code = (
        "import sys, attestation_schema, attestation_schema.integrations,"
        " attestation_schema.integrations._core\n"
        "bad = sorted(m for m in ('torch', 'transformers', 'lightning', 'accelerate')"
        " if m in sys.modules)\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_missing_framework_names_the_extra():
    from attestation_schema.integrations import missing_extra

    err = missing_extra("huggingface", ModuleNotFoundError("x", name="transformers"))
    assert 'pip install "attestation-schema[hf]"' in str(err)
    assert "transformers" in str(err)


def test_only_the_framework_modules_import_a_framework():
    frameworks = {"torch", "transformers", "lightning", "accelerate", "pytorch_lightning"}
    allowed = {"huggingface.py": {"transformers"}, "lightning.py": {"lightning"}}
    seen = set()
    for path in sorted(INTEGRATIONS.glob("*.py")):
        roots = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        assert roots & frameworks <= allowed.get(path.name, set()), path.name
        seen |= roots & frameworks
    assert seen == {"transformers", "lightning"}  # the scan is not vacuous


# ---- dependency identity -----------------------------------------------------------
# The rule itself is `attestation_schema.identity` (tests/test_identity.py);
# here only what the callbacks add: which modules, and only imported ones.


def test_installed_dependency_reads_this_process(monkeypatch):
    assert core.installed_dependency("nope", "no_such_module_here", "nope") is None
    d = core.installed_dependency("attestation-schema", "attestation_schema", "attestation-schema")
    assert d is not None and d.version == s.__version__
    assert "__version__ == metadata" in d.evidence  # the module was checked, not imported anew
    rec = _record(dependencies=[core.python_dependency(), d])
    assert s.loads(s.dumps(rec)) == rec


# ---- code identity -----------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_code_identity_reads_head_dirty_and_a_credential_free_remote(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.org")
    _git(repo, "config", "user.name", "t")
    _git(repo, "remote", "add", "origin", "https://bob:s3cret@example.org/x/train.git")
    (repo / "train.py").write_text("print(1)\n")
    _git(repo, "add", "train.py")
    _git(repo, "commit", "-q", "-m", "init")
    clean = core.code_identity(repo)
    assert clean.git_sha == _git(repo, "rev-parse", "HEAD")
    assert (clean.dirty, clean.provenance) == (False, "inferred")
    assert clean.repo == "https://example.org/x/train.git"
    assert str(tmp_path) not in (clean.evidence or "")
    (repo / "train.py").write_text("print(2)\n")
    assert core.code_identity(repo).dirty is True
    (repo / "untracked.txt").write_text("x")
    (repo / "train.py").write_text("print(1)\n")
    assert core.code_identity(repo).dirty is False  # untracked files are not the code


def test_no_checkout_is_unknown_with_every_field_null(tmp_path):
    c = core.code_identity(tmp_path)
    assert (c.git_sha, c.dirty, c.repo, c.version, c.provenance) == (
        None,
        None,
        None,
        None,
        "unknown",
    )
    assert c.evidence


# ---- config, values, outputs ------------------------------------------------------


def test_jsonable_never_guesses_a_type():
    import enum

    class Mode(enum.Enum):
        A = "a"

    out = core.jsonable(
        {"m": Mode.A, "p": Path("x/y"), "t": (1, 2), "nan": float("nan"), "big": 2**60, 3: None}
    )
    assert out == {"m": "a", "p": "x/y", "t": [1, 2], "nan": "nan", "big": str(2**60), "3": None}
    assert s.canonical.canonicalize(out)


def test_redact_replaces_what_the_conformance_rule_flags():
    config = {
        "tracking_uri": "https://user:pw@mlflow.example.org",
        "nested": {"api_key": "abc123", "hub_token": "<HUB_TOKEN>"},
        "list": ["hf_" + "a" * 34, "fine"],
        "lr": 0.1,
    }
    clean, paths = core.redact(config, path="$.config")
    assert paths == ["$.config.tracking_uri", "$.config.nested.api_key", "$.config.list[0]"]
    assert clean["nested"]["hub_token"] == "<HUB_TOKEN>" and clean["lr"] == 0.1
    assert conformance.credential_findings(clean) == []
    assert conformance.credential_findings(config)  # the control: the raw config is flagged


def test_number_takes_numbers_only():
    assert core.number(1) == 1 and core.number(0.5) == 0.5
    assert core.number(True) is None and core.number("1") is None and core.number([1]) is None
    assert core.number(2**60) == float(2**60)

    class Scalar:
        def item(self):
            return 0.25

    assert core.number(Scalar()) == 0.25


def test_merge_directions_drops_a_disagreement():
    assert core.merge_directions([("a", "min"), ("b", "max"), ("a", "min")]) == {
        "a": "min",
        "b": "max",
    }
    assert core.merge_directions([("a", "min"), ("a", "max"), ("a", "min")]) == {"a": None}
    assert core.merge_directions([("a", "up")]) == {}


def test_output_entries_digest_files_and_skip_what_is_gone(tmp_path):
    root = tmp_path / "out"
    (root / "checkpoint-2").mkdir(parents=True)
    (root / "checkpoint-2" / "model.bin").write_bytes(b"weights")
    (root / "checkpoint-2" / "sub").mkdir()
    (root / "checkpoint-2" / "sub" / "opt.pt").write_bytes(b"opt")
    (root / "last.ckpt").write_bytes(b"ckpt")
    (tmp_path / "elsewhere.bin").write_bytes(b"not this run's")
    (root / "link").symlink_to(tmp_path / "elsewhere.bin")  # never followed
    entries = core.output_entries(
        [root / "checkpoint-2", root / "checkpoint-4", root / "last.ckpt", root / "link"], root
    )
    assert [(e.name, e.role) for e in entries] == [
        ("checkpoint-2/model.bin", "output"),
        ("checkpoint-2/sub/opt.pt", "output"),
        ("last.ckpt", "output"),
    ]
    assert entries[0].sha256 == hashlib.sha256(b"weights").hexdigest()
    outside = core.output_name(tmp_path / "elsewhere" / "a.ckpt", root)
    assert outside == "../elsewhere/a.ckpt"


# ---- the recorder ------------------------------------------------------------------


def _record(**overrides) -> s.RunRecord:
    base = dict(
        run_id="r",
        family=None,
        arm=None,
        status="complete",
        started_at=None,
        finished_at=None,
        producer=s.Producer("t", None),
        code=s.UNKNOWN_CODE,
        config={},
        data=[],
        metrics=[],
        history=None,
        seeds=None,
        dependencies=None,
        ext={},
    )
    return s.RunRecord(**{**base, **overrides})


def _finish(rec: core.Recorder, status="complete", **kw):
    args = dict(
        status=status,
        config={"lr": 0.1, "token_url": "https://a:b@h"},
        data=rec.data(False, []),
        metrics=rec.metrics(),
        seeds={"seed": 1},
        dependencies=[core.python_dependency()],
        ext={"world_size": 1},
    )
    args.update(kw)
    return rec.finish(**args)


def test_the_recorder_writes_the_layout_in_order_and_links_the_dataset(tmp_path):
    dataset = make_dataset(tmp_path / "ds")
    rec = core.Recorder(
        core.Options(datasets=[(tmp_path / "ds", SPLIT)], directions={"loss": "min"}),
        "test",
        "test-ns",
    )
    rec.begin(tmp_path / "out")
    assert rec.run_dir == tmp_path / "out" / "attestation" / f"run-{rec.run_id}"
    rec.log({"loss": 1.0, "flag": True, "name": "x", "acc": float("nan")}, 1)
    rec.log({"loss": 0.5}, 2)
    path = _finish(rec)
    assert path == rec.run_dir / "attestation" / "run.json"
    assert conformance.check_run_dir(rec.run_dir).ok
    record = s.read(rec.run_dir)
    assert [d.name for d in record.data] == [
        DATASET_ID,
        f"{DATASET_ID}/split-random.json",
        "attestation/digests.json",
    ]
    assert s.resolve_lineage(record, [dataset]).verified
    final = {m.name: (m.value, m.step, m.direction) for m in record.metrics}
    assert final["loss"] == (0.5, 2, "min")
    assert str(final["acc"][0]) == "nan" and final["acc"][2] is None
    assert "flag" not in final and "name" not in final
    assert record.config["token_url"] == core.REDACTED
    assert record.ext["test-ns"]["redacted"] == ["$.config.token_url"]
    assert record.seeds == {"seed": 1}
    assert _finish(rec) == path  # written once


def test_nothing_logged_is_a_null_history_and_given_seeds_win(tmp_path):
    rec = core.Recorder(core.Options(seeds={"mine": 3}), "test", "test-ns")
    rec.begin(tmp_path)
    _finish(rec, status="failed")
    record = s.read(rec.run_dir)
    assert record.history is None and record.status == "failed"
    assert record.seeds == {"mine": 3}
    assert [d.name for d in record.data if d.sha256 is None] == ["train"]


def test_an_existing_record_is_refused_at_begin(tmp_path):
    first = core.Recorder(core.Options(run_dir=tmp_path / "run"), "test", "test-ns")
    first.begin(tmp_path)
    _finish(first)
    again = core.Recorder(core.Options(run_dir=tmp_path / "run"), "test", "test-ns")
    with pytest.raises(FileExistsError):
        again.begin(tmp_path)
    assert not again.begun


def test_abort_leaves_no_temporary_history(tmp_path):
    rec = core.Recorder(core.Options(), "test", "test-ns")
    rec.begin(tmp_path)
    rec.log({"loss": 1.0}, 0)
    rec.abort()
    assert list((rec.run_dir / "attestation").iterdir()) == []


def test_error_status_and_ext():
    assert core.error_status(KeyboardInterrupt()) == "cancelled"
    assert core.error_status(ValueError("x")) == "failed"
    assert core.error_ext(ValueError("line one\nline two")) == {
        "type": "ValueError",
        "message": "line one",
    }
