"""The MLflow mapping and the run-context provider's tags, without mlflow.

Standard library only, so it runs everywhere (the bare 3.11 job included):
what the provider observes, that it never raises, and how its tags read back
into a record. The provider class itself, on a real MLflow store, is
test_integrations_mlflow.py. Spec:
docs/superpowers/specs/2026-11-20-mlflow-plugin-design.md.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import attestation_schema as s
import pytest
from attestation_schema.canonical import canonicalize
from attestation_schema.integrations import mlflow_mapping as m

SHA = "a" * 40
OTHER = "b" * 40


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _tags(**over: str) -> dict[str, str]:
    """The provider's tags for a clean checkout at SHA and two dependencies."""
    code = s.CodeIdentity(SHA, False, "https://example.org/x.git", None, "inferred", "git HEAD")
    deps = [
        s.Dependency("python", "3.12.1", None, None, "verified", "sys.version_info", None),
        s.Dependency("numpy", "2.1.0", None, None, "verified", "INSTALLER=pip", "index"),
        s.Dependency("lab", "0.1", None, None, "inferred", None, "file"),
    ]
    tags = {
        m.FORMAT_TAG: m.FORMAT,
        m.CODE_TAG: canonicalize(asdict(code)).decode(),
        m.DEPENDENCIES_TAG: canonicalize([asdict(d) for d in deps]).decode(),
    }
    return {**tags, **over}


# ---- the shared mapping ---------------------------------------------------------


def test_timestamps_latest_and_grouping():
    assert m.timestamp_ms(1787956426024) == "2026-08-28T22:33:46.024000Z"
    assert m.timestamp_ms(None) is None
    P = SimpleNamespace
    pts = [P(key="a", step=1, timestamp=5, value=1.0, ds=None),
           P(key="a", step=1, timestamp=5, value=2.0, ds=None),
           P(key="a", step=0, timestamp=9, value=7.0, ds=None),
           P(key="a", step=0, timestamp=1, value=3.0, ds="test")]  # fmt: skip
    # MLflow's key: greatest step, then latest timestamp, then largest value.
    assert m.latest(pts).value == 2.0
    grouped = m.group_points(pts, lambda p: p.ds)
    assert m.series_order(grouped) == [("a", None), ("a", "test")]
    assert [p.value for p in grouped[("a", None)]] == [1.0, 2.0, 7.0]  # stable by timestamp


# ---- observing ---------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_observe_code_reads_the_checkout_and_its_dirty_bit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.org")
    _git(repo, "config", "user.name", "t")
    _git(repo, "remote", "add", "origin", "https://bob:s3cret@example.org/x/train.git")
    (repo / "train.py").write_text("print(1)\n")
    _git(repo, "add", "train.py")
    _git(repo, "commit", "-q", "-m", "init")
    code = m.observe_code(str(repo))
    assert code.git_sha == _git(repo, "rev-parse", "HEAD")
    assert (code.dirty, code.provenance, code.repo) == (
        False,
        "inferred",
        "https://example.org/x/train.git",
    )
    (repo / "train.py").write_text("print(2)\n")
    assert m.observe_code(str(repo)).dirty is True
    none = m.observe_code(str(tmp_path / "nowhere"))
    assert (none.git_sha, none.provenance) == (None, "unknown")


def test_only_imported_packages_are_named_and_nothing_is_imported(monkeypatch):
    for _, module, _ in m.KEY_PACKAGES:
        monkeypatch.delitem(sys.modules, module, raising=False)
    deps = m.observe_dependencies({})
    assert [d.name for d in deps] == ["python"]
    assert deps[0].provenance == "verified"
    assert not any(module in sys.modules for _, module, _ in m.KEY_PACKAGES)
    # An imported key package is identified through the one rule.
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(__version__="0"))
    monkeypatch.setattr(m, "_installed", lambda dists: dists[0])
    seen = {}

    def identity(dist, module, *, name):
        seen[name] = (dist, module)
        return s.Dependency(name, "0", None, None, "inferred", "stub", None)

    monkeypatch.setattr(m, "package_identity", identity)
    assert [d.name for d in m.observe_dependencies({})] == ["python", "numpy"]
    assert seen["numpy"][1] is sys.modules["numpy"]  # the module object, never a name


def test_declared_packages_are_named_from_metadata_when_not_imported():
    assert m.declared_packages({m.PACKAGES_ENV: " my-lib , other:pkg_mod,,"}) == [
        ("my-lib", "my_lib"),
        ("other", "pkg_mod"),
    ]
    deps = m.observe_dependencies({m.PACKAGES_ENV: "attestation-schema:attestation_schema"})
    ours = next(d for d in deps if d.name == "attestation-schema")
    assert ours.version == s.__version__
    missing = m.observe_dependencies({m.PACKAGES_ENV: "no-such-dist-here"})[-1]
    assert (missing.name, missing.provenance) == ("no-such-dist-here", "unknown")


def test_tags_round_trip_through_the_reader():
    tags = m.observe_tags({}, code=lambda: s.CodeIdentity(SHA, True, None, None, "inferred", "x"))
    assert tags[m.FORMAT_TAG] == m.FORMAT
    assert json.loads(tags[m.PRODUCER_TAG])["version"] == s.__version__
    assert m.ERROR_TAG not in tags
    assert all(k.startswith(m.PREFIX) for k in tags)  # never an mlflow.* tag
    assert all(len(v) <= m.MAX_TAG_VALUE for v in tags.values())
    reading = m.read_tags(tags)
    assert reading.code.git_sha == SHA and reading.code.dirty is True
    assert reading.dependencies and reading.dependencies[0].name == "python"
    assert reading.problems == []


def test_a_failing_probe_is_reported_on_the_run_and_the_rest_kept():
    def boom():
        raise RuntimeError("git exploded\nsecond line")

    tags = m.observe_tags({}, code=boom)
    assert m.CODE_TAG not in tags and m.DEPENDENCIES_TAG in tags
    assert tags[m.ERROR_TAG] == "code not observed: RuntimeError: git exploded"

    def deps_boom(environ):
        raise OSError("no metadata")

    tags = m.observe_tags({}, code=lambda: m.observe_code("/"), dependencies=deps_boom)
    assert m.CODE_TAG in tags and m.DEPENDENCIES_TAG not in tags
    assert "dependencies not observed: OSError: no metadata" in tags[m.ERROR_TAG]
    assert "the run-context provider reported" in m.read_tags(tags).problems[-1]


def test_dependencies_past_the_tag_limit_are_dropped_and_said():
    big = [
        s.Dependency(f"pkg{i}", "1.0", None, None, "inferred", "x" * 400, None) for i in range(40)
    ]
    tags = m.observe_tags({}, dependencies=lambda env: big)
    kept = json.loads(tags[m.DEPENDENCIES_TAG])
    assert 1 <= len(kept) < 40 and len(tags[m.DEPENDENCIES_TAG]) <= m.MAX_TAG_VALUE
    assert f"{len(kept)} of 40 dependencies fit" in tags[m.ERROR_TAG]
    assert "pkg39" in tags[m.ERROR_TAG]


def test_the_opt_out():
    assert m.enabled({}) is True
    for off in ("0", "false", "No", " off "):
        assert m.enabled({m.OPT_OUT_ENV: off}) is False
    assert m.enabled({m.OPT_OUT_ENV: "1"}) is True


def _dep() -> s.Dependency:
    """`unknown`: a dependency naming nothing (`verified` would claim to)."""
    return s.Dependency("x", None, None, None, "unknown", None, None)


def _record_dict() -> dict:
    return s.RunRecord(
        run_id="r",
        family=None,
        arm=None,
        status="complete",
        started_at=None,
        finished_at=None,
        producer=s.Producer("t", None),
        code=s.CodeIdentity(None, None, None, None, "unknown", None),
        config={},
        data=[],
        metrics=[],
        history=None,
        seeds=None,
        dependencies=None,
        ext={},
    ).to_dict()


# ---- reading back ----------------------------------------------------------------


def test_the_provider_and_mlflow_agreeing_keeps_the_dirty_bit():
    code = m.code_from_tags(_tags(**{m.GIT_COMMIT: SHA}))
    assert (code.git_sha, code.dirty, code.provenance) == (SHA, False, "inferred")
    assert code.evidence and code.evidence.endswith(f"{m.GIT_COMMIT} agrees")


def test_two_commits_are_unknown_and_both_named():
    code = m.code_from_tags(_tags(**{m.GIT_COMMIT: OTHER}))
    assert (code.git_sha, code.dirty, code.provenance) == (None, None, "unknown")
    assert SHA in (code.evidence or "") and OTHER in (code.evidence or "")
    assert s.from_dict({**_record_dict(), "code": asdict(code)}).code == code


def test_one_source_alone_and_neither():
    only_mlflow = m.code_from_tags({m.GIT_COMMIT: SHA, m.GIT_REPO: "https://t0k@example.org/x"})
    assert (only_mlflow.provenance, only_mlflow.dirty) == ("inferred", None)
    assert only_mlflow.repo == "https://example.org/x"  # credentials stripped
    assert m.code_from_tags(_tags()).dirty is False
    neither = m.code_from_tags({})
    assert (neither.git_sha, neither.provenance, neither.evidence) == (None, "unknown", None)
    # The provider saw no checkout, MLflow saw one: MLflow's.
    unknown = asdict(s.CodeIdentity(None, None, None, None, "unknown", "no checkout"))
    tags = _tags(**{m.CODE_TAG: json.dumps(unknown), m.GIT_COMMIT: SHA})
    assert m.code_from_tags(tags).git_sha == SHA


def test_dependencies_keep_the_run_time_provenance_and_name_the_tag():
    deps = m.dependencies_from_tags(_tags())
    assert deps is not None
    assert [(d.name, d.provenance) for d in deps] == [
        ("python", "verified"),
        ("numpy", "verified"),
        ("lab", "inferred"),
    ]
    assert all(d.evidence and d.evidence.endswith(m.TAG_EVIDENCE) for d in deps)
    assert deps[2].evidence == m.TAG_EVIDENCE
    rec = s.from_dict({**_record_dict(), "dependencies": [asdict(d) for d in deps]})
    assert s.loads(s.dumps(rec)) == rec
    # No tag: not observed (null), never "observed, none" ([]).
    assert m.dependencies_from_tags({m.GIT_COMMIT: SHA}) is None


@pytest.mark.parametrize(
    ("over", "problem"),
    [
        ({m.DEPENDENCIES_TAG: "{not json"}, "is not JSON"),
        ({m.DEPENDENCIES_TAG: '[{"name":"x"}]'}, "is not a core dependency list"),
        (
            {m.DEPENDENCIES_TAG: json.dumps([{**asdict(_dep()), "provenance": "verified"}])},
            "is not a core dependency list",
        ),
        ({m.CODE_TAG: '{"git_sha":"short"}'}, "is not a core code object"),
        ({m.FORMAT_TAG: "attestation.mlflow-tags/9"}, "attestation tags ignored"),
        ({m.FORMAT_TAG: ""}, "attestation tags ignored"),
    ],
)
def test_a_tag_that_does_not_parse_is_ignored_and_said(over, problem):
    reading = m.read_tags({**_tags(), m.GIT_COMMIT: SHA, **over})
    assert any(problem in p for p in reading.problems), reading.problems
    assert len(reading.problems) == len(set(reading.problems))  # each said once
    # What could still be read is: MLflow's own commit at least.
    assert reading.code.git_sha == SHA
