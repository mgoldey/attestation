"""The release workflow publishes two distributions from one file, so the
properties that keep them from crossing are pinned here, parsed from
.github/workflows/release.yml rather than grepped: tag forms, version guards,
trusted publishing (OIDC, no tokens), and a tag filter pair that cannot both
match one tag. Spec: docs/superpowers/specs/2026-11-19-core-run-record-design.md,
Open decision 2.

Since roadmap 39 (2026-11-20-shared-ci-templates-design.md) each build job
CALLS the shared reusable-wheel-smoke.yml, so a guard is pinned as the inputs a
build job passes plus the reusable step that acts on them; the publish jobs
stay top-level (PyPI trusted publishing does not work from a reusable
workflow)."""

import re
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github/workflows"
ROOT = WORKFLOWS.parent.parent
TEXT = (WORKFLOWS / "release.yml").read_text()
WF = yaml.safe_load(TEXT)
REUSABLE_PATH = "./.github/workflows/reusable-wheel-smoke.yml"
REUSABLE_TEXT = (WORKFLOWS / "reusable-wheel-smoke.yml").read_text()
REUSABLE = yaml.safe_load(REUSABLE_TEXT)
FLOOR_SCRIPT = "scripts/check_schema_floor.py"
JOBS = WF["jobs"]
TAGS = (WF.get("on") or WF[True])["push"]["tags"]  # PyYAML reads a bare `on` as True

ATT = {"build": "build", "publish": "publish"}
SCH = {"build": "build-schema", "publish": "publish-schema"}


def _with(job: str) -> dict:
    assert JOBS[job]["uses"] == REUSABLE_PATH, job
    return JOBS[job]["with"]


def _reusable_step(name: str) -> dict:
    steps = [s for s in REUSABLE["jobs"]["build"]["steps"] if s.get("name") == name]
    assert len(steps) == 1, name
    return steps[0]


def _job_text(job: str) -> str:
    """Everything a build job runs: its inputs, plus any script one names."""
    w = _with(job)
    text = "\n".join(f"{k}: {v}" for k, v in w.items())
    if FLOOR_SCRIPT in str(w.get("pre-build", "")):
        text += "\n" + (ROOT / FLOOR_SCRIPT).read_text()
    return text


def _glob(pattern: str, tag: str) -> bool:
    # GitHub: `*` matches any run of characters except `/`, anchored both ends.
    return re.fullmatch(re.escape(pattern).replace(r"\*", "[^/]*"), tag) is not None


def test_both_tag_forms_and_nothing_else():
    assert sorted(TAGS) == ["schema-v*", "v*"]


def test_tag_filters_are_disjoint():
    samples = ["v0.5.0", "v1", "vfoo", "schema-v0.1.0", "schema-v", "schema-vv1", "v1/schema-v1"]
    for tag in samples:
        assert sum(_glob(p, tag) for p in TAGS) <= 1, tag
    assert not _glob("v*", "schema-v0.1.0")
    assert not _glob("schema-v*", "v0.5.0")
    # the job-level guards agree with the filters
    for tag in samples:
        a = tag.startswith("v")
        s = tag.startswith("schema-v")
        assert not (a and s), tag


def test_jobs_are_split_by_tag_prefix():
    for names, prefix in ((ATT, "'v'"), (SCH, "'schema-v'")):
        for job in names.values():
            cond = JOBS[job]["if"]
            assert f"startsWith(github.ref_name, {prefix})" in cond, (job, cond)
    assert set(JOBS) == {"build", "publish", "build-schema", "publish-schema"}


def test_version_guards_name_each_packages_own_pyproject():
    att, sch = _with("build"), _with("build-schema")
    assert att["tag-prefix"] == "v" and att["version-file"] == "pyproject.toml"
    assert sch["tag-prefix"] == "schema-v"
    assert sch["version-file"] == "packages/attestation-schema/pyproject.toml"
    # the reusable guard: tag minus the prefix vs that file's [project].version
    static = _reusable_step("Tag matches the static version")
    assert "inputs.tag-prefix != ''" in static["if"] and "inputs.version-file != ''" in static["if"]
    run = static["run"]
    assert 'GITHUB_REF_NAME#"$PREFIX"}' in run and '"$VERSION_FILE"' in run
    assert 'test "$tag" = "$version"' in run and "exit 1" in run
    # and every built file carries the tag's version, with no local segment
    built = _reusable_step("Every built file carries the tag's version (no local segment)")
    assert built["if"] == "${{ inputs.tag-prefix != '' }}"
    assert "version != tag" in built["run"] and '"+" in version' in built["run"]
    assert "sys.exit" in built["run"]
    # guards run before the upload a publish job downloads
    names = [s.get("name") or s.get("uses", "") for s in REUSABLE["jobs"]["build"]["steps"]]
    upload = next(i for i, n in enumerate(names) if n.startswith("actions/upload-artifact@"))
    assert names.index(static["name"]) < names.index("Build") < names.index(built["name"]) < upload


def test_attestation_release_checks_the_schema_floor_on_pypi():
    assert _with("build")["pre-build"] == f"python3 {FLOOR_SCRIPT}"
    assert _reusable_step("Before build")["if"] == "${{ inputs.pre-build != '' }}"
    text = _job_text("build")
    assert "https://pypi.org/pypi/attestation-schema/json" in text
    assert "attestation-schema" in text and ">=" in text
    assert "cannot verify" in text  # unreachable PyPI fails, never passes silently
    assert "is not published on PyPI" in text
    # and the schema build does not need it
    assert "pre-build" not in _with("build-schema")
    assert "pypi.org/pypi" not in _job_text("build-schema")


def test_schema_wheel_is_tested_on_311_before_publish():
    w = _with("build-schema")
    assert w["build-command"].startswith("uv build --package attestation-schema")
    assert w["dist-dir"] == "dist-schema" and w["smoke-python-versions"] == '["3.11"]'
    text = w["smoke-command"]
    assert "--python 3.11" in text and "dist-schema/*.whl" in text and "pytest" in text
    # each publish downloads the artifact its own build uploaded
    for build, publish in (("build", "publish"), ("build-schema", "publish-schema")):
        dl = next(s for s in JOBS[publish]["steps"] if "download-artifact" in s.get("uses", ""))
        assert dl["with"]["name"] == _with(build)["artifact-name"]
    assert JOBS["publish-schema"]["needs"] == "build-schema"
    assert JOBS["publish"]["needs"] == "build"


def test_both_publish_jobs_use_trusted_publishing():
    for job in ("publish", "publish-schema"):
        j = JOBS[job]
        assert j["environment"] == "pypi"
        assert j["permissions"] == {"id-token": "write"}
        assert "uses" not in j  # top-level: trusted publishing cannot run in a reusable workflow
        uses = [s["uses"] for s in j["steps"] if "uses" in s]
        pub = [u for u in uses if u.startswith("pypa/gh-action-pypi-publish@")]
        assert len(pub) == 1 and re.fullmatch(r"pypa/gh-action-pypi-publish@[0-9a-f]{40}", pub[0])
        for s in j["steps"]:
            assert "with" not in s or "password" not in s["with"]


def test_no_secrets_or_tokens():
    for text in (TEXT, REUSABLE_TEXT):
        assert "secrets." not in text
        assert not re.search(r"(?i)\b(password|api[_-]?token|PYPI_TOKEN|TWINE_)", text)
    for job in JOBS.values():
        assert "secrets" not in job
