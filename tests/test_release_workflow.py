"""The release workflow publishes two distributions from one file, so the
properties that keep them from crossing are pinned here, parsed from
.github/workflows/release.yml rather than grepped: tag forms, version guards,
trusted publishing (OIDC, no tokens), and a tag filter pair that cannot both
match one tag. Spec: docs/superpowers/specs/2026-11-19-core-run-record-design.md,
Open decision 2."""

import re
from pathlib import Path

import yaml

TEXT = (Path(__file__).resolve().parent.parent / ".github/workflows/release.yml").read_text()
WF = yaml.safe_load(TEXT)
JOBS = WF["jobs"]
TAGS = (WF.get("on") or WF[True])["push"]["tags"]  # PyYAML reads a bare `on` as True

ATT = {"build": "build", "publish": "publish"}
SCH = {"build": "build-schema", "publish": "publish-schema"}


def _steps_text(job: str) -> str:
    return "\n".join(str(s.get("run", "")) for s in JOBS[job]["steps"])


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
    att = _steps_text("build")
    assert "GITHUB_REF_NAME#v}" in att and "'pyproject.toml'" in att
    sch = _steps_text("build-schema")
    assert "GITHUB_REF_NAME#schema-v}" in sch
    assert "packages/attestation-schema/pyproject.toml" in sch
    for text in (att, sch):
        assert 'test "$tag" = "$version"' in text and "exit 1" in text


def test_attestation_release_checks_the_schema_floor_on_pypi():
    text = _steps_text("build")
    assert "https://pypi.org/pypi/attestation-schema/json" in text
    assert "attestation-schema" in text and ">=" in text
    assert "cannot verify" in text  # unreachable PyPI fails, never passes silently
    assert "is not published on PyPI" in text
    # and the schema build does not need it
    assert "pypi.org/pypi" not in _steps_text("build-schema")


def test_schema_wheel_is_tested_on_311_before_publish():
    text = _steps_text("build-schema")
    assert "uv build --package attestation-schema" in text
    assert "--python 3.11" in text and "dist-schema/*.whl" in text and "pytest" in text
    assert JOBS["publish-schema"]["needs"] == "build-schema"
    assert JOBS["publish"]["needs"] == "build"


def test_both_publish_jobs_use_trusted_publishing():
    for job in ("publish", "publish-schema"):
        j = JOBS[job]
        assert j["environment"] == "pypi"
        assert j["permissions"] == {"id-token": "write"}
        uses = [s["uses"] for s in j["steps"] if "uses" in s]
        assert "pypa/gh-action-pypi-publish@release/v1" in uses
        for s in j["steps"]:
            assert "with" not in s or "password" not in s["with"]


def test_no_secrets_or_tokens():
    assert "secrets." not in TEXT
    assert not re.search(r"(?i)\b(password|api[_-]?token|PYPI_TOKEN|TWINE_)", TEXT)
