"""The stdlib conformance checker (`attestation_schema.conformance`).

Each broken golden fixture is the mutant for its rule's check: the valid
fixtures must produce no violation of that rule AND the broken one must
produce exactly it, so a check that always passes or always fails is caught
(docs/measurement-lessons.md, 4). Spec:
docs/superpowers/specs/2026-11-19-conformance-suite-design.md.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import attestation_schema as s
import pytest
from attestation_schema import conformance as c
from attestation_schema import layout
from conformance_fixtures import (
    BROKEN_RULES,
    DATA,
    EXPECTED,
    SPEC_CONFLICT,
    build,
    spec_for,
)

VALID = sorted((DATA / "valid").iterdir())


def _tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_fixtures_are_what_the_writers_produce(tmp_path):
    build(tmp_path)
    assert _tree(tmp_path) == _tree(DATA), (
        "tests/data/conformance is stale: run python packages/attestation-schema/tests/"
        "conformance_fixtures.py and commit the result"
    )


def test_every_rule_but_spec_has_a_broken_fixture():
    assert set(BROKEN_RULES) | {"spec"} == set(c.RULES)
    assert {p.name for p in (DATA / "broken").iterdir()} == set(BROKEN_RULES)


@pytest.mark.parametrize("run_dir", VALID, ids=lambda p: p.name)
def test_valid_fixtures_conform(run_dir):
    report = c.check_run_dir(run_dir)
    assert report.ok, report.format()


@pytest.mark.parametrize("rule", BROKEN_RULES)
def test_each_broken_fixture_breaks_exactly_its_rule(rule):
    run_dir = DATA / "broken" / rule
    report = c.check_run_dir(run_dir)
    assert [v.rule for v in report.violations] == [rule], report.format()
    (v,) = report.violations
    file, json_path, fragment = EXPECTED[rule]
    assert v.file == str(run_dir / file)
    assert v.json_path == json_path
    assert fragment in v.message
    # The rendered message names the record path and the JSON path.
    assert str(run_dir / file) in str(v) and json_path in str(v) and f"[{rule}]" in str(v)


def test_spec_rule_needs_a_spec_and_catches_a_conflict():
    run_dir = DATA / "valid" / "job-complete"
    assert c.check_run_dir(run_dir, spec=spec_for()).ok
    report = c.check_run_dir(run_dir, spec=SPEC_CONFLICT)
    assert [(v.rule, v.json_path) for v in report.violations] == [("spec", "$.config")]
    report = c.check_run_dir(run_dir, spec=spec_for(arm="lr=1e-2"))
    assert [(v.rule, v.json_path) for v in report.violations] == [("spec", "$.arm")]


def test_assert_conformant_raises_an_assertion_listing_every_violation(tmp_path):
    with pytest.raises(c.ConformanceError) as exc:
        c.assert_conformant(DATA / "broken" / "credentials")
    assert isinstance(exc.value, AssertionError)
    assert "$.config.tracking_uri" in str(exc.value)
    assert c.assert_conformant(DATA / "valid" / "minimal").ok


def test_every_violation_is_reported_not_just_the_first(tmp_path):
    rd = tmp_path / "r"
    shutil.copytree(DATA / "broken" / "source", rd)
    raw = (rd / layout.RUN_JSON).read_text()
    raw = raw.replace('"ext":{}', '"ext":{"Bad":{},"attestation.x":{}}')
    raw = raw.replace('"config":{"note":"bare"}', '"config":{"db_password":"pw1"}')
    (rd / layout.RUN_JSON).write_text(raw)
    report = c.check_run_dir(rd)
    got = sorted((v.rule, v.json_path) for v in report.violations)
    assert got == [
        ("credentials", "$.config.db_password"),
        ("ext-namespace", "$.ext.Bad"),
        ("ext-namespace", '$.ext["attestation.x"]'),
        ("source", "$.dependencies[0].source"),
    ], report.format()


def test_not_a_run_directory_is_a_violation(tmp_path):
    report = c.check_run_dir(tmp_path)
    assert report.rules == ["read"]
    assert "not a run directory" in report.violations[0].message


def test_an_unbound_manifest_and_an_unbound_sidecar_are_violations(tmp_path):
    rd = tmp_path / "r"
    (rd / "out.txt").parent.mkdir(parents=True)
    (rd / "out.txt").write_text("x")
    layout.write_manifest(rd)
    s.write(_bare("r"), rd)  # data: [] -- the manifest is not bound
    (rd / layout.HISTORY_JSONL).write_text('{"schema":"attestation.history/1"}\n')
    report = c.check_run_dir(rd)
    assert sorted(report.rules) == ["history", "manifest"], report.format()


def test_a_bound_manifest_that_is_missing_is_a_violation(tmp_path):
    rd = tmp_path / "r"
    entry = layout.write_manifest(rd)
    s.write(dataclasses.replace(_bare("r"), data=[entry]), rd)
    (rd / layout.DIGESTS_JSON).unlink()
    report = c.check_run_dir(rd)
    assert [(v.rule, v.json_path) for v in report.violations] == [("manifest", "$.data[0]")]


def test_verify_outputs_false_skips_only_the_rehash():
    rd = DATA / "broken" / "manifest"
    assert c.check_run_dir(rd, verify_outputs=False).ok


def test_a_job_record_with_null_status_and_a_disagreeing_state(tmp_path):
    rd = tmp_path / "r"
    shutil.copytree(DATA / "valid" / "job-complete", rd)
    raw = (rd / layout.RUN_JSON).read_text().replace('"status":"complete"', '"status":"failed"')
    (rd / layout.RUN_JSON).write_text(raw)
    report = c.check_run_dir(rd)
    assert ("terminal", "$.state") in [(v.rule, v.json_path) for v in report.violations]
    raw = raw.replace('"status":"failed"', '"status":null')
    (rd / layout.RUN_JSON).write_text(raw)
    report = c.check_run_dir(rd)
    assert ("terminal", "$.status") in [(v.rule, v.json_path) for v in report.violations]


def test_a_lagging_state_beside_a_record_is_allowed(tmp_path):
    """state.json may lag; run.json is authoritative (core spec)."""
    rd = tmp_path / "r"
    s.init_state(rd, run_id="r", backend="local", job_id="j", at="2026-11-19T10:00:00.000000Z")
    s.write(_bare("r"), rd)
    assert c.check_run_dir(rd).ok


def _bare(run_id: str) -> s.RunRecord:
    return s.RunRecord(
        run_id=run_id,
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
    )


# Token shapes are assembled at runtime so no literal in the repository looks
# like a real secret to a scanner.
_POSITIVE = [
    "https://user:" + "pw@example.invalid/x",
    "https://token" + "@example.invalid/x.git",
    "git+https://u:" + "p@example.invalid/x",
    "https://bucket.example.invalid/f.whl?X-Amz-Signature=" + "ab12",
    "https://example.invalid/api?access_token=" + "abc",
    "AKIA" + "ABCDEFGHIJKLMNOP",
    "ghp_" + "a" * 36,
    "xoxb-" + "1234567890-abc",
    "hf_" + "b" * 34,
    "sk-" + "c" * 32,
    "-----BEGIN RSA " + "PRIVATE KEY-----",
]
_NEGATIVE = [
    "https://example.invalid/x.git",
    "ssh://git@example.invalid/x.git",
    "git@example.invalid:x.git",
    "https://example.invalid/data?page=2",
    "sk-learn",
    "the tokenizer is bpe",
    "https://example.invalid/#sha256=" + "d" * 64,
]


@pytest.mark.parametrize("text", _POSITIVE)
def test_credential_shapes_are_found(text):
    assert c.credential_in(text), text


@pytest.mark.parametrize("text", _NEGATIVE)
def test_ordinary_strings_are_not_credentials(text):
    assert c.credential_in(text) is None


def test_secret_named_keys_unless_redacted():
    found = c.credential_findings(
        {
            "db": {"password": "pw1", "PASSWORD_hint": "x"},
            "wandb_api_key": "k",
            "api_key": "${WANDB_API_KEY}",
            "client_secret": "***",
            "token": "<pad>",
            "pad_token": "<pad>",
            "secret_key": "REDACTED",
            "private_key": "",
        }
    )
    assert sorted(p for p, _ in found) == ["$.db.password", "$.wandb_api_key"]


def test_find_targets_reports_a_pattern_matching_nothing(tmp_path):
    dirs, empty = c.find_targets(["valid/*", "nope/*", "broken/terminal"], DATA)
    assert empty == ["nope/*"]
    assert [d.name for d in dirs] == ["in-flight", "job-complete", "minimal", "terminal"]
    dirs, empty = c.find_targets(["**/run.json"], DATA / "valid")
    assert sorted(d.name for d in dirs) == ["job-complete", "minimal"] and empty == []
    dirs, _ = c.find_targets(["."], DATA / "valid")
    assert len(dirs) == 3
