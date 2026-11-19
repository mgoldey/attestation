"""The conformance pytest plugin (`attestation_schema.pytest_plugin`), driven
through `pytester` exactly as a producer's CI would run it. Spec:
docs/superpowers/specs/2026-11-19-conformance-suite-design.md.
"""

from __future__ import annotations

import importlib.metadata as metadata
import tomllib
from pathlib import Path

import pytest
from conformance_fixtures import BROKEN_RULES, DATA, EXPECTED

pytest_plugins = ["pytester"]

PLUGIN = "attestation_schema.pytest_plugin"
PKG = Path(__file__).resolve().parents[1]


def _run(pytester: pytest.Pytester, *args: str) -> pytest.RunResult:
    return pytester.runpytest("-p", PLUGIN, "-p", "no:cacheprovider", *args)


def test_valid_fixtures_pass_one_item_each(pytester):
    result = _run(pytester, "--attestation-runs", str(DATA / "valid" / "*"), "-v")
    result.assert_outcomes(passed=3)
    result.stdout.fnmatch_lines(["*attestation-conformance::*valid/job-complete PASSED*"])


@pytest.mark.parametrize("rule", BROKEN_RULES)
def test_each_broken_fixture_fails_naming_rule_file_and_json_path(pytester, rule):
    run_dir = DATA / "broken" / rule
    result = _run(pytester, "--attestation-runs", str(run_dir))
    result.assert_outcomes(failed=1)
    file, json_path, fragment = EXPECTED[rule]
    out = result.stdout.str()
    assert f"{run_dir / file} at {json_path}: [{rule}]" in out, out
    assert fragment in out


def test_a_pattern_matching_nothing_fails_rather_than_passing_vacuously(pytester):
    result = _run(pytester, "--attestation-runs", str(DATA / "no-such" / "*"))
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*no run directory matched*no-such*"])


def test_ini_list_applies_without_explicit_paths_only(pytester):
    pytester.makeini(f"[pytest]\nattestation_runs =\n    {DATA / 'valid' / 'minimal'}\n")
    pytester.makepyfile(test_other="def test_other():\n    pass\n")
    _run(pytester).assert_outcomes(passed=2)
    _run(pytester, "test_other.py").assert_outcomes(passed=1)
    # The command line replaces the ini list, paths or not.
    _run(
        pytester, "test_other.py", "--attestation-runs", str(DATA / "broken" / "read")
    ).assert_outcomes(passed=1, failed=1)


def test_marker_selects_only_the_generated_items(pytester):
    pytester.makepyfile(test_other="def test_other():\n    pass\n")
    result = _run(
        pytester, "-m", "attestation_conformance", "--attestation-runs", str(DATA / "valid")
    )
    result.assert_outcomes(passed=3, deselected=1)


def test_fixture_checks_a_record_made_in_the_test(pytester):
    pytester.makepyfile(
        test_producer=f"""
import sys
sys.path.insert(0, {str(Path(__file__).parent)!r})
import pytest
import attestation_schema as s
from conformance_fixtures import _record, spec_for, SPEC_CONFLICT, DATA

def test_fresh_record_conforms(tmp_path, attestation_conformance):
    s.write(_record("fresh"), tmp_path)
    report = attestation_conformance(tmp_path)
    assert report.ok and report.record.run_id == "fresh"

def test_job_against_its_spec(attestation_conformance):
    attestation_conformance(DATA / "valid" / "job-complete", spec_for())

def test_job_against_another_spec_fails(attestation_conformance):
    attestation_conformance(DATA / "valid" / "job-complete", SPEC_CONFLICT)

def test_report_does_not_fail(attestation_conformance):
    assert attestation_conformance.report(DATA / "broken" / "read").rules == ["read"]
"""
    )
    result = _run(pytester)
    result.assert_outcomes(passed=3, failed=1)
    # fnmatch would read `[spec]` as a character class: compare text.
    assert "job-complete/attestation/run.json at $.config: [spec] config is not the" in (
        result.stdout.str()
    )


def test_without_p_the_plugin_is_absent(pytester):
    """Opt-in: a suite that merely has the package installed is unchanged."""
    pytester.makepyfile(test_x="def test_x(attestation_conformance):\n    pass\n")
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*fixture 'attestation_conformance' not found*"])
    result = pytester.runpytest("-p", "no:cacheprovider", "--attestation-runs", "x")
    assert result.ret == pytest.ExitCode.USAGE_ERROR


def test_no_pytest11_entry_point_and_pytest_only_as_an_extra():
    project = tomllib.loads((PKG / "pyproject.toml").read_text())["project"]
    assert "entry-points" not in project and "entry_points" not in project
    assert project["dependencies"] == []
    assert project["optional-dependencies"] == {"conformance": ["pytest>=7.2"]}
    installed = [
        ep for ep in metadata.entry_points(group="pytest11") if "attestation_schema" in ep.value
    ]
    assert installed == []
