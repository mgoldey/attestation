"""The shared CI templates (roadmap 39): reusable workflows in this repository
that attestation calls by local path and smeltery and ferric call by a pinned
commit SHA. Parsed with yaml, not grepped. What is pinned:

- every workflow here pins each third-party action by a full commit SHA with a
  version comment (a moved tag cannot change what another repository runs);
- every reusable-*.yml is `on: workflow_call` ONLY, with typed inputs, a
  read-only token, and no secrets (a caller passes none and inherits none);
- an input a reusable workflow reads exists, and an input a caller passes is
  declared (a typo is otherwise silently ignored by GitHub at call time);
- attestation's own ci.yml and release.yml call each of them, so this
  repository's CI exercises the same files the other two call.

Spec: docs/superpowers/specs/2026-11-20-shared-ci-templates-design.md.
"""

import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github/workflows"
ALL = sorted(WORKFLOWS.glob("*.yml"))
REUSABLE = sorted(WORKFLOWS.glob("reusable-*.yml"))
CALLERS = [p for p in ALL if p not in REUSABLE]
INPUT_TYPES = {"string", "number", "boolean"}
PINNED = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
INPUT_REF = re.compile(r"inputs\.([A-Za-z0-9_-]+)")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _on(wf: dict) -> dict:
    return wf.get("on") or wf[True]  # PyYAML reads a bare `on` as True


def _step_uses(path: Path):
    """(uses, raw line) for every step-level `uses:` in the file."""
    lines = path.read_text().splitlines()
    for job in _load(path)["jobs"].values():
        for step in job.get("steps", []):
            if "uses" in step:
                raw = next(
                    ln for ln in lines if re.search(rf"uses:\s*{re.escape(step['uses'])}", ln)
                )
                yield step["uses"], raw


def test_the_three_templates_exist():
    assert [p.name for p in REUSABLE] == [
        "reusable-pre-commit.yml",
        "reusable-python-gates.yml",
        "reusable-wheel-smoke.yml",
    ]


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_every_third_party_action_is_sha_pinned_with_a_version_comment(path):
    seen = 0
    for uses, raw in _step_uses(path):
        assert not uses.startswith("./"), f"{path.name}: local action {uses}"
        assert PINNED.match(uses), f"{path.name}: {uses} is not pinned by a full commit SHA"
        assert re.search(r"#\s*v\d", raw), f"{path.name}: {uses} has no `# vX.Y.Z` comment"
        seen += 1
    if path in REUSABLE:
        assert seen, path.name


def test_one_sha_per_action_across_all_workflows():
    shas: dict[str, set[str]] = {}
    for path in ALL:
        for uses, _ in _step_uses(path):
            action, sha = uses.split("@")
            shas.setdefault(action, set()).add(sha)
    assert {a: s for a, s in shas.items() if len(s) > 1} == {}


@pytest.mark.parametrize("path", REUSABLE, ids=lambda p: p.name)
def test_reusable_workflows_are_workflow_call_only_with_typed_inputs(path):
    wf = _load(path)
    on = _on(wf)
    assert set(on) == {"workflow_call"}, path.name
    inputs = on["workflow_call"]["inputs"]
    assert inputs, path.name
    for name, spec in inputs.items():
        assert spec.get("type") in INPUT_TYPES, (path.name, name)
        assert spec.get("required") is True or "default" in spec, (path.name, name)
    assert "secrets" not in on["workflow_call"], path.name
    assert wf["permissions"] == {"contents": "read"}, path.name
    text = path.read_text()
    assert "secrets." not in text and "secrets: inherit" not in text
    # each input it reads is declared, and each declared input is read
    used = set(INPUT_REF.findall(text))
    assert used <= set(inputs), (path.name, used - set(inputs))
    assert set(inputs) <= used, (path.name, set(inputs) - used)


def _calls():
    for path in CALLERS:
        for job_id, job in _load(path)["jobs"].items():
            if "uses" in job:
                yield path, job_id, job


def test_callers_pass_only_declared_inputs_and_every_required_one():
    calls = list(_calls())
    assert calls
    for path, job_id, job in calls:
        target = WORKFLOWS / Path(job["uses"]).name
        assert job["uses"] == f"./.github/workflows/{target.name}", (path.name, job_id)
        inputs = _on(_load(target))["workflow_call"]["inputs"]
        given = set(job.get("with", {}))
        assert given <= set(inputs), (path.name, job_id, given - set(inputs))
        required = {n for n, s in inputs.items() if s.get("required")}
        assert required <= given, (path.name, job_id, required - given)
        assert "secrets" not in job, (path.name, job_id)


def test_attestation_ci_exercises_every_template():
    called = {Path(job["uses"]).name for _, _, job in _calls()}
    assert called == {p.name for p in REUSABLE}
    ci = {Path(j["uses"]).name for j in _load(WORKFLOWS / "ci.yml")["jobs"].values() if "uses" in j}
    assert ci == {p.name for p in REUSABLE}


def test_visible_check_names_keep_their_prefix():
    """A called job's check reads '<caller name> / <called name>'; the caller
    names are the ones branch protection saw before roadmap 39."""
    jobs = _load(WORKFLOWS / "ci.yml")["jobs"]
    assert jobs["gates"]["name"] == "gates (${{ matrix.os }}, py${{ matrix.python-version }})"
    assert jobs["wheel-smoke"]["name"] == "wheel smoke test"
    assert jobs["docs"]["name"] == "docs site"
    assert jobs["flows"]["name"] == "example flows (offline)"
    assert jobs["schema-py311"]["name"] == "attestation-schema on py3.11"
    assert jobs["schema-integrations"]["name"] == "attestation-schema training callbacks (py3.11)"


def test_gates_keep_every_gate_the_old_job_ran():
    w = _load(WORKFLOWS / "ci.yml")["jobs"]["gates"]["with"]
    assert w["python-preference"] == "only-managed"  # sqlite-vec on macOS
    assert w["sync-args"] == "--all-extras --dev"
    assert w["ty-command"] == "uv run --no-sync ty check src/ packages/attestation-schema/src/"
    assert w["lock-check"] is True
    for gate in (
        "scripts/check_complexity.py",
        "bandit -q -c pyproject.toml",
        "xenon --max-modules B",
    ):
        assert gate in w["checks"], gate
    assert w["test-command"] == "uv run --no-sync pytest -q"
    gates = _load(WORKFLOWS / "reusable-python-gates.yml")["jobs"]["gates"]["steps"]
    runs = "\n".join(str(s.get("run", "")) for s in gates)
    assert "format --check ." in runs and '"${ruff[@]}" check .' in runs
    # ruff and ty are blocking by default; report-only is an explicit opt-out
    inputs = _on(_load(WORKFLOWS / "reusable-python-gates.yml"))["workflow_call"]["inputs"]
    assert inputs["ruff-check-blocking"]["default"] is True
    assert inputs["ty-blocking"]["default"] is True
    assert inputs["lock-check"]["default"] is True


def test_no_template_publishes():
    """PyPI trusted publishing does not work from a reusable workflow
    (docs.pypi.org/trusted-publishers/troubleshooting/): publishing stays in
    each caller's top-level release workflow."""
    for path in REUSABLE:
        for uses, _ in _step_uses(path):
            assert not uses.startswith("pypa/gh-action-pypi-publish@"), path.name
        for job in _load(path)["jobs"].values():
            assert "id-token" not in (job.get("permissions") or {}), path.name
            assert "environment" not in job, path.name
