"""The executed demo notebooks that ship inside `attestation-provenance`.

A notebook is worth shipping only if it is (a) really executed -- outputs a
person typed would teach a verdict the tool never gave, (b) stable -- a path,
user name or timestamp in an output makes every regeneration a noisy diff and
leaks a machine, (c) delivered -- the installer once copied only SKILL.md and
scripts/, so a notebook beside them never reached the researcher, and (d)
truthful about the CLI -- a notebook that runs `attest claims_check` teaches
the same wrong command the skill's own test guards against.

`scripts/build_skill_notebooks.py` regenerates the notebooks and RESULTS.md by
executing them; these tests fail when the committed files and a fresh run
disagree.
"""

import getpass
import importlib.util
import re
import shlex
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

import attestation.install as install
from attestation.cli import build_parser

_REPO_ROOT = Path(install.__file__).resolve().parent.parent.parent
SKILL = "attestation-provenance"
NOTEBOOK_DIR = install._skill_source_dir(SKILL) / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
EXPECTED = {
    "check-a-drafts-claims.ipynb",
    "which-arm-won.ipynb",
    "make-your-own-claim.ipynb",
}


def _read(path: Path) -> nbformat.NotebookNode:
    return nbformat.read(path, as_version=4)


def _code_cells(nb):
    return [c for c in nb.cells if c.cell_type == "code"]


def test_the_expected_notebooks_and_a_rendered_result_ship():
    assert {p.name for p in NOTEBOOKS} == EXPECTED
    assert (NOTEBOOK_DIR / "RESULTS.md").is_file()


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_nbformat_and_fully_executed(path):
    nb = _read(path)
    nbformat.validate(nb)
    cells = _code_cells(nb)
    assert cells, "a demo notebook with no code teaches nothing"
    for cell in cells:
        assert cell.execution_count, f"{path.name}: {cell.id} was never executed"
        assert cell.outputs, f"{path.name}: {cell.id} has no saved output"
        assert not [o for o in cell.outputs if o.output_type == "error"], (
            f"{path.name}: {cell.id} saved a traceback"
        )


@pytest.mark.parametrize("path", NOTEBOOKS + [NOTEBOOK_DIR / "RESULTS.md"], ids=lambda p: p.name)
def test_notebook_carries_no_machine_paths_user_names_or_timestamps(path):
    text = path.read_text()
    assert not re.search(r"(?<![\w.<~])/(home|tmp|Users|root|var|private)/", text), "machine path"
    assert not re.search(r"[A-Za-z]:\\\\(Users|Documents)", text), "windows path"
    for name in {getpass.getuser(), "matt", "mgoldey"}:
        assert not re.search(rf"\b{re.escape(name)}\b", text, re.I), f"user name {name!r}"
    assert not re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", text), "timestamp"
    assert not re.search(r"\b\d{1,2}:\d{2}:\d{2}\b", text), "clock time"
    assert not re.search(r"attest-demo-\w+", text), "temp folder name leaked"
    if path.suffix == ".ipynb":
        nb = _read(path)
        for cell in nb.cells:
            assert "execution" not in cell.metadata, "per-cell timing makes every run a diff"


def _attest_commands(nb) -> list[str]:
    """Commands a code cell RUNS (parsed with their arguments) and commands the
    prose NAMES (parsed with `--help`: prose says `attest runs compare`, no family)."""
    found = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            found += re.findall(r'\battest\("([^"]+)"', cell.source)
        else:
            found += [f"{c} --help" for c in re.findall(r"`attest ([a-z][^`]*)`", cell.source)]
    return found


def test_every_attest_command_a_notebook_runs_parses_against_the_cli():
    """Parsed with its real arguments, not just `--help`: the notebooks call
    the commands, so a renamed flag must fail here and not in a reader's kernel."""
    parser = build_parser()
    seen = 0
    for path in NOTEBOOKS:
        for command in _attest_commands(_read(path)):
            seen += 1
            try:
                parser.parse_args(shlex.split(command))
            except SystemExit as exc:
                if exc.code == 0:  # `--help` on a real command exits 0
                    continue
                pytest.fail(f"{path.name} runs `attest {command}`, which the CLI rejects ({exc})")
    assert seen >= 8, "expected the notebooks' attest calls to be found and checked"


def test_the_installer_copies_the_notebooks_with_the_skill(monkeypatch, tmp_path):
    """Pattern of tests/test_install_skills.py: install into a fake home and
    look for the files where Hermes reads them -- including a profile's tree."""
    fake_home = tmp_path / "home"
    profile_skills = fake_home / ".hermes" / "profiles" / "research" / "skills"
    profile_skills.mkdir(parents=True)
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)

    install.step_skill_copy("agenthermes", check=False)

    for root in (fake_home / ".hermes" / "skills", profile_skills):
        for name in [*EXPECTED, "RESULTS.md"]:
            copied = root / SKILL / "notebooks" / name
            assert copied.is_file(), f"{copied.relative_to(fake_home)} was not installed"
            assert copied.read_bytes() == (NOTEBOOK_DIR / name).read_bytes()
    # and `--check` agrees the tree is current, rather than reporting the
    # notebooks stale forever
    assert install.step_skill_copy("agenthermes", check=True).status == install.Status.OK


def test_check_reports_a_missing_notebook(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)
    install.step_skill_copy("agenthermes", check=False)
    (fake_home / ".hermes" / "skills" / SKILL / "notebooks" / "which-arm-won.ipynb").unlink()

    assert install.step_skill_copy("agenthermes", check=True).status == install.Status.BROKEN


def test_the_skill_tells_an_agent_about_every_notebook():
    text = install._skill_source_dir(SKILL).joinpath("SKILL.md").read_text()
    assert "## Notebook demos" in text
    assert "notebooks/RESULTS.md" in text
    assert "~/.hermes/workspace/" in text


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_skill_notebooks", _REPO_ROOT / "scripts" / "build_skill_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_results_md_is_rendered_from_the_saved_outputs():
    """RESULTS.md is never typed: it must equal what the builder renders from
    the notebooks as committed."""
    builder = _load_builder()
    assert (NOTEBOOK_DIR / "RESULTS.md").read_text() == builder.render_results()


def _rerun(path: Path) -> nbformat.NotebookNode:
    nb = _read(path)
    NotebookClient(
        nb,
        timeout=60,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    return nb


def _printed(nb) -> list[str]:
    out = []
    for cell in _code_cells(nb):
        for o in cell.outputs:
            if o.output_type == "stream":
                out.append(o.text)
            elif "text/markdown" in o.get("data", {}):
                out.append(o.data["text/markdown"])
    return ["".join(out)]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_saved_outputs_are_what_the_notebook_prints_today(path, monkeypatch):
    """A fresh run (offline, a few seconds) against the repo's example
    workspace must print what was committed. Fails when the claim checker, the
    ledger or the example changes and the notebooks were not regenerated."""
    monkeypatch.delenv("ATTEST_EXAMPLE_WORKSPACE", raising=False)
    assert _printed(_rerun(path)) == _printed(_read(path)), (
        f"{path.name} is stale: run `uv run python scripts/build_skill_notebooks.py`"
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_still_runs_where_the_repo_examples_are_absent(path, monkeypatch):
    """Installed on a machine without examples/: it must say so and use a
    stand-in rather than fail."""
    monkeypatch.setenv("ATTEST_EXAMPLE_WORKSPACE", "none")
    nb = _rerun(path)
    text = _printed(nb)[0]
    if path.name != "make-your-own-claim.ipynb":
        assert "is not on this machine" in text
    if path.name == "check-a-drafts-claims.ipynb":
        for verdict in ("supported", "contradicted", "unsupported", "malformed"):
            assert verdict in text
