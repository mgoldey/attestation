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
import shutil
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
    assert "Hermes workspace" in text
    assert "~/.hermes/workspace" not in text, "ignores HERMES_HOME"
    # the run instruction must name an environment that has attestation in it
    assert "uv run --with jupyter jupyter nbconvert" in text
    assert "any" in text and "failure" in text and "RESULTS.md" in text
    assert "never re-runs, edits, or writes a document" not in text


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


@pytest.fixture
def private_tmp(monkeypatch, tmp_path):
    """The notebooks make a temp folder per run; keep it under pytest's tmp_path
    so runs do not pile up in the system temp dir."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_still_runs_where_the_repo_examples_are_absent(path, monkeypatch, private_tmp):
    """Installed on a machine without examples/: it must say so and use a
    stand-in rather than fail."""
    monkeypatch.setenv("ATTEST_EXAMPLE_WORKSPACE", "none")
    text = _printed(_rerun(path))[0]
    if path.name != "make-your-own-claim.ipynb":
        assert "is not on this machine" in text


def test_the_stand_in_workspace_teaches_the_same_lessons(monkeypatch, private_tmp):
    """What every installed-wheel reader gets. `attest()` never raises on a
    nonzero exit, so renaming a metric in the stand-in once printed `winner:
    None` and every test still passed: pin the lines that carry the lesson."""
    monkeypatch.setenv("ATTEST_EXAMPLE_WORKSPACE", "none")
    which = _printed(_rerun(NOTEBOOK_DIR / "which-arm-won.ipynb"))[0]
    for line in (
        "winner: kdsweep_t4",
        "caveat: the top two arms differ by 0.0017 (2.6%) -- too close to call",
        "caveat: each arm is a single run; no seed replication",
        "unknown direction for metric 'n_records' -- refusing to rank.",
    ):
        assert line in which, line
    assert "Skipped" not in which, "the refusal must be shown in both modes"
    check = _printed(_rerun(NOTEBOOK_DIR / "check-a-drafts-claims.ipynb"))[0]
    for verdict in ("supported", "contradicted", "unsupported", "malformed"):
        assert re.search(rf"\*\*{verdict}\*\*", check), verdict
    assert "1 contradicted" in check and "[exit 1]" in check


def test_a_checkouts_dotenv_and_hostile_env_cannot_change_the_refusal(monkeypatch, private_tmp):
    """Every `attest` subprocess calls load_env(), which fills in from the
    checkout's .env any variable the notebook merely popped. Pointing the
    variables at the notebook's own temp folder makes them win instead (the
    same lesson tests/conftest.py records: repoint, do not delete)."""
    hostile = private_tmp / "hostile_direction.toml"
    hostile.write_text('[metric_direction]\nn_records = "higher_is_better"\n')
    env_file = _REPO_ROOT / ".env"
    if env_file.exists():
        pytest.skip("this checkout has a real .env; not overwriting it")
    env_file.write_text(
        f"LEDGER_METRIC_DIRECTION_FILE={hostile}\n"
        f"LEDGER_CORPUS_FILE={private_tmp / 'hostile_corpus.toml'}\n"
        f"RESEARCH_ROOT={private_tmp}\n"
    )
    try:
        for var in ("LEDGER_METRIC_DIRECTION_FILE", "LEDGER_CORPUS_FILE", "RESEARCH_ROOT"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("ATTEST_EXAMPLE_WORKSPACE", raising=False)
        text = _printed(_rerun(NOTEBOOK_DIR / "which-arm-won.ipynb"))[0]
    finally:
        env_file.unlink()  # the file this test wrote, nothing else
    assert "refusing to rank" in text
    assert "winner: kdsweep_t4" in text
    assert str(private_tmp / "hostile") not in text


def test_a_missing_attest_says_which_environment_to_use_not_pip_install(monkeypatch, private_tmp):
    nb = _read(NOTEBOOK_DIR / "make-your-own-claim.ipynb")
    setup = _code_cells(nb)[0].source
    for var in ("ATTEST_DB", "HERMES_HOME", "RESEARCH_ROOT"):
        monkeypatch.setenv(var, "x")
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    with pytest.raises(RuntimeError) as err:
        exec(setup, {"__name__": "nb"})
    message = str(err.value)
    assert "pip install" not in message
    assert "attestation environment" in message


def test_every_notebook_says_to_copy_it_before_editing():
    for path in NOTEBOOKS:
        assert "Copy this notebook before editing" in _read(path).cells[0].source, path.name


def test_the_notebooks_teach_verdicts_as_the_checker_defines_them():
    text = (NOTEBOOK_DIR / "check-a-drafts-claims.ipynb").read_text()
    assert "stale: the run now says otherwise" not in text
    for needle in ("split=", "no such metric", "uncited"):
        assert needle in text, needle
    arm = (NOTEBOOK_DIR / "which-arm-won.ipynb").read_text()
    assert "cannot tell" not in arm, "goes beyond what the tool's caveat says"


# --------------------------------------------------------------------------
# the installer ships an ALLOWLIST of a skill's files
# --------------------------------------------------------------------------


def _install_into_fake_home(monkeypatch, tmp_path, src_root: Path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)
    monkeypatch.setattr(install, "_skills_source_root", lambda: src_root)
    result = install.step_skill_copy("agenthermes", check=False)
    return fake_home / ".hermes" / "skills", result


@pytest.fixture
def source_skills(tmp_path):
    """A copy of the real skills tree the test is free to litter."""
    root = tmp_path / "src-skills"
    shutil.copytree(install._skills_source_root(), root)
    return root


def test_a_merge_leftover_skill_md_is_not_copied_and_does_not_disable_the_skill(
    monkeypatch, tmp_path, source_skills
):
    (source_skills / SKILL / "SKILL.md.orig").write_text("a merge leftover")
    dest, _ = _install_into_fake_home(monkeypatch, tmp_path, source_skills)
    assert not (dest / SKILL / "SKILL.md.orig").exists()
    assert not install._skill_disabled(dest / SKILL)
    assert install.step_skill_copy("agenthermes", check=True).status == install.Status.OK


def test_symlinks_and_editor_litter_are_not_copied(monkeypatch, tmp_path, source_skills):
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("not part of the skill")
    notebooks = source_skills / SKILL / "notebooks"
    (notebooks / "link.md").symlink_to(outside)
    (notebooks / "Untitled~").write_text("autosave")
    (notebooks / "stray.pyc").write_bytes(b"\0")
    (notebooks / ".ipynb_checkpoints").mkdir()
    (notebooks / ".ipynb_checkpoints" / "x-checkpoint.ipynb").write_text("{}")
    (notebooks / "__pycache__").mkdir()
    (notebooks / "__pycache__" / "x.py").write_text("")
    (source_skills / SKILL / ".hidden.md").write_text("dot")
    dest, _ = _install_into_fake_home(monkeypatch, tmp_path, source_skills)
    shipped = {p.name for p in (dest / SKILL).rglob("*")}
    for litter in (
        "link.md",
        "Untitled~",
        "stray.pyc",
        ".ipynb_checkpoints",
        "__pycache__",
        ".hidden.md",
    ):
        assert litter not in shipped, litter
    assert "SKILL.md" in shipped and "RESULTS.md" in shipped
    assert (dest / "attestation-setup" / "scripts" / "setup.sh").is_file()
    for name in EXPECTED:
        assert (dest / SKILL / "notebooks" / name).is_file()


def test_an_edited_installed_notebook_is_a_note_not_a_broken_check(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)
    install.step_skill_copy("agenthermes", check=False)
    copy = fake_home / ".hermes" / "skills" / SKILL / "notebooks" / "which-arm-won.ipynb"
    copy.write_text(copy.read_text() + " ")
    result = install.step_skill_copy("agenthermes", check=True)
    assert result.status == install.Status.OK
    assert "which-arm-won.ipynb" in result.detail
