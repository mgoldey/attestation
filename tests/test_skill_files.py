"""Content checks on the shipped research-provenance skill files.

No existing test reads setup.sh or SKILL.md content -- test_install.py and
test_install_e2e.py only assert the files *exist* (and exercise their own
fixture SKILL.md, never the shipped one). This file closes that gap for the
one failure mode that matters most: a skill file invoking a console script
pyproject.toml never declared, which execs silently on machines that happen
to have an unrelated same-named binary on PATH.

Checked against pyproject.toml's actual [project.scripts] via tomllib rather
than a hardcoded "hermes"/"attest" string, so this keeps working through the
next rename instead of quietly going stale itself.
"""

import re
import tomllib
from pathlib import Path

import attestation.install as install

_REPO_ROOT = Path(install.__file__).resolve().parent.parent.parent


def _declared_console_scripts() -> set[str]:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    return set(pyproject["project"]["scripts"])


def _skill_dir() -> Path:
    src = install._skill_source_dir()
    assert src.is_dir(), "shipped skill source dir must exist"
    return src


def _undeclared_exec_targets(text: str, declared: set[str]) -> set[str]:
    """Binary names actually invoked via `uv run [--project <dir>] <bin> ...`
    or `uvx --from <pkg> <bin> ...`, that are not among the declared console
    scripts.

    Deliberately narrow (requires a trailing subcommand/flag like `install`,
    `serve`, `--check`, `--yes`) so it matches only real invocation lines,
    not prose that merely mentions "uv run" or "uvx --from" while explaining
    the command shape (e.g. SKILL.md's "`uvx --from <package>` takes the
    *package*, and the trailing word is the *executable*").
    """
    found = set()
    pattern = re.compile(
        r"\buv run(?:\s+--project\s+\S+)?\s+(\w+)\s+(?:install|serve)\b"
        r"|\buvx --from\s+\S+\s+(\w+)\s+(?:install|serve)\b"
    )
    for match in pattern.finditer(text):
        word = match.group(1) or match.group(2)
        if word not in declared:
            found.add(word)
    return found


def test_setup_sh_invokes_only_declared_console_scripts():
    declared = _declared_console_scripts()
    setup_sh = (_skill_dir() / "scripts" / "setup.sh").read_text()

    bad = _undeclared_exec_targets(setup_sh, declared)

    assert not bad, (
        f"setup.sh execs {bad}, not among pyproject's declared [project.scripts] {declared}"
    )


def test_setup_sh_has_no_hermes_install_invocation():
    """Regression guard for the specific incident: both exec paths called
    `hermes install --yes`, a binary pyproject never shipped."""
    setup_sh = (_skill_dir() / "scripts" / "setup.sh").read_text()

    assert "hermes install" not in setup_sh


def test_skill_md_invokes_only_declared_console_scripts():
    declared = _declared_console_scripts()
    skill_md = (_skill_dir() / "SKILL.md").read_text()

    bad = _undeclared_exec_targets(skill_md, declared)

    assert not bad, (
        f"SKILL.md documents running {bad}, not among pyproject's declared "
        f"[project.scripts] {declared}"
    )


def test_skill_md_has_no_hermes_install_invocation():
    skill_md = (_skill_dir() / "SKILL.md").read_text()

    assert "hermes install" not in skill_md
