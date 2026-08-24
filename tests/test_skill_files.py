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


def _live_tool_names() -> set[str]:
    """Every tool name the MCP server actually registers.

    Derived from the live surface rather than a literal list. A literal list
    is exactly what let SKILL.md drift: it documented the tools someone
    remembered, and `cite.*` -- four tools, a whole namespace -- was never
    written down at all.
    """
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from attestation.mcp import register_all

    server = FastMCP("skill-surface-check")
    register_all(server)
    return {t.name for t in asyncio.run(server.list_tools())}


def test_skill_md_documents_every_live_namespace():
    """SKILL.md is what the agent reads; a namespace missing from it is invisible.

    Four reviewers found this file describing a world that no longer existed:
    `cite.*` was undocumented entirely, and the `.ask` routers -- the measured
    entry point, 13/15 against 8/15 for the flat surface -- appeared nowhere.
    Both were live when the file was last edited.

    Asserting on `<ns>.` catches the namespace being named as a tool prefix
    rather than merely as a word: "citations" in prose must not satisfy a
    check for `cite.*`.
    """
    skill_md = (_skill_dir() / "SKILL.md").read_text()
    namespaces = {name.split(".", 1)[0] for name in _live_tool_names()}

    missing = sorted(ns for ns in namespaces if f"{ns}." not in skill_md)

    assert not missing, (
        f"SKILL.md documents no {missing} tools, but the live MCP surface "
        f"registers that namespace. Add a section, or the agent reading this "
        f"skill will never know those tools exist."
    )


def test_skill_md_teaches_every_ask_router():
    """Each router, by name.

    Separate from the namespace check because a namespace can be documented
    thoroughly while its router is not: the flat tools are what an agent
    reaches for by default, and they measured worse.
    """
    skill_md = (_skill_dir() / "SKILL.md").read_text()

    missing = sorted(n for n in _live_tool_names() if n.endswith(".ask") and n not in skill_md)

    assert not missing, f"SKILL.md never mentions {missing}; agents will pick flat tools instead"


def test_the_presentation_example_covers_slack_not_just_markdown():
    """A worked example outweighs the rule written next to it.

    A Slack reader got their feed as five Markdown links and replied "the
    links weren't clickable": Slack `mrkdwn` renders `[title](url)` as literal
    text. The urls were all present -- the syntax was for another surface.

    The first fix added a paragraph saying so and left the single Markdown
    example above it. Measured on gemma4:e2b against the real file: 5 of 5
    items rendered, 0 of 5 clickable in Slack -- the model copied the example
    and the prose changed nothing. Only splitting the example block by surface
    moved it, to 5 of 5 clickable across four runs, while the Markdown surface
    stayed correct.

    So the guard is on the EXAMPLE, not on the prose. A future edit that
    collapses the block back to one Markdown line reintroduces the bug.
    """
    skill_md = (_skill_dir() / "SKILL.md").read_text()
    start = skill_md.index("**Present each item as one line")
    block = skill_md[start : skill_md.index("```", skill_md.index("```", start) + 3)]

    assert "<https://" in block and "|" in block, (
        "the presentation example shows no Slack-syntax link; a model in Slack "
        "copies the Markdown form and the reader gets unclickable text"
    )
    assert "](http" in block, "the Markdown example was dropped; it is still correct elsewhere"
    assert "Slack" in block, "the example block must label which line is for which surface"
