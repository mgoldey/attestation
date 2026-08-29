"""Every golden path has the same shape, and the documented commands are the
tested commands. Discovery is by directory: adding examples/<name>/README.md
enrols a path in every check here, with no edit to this file.

Spec: docs/superpowers/specs/2026-08-28-golden-paths-design.md."""

import os
import re
import subprocess
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"
CATALOGUE = EXAMPLES / "README.md"
SECTIONS = [
    "What you get",
    "Prerequisites",
    "Run it",
    "What it prints",
    "What it demonstrates",
    "When it goes wrong",
    "Next",
]
LABELS = {"none — pure local computation", "a model server at LLM_BASE_URL", "network"}
COMMAND_PREFIXES = ("uv run", "attest", "./run.sh", "export ", "ATTEST_")
FORBIDDEN = ("/home/", "github.com", "mlflow.user", "git@")
TEXT_SUFFIXES = {
    ".md",
    ".sh",
    ".py",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".txt",
    ".bib",
    ".csv",
    ".lock",
    "",
}


def paths() -> list[Path]:
    return sorted(p for p in EXAMPLES.glob("*/README.md"))


def sections(text: str) -> list[str]:
    return re.findall(r"^## (.+?)\s*$", text, re.M)


def _section(text: str, name: str) -> str:
    m = re.search(rf"^## {re.escape(name)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


def prerequisite(text: str) -> str:
    body = _section(text, "Prerequisites").strip().splitlines()
    return body[0].strip().strip("`") if body else ""


def _fenced(block: str) -> list[str]:
    out, inside = [], False
    for line in block.splitlines():
        if line.startswith("```"):
            inside = not inside
            continue
        if inside:
            out.append(line.rstrip())
    return out


def run_commands(text: str) -> list[str]:
    return [
        line.strip()
        for line in _fenced(_section(text, "Run it"))
        if line.strip().startswith(COMMAND_PREFIXES)
    ]


def pinned_line(text: str) -> str:
    lines = [line for line in _fenced(_section(text, "What it prints")) if line.strip()]
    return lines[0].strip() if lines else ""


def catalogue_rows() -> dict[str, str]:
    rows = {}
    for line in CATALOGUE.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0].startswith("`") and cells[0].endswith("/`"):
            rows[cells[0].strip("`").rstrip("/")] = cells[2].strip("`")
    return rows


@pytest.mark.parametrize("readme", paths(), ids=lambda p: p.parent.name)
def test_the_seven_sections_in_order(readme):
    assert sections(readme.read_text()) == SECTIONS, readme


@pytest.mark.parametrize("readme", paths(), ids=lambda p: p.parent.name)
def test_the_prerequisite_is_one_of_three_honest_labels(readme):
    assert prerequisite(readme.read_text()) in LABELS, readme


@pytest.mark.parametrize("readme", paths(), ids=lambda p: p.parent.name)
def test_the_readme_commands_are_the_run_sh_commands(readme):
    script = readme.parent / "run.sh"
    assert script.is_file() and os.access(script, os.X_OK), f"{script} missing or not executable"
    text = script.read_text()
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text
    commands = run_commands(readme.read_text())
    assert commands, f"{readme}: Run it has no commands"
    for cmd in commands:
        assert cmd in text, f"{readme.parent.name}: README command not in run.sh: {cmd!r}"


@pytest.mark.parametrize("readme", paths(), ids=lambda p: p.parent.name)
def test_every_path_has_a_catalogue_row_with_the_same_label(readme):
    rows = catalogue_rows()
    name = readme.parent.name
    assert name in rows, f"{name} has no row in examples/README.md"
    assert rows[name] == prerequisite(readme.read_text()), name


def test_the_catalogue_lists_only_paths_that_exist():
    for name in catalogue_rows():
        assert (EXAMPLES / name / "README.md").is_file(), name


# Two rules narrow this guard so it catches real leaks, not collisions with
# unrelated content: (1) the username matches only as a whole word, case
# sensitive -- a bare substring match on a short name (e.g. "matt") also
# matches ordinary English ("mattered", "matters"), which is not attribution.
# (2) `mlflow.user` and `git@` are checked only outside .py/.sh source: a
# scrubber's own code and docs must be free to NAME the tag or prefix it
# strips (train_mlflow.py's _SCRUB_TAGS names "mlflow.user" verbatim, which
# is the tag's real name, not a leaked value) -- `/home/` and `github.com`
# have no such legitimate source-code use and still apply to every file.
def test_no_committed_example_carries_attribution_or_machine_paths():
    user = os.environ.get("USER", "")
    hits = []
    for f in EXAMPLES.rglob("*"):
        if not f.is_file() or f.suffix not in TEXT_SUFFIXES or "__pycache__" in f.parts:
            continue
        text = f.read_text(errors="replace")
        is_source = f.suffix in (".py", ".sh")
        needles = tuple(n for n in FORBIDDEN if n not in ("mlflow.user", "git@") or not is_source)
        for needle in needles:
            if needle in text:
                hits.append(f"{f.relative_to(EXAMPLES)}: {needle}")
        if len(user) >= 4 and re.search(rf"\b{re.escape(user)}\b", text):
            hits.append(f"{f.relative_to(EXAMPLES)}: {user}")
    assert not hits, "\n".join(hits)


def _offline(readme: Path) -> bool:
    return prerequisite(readme.read_text()).startswith("none")


@pytest.mark.parametrize("readme", [p for p in paths() if _offline(p)], ids=lambda p: p.parent.name)
def test_an_offline_path_runs_green_and_prints_its_pinned_line(readme, tmp_path):
    env = {**os.environ, "HOME": str(tmp_path), "LLM_BASE_URL": "http://127.0.0.1:9/v1"}
    for var in ("ATTEST_DB", "RSS_DB", "RESEARCH_ROOT"):
        env.pop(var, None)
    proc = subprocess.run(
        [str(readme.parent / "run.sh")], env=env, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, (
        f"{readme.parent.name} run.sh failed:\n{proc.stdout}\n{proc.stderr}"
    )
    pin = pinned_line(readme.read_text())
    assert pin, f"{readme.parent.name}: What it prints has no fenced line to pin"
    assert pin in proc.stdout, f"{readme.parent.name}: pinned line not in output: {pin!r}"
