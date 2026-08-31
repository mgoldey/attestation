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


def _executed_lines(script_text: str) -> list[str]:
    return [
        line
        for line in script_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize("readme", paths(), ids=lambda p: p.parent.name)
def test_the_readme_commands_are_the_run_sh_commands(readme):
    script = readme.parent / "run.sh"
    assert script.is_file() and os.access(script, os.X_OK), f"{script} missing or not executable"
    text = script.read_text()
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text
    commands = run_commands(readme.read_text())
    assert commands, f"{readme}: Run it has no commands"
    executed = _executed_lines(text)
    for cmd in commands:
        assert any(cmd in line for line in executed), (
            f"{readme.parent.name}: README command appears only in a comment or not at all"
            f" in run.sh: {cmd!r}"
        )


@pytest.mark.parametrize("readme", paths(), ids=lambda p: p.parent.name)
def test_every_path_has_a_catalogue_row_with_the_same_label(readme):
    rows = catalogue_rows()
    name = readme.parent.name
    assert name in rows, f"{name} has no row in examples/README.md"
    assert rows[name] == prerequisite(readme.read_text()), name


def test_the_catalogue_lists_only_paths_that_exist():
    for name in catalogue_rows():
        assert (EXAMPLES / name / "README.md").is_file(), name


def test_the_catalogue_is_ordered_by_prerequisite_then_name():
    # `none` paths first, alphabetical within each prerequisite group.
    # catalogue_rows() is a dict, so its iteration order is the table's own
    # row order -- this asserts against that order directly rather than
    # re-deriving it, so a future re-sort of the table is what this test
    # exists to catch.
    order = {"none — pure local computation": 0, "network": 1, "a model server at LLM_BASE_URL": 2}
    rows = list(catalogue_rows().items())
    expected = sorted(rows, key=lambda row: (order[row[1]], row[0]))
    assert rows == expected, rows


# Two rules narrow this guard so it catches real leaks, not collisions with
# unrelated content: (1) the username matches only as a whole word, case
# sensitive -- a bare substring match on a short name (e.g. "matt") also
# matches ordinary English ("mattered", "matters"), which is not attribution.
# (2) `mlflow.user` and `git@` are checked only outside .py/.sh/.md source: a
# scrubber's own code and its docs must be free to NAME the tag or prefix it
# strips (train_mlflow.py's _SCRUB_TAGS names "mlflow.user" verbatim, which
# is the tag's real name, not a leaked value, and a README explaining the
# scrub needs the same freedom in prose) -- `/home/` and `github.com` have
# no such legitimate source-or-prose use and still apply to every file.
#
# A non-text suffix (e.g. tensorboard's `events.out.tfevents.<ts>.v2`) is
# never decoded as text -- it is scanned as raw bytes instead, for `/home/`
# and the username as a whole word, since a binary format can still embed a
# plain attribution string (a hostname, a username) even though it is not
# itself readable text.
#
# The ambient-username needle exists to catch the developer's own name
# leaking onto their own machine's clone -- it is not meant to fire on a CI
# runner's generic service account. GitHub Actions sets `$USER` to `runner`,
# and "runner" is an ordinary English word that appears in prose (e.g.
# hydra/README.md's "GitHub Actions runner") -- run 33233059347 failed on
# exactly that collision. So the ambient-username check is skipped outright
# under `CI` (any value; that env var is set by every common CI provider,
# not just GitHub Actions), and also for generic account names that are not
# a person's identity even off CI. The fixed needles (`/home/`, `github.com`,
# and the tag-name ones with their .py/.sh/.md exemption) still apply
# everywhere, on CI or not -- only the ambient-username heuristic is scoped.
_GENERIC_ACCOUNT_NAMES = {"runner", "root", "user", "ubuntu", "admin", "ci"}


def _skip_ambient_username(user: str) -> bool:
    return (
        len(user) < 4 or os.environ.get("CI") is not None or user.lower() in _GENERIC_ACCOUNT_NAMES
    )


def test_no_committed_example_carries_attribution_or_machine_paths():
    user = os.environ.get("USER", "")
    check_user = not _skip_ambient_username(user)
    user_bytes_re = re.compile(rb"\b" + re.escape(user.encode()) + rb"\b") if check_user else None
    hits = []
    for f in EXAMPLES.rglob("*"):
        if not f.is_file() or "__pycache__" in f.parts:
            continue
        if f.suffix not in TEXT_SUFFIXES:
            data = f.read_bytes()
            if b"/home/" in data:
                hits.append(f"{f.relative_to(EXAMPLES)}: /home/")
            if user_bytes_re and user_bytes_re.search(data):
                hits.append(f"{f.relative_to(EXAMPLES)}: {user}")
            continue
        text = f.read_text(errors="replace")
        is_source = f.suffix in (".py", ".sh", ".md")
        needles = tuple(n for n in FORBIDDEN if n not in ("mlflow.user", "git@") or not is_source)
        for needle in needles:
            if needle in text:
                hits.append(f"{f.relative_to(EXAMPLES)}: {needle}")
        if check_user and re.search(rf"\b{re.escape(user)}\b", text):
            hits.append(f"{f.relative_to(EXAMPLES)}: {user}")
    assert not hits, "\n".join(hits)


def test_the_ambient_username_guard_is_skipped_on_ci(monkeypatch):
    # hydra/README.md's prose contains "runner" (as in "a GitHub Actions
    # runner"); on a CI box where $USER=runner, that word must not be
    # flagged as leaked attribution. Reproduce the CI run's own conditions
    # by monkeypatching CI/USER rather than depending on the real ambient
    # environment, so this test means the same thing locally and in CI.
    monkeypatch.setenv("CI", "1")
    monkeypatch.setenv("USER", "runner")
    assert "runner" in (EXAMPLES / "hydra" / "README.md").read_text()
    test_no_committed_example_carries_attribution_or_machine_paths()


def _offline(readme: Path) -> bool:
    return prerequisite(readme.read_text()).startswith("none")


def run_the_path(readme: Path, tmp_path: Path) -> str:
    env = {**os.environ, "HOME": str(tmp_path), "LLM_BASE_URL": "http://127.0.0.1:9/v1"}
    for var in ("ATTEST_DB", "RSS_DB", "RESEARCH_ROOT"):
        env.pop(var, None)
    proc = subprocess.run(
        [str(readme.parent / "run.sh")], env=env, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, (
        f"{readme.parent.name} run.sh failed:\n{proc.stdout}\n{proc.stderr}"
    )
    return proc.stdout


@pytest.mark.parametrize("readme", [p for p in paths() if _offline(p)], ids=lambda p: p.parent.name)
def test_an_offline_path_runs_green_and_prints_its_pinned_line(readme, tmp_path):
    stdout = run_the_path(readme, tmp_path)
    pin = pinned_line(readme.read_text())
    assert pin, f"{readme.parent.name}: What it prints has no fenced line to pin"
    assert pin in stdout, f"{readme.parent.name}: pinned line not in output: {pin!r}"


_ELISION_LINES = {"...", "[...]", ""}


@pytest.mark.parametrize("readme", [p for p in paths() if _offline(p)], ids=lambda p: p.parent.name)
def test_every_output_block_of_an_offline_path_is_real(readme, tmp_path):
    stdout = run_the_path(readme, tmp_path)
    checked = 0
    for line in _fenced(_section(readme.read_text(), "What it prints")):
        if line.strip() in _ELISION_LINES:
            continue
        assert line in stdout, f"{readme.parent.name}: {line!r} is not in real stdout"
        checked += 1
    # A block that is ALL elision (every line "..."/"[...]"/blank) makes the
    # loop above assert nothing at all, so the test passes vacuously -- it
    # never actually checked this README's output against anything real.
    # `pinned_line` above already requires a non-elision FIRST line, but this
    # loop walks every line, and nothing stopped a later line from eliding
    # everything after it.
    assert checked > 0, (
        f"{readme.parent.name}: every line of 'What it prints' is an elision"
        " (.../[...] /blank) -- nothing in this block was actually checked"
        " against real output"
    )


@pytest.mark.parametrize("readme", paths(), ids=lambda p: p.parent.name)
def test_every_readme_opens_with_the_checked_by_pointer(readme):
    assert (
        readme.read_text().lstrip().startswith("<!-- checked by tests/test_golden_paths.py -->")
    ), f"{readme.parent.name}: README does not open with the checked-by pointer"
