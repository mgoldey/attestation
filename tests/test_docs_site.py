"""The docs site's generated pages are tested the same way the golden paths
are: a committed file that a script writes gets a test asserting it equals a
fresh render, so a flag added to the CLI or a spec added to the directory
cannot land without the reference page that describes it.

`mkdocs build --strict` itself only runs under `pytest.importorskip("mkdocs")`
-- CI's `gates` job does not install the `docs` dependency group, only the
`docs` job does, and that job runs the build directly. This suite still
checks the nav paths and the API-page-per-module rule without mkdocs
installed, via a regex over `mkdocs.yml` (the project has no PyYAML
dependency).

Spec: docs/superpowers/specs/2026-08-29-package-docs-design.md."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_cli_reference_is_a_fresh_render(monkeypatch):
    """`docs/reference/cli.md` is generated from the live parser; a hand
    edit here would silently drift from `--help` the way a hand-transcribed
    reference always does.
    """
    monkeypatch.setenv("COLUMNS", "80")
    sys.path.insert(0, str(ROOT / "scripts"))
    import render_cli_reference

    committed = (ROOT / "docs" / "reference" / "cli.md").read_text()
    assert committed == render_cli_reference.render(), (
        "docs/reference/cli.md is stale -- run "
        "`uv run python scripts/render_cli_reference.py` and commit the result"
    )


def test_spec_index_is_a_fresh_render():
    """`docs/site/specs.md` lists every design spec; a hand edit here would
    silently drift the moment a spec is added.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import render_spec_index

    committed = (ROOT / "docs" / "site" / "specs.md").read_text()
    assert committed == render_spec_index.render(), (
        "docs/site/specs.md is stale -- run "
        "`uv run python scripts/render_spec_index.py` and commit the result"
    )


def test_every_public_module_has_an_api_reference_page():
    """Every module in `attestation.__all__` gets an mkdocstrings page. A
    module added to `__all__` with no page would build a site silently
    missing it -- mkdocs does not fail just because a page could exist and
    doesn't.
    """
    import attestation

    api_dir = ROOT / "docs" / "reference" / "api"
    for module in attestation.__all__:
        page = api_dir / f"{module}.md"
        assert page.is_file(), f"no API reference page for attestation.{module}: {page}"
        assert f"::: attestation.{module}" in page.read_text(), (
            f"{page} does not contain a mkdocstrings directive for attestation.{module}"
        )


def test_nav_paths_in_mkdocs_yml_exist():
    """Every `.md` path named in `mkdocs.yml`'s nav resolves to a real file
    under `docs/`. Checked by regex rather than a YAML parse: the project
    carries no PyYAML dependency, and `mkdocs build --strict` already covers
    this properly once the `docs` group is installed -- this is the cheap
    version that runs in the default suite.
    """
    text = (ROOT / "mkdocs.yml").read_text()
    paths = re.findall(r":\s*([\w./-]+\.md)\s*$", text, re.MULTILINE)
    assert paths, "no nav paths found in mkdocs.yml -- did the nav format change?"
    docs_dir = ROOT / "docs"
    missing = [p for p in paths if not (docs_dir / p).is_file()]
    assert not missing, f"mkdocs.yml nav points at missing file(s): {missing}"


def test_mkdocs_build_strict():
    """The real build, exactly as CI's `docs` job runs it. Skipped when the
    `docs` dependency group is not installed (the default `uv sync`), since
    that is also what CI's `gates` job does not install.
    """
    pytest.importorskip("mkdocs")
    pytest.importorskip("mkdocstrings")
    result = subprocess.run(
        ["mkdocs", "build", "--strict"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"mkdocs build --strict failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_the_readme_is_a_front_door():
    """The README used to be the whole manual -- 778 lines, twelve sections,
    the pitch and the install guide and the hermes-agent integration manual
    and the ledger manual all in one file. A newcomer could not tell in a
    minute what to run or where their question's answer lived.

    Front-door shape, mechanically checked: short enough to read in a
    minute, every relative link inside it resolves, and the quickstart is
    still there (its byte-identical content is `test_examples.py`'s job, not
    this test's).
    """
    import re

    text = (ROOT / "README.md").read_text()
    lines = text.splitlines()
    assert len(lines) <= 200, f"README is {len(lines)} lines, want <= 200"

    assert "Try it in 60 seconds" in text, "README dropped its quickstart heading"

    link_re = re.compile(r"\]\((docs/[^)#\s]+|CONTRIBUTING\.md|examples/[^)#\s]+)")
    missing = []
    for target in link_re.findall(text):
        if not (ROOT / target).exists():
            missing.append(target)
    assert not missing, f"README links to missing target(s): {missing}"


def test_every_guide_is_in_the_nav_and_leads_with_an_answer():
    """Each guide under `docs/guides/` needs two things to work as a front
    door's second click: the site nav has to know it exists, and it has to
    open with a plain-language answer before the reader hits a `##` heading
    -- the same "lead with the answer" shape as the README, at guide scale.
    """
    nav_text = (ROOT / "mkdocs.yml").read_text()
    guides = sorted((ROOT / "docs" / "guides").glob("*.md"))
    assert guides, "no guides found under docs/guides/"

    for guide in guides:
        rel = f"guides/{guide.name}"
        assert rel in nav_text, f"{rel} is not referenced in mkdocs.yml's nav"

        lines = guide.read_text().splitlines()
        # skip the title (`# ...`) and any blank lines, then the next
        # non-blank line must be the answer paragraph, appearing strictly
        # before the first `## ` section heading.
        body = [ln for ln in lines if not ln.startswith("# ")]
        first_heading_idx = next(
            (i for i, ln in enumerate(body) if ln.startswith("## ")), len(body)
        )
        answer_lines = [ln for ln in body[:first_heading_idx] if ln.strip()]
        assert answer_lines, f"{guide.name} has no answer paragraph before its first ## heading"
        answer_paragraph = " ".join(answer_lines).rstrip()
        assert answer_paragraph.endswith("."), (
            f"{guide.name}'s answer paragraph does not end with a period: {answer_paragraph!r}"
        )


def test_the_concepts_page_defines_the_first_ten_minutes():
    """`docs/concepts.md` is the glossary for the words a newcomer meets
    early: run, family, arm, spec, claim and its verdict kinds, corpus,
    persona, click provenance, surface, golden path, tracker convention.
    Each term must actually be defined here, not just mentioned in passing
    -- checked as a bolded or heading term.
    """
    text = (ROOT / "docs" / "concepts.md").read_text()
    terms = [
        "run",
        "family",
        "arm",
        "spec",
        "claim",
        "verdict",
        "corpus",
        "persona",
        "provenance",
        "surface",
        "golden path",
        "convention",
    ]
    missing = []
    for term in terms:
        bold = re.search(rf"\*\*{re.escape(term)}", text, re.IGNORECASE)
        heading = re.search(rf"^###\s+{re.escape(term)}", text, re.IGNORECASE | re.MULTILINE)
        if not (bold or heading):
            missing.append(term)
    assert not missing, f"docs/concepts.md does not define: {missing}"
