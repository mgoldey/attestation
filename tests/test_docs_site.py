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

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_cli_reference_is_a_fresh_render():
    """`docs/reference/cli.md` is generated from the live parser; a hand
    edit here would silently drift from `--help` the way a hand-transcribed
    reference always does.
    """
    os.environ["COLUMNS"] = "80"
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
