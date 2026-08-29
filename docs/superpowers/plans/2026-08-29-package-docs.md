# Package Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The package states its surface (`__init__`, `py.typed`, `[project.urls]`), every public def has a docstring with a ratchet test that keeps it so, `CONTRIBUTING.md` and `CHANGELOG.md` exist, and an mkdocs site with a generated CLI reference and a mkdocstrings API reference builds `--strict` in CI.

**Architecture:** Three independent tasks. Task 1 touches the package and `cli.py` (docstrings from argparse help) plus the ratchet test. Task 2 is documents only. Task 3 adds `mkdocs.yml`, a `docs` dependency group, two generator scripts with equality tests, and a CI job. No runtime dependency changes; `src/` imports nothing new.

**Tech Stack:** Python ≥3.12, `importlib.metadata`, `mkdocs-material`, `mkdocstrings[python]` (docs group only), argparse introspection.

**Spec:** `docs/superpowers/specs/2026-08-29-package-docs-design.md`

## Global Constraints

- Nothing under `src/` imports mkdocs/mkdocstrings (`tests/test_tag_prompt.py`'s guard gains `mkdocs`).
- Docstring ratchet: public = module, class or function whose name does not start with `_`, under `src/attestation/**`; baseline pinned in the test; only ratchets down; failure names `file:line`.
- `cmd_*` docstrings and argparse `help=` are ONE source; a test asserts equality for every subcommand.
- `docs/reference/cli.md` is generated; a test asserts the committed file equals a fresh render.
- `mkdocs build --strict` must pass locally and in CI; the site is built, not deployed.
- The README is not split; it gains one "Documentation" line.
- Line length 100; `*.md` excluded from ruff; gates after `git add`; commit by pathspec; message style: plain sentence, blank line, `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; new test files and new dirs into CLAUDE.md's docs index.

---

### Task 1: The surface, the docstrings, and the ratchet

**Files:**
- Modify: `src/attestation/__init__.py`, `pyproject.toml` (`[project.urls]`; ensure `py.typed` is packaged — check `[tool.hatch]`/build config for package data), `src/attestation/cli.py`, `src/attestation/citations.py`, any other module the ratchet lists, `.github/workflows/ci.yml` (wheel-smoke asserts `py.typed`), `CLAUDE.md` (index)
- Create: `src/attestation/py.typed` (empty), `tests/test_docstring_ratchet.py`, `tests/test_package_surface.py`

- [ ] **Step 1: Failing tests.** `tests/test_package_surface.py`: `attestation.__version__` equals `importlib.metadata.version("attestation")`; `attestation.__all__` lists exactly the modules in the spec and each imports; `attestation.__doc__` is non-empty; `Path(attestation.__file__).with_name("py.typed").exists()`. `tests/test_docstring_ratchet.py`: walk `src/attestation/**/*.py` with `ast`, collect undocumented public defs as `path:line name`, assert `len(missing) <= BASELINE` with the list in the message; `BASELINE = 0` as the target — set it to the real count first, ratchet to 0 by the end of the task. Plus `test_every_cli_handler_docstring_is_its_help_text`: build the parser, for each subparser action map `dest` → `help`, and assert `getattr(cli, f"cmd_{name.replace('-', '_')}").__doc__.strip() == help` (read how `build_parser` names things; `runs` has sub-subcommands — cover them too).
- [ ] **Step 2: Implement.** `__init__.py` docstring + `__version__` (`importlib.metadata.version`, with a `PackageNotFoundError` fallback to `"0+unknown"`) + `__all__`; `py.typed`; `[project.urls]`; wheel-smoke step asserts `py.typed` in the installed package (`importlib.resources.files("attestation").joinpath("py.typed").is_file()`), mirroring the `feeds.toml` step. In `cli.py`, a helper `_help(fn)` or a table `HELP = {...}` used by both `add_parser(..., help=HELP[name])` and `cmd_x.__doc__ = HELP[name]` — whichever keeps `build_parser` readable; keep behaviour identical (`attest --help` output unchanged — assert in a test by comparing against the current output captured once). Write the remaining docstrings (citations readers, claims dataclasses, etc.) from their specs/tests: say what the thing is for and any measured rationale, not the signature.
- [ ] **Step 3:** ratchet baseline to 0; gates; commit.

---

### Task 2: CONTRIBUTING.md and CHANGELOG.md

**Files:** Create `CONTRIBUTING.md`, `CHANGELOG.md`; modify `README.md` (one "Documentation" line near the end pointing at `CONTRIBUTING.md`, `CHANGELOG.md`, and `uv run --group docs mkdocs serve`), `CLAUDE.md` (index root list gains the two files).

- [ ] Write `CONTRIBUTING.md` per the spec's list, from `CLAUDE.md`'s "Working here" and `docs/superpowers/specs/2026-08-28-golden-paths-design.md`'s "What a golden path is" — a contributor's voice, ~120 lines, every claim checkable (name the test that enforces each rule). Write `CHANGELOG.md` (Keep a Changelog header; `## [Unreleased]` with subsections Added / Changed / Fixed; one line per change with the short SHA in parentheses, drawn from `git log --oneline 573e42c..HEAD` grouped by area: CI, example flows, golden paths, ledger conventions, corpora, CLI; a closing line that history before 2026-08-28 is not reconstructed). Commit.

---

### Task 3: The docs site and the CLI reference

**Files:** Create `mkdocs.yml`, `docs/index.md` (includes README via a snippet or symlink — check what mkdocs-material supports without a plugin; a one-line `--8<-- "README.md"` needs `pymdownx.snippets`, which ships with pymdown-extensions, a mkdocs-material dependency), `docs/reference/cli.md` (generated), `docs/reference/api/<module>.md` (one per `__all__` entry: `::: attestation.<module>`), `docs/site/specs.md` (generated list), `scripts/render_cli_reference.py`, `scripts/render_spec_index.py`; modify `pyproject.toml` (group `docs = ["mkdocs-material>=9", "mkdocstrings[python]>=0.26"]`), `uv.lock` (`uv lock`), `.github/workflows/ci.yml` (job `docs`: `uv sync --group docs`, `uv run --group docs mkdocs build --strict`, `actions/upload-artifact` of `site/`), `.gitignore` (`site/`), `tests/test_docs_site.py`, `tests/test_tag_prompt.py` (guard gains `mkdocs`), `CLAUDE.md`.

- [ ] **Step 1: Failing tests** in `tests/test_docs_site.py`: `render_cli_reference.render() == (repo/"docs/reference/cli.md").read_text()`; `render_spec_index.render()` equals the committed `docs/site/specs.md`; every module in `attestation.__all__` has `docs/reference/api/<module>.md` containing `::: attestation.<module>`; `mkdocs.yml` parses and every nav entry's file exists (use `yaml` — is PyYAML available to tests? `mkdocs` depends on it; the test can `pytest.importorskip("yaml")`... no: make the test read `mkdocs.yml` with `tomllib`? It is YAML. Ruling: the test runs `uv run --group docs mkdocs build --strict` via subprocess only when `mkdocs` is importable (`importorskip`), otherwise checks the nav files by a regex over `mkdocs.yml` lines `- <title>: <path>`).
- [ ] **Step 2:** the two renderers (`render_cli_reference.py`: walk `build_parser()`'s subparsers recursively, emit `## attest <cmd>` + the parser's `format_help()` in a fenced block; `render_spec_index.py`: list `docs/superpowers/specs/*.md` newest first with each file's first `#` heading and Status line), `mkdocs.yml` (site_name, `docs_dir: docs`, `theme: material`, `plugins: [search, mkdocstrings]`, nav per the spec, `markdown_extensions: [pymdownx.snippets, admonition, tables]`, `strict: true`), the API pages, the CI job, `.gitignore`.
- [ ] **Step 3:** `uv run --group docs mkdocs build --strict` green locally; fix every warning it reports (a docstring cross-ref that does not resolve is a real finding — fix the docstring, never suppress); gates; commit.

---

## Self-review
Spec coverage: surface + `py.typed` + urls + ratchet + one-source CLI docstrings (T1); CONTRIBUTING/CHANGELOG + README line (T2); site + CLI reference + spec index + API pages + CI job + guard (T3). Placeholders: none — each step names files, functions, tests and commands. Type consistency: `render()` functions return `str` and the tests compare to committed files; `__all__` drives the API pages and the surface test.
