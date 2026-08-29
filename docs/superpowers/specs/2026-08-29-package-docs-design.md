# Package documentation: the surface, the reference, and the ratchets

**Date:** 2026-08-29
**Status:** design; implementation follows in the plan of the same date.
**Depends on:** golden paths (`2026-08-28-golden-paths-design.md`), the
architecture roadmap (`2026-08-21-architecture-roadmap.md`).

## Problem

Measured on 2026-08-29 against `main` at `e5e511b`:

- `src/attestation/__init__.py` is empty: no package docstring, no
  `__version__`, no statement of what is public. `pyproject.toml` has a
  good description, license, keywords and classifiers but no
  `[project.urls]`, and the wheel ships no `py.typed`, so a downstream type
  checker ignores annotations this repo checks with `ty` on every commit.
- 190 of 292 public defs carry a docstring (65%). The 103 without are
  concentrated in `cli.py`'s `cmd_*` handlers — whose one-line purpose
  already exists as argparse `help=` text, a second copy of the same
  fact — and in the citation readers' `all`/`lookup` methods.
- There is no changelog and no contributor guide. The conventions that
  matter here (a spec before code, the eight gates, the docs index that
  `test_architecture.py` enforces, the golden-path shape, the `noqa`
  policy, no attribution in committed fixtures, the commit-message style)
  live in `CLAUDE.md`, which is written for an agent, not a contributor.
- The README is 755 lines in 11 sections. It is the front door, the
  install guide, the hermes-agent integration manual, the ledger manual,
  the claims manual, the ranking explainer and the test guide at once.
  `docs/` holds 27 design specs, an architecture narrative and the
  measurement-lessons record — the material a docs site is made of — with
  no site, no API reference, and no CLI reference beyond `--help`.

The repo's own rule is that a documented fact is a tested fact. Package
documentation should be held to it: coverage that only ratchets down, a
reference that fails the build when a cross-reference breaks, and a CLI
page generated from the parser rather than transcribed.

## Design

### The package surface

`attestation/__init__.py` gets a docstring stating what the package is
(one paragraph, the same sentence `pyproject.toml` uses), `__version__`
read from `importlib.metadata` (so `pyproject.toml` stays the one source),
and an `__all__` naming the modules a user is meant to import
(`ledger`, `claims`, `citations`, `rank`, `ingest`, `features`, `simulate`,
`explain`, `symbolic`, `kg`, `emit`, `install`, `llm`, `embed`, `db`,
`ports`). Nothing is re-exported: the modules are the API, as
`docs/superpowers/specs/2026-08-21-onion-refactor-design.md` decided.
A `py.typed` marker ships in the wheel (the wheel smoke test asserts it).
`[project.urls]` gains Homepage, Repository, Issues and Changelog.

### Docstrings: one source, and a ratchet

`cli.py`'s handlers keep their argparse `help=` as the one source: each
`cmd_*` function's docstring is set from the same string at definition
(`build_parser` already has the strings; a small helper attaches them), so
`--help` and the docstring cannot drift. The citation readers and the
remaining public functions get real docstrings written from their tests
and specs — the rationale-carrying kind this repo already writes, not
restatements of the signature.

`tests/test_docstring_ratchet.py` counts public defs (module, class,
function; names not starting with `_`; under `src/attestation`) with no
docstring and asserts the count is at most the pinned baseline, the way
`scripts/check_complexity.py` pins complexity. The baseline starts at
whatever this work leaves (target: 0) and may only go down; a new public
function without a docstring fails the suite with the offender's
`file:line`.

### CONTRIBUTING.md and CHANGELOG.md

`CONTRIBUTING.md` is the contributor's view of `CLAUDE.md`'s "Working
here": clone → `uv sync` → `uv run pre-commit install`; the eight gates and
what each catches; a spec in `docs/superpowers/specs/` before a feature;
the docs index in `CLAUDE.md` that the architecture test enforces; how to
add a golden path (the seven sections, `run.sh`, the catalogue row, the
generator-and-scrub rule); the `# noqa: BLE001` policy (seven sites, each
with its reason); no attribution or machine paths in committed files (and
the guard that checks); commit messages that say what changed and why in
a sentence.

`CHANGELOG.md` follows Keep a Changelog. An `Unreleased` section digests
what landed since the last tagged state, by area, with one line per
change and the commit that carries its reasoning — commits here are
narrative and the changelog points at them rather than repeating them.
Nothing before 2026-08-28 is reconstructed; the section says so.

### The docs site

`mkdocs.yml` at the root, `mkdocs-material` + `mkdocstrings[python]` in a
new `docs` dependency group (never a runtime dependency; the `src/`-import
guard gains `mkdocs`). Navigation:

- **Home** — `README.md` (included, not copied; `mkdocs` `--strict` refuses
  a broken link).
- **Getting started** — the golden-path catalogue (`examples/README.md`
  included) and one page per path? No: the catalogue links to each
  README on the repository, which already has the seven sections; a copy
  would drift.
- **Guides** — pages that exist today under `docs/`: the architecture
  narrative, measurement lessons, recommendation refinements, the
  hermes-agent research.
- **Design records** — the 27 specs, listed by date (generated from the
  directory at build time by a tiny plugin-free script that writes
  `docs/site/specs.md`; the script is run by the build and by a test).
- **Reference** — the CLI (`docs/reference/cli.md`, generated from
  `build_parser()` by `scripts/render_cli_reference.py`, with a test that
  the committed page equals a fresh render — the same "docs are tested"
  rule as the golden paths) and the API (one `mkdocstrings` page per
  module in `__all__`, `show_source: false`, docstring style left as
  written — this repo's docstrings are prose, and `mkdocstrings` renders
  prose).

`mkdocs build --strict` runs as a fourth CI job, `docs`, so a docstring
that breaks a cross-reference or a nav entry that points nowhere fails
the push. The site is not deployed by this spec (no Pages workflow); the
artifact is uploaded so a reviewer can open it.

### The README, and what does not move yet

The README stays the front door and is not split by this spec. The
judgement of which 300-line sections become docs pages (the hermes-agent
integration manual is the obvious one) is the user's; the site mirrors
the README so nothing is lost either way, and a follow-up spec can move
sections once the site exists to receive them. What this spec does change
in the README: one "Documentation" line pointing at the site's build
command and `CONTRIBUTING.md`.

## Not in scope

- Deploying the site (GitHub Pages) — a workflow with permissions the
  user should turn on deliberately.
- Splitting the README (above).
- Rewriting existing docstrings for style. Coverage, not house style;
  the ratchet counts presence.

## Success criteria

- `uv run python -c "import attestation; print(attestation.__version__,
  attestation.__all__)"` prints the version from `pyproject.toml` and the
  module list; the wheel contains `py.typed` (asserted in the CI wheel
  smoke job).
- `tests/test_docstring_ratchet.py` passes at a baseline of 0 undocumented
  public defs, and fails — naming the offender — when one is added.
- `attest <cmd> --help` and `cmd_<cmd>.__doc__` are the same text for
  every subcommand (one test).
- `uv run --group docs mkdocs build --strict` exits 0; CI job `docs` is
  green; `docs/reference/cli.md` is byte-equal to a fresh render (one
  test).
- `CONTRIBUTING.md` and `CHANGELOG.md` exist and are linked from the
  README; `[project.urls]` present.
