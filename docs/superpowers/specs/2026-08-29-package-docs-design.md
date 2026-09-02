# Package documentation: the surface, the reference, and the ratchets

**Date:** 2026-08-29
**Status:** implemented 2026-08-29, with deviations below.
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
  every subcommand (one test). **Amended 2026-08-29:** `cmd_*` docstrings
  carry rationale paragraphs beyond the one-line help text (see
  Deviations), so "the same text" means the same FIRST line, by
  construction — `_documented` sets it from `HELP`, and the test checks
  only that line.
- `uv run --group docs mkdocs build --strict` exits 0; CI job `docs` is
  green; `docs/reference/cli.md` is byte-equal to a fresh render (one
  test).
- `CONTRIBUTING.md` and `CHANGELOG.md` exist and are linked from the
  README; `[project.urls]` present.

## Deviations and findings

**`docs/CONTRIBUTING.md`, `docs/CHANGELOG.md`, and `docs/examples` are
symlinks to the repo-root files/directory, not copies or mkdocs
`include-markdown` includes.** `mkdocs` serves `docs_dir` as its own root,
so a root-relative link inside a file included via `pymdownx.snippets`
(the README's own links to `CONTRIBUTING.md`, `examples/`, and so on)
resolves against `docs/`, not the repo root — a copy would need every such
link rewritten, and would drift the moment the source file's links
changed. A symlink sidesteps both problems: `docs/CONTRIBUTING.md` IS
`CONTRIBUTING.md`, so a link written relative to the repo root already
lands correctly once `docs/` is the serving root. The cost is Windows:
without `git config core.symlinks true` (or Developer Mode enabled at
clone time), Windows checks a symlink out as a plain text file containing
the link's target path rather than the linked content, and `mkdocs build`
would either 404 on it or render the path as literal text. `mkdocs.yml`'s
new comment block above `nav` and `CONTRIBUTING.md`'s new "The docs site"
section both name this caveat.

**`cmd_*` docstrings are not literally identical to their `help=` text —
only their first line is.** The design's "one source" section anticipated
`HELP` feeding both `--help` and the docstring; what shipped
(`_documented`, `cli.py`) sets the docstring's first line from `HELP` and
keeps any rationale paragraph already written in the function's own
literal docstring below it (blank line, then prose) — `argparse`'s
`help=` stays a one-line summary, matching how every other subcommand's
help text reads, while the docstring can still carry the "why" this
codebase's docstrings are written to carry (see `CONTRIBUTING.md`'s
"Docstrings on every public def"). `tests/test_cli.py::
test_every_cmd_docstring_is_its_helps_first_line` checks exactly that: the
docstring's first line equals `HELP[name]`, not the whole string. The
Success criteria entry above is amended in place to say so, since "the
same text" as originally written was already wrong the day this shipped.

**The docstring ratchet's runtime-`__doc__` fallback originally accepted
ANY decorator, not just `@_documented` — narrowed 2026-08-29 (see follow-up
item 2).** As written for this spec, `tests/test_docstring_ratchet.py`
treated a module-level def as documented if it carried any decorator at
all and its runtime `__doc__` was non-empty, reasoning that a decorator is
the only thing that can set `__doc__` without a literal string in the
body. That reasoning missed `@dataclass`, which synthesizes a `__doc__`
of the form `Foo(a: int)` on the class itself with no literal docstring
anywhere — so an undocumented public dataclass passed the ratchet.
Narrowed the fallback to require `_documented` specifically among the
decorator's names (`_decorator_names()` resolves `@foo`, `@foo(...)`,
`@mod.foo`, and `@mod.foo(...)` to their base name), and added two
regression tests: an undocumented `@dataclass` is reported, and a
`@_documented`-decorated def with no literal docstring is still accepted.
The collector (`_undocumented`) already took an arbitrary `path` and read
it directly, so no path-handling refactor was needed; what did need
generalizing was the module-loading step it uses to check a decorator's
runtime `__doc__`, which assumed the path was importable as
`attestation.<...>` — `_load_module_for_import_check()` now falls back to
loading an out-of-package path directly via `importlib.util`, which is
what lets the regression tests point the collector at a temp module
outside `src/attestation` instead of only asserting against the real tree.

**The getting-started page includes the golden-path catalogue by
snippet, matching the pattern the design already set for the README.**
`getting-started.md` is `examples/README.md`'s catalogue via
`pymdownx.snippets`, the same "included, not copied" rule `mkdocs.yml`'s
Home entry applies to the README — a hand-copied catalogue would drift
the first time a golden path was added or reordered, which
`tests/test_golden_paths.py` already guards on the `examples/` side but
would have no counterpart on the docs-site side.

**Two dead-docstring bugs found in `server.py` during this work:
`require_user` and `reader` each had a string literal placed AFTER their
first statement (`conn = connection()`), not before it — a comment-shaped
piece of dead code, not a docstring, since Python only recognizes a
literal string as `__doc__` when it is the first statement in the body.**
Both had real, substantial rationale written (why writes refuse rather
than autocreate; why the web UI's read path autocreates and only guards
the write against cross-origin), and both were silently inert: `ast.
get_docstring` and `help()` alike would have reported `None`, and the
docstring ratchet itself would have flagged them as undocumented public
functions had this landed before the fix rather than as part of it. Fixed
in `7bc20cf` by moving `conn = connection()` below the docstring in both
functions — the same commit that added the ratchet, so the ratchet never
saw the bug. Recorded here because a docstring rewrite that reorders
statements is exactly the kind of change a diff reviewer skims past.

**Deployment, 2026-09-01.** Turned on by instruction ("make docs via github"):
`.github/workflows/docs.yml` runs the same `mkdocs build --strict` as the
`docs site` CI job and deploys the result to GitHub Pages from `main`;
Pages was switched to the Actions build type with
`gh api -X POST repos/mgoldey/attestation/pages -f build_type=workflow`. The
account's user-pages custom domain means the project site is served at
`http://matthew.thegoldeys.com/attestation/` (HTTPS is not enforced on that
domain today), which `site_url` now states.
