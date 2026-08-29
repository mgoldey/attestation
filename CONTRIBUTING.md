# Contributing

This is a solo-maintained local-first tool, not a project soliciting a
community — but if you're reading this to send a patch, here's how the repo
actually works.

## Getting started

```bash
git clone <this repo>
cd attestation
uv sync
uv run pre-commit install
```

`uv sync` builds `.venv/`. If `uv run pytest` (or any `uv run` command) ever
fails with something like "No module named 'attestation'" or "Failed to
spawn," suspect a stale `.venv/`, not a broken `uv` — see CLAUDE.md's
"Working here" for the story of the last time this happened (a venv built at
the project's previous path, with 26 console scripts pointing their shebang
at an interpreter that no longer existed). `uv sync` rebuilds it.

`uv run pre-commit install` activates the local git hook so the gates below
run on every commit rather than only when you remember to run them.

## The gates

`uv run --frozen pre-commit run --all-files` runs eight local hooks, defined
in `.pre-commit-config.yaml`. Each one exists because something specific
broke without it:

1. **ruff format** — formats Python (and, since ruff 0.16, Python code
   fences inside Markdown — `*.md` itself is excluded, see below).
2. **ruff check** — lints `E, F, W, I, BLE` at line length 100. `BLE`
   (bare/broad `except`) is not a blanket ban — see "The noqa policy" below.
3. **ty** — type checks `src/` as a whole tree, not staged-files-only:
   `ty` resolves imports across modules, so a partial check both under- and
   over-reports relative to a full run.
4. **uv.lock matches pyproject.toml** — `uv lock --check`, so a dependency
   edit that forgets to update the lock fails here instead of on someone
   else's clone.
5. **complexity ratchet** (`scripts/check_complexity.py`, via radon) — pins
   each file's worst-function complexity at its own measured value. It's
   per-file on purpose: a single global threshold loose enough to pass a
   clean checkout also permits any file to grow a function up to that same
   rank.
6. **bandit** (security, medium severity and above) — baseline is 0 findings.
7. **xenon** (maintainability) — module and average grades only, not
   per-function absolute: that's what the complexity ratchet already covers.
8. **pytest** — the full suite, ~50-70s. Slow for a hook, worth it: this
   repo's recurring failure mode has been a test that passes against the
   bug it was written to catch, so a green suite is the one signal that
   has actually mattered.

Two things that catch people out:

- **`pre-commit run --all-files` only looks at tracked files.** If you added
  a new file, `git add` it first, or the hook won't see it and you'll get a
  green run that proves nothing about your new code.
- **Read the per-hook `Passed`/`Failed` lines, not just the tail.** A hook
  that fails still lets later hooks run, so the last thing printed is
  often an unrelated hook's success, not the failure you need to fix.

CI runs the same eight gates, on Linux and macOS, across Python 3.12 and
3.13, plus a wheel smoke test — so a bypassed local hook (`--no-verify`,
which nothing here asks you to use) is caught on push rather than never.

## A spec before a feature

Every non-trivial feature has a design doc in `docs/superpowers/specs/`,
written *before* the code, and a plan in `docs/superpowers/plans/`. The spec
records *why*, not just *what* — read the relevant one before touching a
subsystem, and write one before adding a new subsystem. `CLAUDE.md`'s "Docs
Index" section names which spec goes with which code area.

## The docs index

`CLAUDE.md` carries a compressed file-tree index — every source file under
`src/attestation/**` and every test under `tests/*.py`, grouped by
directory. `tests/test_architecture.py::test_the_docs_index_lists_every_source_and_test_file`
asserts every such file's name appears somewhere in that index; it once
drifted 11 files behind (including a persona-split that left the index
pointing at the wrong module), so this isn't optional bookkeeping — it's a
gate. **Add your new file's name to the appropriate `{...}` group in
`CLAUDE.md` in the same commit that adds the file.**

## The docs site

`mkdocs.yml`'s comment block above `nav` states which pages are generated
(never hand-edit), hand-written, snippet-included, or symlinked — read it
before touching a page — and `uv run --group docs mkdocs build --strict`
(`tests/test_docs_site.py` covers the parts of it that don't need `mkdocs`
installed) is how you check a nav or cross-reference change before pushing.

## Adding a golden path

`examples/<name>/` is this repo's worked-example format
(`docs/superpowers/specs/2026-08-28-golden-paths-design.md`,
`tests/test_golden_paths.py`). Every path needs:

1. **`README.md` with exactly these seven `## ` sections, in order:**
   *What you get*, *Prerequisites*, *Run it*, *What it prints*, *What it
   demonstrates*, *When it goes wrong*, *Next*.
2. **`Prerequisites` names one of exactly three labels**, verbatim:
   `none — pure local computation`, `a model server at LLM_BASE_URL`, or
   `network`.
3. **`run.sh`**: `#!/usr/bin/env bash`, `set -euo pipefail`, executable, and
   it must contain — verbatim — every fenced command that starts with
   `uv run`, `attest`, `./run.sh`, `export `, or `ATTEST_` in the README's
   *Run it* section. The two are checked against each other; they cannot
   drift.
4. **A row in `examples/README.md`'s catalogue table**, with the same
   prerequisite label, in the table's sort order (`none` first, then
   `network`, then the model-server paths; alphabetical within each group).
5. **If the path's inputs come from a real third-party library**, they're
   produced by a committed `generate.py`/`generate.sh` that runs that real
   library and pins its version (refusing to run under any other), then
   scrubs attribution and machine paths — never hand-written to look real.
6. **No committed file may contain** an absolute `/home/` path, a
   `github.com` URL, or (outside this repo's own scrubber source/docs) the
   `mlflow.user` tag name or a `git@` remote, checked byte-for-byte even in
   binary files. `tests/test_golden_paths.py::test_no_committed_example_carries_attribution_or_machine_paths`
   enforces this across all of `examples/**`; it's skipped for the
   ambient-`$USER` check specifically on CI and for generic account names
   (`runner`, `root`, `ci`, etc.), because a CI runner's own service account
   is not a person's identity.

Discovery is automatic: `tests/test_golden_paths.py` globs `examples/*/README.md`
and runs every check above with no edit to the test file itself. If a path's
prerequisite is `none`, its `run.sh` is also actually executed in CI and one
pinned line from *What it prints* is asserted against real stdout — so
that section can't just be aspirational prose.

## The noqa policy

`# noqa: BLE001` (suppressing ruff's bare/broad-except lint) is not banned,
but it is not free either: there is deliberately no `[tool.ruff.lint.
per-file-ignores]` blanket exemption anywhere in `pyproject.toml` (there
used to be one, silently pointing at a path from before a rename, enforcing
nothing). Every site needs an inline comment explaining why that particular
catch is a policy, not a swallow. The total count is asserted by
`tests/test_architecture.py::test_claude_md_noqa_inventory_matches_the_tree`
against the number `CLAUDE.md`'s "Reliability contract" line states — add a
site and forget to bump that count, and this test fails. (Line numbers are
deliberately not pinned anywhere, on purpose — citing them is what let a
previous version of this policy rot.)

## No attribution or machine paths in committed files

Applies beyond `examples/` in spirit, but is mechanically enforced there:
nothing checked in should carry your home directory path, your username, or
a repo URL baked into a fixture. If you're generating a fixture from a real
tool (wandb, mlflow, sacred, dvc, hydra, tensorflow, bibtexparser, …), run
the scrub step and verify the guard test passes before committing — don't
hand-edit a fixture to *look* real.

## Docstrings on every public def

Every module, class, and function under `src/attestation/**` whose name
doesn't start with `_` needs a docstring — at any nesting depth, including
closures and decorator-wrapped inner functions. `tests/test_docstring_ratchet.py::test_every_public_def_has_a_docstring`
counts violations by walking the AST and fails if the count exceeds its
pinned `BASELINE` (currently 0), naming the offending `file:line`. The
ratchet only ever goes down: adding an undocumented public function fails
the suite immediately, on your machine, before it fails anyone else's.

One exception that isn't really an exception: `cli.py`'s `cmd_*` handlers
get their docstring's first line from the same string already passed to
argparse's `help=`, so the two can't drift — see
`tests/test_cli.py::test_every_cmd_docstring_is_its_helps_first_line`. If
you add a subcommand, write the `help=` string once and reuse it as the
docstring's first line; add rationale after a blank line if there's more to
say.

## Commit messages

A plain sentence saying what changed and why — not what files moved, since
`git diff` already shows that. Look at `git log` for the voice this repo
uses: narrative, past-tense, one line, no bullet-point changelog crammed
into the subject. The `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
trailer is a convention for commits an agent authors in this repo, not a
requirement for a human contributor's commit.

## Pushing to main

This repo has no long-lived feature branches or PR review gate — work lands
on `main` directly once the gates above are green. That's a description of
how this particular repo operates, not a recommendation for projects with
more than one contributor actively pushing at once.
