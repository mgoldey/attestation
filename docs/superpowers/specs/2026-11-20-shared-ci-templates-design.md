# Shared CI templates: three reusable workflows, called by all three repositories

**Date:** 2026-11-20
**Status:** stage 1 implemented 2026-11-20 (roadmap item 39): the templates
live here and attestation's own CI and release call them. Stage 2 (smeltery
and ferric call them by a pinned SHA) is the snippets under "Stage 2".
**Not this:** a shared composite action, a template repository, or moving
ferric's Rust/libint2/maturin build anywhere.

## Problem

attestation, smeltery and ferric each carry their own copy of the same CI
plumbing: check out, install uv, `uv sync` some extras, ruff format/check, a
type check, `uv lock --check`, run the hooks, build a wheel the release way and
install it into a clean venv, refuse a tag the package does not carry. The
copies have drifted: different action majors (attestation `checkout@v4`
by tag, smeltery and ferric by SHA, at different SHAs), different ruff pins
enforced differently, and two of three smoke tests written as YAML steps.

## Decision

Three reusable workflows in **mgoldey/attestation** (public), each
`on: workflow_call` only, typed inputs, `permissions: contents: read`, no
secrets declared or inherited. attestation calls them by local path
(`uses: ./.github/workflows/reusable-x.yml`), so every PR here exercises the
exact files the other two call; smeltery and ferric call them by full commit
SHA (`uses: mgoldey/attestation/.github/workflows/reusable-x.yml@<sha>`), so a
change here reaches them only when they bump the SHA.

| Workflow | What it is | attestation | smeltery | ferric |
|---|---|---|---|---|
| `reusable-python-gates.yml` | one matrix cell: uv sync, ruff format --check, ruff check, type check, `uv lock --check`, caller checks, tests | `gates` (4 cells) | `test` (2 cells) | `python-quality` (optional, see below) |
| `reusable-pre-commit.yml` | `pre-commit run --all-files` with the project's pre-commit or a pinned `uvx pre-commit@X` | new `pre-commit` job | `pre-commit` | `pre-commit` |
| `reusable-wheel-smoke.yml` | build (the release command) + optional tag guard, one artifact, then a smoke matrix (python x cells the BUILT wheel computes) | `wheel smoke test`, and the build half of both release lines | `wheel-build` + `wheel-smoke`, and the build half of `release` | not used |

Each was checked against all three repositories before it was written; what
was not common stayed out (below).

### What is genuinely common, and the interface it got

- **The gates cell.** attestation's `gates` and smeltery's `test` are the same
  shape: checkout, setup-uv at a python, `uv sync <args>`, extra installs,
  ruff, type check, lock check, then the repository's own checks and tests.
  The MATRIX stays in the caller (`strategy.matrix` on the calling job): the
  axes differ per repository (attestation os x 3.12/3.13, smeltery 3.11/3.12),
  and a reusable that owned the matrix would need a JSON input for every axis.
  ruff runs either as the project's locked ruff (`ruff-version: ""`,
  attestation) or as an exact `uvx ruff@X` (smeltery 0.16.10, ferric 0.15.8):
  both repos pin it exactly because `ruff format` output moves between
  releases. Report-only is an explicit input (`ruff-check-blocking`,
  `ty-blocking`, default blocking) because ferric's ruff check and ty are
  report-only on purpose, with an exit criterion in their step comments; a
  report-only step tees its output to the step summary rather than hiding
  behind `continue-on-error`. `pre-sync` exists for ferric's bandit, which
  must run before `.venv` exists or it scans third-party code.
- **pre-commit as CI.** smeltery and ferric both run their hooks as a CI job
  so "what a contributor's hooks check and what CI checks cannot drift"
  (smeltery's words; ferric's says the same of `--no-verify`). The two ways
  of getting pre-commit are the one real difference: smeltery's hooks are
  `repo: local` calling `uv run`, so pre-commit must come from the synced
  project; ferric's hooks are remote repos with a pinned
  `pre-commit==4.6.2`. One input (`pre-commit-version`, empty = the
  project's) covers both. attestation had no such job (its `gates` runs the
  same tools as separate steps); it gains one, with `SKIP=pytest` because
  gates already runs the suite four times. That job is new coverage, not a
  rename: the xenon hook excludes `*.ipynb` and gates does not, which is the
  kind of drift it now reports.
- **Build the release way, smoke the built wheel.** attestation (two wheels,
  plain pip into a clean venv) and smeltery (one wheel, one smoke cell per
  extra the wheel's own metadata declares, `scripts/wheel_smoke.py`) both
  build in one job and install elsewhere. The template builds with the
  caller's command (`uv build` builds the sdist and the wheel FROM it -- the
  property smeltery's comment says v0.1.0 was lost to), uploads one
  artifact, then runs `smoke-command` per (python, cell); cells come from
  `matrix-command` run against the BUILT files, so smeltery's
  `list-extras dist/*.whl` drops in unchanged. Node setup is an input because
  smeltery's wheel ships a built UI.
- **The release guard.** Every release here refuses a tag its files do not
  carry. With `tag-prefix` set, the template checks every built file's name
  (wheel and sdist) for exactly the tag's version and no local segment
  (smeltery's guard, which works for a VCS-derived version); with
  `version-file` it also checks the static `[project].version` (attestation's
  guard, kept: test_release_workflow.py pins it) before building.
  `require-clean-tree` and `twine-check` are smeltery's two other refusals.

### What deliberately stayed per repository

- **Publishing.** PyPI trusted publishing does not work from a reusable
  workflow. docs.pypi.org/trusted-publishers/troubleshooting/, "Reusable
  workflows on GitHub": "Reusable workflows cannot currently be used as the
  workflow in a Trusted Publisher. This is a practical limitation, and is
  being tracked in warehouse#11096." pypa/gh-action-pypi-publish's README:
  "Trusted publishing cannot be used from within a reusable workflow at this
  time. It is recommended to instead create a non-reusable workflow that
  contains a job calling your reusable workflow, and then do the trusted
  publishing step from a separate job within that non-reusable workflow."
  (Both read 2026-11-20 from the warehouse and pypa repositories' sources;
  docs.pypi.org itself was not reachable from the build machine.) So every
  release workflow keeps a top-level `publish` job (`environment: pypi`,
  `id-token: write`, download artifact, `pypa/gh-action-pypi-publish` by SHA)
  that `needs:` the job calling the template. The PyPI publishers stay as
  registered (each names its repository's own release workflow file and
  environment `pypi`); nothing on PyPI changes. tests/test_shared_workflows.py
  pins that no template publishes or asks for `id-token`.
- **ferric's native build.** Rust toolchain, libint2 (two variants), the
  manylinux_2_28 container, maturin abi3, auditwheel/delocate/delvewheel,
  `release-check.py`, MPI wheels: none of it exists in the other two, and a
  template with that many inputs would be ferric's workflow with extra
  indirection. ferric's `wheels.yml`, `wheels-macwin.yml`,
  `native-deps-macwin.yml`, `platform-tests.yml`, `ci.yml`'s Rust jobs and
  its release stay in ferric. ferric does not use `reusable-wheel-smoke.yml`:
  its smoke test needs the wheel variant matrix and the libint2 identity
  checks.
- **Docs.** attestation builds mkdocs, ferric mdBook (book.yml), smeltery
  none: nothing shared. The rest of attestation's CI (example flows, the 3.11
  schema job, training callbacks, Pages deploy) is attestation's own.
- **smeltery's specialised jobs** (bench coverage floor, docking, temporal,
  ml-potential, ui, docker, cross-server, stack, compat) are the gates shape
  with one-off assertions ("a skip here is a failure"); they could call
  `reusable-python-gates.yml` with lint off, but that is stage 2's choice, not
  a requirement of it.

## Pins

Every third-party action in every workflow file here is pinned by full commit
SHA with a `# vX.Y.Z` comment, ONE SHA per action across all files
(tests/test_shared_workflows.py). The SHAs are the ones smeltery already runs,
each verified with `git ls-remote` against its tag on 2026-11-20:

| Action | Tag | SHA |
|---|---|---|
| actions/checkout | v7.0.1 | 3d3c42e5aac5ba805825da76410c181273ba90b1 |
| astral-sh/setup-uv | v10.3.3 | 68559e99cc807c1eb7e9238fd43a15bba460eec2 |
| actions/setup-python | v7.1.0 | 6d4fd22b20465ccc0f7354476588e8df7c9ca0e0 |
| actions/setup-node | v7.1.0 | 949feb2413d6458794dcd2491c4babbbce0c15c1 |
| actions/upload-artifact | v7.0.2 | cf430e030ddbb5b0abf93d22962f4752f3646cd9 |
| actions/download-artifact | v8.0.2 | 9000827ccba6bdab643e8b6fd33ac0654aef8333 |
| pypa/gh-action-pypi-publish | v1.14.3 | 7245906fb61fa33a13554f9d5fb856fe61758086 (commit; the tag is annotated) |
| actions/upload-pages-artifact | v3.0.1 | 56afc609e74202658d3ffba0e8f6dda462b719fa |
| actions/deploy-pages | v4.0.5 | d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e |

This moves attestation from `checkout@v4`/`setup-uv@v5`/`upload-artifact@v4`/
`download-artifact@v4`/`pypi-publish@release/v1` (tags) to those SHAs.

## Check names

A job that calls a reusable workflow reports `<caller job name> / <called job
name>`; GitHub offers no way to drop the suffix. The caller names are kept, so
every old name survives as a prefix:

| Before | After |
|---|---|
| `gates (ubuntu-latest, py3.12)` (x4 cells) | `gates (ubuntu-latest, py3.12) / gates` (x4) |
| `wheel smoke test` | `wheel smoke test / build`, `wheel smoke test / smoke (py3.12, default)` |
| -- | `pre-commit / pre-commit` (new) |
| `docs site`, `example flows (offline)`, `attestation-schema on py3.11`, `attestation-schema training callbacks (py3.11)` | unchanged (not converted) |
| release: `build + smoke (attestation)` / `(attestation-schema)` | `... / build`, `... / smoke (py3.12, default)` (py3.11 for the schema); publish jobs unchanged |

attestation has no repository rulesets (`gh api repos/mgoldey/attestation/rulesets`
returned `[]`); classic branch protection could not be read with the session's
token (403). If it lists required checks, the five changed names above need
updating there when this merges.

## Behaviour kept, and what moved into files

- attestation's wheel smoke checks moved verbatim from YAML steps into
  `scripts/wheel_smoke.sh` (one `smoke-command`); both wheels now build into
  one `dist/` (the globs `attestation-*.whl` and `attestation_schema-*.whl`
  cannot cross). feeds.toml and py.typed are one check instead of two.
- The schema-floor check moved from an inline heredoc to
  `scripts/check_schema_floor.py` (the release's `pre-build`), unchanged in
  behaviour; its broad `except Exception` became `except (OSError,
  ValueError)` (network failure or non-JSON body), so it needs no BLE001
  site. tests/test_release_workflow.py reads the script for the strings it
  pinned in the YAML before.
- The release guard gained the built-file check (tag == every file's version,
  no local segment), which attestation did not have.
- tests/test_release_workflow.py keeps every behaviour it pinned (both tag
  forms and nothing else, disjoint filters, jobs split by prefix, each
  package's own pyproject in its guard, the PyPI floor check and its
  `cannot verify` refusal, the 3.11 schema test before publish, trusted
  publishing with `environment: pypi` and `id-token: write` only, no secrets)
  against the new structure: the inputs a build job passes plus the template
  step that acts on them.

## Stage 2: how smeltery and ferric call them

Replace `<SHA>` with the attestation commit that carries this change (the
merge commit on main). A called workflow checks out the CALLER's repository
(it runs in the caller's `github` context), so `scripts/...` in an input is
the caller's own file. Inputs are validated by tests/test_shared_workflows.py
only for attestation's callers; a caller elsewhere passing an undeclared input
fails at workflow start ("Invalid input"), which is the check there.

### smeltery `.github/workflows/test.yml`

```yaml
  pre-commit:
    uses: mgoldey/attestation/.github/workflows/reusable-pre-commit.yml@<SHA>
    with:
      runs-on: ubuntu-24.04
      python-version: "3.11"
      fetch-depth: 0            # hatch-vcs
      sync-args: --extra dev --extra bench --extra rdkit --no-sources
      args: run --all-files     # was: no --show-diff-on-failure

  test:
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    uses: mgoldey/attestation/.github/workflows/reusable-python-gates.yml@<SHA>
    with:
      name: test
      runs-on: ubuntu-24.04
      python-version: ${{ matrix.python-version }}
      fetch-depth: 0
      timeout-minutes: 90
      enable-cache: "true"
      sync-args: >-
        --extra dev --extra posebusters --extra experiments --extra bench
        --extra attestation --extra dataset --no-sources
      setup: |
        uv run --no-sync python -c "from smeltery import core_record as c; assert c.AVAILABLE, c.NOT_INSTALLED"
        uv pip install pdb2pqr
      ruff-version: "0.16.10"   # tests/test_ruff_pin.py: read this input instead of env RUFF_VERSION
      lock-check: false         # the pre-commit job's uv-lock hook owns it
      test-command: |
        uv run --no-sync pytest -q
        uv run --no-sync pytest -q experiments
        uv run --no-sync python examples/mwe_benzoic.py
```

`OPENBLAS_NUM_THREADS` is workflow-level `env:` in test.yml, which does not
reach a called workflow: put `OPENBLAS_NUM_THREADS=1` in front of the test
commands or export it in `setup` via `$GITHUB_ENV`. The `ty` job's
`uvx ty@0.0.96 check` can move into `ty-command` of the 3.11 cell only if the
`rdkit` extra joins its sync; leaving `ty` as its own job is fine.

```yaml
  wheel:
    uses: mgoldey/attestation/.github/workflows/reusable-wheel-smoke.yml@<SHA>
    with:
      runs-on: ubuntu-24.04
      build-python-version: "3.11"
      fetch-depth: 0
      node-version: "22"
      node-cache-dependency-path: ui/package-lock.json
      pre-build: npm ci --prefix ui && npm run build --prefix ui
      build-command: python -m pip install --upgrade build && python -m build
      twine-check: true
      artifact-name: wheel
      matrix-command: python scripts/wheel_smoke.py list-extras dist/*.whl
      smoke-python-versions: '["3.11"]'
      smoke-command: |
        export OPENBLAS_NUM_THREADS=1
        python scripts/wheel_smoke.py check dist/*.whl --extra "$SMOKE_CELL" --config examples/mwe_benzoic.toml
        if [ "$SMOKE_CELL" = none ]; then
          python scripts/wheel_smoke.py check dist/*.whl --extra none --example examples/mwe_benzoic.py
        fi
```

### smeltery `.github/workflows/release.yml`

```yaml
jobs:
  build:
    uses: mgoldey/attestation/.github/workflows/reusable-wheel-smoke.yml@<SHA>
    with:
      runs-on: ubuntu-24.04
      build-python-version: "3.11"
      fetch-depth: 0
      require-clean-tree: true
      tag-prefix: v             # version-file empty: hatch-vcs derives it
      node-version: "22"
      node-cache-dependency-path: ui/package-lock.json
      pre-build: npm ci --prefix ui && npm run build --prefix ui
      build-command: python -m pip install --upgrade build && python -m build
      twine-check: true
      artifact-name: dist
      matrix-command: python scripts/wheel_smoke.py list-extras dist/*.whl
      smoke-python-versions: '["3.11"]'
      smoke-command: |
        export OPENBLAS_NUM_THREADS=1
        python scripts/wheel_smoke.py check dist/*.whl --extra "$SMOKE_CELL" --config examples/mwe_benzoic.toml --full
        if [ "$SMOKE_CELL" = none ]; then
          python scripts/wheel_smoke.py check dist/*.whl --extra none --example examples/mwe_benzoic.py
          python -m venv /tmp/venv && /tmp/venv/bin/pip install dist/*.whl pytest
          mkdir /tmp/t && cp tests/*.py examples/mwe_benzoic.py /tmp/t/ && cd /tmp/t
          /tmp/venv/bin/python -m pytest -q -p no:cacheprovider test_funnel.py test_anchors.py
          /tmp/venv/bin/python mwe_benzoic.py
        fi

  publish:                      # unchanged: top-level, trusted publishing
    needs: build
    runs-on: ubuntu-24.04
    environment: {name: pypi, url: https://pypi.org/project/smeltery/}
    permissions: {id-token: write}
    steps:
      - uses: actions/download-artifact@9000827ccba6bdab643e8b6fd33ac0654aef8333 # v8.0.2
        with: {name: dist, path: dist/}
      - uses: pypa/gh-action-pypi-publish@7245906fb61fa33a13554f9d5fb856fe61758086 # v1.14.3
```

The PyPI publisher (owner mgoldey, repo smeltery, workflow release.yml,
environment pypi) stays valid: the publish job is still in release.yml.

### ferric `.github/workflows/ci.yml`

```yaml
  pre-commit:
    uses: mgoldey/attestation/.github/workflows/reusable-pre-commit.yml@<SHA>
    with:
      runs-on: ubuntu-22.04
      python-version: "3.11"
      timeout-minutes: 10
      pre-commit-version: "4.6.2"   # remote-repo hooks; no project install
```

ferric's `python-quality` comment rejects `astral-sh/setup-uv` ("one fewer
supply-chain surface"). The templates use it, SHA-pinned, so adopting them is
a policy decision for ferric: the surface moves to one reviewed pin here, not
away. If ferric accepts it, `python-quality` becomes:

```yaml
  python-quality:
    uses: mgoldey/attestation/.github/workflows/reusable-python-gates.yml@<SHA>
    with:
      name: python-quality
      runs-on: ubuntu-22.04
      python-version: "3.11"
      timeout-minutes: 15
      pre-sync: uvx bandit@1.9.4 -ll -q -r . -x terf-tables/generate_interpolation_tables.py
      sync-args: --extra dev --no-install-project
      ruff-version: "0.15.8"        # keep equal to .pre-commit-config.yaml's ruff rev
      ruff-check-blocking: false    # report-only until the exit criterion in ci.yml is met
      ty-command: uvx ty@0.0.81 check --exclude 'terf-tables/generate_interpolation_tables.py'
      ty-blocking: false
      lock-check: false             # ci.yml never ran it; turn on once it passes
```

If not, ferric keeps `python-quality` and calls only `reusable-pre-commit.yml`
(which also uses setup-uv; with the same objection ferric keeps that job too
and stage 2 changes nothing in ferric). Everything else in ferric stays (see
"What deliberately stayed per repository").

## Verification

- `actionlint` 1.7.12 (`uvx --from actionlint-py actionlint`, with shellcheck
  from `shellcheck-py`) reports nothing on every workflow file.
- tests/test_shared_workflows.py: SHA pins with version comments, one SHA per
  action, `workflow_call`-only with typed inputs and a default or `required`,
  every declared input read and every read input declared, callers pass only
  declared inputs and every required one, ci.yml calls all three, the visible
  caller names, every old gate still configured, no template publishes.
- The workflows themselves run only on GitHub: the first run of this change
  is the PR that carries it.
