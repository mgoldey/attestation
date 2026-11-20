# Tests

How do I know it still works? `uv run pytest` plus three static gates
locally, the same eight checks in CI across two OSes and two Python
versions, and `tests/test_golden_paths.py` runs every `examples/` path for
real and pins a line of its output — so the docs are what the suite asserts.

## Tests

    uv run pytest
    uv run ruff check .                     # lint (E, F, W, I, BLE; line length 100)
    uv run ty check                         # type check
    uv run radon cc -s -n C src/attestation      # complexity report (empty = nothing worse than B)

`tests/test_golden_paths.py` runs every `examples/` path whose prerequisite is
`none — pure local computation` and pins one line of its output, so the
README under each path is asserted, not just written. CI's `flows` job runs
`examples/flows/run_all.py --offline` on every push.

## CI jobs

These jobs run on every push and pull request (`.github/workflows/ci.yml`).
The first three call the shared reusable workflows in the same directory
(`reusable-python-gates.yml`, `reusable-pre-commit.yml`,
`reusable-wheel-smoke.yml`), which smeltery and ferric call too by a pinned
commit SHA, so their checks read `<job> / <called job>`
([design](../superpowers/specs/2026-11-20-shared-ci-templates-design.md)):

- **`gates`** — the full local gate set (`ruff format --check`, `ruff check`,
  `ty check`, `uv.lock` sync, the complexity ratchet, `bandit`, `xenon`,
  `pytest`) on Linux and macOS, across Python 3.12 and 3.13.
- **`pre-commit`** — `.pre-commit-config.yaml`'s hooks as-is (`SKIP=pytest`:
  gates runs the suite), so the local hooks and CI cannot drift.
- **`wheel-smoke`** — builds both wheels the release way, installs them into
  clean venvs, and runs `scripts/wheel_smoke.sh`: `attest --help` exits 0,
  the package imports cleanly, `feeds.toml`/`py.typed` are packaged, and the
  attestation-schema wheel installs alone and round-trips a record.
- **`flows`** — every example flow in `examples/flows/` against the stub
  model server (`--offline`), so the golden paths that need no live LLM are
  checked end to end on every push.
- **`docs`** — `mkdocs build --strict` with the `docs` dependency group
  installed (the `gates` job does not install it), and uploads the built
  site as an artifact.
- **`schema-py311`** and **`schema-integrations`** — the attestation-schema
  wheel's own tests on its 3.11 floor, bare and with the training-callback
  extras.

A fifth workflow, `docs` (`.github/workflows/docs.yml`), runs only on a
push to `main`: the same strict build, then a deploy to GitHub Pages at
<http://matthew.thegoldeys.com/attestation/>. A page that would fail the
`docs` CI job never deploys, because both run the same command.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for what each local gate exists
to catch and the "Where things live" map of which test guards which
directory.
