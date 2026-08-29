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

Four jobs run on every push and pull request (`.github/workflows/ci.yml`):

- **`gates`** — the full local gate set (`ruff format --check`, `ruff check`,
  `ty check`, `uv.lock` sync, the complexity ratchet, `bandit`, `xenon`,
  `pytest`) on Linux and macOS, across Python 3.12 and 3.13.
- **`wheel-smoke`** — builds the wheel, installs it into a clean venv, and
  checks `attest --help` exits 0, the package imports cleanly, and
  `feeds.toml`/`py.typed` are actually packaged.
- **`flows`** — every example flow in `examples/flows/` against the stub
  model server (`--offline`), so the golden paths that need no live LLM are
  checked end to end on every push.
- **`docs`** — `mkdocs build --strict` with the `docs` dependency group
  installed (the `gates` job does not install it), and uploads the built
  site as an artifact.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for what each local gate exists
to catch and the "Where things live" map of which test guards which
directory.
