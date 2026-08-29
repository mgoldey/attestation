# Golden paths

Each directory here is a use case you can run from a clean clone: a README with
the same seven sections, a `run.sh` that runs the README's own commands, and
its inputs on disk. `tests/test_golden_paths.py` runs every path whose
prerequisite is `none` and pins one line of its output, so the docs are what
the suite asserts.

| path | what it shows | prerequisite | runtime |
|---|---|---|---|
| `workspace/` | the run ledger and claim checker over a two-project workspace with three claims deliberately wrong | `none — pure local computation` | ~1 s |
| `flows/` | forty labelled items scored for two personas, every MCP tool driven over stdio, four MLflow arms read back by the ledger | `none — pure local computation` | ~75 s |
