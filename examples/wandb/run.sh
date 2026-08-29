#!/usr/bin/env bash
# The commands examples/wandb/README.md shows, run end to end.
# tests/test_golden_paths.py asserts the two agree and runs this.
set -euo pipefail
cd "$(dirname "$0")"
export ATTEST_DB="$(mktemp -d)/attest.db"
uv run attest runs scan --root .
uv run attest runs list
uv run attest runs compare generate --metric auc
