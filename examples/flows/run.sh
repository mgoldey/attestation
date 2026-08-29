#!/usr/bin/env bash
# The command examples/flows/README.md shows, run end to end.
# tests/test_golden_paths.py asserts the two agree and runs this.
set -euo pipefail
cd "$(dirname "$0")"
export ATTEST_DB="$(mktemp -d)/attest.db"
uv run --group examples python run_all.py --offline
