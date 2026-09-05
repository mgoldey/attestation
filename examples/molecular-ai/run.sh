#!/usr/bin/env bash
# The commands examples/molecular-ai/README.md shows, run end to end.
# tests/test_golden_paths.py asserts the two agree and runs this.
set -euo pipefail
cd "$(dirname "$0")"
export ATTEST_DB="$(mktemp -d)/attest.db"
export ATTEST_BIB_PATHS=$PWD/references.bib
uv run attest library sync --sources bibtex
uv run attest library search "force field" --limit 5
uv run attest library related batzner2021equivariant
uv run attest kg-report
