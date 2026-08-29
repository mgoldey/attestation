#!/usr/bin/env bash
# The commands examples/citations/README.md shows, run end to end.
# tests/test_golden_paths.py asserts the two agree and runs this.
set -euo pipefail
cd "$(dirname "$0")"
export ATTEST_DB="$(mktemp -d)/attest.db"
export RESEARCH_ROOT=$PWD/../workspace
uv run attest runs scan --root ../workspace
uv run attest claims DRAFT.md || true
uv run python check_citations.py
