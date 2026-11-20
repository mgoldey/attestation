#!/usr/bin/env bash
# The commands examples/model-registry/README.md shows, run end to end.
# tests/test_golden_paths.py asserts the two agree and runs this.
set -euo pipefail
cd "$(dirname "$0")"
export ATTEST_DB="$(mktemp -d)/attest.db"
uv run attest model trace hf/model
uv run attest model trace models:/iris-clf@champion
uv run attest runs scan --root .. --project model-registry
uv run attest claims DRAFT.md
uv run attest model trace models:/iris-clf/2 || true
