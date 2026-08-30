#!/usr/bin/env bash
# The command examples/ranking/README.md shows, run end to end.
# tests/test_golden_paths.py asserts the two agree and runs this.
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python examples/ranking/rank_rows.py
