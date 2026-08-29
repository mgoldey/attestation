#!/usr/bin/env bash
# The commands examples/ranking/README.md shows, run end to end:
#   ./run.sh
# or, from the repo root:
#   uv run python examples/ranking/rank_rows.py
# tests/test_golden_paths.py asserts the two agree and runs this.
set -euo pipefail
cd "$(dirname "$0")"
uv run python rank_rows.py
