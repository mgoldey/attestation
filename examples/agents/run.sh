#!/usr/bin/env bash
# The commands examples/agents/README.md shows, run end to end.
# tests/test_golden_paths.py asserts the two agree and runs this.
set -euo pipefail
cd "$(dirname "$0")"
export ATTEST_DB="$(mktemp -d)/attest.db"
uv run attest install --check || true
uv run attest emit
ATTEST_TOOLS=provenance ATTEST_EXPAND=1 uv run python ../flows/mcp_e2e.py --surface provenance --offline
