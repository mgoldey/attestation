#!/usr/bin/env bash
# The commands examples/citations/README.md shows, echoed and run one at a
# time. Not run directly -- record.sh invokes this under `asciinema rec`.
set -euo pipefail
CITATIONS="$1"
cd "$CITATIONS"
export ATTEST_DB="$(mktemp -d)/attest.db"
export RESEARCH_ROOT=$PWD/../workspace
clear

run() {
  echo "\$ $*"
  "$@"
  echo
  sleep 1
}

run uv run attest runs scan --root ../workspace
echo "\$ attest claims DRAFT.md"
uv run attest claims DRAFT.md || true
echo
sleep 1
echo "\$ uv run python check_citations.py"
uv run python check_citations.py 2>/dev/null
echo
sleep 2
