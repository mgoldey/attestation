#!/usr/bin/env bash
# The commands examples/workspace/README.md shows, echoed and run one at a
# time so the recording reads like a walkthrough rather than a log dump.
# Not run directly -- record.sh invokes this under `asciinema rec`.
set -euo pipefail
WORKSPACE="$1"
cd "$WORKSPACE"
export ATTEST_DB="$(mktemp -d)/attest.db"
export RESEARCH_ROOT=$PWD
clear

run() {
  echo "\$ $*"
  "$@"
  echo
  sleep 1
}

run uv run attest runs scan --root .
run uv run attest runs list
run uv run attest runs compare kdsweep --metric wer
echo "\$ attest claims speech-distill/FINDINGS.md"
uv run attest claims speech-distill/FINDINGS.md || true
echo
sleep 1
run uv run attest claims speech-distill/FINDINGS.md --coverage
sleep 2
