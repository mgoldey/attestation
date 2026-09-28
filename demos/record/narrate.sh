#!/usr/bin/env bash
# `attest runs record` writing a sweep's results in the shape `runs scan`
# reads, then scanning and comparing it in the same call -- the write side of
# the ledger, for a run the researcher only has numbers for. Runs in a fresh
# temp directory against a throwaway database. Not run directly -- record.sh
# invokes this under `asciinema rec`.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$(mktemp -d)"
export ATTEST_DB="$PWD/attest.db"
clear

run() {
  echo "\$ $*"
  "$@"
  echo
  sleep 1
}
attest() { uv run -q --project "$REPO" attest "$@"; }

echo "# Three warmup schedules, and only their word error rates to show for it."
echo
run attest runs record warmup --arm short wer=0.182 --arm long wer=0.176 --arm none wer=0.241 --corpus librispeech-dev --scan
echo "\$ find results configs -type f && cat results/warmup_long.json"
find results configs -type f | sort
cat results/warmup_long.json
echo
sleep 3
