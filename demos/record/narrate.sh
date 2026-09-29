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
export LEDGER_METRIC_DIRECTION_FILE="$PWD/metric_direction.toml"  # --direction writes here
clear

run() {
  echo "\$ $*"
  "$@"
  echo
  sleep 1
}
attest() { uv run -q --project "$REPO" attest "$@"; }

echo "# Three annealing temperatures, and a conductivity for each in a lab notebook."
echo
run attest runs record anneal --arm 300k conductivity=412 --arm 450k conductivity=655 --arm 600k conductivity=631 --direction conductivity=higher_is_better --scan
echo "\$ find results configs -type f && cat results/anneal_450k.json"
find results configs -type f | sort
cat results/anneal_450k.json
echo
sleep 3
