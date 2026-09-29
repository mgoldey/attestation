#!/usr/bin/env bash
# The run ledger and claim checker over an experimental and a computational
# project -- a catalyst screen (yield, %) and a basis-set convergence check
# (MAE, kcal/mol) -- with a draft carrying one stale number, one claim about
# an experiment never run, and one citation key no reference list holds. workspace/FINDINGS.md says the numbers are
# invented for the demo. No model, no network. Not run directly -- record.sh
# invokes this under `asciinema rec`.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$(dirname "$0")/workspace"
export ATTEST_DB="$(mktemp -d)/attest.db"
export LEDGER_METRIC_DIRECTION_FILE="$PWD/metric_direction.toml"  # yield: higher is better
export ATTEST_BIB_PATHS="$PWD/references.bib"  # what cite= keys resolve against
clear

run() {
  echo "\$ $*"
  "$@"
  echo
  sleep 1
}
attest() { uv run -q --project "$REPO" attest "$@"; }

run attest runs scan --root .
run attest runs compare screen --metric yield
run attest runs compare basis --metric mae
echo "\$ attest claims FINDINGS.md"
attest claims FINDINGS.md || true
echo
sleep 1
run attest claims FINDINGS.md --coverage
sleep 2
