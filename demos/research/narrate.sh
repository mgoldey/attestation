#!/usr/bin/env bash
# Going out to look: `attest research` searches PubMed and CrossRef and keeps
# the hits in the reference library, then `attest sources add` makes the same
# query a standing topic that every ingest re-runs. Needs the network (arXiv's
# export API is left out because it throttles hard; add it with --sources).
# Not run directly -- record.sh invokes this under `asciinema rec`.
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

run attest research "equivariant neural network interatomic potentials" --sources pubmed,crossref --limit 3
run attest library status
run attest sources add "research:pubmed,crossref?q=equivariant+interatomic+potentials" --title "Equivariant potentials"
sleep 3
