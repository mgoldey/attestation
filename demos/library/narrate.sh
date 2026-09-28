#!/usr/bin/env bash
# The reference library over examples/molecular-ai/references.bib: sync a
# BibTeX file into the deduplicated store, search it, follow a reference's
# citations, and write a filtered .bib back out. No model, no network. Not run
# directly -- record.sh invokes this under `asciinema rec`.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$(mktemp -d)"
export ATTEST_DB="$WORK/attest.db"
export ATTEST_BIB_PATHS="$REPO/examples/molecular-ai/references.bib"
cd "$WORK"
clear

run() {
  echo "\$ $*"
  "$@"
  echo
  sleep 1
}
attest() { uv run -q --project "$REPO" attest "$@"; }

run attest library sync --sources bibtex
run attest library status
run attest library search "force field" --limit 5
run attest library related batzner2022equivariant
run attest library export --bib equivariant.bib --year 2022
echo "\$ grep -c '^@' equivariant.bib"
grep -c '^@' equivariant.bib
sleep 3
