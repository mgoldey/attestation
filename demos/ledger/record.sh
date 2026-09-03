#!/usr/bin/env bash
# Records an asciinema .cast of the run-ledger walkthrough by narrating and
# running examples/workspace/run.sh's own commands (the ones
# examples/workspace/README.md documents and test_golden_paths.py checks
# against that script), then converts the .cast to a GIF with agg. Neither
# output is committed -- rerun this to regenerate. See
# demos/README.md for setup.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-../../demo/ledger}"
mkdir -p "$(dirname "$OUT")"
WORKSPACE="$(cd ../../examples/workspace && pwd)"

asciinema rec --overwrite --command "bash '$PWD/narrate.sh' '$WORKSPACE'" "$OUT.cast"
agg "$OUT.cast" "$OUT.gif"
echo "wrote $OUT.cast and $OUT.gif"
