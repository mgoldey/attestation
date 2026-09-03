#!/usr/bin/env bash
# Records an asciinema .cast of the claims + citations walkthrough (the
# commands examples/citations/README.md documents), then converts it to a
# GIF with agg. Neither output is committed -- rerun this to regenerate.
# See demos/README.md for setup.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-../../demo/claims}"
mkdir -p "$(dirname "$OUT")"
CITATIONS="$(cd ../../examples/citations && pwd)"

asciinema rec --overwrite --command "bash '$PWD/narrate.sh' '$CITATIONS'" "$OUT.cast"
agg "$OUT.cast" "$OUT.gif"
echo "wrote $OUT.cast and $OUT.gif"
