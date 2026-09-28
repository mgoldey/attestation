#!/usr/bin/env bash
# Records an asciinema .cast of narrate.sh and converts it to a GIF with agg.
# Neither output is committed here -- rerun this to regenerate; the GIF shown
# in docs/guides/demos.md is copied to docs/media/. See demos/README.md.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-../../demo/record}"
mkdir -p "$(dirname "$OUT")"

asciinema rec --overwrite --idle-time-limit 2 --cols 100 --rows 34 --command "bash '$PWD/narrate.sh'" "$OUT.cast"
agg "$OUT.cast" "$OUT.gif"
echo "wrote $OUT.cast and $OUT.gif"
