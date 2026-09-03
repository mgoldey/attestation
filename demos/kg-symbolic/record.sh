#!/usr/bin/env bash
# Records an asciinema .cast of the kg.* + sym.* walkthrough (demo.py),
# then converts it to a GIF with agg. Needs a database seeded by
# seed_kg_db.py first -- kg.* has nothing to show against an empty graph.
# Neither the .cast nor the .gif is committed. See demos/README.md.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-../../demo/kg-symbolic}"
mkdir -p "$(dirname "$OUT")"

if [ -z "${ATTEST_DB:-}" ] || [ ! -f "$ATTEST_DB" ]; then
  echo "ATTEST_DB must point at a database seeded by seed_kg_db.py" >&2
  echo "  uv run python seed_kg_db.py /path/to/demo.db" >&2
  echo "  ATTEST_DB=/path/to/demo.db $0" >&2
  exit 1
fi

asciinema rec --overwrite --command "bash -c 'clear; echo \"\$ uv run python demo.py\"; ATTEST_DB=$ATTEST_DB uv run python demo.py 2>/dev/null; sleep 2'" "$OUT.cast"
agg "$OUT.cast" "$OUT.gif"
echo "wrote $OUT.cast and $OUT.gif"
