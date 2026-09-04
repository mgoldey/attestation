#!/usr/bin/env bash
# Records an asciinema .cast of a real hermes chat session calling
# runs.ask, then converts it to a GIF with agg. Needs the attestation MCP
# server in ~/.hermes/config.yaml pointed at a scratch database seeded from
# examples/workspace -- see README.md. Neither the .cast nor the .gif is
# committed. Not run by any test: this is the one demo driving a real
# agent, which needs a live Hermes install this repo cannot assume.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-../../demo/hermes}"
mkdir -p "$(dirname "$OUT")"

asciinema rec --overwrite --command "bash -c '
  cd \"\$(cd ../../examples/workspace && pwd)\"
  clear
  echo \"\$ hermes chat -q \\\"Compare the kdsweep arms by wer and tell me the winner.\\\" -s attestation-provenance\"
  hermes chat -q \"Compare the kdsweep arms by wer and tell me the winner.\" --cli -s attestation-provenance -m gemma4:e2b-it-q4_K_M
  sleep 2
'" "$OUT.cast"

agg "$OUT.cast" "$OUT.gif"
echo "wrote $OUT.cast and $OUT.gif"
