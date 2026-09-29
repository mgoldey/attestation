#!/usr/bin/env bash
# Records one real agent turn: record-agent.sh NAME SURFACE MODEL QUESTION
#   HERMES_HOME=/path/to/isolated/home record-agent.sh hermes-knowledge \
#     attestation-knowledge gemma4:e2b-it-q4_K_M "What are the main areas I read about?"
# Writes ../../demo/NAME.cast and .gif. Needs a live Hermes install whose
# config has the attestation-<surface> MCP servers and a model server.
set -euo pipefail
cd "$(dirname "$0")"
NAME="$1"; SURFACE="$2"; MODEL="$3"; QUESTION="$4"
OUT="../../demo/$NAME"
mkdir -p "$(dirname "$OUT")"
asciinema rec --overwrite --idle-time-limit 2 --cols 100 --rows 30 \
  --command "bash '$PWD/narrate-agent.sh' '$SURFACE' '$MODEL' \"$QUESTION\"" "$OUT.cast"
agg "$OUT.cast" "$OUT.gif"
echo "wrote $OUT.cast and $OUT.gif"
