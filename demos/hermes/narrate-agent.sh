#!/usr/bin/env bash
# One real `hermes chat` turn on one attestation surface, for record-agent.sh.
#   narrate-agent.sh SURFACE MODEL QUESTION
# `-t SURFACE` restricts Hermes to that one MCP server (see record-feed.sh for
# why), and HERMES_HOME, when set, keeps the turn out of the operator's real
# Hermes home. Not run directly -- record-agent.sh runs it under asciinema.
set -euo pipefail
SURFACE="$1"; MODEL="$2"; QUESTION="$3"
cd "$(mktemp -d)"  # hermes loads CLAUDE.md/AGENTS.md from the cwd
clear
echo "\$ hermes chat -q \"$QUESTION\" -t $SURFACE -m $MODEL"
hermes chat -q "$QUESTION" --cli -s "$SURFACE" -t "$SURFACE" -m "$MODEL" 2>&1 \
  | grep -v 'RuntimeWarning\|isawaitable\|tracemalloc'
sleep 3
