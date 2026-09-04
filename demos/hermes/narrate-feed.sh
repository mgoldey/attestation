#!/usr/bin/env bash
# The hermes chat call under recording. Not run directly -- record-feed.sh
# invokes this under `asciinema rec`, avoiding the doubly-nested quoting
# that broke embedding this command inline in record-feed.sh's own
# `asciinema rec --command "bash -c '...'"` string.
set -euo pipefail
clear
QUESTION="I'm demo-reader. What should I read today?"
echo "\$ hermes chat -q \"$QUESTION\" -s attestation-feed -t attestation-feed"
hermes chat -q "$QUESTION" --cli -s attestation-feed -t attestation-feed -m gemma4:e2b-it-q4_K_M
sleep 2
