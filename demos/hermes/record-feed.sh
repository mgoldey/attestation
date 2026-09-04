#!/usr/bin/env bash
# Records an asciinema .cast of a real hermes chat session calling
# feed.ask, then converts it to a GIF with agg. Needs the attestation-feed
# MCP server in ~/.hermes/config.yaml pointed at a database seeded by
# ../feed/seed_feed_db.py -- see README.md. Neither the .cast nor the .gif
# is committed. Not run by any test: this is a demo driving a real agent,
# which needs a live Hermes install this repo cannot assume.
#
# `-t attestation-feed` restricts hermes's OWN built-in toolset
# (filesystem, terminal, browser, ...) to just this MCP server. Without
# it, live testing (2026-09-03) showed gemma4:e2b under ~16k prompt
# tokens of unrelated tool schemas either inventing a missing precondition
# ("I need to know which feeds they subscribe to first" -- not asked for)
# before calling a tool it had already correctly identified, or reasoning
# to the right call and then printing it as text instead of invoking it.
# Cutting the built-in toolset (~11k-14k tokens instead), together with
# the skill text's explicit "call it now" and "this is documentation, not
# something to type" additions, made both failures go away in repeated
# live testing.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-../../demo/hermes-feed}"
mkdir -p "$(dirname "$OUT")"

asciinema rec --overwrite --command "bash '$PWD/narrate-feed.sh'" "$OUT.cast"
agg "$OUT.cast" "$OUT.gif"
echo "wrote $OUT.cast and $OUT.gif"
