#!/usr/bin/env bash
# Regenerate workspace/ from smeltery's own prebuilt runs, written by
# smeltery's own core-record writer (`smeltery record core`). Never edit the
# output by hand: tests/test_smeltery_records.py reads it as a real producer's.
#
#   SMELTERY=/path/to/smeltery tests/data/smeltery/generate.sh
#
# The one change made before smeltery writes: the legacy run-<digest12>.json
# names the machine it ran on (host.node) and a checkout path under /home/.
# Both are scrubbed IN THE INPUT, so the core record, the manifest
# (attestation/digests.json) and every sha256 in `data` are computed by
# smeltery over the scrubbed bytes and still bind each other.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
smeltery="$(cd "${SMELTERY:?set SMELTERY to a smeltery checkout}" && pwd)"
out="$here/workspace/mwe-benzoic"
rm -rf "$here/workspace"
mkdir -p "$out"
for name in benzoic-rhf benzoic-pbe; do
  cp -R "$smeltery/src/smeltery/prebuilt/$name/run" "$out/$name"
  python3 - "$out/$name"/run-*.json <<'PY'
import re, sys
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
text = re.sub(r'("node": )"[^"]*"', r'\1"scrubbed"', text)
text = re.sub(r"/home/[^/\"]+/", "/scrubbed/", text)
open(path, "w", encoding="utf-8").write(text)
PY
  (cd "$smeltery" && uv run --isolated --frozen --extra attestation smeltery record core "$out/$name")
done
