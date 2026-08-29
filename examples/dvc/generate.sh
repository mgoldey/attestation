#!/usr/bin/env bash
# examples/dvc/generate.sh
#
# Runs the real `dvc` CLI (dvc 3.x, confirmed by the pinned version below)
# against this directory's dvc.yaml/params.yaml/train.py, then commits what
# it wrote. Regenerating rewrites dvc.lock and metrics/*.json in place --
# `lr` in {0.01, 0.1, 1, 10} is fixed, train.py's random_state and split are
# fixed, so the four arms and their metric values are stable across a
# regeneration; only dvc.lock's content-addressed md5 hashes change.
#
# `dvc init --no-scm -q` skips DVC's usual requirement of a surrounding git
# repo (this directory is not one) and its own `.git`-style bookkeeping;
# `dvc repro -q` runs every stage dvc.yaml declares and writes dvc.lock.
#
#     uv run --with dvc --with scikit-learn --no-project bash -c \
#         'dvc init --no-scm -q && dvc repro -q'
#
# `dvc init` writes `.dvc/config` (small, path-free: `[core]\n no_scm =
# true\n` -- kept, see below), `.dvc/cache` (a content-addressed blob store,
# a full copy of every metrics/*.json keyed by hash -- deleted, pure
# regeneratable cache) and `.dvc/tmp` (lock files and a boot-time marker for
# this one invocation -- deleted, transient). `_dvc_runs` in
# ledger_adapters/generic.py never reads any file under `.dvc/`: dvc.yaml,
# params.yaml and dvc.lock are the record.
set -euo pipefail
cd "$(dirname "$0")"

DVC_VERSION="3.67.1" # pin used to produce the committed fixture -- informational only

rm -rf dvc.lock metrics .dvc .dvcignore

uv run --with dvc --with scikit-learn --no-project bash -c \
  'dvc init --no-scm -q && dvc repro -q'

rm -rf .dvc/cache .dvc/tmp
rm -f .dvcignore

echo "dvc $DVC_VERSION wrote:"
echo "  dvc.lock"
for f in metrics/*.json; do
  echo "  $f"
done
[ -f .dvc/config ] && echo "  .dvc/config (kept: small, path-free)"
echo "regenerated $(find metrics -name '*.json' | wc -l | tr -d ' ') metric file(s)"
