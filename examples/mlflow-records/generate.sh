#!/usr/bin/env bash
# Regenerate records/ from the committed MLflow sweep in ../flows/training/mlruns.
# tests/test_mlflow_records.py re-runs the same export and asserts the bytes
# are identical, so a change to the mapping shows up as a diff here.
#
# git cannot commit an empty directory, and every run in the committed sweep
# has an empty artifacts/ (it logged none). MLflow 3's file store refuses a
# run directory without one ("Run ... not found"), so the store is copied and
# those directories put back -- the store MLflow wrote, as MLflow wrote it.
set -euo pipefail
cd "$(dirname "$0")"
STORE="$(mktemp -d)"
cp -R ../flows/training/mlruns "$STORE/mlruns"
for run in "$STORE"/mlruns/*/*/; do mkdir -p "${run}artifacts"; done
rm -rf records
uv run --group examples python export_records.py --tracking-uri "$STORE/mlruns" --out records --arm-param C --seed-param seed --direction auc=max
