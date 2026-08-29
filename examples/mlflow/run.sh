#!/usr/bin/env bash
# The commands examples/mlflow/README.md shows, run end to end.
# tests/test_golden_paths.py asserts the two agree and runs this.
set -euo pipefail
cd "$(dirname "$0")"
export ATTEST_DB="$(mktemp -d)/attest.db"
uv run attest runs scan --root ../flows --project training
uv run attest runs compare c_sweep --metric auc
uv run attest claims ../flows/training/FINDINGS.md || true
uv run --group examples python ../flows/training/train_mlflow.py --out "$(mktemp -d)"
