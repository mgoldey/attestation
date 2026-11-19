#!/usr/bin/env bash
# The commands examples/mlflow-records/README.md shows, run end to end.
# tests/test_golden_paths.py asserts the two agree and runs this.
set -euo pipefail
cd "$(dirname "$0")"
STORE="$(mktemp -d)"
cp -R ../flows/training/mlruns "$STORE/mlruns"
for run in "$STORE"/mlruns/*/*/; do mkdir -p "${run}artifacts"; done
uv run --group examples python export_records.py --tracking-uri "$STORE/mlruns" --out "$STORE/records" --arm-param C --seed-param seed --direction auc=max
diff -r records "$STORE/records" && echo "byte-identical to the committed records/"
uv run python -c "import attestation_schema as s; r = s.read('records/run-06f405c125804496876ebfa098ad4c94'); print(r.schema, r.family, r.arm, [(m.name, m.value, m.direction) for m in r.metrics if m.name == 'auc'], len(r.history[-1].points), 'train_loss points')"
