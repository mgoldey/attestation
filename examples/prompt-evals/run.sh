#!/usr/bin/env bash
# The commands examples/prompt-evals/README.md shows, run end to end.
# tests/test_golden_paths.py asserts the two agree; this path needs a live
# model server, so the suite checks README <-> run.sh agreement only and
# never executes it.
set -euo pipefail
cd "$(dirname "$0")/../.."
export ATTEST_DB="$(mktemp -d)/attest.db"
uv run python evals/run_tagging_eval.py --split dev
uv run python evals/transfer_matrix.py --artifact evals/prompts/tagging-2026-08-27.json
uv run python evals/run_reaction_eval.py --split dev
uv run python evals/run_explanation_eval.py --split dev
