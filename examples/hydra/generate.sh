#!/usr/bin/env bash
# examples/hydra/generate.sh
#
# Runs the real Hydra multirun sweep (hydra-core 1.3.5, confirmed by the
# pinned version below) against this directory's conf/config.yaml and
# train.py, then scrubs what it wrote and commits it. Regenerating rewrites
# multirun/ in place: lr in {0.01, 0.1, 1, 10} is fixed, train.py's
# random_state and split are fixed, so the four arms and their metric
# values are stable across a regeneration; only the date/time sweep
# directory Hydra names changes.
#
# `hydra.job.chdir=True` is required -- see train.py's module docstring for
# why: hydra-core 1.3.5 does not chdir into each job's own output directory
# by default, so without this override all four arms would overwrite one
# top-level metrics.json instead of writing into multirun/<date>/<time>/<n>/.
#
#     uv run --with hydra-core==1.3.5 --with scikit-learn --no-project python \
#         train.py --multirun lr=0.01,0.1,1,10 hydra.job.chdir=True
#
# The pin matches HYDRA_VERSION below and is load-bearing, not cosmetic, the
# same way examples/wandb/generate.py pins wandb: after the sweep runs, this
# script checks the installed version against HYDRA_VERSION and exits 1
# rather than silently commit a fixture this file no longer accurately
# describes.
#
# Hydra writes, per arm, multirun/<date>/<time>/<n>/{.hydra/{config.yaml,
# hydra.yaml,overrides.yaml}, metrics.json, train.log} plus one
# multirun/<date>/<time>/multirun.yaml for the sweep as a whole. `_hydra_runs`
# in ledger_adapters/generic.py never reads train.log or multirun.yaml --
# only .hydra/config.yaml, .hydra/hydra.yaml (for hydra.job.name), and
# metrics.json. scrub.py deletes both unread files outright (train.log is
# empty here since train.py prints to stdout, not Hydra's job logger;
# multirun.yaml duplicates the same absolute paths hydra.yaml carries).
set -euo pipefail
cd "$(dirname "$0")"

HYDRA_VERSION="1.3.5" # pin used to produce the committed fixture -- enforced below, not just echoed

rm -rf multirun

uv run --with hydra-core=="$HYDRA_VERSION" --with scikit-learn --no-project python \
  train.py --multirun lr=0.01,0.1,1,10 hydra.job.chdir=True

INSTALLED_HYDRA_VERSION="$(uv run --with hydra-core=="$HYDRA_VERSION" --no-project python -c \
  'import hydra; print(hydra.__version__)')"
if [ "$INSTALLED_HYDRA_VERSION" != "$HYDRA_VERSION" ]; then
  echo "generate.sh was verified against hydra-core==$HYDRA_VERSION and refuses to commit a" >&2
  echo "fixture produced by hydra-core==$INSTALLED_HYDRA_VERSION -- install the pinned version." >&2
  exit 1
fi

uv run --with hydra-core=="$HYDRA_VERSION" --no-project python scrub.py

echo "hydra $HYDRA_VERSION wrote:"
for f in multirun/*/*/*/metrics.json; do
  echo "  $f"
done
echo "regenerated $(find multirun -name 'metrics.json' | wc -l | tr -d ' ') arm(s)"
