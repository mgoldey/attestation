# examples/sacred/generate.py
"""Four Sacred runs in one experiment, in well under thirty seconds.

LogisticRegression on scikit-learn's bundled breast-cancer set (569 rows, no
download), lr in {0.001, 0.01, 0.1, 1.0} -- the same surrogate-vs-reported
split `examples/wandb/generate.py` and `examples/flows/training/
train_mlflow.py` use: an SGDClassifier trained incrementally for ten steps
supplies the `train_loss` curve `_run.log_scalar` logs at each step (Sacred
has no incremental API for LogisticRegression either), and the actual
reported number -- `auc`, also `@ex.main`'s return value, which Sacred
records separately as `run.json`'s `result` -- always comes from the
LogisticRegression fit on the same split. Run via:

    uv run --python 3.12 --with sacred==0.8.7 --with scikit-learn --no-project python generate.py

The version pin matches the SACRED_VERSION constant below and is
load-bearing, not cosmetic, the same way examples/wandb/generate.py pins
wandb: main() refuses to run under any other installed sacred version
rather than silently produce a fixture this file's own docstring no longer
accurately describes.

`Experiment("lr_sweep")` with a `FileStorageObserver("sacred_runs")`, run
four times via `ex.run(config_updates={"lr": ...})`. FileStorageObserver
writes one numbered directory per run, starting at 1, with no separate
"offline" mode to opt into -- unlike W&B, everything here already stays on
disk with no account and no network call.

Sacred needs Python 3.12, not the repo's other pinned versions: `sacred`
imports `pkg_resources` (deprecated, and Python 3.13 does not bundle
`setuptools`, so `pkg_resources` is absent unless something else installs
it) at import time. Under 3.12 it imports fine, with the pkg_resources
deprecation warning `generate.py` and `run.sh` accept as harmless. Confirmed
against sacred 0.8.7.

The sacred_runs/ this writes is committed beside it: the first real Sacred
directory the ledger reader (_sacred_runs in ledger_adapters/generic.py) has
read. Regenerating changes run ids only in the sense that Sacred numbers
runs by how many already exist in the target directory -- run() with a
clean sacred_runs/ always numbers 1-4, so the four `<experiment.name>/<n>`
run names are actually stable across a regeneration, unlike W&B's or
MLflow's hash-derived ids. Running this file deletes and rewrites the
committed sacred_runs/ in place.

Sacred's FileStorageObserver writes: this machine's hostname, CPU model, and
GPU inventory, plus the OS string and Python version (all under run.json's
`host` key); this repo's absolute path (`experiment.base_dir` and, for every
logged source file, the second element of `experiment.sources`, and
`experiment.repositories` if the driver script sits in a git repo);
captured stdout/stderr (`cout.txt`); and a content-hashed copy of this very
script under `_sources/`. `_sacred_runs` reads only `run.json`'s
`experiment.name`, `status`, `start_time`, `stop_time` and `result`, plus
`config.json` and `metrics.json` in full -- never `host`, `base_dir`,
`sources`, `repositories`, `cout.txt`, or `_sources/` -- so scrubbing them
costs the reader nothing. `scrub()` below removes them after training.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

FAMILY = "lr_sweep"
ARMS = (0.001, 0.01, 0.1, 1.0)
HERE = Path(__file__).resolve().parent

# The sacred version this fixture was produced with, and the only one this
# file will run under -- see _require_pinned_sacred_version(). The Python
# version note is separate, see the module docstring's pkg_resources
# paragraph; that constraint is applied in run.sh, not here.
SACRED_VERSION = "0.8.7"

# Keys run.json's `experiment` object writes that name this machine or this
# repository's location. `name` is the only one _sacred_runs or a reader of
# this fixture could ever want.
_SCRUB_EXPERIMENT_KEYS = ("base_dir", "sources", "repositories")


def scrub(sacred_dir: Path) -> None:
    """Strip attribution, machine paths, and captured output Sacred writes.

    The reader (_sacred_runs in ledger_adapters/generic.py) reads only
    run.json's `experiment.name`, `status`, `start_time`, `stop_time` and
    `result`, plus config.json and metrics.json in full -- never `host`,
    `experiment.base_dir`/`sources`/`repositories`, `cout.txt`, or
    `_sources/`. Nothing here is load-bearing for the ledger; committing it
    anyway would put this machine's hostname, CPU/GPU inventory, and this
    repo's absolute path into version control.
    """
    sacred_dir = sacred_dir.resolve()
    shutil.rmtree(sacred_dir / "_sources", ignore_errors=True)
    for run_dir in sorted(p for p in sacred_dir.iterdir() if p.is_dir()):
        (run_dir / "cout.txt").unlink(missing_ok=True)

        run_json_path = run_dir / "run.json"
        if not run_json_path.is_file():
            continue
        run_json = json.loads(run_json_path.read_text())
        run_json.pop("host", None)
        experiment = run_json.get("experiment")
        if isinstance(experiment, dict):
            for key in _SCRUB_EXPERIMENT_KEYS:
                experiment.pop(key, None)
        # meta.command is always the literal subcommand ("run"), never a
        # path, but meta.options carries whatever was passed on the command
        # line -- none of it here, since this script calls ex.run() directly
        # rather than through Sacred's CLI, so there is nothing machine- or
        # person-specific left to strip from it.
        run_json_path.write_text(json.dumps(run_json, indent=2, sort_keys=True) + "\n")


def _require_pinned_sacred_version(sacred_module) -> None:
    """Refuse to run under any sacred but the one this file was verified against.

    SACRED_VERSION is not decorative -- see examples/wandb/generate.py's
    identical guard for the precedent this follows. Failing loudly here,
    before any run is created, is cheaper than debugging a fixture whose
    shape silently drifted from what this file's docstring describes.
    """
    if sacred_module.__version__ != SACRED_VERSION:
        raise SystemExit(
            f"generate.py was verified against sacred=={SACRED_VERSION} and refuses to"
            f" run under sacred=={sacred_module.__version__}. install the pinned version,"
            f" e.g.:\n\n"
            f"    uv run --python 3.12 --with sacred=={SACRED_VERSION}"
            f" --with scikit-learn --no-project python generate.py"
        )


def train(out: Path, seed: int = 0) -> list[dict]:
    import numpy as np
    import sacred as sacred_module
    from sacred import Experiment
    from sacred.observers import FileStorageObserver
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    _require_pinned_sacred_version(sacred_module)

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    ex = Experiment(FAMILY)
    ex.observers.append(FileStorageObserver(str(out / "sacred_runs")))

    @ex.config
    def _cfg():
        lr = 0.01  # noqa: F841 -- overridden per arm by config_updates below
        seed = 0  # noqa: F841

    @ex.main
    def _run(_run, lr, seed):
        sgd = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=lr, random_state=seed)
        classes = np.array([0, 1])
        for step in range(10):
            sgd.partial_fit(X_tr, y_tr, classes=classes)
            p = np.clip(sgd.predict_proba(X_tr)[:, 1], 1e-6, 1 - 1e-6)
            loss = float(-np.mean(y_tr * np.log(p) + (1 - y_tr) * np.log(1 - p)))
            _run.log_scalar("train_loss", loss, step)

        clf = LogisticRegression(C=1.0 / lr, max_iter=2000, random_state=seed).fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, prob))
        _run.log_scalar("auc", auc)
        return auc

    results = []
    for lr in ARMS:
        run = ex.run(config_updates={"lr": lr, "seed": seed})
        results.append({"lr": lr, "run_id": run._id, "auc": run.result})
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=HERE, help="directory to hold sacred_runs/")
    ap.add_argument("--json", type=Path, help="also write a machine-readable summary here")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    sacred_dir = args.out / "sacred_runs"
    if sacred_dir.exists():
        shutil.rmtree(sacred_dir)

    results = train(args.out)
    scrub(sacred_dir)
    elapsed = time.perf_counter() - t0

    for r in results:
        print(f"lr={r['lr']:<6} auc {r['auc']:.4f}")
    print(f"{len(results)} runs logged to {sacred_dir} in {elapsed:.1f}s")
    if args.json:
        args.json.write_text(
            json.dumps({"flow": "generate_sacred", "seconds": elapsed, "arms": results}, indent=2)
        )
    if elapsed > 30:
        print("FAILED: over the 30 s budget the README promises", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("SACRED_CAPTURE_MODE", "no")  # no cout.txt capture noise
    sys.exit(main())
