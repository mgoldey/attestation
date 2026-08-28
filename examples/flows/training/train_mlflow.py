# examples/flows/training/train_mlflow.py
"""Four training runs in one MLflow family, in well under thirty seconds.

LogisticRegression on scikit-learn's bundled breast-cancer set (569 rows,
no download), C in {0.01, 0.1, 1, 10}, one fixed stratified split. Each
arm logs its params and its held-out accuracy, precision, recall and AUC
to a local MLflow file store, plus a ten-step train_loss curve so the
ledger's "last line of each metric file" rule meets a real multi-line
file. run_name is the family name: that is how the ledger groups arms.

The mlruns/ this writes is committed beside it: it is the first real MLflow
directory the ledger reader has read, and tests/test_examples.py pins it
without needing mlflow installed. Regenerating changes the run ids and is
a deliberate act.

    uv run --group examples python examples/flows/training/train_mlflow.py
    uv run attest runs scan --root examples/flows --project training
    uv run attest runs compare c_sweep --metric auc
    uv run attest claims examples/flows/training/FINDINGS.md      # exit 1: one is wrong
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

FAMILY = "c_sweep"
ARMS = (0.01, 0.1, 1.0, 10.0)
HERE = Path(__file__).resolve().parent

# mlflow-skinny 3.x refuses a `file:` tracking URI unless this is set --
# the local filesystem backend is in maintenance mode upstream, not removed.
# Set before `import mlflow` reads it, so the documented one-line command
# needs no extra environment variable.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def train(tracking_dir: Path, seed: int = 0) -> list[dict]:
    import mlflow
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    mlflow.set_tracking_uri(f"file:{tracking_dir}")
    mlflow.set_experiment("flows")
    results = []
    for C in ARMS:
        with mlflow.start_run(run_name=FAMILY) as run:
            mlflow.log_params({"C": C, "seed": seed, "dataset": "sklearn:breast_cancer"})
            # A short curve, so the ledger reads a genuine multi-line metric file.
            sgd = SGDClassifier(loss="log_loss", alpha=1.0 / (C * len(X_tr)), random_state=seed)
            classes = np.unique(y_tr)
            for step in range(10):
                sgd.partial_fit(X_tr, y_tr, classes=classes)
                p = np.clip(sgd.predict_proba(X_tr)[:, 1], 1e-6, 1 - 1e-6)
                loss = float(-np.mean(y_tr * np.log(p) + (1 - y_tr) * np.log(1 - p)))
                mlflow.log_metric("train_loss", loss, step=step)
            clf = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            pred = (prob >= 0.5).astype(int)
            metrics = {
                "accuracy": float(accuracy_score(y_te, pred)),
                "precision": float(precision_score(y_te, pred)),
                "recall": float(recall_score(y_te, pred)),
                "auc": float(roc_auc_score(y_te, prob)),
            }
            mlflow.log_metrics(metrics)
            results.append({"C": C, "run_id": run.info.run_id, **metrics})
    return results


def write_findings(path: Path, results: list[dict], project: str) -> None:
    """One claim per arm, plus one deliberately stale, under its own heading."""
    best = max(results, key=lambda r: r["auc"])
    lines = [
        "# c_sweep findings",
        "",
        f"Four arms of `LogisticRegression` on scikit-learn's breast-cancer set,"
        f" C in {list(ARMS)}.",
        f"The best held-out AUC was C={best['C']} at {best['auc']:.4f}.",
        "",
    ]
    for r in results:
        name = f"{FAMILY}/{r['run_id'][:8]}"
        # value= is the AUC rounded to 4 places for the prose; claims.DEFAULT_TOL
        # is effectively exact (1e-9), so a rounded number needs tol= to say so
        # -- half of the last displayed decimal place, wider than any float
        # rounding at that precision but far tighter than the deliberately
        # wrong claim's 0.05 offset below.
        lines.append(
            f"- C={r['C']}: AUC {r['auc']:.4f}, precision {r['precision']:.4f},"
            f" recall {r['recall']:.4f}"
            f" <!-- claim: {project}/{name} metric=auc value={r['auc']:.4f} tol=0.00005 -->"
        )
    stale = best["auc"] - 0.05
    lines += [
        "",
        "### Deliberately wrong claim, for the demo",
        "",
        f"- The best arm reached AUC {stale:.4f}"
        f" <!-- claim: {project}/{FAMILY}/{best['run_id'][:8]}"
        f" metric=auc value={stale:.4f} -->",
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--out", type=Path, default=HERE, help="directory to hold mlruns/ and FINDINGS.md"
    )
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    tracking = args.out / "mlruns"
    if tracking.exists():
        shutil.rmtree(tracking)
    results = train(tracking)
    write_findings(args.out / "FINDINGS.md", results, project=args.out.name)
    elapsed = time.perf_counter() - t0

    for r in results:
        print(
            f"C={r['C']:<6} accuracy {r['accuracy']:.4f}  precision {r['precision']:.4f}"
            f"  recall {r['recall']:.4f}  auc {r['auc']:.4f}"
        )
    print(f"{len(results)} runs logged to {tracking} in {elapsed:.1f}s")
    if args.json:
        args.json.write_text(
            json.dumps({"flow": "train_mlflow", "seconds": elapsed, "arms": results}, indent=2)
        )
    if elapsed > 30:
        print("FAILED: over the 30 s budget the README promises", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
