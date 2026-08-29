# examples/dvc/train.py
"""One arm of the `train` DVC stage: LogisticRegression at one learning rate.

Invoked by dvc.yaml's foreach stage as `python train.py ${item}`, once per
value in params.yaml's `lr` list. Trains on scikit-learn's bundled
breast-cancer set (569 rows, no download, no randomness beyond a fixed
random_state), the same surrogate this repo's other tracker examples use
(examples/wandb/generate.py, examples/sacred/generate.py,
examples/flows/training/train_mlflow.py), minus the synthetic curve those
need for their own step-logging APIs -- DVC metrics are a snapshot per
`dvc repro`, not a log, so there is nothing here to log incrementally.

Writes metrics/<lr>.json, where <lr> is the literal argv[1] string, not
`str(float(argv[1]))` -- dvc.yaml declares the output as
metrics/${item}.json, and DVC substitutes ${item} with params.yaml's list
entry exactly as written (`1`, not `1.0`). Using the parsed float here
produced metrics/1.0.json for the arm dvc.yaml calls train@1, and dvc repro
failed with "output 'metrics/1.json' does not exist" -- discovered by
running this stage for real, not by reading DVC's docs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main(argv: list[str]) -> int:
    lr_text = argv[0]
    lr = float(lr_text)

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    clf = LogisticRegression(C=1.0 / lr, max_iter=2000, random_state=0).fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    prob = clf.predict_proba(X_te)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "auc": roc_auc_score(y_te, prob),
    }

    out_dir = Path("metrics")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{lr_text}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"lr={lr_text:<6} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
