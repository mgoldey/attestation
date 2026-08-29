# examples/hydra/train.py
"""One arm of a Hydra multirun sweep: LogisticRegression at one learning rate.

Invoked by generate.sh as `python train.py --multirun lr=0.01,0.1,1,10
hydra.job.chdir=True`, once per value in the sweep. Trains on scikit-learn's
bundled breast-cancer set (569 rows, no download, no randomness beyond a
fixed random_state), the same surrogate this repo's other tracker examples
use (examples/wandb/generate.py, examples/sacred/generate.py, examples/dvc/
train.py, examples/tensorflow/generate.py).

`hydra.job.chdir=True` is required and not the default. hydra-core 1.3.5 no
longer changes the working directory per job unless asked -- Hydra 1.1 and
earlier always chdir'd, and 1.2 through at least 1.3.5 keep the setting but
default `hydra.job.chdir` to `null` (meaning: warn and behave as `False`)
for backward-compatibility with configs that read relative paths from the
launch directory. Without the override, every arm's `os.getcwd()` is the
directory generate.sh was run from, and all four arms silently overwrite
the same top-level metrics.json instead of writing into their own
`multirun/<date>/<time>/<n>/` -- found by running this for real (a bare
`--multirun lr=...` produced one metrics.json, not four), not by reading
Hydra's release notes first.

Writes metrics.json into os.getcwd() -- with hydra.job.chdir=True, that is
this arm's own numbered output directory, exactly like the sibling
examples write into their own arm's directory (Sacred's sacred_runs/<n>/,
DVC's foreach expansion).
"""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    lr = float(cfg.lr)

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
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    print(f"lr={lr:<6} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
