# examples/tensorflow/generate.py
"""Four Keras training runs -- one per learning rate -- read back by the
run ledger through nothing more exotic than CSVLogger's CSV and a plain
JSON dict, no TensorFlow-specific reader needed.

A two-dense-layer classifier on scikit-learn's bundled breast-cancer set
(569 rows, no download -- `tf.keras.datasets` reaches the internet, so this
uses `sklearn.datasets.load_breast_cancer` instead, the same surrogate
`examples/sacred/generate.py` and `examples/dvc/train.py` use), four arms
over `learning_rate in {1e-3, 3e-3, 1e-2, 3e-2}`, five epochs each, CPU
only. Run via:

    uv run --with tensorflow-cpu --with tensorboard --with scikit-learn \
        --no-project python generate.py

`tensorboard` is required alongside `tensorflow-cpu` -- without it,
`tf.keras.callbacks.TensorBoard` fails at the first `model.fit()` with
`TBNotInstalledError: TensorBoard is not installed, missing implementation
for tf.summary.scalar` (found by running this for real, not by reading
TensorFlow's docs). Installing both is ~500 MB and can take a few minutes
on a cold `uv` cache; the committed results/, tb/ and this script mean
nobody else needs to run it just to read this example.

Per arm, `tf.keras.callbacks.CSVLogger("results/lr_<lr>.csv")` logs the
epoch-by-epoch curve (loss/accuracy/val_loss/val_accuracy per epoch) -- the
way to make a Keras run legible to `ledger_adapters/generic.py` without a
TensorFlow-specific reader, since CSVLogger's output is exactly the
`results/*.csv` shape the generic adapter already expects. It has no column
naming which arm a row belongs to (there's only one arm's rows per file,
never a `config_name` column), so `_label_of` finds nothing and every row
is silently unlabelled -- the CSV is written for a human (or TensorBoard)
to read, not for the ledger. The final `accuracy`, `precision`, `recall`,
`auc` on the held-out split -- computed with `sklearn.metrics`, not
`tf.keras.metrics.AUC`/`Precision`/`Recall`, because Keras's streaming
metric objects report a running average across batches rather than a
single clean number computed once against `y_test`, and the two disagree
in the third decimal place -- are what `results/lr_<lr>.json` carries, and
that JSON dict of scalars is exactly the shape `metrics_from_payload`
already reads as one recorded run. `family_of("lr_0.001")` groups the four
arms into family `lr` (see the fix in ledger_adapters/generic.py this path
found and required, described in the README).

One TensorBoard run is also written, via `tf.keras.callbacks.TensorBoard
("tb/")`, for the lr=1e-2 arm only -- the ledger does not read event files
(binary protobuf, out of scope by the golden-paths spec) so one is enough
to show the convention exists; committing four would only be four times as
much binary for no additional coverage. The event file TensorBoard writes
embeds this machine's hostname and this process's pid in its own filename
(`events.out.tfevents.<timestamp>.<host>.<pid>.v2`) -- scrub() below renames
it to drop both, keeping the timestamp (a run identifier, not attribution).

Determinism: `tf.random.set_seed`, `np.random.seed`, `random.seed` are all
set, plus `TF_DETERMINISTIC_OPS=1` (must be set before TensorFlow is
imported) and `tf.config.set_visible_devices([], "GPU")` to force CPU. The
scikit-learn split uses a fixed `random_state`. On the same machine and the
same TensorFlow build this reproduces the same numbers; different hardware
or a different TensorFlow/XLA version is not guaranteed to reproduce them
bit-for-bit, which is why the committed results/ -- not a fresh run -- is
what the README's pinned line and run.sh depend on. Confirmed against
tensorflow-cpu 2.20.0.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TF's own startup banner

HERE = Path(__file__).resolve().parent
ARMS = (1e-3, 3e-3, 1e-2, 3e-2)
EPOCHS = 5
SEED = 0
TB_ARM = 1e-2  # the one arm that also gets a TensorBoard run


def _fmt(lr: float) -> str:
    return f"{lr:g}"


def scrub_tfevents(tb_dir: Path) -> list[Path]:
    """Rename every committed event file to drop hostname and pid.

    TensorBoard names each `events.out.tfevents.<ts>.<host>.<pid>[.<n>].v2`
    -- a `TensorBoard("tb/")` callback writes at least two of these, one
    under `tb/train/` and one under `tb/validation/`, and the validation
    writer's real name observed here (`...1739873.1.v2`) carries an extra
    numeric segment beyond host/pid. Keep only what identifies the run
    (`events`, `out`, `tfevents`, the timestamp) and the trailing format
    suffix (`v2`); drop every segment in between, whatever its count --
    that is this machine's hostname, this process's pid, and any writer
    index, none of which anything reads.
    """
    dests = []
    for src in sorted(tb_dir.rglob("events.out.tfevents.*")):
        parts = src.name.split(".")
        if len(parts) >= 5:
            ts, suffix = parts[3], parts[-1]
            dest = src.with_name(f"events.out.tfevents.{ts}.{suffix}")
        else:
            dest = src
        if dest != src:
            src.rename(dest)
        dests.append(dest)
    return dests


def train_one(lr: float, out_dir: Path, with_tensorboard: bool) -> dict:
    import numpy as np
    import tensorflow as tf
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    tf.config.set_visible_devices([], "GPU")
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_tr.shape[1],)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    tag = _fmt(lr)
    callbacks = [tf.keras.callbacks.CSVLogger(str(out_dir / "results" / f"lr_{tag}.csv"))]
    if with_tensorboard:
        callbacks.append(tf.keras.callbacks.TensorBoard(str(out_dir / "tb")))

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_te, y_te),
        epochs=EPOCHS,
        batch_size=32,
        verbose=0,
        callbacks=callbacks,
        shuffle=False,
    )

    prob = model.predict(X_te, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred)),
        "recall": float(recall_score(y_te, pred)),
        "auc": float(roc_auc_score(y_te, prob)),
    }
    (out_dir / "results" / f"lr_{tag}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return {"lr": lr, **metrics}


def main() -> int:
    out_dir = HERE
    results_dir = out_dir / "results"
    tb_dir = out_dir / "tb"
    shutil.rmtree(results_dir, ignore_errors=True)
    shutil.rmtree(tb_dir, ignore_errors=True)
    results_dir.mkdir(parents=True)

    t0 = time.perf_counter()
    arms = [train_one(lr, out_dir, with_tensorboard=(lr == TB_ARM)) for lr in ARMS]
    elapsed = time.perf_counter() - t0

    event_files = scrub_tfevents(tb_dir)

    print("tensorflow wrote:")
    for a in arms:
        tag = _fmt(a["lr"])
        print(f"  results/lr_{tag}.csv")
        print(f"  results/lr_{tag}.json")
        print(
            f"lr={tag:<6} accuracy={a['accuracy']:.4f} precision={a['precision']:.4f} "
            f"recall={a['recall']:.4f} auc={a['auc']:.4f}"
        )
    for event_file in event_files:
        print(f"  {event_file.relative_to(out_dir)}")
    print(f"{len(arms)} arm(s) trained in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
