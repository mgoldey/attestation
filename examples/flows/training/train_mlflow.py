# examples/flows/training/train_mlflow.py
"""Four training runs in one MLflow family, in well under thirty seconds.

LogisticRegression on scikit-learn's bundled breast-cancer set (569 rows,
no download), C in {0.01, 0.1, 1, 10}, one fixed stratified split. Each
arm logs its params and its held-out accuracy, precision, recall and AUC
to a local MLflow file store, plus a ten-step train_loss curve so the
ledger's "last line of each metric file" rule meets a real multi-line
file. run_name is the family name: that is how the ledger groups arms.
The curve comes from a surrogate SGDClassifier fit incrementally alongside
the LogisticRegression -- mlflow has no incremental API for the latter, and
the reported accuracy/precision/recall/auc are always the LogisticRegression's,
never the surrogate's.

The mlruns/ this writes is committed beside it: it is the first real MLflow
directory the ledger reader has read, and tests/test_examples.py pins it
without needing mlflow installed. Regenerating changes the run ids and is a
deliberate act -- running with no --out deletes and rewrites the committed
mlruns/ and FINDINGS.md in place. The committed fixture was produced
against mlflow-skinny 3.15.2, the version uv.lock currently resolves for
the `examples` dependency group (`pyproject.toml` pins `mlflow-skinny>=2.20`,
a range rather than an exact version -- unlike wandb/sacred/dvc/hydra-core,
there is no known regression across mlflow-skinny releases in the small
surface this reader touches, so this file does not refuse to run under a
different resolved version the way those generators do).

mlflow also writes personal attribution and machine-specific absolute paths
(tags/mlflow.user, the git remote URL, this machine's home directory in
every artifact_uri) that this repo does not commit -- the reader
(_mlflow_runs) reads only lifecycle_stage, run_name, metrics/*, params/*, so
scrubbing them costs the reader nothing. `scrub()` below removes them after
training.

    uv run --group examples python examples/flows/training/train_mlflow.py
    uv run attest runs scan --root examples/flows --project training
    uv run attest runs compare c_sweep --metric auc
    uv run attest claims examples/flows/training/FINDINGS.md      # exit 1: one is wrong
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

FAMILY = "c_sweep"
ARMS = (0.01, 0.1, 1.0, 10.0)
HERE = Path(__file__).resolve().parent

# Half of the last displayed decimal place in write_findings' `.4f` claim
# values -- wide enough to absorb the float/rounded-string gap, far tighter
# than the deliberately-wrong claim's 0.05 offset.
CLAIM_TOL = 0.00005

# mlflow-skinny 3.x refuses a `file:` tracking URI unless this is set --
# the local filesystem backend is in maintenance mode upstream, not removed.
# Set before `import mlflow` reads it, so the documented one-line command
# needs no extra environment variable.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# Tag files that carry who ran this and from where, not anything the reader
# uses. mlflow.runName (the family) and mlflow.source.type (LOCAL vs. a
# tracked job) are the only ones _mlflow_runs-adjacent tooling could ever
# want; the rest -- user, and every git.* identifying this machine's clone
# and its remote -- are deleted outright rather than genericised, since a
# placeholder invites someone to "restore" them from their own machine.
_SCRUB_TAGS = {
    "mlflow.user",
    "mlflow.source.name",
    "mlflow.source.git.branch",
    "mlflow.source.git.commit",
    "mlflow.source.git.repoURL",
}
_URI_FIELD_RE = re.compile(
    r"^(artifact_uri|artifact_location): file://(?P<path>/\S*)$", re.MULTILINE
)


def scrub(tracking_dir: Path) -> None:
    """Strip attribution and machine-specific paths mlflow writes by default.

    The reader (_mlflow_runs in ledger_adapters/generic.py) reads only
    lifecycle_stage, run_name, metrics/*, params/* from meta.yaml and the run
    directory -- never artifact_uri, never a tag other than run_name. Nothing
    here is load-bearing for the ledger; committing it anyway would put a
    real person's name, this machine's home directory, and this repo's git
    remote into version control, which is exactly the class of content this
    repo has previously rewritten history to remove.
    """
    tracking_dir = tracking_dir.resolve()
    for tags_dir in tracking_dir.glob("*/*/tags"):
        for name in _SCRUB_TAGS:
            (tags_dir / name).unlink(missing_ok=True)

    def _relativize(match: re.Match) -> str:
        # Whatever mlflow appended (a run's own "/artifacts", nothing for an
        # experiment's own directory) is preserved -- only the machine-specific
        # prefix up to tracking_dir is replaced with a path relative to it.
        field, absolute = match.group(1), Path(match.group("path"))
        rel = os.path.relpath(absolute, tracking_dir)
        return f"{field}: file:./mlruns/{rel}"

    for meta_path in tracking_dir.glob("**/meta.yaml"):
        text = meta_path.read_text()
        scrubbed = _URI_FIELD_RE.sub(_relativize, text)
        scrubbed = re.sub(r"^user_id: .*$", "user_id: flows", scrubbed, flags=re.MULTILINE)
        meta_path.write_text(scrubbed)


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
    """One claim per arm, plus one deliberately stale, under its own heading.

    Raises ValueError if the deliberately-wrong claim happens to land within
    CLAIM_TOL of any arm's real AUC -- under a different sklearn version or a
    different seed the four real values shift, and nothing else checks that
    the demo still yields exactly one contradiction. Committing a FINDINGS.md
    whose "deliberately wrong" claim turned out to be supported would silently
    break the exact thing this file exists to demonstrate; failing loudly here
    means that regeneration stops before anything is written, not after.
    """
    best = max(results, key=lambda r: r["auc"])
    stale = best["auc"] - 0.05
    for r in results:
        if abs(stale - r["auc"]) <= CLAIM_TOL:
            raise ValueError(
                f"the deliberately-wrong claim (auc={stale:.4f}) landed within"
                f" CLAIM_TOL={CLAIM_TOL} of arm C={r['C']}'s real auc={r['auc']:.4f}"
                " -- it would be SUPPORTED instead of CONTRADICTED. Regeneration"
                " produced values too close together for the demo to work;"
                " widen the offset in write_findings or pick a different seed."
            )

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
        # -- CLAIM_TOL is wider than any float rounding at that precision but
        # far tighter than the deliberately wrong claim's 0.05 offset below,
        # which the check above just confirmed holds.
        lines.append(
            f"- C={r['C']}: AUC {r['auc']:.4f}, precision {r['precision']:.4f},"
            f" recall {r['recall']:.4f}"
            f" <!-- claim: {project}/{name} metric=auc value={r['auc']:.4f} tol={CLAIM_TOL:.5f} -->"
        )
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
    ap.add_argument("--json", type=Path, help="also write a machine-readable summary here")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    tracking = args.out / "mlruns"
    if tracking.exists():
        shutil.rmtree(tracking)
    results = train(tracking)
    scrub(tracking)
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
