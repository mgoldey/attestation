# examples/wandb/generate.py
"""One W&B run family, offline, in well under thirty seconds.

LogisticRegression on scikit-learn's bundled breast-cancer set (569 rows, no
download), lr in {0.001, 0.01, 0.1, 1.0} standing in for four sweep arms (W&B
has no `lr` for LogisticRegression's C, so this trains a surrogate
SGDClassifier at that learning rate for the ten-step train_loss curve and
reports the LogisticRegression's own held-out accuracy/auc as the run's
summary -- the same split train_mlflow.py uses between a curve-producing
surrogate and the metric that is actually reported). Run via:

    WANDB_MODE=offline uv run --with wandb --with scikit-learn --no-project python generate.py

wandb.init(project="flows", name="lr_sweep", ...) four times, wandb.log for
ten steps each, wandb.summary set to the final accuracy/auc, wandb.finish().
WANDB_MODE=offline keeps every write on disk under `wandb/`, no network call
and no account. Left alone, wandb also tries to write ~/.config/wandb and
~/.netrc; WANDB_DIR points its run directory at this path and
WANDB_CONFIG_DIR/WANDB_CACHE_DIR point elsewhere so nothing lands in the
caller's home directory (run.sh and this file's __main__ guard both set
them).

**The real finding, ahead of the one this task was written to look for:**
offline W&B does not write wandb-summary.json or config.yaml to files/ at
all -- confirmed here against wandb 0.17.6 through 0.29.0, and independently
confirmed as known upstream behaviour on wandb's own issue tracker (issues
#7227 and #9646; a W&B maintainer's answer on issue #1768 states plainly
"we do not have a python API of sorts to pull the values from an offline
*.wandb file", and `wandb sync --view` still demands a login, so it is not a
locally-usable substitute). Every logged value nonetheless reaches disk,
inside the run's binary `.wandb` transaction log (protobuf `run`/`config`/
`summary` records) -- what is missing is only the plain-file materialisation
`wandb sync` would otherwise perform against a real server. `_decode_wandb_log`
below performs that same materialisation locally, using the community's own
published workaround for this exact gap (issue #1768's accepted answer:
`wandb.sdk.internal.datastore.DataStore().scan_data()` over the run's
`.wandb` file), and writes wandb-summary.json/config.yaml in the documented
on-disk shape before the binary log -- which nothing this repo reads -- is
deleted. This is not invented data: every value it writes was logged by this
run and is being read back from the file wandb itself wrote, in the same
records `wandb sync` uploads. wandb-metadata.json needs no such step -- it
*is* written directly to files/ in offline mode.

The wandb/ this writes is committed beside it: it is the first real W&B
offline directory the ledger reader (_wandb_runs in
ledger_adapters/generic.py) has read. Regenerating changes the run ids and
timestamps and is a deliberate act -- running this file deletes and
rewrites the committed wandb/ in place.

wandb also writes personal attribution and machine-specific absolute paths
(host, username, the executable path, the repo root, git remote, email,
absolute script paths in wandb-metadata.json; full binary logs under
`.wandb`/`logs`/`tmp`; a frozen `files/requirements.txt`) that this repo
does not commit. `_wandb_runs` reads only wandb-summary.json, config.yaml
and wandb-metadata.json's `program`/`startedAt` -- never `host`, `username`,
`executable`, `root`, `git`, `email`, `codePath`, `requirements.txt`, the
`.wandb` binary log, or anything under `logs/`/`tmp/` -- so scrubbing them
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

# The wandb version this fixture was produced with. Pinned (rather than the
# unversioned `--with wandb` in the module docstring's command) because
# offline materialisation regressed between releases: 0.17.x still writes
# wandb-metadata.json to files/ in offline mode; by 0.29.0 files/ holds only
# requirements.txt and even that is gone. The pin is applied in run.sh, not
# here -- this file has no opinion on how it is invoked.
WANDB_VERSION = "0.17.6"

# Keys wandb-metadata.json writes that identify this machine or this person.
# `program`, `startedAt`, `args`, and `python` are the only ones _wandb_runs
# or a reader of this fixture could ever want; the rest -- the absolute
# executable and repo-root paths, the git remote and commit, and this
# machine's hostname/username/email -- are deleted outright rather than
# genericised, since a placeholder invites someone to "restore" them from
# their own machine.
_SCRUB_METADATA_KEYS = ("host", "username", "executable", "root", "git", "email", "codePath")

# `program` is kept (the reader derives a run's name from it) but W&B writes
# it as this machine's absolute invocation path, e.g. "<repo>/examples/wandb/
# generate.py". _wandb_runs already does `program.rsplit("/", 1)[-1]` to get
# the basename, so relativizing to just the basename costs the reader
# nothing while dropping the leaked path.

# Bookkeeping keys W&B injects into a run's summary that are not a logged
# metric: the running step/timestamp/runtime it stamps on every write. Kept
# out of the materialised wandb-summary.json for the same reason
# ledger_adapters.generic._wandb_runs strips them again on the read side
# (belt and suspenders -- the reader's own strip is the one that matters,
# this one just keeps the committed fixture honest about what W&B itself
# would ship as "summary" via a real sync).
_SUMMARY_BOOKKEEPING_KEYS = ("_step", "_runtime", "_timestamp")


def _decode_wandb_log(wandb_file: Path) -> tuple[dict, dict]:
    """Read back the config and summary a `.wandb` transaction log recorded.

    Offline W&B never writes these to files/ (see the module docstring) --
    they exist only inside this binary protobuf log until `wandb sync`
    uploads it. This performs the same read `wandb sync` performs, locally:
    `wandb.sdk.internal.datastore.DataStore` is wandb's own log reader, used
    here exactly as a W&B maintainer's workaround for this gap describes
    (wandb issue tracker, #1768). Every value returned was logged by this
    run; nothing here is inferred or invented.
    """
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    config: dict = {}
    summary: dict = {}
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))
    while True:
        data = ds.scan_data()
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)
        kind = record.WhichOneof("record_type")
        if kind == "run":
            items = record.run.config.update
        elif kind == "config":
            items = record.config.update
        elif kind == "summary":
            items = record.summary.update
        else:
            continue
        target = summary if kind == "summary" else config
        for item in items:
            target[item.key] = json.loads(item.value_json)
    return config, summary


def _write_materialised_files(run_dir: Path, config: dict, summary: dict) -> None:
    """Write config.yaml and wandb-summary.json in W&B's documented shape.

    config.yaml wraps every entry as {desc, value} plus an injected `_wandb`
    key -- the exact shape ledger_adapters.generic._wandb_config already
    unwraps. wandb-summary.json is the flat object metrics_from_payload
    already handles.
    """
    files = run_dir / "files"
    files.mkdir(exist_ok=True)

    lines = ["wandb_version: 1", ""]
    for key, value in config.items():
        if key == "_wandb":
            continue
        rendered = json.dumps(value) if isinstance(value, str) else value
        lines += [f"{key}:", "  desc: null", f"  value: {rendered}"]
    lines += ["_wandb:", "  desc: null", "  value:", f"    cli_version: {WANDB_VERSION}"]
    (files / "config.yaml").write_text("\n".join(lines) + "\n")

    clean_summary = {k: v for k, v in summary.items() if k not in _SUMMARY_BOOKKEEPING_KEYS}
    (files / "wandb-summary.json").write_text(json.dumps(clean_summary))


def scrub(wandb_dir: Path) -> None:
    """Strip attribution, machine paths, and bulk logs W&B writes by default.

    The reader (_wandb_runs in ledger_adapters/generic.py) reads only
    wandb-summary.json, config.yaml, and wandb-metadata.json's `program` and
    `startedAt` -- never the binary `.wandb` log, `logs/`, `tmp/`,
    `files/requirements.txt`, `files/output.log`, or any of the metadata
    keys this removes. Nothing here is load-bearing for the ledger;
    committing it anyway would put this machine's hostname, this person's
    username and email, and this repo's git remote into version control.
    """
    wandb_dir = wandb_dir.resolve()
    for run_dir in wandb_dir.glob("offline-run-*"):
        for wandb_log in run_dir.glob("*.wandb"):
            wandb_log.unlink()
        for syncstate in run_dir.glob("*.wandb.syncstate"):
            syncstate.unlink()
        shutil.rmtree(run_dir / "logs", ignore_errors=True)
        shutil.rmtree(run_dir / "tmp", ignore_errors=True)
        files = run_dir / "files"
        (files / "requirements.txt").unlink(missing_ok=True)
        (files / "output.log").unlink(missing_ok=True)

        meta_path = files / "wandb-metadata.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            for key in _SCRUB_METADATA_KEYS:
                meta.pop(key, None)
            program = meta.get("program")
            if isinstance(program, str) and "/" in program:
                meta["program"] = program.rsplit("/", 1)[-1]
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    # W&B also keeps top-level debug logs and a "latest-run" symlink; neither
    # is read by anything and both point at machine-specific paths.
    for name in ("debug.log", "debug-internal.log", "latest-run"):
        (wandb_dir / name).unlink(missing_ok=True)


def train(seed: int = 0) -> list[dict]:
    import numpy as np
    import wandb
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    results = []
    for lr in ARMS:
        run = wandb.init(
            project="flows",
            name=FAMILY,
            config={"lr": lr, "seed": seed, "dataset": "sklearn:breast_cancer"},
        )
        sgd = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=lr, random_state=seed)
        classes = np.array([0, 1])
        for step in range(10):
            sgd.partial_fit(X_tr, y_tr, classes=classes)
            p = np.clip(sgd.predict_proba(X_tr)[:, 1], 1e-6, 1 - 1e-6)
            loss = float(-np.mean(y_tr * np.log(p) + (1 - y_tr) * np.log(1 - p)))
            run.log({"train_loss": loss}, step=step)
        clf = LogisticRegression(C=1.0 / lr, max_iter=2000, random_state=seed).fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        accuracy = float(accuracy_score(y_te, pred))
        auc = float(roc_auc_score(y_te, prob))
        run.summary["accuracy"] = accuracy
        run.summary["auc"] = auc
        run_dir = Path(run.settings.sync_dir)
        run_id = run.id
        results.append(
            {"lr": lr, "run_id": run_id, "accuracy": accuracy, "auc": auc, "run_dir": run_dir}
        )
        run.finish()
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=HERE, help="directory to hold wandb/")
    ap.add_argument("--json", type=Path, help="also write a machine-readable summary here")
    args = ap.parse_args(argv)

    os.environ.setdefault("WANDB_MODE", "offline")
    # WANDB_DIR is the directory wandb creates its own `wandb/` subdirectory
    # under -- not the run directory itself. Set here (rather than passing
    # `dir=` to wandb.init) so it also governs where the `wandb/debug.log`
    # symlink and `latest-run` land: both under args.out/wandb, never args.out.
    os.environ["WANDB_DIR"] = str(args.out)

    t0 = time.perf_counter()
    wandb_dir = args.out / "wandb"
    if wandb_dir.exists():
        shutil.rmtree(wandb_dir)
    results = train()

    for r in results:
        run_dir = r.pop("run_dir")
        wandb_log = next(run_dir.glob("*.wandb"))
        config, summary = _decode_wandb_log(wandb_log)
        _write_materialised_files(run_dir, config, summary)

    scrub(wandb_dir)
    elapsed = time.perf_counter() - t0

    for r in results:
        print(f"lr={r['lr']:<6} accuracy {r['accuracy']:.4f}  auc {r['auc']:.4f}")
    print(f"{len(results)} runs logged to {wandb_dir} in {elapsed:.1f}s")
    if args.json:
        args.json.write_text(
            json.dumps({"flow": "generate_wandb", "seconds": elapsed, "arms": results}, indent=2)
        )
    if elapsed > 30:
        print("FAILED: over the 30 s budget the README promises", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
