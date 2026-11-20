# examples/model-registry/generate.py
"""Regenerate this example's fixtures with the REAL libraries, then scrub them.

Two parts, each rebuilding only its own subtree, because they need different
environments:

    uv run --group examples --with pandas --with skops python generate.py mlflow
    <python with transformers, torch and attestation-schema[hf]> generate.py hf

`mlflow` (mlflow-skinny 3.15.2 from the `examples` group; its sklearn flavor
needs pandas and skops, which the group does not carry, hence `--with`):

    data/iris/1/            iris.csv (scikit-learn's bundled copy, byte for
                            byte) as an attestation.dataset/1 directory
    mlruns/                 a real MLflow file store: one run in experiment
                            `iris` that fits a LogisticRegression, logs its
                            test accuracy, its dataset input and the model
                            (mlflow.sklearn.log_model), then registers that
                            model as `iris-clf` version 1, alias `champion`
    mlflow-records/         `attest mlflow sync` over that store: the run's
                            core record, with the logged model's files as
                            `output` data by sha256

`hf` (transformers' Trainer with attestation_schema's AttestationCallback,
on CPU, no download -- the model is built from a config):

    hf/prep/                a native core run that wrote sequences.csv
    data/toy-seq/1/         that file as a dataset record naming prep as its
                            source run by record digest
    hf/train/               the Trainer's output_dir: checkpoint-<N>/ and the
                            callback's run record (checkpoint files as
                            `output` data, the dataset by content digest)
    hf/model/               trainer.save_model(): the saved model the claims
                            in DRAFT.md name

Nothing contacts a registry or the Hub. MLflow writes this machine's absolute
paths into every meta.yaml and MLmodel and the user into tags/mlflow.user;
`scrub_mlflow` removes them BEFORE the sync, so the record digests the bytes
that are committed. The HF part runs in a temporary directory outside any git
checkout with relative paths, so neither the pickled TrainingArguments nor
the record carries a machine path, and states the code identity itself
(this checkout's HEAD, no remote URL).

Regenerating changes run ids, model ids and timestamps; README.md's pinned
line does not depend on them.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRODUCER = "attestation model-registry example"

# Tag files naming who ran this and from which checkout (train_mlflow.py's
# list, plus the logged model's own copies).
_SCRUB_TAGS = {
    "mlflow.user",
    "mlflow.source.name",
    "mlflow.source.git.branch",
    "mlflow.source.git.commit",
    "mlflow.source.git.repoURL",
    "mlflow.gitRepoURL",
}
_URI_FIELD = re.compile(
    r"^(\s*(?:artifact_uri|artifact_location|storage_location|artifact_path|source|uri)):"
    r" (?:file://)?(?P<path>/\S+)$",
    re.MULTILINE,
)


def _schema():
    import attestation_schema as s

    return s


def _code():
    """This checkout's HEAD and dirty flag -- never its remote URL, which would
    put a hosting account into a committed fixture."""
    s = _schema()
    git = s.git_state(HERE)
    if git.sha is None:
        return s.CodeIdentity(None, None, None, None, "unknown", "not run from a git checkout")
    return s.CodeIdentity(
        git_sha=git.sha,
        dirty=git.dirty,
        repo=None,
        version=None,
        provenance="inferred",
        evidence="git HEAD of the checkout holding generate.py; remote URL left out on purpose",
    )


def _producer():
    s = _schema()
    return s.Producer(PRODUCER, s.__version__)


def write_dataset(root: Path, dataset_id: str, name: str, rows: int, source_runs) -> None:
    """`root/` (its data files already written) as an attestation.dataset/1 dir."""
    s = _schema()
    content = s.write_dataset_manifest(root)
    record = s.DatasetRecord(
        dataset_id=dataset_id,
        name=name,
        version="1",
        created_at=None,
        producer=_producer(),
        code=_code(),
        content_sha256=content,
        files=s.dataset_files(root),
        rows=rows,
        columns=None,
        splits=[],
        source_runs=source_runs,
        license=None,
        ext={},
    )
    s.write_dataset(record, root)


# ---- mlflow ---------------------------------------------------------------------


def scrub_mlflow(store: Path) -> None:
    """Drop attribution tags and turn every absolute path MLflow wrote into
    `file:./mlruns/<path inside the store>` (the reader maps that tail into
    whichever store it is reading)."""
    store = store.resolve()
    for tags in [*store.glob("*/*/tags"), *store.glob("*/models/*/tags")]:
        for name in _SCRUB_TAGS:
            (tags / name).unlink(missing_ok=True)

    def relative(match: re.Match) -> str:
        path = Path(match.group("path"))
        try:
            rel = path.relative_to(store)
        except ValueError:
            rel = Path(os.path.relpath(path, HERE))
            return f"{match.group(1)}: {rel.as_posix()}"
        return f"{match.group(1)}: file:./mlruns/{rel.as_posix()}"

    for meta in [*store.rglob("meta.yaml"), *store.rglob("MLmodel")]:
        text = meta.read_text()
        text = _URI_FIELD.sub(relative, text)
        text = re.sub(r"^user_id: .+$", "user_id: null", text, flags=re.MULTILINE)
        meta.write_text(text)


def mlflow_part() -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    # The run-context provider would tag this checkout's git remote; the
    # store keeps MLflow's own mapping only (as tests/test_mlflow_records.py).
    os.environ["ATTESTATION_MLFLOW_TAGS"] = "0"
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    import sklearn
    from mlflow.tracking import MlflowClient
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    for sub in ("mlruns", "mlflow-records", "data/iris"):
        shutil.rmtree(HERE / sub, ignore_errors=True)
    data_dir = HERE / "data" / "iris" / "1"
    data_dir.mkdir(parents=True)
    bundled = Path(sklearn.__file__).parent / "datasets" / "data" / "iris.csv"
    shutil.copyfile(bundled, data_dir / "iris.csv")
    # scikit-learn's copy has a count header (150,4,names...), not column names.
    table = pd.read_csv(data_dir / "iris.csv", skiprows=1, header=None)
    write_dataset(data_dir, "iris", "Iris (scikit-learn's bundled copy)", len(table), [])

    os.chdir(HERE)
    mlflow.set_tracking_uri(f"file:{HERE / 'mlruns'}")
    mlflow.set_experiment("iris")
    X, y = table.iloc[:, :4].to_numpy(), table.iloc[:, 4].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    with mlflow.start_run(run_name="logreg") as run:
        model = LogisticRegression(C=1.0, max_iter=500).fit(X_tr, y_tr)
        mlflow.log_params({"C": 1.0, "seed": 0, "test_size": 0.3})
        mlflow.log_metric("accuracy", float(model.score(X_te, y_te)))
        source = "data/iris/1/iris.csv"
        mlflow.log_input(mlflow.data.from_pandas(table, source=source, name="iris"), "training")
        info = mlflow.sklearn.log_model(model, name="model")
    version = mlflow.register_model(info.model_uri, "iris-clf")
    MlflowClient().set_registered_model_alias("iris-clf", "champion", version.version)
    scrub_mlflow(HERE / "mlruns")

    from attestation.cli import main

    rc = main(["mlflow", "sync", "--tracking-dir", "mlruns", "--out", "mlflow-records"])
    if rc != 0:
        sys.exit(f"attest mlflow sync exited {rc}")
    print(f"mlflow: run {run.info.run_id}, iris-clf version {version.version} (alias champion)")


# ---- hf ---------------------------------------------------------------------------

VOCAB = 16
SEQ = 8
ROWS = 64


def write_prep(work: Path) -> object:
    """A native core run that wrote sequences.csv: the dataset's source run."""
    import random

    s = _schema()
    run_dir = work / "hf" / "prep"
    run_dir.mkdir(parents=True)
    rng = random.Random(0)
    with (run_dir / "sequences.csv").open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow([f"t{i}" for i in range(SEQ)])
        for _ in range(ROWS):
            start = rng.randrange(VOCAB)
            out.writerow([(start + i) % VOCAB for i in range(SEQ)])  # count upwards, mod 16
    started = s.format_timestamp(_now())
    record = s.from_dict(
        {
            "schema": s.SCHEMA,
            "run_id": "prep-0",
            "family": "toy-seq-prep",
            "arm": None,
            "status": "complete",
            "started_at": started,
            "finished_at": s.format_timestamp(_now()),
            "producer": {"name": PRODUCER, "version": s.__version__},
            "code": asdict(_code()),
            "config": {"rows": ROWS, "seq": SEQ, "vocab": VOCAB, "seed": 0},
            "data": [
                {
                    "name": "sequences.csv",
                    "sha256": s.sha256_file(run_dir / "sequences.csv"),
                    "role": "output",
                }
            ],
            "metrics": [],
            "history": None,
            "seeds": {"seed": 0},
            "dependencies": None,
            "ext": {},
        }
    )
    s.write(record, run_dir)
    return record


def _now():
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc)


def hf_part() -> None:
    import torch
    from attestation_schema.integrations.huggingface import AttestationCallback
    from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

    s = _schema()
    work = Path(tempfile.mkdtemp(prefix="model-registry-"))
    prep = write_prep(work)
    data_dir = work / "data" / "toy-seq" / "1"
    data_dir.mkdir(parents=True)
    shutil.copyfile(work / "hf" / "prep" / "sequences.csv", data_dir / "sequences.csv")
    source = s.SourceRun(prep.run_id, prep.family, s.record_digest(prep))
    write_dataset(data_dir, "toy-seq", "count-up token sequences", ROWS, [source])

    with (data_dir / "sequences.csv").open() as fh:
        rows = [[int(v) for v in row] for row in list(csv.reader(fh))[1:]]
    ids = torch.tensor(rows)
    train = [{"input_ids": r, "labels": r} for r in ids]

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=VOCAB, n_positions=SEQ, n_embd=16, n_layer=1, n_head=2, bos_token_id=0
    )
    model = GPT2LMHeadModel(config)
    os.chdir(work)  # relative output_dir: the pickled TrainingArguments name no machine path
    args = TrainingArguments(
        output_dir="hf/train",
        num_train_epochs=4,
        per_device_train_batch_size=16,
        learning_rate=5e-3,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=4,
        report_to="none",
        seed=0,
        use_cpu=True,
        disable_tqdm=True,
    )
    callback = AttestationCallback(
        family="toy-seq-gpt2", datasets=[data_dir], code=_code(), run_id="toy-seq-gpt2-0"
    )
    trainer = Trainer(model=model, args=args, train_dataset=train, callbacks=[callback])
    trainer.train()
    trainer.save_model("hf/model")

    for sub in ("hf", "data/toy-seq"):
        shutil.rmtree(HERE / sub, ignore_errors=True)
    shutil.copytree(work / "hf", HERE / "hf", symlinks=True)
    shutil.copytree(data_dir, HERE / "data" / "toy-seq" / "1")
    shutil.rmtree(work)
    print(f"hf: record {callback.record_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("part", choices=("mlflow", "hf"))
    args = parser.parse_args(argv)
    mlflow_part() if args.part == "mlflow" else hf_part()
    leaks = subprocess.run(
        ["grep", "-rIl", *(f"-e/{d}/" for d in ("home", "tmp", "root")), str(HERE)],
        capture_output=True,
        text=True,
    ).stdout.split()
    leaks = [p for p in leaks if not p.endswith(".py") and not p.endswith(".md")]
    if leaks:
        print("machine paths left in:\n  " + "\n  ".join(leaks), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
