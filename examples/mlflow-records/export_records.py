# examples/mlflow-records/export_records.py
"""Write every finished run in an MLflow tracking store as an attestation.run/1 record.

A producer, not an adapter: it reads MLflow through MLflow's own client (so a
file store, a sqlite store and a tracking server all read the same way) and
writes ONLY through `attestation_schema` -- `layout.write_manifest` and
`write`, never hand-rolled JSON. It imports nothing from `attestation`, the
way smeltery does not: a producer depends on the schema, not on the observer.

The mapping, and why each line is what it is, is the "Producer findings"
section of docs/superpowers/specs/2026-11-19-core-run-record-design.md:

    experiment name          -> family
    (no MLflow concept)      -> arm: null, unless --arm-param names the params
                                that define one (a run name is NOT an arm: the
                                real sweep reuses one run name for four arms)
    run id                   -> run_id; run dir is <out>/run-<run_id>
    status                   -> FINISHED complete, FAILED failed, KILLED cancelled;
                                RUNNING/SCHEDULED and deleted runs get no record
    start/end time (ms)      -> started_at/finished_at via format_timestamp
    params (strings)         -> config, verbatim -- never re-typed
    every logged point       -> history, one series per (metric, linked dataset)
    MLflow's latest point    -> metrics (MLflow's key: step, timestamp, value);
                                direction null unless --direction declares it;
                                unit null (MLflow has none); labels {}
    linked dataset name      -> split (MLflow 3 log_metric(dataset=...))
    mlflow.source.git.commit -> code.git_sha, provenance "inferred" (HEAD at run start);
                                dirty null (MLflow never records it); version null
    mlflow.source.git.repoURL-> code.repo, credentials stripped
    artifacts                -> data, role "output", sha256 of the stored file
    dataset inputs           -> data, role = mlflow.data.context, sha256 null
                                (MLflow's digest is not a sha256 of any bytes
                                a reader can fetch; it stays in ext.mlflow)
    --seed-param NAME        -> seeds {NAME: int}; null when none is declared
    (no environment recorded)-> dependencies: mlflow, provenance "unknown"
    everything else          -> ext.mlflow (experiment id, run name, every tag --
                                the repo URL tag with its credentials stripped
                                too -- and every dataset input verbatim)

    uv run --group examples python export_records.py \\
        --tracking-uri ../flows/training/mlruns --out "$(mktemp -d)" \\
        --arm-param C --seed-param seed --direction auc=max
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from attestation_schema import (
    CodeIdentity,
    DataDigest,
    Dependency,
    Metric,
    Point,
    Producer,
    RunRecord,
    Series,
    format_timestamp,
    layout,
    read,
    write,
)

# mlflow-skinny 3.x refuses a `file:` tracking URI unless this is set (the
# file store is deprecated upstream, not removed). Set before mlflow reads it.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
# Hashing artifacts downloads each one; a progress bar per file is noise.
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

PRODUCER = Producer(name="attestation-examples/mlflow-records", version="0.1.0")
STATUS = {"FINISHED": "complete", "FAILED": "failed", "KILLED": "cancelled"}
GIT_COMMIT = "mlflow.source.git.commit"
GIT_REPO = "mlflow.source.git.repoURL"
CONTEXT_TAG = "mlflow.data.context"


@dataclass(frozen=True)
class Declared:
    """What the person running the export states; MLflow records none of it."""

    arm_params: tuple[str, ...] = ()
    seed_params: tuple[str, ...] = ()
    directions: dict[str, str] = field(default_factory=dict)


def timestamp_ms(ms: int | None) -> str | None:
    """MLflow's epoch milliseconds as the record's one spelling of that instant."""
    if ms is None:
        return None
    whole, frac = divmod(int(ms), 1000)
    return format_timestamp(datetime.fromtimestamp(whole, UTC).replace(microsecond=frac * 1000))


def strip_credentials(url: str | None) -> str | None:
    """A repo URL without `user:token@` -- the schema refuses one that has it."""
    if not url:
        return None
    parts = urlsplit(url)
    if parts.scheme in ("http", "https") and (parts.username or parts.password):
        host = parts.hostname or ""
        netloc = f"{host}:{parts.port}" if parts.port else host
        return urlunsplit(parts._replace(netloc=netloc))
    return url


def history(points: Iterable[Any]) -> dict[tuple[str, str | None], list[Any]]:
    """Every logged point, grouped by (metric, linked dataset) in logging order.

    Logging order is timestamp order; the sort is stable, so points MLflow
    logged in the same millisecond keep the order the store returned them.
    """
    series: dict[tuple[str, str | None], list[Any]] = {}
    for p in sorted(points, key=lambda p: p.timestamp):
        series.setdefault((p.key, p.dataset_name or None), []).append(p)
    return series


def latest(points: Sequence[Any]) -> Any:
    """MLflow's own 'latest' point, by MLflow's own key: greatest step, then
    latest timestamp, then largest value (file_store._get_metric_from_file).
    The value tiebreak is MLflow's, kept so `metrics` agrees with what MLflow
    reports for the run when two values share a step and a millisecond."""
    return max(points, key=lambda p: (p.step, p.timestamp, p.value))


def _arm(params: dict[str, str], names: Sequence[str], run_id: str) -> str | None:
    missing = [n for n in names if n not in params]
    if missing:
        raise ValueError(f"run {run_id}: --arm-param {missing} not among its params")
    return ",".join(f"{n}={params[n]}" for n in names) or None


def _seeds(params: dict[str, str], names: Sequence[str], run_id: str) -> dict[str, int] | None:
    if not names:
        return None
    seeds = {}
    for name in names:
        try:
            seeds[name] = int(params[name])
        except KeyError:
            raise ValueError(f"run {run_id}: --seed-param {name!r} is not a param") from None
        except ValueError:
            raise ValueError(
                f"run {run_id}: param {name}={params[name]!r} is not an integer seed"
            ) from None
    return seeds


def _code(tags: dict[str, str]) -> CodeIdentity:
    """MLflow's git tags: the HEAD of the script's checkout when the run
    started -- inferred, since MLflow never records uncommitted changes."""
    sha = tags.get(GIT_COMMIT) or None
    return CodeIdentity(
        git_sha=sha,
        dirty=None,
        repo=tags.get(GIT_REPO) or None,
        version=None,
        provenance="inferred" if sha else "unknown",
        evidence=f"MLflow tag {GIT_COMMIT}: HEAD of the source's checkout at run start; "
        "uncommitted changes are not recorded"
        if sha
        else None,
    )


def _artifacts(client: Any, run_id: str, path: str | None = None) -> list[str]:
    out: list[str] = []
    for info in client.list_artifacts(run_id, path):
        out += _artifacts(client, run_id, info.path) if info.is_dir else [info.path]
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _data(client: Any, run: Any) -> tuple[list[DataDigest], list[dict[str, Any]]]:
    """Dataset inputs (named, undigested) and artifacts (digested), plus the
    dataset inputs verbatim for ext.mlflow."""
    entries: list[DataDigest] = []
    datasets: list[dict[str, Any]] = []
    for inp in run.inputs.dataset_inputs if run.inputs else []:
        ds = inp.dataset
        tags = {t.key: t.value for t in inp.tags}
        datasets.append(
            {
                "name": ds.name,
                "digest": ds.digest,
                "source_type": ds.source_type,
                "source": ds.source,
                "schema": ds.schema,
                "profile": ds.profile,
                "tags": tags,
            }
        )
        role = tags.get(CONTEXT_TAG) or "input"
        entries.append(DataDigest(name=f"datasets/{ds.name}@{ds.digest}", sha256=None, role=role))
    entries.sort(key=lambda d: d.name)
    datasets.sort(key=lambda d: (d["name"], d["digest"]))
    paths = sorted(_artifacts(client, run.info.run_id))
    if paths:
        with tempfile.TemporaryDirectory() as tmp:
            for rel in paths:
                local = Path(client.download_artifacts(run.info.run_id, rel, tmp))
                entries.append(
                    DataDigest(name=f"artifacts/{rel}", sha256=_sha256(local), role="output")
                )
    return entries, datasets


def record_for(client: Any, run: Any, experiment: Any, declared: Declared) -> RunRecord | None:
    """The core record for one MLflow run, or None for a run that has not ended."""
    info = run.info
    status = STATUS.get(info.status)
    if status is None or info.lifecycle_stage != "active":
        return None
    params = dict(run.data.params)
    tags = dict(run.data.tags)
    if GIT_REPO in tags:  # MLflow stores the remote verbatim, credentials and all
        tags[GIT_REPO] = strip_credentials(tags[GIT_REPO]) or ""

    points = [
        p for key in sorted(run.data.metrics) for p in client.get_metric_history(info.run_id, key)
    ]
    curves = history(points)
    metrics, series = [], []
    for name, split in sorted(curves, key=lambda k: (k[0], k[1] or "")):
        logged = curves[(name, split)]
        final = latest(logged)
        metrics.append(
            Metric(
                name=name,
                value=final.value,
                direction=declared.directions.get(name),
                step=final.step,
                split=split,
                unit=None,
                labels={},
            )
        )
        series.append(
            Series(
                name=name,
                split=split,
                labels={},
                points=[Point(p.step, p.value, timestamp_ms(p.timestamp)) for p in logged],
            )
        )

    data, datasets = _data(client, run)
    return RunRecord(
        run_id=info.run_id,
        family=experiment.name,
        arm=_arm(params, declared.arm_params, info.run_id),
        status=status,
        started_at=timestamp_ms(info.start_time),
        finished_at=timestamp_ms(info.end_time),
        producer=PRODUCER,
        code=_code(tags),
        config=params,
        data=data,
        metrics=metrics,
        history=series,
        seeds=_seeds(params, declared.seed_params, info.run_id),
        dependencies=[
            Dependency(
                name="mlflow",
                version=None,
                build_sha=None,
                dirty=None,
                provenance="unknown",
                evidence="the run logged to MLflow; MLflow records no client version for a run",
                source=None,
            )
        ],
        ext={
            "mlflow": {
                "experiment_id": experiment.experiment_id,
                "run_name": info.run_name,
                "tags": tags,
                "datasets": datasets,
            }
        },
    )


def export(
    tracking_uri: str,
    out: Path,
    declared: Declared = Declared(),
    experiments: Sequence[str] = (),
) -> tuple[list[Path], list[str]]:
    """Write one run directory per ended MLflow run under `out`.

    Returns (run directories written, ids of runs skipped as not ended).
    Each directory gets `attestation/digests.json` and then
    `attestation/run.json` -- the layout's write order.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri)
    written: list[Path] = []
    skipped: list[str] = []
    for experiment in sorted(client.search_experiments(), key=lambda e: e.experiment_id):
        if experiments and experiment.name not in experiments:
            continue
        runs = client.search_runs([experiment.experiment_id], max_results=50000)
        for run in sorted(runs, key=lambda r: r.info.run_id):
            record = record_for(client, run, experiment, declared)
            if record is None:
                if run.info.lifecycle_stage == "active":
                    skipped.append(run.info.run_id)
                continue
            run_dir = out / f"run-{record.run_id}"
            manifest = layout.write_manifest(run_dir)
            write(_with_manifest(record, manifest), run_dir)
            written.append(run_dir)
    return written, skipped


def _with_manifest(record: RunRecord, manifest: DataDigest) -> RunRecord:
    from dataclasses import replace

    return replace(record, data=[*record.data, manifest])


def _directions(pairs: Sequence[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        name, _, direction = pair.partition("=")
        if direction not in ("min", "max") or not name:
            raise SystemExit(f"--direction takes NAME=min|max, got {pair!r}")
        out[name] = direction
    return out


def _uri(text: str) -> str:
    """A bare directory is a file store; anything with a scheme passes through."""
    return text if "://" in text or text.startswith("file:") else Path(text).resolve().as_uri()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--tracking-uri", required=True, help="an mlruns/ directory or any MLflow URI")
    ap.add_argument("--out", type=Path, required=True, help="where run-<run_id>/ directories go")
    ap.add_argument("--experiment", action="append", default=[], help="only these experiments")
    ap.add_argument("--arm-param", action="append", default=[], help="params that define the arm")
    ap.add_argument("--seed-param", action="append", default=[], help="params that are RNG seeds")
    ap.add_argument("--direction", action="append", default=[], help="METRIC=min|max, declared")
    args = ap.parse_args(argv)

    declared = Declared(
        arm_params=tuple(args.arm_param),
        seed_params=tuple(args.seed_param),
        directions=_directions(args.direction),
    )
    written, skipped = export(_uri(args.tracking_uri), args.out, declared, args.experiment)
    for run_dir in written:
        rec = read(run_dir)
        points = sum(len(s.points) for s in rec.history)
        print(
            f"{run_dir.name[:12]}  {rec.status:<9} family={rec.family} arm={rec.arm}"
            f"  {len(rec.metrics)} metrics, {len(rec.history)} series, {points} points,"
            f" {len(rec.data)} data"
        )
    for run_id in skipped:
        print(f"skipped {run_id}: not ended, so no record (a record is written once, at the end)")
    print(f"{len(written)} attestation.run/1 record(s) written and read back")
    return 0


if __name__ == "__main__":
    sys.exit(main())
