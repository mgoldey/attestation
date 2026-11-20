"""MLflow's file store, read directly and said as a core run record.

The mapping is the one `examples/mlflow-records/export_records.py` proved
against MLflow's own client (core spec, Producer findings 1): experiment ->
family, run id -> run_id, FINISHED/FAILED/KILLED -> complete/failed/
cancelled, params -> config verbatim, every logged point -> history, MLflow's
latest point -> metrics, git tags -> code (inferred), artifacts -> data
(sha256, role output), dataset inputs -> data (named, sha256 null). This
module reads `mlruns/` with no mlflow installed: `meta.yaml` with
`yaml.safe_load` (what MLflow's `read_yaml` calls), everything else as the
plain files the store is. Where MLflow's semantics matter they are mirrored
and the source is named; the one deliberate divergence is the dataset-step
bug (see `_point`).

The pure pieces both producers share -- status, tag names, timestamps,
MLflow's latest point, the grouping, the git-tag and `attestation.*`-tag
rules for `code` and `dependencies` -- are `attestation_schema.integrations.
mlflow_mapping`, called by the exporter too, so there is one mapping.

Specs: docs/superpowers/specs/2026-11-20-mlflow-adapter-design.md,
docs/superpowers/specs/2026-11-20-mlflow-plugin-design.md.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import attestation_schema as schema
import yaml
from attestation_schema.integrations import mlflow_mapping as mapping

from attestation.compute import WALL_METRIC
from attestation.ledger_adapters import timing
from attestation.ledger_adapters.found import IMPORT_NAMESPACE, Found, FoundMetric, history_ref

STORE_DIR = "mlruns"
ADAPTER = "mlflow"
# meta.yaml stores the status as RunStatus's protobuf integer; older or
# hand-written stores spell the name. MLflow's default when absent is RUNNING.
_STATUS_NAMES = {1: "RUNNING", 2: "SCHEDULED", 3: "FINISHED", 4: "FAILED", 5: "KILLED"}
TERMINAL = mapping.STATUS
GIT_COMMIT = mapping.GIT_COMMIT
GIT_REPO = mapping.GIT_REPO
RUN_NAME_TAG = mapping.RUN_NAME_TAG
CONTEXT_TAG = mapping.CONTEXT_TAG
# Not experiments: the trash holds deleted ones, `models` is the registry.
_NOT_EXPERIMENTS = {".trash", "models"}


@dataclass(frozen=True)
class LoggedPoint:
    """One line of a metric file, as MLflow logged it."""

    key: str
    value: float
    timestamp: int
    step: int
    dataset: str | None


@dataclass
class MlflowFound(Found):
    """A run as the legacy readers' `Found` (what `generic.discover` callers
    read: name, family, config, finite final values, wall time) plus what the
    store says that a `Found` has no field for. `error` is why MLflow's own
    client could not read the run; such a run is refused, never half-read."""

    run_id: str = ""
    run_status: str = "RUNNING"
    start_ms: int | None = None
    end_ms: int | None = None
    params: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    points: list[LoggedPoint] = field(default_factory=list)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[tuple[str, Path]] = field(default_factory=list)
    artifact_note: str | None = None
    wall: float | None = None
    error: str | None = None

    @property
    def series(self) -> list[schema.Series]:
        """Every logged point as the record's curves, the exporter's order."""
        return [s for _, s in _curves(self)]


# ---- reading the store --------------------------------------------------------


def _yaml(path: Path) -> dict | None:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def _files(folder: Path) -> dict[str, Path]:
    """`{key: file}` under one of a run's folders, recursively: a key with `/`
    is a nested file (`FileStore._get_resource_files`)."""
    if not folder.is_dir():
        return {}
    out = {}
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            out[path.relative_to(folder).as_posix()] = path
    return out


def _read_text(path: Path) -> str:
    """A param or tag value verbatim, as MLflow's `read_file` returns it."""
    return path.read_text(encoding="utf-8")


def _point(key: str, line: str) -> LoggedPoint:
    """One metric line, by `FileStore._get_metric_from_line`'s rules: split on
    single spaces into 2, 3 or 5 fields, anything else malformed.

    DIVERGES from MLflow in one place: MLflow reads the step only when there
    are exactly 3 fields, so a dataset-linked point (5 fields) comes back at
    step 0 whatever was logged (3.15.2; the file holds the right step). The
    file's step is read here -- reporting a value logged at step 5 as step 0
    is the bug, not the semantics (spec, "The dataset-step bug")."""
    parts = line.strip().split(" ")
    if len(parts) not in (2, 3, 5):
        raise ValueError(f"{len(parts)} field(s); MLflow expects 2, 3 or 5")
    step = int(parts[2]) if len(parts) >= 3 else 0
    return LoggedPoint(
        key=key,
        value=float(parts[1]),
        timestamp=int(parts[0]),
        step=step,
        dataset=parts[3] if len(parts) == 5 else None,
    )


def _points(key: str, path: Path) -> list[LoggedPoint]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        # MLflow: "Metric '<key>' is malformed. No data found."
        raise ValueError(f"metrics/{key} is empty")
    out = []
    for n, line in enumerate(lines, 1):
        try:
            out.append(_point(key, line))
        except ValueError as exc:
            raise ValueError(f"metrics/{key} line {n} is malformed: {exc}") from None
    return out


def _dataset_input(input_dir: Path, exp_dir: Path) -> dict[str, Any] | None:
    """One input's dataset (its own `meta.yaml`) plus the input's tags, or
    None for a model input or a dangling reference (MLflow skips it too)."""
    link = _yaml(input_dir / "meta.yaml") or {}
    if link.get("source_type") != "DATASET":
        return None
    dataset = _yaml(exp_dir / "datasets" / str(link.get("source_id")) / "meta.yaml")
    if dataset is None or not dataset.get("name"):
        return None
    tags = link.get("tags") or {}
    return {**dataset, "tags": {str(k): str(v) for k, v in tags.items()}}


def _datasets(run_dir: Path, exp_dir: Path) -> list[dict[str, Any]]:
    """The run's dataset inputs (`FileStore._get_dataset_inputs`; model
    inputs are not mapped)."""
    inputs = run_dir / "inputs"
    if not inputs.is_dir():
        return []
    found = (_dataset_input(d, exp_dir) for d in sorted(inputs.iterdir()) if d.is_dir())
    return [d for d in found if d is not None]


def _artifact_root(meta: dict, run_dir: Path) -> tuple[Path | None, str | None]:
    """Where the run's artifacts are, or why they cannot be read.

    MLflow's default location is `<exp>/<run>/artifacts` inside the store: when
    the URI ends there, the copy in the store being scanned is read, wherever
    the store now lives (a moved store's URI names another machine's path).
    Any other local path is read where it says; anything else is not guessed."""
    uri = str(meta.get("artifact_uri") or "")
    tail = f"{run_dir.parent.name}/{run_dir.name}/artifacts"
    if not uri or uri.rstrip("/").endswith(tail):
        return run_dir / "artifacts", None
    parts = urlsplit(uri)
    if parts.scheme in ("", "file"):
        path = Path(unquote(parts.path))
        if path.is_dir():
            return path, None
    return None, f"artifacts at {uri} were not read (not a local directory here)"


def _ms(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return None
    return ms or None


def _status(value: Any) -> str:
    if value is None:
        return "RUNNING"
    if isinstance(value, int) and not isinstance(value, bool):
        return _STATUS_NAMES.get(value, str(value))
    return str(value)


def _ended_meta(run_dir: Path) -> dict | None:
    """The run's meta.yaml, or None for a run the ledger does not read: one
    deleted in the MLflow UI (still on disk; resurrecting it in a ledger read
    for provenance is worse than missing it) or one in flight (a record is
    written once, when the run ends)."""
    meta = _yaml(run_dir / "meta.yaml")
    if meta is None or meta.get("lifecycle_stage", "active") != "active":
        return None
    return meta if _status(meta.get("status")) in TERMINAL else None


def _read_logged(found: MlflowFound, run_dir: Path, metric_files: dict[str, Path]) -> None:
    """Params and every metric point, or `found.error` saying why MLflow's own
    client could not read the run either."""
    try:
        found.params = {k: _read_text(p) for k, p in _files(run_dir / "params").items()}
        for key, path in sorted(metric_files.items()):
            found.points.extend(_points(key, path))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        found.error = f"MLflow cannot read this run either: {exc}"


def _legacy_metrics(found: MlflowFound) -> None:
    """The `Found` view `generic.discover` callers read: each series' final
    value when finite (never a NaN), plus wall time."""
    for (key, split), points in _grouped(found).items():
        final = _latest(points)
        if math.isfinite(final.value):
            found.metrics.append(FoundMetric(key, final.value, step=final.step, split=split))
    timing.add_wall(found.metrics, found.wall)


def read_run(
    root: Path, exp_dir: Path, run_dir: Path, family: str | None, *, measured_only: bool = True
) -> MlflowFound | None:
    """One run directory, or None for a run not ended (or deleted). With
    `measured_only` (the ledger's rule: every reader's), a run that logged no
    metric is None too; `attest mlflow sync` records it anyway."""
    meta = _ended_meta(run_dir)
    metric_files = _files(run_dir / "metrics") if meta is not None else {}
    if meta is None or (measured_only and not metric_files):
        return None  # not ended, or nothing measured
    tags = {k: _read_text(p) for k, p in _files(run_dir / "tags").items()}
    run_name = meta.get("run_name") or tags.get(RUN_NAME_TAG) or None
    found = MlflowFound(
        project=root.name,
        name=f"{run_name}/{run_dir.name[:8]}" if run_name else run_dir.name,
        source_path=str(run_dir),
        family=family,
        status="recorded",
        adapter=ADAPTER,
        run_id=run_dir.name,
        run_status=_status(meta.get("status")),
        start_ms=_ms(meta.get("start_time")),
        end_ms=_ms(meta.get("end_time")),
        tags=tags,
    )
    if str(meta.get("run_id", run_dir.name)) != run_dir.name:
        found.error = f"meta.yaml names run {meta.get('run_id')!r}, not its directory"
    else:
        _read_logged(found, run_dir, metric_files)
    if not found.error:
        _read_inputs_and_outputs(found, meta, run_dir, exp_dir)
    return found


def _read_inputs_and_outputs(found: MlflowFound, meta: dict, run_dir: Path, exp_dir: Path) -> None:
    """Config, dataset inputs, artifact paths, wall time and the legacy view."""
    found.config = found.params or None
    found.datasets = _datasets(run_dir, exp_dir)
    art_root, found.artifact_note = _artifact_root(meta, run_dir)
    if art_root is not None:
        found.artifacts = list(_files(art_root).items())
    found.wall = timing.epoch_ms_duration(found.start_ms, found.end_ms)
    _legacy_metrics(found)


def experiments(base: Path) -> list[tuple[Path, str | None]]:
    """(directory, name) of every active experiment in the store. The trash
    and the model registry are not experiments; a missing name is None."""
    out = []
    for exp_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if exp_dir.name in _NOT_EXPERIMENTS or exp_dir.name.startswith("."):
            continue
        experiment = _yaml(exp_dir / "meta.yaml") or {}
        if experiment.get("lifecycle_stage", "active") == "active":
            name = experiment.get("name")
            out.append((exp_dir, str(name) if name not in (None, "") else None))
    return out


def run_dirs(exp_dir: Path) -> list[Path]:
    """An experiment's run directories (its `datasets/` is not a run)."""
    return sorted(p for p in exp_dir.iterdir() if p.is_dir() and p.name != "datasets")


def mlflow_runs(root: Path, seen: set[str]) -> list[MlflowFound]:
    """Every ended, active run under `<root>/mlruns/<experiment_id>/<run_id>/`.

    Named `<run_name>/<run id[:8]>` (the id alone when the run has no name),
    the ledger's name for MLflow runs since the first reader; a name already
    in `seen` is skipped, as every legacy reader does. So is a run whose id
    is in `seen`: the core adapter, which runs first, claimed it for the
    run's own core record (`attest mlflow sync`'s), which outranks a
    conversion of the store -- one run, read once (plugin spec, Decision 3)."""
    base = Path(root) / STORE_DIR
    if not base.is_dir():
        return []
    out: list[MlflowFound] = []
    for exp_dir, family in experiments(base):
        for run_dir in run_dirs(exp_dir):
            if run_dir.name in seen:
                continue
            found = read_run(Path(root), exp_dir, run_dir, family)
            if found is not None and found.name not in seen:
                seen.add(found.name)
                out.append(found)
    return out


# ---- the mapping ---------------------------------------------------------------


timestamp_ms = mapping.timestamp_ms


def _grouped(found: MlflowFound) -> dict[tuple[str, str | None], list[LoggedPoint]]:
    """Points per (metric, linked dataset) in logging order (the shared rule)."""
    return mapping.group_points(found.points, lambda p: p.dataset)


def _latest(points: list[LoggedPoint]) -> LoggedPoint:
    """MLflow's own final value (the shared rule)."""
    return mapping.latest(points)


def _curves(found: MlflowFound) -> list[tuple[LoggedPoint, schema.Series]]:
    """(final point, curve) per series, sorted by (metric, split)."""
    grouped = _grouped(found)
    out = []
    for key, split in mapping.series_order(grouped):
        logged = grouped[(key, split)]
        points = [schema.Point(p.step, p.value, timestamp_ms(p.timestamp)) for p in logged]
        out.append((_latest(logged), schema.Series(key, split, {}, points)))
    return out


def _data(found: MlflowFound) -> list[dict]:
    """Dataset inputs (named, undigested, sorted by name), then artifacts
    (sha256 of the bytes, sorted by path) -- the exporter's order."""
    entries: list[dict[str, Any]] = []
    for ds in found.datasets:
        role = ds["tags"].get(CONTEXT_TAG) or "input"
        name = f"datasets/{ds['name']}@{ds.get('digest')}"
        entries.append({"name": name, "sha256": None, "role": role})
    entries.sort(key=lambda d: d["name"])
    for rel, path in sorted(found.artifacts):
        entries.append(
            {"name": f"artifacts/{rel}", "sha256": schema.sha256_file(path), "role": "output"}
        )
    return entries


def to_core_record(
    found: MlflowFound, *, caveats: list[str], replicate: tuple[str, int] | None
) -> schema.RunRecord:
    """The core record for one MLflow run, validated by `from_dict`.

    `replicate` is the ledger-wide `<config>_seed<N>` naming convention
    (`trackers.replicate_of`), the only arm the ledger knows without a
    declaration. Raises `RecordError` for a run MLflow itself cannot read."""
    if found.error:
        raise schema.RecordError("", f"{found.name}: {found.error}")
    curves = _curves(found)
    metrics = [
        {
            "name": s.name,
            "value": schema.spell_value(final.value),
            "direction": None,
            "step": final.step,
            "split": s.split,
            "unit": None,
            "labels": {},
        }
        for final, s in curves
    ]
    if found.wall is not None and all(m["name"] != WALL_METRIC for m in metrics):
        metrics.append(
            {
                "name": WALL_METRIC,
                "value": found.wall,
                "direction": None,
                "step": None,
                "split": None,
                "unit": timing.WALL_UNIT,
                "labels": {},
            }
        )
    from attestation import __version__

    identity = mapping.read_tags(found.tags)
    run_caveats = [f"{found.name}: {found.artifact_note}"] if found.artifact_note else []
    run_caveats += [f"{found.name}: {p}" for p in identity.problems]
    deps = identity.dependencies
    return schema.from_dict(
        {
            "schema": schema.SCHEMA,
            "run_id": found.run_id,
            "family": found.family,
            "arm": replicate[0] if replicate else None,
            "status": TERMINAL[found.run_status],
            "started_at": timestamp_ms(found.start_ms),
            "finished_at": timestamp_ms(found.end_ms),
            "producer": {"name": f"attestation {ADAPTER}-import", "version": __version__},
            "code": asdict(identity.code),
            "config": dict(found.params),
            "data": _data(found),
            "metrics": metrics,
            "history": history_ref([s for _, s in curves]),
            "seeds": {"seed": replicate[1]} if replicate else None,
            "dependencies": [asdict(d) for d in deps] if deps is not None else None,
            "ext": {
                IMPORT_NAMESPACE: {
                    "adapter": ADAPTER,
                    "status": found.status,
                    "notes": None,
                    "corpus": None,
                    "caveats": caveats,
                    "run_caveats": run_caveats,
                }
            },
        }
    )


__all__ = [
    "MlflowFound",
    "LoggedPoint",
    "experiments",
    "mlflow_runs",
    "read_run",
    "run_dirs",
    "timestamp_ms",
    "to_core_record",
]
