"""W&B run directories, read directly and said as a core run record.

MLflow's adapter is the template (`mlflow_store`): the whole run maps to a
core record -- project -> family, the log's run id -> run_id, exit code ->
status, config, summary -> metrics, every logged row -> history (a reference
computed by the schema's own writer), git -> code (inferred), `files/` the
user saved -> data (sha256, role output), `_runtime` -> wall_time_s. The
source is the run's `.wandb` log (`wandb_log`, decoded with no wandb
installed), because an offline run writes almost nothing else; a directory
with no log (synced, or materialised by a script) is read from `files/` and
says what that cannot tell.

Spec: docs/superpowers/specs/2026-11-20-wandb-hydra-adapters-design.md.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import attestation_schema as schema
import yaml

from attestation.compute import WALL_METRIC
from attestation.ledger_adapters import timing, wandb_log
from attestation.ledger_adapters.found import (
    IMPORT_NAMESPACE,
    Found,
    FoundMetric,
    history_ref,
    json_safe,
)

BASE_DIR = "wandb"
ADAPTER = "wandb"
LOG_GLOB = "*.wandb"
# W&B's own files in files/, never the user's: not mapped to `data`.
OWN_FILES = {
    "wandb-summary.json",
    "wandb-metadata.json",
    "config.yaml",
    "requirements.txt",
    "output.log",
    "conda-environment.yaml",
    "wandb-history.jsonl",
    "wandb-events.jsonl",
}
OWN_DIRS = {"code"}
GIT_EVIDENCE = (
    "W&B run record git: HEAD of the source's checkout at wandb.init; uncommitted"
    " changes are not in the record (W&B saves them as diff.patch, not read)"
)


@dataclass
class WandbFound(Found):
    """A run as the legacy readers' `Found` (name, family, config, finite
    final values, wall time) plus what W&B recorded that a `Found` has no
    field for. `error` is why the log cannot be read; such a run is refused,
    never half-read. `log` is None for a directory with no `.wandb` log."""

    run_id: str = ""
    log: wandb_log.WandbLog | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    raw_config: dict[str, Any] = field(default_factory=dict)
    start: datetime | None = None
    git_commit: str | None = None
    git_remote: str | None = None
    outputs: list[tuple[str, Path]] = field(default_factory=list)
    wall: float | None = None
    error: str | None = None

    @property
    def series(self) -> list[schema.Series]:
        """Every logged row as the record's curves (empty without a log)."""
        return _series(self.log.rows) if self.log is not None else []


# ---- values ---------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def numeric_leaves(node: dict, prefix: str = "") -> list[tuple[str, int | float]]:
    """`(dotted name, value)` of every number in a summary or history row,
    sorted by name. `_`-prefixed keys are W&B's bookkeeping (`_step`,
    `_runtime`, `_timestamp`, `_wandb`) and a dict with `_type` is a media
    object (an image, a table): neither is a measurement."""
    out: list[tuple[str, int | float]] = []
    for key, value in node.items():
        if str(key).startswith("_"):
            continue
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            if "_type" not in value:
                out.extend(numeric_leaves(value, name))
        elif _is_number(value):
            out.append((name, value))
    return sorted(out, key=lambda kv: kv[0])


def _when(epoch: Any) -> str | None:
    """W&B's `_timestamp` (epoch seconds, a float) in the record's spelling."""
    if not _is_number(epoch) or not math.isfinite(epoch):
        return None
    return schema.format_timestamp(datetime.fromtimestamp(epoch, UTC))


def _row_step(row: wandb_log.HistoryRow, index: int) -> int:
    step = row.items.get(("_step",))
    if isinstance(step, int) and not isinstance(step, bool) and step >= 0:
        return step
    return row.step if row.step is not None and row.step >= 0 else index


def _series(rows: list[wandb_log.HistoryRow]) -> list[schema.Series]:
    """One series per numeric key, points in log order, sorted by name."""
    points: dict[str, list[schema.Point]] = {}
    for index, row in enumerate(rows):
        nested: dict = {}
        for path, value in row.items.items():
            wandb_log.set_path(nested, path, value)
        at = _when(nested.get("_timestamp"))
        step = _row_step(row, index)
        for name, value in numeric_leaves(nested):
            points.setdefault(name, []).append(schema.Point(step, value, at))
    return [schema.Series(name, None, {}, points[name]) for name in sorted(points)]


# ---- reading the run directory ---------------------------------------------------


def _json_file(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def files_config(path: Path) -> dict[str, Any]:
    """`files/config.yaml` with W&B's `{desc, value}` wrapper unwrapped and
    its bookkeeping (`wandb_version`, `_`-keys) dropped."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for key, entry in data.items():
        if str(key).startswith("_") or key == "wandb_version":
            continue
        out[str(key)] = entry.get("value") if isinstance(entry, dict) else entry
    return out


def _outputs(files: Path) -> list[tuple[str, Path]]:
    """Files the user saved into the run (`wandb.save`, logged media):
    everything under files/ but W&B's own files and saved code."""
    if not files.is_dir():
        return []
    out = []
    for dirpath, dirnames, filenames in os.walk(files):
        dirnames[:] = sorted(d for d in dirnames if Path(dirpath) != files or d not in OWN_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = path.relative_to(files).as_posix()
            if rel not in OWN_FILES and not name.endswith(".patch"):
                out.append((rel, path))
    return sorted(out)


def _from_log(found: WandbFound, log_path: Path) -> None:
    """Fill the run from its log, or set `found.error` for a damaged one."""
    try:
        log = wandb_log.read_log(log_path)
    except (OSError, wandb_log.LogError) as exc:
        found.error = f"{log_path.name} cannot be read: {exc}"
        return
    found.log = log
    found.run_id = log.run_id or found.run_id
    found.family = log.project
    found.summary = log.summary
    found.raw_config = {k: v for k, v in log.config.items() if not str(k).startswith("_")}
    found.start = log.started
    found.git_commit, found.git_remote = log.git_commit, log.git_remote


def _from_files(found: WandbFound, files: Path, meta: dict) -> None:
    """A directory with no log: what files/ holds (a sync's materialisation)."""
    found.summary = _json_file(files / "wandb-summary.json") or {}
    found.raw_config = files_config(files / "config.yaml")
    found.start = _parse_naive_utc(meta.get("startedAt"))


def _parse_naive_utc(text: Any) -> datetime | None:
    """W&B's `startedAt`: ISO 8601, written in UTC without a zone."""
    try:
        when = datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None
    return when if when.tzinfo else when.replace(tzinfo=UTC)


def _metadata_git(found: WandbFound, meta: dict) -> None:
    """A synced directory's metadata names git when the log does not."""
    git = meta.get("git")
    if isinstance(git, dict):
        found.git_commit = found.git_commit or (str(git.get("commit") or "") or None)
        found.git_remote = found.git_remote or (str(git.get("remote") or "") or None)


def _legacy_view(found: WandbFound) -> None:
    """The `Found` view `generic.discover` callers read: finite summary
    values and wall time; config as the record's."""
    found.config = json_safe(found.raw_config) or None
    found.metrics = [FoundMetric(n, float(v)) for n, v in numeric_leaves(found.summary)]
    found.metrics = [m for m in found.metrics if math.isfinite(m.value)]
    found.wall = timing.seconds(found.summary.get("_runtime"))
    timing.add_wall(found.metrics, found.wall)


def _measured(found: WandbFound) -> bool:
    if found.error:
        return True  # refused, and listed -- never silently skipped
    return bool(numeric_leaves(found.summary)) or bool(found.series)


def _new_found(root: Path, run_dir: Path, meta: dict) -> WandbFound:
    """The run's legacy identity: named `<program stem>/<run id>`."""
    short_id = run_dir.name.rsplit("-", 1)[-1]
    program = str(meta.get("program") or "").rsplit("/", 1)[-1]
    stem = program[:-3] if program.endswith(".py") else program
    return WandbFound(
        project=root.name,
        name=f"{stem}/{short_id}" if stem else run_dir.name,
        source_path=str(run_dir),
        family=stem or None,  # the fallback; a log's project replaces it
        status="recorded",
        started=meta.get("startedAt"),
        adapter=ADAPTER,
        run_id=short_id,
    )


def _read_run(root: Path, run_dir: Path) -> WandbFound | None:
    files = run_dir / "files"
    meta = _json_file(files / "wandb-metadata.json") or {}
    logs = sorted(run_dir.glob(LOG_GLOB))
    if not logs and not (files / "wandb-summary.json").is_file():
        return None  # nothing recorded here
    found = _new_found(root, run_dir, meta)
    if len(logs) > 1:
        found.error = f"{len(logs)} .wandb logs in one run directory; W&B writes one"
    elif logs:
        _from_log(found, logs[0])
    else:
        _from_files(found, files, meta)
    _metadata_git(found, meta)
    found.outputs = _outputs(files)
    _legacy_view(found)
    return found if _measured(found) else None


def wandb_runs(root: Path, seen: set[str]) -> list[WandbFound]:
    """Every run directory under `<root>/wandb/` holding a log or a summary.

    Named `<program stem>/<run id>` (the directory name when the metadata
    names no program), the ledger's name for W&B runs since the first
    reader; a name already in `seen` is skipped. `latest-run` is a symlink
    to a run already listed and is not read twice."""
    base = Path(root) / BASE_DIR
    if not base.is_dir():
        return []
    out: list[WandbFound] = []
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir() and not p.is_symlink()):
        found = _read_run(Path(root), run_dir)
        if found is not None and found.name not in seen:
            seen.add(found.name)
            out.append(found)
    return out


# ---- the mapping ---------------------------------------------------------------


def _status(found: WandbFound) -> tuple[str | None, str | None]:
    """(status, run caveat). Only the exit record says how a run ended."""
    if found.log is None:
        return None, (
            "no .wandb log: read from files/, which records no exit code, no curve"
            " and no project (family is the program name)"
        )
    if found.log.exit_code is None:
        return None, (
            "the .wandb log has no exit record: the run is still in flight, or the"
            " process died before wandb.finish; W&B cannot say which"
        )
    return ("complete" if found.log.exit_code == 0 else "failed"), None


def _metric(name: str, value: Any, unit: str | None = None) -> dict:
    return {
        "name": name,
        "value": schema.spell_value(value),
        "direction": None,
        "step": None,
        "split": None,
        "unit": unit,
        "labels": {},
    }


def _metrics(found: WandbFound) -> list[dict]:
    metrics = [_metric(n, v) for n, v in numeric_leaves(found.summary)]
    if found.wall is not None and all(m["name"] != WALL_METRIC for m in metrics):
        metrics.append(_metric(WALL_METRIC, found.wall, timing.WALL_UNIT))
    return metrics


def _code(found: WandbFound) -> dict:
    sha = found.git_commit
    repo = found.git_remote
    return {
        "git_sha": sha,
        "dirty": None,
        "repo": schema.strip_credentials(repo) if repo else None,
        "version": None,
        "provenance": "inferred" if sha else "unknown",
        "evidence": GIT_EVIDENCE if sha else None,
    }


def _data(found: WandbFound) -> list[dict]:
    return [
        {"name": f"files/{rel}", "sha256": schema.sha256_file(path), "role": "output"}
        for rel, path in found.outputs
    ]


def _run_caveats(found: WandbFound, lossy: list[str]) -> list[str]:
    out = []
    status_note = _status(found)[1]
    if status_note:
        out.append(f"{found.name}: {status_note}")
    if found.log is not None and found.log.truncated:
        out.append(
            f"{found.name}: the .wandb log ends in a partial record (still being"
            " written, or the process died mid-write); read up to it"
        )
    if lossy:
        out.append(
            f"{found.name}: non-finite value(s) stored as strings, not numbers: {', '.join(lossy)}"
        )
    return out


def to_core_record(
    found: WandbFound, *, caveats: list[str], replicate: tuple[str, int] | None
) -> schema.RunRecord:
    """The core record for one W&B run, validated by `from_dict`.

    `replicate` is the ledger-wide `<config>_seed<N>` naming convention
    (`trackers.replicate_of`). Raises `RecordError` for a damaged log."""
    if found.error:
        raise schema.RecordError("", f"{found.name}: {found.error}")
    from attestation import __version__

    lossy: list[str] = []
    config = json_safe(found.raw_config, "config", lossy)
    return schema.from_dict(
        {
            "schema": schema.SCHEMA,
            "run_id": found.run_id,
            "family": found.family,
            "arm": replicate[0] if replicate else None,
            "status": _status(found)[0],
            "started_at": schema.format_timestamp(found.start) if found.start else None,
            "finished_at": None,
            "producer": {"name": f"attestation {ADAPTER}-import", "version": __version__},
            "code": _code(found),
            "config": config,
            "data": _data(found),
            "metrics": _metrics(found),
            "history": history_ref(found.series),
            "seeds": {"seed": replicate[1]} if replicate else None,
            "dependencies": None,
            "ext": {
                IMPORT_NAMESPACE: {
                    "adapter": ADAPTER,
                    "status": found.status,
                    "notes": None,
                    "corpus": None,
                    "caveats": caveats,
                    "run_caveats": _run_caveats(found, lossy),
                }
            },
        }
    )


__all__ = ["WandbFound", "files_config", "numeric_leaves", "to_core_record", "wandb_runs"]
