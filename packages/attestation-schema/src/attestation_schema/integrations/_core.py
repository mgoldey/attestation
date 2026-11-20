"""What every training-loop callback shares: the recorder and its observers.

Standard library only (it runs in the bare 3.11 job). The framework modules
(`huggingface.py`, `lightning.py`) translate their trainer's hooks into the
calls below; everything that can be decided without a framework is decided
here, so it is tested without one.

    Recorder.begin()          at train start (rank 0): refuse an existing
                              record, open the history sidecar
    Recorder.log(...)         each logged value -> one history point
    Recorder.finish(...)      at the end: history closed -> digests.json ->
                              run.json (the layout's write order)

Spec: docs/superpowers/specs/2026-11-19-training-callbacks-design.md.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any, TypeAlias, cast

from attestation_schema import layout
from attestation_schema.canonical import MAX_SAFE_INTEGER
from attestation_schema.conformance import _REDACTED, _secret_key, credential_in
from attestation_schema.dataset import DatasetRecord, dataset_lineage_entries, read_dataset
from attestation_schema.files import sha256_file
from attestation_schema.history import HistoryWriter
from attestation_schema.identity import git_state, package_identity
from attestation_schema.record import (
    STATUSES,
    CodeIdentity,
    DataDigest,
    Dependency,
    Metric,
    Producer,
    RunRecord,
    format_timestamp,
    write,
)

OUTPUT_ROLE = "output"
REDACTED = "<redacted>"

# A dataset argument: a dataset directory, its dataset.json, a record, or any
# of those paired with the name of the split the run trained on.
DatasetLike: TypeAlias = str | os.PathLike[str] | DatasetRecord
DatasetArg: TypeAlias = DatasetLike | tuple[DatasetLike, str]


def now() -> str:
    """The current instant in the record's one spelling."""
    return format_timestamp(datetime.now(UTC))


def new_run_id() -> str:
    """A fresh run id: one per execution, so a default run directory never collides."""
    return uuid.uuid4().hex[:12]


def producer(kind: str) -> Producer:
    """`attestation-schema <kind>` at the installed package version."""
    try:
        version: str | None = metadata.version("attestation-schema")
    except metadata.PackageNotFoundError:
        version = None
    return Producer(name=f"attestation-schema {kind}", version=version)


# ---- datasets ----------------------------------------------------------------


def dataset_entries(datasets: Iterable[DatasetArg], role: str = "train") -> list[DataDigest]:
    """`data[]` entries for the datasets a caller named: the content digest
    under `dataset_id` and, for a `(dataset, split)` pair, the split file by
    its digest (`dataset_lineage_entries`) -- what `resolve_lineage` links by."""
    out: list[DataDigest] = []
    for item in datasets:
        if isinstance(item, tuple):
            target, split = cast("tuple[DatasetLike, str]", item)
        else:
            target, split = item, None
        record = target if isinstance(target, DatasetRecord) else read_dataset(target)
        out += dataset_lineage_entries(record, role=role, split=split)
    return out


def undigested_data(evaluated: bool) -> list[DataDigest]:
    """What the callback observed when no dataset was named: the run read
    training data (and evaluation data, if it evaluated), bytes not digested.
    Never `[]`: a record with no data entry says the run read nothing."""
    out = [DataDigest(name="train", sha256=None, role="train")]
    if evaluated:
        out.append(DataDigest(name="eval", sha256=None, role="eval"))
    return out


# ---- outputs -----------------------------------------------------------------


def output_name(path: Path, root: Path) -> str:
    """A checkpoint's `data[]` name: its path relative to the trainer's output
    root (never absolute -- an absolute path names a machine)."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def output_entries(paths: Iterable[Path], root: Path) -> list[DataDigest]:
    """One `output` entry per regular file in `paths` (a directory contributes
    every file under it, sorted), each by the sha256 of its bytes. A path that
    no longer exists contributes nothing; a symlink is not followed."""
    files: list[Path] = []
    for p in paths:
        if p.is_symlink() or not p.exists():
            continue
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.is_file() and not f.is_symlink())
        else:
            files.append(p)
    out: dict[str, DataDigest] = {}
    for f in files:
        name = output_name(f, root)
        out.setdefault(name, DataDigest(name=name, sha256=sha256_file(f), role=OUTPUT_ROLE))
    return list(out.values())


# ---- config ------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    """`value` as JSON canonicalization can write it, without guessing types:
    enums by value, paths and other objects as `str()`, tuples and sets as
    lists, non-finite floats and integers past 2^53 as their Python spelling."""
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, int):
        return value if abs(value) <= MAX_SAFE_INTEGER else str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted((jsonable(v) for v in value), key=repr)
    item = getattr(value, "item", None)  # a numpy or torch scalar
    if callable(item):
        try:
            return jsonable(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def redact(value: Any, key: str | None = None, path: str = "$") -> tuple[Any, list[str]]:
    """Replace every string the conformance `credentials` rule would flag with
    `<redacted>`, returning the paths replaced. A record's digest is its
    identity, so a leaked key cannot be edited out after it is written."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        hits: list[str] = []
        for k, v in value.items():
            out[k], found = redact(v, k, f"{path}.{k}")
            hits += found
        return out, hits
    if isinstance(value, list):
        items, hits = [], []
        for i, v in enumerate(value):
            item, found = redact(v, None, f"{path}[{i}]")
            items.append(item)
            hits += found
        return items, hits
    if isinstance(value, str) and value.strip():
        secret_named = key is not None and _secret_key(key)
        if credential_in(value) or (secret_named and not _REDACTED.fullmatch(value.strip())):
            return REDACTED, [path]
    return value, []


# ---- identities --------------------------------------------------------------
# One rule for every producer: `attestation_schema.identity` (0.2.0). These
# wrappers only decide WHICH packages a callback names.


def installed_dependency(name: str, module: str, dist: str) -> Dependency | None:
    """`identity.package_identity` for an imported module; None when this
    process never imported it (a list is observed: what the run did not load
    is not named). The module object is passed, so nothing is imported here."""
    mod = sys.modules.get(module)
    if mod is None:
        return None
    return package_identity(dist, mod, name=name)


def python_dependency() -> Dependency:
    """The interpreter: its version is observed exactly; where it came from is not."""
    return Dependency(
        name="python",
        version=platform.python_version(),
        build_sha=None,
        dirty=None,
        provenance="verified",
        evidence="sys.version_info",
        source=None,
    )


def dependencies(modules: Sequence[tuple[str, str, str]]) -> list[Dependency]:
    """Python, then each `(name, module, distribution)` this process imported."""
    out = [python_dependency()]
    for name, module, dist in modules:
        dep = installed_dependency(name, module, dist)
        if dep is not None:
            out.append(dep)
    return out


def script_directory() -> Path:
    """The running script's directory (`sys.argv[0]`), or the working
    directory when there is no script file (a notebook, `python -c`)."""
    script = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    if script is not None and script.is_file():
        return script.resolve().parent
    return Path.cwd()


def code_identity(directory: Path | None = None) -> CodeIdentity:
    """The training code's git checkout, read now: HEAD, whether tracked
    files differ from it, and the origin URL without credentials.
    `inferred` -- HEAD names the checkout, not the bytes Python loaded."""
    where = directory if directory is not None else script_directory()
    git = git_state(where)
    if git.sha is None:
        return CodeIdentity(
            git_sha=None,
            dirty=None,
            repo=None,
            version=None,
            provenance="unknown",
            evidence="no git checkout found at the training script (or git is not installed)",
        )
    return CodeIdentity(
        git_sha=git.sha,
        dirty=git.dirty,
        repo=git.remote,
        version=None,
        provenance="inferred",
        evidence="git HEAD of the training script's checkout at train start;"
        " dirty = tracked files differ from HEAD (git status)",
    )


# ---- values ------------------------------------------------------------------


def number(value: Any) -> int | float | None:
    """A logged value as a metric value, or None if it is not a number
    (strings, booleans and containers are not metrics)."""
    if isinstance(value, bool):
        return None
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, int | float):
        try:
            value = item()
        except (TypeError, ValueError, RuntimeError):
            return None
        if isinstance(value, bool):
            return None
    if isinstance(value, int):
        return value if abs(value) <= MAX_SAFE_INTEGER else float(value)
    if isinstance(value, float):
        return value
    return None


def merge_directions(declared: Iterable[tuple[str, str]]) -> dict[str, str | None]:
    """Directions the user declared (a checkpoint's `mode`, HF's
    `greater_is_better`); a name declared both ways gets None."""
    out: dict[str, str | None] = {}
    for name, direction in declared:
        if direction not in ("min", "max"):
            continue
        out[name] = direction if out.get(name, direction) == direction else None
    return out


# ---- the recorder ------------------------------------------------------------


@dataclass
class Options:
    """What the caller states; every field optional (see the callbacks)."""

    run_dir: str | os.PathLike[str] | None = None
    run_id: str | None = None
    family: str | None = None
    arm: str | None = None
    datasets: Sequence[DatasetArg] | None = None
    dataset_role: str = "train"
    data: Sequence[DataDigest] = ()
    seeds: dict[str, int | list[int]] | None = None
    code: CodeIdentity | None = None
    directions: Mapping[str, str] = field(default_factory=dict)


class Recorder:
    """One run's record, built across the training loop and written once."""

    def __init__(self, options: Options, kind: str, namespace: str):
        self.options = options
        self.run_id = options.run_id or new_run_id()
        self.producer = producer(kind)
        self.namespace = namespace
        self.run_dir: Path | None = None
        self.started_at: str | None = None
        self.written: Path | None = None
        self._history: HistoryWriter | None = None
        self._series: dict[str, int] = {}
        self.last: dict[str, tuple[int | float, int]] = {}  # name -> (value, step)
        self.code: CodeIdentity | None = None

    def default_run_dir(self, output_root: str | os.PathLike[str]) -> Path:
        """`<output_root>/attestation/run-<run_id>`."""
        return Path(output_root) / layout.ATTESTATION_DIR / f"run-{self.run_id}"

    @property
    def begun(self) -> bool:
        return self.run_dir is not None

    def begin(self, output_root: str | os.PathLike[str]) -> None:
        """Train start: fix the run directory, refuse one that already holds a
        record (before any training, never after), read the code identity
        while the checkout is as it was at start, open the history."""
        given = self.options.run_dir
        rd = Path(given) if given is not None else self.default_run_dir(output_root)
        if (rd / layout.RUN_JSON).exists():
            raise FileExistsError(f"{rd / layout.RUN_JSON} exists; a run directory holds one run")
        self.run_dir = rd
        self.started_at = now()
        self.code = self.options.code or code_identity()
        self._history = HistoryWriter(rd)

    def log(self, values: Mapping[str, Any], step: int) -> None:
        """Every numeric value in `values` becomes a history point at `step`."""
        if self._history is None:
            return
        at = now()
        for name, raw in values.items():
            value = number(raw)
            if value is None or not isinstance(name, str) or not name:
                continue
            if name not in self._series:
                self._series[name] = self._history.series(name)
            self._history.point(self._series[name], step, value, at)
            self.last[name] = (value, step)

    def abort(self) -> None:
        """Drop the history's temporary file (no record will be written)."""
        if self._history is not None:
            self._history.abort()
            self._history = None

    def metrics(
        self,
        final: Mapping[str, tuple[int | float, int | None]] | None = None,
        declared: Mapping[str, str | None] | None = None,
    ) -> list[Metric]:
        """Final values: `final` when the trainer holds its own, else each
        series' last point. Direction: the caller's `directions=`, else what
        the user declared to the trainer, else None -- never inferred."""
        values = final if final is not None else dict(self.last)
        directions = {**(declared or {}), **self.options.directions}
        return [
            Metric(
                name=name,
                value=value,
                direction=directions.get(name),
                step=step,
                split=None,
                unit=None,
                labels={},
            )
            for name, (value, step) in values.items()
        ]

    def data(self, evaluated: bool, outputs: list[DataDigest]) -> list[DataDigest]:
        """The datasets named (or what was observed of the data), the caller's
        own entries, then the outputs."""
        o = self.options
        named = (
            undigested_data(evaluated)
            if o.datasets is None
            else dataset_entries(o.datasets, o.dataset_role)
        )
        out: dict[str, DataDigest] = {}
        for entry in [*named, *o.data, *outputs]:
            out.setdefault(entry.name, entry)
        return list(out.values())

    def finish(
        self,
        *,
        status: str,
        config: Mapping[str, Any],
        data: list[DataDigest],
        metrics: list[Metric],
        seeds: dict[str, int | list[int]] | None,
        dependencies: list[Dependency] | None,
        ext: dict[str, Any],
    ) -> Path:
        """Close the history, write the manifest, then the record."""
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
        if self.run_dir is None:
            raise RuntimeError("finish() before begin()")
        if self.written is not None:
            return self.written
        rd = self.run_dir
        history = self._history.close() if self._history is not None else None
        self._history = None
        clean, redacted = redact(jsonable(dict(config)), path="$.config")
        extra, redacted_ext = redact(jsonable(ext), path=f"$.ext.{self.namespace}")
        if redacted or redacted_ext:
            extra = {**extra, "redacted": [*redacted, *redacted_ext]}
        manifest = layout.write_manifest(rd)
        record = RunRecord(
            run_id=self.run_id,
            family=self.options.family,
            arm=self.options.arm,
            status=status,
            started_at=self.started_at,
            finished_at=now(),
            producer=self.producer,
            code=self.code or code_identity(),
            config=clean,
            data=[*data, manifest],
            metrics=metrics,
            history=history,
            seeds=self.options.seeds if self.options.seeds is not None else seeds,
            dependencies=dependencies,
            ext={self.namespace: extra},
        )
        self.written = write(record, rd)
        return self.written


def error_status(exc: BaseException) -> str:
    """`cancelled` for an interrupt someone sent, `failed` for anything else."""
    return "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"


def error_ext(exc: BaseException) -> dict[str, str]:
    """The exception's type and first line, for `ext.<namespace>.error`."""
    text = str(exc).strip().splitlines()
    return {"type": type(exc).__name__, "message": text[0][:500] if text else ""}
