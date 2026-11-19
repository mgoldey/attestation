"""The on-disk run directory: fixed paths attestation scans.

    <run_dir>/attestation/run.json      core record, written once, last
    <run_dir>/attestation/state.json    in-flight job state (backend-owned)
    <run_dir>/attestation/digests.json  sha256 of every file outside attestation/
    <run_dir>/attestation/history.jsonl logged curves; bound by the record's sha256
    <run_dir>/steps/NN-<name>/          one directory per step / tier, in order
    <run_dir>/logs/run.log              whole-run log
    <run_dir>/logs/NN-<name>.log        one per step

Only attestation/ is required: steps/ and logs/ are optional, and outputs may
live anywhere under the run directory (the manifest covers every file).
Write order is steps, logs and history.jsonl -> digests.json -> run.json, so a
directory with run.json is complete and one with only state.json is in
flight. history.jsonl is under attestation/, so the manifest does not list it:
the record binds it directly (`history.sha256`), once. Spec:
docs/superpowers/specs/2026-11-19-core-run-record-design.md.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from attestation_schema.canonical import canonicalize
from attestation_schema.record import HISTORY_PATH, DataDigest

ATTESTATION_DIR = "attestation"
RUN_JSON = f"{ATTESTATION_DIR}/run.json"
STATE_JSON = f"{ATTESTATION_DIR}/state.json"
DIGESTS_JSON = f"{ATTESTATION_DIR}/digests.json"
HISTORY_JSONL = HISTORY_PATH  # attestation/history.jsonl, owned by history.py
STEPS_DIR = "steps"
LOGS_DIR = "logs"
RUN_LOG = f"{LOGS_DIR}/run.log"

DIGESTS_SCHEMA = "attestation.digests/1"
STATE_SCHEMA = "attestation.state/1"
MANIFEST_ROLE = "manifest"

_STEP_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")


class LayoutError(ValueError):
    """A run directory, or a name in one, that breaks the layout."""


def step_name(index: int, name: str) -> str:
    """`NN-<name>`: the directory (and log stem) of step `index`."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise LayoutError(f"step index must be an integer >= 0, got {index!r}")
    if not isinstance(name, str) or not _STEP_NAME.fullmatch(name):
        raise LayoutError(f"step name {name!r} does not match {_STEP_NAME.pattern}")
    return f"{index:02d}-{name}"


def step_dir(run_dir: str | os.PathLike[str], index: int, name: str) -> Path:
    """Where step `index` writes its outputs."""
    return Path(run_dir) / STEPS_DIR / step_name(index, name)


def step_log(run_dir: str | os.PathLike[str], index: int, name: str) -> Path:
    """Where step `index` writes its log."""
    return Path(run_dir) / LOGS_DIR / f"{step_name(index, name)}.log"


def is_run_dir(path: str | os.PathLike[str]) -> bool:
    """A run has a record (finished) or a state file (in flight, or died in flight)."""
    p = Path(path)
    return (p / RUN_JSON).is_file() or (p / STATE_JSON).is_file()


def find_runs(root: str | os.PathLike[str]) -> list[Path]:
    """Every run directory under `root` (itself included), sorted.

    Runs do not nest: the walk stops descending at a run. Hidden directories
    are skipped, and symlinked directories are not followed.
    """
    found: list[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        if is_run_dir(dirpath):
            found.append(Path(dirpath))
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
    return sorted(found)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """sha256 of every regular file under `run_dir` except `attestation/`.

    A symlink is refused, not followed: a link out of the run directory would
    make the manifest describe someone else's file.
    """
    root = Path(run_dir)
    files: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        if here == root:
            dirnames[:] = [d for d in dirnames if d != ATTESTATION_DIR]
        for name in [*dirnames, *filenames]:
            if (here / name).is_symlink():
                raise LayoutError(f"{(here / name).relative_to(root)} is a symlink")
        for name in filenames:
            path = here / name
            files[path.relative_to(root).as_posix()] = _sha256_file(path)
    return {"schema": DIGESTS_SCHEMA, "files": files}


def write_manifest(run_dir: str | os.PathLike[str]) -> DataDigest:
    """Write `attestation/digests.json`; return its `data` entry for the record.

    Refused once `run.json` exists: the record already binds the old manifest.
    """
    root = Path(run_dir)
    if (root / RUN_JSON).exists():
        raise LayoutError(f"{root / RUN_JSON} exists; the manifest is written before the record")
    payload = canonicalize(build_manifest(root))
    target = root / DIGESTS_JSON
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, target)
    return DataDigest(
        name=DIGESTS_JSON, sha256=hashlib.sha256(payload).hexdigest(), role=MANIFEST_ROLE
    )
