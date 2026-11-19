"""The native adapter: a directory that already holds a core record.

smeltery writes `<run_dir>/attestation/run.json` itself, and the MLflow
exporter writes one per run; those are read with `attestation_schema.read`,
no conversion. A record the reader refuses is reported by the pipeline,
never coerced; a directory with only `state.json` is a run in flight and is
skipped (the ledger records runs that ended).

Spec: docs/superpowers/specs/2026-11-19-ledger-reads-core-records-design.md.
"""

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import attestation_schema as schema
from attestation_schema import layout

# How far below a project directory a run directory is looked for. The
# known producers put runs one or two levels down (`records/run-<id>/`,
# `runs/<campaign>/run-<id>/`); an unbounded walk would visit every file of
# every dataset in the workspace on every scan, to find nothing.
MAX_DEPTH = 4


@dataclass(frozen=True)
class NativeRun:
    """A run directory holding a record: the record, or why it was refused."""

    source_path: str
    name: str
    record: schema.RunRecord | None
    error: schema.RecordError | None


def run_dirs(root: Path) -> list[Path]:
    """Run directories under `root` (itself included) that hold a terminal
    record, at most `MAX_DEPTH` levels down, sorted. Runs do not nest, hidden
    directories are skipped, and symlinks are not followed -- the rules of
    `attestation_schema.layout.find_runs`, with a depth bound."""
    root = Path(root)
    found: list[Path] = []
    base = len(root.parts)
    for dirpath, dirnames, _ in os.walk(root):
        if layout.is_run_dir(dirpath):
            dirnames[:] = []
            if schema.record_path(dirpath).is_file():
                found.append(Path(dirpath))
            continue
        if len(Path(dirpath).parts) - base >= MAX_DEPTH:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
    return sorted(found)


def _read(run_dir: Path) -> tuple[schema.RunRecord | None, schema.RecordError | None]:
    try:
        return schema.read(run_dir), None
    except schema.RecordError as exc:
        return None, exc


def _names(records: list[schema.RunRecord | None], taken: set[str]) -> list[str]:
    """The ledger name of each record: its `run_id`, unless that is shared.

    `run_id` is not unique (re-executions share it). When two records share
    one, or one collides with a name an earlier adapter took, EVERY copy is
    named `<run_id>@<digest12>`: suffixing only the second would let a name
    lookup silently resolve to whichever record sorted first."""
    counts = Counter(r.run_id for r in records if r is not None)
    out = []
    for r in records:
        if r is None:
            out.append("")
        elif counts[r.run_id] > 1 or r.run_id in taken:
            out.append(f"{r.run_id}@{schema.record_digest(r)[:12]}")
        else:
            out.append(r.run_id)
    return out


class CoreAdapter:
    """`Adapter` for directories holding `attestation/run.json`."""

    name = "core"

    def detect(self, path: Path) -> bool:
        """True when a terminal record sits within `MAX_DEPTH` of `path`."""
        return bool(run_dirs(path))

    def scan(self, path: Path, taken: set[str] | None = None) -> list[NativeRun]:
        """Every record under `path`, read, named, its names claimed in `taken`."""
        taken = set() if taken is None else taken
        dirs = run_dirs(path)
        read = [_read(d) for d in dirs]
        names = _names([r for r, _ in read], taken)
        taken.update(n for n in names if n)
        return [
            NativeRun(str(d), n, record, error)
            for d, n, (record, error) in zip(dirs, names, read, strict=True)
        ]

    def to_core_record(self, found: NativeRun) -> schema.RunRecord:
        """The record as read -- no conversion -- or the reader's refusal."""
        if found.record is None:
            assert found.error is not None
            raise found.error
        return found.record


CORE = CoreAdapter()
