"""The native adapter: a directory that already holds a core record.

smeltery writes `<run_dir>/attestation/run.json` itself, and the MLflow
exporter writes one per run; those are read with `attestation_schema.read`,
no conversion. Every run directory holding a record is first checked against
the whole conformance suite (`attestation_schema.conformance`: the reader,
canonical bytes, the manifest's binding, the history sidecar, `state.json`,
credentials, ext namespaces) and ANY violation quarantines it: it is never
imported and the pipeline reports it loudly, never coerced. A directory with
only `state.json` is a run in flight and is skipped (the ledger records runs
that ended; `attestation.inflight` reports those, read at query time).

Specs: docs/superpowers/specs/2026-11-19-ledger-reads-core-records-design.md,
docs/superpowers/specs/2026-11-19-conformance-suite-design.md (Decision 3).
"""

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import attestation_schema as schema
from attestation_schema import conformance, layout

# How far below a project directory a run directory is looked for. The
# known producers put runs one or two levels down (`records/run-<id>/`,
# `runs/<campaign>/run-<id>/`); an unbounded walk would visit every file of
# every dataset in the workspace on every scan, to find nothing.
MAX_DEPTH = 4


class NonConformingRecord(schema.RecordError):
    """A run directory that broke at least one conformance rule. `report`
    holds every violation; the message lists them relative to the run."""

    def __init__(self, report: conformance.Report):
        self.report = report
        self.rules = report.rules
        self.reasons = [_relative(report.run_dir, v) for v in report.violations]
        super().__init__("", "; ".join(self.reasons))


def _relative(run_dir: Path, v: conformance.Violation) -> str:
    try:
        file = Path(v.file).relative_to(run_dir).as_posix()
    except ValueError:
        file = v.file
    return f"{file} at {v.json_path}: [{v.rule}] {v.message}"


@dataclass(frozen=True)
class NativeRun:
    """A run directory holding a record: the record, or why it was refused."""

    source_path: str
    name: str
    record: schema.RunRecord | None
    error: schema.RecordError | None


def all_run_dirs(root: Path) -> list[Path]:
    """Every run directory under `root` (itself included) -- a record, a
    `state.json`, or both -- at most `MAX_DEPTH` levels down, sorted. Runs do
    not nest, hidden directories are skipped, and symlinks are not followed --
    the rules of `attestation_schema.layout.find_runs`, with a depth bound.
    `attestation.inflight` reads the ones without a record from here, so the
    ledger and the status view walk one layout by one rule."""
    root = Path(root)
    found: list[Path] = []
    base = len(root.parts)
    for dirpath, dirnames, _ in os.walk(root):
        if layout.is_run_dir(dirpath):
            dirnames[:] = []
            found.append(Path(dirpath))
            continue
        if len(Path(dirpath).parts) - base >= MAX_DEPTH:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
    return sorted(found)


def run_dirs(root: Path) -> list[Path]:
    """The run directories under `root` that hold a terminal record."""
    return [d for d in all_run_dirs(root) if schema.record_path(d).is_file()]


def _read(run_dir: Path) -> tuple[schema.RunRecord | None, schema.RecordError | None]:
    """The record if the run directory conforms, else why it is quarantined.

    `verify_outputs=False`: re-hashing every output of every run on every
    scan costs time proportional to the outputs, which the ledger does not
    store; the manifest's own format and its binding to the record are still
    checked. Re-hashing is the producer's CI's job (the pytest plugin)."""
    report = conformance.check_run_dir(run_dir, verify_outputs=False)
    if not report.ok or report.record is None:
        return None, NonConformingRecord(report)
    return report.record, None


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
