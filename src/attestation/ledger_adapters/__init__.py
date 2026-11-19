"""Adapters that turn what is on disk into core run records for the ledger.

The ledger reads ONLY `attestation_schema.RunRecord`. Everything that knows a
tracker's layout lives in this package, behind one protocol (`Adapter`):

- `core` reads a directory that already holds `attestation/run.json`
  (smeltery, the MLflow exporter) with no conversion;
- `generic` reads the conventions ML/science repos already share
  (`results/`, `logs/`, `configs/`, ... holding JSON/JSONL/CSV/YAML/TOML);
- one adapter per tracker layout (`trackers.py`), each wrapping its reader in
  `generic` and converting what it found with `trackers.found_to_core`.

No project is named here and none needs to be: `ledger.scan` walks a
workspace root, treats every subdirectory as a project, and imports it with
`import_project`. Prefer teaching a reader a new convention over adding a
project-specific adapter -- a convention helps every project.

Spec: docs/superpowers/specs/2026-11-19-ledger-reads-core-records-design.md.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import attestation_schema as schema

from attestation.ledger_adapters import generic
from attestation.ledger_adapters.core import CORE
from attestation.ledger_adapters.found import IMPORT_NAMESPACE
from attestation.ledger_adapters.trackers import LEGACY_ADAPTERS

__all__ = [
    "ADAPTERS",
    "IMPORT_NAMESPACE",
    "LEGACY_ADAPTERS",
    "Adapter",
    "ImportResult",
    "Imported",
    "Quarantined",
    "diagnose_empty",
    "import_project",
]


class Adapter(Protocol):
    """One way of finding runs on disk and saying them as core records.

    `scan` yields items carrying at least `name` (the ledger's key for the
    run within its project) and `source_path` (where it was found); `taken`
    is the set of names earlier adapters already claimed, shared so a
    results measurement outranks a config spec of the same name and a
    tracker run colliding with a results run is recorded once.
    `to_core_record` raises `attestation_schema.RecordError` for an item the
    core record cannot express; the pipeline reports it rather than crash."""

    @property
    def name(self) -> str:
        """The adapter's name, stored in `runs.adapter`."""
        ...

    def detect(self, path: Path) -> bool:
        """Cheaply: does `path` hold anything this adapter reads?"""
        ...

    def scan(self, path: Path, taken: set[str] | None = None) -> Iterable[Any]:
        """What this adapter found under `path`, each with `name` and
        `source_path`, adding the names it claims to `taken`."""
        ...

    def to_core_record(self, found: Any) -> schema.RunRecord:
        """One found item as a validated core record (or `RecordError`)."""
        ...


# Native records first, then the legacy readers in their historical order.
ADAPTERS: tuple[Adapter, ...] = (CORE, *LEGACY_ADAPTERS)


@dataclass(frozen=True)
class Imported:
    """A core record and where the ledger found it. `name` and
    `source_path` belong to the import, not to the run."""

    record: schema.RunRecord
    name: str
    source_path: str
    adapter: str


@dataclass(frozen=True)
class Quarantined:
    """A record that failed conformance: never imported, always reported.
    `rules` are the conformance rules it broke (`read` for a legacy run the
    core record cannot express); `reasons` say where and why."""

    path: str
    adapter: str
    rules: list[str]
    reasons: list[str]


@dataclass
class ImportResult:
    """A project's imports; the records quarantined (also as `refused`,
    `<path>: <error>`, the older shape); and what the readers skipped."""

    imported: list[Imported] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    quarantined: list[Quarantined] = field(default_factory=list)
    lossy: list[str] = field(default_factory=list)


def _quarantine(found: Any, adapter: str, exc: schema.RecordError) -> Quarantined:
    rules = getattr(exc, "rules", None) or ["read"]
    reasons = getattr(exc, "reasons", None) or [str(exc)]
    return Quarantined(str(found.source_path), adapter, list(rules), list(reasons))


def import_project(root: Path, adapters: Iterable[Adapter] = ADAPTERS) -> ImportResult:
    """Every run under one project directory, as validated core records.

    A record that fails conformance is quarantined -- listed with its path and
    reasons, never imported, never coerced (conformance spec, Decision 3)."""
    root = Path(root)
    out = ImportResult()
    taken: set[str] = set()
    for adapter in adapters:
        if not adapter.detect(root):
            continue
        with generic.skipped_non_finite() as skipped:
            found_items = list(adapter.scan(root, taken))
        if skipped:
            names = sorted(set(skipped))
            shown = ", ".join(names[:8]) + (", ..." if len(names) > 8 else "")
            out.lossy.append(
                f"the {adapter.name} reader skipped {len(skipped)} non-finite value(s)"
                f" (NaN/inf: not measured, not stored): {shown}"
            )
        for found in found_items:
            try:
                record = adapter.to_core_record(found)
            except schema.RecordError as exc:
                out.refused.append(f"{found.source_path}: {exc}")
                out.quarantined.append(_quarantine(found, adapter.name, exc))
                continue
            out.imported.append(Imported(record, found.name, found.source_path, adapter.name))
    return out


def diagnose_empty(root: Path) -> str:
    """Why a project yielded no runs -- see `generic.diagnose_empty`."""
    return generic.diagnose_empty(root)
