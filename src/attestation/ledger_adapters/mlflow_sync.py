"""`attest mlflow sync`: an MLflow file store, written down as core records.

Each ended run in `--tracking-dir` becomes `<out>/run-<run_id>/` in the
layout's order (`attestation/history.jsonl`, `digests.json`, `run.json`).
The record is the ledger adapter's own (`trackers.MLFLOW.to_core_record`:
one mapping, the exporter's), with `producer` `attestation mlflow-sync` and
`ext.mlflow-sync` in place of `ext.ledger-import` -- so the ledger reads the
synced record as native, and its code and dependency provenance (from the
run-context provider's `attestation.*` tags) reach it. No mlflow needed.

Idempotent: a run whose record is already there byte for byte (or differs
only in `producer`, an attestation upgrade) is left alone; one that differs
otherwise is REFUSED, never silently overwritten, unless `--force`. Every
directory is built beside its target, checked with the conformance suite,
then renamed into place.

The command lives here, not in cli.py: no tracker is named outside this
package (tests/test_tracker_names.py). cli.py registers `COMMANDS`
generically. Spec: docs/superpowers/specs/2026-11-20-mlflow-plugin-design.md.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import attestation_schema as schema
from attestation_schema import conformance, layout

from attestation.ledger_adapters import mlflow_store
from attestation.ledger_adapters.found import IMPORT_NAMESPACE

NAMESPACE = "mlflow-sync"
PRODUCER_NAME = "attestation mlflow-sync"
DEFAULT_OUT = "mlflow-records"

HELP = {
    "mlflow": "MLflow runs as core records (provenance tags read back)",
    "mlflow.sync": "write a core record per ended MLflow run; refuses a changed run",
}

# What happened to one run. `refused`, `unreadable` and `nonconforming` make
# the command exit 1.
WRITTEN, UNCHANGED, REPLACED = "written", "unchanged", "replaced"
REFUSED, UNREADABLE, NONCONFORMING = "refused", "unreadable", "nonconforming"
FAILURES = frozenset({REFUSED, UNREADABLE, NONCONFORMING})
STATES = (WRITTEN, REPLACED, UNCHANGED, REFUSED, UNREADABLE, NONCONFORMING)


@dataclass(frozen=True)
class Outcome:
    """One run's sync: what happened, where, and why (for anything but written)."""

    run_id: str
    state: str
    path: Path
    detail: str = ""
    record: schema.RunRecord | None = None


@dataclass
class SyncResult:
    """Every run's outcome, and how many runs had no record to write."""

    outcomes: list[Outcome]
    skipped: int  # not ended, or deleted: no record, as every MLflow reader

    @property
    def ok(self) -> bool:
        """No run was refused, unreadable or non-conforming."""
        return not any(o.state in FAILURES for o in self.outcomes)


def synced_record(found: mlflow_store.MlflowFound) -> schema.RunRecord:
    """The adapter's record for `found`, as `sync` writes it: its own
    producer, and `ext.mlflow-sync` instead of the importer's namespace.
    Raises `RecordError` for a run MLflow itself cannot read."""
    from attestation import __version__
    from attestation.ledger_adapters.trackers import MLFLOW

    record = MLFLOW.to_core_record(found)
    imported = record.ext.get(IMPORT_NAMESPACE, {})
    run_name = found.name.rsplit("/", 1)[0] if found.name != found.run_id else None
    ext = {
        NAMESPACE: {
            "experiment_id": Path(found.source_path).parent.name,
            "run_name": run_name,
            "notes": list(imported.get("run_caveats") or []),
        }
    }
    producer = schema.Producer(name=PRODUCER_NAME, version=__version__)
    return replace(record, producer=producer, ext=ext)


def _build(found: mlflow_store.MlflowFound, record: schema.RunRecord, where: Path) -> None:
    """The run directory at `where`, in the layout's write order."""
    history = schema.write_history(where, found.series)
    if history != record.history:  # one pure function of the store, computed twice
        raise schema.RecordError("history", "the sidecar differs from the record's reference")
    manifest = layout.write_manifest(where)
    schema.write(replace(record, data=[*record.data, manifest]), where)


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _differences(old: Path, new: Path) -> list[str]:
    """Top-level record fields (and files) that differ between two run dirs."""
    try:
        before = schema.read(old).to_dict()
    except (OSError, schema.RecordError) as exc:
        return [f"the existing record is unreadable ({exc})"]
    after = schema.read(new).to_dict()
    fields = sorted(k for k in after.keys() | before.keys() if before.get(k) != after.get(k))
    if not fields:
        fields = ["files beside the record"]
    return fields


def _swap(tmp: Path, target: Path) -> None:
    """Replace `target` with `tmp`: rename the old aside, the new in, then delete."""
    old = target.with_name(f".{target.name}.old-{secrets.token_hex(4)}")
    os.replace(target, old)
    os.replace(tmp, target)
    shutil.rmtree(old)


def _place(found: mlflow_store.MlflowFound, out: Path, force: bool) -> Outcome:
    run_id = found.run_id
    target = out / f"run-{run_id}"
    try:
        record = synced_record(found)
    except schema.RecordError as exc:
        return Outcome(run_id, UNREADABLE, Path(found.source_path), str(exc))
    tmp = out / f".run-{run_id}.sync-{secrets.token_hex(4)}"
    try:
        _build(found, record, tmp)
        report = conformance.check_run_dir(tmp)
        if not report.ok:
            reasons = "; ".join(f"[{v.rule}] {v.message}" for v in report.violations)
            return Outcome(run_id, NONCONFORMING, target, reasons)
        if not target.exists():
            os.replace(tmp, target)
            return Outcome(run_id, WRITTEN, target, record=record)
        if _files(target) == _files(tmp):
            return Outcome(run_id, UNCHANGED, target, record=record)
        changed = _differences(target, tmp)
        if changed == ["producer"]:
            return Outcome(run_id, UNCHANGED, target, "written by another producer version", record)
        if not force:
            detail = f"the run changed since it was synced ({', '.join(changed)} differ);"
            return Outcome(run_id, REFUSED, target, f"{detail} --force replaces it")
        _swap(tmp, target)
        return Outcome(run_id, REPLACED, target, ", ".join(changed), record)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def sync(tracking_dir: Path, out: Path, *, force: bool = False) -> SyncResult:
    """Write (or check) one run directory per ended run in the file store
    at `tracking_dir`. Never overwrites a record that differs unless `force`."""
    store = Path(tracking_dir)
    if not store.is_dir():
        raise FileNotFoundError(f"{store} is not a directory (an MLflow file store, e.g. mlruns/)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    outcomes: list[Outcome] = []
    skipped = 0
    for exp_dir, family in mlflow_store.experiments(store):
        for run_dir in mlflow_store.run_dirs(exp_dir):
            found = mlflow_store.read_run(
                store.parent, exp_dir, run_dir, family, measured_only=False
            )
            if found is None:
                skipped += 1
                continue
            outcomes.append(_place(found, out, force))
    return SyncResult(outcomes, skipped)


# ---- the command ---------------------------------------------------------------


def register(sub) -> None:
    """Attach `attest mlflow sync` to the top-level subparsers."""
    sp = sub.add_parser("mlflow", help=HELP["mlflow"])
    ml = sp.add_subparsers(dest="mlflow_command", required=True)
    p = ml.add_parser("sync", help=HELP["mlflow.sync"])
    p.add_argument(
        "--tracking-dir",
        default="mlruns",
        help="the MLflow file store (default: ./mlruns); no mlflow is needed to read it",
    )
    p.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"where run-<run_id>/ directories go (default: ./{DEFAULT_OUT})",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="replace a record whose run changed since it was synced",
    )
    p.set_defaults(func=cmd_mlflow_sync)


def cmd_mlflow_sync(args: argparse.Namespace) -> int:
    """write a core record per ended MLflow run; refuses a changed run

    The first line is HELP["mlflow.sync"] (test_cli pins the two equal).
    Exits 1 when any run is refused (changed since it was synced, without
    --force), unreadable, or would not conform."""
    try:
        result = sync(Path(args.tracking_dir), Path(args.out), force=args.force)
    except FileNotFoundError as exc:
        print(f"attest mlflow sync: {exc}")
        return 1
    counts: dict[str, int] = {}
    for o in result.outcomes:
        counts[o.state] = counts.get(o.state, 0) + 1
        line = f"{o.state:<13} {o.run_id}"
        if o.record is not None:
            deps = o.record.dependencies
            line += (
                f"  family={o.record.family} status={o.record.status}"
                f" code={o.record.code.provenance}"
                f" dependencies={'null' if deps is None else len(deps)}"
            )
        if o.detail:
            line += f"\n              {o.detail}"
        print(line)
    summary = ", ".join(f"{counts[st]} {st}" for st in STATES if st in counts) or "no runs"
    print(f"{summary}; {result.skipped} skipped (not ended or deleted) -> {args.out}")
    return 0 if result.ok else 1


__all__ = ["HELP", "Outcome", "SyncResult", "register", "sync", "synced_record"]
