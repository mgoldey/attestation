"""The legacy readers as adapters: each turns what it found into a core record.

Every reader in `generic` -- the plain results/configs conventions and the
five tracker layouts -- is wrapped here as one `TrackerAdapter` behind the
`ledger_adapters.Adapter` protocol. The readers themselves are unchanged
(spec: docs/superpowers/specs/2026-11-19-ledger-reads-core-records-design.md);
what is new is `found_to_core`, the one place a `Found` becomes an
`attestation_schema.RunRecord`, and the caveats, which moved here from
`ledger.py` and now travel inside each record as data.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import attestation_schema as schema

from attestation.ledger_adapters import generic
from attestation.ledger_adapters.found import IMPORT_NAMESPACE, Found

# What each reader cannot do, stated where a reader of the run will see it.
# Moved verbatim from `ledger.ADAPTER_CAVEATS`: a wandb-derived run looked
# identical in `runs.list` and `runs.detail` to a hand-written JSON one, so
# the reader had no way to know that the number in front of them is the
# FINAL logged value rather than the best step, or that an offline W&B run
# carries no metrics at all in files/ until it is synced (see
# generic._wandb_runs). The generic reader has none: a caveat on every run
# is boilerplate, and boilerplate trains a reader to skip the ones that matter.
WANDB_CAVEAT = (
    "read by the wandb adapter: values come from wandb-summary.json, which"
    " holds each metric's final logged value rather than its curve or its"
    " best step. Offline W&B does not write that file until a run is"
    " synced -- see examples/wandb/generate.py for how the committed"
    " fixture's was materialised without a network call"
)
MLFLOW_CAVEAT = (
    "read by the mlflow adapter: each metric is the FINAL line of its"
    " metrics file, not the curve and not the best step"
)
SACRED_CAVEAT = (
    "read by the sacred adapter: each metric is the LAST value in"
    " metrics.json's series, not the curve and not the best step"
)
DVC_CAVEAT = (
    "read by the dvc adapter: each metric file is a snapshot overwritten"
    " on every `dvc repro`, not a curve -- there is no history of a"
    " prior run's value once a stage reruns"
)
HYDRA_CAVEAT = (
    "read by the hydra adapter: metrics come from whatever JSON/CSV file"
    " an arm's directory holds, read as a final value the same way as"
    " every other tracker here, not a curve or a best step"
)

_UNKNOWN_CODE = {
    "git_sha": None,
    "dirty": None,
    "repo": None,
    "version": None,
    "provenance": "unknown",
    "evidence": None,
}


def _version() -> str | None:
    from attestation import __version__

    return __version__


def _started_at(text: str | None, assume_utc: bool) -> str | None:
    """A tracker's start time in the core spelling, or None.

    A naive time is read as UTC only when the adapter says its tracker writes
    UTC (W&B's `startedAt`, Sacred's `start_time`); anything unparseable is
    `null` -- "did not say" -- rather than a guess."""
    if not text:
        return None
    try:
        when = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        if not assume_utc:
            return None
        when = when.replace(tzinfo=UTC)
    return schema.format_timestamp(when)


def _json_safe(value: Any, path: str = "", lossy: list[str] | None = None) -> Any:
    """`value` as JSON the record accepts: a non-finite float becomes its
    string (the rule `generic._coerce` already applies to YAML config -- a
    CSV cell `nan` is otherwise the one way a NaN reaches a config). Each
    such path is appended to `lossy`, so the change is a caveat, not silent."""
    if isinstance(value, float) and not math.isfinite(value):
        if lossy is not None:
            lossy.append(path or "(value)")
        return str(value)
    if isinstance(value, dict):
        return {
            str(k): _json_safe(v, f"{path}.{k}" if path else str(k), lossy)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, f"{path}[{i}]", lossy) for i, v in enumerate(value)]
    return value


def _lossy_caveats(
    found: Found, lossy: list[str], dropped: list[str], started: str | None
) -> list[str]:
    """What the conversion changed or dropped, said where a reader of the run
    will see it (conformance spec, Decision 4: nothing coerces silently)."""
    out = []
    if lossy:
        out.append(
            f"{found.name}: non-finite value(s) stored as strings, not numbers: {', '.join(lossy)}"
        )
    if dropped:
        out.append(
            f"{found.name}: duplicate metric value(s) dropped, the first of each kept: "
            + ", ".join(dropped)
        )
    if found.started and started is None:
        out.append(f"{found.name}: start time {found.started!r} not recorded (no UTC instant)")
    return out


def _metrics(found: Found, dropped: list[str] | None = None) -> list[dict]:
    """The core metric list. Duplicates on (name, step, split) keep the first,
    which is what the ledger's insert loop always did (each dropped one is
    appended to `dropped`); direction is never inferred, and a legacy reader
    knows no unit or label."""
    out: list[dict] = []
    seen: set[tuple] = set()
    for m in found.metrics:
        key = (m.metric, m.step, m.split or None)
        if key in seen:
            if dropped is not None:
                dropped.append(f"{m.metric}={m.value!r}")
            continue
        seen.add(key)
        out.append(
            {
                "name": m.metric,
                "value": m.value,
                "direction": None,
                "step": m.step,
                "split": m.split or None,
                "unit": None,
                "labels": {},
            }
        )
    return out


def found_to_core(
    found: Found,
    *,
    adapter: str,
    caveats: list[str],
    run_caveats: list[str],
    started_is_utc: bool = False,
    observes_completion: bool = False,
) -> schema.RunRecord:
    """The core record for one legacy run, validated by `from_dict` (the
    reader's own rules), so an import is held to everything a written
    record is. Raises `attestation_schema.RecordError` for a run the core
    cannot express. What the core has no field for -- the ledger's own
    status word, notes, the corpus an artifact named, the caveats -- goes in
    `ext.ledger-import`. A lossy step (a duplicate dropped, a NaN config value
    stringified, a start time dropped) becomes a run caveat."""
    lossy: list[str] = []
    dropped: list[str] = []
    started = _started_at(found.started, started_is_utc)
    config = _json_safe(found.config or {}, "config", lossy)
    corpus = _json_safe(found.corpus, "corpus", lossy)
    metrics = _metrics(found, dropped)
    run_caveats = [*run_caveats, *_lossy_caveats(found, lossy, dropped, started)]
    return schema.from_dict(
        {
            "schema": schema.SCHEMA,
            "run_id": found.name,
            "family": found.family or None,
            "arm": None,
            # Terminal status is not observable from most legacy artifacts:
            # null says so (core spec, Producer finding 4.1, decided at the v1
            # freeze); a reader that keeps only finished runs says `complete`.
            # The ledger's own word (recorded/spec) is in ext.
            "status": "complete" if observes_completion else None,
            "started_at": started,
            "finished_at": None,
            "producer": {"name": f"attestation {adapter}-import", "version": _version()},
            "code": dict(_UNKNOWN_CODE),
            "config": config,
            "data": [],
            "metrics": metrics,
            "history": None,
            "seeds": None,
            "dependencies": None,
            "ext": {
                IMPORT_NAMESPACE: {
                    "adapter": found.adapter or adapter,
                    "status": found.status,
                    "notes": found.notes,
                    "corpus": corpus,
                    "caveats": caveats,
                    "run_caveats": run_caveats,
                }
            },
        }
    )


@dataclass(frozen=True)
class TrackerAdapter:
    """One legacy reader behind the `Adapter` protocol.

    `reader(root, taken)` is the unchanged function from `generic`; `marker`
    is the cheap `detect`. `notes_are_caveats` marks a reader whose `notes`
    only ever say what is wrong with that run (DVC's lock-hash mismatch),
    which become the run's own caveats."""

    name: str
    reader: Callable[[Path, set[str]], list[Found]]
    marker: Callable[[Path], bool]
    caveat: str | None = None
    started_is_utc: bool = False
    notes_are_caveats: bool = False
    # The reader keeps only runs its tracker marked finished (Sacred's
    # `COMPLETED`), so it observed completion; every other reader did not.
    observes_completion: bool = False

    def detect(self, path: Path) -> bool:
        """The tracker's marker (its directory or file) is present."""
        return self.marker(Path(path))

    def scan(self, path: Path, taken: set[str] | None = None) -> list[Found]:
        """The unchanged legacy reader, sharing `taken` as its `seen`."""
        return self.reader(Path(path), set() if taken is None else taken)

    def to_core_record(self, found: Found) -> schema.RunRecord:
        """`found_to_core` with this adapter's caveats and start-time rule."""
        run_caveats = [found.notes] if self.notes_are_caveats and found.notes else []
        return found_to_core(
            found,
            adapter=self.name,
            caveats=[self.caveat] if self.caveat else [],
            run_caveats=run_caveats,
            started_is_utc=self.started_is_utc,
            observes_completion=self.observes_completion,
        )


def _has_dir(name: str) -> Callable[[Path], bool]:
    return lambda root: (root / name).is_dir()


GENERIC = TrackerAdapter(generic.ADAPTER_NAME, generic.plain_runs, generic.has_plain_layout)
WANDB = TrackerAdapter(
    "wandb", generic._wandb_runs, _has_dir("wandb"), WANDB_CAVEAT, started_is_utc=True
)
MLFLOW = TrackerAdapter("mlflow", generic._mlflow_runs, _has_dir("mlruns"), MLFLOW_CAVEAT)
SACRED = TrackerAdapter(
    "sacred",
    generic._sacred_runs,
    _has_dir("sacred_runs"),
    SACRED_CAVEAT,
    started_is_utc=True,
    observes_completion=True,
)
DVC = TrackerAdapter(
    "dvc",
    generic._dvc_runs,
    lambda root: (root / "dvc.yaml").is_file(),
    DVC_CAVEAT,
    notes_are_caveats=True,
)
HYDRA = TrackerAdapter("hydra", generic._hydra_runs, _has_dir("multirun"), HYDRA_CAVEAT)

# The old `generic.discover` order: plain results and configs first (a
# measurement outranks a spec of the same name), then each tracker.
LEGACY_ADAPTERS: tuple[TrackerAdapter, ...] = (GENERIC, WANDB, MLFLOW, SACRED, DVC, HYDRA)
