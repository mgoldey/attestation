"""The legacy readers as adapters: each turns what it found into a core record.

Every reader in `generic` -- the plain results/configs conventions and the
five tracker layouts -- is wrapped here as one `TrackerAdapter` behind the
`ledger_adapters.Adapter` protocol. The readers themselves are unchanged
(spec: docs/superpowers/specs/2026-11-19-ledger-reads-core-records-design.md);
what is new is `found_to_core`, the one place a `Found` becomes an
`attestation_schema.RunRecord`, and the caveats, which moved here from
`ledger.py` and now travel inside each record as data.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import attestation_schema as schema

from attestation import replicates
from attestation.ledger_adapters import generic, hydra_store, mlflow_store, wandb_store
from attestation.ledger_adapters.found import IMPORT_NAMESPACE, Found
from attestation.ledger_adapters.found import core_metrics as _metrics
from attestation.ledger_adapters.found import json_safe as _json_safe

# What each reader cannot do, stated where a reader of the run will see it.
# Each says what its reader cannot do, where a reader of the run will see
# it: a tracker-derived run looked identical in `runs.list` and `runs.detail`
# to a hand-written JSON one, so the reader had no way to know the number in
# front of them is a FINAL value rather than the best step. The generic
# reader has none: a caveat on every run is boilerplate, and boilerplate
# trains a reader to skip the ones that matter.
WANDB_CAVEAT = (
    "read by the wandb adapter: each metric is the run's final summary value (by"
    " default the last value logged), not the best step; the curve is in the"
    " run's core record (history), not in the ledger. Read from the run's"
    " .wandb log; a run without one is read from files/ and has no status or"
    " curve"
)
MLFLOW_CAVEAT = (
    "read by the mlflow adapter: each metric is MLflow's own latest value"
    " (greatest step, then latest timestamp), not the best step; the full"
    " curve is in the run's core record (history), not in the ledger"
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
    " every other tracker here, not a curve or a best step; Hydra records"
    " no outcome and no timing, so status and wall time are not recorded"
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


def replicate_of(name: str, *, run_id_suffix: bool = False) -> tuple[str, int] | None:
    """`(arm, seed)` for a legacy run named `<configuration>_seed<N>`, else None.

    The legacy artifacts carry no arm and no seed field; the run NAME is the
    only place a person said "this is seed N of configuration X" (the naming
    the attestation-record skill teaches), so this is where that is read --
    once, at import -- and written into the core record's own `arm` and
    `seeds`. From there on the ledger pools by the record's fields
    (`replicates.group`), the same rule a native record with an arm and
    seeds gets, and never re-parses a name.

    `run_id_suffix` is set by the readers that append their tracker's run id
    to a run's name (`<run_name>/<id>`): that id is unique per run, so it is
    dropped before the seed is looked for -- left in, every seed of
    `clf_seed<N>/<id>` became its own arm and nothing pooled. A name with no
    single seed token, or nothing left once it is removed (a bare `seed1`
    names no configuration), has no arm.
    """
    head = name
    if run_id_suffix and "/" in name:
        head = name.rsplit("/", 1)[0]
    parsed = replicates.strip_seed(head)
    if parsed is None or not parsed[0]:
        return None
    return parsed


def found_to_core(
    found: Found,
    *,
    adapter: str,
    caveats: list[str],
    run_caveats: list[str],
    started_is_utc: bool = False,
    observes_completion: bool = False,
    run_id_suffix: bool = False,
) -> schema.RunRecord:
    """The core record for one legacy run, validated by `from_dict` (the
    reader's own rules), so an import is held to everything a written
    record is. Raises `attestation_schema.RecordError` for a run the core
    cannot express. What the core has no field for -- the ledger's own
    status word, notes, the corpus an artifact named, the caveats -- goes in
    `ext.ledger-import`. A lossy step (a duplicate dropped, a NaN config value
    stringified, a start time dropped) becomes a run caveat.

    `arm` and `seeds` are set only for a run named as a seed replicate
    (`replicate_of`): `arm` the configuration's name, `seeds` `{"seed": N}`.
    Every other legacy run keeps both null -- "did not say"."""
    replicate = replicate_of(found.name, run_id_suffix=run_id_suffix)
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
            "arm": replicate[0] if replicate else None,
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
            "seeds": {"seed": replicate[1]} if replicate else None,
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
    reader: Callable[[Path, set[str]], Sequence[Found]]
    marker: Callable[[Path], bool]
    caveat: str | None = None
    started_is_utc: bool = False
    notes_are_caveats: bool = False
    # The reader keeps only runs its tracker marked finished (Sacred's
    # `COMPLETED`), so it observed completion; every other reader did not.
    observes_completion: bool = False
    # The reader names a run `<label>/<its tracker's run id>` (see
    # `replicate_of`): the id never takes part in pooling seeds.
    run_id_suffix: bool = False

    def detect(self, path: Path) -> bool:
        """The tracker's marker (its directory or file) is present."""
        return self.marker(Path(path))

    def scan(self, path: Path, taken: set[str] | None = None) -> list[Found]:
        """The unchanged legacy reader, sharing `taken` as its `seen`."""
        return list(self.reader(Path(path), set() if taken is None else taken))

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
            run_id_suffix=self.run_id_suffix,
        )


def _has_dir(name: str) -> Callable[[Path], bool]:
    return lambda root: (root / name).is_dir()


GENERIC = TrackerAdapter(generic.ADAPTER_NAME, generic.plain_runs, generic.has_plain_layout)


@dataclass(frozen=True)
class StoreAdapter(TrackerAdapter):
    """A tracker whose reader maps the whole run itself (`*_store`: status,
    curves, code, outputs), so its record is not built by `found_to_core`;
    name, marker, registry order and the run-id suffix are a
    `TrackerAdapter`'s. `store` is the module with `to_core_record`, `kind`
    the `Found` subclass its reader returns. Specs:
    2026-11-20-mlflow-adapter-design.md, 2026-11-20-wandb-hydra-adapters-design.md."""

    store: Any = None
    kind: type = Found

    def to_core_record(self, found: Found) -> schema.RunRecord:
        """The store's `to_core_record`, with this adapter's caveat and the
        ledger-wide `<config>_seed<N>` arm (`replicate_of`)."""
        if not isinstance(found, self.kind):
            kind = type(found).__name__
            raise TypeError(f"the {self.name} adapter maps its own runs, not {kind}")
        return self.store.to_core_record(
            found,
            caveats=[self.caveat] if self.caveat else [],
            replicate=replicate_of(found.name, run_id_suffix=self.run_id_suffix),
        )


WANDB = StoreAdapter(
    "wandb",
    wandb_store.wandb_runs,
    _has_dir(wandb_store.BASE_DIR),
    WANDB_CAVEAT,
    run_id_suffix=True,
    store=wandb_store,
    kind=wandb_store.WandbFound,
)


MLFLOW = StoreAdapter(
    "mlflow",
    mlflow_store.mlflow_runs,
    _has_dir(mlflow_store.STORE_DIR),
    MLFLOW_CAVEAT,
    run_id_suffix=True,
    store=mlflow_store,
    kind=mlflow_store.MlflowFound,
)
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
HYDRA = StoreAdapter(
    "hydra",
    hydra_store.hydra_runs,
    _has_dir(hydra_store.BASE_DIR),
    HYDRA_CAVEAT,
    store=hydra_store,
    kind=hydra_store.HydraFound,
)

# The old `generic.discover` order: plain results and configs first (a
# measurement outranks a spec of the same name), then each tracker.
LEGACY_ADAPTERS: tuple[TrackerAdapter, ...] = (GENERIC, WANDB, MLFLOW, SACRED, DVC, HYDRA)
