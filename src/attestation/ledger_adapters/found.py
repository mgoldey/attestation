"""What an importer of a legacy artifact found, before it is a core record.

Formerly `ledger.RunRecord` / `ledger.Metric`. They were never the ledger's
shape -- they were the readers' -- and since the ledger now reads only
`attestation_schema.RunRecord`, they live with the readers that build them.
`trackers.found_to_core` is the one place a `Found` becomes a core record.
"""

from dataclasses import dataclass, field


@dataclass
class FoundMetric:
    """One final value read from a run's artifacts -- not a logged curve;
    see `generic`'s per-tracker conventions for what "final" means per
    reader. `step`/`split` disambiguate a metric a run recorded more than
    once (a checkpoint, a train vs. eval split)."""

    metric: str
    value: float
    step: int | None = None
    split: str | None = None
    # The core metric's `unit`. Only the readers that emit wall time set one
    # (`timing.add_wall`: "s"); every other value's unit is not on disk.
    unit: str | None = None


@dataclass
class Found:
    """One experiment run, as a legacy reader discovered it on disk."""

    project: str
    name: str
    source_path: str
    family: str | None = None
    status: str = "unknown"
    started: str | None = None
    config: dict | None = None
    notes: str | None = None
    metrics: list[FoundMetric] = field(default_factory=list)
    # What the artifact said about the data, if anything. None means "did not
    # say" -- never "no corpus" and never "the default corpus".
    corpus: dict | None = None
    # Which reader produced this run. Set by the reader, never inferred; it is
    # what lets each adapter attach its own caveats to exactly its runs.
    adapter: str | None = None


# The extension namespace every importer of a legacy artifact writes, and the
# ledger reads, for what the core record has no field for: the ledger's own
# status word, notes, the corpus an artifact named, which reader made it, and
# its caveats. ONE namespace for the importer (one producer), not one per
# tracker: `ext.mlflow` already belongs to the MLflow exporter in another
# shape. Not under `attestation.*`, which the schema reserves.
IMPORT_NAMESPACE = "ledger-import"
