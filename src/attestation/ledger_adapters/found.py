"""What an importer of a legacy artifact found, before it is a core record.

Formerly `ledger.RunRecord` / `ledger.Metric`. They were never the ledger's
shape -- they were the readers' -- and since the ledger now reads only
`attestation_schema.RunRecord`, they live with the readers that build them.
`trackers.found_to_core` is the one place a `Found` becomes a core record.
"""

import math
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import attestation_schema as schema


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


def json_safe(value: Any, path: str = "", lossy: list[str] | None = None) -> Any:
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
            str(k): json_safe(v, f"{path}.{k}" if path else str(k), lossy) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(v, f"{path}[{i}]", lossy) for i, v in enumerate(value)]
    return value


def core_metrics(found: Found, dropped: list[str] | None = None) -> list[dict]:
    """The core metric list. Duplicates on (name, step, split) keep the first,
    which is what the ledger's insert loop always did (each dropped one is
    appended to `dropped`); direction is never inferred, a legacy reader knows
    no label, and a unit only where the reader set one (wall time,
    `timing.add_wall`)."""
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
                "unit": m.unit,
                "labels": {},
            }
        )
    return out


def history_ref(series: Iterable[schema.Series]) -> dict | None:
    """The reference the record carries for these curves, computed by the
    schema's own writer (in a directory thrown away after): the sidecar's
    bytes are a pure function of what the tracker logged, so the digest names
    bytes anyone can regenerate. The ledger stores no curve (MLflow adapter
    spec, "Curves"); every adapter that reads a curve shares this."""
    with tempfile.TemporaryDirectory() as tmp:
        ref = schema.write_history(tmp, series)
    if ref is None:
        return None
    return {"path": ref.path, "sha256": ref.sha256, "series": ref.series, "points": ref.points}
