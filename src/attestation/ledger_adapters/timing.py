"""A run's wall time, read from each tracker's own timestamps.

The per-tracker half of compute tracking (the ledger half is `compute.py`,
which names no tracker -- tests/test_tracker_names.py). Each reader that sees
a duration emits it as an ordinary core metric: `wall_time_s`, unit `"s"`,
`direction` null. A metric rather than an `ext` field because the core record
already has the shape -- a named value with a unit -- and a native producer
that times its own runs writes the very same metric, so `compute` reads one
name whichever adapter delivered the run. Never a direction: shorter is not
"better" in a ranking, and nothing here ranks by it.

WALL TIME IS NOT COMPUTE: see compute.py. Spec:
docs/superpowers/specs/2026-11-19-compute-tracking-design.md.
"""

from datetime import datetime

from attestation.compute import WALL_METRIC, seconds
from attestation.ledger_adapters.found import FoundMetric

WALL_UNIT = "s"


def iso_duration(start, stop) -> float | None:
    """Seconds from one ISO-8601 timestamp to another (Sacred's `start_time`
    and `stop_time`), or None when either is absent, unparseable or the stop is
    before the start. Mixing a naive and an aware timestamp is None, not a guess."""
    try:
        a, b = datetime.fromisoformat(str(start)), datetime.fromisoformat(str(stop))
        return seconds((b - a).total_seconds())
    except (TypeError, ValueError):
        return None


def epoch_ms_duration(start, end) -> float | None:
    """Seconds between two millisecond epoch timestamps (MLflow's `start_time`
    and `end_time`), or None when either is missing, zero or the end precedes the
    start. MLflow writes `end_time: 0` or omits it for a run still going."""
    a, b = seconds(start), seconds(end)
    if not a or not b:
        return None
    return seconds((b - a) / 1000.0)


def add_wall(metrics: list[FoundMetric], wall: float | None) -> None:
    """Append the run's wall time as `wall_time_s` (unit `s`) when the tracker
    recorded one -- and only then: nothing is inferred, and a metric the
    tracker itself named `wall_time_s` is never shadowed."""
    if wall is not None and all(m.metric != WALL_METRIC for m in metrics):
        metrics.append(FoundMetric(WALL_METRIC, wall, unit=WALL_UNIT))


__all__ = ["WALL_UNIT", "add_wall", "epoch_ms_duration", "iso_duration", "seconds"]
