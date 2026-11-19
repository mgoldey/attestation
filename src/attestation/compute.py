"""Wall time per arm, and a warning when a comparison is not compute-matched.

WALL TIME IS NOT COMPUTE. What this module handles is the seconds between a run's
own start and stop timestamps on whatever machine ran it: it includes queueing,
data loading and a cold cache, differs between a laptop and an A100, and says
nothing about FLOPs, energy, memory or cost. The one claim made from it is that
two arms took visibly different time, so they may not have had the same budget --
a prompt to check, never a measure, and never an adjustment to a ranking.

Pure and stdlib-only except `walls_by_run`, the one read of the ledger (it takes
an open connection and imports nothing for it). `ledger._compare` calls
`attach` and `caveats`; the tracker readers call the duration parsers. See
docs/superpowers/specs/2026-11-19-compute-tracking-design.md.
"""

from datetime import datetime

from attestation import replicates

# A duration is stored as an ordinary run_metrics row under this name -- the
# name an explicit key in a JSON/CSV result already gets -- so there is no
# column and no migration. `runtime_s` is accepted as a synonym; nothing else
# (`elapsed`, `duration`, `_timestamp`) is, because its unit would be a guess.
WALL_METRIC = "wall_time_s"
WALL_METRICS = (WALL_METRIC, "runtime_s")

# The longer of the top two arms must take at least this many times the shorter.
COMPUTE_RATIO_WARN = 2.0
# ...and must itself have run this long. A 90 ms run beside a 40 ms run is
# scheduler jitter, not a budget difference (the committed tracker fixtures run
# for 24-155 ms), and a warning there would be a false alarm.
MIN_WALL_S = 1.0
# Arm names listed in one caveat; the rest are counted.
MAX_NAMES = 5

NOT_COMPUTE = (
    "Wall time is not compute -- hardware, load and data loading differ -- so treat this as"
    " a prompt to check the budgets, not a measure of cost"
)


def seconds(value) -> float | None:
    """`value` as a non-negative finite number of seconds, else None. Strings
    are accepted (a YAML reader returns them); booleans and NaN are not."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if 0.0 <= out < float("inf") else None


def iso_duration(start, stop) -> float | None:
    """Seconds from one ISO-8601 timestamp to another (sacred's `start_time` and
    `stop_time`), or None when either is absent, unparseable or the stop is
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


def run_wall(rows: list[dict]) -> float | None:
    """One run's wall time from its recorded rows (`metric`, `value`), or None.

    `wall_time_s` beats `runtime_s`. A run with several rows of the one name (a
    log with one per step) takes the LARGEST: the series is cumulative and the
    total is its end.
    """
    for name in WALL_METRICS:
        found = [seconds(r["value"]) for r in rows if r["metric"] == name]
        found = [v for v in found if v is not None]
        if found:
            return max(found)
    return None


def walls_by_run(conn, run_ids: list[int]) -> dict[int, float]:
    """`{run_id: wall seconds}` for the runs that recorded one. The only ledger
    read here: one query for every run, whatever the family's size."""
    if not run_ids:
        return {}
    marks = ",".join("?" * len(run_ids))
    names = ",".join("?" * len(WALL_METRICS))
    rows: dict[int, list[dict]] = {}
    for r in conn.execute(
        f"SELECT run_id, metric, value FROM run_metrics WHERE run_id IN ({marks})"
        f" AND metric IN ({names})",
        (*run_ids, *WALL_METRICS),
    ):
        rows.setdefault(r["run_id"], []).append({"metric": r["metric"], "value": r["value"]})
    walls = {rid: run_wall(rs) for rid, rs in rows.items()}
    return {rid: w for rid, w in walls.items() if w is not None}


def run_walls(runs: list[dict], walls: dict[int, float] | None) -> dict[str, float]:
    """`{run name: wall seconds}`. A nested arm is named `<run>[<split>]`, so it
    finds nothing here: the file's duration belongs to the file, and giving it to
    every arm inside would multiply it."""
    walls = walls or {}
    return {r["name"]: walls[r["id"]] for r in runs if r["id"] in walls}


def _members(arm: dict, raw_arms: list[dict]) -> list[dict]:
    """The pre-pooling arms a replicate arm was built from: the valued arms
    whose name is `arm`'s base once the seed token is removed."""
    out = []
    for raw in raw_arms:
        parsed = replicates.seed_of(raw["name"]) if raw.get("value") is not None else None
        if parsed is not None and parsed[0] == arm["name"]:
            out.append(raw)
    return out


def attach(arms: list[dict], raw_arms: list[dict], walls: dict[str, float]) -> None:
    """Set `wall_time_s` on each arm whose time is fully known, in place.

    A single run takes its own. A pooled arm takes the SUM over every run it
    pooled -- found through `raw_arms`, since its `runs` list is capped at five
    names -- plus `wall_time_mean_s`. If any member has no time the arm gets none:
    a partial sum would read as the arm's total. Arms with no time gain no key,
    so a family without timings keeps exactly the payload it had.
    """
    for arm in arms:
        if arm.get("n_runs"):
            members = _members(arm, raw_arms)
            times = [walls.get(m["name"]) for m in members]
            if len(members) != arm["n_runs"] or None in times:
                continue
            total = sum(t for t in times if t is not None)
            arm["wall_time_s"] = total
            arm["wall_time_mean_s"] = total / arm["n_runs"]
        elif arm["name"] in walls:
            arm["wall_time_s"] = walls[arm["name"]]


def per_run(arm: dict) -> float | None:
    """Wall seconds for ONE run of the arm: the mean for a pooled arm, the time
    itself for a single run. Totals are not comparable across arms with
    different replicate counts; per-run time is."""
    return arm.get("wall_time_mean_s", arm.get("wall_time_s"))


def compute_ratio(arms: list[dict]) -> float | None:
    """Longer over shorter per-run wall time of the first two arms, or None when
    fewer than two or either has no time. Infinite when the shorter is zero."""
    if len(arms) < 2:
        return None
    a, b = per_run(arms[0]), per_run(arms[1])
    if a is None or b is None:
        return None
    hi, lo = max(a, b), min(a, b)
    return hi / lo if lo > 0 else float("inf")


def _fmt(value: float | None) -> str:
    """Seconds for a caveat: four significant digits."""
    return f"{value:.4g}s" if value is not None else "?"


def _listed(names: list[str]) -> str:
    """Names for a caveat, capped at MAX_NAMES with the remainder counted."""
    extra = len(names) - MAX_NAMES
    shown = ", ".join(names[:MAX_NAMES])
    return f"{shown} and {extra} more" if extra > 0 else shown


def _ratio_caveat(top2: list[dict]) -> list[str]:
    """The top-two ratio caveat, or `[]` below the threshold or the floor."""
    ratio = compute_ratio(top2)
    first, second = top2[0], top2[1]
    longest = max(per_run(first) or 0.0, per_run(second) or 0.0)
    if ratio is None or ratio < COMPUTE_RATIO_WARN or longest < MIN_WALL_S:
        return []
    return [
        f"the top two arms did not take the same wall time per run: {first['name']}"
        f" {_fmt(per_run(first))} vs {second['name']} {_fmt(per_run(second))}"
        f" ({ratio:.1f}x); this may not be a compute-matched comparison. {NOT_COMPUTE}"
    ]


def caveats(scored: list[dict]) -> list[str]:
    """Compute caveats for a ranked list; `[]` when no arm has a time, so a
    family that predates this feature renders exactly as it did.

    Arms with no time are named only once some arm has one (otherwise it is
    every family's state and would be noise). The ratio is the top two arms'
    per-run time, and is silent below COMPUTE_RATIO_WARN or when the longer arm
    ran under MIN_WALL_S.
    """
    if not any(per_run(a) is not None for a in scored):
        return []
    out = _ratio_caveat(scored[:2]) if len(scored) > 1 else []
    missing = sorted(a["name"] for a in scored if per_run(a) is None)
    if missing:
        out.append(
            f"compute not recorded for {_listed(missing)} while other arms have a wall time,"
            " so whether these arms were compute-matched cannot be checked"
        )
    return out
