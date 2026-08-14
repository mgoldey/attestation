"""A ledger of experiment runs, read from artifacts already on disk.

Deliberately NOT an experiment tracker. It does not wrap training, does not ask
for `log_metric()` calls, and requires no change to how anything is run. Most
research projects already leave structured artifacts behind -- config files,
per-eval JSON dumps, benchmark results -- and the numbers that end up in a
README were transcribed from them by hand. This reads those artifacts so the
numbers become derivable, and re-derivable when they change.

Adoption cost is the design constraint. A tool that requires new discipline
gets used for a week; one that reads what is already there keeps working while
you forget it exists.

Two rules keep it honest:

**Record what is unambiguous, refuse to guess the rest.** A config file is a
specification with no result attached, so it is stored with no metrics rather
than an invented one. An unrecognised file shape yields no run rather than a
wrong one.

**Never rank a metric whose direction is undeclared.** WER 0.043 -> 0.053 is a
regression; accuracy 0.90 -> 0.94 is an improvement. A comparison that guesses
will confidently order an ablation backwards, which is worse than refusing.
"""

import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


def workspace_root(explicit: str | None = None) -> Path | None:
    """Where the projects live. Explicit argument, then RESEARCH_ROOT, else None.

    Deliberately no default. Guessing a directory would either scan something
    the user did not mean or silently find nothing; returning None lets the
    caller say "set RESEARCH_ROOT" instead of reporting an empty success.
    """
    value = explicit or os.environ.get("RESEARCH_ROOT")
    return Path(value).expanduser() if value else None


# Which direction is "better" for a metric. Ranking without this is worse than
# useless: WER 0.0433 -> 0.0527 is a regression, accuracy 0.90 -> 0.94 is an
# improvement, and a comparison tool that guesses will confidently rank
# ablation arms backwards. Adapters extend this; `compare` refuses to rank on a
# metric absent from it rather than assuming.
METRIC_DIRECTION: dict[str, str] = {
    "wer": "lower_is_better",
    "cer": "lower_is_better",
    "loss": "lower_is_better",
    "val_loss": "lower_is_better",
    "mae": "lower_is_better",
    "rmse": "lower_is_better",
    "error": "lower_is_better",
    "accuracy": "higher_is_better",
    "r_squared": "higher_is_better",
    "f1": "higher_is_better",
}


@dataclass
class Metric:
    metric: str
    value: float
    step: int | None = None
    split: str | None = None


@dataclass
class RunRecord:
    """One experiment run, as discovered on disk."""

    project: str
    name: str
    source_path: str
    family: str | None = None
    status: str = "unknown"
    started: str | None = None
    config: dict | None = None
    notes: str | None = None
    metrics: list[Metric] = field(default_factory=list)


def scan(conn: sqlite3.Connection, root: Path, project: str | None = None) -> dict:
    """Read artifacts under a workspace `root` into the ledger. Idempotent.

    Every subdirectory of `root` is treated as a project and read by the
    convention-based adapter -- no project needs to be known in advance. A
    directory yielding no recognisable runs is reported in `empty` rather than
    silently omitted, so "found nothing" is never mistaken for "nothing there".

    Replaces a project's rows wholesale rather than merging: the artifacts on
    disk are the source of truth, so a run that vanished there vanishes here.
    """
    from attestation.ledger_adapters import adapter_for

    root = Path(root).expanduser()
    if not root.is_dir():
        return {"scanned": {}, "empty": [], "message": f"no such directory: {root}"}

    if project:
        candidates = [root / project]
    else:
        candidates = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))

    scanned: dict[str, int] = {}
    empty: list[str] = []
    for project_root in candidates:
        if not project_root.is_dir():
            empty.append(f"{project_root.name} (no such directory)")
            continue
        records = adapter_for(project_root.name).discover(project_root)
        if not records:
            empty.append(project_root.name)
            continue
        _replace_project(conn, project_root.name, records)
        scanned[project_root.name] = len(records)

    conn.commit()
    return {"scanned": scanned, "empty": empty}


def _replace_project(conn: sqlite3.Connection, project: str, records: list[RunRecord]) -> None:
    conn.execute(
        "DELETE FROM run_metrics WHERE run_id IN (SELECT id FROM runs WHERE project = ?)",
        (project,),
    )
    conn.execute("DELETE FROM runs WHERE project = ?", (project,))
    for r in records:
        cur = conn.execute(
            "INSERT INTO runs(project, name, family, status, started, source_path,"
            " config_json, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                r.project,
                r.name,
                r.family,
                r.status,
                r.started,
                r.source_path,
                json.dumps(r.config) if r.config is not None else None,
                r.notes,
            ),
        )
        run_id = cur.lastrowid
        seen: set[tuple] = set()
        for m in r.metrics:
            key = (m.metric, m.step, m.split)
            if key in seen:  # last write wins; the PK would reject a duplicate
                continue
            seen.add(key)
            conn.execute(
                "INSERT INTO run_metrics(run_id, metric, value, step, split)"
                " VALUES (?, ?, ?, ?, ?)",
                (run_id, m.metric, m.value, m.step, m.split),
            )


def list_runs(
    conn: sqlite3.Connection,
    project: str | None = None,
    family: str | None = None,
    limit: int = 20,
) -> list[dict]:
    sql = "SELECT id, project, name, family, status, started, source_path FROM runs"
    clauses, params = [], []
    if project:
        clauses.append("project = ?")
        params.append(project)
    if family:
        clauses.append("family = ?")
        params.append(family)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY project, family, name LIMIT ?"
    params.append(max(1, int(limit)))
    return [dict(r) for r in conn.execute(sql, params)]


def detail(conn: sqlite3.Connection, project: str, name: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM runs WHERE project = ? AND name = ?", (project, name)
    ).fetchone()
    if row is None:
        return None
    out = dict(row)
    if out.get("config_json"):
        out["config"] = json.loads(out.pop("config_json"))
    else:
        out.pop("config_json", None)
        out["config"] = None
    out["metrics"] = [
        dict(m)
        for m in conn.execute(
            "SELECT metric, value, step, split FROM run_metrics WHERE run_id = ?"
            " ORDER BY metric, step",
            (row["id"],),
        )
    ]
    return out


def _best_step(values: list[dict], direction: str) -> dict | None:
    """The run's best value for a metric, not its last -- a training run that
    diverges late should not be judged by where it ended up."""
    if not values:
        return None
    pick = min if direction == "lower_is_better" else max
    return pick(values, key=lambda v: v["value"])


def compare(conn: sqlite3.Connection, family: str, metric: str | None = None) -> dict:
    """Rank every arm of an ablation family by a metric.

    The question this exists for: a sweep of N named config variants is a
    designed experiment, and which arm won usually lives only in filenames and
    memory. Arms with no value for the metric are listed in `without_metric`
    rather than dropped -- an arm that was never evaluated is a finding.
    """
    runs = [
        dict(r)
        for r in conn.execute(
            "SELECT id, project, name, status, source_path FROM runs WHERE family = ?"
            " ORDER BY name",
            (family,),
        )
    ]
    if not runs:
        return {"family": family, "metric": metric, "arms": [], "winner": None}

    if metric is None:
        counts: dict[str, int] = {}
        for r in runs:
            for m in conn.execute(
                "SELECT DISTINCT metric FROM run_metrics WHERE run_id = ?", (r["id"],)
            ):
                counts[m["metric"]] = counts.get(m["metric"], 0) + 1
        # the metric the most arms share, so the comparison covers the family
        known = {k: v for k, v in counts.items() if k in METRIC_DIRECTION}
        if not known:
            raise ValueError(
                f"no metric with a known direction in family {family!r};"
                f" found {sorted(counts) or 'none'}"
            )
        metric = sorted(known.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    direction = METRIC_DIRECTION.get(metric)
    if direction is None:
        raise ValueError(
            f"unknown direction for metric {metric!r} -- refusing to rank."
            " Declare it in ledger.METRIC_DIRECTION; guessing would rank"
            " ablation arms backwards."
        )

    arms = []
    for r in runs:
        values = [
            dict(v)
            for v in conn.execute(
                "SELECT value, step, split FROM run_metrics WHERE run_id = ? AND metric = ?",
                (r["id"], metric),
            )
        ]
        best = _best_step(values, direction)
        n_row = conn.execute(
            "SELECT value FROM run_metrics WHERE run_id = ? AND metric = 'n_records'"
            " ORDER BY value DESC LIMIT 1",
            (r["id"],),
        ).fetchone()
        arms.append(
            {
                "name": r["name"],
                "status": r["status"],
                "value": best["value"] if best else None,
                "step": best["step"] if best else None,
                # provenance: every number must be traceable to the file it came
                # from, or the comparison cannot be audited
                "source_path": r["source_path"],
                "n": int(n_row["value"]) if n_row else None,
            }
        )

    scored = [a for a in arms if a["value"] is not None]

    def rank_key(arm: dict) -> tuple[float, str]:
        # Bind through a local float so the None-ness is resolved once, where
        # the `scored` filter above already guarantees it. Negating inside the
        # lambda read as `-None` to a type checker, and it was right to ask.
        value = float(arm["value"])
        return (value if direction == "lower_is_better" else -value, arm["name"])

    scored.sort(key=rank_key)
    missing = sorted((a for a in arms if a["value"] is None), key=lambda a: a["name"])

    return {
        "family": family,
        "metric": metric,
        "direction": direction,
        "arms": scored + missing,
        "winner": scored[0]["name"] if scored else None,
        "without_metric": [a["name"] for a in missing],
        "caveats": _caveats(scored, metric),
    }


# Below this many samples, a difference between arms is not worth reading as a
# result. Not a significance test -- it is a prompt to go look, which is what an
# auditor needs when the tool cannot know the variance.
SMALL_SAMPLE = 30


def _caveats(scored: list[dict], metric: str) -> list[str]:
    """Reasons to distrust the ranking, stated with the ranking.

    A comparison that prints four decimal places and names a winner implies a
    confidence it has not earned. The tool cannot run a significance test (it
    has aggregates, not per-sample values, for most shapes), so it says plainly
    what it does not know instead of implying it does.
    """
    out: list[str] = []
    if len(scored) < 2:
        return out

    sizes = [a["n"] for a in scored if a["n"] is not None]
    if sizes and max(sizes) < SMALL_SAMPLE:
        out.append(
            f"every arm has n < {SMALL_SAMPLE} (largest {max(sizes)});"
            f" differences in {metric} at this size are likely noise"
        )
    if sizes and len(set(sizes)) > 1:
        out.append(f"arms were evaluated on different sample sizes ({min(sizes)}-{max(sizes)})")
    if any(a["n"] is None for a in scored):
        out.append("some arms report no sample size, so their weight is unknown")

    top, second = scored[0], scored[1]
    if top["value"] is not None and second["value"] is not None:
        spread = abs(second["value"] - top["value"])
        scale = max(abs(top["value"]), 1e-12)
        if spread / scale < 0.05:
            out.append(
                f"the top two arms differ by {spread:.4g} ({100 * spread / scale:.1f}%)"
                " -- too close to call from these numbers alone"
            )
    steps = {a["step"] for a in scored if a["step"] is not None}
    if len(steps) > 1:
        out.append(
            "arms are at different training steps, so this compares checkpoints,"
            " not just configurations"
        )
    return out


def families(conn: sqlite3.Connection, project: str | None = None) -> list[dict]:
    sql = (
        "SELECT project, family, COUNT(*) n FROM runs WHERE family IS NOT NULL"
        + (" AND project = ?" if project else "")
        + " GROUP BY project, family ORDER BY n DESC, family"
    )
    return [dict(r) for r in conn.execute(sql, (project,) if project else ())]
