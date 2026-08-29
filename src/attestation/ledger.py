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
import re
import sqlite3
import tomllib
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
# ablation arms backwards. `compare` refuses to rank on a metric absent from
# it rather than assuming. Extend it without editing this file by writing a
# `[metric_direction]` table to the TOML at METRIC_DIRECTION_PATH (see below)
# -- entries there are merged over these defaults.
METRIC_DIRECTION: dict[str, str] = {
    "wer": "lower_is_better",
    "cer": "lower_is_better",
    "loss": "lower_is_better",
    "val_loss": "lower_is_better",
    # Perplexity is exp(loss) and NLL is loss under another name: lower is
    # better by definition, not by convention. They are the metrics a language
    # model reports, and omitting them made the ledger refuse to compare the
    # runs it had just discovered.
    "ppl": "lower_is_better",
    "perplexity": "lower_is_better",
    "nll": "lower_is_better",
    "mae": "lower_is_better",
    "rmse": "lower_is_better",
    "error": "lower_is_better",
    "accuracy": "higher_is_better",
    "r_squared": "higher_is_better",
    "f1": "higher_is_better",
    # Widened after measuring the real corpus at ~/qc rather than the example
    # fixture: 17 of 44 multi-run families refused to rank, and the refusals
    # were not edge cases -- they were ordinary retrieval and classification
    # metrics missing from a 15-entry table. Every name here has ONE
    # direction by definition across every field that uses it; anything whose
    # direction depends on context (a "rate", a "count", a "gap") stays out and
    # is declared per-user, which is what the TOML override exists for.
    "auc": "higher_is_better",
    "roc_auc": "higher_is_better",
    "precision": "higher_is_better",
    "recall": "higher_is_better",
    "ndcg": "higher_is_better",
    "map": "higher_is_better",
    "mrr": "higher_is_better",
    "bleu": "higher_is_better",
    "rouge": "higher_is_better",
    "iou": "higher_is_better",
    "dice": "higher_is_better",
    "psnr": "higher_is_better",
    "ssim": "higher_is_better",
    "mse": "lower_is_better",
    "mape": "lower_is_better",
    "fid": "lower_is_better",
    "eer": "lower_is_better",
}

# Optional user-supplied TOML holding a `[metric_direction]` table, merged over
# METRIC_DIRECTION above. Explicit env var, then a per-user config file --
# never edit installed package source to teach the ledger a new metric.
METRIC_DIRECTION_PATH_ENV = "LEDGER_METRIC_DIRECTION_FILE"
_DEFAULT_METRIC_DIRECTION_PATH = Path.home() / ".hermes" / "metric_direction.toml"

# Split/phase affixes stripped from a metric name before the METRIC_DIRECTION
# lookup. The generic adapter extracts metric names verbatim from artifacts,
# where they are overwhelmingly prefixed or suffixed (`test_accuracy`,
# `top1_accuracy`, `f1_macro`) rather than bare. Stripping a *known* affix and
# looking up the *declared* direction for the stem is still a declaration, not
# a guess -- an undeclared stem still refuses.
# `best_`/`final_` say *which* value of a metric was taken, not what it
# measures, and they stack with a split prefix -- `best_val_loss` is the single
# most common metric name a training loop writes. Stripping only one affix left
# it undeclared, so a real repo's runs were discovered and then refused for
# comparison.
_METRIC_PREFIXES = ("train_", "val_", "valid_", "test_", "eval_", "top1_", "best_", "final_")
_METRIC_SUFFIXES = ("_macro", "_micro")
# `_at_10`, `_at_5`: the standard cutoff notation for the ranking metrics
# (ndcg, map, mrr, recall, precision) whose direction does not depend on k.
# Stripped by regex rather than a literal list because k is unbounded.
_METRIC_CUTOFF_RE = re.compile(r"_at_\d+$")


def _metric_stem(metric: str) -> str:
    """`metric` with known split/qualifier affixes removed.

    Loops until no affix matches, since they compose. This stays a declaration
    rather than a guess: an unknown stem is still refused, so `best_val_score`
    resolves to nothing exactly as `score` does.
    """
    stem = metric
    changed = True
    while changed:
        changed = False
        stripped = _METRIC_CUTOFF_RE.sub("", stem)
        if stripped != stem and stripped:
            stem = stripped
            changed = True
        for prefix in _METRIC_PREFIXES:
            if stem.startswith(prefix) and len(stem) > len(prefix):
                stem = stem[len(prefix) :]
                changed = True
                break
    for suffix in _METRIC_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _metric_direction_path() -> Path:
    value = os.environ.get(METRIC_DIRECTION_PATH_ENV)
    return Path(value).expanduser() if value else _DEFAULT_METRIC_DIRECTION_PATH


def metric_directions() -> dict[str, str]:
    """Built-in METRIC_DIRECTION, overlaid with a user's TOML file if present.

    Read fresh on every call (not cached at import time) so tests and a user
    editing the file are both respected, matching db.embed_dims()'s pattern.
    """
    path = _metric_direction_path()
    if not path.is_file():
        return dict(METRIC_DIRECTION)
    overrides = tomllib.loads(path.read_text()).get("metric_direction", {})
    return {**METRIC_DIRECTION, **overrides}


def _metric_direction(metric: str, directions: dict[str, str]) -> str | None:
    """Declared direction for `metric`, matching the stem when the exact name
    is absent. Still a declaration -- an unrecognised stem returns None."""
    direct = directions.get(metric)
    if direct is not None:
        return direct
    return directions.get(_metric_stem(metric))


@dataclass
class Metric:
    """One final value read from a run's artifacts -- not a logged curve;
    see the module docstring's per-tracker conventions for what "final" means
    per adapter. `step`/`split` disambiguate a metric a run recorded more
    than once (a checkpoint, a train vs. eval split)."""

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
    # What the artifact said about the data, if anything. None means "did not
    # say" -- never "no corpus" and never "the default corpus".
    corpus: dict | None = None
    # Resolved at scan time by _link_corpora; NULL until then.
    corpus_id: int | None = None
    # Which reader produced this record: "generic", "wandb", "mlflow", or a
    # named adapter. Set by the adapter, never inferred here. NULL/None means
    # "recorded before anything tracked this", which is not the same as
    # "generic" -- see ADAPTER_CAVEATS for why the distinction earns a column.
    adapter: str | None = None


def scan(conn: sqlite3.Connection, root: Path, project: str | None = None) -> dict:
    """Read artifacts under a workspace `root` into the ledger. Idempotent.

    Every subdirectory of `root` is treated as a project and read by the
    convention-based adapter -- no project needs to be known in advance. A
    directory yielding no recognisable runs is reported in `empty` rather than
    silently omitted, so "found nothing" is never mistaken for "nothing there".

    Replaces a project's rows wholesale rather than merging: the artifacts on
    disk are the source of truth, so a run that vanished there vanishes here.
    """
    from attestation import corpus
    from attestation.ledger_adapters import adapter_for

    root = Path(root).expanduser()
    if not root.is_dir():
        return {
            "scanned": {},
            "empty": [],
            "diagnostics": {},
            "message": f"no such directory: {root}",
        }

    fallback_roots: list[Path] = []
    if project:
        candidates = [root / project]
    else:
        candidates = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        # `root` may be a single repo rather than a directory of them. Its own
        # `results/` is then a project root of its own, and the adapter looks
        # for `results/results/` -- so pointing --root at a repo, which is the
        # obvious thing to do, reported "0 runs" and blamed the user's
        # directory names. Read the root itself too, as a fallback rather than
        # a peer: it is only reported when it yields runs, since a workspace of
        # projects would otherwise always list its own parent as empty.
        fallback_roots = [root]

    # Manifest is the highest-precedence source: it is the only one that can
    # state intent ("these arms were meant to share a corpus").
    manifest, assignments = corpus.load_manifest(root)

    scanned: dict[str, int] = {}
    empty: list[str] = []
    # Why each empty project was empty. "0 run(s)" with no reason is the one
    # failure this tool cannot afford -- an unrecognised-but-ordinary layout
    # looks identical to an empty workspace, and the user has no next step.
    diagnostics: dict[str, str] = {}
    # Directories whose files the root-level scan already claimed. Reporting
    # one of these as an empty project too would tell the reader to go fix a
    # layout that just worked.
    consumed: set[str] = set()
    # Fallback first, so `consumed` is known before the per-project pass.
    for project_root in [*fallback_roots, *candidates]:
        if not project_root.is_dir():
            empty.append(project_root.name)
            diagnostics[project_root.name] = "no such directory"
            continue
        adapter = adapter_for(project_root.name)
        records = adapter.discover(project_root)
        if not records:
            if project_root in fallback_roots:
                continue  # a fallback that found nothing is not a project
            if project_root.name in consumed:
                continue
            empty.append(project_root.name)
            explain = getattr(adapter, "diagnose_empty", None)
            diagnostics[project_root.name] = (
                explain(project_root) if explain else "no recognisable runs"
            )
            continue
        if project_root in fallback_roots:
            for record in records:
                try:
                    rel = Path(record.source_path).resolve().relative_to(project_root.resolve())
                except ValueError:
                    continue
                if len(rel.parts) > 1:
                    consumed.add(rel.parts[0])
        _link_corpora(conn, records, manifest, assignments)
        _replace_project(conn, project_root.name, records)
        # NO commit here. Corpora rows and the runs whose corpus_id references
        # them are written across this loop and committed once below, so a
        # per-project commit tears that write: see
        # test_a_scan_that_fails_partway_leaves_no_torn_write.
        scanned[project_root.name] = len(records)

    conn.commit()
    return {"scanned": scanned, "empty": empty, "diagnostics": diagnostics}


def _link_corpora(
    conn: sqlite3.Connection, records: list, manifest: dict, assignments: dict
) -> None:
    """Attach a corpus to each record, manifest first, then artifact fields.

    Precedence is declared rather than incidental: a human declaration is the
    only source that can say "these arms were meant to share a corpus", so it
    outranks whatever a result file happened to record.
    """
    from attestation import corpus

    by_family = assignments.get("family") or {}
    by_run = assignments.get("run") or {}
    for record in records:
        entry = None
        declared = by_run.get(record.name) or by_family.get(record.family or "")
        if declared and declared in manifest:
            entry = manifest[declared]
        elif record.corpus:
            entry = record.corpus
        if entry:
            record.corpus_id = corpus.upsert(conn, entry)


def _replace_project(conn: sqlite3.Connection, project: str, records: list[RunRecord]) -> None:
    conn.execute(
        "DELETE FROM run_metrics WHERE run_id IN (SELECT id FROM runs WHERE project = ?)",
        (project,),
    )
    conn.execute("DELETE FROM runs WHERE project = ?", (project,))
    for r in records:
        cur = conn.execute(
            "INSERT INTO runs(project, name, family, status, started, source_path,"
            " config_json, notes, corpus_id, adapter) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                r.project,
                r.name,
                r.family,
                r.status,
                r.started,
                r.source_path,
                json.dumps(r.config) if r.config is not None else None,
                r.notes,
                r.corpus_id,
                r.adapter,
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


def sample_runs(
    conn: sqlite3.Connection,
    project: str | None = None,
    family: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Runs to show for a listing: spread across projects when not filtered.

    `list_runs` orders by (project, family, name), so a LIMIT let the
    alphabetically-first project fill the whole answer -- the live ledger has
    18 projects and 858 runs, and a default listing returned 10, all
    `ablation`. The count said "10 of 858" and not that they were one project,
    so an honest count made a partial answer look complete.

    A filtered call is already narrowed by the caller and passes straight
    through.
    """
    if project or family:
        return list_runs(conn, project=project, family=family, limit=limit)

    # Rank within each project in SQL rather than paging a prefix into memory.
    # A pool taken with list_runs is ordered by project, so on the live ledger
    # a 320-row pool held 4 of 18 projects and the round-robin still returned
    # one -- a bigger pool is a fragile fix for the wrong shape.
    rows = conn.execute(
        """
        SELECT id, project, name, family, status, started, source_path
          FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY project ORDER BY family, name) AS rn
              FROM runs
          )
         ORDER BY rn, project
         LIMIT ?
        """,
        (limit,),
    )
    return [dict(r) for r in rows]


def count_runs(
    conn: sqlite3.Connection,
    project: str | None = None,
    family: str | None = None,
) -> int:
    """How many runs match, ignoring any limit.

    `list_runs` truncates and the caller could not tell: 212 of 222 runs
    vanished from a runs.list response whose message read "10 run(s)", while
    the families truncation in the same response was reported exactly.
    """
    sql = "SELECT COUNT(*) FROM runs"
    clauses, params = [], []
    if project:
        clauses.append("project = ?")
        params.append(project)
    if family:
        clauses.append("family = ?")
        params.append(family)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    return int(conn.execute(sql, params).fetchone()[0])


def list_runs(
    conn: sqlite3.Connection,
    project: str | None = None,
    family: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Runs in the ledger, optionally filtered by project/family. Ordered
    `project, family, name` -- grouped for display so related runs sit
    together, not `started DESC`, so this is not a "most recent runs" query;
    a caller wanting recency order does its own sort on the result."""
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


# What each reader cannot do, stated where a reader of the run will see it.
#
# These are not new facts: both were written down in
# `ledger_adapters/generic.py` when the readers were built, in source comments
# no user of the tool ever reads. A wandb-derived run looked identical in
# `runs.list` and `runs.detail` to a hand-written JSON one, so the reader had
# no way to know that the number in front of them is the FINAL logged value
# rather than the best step, or that an offline W&B run carries no metrics at
# all in files/ until it is synced (see ledger_adapters/generic.py's
# _wandb_runs docstring).
#
# Keyed by adapter name, so a reader with no caveats (the generic one) attaches
# none. A caveat on every run is boilerplate, and boilerplate is what teaches a
# reader to skip the caveats that matter.
ADAPTER_CAVEATS: dict[str, str] = {
    "wandb": (
        "read by the wandb adapter: values come from wandb-summary.json, which"
        " holds each metric's final logged value rather than its curve or its"
        " best step. Offline W&B does not write that file until a run is"
        " synced -- see examples/wandb/generate.py for how the committed"
        " fixture's was materialised without a network call"
    ),
    "mlflow": (
        "read by the mlflow adapter: each metric is the FINAL line of its"
        " metrics file, not the curve and not the best step"
    ),
    "sacred": (
        "read by the sacred adapter: each metric is the LAST value in"
        " metrics.json's series, not the curve and not the best step"
    ),
    "dvc": (
        "read by the dvc adapter: each metric file is a snapshot overwritten"
        " on every `dvc repro`, not a curve -- there is no history of a"
        " prior run's value once a stage reruns"
    ),
    "hydra": (
        "read by the hydra adapter: metrics come from whatever JSON/CSV file"
        " an arm's directory holds, read as a final value the same way as"
        " every other tracker here, not a curve or a best step"
    ),
}


def adapter_caveats(adapters) -> list[str]:
    """Caveats for the readers that produced a set of runs, deduplicated.

    Takes adapter names rather than rows so it is callable from anywhere and
    testable without a database. Unknown and NULL adapters contribute nothing:
    NULL means "recorded before this was tracked", and inventing a caveat for
    a reader we cannot name would be worse than saying nothing.
    """
    out: list[str] = []
    for adapter in sorted({a for a in adapters if a}):
        note = ADAPTER_CAVEATS.get(adapter)
        if note and note not in out:
            out.append(note)
    return out


def detail(conn: sqlite3.Connection, project: str, name: str) -> dict | None:
    """One run in full: its row, decoded config, every recorded metric, and
    the caveat (if any) for the adapter that read it -- `None` if no such
    run exists rather than an empty dict, so a caller can tell "not found"
    from "found, nothing to report"."""
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
    # Empty for an ordinary run; the caveat is attached only when the reader
    # that produced this one has a limitation the number does not show.
    out["caveats"] = adapter_caveats([out.get("adapter")])
    return out


# Splits a model is judged ON, in the order a reader would trust them. A run
# reporting several is judged by the first it has, never by whichever number
# happened to be best.
_EVAL_SPLITS = ("test", "eval", "val", "valid", "validation", "dev", "holdout")
_TRAIN_SPLITS = ("train", "training", "fit")


def _split_rank(split: str | None) -> int:
    """Lower sorts first. Unlabelled sits between eval and train: it is usually
    a headline number, but nothing says so, and it must not outrank an explicit
    test score."""
    if split is None:
        return len(_EVAL_SPLITS)
    lowered = split.lower()
    for i, name in enumerate(_EVAL_SPLITS):
        if lowered == name or lowered.startswith(name):
            return i
    if any(lowered.startswith(t) for t in _TRAIN_SPLITS):
        return len(_EVAL_SPLITS) + 1
    return len(_EVAL_SPLITS)


def collapse_to_last(metrics: list[dict]) -> list[dict]:
    """A step series collapsed to its last row per metric NAME, first-appearance
    order kept. Keyed on the name alone: on the live worst case `split` carried
    the sweep coordinate, so keying on (metric, split) collapsed nothing and 4
    of 33 names survived a row cap. Truncation is the caller's budget, not this
    function's."""
    last: dict[str, dict] = {}
    for row in metrics:
        last[row["metric"]] = row
    return list(last.values())


def nested_arms(values: list[dict]) -> dict[str, list[dict]]:
    """Group a run's metric rows by nested split key, when it has several.

    Real artifacts record the arms of an experiment INSIDE one file, under
    keys like `arms.Baseline` / `arms.Treatment_Eigen`, or lm-eval-harness's
    `results.mmlu_marketing`. The ledger's unit is the file, so all of those
    collapse into one run -- and `_split_rank` returns the same rank for every
    one of them, so `_best_step` takes the max across siblings.

    Measured on a real corpus: a run reporting Baseline 0.000, Control_RAG
    0.644, Oracle_Post 0.988 and Treatment_Eigen 0.655 scored 0.988. Every arm
    was credited with its own oracle upper bound, and on ~/nota's lm-eval
    output each model was ranked on its own easiest subtask, which reordered
    the bottom half of the table.

    Returns {} when there is nothing to fan out -- one nested group, or none.
    """
    groups: dict[str, list[dict]] = {}
    for value in values:
        split = value.get("split") or ""
        prefix, _, rest = split.partition(".")
        if not rest or _split_rank(split) != len(_EVAL_SPLITS):
            # Either not nested, or a recognised split like `test.clean` whose
            # rank already orders it. Only unrecognised nested keys are arms.
            return {}
        groups.setdefault(split, []).append(value)
    return groups if len(groups) > 1 else {}


def _best_step(values: list[dict], direction: str) -> dict | None:
    """The run's best value for a metric, not its last -- a training run that
    diverges late should not be judged by where it ended up.

    Best *within one split*, though. Picking the extreme across every split
    ranked an arm reporting train 0.01 / test 0.90 at 0.01, beating an arm
    whose test loss was 0.50 -- the ablation came out backwards, silently,
    which is the failure this module's docstring says it exists to prevent.
    So the most trustworthy split a run reports is chosen first, and the
    best step is taken only within it.
    """
    if not values:
        return None
    best_split = min(_split_rank(v.get("split")) for v in values)
    candidates = [v for v in values if _split_rank(v.get("split")) == best_split]
    pick = min if direction == "lower_is_better" else max
    return pick(candidates, key=lambda v: v["value"])


def _family_rows(conn: sqlite3.Connection, family: str, project: str | None) -> list[dict]:
    """The runs of one family, optionally narrowed to a project."""
    sql = (
        "SELECT r.id, r.project, r.name, r.status, r.source_path, r.adapter,"
        # c.source too: upsert records a CONTESTED marker there when one
        # corpus name carries two conflicting declarations, and a comparison
        # must not vouch for a corpus whose own definition is disputed.
        " c.name AS corpus, c.source AS corpus_source"
        " FROM runs r LEFT JOIN corpora c ON c.id = r.corpus_id"
        " WHERE r.family = ?"
    )
    params: tuple = (family,)
    if project:
        sql += " AND r.project = ?"
        params = (family, project)
    return [dict(r) for r in conn.execute(sql + " ORDER BY r.name", params)]


def _refuse_cross_project(family: str, rows: list[dict]) -> None:
    """Refuse rather than pick when a family name spans projects.

    Silently choosing one would be the same guess in a quieter voice, and
    ranking across them is what produced a named winner over English and
    Mandarin ASR arms pooled together.
    """
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["project"]] = counts.get(row["project"], 0) + 1
    if len(counts) > 1:
        # Ordered by arm count, not alphabetically. On a real corpus this
        # refusal listed 13 project names for one family and asked the reader
        # to pick, giving them nothing to pick ON -- and alphabetical order put
        # a dated backup directory second. Arm counts are the one signal
        # already in hand that distinguishes the main line of work from a
        # worktree that ran a subset.
        #
        # Collapsing git worktrees onto their main checkout was tried and
        # reverted: 58 of 62 run names in those directories COLLIDE, because
        # they are the same experiment re-run on different branches rather than
        # arms of a sweep. Merging them would have presented twelve copies of
        # one run as comparable arms, which is worse than refusing. The
        # refusal is right; it was only uninformative.
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        listed = ", ".join(f"{name} ({n})" for name, n in ranked)
        raise ValueError(
            f"family {family!r} exists in {len(counts)} projects, by arm count: "
            f"{listed} -- pass project= to say which. "
            "Arms from different projects are not comparable."
        )


def _likeliest_metric(counts: dict[str, int]) -> str | None:
    """A metric worth naming in the refusal, or None if nothing is safe to name.

    Only names whose SEGMENTS include one whose direction is already known:
    `test_acc` splits to {test, acc}, `consecutive_detections` shares nothing.
    Substring matching was worse -- it picked `nll_missing_rate` out of a real
    43-metric family because the name contains "nll", and that is a
    missing-data rate where "higher_is_better" would be exactly backwards.

    Even segment matching cannot fully separate `nll_missing_rate` from a loss,
    so the suggestion is deliberately hedged as "looks like a result metric"
    rather than asserted, and a name is offered only when exactly one candidate
    stands out. Three versions of this were measured against the real corpus at
    ~/qc rather than the example fixture; each earlier one confidently named a
    metric that would have ranked an ablation backwards or by dataset size.

    Returning None matters as much as returning a name: this refusal exists to
    stop confident wrong answers, so an unjustifiable suggestion is the one
    thing it must not emit.
    """
    if not counts:
        return None
    known = {k.lower() for k in METRIC_DIRECTION} | {"acc", "auc"}
    plausible = {
        name: n for name, n in counts.items() if known & set(re.split(r"[^a-z0-9]+", name.lower()))
    }
    if len(plausible) != 1:
        # Zero candidates, or several that name-shape cannot rank between.
        # Listing the family's metrics (which the caller already does) beats
        # picking one of them on a coin flip.
        return None
    return next(iter(plausible))


def _no_direction_message(family: str, counts: dict[str, int]) -> str:
    """The refusal a new user meets first, with the remedy in it.

    `runs compare` on the shipped example workspace used to say only what it
    found and stop, which reads as a dead end rather than a step -- while its
    sibling refusal for a single named metric already pointed at the file to
    edit.
    """
    if not counts:
        # No metrics at all is a different problem from an undeclared
        # direction, and telling this reader to edit metric_direction.toml
        # sends them to a file where there is nothing to write. Reached on a
        # real corpus by following this refusal's own advice: `runs compare
        # water --project ferric` said "found none" and then advised declaring
        # one of them.
        return (
            f"family {family!r} has no recorded metrics, so there is nothing to"
            " rank. The runs were found but their artifacts carried no numeric"
            " results -- check what `attest runs show` reports for one of them."
        )
    suggestion = _likeliest_metric(counts)
    example = (
        f" -- `{suggestion}` looks like a result metric, so perhaps"
        f' `{suggestion} = "higher_is_better"`'
        if suggestion
        else " for whichever of those is the result you care about"
    )
    return (
        f"no metric with a known direction in family {family!r};"
        f" found {sorted(counts) or 'none'}."
        f" Declare one under [metric_direction] in"
        f" {_metric_direction_path()}{example};"
        " guessing would rank ablation arms backwards."
    )


def _arms_for_run(run, values: list[dict], n_value, direction: str) -> list[dict]:
    """The comparable arms one run contributes -- usually itself, sometimes many.

    A run recording several unrecognised nested splits is not one arm; it is a
    whole experiment in one file, with the arms under keys like
    `arms.Treatment_Eigen`. Ranked as a single run it scored its own best
    sibling, so every arm was credited with the oracle upper bound sitting
    beside it. Each nested split becomes its own arm, named `<run>[<split>]`
    so the file it came from stays visible.
    """
    n = int(n_value) if n_value is not None else None
    groups = nested_arms(values)
    if groups:
        out = []
        for split, rows in sorted(groups.items()):
            best = _best_step(rows, direction)
            if best is not None:
                out.append(
                    {
                        "name": f"{run['name']}[{split}]",
                        "status": run["status"],
                        "value": best["value"],
                        "step": best["step"],
                        "split": split,
                        "source_path": run["source_path"],
                        "n": n,
                    }
                )
        return out
    best = _best_step(values, direction)
    return [
        {
            "name": run["name"],
            "status": run["status"],
            "value": best["value"] if best else None,
            "step": best["step"] if best else None,
            # which split the number came from: a reader comparing arms needs
            # to know whether they are looking at test or train
            "split": best["split"] if best else None,
            # provenance: every number must be traceable to the file it came
            # from, or the comparison cannot be audited
            "source_path": run["source_path"],
            "n": n,
        }
    ]


def _compare_rows(
    conn: sqlite3.Connection, family: str, project: str | None, *, need_counts: bool
) -> tuple[list[dict], dict[str, int]]:
    """The runs of one family plus, when `need_counts` is true, a count of how
    many arms report each metric name.

    Metric discovery needs `counts` before the values query can run (the
    metric is not known until one is picked), so this reader returns both
    rather than splitting on a boundary the caller cannot use independently.
    `need_counts=False` is what a caller who already named a metric passes: the
    GROUP BY below never runs for it, which is the same one-fewer-round-trip
    behaviour `compare()` had before this split, when the counts query was
    guarded by `if metric is None:`.
    """
    rows = _family_rows(conn, family, project)
    _refuse_cross_project(family, rows)
    if not rows or not need_counts:
        return rows, {}
    run_ids = [r["id"] for r in rows]
    placeholders = ",".join("?" * len(run_ids))
    counts: dict[str, int] = {}
    for m in conn.execute(
        f"SELECT run_id, metric FROM run_metrics WHERE run_id IN ({placeholders})"
        " GROUP BY run_id, metric",
        run_ids,
    ):
        counts[m["metric"]] = counts.get(m["metric"], 0) + 1
    return rows, counts


def _compare_values(
    conn: sqlite3.Connection, run_ids: list[int], metric: str
) -> tuple[dict[int, list[dict]], dict[int, float]]:
    """Every recorded value of `metric`, and the `n_records` count, for each
    run in `run_ids`.

    Two grouped queries instead of two-per-arm: run_id IN (...) covers every
    arm at once, so an N-arm family costs a constant number of round trips
    rather than 2N.
    """
    placeholders = ",".join("?" * len(run_ids))
    values_by_run: dict[int, list[dict]] = {rid: [] for rid in run_ids}
    for v in conn.execute(
        f"SELECT run_id, value, step, split FROM run_metrics"
        f" WHERE run_id IN ({placeholders}) AND metric = ?",
        (*run_ids, metric),
    ):
        values_by_run[v["run_id"]].append(dict(v))
    n_by_run: dict[int, float] = {}
    for row in conn.execute(
        f"SELECT run_id, MAX(value) AS value FROM run_metrics"
        f" WHERE run_id IN ({placeholders}) AND metric = 'n_records' GROUP BY run_id",
        run_ids,
    ):
        n_by_run[row["run_id"]] = row["value"]
    return values_by_run, n_by_run


def _pick_metric(counts: dict[str, int], directions: dict[str, str], family: str) -> str:
    """The metric to rank by when the caller did not name one: whichever
    directed metric the most arms share, so the comparison covers the family.

    Raises rather than guessing when no recorded metric has a known
    direction -- ranking one would risk ordering an ablation backwards.
    """
    known = {k: v for k, v in counts.items() if _metric_direction(k, directions)}
    if not known:
        # Names the fix, not just the problem. Its sibling refusal below
        # (single named metric) already pointed at the file to edit; this
        # one listed what it found and stopped, and it is the one a new
        # user hits FIRST -- `runs compare` on the shipped example data
        # refuses this way, which is the flagship capability's first
        # impression. An error that states a rule without stating the
        # remedy reads as a dead end rather than a step.
        raise ValueError(_no_direction_message(family, counts))
    return sorted(known.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _compare(
    runs: list[dict],
    values_by_run: dict[int, list[dict]],
    n_by_run: dict[int, float],
    metric: str,
    directions: dict[str, str],
    family: str,
) -> dict:
    """Rank every arm of an ablation family by `metric` -- the pure decision
    half of `compare()`. Takes rows and already-fetched values so it can be
    exercised with literal dicts and no database.
    """
    direction = _metric_direction(metric, directions)
    if direction is None:
        raise ValueError(
            f"unknown direction for metric {metric!r} -- refusing to rank."
            f" Declare it under [metric_direction] in {_metric_direction_path()};"
            " guessing would rank ablation arms backwards."
        )

    arms = []
    for r in runs:
        arms.extend(_arms_for_run(r, values_by_run[r["id"]], n_by_run.get(r["id"]), direction))

    scored = [a for a in arms if a["value"] is not None]

    def rank_key(arm: dict) -> tuple[float, str]:
        """Sort key: best value first regardless of direction, ties broken
        by name. See the comment above the `float()` call for why the None
        case is resolved here rather than inline in a lambda."""
        # Bind through a local float so the None-ness is resolved once, where
        # the `scored` filter above already guarantees it. Negating inside the
        # lambda read as `-None` to a type checker, and it was right to ask.
        value = float(arm["value"])
        return (value if direction == "lower_is_better" else -value, arm["name"])

    scored.sort(key=rank_key)
    missing = sorted((a for a in arms if a["value"] is None), key=lambda a: a["name"])

    shared, corpus_caveats = _corpus_agreement(runs, metric)
    return {
        "family": family,
        "metric": metric,
        "direction": direction,
        "arms": scored + missing,
        "winner": scored[0]["name"] if scored else None,
        "without_metric": [a["name"] for a in missing],
        "corpus": shared,
        # Reader caveats last: they qualify how every number here was
        # obtained, which is context for the ranking rather than a reason to
        # distrust one arm over another.
        "caveats": _all_caveats(scored, metric, corpus_caveats, runs),
    }


def compare(
    conn: sqlite3.Connection,
    family: str,
    metric: str | None = None,
    project: str | None = None,
) -> dict:
    """Rank every arm of an ablation family by a metric, within ONE project.

    The question this exists for: a sweep of N named config variants is a
    designed experiment, and which arm won usually lives only in filenames and
    memory. Arms with no value for the metric are listed in `without_metric`
    rather than dropped -- an arm that was never evaluated is a finding.

    A family name is unique per PROJECT, not globally: `families()` and
    runs.list both present (project, family) as the unit. This selected
    `WHERE family = ?` alone, so an English-ASR sweep and a Mandarin-ASR sweep
    that both called their arms `asr_baseline`/`asr_biglm` were pooled into one
    ranking with a named winner -- and, worse, _corpus_agreement keys on run
    name, so the collision ERASED the disagreement and the tool reported "all
    arms on aishell-1" for arms half of which ran on librispeech.
    """
    runs, counts = _compare_rows(conn, family, project, need_counts=metric is None)
    if not runs:
        # A dead end here is the same failure as an unexplained empty scan.
        # `compare <project>` is the intuitive first guess and finds nothing,
        # because families are derived from filename prefixes rather than from
        # the project directory -- so name the families that do exist.
        available = [
            r["family"]
            for r in conn.execute(
                "SELECT DISTINCT family FROM runs WHERE family IS NOT NULL ORDER BY family"
            )
        ]
        in_project = [
            r["name"]
            for r in conn.execute(
                "SELECT name FROM runs WHERE project = ? ORDER BY name", (family,)
            )
        ]
        if in_project:
            message = (
                f"no family {family!r}, but it is a project with"
                f" {len(in_project)} run(s). Families group arms by a shared"
                f" filename prefix, not by project"
            )
        elif available:
            # A project filter that excluded it, not a missing family. The old
            # message denied the family and then listed it as available in the
            # same sentence, never mentioning the argument that actually
            # failed, which reads as the tool being broken.
            elsewhere = [
                r["project"]
                for r in conn.execute(
                    "SELECT DISTINCT project FROM runs WHERE family = ? ORDER BY project",
                    (family,),
                )
            ]
            if project and elsewhere:
                message = (
                    f"family {family!r} exists, but not in project {project!r}."
                    f" It is in: {', '.join(elsewhere)}"
                )
            else:
                message = f"no family {family!r}"
        elif not conn.execute("SELECT 1 FROM runs LIMIT 1").fetchone():
            # An EMPTY ledger, not a naming problem. This branch fired for both
            # and described only the second, so the first thing a caller sees
            # after a fresh install blamed their filenames for a database with
            # nothing in it. runs.list gets this right; runs.compare is where a
            # model naturally lands first, and both gemma models dead-ended
            # here -- asking the user for a family name that already existed on
            # disk. Patching only this text made gemma4:e2b scan and succeed.
            message = (
                "the ledger is EMPTY -- no runs have been read yet."
                " Call runs.scan(confirm=true) to read the artifacts already on"
                " disk, then runs.list to see the families it found"
            )
        else:
            message = (
                f"no family {family!r}, and no run has one: families are derived"
                f" from a shared filename prefix, so arms need names like"
                f" `asr_baseline` / `asr_biglm` to be compared as a unit"
            )
        if available:
            message += f". Available: {', '.join(available)}"
        return {
            "family": family,
            "metric": metric,
            "arms": [],
            "winner": None,
            "available_families": available,
            "message": message,
        }

    directions = metric_directions()
    metric = metric or _pick_metric(counts, directions, family)
    run_ids_all = [r["id"] for r in runs]
    values_by_run, n_by_run = _compare_values(conn, run_ids_all, metric)
    return _compare(runs, values_by_run, n_by_run, metric, directions, family)


# Below this many samples, a difference between arms is not worth reading as a
# result. Not a significance test -- it is a prompt to go look, which is what an
# auditor needs when the tool cannot know the variance.
SMALL_SAMPLE = 30


# Metrics whose value has no meaning across different data. Ranking a loss or
# a perplexity computed on two corpora compares nothing; an accuracy at least
# shares a scale, so the caveat is worded less absolutely for the rest.
_CORPUS_SENSITIVE = frozenset({"loss", "ppl", "perplexity", "nll"})


def _corpus_agreement(runs: list[dict], metric: str) -> tuple[str | None, list[str]]:
    """`(shared_corpus_name, caveats)` for the arms being compared.

    Three cases, and telling them apart is the point. All arms agree: name it,
    so the reader learns the comparison was *checked* rather than assumed.
    Arms differ: say which saw what. Any arm unknown: say the comparison is
    unverified -- unknown is never silently treated as agreement, which is
    exactly the assumption every comparison made before this existed.
    """
    # Keyed on (project, name), which is what `UNIQUE (project, name)` on the
    # runs table already says the identity is. Keying on name alone let a
    # collision overwrite one arm's corpus with another's, so two corpora
    # looked like one and this function reported agreement it had not checked
    # -- failing closed to a confident falsehood. `compare` now refuses to
    # span projects, but a helper must not depend on its caller's discipline.
    named = {(r["project"], r["name"]): r.get("corpus") for r in runs}
    known = {c for c in named.values() if c}
    unknown = sorted(name for (_project, name), c in named.items() if not c)

    # A corpus whose own definition is disputed cannot ground an agreement
    # claim. corpus.upsert marks the row CONTESTED when two declarations of one
    # name disagree; naming it here would be the same false confidence the
    # name-collision bug produced, one layer down.
    contested = sorted(
        {
            str(source)
            for r in runs
            if (source := r.get("corpus_source")) and str(source).startswith("CONTESTED")
        }
    )
    if contested:
        return None, [
            "arms name the same corpus but its definition is disputed"
            f" ({'; '.join(contested)}) -- the comparison is unverified"
        ]

    if len(known) > 1:
        sensitive = _metric_stem(metric) in _CORPUS_SENSITIVE
        detail = ", ".join(f"{name} saw {c}" for (_p, name), c in sorted(named.items()) if c)
        note = (
            f"arms did not share a corpus ({detail});"
            f" {metric} is not comparable across different data"
            if sensitive
            else f"arms did not share a corpus ({detail}); check that {metric} is comparable"
        )
        return None, [note]

    if unknown and known:
        # Only a *partial* record is a finding. When no arm names a corpus
        # there is no inconsistency to report -- it is a ledger without corpus
        # data, and warning on every comparison would make the caveats
        # boilerplate, which is precisely what trains a reader to ignore them.
        return None, [
            f"corpus only partly recorded: {', '.join(unknown)}"
            f" omit{'s' if len(unknown) == 1 else ''} the data they used,"
            f" so agreement with {sorted(known)[0]} is unverified"
        ]

    return (next(iter(known)) if known else None), []


def _all_caveats(
    scored: list[dict], metric: str, corpus_caveats: list[str], runs: list[dict]
) -> list[str]:
    """Every reason to qualify a comparison, in order of how directly it bears
    on the ranking: the ranking's own caveats, then the corpus ones, then the
    limitations of the readers that produced the numbers.

    Reader caveats come last because they qualify how EVERY number here was
    obtained -- context for the whole comparison rather than a reason to
    distrust one arm over another.
    """
    return (
        _caveats(scored, metric)
        + corpus_caveats
        + adapter_caveats([r.get("adapter") for r in runs])
    )


def _caveats(scored: list[dict], metric: str) -> list[str]:
    """Reasons to distrust the ranking, stated with the ranking.

    A comparison that prints four decimal places and names a winner implies a
    confidence it has not earned. The tool cannot run a significance test (it
    has aggregates, not per-sample values, for most shapes), so it says plainly
    what it does not know instead of implying it does.
    """
    out: list[str] = []

    splits = {a.get("split") for a in scored}
    ranks_seen = {_split_rank(sp) for sp in splits}
    if len(ranks_seen) > 1:
        # Arms judged at different levels of trust. This fires for train
        # against test AND for train against unlabelled -- an earlier version
        # checked "all train" and "more than one named split" and fell through
        # the gap between them, letting a training-loss arm beat an unlabelled
        # one with no caveat at all.
        named = ", ".join(sorted(sp if sp else "(unlabelled)" for sp in splits))
        out.append(
            f"arms are judged on different splits ({named}); the comparison is"
            " only as sound as those splits are comparable"
        )
    if splits and all(_split_rank(sp) > len(_EVAL_SPLITS) for sp in splits):
        # Every arm is being judged on training data. That may be all that was
        # recorded, but a training loss ranks how well an arm memorised, not
        # how well it generalises, and a reader must not mistake one for the
        # other.
        out.append(
            f"every arm's {metric} comes from a training split; this ranks fit, "
            "not generalisation -- record an eval/test split to compare properly"
        )

    if len(scored) == 1:
        # A one-arm family always "wins". Reporting that without comment reads
        # as the result of a comparison, when nothing was compared -- usually a
        # sign the family heuristic split a sweep on inconsistent arm names.
        return [
            "only one arm in this family, so nothing was compared;"
            " arms group by a shared filename prefix"
        ]
    if not scored:
        return out

    sizes = [a["n"] for a in scored if a["n"] is not None]
    if scored[0]["n"] is not None and scored[0]["n"] < SMALL_SAMPLE:
        out.append(
            f"the winning arm's value rests on n={scored[0]['n']} samples"
            f" (< {SMALL_SAMPLE}); treat it as provisional"
        )
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
    # Each arm here is exactly one run -- the schema records no seed
    # replication (seed lives in `config`, never as a distinct run of the same
    # configuration). A ranking over single runs cannot separate the effect of
    # the configuration from ordinary run-to-run variance, regardless of how
    # large n_records is for any individual arm.
    out.append(
        "each arm is a single run; no seed replication, so this ranking cannot"
        " separate configuration from run-to-run variance"
    )
    return out


def families(conn: sqlite3.Connection, project: str | None = None) -> list[dict]:
    """Every named run family and its run count, largest first -- what
    `attest runs list` prints below the individual runs so a reader knows
    what is comparable with `runs compare`."""
    sql = (
        "SELECT project, family, COUNT(*) n FROM runs WHERE family IS NOT NULL"
        + (" AND project = ?" if project else "")
        + " GROUP BY project, family ORDER BY n DESC, family"
    )
    return [dict(r) for r in conn.execute(sql, (project,) if project else ())]
