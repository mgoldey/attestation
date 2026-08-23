"""Convention-based discovery: works on any project, knows no project's name.

This is the default adapter and the one that should stay the default. It reads
directory and file conventions that ML/science repos share -- `results/`,
`logs/`, `configs/`, `outputs/`, `metrics/` holding JSON, JSONL, YAML or TOML --
rather than any particular project's layout.

The rule it follows: **record what is unambiguous, refuse to guess the rest.**

- A JSON object of scalars (`{"wer": 0.31, "epochs": 40}`) is a metrics record.
- A JSON list of objects with shared numeric fields (a per-item eval dump) is
  aggregated to the mean of each numeric field, plus `n_records`.
- A JSON object of objects (`{"key": {"method": value}}`) is flattened, with the
  outer key kept in `split` so a number traces back to what produced it.
- A config file is a specification with no result attached, so it gets status
  `spec` and no metrics -- never an invented one.

Numbers are extracted; meaning is not inferred. A field named `score` is stored
as `score`, and `compare` will refuse to rank it until someone declares which
direction is better, because guessing ranks ablations backwards.
"""

import json
import math
import re
import statistics
from pathlib import Path

from attestation import corpus
from attestation.ledger import Metric, RunRecord

# Where ML/science projects conventionally put results and configuration.
RESULT_DIRS = ("results", "logs", "outputs", "metrics", "eval", "evals", "benchmarks", "reports")
CONFIG_DIRS = ("configs", "config", "examples", "experiments", "conf")
CONFIG_SUFFIXES = (".yaml", ".yml", ".toml", ".json", ".ini", ".cfg")

# step_1000 / step-1000 / epoch_40 / iter50 / 22000steps / _10k
_STEP = re.compile(
    r"(?:step|epoch|iter|it|ckpt)[-_]?(\d+)(k?)|(\d+)(k?)[-_]?(?:steps?|epochs?)", re.I
)
# common ways a run records a variant: cfg2.0, seed=3, lr1e-4, fold_2
_SPLIT = re.compile(r"(?:^|[-_])(cfg|seed|fold|split|lr|temp|scale)[-_=]?([\w.]+)", re.I)

_SKIP_KEYS = {"step", "epoch", "iteration", "iter", "global_step", "seed", "index", "id"}
# Corpus fields name what a run *read*, never what it measured. Ranking
# `seq_len` or `vocab_size` as a metric is the same over-extraction bug as
# reading conformer names as measurements.
_SKIP_KEYS |= {k.lower() for k in corpus.CORPUS_KEYS}

# A metrics record has a handful of named quantities. A mapping with hundreds
# of numeric keys is a lookup table -- a tokenizer vocabulary, a char-to-index
# map, an embedding -- and reading it as metrics invents nonsense: one real
# vocab.json produced 50,258 "metrics", including a key literally named `wer`
# that then ranked as a WER of 1554. Refusing a shape we cannot interpret is
# the whole discipline here.
MAX_METRIC_KEYS = 60

# Words that appear in the name of a *quantity*. A metrics record names what it
# measured (`best_val_loss`, `n_params`, `wer`); a lookup table names the
# entities it measured them for (`H_ttt`, `P_gg` -- molecular conformers).
# Real example: benchmarks/gmtkn30/aconf_pyscf_energies.json maps 18 conformers
# to energies, and the ledger read all 18 names as rankable metrics. Only 18
# keys, so MAX_METRIC_KEYS never fired.
_METRIC_WORDS = frozenset(
    """acc accuracy auc best cer coef correlation count dim epochs err error eval f1
    final iters latency loss mae max mean median min mse n nll num params perplexity
    ppl prec precision r2 rate recall rmse score sd seed size std steps test throughput
    time train val valid validation var wer""".split()
)

# Below this, a record is too small to infer anything from its key vocabulary:
# two oddly-named fields are as likely a terse result as a table.
MIN_KEYS_FOR_ENTITY_CHECK = 6


def _is_lookup_table(node: dict) -> bool:
    """True when a mapping is data rather than a metrics record.

    Two shapes qualify. A *vocabulary* is huge and all-numeric (one real
    vocab.json produced 50,258 "metrics", including a key named `wer` that
    ranked as a WER of 1554). An *entity table* is small but keyed by the
    things measured rather than the measurements (`H_ttt`, `P_gg`).
    """
    if not node:
        return False
    numeric = [k for k, v in node.items() if isinstance(v, (int, float))]
    if len(node) > MAX_METRIC_KEYS and len(numeric) == len(node):
        return True
    return _is_entity_table(numeric)


def _is_entity_table(keys: list[str]) -> bool:
    """True when keys name entities rather than quantities.

    A metrics record's keys share a vocabulary with every other metrics record
    ever written; a table of per-entity values shares none of it. Requiring
    *zero* metric words keeps this conservative -- one `loss` or `_mean`
    anywhere and the record is read normally, so the cost of a false positive
    is bounded to files that look nothing like results.
    """
    if len(keys) < MIN_KEYS_FOR_ENTITY_CHECK:
        return False
    words: set[str] = set()
    for key in keys:
        words |= set(re.split(r"[^a-z0-9]+", key.lower()))
    return not (words & _METRIC_WORDS)


def _step_from(stem: str) -> int | None:
    m = _STEP.search(stem)
    if not m:
        return None
    digits, kilo = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    return int(digits) * (1000 if kilo else 1)


def _split_from(stem: str) -> str | None:
    m = _SPLIT.search(stem)
    return f"{m.group(1).lower()}{m.group(2)}" if m else None


def _numeric_items(obj: dict) -> list[tuple[str, float]]:
    """Numeric fields of a result object, excluding non-finite ones.

    NaN and inf are ordinary in recorded results -- a t-test between two
    identical arms is NaN, a diverged loss is inf -- and `json.dump` writes
    them as bare `NaN`/`Infinity` tokens that `json.loads` accepts. They must
    not become metrics: NaN compares false to everything, so a NaN would lose
    every ranking it appeared in and be reported as a legitimate last place.
    Dropping the field says "not measured", which is what happened.

    This is the one place every numeric metric passes through, so filtering
    here also protects the aggregate branch, where `statistics.pstdev` raises
    on a NaN and the traceback escaped `scan()` -- one such file took down
    every project in the workspace.
    """
    return [
        (k.lower(), float(v))
        for k, v in obj.items()
        if isinstance(v, (int, float))
        and not isinstance(v, bool)
        and k.lower() not in _SKIP_KEYS
        and math.isfinite(v)
    ]


# How deep to follow nested result structures. Deep enough for the shapes that
# occur (summary -> variants -> raw -> value is 3), shallow enough that a
# pathological document cannot produce thousands of spurious metrics.
MAX_DEPTH = 4


def _descend(node: dict, step: int | None, prefix: str, depth: int = 0) -> list[Metric]:
    """Numeric leaves of a nested mapping, with their path kept as `split`."""
    if depth > MAX_DEPTH:
        return []
    out: list[Metric] = []
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            if _is_lookup_table(value):
                continue  # a lookup table, not results
            out.extend(_descend(value, step, path, depth + 1))
    if depth > 0:  # top-level scalars are handled by the caller
        out.extend(Metric(k, v, step=step, split=prefix or None) for k, v in _numeric_items(node))
    return out


def _seed_config(payload) -> dict | None:
    """A top-level `seed` on a dict payload, kept as provenance -- not a metric.

    `seed` determines reproducibility, not performance: it belongs in `config`
    (what produced the run), never in `run_metrics` (what the run measured),
    or `compare` would have a rankable-looking column with no declared
    direction.
    """
    if not isinstance(payload, dict):
        return None
    seed = payload.get("seed")
    if seed is None or isinstance(seed, bool) or not isinstance(seed, (int, float, str)):
        return None
    return {"seed": str(seed)}


def _row_step(row: dict) -> int | None:
    """A row's step, under any of the names training loops conventionally use."""
    for key in ("step", "global_step", "iteration", "iter", "epoch"):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return int(value)
    return None


def metrics_from_payload(payload, step: int | None, split: str | None) -> list[Metric]:
    """Extract metrics from the three JSON shapes that actually recur.

    Shapes not covered here yield nothing, which is the correct outcome: an
    unrecognised layout should produce no run rather than a wrong one.
    """
    out: list[Metric] = []

    if isinstance(payload, dict):
        if _is_lookup_table(payload):
            return []  # a lookup table, not a metrics record
        flat = _numeric_items(payload)
        if flat:
            # {"wer": 0.31, "loss": 1.2} -- a metrics record
            out.extend(Metric(k, v, step=step, split=split) for k, v in flat)
            if payload.get("step") is not None and step is None:
                try:
                    fixed = int(payload["step"])
                    out = [Metric(m.metric, m.value, step=fixed, split=m.split) for m in out]
                except (TypeError, ValueError):
                    pass
        # Results nest. A real file puts its headline number at
        # summary.variants.raw.rho_pooled while bookkeeping counts sit at the
        # top; flattening only one level captured n_molecules and missed every
        # actual result. Descend, keeping the path as `split` so a value stays
        # traceable to where it sat.
        out.extend(_descend(payload, step=step, prefix=""))

    elif isinstance(payload, list):
        rows = [r for r in payload if isinstance(r, dict)]
        # A list of dicts is two different things. Rows carrying distinct
        # `step`s are ONE run measured over time -- a training log -- and
        # averaging a descending loss curve reports a number the run never had,
        # while calling its checkpoints `n_records` claims a sample size it does
        # not have. Rows without steps are independent samples, where the mean
        # is exactly right. Tell them apart by the steps themselves.
        steps = [_row_step(r) for r in rows]
        if (
            rows
            and len({s for s in steps if s is not None}) > 1
            and all(s is not None for s in steps)
        ):
            for row, row_step in zip(rows, steps, strict=True):
                out.extend(Metric(k, v, step=row_step, split=split) for k, v in _numeric_items(row))
            return out
        if rows:
            # a per-item eval dump: aggregate to the mean of each numeric field
            fields: dict[str, list[float]] = {}
            for row in rows:
                for k, v in _numeric_items(row):
                    fields.setdefault(k, []).append(v)
            for k, vs in sorted(fields.items()):
                out.append(Metric(k, float(statistics.fmean(vs)), step=step, split=split))
                # pstdev of a single value is 0.0 (stdlib-guaranteed), not an
                # error -- a bimodal [0.01, 0.01, 0.99] and a tight
                # [0.34, 0.34, 0.33] both mean ~0.337; without this the spread
                # that distinguishes them is unrecoverable once aggregated.
                out.append(Metric(f"{k}_std", float(statistics.pstdev(vs)), step=step, split=split))
            if fields:
                out.append(Metric("n_records", float(len(rows)), step=step, split=split))

    return out


def _csv_rows(path: Path) -> list[dict]:
    """A results CSV as rows, with numeric columns coerced.

    CSV is as common as JSON for recorded results and was missing at first --
    a 48-config sweep sat invisible in `results.csv` while the ledger reported
    the project had no results. csv is stdlib, so this costs no dependency.
    """
    import csv

    try:
        with path.open(newline="", errors="replace") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []

    out: list[dict] = []
    for row in rows:
        coerced: dict = {}
        for key, value in row.items():
            if key is None or value is None:
                continue
            try:
                coerced[key.strip()] = float(value)
            except (TypeError, ValueError):
                coerced[key.strip()] = value
        if coerced:
            out.append(coerced)
    return out


def _label_of(row: dict) -> str | None:
    """The column naming what a row is -- a sweep's arm identity.

    Without this every row of a 49-row sweep would collapse into one mean, and
    "which config won" would be unanswerable from the ledger.
    """
    for key in ("config_name", "config", "name", "run", "variant", "arm", "label", "id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _load(path: Path):
    text = path.read_text(errors="replace")
    if path.suffix == ".jsonl":
        # Skip unparseable lines rather than discarding the file. JSONL is
        # line-delimited precisely so it survives a partial write, and a run
        # killed mid-flush leaves a truncated final line -- the commonest way
        # these files end. Dropping thousands of good rows for one bad one is
        # silent data loss. A file that is not JSONL at all yields no valid
        # rows and so still produces no run.
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows or None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _header_comment(path: Path) -> str | None:
    """The leading comment block of a config -- often the stated hypothesis.

    Stored verbatim, never parsed: interpreting a hypothesis is a judgement,
    and this layer does not make judgements.
    """
    lines: list[str] = []
    for raw in path.read_text(errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            lines.append(stripped.lstrip("#").strip())
        elif not stripped and lines:
            continue
        elif lines:
            break
    return "\n".join(lines).strip() or None


def _config_shape(path: Path) -> dict | None:
    """Top-level keys/sections, without a YAML or TOML parser.

    Deliberately shallow: the point is to identify a config, not to reproduce
    it, and leaning on an undeclared parser dependency is how this codebase got
    burned before (networkx).
    """
    out: dict = {}
    section = None
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]")
        elif line.startswith("#") or not line:
            continue
        elif "=" in line and section:
            key, _, value = line.partition("=")
            out[f"{section}.{key.strip()}"] = value.strip().strip("\"'")
        elif ":" in line and not raw.startswith((" ", "\t")):
            out.setdefault("sections", []).append(line.split(":", 1)[0])
    return out or None


def family_of(stem: str) -> str | None:
    """Group sibling runs by their shared prefix, so a sweep compares as a unit.

    Two shapes, both common:

    - a *sweep*: `dit_small_rope_crossattn` / `dit_small_rope_melmask` differ in
      a trailing variant token, so the family is everything before it.
    - a *series*: `eval_step_22000` / `eval_step_18000_cfg2.0` are the same run
      at different steps, so stripping the step and variant tokens leaves the
      family directly.

    A heuristic, and treated as one. It groups names that follow a separator
    convention; anything else gets no family rather than a wrong one, and an
    ungrouped run is still listed, just not auto-compared.
    """
    stripped = re.sub(r"[-_]?(?:step|epoch|iter|it|ckpt)[-_]?\d+k?", "", stem, flags=re.I)
    stripped = _SPLIT.sub("", stripped)
    parts = [p for p in re.split(r"[-_]", stripped) if p]

    if not parts:
        return None
    if stripped != stem:
        # a series: the step/variant token was the distinguishing part, so what
        # remains already names the family
        return "-".join(parts)
    # a sweep: drop the trailing variant token. Two parts is enough --
    # `benchmark_adz` / `benchmark_atz` (basis sets) and `run_a` / `run_b` are
    # the commonest sweep shape there is, and requiring three would miss them.
    return "-".join(parts[:-1]) if len(parts) >= 2 else None


def diagnose_empty(root: Path) -> str:
    """Say why `discover` found nothing here. Called only when it found nothing.

    A scan that reports "0 run(s)" and no reason is the failure this tool can
    least afford: adoption cost is the design constraint, and a researcher whose
    layout is ordinary-but-unrecognised sees a successful-looking command that
    found nothing, with no next step. Every branch below names a real state and
    the action that resolves it.
    """
    files = [p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")]
    if not files:
        return "no files in this directory"

    readable = [p for p in files if p.suffix in (*CONFIG_SUFFIXES, ".jsonl", ".csv")]
    if not readable:
        kinds = sorted({p.suffix or "(no extension)" for p in files})
        return (
            f"{len(files)} file(s), none in a readable format"
            f" (found {', '.join(kinds[:5])};"
            f" expected {', '.join((*CONFIG_SUFFIXES, '.jsonl', '.csv'))})"
        )

    # Readable files exist, so the miss is about WHERE they are, or what is in
    # them. Distinguish those: they need different fixes.
    in_result_dir = [
        p for p in readable if any(part in RESULT_DIRS for part in p.relative_to(root).parts[:-1])
    ]
    if not in_result_dir:
        in_config_dir = [
            p
            for p in readable
            if any(part in CONFIG_DIRS for part in p.relative_to(root).parts[:-1])
        ]
        if in_config_dir and len(in_config_dir) == len(readable):
            return (
                f"{len(in_config_dir)} config file(s) but no results:"
                f" a spec with no result attached is recorded as a run with no"
                f" metrics, so put eval output in one of"
                f" {'/, '.join(RESULT_DIRS[:4])}/ to give it numbers"
            )
        where = sorted({str(p.relative_to(root).parent) for p in readable})[:3]
        return (
            f"{len(readable)} readable file(s), none under a recognised results"
            f" directory (found them in: {', '.join(where)};"
            f" expected {'/, '.join(RESULT_DIRS)}/)"
        )

    # A CSV whose numeric columns are fine but which names no arm is a real
    # and deliberate refusal (there is nothing to name the runs after), so it
    # needs its own message -- "no metrics record" misdescribes it.
    csvs = [p for p in in_result_dir if p.suffix.lower() == ".csv"]
    unlabelled = [p for p in csvs if (rows := _csv_rows(p)) and not any(map(_label_of, rows))]
    if unlabelled and len(unlabelled) == len(in_result_dir):
        return (
            f"{len(unlabelled)} CSV(s) with numeric columns but no column naming"
            f" each row's arm, so the runs cannot be named -- add one of:"
            f" config_name, config, name, run, variant, arm, label, id"
        )

    return (
        f"{len(in_result_dir)} file(s) under a results directory, but none held"
        f" a metrics record -- expected a JSON object of scalars like"
        f' {{"wer": 0.05}}, a list of such objects, or a CSV with numeric columns'
    )


# Experiment trackers put runs in a fixed directory name of their own choosing,
# which makes them conventions in the strict sense: every project using the tool
# produces the layout, and the user does not pick the name. That is why these
# live here rather than as named adapters -- see ledger_adapters/__init__.py.
#
# NEITHER READER HAS BEEN RUN AGAINST A REAL DIRECTORY. There was no wandb/ or
# mlruns/ on the machine where they were written, so both are built from the
# tools' published layouts and tested against transcribed fixtures. They are
# plausible, not verified. If you have a real one, point this at it: the shape-
# tolerance tests in tests/test_tracker_adapters.py say what should happen when
# the layout differs, but only a real library proves what it actually does.
TRACKER_DIRS = ("wandb", "mlruns")

# Recorded on every RunRecord this module's convention-based reader produces,
# so `runs.detail` can tell a hand-written results/ tree apart from a tracker
# directory. The tracker readers below label themselves "wandb" and "mlflow"
# instead, which is what lets ledger.ADAPTER_CAVEATS attach their limitations
# to exactly the runs they apply to.
ADAPTER_NAME = "generic"


def _yaml_scalars(path: Path) -> dict[str, str]:
    """Top-level `key: value` pairs from a small machine-written YAML file.

    Not a YAML parser and not trying to be. Both files this reads (`meta.yaml`,
    `config.yaml`) are generated, flat at the level we need, and unquoted. The
    module already refuses an undeclared parser dependency for exactly this --
    see `_config_shape` and the networkx note there.
    """
    out: dict[str, str] = {}
    for raw in path.read_text(errors="replace").splitlines():
        if raw.startswith((" ", "\t", "#")) or ":" not in raw:
            continue
        key, _, value = raw.partition(":")
        value = value.strip().strip("'\"")
        if value:
            out[key.strip()] = value
    return out


def _wandb_config(path: Path) -> dict | None:
    """Unwrap W&B's `{desc, value}` wrapper around every config entry.

    The file looks like:

        lr:
          desc: null
          value: 0.0003

    Storing it raw would make every config value a two-key dict. Keys starting
    with `_` are W&B's own bookkeeping (`_wandb`) and are dropped.
    """
    out: dict = {}
    key = None
    for raw in path.read_text(errors="replace").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw.startswith((" ", "\t")):
            key = raw.partition(":")[0].strip()
            continue
        stripped = raw.strip()
        if key and not key.startswith("_") and stripped.startswith("value:"):
            raw_value = stripped.partition(":")[2].strip().strip("'\"")
            if raw_value:
                out[key] = _coerce(raw_value)
    return out or None


def _coerce(text: str):
    """A YAML scalar as int, float or str -- in that order.

    A non-finite float stays a string. `float("nan")` and `float("inf")` both
    parse, so a config value literally spelled `nan` would otherwise become a
    NaN silently. Config is not ranked, so the stakes are lower than in
    `_mlflow_metric`, but it is the same mistake.
    """
    try:
        return int(text)
    except ValueError:
        pass
    try:
        value = float(text)
    except ValueError:
        return text
    return value if math.isfinite(value) else text


def _wandb_runs(root: Path, seen: set[str]) -> list[RunRecord]:
    """Runs under `wandb/run-<timestamp>-<id>/files/`.

    `wandb-summary.json` is a flat object of scalars, which
    `metrics_from_payload` already handles -- the mapping is nearly free. The
    work here is naming: `run-20260814_101133-a1b2c3d4` is a timestamp and a
    hash, and a ledger listing forty of those is unreadable.
    """
    base = root / "wandb"
    if not base.is_dir():
        return []
    records: list[RunRecord] = []
    for run_dir in sorted(base.iterdir()):
        files = run_dir / "files"
        summary = files / "wandb-summary.json"
        if not summary.is_file():
            continue
        payload = _load(summary)
        if not isinstance(payload, dict):
            continue
        # W&B writes its own bookkeeping into the summary: _step, _runtime,
        # _timestamp, _wandb. `_wandb_config` already drops underscore keys on
        # the config side; without the same filter here a wall-clock timestamp
        # is recorded as a measurement. compare() refuses to rank it, so it was
        # noise rather than a wrong verdict -- but a ledger listing
        # `_timestamp: 170000001.0` beside a real metric invites the misreading
        # the direction rule exists to prevent.
        payload = {k: v for k, v in payload.items() if not k.startswith("_")}
        metrics = metrics_from_payload(payload, None, None)
        if not metrics:
            continue

        meta_path = files / "wandb-metadata.json"
        meta = _load(meta_path) if meta_path.is_file() else None
        meta = meta if isinstance(meta, dict) else {}

        # run-20260814_101133-a1b2c3d4 -> a1b2c3d4
        short_id = run_dir.name.rsplit("-", 1)[-1]
        program = str(meta.get("program") or "").rsplit("/", 1)[-1]
        stem = program[:-3] if program.endswith(".py") else program
        name = f"{stem}/{short_id}" if stem else run_dir.name

        if name in seen:
            continue
        seen.add(name)

        config_path = files / "config.yaml"
        records.append(
            RunRecord(
                project=root.name,
                name=name,
                source_path=str(run_dir),
                family=stem or None,
                status="recorded",
                started=meta.get("startedAt"),
                config=_wandb_config(config_path) if config_path.is_file() else None,
                metrics=metrics,
                adapter="wandb",
            )
        )
    return records


def _mlflow_metric(path: Path) -> tuple[float, int | None] | None:
    """The final value of one MLflow metric file, and the step it was logged at.

    Each line is `<timestamp_ms> <value> <step>`. We read the last parseable
    one: this adapter records final values, not curves. Recording the whole
    history would make MLflow runs structurally unlike every other run in the
    ledger and would put one row per logged step into run_metrics -- a
    200-epoch run becomes 200 rows per metric.

    The consequence is worth stating plainly: **a user who wants training
    curves is not served by this ledger.**
    """
    for line in reversed(path.read_text(errors="replace").splitlines()):
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        # A diverged run's last logged loss is `nan` or `inf`, and float()
        # accepts both. `_numeric_items` filters these on the JSON path and its
        # docstring calls itself "the one place every numeric metric passes
        # through" -- this reader made that untrue by building Metric directly.
        # A NaN compares false to everything, so it loses every ranking it
        # appears in and is reported as a legitimate last place.
        #
        # Return None rather than falling through to an earlier finite line:
        # reporting a diverged run's mid-training loss as its result would be
        # worse than reporting nothing. The run diverged; "not measured" is
        # what happened.
        if not math.isfinite(value):
            return None
        step = None
        if len(parts) > 2:
            try:
                step = int(parts[2])
            except ValueError:
                step = None
        return value, step
    return None


def _mlflow_runs(root: Path, seen: set[str]) -> list[RunRecord]:
    """Runs under `mlruns/<experiment_id>/<run_id>/`.

    The metric-per-file layout is genuinely unlike anything else this adapter
    reads: not JSON, not CSV, one file per metric with one line per logged
    step. See `_mlflow_metric` for what is taken from it and why.
    """
    base = root / "mlruns"
    if not base.is_dir():
        return []
    records: list[RunRecord] = []
    for exp_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            meta_path = run_dir / "meta.yaml"
            if not meta_path.is_file():
                continue
            meta = _yaml_scalars(meta_path)
            # A run deleted in the MLflow UI is still on disk. Resurrecting it
            # in a ledger the user reads for provenance is worse than missing
            # it -- they decided it was not part of the record.
            if meta.get("lifecycle_stage", "active") != "active":
                continue

            metrics = []
            for metric_file in sorted((run_dir / "metrics").glob("*")):
                if not metric_file.is_file():
                    continue
                final = _mlflow_metric(metric_file)
                if final is not None:
                    metrics.append(Metric(metric_file.name, final[0], step=final[1]))
            if not metrics:
                continue

            label = meta.get("run_name") or run_dir.name
            name = f"{label}/{run_dir.name[:8]}" if meta.get("run_name") else run_dir.name
            if name in seen:
                continue
            seen.add(name)

            config = {}
            for param_file in sorted((run_dir / "params").glob("*")):
                if param_file.is_file():
                    config[param_file.name] = param_file.read_text(errors="replace").strip()

            records.append(
                RunRecord(
                    project=root.name,
                    name=name,
                    source_path=str(run_dir),
                    family=label if meta.get("run_name") else None,
                    status="recorded",
                    config=config or None,
                    metrics=metrics,
                    adapter="mlflow",
                )
            )
    return records


def _result_name(stem: str, result: Path, base: Path, dirname: str, seen: set[str]) -> str:
    """A run name that does not collide with one already recorded.

    Qualified by its directory when the bare stem is taken:
    `results/baseline.json` and `eval/baseline.json` are two runs, and
    recording only one silently discarded the other's numbers -- in a sweep
    where final scores moved to eval/, that ranks against stale ones with no
    caveat, because the tool cannot see what it dropped.
    """
    if result.parent != base:
        return f"{result.parent.name}/{stem}"
    return f"{dirname}/{stem}" if stem in seen else stem


def discover(root: Path) -> list[RunRecord]:
    project = root.name
    records: list[RunRecord] = []
    seen: set[str] = set()

    # RESULTS FIRST, then configs. Both share `seen`, and walking configs
    # first let `configs/asr_biglm.yaml` claim the name `asr_biglm` so
    # `results/asr_biglm.json` was skipped -- compare then reported the real
    # winner (WER 0.05) as "an arm that was never evaluated" and named the
    # loser. A config is a SPEC and a result is a MEASUREMENT; when both exist
    # for one name the measurement is the run, and the spec is what it was
    # going to be.
    for dirname in RESULT_DIRS:
        base = root / dirname
        if not base.is_dir():
            continue
        for table in sorted(list(base.glob("*.csv")) + list(base.glob("*/*.csv"))):
            rows = _csv_rows(table)
            labelled = [(r, _label_of(r)) for r in rows]
            if not any(label for _, label in labelled):
                continue  # no arm identity: nothing to name a run after
            for row, label in labelled:
                if not label:
                    continue
                # one run per labelled row: a 49-row sweep is 49 arms, not one
                # aggregate, or "which config won" is unanswerable
                name = f"{table.stem}/{label}"
                if name in seen:
                    continue
                seen.add(name)
                metrics = [Metric(k, v) for k, v in _numeric_items(row)]
                if not metrics:
                    continue
                records.append(
                    RunRecord(
                        project=project,
                        name=name,
                        source_path=str(table),
                        family=table.stem,
                        status="recorded",
                        config={k: v for k, v in row.items() if isinstance(v, (str, int, float))},
                        metrics=metrics,
                        adapter=ADAPTER_NAME,
                    )
                )

        candidates = (
            list(base.glob("*.json")) + list(base.glob("*/*.json")) + list(base.glob("*.jsonl"))
        )
        for result in sorted(candidates):
            payload = _load(result)
            if payload is None:
                continue
            stem = result.stem
            metrics = metrics_from_payload(payload, _step_from(stem), _split_from(stem))
            if not metrics:
                continue
            name = _result_name(stem, result, base, dirname, seen)
            if name in seen:
                continue
            seen.add(name)
            records.append(
                RunRecord(
                    project=project,
                    name=name,
                    source_path=str(result),
                    family=result.parent.name if result.parent != base else family_of(stem),
                    status="recorded",
                    config=_seed_config(payload),
                    metrics=metrics,
                    corpus=corpus.from_payload(
                        payload[0] if isinstance(payload, list) and payload else payload
                    ),
                    adapter=ADAPTER_NAME,
                )
            )

    # Tracker directories last, sharing `seen`: a project may have both a
    # results/ tree and a wandb/ one, and the same run must not appear twice.
    for dirname in CONFIG_DIRS:
        for cfg in sorted((root / dirname).glob("*")):
            if cfg.suffix.lower() not in CONFIG_SUFFIXES or not cfg.is_file():
                continue
            if cfg.suffix == ".json" and _load(cfg) is not None:
                continue  # a JSON config that parses as data is handled as a result
            name = cfg.stem
            if name in seen:
                continue
            seen.add(name)
            records.append(
                RunRecord(
                    project=project,
                    name=name,
                    source_path=str(cfg),
                    family=family_of(name),
                    status="spec",
                    config=_config_shape(cfg),
                    notes=_header_comment(cfg),
                    adapter=ADAPTER_NAME,
                )
            )

    records.extend(_wandb_runs(root, seen))
    records.extend(_mlflow_runs(root, seen))
    return records
