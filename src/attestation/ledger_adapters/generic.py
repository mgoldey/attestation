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

import hashlib
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

    Three shapes, all common:

    - a *sweep*: `dit_small_rope_crossattn` / `dit_small_rope_melmask` differ in
      a trailing variant token, so the family is everything before it --
      returned hyphen-joined regardless of which separator (`-` or `_`) the
      input used, so this pair's family is `dit-small-rope`, not
      `dit_small_rope`.
    - a *series*: `eval_step_22000` / `eval_step_18000_cfg2.0` are the same run
      at different steps, so stripping the step and variant tokens leaves the
      family directly.
    - a *bare split-token stem*: `lr_0.001` / `lr_0.01` have no separate shared
      prefix at all -- the recognised token (`lr`) IS the whole stem, so
      stripping it the way the sweep/series cases do empties the name. The
      token itself is the family here (real example: a four-arm learning-rate
      sweep named `results/lr_<lr>.json`, which `attest runs compare lr` must
      group as family `lr`).

    A heuristic, and treated as one. It groups names that follow a separator
    convention; anything else gets no family rather than a wrong one, and an
    ungrouped run is still listed, just not auto-compared.
    """
    stripped = re.sub(r"[-_]?(?:step|epoch|iter|it|ckpt)[-_]?\d+k?", "", stem, flags=re.I)
    stripped = _SPLIT.sub("", stripped)
    parts = [p for p in re.split(r"[-_]", stripped) if p]

    if not parts:
        # Nothing survived stripping. If a recognised split token consumed the
        # *entire* stem (not just a trailing step/variant suffix on top of a
        # real prefix), that token's own name is the family -- `lr_0.001` has
        # no prefix to fall back to, but `lr` is still a meaningful group.
        m = _SPLIT.match(stem)
        if m and m.end() == len(stem):
            return m.group(1).lower()
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
# The MLflow reader was run against a real directory on 2026-08-28
# (examples/flows/training/mlruns, written by mlflow-skinny 3.x via
# train_mlflow.py) and read four runs with final values and steps. The
# only surprise: mlflow-skinny 3.x refuses a `file:` tracking URI unless
# MLFLOW_ALLOW_FILE_STORE=true is set (the local filesystem backend is in
# maintenance mode upstream); run_name did land in meta.yaml as documented,
# so no fallback to tags/mlflow.runName was needed. The W&B reader was run
# against a real offline directory on 2026-08-28 (examples/wandb/wandb,
# written by wandb 0.17.6 via generate.py): the run directory is named
# `offline-run-<timestamp>-<id>`, not `run-<timestamp>-<id>` as the prior
# docstring assumed, but `_wandb_runs` never filtered on that prefix, so both
# already worked -- see its docstring for the naming detail and for the
# larger surprise, that offline W&B does not write wandb-summary.json or
# config.yaml to files/ at all until a run is synced.
#
# Hydra's reader was added 2026-08-28, verified against a real `--multirun`
# sweep (hydra-core 1.3.5, examples/hydra/generate.sh): the one surprise was
# that hydra-core 1.3.5 no longer changes into each arm's own output
# directory by default (`hydra.job.chdir` defaults to null, not True, as of
# Hydra 1.2 -- a backward-compatibility change from Hydra 1.1 and earlier's
# unconditional chdir), so `--multirun lr=...` alone wrote one shared
# metrics.json instead of four -- see examples/hydra/train.py's docstring.
TRACKER_DIRS = ("wandb", "mlruns", "sacred_runs", "multirun")

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


def _yaml_path_index(lines: list[tuple[int, str, str | None]], path: tuple[str, ...]) -> int | None:
    """The line index of the nested key at `path`, walking indentation.

    A small generalisation of `_yaml_scalars` for a file where the value
    this reader wants is not top-level -- `.hydra/hydra.yaml`'s
    `hydra.job.name` sits three levels deep under `hydra:` -> `job:` ->
    `name:`. Built on `_indented_lines` rather than a new parser, the same
    reuse `_dvc_stages`/`_dvc_lock_params` already make of it for a nested
    `stages: <name>: params: ...` shape -- this is that same walk, made
    reusable for an arbitrary dotted path instead of one fixed shape.
    Returns None if any segment of `path` is missing, so a caller degrades
    to a fallback rather than raising on a file an older or newer Hydra
    version writes slightly differently.
    """
    idx = 0
    indent: int | None = None
    found: int | None = None
    for key in path:
        found = None
        target_indent: int | None = None
        i = idx
        while i < len(lines):
            line_indent, line_key, _ = lines[i]
            if indent is not None and line_indent <= indent:
                break
            if target_indent is None:
                target_indent = line_indent
            if line_indent == target_indent and line_key == key:
                found = i
                break
            i += 1
        if found is None:
            return None
        indent = target_indent
        idx = found + 1
    return found


def _yaml_path_scalar(
    lines: list[tuple[int, str, str | None]], path: tuple[str, ...]
) -> str | None:
    """The scalar value at a nested `path`, or None -- see `_yaml_path_index`."""
    i = _yaml_path_index(lines, path)
    return lines[i][2] if i is not None else None


def _yaml_path_list(lines: list[tuple[int, str, str | None]], path: tuple[str, ...]) -> list[str]:
    """The block-list items under a nested `path`, or `[]` -- see
    `_yaml_path_index`. Hydra's own YAML dumper writes a list's `- item`
    lines at the SAME indent as the key introducing them (`task:` and its
    `- lr=0.01` line are both indent 4), not one level deeper the way this
    module's DVC helpers assume -- so this scans forward while the line is
    a list item, rather than requiring a deeper indent as `_dvc_stage_body`
    does for DVC's own block-list style."""
    i = _yaml_path_index(lines, path)
    if i is None:
        return []
    out: list[str] = []
    for _, key, value in lines[i + 1 :]:
        if key != "-":
            break
        if value is not None:
            out.append(value)
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
    """Runs under `wandb/<any-dir>/files/`.

    Verified 2026-08-28 against a real offline directory (wandb 0.17.6,
    examples/wandb/generate.py): the run directory is named
    `offline-run-<timestamp>-<id>`, not `run-<timestamp>-<id>` as an earlier
    version of this docstring claimed -- but this function never filtered by
    that prefix. It walks every child of `wandb/` looking for `files/
    wandb-summary.json`, so both names, and any future one, already worked;
    only the docstring's claim was too narrow.

    `wandb-summary.json` is a flat object of scalars, which
    `metrics_from_payload` already handles -- the mapping is nearly free. The
    work here is naming: `run-20260814_101133-a1b2c3d4` is a timestamp and a
    hash, and a ledger listing forty of those is unreadable.

    The real surprise was upstream of naming: **offline W&B does not write
    wandb-summary.json or config.yaml to files/ at all.** Every logged value
    reaches disk, but only inside the run's binary `.wandb` transaction log;
    the plain files this function reads exist only after `wandb sync`
    uploads to a real server (confirmed against wandb 0.17.6 through 0.29.0;
    also github.com/wandb/wandb issues #7227, #9646, and a maintainer's own
    answer on #1768 that no local API exists for this). examples/wandb/
    generate.py materialises them locally by decoding the `.wandb` log with
    `wandb.sdk.internal.datastore` -- the community's own workaround for
    this exact gap -- so the committed fixture is what a real synced run's
    files/ looks like, built from values wandb itself logged. A directory
    from `wandb.init(mode="offline")` alone, never synced and never run
    through that decode step, will scan to zero runs: not a bug in this
    reader, but the documented behaviour of the tool it reads.
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

        # offline-run-20260814_101133-a1b2c3d4 -> a1b2c3d4 (also works for
        # the older run-20260814_101133-a1b2c3d4 form -- both just split on "-")
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


def _sacred_metrics(metrics_path: Path) -> list[Metric]:
    """`metrics.json`'s `{name: {"steps": [...], "values": [...]}}` shape.

    Same decision as MLflow's metric-per-file log: the final value and the
    step it was logged at, not the curve -- a ten-point train_loss series
    becomes one Metric, not ten `run_metrics` rows. `values`/`steps` are
    parallel arrays kept in logged order, so the last element of each is the
    final one; a `timestamps` array is also present and unused here.
    """
    payload = _load(metrics_path)
    if not isinstance(payload, dict):
        return []
    out: list[Metric] = []
    for name, series in payload.items():
        if not isinstance(series, dict):
            continue
        values = series.get("values")
        if not isinstance(values, list) or not values:
            continue
        value = values[-1]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            continue
        steps = series.get("steps")
        step = (
            steps[-1] if isinstance(steps, list) and steps and isinstance(steps[-1], int) else None
        )
        out.append(Metric(name, float(value), step=step))
    return out


def _sacred_result_metric(run_json: dict) -> Metric | None:
    """`run.json`'s own `result` -- the value `@ex.main` returned.

    Sacred lets a main function return anything JSON-serialisable: a dict, a
    list, a string. Only a number is a metric; the field is simply absent
    from `metrics.json` (it is not something `_run.log_scalar` ever wrote),
    so this is the one place it can be picked up at all.
    """
    result = run_json.get("result")
    if (
        isinstance(result, bool)
        or not isinstance(result, (int, float))
        or not math.isfinite(result)
    ):
        return None
    return Metric("result", float(result))


def _sacred_runs(root: Path, seen: set[str]) -> list[RunRecord]:
    """Runs under `sacred_runs/<n>/`, one numbered directory per run.

    Verified 2026-08-28 against a real directory (sacred 0.8.7,
    examples/sacred/generate.py): `FileStorageObserver` writes `config.json`,
    `metrics.json`, `run.json`, `cout.txt` (captured stdout/stderr) and a
    shared `_sources/` of hashed copies of the driver script into each
    numbered run directory it creates, starting at `1`. `run.json` carries
    `experiment.name`, `status`, and -- unlike W&B or MLflow -- a `result`
    field holding whatever `@ex.main` returned. `cout.txt` and `_sources/`
    are not read here: see examples/sacred/generate.py's `scrub()` for why
    they are not committed either.
    """
    base = root / "sacred_runs"
    if not base.is_dir():
        return []
    records: list[RunRecord] = []
    for run_dir in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name):
        run_json = _load(run_dir / "run.json")
        if not isinstance(run_json, dict):
            continue
        # A run.json with status FAILED or INTERRUPTED is still on disk after
        # a crash. Recording it the same way as a completed run would
        # misreport a crash as a measurement.
        if run_json.get("status") != "COMPLETED":
            continue
        experiment = run_json.get("experiment")
        family = experiment.get("name") if isinstance(experiment, dict) else None
        if not isinstance(family, str) or not family:
            continue

        metrics_path = run_dir / "metrics.json"
        metrics = _sacred_metrics(metrics_path) if metrics_path.is_file() else []
        result_metric = _sacred_result_metric(run_json)
        if result_metric is not None:
            metrics.append(result_metric)
        if not metrics:
            continue  # a spec with no measurement attached -- same as MLflow

        name = f"{family}/{run_dir.name}"
        if name in seen:
            continue
        seen.add(name)

        config_path = run_dir / "config.json"
        config = _load(config_path) if config_path.is_file() else None
        config = config if isinstance(config, dict) else None

        records.append(
            RunRecord(
                project=root.name,
                name=name,
                source_path=str(run_dir),
                family=family,
                status="recorded",
                started=run_json.get("start_time"),
                config=config,
                metrics=metrics,
                adapter="sacred",
            )
        )
    return records


# indent, key, inline-value (None when the value is a nested block on
# following lines) -- the atom `_dvc_stages`/`_dvc_lock_stages` build a tree
# from. Comments and blank lines are dropped; DVC's own writer never quotes
# a plain scalar or emits flow-style mappings in these files, so a full YAML
# grammar buys nothing a machine-written file needs. A hand-edited dvc.yaml
# can still carry a trailing `# comment` on a line, or a flow-style list
# (`metrics: [a, b]`) instead of DVC's own block style -- both are handled
# rather than left to degrade silently into a misattributed generic run
# (see `_strip_inline_comment` and the flow-list check in `_dvc_stage_body`).
def _strip_inline_comment(text: str) -> str:
    """A trailing ` #...` comment, dropped -- never a `#` inside a value.

    DVC's own writer never emits one, but a hand-edited dvc.yaml commonly
    does (`- metrics/${item}.json  # note`). Only a `#` preceded by
    whitespace counts AND outside a quoted segment: an earlier version of
    this function used a plain whitespace-then-hash regex and truncated a
    single-quoted `1 # not a comment` to `1`, and a double-quoted `see docs
    # section 3` to `see docs` -- found by review, not by running a real
    dvc.yaml, since DVC's own writer never quotes a scalar in the first
    place (this module's own long-standing assumption, stated on
    `_indented_lines` above). A hand-edited file can still quote one, so a
    `#` is tracked against open/close quote state one character at a time
    rather than assumed to never occur inside one.
    """
    quote: str | None = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == "#" and (i == 0 or text[i - 1].isspace()):
            return text[:i].rstrip()
    return text.rstrip()


def _indented_lines(text: str) -> list[tuple[int, str, str | None]]:
    out: list[tuple[int, str, str | None]] = []
    for raw in text.splitlines():
        line = _strip_inline_comment(raw.rstrip())
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if stripped.startswith("- "):
            out.append((indent, "-", stripped[2:].strip() or None))
            continue
        key, sep, value = stripped.partition(":")
        if not sep:
            continue
        out.append((indent, key.strip(), value.strip() or None))
    return out


def _dvc_stages(dvc_yaml: Path) -> dict[str, dict]:
    """`stages:` from `dvc.yaml`, only stages that declare `metrics:` (a
    plain stage's own, or a `foreach`/`do` stage's).

    A stage with no `metrics:` is a data-prep or training step with nothing
    measured attached -- the same refusal the generic reader makes for a
    config file with no result, so it is silently excluded rather than
    reported as an empty run.

    Returns `{stage_name: {"foreach": param_name | None, "params": [...],
    "metrics": [...]}}`. `metrics` entries keep the literal `${item}` token
    unexpanded for a foreach stage -- expansion happens in `_dvc_runs`, once
    per item, not here.
    """
    lines = _indented_lines(dvc_yaml.read_text(errors="replace"))
    stages: dict[str, dict] = {}
    i = 0
    # Find "stages:" at indent 0, then each direct child is one stage name.
    while i < len(lines) and lines[i][:2] != (0, "stages"):
        i += 1
    i += 1
    stage_indent = lines[i][0] if i < len(lines) else None
    while i < len(lines) and lines[i][0] == stage_indent:
        name = lines[i][1]
        i += 1
        body_indent = lines[i][0] if i < len(lines) and lines[i][0] > stage_indent else None
        block: list[tuple[int, str, str | None]] = []
        while i < len(lines) and (body_indent is None or lines[i][0] >= body_indent):
            block.append(lines[i])
            i += 1
        stages[name] = _dvc_stage_body(block)
    return {name: body for name, body in stages.items() if body["metrics"]}


def _dvc_stage_body(block: list[tuple[int, str, str | None]]) -> dict:
    """One stage's `foreach`/`params`/`metrics`, from a plain stage or from
    inside its `do:` block -- both shapes are the same list of (indent, key,
    value) lines, just nested one level deeper for `do:`.

    `params:`/`metrics:` accept DVC's own block-list style (`metrics:` then
    `- a` / `- b` on following lines) and a flow-style list on the key's own
    line (`metrics: [a, b]`) -- both were confirmed against real `dvc repro`
    output for the block style, and the flow style is DVC-legal YAML a
    hand-edited dvc.yaml can use even though `dvc repro` itself never
    writes it. A trailing `# comment` anywhere in the file, block or flow,
    is stripped by `_indented_lines`/`_strip_inline_comment` before this
    function ever sees a line.
    """
    foreach = next((v for _, k, v in block if k == "foreach"), None)
    # `foreach: ${lr}` -> "lr"
    foreach_param = foreach.strip("${}") if foreach else None

    def _list_after(key: str) -> list[str]:
        for idx, (_, k, v) in enumerate(block):
            if k != key:
                continue
            if v is not None:
                # `metrics: [metrics/${item}.json]` -- a flow-style list
                # given inline on the key's own line, rather than DVC's own
                # block style (`metrics:` then `- ...` lines below). Parsed
                # the same way params.yaml's flow lists already are, so a
                # hand-edited dvc.yaml using either style is read, not
                # silently dropped into no run at all.
                return _dvc_flow_list(v)
            out = []
            for _, item_key, item_val in block[idx + 1 :]:
                if item_key != "-":
                    break
                if item_val is not None:
                    out.append(item_val)
            return out
        return []

    return {
        "foreach": foreach_param,
        "params": _list_after("params"),
        "metrics": [m.rstrip(":") for m in _list_after("metrics")],
    }


def _dvc_flow_list(text: str) -> list[str]:
    """`[0.01, 0.1, 1, 10]` as its literal item tokens, unparsed -- DVC prints
    `${item}` back into the command and the metric filename using the exact
    text `params.yaml` wrote (`1`, not `1.0`), so this must not round-trip
    through `float()`."""
    inner = text.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
        return [p.strip() for p in inner.split(",") if p.strip()]
    return [inner] if inner else []


def _dvc_params_yaml(params_yaml: Path) -> dict[str, str | list[str]]:
    """Top-level `key: value` and `key: [a, b, c]` pairs from `params.yaml`.

    Only top-level scalars and flow-style lists are read -- the two shapes
    `dvc.yaml`'s `params:` stanza and `foreach: ${key}` actually reference.
    A nested `params.yaml` (dot-path keys) is out of scope: `dvc.lock`'s own
    recorded values, read by `_dvc_lock_params`, are the fallback either way.
    """
    out: dict[str, str | list[str]] = {}
    for indent, key, value in _indented_lines(params_yaml.read_text(errors="replace")):
        if indent == 0 and key != "-" and value is not None:
            out[key] = _dvc_flow_list(value) if value.startswith("[") else value
    return out


def _dvc_lock_params(dvc_lock: Path) -> dict[str, dict[str, str]]:
    """Each stage instance's recorded `params.yaml` values from `dvc.lock`.

    `dvc.lock` nests as `stages: <name>: params: params.yaml: <key>: <value>`.
    For an ordinary key this value is the scalar actually used; for the key a
    `foreach` stage iterates over, DVC echoes `params.yaml`'s whole list
    instead of the one item that stage ran with (it is quoting the same
    source value for every instance of the sweep), so `_dvc_runs` overrides
    that one key from the stage-instance name rather than trusting it here.
    """
    lines = _indented_lines(dvc_lock.read_text(errors="replace"))
    result: dict[str, dict[str, str]] = {}
    i = 0
    while i < len(lines) and lines[i][:2] != (0, "stages"):
        i += 1
    i += 1
    stage_indent = lines[i][0] if i < len(lines) else None
    while i < len(lines) and lines[i][0] == stage_indent:
        stage_name = lines[i][1]
        i += 1
        params: dict[str, str] = {}
        depth = lines[i][0] if i < len(lines) and lines[i][0] > stage_indent else None
        in_params_file = False
        file_key_indent = None
        while i < len(lines) and (depth is None or lines[i][0] >= depth):
            indent, key, value = lines[i]
            if key == "params" and value is None:
                in_params_file = True
            elif in_params_file and file_key_indent is None and value is None:
                file_key_indent = indent  # the "params.yaml:" line itself
            elif in_params_file and file_key_indent is not None:
                if indent == file_key_indent + 2 and key != "-" and value is not None:
                    # A scalar param recorded directly, e.g. `epochs: 10`.
                    # `key == "-"` is the foreach param's own value instead
                    # spelled as a YAML list (`lr:` with `- 0.01` etc. on the
                    # following lines) -- not one scalar to record here, so
                    # it is skipped; `_dvc_runs` sources that key from the
                    # stage-instance name, which is the one true value.
                    params[key] = value
                elif indent <= file_key_indent:
                    in_params_file = False
            i += 1
        result[stage_name] = params
    return result


def _dvc_lock_outs(dvc_lock: Path) -> dict[str, dict[str, str]]:
    """Each stage instance's recorded `outs:` digests from `dvc.lock`.

    `dvc.lock` nests as `stages: <name>: outs: - path: <relpath> / hash: md5
    / md5: <digest> / size: <n>`. DVC's whole integrity guarantee is this
    digest: it is recomputed and compared on every `dvc repro`/`dvc status`,
    and reading it here for the same purpose costs nothing extra -- the file
    is already open and walked for `_dvc_lock_params`.

    Returns `{stage_name: {relpath: md5}}`, one entry per `outs:` item that
    carries an md5 (DVC's default hash; a stage using a different algorithm
    is skipped rather than compared against the wrong digest).
    """
    lines = _indented_lines(dvc_lock.read_text(errors="replace"))
    result: dict[str, dict[str, str]] = {}
    i = 0
    while i < len(lines) and lines[i][:2] != (0, "stages"):
        i += 1
    i += 1
    stage_indent = lines[i][0] if i < len(lines) else None
    while i < len(lines) and lines[i][0] == stage_indent:
        stage_name = lines[i][1]
        i += 1
        outs: dict[str, str] = {}
        depth = lines[i][0] if i < len(lines) and lines[i][0] > stage_indent else None
        in_outs = False
        outs_indent = None
        current_path: str | None = None
        while i < len(lines) and (depth is None or lines[i][0] >= depth):
            indent, key, value = lines[i]
            if key == "outs" and value is None:
                in_outs = True
                outs_indent = None
            elif in_outs and outs_indent is None and key == "-":
                outs_indent = indent  # the "- path: ..." line itself
                current_path = value.split(":", 1)[1].strip() if value else None
            elif in_outs and outs_indent is not None:
                if indent == outs_indent and key == "-":
                    current_path = value.split(":", 1)[1].strip() if value else None
                elif indent > outs_indent and key == "md5" and current_path and value is not None:
                    outs[current_path] = value
                elif indent <= stage_indent:
                    in_outs = False
            i += 1
        result[stage_name] = outs
    return result


def _dvc_metric_paths(root: Path) -> set[Path]:
    """Every metric file `dvc.yaml`'s declared stages point at, expanded --
    used by `discover` to keep the plain `metrics/` results scan from
    double-counting files DVC already accounts for as its own runs."""
    dvc_yaml = root / "dvc.yaml"
    if not dvc_yaml.is_file():
        return set()
    paths: set[Path] = set()
    for stage in _dvc_stages(dvc_yaml).values():
        items = _dvc_flow_list_from_params(root, stage["foreach"]) if stage["foreach"] else [None]
        for item in items:
            for metric in stage["metrics"]:
                rel = metric.replace("${item}", item) if item is not None else metric
                paths.add((root / rel).resolve())
    return paths


def _dvc_flow_list_from_params(root: Path, param: str) -> list[str]:
    params = _dvc_params_yaml(root / "params.yaml") if (root / "params.yaml").is_file() else {}
    value = params.get(param, [])
    return value if isinstance(value, list) else [value]


def _dvc_runs(root: Path, seen: set[str]) -> list[RunRecord]:
    """Runs declared by `dvc.yaml`'s stages, one per `foreach` item or one
    for a plain stage -- read without ever invoking `dvc` itself.

    Verified 2026-08-28 against a real `dvc repro` (dvc 3.x, examples/dvc/
    generate.sh): a `foreach: ${lr}` stage over `params.yaml`'s `lr` list
    expands to `dvc.lock` stage keys `train@0.01`, `train@0.1`, ... -- the
    `@item` suffix names the run, and the part before `@` is the family, the
    same split `_sacred_runs` makes on `/` for `experiment.name`. A plain
    (non-`foreach`) stage is a single run named for the stage itself with no
    family -- there is no sibling to group it with, the same reason
    `_mlflow_runs` leaves `family` unset for a run with no `run_name`. Only a
    stage declaring `metrics:` (directly, or under `do:`) is read; a
    data-prep stage with no metrics is not a run, the same refusal the
    generic reader makes for a config file with no measurement attached.

    Config comes from two places: `params.yaml`'s `params:` stanza names
    which keys a stage reads, and `dvc.lock` records the value actually used
    -- except the `foreach` key itself, whose `dvc.lock` entry is the whole
    source list rather than the one item that ran, so that one key is taken
    from the stage-instance name instead (see `_dvc_lock_params`).

    Tolerant of a missing `dvc.lock` (read before `dvc repro` has ever run,
    or deliberately not committed): the metric files `dvc.yaml` declares are
    still read if they exist, just with no recorded config. A stage listing
    four `foreach` items when only two have a metric file on disk (an
    in-progress sweep) yields two runs, not four with two broken.

    Also checks each metric file against `dvc.lock`'s own `outs:` digest --
    DVC's entire integrity guarantee, recomputed here rather than trusted
    blindly, since the ledger reads content mtime never touches. A file that
    no longer hashes to what `dvc.lock` recorded (hand-edited after `dvc
    repro`, or a stale checkout) gets its value read exactly as before, but
    `RunRecord.notes` names the mismatch so `compare()` can surface it as a
    caveat instead of presenting a silently-diverged number as trustworthy.
    """
    dvc_yaml = root / "dvc.yaml"
    if not dvc_yaml.is_file():
        return []
    stages = _dvc_stages(dvc_yaml)
    if not stages:
        return []
    params_yaml = root / "params.yaml"
    top_params = _dvc_params_yaml(params_yaml) if params_yaml.is_file() else {}
    dvc_lock = root / "dvc.lock"
    lock_params = _dvc_lock_params(dvc_lock) if dvc_lock.is_file() else {}
    lock_outs = _dvc_lock_outs(dvc_lock) if dvc_lock.is_file() else {}

    records: list[RunRecord] = []
    for stage_name, stage in sorted(stages.items()):
        foreach_param = stage["foreach"]
        items = top_params.get(foreach_param, []) if foreach_param else [None]
        if not isinstance(items, list):
            items = [items]
        for item in items:
            instance = f"{stage_name}@{item}" if item is not None else stage_name
            metrics: list[Metric] = []
            mismatches: list[str] = []
            for metric_rel in stage["metrics"]:
                rel = metric_rel.replace("${item}", item) if item is not None else metric_rel
                metric_path = root / rel
                if not metric_path.is_file():
                    continue
                payload = _load(metric_path)
                if payload is not None:
                    metrics.extend(metrics_from_payload(payload, None, None))
                recorded_md5 = lock_outs.get(instance, {}).get(rel)
                if recorded_md5:
                    # md5, matching dvc.lock's own recorded algorithm -- an
                    # integrity comparison against DVC's own digest, not a
                    # security hash, so usedforsecurity=False.
                    actual_md5 = hashlib.md5(
                        metric_path.read_bytes(), usedforsecurity=False
                    ).hexdigest()
                    if actual_md5 != recorded_md5:
                        mismatches.append(rel)
            if not metrics:
                continue  # declared but not yet produced -- not a broken run

            if instance in seen:
                continue
            seen.add(instance)

            config = dict(lock_params.get(instance, {}))
            for key in stage["params"]:
                top_value = top_params.get(key)
                if key not in config and isinstance(top_value, str):
                    config[key] = top_value
            if foreach_param and item is not None:
                config[foreach_param] = item

            notes = (
                "; ".join(
                    f"dvc.lock hash mismatch: {rel} changed since dvc repro" for rel in mismatches
                )
                or None
            )

            records.append(
                RunRecord(
                    project=root.name,
                    name=instance,
                    source_path=str(root / stage["metrics"][0].replace("${item}", item or "")),
                    family=stage_name if foreach_param else None,
                    status="recorded",
                    config=config or None,
                    notes=notes,
                    metrics=metrics,
                    adapter="dvc",
                )
            )
    return records


def _hydra_job_name(hydra_yaml: Path) -> str | None:
    """`hydra.job.name` from an arm's `.hydra/hydra.yaml`, or None.

    Hydra sets this to the driver script's stem (`train.py` -> `train`)
    unless a config overrides it, and writes it to every arm of a sweep --
    unlike W&B's `run.name`, never written locally in offline mode, this
    one is on disk from the start, the same as Sacred's `experiment.name`.
    """
    text = hydra_yaml.read_text(errors="replace")
    return _yaml_path_scalar(_indented_lines(text), ("hydra", "job", "name"))


def _hydra_config(config_yaml: Path) -> dict | None:
    """An arm's resolved config, from `.hydra/config.yaml`.

    Flat top-level `key: value` pairs -- `_yaml_scalars` already reads
    exactly this shape, coerced the same way `_wandb_config` coerces its
    unwrapped values, since Hydra's own dumper leaves ints/floats unquoted.
    """
    scalars = _yaml_scalars(config_yaml)
    return {k: _coerce(v) for k, v in scalars.items()} or None


def _hydra_arm_metrics(arm_dir: Path) -> list[Metric]:
    """Metrics from any JSON/JSONL/CSV file directly in an arm's directory.

    `.hydra/` is excluded -- it holds config and Hydra's own bookkeeping,
    never a result. Reuses `metrics_from_payload`/`_csv_rows`, the same
    readers the ordinary `results/` scan in `discover` uses, rather than a
    Hydra-specific parser: a JSON object of scalars (`metrics.json`, what
    `examples/hydra/train.py` writes) or a metrics CSV are both already
    handled shapes.
    """
    metrics: list[Metric] = []
    for path in sorted(arm_dir.glob("*")):
        if not path.is_file() or path.parent.name == ".hydra":
            continue
        if path.suffix in (".json", ".jsonl"):
            payload = _load(path)
            if payload is not None:
                metrics.extend(metrics_from_payload(payload, None, None))
        elif path.suffix.lower() == ".csv":
            rows = _csv_rows(path)
            if rows:
                metrics.extend(metrics_from_payload(rows, None, None))
    return metrics


def _hydra_runs(root: Path, seen: set[str]) -> list[RunRecord]:
    """Runs under `multirun/<date>/<time>/<n>/`, one numbered directory per
    arm of a sweep -- Hydra's own `--multirun` layout.

    Verified 2026-08-28 against a real sweep (hydra-core 1.3.5,
    examples/hydra/generate.sh): `python train.py --multirun
    lr=0.01,0.1,1,10 hydra.job.chdir=True` writes `multirun/<date>/<time>/
    <n>/{.hydra/{config.yaml,hydra.yaml,overrides.yaml}, metrics.json,
    train.log}` for each of the four arms, plus one `multirun.yaml`
    summarising the sweep -- see examples/hydra/train.py's own docstring
    for why `hydra.job.chdir=True` is required: hydra-core 1.3.5 no longer
    changes the working directory into each arm's own output directory by
    default, so without it every arm overwrites the same top-level
    metrics.json instead of writing its own.

    Naming: `<job.name>/<date>/<time>/<n>` is unreadable, so a run is named
    `<job.name>/<n>` with family `job.name` (read from `.hydra/
    hydra.yaml`, falling back to the sweep directory's own name when that
    file is missing or unreadable -- an older Hydra version, or a directory
    edited by hand). Two sweeps of the same job name collide on `<n>`
    alone, so a second sweep's arms are qualified with their time
    directory (`<job.name>/<time>/<n>`) rather than silently dropped --
    `seen` is shared with every other reader in this module, the same
    dedup `_wandb_runs`/`_mlflow_runs`/`_sacred_runs`/`_dvc_runs` already
    participate in.

    Tolerant of a missing `.hydra/hydra.yaml` (family falls back) and of no
    metrics file in an arm's directory (the arm is skipped, same as an
    MLflow run with no metric files) -- `_hydra_arm_metrics` degrades to
    fewer metrics rather than raising on a shape it does not recognise.
    """
    sweeps = sorted(p for p in root.glob("multirun/*/*") if p.is_dir())
    records: list[RunRecord] = []
    for sweep_dir in sweeps:
        arm_dirs = sorted(
            (
                p
                for p in sweep_dir.iterdir()
                if p.is_dir() and (p / ".hydra" / "config.yaml").is_file()
            ),
            key=lambda p: p.name,
        )
        if not arm_dirs:
            continue

        hydra_yaml = arm_dirs[0] / ".hydra" / "hydra.yaml"
        family = _hydra_job_name(hydra_yaml) if hydra_yaml.is_file() else None
        family = family or sweep_dir.name

        for arm_dir in arm_dirs:
            metrics = _hydra_arm_metrics(arm_dir)
            if not metrics:
                continue  # a spec with no measurement attached -- same as MLflow/DVC

            name = f"{family}/{arm_dir.name}"
            if name in seen:
                # a second sweep of the same job name -- qualify with the
                # time directory rather than silently dropping this arm
                name = f"{family}/{sweep_dir.name}/{arm_dir.name}"
            if name in seen:
                continue
            seen.add(name)

            records.append(
                RunRecord(
                    project=root.name,
                    name=name,
                    source_path=str(arm_dir),
                    family=family,
                    status="recorded",
                    config=_hydra_config(arm_dir / ".hydra" / "config.yaml"),
                    metrics=metrics,
                    adapter="hydra",
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
    """Every run this adapter recognises under one project directory,
    trying every tracker convention this module knows (wandb, mlflow,
    sacred, dvc, hydra, plain results/configs) and deduplicating by name via
    `seen` -- see the comment below for why results are read before configs."""
    project = root.name
    records: list[RunRecord] = []
    seen: set[str] = set()
    # dvc.yaml stages routinely write into metrics/, one of RESULT_DIRS --
    # unlike wandb/mlruns/sacred_runs, whose directory names never collide
    # with a plain-results convention. Without this a project with a DVC
    # sweep is scanned twice: once by _dvc_runs as `train@0.1`, and again by
    # the ordinary metrics/ walk below as a bare `0.1`, for the same file.
    dvc_metric_paths = _dvc_metric_paths(root)

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
            if result.resolve() in dvc_metric_paths:
                continue  # DVC's own output -- _dvc_runs below reads it
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
    records.extend(_sacred_runs(root, seen))
    records.extend(_dvc_runs(root, seen))
    records.extend(_hydra_runs(root, seen))
    return records
