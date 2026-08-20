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
import re
import statistics
from pathlib import Path

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

# A metrics record has a handful of named quantities. A mapping with hundreds
# of numeric keys is a lookup table -- a tokenizer vocabulary, a char-to-index
# map, an embedding -- and reading it as metrics invents nonsense: one real
# vocab.json produced 50,258 "metrics", including a key literally named `wer`
# that then ranked as a WER of 1554. Refusing a shape we cannot interpret is
# the whole discipline here.
MAX_METRIC_KEYS = 60


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
    return [
        (k.lower(), float(v))
        for k, v in obj.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k.lower() not in _SKIP_KEYS
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
            if len(value) > MAX_METRIC_KEYS and all(
                isinstance(v, (int, float)) for v in value.values()
            ):
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


def metrics_from_payload(payload, step: int | None, split: str | None) -> list[Metric]:
    """Extract metrics from the three JSON shapes that actually recur.

    Shapes not covered here yield nothing, which is the correct outcome: an
    unrecognised layout should produce no run rather than a wrong one.
    """
    out: list[Metric] = []

    if isinstance(payload, dict):
        if len(payload) > MAX_METRIC_KEYS and all(
            isinstance(v, (int, float)) for v in payload.values()
        ):
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
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    return None
        return rows
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

    return (
        f"{len(in_result_dir)} file(s) under a results directory, but none held"
        f" a metrics record -- expected a JSON object of scalars like"
        f' {{"wer": 0.05}}, a list of such objects, or a CSV with numeric columns'
    )


def discover(root: Path) -> list[RunRecord]:
    project = root.name
    records: list[RunRecord] = []
    seen: set[str] = set()

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
                )
            )

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
                metrics = [
                    Metric(k.lower(), float(v))
                    for k, v in row.items()
                    if isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and k.lower() not in _SKIP_KEYS
                ]
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
            name = stem if result.parent == base else f"{result.parent.name}/{stem}"
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
                )
            )

    return records
