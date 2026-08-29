# The experiment ledger

Will it read my runs, and how does it rank them? Yes, from the artifacts
already on disk — results files and the layouts W&B, MLflow, Sacred, DVC,
and Hydra already write — with no instrumentation and no invented ranking
when a metric's direction is undeclared.

## The experiment ledger

Research generates artifacts — config files, eval dumps, benchmark tables — and
the numbers that end up in a README get transcribed from them by hand, where
nothing checks them again. The ledger reads those artifacts so the numbers stay
derivable.

```bash
export RESEARCH_ROOT=~/projects
uv run attest runs scan                    # read runs from artifacts on disk
uv run attest runs list                    # what exists, and what groups into families
uv run attest runs compare <family>        # rank the arms of a sweep
uv run attest runs show <project> <name>   # one run, with its source path
```

Two conventions decide whether a project is read, and both are worth knowing
before the first scan:

```
~/projects/
  asr-ablation/            <- a project: any directory under RESEARCH_ROOT
    results/               <- results live IN a recognised directory, not beside the run
      asr_baseline.json    <- arms share a prefix; `asr` is the family
      asr_biglm.json          {"wer": 0.0433} , or a list of per-sample records
      asr_moredata.json       (a list also gives the comparison its sample size)
    configs/               <- optional: recorded as provenance, never as a metric
      asr_baseline.json
```

Recognised results directories are `results/`, `logs/`, `outputs/`, `metrics/`,
`eval/`, `evals/`, `benchmarks/` and `reports/`; recognised config directories
are `configs/`, `config/`, `conf/`, `experiments/` and `examples/`. A scan that
finds nothing says which of these it looked for and where your files actually
were, rather than reporting an empty success — and `runs compare` on something
that is not a family names the families that exist.

**This is deliberately not an experiment tracker.** MLflow, Sacred, W&B and DVC
all instrument runs at the moment they happen. That does not help a corpus of
runs which already finished, across many projects, in several languages, some
dormant for months. Adoption cost is the design constraint: a tool needing new
discipline gets used for a week, while one that reads what is already there
keeps working after you forget it exists.

That argument cuts the other way for trackers you already run, so a scan also
reads five tracker layouts as conventions of their own, since the tool picks
the directory name, not you — all read as **final values, not curves**,
because a ledger that compares finished arms has no use for the whole series:

| tracker | layout | one-line caveat |
|---|---|---|
| W&B | `wandb/<run>/files/wandb-summary.json` | offline mode writes no summary/config files until synced |
| MLflow | `mlruns/<exp>/<run>/{metrics,params,tags}` | each metric is the final line of its per-step log file |
| Sacred | `FileStorageObserver` dirs, `run.json` + `metrics.json` | `run.json`'s own `result` field is read as a metric too |
| DVC | `dvc.yaml`'s declared `metrics:` files + `params.yaml` | each metric file is a snapshot overwritten on every `dvc repro` |
| Hydra | `--multirun` sweep dirs | needs `hydra.job.chdir=true` or all arms overwrite one `metrics.json` |

Every one of these was verified against a real directory produced by the real
tool (see `docs/superpowers/specs/2026-08-22-tracker-adapters-design.md` for
the per-tracker findings) — earlier drafts of this ledger read only the
published layouts, and some of what they assumed turned out to be wrong once
run for real.

One more grouping rule worth knowing: `family_of()` groups sibling runs by
their shared filename prefix, hyphen-joined regardless of which separator the
input used, so `dit_small_rope_crossattn` and `dit_small_rope_melmask` group
under `dit-small-rope`. A **bare hyperparameter stem** with no separate
prefix — `lr_0.001`, `lr_0.01` — has nowhere to strip down to, so the
recognised token itself (`lr`) becomes the family, letting `attest runs
compare lr` group a learning-rate sweep that has no shared name beyond the
parameter varied.

It reads the conventions research repos already use — `results/`, `logs/`,
`configs/`, `outputs/`, `benchmarks/` holding JSON, JSONL, CSV, YAML or TOML —
and no project is registered in advance. On the author's machine this found
**849 runs across 16 projects** with zero instrumentation.

Two rules keep it honest:

- **Record what is unambiguous, refuse the rest.** A config file is a
  specification with no result attached, so it gets no metrics rather than an
  invented one. An unrecognised shape yields no run. A mapping of hundreds of
  numeric keys is a lookup table, not a metrics record — a real tokenizer
  `vocab.json` was briefly read as 50,258 metrics, one of them keyed `wer`.
- **Never rank a metric whose direction is undeclared.** `compare` raises on
  total energies, because "lower is better" is false across different systems.
  Guessing would order a sweep backwards with total confidence.

Comparisons carry provenance and caveats. Every arm shows its `source_path` and
sample size, and the result warns when all arms are small-n, when arms differ in
sample size, when the top two are within 5%, or when arms sit at different
training steps. A healthy comparison emits no caveats — a tool that always warns
trains you to ignore it.

## Browsing the ledger

```bash
uv run attest browse            # read-only Datasette at :8898
uv run attest kg-report         # graph health + topic clusters
```

`browse` opens the database in [Datasette](https://datasette.io) with
`--immutable`, so a viewer can never write. The canned queries in
`datasette.yml` are the point: named SQL with shareable URLs, so a reviewer can
open the exact query behind a number, change one parameter, and watch the answer
move. `metric_over_time` is the time-series view; `unevaluated_configs` lists
sweeps that were specified but never run.

Datasette is a dev dependency and a separate process, never imported, so a
`uvx` install does not pay for it. It needs `--load-extension` for `sqlite-vec`
or it refuses to open the database at all — `attest browse` handles that.
