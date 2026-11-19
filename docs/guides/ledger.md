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
that is not a family names the families that exist. A results CSV needs one
column naming each row's arm — `config_name`, `config`, `name`, `run`,
`variant`, `arm`, `label`, or `id` — or its rows have nothing to be named
after and the file scans to zero runs.

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

### Declaring a corpus

Corpus detection reads driver-script syntax (AST), not the model's own
claims about itself — see the [concepts glossary](../concepts.md). If
detection finds nothing, or finds the wrong thing (a loader called through a
wrapper function, or a path read from an env var), declare it instead of
editing the driver script to satisfy the detector. Put a `corpora.toml` next
to your workspace, or point `LEDGER_CORPUS_FILE` at one:

```toml
[corpus.wikitext2]
source = "Salesforce/wikitext"
tokenizer = "gpt2"
seq_len = 256

[assign]
family.lm = "wikitext2"
```

`[corpus.<name>]` declares a corpus; `[assign]`'s `family.<family>` (or
`run.<run-name>`, for one run rather than a whole family) links it to runs
already on disk. A declaration is never silently replaced by a later scan —
it fills gaps in what a weaker, artifact-detected value would have said, but
an existing declared value stands even when a subsequent scan disagrees.

### Recording a run

Writing the files above by hand is a five-step procedure — one JSON per arm,
a config with the exact same stem, a `[metric_direction]` entry for any
metric the ledger does not already know, an optional `corpora.toml`, then
scan and compare — and the direction step is the one a model following it
forgets: measured 0/15 on small local models, against ≥0.91 on every
file-shape step. `attest runs record` writes the files deterministically
instead of asking anyone to remember the rule:

```bash
attest runs record asr --arm baseline wer=0.12 --arm biglm wer=0.08 \
  --corpus librispeech --scan
```

writes `results/asr_baseline.json`, `results/asr_biglm.json`, a matching
`configs/*.yaml` per arm (provenance only — family, arm, corpus,
`recorded_at`, never a metric value), and a `corpora.toml` entry for
`librispeech` — then, because of `--scan`, reads them straight back into the
ledger and prints `runs compare`. A metric not already in the built-in table
needs `--direction METRIC=lower_is_better|higher_is_better`; leaving it out
is a refusal, with the identical sentence `runs compare` itself would print,
not a guess. `--dry-run` prints the manifest it would write — `{"files":
{relpath: content}}` — without touching disk, and every target is checked to
be a **new** file before anything is written; an existing target refuses the
whole call unless `--force`.

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

## Seed replicates

Name each seeded run `<config>_seed<N>` (`seed_N`, `seed-N` and `seed=N` work
too) and `compare` pools the seeds of one configuration into ONE arm instead
of ranking them against each other. A seed token never changes the family:
`clf_knn_seed3` groups exactly as `clf_knn` would, so `knn` and `logreg`
compare as arms of `clf`. A real five-seed sweep (sklearn digits):

```
  clf_knn                                          0.9829    540            mean of 5 runs, first .../clf_knn_seed0.json
  clf_logreg                                       0.9696    540            mean of 5 runs, first .../clf_logreg_seed0.json
  clf_sgd                                          0.9452    540            mean of 5 runs, first .../clf_sgd_seed0.json

  replicates:
  clf_knn: mean 0.9829 sd 0.0033 over 5 runs, 95% CI [0.9803, 0.9855] -- clf_knn_seed0, ...
  ...
  top two separable: yes, Cohen's d 3.225
```

A pooled arm reports `n_runs`, `mean` (also its `value`), `sd`, `min`/`max`,
`ci_low`/`ci_high` and up to five member names in `runs`. The interval is a
95% percentile bootstrap of the mean with a fixed seed, so the same runs
always print the same interval; under three runs there is no interval, only
the range. `separable` says whether the top two arms' intervals are disjoint
(`None` when either has none). None of this is a significance test, and the
caveats say so: with fewer than ten runs a bootstrap interval is too narrow.

A config can stop a merge but never start one. Seeds whose configs differ in
anything but `seed`, that claim the same seed, or that report different
splits stay separate arms, with a caveat naming the reason. Tracker runs that
carry no seed token in their name (`train/0`, `train/1`) are never pooled;
when their configs differ only in seed, a caveat says so. A family of single
runs prints exactly what it always did, ending with the single-run caveat.
Design record: `docs/superpowers/specs/2026-11-19-replicate-statistics-design.md`.
## Preregistering a hypothesis

The one thing a reviewer cannot reconstruct afterwards is what you predicted
before the runs existed. `attest prereg add` writes the prediction down and
signs it with the same key as a run record:

```bash
attest prereg add kdsweep --metric wer --expect "kdsweep_t4 < kdsweep_baseline" \
  --by 0.005 --note "distillation at T=4 beats the baseline"
attest prereg check            # exit 1 on refuted, or on altered text
attest prereg list             # the registrations, no database needed

`--expect` orders arms by the metric's raw values (`a < b`: a's value is
lower), chains allowed (`a < b < c`); `--by` is the minimum gap per pair, in
the metric's units. The metric needs a declared direction or `add` refuses
with the sentence `runs compare` prints. Files go to `--root`, else
`RESEARCH_ROOT`, else the current directory: `hypotheses.toml` (the index you
commit and diff) and `prereg/` (the canonical text, whose sha256 is the id,
and its signed record — `attest record verify prereg/<id>.json` checks it).
An edit is a new registration: `--supersedes <id>` keeps the old one and
flags it `superseded`.

`check` re-runs `runs compare` on each hypothesis's own metric and reports
`confirmed`, `refuted`, `inconclusive` or `untested`. With replicated arms
(`ci_low`/`ci_high` present) it compares intervals; with single runs, point
values decide only if you registered with `--allow-single-run`. An arm that
is missing is `untested`, never `refuted`, and naming one seed of a replicated
arm is `inconclusive`. Flags sit beside the verdict: `post_hoc` (registered
after a named arm's artifact was last written), `altered` (anything no longer
matches what was signed — the verdict is computed from the signed text).

What it proves, and what it cannot, is printed with every result:

- **proved** — this key signed exactly this text, and it has not changed;
- **evidence only** — `registered_at` is this machine's clock and `post_hoc`
  compares it with this disk's mtimes; `post_hoc` unset proves nothing;
- **cannot check** — that the prediction came before anyone saw the data, that
  no registration was deleted along with its record, or who holds the key
  unless you pin it. Commit `prereg/` somewhere you do not control to make
  deletion and backdating visible.

`runs.compare` carries the same verdicts as a `hypotheses` block (with
`prereg_limits`) when the family has registrations under `RESEARCH_ROOT`;
registering stays a CLI act. Design:
`docs/superpowers/specs/2026-11-19-preregistration-design.md`.

## Wall time per arm

**Wall time is not compute.** It is the seconds between a run's own start and
stop on whatever machine ran it: it includes queueing and data loading, differs
between a laptop and a GPU node, and says nothing about FLOPs, energy or cost.
Read it only as "these arms may not have had the same budget".

When a tracker recorded a duration (Sacred `start_time`/`stop_time`, MLflow
`start_time`/`end_time`, W&B `_runtime`) or a result file carries an explicit
`wall_time_s` / `runtime_s` key, `runs compare` shows a `wall_s` column and
`runs.compare` arms carry `wall_time_s`. A pooled replicate arm sums every run it
pooled and also reports `wall_time_mean_s`; if any of its runs has no duration it
reports none rather than a partial sum. Nothing is inferred: Hydra and DVC record
no duration, so those runs have none.

```
  caveat: the top two arms did not take the same wall time per run: m_big 32s vs m_small 12s (2.7x); ...
  caveat: compute not recorded for m_tiny while other arms have a wall time, ...
```

The first caveat fires when the top two arms differ 2x or more per run and the
longer took at least a second; the second only when some arm is timed and another
is not. A family with no durations prints exactly what it always did. Run records
(`attest record`, `attest claims --record`) also carry `predicate.environment`:
Python version, command line, the `uv.lock` hash and any GPU names. Design record:
`docs/superpowers/specs/2026-11-19-compute-tracking-design.md`.

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
