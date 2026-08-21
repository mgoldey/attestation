# Run ledger — design

**Date:** 2026-08-12
**Status:** proposed

## Problem

hermes-rss reads other people's science. It has no idea what its user *does*.

The user's own work lives in `~/qc` as eight projects whose results are
recorded in exactly one place: a hand-maintained `INDEX.md` written in prose.
That file is genuinely good — it grades each project honestly ("Retired on a
decisive negative", "Architecture only — zero results", "unsubmitted draft,
benchmark incomplete") — but every number in it was transcribed by hand from
artifacts scattered across log directories, and it goes stale the moment a run
finishes.

Meanwhile the artifacts themselves are already structured:

| Project | Artifact | Count |
|---|---|---|
| mel-flow-tts | `configs/*.yaml` with prose hypothesis headers | 9 in one ablation family |
| mel-flow-tts | `logs/*.json` — per-utterance WER/CER | 29 |
| mel-flow-tts | `logs/auto_eval_step_*.log` with a `N=.. mean WER=..` summary line | ~20 |
| ferric | `examples/*.toml` input decks | dozens |
| ferric | `benchmarks/**/results*.json` | several |

The question a working scientist cannot answer today without grepping:
**"Of the nine `libritts_r_dit_small_rope_*` ablation arms, which won, on what
metric, and by how much?"** That is a designed ablation whose outcome exists
only in filenames and memory.

## What this is not

Not an experiment *tracker*. It does not wrap training, does not ask the user
to call `log_metric()`, and does not need any change to how the projects run.
Adoption cost must be zero: the artifacts already exist, and a tool that
requires new discipline will not be used. This reads what is on disk.

Not a replacement for `INDEX.md`. The goal is to make the *numbers* in it
derivable, so the prose judgement can stay hand-written where it belongs.

## Decisions

### Storage: `runs` and `run_metrics` in `hermes.db`

Two tables in the existing database, not a new store. A run is identified by
`(project, name)` — the natural key on disk — and carries a `source_path` so
every row can be traced back to the file it came from.

```
runs(id, project, name, family, status, started, source_path, config_json, notes)
run_metrics(run_id, metric, value, step, split)
```

`family` is the ablation group (`libritts_r_dit_small_rope`), derived by
stripping the trailing variant token from the name. It is what makes
"compare the arms" a single query.

`run_metrics` is long-format (one row per metric per step) rather than wide,
because different projects report entirely different metrics and a wide table
would need a migration per project.

### Adapters, not a universal parser

**The central design decision.** A single parser that claims to read every
project's results would be a lie: ferric's `benchmarks/a24-subset/results.json`
is a dict keyed `"2|0.1|dimer"` — geometry, scaling factor, fragment — while
mel-flow-tts's `logs/whisper_auto_step_22000.json` is a list of per-utterance
`{file, target, transcript, wer, cer}` records. These do not share a schema and
never will.

So `ledger.py` defines a small adapter protocol, and each project gets an
adapter that knows its conventions:

```python
def discover(root: Path) -> list[RunRecord]   # what runs exist
```

Adapters live in `src/hermes/ledger_adapters/`. Shipping two (mel-flow-tts,
ferric) proves the protocol against genuinely different shapes. A project with
no adapter is reported as unsupported rather than silently skipped — the
failure mode this codebase keeps hitting is tools that claim success for work
they did not do.

### Metric direction must be declared

`WER 0.0433 → 0.0527` is worse; `accuracy 0.90 → 0.94` is better. A comparison
tool that does not know which way is up will confidently rank ablation arms
backwards. Each metric carries a direction (`lower_is_better` /
`higher_is_better`), declared per adapter, and `runs_compare` refuses to rank
on a metric whose direction it does not know rather than guessing.

### Read-only against the projects

The ledger never writes to `~/qc`. It reads artifacts and writes only to
`hermes.db`. A scan is idempotent: re-scanning replaces a project's rows
rather than duplicating them.

## MCP tools

Four, bringing the surface from 28 to 32.

| Tool | Behavior |
|---|---|
| `runs_scan(project=None, confirm)` | Re-read artifacts from disk into the ledger. Mutating, so it needs `confirm=true`. |
| `runs_list(project=None, family=None, limit=20)` | What runs exist, with status and headline metric. |
| `runs_compare(family, metric=None)` | The ablation question: every arm side by side, ranked, with the winning arm named. Refuses unknown-direction metrics. |
| `runs_detail(project, name)` | One run: config, all metrics, source path, the hypothesis header if the adapter found one. |

## Testing

- Adapters against **real fixture files copied from `~/qc`**, not invented ones
  — a parser tested only on synthetic input proves nothing about the format it
  claims to read.
- `runs_compare` ranks a known ablation correctly, and **fails** when handed a
  metric whose direction is undeclared.
- Direction handling: a lower-is-better metric must not rank ascending.
- Idempotency: scanning twice yields identical rows.
- A project with no adapter reports unsupported, not empty success.
- All existing tests pass untouched.

## Out of scope

Wrapping training runs; live metric streaming; plots; hyperparameter search;
anything that requires the user to change how projects are run; parsing the
prose hypothesis headers into structured claims (the header is stored verbatim
and shown, not interpreted).

## Sequencing

1. `ledger.py` — schema, `RunRecord`, adapter protocol, scan/idempotency.
2. The two adapters, against copied fixtures.
3. MCP tools + docs.

Task 1 is useful alone: a queryable ledger with one adapter answers the
ablation question.
