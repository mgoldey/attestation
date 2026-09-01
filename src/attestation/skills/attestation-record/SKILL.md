---
name: attestation-record
description: "Record the outputs of an experiment or evaluation you just ran so attestation's ledger can read them: final values as one JSON per arm in a recognised results directory, config filed as provenance, a metric direction declared before comparing, and a Hydra chdir fix for sweeps. Leaves files; never instruments a run."
version: 1.0.0
author: attestation project
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [experiments, ledger, provenance, reproducibility, local-api]
    related_skills: [attestation-provenance, attestation-annotate]
---

# attestation: recording a run so it can be read back

Use this right after finishing an experiment, evaluation, or sweep -- via
`claude-code`, `codex`, a harness like `evaluating-llms-harness`, or your
own script. `runs.scan` only sees artifacts that already follow a
convention; nothing here launches or instruments anything. If you did not
run the thing yourself, or want to check a manuscript rather than record a
run, hand off to `attestation-provenance` or `attestation-annotate`.

## When NOT to use this

- Checking whether a draft's numbers are still true, or whether a claim is
  supported: that is `attestation-provenance` (reading) and
  `attestation-annotate` (writing the claim comments).
- Instrumenting a run with W&B, MLflow, Sacred, DVC or Hydra tracking as it
  runs: the ledger reads five tracker layouts after the fact (see below);
  it does not add discipline to the run itself. Recording alongside one of
  those trackers means leaving the files it already writes where they are.

## The shape a scan will read

`runs.scan(root, project, confirm)` walks a workspace, treats each
subdirectory as a project, and reads a fixed list of directory names:
`results`, `logs`, `outputs`, `metrics`, `eval`, `evals`, `benchmarks`,
`reports` (`ledger_adapters/generic.py`'s `RESULT_DIRS` -- not `data/`,
not the project root). Inside one of those, write **one JSON file per
arm** holding final scalar values:

```
results/kdsweep_t4.json      {"wer": 0.061, "epochs": 40}
results/kdsweep_t8.json      {"wer": 0.057, "epochs": 40}
```

Final values only, never a training curve: a diverged run whose last value
is `nan` should record nothing for that metric rather than a mid-run number.
JSONL, CSV, YAML and TOML are also read, and a JSON object of objects (e.g.
one file holding several named splits) is flattened with the outer key kept
as `split`. A mapping with dozens of numeric-looking keys is refused as a
metrics record on purpose (a vocabulary or lookup table, not a result) --
keep a results file to a handful of named quantities.

**One JSON per arm, nothing else numeric in that directory.** `runs.scan`
reads *every* recognised file as a run, so it cannot tell a summary or
aggregate apart from an arm -- two arms means exactly two files, no
`best.json`/`summary.json`/`all_results.json` alongside them, even if one
only restates numbers the per-arm files already have (a real scan on two
arms plus one summary file read three runs, not two). Want a summary?
Write it *outside* the recognised directories -- project root or `notes/`,
anywhere `runs.scan` does not look.

**Name arms so they share a prefix.** `family_of` groups sibling runs by
stripping a trailing step/variant token, so `runs.compare` can rank them as
one sweep:

- Good: `kdsweep_t4.json` / `kdsweep_t8.json` -> family `kdsweep`.
- Good: `eval_step_18000.json` / `eval_step_22000_cfg2.0.json` -> family
  `eval` (the step and variant tokens are stripped before grouping).
- Bad: `run1.json` / `results_final.json` -- no shared, recognisable prefix,
  so each lands in the ledger ungrouped and `runs.compare` has nothing to
  rank. If a bare split token (`lr_0.001.json`, `lr_0.01.json`) *is* the
  whole stem, that token itself becomes the family (`lr`) -- do not add
  another prefix on top of it.

**File the config beside it, never inside it.** A results file states what
was *measured*; a config states what was *asked for*. Put the run's config
in `configs/` (or `config/`, `conf/`, `experiments/`), not merged into the
results JSON -- a hyperparameter recorded as a metric shows up as a
rankable number (`compare` will try to rank `seq_len` next to `wer` if you
let it). A one-line comment header at the top of a YAML/TOML config kept
verbatim by `runs.detail` is often the only place a hypothesis survives.

**MUST: the config's stem must exactly match its result's stem** --
`discover()` pairs them by exact stem equality, nothing fuzzier. For
`results/asr_baseline.json`:

- Right: `configs/asr_baseline.yaml` (stem `asr_baseline`, exact match).
- Wrong: `configs/asr_baseline_config.yaml` (no `_config`/`_cfg` suffix,
  ever) or one shared `configs/config.yaml` for every arm (stem `config`
  matches nothing).

A stem that matches nothing is not folded into that run -- it becomes an
unevaluated run of its own (same rule as "an arm never evaluated is a
finding"), so a two-arm sweep with mismatched config names scans as four
runs, not two.

## Declare unfamiliar metrics before comparing

`runs.compare` **refuses** to rank a metric it does not have a direction
for, rather than guessing -- ranking WER as if higher were better names the
worst arm the winner. **Only these are already known** (the live built-in
table, `ledger.METRIC_DIRECTION`):

- lower is better: `wer`, `cer`, `loss`, `val_loss`, `ppl`, `perplexity`,
  `nll`, `mae`, `rmse`, `error`, `mse`, `mape`, `fid`, `eer`
- higher is better: `accuracy`, `r_squared`, `f1`, `auc`, `roc_auc`,
  `precision`, `recall`, `ndcg`, `map`, `mrr`, `bleu`, `rouge`, `iou`,
  `dice`, `psnr`, `ssim`

**Anything else you write -- `novelty_rate`, `hallucination_score`,
`coherence_index`, a metric your harness invented -- you must declare
yourself**, before the first `runs.compare` call, in
`~/.hermes/metric_direction.toml`:

```toml
[metric_direction]
my_custom_score = "higher_is_better"
```

**When in doubt, declare it.** A redundant declaration for an
already-built-in metric is harmless -- it just repeats the answer
`runs.compare` already had; a missing one makes `runs.compare` refuse
outright. There is no case where declaring costs you anything, so if the
name is not in the two lists above, write the TOML entry rather than
guessing it "sounds standard enough" to be known. `lower_is_better` and
`higher_is_better` are the only two valid values.

## Declare the corpus when it can't be detected

`corpus.detect_in_source()` reads which corpus a run used from the driver
script's own syntax (an AST read, not a model), so most runs need nothing
extra. When the script does not make the corpus detectable -- a notebook,
a shell one-liner, a harness that hides the dataset behind a flag --
declare it in `corpora.toml` next to the config, in the shape
`corpus.load_manifest` reads:

```toml
[corpus.speech-clean-16k]
source = "librispeech"
config = "clean"
seq_len = 16000

[assign.family]
kdsweep = "speech-clean-16k"
```

`[assign.family]` (or `[assign.run]` for a single run) links the corpus to
the runs it belongs to, so `runs.compare` can guard against ranking arms
that trained on different data as if they were comparable.

## Hydra sweeps need one override

Hydra 1.2+ does **not** chdir per job by default (1.1 did). A `--multirun`
sweep without `hydra.job.chdir=true` has every arm overwrite the same
top-level results file -- a real four-arm sweep produced one `metrics.json`,
not four. Always pass it explicitly:

```bash
python train.py --multirun lr=0.01,0.1,1,10 hydra.job.chdir=true
```

## Finish in the same session

Once the files are on disk:

```
runs.scan(confirm=true)
runs.compare(family="kdsweep", metric="wer")
```

`confirm=true` is required because a scan replaces each scanned project's
rows -- run it once you're done writing, not mid-run. `runs.compare` will
name a caveat (small sample, a top-two within a few percent, arms on
different corpora) if one applies; relay it, don't drop it. If `compare`
refuses on the metric's direction, that's the signal to go back to the
`[metric_direction]` step above -- it is not a bug to work around.

## Instrumented trackers: hand-off, not competition

If the run already produces W&B (offline mode), MLflow, Sacred, DVC or Hydra
sweep-dir output, leave those files where the tracker writes them --
`runs.scan` reads all five layouts already. **It leaves files, it never
adds instrumentation**: don't wire up a tracker just to satisfy this
skill; write the plain results/config files above instead.
