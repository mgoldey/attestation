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
own script. If you did not run the thing yourself, or want to check a
manuscript rather than record a run, hand off to `attestation-provenance`
or `attestation-annotate`.

## Run one command

If a shell is available, `attest runs record` writes the files below for
you and refuses rather than guesses a direction it doesn't know:

```bash
attest runs record asr --arm baseline wer=0.12 --arm biglm wer=0.08 \
  --corpus librispeech --scan
```

writes `results/asr_baseline.json` / `results/asr_biglm.json`, a matching
`configs/*.yaml` per arm (provenance only), and a `corpora.toml` entry for
`librispeech` -- then `--scan` reads them straight back and prints
`runs.compare`. A metric not already built in (see the list below) needs a
declaration, e.g. for a family scored on `novelty_rate`:

```bash
attest runs record lora --arm rank4 novelty_rate=0.31 --arm rank8 novelty_rate=0.44 \
  --direction novelty_rate=higher_is_better --scan
```

Leaving out `--direction` for an unfamiliar metric is a **refusal**, not a
guess, with the same sentence `runs.compare` itself would print. `--dry-run`
prints the manifest (`{"files": {relpath: content}}`) without writing
anything; every target must be a NEW file or the whole call refuses, unless
`--force`. No shell? Follow "The shape a scan will read" below and write
the files yourself -- everything the command does for you is stated there
as a rule.

## When NOT to use this

- Checking whether a draft's numbers are still true, or whether a claim is
  supported: that is `attestation-provenance` (reading) and
  `attestation-annotate` (writing the claim comments).
- Instrumenting a run with W&B, MLflow, Sacred, DVC or Hydra tracking as it
  runs: the ledger reads five tracker layouts after the fact (see below);
  it does not add discipline to the run itself.

## The shape a scan will read

`runs.scan(root, project, confirm)` walks a workspace and reads a fixed
list of directory names: `results`, `logs`, `outputs`, `metrics`, `eval`,
`evals`, `benchmarks`, `reports` (`ledger_adapters/generic.py`'s
`RESULT_DIRS`). Inside one, **one JSON file per arm** holding final scalar
values -- never a training curve, never a `best.json`/`summary.json`
alongside the per-arm files (a real scan on two arms plus one summary read
three runs, not two). JSONL, CSV, YAML and TOML are also read.

**Name arms so they share a prefix.** `family_of` groups sibling runs by
stripping a trailing step/variant token:

- Good: `kdsweep_t4.json` / `kdsweep_t8.json` -> family `kdsweep`.
- Bad: `run1.json` / `results_final.json` -- no shared prefix, nothing to
  rank as one sweep. A bare split token (`lr_0.001.json`) becomes its own
  family (`lr`).

**File the config beside it, never inside it**, in `configs/` (or
`config/`, `conf/`, `experiments/`). **MUST: the config's stem must
exactly match its result's stem** -- `discover()` pairs them by exact
stem equality, nothing fuzzier. For `results/asr_baseline.json`:

- Right: `configs/asr_baseline.yaml`.
- Wrong: `configs/asr_baseline_config.yaml` (no `_config` suffix, ever) or
  one shared `configs/config.yaml` for every arm.

A stem that matches nothing becomes an unevaluated run of its own, so a
two-arm sweep with mismatched config names scans as four runs, not two.

## Declare unfamiliar metrics before comparing

`runs.compare` **refuses** to rank a metric it does not have a direction
for. **Only these are already known** (`ledger.METRIC_DIRECTION`):

- lower is better: `wer`, `cer`, `loss`, `val_loss`, `ppl`, `perplexity`,
  `nll`, `mae`, `rmse`, `error`, `mse`, `mape`, `fid`, `eer`
- higher is better: `accuracy`, `r_squared`, `f1`, `auc`, `roc_auc`,
  `precision`, `recall`, `ndcg`, `map`, `mrr`, `bleu`, `rouge`, `iou`,
  `dice`, `psnr`, `ssim`

Anything else -- `novelty_rate`, `hallucination_score`, a metric your
harness invented -- needs `--direction METRIC=...` (the command) or, by
hand, an entry in `~/.hermes/metric_direction.toml`:

```toml
[metric_direction]
my_custom_score = "higher_is_better"
```

When in doubt, declare it: a redundant declaration for an already-built-in
metric is harmless; a missing one makes `runs.compare` refuse outright.

## Declare the corpus when it can't be detected

`corpus.detect_in_source()` reads which corpus a run used from the driver
script's own syntax (an AST read, not a model). When the script hides it
-- a notebook, a shell one-liner, a flag -- declare it in `corpora.toml`:

```toml
[corpus.speech-clean-16k]
source = "librispeech"
config = "clean"

[assign.family]
kdsweep = "speech-clean-16k"
```

## Hydra sweeps need one override

Hydra 1.2+ does **not** chdir per job by default. A `--multirun` sweep
without `hydra.job.chdir=true` has every arm overwrite the same top-level
results file:

```bash
python train.py --multirun lr=0.01,0.1,1,10 hydra.job.chdir=true
```

## Finish in the same session

```
runs.scan(confirm=true)
runs.compare(family="kdsweep", metric="wer")
```

`runs.compare` names a caveat (small sample, a top-two within a few
percent, arms on different corpora) if one applies; relay it, don't drop
it. If `compare` refuses on direction, that's the signal to go back to the
declare-a-direction step -- not a bug to work around.

## Instrumented trackers: hand-off, not competition

If the run already produces W&B (offline mode), MLflow, Sacred, DVC or
Hydra sweep-dir output, leave those files where the tracker writes them --
`runs.scan` reads all five layouts already. **It leaves files, it never
adds instrumentation**: write the plain results/config files above
instead of wiring up a tracker just to satisfy this skill.
