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

Use this right after you finish running an experiment, evaluation, or sweep
-- through `claude-code`, `codex`, a harness like `evaluating-llms-harness`,
or a script you wrote yourself. `runs.scan` only sees artifacts that already
follow a convention; nothing here is registered in advance, and nothing here
launches or instruments anything. If you did not run the thing yourself, or
the reader wants to check a manuscript rather than record a run, hand off to
`attestation-provenance` or `attestation-annotate` instead.

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
subdirectory as a project, and reads what it finds in a fixed list of
directory names: `results`, `logs`, `outputs`, `metrics`, `eval`, `evals`,
`benchmarks`, `reports` (the real list, `ledger_adapters/generic.py`'s
`RESULT_DIRS` -- not `data/`, not the project root). Inside one of those,
write **one JSON file per arm** holding final scalar values:

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
let it). A one-line comment header at the top of a YAML/TOML config is kept
verbatim by `runs.detail` and is often the only place a hypothesis survives
-- write the one sentence explaining what this arm changed.

## Declare unfamiliar metrics before comparing

`runs.compare` **refuses** to rank a metric it does not have a direction
for, rather than guessing -- ranking WER as if higher were better names the
worst arm the winner. If the metric you just wrote is not one of the
built-in ones (loss, accuracy, wer, and similar familiar names), add it
**before** the first `runs.compare` call, in
`~/.hermes/metric_direction.toml`:

```toml
[metric_direction]
my_custom_score = "higher_is_better"
```

Do this now, not when the refusal appears: you are the one who knows which
way the metric points, and the refusal exists so nobody downstream guesses
wrong on your behalf. `lower_is_better` is the other valid value.

## Declare the corpus when it can't be detected

`corpus.detect_in_source()` reads which corpus a run used from the driver
script's own syntax (an AST read, not a model), so most runs need nothing
extra. When the driver script does not make the corpus detectable -- a
notebook, a shell one-liner, a harness that hides the dataset behind a flag
-- declare it in `corpora.toml` next to the config, using the same table
shape `corpus.load_manifest` reads:

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

Hydra 1.2+ does **not** change directory per job by default (it did through
1.1). Run a `--multirun` sweep without `hydra.job.chdir=true` and every arm
overwrites the same top-level results file -- a real four-arm learning-rate
sweep produced one `metrics.json`, not four, this way. Always pass it
explicitly on a sweep:

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
`runs.scan` reads all five layouts already. This skill is about the case
where nothing is instrumenting the run: **it leaves files, it never adds
instrumentation**. Don't wire up a tracker just to satisfy this skill; write
the plain results/config files above instead.
