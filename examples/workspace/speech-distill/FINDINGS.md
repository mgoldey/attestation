# Knowledge distillation for a small speech model

A worked example for `attest`. Every number below carries a `claim:` annotation
naming the run it came from, so `claims_check` can re-derive it from the
artifacts in `results/` rather than trusting the prose.

Three of these claims are deliberately wrong. Finding them is the demo.

## Setup

All arms train on LibriSpeech 100h, character tokenizer, sequence length 1024,
declared in `configs/corpus.toml`. The model is an 8-layer, 512-wide encoder at
41.3M parameters.

## Results

The baseline reaches a word error rate of 0.0731.
<!-- claim: speech-distill/kdsweep_baseline metric=wer value=0.0731 -->

Distillation at temperature 2 improves this to 0.0688.
<!-- claim: speech-distill/kdsweep_t2 metric=wer value=0.0688 -->

Temperature 4 is better still, at 0.0642 -- a 12.2% relative reduction over the
baseline.
<!-- claim: speech-distill/kdsweep_t4 metric=wer value=0.0642 -->

Validation loss follows the same ordering, ending at 2.18 for the best arm.
<!-- claim: speech-distill/kdsweep_t4 metric=val_loss value=2.18 -->

### Deliberately wrong claims, for the demo

This one is stale: it names a value that no longer matches the artifact.
<!-- claim: speech-distill/kdsweep_t2 metric=wer value=0.0701 -->

This one names a run that does not exist.
<!-- claim: speech-distill/kdsweep_t8 metric=wer value=0.0600 -->

This one is malformed -- no metric field at all.
<!-- claim: speech-distill/kdsweep_baseline value=0.0731 -->

## Seed sensitivity

Re-running the best arm with a different seed gives 0.0659, which is a swing of
0.0017 -- larger than the gap between several of the arms above. Any ranking of
temperatures 2 and 4 should be read with that in mind.
<!-- claim: speech-distill/kdsweep_t4b metric=wer value=0.0659 -->

## Numbers with no claim

The model has 41.3M parameters and trains for 20 epochs at a batch size of 64.
Those last three numbers are uncovered on purpose: `claims_coverage` should
list them, since a number in prose with nothing behind it is exactly what this
tool exists to surface.
