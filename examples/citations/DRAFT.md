# Draft: temperature-scaled knowledge distillation for a small speech model

Notes toward a paper, checked against the `speech-distill` runs in
`examples/workspace/` and against `references.bib` for anything cited by key.

## Background

The transformer architecture we distil from uses self-attention throughout,
following Vaswani et al.
<!-- claim: speech-distill/kdsweep_baseline metric=wer value=0.0731 cite=vaswani2017attention -->
The baseline model reaches a word error rate of 0.0731 before distillation.

Optimisation for every arm uses Adam, as originally described by Kingma and
Ba.
<!-- claim: speech-distill/kdsweep_t2 metric=wer value=0.0688 cite=kingma2015adam -->
Temperature-2 distillation improves this to 0.0688.

The distillation objective itself follows Hinton, Vinyals and Dean's
temperature-scaled soft-target formulation.
<!-- claim: speech-distill/kdsweep_t4 metric=wer value=0.0642 cite=hinton2015distilling -->
Temperature 4 does better still, at 0.0642 -- a relative reduction of 12.2%
over the baseline.

## A claim this draft cannot yet back with a reference

A related density-functional argument for why lower-entropy soft targets
transfer more efficiently is sketched in an unpublished note.
<!-- claim: speech-distill/kdsweep_t4b metric=wer value=0.0659 cite=doe2099imaginary -->
The seed-replication arm records 0.0659 on the same metric; the citation key
above names no entry in `references.bib` or any other configured source, on
purpose -- `attest claims` (and `cite.check` over MCP) reports it as
`uncited`, distinct from the run-level verdict on the number itself, which
resolves normally.

## Numbers with no claim at all

The encoder underneath all four arms has 41.3M parameters. This number is
deliberately uncovered: `attest claims --coverage` lists it as prose with no
`claim:` annotation behind it, the same rule `examples/workspace/` demonstrates.
