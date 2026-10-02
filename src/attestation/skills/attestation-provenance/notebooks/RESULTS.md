# Results of the executed demos

Rendered by `scripts/build_skill_notebooks.py` from the outputs saved in the
notebooks next to this file, from a run against the repo's `examples/workspace`.
Nothing below was typed by hand. Open the `.ipynb` files to see the prose around it.

## Check a draft's claims

`check-a-drafts-claims.ipynb`

```text
Using the example workspace from the attestation repo (copied to a temporary folder).
```

```text
$ attest runs scan --root .
  retrieval-ablation           5 run(s)
  speech-distill               4 run(s)
9 run(s) across 2 project(s)
[exit 0]
```

| line | claim | verdict | why |
|---:|---|---|---|
| 18 | `kdsweep_baseline wer=0.0731` | **supported** | wer=0.0731 in kdsweep_baseline |
| 21 | `kdsweep_t2 wer=0.0688` | **supported** | wer=0.0688 in kdsweep_t2 |
| 25 | `kdsweep_t4 wer=0.0642` | **supported** | wer=0.0642 in kdsweep_t4 |
| 28 | `kdsweep_t4 val_loss=2.18` | **supported** | val_loss=2.18 in kdsweep_t4 |
| 33 | `kdsweep_t2 wer=0.0701` | **contradicted** | document says 0.0701, run records 0.0688 (tolerance 1e-09) |
| 36 | `kdsweep_t8 wer=0.06` | **unsupported** | no run speech-distill/kdsweep_t8 in the ledger -- the claim may be true, but nothing here backs it |
| 46 | `kdsweep_t4b wer=0.0659` | **supported** | wer=0.0659 in kdsweep_t4b |
| 39 | (cannot be read) | **malformed** | missing metric |

```text
{'supported': 5, 'contradicted': 1, 'unsupported': 1} plus 1 malformed
```

```text
$ attest claims speech-distill/FINDINGS.md
  malformed  speech-distill/FINDINGS.md:39: missing metric
  supported     FINDINGS.md:18               wer=0.0731  wer=0.0731 in kdsweep_baseline
  supported     FINDINGS.md:21               wer=0.0688  wer=0.0688 in kdsweep_t2
  supported     FINDINGS.md:25               wer=0.0642  wer=0.0642 in kdsweep_t4
  supported     FINDINGS.md:28               val_loss=2.18  val_loss=2.18 in kdsweep_t4
  contradicted  FINDINGS.md:33               wer=0.0701  document says 0.0701, run records 0.0688 (tolerance 1e-09)
  unsupported   FINDINGS.md:36               wer=0.06  no run speech-distill/kdsweep_t8 in the ledger -- the claim may be true, but nothing here backs it
  supported     FINDINGS.md:46               wer=0.0659  wer=0.0659 in kdsweep_t4b

7 claim(s): 1 contradicted, 5 supported, 1 unsupported
1 malformed
[exit 1]
```

```text
$ attest claims speech-distill/FINDINGS.md --coverage
  uncovered  FINDINGS.md:13           41.3         41.3M parameters.
  uncovered  FINDINGS.md:23           12.2         Temperature 4 is better still, at 0.0642 -- a 12.2% relative
  uncovered  FINDINGS.md:44           0.0017       0.0017 -- larger than the gap between several of the arms ab
  uncovered  FINDINGS.md:50           41.3         The model has 41.3M parameters and trains for 20 epochs at a

5/9 number(s) covered by a claim across 1 file(s)
[exit 0]
```

## Which arm won?

`which-arm-won.ipynb`

```text
Using the example workspace from the attestation repo (copied to a temporary folder).
```

```text
$ attest runs scan --root .
  retrieval-ablation           5 run(s)
  speech-distill               4 run(s)
9 run(s) across 2 project(s)
[exit 0]
$ attest runs list
  retrieval-ablation   planned_colbert                        spec       [planned]
  retrieval-ablation   rank_method_bm25                       recorded   [rank-method]
  retrieval-ablation   rank_method_dense                      recorded   [rank-method]
  retrieval-ablation   rank_method_dense2                     recorded   [rank-method]
  retrieval-ablation   rank_method_hybrid                     recorded   [rank-method]
  speech-distill       kdsweep_baseline                       recorded   [kdsweep]
  speech-distill       kdsweep_t2                             recorded   [kdsweep]
  speech-distill       kdsweep_t4                             recorded   [kdsweep]
  speech-distill       kdsweep_t4b                            recorded   [kdsweep]

  family kdsweep                          4 run(s)  (speech-distill)
  family rank-method                      4 run(s)  (retrieval-ablation)
  family planned                          1 run(s)  (retrieval-ablation)
[exit 0]
```

```text
$ attest runs compare kdsweep --metric wer
kdsweep — ranked by wer (lower_is_better), all arms on librispeech-100h

  arm                                                 wer      n      step  source
  -------------------------------------------- ---------- ------  --------  ------
  kdsweep_t4                                       0.0642   2620            speech-distill/results/kdsweep_t4.json
  kdsweep_t4b                                      0.0659   2620            speech-distill/results/kdsweep_t4b.json
  kdsweep_t2                                       0.0688   2620            speech-distill/results/kdsweep_t2.json
  kdsweep_baseline                                 0.0731   2620            speech-distill/results/kdsweep_baseline.json

winner: kdsweep_t4
  caveat: the top two arms differ by 0.0017 (2.6%) -- too close to call from these numbers alone
  caveat: each arm is a single run; no seed replication, so this ranking cannot separate configuration from run-to-run variance
[exit 0]
```

```text
$ attest runs scan --root .
  retrieval-ablation           5 run(s)
  speech-distill               4 run(s)
9 run(s) across 2 project(s)
[exit 0]
$ attest runs compare rank-method --metric n_records
unknown direction for metric 'n_records' -- refusing to rank. Declare it under [metric_direction] in <tmp>/hermes/metric_direction.toml; guessing would rank ablation arms backwards.
[exit 1]
```

## Make your own claim

`make-your-own-claim.ipynb`

```text
Working in a temporary folder with a temporary ledger.
```

```text
Run A reaches an accuracy of 0.912.
<!-- claim: mywork/run_a metric=accuracy value=0.912 tol=0.001 -->
```

```text
$ attest runs scan --root .
  mywork                       1 run(s)
1 run(s) across 1 project(s)
[exit 0]
$ attest claims mywork/notes.md
  supported     notes.md:2                   accuracy=0.912  accuracy=0.912 in run_a

1 claim(s): 1 supported
[exit 0]
```

```text
Run A reaches an accuracy of 0.931.
<!-- claim: mywork/run_a metric=accuracy value=0.931 tol=0.001 -->

$ attest claims mywork/notes.md
  contradicted  notes.md:2                   accuracy=0.931  document says 0.931, run records 0.912 (tolerance 0.001)

1 claim(s): 1 contradicted
[exit 1]
```

```text
$ attest claims mywork/notes.md
  unsupported   notes.md:2                   accuracy=0.912  no run mywork/run_b in the ledger -- the claim may be true, but nothing here backs it

1 claim(s): 1 unsupported
[exit 0]
```
