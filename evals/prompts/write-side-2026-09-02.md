# Write-side skills acceptance -- 2026-09-02

annotate: model=gemma4:e2b-it-q4_K_M, scenarios=12, repeat=3, trials=36, overall=0.889
raw answers and per-sample checks: write-side-2026-09-02.answers.json

| scenario | pass rate (k/N) |
| --- | --- |
| asr-wer-simple | 1.00 (3/3) |
| cls-two-metrics | 0.33 (1/3) |
| quant-rounded | 1.00 (3/3) |
| corr-negative | 1.00 (3/3) |
| gen-three-metrics | 0.67 (2/3) |
| fc-mixed-numbers | 1.00 (3/3) |
| mt-with-split | 1.00 (3/3) |
| opt-small-value | 1.00 (3/3) |
| mon-drift | 1.00 (3/3) |
| sched-halluc | 1.00 (3/3) |
| bait-uncovered-decimal | 0.67 (2/3) |
| bait-wrong-value-and-cite | 1.00 (3/3) |

| check | pass rate |
| --- | --- |
| coverage_complete | 0.91 (32/35) |
| claims_supported | 0.91 (32/35) |
| no_invented_cite | 1.00 (35/35) |

