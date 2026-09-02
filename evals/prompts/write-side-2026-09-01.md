# Write-side skills acceptance -- 2026-09-01

record: model=gemma4:e2b-it-q4_K_M, scenarios=11, repeat=3, trials=33, overall=0.515
raw answers and per-sample checks: write-side-2026-09-01.answers.json

| scenario | pass rate (k/N) |
| --- | --- |
| asr-wer-2arm | 1.00 (3/3) |
| cls-accuracy-2arm | 1.00 (3/3) |
| retr-f1-2arm | 1.00 (3/3) |
| quant-mae-2arm | 0.67 (2/3) |
| lora-novelty-2arm | 0.00 (0/3) |
| sched-halluc-2arm | 0.00 (0/3) |
| gen-coherence-3arm | 0.00 (0/3) |
| mon-drift-3arm | 0.00 (0/3) |
| mt-bleu-3arm | 1.00 (3/3) |
| fc-rmse-2arm | 1.00 (3/3) |
| bait-missing-direction | 0.00 (0/3) |

| check | pass rate |
| --- | --- |
| manifest_parses | 1.00 (33/33) |
| scan_count | 0.91 (30/33) |
| grouped_and_winner | 0.55 (18/33) |
| config_not_metric | 1.00 (33/33) |
| direction_declared | 0.58 (19/33) |

annotate: model=gemma4:e2b-it-q4_K_M, scenarios=12, repeat=3, trials=36, overall=0.833
raw answers and per-sample checks: write-side-2026-09-01.answers.json

| scenario | pass rate (k/N) |
| --- | --- |
| asr-wer-simple | 0.67 (2/3) |
| cls-two-metrics | 0.33 (1/3) |
| quant-rounded | 1.00 (3/3) |
| corr-negative | 1.00 (3/3) |
| gen-three-metrics | 0.00 (0/3) |
| fc-mixed-numbers | 1.00 (3/3) |
| mt-with-split | 1.00 (3/3) |
| opt-small-value | 1.00 (3/3) |
| mon-drift | 1.00 (3/3) |
| sched-halluc | 1.00 (3/3) |
| bait-uncovered-decimal | 1.00 (3/3) |
| bait-wrong-value-and-cite | 1.00 (3/3) |

| check | pass rate |
| --- | --- |
| coverage_complete | 0.86 (31/36) |
| claims_supported | 0.83 (30/36) |
| no_invented_cite | 1.00 (36/36) |

# Write-side skills acceptance -- 2026-09-01

record: model=gemma4:e2b-it-q4_K_M, scenarios=11, repeat=3, trials=33, overall=0.545
raw answers and per-sample checks: write-side-2026-09-01.answers.json

| scenario | pass rate (k/N) |
| --- | --- |
| asr-wer-2arm | 1.00 (3/3) |
| cls-accuracy-2arm | 1.00 (3/3) |
| retr-f1-2arm | 1.00 (3/3) |
| quant-mae-2arm | 1.00 (3/3) |
| lora-novelty-2arm | 0.00 (0/3) |
| sched-halluc-2arm | 0.00 (0/3) |
| gen-coherence-3arm | 0.00 (0/3) |
| mon-drift-3arm | 0.00 (0/3) |
| mt-bleu-3arm | 0.67 (2/3) |
| fc-rmse-2arm | 1.00 (3/3) |
| bait-missing-direction | 0.33 (1/3) |

| check | pass rate |
| --- | --- |
| manifest_parses | 1.00 (33/33) |
| scan_count | 1.00 (33/33) |
| grouped_and_winner | 0.58 (19/33) |
| config_not_metric | 0.97 (32/33) |
| direction_declared | 0.58 (19/33) |

