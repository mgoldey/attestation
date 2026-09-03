# Skill trigger transfer matrix: symbolic (2026-09-03)

Split `all`, 4 case(s). Primary model `gemma4:e2b`.

| candidate | gemma4:e2b | gemma4:e4b | hermes3:8b | spread |
|---|---|---|---|---|
| baseline | 0.750 | 0.250 | 0.000 | 0.750 |
| symbolic-trigger-2026-09-03 | 1.000 | 1.000 | 0.000 | 1.000 |

## Gate

- **symbolic-trigger-2026-09-03**: FAIL
  - beats the baseline on 1 other model(s) ['gemma4:e4b']; needs 2
  - spread across models 1.000 is wider than the baseline's 0.750
