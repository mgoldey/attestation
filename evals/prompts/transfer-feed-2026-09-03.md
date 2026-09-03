# Skill trigger transfer matrix: feed (2026-09-03)

Split `all`, 7 case(s). Primary model `gemma4:e2b`.

| candidate | gemma4:e2b | gemma4:e4b | hermes3:8b | spread |
|---|---|---|---|---|
| baseline | 0.500 | 0.500 | 0.286 | 0.214 |
| feed-trigger-2026-09-03 | 0.786 | 0.786 | 0.214 | 0.571 |

## Gate

- **feed-trigger-2026-09-03**: FAIL
  - beats the baseline on 1 other model(s) ['gemma4:e4b']; needs 2
  - spread across models 0.571 is wider than the baseline's 0.214
