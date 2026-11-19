# attestation-schema

The core run record that experiment producers write and
[attestation](https://github.com/mgoldey/attestation) reads:
`attestation.run/1`, **v1-DRAFT (not frozen)**. Standard library only.

```python
from attestation_schema import (
    RunRecord, Producer, CodeIdentity, Metric, HistoryWriter, write, read, read_history,
)

run_dir = "runs/run-3f2a9c01b7de"
with HistoryWriter(run_dir) as hist:            # streams attestation/history.jsonl
    loss = hist.series("loss", split="eval")
    for step, value in [(0, 0.9), (500, 0.4), (1000, 0.25)]:
        hist.point(loss, step, value)           # at=format_timestamp(...) if known

record = RunRecord(
    run_id="run-3f2a9c01b7de", family="lr-sweep", arm="lr=1e-3", status="complete",
    started_at="2026-11-19T10:00:00Z", finished_at="2026-11-19T10:42:13Z",
    producer=Producer(name="train.py", version=None),
    code=CodeIdentity(git_sha=None, dirty=None, repo=None, version=None,
                      provenance="unknown", evidence=None),
    config={"lr": 0.001}, data=[],
    metrics=[Metric(name="loss", value=0.25, direction="min", step=1000, split="eval",
                    unit=None, labels={})],
    history=hist.ref,      # {path, sha256, series, points}; None = no curve recorded
    seeds={"python": 0},
    dependencies=None,     # None = could not observe the environment; [] = observed, none
    ext={},
)
write(record, run_dir)     # -> runs/run-3f2a9c01b7de/attestation/run.json
assert read(run_dir) == record          # never opens history.jsonl
curves = read_history(run_dir)          # verifies the sha256, then parses
```

- `dumps`/`write` are the only sanctioned writers: validate, then RFC 8785
  canonical JSON, so the same record is always the same bytes.
- `loads`/`read` refuse anything non-conforming with a `RecordError` naming
  the path (`metrics[2].direction`); nothing is coerced or defaulted. They
  dispatch on `schema`, so later versions can still read older records.
- **Curves never go in `run.json`.** `metrics` holds final values; logged
  curves go to the sidecar `attestation/history.jsonl` (`attestation.history/1`:
  JSON Lines, one canonical line per series declaration or point, streamable
  and deterministic), written with `HistoryWriter` (incrementally, finalized
  atomically on close) or `write_history(run_dir, series)`, and bound to the
  record by sha256. `read_history` raises `HistoryMissingError`,
  `HistoryDigestError` or `HistoryFormatError` (all `RecordError`). Write
  order: outputs and history -> `layout.write_manifest` -> `write`.
- `ext.<namespace>` carries producer data the core does not model, untouched.
- `attestation_schema.layout` fixes the run directory (`attestation/run.json`,
  `attestation/state.json`, `attestation/digests.json`,
  `attestation/history.jsonl`, `steps/NN-<name>/`, `logs/`);
  `attestation_schema.jobs` is the job-backend `Protocol`.
- `attestation_schema/schemas/run-1.schema.json` (the record) and
  `history-1.schema.json` (one sidecar line) are JSON Schemas for other
  languages: necessary, not sufficient — the Python reader is the reference.

Design: `docs/superpowers/specs/2026-11-19-core-run-record-design.md` in the
attestation repository.
