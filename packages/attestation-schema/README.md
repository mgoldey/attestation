# attestation-schema

The core run record that experiment producers write and
[attestation](https://github.com/mgoldey/attestation) reads:
`attestation.run/1`, **v1-DRAFT (not frozen)**. Standard library only.

```python
from attestation_schema import RunRecord, Producer, CodeIdentity, Metric, write, read

record = RunRecord(
    run_id="run-3f2a9c01b7de", family="lr-sweep", arm="lr=1e-3", status="complete",
    started_at="2026-11-19T10:00:00Z", finished_at="2026-11-19T10:42:13Z",
    producer=Producer(name="train.py", version=None),
    code=CodeIdentity(git_sha=None, dirty=None, repo=None),
    config={"lr": 0.001}, data=[],
    metrics=[Metric(name="loss", value=0.25, direction="min", step=1000, split="eval")],
    seeds={"python": 0}, dependencies=[], ext={},
)
write(record, "runs/run-3f2a9c01b7de")   # -> runs/run-3f2a9c01b7de/attestation/run.json
assert read("runs/run-3f2a9c01b7de") == record
```

- `dumps`/`write` are the only sanctioned writers: validate, then RFC 8785
  canonical JSON, so the same record is always the same bytes.
- `loads`/`read` refuse anything non-conforming with a `RecordError` naming
  the path (`metrics[2].direction`); nothing is coerced or defaulted. They
  dispatch on `schema`, so later versions can still read older records.
- `ext.<namespace>` carries producer data the core does not model, untouched.
- `attestation_schema.layout` fixes the run directory (`attestation/run.json`,
  `attestation/state.json`, `attestation/digests.json`, `steps/NN-<name>/`,
  `logs/`); `attestation_schema.jobs` is the job-backend `Protocol`.
- `attestation_schema/schemas/run-1.schema.json` is a JSON Schema for other
  languages: necessary, not sufficient — the Python reader is the reference.

Design: `docs/superpowers/specs/2026-11-19-core-run-record-design.md` in the
attestation repository.
