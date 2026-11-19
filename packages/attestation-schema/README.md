# attestation-schema

The core run record that experiment producers write and
[attestation](https://github.com/mgoldey/attestation) reads:
`attestation.run/1`, **frozen 2026-11-19**: a record valid today stays valid,
and any change -- an additive optional field included -- is a new version
(`attestation.run/2`) that readers dispatch on. Standard library only.

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
  `attestation_schema.jobs` is the job-backend `Protocol`, and
  `attestation_schema.state` is what every backend shares: `state.json`
  (`attestation.state/1`: `read_state`, `init_state`, `transition`), its
  cross-process lock (`state_lock`, POSIX `flock`; refused on Windows), and
  `settle` -- end a job exactly once, an existing record deciding, writing
  `terminal_record` for a job that died without one. A job's record carries
  its `JobSpec`'s `run_id`, `family`, `arm`, `config` and `ext` verbatim.
- `attestation_schema/schemas/run-1.schema.json` (the record),
  `history-1.schema.json` (one sidecar line) and `state-1.schema.json`
  (`state.json`) are JSON Schemas for other languages: necessary, not
  sufficient — the Python reader is the reference. They are frozen and
  pinned by digest.
- Package version 0.1.0: the formats are frozen; the Python API may still
  change in a 0.x minor release until the job-state API has a consumer.
 
## Conformance

The reader stops at the first problem in `run.json`; a run directory has more
rules than that. The conformance suite checks all of them and reports every
violation with the file and the JSON path it is in: the record reads under
the frozen reader; its bytes are the canonical ones (`write` round-trips them
byte for byte); `digests.json` matches the files and is bound by the record;
the history sidecar matches its digest and format; `state.json` is
well-formed with legal transitions and agrees with the record; a terminal
state has its record; the record matches its `JobSpec` (when you pass one);
every `dependencies[].source` is in the vocabulary; no credentials anywhere
in the schema-owned files; every `ext` key is a valid namespace.

Run it in your CI -- three lines:

```
pip install "attestation-schema[conformance]"
pytest -p attestation_schema.pytest_plugin --attestation-runs 'runs/*'
```

One test per run directory (marker `attestation_conformance`). A pattern
that matches no run directory **fails**: a typo never turns the suite into a
green no-op. Or configure it once:

```toml
[tool.pytest.ini_options]
addopts = "-p attestation_schema.pytest_plugin"
attestation_runs = ["runs"]          # used when pytest runs without explicit paths
```

A failure names the record and the path inside it:

```
runs/r1/attestation/run.json at $.config.tracking_uri: [credentials] a URL with credentials (user:password@); strip them
```

To check a record your test just produced (and, for a job, the spec it ran
with), use the fixture:

```python
def test_my_producer_writes_a_conforming_record(tmp_path, attestation_conformance):
    run_my_pipeline(out=tmp_path)
    attestation_conformance(tmp_path, spec)        # spec: the JobSpec, or omit it
```

Without pytest: `attestation_schema.conformance.check_run_dir(run_dir)`
returns a `Report` (`.ok`, `.violations`, `.format()`), and
`assert_conformant(run_dir)` raises `ConformanceError`. The plugin is
**opt-in** (`-p`, no `pytest11` entry point), so a suite that merely has this
package installed is unchanged, and pytest is only the `[conformance]`
extra's: the package itself still depends on nothing. attestation's ledger
runs the same checks on every scan and quarantines any record that fails.
Golden fixtures (a conforming record for each shape, and one record broken
per rule) are in `tests/data/conformance/`.

## Releasing

Tag `schema-v<version>` (the version in this package's `pyproject.toml`, which
the workflow checks), independent of attestation's `v<version>` tags:

```
git tag schema-v0.1.0 && git push origin schema-v0.1.0
```

`.github/workflows/release.yml` builds the sdist and wheel, runs this
package's tests on CPython 3.11 against the wheel, and publishes to PyPI
through trusted publishing (environment `pypi`, no token). Release it before
an attestation release whose `attestation-schema>=X` floor is new.

Design: `docs/superpowers/specs/2026-11-19-core-run-record-design.md` in the
attestation repository.
