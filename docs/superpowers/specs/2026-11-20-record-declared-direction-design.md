# Record-declared metric direction: rank by what the producer said

**Date:** 2026-11-20
**Status:** implemented 2026-11-20 on `next/record-direction` (follow-up found
by roadmap 13 and 27).
**Depends on:** `2026-08-12-run-ledger-design.md` ("Metric direction must be
declared"), `2026-11-19-core-run-record-design.md` (`metrics[].direction`:
`"min"` | `"max"` | `null`, null = the producer did not say, never inferred;
one direction per metric name per record),
`2026-11-19-ledger-reads-core-records-design.md` (which left `direction`
unused: "making a record's own direction outrank that table is a ranking
change for its own spec" -- this is that spec),
`2026-11-19-preregistration-design.md` (`add` refuses an undeclared metric).
**Not this:** inferring a direction from a metric name, a unit or a value;
changing the built-in table or `metric_direction.toml`; changing what a
legacy importer emits (still `direction: null`); a direction in claims (the
claim checker compares values, it never ranks).

## Problem

A native producer already says which way is better. smeltery writes
`paired_ddE.mean` with `direction: "min"`; the HF/Lightning callbacks write a
direction wherever the trainer declared one; the MLflow exporter writes one
under `--direction`. The ledger ignored all of it: `runs.compare` and
`attest runs compare` read directions only from the built-in table and
`metric_direction.toml` (`LEDGER_METRIC_DIRECTION_FILE`), so a family of
smeltery runs refused to rank until a person transcribed into a side file
what the records already carried. That is the hand-transcription the ledger
exists to remove, and the side file is a second source of truth that can
silently disagree with the first.

## Decision 1: precedence -- the declaration wins, a disagreement is said

Three sources can speak, and the order is:

1. **The declaration** -- the built-in `METRIC_DIRECTION` table merged with
   `metric_direction.toml`, matched by stem exactly as before. When it names
   a direction, that direction is used, `direction_source` is `"declared"`.
   If any record that reports the metric declares the *other* direction, the
   comparison carries a caveat naming those runs (≤5 names, `+N more`), first
   among its caveats because it bears on the ranking itself. Records that are
   silent (null) are not a disagreement.
2. **The records** -- with no declaration, the direction is used only when
   EVERY run in the family that reports a stored value for the metric
   declares it, and they all declare the same one (`min` ->
   `lower_is_better`, `max` -> `higher_is_better`). `direction_source` is
   `"records"`.
3. **Neither** -- refuse, exactly as before. When records were involved the
   refusal says why they could not decide: they disagree (`a, b declare min;
   c declares max`) or only some declare (`a declares min; c declares
   none`). Then the same remedy sentence as today: declare it under
   `[metric_direction]` in the path.

Why the declaration outranks the record rather than the other way round:

- **It is the only source a reader of the ledger controls.** A record is
  frozen once written (`attestation.run/1` is frozen, run.json is bound by
  digests); if a producer declared the wrong direction, the TOML is the one
  place a person can correct it without rewriting artifacts. Letting the
  record win would make a wrong record uncorrectable short of re-running.
- **The built-in table names only metrics with one direction by
  definition** (WER, loss, accuracy). A record that disagrees with it is far
  likelier wrong than the table, and the caveat makes the conflict visible
  instead of picking silently in either direction.
- **Nothing that ranks today changes.** Every family the ledger could rank
  before ranks the same way after; the only behaviour change is that families
  it refused can now rank, plus a caveat where records contradict a
  declaration.

Why unanimity and not a majority among records: a family whose runs disagree
about which way is better is not a ranking problem with a vote, it is a
producer bug or two different metrics under one name. A partly declared family
is the same: the silent runs did not say, and treating silence as consent is
the inference the core schema forbids ("null = the producer did not say;
never inferred"). Refusing with the run names gives the person the exact
runs to look at.

Never from the name: the record's word is a declaration made by the program
that computed the value. A name, a unit or a value range is not, and nothing
here looks at them.

## Decision 2: storage -- migration 017, `run_metrics.direction`

The direction is a property of a metric as one run reported it, so it lives
beside the value: `ALTER TABLE run_metrics ADD COLUMN direction TEXT`,
holding the record's word verbatim (`min` / `max` / NULL). Considered and
rejected:

- **Recompute from the record at compare time.** compare reads the DB, not
  the files; reopening every run.json per comparison breaks the ledger's
  contract that the scan is the only reader of artifacts (and `runs.compare`
  works on a DB whose workspace is not mounted).
- **`runs.directions_json`.** A per-run map duplicates the metric name
  outside `run_metrics`, and the metric-discovery query (`GROUP BY run_id,
  metric`) would need a second read to join it back. A column on the row it
  qualifies is readable in Datasette as-is.
- **`caveats_json`.** A direction is data, not a caveat.

A record carries one direction per name, so every row of a (run, metric) has
the same value; the discovery query takes `MAX(direction)` per group, which is
that one value. Columns only: the table count (24) and the fresh-file
`sqlite_master` count (35) are unchanged. A DB scanned before 017 has NULL
everywhere and behaves exactly as before until re-scanned -- the rule
migrations 015 and 016 follow (the values come from the records, not the DB).

A metric whose value is NaN/inf is not stored (existing rule), so its
direction is not stored either: that run does not "report" the metric for
ranking, so it neither votes nor dissents.

## Decision 3: surfacing the source

`compare()`'s result gains `direction_source: "records"` **only** when the
records supplied the direction. When the declaration did, the key is absent,
so every family that ranked before returns exactly the keys it always had
(`test_single_run_family_output_is_unchanged` pins kdsweep's key set; the
golden paths pin CLI lines). This mirrors `separable`/`effect_size`, which
appear only when some arm pooled. The CLI header reads
`<family> — ranked by <metric> (<direction>, declared by its records)` in that
case and is unchanged otherwise. The MCP payload is the dict as returned
(one ~30-char key), well inside the 7000-char budget (pinned in
`test_response_size.py`).

Metric discovery (`_pick_metric`, compare with no metric named) counts a
metric as directed when *either* source resolves it, so a smeltery family
with no TOML picks `paired_ddE.mean` by itself. Metrics whose records
contradict each other are not candidates.

## Decision 4: preregistration still requires a declaration

`attest prereg add` keeps refusing a metric the declaration does not name,
with `unknown_direction_message`'s sentence, unchanged. Reasons:

- **A registration precedes its runs.** At `add` time there are normally no
  records to read; consulting the ledger would accept a hypothesis only when
  its evidence already exists -- exactly the registration prereg flags as
  `post_hoc`. A guard that passes only for post-hoc registrations is worse
  than none.
- **The verdict is direction-independent.** `--expect "a < b"` compares raw
  values, so the direction never decides the verdict; the guard only promises
  the metric will be rankable. Accepting "some future record will declare it"
  is a promise nothing can check at registration.
- **Evaluation does use records.** `prereg check` and the `hypotheses` block
  read `ledger.compare`, which now ranks by records, so a registration made
  with a TOML declaration still evaluates if the TOML later goes away and the
  records declare the metric.

A user preregistering a smeltery metric therefore declares it once in the
TOML; the records then agree with it (no caveat) or the disagreement caveat
fires.

## What does not change

- Legacy readers (generic, sacred, dvc, wandb, mlflow store, hydra) emit
  `direction: null`; their families rank exactly as before.
- `attest runs record` / `runs.record` write legacy result files and keep
  requiring a declared direction (`--direction` or the table).
- `runs.ask`'s metric-in-question matcher still matches names from the
  declaration only (a record-only metric is picked by `_pick_metric`'s count
  instead). Noted as a gap, not fixed here.

## Tests

`tests/test_record_direction.py`: native records declaring `min` rank with no
TOML (`direction_source == "records"`, CLI header says so); a TOML that
agrees ranks with no caveat and no `direction_source`; a TOML that disagrees
ranks by the TOML with a caveat naming the runs; records that contradict each
other, or partly declare, refuse with the run names; legacy (null) families
are unchanged; discovery picks a record-directed metric; migration 017 adds
the column to an old DB; prereg `add` still refuses a record-only metric while
`check` evaluates through record directions; the MCP payload stays under the
budget. `tests/test_smeltery_records.py` gains the smeltery fixture ranking
with no TOML.
