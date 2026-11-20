# smeltery's runs in the ledger: no new reader, a provenance caveat for every producer

**Date:** 2026-11-20
**Status:** implemented 2026-11-20 (roadmap item 13).
**Depends on:** `2026-11-19-ledger-reads-core-records-design.md` (the `core`
adapter, `runs.caveats_json`, `ext.ledger-import`),
`2026-11-19-core-run-record-design.md` (Producer findings 2: the smeltery
mapping; `code` and `dependencies[]` on one provenance ladder),
`2026-11-19-dependency-identity-design.md` (what `verified` means, and
ferric's `__build__` stamp).
**Not this:** ranking by a record's own metric `direction` (the ledger still
reads directions from `metric_direction.toml`; a record's direction
outranking that table is a ranking change for its own spec), anything
chemistry-shaped in the ledger, a reader for smeltery's legacy
`run-<digest12>.json`.

## Problem

The roadmap item reads: "smeltery adapter. Funnel tiers to `ext.smeltery`,
ferric `__build__` to a dependency identity. Unverified or dirty provenance
is a caveat in `attest claims`, never a contradiction."

The first two halves are already done, by smeltery. `smeltery.core_record`
writes `<OUT>/attestation/run.json` through `attestation_schema.write` on
every run with its `attestation` extra installed (and `smeltery record core
OUT` converts a run made before that): the funnel's tiers sit under
`ext.smeltery.tiers`, ferric is a `dependencies[]` entry whose `build_sha`,
`dirty` and `provenance: "verified"` come from its `__build__` stamp, and the
ledger's `core` adapter imports the run directory with no conversion. Checked
on smeltery's own prebuilt runs (below): `attest runs scan` reads both as
`complete` runs of family `mwe-benzoic-cli`, with every funnel count and
paired ddE in `run_metrics`.

What is missing is the third half, and it is not smeltery's to fix. The
ledger drops `code` and `dependencies` on the floor ("the ledger's tables
have no place for them"), so a run whose code was a dirty checkout, or whose
`rdkit` the record could not identify, ranks and supports claims exactly like
one whose every identity is verified. smeltery's prebuilt runs are both: their
`code` is `inferred` and `dirty: true` (an editable checkout with uncommitted
changes), and `rdkit`/`numpy` are `unknown` (the legacy record never said
which). Nothing in `runs.compare`, `runs.detail` or `attest claims` says so.

A second gap surfaced only when a real smeltery record went through the
claim checker: `claims._parse_body` lowercases the claimed metric name (the
legacy readers store keys lowercased), and the lookup was case-sensitive, so a
claim on smeltery's `paired_ddE.mean` could never find its row -- every one
read `unsupported`. Core records keep a metric's name as written.

## Decision 1: no smeltery reader

The `core` adapter is the smeltery adapter. A tracker-specific reader for the
legacy `run-<digest12>.json` was considered and rejected:

- smeltery owns the mapping (Producer findings 2), and a second implementation
  in attestation is the drift the core record exists to remove -- the same
  argument that made the MLflow adapter mirror the exporter line for line.
- The only legacy-only run directories found are smeltery's five `prebuilt/`
  demos, which `smeltery record core OUT` converts in one command; every run
  smeltery makes now writes its record itself. A run made before the
  `attestation` extra existed is invisible to the ledger and `attest runs scan`
  says why (`no runs in <project>: ... none under a recognised results
  directory`). Converting it is smeltery's command, not a guess here.
- attestation must not import smeltery (`tests/test_boundaries.py`), so a
  reader could not reuse smeltery's mapping; it would re-derive it.

## Decision 2: provenance findings, schema-driven, stored with the run's caveats

A pure function, `run_identity.provenance_findings(record)`, reads only the
frozen schema's fields, so it applies to every producer that writes a core
record -- smeltery, the MLflow exporter, the HF/Lightning callbacks -- and
names none of them:

| record says | finding |
|---|---|
| `code.provenance` `inferred` / `unknown` | `code inferred` / `code unknown` |
| `code.dirty: true` | `code dirty` (folded: `code inferred, dirty`) |
| `dependencies: null` | `dependencies not recorded` |
| a dependency `inferred` / `unknown` | `dependencies inferred: a, b` / `dependencies unknown: c` (names sorted) |
| a dependency `dirty: true` (any provenance) | `dependencies dirty: d` |

`verified` and clean says nothing, and `dirty: null` says nothing: the schema
spells "not observed" as null and the provenance level already says how much
is known, so a null `dirty` beside a `verified` identity is not a second
finding. A ferric build stamped `dirty: true` is `verified` (the stamp names
the build exactly) *and* a finding (the build came from uncommitted source).

**Native records only.** A record carrying `ext.ledger-import` (a legacy
artifact converted in memory by an importer) gets no findings: its `code`
and `dependencies` are the converter's "did not say", identical for every run
that reader produces, so a per-run caveat would be a property of the reader
repeated once per arm -- the boilerplate that `_corpus_agreement`'s comment
warns trains readers to skip caveats. The two adapters that do read an
identity (the MLflow store's git tag, `inferred`) are importers too, and their
reader caveat already says what they read. A record a producer wrote is the
producer's statement about the run, so its gaps are the run's.

**Stored, not recomputed.** The ledger keeps no record after a scan, so the
findings ride in `runs.caveats_json` beside `reader` and `run`, under
`provenance` (a list of the phrases above). No migration: the column is JSON
and a missing key reads as `[]`. A run scanned before this change shows no
findings until it is re-scanned, as with migration 015.

## Decision 3: where the findings surface -- always as a caveat, never a verdict

- **`runs.detail`** (and `attest runs show`): a `provenance` list beside
  `caveats`, printed as one `provenance:` line when non-empty.
- **`runs.compare`**: one caveat per distinct set of findings, naming the runs
  that share it (at most five, then `and N more`), after the runs' own caveats
  and before the reader caveats: `provenance not verified for a, b: code
  inferred, dirty; dependencies unknown: numpy, rdkit -- the values stand, but
  which code and environment produced them is not established (runs.detail
  lists each run's)`. Grouping keeps a pooled arm of twenty seed replicates
  from contributing twenty identical lines. Ranking is unchanged.
- **`attest claims` / `runs.claims_check`**: a `Verdict` gains `caveats`, the
  matched run's own caveats (`caveats_json.run`: did not complete, a
  non-finite value) and its provenance caveat. The verdict is computed
  exactly as before: a claim whose number agrees with a dirty run is
  `supported`, with the caveat beside it; one whose number disagrees is
  `contradicted` because of the number, never because of the provenance. The
  CLI prints each caveat on its own `caveat:` line under the verdict, the exit
  status ignores them, and the pinned `N claim(s): ...` summary is untouched.
  The MCP row carries `caveats` on every row (empty list when none) so its
  shape does not change between verdicts. Only a single matched run carries
  caveats: an `unsupported` or `ambiguous` verdict has no one run to qualify.

Rejected: a new verdict (`unverified`), or demoting `supported` to `stale`.
A verdict answers "does the document agree with the run"; provenance answers
"how sure are we which code produced the run". Folding the second into the
first makes a true number read as a problem with the document, and the
roadmap says it outright: a caveat, never a contradiction.

## Decision 4: `ext.<namespace>` is not surfaced

The ledger reads no producer's namespace and lists none. `ext.smeltery` holds
the funnel's tiers, inputs and input digest -- chemistry the ledger cannot
interpret, and the core spec's reason for `ext` is exactly that the core does
not model it. Listing the namespace names in `runs.detail` was considered:
it tells a reader nothing they can act on that `source_path` (the run
directory, whose `attestation/run.json` holds the namespace) does not already
give them. A tool that reads one producer's namespace is that producer's.

## Decision 5: claim metric names match case-insensitively

`check_claim` compares `lower(metric)` to the claim's (already lowercased)
metric. Legacy rows are stored lowercase, so nothing they match changes. A
run that records two metrics differing only in case now yields two rows for
one claim, which is `ambiguous` with both listed -- the checker's existing
rule for more than one candidate row, never a silent pick.

## Fixture: smeltery's own runs, written by smeltery

`tests/data/smeltery/workspace/mwe-benzoic/{benzoic-rhf,benzoic-pbe}/` are
smeltery's prebuilt `benzoic-rhf` and `benzoic-pbe` runs (one campaign, two
methods: RHF and PBE) with the core record `smeltery record core` writes.
`tests/data/smeltery/generate.sh` regenerates them from a smeltery checkout
(`SMELTERY=/path/to/smeltery tests/data/smeltery/generate.sh`, with
`uv run --isolated --frozen --extra attestation`, so smeltery's own lockfile
picks the schema version and the checkout's venv is not touched). Running it
twice gives identical bytes. The one edit made: the legacy record names the
machine it ran on (`host.node`) and a checkout path under `/home/`; both are
scrubbed in the *input*, before smeltery writes, so the manifest and every
`data[].sha256` are smeltery's own over the scrubbed bytes and conformance
still passes. The records are never edited after smeltery writes them.

They are under `tests/data/`, not `examples/`: a golden path needs a seven-
section README and a pinned output, and this is a regression fixture, not a
walkthrough.

## Testing

`tests/test_smeltery_records.py`:

- the fixture is what smeltery wrote (conformance passes on both runs; producer
  name; the scrub left no `/home/`);
- `attest runs scan` imports both runs through the `core` adapter, with no
  quarantine, as family `mwe-benzoic-cli`, labelled metrics folded into split;
- `runs.detail` lists the findings (`code inferred, dirty`, `dependencies
  unknown: numpy, rdkit`), and ferric's verified clean stamp contributes none;
- `runs.compare` on `paired_ddE.mean` (direction declared in a test
  `metric_direction.toml`) emits ONE grouped provenance caveat naming both runs;
- a document with a claim that agrees and one that disagrees: `supported` and
  `contradicted` as before, both carrying the provenance caveat, exit status
  1 only for the contradiction; the same through `runs.claims_check`.

`tests/test_run_identity.py`: the findings table row by row on records built
with `attestation_schema.from_dict`, a fully verified clean record yields
none, an importer's record yields none, and the module names no tracker or
product (`tests/test_tracker_names.py` already scans it).

Existing suites are the behaviour-preservation evidence: legacy imports get no
findings, so every golden path and every pinned caveat list is unchanged.
