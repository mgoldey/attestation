# A conformance suite with teeth

**Date:** 2026-11-19
**Status:** implemented 2026-11-19 on `next/conformance` (roadmap item 15).
**Depends on:** `2026-11-19-core-run-record-design.md` (the frozen formats and
their compatibility policy), `2026-11-19-ledger-reads-core-records-design.md`
(the import pipeline whose refusals this makes loud).
**Changes no format.** `attestation.run/1`, `attestation.history/1`,
`attestation.state/1` and `attestation.digests/1` stay exactly as frozen;
`tests/test_frozen.py` is untouched. Everything here is new Python API in
`attestation-schema` (a 0.x minor addition, allowed by the policy) and new
behaviour in attestation's ledger. Format problems the suite surfaced are
listed under "v2 candidates", not fixed.

## Problem

The reader (`attestation_schema.read`) is strict, but it is only one of the
rules a run directory must keep, and nothing runs the rest anywhere a
producer would notice:

- A record can read and still not be what `write` would have written
  (pretty-printed, keys out of order, a trailing newline). Its sha256 is
  then not its `record_digest`, so its identity is ambiguous.
- `digests.json` can disagree with the files it lists, or not be bound by
  the record at all; `history.jsonl` can be bound by a digest it no longer
  has; `state.json` can say `failed` while no record exists, or `complete`
  while the record says `failed`. The reader opens none of these files.
- Credentials are refused in `code.repo` and `dependencies[].source` only.
  A tracking URI with `user:password@` in `config` reads fine.
- Each producer would have to re-implement all of that in its own CI, and
  would get it subtly wrong.

On the observer side, the ledger's core adapter refuses records the reader
refuses, but **quietly**: the refusal lands in `scan()`'s `diagnostics`, and
`attest runs scan` prints diagnostics only for projects that yielded *no*
runs. A project with nine good records and one refused one printed
`9 run(s)` and exit 0 -- the refusal was visible only to someone who read the
JSON of the MCP envelope. Measured on this branch before the change: the
`test_a_record_the_reader_refuses_is_a_diagnostic_not_a_crash` scenario,
driven through `attest runs scan`, printed `p  1 run(s)` and exited 0.

## Decision 1: a stdlib checker, and a pytest plugin over it

Two modules, so the checker stays usable without pytest:

- `attestation_schema.conformance` -- **standard library only**, like the
  rest of the package. `check_run_dir(run_dir, *, spec=None,
  verify_outputs=True) -> Report`, `assert_conformant(...)` (raises
  `ConformanceError`, an `AssertionError`), `find_targets(patterns, base)`
  and `RULES`. A unittest-based producer, or a release script, calls these
  directly.
- `attestation_schema.pytest_plugin` -- the **only** module in the package
  that imports pytest. It turns targets into test items and offers the
  fixture. pytest becomes an optional extra, `attestation-schema
  [conformance]` (`pytest>=7.2`, the release that added `config.args_source`,
  which the plugin reads); the core install stays dependency-free and the
  boundary test (`tests/test_boundaries.py`) is amended to allow exactly
  that one module and that one extra, nothing else.

### Registration: opt-in `-p`, not a `pytest11` entry point

Chosen: **opt-in**. A producer loads it with `-p
attestation_schema.pytest_plugin` (on the command line, in `addopts`, or
as `pytest_plugins` in its root `conftest.py`). Rejected: a `pytest11`
entry point, because pytest loads every installed entry-point plugin into
every test session of that environment, and the package is installed in
environments whose suites have nothing to do with records (attestation's
own users, a smeltery worker image, anything that depends on a producer):

- an auto-loaded plugin's import runs at every pytest start in that
  environment; a failure in it (a pytest API change, a broken install)
  breaks unrelated suites, which is the opposite of "must not change their
  behaviour";
- its fixture names and ini keys enter every suite's namespace (a producer
  with its own `attestation_conformance` fixture would be shadowed or
  shadow ours, depending on load order);
- the wheel's entry point exists whether or not the `[conformance]` extra was
  installed, so the plugin would be loaded where its own declared
  requirement might be unmet;
- the only cost of opt-in is one `-p` in the producer's CI config, which is
  also where a reader of that config learns the suite runs at all.

`packages/attestation-schema/tests/test_conformance_plugin.py` asserts the package declares no
`pytest11` entry point and that a session without `-p` has no such plugin,
option or fixture.

### Pointing it at runs

- `--attestation-runs PATTERN` (repeatable): a run directory, a directory
  holding run directories (`layout.find_runs`), a `run.json`/`state.json`
  path, or a glob (`**` allowed). Relative to the invocation directory.
- ini `attestation_runs` (line list): the same, relative to the ini file's
  directory. Used only when pytest was invoked **without** explicit paths
  (`config.args_source` is not `ARGS`), so `pytest tests/test_x.py` runs that
  file and nothing else; `--attestation-runs` on the command line always
  applies and replaces the ini list.
- A pattern that matches no run directory is a **failing** item
  (`no run directory matched 'runs/*'`), never zero items. A typo in the
  glob would otherwise turn the whole suite into a green no-op, which is
  the "measuring the fixture, not the system" failure this repo keeps
  recording (`docs/measurement-lessons.md`, 3 and 4).
- One item per run directory, named by its path relative to the rootdir,
  under a collector `attestation-conformance`, each carrying the marker
  `attestation_conformance` (registered by the plugin) so `-m` selects them.
  Items are added in `pytest_collection_modifyitems(tryfirst=True)`, before
  `-k`/`-m` deselection runs.

### The fixture

`attestation_conformance` returns a checker: `attestation_conformance(run_dir,
spec=None, verify_outputs=True)` fails the test with the full report when
anything is violated and returns the `Report` otherwise;
`attestation_conformance.report(run_dir, ...)` returns the report without
failing, for asserting on a deliberately broken directory. This is how a
producer generates a record in-test (`tmp_path`) and asserts it conforms,
and the only route by which a `JobSpec` reaches the `spec` rule: the spec is
an in-memory value the submitter holds, not a file in the run directory
(the core spec, 5.C.3: "no file is added to `attestation/`").

## Decision 2: the rules

Every check runs; a report lists every violation, not the first. Each
violation names its **rule**, the **file** it is in (an absolute path under
the run directory) and the **JSON path** inside it (`$.metrics[2].direction`,
`$.files["out/a.csv"]`, `line 17: $.value` for the sidecar), so the failure
message alone says where to look:

```
runs/r1/attestation/run.json at $.config.tracking_uri: [credentials] a URL with credentials (user:password@); strip them
```

| Rule | Checks | Notes |
|---|---|---|
| `read` | `run.json` parses strictly and validates under the frozen reader (`from_dict`). | A run directory with neither `run.json` nor `state.json` is a `read` violation. Reader errors whose path belongs to another rule (`dependencies[i].source`, an `ext` namespace, a credential in a URL) are reported under that rule instead. |
| `canonical` | The file's bytes are exactly `dumps(read(file))`: RFC 8785, no trailing newline. | Names the first differing byte. A record that reads but is not canonical has a sha256 that is not its `record_digest`. |
| `manifest` | If `digests.json` exists: strict JSON, exactly `{schema, files}`, `schema` `attestation.digests/1`, every digest 64 lowercase hex, canonical bytes; the record lists it in `data` with `role: "manifest"` and its sha256; with `verify_outputs`, `files` equals `layout.build_manifest(run_dir)` (each missing, extra or changed file is its own violation; a symlink is one). If the record names `attestation/digests.json` and the file is missing, that too. | A manifest is optional (the layout requires only `attestation/`, and `settle` omits one when a symlink refuses it). An unbound manifest binds nothing, so it is a violation, not a curiosity. |
| `history` | If the record's `history` is set: `read_history` succeeds (exists, sha256, every line, counts). If it is `null`: no `history.jsonl` exists. | A stray sidecar is a curve nobody bound. |
| `state` | If `state.json` exists: `read_state` succeeds (strict, exact keys, legal transitions, time order, last entry = state), its bytes are canonical, and its `run_id` is the record's. | |
| `terminal` | A terminal `state.json` has a `run.json` beside it; when both exist and the state is terminal, the record's `status` is that state; a job's record (a `state.json` exists) never has `status: null`. | `state.json` may lag, so a non-terminal state beside a record is allowed (the core spec: `run.json` is authoritative). |
| `spec` | With a `JobSpec`: `state.spec_conflicts(spec, record)` is empty. | Fixture / API only (see above). |
| `source` | Every `dependencies[i].source` is in the closed vocabulary. | Checked on the parsed JSON independently of the reader, so a record with three bad sources reports three. |
| `credentials` | No credential in any string of `run.json`, `state.json`, `digests.json` or `history.jsonl`: a URL with `user:password@` (or any userinfo on http(s)), a query parameter that carries one (`token`, `access_token`, `api_key`, `key`, `sig`, `signature`, `password`, `secret`, `X-Amz-Signature`, `X-Amz-Credential`, `X-Goog-Signature`, `X-Goog-Credential`), a known token shape (AWS access key id, GitHub, GitLab, Slack, Hugging Face, OpenAI-style `sk-`, a PEM private key), or a non-empty value under a secret-named key (`password`, `secret`, `api_key`, `access_token`, `private_key`, ... and `*_<those>`), unless it is a redaction (`***`, `REDACTED`, `<redacted>`, `${ENV_VAR}`). | "Anywhere" means the schema-owned files; a producer's own outputs (logs, `steps/`) are its business. Patterns are heuristics with false positives possible on exotic configs; a false positive is fixed by renaming or redacting the value, which costs less than a leaked key in a record whose digest is its identity and therefore cannot be edited. |
| `ext-namespace` | Every `ext` key matches the namespace grammar, is not `attestation`/`attestation.*`, and holds an object. | Checked independently of the reader, so every bad namespace is reported. |

Not rules: `metrics` vs history agreement (the core spec rejected it), and
uniqueness of `run_id` across directories (not required unique).

## Decision 3: the ledger quarantines, loudly, and fails the scan by default

- The core adapter runs `conformance.check_run_dir(run_dir,
  verify_outputs=False)` on every run directory holding a record, in place
  of the bare `read`. **Any violation quarantines the record**: it never
  reaches `runs`/`run_metrics`, and its path, rules and messages are
  returned. `verify_outputs=False` because rehashing every output of every
  run on every scan costs time proportional to the outputs (smeltery's
  runs carry receptor and pose files), while the ledger stores none of
  them; the manifest's own format and its binding to the record are still
  checked. Re-hashing the outputs is the plugin's job, in the producer's CI,
  where the outputs were just written.
- A legacy run the converter refuses (`found_to_core` raises) is quarantined
  the same way, under rule `read`.
- `ledger.scan` returns `quarantined: [{project, path, adapter, rules,
  reasons}]` beside `diagnostics` (which keeps its `N record(s) refused`
  text, so nothing reading it breaks).
- `attest runs scan` prints, after the per-project counts, a
  `QUARANTINED (N record(s) -- not imported)` section listing each path and
  each reason, **and exits 1**. `--strict/--no-strict` (default strict):
  `--no-strict` keeps the section and exits 0. Strict by default because the
  scan is what a script or CI job runs, and a scan that exits 0 while
  refusing records is exactly the silence this item removes; a person who
  wants the partial ledger anyway says so with one flag. The rows that did
  import are written either way (quarantine is per record, not per scan).
  `attest runs record --scan` follows the same default.
- MCP `runs.scan` adds `quarantined` to its envelope (the `empty=` failure
  envelope too) and leads its `message` with the count when non-zero; it
  does not raise, because the tool's caller is a model that should be told
  which records were refused, not handed a bare error.

## Decision 4: every lossy mapping is documented, and none is silent

What the import pipeline does to a record on its way into the ledger's
tables. "Visible" means a run caveat (stored in `runs.caveats_json`, shown
by `runs.detail`, emitted by `compare`) or the QUARANTINED section.

| Where | Mapping | Lossy? | Visible as |
|---|---|---|---|
| `ledger.row_from_core` | metric `labels` folded into `split` as `split,k=v,...` | yes: a split or label containing `,`/`=` makes the fold ambiguous | documented here and in `_metric_split`; no per-run caveat (it would repeat on every labelled run of every compare, and the information is preserved for the common case). |
| `ledger.row_from_core` | non-finite metric value not stored | yes | run caveat (existing). |
| `ledger.row_from_core` | `status: null` stored as `unknown` | no (the column's word for null) | run caveat (existing). |
| `ledger.row_from_core` | `status` `failed`/`cancelled` on a record that also carries `ext.ledger-import` | was silent: the ext's status word replaced it and the "did not complete" caveat was skipped | **fixed**: that caveat is now added whenever the record's own status is `failed`/`cancelled`, whatever ext says. |
| `ledger.row_from_core` | `config: {}` stored as NULL | no | documented. |
| `ledger.row_from_core` | `data`, `code`, `seeds`, `dependencies`, `history`, other `ext` namespaces not stored | not stored (no column), not changed | documented (ledger spec). |
| `trackers.found_to_core` | duplicate `(name, step, split)` metrics: first kept | was silent | **run caveat** naming each dropped duplicate. |
| `trackers.found_to_core` | non-finite float in `config`/corpus becomes its string (`"nan"`) | was silent | **run caveat** naming each JSON path. |
| `trackers.found_to_core` | unparseable start time, or a naive one from a tracker not known to write UTC, becomes `null` | was silent | **run caveat** quoting the text. |
| `trackers.found_to_core` | `split ""` becomes `null`; `config None` becomes `{}` | no (the readers never distinguished them) | documented. |
| legacy readers (`generic`) | non-finite values in result files are not read as metrics (`_numeric_items`, MLflow, Sacred) | yes ("not measured") | **LOSSY diagnostic** per project and reader (below). |

The last row is the one place attribution is coarser than a run. The readers
are ported unchanged (ledger spec) and cannot attribute a skipped leaf to
the run it came from without being rewritten, so each skip is recorded in a
scan-scoped collector (`generic.skipped_non_finite`, a `ContextVar`) and
`import_project` reports it per reader: `ledger.scan` returns `lossy:
{project: ["the generic reader skipped 3 non-finite value(s) (NaN/inf: not
measured, not stored): loss, ..."]}`, `attest runs scan` prints a `LOSSY
(<project>):` section, and MCP `runs.scan` returns `lossy`. It does not fail
the scan: a skipped NaN is the reader's documented interpretation, not a
non-conforming record. Per-run attribution is left to roadmap 16, which
rewrites those readers anyway.

## Decision 5: attestation runs its own suite

- `pyproject.toml`'s pytest `addopts` gains `-p
  attestation_schema.pytest_plugin`, and `attestation_runs =
  ["examples/mlflow-records/records"]`, so every `uv run pytest` (the
  pre-commit hook and CI's `gates` job) checks the four committed MLflow
  records as four items.
- CI's `schema-py311` job installs the wheel **with the extra**
  (`dist-schema/*.whl[conformance]`), runs the package tests (which now
  include the `pytester` golden-fixture tests) and runs the plugin against
  `examples/mlflow-records/records/*` from the installed wheel on 3.11 -- the
  exact command a producer runs.
- The wheel smoke job's "declares no dependencies" check becomes "declares
  no unconditional dependency": every requirement carries `extra ==`.
- `tests/test_mlflow_records.py` guards the wiring (the plugin is loaded
  in attestation's own session, the ini lists the records, ci.yml runs the
  plugin), because a deleted ini line would make the suite stop running
  with nothing going red.

## Golden fixtures

`packages/attestation-schema/tests/data/conformance/`:

- `valid/minimal/` -- a record and nothing else (no history, no manifest).
- `valid/job-complete/` -- a settled job: `steps/00-train/`, `logs/`,
  `history.jsonl`, `digests.json` bound in `data`, `state.json`
  `submitted -> running -> complete`, dependencies with every `source` kind
  family, two `ext` namespaces.
- `valid/in-flight/` -- `state.json` `running`, no record.
- `broken/<rule>/` -- one per rule except `spec`, each breaking exactly
  that rule (and the `spec` case reuses `valid/job-complete` with a
  conflicting `JobSpec` given through the fixture).

They are **built by code** (`tests/conformance_fixtures.py:build`) with the
package's own writers, and committed; a test rebuilds them into `tmp_path`
and asserts the committed tree is byte-identical, so they cannot drift from
the writers. The credentials fixture uses `ci:hunter2@example.invalid`, not a
real-looking token, so no secret scanner trips on the repository; the token
shapes are tested with strings assembled at runtime.

`tests/test_conformance_plugin.py` drives the plugin through `pytester`:
every valid fixture passes; every broken fixture fails with exactly its
rule, its file and its JSON path in the output; a pattern matching nothing
fails; the ini is ignored when paths are given and used otherwise; the
fixture API passes a fresh record and fails a broken one; `-m
attestation_conformance` selects the items; no `-p`, no plugin. Per the
measurement lessons (4), each broken fixture is also the mutant for its
rule's check: the test asserts the valid fixture produces no violation of
that rule *and* the broken one produces it, so a check that always passes
or always fails is caught.

## v2 candidates (format findings; nothing changed)

1. **The manifest is not required, and nothing says whether one was
   attempted.** A record without a `role: "manifest"` entry cannot be told
   apart from one whose manifest was refused (a symlink) other than by
   `state.json`'s message. v2 could require the entry with `sha256: null`
   meaning "attempted and refused". Not done: any change to what v1 accepts
   is v2.
2. **`digests.json` has no reader in the package.** Its format is frozen by
   prose and `build_manifest`, but there is no `read_manifest` refusing a
   malformed one; the conformance checker implements that validation. A
   `read_manifest` is API, not format, so it can land in a 0.x minor.
3. **Credentials are refused in two fields only.** The v1 reader cannot
   start refusing credentials in `config` (that changes what it accepts);
   the conformance rule is the producer-side guard until v2 decides whether
   the reader should.
4. **Labels have no column in the ledger and fold into `split`**; a label or
   split containing `,` or `=` folds ambiguously. Not a record format
   problem -- a ledger schema one (a `run_metric_labels` table).

## Testing

- `packages/attestation-schema/tests/test_conformance.py` -- the stdlib
  checker: each rule's mutant against its fixture, every violation carries
  rule/file/JSON path, the credentials scanner's positives and negatives
  (assembled at runtime), the fixtures rebuild byte-identically.
- `packages/attestation-schema/tests/test_conformance_plugin.py` -- the
  plugin via `pytester`, as above. Runs on 3.11 in `schema-py311`.
- `tests/test_core_adapter.py` -- five records the bare reader
  accepts (non-canonical bytes, a stray history sidecar, a terminal
  `state.json` disagreeing with its record, an unbound manifest, a credential
  in `config`) are each quarantined and never stored; `attest runs scan`
  prints `QUARANTINED` with path and reason and exits 1, `--no-strict` exits
  0; MCP `runs.scan` leads its message with the quarantine; the converter's
  lossy steps are run caveats; a reader's skipped NaN is a `LOSSY` line.
  Measured as a mutant: with the core adapter reverted to the bare `read`,
  7 of its 11 tests fail.
- `tests/test_mlflow_records.py` -- the wiring guards.
