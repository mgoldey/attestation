# MLflow runs carry their provenance, and `attest mlflow sync` writes it down

**Date:** 2026-11-20
**Status:** implemented 2026-11-20 on `next/mlflow-plugin` (roadmap item 35).
**Depends on:** `2026-11-20-mlflow-adapter-design.md` (the store reader and
its mapping), `2026-11-19-dependency-identity-design.md` (`package_identity`,
the one rule), `2026-11-19-training-callbacks-design.md` (framework
integrations behind extras, `ATTESTATION_REQUIRE_INTEGRATIONS`, the
`schema-integrations` CI job), `2026-11-19-core-run-record-design.md`
(Producer findings 1 and 3: MLflow records no dependency, and no dirty bit).
**Not this:** a tracking-server or sqlite reader for `sync` (it reads the
file store, as the ledger does), an MLflow *artifact* or *model* plugin, a
`runs.sync` MCP tool, deleting records whose MLflow run was deleted.

## Problem

An MLflow run records which commit ran (`mlflow.source.git.commit`) and
nothing else about identity: not whether the tree was dirty, not which build
of torch or scikit-learn the process imported. The core record has fields
for both (`code.dirty`, `dependencies[]` with a provenance ladder), so every
MLflow-derived record -- the exporter's, the ledger adapter's -- says
`dirty: null, dependencies: null`. Those are true ("not observed"), and they
can only become anything else if something observes them *while the run's
process is alive*: after the run, the environment is gone.

Second, a ledger import of an MLflow store is in memory only. There is no
file a collaborator can be handed, a conformance check can run over, or
`attest runs lineage` can follow. The exporter writes such files, but it is
an example (it needs mlflow, and it lives under `examples/`).

## Decision 1: a run-context provider, in `attestation_schema.integrations`

MLflow's fluent `mlflow.start_run` asks every registered *run-context
provider* for tags (`mlflow.tracking.context.registry.resolve_tags`); a
package registers one through the `mlflow.run_context_provider` entry point.
The provider is `attestation_schema.integrations.mlflow:AttestationRunContext`,
declared in `packages/attestation-schema/pyproject.toml`, behind a new extra
`[mlflow]` (`mlflow-skinny>=2.20`, which the full `mlflow` package also
satisfies).

**Why there.** It is producer code: it runs inside the user's training
process, so it must not drag in the observer (`attestation` imports FastAPI,
sqlite-vec, httpx...). The boundary rules already say where producer code
lives: `attestation-schema`, stdlib only, each framework imported by exactly
one module behind its own extra -- the training-callbacks precedent. So:

- `integrations/mlflow.py` is the ONLY module that imports mlflow (for the
  `RunContextProvider` base class); `tests/test_boundaries.py` pins it.
- Everything else -- what the tags say, how they are observed, and how they
  are read back -- is `integrations/mlflow_mapping.py`, standard library
  only, so it runs (and is tested) without mlflow, in the bare 3.11 job, and
  `attestation` imports it with no mlflow installed.
- The tracker-name guard (`tests/test_tracker_names.py`) exempts the schema's
  `integrations/` directory the way it exempts `ledger_adapters/`: a module
  that plugs into a framework has to name it. The rule it protects -- the
  ledger and claims code knowing no tracker -- is unchanged.

### The tags

| tag | value |
|---|---|
| `attestation.format` | `attestation.mlflow-tags/1` -- the layout of these tags, so a reader can refuse one it does not know |
| `attestation.producer` | compact JSON `{"name": "attestation-schema mlflow-run-context", "version": <attestation-schema version>}` |
| `attestation.code` | compact JSON (RFC 8785) of a core `code` object: `git_state` of the script's checkout at run start -- HEAD, **dirty** (tracked files differ), origin without credentials; `inferred` (HEAD names the checkout, not the bytes Python loaded), `unknown` outside a checkout |
| `attestation.dependencies` | compact JSON of a core `dependencies` list: python (`sys.version_info`, verified) and every *already imported* key package -- mlflow, numpy, pandas, scikit-learn, scipy, torch, tensorflow, keras, jax, transformers, lightning, xgboost, lightgbm, catboost -- plus any named in `ATTESTATION_MLFLOW_PACKAGES` (`dist[:module],...`), each through `identity.package_identity` |
| `attestation.error` | present only when something could not be observed or did not fit: the exception type and first line, or "N of M dependencies fit" |

Only packages already in `sys.modules` are named, and the module object is
passed to `package_identity`, so the provider **imports nothing** the run did
not (the callbacks' rule: what the run did not load is not named). A package
named in `ATTESTATION_MLFLOW_PACKAGES` that is not imported is identified
from its metadata alone (`inferred` at best, the rule's own outcome).

MLflow caps a tag value (`MAX_TAG_VAL_LENGTH`, 8000 in 3.x, 5000 in older
servers). The provider keeps each value under 5000 characters: dependencies
are dropped from the end of the list (python stays) and `attestation.error`
says how many fit. Tags a user passes to `start_run(tags=...)` override
provider tags (MLflow applies them last), so a user can always replace ours.

### Never fail a run, never swallow silently

`tags()` never raises. Each probe (code, dependencies) is guarded on its own:
a failure leaves that tag out and writes `attestation.error` with the
exception's type and first line, and the other tags are still returned. (MLflow
would catch an exception from `tags()` itself and log a warning -- but then
*every* attestation tag would be lost, and the cause would be in a log
nobody reads, not on the run.) The git probe uses a 2 s timeout per call
instead of the identity module's 10 s: a run start must not hang on a slow
network filesystem.

### `in_context()`: on by default, with an opt-out

MLflow instantiates every installed provider when it is imported and asks
each one, for every fluent run, whether it applies. The entry point is in
`attestation-schema`'s metadata whatever extras were chosen, so the provider
is active wherever `attestation-schema` and mlflow are both installed --
including where attestation pulled the schema in as a dependency.

`in_context()` returns True unless `ATTESTATION_MLFLOW_TAGS` is `0`/`false`/
`no`/`off`. It reads that one variable and does no I/O. Considered and
rejected: opt-in (`ATTESTATION_MLFLOW_TAGS=1` required). The point of the
roadmap item is that runs carry provenance *automatically*; an opt-in switch
is what nobody sets (the reason the schema package has no dependencies: "a
dependency it must install is a dependency that gets skipped"). MLflow's own
built-in providers -- git, source, user, system environment -- are on by
default and probe the same things. The conservatism is in what the provider
does, not in the switch: no import, no network, bounded time, additive tags
under its own prefix only (it never writes an `mlflow.*` tag), never an
exception, and one documented variable to turn it off. A test pins that
`in_context()` performs no subprocess call.

## Decision 2: one mapping, in the schema package

The exporter (`examples/mlflow-records/export_records.py`, through
`MlflowClient`) and the ledger adapter (`ledger_adapters/mlflow_store.py`,
from files) each held a copy of the same pure pieces: the status map, the
git tag names, `timestamp_ms`, MLflow's latest-point key, the (metric,
dataset) grouping and the git-tag -> `code` rule. Adding the attestation tags
to both would have made a third copy. The pure pieces now live once in
`attestation_schema.integrations.mlflow_mapping` (stdlib; the exporter can
import it because a producer depends on the schema, and so can attestation),
and both call it. The exporter's own `strip_credentials` (http/https only)
is replaced by the schema's (the reader's rule inverted), so an
`ssh://u:p@` remote no longer reaches `ext.mlflow` either. The committed
records are byte-identical (`tests/test_mlflow_records.py` re-exports them).

### Reading the tags back: `code_from_tags`, `dependencies_from_tags`

Both validate with the record reader's own field rules (`record._code`,
`record._dependencies`); a tag that does not parse, or a format they do not
know, is ignored with a *problem* string (never a crash, never a guess).

**Code.** The provider's `attestation.code` and MLflow's
`mlflow.source.git.commit` are two observations of one checkout:

- both name the same sha: the provider's object (it adds `dirty` and the
  repo), its evidence noting that MLflow's tag agrees;
- they disagree: `unknown`, `git_sha` null, evidence naming both -- the core
  record's rule for mixed commits (MLflow resolves the repository from the
  script's path the same way, so a disagreement means someone edited a tag);
- only one: that one, as before (`inferred` from MLflow's tag);
- neither: `unknown`.

**Dependencies: the provenance the rule gave at run time, unchanged.**
`verified` is a property of an *observation*: the identity rule says
`verified` when the module the process imported agrees with an immutable
release (or carries a build stamp, or came from a commit). That observation
was made by the provider, in the run's own process, at run start -- the only
moment it could be made. `sync` and the exporter only *transcribe* it, so
they neither raise nor lower it: a dependency the rule called `verified` at
run start stays `verified`, `inferred` stays `inferred`. Each dependency's
`evidence` gains "; observed at MLflow run start by attestation-schema's
run-context provider (tag attestation.dependencies)", so the record says
where the claim came from. What this trusts is the store: an MLflow tag can
be edited after the run, exactly as a metric file or `meta.yaml` can, and
every MLflow-derived record already vouches for the store as it is (a record
is a statement about what the store says, made by whoever reads it). A
tampered tag is no more and no less detectable than a tampered metric.
Without the provider's tags, `dependencies` stays `null` (not observed) --
never `[]`.

The ledger adapter maps the tags too (one mapping), though the ledger reads
`code`/`dependencies` only from native records -- which is what `sync`
produces.

## Decision 3: `attest mlflow sync`

    attest mlflow sync [--tracking-dir mlruns] [--out mlflow-records] [--force]

Reads the file store with `mlflow_store` (no mlflow, as the ledger), and for
every ended run writes `<out>/run-<run_id>/` in the layout's order:
`attestation/history.jsonl` (`write_history`), `attestation/digests.json`
(`write_manifest`), `attestation/run.json` (`write`). The record is the
adapter's (`trackers.MLFLOW.to_core_record`: the same family, status, curves,
metrics, data, code, dependencies and `<config>_seed<N>` arm) with two
changes: `producer` is `attestation mlflow-sync`, and `ext.ledger-import` is
replaced by `ext.mlflow-sync` (`experiment_id`, `run_name`, `notes` -- the
artifact-URI note and any tag problem). Dropping `ledger-import` is the
point: the ledger reads a record without it as *native*, so the code and
dependency provenance reach `provenance_findings`.

Unlike the ledger, `sync` writes a record for an ended run that logged no
metric (the exporter does too): the record is about the run, not the
ranking.

**Idempotent, never silently overwriting.** Each run is built in a
temporary directory beside its target and compared with what is there:

- no record yet: renamed into place (`written`);
- byte-identical files: left alone (`unchanged`);
- identical except `producer` (an attestation upgrade): left alone
  (`unchanged`) -- the old record is still a true statement about the same
  run;
- anything else differs (the run changed upstream: a tag, a late metric, a
  re-terminated run; or the mapping changed): **refused**, the differing
  top-level fields listed, exit 1, nothing written -- unless `--force`, which
  replaces the directory atomically (`replaced`).

Every written directory is checked with `conformance.check_run_dir` before
it is moved into place; a non-conforming one is reported and never written.
`attest mlflow sync` exits 1 on any refusal or non-conforming run.

**The ledger reads a synced run once.** The core adapter runs first and
claims each record's `run_id`; the mlflow adapter now skips a store run whose
id a core record in the same project already claimed. So syncing into the
project directory (the default `--out` is `./mlflow-records`) does not
double-count. The synced run's ledger name is its run id (the core adapter's
naming), not `<run_name>/<id8>`: a claim that cites the old name must be
updated, or the records written outside the scanned project.

**Where the command lives.** `test_tracker_names.py` forbids tracker names
in `src/attestation` outside `ledger_adapters/`, `cli.py` included. The
command (its parser, HELP strings and handler) is
`ledger_adapters/mlflow_sync.py`; `cli.py` gains a generic three-line hook
that registers the adapters package's commands and merges their HELP
entries, naming no tracker.

## Testing

- `packages/attestation-schema/tests/test_mlflow_mapping.py` (stdlib, runs
  everywhere): tag observation in a real git checkout (clean, dirty, none),
  only-imported packages named, the env-var package list, the size cap,
  the never-raise guard per probe (`attestation.error`), the opt-out, every
  read-back branch (agree, disagree, one, neither, malformed, unknown
  format), provenance preserved and evidence appended.
- `packages/attestation-schema/tests/test_integrations_mlflow.py`: a REAL
  local file store written by `mlflow.start_run` in a subprocess inside a
  git checkout -- the entry point is discovered, the tags are on the run,
  the opt-out removes them, and the exporter's record carries the tagged
  dependencies with their run-time provenance. Skips without mlflow, fails
  under `ATTESTATION_REQUIRE_INTEGRATIONS=1`; CI's `schema-integrations` job
  installs `[mlflow]` and runs it.
- `tests/test_mlflow_sync.py`: sync of the committed sweep
  (`examples/flows/training/mlruns`) -- every record conforms, equals the
  adapter's record but for producer/ext, is native to the ledger, and is
  read once; re-sync byte-identical; a changed run refused (and `--force`
  replaces it); a producer-only difference left alone; tags in a synthetic
  store reach `code`/`dependencies`.
- `tests/test_boundaries.py`: `integrations/mlflow.py` may import mlflow and
  nothing else; `mlflow_mapping.py` is stdlib; the `[mlflow]` extra.
