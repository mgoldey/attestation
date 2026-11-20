# One dependency-identity rule, shared helpers, and attestation-schema 0.2.0

**Date:** 2026-11-19
**Status:** implemented 2026-11-19 (attestation-schema 0.2.0; task-44 audit items 1-6).
**Depends on:** `2026-11-19-core-run-record-design.md` (the frozen
`attestation.run/1`: `dependencies[]`, the `source` vocabulary, the reader's
credential rule, the layout's write order), `2026-11-19-training-callbacks-design.md`
(the callbacks, which carried the schema's private copy of the rule).
**Changes no format.** `tests/test_frozen.py` is untouched and passes: every
record, manifest, state file and history sidecar is byte-identical to what
0.1.0 wrote. Everything here is Python API.

## Problem

The task-44 duplicate-code audit (smeltery / attestation / ferric) found two
implementations of "which build of this package ran":

- smeltery's `record.package_identity` / `_distribution_identity`, mapped to
  the core record by `core_record.dependency` / `core_record.source`;
- this package's private `integrations/_core.py` `dependency_identity`, used
  by the training callbacks.

They disagree on seven inputs, so the same ferric install got a different
`provenance` and `source` depending on which producer wrote the record. A
lineage that reads `dependencies("ferric")` across a dataset's source runs and
a training run then compares two rules, not two builds. The audit also found
smeltery's credential stripper handling only http/https (an
`ssh://u:p@` or `git+https://tok@` remote makes smeltery's write refused by
the schema reader, so no record is written at all), the schema's two manifest
writers not fsyncing while `run.json` did, and two copies of the manifest
writer inside the schema.

## Decision 1: one public function, `attestation_schema.identity.package_identity`

```python
from attestation_schema import package_identity
dep = package_identity("ferric", "ferric")          # -> Dependency
dep = package_identity("torch", sys.modules["torch"], name="torch")
```

`package_identity(dist_name, module=None, *, name=None, probe_git=True)`
observes the process (distribution metadata, `direct_url.json`,
`INSTALLER`, the module's `__version__` and `__build__`, a directory
install's git state) and hands everything to `dependency_identity(...)`,
which is **pure** and holds the whole rule, so every branch is tested without
installing anything. `module` is a module object or a name; a name not yet in
`sys.modules` is imported (smeltery's behaviour: it reads ferric's stamp),
and the callbacks pass the object so nothing is imported that the run did
not load. Without a module only the metadata is read, which can at best be
`inferred` (Decision 2.1). Never raises: what could not be observed is null
and `evidence` says why.

The ladder, first match wins: build stamp -> VCS commit -> index release the
module agrees with -> anything naming a version or commit (`inferred`) ->
nothing (`unknown`). `source` is read beside the ladder, not from it.

## Decision 2: each divergence

| input | smeltery | schema 0.1.0 (callbacks) | **0.2.0** |
|---|---|---|---|
| index install (no direct_url, INSTALLER pip/uv) | `index`, verified, dirty False | `index`, verified iff `__version__` == metadata | `index`; verified iff the module's `__version__` agrees; dirty null |
| no `INSTALLER` file | treated as index: verified | source null, verified iff agree | source null, **never verified** |
| conda / mamba / micromamba | `conda` (all three), inferred | `conda` only for conda, verified iff agree | `conda` (all three), inferred |
| other installers | `system`, inferred | source null, verified iff agree | `system` only for rpm/debian; else null; inferred |
| `dir_info` | `file`, git probe -> commit + dirty, inferred | `file`, no probe, inferred | `file`, git probe -> build_sha + dirty, inferred |
| `vcs_info` without `commit_id` | unknown, source null | `<vcs>+<url>`, verified iff agree | `<vcs>+<url>`, inferred |
| `__build__` stamp | verified, source null | not read | verified; source still read |

1. **An index install is verified only when the loaded module agrees.**
   smeltery's argument (an index release is immutable, so the version pins
   the bytes) is right about the index and silent about the process: the
   metadata says what was installed, not what this interpreter imported (a
   checkout earlier on `sys.path`, a stale `.egg-link`). The module's
   `__version__` agreeing with the metadata is the observation that ties the
   two (a PEP 440 local label only the module reports agrees: torch's
   `2.15.0+cu132`, measured, kept from the callbacks' rule). No module, or a
   module with no `__version__`: `inferred`, `evidence` says so. smeltery
   passes `module="ferric"`. `dirty` is null: with no `build_sha` there is no
   tree to differ from (smeltery wrote False, a claim nobody observed).
2. **A missing `INSTALLER` is no source and never verified.** Every PEP 376
   installer writes one; a distribution without one came from
   `setup.py develop`, a hand-copied tree or a tool that does not follow the
   spec -- exactly the mutable cases where a version proves least. smeltery
   read it as an index install, which made the strongest claim on the
   weakest evidence.
3. **conda, mamba and micromamba are `conda`, and `inferred`.** All three
   install conda packages (micromamba and mamba write their own names, which
   0.1.0 missed). A conda version names several builds (the build string), so
   the version does not pin bytes: smeltery's `inferred` stands. The channel
   is not recorded (`conda`, never `conda:<guess>`).
4. **Other installers say nothing about where.** `system` is a claim (a
   distribution package manager); smeltery made it for every unknown
   installer, including poetry and pdm, which install from an index. Only
   `rpm` (Fedora's RPM macros) and `debian` map to `system`; anything else is
   source null with the installer named in `evidence`, `inferred`.
5. **`dir_info` takes the git probe and stays `inferred`.** The callbacks'
   rule recorded nothing about a local tree; smeltery read its HEAD and
   status, which is the only identity a directory has. Kept, with the reason
   it is not `verified` in `evidence`: HEAD is read now, not at install, and
   an editable install imports whatever the tree holds. `probe_git=False`
   skips it. The path is never written (`file`): it names a machine.
6. **`vcs_info` without a commit names the source and is `inferred`.** PEP
   610 requires `commit_id`, so its absence (or a non-sha such as an svn
   revision) means a malformed or partial record. Where it came from is still
   observed (`<vcs>+<url>`, smeltery dropped it); what was built is not, and a
   version string from a VCS install is not an immutable release, so 0.1.0's
   "verified iff the versions agree" overclaimed. A `vcs` outside
   git/hg/svn/bzr has no spelling in the vocabulary: source null (0.1.0 wrote
   it anyway, and the reader would have refused the record). With a commit:
   `verified`, `build_sha` = commit, `dirty` False (the installer built a
   fresh checkout of that commit).
7. **A `__build__` stamp is generic.** Any module exposing `__build__ ==
   {"git_sha": <40/64 lowercase hex>, "dirty": <bool>}` (extra keys allowed)
   is a verified build stamp: ferric's shape (`ferric-python/src/integrals.rs`,
   `add_build_stamp`), which ferric sets to `None` when the build had no git
   metadata. A malformed stamp (no `dirty`, a short or upper-case sha, a
   tuple) counts as none rather than being repaired: it is a claim the build
   made about itself. The stamp overrides a VCS commit (noted in `evidence`).
   Unlike smeltery, `source` is still read: the stamp says WHICH build, the
   install says from WHERE, and both are observed.

Archive installs (`archive_info`, not one of the audit's seven but a branch
both rules had): `url:<url>` (credentials and query stripped, a `#sha256=`
fragment kept) or `file`, and `inferred` with the archive hash pip recorded
in `evidence`. 0.1.0 verified them on version agreement; a locally built
wheel reuses its version freely, and the archive hash names the file, not a
commit, so it cannot be `build_sha`.

## Decision 3: `strip_credentials` is the reader's rule, inverted

`record._carries_credentials(location)` is now the reader's one credential
predicate (a password for any scheme; any user name for a web scheme,
`git+https` included). `identity.strip_credentials(url)` removes the whole
userinfo exactly when that predicate holds, so `ssh://git@host` (an address,
not a secret) and scp-style `git@host:path` come back unchanged and
`ssh://u:p@host` loses its userinfo. A `dependencies[].source` value is
accepted too: the `index:`/`conda:`/`url:` prefix is kept, and a `url:`
source also loses its query string (a signed URL's credential) and keeps its
fragment. `tests/test_identity.py` checks it against the reader itself over a
URL corpus: the reader accepts every output, and an input comes back
unchanged exactly when the reader already accepts it.

## Decision 4: one durable write (`attestation_schema.files`)

`files._atomic_write(target, data, *, exclusive, fsync=True)` is the only
write path: a temporary file named by pid AND a random token (two writers in
one process cannot collide), fsynced, then `os.replace` or, when
`exclusive`, `os.link` (an atomic no-clobber), then the directory fsynced so
the rename survives a crash. `write` (run.json), `state.json`, both manifest
writers and both dataset writers use it; the streamed history sidecar uses
its commit step (`_commit`). `layout.write_manifest` and
`dataset.write_dataset_manifest` share one body (`layout._write_manifest_file`).
The manifest writers now fsync: the write order digests.json -> run.json
only means something if the first write is durable before the second. Bytes
are unchanged (test_frozen's pins, plus a byte-level manifest test).

## Decision 5: shared helpers, and attestation's copies

`sha256_file(path)` (1 MiB chunks) and `git_state(dir) -> GitState(sha,
dirty, remote)` are public. `git_state` never raises: no git, no checkout or
a timeout reads as None; `dirty` counts tracked files only; `remote` is
credential-free. attestation's own copies now call them:
`run_record/verify.py` and `figures.py` (sha256), `run_record/build.py` and
`prereg.py` (git, timeout 5 s as before). verify.py imports
`attestation_schema.files`, which `test_verifier_imports_nothing_heavy`
allows beside `attestation_schema.canonical` (both stdlib). attestation's
floor becomes `attestation-schema>=0.2.0`.

## Decision 6: the version

0.2.0 is the first release with `dataset.py`, `lineage.py`, `conformance`,
the pytest plugin, the callbacks and `identity` (smeltery's dataset finding
8: it could only detect `dataset.py` by trying the import).
`attestation_schema.__version__` reads the installed metadata and falls back
to `_FALLBACK_VERSION`, which a test keeps equal to `pyproject.toml`.

## What smeltery does with it (not done here)

smeltery's `core_record.dependency` should call `package_identity(dist,
module)` and drop its sentence-parsing `source()`; `strip_credentials`
replaces its http-only copy. That needs smeltery's `attestation` extra floor
raised to `>=0.2.0` -- which the legacy `record.py` (not gated on the extra)
cannot use, so its sentences stay until the legacy record is retired.

## Testing

`packages/attestation-schema/tests/test_identity.py`: one test per decision
above against the pure rule; every shape the rule produces read back by the
frozen reader; `package_identity` against real dist-info directories on
`sys.path` (index, stamp, missing everything, a directory install probed with
a real git checkout); `build_stamp` shape; `git_state`; `strip_credentials`
against `record._repo` and `record._source` over a corpus; `sha256_file`
reading in chunks; `_atomic_write` exclusive and clean; every writer
fsyncing; manifest bytes unchanged; `__version__` == pyproject.
