# Golden paths: documented use cases on disk

**Date:** 2026-08-28
**Status:** implemented 2026-08-29, with deviations below.
**Depends on:** the example flows (`2026-08-28-example-flows-design.md`), the
tracker adapters (`2026-08-22-tracker-adapters-design.md`), the citations
domain (`2026-08-22-citations-domain-design.md`), the config emitters
(`2026-08-22-config-emitters-design.md`).

## Problem

Someone arriving at this repo with a real question — "will this read my
runs?", "can it check my draft?", "how do I point it at my model server?" —
finds the answer spread across a 750-line README, `examples/README.md`, the
`evals` section, and a 700-line SKILL.md. Four things are true today:

- **Three golden paths exist and are pinned by tests**, which is this repo's
  best convention ("a documented path is a tested path"): the README
  quickstart over `examples/workspace`, the workspace walkthrough, and the
  prompt eval → optimize → transfer gate. A fourth, the example flows, landed
  today. None of them has the same shape, and there is no index that says
  which needs a model server and which runs in a second on a clean clone.
- **The install-and-connect-an-agent path is documented across 300 README
  lines** and verified only by stubs. It is the path every agent user takes.
- **No third-party integration has a worked example.** The ledger reads
  MLflow (verified today, against a real directory) and W&B (never run
  against one). Sacred, DVC and Hydra — the other layouts a researcher's
  runs actually live in — have no convention at all. The citation checker
  reads `.bib` files and Zotero, with no example of either. `LLM_BASE_URL`
  accepts any OpenAI-compatible server and the README only ever names
  Ollama.
- **Every fixture that exists was written by the code's own author** — the
  failure mode CLAUDE.md names. The one exception is `examples/flows/
  training/mlruns`, produced by the real library, and it is the only reader
  this repo can call verified.

## What a golden path is

A golden path is a directory under `examples/<name>/` that a newcomer can
run from a clean clone, with everything it needs on disk, and that the test
suite runs the same way the README says to. Concretely, every path has:

1. **`README.md` with fixed sections, in this order:** *What you get* (one
   paragraph, the outcome); *Prerequisites* (one of exactly three honest
   labels: `none — pure local computation`, `a model server at
   LLM_BASE_URL`, `network`); *Run it* (the commands, copy-pasteable, at most
   six, or one `./run.sh`); *What it prints* (the real output, abridged, from
   the last time it was run); *What it demonstrates* (which rule or caveat
   fires and why that is the point); *When it goes wrong* (the two or three
   failure modes a newcomer hits and the message each produces); *Next*
   (one link).
2. **`run.sh`**: `#!/usr/bin/env bash`, `set -euo pipefail`, a temporary
   `ATTEST_DB`, and the *same commands the README shows* — a test asserts
   that every fenced `uv run …` line in the README appears verbatim in
   `run.sh`, so the two cannot drift. Exit 0 is green.
3. **Inputs on disk**, either hand-written (and said to be) or produced by a
   committed script using the real third-party library, then scrubbed of
   attribution and machine paths (the rule `examples/flows/training` set:
   `test_the_committed_mlruns_carries_no_personal_or_machine_attribution`).
   Regeneration is a deliberate act and the script says so.
4. **A test** in `tests/test_golden_paths.py`: for a path whose prerequisite
   is `none`, run `run.sh` and pin one line of its output; for the others,
   the README/`run.sh` agreement check only (the suite never touches a
   model server). The README's *What it prints* block is the source of the
   pinned line, so the docs are what the test asserts.
5. **A row in the catalogue** — `examples/README.md` becomes an index table:
   path, one line of what it shows, prerequisite label, measured runtime —
   ordered by prerequisite, `none` first. The repo README's "Try it in 60
   seconds" stays as it is and gains one line pointing at the catalogue.

These are the best practices, stated once here and enforced by
`test_golden_paths.py` rather than remembered: fixed sections (a test checks
the headings), README ⇔ `run.sh` agreement, offline-first, scrubbed
artifacts, a catalogue row.

## Scope

### Retrofit the paths that exist (no renames)

Renaming `examples/workspace` would break the README quickstart, three tests
and SKILL.md; the shape is what matters, not the name.

- `examples/workspace/` — the ledger + claims path. README gets the fixed
  sections and a `run.sh`; its walkthrough prose is already right.
- `examples/flows/` — the demonstration suite. README gets the fixed
  sections; `run.sh` is `run_all.py --offline`.
- `examples/prompt-evals/` — a README + `run.sh` for the eval → optimize →
  transfer path, pointing at the scripts in `evals/` (which stay where they
  are: `test_tag_prompt.py` and the optimizer's group depend on the path).
  Prerequisite: a model server. No new fixture.
- `examples/agents/` — "connect an agent": `attest install --check`,
  `attest emit`, and one MCP surface driven over stdio via
  `examples/flows/mcp_e2e.py --surface provenance --offline`. Prerequisite:
  `none` (the stub). This is the path that replaces reading README §
  "Launching alongside hermes-agent" to get started; that section stays as
  the reference.

### New third-party integration paths

Each produces its artifacts with the real library (all four install
ephemerally in under five seconds: `uv run --with <lib>`), commits them
scrubbed, and turns the corresponding reader from plausible to verified —
or adds the reader, as a convention in `generic.py` per the adapter rule.

| path | library | reader | src change |
|---|---|---|---|
| `examples/wandb/` | `wandb` offline mode | exists, unverified | expected: offline runs live in `wandb/offline-run-*`, the reader globs `run-*` — a real-directory finding, fixed with a test if confirmed |
| `examples/sacred/` | `sacred` `FileStorageObserver` | none | new `_sacred_runs`: `<dir>/<id>/{config.json, metrics.json, run.json}`; `run.json` `status`/`experiment.name`; `metrics.json` final value + step |
| `examples/dvc/` | `dvc` | none | new `_dvc_runs`: `dvc.yaml`'s declared `metrics:` files plus `params.yaml`; one run per stage that declares metrics |
| `examples/hydra/` | `hydra-core` multirun | none | new `_hydra_runs`: `multirun/<date>/<time>/<n>/.hydra/config.yaml` + the metrics file the job wrote; the sweep directory is the family |
| `examples/citations/` | none (`.bib` on disk) | exists | none: a `.bib`, a draft with `cite=` claims, `cite.sources` / `cite.check` / `cite.lookup`; one key deliberately unresolvable |
| `examples/model-servers/` | none | n/a | none: "bring your own OpenAI-compatible server" — `LLM_BASE_URL` for vLLM, llama.cpp, LM Studio, documented; `run.sh` uses `examples/flows/stub_openai.py` as the runnable server |

Rules for the three new conventions, inherited from the tracker spec: final
values not curves; no metric-direction inference; no named adapters (`NAMED`
stays empty); each in its own function that `discover()` calls; a
shape-tolerance test (missing optional files degrade, never raise); the
reader's docstring names the library version that produced the committed
fixture.

### Not in scope

- TensorBoard event files (binary protobuf; not a convention `generic`
  can read without a dependency).
- Network trackers (W&B cloud, MLflow tracking servers) — the tracker spec
  forbids network reads.
- Changing any golden path's *behaviour* to make its test easier. A path
  that exposes a bug fixes the bug in its own commit with its own test.

## Execution

The framework and the retrofits are one sequence (they share
`tests/test_golden_paths.py` and `examples/README.md`). The six integration
paths are independent of one another except that three touch
`generic.py`'s `discover()`; those three run one after another, the other
three in parallel, each on its own subagent with its own brief.

## Success criteria

- `examples/README.md` lists ten paths with prerequisite labels; the
  `none` ones run green from `tests/test_golden_paths.py` on a clean clone
  in CI with no model server.
- Every path's README has the seven sections in order and its fenced
  commands appear in its `run.sh`, asserted by one test.
- `generic.py`'s W&B reader has been run against a real directory; Sacred,
  DVC and Hydra directories scan to runs with final metric values; `NAMED`
  is still empty; the complexity ratchet holds.
- No committed artifact contains a username, an absolute home path, or a
  repository URL; the flows' attribution test is generalised to every
  `examples/**` directory.
- The repo README's "Try it in 60 seconds" is unchanged except for one
  line pointing at the catalogue.

## Deviations and findings

**Twelve paths shipped, not ten.** The scope table above lists six new
integration paths plus the four retrofits, for ten; `mlflow/` and
`tensorflow/` were added during implementation at the user's request, and
`citations/` grew a BibTeX-software amendment (generate real `.bib` files
with `bibtexparser` rather than typing one to look like real output) also
at the user's request, so the "real library, not hand-typed" rule the spec
already stated for wandb/sacred/dvc/hydra was extended to citations' own
fixture instead of being the one hand-written exception. The catalogue,
`examples/README.md`, lists all twelve; the success-criteria bullet above
still says "ten" as a record of what the spec asked for, not a bug.

- **The expected wandb finding was false; the real one was upstream of
  it.** The spec's table guessed `_wandb_runs` globs `run-*` while real
  offline runs are named `offline-run-*`. Reading the code: it walks every
  child of `wandb/` with no name filter, so both names already worked —
  only the reader's docstring described a narrower pattern than the code
  implemented. The actual finding: offline W&B never writes
  `wandb-summary.json` or `config.yaml` to `files/` at all (confirmed
  against wandb 0.17.6 through 0.29.0; corroborated on wandb's own issue
  tracker, #7227, #9646, #1768). Every logged value still reaches disk,
  inside the run's binary `.wandb` transaction log. `examples/wandb/
  generate.py` materialises the two missing files locally by decoding that
  log with `wandb.sdk.internal.datastore` — the community's published
  workaround — and pins `wandb==0.17.6` (offline materialisation is not
  stable across releases; `generate.py` refuses to run under any other
  installed version rather than silently produce a fixture its own
  docstring no longer describes).
- **MLflow's committed fixture reports final values, never curves** — one
  `metrics/<name>` file per run holding the last-logged value, matching the
  ledger-wide convention (final values, no direction inference) rather than
  MLflow's own append-only metric history.
- **The `attest claims` CLI never ran the citation lint.** `cmd_claims`
  called `claims.check()` with no resolver, so a claim citing a key no
  configured source could resolve printed as plain "supported" — only
  `runs.claims_check` and `cite.check` over MCP ever surfaced the uncited
  verdict. Fixed in `4fb6007` (`cmd_claims` now builds a resolver and passes
  it through), found by `examples/citations/`'s own path, not a synthetic
  test.
- **`family_of` returned `None` for a bare `<token>_<value>` stem.**
  `examples/tensorflow/`'s four arms are named `results/lr_0.001.json` ..
  `lr_0.03.json`; the existing `_SPLIT` regex recognised `lr` as a
  variant-token prefix but, for a stem that is *only* that token plus its
  value, stripping the token emptied the whole stem and every arm grouped
  to no family at all — `attest runs compare lr` failed outright. Fixed in
  `ca08646`: when stripping the recognised token leaves nothing behind, the
  token's own name is the family (`lr_0.001` groups as `lr`), and the
  existing sweep/series shapes are unchanged.
- **Hydra 1.3 does not chdir into each arm's own output directory by
  default.** `hydra.job.chdir=True` is required or `train.py`'s
  `open("metrics.json", "w")` writes into the directory the sweep was
  launched from, and all four arms silently overwrite the same file — a
  real `--multirun lr=0.01,0.1,1,10` run with no override produced one
  `metrics.json`, not four. `examples/hydra/generate.sh` passes the
  override explicitly; `_hydra_runs` reads only the layout that override
  produces.
- **DVC's `dvc.lock` records the whole swept `foreach` list, not the one
  value each stage instance ran with** — every `train@<lr>` entry carries
  the same `params.yaml: lr: [0.01, 0.1, 1, 10]` literal, because DVC is
  quoting the source the instance was generated from, not the value it was
  instantiated with. `_dvc_runs` takes each arm's `lr` from the stage
  instance name itself (`train@0.1` implies `lr=0.1`) rather than trusting
  `dvc.lock`'s echoed list. Relatedly, `metrics/` is already one of
  `generic.py`'s own `RESULT_DIRS`, so an unguarded scan would read
  `metrics/0.1.json` twice — once as a bare-named run, once as
  `_dvc_runs`'s `train@0.1`; `discover()` now computes which metric files
  `dvc.yaml`'s stages claim before the ordinary scan runs, and skips them.
- **`conf/` collides with `CONFIG_DIRS`.** Hydra's own config directory
  (`conf/config.yaml`) is one of `generic.py`'s existing `CONFIG_DIRS`, so
  `attest runs scan` reads it as an ordinary config spec named `config`
  alongside the four sweep arms — the same "a spec with no result attached
  is recorded as a run with no metrics" honesty every config file gets.
  Not a bug: `conf/` really is a config directory, and the ledger has no
  way to know it is a Hydra input rather than an ordinary one.
- **Attribution-guard rulings**, both narrowing `test_no_committed_
  example_carries_attribution_or_machine_paths` to catch real leaks rather
  than collisions with unrelated content: the ambient username matches only
  as a whole word, case-sensitive (a bare substring on a short name like
  "matt" also matches "mattered"); and `mlflow.user`/`git@` are checked
  only outside `.py`/`.sh`/`.md` source, since a scrubber's own code and
  its docs must be free to name the tag or prefix it strips
  (`train_mlflow.py`'s `_SCRUB_TAGS` names `mlflow.user` verbatim — the
  tag's real name, not a leaked value). A third ruling landed after CI
  caught a case the first two didn't: GitHub Actions sets `$USER=runner`,
  and "runner" is an ordinary word in hydra's own README prose ("a GitHub
  Actions runner") — the ambient-username check is now skipped under `CI`
  or when the username is a generic account name (`runner`, `root`,
  `user`, `ubuntu`, `admin`, `ci`), while the fixed needles (`/home/`,
  `github.com`) still apply everywhere. The guard was also generalised from
  `examples/flows/` alone to all of `examples/**`, and extended to scan
  non-text suffixes (e.g. TensorBoard's `events.out.tfevents.<ts>.v2`) as
  raw bytes for the same two needles, since a binary format can still embed
  a plain attribution string even though it is not itself readable text.

## Deviations and findings

**Twelve paths shipped, not ten.** The scope table above named six new
integration paths plus four retrofits (workspace, flows, prompt-evals,
agents) for a planned total of ten. Two more were added at the user's
request during implementation: `examples/mlflow/` (a standalone front door
over the existing `examples/flows/training/mlruns` fixture, so the ledger's
oldest tracker reader gets its own runnable example rather than living only
inside the flows suite) and `examples/tensorflow/` (a real Keras sweep read
back through `CSVLogger`'s CSV and a plain metrics JSON, needing no new
reader). The citations path was also widened beyond the original "`.bib` on
disk" scope at the user's request: the fixture BibTeX library is generated
by real `bibtexparser` software rather than hand-written, matching the rule
already binding on wandb/sacred/dvc/hydra/tensorflow that a generated
artifact comes from the real library.

**Offline W&B writes no summary/config files at all, confirmed against wandb
0.17.6–0.29.0.** The scope table's listed risk was a possible directory-naming
mismatch (`offline-run-*` vs. the reader's `run-*` glob); reading the real
directory showed the glob was already correct and the naming was never the
problem. The actual gap is upstream: `wandb.init(mode="offline")` never
materialises `wandb-summary.json` or `config.yaml` under `files/` until
`wandb sync` uploads to a server, which this repo's offline guarantee never
does (confirmed against wandb's own issue tracker, #7227/#9646/#1768).
`examples/wandb/generate.py` works around this by decoding the run's own
binary `.wandb` transaction log with `wandb.sdk.internal.datastore.DataStore`
(the community's documented workaround for the same gap) and writing the two
files in their documented on-disk shape before deleting the binary log,
which nothing in this repo reads. The generator is pinned to `wandb==0.17.6`
and refuses to run under any other installed version, since offline
materialisation is not stable across releases.

**The `attest claims` CLI never ran the citation lint** — only the MCP tools
(`cite.check`, `runs.claims_check`) did. `examples/citations/` surfaced this
by exercising both paths against the same draft; fixed in `4fb6007` (`attest
claims now runs the citation lint, matching the MCP tools`), with the CLI's
prior blind spot documented in `check_citations.py`'s docstring
(`d323b52`).

**`family_of` returned `None` for a bare `lr_<value>` stem.** The existing
`_SPLIT` regex recognises `lr` as a variant-token prefix, but for a stem
that is *only* that token plus its value, stripping the token empties the
whole stem — `examples/tensorflow/`'s real four-arm sweep (`results/
lr_0.001.json` .. `lr_0.03.json`) hit this and `attest runs compare lr`
failed outright. Fixed in `ca08646` (`Group a bare split-token stem by its
own name: lr_0.001 is family lr`): when stripping the recognised token
leaves nothing behind, the token's own name becomes the family.

**Hydra 1.3 does not chdir per job by default.** Without
`hydra.job.chdir=true` on the `--multirun` command, all four sweep arms
write to one shared working directory and silently overwrite the same
`metrics.json` in turn, so `attest runs scan` sees only the last arm to
finish. `examples/hydra/README.md`'s *When it goes wrong* documents this as
the first failure mode a newcomer hits, and `run.sh` passes the flag.

**DVC's `foreach` stage echoes the whole swept parameter list into
`dvc.lock`, not each arm's actual value, and its `${item}` is a literal
token in `dvc.yaml`, not one DVC ever expands on disk.** `dvc.yaml`'s
`metrics: [metrics/${item}.json]` keeps `${item}` unexpanded — the generic
reader's `_dvc_stage_block` leaves it that way and `_dvc_runs` performs the
substitution itself, once per item, using `params.yaml`'s own list (the
same list `dvc.lock` echoes verbatim for the `foreach` key rather than
recording per-arm). `examples/dvc/generate.sh` needs no dependency on the
`dvc` package at all — `dvc.yaml`/`dvc.lock`/`params.yaml` are just text
the reader parses by hand.

**`conf/` collides with `generic.py`'s existing `CONFIG_DIRS`.** Hydra's
own config directory convention (`conf/config.yaml`) is one of the strings
`CONFIG_DIRS` already excluded from run discovery (alongside `configs`,
`config`, `examples`, `experiments`) — `examples/hydra/`'s `conf/` is
correctly reported as an empty project with "no runs ... no metrics" rather
than misread as a sweep arm; this is documented as expected behaviour, not
a bug the path needed to fix.

**Attribution-guard rulings, both narrowing the guard to avoid false
positives rather than widening what it catches:** the username needle
matches only as a whole word (`\bmatt\b`), since a bare substring also
matches ordinary English ("mattered"); and the `mlflow.user`/`git@` needles
are skipped inside `.py`/`.sh`/`.md` source, since a scrubber's own code and
docs must be free to name the tag or prefix it strips (`train_mlflow.py`'s
`_SCRUB_TAGS` names `"mlflow.user"` verbatim as the tag's real name, not a
leaked value) — `/home/` and `github.com` have no such legitimate
source-or-prose use and still apply everywhere. A third ruling landed after
a CI failure (run 33233059347): GitHub Actions sets `$USER=runner`, and
"runner" collides with ordinary prose (`hydra/README.md`'s "a GitHub
Actions runner"), so the ambient-username check is skipped under `CI` and
for a short list of generic account names (`runner`, `root`, `user`,
`ubuntu`, `admin`, `ci`) even off CI; the fixed needles still apply
unconditionally. The guard was also generalised from `examples/flows/`
alone to all of `examples/**`, and extended to scan non-text suffixes (e.g.
tensorboard's `events.out.tfevents.<ts>.v2`) as raw bytes for `/home/` and
the username, rather than skipping binaries outright.

**The repo README's "Demonstrations" paragraph became the golden-paths
section, and "Prompt evals and the optimizer" was reduced to a pointer
paragraph.** The quickstart block itself is byte-identical; only the prose
around it changed to point at `examples/README.md`'s catalogue rather than
describing each flow inline a second time.

**sacred/dvc/hydra generators were unpinned until 2026-08-29 (this
commit).** `examples/wandb/generate.py`'s `WANDB_VERSION` pin-and-refuse
pattern was established first and not propagated to the other three
generated fixtures at the time: `examples/sacred/generate.py` documented a
bare `--with sacred` with no runtime check; `examples/dvc/generate.sh` and
`examples/hydra/generate.sh` both declared a version constant
(`DVC_VERSION`, `HYDRA_VERSION`) that was echoed in output but never passed
to `--with` or checked against what actually ran. All three now pin the
installed command to the constant and refuse (a `SystemExit` for sacred, an
`exit 1` after a version check for dvc/hydra) when the installed version
disagrees, matching wandb's and tensorflow's existing precedent.

**`generic.py` reached ~1500 lines across five tracker conventions
(wandb, mlflow, sacred, dvc, hydra) by the time this pass landed.** A split
into `ledger_adapters/trackers.py`, one reader per tracker, is recommended
but not done here — out of scope for a fix-up pass, and a real
refactor deserves its own review rather than riding along with unrelated
version pins.
