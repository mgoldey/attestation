# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
History before 2026-08-28 is not reconstructed — this file starts where a
narrative commit history and this project's day-by-day session logging
overlap, not at the repo's actual beginning. Each line below points at the
commit that carries the reasoning rather than repeating it; `git log
--oneline <sha>` on any entry shows the full message.

## [Unreleased]

### Fixed

- `runs.ask` (`2026-09-03`): comparing arms by a metric the question named
  silently fell back to whichever metric most arms shared instead, because
  `_runs_ask` called `_compare(family)` with no metric argument at all —
  found via a real Hermes session asking "compare kdsweep by wer" that got
  a caveat computed over a different metric's spread. The same call also
  never surfaced `winner` in its own answer, because `winner` names one
  arm rather than a collection and `_RESULT_KEYS`' generic "named list"
  path never looked for it — a caller asking "which arm won?" got the arm
  list back with no arm marked as the answer. Text-extracting a metric
  from the question alone was not enough either: gemma4:e2b paraphrased
  "using the wer metric, compare..." down to `question="which arm won?"`
  three runs straight before the tool ever saw it, so `runs.ask` gained an
  explicit `metric` parameter, and the provenance skill now tells an agent
  to pass it rather than rely on its own paraphrase carrying it.

### Added

- `demos/hermes/` (`2026-09-03`): a fifth demo, and the only one driving a
  real agent rather than calling the tools directly — an asciinema
  recording of `hermes chat` asking a real question against the
  `attestation-provenance` skill, verified twice byte-identical after the
  fixes above. `record.sh`/`README.md` follow the same convention as the
  other four (script committed, output gitignored); needs a live Hermes
  install and Ollama, so it is not run by any test.
- `demos/` (repo root, `2026-09-03`): recording scripts for four short
  demos — the run ledger + claim checker (`ledger/`, asciinema), citations
  (`claims/`, asciinema), `kg.*`/`sym.*` over MCP (`kg-symbolic/`,
  asciinema, the one pair of MCP-only surfaces with no CLI/UI front end),
  and the HTMX web UI (`feed/`, Playwright, its own `demos` dependency
  group so a plain `uv sync` never installs a browser). Scripts are
  committed; the `.cast`/`.gif`/`.webm` output is not — same convention as
  the existing gitignored `demo/`. Lives at the repo root rather than under
  `examples/`: a README one level under `examples/` is swept into
  `test_golden_paths.py`'s discovery as a golden path needing the seven
  sections and a pinned output line, neither of which a video has.
  `kg-symbolic/` and `feed/` share a seeding path (`seed_kg_db.py`) that
  runs the real ingest+tag pipeline against the flows fixture with a live
  chat model, because the `--offline` stub's schema-shaped placeholder tags
  ("existing", "vocabulary", "title") produce a graph with nothing topical
  to show.
- `attest runs record FAMILY --arm NAME METRIC=VALUE...` (`2026-09-01`)
  writes the results/config JSON+YAML pair, `corpora.toml`, and
  `metric_direction.toml` entries a run needs deterministically, refusing
  before writing anything if a target already exists (`--force` to
  overwrite) or if a metric's ranking direction is undeclared (the same
  refusal sentence `runs.compare` prints) — replacing a five-step manual
  procedure whose declaration step small local models followed 0/15 of the
  time, against ≥0.91 on every file-shape step. `--dry-run` prints the
  `{"files": {relpath: content}}` manifest the command's own acceptance eval
  (`evals/run_record_eval.py --command`, 11/11) scores against the real
  ledger reader; `--scan` folds `runs scan` + `runs compare` into the same
  invocation. `src/attestation/record.py` is pure `plan()`/`undeclared()`
  plus one `write()` I/O function and a `merge_toml_table()` helper, with no
  `sqlite3` or `attestation.llm` import.
- The bundled skill split five ways (`2026-09-01`, landed from an Aug-30
  worktree): `attestation-setup` plus one skill per agent surface replace
  the single 39.5 KB `research-provenance` monolith; `attest install` now
  syncs all five into `~/.hermes/skills/` and every profile's skills tree,
  respects a `SKILL.md.<anything>` disable rename, and retires an installed
  monolith by renaming its `SKILL.md`, never deleting.
- Twelve golden-path worked examples under `examples/`, each runnable from
  a clean clone with a fixed-shape README and a `run.sh`
  (`a3387d5`..`e5e511b`, framework in `eae9e49`): the four retrofitted
  paths (`workspace/`, `flows/`, `prompt-evals/`, `agents/`, `e9f168e`,
  `45718bb`) plus real third-party integrations for MLflow (`836c9fc`,
  `4761a9f`), W&B (`dc496da`), Sacred (`fdaa22f`), DVC (`205e1c3`), Hydra
  (`620f300`), TensorFlow/Keras (`b3f1a2f`), citations (`492af29`), and
  bring-your-own model server (`871436b`). The catalogue and its ordering
  are enforced by `tests/test_golden_paths.py`, added in the same sweep.
- Five tracker read conventions in `ledger_adapters/generic.py`, each
  reading a real on-disk layout with no dependency on the tracker's own
  package: Sacred's `FileStorageObserver` (`be9ca47`), DVC's
  `dvc.yaml`/`dvc.lock` (`ec29ece`), and Hydra's `--multirun` sweep
  directories (`4b72989`) join the existing MLflow and W&B readers.
- Task corpora and model-free scorers for the reaction and explanation
  prompts (`e228d0e`, `d4f5741`), matching the treatment tagging already
  had — each gets a labelled corpus, a scorer independent of any model,
  and one public renderer that both the library and the eval script call
  (`d6dac2e`, `cb1ca3a`, `d3a04b0`).
- A stub OpenAI-compatible server (`365339a`) and a persona-reaction
  evaluation harness (`ae32960`) so the example flows and CI run fully
  offline, with `RESULTS.md` written only by a live run (`05fea1c`).
- The package states its own surface: `attestation/__init__.py` gained a
  docstring, `__version__` (read from installed metadata), and an
  `__all__` naming the modules meant to be imported; a `py.typed` marker
  now ships in the wheel; `[project.urls]` gained Homepage, Repository,
  Issues, and Changelog (`7bc20cf`).
- `tests/test_docstring_ratchet.py`: every public def under
  `src/attestation/**` now carries a docstring (103 were missing; ratchet
  baseline is 0 and only goes down), and `cli.py`'s `cmd_*` handlers share
  one source of truth with their argparse `help=` text (`7bc20cf`).
- This file and `CONTRIBUTING.md` (this change).

### Changed

- `generic.py`'s stem-family grouping now handles a bare split-token stem
  (`lr_0.001`) by falling back to the token's own name as the family,
  instead of returning no family at all — found by `examples/tensorflow/`'s
  real four-arm sweep (`ca08646`).
- The DVC comment stripper is now quote-aware, since a `#` inside a quoted
  YAML scalar is not a comment (`83783fd`); an earlier DVC review round
  also fixed a metrics-directory collision and a trailing-comment
  misattribution, each landing with its own test (`0735cf1`).
- The attribution-and-machine-path guard, originally scoped to
  `examples/flows/`, now covers all of `examples/**`, scans non-text
  files (e.g. TensorBoard's binary `.v2` event files) as raw bytes, and
  skips the ambient-`$USER` check on CI and for generic account names
  (`runner`, `root`, …) so it stops colliding with ordinary English
  words like "runner" in prose (`e5e511b`, and the CI-username finding
  recorded in `2026-08-28-golden-paths-design.md`'s Deviations section).
- The repo README's "Try it in 60 seconds" area became a "Golden paths"
  section pointing at the `examples/README.md` catalogue instead of
  describing each flow inline a second time; the quickstart block itself
  is unchanged (`e5e511b`).
- CI: a wheel-smoke step now asserts `py.typed` ships in the built wheel
  (`7bc20cf`); an earlier fix made the local ruff-format hook agree with
  CI's Markdown-fence formatting, after CI's `ruff format --check .`
  failed on design specs no local gate had ever touched (`ccf878f`).

### Fixed

- `attest claims` and `coverage` scanned to zero files in any checkout
  under a dotted directory (a git worktree in `.claude/worktrees/`):
  hidden-directory filtering now judges paths relative to the scanned
  root, not the absolute path.
- `attest claims` (the CLI) never ran the citation lint that the MCP tools
  (`cite.check`, `runs.claims_check`) already ran — found by
  `examples/citations/` exercising both paths against the same draft, and
  fixed to match (`4fb6007`, documented in `check_citations.py`'s
  docstring by `d323b52`).
- Sacred, DVC, and Hydra fixture generators were unpinned from the library
  version they were verified against; all three now pin and refuse to run
  under a different installed version, matching the convention W&B's and
  TensorFlow's generators already followed (`96c2386`).
- A macOS CI run staged "no `python3` anywhere," which does not reflect a
  real macOS box (Python 3 ships at `/usr/local/bin` there); the loud-
  failure test's assumption was corrected (`820006e`). A separate macOS
  failure came from a ruff `FAILED` line that ran to 123 columns — long
  enough that only the local terminal's tail hid it (`22e991a`).
- Two earlier CI-only failures: symbolic calls returned an rlimit error
  because the daemon refresh script never ran on macOS (`a3ed03c`), and
  the first green CI run needed a stubbed daemon test plus a Python build
  that can load the `sqlite-vec` extension (`d4ea750`).

[Unreleased]: https://github.com/mgoldey/attestation/compare/573e42c...HEAD
