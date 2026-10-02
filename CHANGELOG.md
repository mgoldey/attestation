# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
History before 2026-08-28 is not reconstructed — this file starts where a
narrative commit history and this project's day-by-day session logging
overlap, not at the repo's actual beginning. Each line below points at the
commit that carries the reasoning rather than repeating it; `git log
--oneline <sha>` on any entry shows the full message.

## [Unreleased]

### Fixed

- **`attestation-provenance` now says what to do without the tools.** An agent
  whose session had no `runs.*` tools guessed at the CLI (`attest
  claims_check`, a tool name) for 17 calls. The skill now gives the four real
  commands, says a claim checked against an empty ledger is `unsupported`
  because nothing was scanned, and that `attest claims` exiting 1 is a
  contradiction, not a failure. A new test parses every `attest <command>` a
  skill documents against the CLI.

## [0.3.0] - 2026-09-30

The Reading desk: a phone page of ranked papers whose verdicts come back as
clicks, the first new human signal the ranker has had since 2026-08-22.

### Added

- **The reader's bibliography.** Papers a reader rates useful, reads in full
  or asks about, and papers they ask the agent to save, are kept per persona
  in one generated `.bib` (`ATTEST_BIB_OUT`), each with a citation key that
  never changes and the reason it is there; a removal sticks. `cite.save`
  (the 50th tool) and `feed.ask(question="save this", item_id=...)` save and
  remove; `attest library bib` and the hourly refresh write the files.
  Migration 013. `ATTEST_BIB_PATHS` entries may now be folders, so an
  uploaded `.bib` is read without editing `.env`. See
  `docs/superpowers/specs/2026-09-30-bibliography-design.md`.
- **The Reading desk.** `attest desk build|import|refresh` renders today's
  ranked papers as one self-contained page with Useful / Not my area on each,
  and records the verdicts a reader gives there as clicks before the next
  ranking. The ranker had no new human signal since 2026-08-22; this is a
  gesture surface on the reader's phone. See
  `docs/superpowers/specs/2026-09-29-reading-desk-design.md`.
- **Migration 012: `users.feedback_since`.** The desk's import cutoff, set
  when a persona is created and whenever its feedback is purged, so
  `feed.persona_reset` is no longer undone by the page's saved verdicts.
  Existing personas get no cutoff.

## [0.2.5] - 2026-09-29

Two fixes from one live Discord turn: an answer's links now travel with their
titles, and a persona merged away stays merged.

### Fixed

- **A merged persona kept coming back.** Discord prefixes each message with
  the sender's display name, so the agent passed `user="Matthew Goldey"`
  instead of the persona `matt`; `personas.merge` folded that duplicate in
  and deleted the name, and the next read autocreated it again, empty. It
  regrew three times, and ratings made in Discord trained a persona nobody
  read from. Merged names are now aliases (migration 011, `persona_aliases`):
  reads, ratings and autocreate under them reach the kept persona.
  `attest persona-merge matt "Matthew Goldey"` does the merge from the CLI.
- **A listed paper could be linked to a different paper.** An answer named
  its titles and `refs` held their urls separately, so the agent paired them
  up itself -- and in a live Discord turn linked "Learning to Optimize through
  Solver-Grounded Self-Play" to 2609.12105, another paper from earlier in the
  conversation, instead of the 2609.34205 the tool returned. Each title in an
  answer is now a Markdown link to its own url, and the feed skill says to copy
  those links rather than re-pair titles with `refs`.

## [0.2.4] - 2026-09-29

Feed-ranker first on a hosted machine with no GPU: embeddings from a small
CPU embedder on the machine while chat goes to the customer's provider, a
first feed list an installer can choose, and a reader who can say what they
work on.

### Added

- **Embeddings from their own server.** `EMBED_BASE_URL` / `EMBED_API_KEY`
  send embeddings to a different server from chat -- a small CPU embedder on
  a hosted machine with no GPU (embeddinggemma measured 20 items/s on 2 vCPUs,
  380 MB resident) beside the customer's chat provider. Unset, embeddings
  follow `LLM_BASE_URL` as before. The chat key is never sent to a different
  embedding host.
- **`ATTEST_FEEDS`** names the feed list a first ingest subscribes to, so an
  installer can seed a field's journals instead of the packaged, mostly
  machine-learning list.
- **"I work on X" sets what the feed ranks for.** `feed.ask` routes "I work
  on", "I'm interested in", "my research is on" and "my field is" to the
  reader's interests, creating a first-time reader from their own words. On
  a fresh hosted machine nothing is tagged yet, so a reader created on sight
  started from a placeholder.

## [0.2.3] - 2026-09-28

Found by recording a demo of every surface for experimental and computational
scientists: research results carried the wrong DOIs and off-topic CrossRef
hits, `runs record` could not find what it had recorded, and `runs.ask` only
understood machine learning's way of asking which run won.

### Added

- **Demo recordings for every surface, and a narrated film.** The demos page
  now covers recording results, the reference library, literature research,
  a lab-and-simulation ledger example (a catalyst screen and a basis-set
  study, numbers labelled invented), real agent turns on each `*.ask` router,
  and the AgentMarkit install. `demos/film/build.py` stitches them into one
  film with a locally synthesized voiceover and captions.

### Fixed

- **`runs.ask` only understood ML's way of asking which run won.** "Which
  basis set did best?" was asked back ("comparing arms, listing runs, or
  checking a draft?") because the comparison rule knew "won", "winner" and
  "sweep" but not "did best", "worked best", "highest", "lowest" or
  "outperform". Those now route to `runs.compare`.
- **`attest runs record` could not find what it had just recorded.** The scan
  re-derived each run's family from its file name and ignored the `family:`
  its config declares, so `record lr-sweep --arm lr_3e4` scanned back as
  family `lr` and `record my_sweep` as `my-sweep`: `--scan` printed no
  comparison and `runs compare lr-sweep` found nothing. A family a run's own
  config declares now wins over the name heuristic, and `--scan` with nothing
  to compare says so instead of printing nothing.
- **Research hits carried another paper's DOI.** PubMed identifiers were read
  with `iter("ArticleId")`, which also walks the reference list, so a paper
  came back with the DOI (and could come back with the PMC id) of the last
  paper it cites. They now come from the article's own `ArticleIdList`.
- **CrossRef research returned off-topic papers dated 2115.** Sorting by
  publication date made the query match on any one word and put bogus future
  dates first ("Social Network Sites' usage among Greek students" for an
  interatomic-potentials query). CrossRef searches now rank by relevance, cap
  at today's date, and skip peer-review records.

## [0.2.2] - 2026-09-28

Found by installing the AgentMarkit research-assistant starter on the real
Agent37 image and asking it real questions: the install went green while the
agent's tools read an empty database. Plus the feed skill rewritten around
`feed.ask`, and feedback made reachable through it.

### Fixed

- **`runs.ask` asked "which family?" about the family it had just been
  told.** "Compare the kdsweep runs" routed to `runs.compare` and then
  declined for want of a `family` argument; gemma4:e2b passed only the
  question, twice. The family now comes from the question when the argument is
  absent (two named at once still asks back), and caveats are joined with `; `
  so two of them no longer run together into one sentence.
- **On a fresh machine the agent's tools read an empty database.** With no
  `ATTEST_DB` and no legacy skill-data database, the path falls back to
  `./hermes.db`. Ingest and the refresh script run in the checkout, but Hermes
  starts `attest-mcp` from its own directory and strips the parent
  environment, so every feed answer came from an empty file beside the one
  ingest had filled. That was measured on the Agent37 image: 1416 items
  ingested, none visible to the agent. `attest install` now pins
  `ATTEST_DB=<checkout>/hermes.db` in the checkout's `.env` when nothing else
  fixes the path, and `--check` reports the unpinned state.
- **A failed ingest skipped tagging.** The refresh script `attest install`
  writes exited as soon as `attest ingest` did. One refusing feed (arXiv's
  export API answering 406 for days) is enough to fail ingest while every
  other feed's items land, and those items then sat untagged. Tagging now runs
  regardless, and ingest's exit status is still the script's, so cron reports
  the failure.
- **The feed skill named twelve tools to a session that could call two.** The
  deployed feed surface serves `feed.ask` and `feed.tools` only (hiding the
  specifics was measured: the router chosen 26/26 alone, 1/26 beside them), but
  `attestation-feed/SKILL.md` taught `feed.list`, `feed.digest`, `feed.rate`
  and nine more. In real `hermes chat` turns the model called them, Hermes
  rewrote each name to `feed.ask` with the wrong arguments, every call failed
  validation, and the reader was told the feed was "unreachable". The skill is
  rewritten around `feed.ask` (293 → 163 lines) and a ratchet in
  `test_skill_files.py` counts hidden tools each surface skill names against
  its DEPLOYED session: feed 0, provenance 7, knowledge 12, symbolic 7, only
  down.
- **Feedback was unreachable.** `feed.ask` takes an optional `item_id` and
  `url`: `question="useful"` / `"not useful"` with an item records the verdict,
  "why is this here?" explains it, anything else opens it; a `url` subscribes or
  previews. On a surface that serves only `feed.ask`, "tell me which item, then
  I will call feed.rate" named a tool the agent could not call -- and no click
  had been recorded since 2026-08-22.
- **A digest made the agent invent papers.** Its answer named only the topic,
  and gemma4:e2b filled the list with titles that do not exist; named a few
  titles against twelve refs, it repeated them. A digest now answers like
  `feed.list`: item rows tagged with their topic, and `refs` exactly those rows.
- Reading an item answers with its text, not only its title.

Measured on 12 real `hermes chat` turns (gemma4:e2b, isolated `HERMES_HOME`,
a copy of the live database): 8/12 before. After the skill rewrite all 12
called only `feed.ask` with zero rewritten tool names or errors -- but the
digest turn still invented titles, so 11/12 answered correctly. After the
digest alignment fix, two re-runs of that turn each listed five real, distinct
titles.

## [0.2.1] - 2026-09-25

A hotfix, found by driving 39 realistic questions — nine lifted from real
Hermes transcripts that had gone wrong — through the real `attest-mcp` over
stdio against a copy of the live database. The first run passed 25 of 38, and
five of those passes were wrong on inspection. Every finding below is pinned by
a test at the layer it lived in, and three of those tests were mutation-checked
(each one fails with its fix reverted).

### Fixed

- **`feed.ask` raised a validation error on every research question.** 0.2.0's
  paper refs carried `paper_id` and no `item_id`, and `Ref.item_id` was
  required, so `Answer` rejected them — after the papers had already been
  stored. The regression test only checked `_compose`'s dict and never built
  `Answer`; the new one drives the registered tool on a real FastMCP server.
- **A failed research client was invisible.** The headline kept only the text
  before its first `;`, cutting "1 client(s) failed", and `caveat` was empty
  while arXiv returned 406 — so PubMed papers on Alzheimer's were presented as
  if they were the arXiv results. Failed clients and their HTTP status now reach
  `caveat`.
- **Answers that counted instead of naming.** `feed.digest` ("16 item(s) in 2
  topic(s)", zero refs though every nested item had an id and url),
  `runs.claims_check` and `runs.claims_coverage` now name what they found;
  contradicted and unsupported claims lead. `runs.detail` names its metric
  values; "what are my interests?" includes the interests text.
- **`sym.verify` answered "0" to a true identity** — `result` is lhs − rhs. It
  now answers with the verdict. It also takes an equation written with `==`
  instead of asking for both sides separately.
- **`runs.ask` reported `tool_used="runs.record"` while returning a run
  listing** — nothing had been written. It now says runs.record needs
  arguments `runs.ask` cannot carry.
- **Routing.** Questions that name a subject ("latest in memory systems for
  LLMs", "what is known about X") fell through to a clarifier the feed surface
  cannot act on, so the agent reported nothing found; they now search.
  "what feeds am I subscribed to?" was answered with *suggestions*, then — once
  that was fixed — routed to *add a feed*, because `"subscribe"` matched
  `"subscribed"`. `"sweep"` matched run names (`kdsweep_t4` went to compare).
  "the top of my feeds", "when was your recent scrape?", "are you learning?"
  and "my main research areas" each reach the right tool.
- **An unknown concept is refused with the concepts the reader probably
  meant** ("Memory System" → memory, working-memory…) instead of "call
  kg.concepts()", which lists 1403 names.
- **`feed.sources` says when feeds were last fetched**, newest and stalest.

### Known, not fixed here

arXiv's export API is returning 406 to this machine on every paced request
(nine at 20 s intervals). That is throttling on arXiv's side, not a malformed
request; 0.2.1 makes it visible rather than silent. `sym.*` calls take 5–9 s
because the sandbox's `spawn` child re-imports the whole server — measured, and
tracked separately.

## [0.2.0] - 2026-09-18

First published release. Nothing was released under 0.1.0 — the version
existed in `pyproject.toml` but no tag or PyPI artifact was ever cut, so
this is the repo's first public artifact despite the number.

### Added

- PyPI release path (`2026-09-11`): `.github/workflows/release.yml` builds
  and publishes on a `v*` tag through trusted publishing, refusing a tag that
  does not match `pyproject.toml`; a second console script named
  `attestation` makes `uvx attestation install` the whole install command.
  Version bumped to 0.2.0 for the first release.
- `attest install` hosted-models step (`2026-09-11`): with a non-Ollama
  `LLM_BASE_URL` the Ollama steps skip and `hosted_models` makes one
  embedding and one one-token chat request, reporting the server's own
  reason on failure (measured on NVIDIA NIM: 82 models listed, most chat
  models tried answered 410 end-of-life or 404 not-enabled, so a catalogue
  lookup would have passed a configuration that cannot run) and refusing an
  embedding model narrower than `EMBED_DIMS`. `.env.sample` and the install
  guide gain the hosted tier.

### Fixed

- `feed.research` / `attest research` with `--journal` on CrossRef
  (`2026-09-11`): the journal was passed as `query.container-title`, which
  CrossRef treats as a ranking hint, so a search scoped to the Journal of
  Chemical Physics returned Chemical Engineering Science 12 of 12 times. The
  client now resolves the journal name to an ISSN through `/journals` (title
  equality after normalisation, one cached request per name) and filters
  with `issn:`; an unresolved abbreviation falls back to the boost plus a
  client-side container-title match, which returns nothing rather than the
  wrong journal. PubMed's `[Journal]` term was already a filter and is
  unchanged.
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
- `attestation-feed` skill text (`2026-09-03`): two failures found testing
  `feed.ask` the same way, both live-verified fixed. With no toolset
  restriction on the hermes session, gemma4:e2b carried ~16k prompt tokens
  of hermes's own built-in tools alongside attestation's two, and either
  invented a missing precondition ("I need to know which feeds they
  subscribe to first") before calling a tool it had already correctly
  identified, or reasoned to the right call and then printed it as literal
  text instead of invoking it. The skill now says explicitly that `user`
  and `question` are always enough and that the example call syntax is
  documentation, not something to type; `demos/hermes/record.sh` passes
  `-t <mcp-server-name>` to restrict hermes to just that server's tools,
  which made both failures go away in repeated live testing. Neither fix
  alone was reliable in this testing; both together were, twice.

### Added

- Paper research (`2026-09-10`, spec `2026-09-10-paper-research-design.md`):
  standing topics as `research:` feeds (`attest sources add`,
  `feed.source_add`), searched every ingest through arXiv, PubMed and
  CrossRef; the `feed.research` tool and `attest research` for ad hoc search
  into the library; `reference_fulltext` filled by `attest library fulltext`
  and served by `cite.lookup` in windows; BibTeX rendered from library rows
  (`cite.lookup`, `attest library export`); items dedup by DOI/arXiv id
  across feeds; migration 009; `ATTEST_RESEARCH_WEB` on by default. 49 tools.
  Supersedes the unmerged 2026-09-04 design.
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

[Unreleased]: https://github.com/mgoldey/attestation/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/mgoldey/attestation/compare/v0.2.5...v0.3.0
[0.2.5]: https://github.com/mgoldey/attestation/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/mgoldey/attestation/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/mgoldey/attestation/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/mgoldey/attestation/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/mgoldey/attestation/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/mgoldey/attestation/compare/573e42c...v0.2.0
