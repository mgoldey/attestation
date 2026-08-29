# Contributing

Contributions are welcome, and the most useful one is small: tell us about
a run directory, a draft, or a feed this tool should have handled and did
not. attestation is meant to read the research you already have on disk
without asking you to change how you work, so every layout it cannot read
is a bug report waiting to happen. Everything here runs locally under the
MIT licence; nothing you send it leaves your machine, and nothing you
contribute should carry anyone's machine paths or names.

## Ways in, easiest first

| you want to… | do this | what guards it |
|---|---|---|
| **report a run layout it cannot read** | open an issue with the output of `attest runs scan <dir>` and a *scrubbed* `find <dir> -maxdepth 3` — never the raw tree (it carries your paths) | — |
| **report a claim it checked wrongly** | open an issue with the claim line, the metric it should have matched, and `attest claims` output | — |
| **add a feed source** | a `[[feeds]]` entry in `src/attestation/feeds.toml` (seeds the first ingest) or a `[[candidates]]` entry in `feed_candidates.toml` (offered by `feed.source_suggest`, never subscribed unasked) | `tests/test_feeds.py` |
| **add a golden path** (a runnable, documented worked example) | `examples/<name>/` — recipe below | `tests/test_golden_paths.py` |
| **teach the ledger a tracker's directory layout** | a reader in `src/attestation/ledger_adapters/generic.py` — recipe below | `tests/test_ledger_adapters.py` |
| **add a labelled eval case** for a prompt that misbehaved | an entry in `evals/{tagging,reaction,explanation}_cases.json` with a `note` naming the failure | `tests/test_{tagging,reaction,explanation}_eval.py` |
| **add an MCP tool or CLI command** | `src/attestation/mcp/<surface>.py` via `@tool` in `mcp/_tool.py`; `cli.py`'s `HELP` table | `tests/test_architecture.py`, `tests/test_cli.py` |
| **fix or clarify docs** | `README.md`, `docs/guides/*.md`, `docs/concepts.md` — every link is checked by `mkdocs build --strict` | `tests/test_docs_site.py` |
| **add a subsystem** | a design spec in `docs/superpowers/specs/` first (see below), then the code | `tests/test_architecture.py` |

Issues: <https://github.com/mgoldey/attestation/issues>. A pull request
against `main` is the way to send code; the gates below run on it.

## Set up in five minutes

```bash
git clone https://github.com/mgoldey/attestation
cd attestation
uv sync
uv run pre-commit install
uv run pytest -q
```

Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). No model server is
needed to develop: the suite, the ledger, the claim checker and the
symbolic tools are pure local computation. The feed, tagging and
explanation paths talk to any OpenAI-compatible server at `LLM_BASE_URL`
(Ollama, vLLM, llama.cpp, LM Studio) — `docs/guides/install.md` covers
that, and `attest install --check` reports what it finds without changing
anything.

If `uv run pytest` ever fails with "No module named 'attestation'" or
"Failed to spawn", the cause is a stale `.venv/`, not uv — `uv sync`
rebuilds it. (This repo lost weeks to routing around that once; the story
is in `CLAUDE.md`'s "Working here".)

CI runs on Linux and macOS, Python 3.12 and 3.13. Windows is untested: the
docs site uses symlinks under `docs/`, which need `git config core.symlinks
true` or Developer Mode at clone time (`mkdocs.yml` explains). A Windows
report — even "it works" — is a welcome issue.

## Before you open a pull request

`uv run --frozen pre-commit run --all-files` runs the eight gates CI runs,
defined in `.pre-commit-config.yaml`; `docs/guides/testing.md` says what
each catches and why it exists. Two things that catch everyone out:

- **`--all-files` means tracked files.** `git add` a new file first, or the
  hooks never see it and a green run proves nothing about it.
- **Read the per-hook `Passed`/`Failed` lines, not the tail.** A failing
  hook does not stop later hooks, so the last line printed is often an
  unrelated success.

The suite (~70s) is the gate that has mattered most: this repo's recurring
failure mode is a test that passes against the bug it was written to catch,
so a change that fixes a bug should come with the test that failed before
the fix. Nothing here asks for `--no-verify`; CI catches a bypassed hook on
push.

## Where things live

| directory | purpose | guarding test |
|---|---|---|
| `src/attestation/` | the library: ledger, claims, citations, feed ranking, knowledge graph, symbolic math, entry points | `tests/test_architecture.py` (layering rules), `tests/test_docstring_ratchet.py` |
| `src/attestation/mcp/` | the MCP tools, namespaced `feed.*`/`sym.*`/`kg.*`/`runs.*`/`cite.*`, and the four surfaces | `tests/test_mcp_server.py`, `tests/test_agent_surfaces.py`, `tests/test_tool_envelope.py` |
| `src/attestation/ledger_adapters/` | readers for tracker directory conventions (W&B, MLflow, Sacred, DVC, Hydra) and nested result files | `tests/test_ledger_adapters.py` |
| `src/attestation/skills/` | the `research-provenance` skill an agent installs to use the tools well | `tests/test_skill_files.py` |
| `evals/` | labelled corpora and scorers for every model-driven prompt (tagging, reaction, explanation) | `tests/test_tagging_eval.py`, `tests/test_reaction_eval.py`, `tests/test_explanation_eval.py` |
| `examples/` | golden paths: runnable, README-documented worked examples | `tests/test_golden_paths.py` |
| `docs/guides/` | the seven how-to guides a collaborator's question maps to | `tests/test_docs_site.py` |
| `docs/superpowers/` | design specs (written before the code) and their implementation plans | `tests/test_architecture.py::test_the_docs_index_lists_every_source_and_test_file`, `tests/test_docs_site.py::test_spec_index_is_a_fresh_render` |
| `tests/` | the suite itself | `uv run pytest` |
| `scripts/` | generators the docs site depends on (CLI reference, spec index), the complexity ratchet, the mutation-testing helper | `tests/test_docs_site.py` |

## Recipe: teach the ledger a tracker's layout

The ledger reads five trackers' directories as conventions of their own —
one reader each in `src/attestation/ledger_adapters/generic.py`
(`_wandb_runs`, `_mlflow_runs`, `_sacred_runs`, `_dvc_runs`, `_hydra_runs`),
appended in `discover()`. To add a sixth:

1. **Start from a real directory.** Write `examples/<tracker>/generate.py`
   (or `.sh`) that runs the real library, pinned to one version, on real
   data for a few seconds, then scrubs attribution and machine paths. The
   committed fixture is that output — never hand-written to look real,
   because hand-written fixtures encode what you *think* the tool writes.
   `examples/wandb/generate.py` is the model; it even decodes W&B's binary
   log because offline mode writes no summary file.
2. **Write `_<tracker>_runs(root, seen)`** returning `RunRecord`s with final
   metric values (not curves), no direction inference, and the `seen` set
   respected so a directory is never read twice. Record what the format
   cannot tell you in `ledger.py`'s `ADAPTER_CAVEATS` — `runs.compare`
   surfaces those rather than guessing.
3. **Append it in `discover()`** and add cases to
   `tests/test_ledger_adapters.py` that read the fixture and assert the
   arms, families and metrics you know are in it.
4. **Make it a golden path** (next recipe), so a reader can run it and see
   the arms ranked. `docs/guides/ledger.md` gets a row in the tracker table.

## Recipe: add a golden path

`examples/<name>/` is the worked-example format
(`docs/superpowers/specs/2026-08-28-golden-paths-design.md`). Discovery is
by directory — nothing in the tests needs editing — and every path needs:

1. **`README.md` with exactly these seven `## ` sections, in order:** *What
   you get*, *Prerequisites*, *Run it*, *What it prints*, *What it
   demonstrates*, *When it goes wrong*, *Next*.
2. **`Prerequisites` names one of three labels**, verbatim: `none — pure
   local computation`, `a model server at LLM_BASE_URL`, or `network`.
3. **`run.sh`** (`#!/usr/bin/env bash`, `set -euo pipefail`, executable)
   containing verbatim every fenced command in *Run it* that starts with
   `uv run`, `attest`, `./run.sh`, `export ` or `ATTEST_` — the two are
   checked against each other so they cannot drift.
4. **A row in `examples/README.md`'s catalogue** with the same label, in the
   table's order (`none` first, then `network`, then model-server paths;
   alphabetical within each group).
5. **Fixtures from a real library come from a committed generator** that
   pins the library's version and scrubs the output (recipe above).
6. **No committed file carries** an absolute `/home/` path, a `github.com`
   URL, a username, or (outside the scrubber's own source and docs) the
   `mlflow.user` tag or a `git@` remote — checked byte-for-byte, binaries
   included, by
   `tests/test_golden_paths.py::test_no_committed_example_carries_attribution_or_machine_paths`.

If the label is `none`, CI actually runs `run.sh` and asserts one pinned
line from *What it prints* against real stdout, so that section cannot be
aspirational.

## Rules the suite enforces

Each of these exists because its absence once cost something specific; the
test named is the one that will tell you.

- **A spec before a subsystem.** Every non-trivial feature has a design doc
  in `docs/superpowers/specs/` written before the code and a plan in
  `docs/superpowers/plans/`; the spec records *why*. Read the one for a
  subsystem before changing it — `CLAUDE.md`'s docs index maps code to
  spec — and write one before adding a new subsystem. Small fixes and
  docs do not need one.
- **The docs index in `CLAUDE.md` lists every source and test file.**
  `tests/test_architecture.py::test_the_docs_index_lists_every_source_and_test_file`
  fails if a new file's name is missing from the `{...}` group for its
  directory. Add it in the same commit as the file.
- **A docstring on every public def** under `src/attestation/**`, at any
  nesting depth.
  `tests/test_docstring_ratchet.py::test_every_public_def_has_a_docstring`
  pins the count of missing ones at 0 and names the offending `file:line`.
  Write the docstring this repo writes: what it is for and any measured
  rationale, not a restatement of the signature. `cli.py`'s `cmd_*`
  handlers take their first line from the `HELP` table that also feeds
  argparse (`tests/test_cli.py::test_every_cmd_docstring_is_its_helps_first_line`),
  so a new subcommand writes its help string once.
- **MCP tools are namespaced and never repeat their namespace**
  (`kg.path`, not `kg.kg_path`):
  `tests/test_architecture.py::test_every_tool_is_namespaced` and
  `::test_no_tool_repeats_its_own_namespace`. A tool body returns only what
  it computed; `@tool` in `mcp/_tool.py` owns the connection, the user
  lookup and both envelopes. An expected refusal is `raise ToolError(msg)`;
  anything else is a bug.
- **`# noqa: BLE001` is a policy, not a swallow.** There is no blanket
  per-file exemption (there used to be one, pointing at a path from before
  a rename, enforcing nothing). Each site carries an inline reason, and
  `tests/test_architecture.py::test_claude_md_noqa_inventory_matches_the_tree`
  asserts the count against the "Reliability contract" line in `CLAUDE.md`.
- **Nothing offline reaches the network.** The one exception is
  `citations.WebReader`, which exists only when `ATTEST_CITATION_WEB` is
  set, decided at construction time. A change that adds a network call
  anywhere else needs a spec and a flag of the same shape.
- **No attribution or machine paths in committed files.** Mechanically
  enforced under `examples/**`; the rule applies everywhere. Run the scrub
  step for any fixture generated by a real tool.
- **Generated pages are never hand-edited.** `docs/reference/cli.md` and
  `docs/site/specs.md` are rendered by `scripts/`; a test asserts each
  equals a fresh render. `mkdocs.yml`'s comment block says which page comes
  from where; `uv run --group docs mkdocs build --strict` checks every
  cross-reference before you push.

## Local setup, in full

The five-minute setup above is everything the suite needs. The rest is
optional and layered, so you can stop at whatever your work touches:

- **A model server**, only for the feed, tagging, reactions and
  explanations. `LLM_BASE_URL` points at any OpenAI-compatible server —
  Ollama, vLLM, llama.cpp, LM Studio — and nothing in `src/` names one
  vendor. The defaults are Ollama's `gemma4:e2b-it-q4_K_M` for chat and
  `embeddinggemma` for embeddings, chosen because they fit an 8 GB card
  and because every number in `docs/measurement-lessons.md` was measured
  on them. `attest install` sets this up idempotently; `attest install
  --check` only reports. `attest warmup` holds the models loaded for
  `OLLAMA_KEEP_ALIVE` (30 min), deliberately not forever — a pinned model
  once OOM-killed the box.
- **The database** is one SQLite file with the `sqlite-vec` extension.
  `resolve_db_path()` picks, in order: `--db`, `ATTEST_DB`, the hermes
  skill's `data/hermes.db` if it exists, else `./hermes.db`. A fresh file
  is empty — no demo personas — and `attest bootstrap-persona <name>`
  creates one. Tests use `conftest.py`'s `seeded_db()`, never your file.
- **The MCP server** (`attest-mcp`) is spawned once per agent session and
  holds that code until it dies. After editing anything under
  `src/attestation/mcp/`, run `attest reload`: it stops every live server
  and the next tool call respawns one. `hermes mcp test` does not catch
  staleness, because it spawns a fresh process and reports the code on
  disk. `ATTEST_TOOLS=feed|provenance|knowledge|symbolic` serves one
  surface; unset serves all; a typo raises rather than serving everything.
- **Offline everything.** `examples/flows/run_all.py --offline` drives the
  ingest, tagging, reaction and MCP paths against
  `examples/flows/stub_openai.py`, a stdlib server speaking
  `/v1/embeddings` and `/v1/chat/completions`, which is what the `flows`
  CI job runs. Its numbers are about the stub, not a model; `--live` is
  the only mode that writes `examples/flows/RESULTS.md`.

`docs/guides/install.md` has the manual steps behind `attest install`, and
`docs/guides/agents.md` the per-agent registration.

## The onion, as it actually stands

Contributors arriving from a ports-and-adapters background will look for
the layers; here is where they are and, as importantly, where they were
deliberately not built.

- **Ports** (`ports.py`): `ChatPort`, `EmbeddingPort`, `EmbedderPort`,
  `CitationPort` — structural `Protocol`s the domain depends on. `llm.py`
  implements the first two against any OpenAI-compatible server;
  `EmbedderPort` is separate because `embed.py` prompts documents and
  queries asymmetrically. There is **no repository protocol**, on purpose.
- **Domain** (`ledger`, `claims`, `citations`, `rank`, `kg`, `features`,
  `explain`, `simulate`, `symbolic`, `corpus`): the logic. Some of it
  still speaks `sqlite3` directly, and `kg.build_graph()` is the model of
  where it should go — it takes `(item_id, tag)` pairs, not a connection,
  and is tested with no database at all.
- **Presentation** (`cli.py`, `server.py`, `mcp/*.py`): thin. In
  `mcp/_tool.py`, `@tool` owns the whole per-call ritual — the connection,
  the user lookup, the success and failure envelopes — so a tool body
  returns only what it computed. `mcp/routing.py` turns a question into a
  tool with no model call; `mcp/ask.py` fronts it with a typed `Answer`.

What the suite enforces (`tests/test_architecture.py`): the module import
graph is acyclic; each `mcp/` module stays under a measured code-line cap;
every tool body is reachable without FastMCP; no SQLite connection is held
across requests; every tool is namespaced and its schema bounds match the
code. Deferred `from attestation import ...` inside function bodies are
lazy loads that keep `attest --help` at 0.22 s (`test_cli_help_stays_fast`)
— do not "fix" them.

What was rejected, and why it matters for the next seam: the full onion
(`docs/superpowers/specs/2026-08-21-onion-refactor-design.md`, superseded
the same day by `2026-08-21-tool-surface-design.md`) proposed repository
protocols, SQLite implementations, fakes and service facades. A review
through Rich Hickey's simple-versus-easy lens found it relocated the
tangle rather than decomplecting it: a 34-method `FeedRepo` was a bag of
queries wearing an interface, and the one real braid (`build_graph`
taking a connection) cost one signature change to remove. The standing
rule is the one that review left: **a seam is added when a test can name
what it decomplects**, never for symmetry. If you see a braid — domain
code that cannot be exercised without a database, an envelope built by
hand, a policy hidden in a broad `except` — the contribution is the
smallest signature change that lets a DB-free test name it, with the
`# noqa` policy left where the local knowledge lives (`rank.py` serving a
stale cached vector when the embedder is down is a decision no outer layer
can make).

The seams currently proposed under that rule — nine cuts that three
independent review lenses converged on — are in
`docs/superpowers/specs/2026-08-29-onion-seams-design.md`; read it before
proposing a tenth, since its "Refused" list records what a reviewer will
say no to.

## Prompt optimisation with DSPy

Every model-facing prompt in this repo is **data with one renderer**, so an
optimizer and the shipping code cannot disagree about what the model saw:
`features.tag_messages()`, `simulate.reaction_messages()` and
`explain.explanation_messages()` are the only places those prompts are
rendered, and `attest tag`, the eval runners and the optimizer's DSPy
adapter all call them. `ATTEST_TAG_PROMPT` loads an artifact from
`evals/prompts/*.json` (validated before any model call); unset, the
hand-embedded default is used verbatim, and `tests/test_tag_prompt.py`
pins it to the artifact it came from.

Each prompt has a labelled corpus with train/dev splits and a `note` on
every case naming the failure it targets: `evals/tagging_cases.json` (51),
`evals/reaction_cases.json` (100), `evals/explanation_cases.json` (40).
The `bait-*` cases are the live failure mode (a generic, on-vocabulary tag
for an off-vocabulary item; a "match" claimed between a termite-feed paper
and "advanced topics like AI"), and dev holds them so an optimizer never
trains on the thing it is being scored against.

Only tagging has an optimizer today: `evals/optimize_tagging.py` runs
DSPy's GEPA, instruction-only (demonstrations chosen from the labelled
cases and scored on them would be a tautology), with `dspy` in the
`optimize` dependency group — `uv run --group optimize python
evals/optimize_tagging.py`; `tests/test_tag_prompt.py` asserts nothing
under `src/` mentions it. The default budget of 300 metric calls is about
25 minutes on `gemma4:e2b`. Its output is a candidate artifact, not a new
default: `tagging_eval.gate()` decides — **not worse** than the baseline on
the primary model, better on at least two others, and no wider a spread
across them — measured by `evals/transfer_matrix.py` and committed as a
dated record under `evals/prompts/`. That gate is sample-sensitive (a
`repeat=2` run passed and a `repeat=1` re-run failed on the same prompt),
which is why the record is a committed artifact rather than a sentence.

Ways to contribute here, in order of leverage: a new labelled case for a
failure you saw (with its `note`); an optimizer for reaction or
explanation, which have the corpus, the scorer (`score_one`) and the
shared `EvalResult`/`gate`/`spread` but no optimizer yet — copy
`optimize_tagging.py`'s adapter shape so the model still sees the
production prompt; a transfer run on a model family the matrix has not
seen. `docs/guides/evals.md` and `examples/prompt-evals/` show a live run.

## Commits and pull requests

A commit message is a plain sentence saying what changed and why — `git
diff` already lists the files. `git log` shows the voice. The
`Co-Authored-By` trailer on agent-authored commits is that convention, not
something asked of you.

Send a pull request against `main` with the gates green; the maintainer
pushes directly to `main`, which is a description of a small repo's
workflow, not a recommendation. A PR that adds a test which fails before
the change and passes after it is the easiest kind to merge. If you are
unsure whether a change wants a spec, open the issue first — a paragraph
there is cheaper than a spec nobody needed.

## Changelog

`CHANGELOG.md` follows Keep a Changelog: one line per change under
`Unreleased`, by area, pointing at the commit that carries its reasoning.
Add a line for anything a user would notice.
