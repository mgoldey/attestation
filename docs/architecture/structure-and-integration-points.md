# Repository structure: integration points, Cookiecutter DS, and the next reviewers

**Date:** 2026-08-29. A design note, not a spec: it records how this
repository is organised and why, compares that with the layout convention
most data-science projects start from, and proposes which review lenses
to run next on repository and experiment structure. Measured numbers are
as of `main` at `787823d`; the tests named beside them are what keep them
true.

## The answer in one paragraph

Cookiecutter Data Science standardises the *producer*: it tells a
researcher where to put raw data, processed data, models, notebooks and
reports so that the next person can guess. attestation standardises the
*consumer*: it defines what an agent, a CLI, or a test may ask — rank
these runs, check this claim, read this tracker's directory — and reads
whatever layout the producer already chose. The integration points are
the design. Each one is a contract with a test that guards it and a
document that names it, and none of them is a folder position. That is
why an agent can use this tool without learning a layout: it calls
`runs.compare`, and the tool surface, not the directory tree, is what was
measured against a 2B model until it routed.

## Cookiecutter Data Science: the ideas taken, the ideas left, and the difference

Cookiecutter DS is the convention most data-science projects start from,
and several of its opinions are this repo's opinions too. This is not a
compatibility target — attestation is not a project template and does
not promise to read any particular layout — it is the nearest published
philosophy to compare against, so the borrowings and the departures can
be named one by one.

The v2 template generates `data/{external,interim,processed,raw}`,
`docs/`, `models/`, `notebooks/`, `references/`, `reports/figures/`, a
`Makefile`, `pyproject.toml`, and a module with `config.py`,
`dataset.py`, `features.py`, `modeling/{train,predict}.py`, `plots.py`.
Its opinions page states ten principles. Each one, with whether the idea
is taken here, and in what form:

| Cookiecutter DS opinion | taken? | attestation's stance |
|---|---|---|
| "Data analysis is a directed acyclic graph" | taken, inverted | Provenance is a DAG too — runs → claims → citations — but it is *discovered from artifacts* (`ledger.scan`, `claims.check_claim`), never declared in a build file. `compare()` refuses to rank arms across corpora rather than trusting a declared edge. |
| "Raw data is immutable" | taken | The ledger only reads a workspace; it never writes into one. Example fixtures come from a committed generator that ran the real tool, then scrubbed — never edited to look right (`tests/test_golden_paths.py`). |
| "Data should (mostly) not be kept in source control" | half | The database is not. The three eval corpora *are* (`evals/*_cases.json`, 191 cases): small, labelled, and the thing the prompts are optimised against — they are the program, so they are versioned like one. |
| "Tools for DAGs" (Make) | left | None. `dvc.yaml` and Hydra's `multirun/` are read as conventions, not executed; the tool has no pipeline runner and does not want one. |
| "Notebooks are for exploration and communication, source files are for repetition" | left, deliberately | No notebooks. Thirteen golden paths (`examples/`) with a pinned "What it prints" line play the communication role, and `examples/flows/RESULTS.md` the exploration record. This is a gap a notebook-first reviewer would name (below). |
| "Refactor the good parts into source code" | taken | The same rule, applied to the tool itself: a seam is added when a test can name what it decomplects (`docs/superpowers/specs/2026-08-29-onion-seams-design.md`). |
| "Keep your modeling organized" | taken — it is the product | This is the product. The ledger organises runs *after the fact*, for projects that did not — five tracker layouts plus bare `results/` files, with caveats where the format cannot say what a reader needs. |
| "Build from the environment up" | taken | `uv.lock` with `--frozen` everywhere, CI on `only-managed` Pythons, models named exactly (`gemma4:e2b-it-q4_K_M`, `embeddinggemma`), and a docs site that builds `--strict`. |
| "Keep secrets and configuration out of version control" | transposed | Nothing leaves the machine, so there are no service secrets to keep out; the analogue here is *identity*: the attribution guard fails the suite on a home path, a username, or a repo URL in a committed fixture. |
| "Encourage adaptation from a consistent default" | taken | The golden path is the consistent default — seven sections, one of three prerequisite labels, a `run.sh` that must match the README — enforced by discovery, so a new one needs no test edit. |

What distinguishes this approach, stated so it can be argued with:

- **Interfaces over positions.** Cookiecutter's contract is a place
  (`data/processed/` *means* processed); attestation's is an interface
  with a test (`RunRecord` *means* a run, however it was laid out). A
  producer never has to move a file to be read; a consumer never has to
  know where one is.
- **The consumer is standardised, not the producer.** The template shapes
  the project that makes the results; this tool shapes what may be asked
  of results that already exist — and reads five tracker layouts and bare
  `results/` files as conventions of their own, with caveats where a
  format cannot say what a reader needs.
- **Provenance is discovered, then doubted.** A declared DAG is exact;
  a discovered one is caveated. `compare()` refuses to rank arms across
  corpora and `_caveats()` says what the artifacts cannot; the tool would
  rather return a caveat than a verdict it cannot back.
- **The agent is a measured consumer.** Every integration point below is
  a typed, bounded, refusable call that a 2B model was tested against;
  a directory convention has no schema, no envelope and nothing to
  measure.
- **Fully local, and identity stays out of the tree.** Nothing leaves the
  machine, so the secret to keep out of version control is not a
  credential but a person: paths, usernames and remotes are what the
  guard rejects.

## The integration points, catalogued

Every point below is something an outside party plugs into. The test
column is the answer to "how do I know it still holds".

| point | shape | who plugs in | guarded by | documented in |
|---|---|---|---|---|
| MCP tools | 46 tools in five namespaces — `feed.*` 20, `sym.*` 8, `runs.*` 8, `kg.*` 6, `cite.*` 4 — each with a schema whose bounds match the code, and one envelope | any MCP client; hermes-agent and Claude Code are the measured ones | `tests/test_architecture.py` (`test_every_tool_is_namespaced`, `test_tool_schemas_constrain_their_arguments`, `test_claude_md_tool_counts_match_the_live_surface`), `tests/test_tool_envelope.py` | `docs/guides/agents.md` |
| Surfaces | `ATTEST_TOOLS=feed\|provenance\|knowledge\|symbolic`; each shows an ask-router plus one companion unless `ATTEST_EXPAND=1` | an agent whose prompt budget cannot carry 85 KB of schemas | `tests/test_agent_surfaces.py` | `docs/guides/agents.md` |
| Ask routers | `mcp/routing.py`: question → tool, no model call; ambiguity returns options, never a default | a small model that misroutes a flat surface | `tests/test_ask_routing.py` (13/15 routed vs 8/15 flat, measured) | `docs/measurement-lessons.md` |
| The CLI | `attest <cmd>`; help text and docstring share one `HELP` table | shells, cron, `run.sh` in every golden path | `tests/test_cli.py`, `docs/reference/cli.md` equals a fresh render | `docs/reference/cli.md` |
| Model ports | `ports.py`: `ChatPort`, `EmbeddingPort`, `EmbedderPort`, `CitationPort` — structural Protocols | any OpenAI-compatible server; the offline stub in `examples/flows/stub_openai.py` | `tests/test_ports.py`, `tests/test_llm.py` | `docs/guides/install.md` |
| Tracker conventions | one reader per layout in `ledger_adapters/generic.py`, appended in `discover()`, caveats in `ledger.ADAPTER_CAVEATS` | W&B, MLflow, Sacred, DVC, Hydra directories as they are written | `tests/test_ledger_adapters.py`, one golden path each | `docs/guides/ledger.md`, `CONTRIBUTING.md` recipe |
| Result files | `results/`, `logs/`, JSON/CSV with nested metrics; corpus detected from the driver script's AST | projects with no tracker at all | `tests/test_ledger.py`, `examples/workspace/` | `docs/guides/ledger.md` |
| Claims | `claim:` lines in Markdown with an optional `cite=` key; five verdict kinds | a draft, a paper, a README | `tests/test_claims.py`, `tests/test_citations.py` | `docs/guides/claims-and-citations.md` |
| Prompt renderers | `features.tag_messages`, `simulate.reaction_messages`, `explain.explanation_messages` — one renderer per prompt; artifacts via `ATTEST_TAG_PROMPT` | an optimizer (DSPy GEPA today), an eval runner | `tests/test_tag_prompt.py` pins the default to its artifact | `CONTRIBUTING.md` "Prompt optimisation" |
| Eval corpora | `evals/{tagging,reaction,explanation}_cases.json`, train/dev, a `note` per case | anyone who saw a prompt fail | `tests/test_{tagging,reaction,explanation}_eval.py`, `tagging_eval.gate()` | `docs/guides/evals.md` |
| Feeds | `[[feeds]]` in `feeds.toml` seed the first ingest; `[[candidates]]` are offered, never subscribed unasked | RSS/Atom sources | `tests/test_feeds.py` | `docs/guides/feed.md` |
| The skills | `src/attestation/skills/attestation-{setup,feed,provenance,knowledge,symbolic}/SKILL.md` — how an agent should behave with each surface's tools (never ask for a persona name; extract verdicts from discourse; relay caveats verbatim) | hermes-agent, Claude Code | `tests/test_skill_files.py`, `tests/test_install_skills.py` | `docs/guides/agents.md` |
| Config emitters | `emit.py`: `hermes_servers()`, `claude_agents()` — generated agent config, checked against the live tree | an agent's config file | `tests/test_emit.py` | `docs/guides/agents.md` |
| The database | one SQLite file plus `sqlite-vec`; Datasette opens it with the extension | a browser, `datasette.yml` | `tests/test_db.py` | `docs/guides/ledger.md` "Browsing" |
| Golden paths | `examples/<name>/` with seven sections and a `run.sh` that matches | a newcomer; CI, which runs every `none` path | `tests/test_golden_paths.py` | `examples/README.md` |

Fourteen points, and the directory tree is not one of them. That is the
philosophy stated as a list: an outside party integrates by satisfying a
contract that has a test, and never by placing a file where the tool
expects to find it.

### Why MCP is the easy half of the agent interface

The measurements that shaped the surface are the argument:

- A flat surface of 37 tools routed 8 of 15 questions on `gemma4:e2b`; a
  deterministic router in front of it routed 13; an LLM swarm 7.3 at
  twice the latency. The *surface* was the lever, not the model.
- The same 2B model could not render a ten-item feed payload — it looped
  truncate-apologise-redump — so `feed.list` defaults to 4 and caps at
  13, both asserted against a 7000-character ceiling
  (`tests/test_response_size.py`). A tool that returns less than the
  agent can carry is easier to use than a richer one.
- With no persona named in its system prompt, the agent passed
  `user="user"` on 9 of 9 calls; naming it fixed 9 of 9. The prompt is
  part of the interface, which is why the skill ships beside the tools.
- In hermes-agent, 67 tool schemas cost 85 KB *every turn* while 68
  skills cost 7 KB; the `filament` server alone was 46% of the budget.
  Hence surfaces: hide tools, not skills.

None of this is possible with a folder convention, because a folder
convention has no schema, no envelope, and nothing to measure a small
model against. MCP gives the agent a typed, bounded, refusable call; the
tests give the maintainer a way to know the call still means what the
docs say.

## The next reviewers

The seams review used three lenses (Hickey, Hettinger, Karpathy) on
module internals. Repository and experiment *structure* is a different
question and wants different lenses. Candidates, with the question each
would ask of this repo and what it would probably find. These are lenses
— the published positions the person is known for — not impersonations,
and a review under one is read as "what this lens shows", never as the
person's opinion.

| lens | the question it asks here | likely finding | run it? |
|---|---|---|---|
| **Michael Feathers** (*Working Effectively with Legacy Code* — the origin of "seam") | where can behaviour be altered without editing in place? | would re-derive the seams spec and rank the sensing points (what a test can observe) — the right *whole-branch reviewer* for that plan | yes, as the reviewer of the seams plan |
| **Gary Bernhardt** ("Boundaries": functional core, imperative shell) | is every decision in a pure function and every effect in a thin shell? | the same nine cuts, plus which shells (`cli.py`, `mcp/feed.py`) still decide things | merge with Feathers |
| **John Ousterhout** (*A Philosophy of Software Design*: deep modules, define errors out of existence) | are the 46 tools deep or shallow? does the envelope define errors away or move them? | the surface is wide by design and measured; the hand-rolled `{"ok": …}` in `feeds.py` is the error that should not exist (seam 9) | yes — the surface-depth review nobody has run |
| **Hadley Wickham** (tidy data, *tidyverse design principles*) | is every table one observation per row, and do the functions compose? | `RunRecord`/`Metric` are long-format already; tool envelopes are wide dicts a client re-tidies; `family_of` is a parser where a column would do | yes — the data-shape review |
| **Greg Wilson** (*Good Enough Practices in Scientific Computing*) and **Karl Broman** (reproducible research organisation) | what does the ledger *require* of a workspace, and is that the minimum? | the `results/` convention and corpus-from-AST are close to "good enough"; the gap is a stated minimum a project can check itself against — `generic.diagnose_empty()` already explains an empty scan, but no page states the minimum up front | yes — the experiment-structure review |
| **Andrew Ng** (data-centric AI) | are the corpora the artefact being improved, with error analysis per case? | yes for tagging (51 cases with notes, GEPA gate); reaction and explanation have corpora and no optimizer; profile synthesis has neither (seam 10) | later, once seam 10 lands |
| **Chip Huyen** (*Designing ML Systems*: feedback loops, drift) | where does the feed's training signal come from, and can it poison itself? | click provenance already separates `ui/agent/bootstrap/simulated/implicit`; only positives are inferred; the open question is drift of the tag vocabulary frozen 2026-08-27 | later |
| **Simon Willison** (small tools, SQLite everywhere, documentation-driven) | is every piece of state one `datasette` away, and does every feature ship with its docs? | the DB opens in Datasette (`datasette.yml`); the docs-are-tested rule is the same instinct; would ask for a plugin hook where surfaces are | low priority — mostly agreement |
| **Jeremy Howard** (nbdev, notebooks-first) | where is the exploratory record, and why is none of it executable prose? | the dissent: golden paths are prose *with* a pinned output line, but no notebook renders a result inline; would propose `examples/*/README.md` be executed, which `test_golden_paths` half does | run as the designated dissent |
| **Dmitry Petrov** (DVC: declare the DAG) | why discover provenance from artifacts instead of declaring it? | the second dissent: declared DAGs are exact where discovery is caveated; the answer is that the tool serves projects that never declared one, and reads `dvc.yaml` when they did | run as the designated dissent |
| **Peter Bull** (Cookiecutter DS) | which of the ten opinions survive when the producer is not yours to standardise? | the table above is the first answer; a review would test whether "interfaces over positions" holds at every integration point or only at the ledger | later, after the seams land |
| **the 2B agent** (not a person: the measured consumer) | does a small model route, render and refuse correctly? | the lens this repo already runs on every surface change; keep running it | always |

Recommended next round, in order: Ousterhout on the tool surface,
Wickham on data shapes, Wilson + Broman on what a workspace must
provide, with Howard and Petrov as the two designated dissents so the
round argues with itself. Feathers/Bernhardt is reserved as the
whole-branch reviewer of the seams plan. Each review follows the same
protocol as the seams round: read-only, the same measurements, findings
with `file:line`, a "leave alone" list, and a convergence rule before
anything becomes a spec.
