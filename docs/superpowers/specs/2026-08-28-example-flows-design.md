# Example flows: every agent helper, driven end to end

**Date:** 2026-08-28
**Status:** design; implementation follows in the plan of the same date.
**Depends on:** agent surfaces (`2026-08-22-agent-surfaces-design.md`),
tracker adapters (`2026-08-22-tracker-adapters-design.md`), the tagging eval
(`2026-08-23-dspy-prompt-optimization-design.md`), `docs/measurement-lessons.md`.

## Problem

The repo has three documented, tested golden paths (README quickstart, the
`examples/workspace` walkthrough, the prompt eval → transfer gate) and 46 MCP
tools across four agent surfaces. What it does not have:

- **Nothing drives the MCP server as a client.** Every test calls `_impl`
  functions or registers a `FastMCP` in-process. `attest-mcp` over stdio —
  the path every agent actually takes — has never been exercised by code in
  this repo. `CLAUDE.md` records finding two servers running code five
  commits stale; a client-side check is the only kind that would have seen it.
- **No flow exercises the four surfaces as a person would use them.**
  `ATTEST_TOOLS=feed|provenance|knowledge|symbolic` each serve two entry
  points and expand to 21/7+1/6+1+4/8 tools; the surfaces have a live test
  of their *registration* (`test_agent_surfaces.py`) and none of their
  *behaviour* through the server.
- **No precision, recall or AUC is printed anywhere.** `rank.evaluate_user`
  computes a cross-validated AUC and `attest eval` prints it, but only for a
  persona that already has clicks; the LLM-driven half of the feed
  (`simulate.react_to_item`, the tagging prompt, explanations) is scored by a
  rubric (`evals/tagging_eval.py`) or not at all.
- **The MLflow ledger reader has never met a real directory.** The spec says
  so, `generic.py:469` says so in capitals, the test module says so. Its
  fixtures were transcribed from documentation by the reader's own author —
  the repo's named failure mode with extra steps.
- **Every documented number was measured by hand.** The transfer matrix is
  scripted; the routing 13/15, the persona 9/9, the Telegram 5/5 were
  one-off sessions. `measurement-lessons.md` §3 exists because of that.

## Scope

A directory `examples/flows/` containing four scripts and one fixture, a
dependency group, a CI job, and a results document. Everything runs in two
modes:

- **offline** — a stub OpenAI-compatible model server; deterministic; runs in
  CI; proves the *plumbing* end to end. Its numbers are about the stub and
  every output line says so.
- **live** — Ollama at `LLM_BASE_URL`; the mode that produces the numbers
  recorded in `examples/flows/RESULTS.md`.

The offline mode is not a way to make the numbers look good. It is what makes
the flows *un-rottable*: `test_examples.py` pins the README quickstart for the
same reason.

### Not in scope

- A Sacred convention for the ledger. Sacred's `FileStorageObserver` layout
  (`<dir>/<id>/{config,metrics,run}.json`) is a real convention and would be
  welcome under the adapter rule, but it is a ledger feature, not a flow, and
  the corpus-ledger spec already argues Sacred is push-only. Recorded as the
  first candidate follow-up.
- Changing any tool's behaviour to make a flow pass. A flow that fails
  against the real server has found a bug; the bug is fixed in its own
  commit with its own test, and the flow stays as the thing that found it.
- A model-quality benchmark. The persona eval scores agreement with a
  fixture's labels on ~40 items. It is a smoke test with a number attached,
  and is presented as one.

## Components

```
examples/flows/
  README.md              what each flow shows, how to run, what the numbers mean
  RESULTS.md             measured live, dated, model named -- written by run_all --write-results
  corpus/
    labelled.xml         ~40 Atom entries, real-shaped, hand-written (no scraping)
    labels.json          {guid: {persona_name: true|false}} -- the ground truth
    personas.toml        two personas: name + interests text
  stub_openai.py         stdlib http.server: /v1/chat/completions, /v1/embeddings
  persona_eval.py        the LLM-driven eval: precision, recall, AUC
  mcp_e2e.py             stdio client that calls every tool on every surface
  training/
    train_mlflow.py      four arms, one family, < 30 s, logged to ./mlruns
    mlruns/              COMMITTED output of one run (the real directory)
    FINDINGS.md          claims against those runs, one deliberately wrong
  run_all.py             runs the above; --offline | --live; --write-results
```

### The fixture: `corpus/`

`labelled.xml` is an Atom feed of about forty entries across four topics
(synthesis chemistry, ML systems, a third topic neither persona wants, and
a handful of generic-science bait items of the kind the `bait-*` tagging
cases target). Each entry has a title, a two-to-four-sentence summary, a
stable `id` (the guid) and a `link`. It is hand-written, not scraped: the
labels are the point, and labels on real abstracts would be an opinion about
somebody else's paper.

**Amended 2026-08-28: the entries carry no date.** They originally each had
an `<updated>` in August 2026, which made the demonstration decay: `feed.list`
and `rank_items` default to a fourteen-day window, so the fixture would have
shown fewer items every week and none at all by mid-September, with no test
going red. The entry-level `<updated>` elements were removed (the feed-level
one stays), so `run_ingest` falls back to `COALESCE(?, datetime('now'))` and
every ingest dates the corpus to now. `test_flows_fixture.py` asserts no entry
carries a date. `mcp_e2e.py` still passes `since_days: 3650` on `feed.list`
for the same reason, which remains a valid argument.

`labels.json` maps each guid to a verdict per persona. Every item has a
label for both personas; roughly a third are positive for exactly one,
some are negative for both, none is positive for both. `personas.toml`
carries the two personas' interest strings — the same strings a user would
type into the onboarding form.

**Ingest goes through the real path.** `run_ingest(conn, embedder,
feeds_path, parse=feedparser.parse)` accepts a local path as a feed URL
because feedparser does. The scripts write a temporary `feeds.toml` whose one
`url` is the absolute path of `labelled.xml`. No parse hook, no direct
`INSERT INTO items`: the flow embeds through `EmbeddingClient` against
whatever `LLM_BASE_URL` names.

### `stub_openai.py`

A stdlib `http.server` on `127.0.0.1:0` that speaks the two endpoints
`llm.py` calls:

- `POST /v1/embeddings` — a hashed bag-of-words vector (`EMBED_DIMS` wide,
  default 256), so texts sharing vocabulary land near each other. This is
  what lets `rank_items` and the click classifier behave *sensibly* rather
  than randomly under the stub, which matters for exercising code paths
  (single-class guards, blend weights) — not for the numbers.
- `POST /v1/chat/completions` — inspects the request's `json_schema` and
  the last user message and returns an object that satisfies the schema:
  `Reaction` (`reasoning`, `verdict`, `confidence`) by keyword overlap with
  the persona's interests; the tagging schema by picking vocabulary words
  present in the text; the explain schema with a fixed sentence; any other
  schema by filling required fields with typed placeholders. Honours
  `reasoning_effort` by ignoring it, as a server that does not know the
  field would.

It is started by `run_all.py` (and by each script when run alone with
`--offline`) and its URL is exported as `LLM_BASE_URL`. `CHAT_MODEL` and
`EMBED_MODEL` are set to `stub` so every printed line that names a model
names the stub.

The stub is deliberately dumb. A cleverer stub is a second model to be
wrong about (`measurement-lessons.md` §3: the number is about the artifact).

### `persona_eval.py` — precision, recall, AUC

Runs, per persona, against a fresh database in a temporary directory:

1. `run_ingest` the fixture (real embedder against `LLM_BASE_URL`).
2. Create the persona from `personas.toml` through `personas`/`rank` — the
   same call `feed.persona_create` makes.
3. `simulate.simulate_feedback(conn, chat_fn, persona, items)` over **every**
   item: one LLM call per item, the model reacting *as* the persona. This is
   the LLM-invocation half of the eval, driven by the script.
4. Score the verdicts against `labels.json`:
   - **precision / recall** of `verdict` (useful) vs the label, with the
     confusion matrix printed beside them so a reader can see n;
   - **AUC** of a signed confidence score (`+confidence` for useful,
     `-confidence` for not) vs the label — meaningful only if confidence
     varies, and the script prints the confidence histogram because on
     gemma4:e2b it was measured inert (4 or 5 every time) and a reader must
     be able to see that;
   - items the model skipped as unsure are reported, not silently dropped
     from the denominator.
5. Score the **ranker** on the same labels: `rank.evaluate_user` (the
   cross-validated click-classifier AUC that `attest eval` prints, over the
   simulated clicks), and the AUC of `rank_items`' order over all items vs
   the label — the number `attest eval` cannot see, and the one that moves
   when interests change.

Output is one table per persona plus a header that names the mode, the
chat model, the embedding model, item count and elapsed time. `--json`
writes the same to a file for `run_all --write-results`.

**What the numbers mean, stated in the output.** Precision and recall here
are *agreement with the fixture's labels*. High agreement means the model
reads a persona's interests the way the author of `labels.json` did. It is
evidence about the flow, not a model benchmark; forty items is not a
benchmark of anything.

### `mcp_e2e.py` — the server, over stdio, every tool

Spawns `uv run attest-mcp` five times — once per `ATTEST_TOOLS` surface with
`ATTEST_EXPAND=1`, once unrestricted — using the `mcp` package's
`stdio_client` + `ClientSession`, with `ATTEST_DB` pointing at a database
the script prepared the way `persona_eval.py` does (ingested fixture, two
personas, simulated clicks so the classifier can train, tags so the graph
has concepts, and `examples/workspace` scanned so the ledger has runs).

For each spawn it:

1. `list_tools()` and checks the set against `AGENT_SURFACES` and the
   counts in `CLAUDE.md` — re-measured, not quoted.
2. Calls **every** listed tool with a scripted argument set, in the order a
   person would (create before update before delete; scan before compare;
   `source_preview` before `source_add`). Destructive tools are called with
   `confirm=true` on entities the script created.
3. Checks each result is the envelope `_tool.py` promises: `ok`, `message`,
   and for a failure the same keys as a success (`empty=`). For the four
   `.ask` routers it also sends three natural-language questions each,
   including one that must return `options` rather than a default.
4. Checks the response-size contract on `feed.list` (default limit,
   cap of 13) against the ceiling `test_response_size.py` measures.

Prints a matrix: surface × tool × ok/refused/failed, with the message for
anything not ok. Exit status is nonzero if any tool that should succeed did
not, or any tool that should refuse (unknown persona on a destructive tool,
`sym.solve` on an ambiguous input) did not refuse.

`sym.*` needs no database and is called with the same expressions
`test_symbolic_mcp.py` uses, including the runaway that must be cancelled.

**Why a separate process and not `FastMCP` in-process.** In-process is what
every test already does. The failure this flow exists to catch is the one
in-process cannot see: the entry point, the env the server reads at import,
the stdio framing, the schema FastMCP emits from a Pydantic return, the
stale-process problem. `attest reload` is out of scope for the flow but the
flow is what would have noticed its absence.

### `training/train_mlflow.py` — real runs, under 30 seconds

Trains four arms of one family on a dataset bundled with scikit-learn
(`load_breast_cancer`: 569 rows, 30 features, no download), logging to a
local MLflow file store:

- family `c_sweep`: `LogisticRegression` with `C ∈ {0.01, 0.1, 1, 10}`,
  a fixed seed, a fixed 80/20 stratified split;
- per run: params `C`, `seed`, `dataset`; metrics `accuracy`, `precision`,
  `recall`, `auc` on the held-out split, plus `train_loss` logged per epoch
  for ten steps so the reader's "last line of each metric file" rule is
  exercised on a real multi-line file;
- `mlflow.set_tracking_uri("file:" + <training>/mlruns)`; `run_name` is the
  family name, so the ledger groups them (`_mlflow_runs` derives family from
  `run_name`).

`mlflow-skinny` is the dependency (installs in ~4 s measured; full `mlflow`
pulls a UI stack nothing here uses). It lives in a new `examples`
dependency group beside `optimize`, and `tests/test_tag_prompt.py`'s rule
that `src/` mentions no optional dependency extends to it.

The script also writes `FINDINGS.md` with one claim per arm in the
`<!-- claim: training/<name> metric=auc value=... -->` format the claim
checker reads, plus one deliberately stale value under an explicit heading,
so `attest claims` has a contradiction to find — mirroring
`examples/workspace/speech-distill/FINDINGS.md`.

**The output is committed.** `mlruns/` from one run (a few KB of text: one
`meta.yaml`, four param files, five metric files per run) lives in the repo
so that `ledger.scan` has a real MLflow directory to read *in the test
suite*, without mlflow installed. The reader's docstring, the test module
docstring and the tracker spec each get one line: run against a real
directory on 2026-08-28, by this script. Regenerating changes the run ids
and is a deliberate act; the flow compares its fresh run to the committed
one by family and metric, not by id.

The ledger commands the flow then runs and checks: `attest runs scan --root
examples/flows/training` (one project, four runs, adapter `mlflow`), `attest
runs compare c_sweep --metric auc` (a winner and the seed-replication
caveat), `attest claims examples/flows/training/FINDINGS.md` (exit 1, one
contradiction).

Wall clock: measured before the "under 30 seconds" claim is written into
the README; the script prints its own elapsed time and `run_all` fails if it
exceeds 30 s, so the claim is enforced rather than remembered.

### `run_all.py`

Orchestrates: starts the stub unless `--live`, runs the three flows in
order (training first — it needs no model; persona eval; MCP end to end),
collects each one's JSON, prints one summary, exits nonzero on the first
failure. `--write-results` renders `RESULTS.md` from the JSON: date, mode,
models, the persona tables, the MCP matrix summary, the ledger winner and
claim verdicts, elapsed times. Offline runs never write `RESULTS.md`; the
file records live numbers only.

### CI

A third job, `flows`, in `.github/workflows/ci.yml`: ubuntu-latest,
`uv sync --group examples`, `uv run python examples/flows/run_all.py
--offline`. Separate from `gates` so its ~1–2 minutes and its extra
dependency do not slow the matrix, and so a flow failure reads as "the
demonstration broke", distinct from "a unit test broke".

### Tests

Under `tests/`, model-free, in the existing suite:

- `test_flows_fixture.py`: every entry in `labelled.xml` has a label for
  every persona in `personas.toml`; no item is positive for both; each
  persona has at least ten positives and ten negatives (so
  `evaluate_user`'s fold arithmetic has a class to stratify on).
- `test_flows_stub.py`: the stub returns an object satisfying each schema
  it will be asked for, including `Reaction`; embeddings have `EMBED_DIMS`
  entries and equal texts give equal vectors.
- `test_examples.py` gains: `ledger.scan` over `examples/flows/training`
  yields four `mlflow` runs in family `c_sweep` with final metric values and
  their steps; `compare` names a winner; the claims file has exactly one
  contradiction. This is the test that turns the reader from plausible into
  verified.
- `test_flows_scoring.py`: the precision/recall/AUC arithmetic in
  `persona_eval.py` on a hand-computed confusion matrix, and the "unsure
  items are reported, not dropped" rule.

The flows themselves are not pytest tests: they take minutes live and
seconds offline, and their offline run is the CI job.

## Approaches considered for the LLM-driven metric

**A. Persona reactions against a labelled corpus** (chosen). Uses the real
`simulate.react_to_item` prompt, the real client, and labels the author
wrote for the purpose. Precision and recall are well-defined; AUC needs
the confidence to vary and the script says when it does not. The ranker's
AUC comes for free on the same labels.

**B. Tag-set precision/recall against `tagging_cases.json`.** The cases carry
`should_include_any` and `must_not`, not gold tag sets, so precision would
be against a partial oracle. The rubric in `tagging_eval.py` already
encodes what those cases can honestly measure; reusing it as P/R would
overstate it.

**C. Routing precision/recall.** Deterministic — no model call — so it does
not meet the brief, but it is cheap and `mcp_e2e.py` sends natural
questions to the `.ask` routers anyway; their hit rate is printed in the
matrix rather than dressed up as a classifier metric.

## Success criteria

- `uv run python examples/flows/run_all.py --offline` exits 0 on a clean
  checkout with no Ollama and no network, in CI, in under three minutes.
- `--live` on this machine produces `RESULTS.md` with: precision, recall
  and AUC per persona for the reactions and for the ranker; a 5-row MCP
  matrix with every tool called; the `c_sweep` winner; one contradicted
  claim; the training elapsed time under 30 s.
- Every one of the 46 tools (50 under a surface, counting `.tools`) is
  called at least once over stdio.
- `ledger_adapters/generic.py` no longer says neither tracker reader has
  been run against a real directory; it says which one has, and when.
- No new `# noqa: BLE001`; the complexity ratchet and the seven other gates
  hold; nothing under `src/` imports mlflow.
- Anything a flow finds broken is fixed in its own commit with its own
  failing-then-passing test, and the flow's output names it in `RESULTS.md`.
