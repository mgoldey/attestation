# Example flows

## What is here

Three flows that drive the agent-facing halves of this repo end to end, plus
the fixture they all read. `persona_eval.py` ingests a labelled corpus, has
the model react to every item as each persona, and scores those reactions
against hand labels. `mcp_e2e.py` spawns `attest-mcp` over stdio and calls
every tool on every agent surface — the path an agent actually takes, which
until now no test in this repo exercised. `training/train_mlflow.py` trains
four real MLflow arms and lets the run ledger read the directory it wrote.
`run_all.py` runs all three and prints one summary. The fixture is
`corpus/`: forty hand-written Atom entries, two personas, and a label for
every item under every persona. It is hand-written rather than scraped
because the labels are the point, and it deliberately carries no dates —
`feed.list` and `rank_items` default to a fourteen-day window, so dated
entries would have aged the demonstration out of existence with no test
going red.

## How to run

Offline, in about ninety seconds, with nothing installed but the dependency
group — no Ollama, no network:

```bash
uv sync --group examples
uv run --group examples python examples/flows/run_all.py --offline
```

`--offline` points every flow at `stub_openai.py`, a stdlib `http.server`
speaking `/v1/embeddings` and `/v1/chat/completions` with hashed embeddings
and schema-shaped answers. It proves the plumbing, not the models, and every
output line says so. Live, against Ollama at `LLM_BASE_URL` (ten to fifteen
minutes):

```bash
uv run --group examples python examples/flows/run_all.py --live --write-results
```

Each flow also runs alone, and each takes `--json PATH` to write its report.
`--skip NAME` drops one from the run.

## What each flow demonstrates, and what its numbers mean

**`persona_eval.py`** — precision, recall and AUC are *agreement with forty
hand labels*, not a statement about model quality. The corpus is small and
written to be separable; the number that would be alarming is a low one, not
a high one. The confusion matrix, the unsure count and the confidence
histogram are printed beside every score so the reader can see what the
model actually did. The ranker AUCs beneath them are `rank.evaluate_user`
over the same simulated clicks.

**`mcp_e2e.py`** — 127 calls over stdio across the `feed`, `provenance`,
`knowledge` and `symbolic` surfaces and the full server, in the order a
person would call them. Every row is `ok`, `refused` (an expected refusal,
checked as such) or `FAILED`. Refusals are asserted, not tolerated: a tool
that stopped refusing an unknown persona would fail this flow. It also
exercises the response-size contract by calling `feed.list` at limit 4 and at
the cap of 13; the 7000-character ceiling itself is asserted by
`tests/test_response_size.py`.

**`training/train_mlflow.py`** — four `LogisticRegression` arms on
scikit-learn's breast-cancer set, C in `[0.01, 0.1, 1.0, 10.0]`, written to
a real `mlruns/` in under two seconds (1.9 s in the run recorded in
`RESULTS.md`). `attest runs scan` then reads that directory through
`ledger_adapters/generic.py` — the first time that reader has met an MLflow
directory it did not have transcribed from documentation — and
`training/FINDINGS.md` carries one deliberately wrong claim so
`attest claims` demonstrates a contradiction.

## Where the numbers live

`RESULTS.md`, and only from a live run. `run_all.py --write-results` renders
it from the flows' JSON reports and `render_results` raises if any report's
mode is not `live`: offline numbers are facts about the stub server and must
never be filed as results. The file is dated and names the chat and
embedding models it used.

## Regenerating `training/mlruns`

The `mlruns/` directory under `training/` is **committed**, because the
ledger reader needs a real directory to read in CI and in the offline flow.
`run_all.py` runs the trainer into a temporary `--out` so a demonstration
run never rewrites it. Regenerating it is deliberate and explicit:

```bash
uv run --group examples python examples/flows/training/train_mlflow.py
```

That rewrites `mlruns/` and `FINDINGS.md` in place, run ids and all; the
diff is expected to be large and should be reviewed as a fixture change.

## Design

`docs/superpowers/specs/2026-08-28-example-flows-design.md` records why each
flow exists and what it is evidence for.
