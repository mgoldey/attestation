<!-- checked by tests/test_golden_paths.py -->

# Example flows

## What you get

Three flows that drive the agent-facing halves of this repo end to end, plus
the fixture they all read. `persona_eval.py` ingests a labelled corpus, has
the model react to every item as each persona, and scores those reactions
against hand labels. `mcp_e2e.py` spawns `attest-mcp` over stdio and calls
every tool on every agent surface — the path an agent actually takes.
`training/train_mlflow.py` trains four real MLflow arms and lets the run
ledger read the directory it wrote. `run_all.py` runs all three and prints
one summary. The fixture is `corpus/`: forty hand-written Atom entries, two
personas, and a label for every item under every persona. It is
hand-written rather than scraped because the labels are the point, and it
deliberately carries no dates — `feed.list` and `rank_items` default to a
fourteen-day window, so dated entries would have aged the demonstration out
of existence with no test going red.

## Prerequisites

`none — pure local computation`

`--offline` points every flow at `stub_openai.py`, a stdlib `http.server`
speaking `/v1/embeddings` and `/v1/chat/completions` with hashed embeddings
and schema-shaped answers — no Ollama, no network. It proves the plumbing,
not the models, and every output line says so. A `--live` run against
Ollama needs a model server at `LLM_BASE_URL`; see *Next*.

## Run it

```bash
uv sync --group examples
uv run --group examples python run_all.py --offline
```

Relative to this directory. Each flow also runs alone, and each takes
`--json PATH` to write its report; `--skip NAME` drops one from the run.

## What it prints

```
mcp_e2e        ok
```

Abridged — the full run prints each flow's own progress (127 MCP calls with
per-call timing, the persona-eval confusion matrices, the MLflow training
arms) and ends with:

```
=== summary
train_mlflow   ok
persona_eval   ok
mcp_e2e        ok
```

## What it demonstrates

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

`training/mlruns` is **committed**, because the ledger reader needs a real
directory to read in CI and in this offline flow. `run_all.py` runs the
trainer into a temporary `--out` so a demonstration run never rewrites it.
Regenerating it is deliberate and explicit:

```bash
uv run --group examples python training/train_mlflow.py
```

That rewrites `mlruns/` and `FINDINGS.md` in place, run ids and all; the
diff is expected to be large and should be reviewed as a fixture change.

## When it goes wrong

- `--offline` and `--live` are mutually exclusive and one is required;
  omitting both raises an argparse error before anything runs.
- A live run with no reachable `LLM_BASE_URL` fails at the first model call
  inside whichever flow runs first (`train_mlflow` needs no model, so it is
  `persona_eval` that fails); the flow's own report records the failure
  rather than the summary silently going green.
- `--write-results` after a failing flow does not rewrite `RESULTS.md`:
  `render_results` raises if any report's mode is not `live`, so an offline
  run cannot be filed as results by accident.

## Next

Live, against Ollama at `LLM_BASE_URL` (ten to fifteen minutes):

```bash
uv run --group examples python run_all.py --live --write-results
```

See `docs/superpowers/specs/2026-08-28-example-flows-design.md` for why each
flow exists and what it is evidence for, and the catalogue at
`examples/README.md` for the other golden paths.
