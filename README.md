# attestation

attestation makes research provenance auditable and fully local: it reads
the experiment runs you already have on disk (results files, W&B/MLflow/
Sacred/DVC/Hydra directories), checks the numbers in your drafts against
them, keeps a personalised science feed with a knowledge graph, and does
symbolic derivations — all exposed to agents as MCP tools, with nothing
leaving the machine. It is for researchers who keep their own runs and
drafts and want them checked, locally.

## Try it in 60 seconds

No model server, no ingest, no config — the run ledger and claim checker are
pure local computation over files that already exist. This uses the sweep in
`examples/workspace`:

```bash
git clone https://github.com/mgoldey/attestation ~/attestation && cd ~/attestation && uv sync
export ATTEST_DB=/tmp/attest-demo.db

uv run attest runs scan --root examples/workspace
uv run attest runs compare kdsweep
uv run attest claims examples/workspace/speech-distill/FINDINGS.md
```

Measured at **0.94s total** with no LLM backend reachable at all. What it says:

```
kdsweep — ranked by val_loss (lower_is_better), all arms on librispeech-100h
winner: kdsweep_t4
  caveat: the top two arms differ by 0.03 (1.4%) -- too close to call from
          these numbers alone
  caveat: each arm is a single run; no seed replication, so this ranking
          cannot separate configuration from run-to-run variance

7 claim(s): 1 contradicted, 5 supported, 1 unsupported
1 malformed
```

`attest claims` exits 1 on a contradiction (so it can gate a commit), which is
why a `&&` chain stops there.

Every tracker ranks arms. The second caveat — this ranking cannot separate
configuration from noise — is the part that earns its keep, and the
`contradicted` verdict is a number in a document that no longer matches the
artifact it came from.

## What it does

Four things, plus one way to use all of them:

- **The experiment ledger** reads runs from artifacts already on disk — no
  instrumentation, no `log_metric()` calls — and ranks the
  [arms](docs/concepts.md) of a sweep with caveats rather than a silent
  verdict. Five tracker layouts (W&B, MLflow, Sacred, DVC, Hydra) are read
  as conventions of their own. See
  [docs/guides/ledger.md](docs/guides/ledger.md).
- **Verifiable claims and citations** check a number in your prose against
  the run that produced it, one of five verdicts (`supported`,
  `contradicted`, `unsupported`, `ambiguous`, `stale`), and lint a citation
  key against your configured bibliography. See
  [docs/guides/claims-and-citations.md](docs/guides/claims-and-citations.md).
- **The feed and knowledge graph** rank a personalised reading list from
  cosine similarity plus click-trained terms, and derive a concept graph
  from the same tagging pass — no separate content pipeline. See
  [docs/guides/feed.md](docs/guides/feed.md).
- **Symbolic derivations** run in a sandboxed subprocess with a timeout and
  memory cap, never `eval`-ing input. See `sym.*` in
  [docs/guides/agents.md](docs/guides/agents.md).
- **Use it from an agent** — all four are exposed as 46 MCP tools, restricted
  per session into `feed`/`provenance`/`knowledge`/`symbolic` surfaces. See
  [docs/guides/agents.md](docs/guides/agents.md) and the repo's own
  `src/attestation/skills/research-provenance/SKILL.md`.

## Install

Two tiers: Python 3.12+ and [`uv`](https://docs.astral.sh/uv/) alone runs
the ledger and claim checker; add [Ollama](https://ollama.com) for the feed,
tagging, and knowledge graph. Models are optional and pulled for you.

```bash
uv run attest install          # idempotent setup: .env, models, first ingest
uv run attest install --check  # diagnose only, exits 1 on gaps
```

See [docs/guides/install.md](docs/guides/install.md) for prerequisites and
the manual steps `attest install` automates.

## Golden paths

A golden path is a directory under `examples/` with a README in seven fixed
sections, a `run.sh` that runs those README commands verbatim, and its own
inputs on disk — `tests/test_golden_paths.py` runs every one whose
prerequisite is `none` and pins a line of its output, so the docs are what
the suite asserts. `examples/README.md` is the full catalogue, with a runtime
for each. Twelve paths, grouped by prerequisite:

**None — pure local computation:**

- `agents/` — the install doctor, `attest emit`'s configs, one [surface](docs/concepts.md) over stdio
- `citations/` — a BibTeX library, a draft, one citation key that resolves nowhere
- `dvc/` — a real `dvc repro` pipeline, four `foreach` arms ranked
- `flows/` — forty items scored, every MCP tool over stdio, four MLflow arms
- `hydra/` — a real Hydra `--multirun` sweep, four arms ranked
- `mlflow/` — a real MLflow directory, four arms, one contradicted claim
- `model-servers/` — `attest ingest`/`tag` against an in-process stub server
- `sacred/` — a real Sacred `FileStorageObserver` directory, four arms ranked
- `tensorflow/` — a real Keras/CSVLogger run, four learning-rate arms ranked
- `wandb/` — a real offline W&B run directory, four arms ranked
- `workspace/` — the ledger and claim checker, three claims wrong on purpose

**A model server at `LLM_BASE_URL`:** every path above needs no model; this
one does, to score prompts against a running LLM.

- `prompt-evals/` — the tagging prompt's dev score and the transfer gate

See `examples/README.md` for what each demonstrates and how long it takes.
The most thorough is `flows/`:

```bash
uv run --group examples python examples/flows/run_all.py --offline
```

[`examples/flows/README.md`](examples/flows/README.md) explains each flow;
`examples/flows/RESULTS.md` records what the live run measured.

## Documentation

`uv run --group docs mkdocs serve` runs a browsable site over everything
below. The guides, one per question:

- [docs/guides/install.md](docs/guides/install.md) — set up with or without a model server
- [docs/guides/agents.md](docs/guides/agents.md) — MCP tools, surfaces, the skill
- [docs/guides/ledger.md](docs/guides/ledger.md) — reading and ranking runs
- [docs/guides/claims-and-citations.md](docs/guides/claims-and-citations.md) — checking a draft
- [docs/guides/feed.md](docs/guides/feed.md) — how ranking works
- [docs/guides/evals.md](docs/guides/evals.md) — how prompts are measured
- [docs/guides/testing.md](docs/guides/testing.md) — the gates and CI jobs

Plus [docs/concepts.md](docs/concepts.md) for first-ten-minutes vocabulary, the
CLI reference (`docs/reference/cli.md`) and API reference under `docs/`, design
records under `docs/superpowers/specs/`, measurement lessons
(`docs/measurement-lessons.md`), [CONTRIBUTING.md](CONTRIBUTING.md) for the
gates and conventions, and [CHANGELOG.md](CHANGELOG.md) for what's landed.

## Licence

MIT — see [LICENSE](LICENSE).
