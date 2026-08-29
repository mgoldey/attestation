# Golden paths

Each directory here is a use case you can run from a clean clone: a README with
the same seven sections, a `run.sh` that runs the README's own commands, and
its inputs on disk. `tests/test_golden_paths.py` runs every path whose
prerequisite is `none` and pins one line of its output, so the docs are what
the suite asserts.

| path | what it shows | prerequisite | runtime |
|---|---|---|---|
| `workspace/` | the run ledger and claim checker over a two-project workspace with three claims deliberately wrong | `none — pure local computation` | ~1 s |
| `flows/` | forty labelled items scored for two personas, every MCP tool driven over stdio, four MLflow arms read back by the ledger | `none — pure local computation` | ~75 s |
| `model-servers/` | `attest ingest` and `attest tag` run as real subprocesses against an in-process OpenAI-compatible stub, plus a table for pointing `LLM_BASE_URL` at vLLM/llama.cpp/LM Studio/Ollama | `none — pure local computation` | ~2 s |
| `mlflow/` | the run ledger reading a real MLflow directory — four sweep arms ranked, one claim checked against them and contradicted on purpose, the same arms retrained in about two seconds | `none — pure local computation` | ~5 s |
| `citations/` | a four-entry BibTeX library written by bibtexparser, a draft citing three real keys and one that resolves nowhere, and the citation lint that only `cite.check`/`runs.claims_check` over MCP surface — `attest claims` does not | `none — pure local computation` | ~5 s |
| `agents/` | the install doctor's report, the per-surface agent configs `attest emit` generates, and one `AGENT_SURFACES` surface driven over stdio the way hermes-agent actually drives it | `none — pure local computation` | ~10 s |
| `wandb/` | the run ledger reading a real W&B offline directory — four sweep arms ranked, and the finding that offline W&B never writes its own summary/config files until synced | `none — pure local computation` | ~1 s |
| `sacred/` | the run ledger reading a real Sacred `FileStorageObserver` directory — four sweep arms ranked, and `run.json`'s own `result` field read as a metric alongside `metrics.json`'s logged series | `none — pure local computation` | ~1 s |
| `dvc/` | the run ledger reading a real `dvc repro` pipeline — four `foreach` sweep arms ranked, and `dvc.lock`'s own recorded param list read apart from each arm's actual value, with no dependency on the `dvc` package | `none — pure local computation` | ~1 s |
| `tensorflow/` | the run ledger reading a real Keras training run through CSVLogger's CSV and a plain metrics JSON — four learning-rate arms ranked, and the `family_of` fix a bare `lr_<value>` stem required | `none — pure local computation` | ~1 s |
| `hydra/` | the run ledger reading a real Hydra `--multirun` sweep — four sweep arms ranked, and the finding that `hydra.job.chdir=True` is required or all four arms silently overwrite one shared `metrics.json` | `none — pure local computation` | ~1 s |
| `prompt-evals/` | a score for the shipping tagging prompt on 28 dev cases, and the transfer gate that decided whether a GEPA candidate could replace it | `a model server at LLM_BASE_URL` | ~2 min |
