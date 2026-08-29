# Model servers

## What you get

Attestation talks to any OpenAI-compatible server: the only contract is
`POST /v1/chat/completions` with `response_format.json_schema` and
`POST /v1/embeddings`. This path runs the real `attest ingest` and
`attest tag` subprocesses against such a server — `examples/flows/stub_openai.py`,
started in-process by `with_server.py` rather than as a background process —
so the commands below exercise `attestation.llm`'s actual HTTP client, not a
mock of it. Point `LLM_BASE_URL` at vLLM, llama.cpp, LM Studio, or Ollama
instead and the same commands run against your own server.

## Prerequisites

`none — pure local computation`

## Run it

```bash
uv run python with_server.py
```

`with_server.py` starts the stub server on a free port, exports
`LLM_BASE_URL`/`CHAT_MODEL`/`EMBED_MODEL` to point at it, writes a `feeds.toml`
over the `examples/flows/` corpus fixture (via `_common.write_feeds_toml`),
then runs `uv run attest ingest --feeds <that file>` and
`uv run attest tag --limit 5` as subprocesses against a fresh temp
`ATTEST_DB`, and shuts the stub down when they finish (or fail).

## What it prints

```
{'added': 40, 'skipped': 0, 'failed_feeds': 0}
```

The second line, `attest tag`'s own stats dict, follows:
`{'tagged': 5, 'failed': 0, 'model': 'stub', 'prompt': 'default'}`.

## What it demonstrates

**`LLM_BASE_URL` semantics** — `attestation.llm.base_url()` reads it at call
time (constructor arg > env var > default), so pointing it at a different
server is the entire integration: no code in `src/` changes.

**A server for every popular runtime**, all speaking the same two endpoints:

| server | `LLM_BASE_URL` | notes |
|---|---|---|
| vLLM | `http://host:8000/v1` | pass `--served-model-name` and set `CHAT_MODEL`/`EMBED_MODEL` to match |
| llama.cpp server | `http://host:8080/v1` | one model per server process; embeddings need `--embedding` |
| LM Studio | `http://localhost:1234/v1` | load the model in the app first; the local server is off by default |
| Ollama | `http://localhost:11434/v1` | the `/v1` suffix is required — Ollama's native API is a different shape |

**`EMBED_DIMS` must match the embedding model.** `attestation.db.embed_dims()`
reads it once at 256 by default; `attestation.embed.Embedder` truncates every
vector to that many dimensions before it is stored. Changing it after vectors
exist doesn't migrate them — `get_db` refuses to open a database whose stored
vectors have a different width than the current `EMBED_DIMS`, rather than
silently comparing vectors of two sizes.

**`reasoning_effort` is sent, and retried away on a 400.**
`ChatClient.chat_json` puts `"reasoning_effort": "none"` in every chat
request (a reasoning model burns time and tokens on a small schema-bound
JSON reply for no benefit — measured on gemma4:e2b at 19.8s/~500 thinking
tokens with it unset vs 10.5s with it off). A server that doesn't recognise
the field ignores it; a server that rejects unknown fields answers 400, and
`chat_json` retries once with the field removed rather than failing the call.

## When it goes wrong

- **`embedding model unreachable`** — `attest ingest` reports
  `failed_feeds` with `embedder_down: True` in its stats and stops after the
  first feed rather than retrying a dead socket per item; the fix is to check
  `LLM_BASE_URL` and that the server is actually listening there.
- **A server that rejects `response_format`** — some servers 400 on the
  whole payload rather than just `reasoning_effort`, in which case
  `chat_json`'s retry (which only drops `reasoning_effort`) still fails, and
  `attest tag` reports the item as `failed`; the fix lives in the server's
  config, not in attestation.
- **Dims mismatch** — swapping to an embedding model with a different native
  width without also setting `EMBED_DIMS` either truncates useful dimensions
  silently (if the new model's output is wider) or raises
  `model returned a N-dim embedding but M dims are configured` (if narrower).
  A database already holding vectors at one `EMBED_DIMS` refuses to open
  under a different one — start a fresh `ATTEST_DB` when you change it.

## Next

See the catalogue at `examples/README.md` for the other golden paths.
