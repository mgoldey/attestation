# LLM client extraction (OpenAI-compatible endpoints) + .env — design

**Date:** 2026-08-05
**Status:** approved (brainstorming dialogue; both sections approved)

## Problem

Ollama integration is scattered across three files with three different
transport styles: `embed.py` (its own `httpx.Client` against native
`/api/embed`), `explain.py` (`ollama_chat()` against native `/api/chat` with
Ollama-specific `format`/`keep_alive`/`num_ctx` fields), and `cli.py`
(`warmup()` with two inline `httpx.post` calls). Configuration is
env-var-scattered and shell-local: today's live incident — `hermes tag`
running at ~60s/item — happened because `CHAT_MODEL=hermes3:8b`
existed only in an interactive shell, so the run fell back to `gemma4:12b`,
which CPU-offloads on this GPU class. Cron and the hermes-agent-spawned MCP
process are equally exposed.

## Decision

1. Extract all LLM transport into one new module, `src/hermes/llm.py`,
   speaking the **OpenAI-compatible API surface** (`/v1/chat/completions`,
   `/v1/embeddings`) so the backend is swappable by base-URL: Ollama's `/v1`
   layer today; vLLM, llama.cpp-server, OpenRouter, or OpenAI by editing one
   env var.
2. Adopt **python-dotenv**, loaded at process entry points only, with a
   committed `.env.sample` and a git-ignored `.env`.

Chosen over: the `openai` SDK (heavyweight for two loopback POSTs; its
retry policy would fight the engine's own retry-then-skip contracts) and
centralizing on native Ollama endpoints (no portability).

## Module: `src/hermes/llm.py`

```
load_env()                                  # dotenv; see Loading below
ChatClient(base_url=None, model=None, api_key=None, timeout=120, client=None)
    .chat_json(messages: list[dict], schema: dict) -> dict
        # POST {base}/chat/completions
        # response_format: {"type": "json_schema",
        #                   "json_schema": {"name": "response",
        #                                   "schema": schema, "strict": true}}
        # returns json.loads(choices[0].message.content)
EmbeddingClient(base_url=None, model=None, api_key=None, timeout=60, client=None)
    .embed(text: str) -> list[float]        # POST {base}/embeddings, data[0].embedding
default_chat_fn(messages, schema) -> dict   # lazy module-level ChatClient singleton
CHAT_MODEL resolution lives here (moves from explain.py)
```

Config resolution, **at construction time** (never at import), precedence
constructor arg > env > default:

| Env var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible base URL |
| `CHAT_MODEL` | `gemma4:12b` (unchanged) | chat model (explain + tagging) |
| `EMBED_MODEL` | `embeddinggemma` | embedding model |
| `LLM_API_KEY` | unset | sent as `Authorization: Bearer` iff set |
| `EMBED_DIMS` | `256` | stored embedding dimensionality (see below) |

No retries inside the clients — reliability policy (retry-then-skip,
cache fallback) stays with the callers that own it. Clients raise `httpx`
errors exactly as the code they replace did.

## Embedding dims (`EMBED_DIMS`)

Dims are storage geometry, not transport, so the resolver lives in `db.py`:
`embed_dims() -> int` reads `EMBED_DIMS` (default 256) at call time.
Two consumers must agree, and both use this one function:

- **`embed.Embedder`**: default `dims=embed_dims()`; `truncate_normalize`
  truncates the model's output (Matryoshka for embeddinggemma) to that
  length. If the model returns fewer dims than configured,
  `truncate_normalize` raises `ValueError` naming both numbers — a
  misconfigured model/dims pair fails loudly at the first embed, not with a
  cryptic sqlite-vec insert error.
- **`db.get_db`**: `VEC_SCHEMA` is built from `embed_dims()`
  (`vec0(embedding float[N])`). On connect, if an `item_vectors` table
  already exists, its declared dims (parsed from
  `sqlite_master.sql`) are compared against the configured value; a
  mismatch raises a clear error: "database has float[256] vectors but
  EMBED_DIMS=512 — re-ingest into a fresh database or set matching
  dims". No silent migration: changing dims invalidates every stored
  vector, so a rebuild (delete DB + `hermes ingest`) is the only honest
  path, and the error message says so.

`rank.py` needs no change — vector lengths flow from `np.frombuffer` and
stay consistent because every producer (ingest embedder, profile embedder)
resolves the same `embed_dims()`.

## Env loading

`load_env()`: `load_dotenv(<repo-root>/.env, override=False)` (repo root =
`Path(__file__).resolve().parents[2]`, correct for editable installs;
harmlessly absent for site-packages/uvx installs) followed by
`load_dotenv(override=False)` (cwd-upward search). Real environment always
wins (`override=False`).

Called from exactly two places: `cli.main()` and `mcp_server.main()`.
Not from `create_app` or any library import — tests and embedders stay
dotenv-free.

## `.env.sample` (committed) / `.env` (git-ignored)

Two sections:

1. **hermes-read variables** — the five table rows above plus a commented
   `RSS_DB`. Ships with `CHAT_MODEL=hermes3:8b` uncommented
   (the recommended 8GB-GPU model), so `cp .env.sample .env` is itself the
   fix for the slow-model footgun. `EMBED_DIMS` ships commented at
   `256` with the warning: "changing this invalidates every stored vector —
   delete the DB and re-ingest".
2. **Ollama SERVER settings** — commented, clearly labeled "read by the
   ollama daemon, NOT by hermes; set in the environment that launches
   `ollama serve`": `OLLAMA_MAX_LOADED_MODELS=2`, `OLLAMA_KEEP_ALIVE=-1`,
   `OLLAMA_CONTEXT_LENGTH=8192`. The latter two replace the per-request
   `keep_alive: -1` and `options.num_ctx: 8192` fields that the native API
   accepted and `/v1` does not.

`.gitignore` gains `.env`.

## Call-site migration

- **`embed.py`**: class renamed `OllamaEmbedder` → `Embedder` (no alias —
  internal API). Keeps `DOC_PROMPT`/`QUERY_PROMPT` templates and
  `truncate_normalize` (768→256 Matryoshka + unit-norm, client-side, works
  identically over `/v1/embeddings`); transport delegates to an
  injected-or-default `EmbeddingClient`. Call sites updated: `cli.py` (ingest,
  bootstrap-persona), `server.py` (`create_app`), `mcp_server.py`
  (`_get_embedder`).
- **`explain.py`**: `ollama_chat` deleted; `explain()`'s default `chat_fn`
  becomes `llm.default_chat_fn`. The injectable `chat_fn(messages, schema)`
  contract is unchanged — `features.py`, `server.py`, `mcp_server.py`, and
  all tests keep working as-is. `CHAT_MODEL` imports (`cli.warmup`,
  `features.run_tagging` provenance column) now come from `llm.py`.
- **`cli.py`**: `warmup()` stays Ollama-native deliberately (it exists to
  exploit Ollama VRAM pinning): derives the native base by stripping a
  trailing `/v1` from `LLM_BASE_URL`, POSTs `/api/chat` +
  `/api/embed` with `keep_alive: -1`; any HTTP failure prints one line
  ("warmup is Ollama-only; skipping for this backend") and exits 0.
  `main()` calls `llm.load_env()` before dispatch; so does
  `mcp_server.main()`.

## Testing

- `llm.py` unit tests via `httpx.MockTransport` (no network): request path
  and body shape (json_schema envelope, model field), Bearer header present
  iff key set, response parsing, env-fallback precedence
  (arg > env > default).
- `load_env` test: a pre-set env var survives a conflicting `.env`
  (`override=False` proven).
- `.env.sample` drift guard: every `HERMES_*` name in the file is asserted
  to be one the code reads (grep the sample, compare against a canonical
  list exported by `llm.py`, plus `EMBED_DIMS`/`RSS_DB` from
  `db.py`).
- Dims tests: `Embedder` respects `EMBED_DIMS` (monkeypatched env →
  vector length follows); `get_db` on a fresh path creates `float[N]` per
  env; `get_db` against an existing DB with mismatched dims raises the
  clear error; `truncate_normalize` raises when the model output is shorter
  than configured dims.
- All existing tests pass untouched — the `chat_fn` and embedder contracts
  are preserved by design. `FakeEmbedder` unaffected.

## Out of scope (YAGNI)

Streaming, client-side retries, per-call model override, multi-backend
failover, async clients, automatic vector migration on dims change (rebuild
via re-ingest is the supported path).
