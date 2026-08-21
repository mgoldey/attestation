# LLM Client Extraction (OpenAI-compatible) + .env Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse all scattered Ollama HTTP calls into one OpenAI-compatible transport module (`src/hermes/llm.py`), make every knob env-driven with `.env` support, and expose embedding dims as `EMBED_DIMS` with schema-coupled safety.

**Architecture:** New `llm.py` owns transport (`/v1/chat/completions` with strict `json_schema` structured output, `/v1/embeddings`); `embed.py` keeps prompt templates + Matryoshka truncation and delegates transport; `explain.py`/`features.py`/`server.py`/`mcp_server.py` keep the injectable `chat_fn(messages, schema)` contract with `llm.default_chat_fn` as the new default; `db.py` owns `embed_dims()` and refuses dims-mismatched databases. `python-dotenv` loads `.env` at the two process entry points only. Spec: `docs/superpowers/specs/2026-08-05-llm-client-extraction-design.md`.

**Tech Stack:** Python 3.12, httpx (with `httpx.MockTransport` for tests), python-dotenv, numpy, pytest. Work on the existing branch `feat/llm-clients`.

## Global Constraints

- Config resolution precedence: constructor arg > env var > default, resolved **at construction time**, never at import.
- Env vars and defaults: `LLM_BASE_URL` → `http://localhost:11434/v1`; `CHAT_MODEL` → `gemma4:12b`; `EMBED_MODEL` → `embeddinggemma`; `LLM_API_KEY` → unset (Bearer header iff set); `EMBED_DIMS` → `256`.
- The injectable `chat_fn(messages: list[dict], schema: dict) -> dict` contract is preserved everywhere — existing tests for explain/features/server/mcp must pass **unmodified**.
- No retries inside clients; they raise `httpx` errors. Reliability policy stays with callers.
- `load_dotenv(..., override=False)` always — real environment wins.
- `load_env()` is called ONLY from `cli.main()` and `mcp_server.main()` — not `create_app`, not library imports.
- Structured output envelope: `response_format: {"type": "json_schema", "json_schema": {"name": "response", "schema": <schema>, "strict": true}}`.
- `src/hermes/ingest.py` must not be modified. Ruff line length 100 (`uv run ruff check .` clean); full suite green via `uv run pytest` after every task.
- Every commit message ends with trailer line: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: `llm.py` — transport module + python-dotenv dependency

**Files:**
- Create: `src/hermes/llm.py`
- Create: `tests/test_llm.py`
- Modify: `pyproject.toml` (add `"python-dotenv>=1.0",` to `dependencies`, after the `"tqdm>=4.66",` line)

**Interfaces:**
- Consumes: nothing from this plan (first task).
- Produces (later tasks rely on these exact names):
  - `base_url() -> str`, `chat_model() -> str`, `embed_model() -> str` — env-resolving helpers
  - `ENV_VARS: tuple[str, ...]` — the four var names this module reads (drift-guard input)
  - `load_env() -> None`
  - `ChatClient(base_url=None, model=None, api_key=None, timeout=120, transport=None)` with `.chat_json(messages, schema) -> dict`
  - `EmbeddingClient(base_url=None, model=None, api_key=None, timeout=60, transport=None)` with `.embed(text) -> list[float]`
  - `default_chat_fn(messages, schema) -> dict`
- Deviation from spec, intentional: constructors take `transport` (an `httpx.BaseTransport`) rather than a whole `client` — a pre-built client would bypass the Bearer-header logic and make it untestable. The header logic must live in the constructor.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml` `dependencies`, after `"tqdm>=4.66",` add:

```toml
    "python-dotenv>=1.0",
```

Run: `uv sync` (updates `uv.lock`).

- [ ] **Step 2: Write the failing tests** — create `tests/test_llm.py`:

```python
import json

import httpx
import pytest

from hermes.llm import ChatClient, EmbeddingClient, base_url, chat_model, embed_model, load_env


def chat_transport(captured, content='{"ok": true}'):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": json.loads(request.content),
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.MockTransport(handler)


def embed_transport(captured, dims=768):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": json.loads(request.content),
            }
        )
        return httpx.Response(200, json={"data": [{"embedding": [1.0] * dims}]})

    return httpx.MockTransport(handler)


def test_chat_json_request_shape_and_parse():
    captured = []
    client = ChatClient(base_url="http://test/v1", model="m1", transport=chat_transport(captured))
    out = client.chat_json([{"role": "user", "content": "hi"}], {"type": "object"})
    assert out == {"ok": True}
    req = captured[0]
    assert req["url"] == "http://test/v1/chat/completions"
    assert req["body"]["model"] == "m1"
    assert req["body"]["messages"] == [{"role": "user", "content": "hi"}]
    rf = req["body"]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == {"type": "object"}
    assert rf["json_schema"]["strict"] is True
    assert "authorization" not in req["headers"]  # no key set -> no Bearer header


def test_chat_client_env_fallbacks(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "env-model")
    monkeypatch.setenv("LLM_API_KEY", "sekrit")
    captured = []
    client = ChatClient(base_url="http://test/v1", transport=chat_transport(captured))
    client.chat_json([], {})
    assert captured[0]["body"]["model"] == "env-model"
    assert captured[0]["headers"]["authorization"] == "Bearer sekrit"


def test_chat_client_arg_beats_env(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "env-model")
    captured = []
    ChatClient(
        base_url="http://test/v1", model="arg-model", transport=chat_transport(captured)
    ).chat_json([], {})
    assert captured[0]["body"]["model"] == "arg-model"


def test_chat_json_raises_on_http_error():
    transport = httpx.MockTransport(lambda req: httpx.Response(500))
    client = ChatClient(base_url="http://test/v1", model="m", transport=transport)
    with pytest.raises(httpx.HTTPStatusError):
        client.chat_json([], {})


def test_embedding_client_request_and_parse(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "env-embed")
    captured = []
    client = EmbeddingClient(base_url="http://test/v1", transport=embed_transport(captured))
    vec = client.embed("some text")
    assert captured[0]["url"] == "http://test/v1/embeddings"
    assert captured[0]["body"] == {"model": "env-embed", "input": "some text"}
    assert len(vec) == 768 and vec[0] == 1.0


def test_env_helper_defaults(monkeypatch):
    for var in ("LLM_BASE_URL", "CHAT_MODEL", "EMBED_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert base_url() == "http://localhost:11434/v1"
    assert chat_model() == "gemma4:12b"
    assert embed_model() == "embeddinggemma"


def test_load_env_real_environment_wins(tmp_path, monkeypatch):
    import hermes.llm

    (tmp_path / ".env").write_text("CHAT_MODEL=from-dotenv\nEMBED_MODEL=dotenv-embed\n")
    monkeypatch.setattr(
        hermes.llm, "_REPO_ROOT", tmp_path
    )  # hermetic: ignore any real checkout .env
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAT_MODEL", "from-shell")
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    load_env()
    assert chat_model() == "from-shell"  # override=False: shell wins
    assert embed_model() == "dotenv-embed"  # unset var: .env fills it
```

(`_REPO_ROOT` monkeypatching keeps this test hermetic — without it, a real
`.env` at the checkout root, which the post-plan verification itself
creates, would leak into the test.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hermes.llm'`

- [ ] **Step 4: Implement** — create `src/hermes/llm.py`:

```python
"""OpenAI-compatible LLM transport: chat completions + embeddings.

All config resolves at construction/call time (never at import):
constructor arg > env var > default. No retries here — reliability policy
(retry-then-skip, cache fallback) belongs to the callers.
"""

import json
import os
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_CHAT_MODEL = "gemma4:12b"
DEFAULT_EMBED_MODEL = "embeddinggemma"

_REPO_ROOT = Path(__file__).resolve().parents[2]  # editable-install checkout root

# Canonical list of env vars this module reads (drift guard for .env.sample).
ENV_VARS = (
    "LLM_BASE_URL",
    "CHAT_MODEL",
    "EMBED_MODEL",
    "LLM_API_KEY",
)


def base_url() -> str:
    return os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)


def chat_model() -> str:
    return os.environ.get("CHAT_MODEL", DEFAULT_CHAT_MODEL)


def embed_model() -> str:
    return os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL)


def load_env() -> None:
    """Load .env (repo root first, then cwd-upward search); real env always wins.

    Called only from process entry points (cli.main, mcp_server.main) —
    never from library imports, so tests stay dotenv-free.
    """
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=False)
    load_dotenv(override=False)


def _headers(api_key: str | None) -> dict:
    key = api_key if api_key is not None else os.environ.get("LLM_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


_module_base_url = base_url  # constructors' `base_url` param shadows the function


class ChatClient:
    def __init__(self, base_url=None, model=None, api_key=None, timeout=120, transport=None):
        self.model = model or chat_model()
        self.client = httpx.Client(
            base_url=base_url or _module_base_url(),
            timeout=timeout,
            headers=_headers(api_key),
            transport=transport,
        )

    def chat_json(self, messages: list[dict], schema: dict) -> dict:
        resp = self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": schema, "strict": True},
                },
            },
        )
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"])


class EmbeddingClient:
    def __init__(self, base_url=None, model=None, api_key=None, timeout=60, transport=None):
        self.model = model or embed_model()
        self.client = httpx.Client(
            base_url=base_url or _module_base_url(),
            timeout=timeout,
            headers=_headers(api_key),
            transport=transport,
        )

    def embed(self, text: str) -> list[float]:
        resp = self.client.post("/embeddings", json={"model": self.model, "input": text})
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


_default_chat_client: ChatClient | None = None


def default_chat_fn(messages: list[dict], schema: dict) -> dict:
    """Module-level lazy ChatClient; the default `chat_fn` for explain/tagging."""
    global _default_chat_client
    if _default_chat_client is None:
        _default_chat_client = ChatClient()
    return _default_chat_client.chat_json(messages, schema)
```

Note on `_module_base_url`: the constructor parameter `base_url` shadows
the module-level function inside `__init__`, so the module aliases it once.
Do not rename the public `base_url()` function or the constructor
parameter — both names are spec'd.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -v && uv run ruff check .`
Expected: PASS, ruff clean

- [ ] **Step 6: Commit**

```bash
git add src/hermes/llm.py tests/test_llm.py pyproject.toml uv.lock
git commit -m "feat: OpenAI-compatible LLM transport module (llm.py) + python-dotenv"
```

---

### Task 2: `EMBED_DIMS` — dims resolver, schema coupling, mismatch guard

**Files:**
- Modify: `src/hermes/db.py` (add `embed_dims()`; make `VEC_SCHEMA` dynamic; guard in `get_db`)
- Modify: `src/hermes/embed.py` (only `truncate_normalize` in this task)
- Test: `tests/test_db.py`, `tests/test_embed.py`

**Interfaces:**
- Consumes: nothing from other tasks (independent of Task 1).
- Produces: `db.embed_dims() -> int` (env `EMBED_DIMS`, default 256, read at call time). `truncate_normalize(vec, dims: int | None = None)` — `None` resolves via `embed_dims()`; raises `ValueError` when `len(vec) < dims`. Task 3's `Embedder` relies on both.

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_db.py`:

```python
def test_embed_dims_env(monkeypatch):
    from hermes.db import embed_dims

    monkeypatch.delenv("EMBED_DIMS", raising=False)
    assert embed_dims() == 256
    monkeypatch.setenv("EMBED_DIMS", "512")
    assert embed_dims() == 512


def test_vec_schema_follows_dims_env(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBED_DIMS", "128")
    conn = get_db(tmp_path / "small.db")
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'item_vectors'").fetchone()[
        "sql"
    ]
    assert "float[128]" in sql


def test_dims_mismatch_refuses_to_open(tmp_path, monkeypatch):
    import pytest

    monkeypatch.delenv("EMBED_DIMS", raising=False)
    get_db(tmp_path / "t.db").close()  # created at default 256
    monkeypatch.setenv("EMBED_DIMS", "512")
    with pytest.raises(RuntimeError, match=r"float\[256\].*EMBED_DIMS=512"):
        get_db(tmp_path / "t.db")
```

Append to `tests/test_embed.py`:

```python
def test_truncate_normalize_raises_when_model_too_small():
    import pytest

    with pytest.raises(ValueError, match="128.*256"):
        truncate_normalize(np.ones(128, dtype=np.float32), dims=256)


def test_truncate_normalize_default_dims_follows_env(monkeypatch):
    monkeypatch.setenv("EMBED_DIMS", "64")
    out = truncate_normalize(np.ones(768, dtype=np.float32))
    assert out.shape == (64,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db.py tests/test_embed.py -v`
Expected: new tests FAIL (`embed_dims` not defined; `float[128]` absent — schema is hardcoded 256; no ValueError raised; env ignored)

- [ ] **Step 3: Implement.**

In `src/hermes/db.py`: add `import re` to the imports. Replace the module-level `VEC_SCHEMA = "CREATE VIRTUAL TABLE ..."` constant with:

```python
def embed_dims() -> int:
    """Stored embedding dimensionality (EMBED_DIMS, default 256).

    Read at call time so tests and .env loading are respected. Changing it
    invalidates every stored vector — get_db() refuses mismatched databases.
    """
    return int(os.environ.get("EMBED_DIMS", "256"))


def _vec_schema(dims: int) -> str:
    return f"CREATE VIRTUAL TABLE IF NOT EXISTS item_vectors USING vec0(embedding float[{dims}])"
```

In `get_db`, replace the line `conn.execute(VEC_SCHEMA)` with:

```python
dims = embed_dims()
existing = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'item_vectors'").fetchone()
if existing:
    m = re.search(r"float\[(\d+)\]", existing["sql"])
    stored = int(m.group(1)) if m else None
    if stored is not None and stored != dims:
        raise RuntimeError(
            f"database has float[{stored}] vectors but EMBED_DIMS={dims}"
            " — re-ingest into a fresh database or set matching dims"
        )
else:
    conn.execute(_vec_schema(dims))
```

In `src/hermes/embed.py`, replace `truncate_normalize`:

```python
def truncate_normalize(vec: np.ndarray, dims: int | None = None) -> np.ndarray:
    from hermes.db import embed_dims

    dims = dims if dims is not None else embed_dims()
    if len(vec) < dims:
        raise ValueError(
            f"model returned a {len(vec)}-dim embedding but {dims} dims are configured"
            " (EMBED_DIMS) — use a larger model or smaller dims"
        )
    v = vec[:dims].astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v
```

(The local `from hermes.db import embed_dims` avoids a module-level import
cycle risk and keeps `embed.py` importable standalone; `db.py` never
imports `embed.py`.)

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest && uv run ruff check .`
Expected: ALL PASS (existing tests use default 256 throughout), ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/hermes/db.py src/hermes/embed.py tests/test_db.py tests/test_embed.py
git commit -m "feat: EMBED_DIMS env — schema-coupled dims with mismatch guard"
```

---

### Task 3: `Embedder` over `EmbeddingClient` + call-site rename

**Files:**
- Modify: `src/hermes/embed.py` (class rewrite)
- Modify: `src/hermes/cli.py` (two `OllamaEmbedder()` sites)
- Modify: `src/hermes/server.py` (`create_app` default)
- Modify: `src/hermes/mcp_server.py` (`_get_embedder`)
- Test: `tests/test_embed.py` (transport tests rewritten for /v1 shape)

**Interfaces:**
- Consumes: Task 1's `EmbeddingClient(base_url=None, model=None, api_key=None, timeout=60, transport=None)` with `.embed(text) -> list[float]`; Task 2's `embed_dims()` and `truncate_normalize(vec, dims)`.
- Produces: `Embedder(client: EmbeddingClient | None = None, dims: int | None = None)` with unchanged `embed_document(title, text) -> np.ndarray` / `embed_query(text) -> np.ndarray`. The name `OllamaEmbedder` ceases to exist (no alias).

- [ ] **Step 1: Rewrite the transport tests.** In `tests/test_embed.py`, replace the import line and `make_embedder` helper (keep all `truncate_normalize` tests as-is):

```python
from hermes.embed import Embedder, truncate_normalize
from hermes.llm import EmbeddingClient


def make_embedder(captured):
    """Embedder wired to a mock /v1 transport returning a fixed 768-dim vector."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"embedding": [1.0] * 768}]})

    client = EmbeddingClient(
        base_url="http://test/v1", model="embeddinggemma", transport=httpx.MockTransport(handler)
    )
    return Embedder(client=client)
```

The three prompt-format tests (`test_embed_document_prompt_format`,
`test_embed_document_missing_title_uses_none`, `test_embed_query_prompt_format`)
keep their exact assertions — the `/v1/embeddings` body also carries
`"input"` and `"model"` keys, so `captured[0]["input"]`/`["model"]` still hold.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embed.py -v`
Expected: FAIL with `ImportError: cannot import name 'Embedder'`

- [ ] **Step 3: Implement.** Replace the `OllamaEmbedder` class in `src/hermes/embed.py` (docstring line 1 becomes `"""Embedding client wrapper: doc/query prompts + Matryoshka truncation."""`; drop the direct `httpx` import if now unused):

```python
from hermes.llm import EmbeddingClient


class Embedder:
    """Doc/query prompt formatting + truncation over an OpenAI-style embeddings client."""

    def __init__(self, client: EmbeddingClient | None = None, dims: int | None = None):
        from hermes.db import embed_dims

        self.client = client or EmbeddingClient()
        self.dims = dims if dims is not None else embed_dims()

    def _embed(self, prompt: str) -> np.ndarray:
        vec = np.asarray(self.client.embed(prompt), dtype=np.float32)
        return truncate_normalize(vec, self.dims)

    def embed_document(self, title: str, text: str) -> np.ndarray:
        return self._embed(DOC_PROMPT.format(title=title or "none", text=text))

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(QUERY_PROMPT.format(text=text))
```

Update call sites (mechanical — these are ALL of them; verify with
`grep -rn OllamaEmbedder src/ tests/` afterwards, expect zero hits):

- `src/hermes/cli.py`: both `from hermes.embed import OllamaEmbedder` + `OllamaEmbedder()` pairs (ingest and bootstrap-persona dispatch blocks) → `from hermes.embed import Embedder` + `Embedder()`.
- `src/hermes/server.py` `create_app`: `from hermes.embed import OllamaEmbedder` / `embedder = OllamaEmbedder()` → `from hermes.embed import Embedder` / `embedder = Embedder()`.
- `src/hermes/mcp_server.py` `_get_embedder`: `from hermes.embed import OllamaEmbedder` / `_embedder = OllamaEmbedder()` → `from hermes.embed import Embedder` / `_embedder = Embedder()`.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest && uv run ruff check . && grep -rn OllamaEmbedder src/ tests/ | wc -l`
Expected: ALL PASS, ruff clean, grep count `0`

- [ ] **Step 5: Commit**

```bash
git add src/hermes/embed.py src/hermes/cli.py src/hermes/server.py src/hermes/mcp_server.py tests/test_embed.py
git commit -m "refactor: Embedder delegates transport to EmbeddingClient (/v1/embeddings)"
```

---

### Task 4: chat migration — `explain.py` drops `ollama_chat`

**Files:**
- Modify: `src/hermes/explain.py` (delete `ollama_chat` + `CHAT_MODEL`; new default)
- Modify: `src/hermes/features.py` (`run_tagging` lazy imports)
- Modify: `src/hermes/server.py`, `src/hermes/mcp_server.py` (import swaps)
- Modify: `src/hermes/cli.py` (`warmup`'s `CHAT_MODEL` import — minimal touch here; full warmup rewrite is Task 5)
- Test: no new tests — the existing suite IS the test (chat_fn contract preservation)

**Interfaces:**
- Consumes: Task 1's `default_chat_fn(messages, schema) -> dict` and `chat_model() -> str`.
- Produces: `explain.explain(conn, user_id, item_id, chat_fn=default_chat_fn)`; the names `ollama_chat` and `explain.CHAT_MODEL` cease to exist.

- [ ] **Step 1: Implement the migration** (no new tests first — the regression gate is the untouched existing suite):

In `src/hermes/explain.py`:
- Delete the `CHAT_MODEL = os.environ.get(...)` line and the whole `ollama_chat` function; drop now-unused `import httpx`, `import json`, `import os` if nothing else uses them (check: `json` is not used elsewhere in the file; `os` is not; `httpx` is not).
- Add `from hermes.llm import default_chat_fn` to the imports.
- Change the signature: `def explain(conn, user_id: int, item_id: int, chat_fn=default_chat_fn) -> str | None:`.
- Module docstring line 3-4 ("swappable Ollama backend (CHAT_MODEL, default gemma4:12b)") becomes: `the chat model is a swappable OpenAI-compatible backend (see src/hermes/llm.py).`

In `src/hermes/features.py` `run_tagging`, replace the import block at the top of the function:

```python
    if chat_fn is None:
        from hermes.llm import default_chat_fn

        chat_fn = default_chat_fn
    from hermes.llm import chat_model
```

and the call `tag_one_item(conn, item, chat_fn, vocab, CHAT_MODEL)` → `tag_one_item(conn, item, chat_fn, vocab, chat_model())`.

In `src/hermes/server.py`: `from hermes.explain import explain, ollama_chat` → two lines: `from hermes.explain import explain` and `from hermes.llm import default_chat_fn`; in `create_app`, `chat_fn = chat_fn or ollama_chat` → `chat_fn = chat_fn or default_chat_fn`.

In `src/hermes/mcp_server.py`: `from hermes.explain import ollama_chat` → `from hermes.llm import default_chat_fn`; the call `explain_item_fn(conn, row["id"], item_id, chat_fn=ollama_chat)` → `explain_item_fn(conn, row["id"], item_id, chat_fn=default_chat_fn)`.

In `src/hermes/cli.py` `warmup()`: `from hermes.explain import CHAT_MODEL` → `from hermes.llm import chat_model`, and both `CHAT_MODEL` uses become `chat_model()` (the f-string print too). Leave the rest of `warmup` alone — Task 5 rewrites it.

- [ ] **Step 2: Verify nothing references the dead names**

Run: `grep -rn "ollama_chat\|explain import CHAT_MODEL\|from hermes.explain import CHAT_MODEL" src/ tests/ | wc -l`
Expected: `0`

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest && uv run ruff check .`
Expected: ALL PASS unmodified — this is the proof the `chat_fn` contract survived. If any test fails, the migration broke a contract; fix the migration, never the test.

- [ ] **Step 4: Commit**

```bash
git add src/hermes/explain.py src/hermes/features.py src/hermes/server.py src/hermes/mcp_server.py src/hermes/cli.py
git commit -m "refactor: chat calls route through llm.default_chat_fn; ollama_chat removed"
```

---

### Task 5: warmup (Ollama-native, graceful skip) + entry-point `load_env()`

**Files:**
- Modify: `src/hermes/cli.py` (`warmup` rewrite; `main` calls `load_env`)
- Modify: `src/hermes/mcp_server.py` (`main` calls `load_env`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 1's `load_env()`, `base_url()`, `chat_model()`, `embed_model()`.
- Produces: `warmup()` that exits 0 with a one-line skip message on any HTTP failure; `cli.main()` and `mcp_server.main()` both begin with `load_env()`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_cli.py`):

```python
def test_warmup_skips_gracefully_on_non_ollama_backend(capsys, monkeypatch):
    import httpx as _httpx

    import hermes.cli

    def boom(*args, **kwargs):
        raise _httpx.ConnectError("no ollama here")

    monkeypatch.setattr(hermes.cli.httpx, "post", boom)
    rc = main(["warmup"])
    assert rc == 0
    assert "skipping" in capsys.readouterr().out.lower()


def test_main_loads_dotenv(tmp_path, monkeypatch, capsys):
    """main() must call llm.load_env() before dispatch: a .env in cwd is visible."""
    import hermes.llm

    (tmp_path / ".env").write_text("CHAT_MODEL=dotenv-model\n")
    monkeypatch.setattr(hermes.llm, "_REPO_ROOT", tmp_path)  # hermetic vs real checkout .env
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    captured = {}

    import hermes.cli

    def fake_warmup():
        from hermes.llm import chat_model

        captured["model"] = chat_model()

    monkeypatch.setattr(hermes.cli, "warmup", fake_warmup)
    assert main(["warmup"]) == 0
    assert captured["model"] == "dotenv-model"
```

(Note: `main` dispatches `warmup` via the module attribute for this to be
monkeypatchable — Step 3 makes the call site `hermes.cli`-module-attribute
style, mirroring the `hermes.features.run_tagging` pattern already used by
the `tag` subcommand.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: the two new tests FAIL (warmup raises `ConnectError` uncaught; `.env` not loaded)

- [ ] **Step 3: Implement.**

In `src/hermes/cli.py`, replace `warmup()`:

```python
def warmup() -> None:
    """Pin chat + embed models in VRAM. Ollama-specific by design: derives the
    native /api base from the /v1 base URL; other backends don't need pinning."""
    from hermes.llm import base_url, chat_model, embed_model

    native = base_url().rstrip("/").removesuffix("/v1")
    try:
        httpx.post(
            f"{native}/api/chat",
            json={
                "model": chat_model(),
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "keep_alive": -1,
                "options": {"num_ctx": 8192},
            },
            timeout=300,
        ).raise_for_status()
        httpx.post(
            f"{native}/api/embed",
            json={"model": embed_model(), "input": "warmup", "keep_alive": -1},
            timeout=300,
        ).raise_for_status()
    except httpx.HTTPError:
        print("warmup is Ollama-only; skipping for this backend")
        return
    print(f"models loaded and pinned (chat={chat_model()}, keep_alive=-1)")
```

In `main()`: first two lines of the function body become:

```python
    from hermes.llm import load_env

    load_env()
    args = build_parser().parse_args(argv)
```

and the warmup dispatch becomes module-attribute style:

```python
    if args.command == "warmup":
        import hermes.cli

        hermes.cli.warmup()
        return 0
```

In `src/hermes/mcp_server.py` `main()`:

```python
def main() -> None:
    from hermes.llm import load_env

    load_env()
    mcp.run(transport="stdio")
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest && uv run ruff check .`
Expected: ALL PASS, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/hermes/cli.py src/hermes/mcp_server.py tests/test_cli.py
git commit -m "feat: dotenv at entry points; warmup degrades gracefully off-Ollama"
```

---

### Task 6: `.env.sample`, `.gitignore`, drift guard, README

**Files:**
- Create: `.env.sample`
- Modify: `.gitignore` (append `.env`)
- Modify: `README.md` (new Configuration subsection; MCP note)
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: Task 1's `ENV_VARS`.
- Produces: the committed `.env.sample`; no code interfaces.

- [ ] **Step 1: Write the failing drift-guard test** (append to `tests/test_llm.py`):

```python
def test_env_sample_documents_exactly_the_vars_the_code_reads():
    import re
    from pathlib import Path

    from hermes import llm

    sample = (Path(__file__).resolve().parents[1] / ".env.sample").read_text()
    documented = {
        name
        for name in re.findall(r"^#?([A-Z_]+)=", sample, flags=re.M)
        if not name.startswith("OLLAMA_")  # daemon section: documented, not read by hermes
    }
    known = set(llm.ENV_VARS) | {"EMBED_DIMS", "RSS_DB"}
    assert documented == known
```

(The regex requires the name to start at column 0 or right after `#`, so the
indented `#OLLAMA_...` comment explanations don't match; the explicit
`OLLAMA_` filter covers the section's own entries.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm.py::test_env_sample_documents_exactly_the_vars_the_code_reads -v`
Expected: FAIL with `FileNotFoundError` (.env.sample doesn't exist)

- [ ] **Step 3: Create `.env.sample`** (repo root, exact content):

```bash
# hermes-rss configuration — copy to .env and edit:  cp .env.sample .env
# Loaded at startup by the `hermes` CLI and the MCP server (hermes-mcp).
# Real environment variables always win over values in this file.

# OpenAI-compatible endpoint serving chat + embeddings (Ollama's /v1 by default).
# Swap backends by pointing this elsewhere (vLLM, llama.cpp server, OpenRouter...).
LLM_BASE_URL=http://localhost:11434/v1

# API key, sent as a Bearer token when set. Ollama ignores auth; needed for
# OpenRouter / authenticated vLLM / OpenAI.
#LLM_API_KEY=

# Chat model for explanations + tagging. hermes3:8b is strongly recommended on
# 8 GB-class GPUs (the gemma4:12b default partially CPU-offloads: ~60-90s/call vs ~6s).
CHAT_MODEL=hermes3:8b

# Embedding model.
#EMBED_MODEL=embeddinggemma

# Stored embedding dimensionality (Matryoshka truncation, client-side).
# WARNING: changing this invalidates every stored vector — delete the database
# and re-ingest. get_db() refuses to open a mismatched database.
#EMBED_DIMS=256

# Database path override (default resolution order is documented in the README).
#RSS_DB=/home/you/.hermes/skills/science-recommendations/data/hermes.db

# --- Ollama SERVER settings (read by the ollama daemon, NOT by hermes) --------
# Set these in the environment that launches `ollama serve` (e.g. its systemd
# unit), not here. /v1 requests cannot pin models or set context length, so
# these replace the per-request keep_alive/num_ctx knobs of the native API.
#OLLAMA_MAX_LOADED_MODELS=2   # keep chat + embed models co-resident
#OLLAMA_KEEP_ALIVE=-1         # pin models in VRAM (no 10-20s cold loads)
#OLLAMA_CONTEXT_LENGTH=8192   # context window for /v1 requests
```

Append to `.gitignore`:

```
.env
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm.py -v`
Expected: PASS (the regex only counts `HERMES_*` names, so the `OLLAMA_*` comment lines don't trip it)

- [ ] **Step 5: README edits.** In `README.md`:

After the "Install and first run" code block's closing paragraph ("Edit `feeds.toml` ..."), insert:

```markdown
### Configuration (.env)

```bash
cp .env.sample .env    # then edit — hermes3:8b is pre-selected as the chat model
```

The `hermes` CLI and the MCP server load `.env` at startup (real environment
variables always win), so your shell, cron, and hermes-agent-spawned
processes all see the same configuration. All LLM traffic speaks the
OpenAI-compatible API (`LLM_BASE_URL`, default Ollama's
`http://localhost:11434/v1`) — point it at vLLM, llama.cpp server, or
OpenRouter (set `LLM_API_KEY`) to swap backends. See `.env.sample`
for every variable, including the Ollama daemon settings
(`OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_CONTEXT_LENGTH=8192`) that replace the
per-request pinning the native API used to provide.
```

In the MCP section, the line "To pin the explanation model for the MCP process, add `--env CHAT_MODEL=hermes3:8b` to the `hermes mcp add` command." becomes: "The MCP server reads `.env` from the checkout at startup, so `CHAT_MODEL` set there applies — no `--env` flag needed (though `--env` still works and wins over `.env`)."

- [ ] **Step 6: Full suite + commit**

Run: `uv run pytest && uv run ruff check .`
Expected: ALL PASS, ruff clean

```bash
git add .env.sample .gitignore README.md tests/test_llm.py
git commit -m "feat: .env.sample + gitignore .env; README configuration docs"
```

---

### Task 7: hard-rename sweep — `RSS_DB` + legacy `HERMES_*` references

**Files:**
- Modify: `src/hermes/db.py` (`resolve_db_path` env read + docstring)
- Modify: `src/hermes/cli.py` (`--db` help text)
- Modify: `tests/test_mcp_server.py` (autouse `_patch_env_db` fixture)
- Modify: `skills/science-recommendations/scripts/setup.sh`, `skills/science-recommendations/SKILL.md`, `README.md`, `DEMO.md` (doc/script references)
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: the env var `RSS_DB` (old `HERMES_RSS_DB` no longer read — hard rename per user decision); zero remaining `HERMES_RSS_DB`/`HERMES_CHAT_MODEL`/`HERMES_EMBED_MODEL`/`HERMES_LLM_*`/`HERMES_EMBED_DIMS` references outside `docs/superpowers/` (historical specs/plans stay as written).

- [ ] **Step 1: Write the failing test** (append to `tests/test_db.py`; add `resolve_db_path` to the existing `from hermes.db import ...` line if absent):

```python
def test_resolve_db_path_reads_unprefixed_env(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_RSS_DB", raising=False)
    monkeypatch.setenv("RSS_DB", str(tmp_path / "x.db"))
    assert resolve_db_path(None) == tmp_path / "x.db"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py -v`
Expected: the new test FAILS (code still reads `HERMES_RSS_DB`, so it falls through to the default path)

- [ ] **Step 3: Implement.**

- `src/hermes/db.py` `resolve_db_path`: `os.environ.get("HERMES_RSS_DB")` → `os.environ.get("RSS_DB")`; update the docstring's resolution-order text (`HERMES_RSS_DB` → `RSS_DB`).
- `src/hermes/cli.py` `add_db` help text: `HERMES_RSS_DB env var` → `RSS_DB env var`.
- `tests/test_mcp_server.py` `_patch_env_db` fixture: `monkeypatch.setenv("HERMES_RSS_DB", ...)` → `monkeypatch.setenv("RSS_DB", ...)` (docstring too).
- `skills/science-recommendations/scripts/setup.sh`: every `HERMES_RSS_DB=` env assignment → `RSS_DB=`; `CHAT_MODEL="${HERMES_CHAT_MODEL:-hermes3:8b}"` → `CHAT_MODEL="${CHAT_MODEL:-hermes3:8b}"`; the `HERMES_CHAT_MODEL="${CHAT_MODEL}"` exports on serve lines → `CHAT_MODEL="${CHAT_MODEL}"`.
- `skills/science-recommendations/SKILL.md`, `README.md`, `DEMO.md`: replace every `HERMES_RSS_DB` → `RSS_DB`, `HERMES_CHAT_MODEL` → `CHAT_MODEL` (mechanical find-replace; read each hit's line to confirm it's an env-var reference, not prose about the hermes binary).

- [ ] **Step 4: Verify the sweep is complete**

Run: `grep -rn "HERMES_RSS_DB\|HERMES_CHAT_MODEL\|HERMES_EMBED_MODEL\|HERMES_EMBED_DIMS\|HERMES_LLM_" --include="*.py" --include="*.sh" --include="*.md" --include="*.toml" . | grep -v "docs/superpowers/" | grep -v ".superpowers/" | wc -l`
Expected: `0`

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest && uv run ruff check .`
Expected: ALL PASS, ruff clean

- [ ] **Step 6: Commit**

```bash
git add src/hermes/db.py src/hermes/cli.py tests/test_db.py tests/test_mcp_server.py \
        skills/science-recommendations/scripts/setup.sh skills/science-recommendations/SKILL.md \
        README.md DEMO.md
git commit -m "refactor!: unprefixed env vars — RSS_DB, CHAT_MODEL sweep (hard rename)"
```

---

### Task 8: dev tooling — `ty`, `radon`, ruff rule enforcement (user-requested)

**Files:**
- Modify: `pyproject.toml` (dev deps + `[tool.ruff.lint]`)
- Modify: any source/test file with violations the newly-enforced rules surface
- Modify: `README.md` (Tests section)
- Test: the tool runs themselves are the gate (no new pytest tests)

**Interfaces:**
- Consumes: nothing; runs after Task 7 so it lints the final code.
- Produces: enforced tooling — `uv run ruff check .` (with real rule set), `uv run ty check`, and a radon complexity report with no function graded worse than B.

- [ ] **Step 1: Add the dev dependencies**

In `pyproject.toml`, change the dev group to:

```toml
[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6", "ty", "radon>=6"]
```

Run: `uv sync`

- [ ] **Step 2: Enforce a real ruff rule set.** Add to `pyproject.toml` under the existing `[tool.ruff]` section:

```toml
[tool.ruff.lint]
select = ["E", "F", "W", "I", "BLE"]
```

(This activates E501 line-length at the configured 100, import sorting, and
makes the existing `BLE001` per-file-ignores meaningful again. Keep the
existing `[tool.ruff.lint.per-file-ignores]` block — it exists precisely
for this.)

- [ ] **Step 3: Fix the fallout.**

Run: `uv run ruff check . --statistics` to see what the new rules surface.
Known debts to fix: the 109-char module docstring at `src/hermes/rank.py:1`
(rewrap to ≤100), long lines in `tests/test_llm.py` (~5) and any others
E501 reports; `I001` import-order fixes via `uv run ruff check . --fix`.
Fix by hand where `--fix` can't. Do NOT change behavior — formatting-only
edits; the full pytest suite is the behavior gate.

- [ ] **Step 4: Type-check with ty.**

Run: `uv run ty check`
Fix reported type errors with minimal annotations or corrections. If ty
(pre-1.0) reports a false positive that cannot be annotated away, suppress
narrowly with a `# ty: ignore[rule]` comment and note it in the report —
do not blanket-disable rules project-wide without recording why.

- [ ] **Step 5: Complexity gate with radon.**

Run: `uv run radon cc -s -n C src/hermes tests`
Expected: empty output (no function rated C or worse). If anything rates C,
report it in your task report rather than refactoring it — complexity
refactors are out of scope for a tooling task; the finding goes to the
ledger.

- [ ] **Step 6: Document.** In `README.md`'s Tests section, replace the two-line block with:

```
    uv run pytest
    uv run ruff check .                     # lint (E, F, W, I, BLE; line length 100)
    uv run ty check                         # type check
    uv run radon cc -s -n C src/hermes      # complexity report (empty = nothing worse than B)
```

- [ ] **Step 7: Full suite + commit**

Run: `uv run pytest && uv run ruff check . && uv run ty check`
Expected: ALL PASS

```bash
git add pyproject.toml uv.lock README.md <files touched by lint/type fixes>
git commit -m "chore: dev tooling — ty type checks, radon complexity, enforced ruff rules"
```

---

## Post-plan verification (manual, once all tasks land)

1. `cp .env.sample .env`, then `uv run hermes warmup` against live Ollama — expect the pinned-models message with `chat=hermes3:8b`.
2. `uv run hermes tag --limit 3` — expect ~2-6s/item via `/v1` structured output (this also live-verifies Ollama's `/v1` `response_format: json_schema` support; if Ollama rejects it, that's a blocking find — report it, don't work around silently).
3. One hermes-agent chat turn calling `list_feed` + `explain_item` — verifies the MCP process picked up `.env` (fast explain = right model).
4. Confirm `OLLAMA_KEEP_ALIVE=-1` / `OLLAMA_CONTEXT_LENGTH=8192` are set in the ollama daemon's environment (systemd unit or shell that runs `ollama serve`) — without them, models unload after 5 idle minutes and context falls back to the model default.
5. Hard-rename aftermath (user actions): re-copy the skill (`cp -r skills/science-recommendations ~/.hermes/skills/`) so the installed `setup.sh` uses the new names; update any shell exports/cron lines still using `HERMES_CHAT_MODEL`/`HERMES_RSS_DB` — the old spellings are silently ignored now.
