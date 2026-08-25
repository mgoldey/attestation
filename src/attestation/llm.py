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
# e2b over 12b: 2.2 GB resident and fully GPU-resident on 8 GB-class cards,
# where 12b partially CPU-offloads (~60-90s per call vs ~2.2s).
DEFAULT_CHAT_MODEL = "gemma4:e2b-it-q4_K_M"
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


class BackendUnreachable(RuntimeError):
    """The model backend refused or never answered the socket.

    Raised by callers that must stop a whole run on the condition (tagging)
    so the run can catch it narrowly; `backend_unreachable` classifies the
    raw transport error for callers that keep the original exception.
    """


def backend_unreachable(exc: BaseException) -> bool:
    """Whether this failure means the model backend is unreachable.

    Matched on the transport exception rather than on message text: httpx
    raises ConnectError/ConnectTimeout for a refused or unanswered socket,
    which is exactly the "Ollama is not running" case. Shared by ingest (the
    embedder) and tagging (the chat model): both stop at the first such
    failure and say so once, instead of failing every remaining item against
    a dead socket.
    """
    import httpx

    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))


def chat_model() -> str:
    return os.environ.get("CHAT_MODEL", DEFAULT_CHAT_MODEL)


def embed_model() -> str:
    return os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL)


def load_env() -> None:
    """Load .env (repo root first, then cwd-upward search); real env always wins.

    Called only from process entry points (cli.main, mcp_server.main) —
    never from library imports, so tests stay dotenv-free.

    Any OTHER entry point must call this itself. A standalone script that
    imports attestation and skips it gets DEFAULT_CHAT_MODEL rather than the model
    in .env, silently and with no error — a one-off re-tagging script did
    exactly that on 2026-08-11 and ran against the wrong model until the
    banner it printed gave it away.
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
        # reasoning_effort="none": every call here asks for a small, schema-bound
        # JSON object, so chain-of-thought buys nothing and costs a lot. Measured
        # on gemma4:e2b (2026-08-11): 19.8s and ~500 thinking tokens per tagging
        # call by default vs 10.5s with thinking off, for equal-or-better tags.
        # Servers that do not know the field ignore it; those that reject it are
        # retried below without it.
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            },
            "reasoning_effort": "none",
        }
        resp = self.client.post("/chat/completions", json=payload)
        if resp.status_code == 400:
            payload.pop("reasoning_effort")
            resp = self.client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        return _first_json_object(resp.json()["choices"][0]["message"]["content"])


def _first_json_object(text: str) -> dict:
    """The first complete JSON object in a reply, ignoring anything around it.

    Schema-constrained decoding is a request, not a guarantee. gemma4:e2b
    emitted a valid object followed by a second one, and `json.loads` raised
    "Extra data: line 3 column 2" straight out of chat_json -- an explanation
    request crashed rather than degrading, and explain.py's retry could not
    help because the second attempt hits the same behaviour. Prose before the
    object is the same failure wearing a different hat.

    Anything after the first complete object is the model failing to stop, so
    it is dropped. A reply with no object at all is a real failure and raises:
    recovering must not shade into inventing.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    raise ValueError(f"no JSON object in model reply: {text[:200]!r}")


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
