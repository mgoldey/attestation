"""OpenAI-compatible LLM transport: chat completions + embeddings.

All config resolves at construction/call time (never at import):
constructor arg > env var > default. No retries here — reliability policy
(retry-then-skip, cache fallback) belongs to the callers.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from attestation.ports import (
    BackendNotConfigured,
    BackendUnreachable,  # noqa: F401 -- re-exported: callers import them from here
    backend_unreachable,  # noqa: F401
    display_url,
    key_is_header_safe,
    register_secret,
)

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
    "EMBED_BASE_URL",
    "EMBED_API_KEY",
    "ATTEST_LLM_FROM_HERMES",
)

FROM_HERMES_VAR = "ATTEST_LLM_FROM_HERMES"
BUILTIN = "built-in Ollama default"


def _opted_into_hermes() -> bool:
    return os.environ.get(FROM_HERMES_VAR, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ChatTarget:
    """Where chat goes, who makes the call, and where each part came from.

    `kind` is "openai" (this process posts to `base_url` with `api_key`) or
    "hermes" (Hermes' own runtime makes the call -- `base_url` and `api_key`
    are empty and nothing of the credential is ever here). `api_key` is
    excluded from `repr`. `*_source` are human-readable ("env LLM_BASE_URL",
    "Hermes (nvidia, ...)", "built-in Ollama default") and are what every
    unreachable / no-such-model message quotes; `origin` is the one-phrase form.
    """

    base_url: str
    model: str
    api_key: str = field(default="", repr=False)
    url_source: str = BUILTIN
    model_source: str = BUILTIN
    key_source: str = "unset"
    kind: str = "openai"
    provider: str = ""
    model_override: bool = False

    @property
    def origin(self) -> str:
        """Where this target came from, in one phrase, for error messages."""
        if self.kind == "hermes":
            return f"Hermes (provider {self.provider}, its own runtime)"
        return self.url_source


def _hermes_target(hermes, model_env, key_env) -> ChatTarget:
    where = f"Hermes ({hermes.provider}, {hermes.config_path})"
    override = model_env not in (None, "", DEFAULT_CHAT_MODEL)
    if hermes.kind == "hermes":
        return ChatTarget(
            base_url="",
            model=model_env if override else hermes.model,
            url_source=where,
            model_source="env CHAT_MODEL" if override else where,
            key_source=f"held by Hermes ({hermes.why})",
            kind="hermes",
            provider=hermes.provider,
            model_override=override,
        )
    # LLM_API_KEY is NOT used here: it was set for an LLM_BASE_URL, and this URL is
    # Hermes'. Pairing a key with a host it was not set for is how keys leak.
    register_secret(hermes.api_key)
    return ChatTarget(
        base_url=hermes.base_url,
        model=model_env if override else hermes.model,
        api_key=hermes.api_key,
        url_source=where,
        model_source="env CHAT_MODEL" if override else where,
        key_source=(
            f"Hermes ({hermes.provider}: {hermes.key_source})" if hermes.api_key else "unset"
        ),
        provider=hermes.provider,
        model_override=override,
    )


def chat_target() -> ChatTarget:
    """Resolve the chat backend, at call time.

    Per field, first match wins:

    1. the env var (`LLM_BASE_URL`, `CHAT_MODEL`, `LLM_API_KEY`);
    2. the model Hermes is connected to -- ONLY when the host opted in with
       `ATTEST_LLM_FROM_HERMES=1` and `LLM_BASE_URL` is not set to something
       of its own (see below). Whatever the provider: a key provider Attestation
       calls itself, anything else (ChatGPT sign-in, Anthropic, ...) through
       Hermes' own runtime -- see `hermes_model` and `hermes_bridge`;
    3. the built-in Ollama default.

    Opt-in is off by default, so a non-Hermes user's resolution is exactly
    what it was. A hosted Research Desk with no chat model on its loopback
    server is the case it exists for: Hermes' model is the only chat model
    that machine has.

    `LLM_BASE_URL` / `CHAT_MODEL` equal to the built-in defaults count as
    unset for this purpose: `.env.sample` writes exactly those two lines
    uncommented and `attest install` copies it to `.env`, so on a provisioned
    box they are boilerplate, not a decision, and honouring them would leave
    the opt-in with nothing to do. Anything else in LLM_BASE_URL is a decision
    and is final -- Hermes is not consulted at all.

    Raises `BackendNotConfigured` when opted in and Hermes has no usable model.
    """
    url_env = os.environ.get("LLM_BASE_URL")
    model_env = os.environ.get("CHAT_MODEL")
    key_env = os.environ.get("LLM_API_KEY", "")
    if key_env and not key_is_header_safe(key_env):
        raise BackendNotConfigured(
            "LLM_API_KEY is not a valid HTTP header value (it has a space, a newline or a"
            " non-ASCII character); re-enter it"
        )
    register_secret(key_env)
    if _opted_into_hermes() and url_env in (None, "", DEFAULT_BASE_URL):
        from attestation import hermes_model

        return _hermes_target(hermes_model.resolve(), model_env, key_env)
    return ChatTarget(
        base_url=url_env or DEFAULT_BASE_URL,
        model=model_env or DEFAULT_CHAT_MODEL,
        api_key=key_env,
        url_source="env LLM_BASE_URL" if url_env else BUILTIN,
        model_source="env CHAT_MODEL" if model_env else BUILTIN,
        key_source="env LLM_API_KEY" if key_env else "unset",
    )


def base_url() -> str:
    """The chat server root, resolved at call time -- see `chat_target`.
    Never cached, so a `.env` change or a new Hermes model takes effect
    without restarting anything that only imports this module."""
    return chat_target().base_url


def chat_model() -> str:
    """The chat model name, resolved at call time -- see `chat_target`."""
    return chat_target().model


def chat_api_key() -> str:
    """The chat Bearer key ("" when none), resolved at call time. Callers put
    it in a header and nowhere else."""
    return chat_target().api_key


def describe_chat() -> str:
    """ "<host> model <name> (url from ..., model from ...)" -- or the reason
    there is none. For `attest install --check` and every chat failure message:
    hosts and names only, never a key."""
    try:
        t = chat_target()
    except BackendNotConfigured as exc:
        return f"not resolved: {exc}"
    where = f"url: {t.url_source}; model: {t.model_source}; key: {t.key_source}"
    if t.kind == "hermes":
        return f"via Hermes' own runtime, provider {t.provider}, model {t.model} ({where})"
    return f"{display_url(t.base_url)} model {t.model} ({where})"


def chat_failure_message(detail: str | None = None) -> str:
    """The stderr text for a chat run that had to stop.

    Says what failed, WHERE the url and model came from, and how to change
    them -- replacing the bare "is ollama running?" that sent a user with a
    hosted model to start a daemon they never wanted.
    """
    try:
        t = chat_target()
    except BackendNotConfigured as exc:
        return str(exc)
    at = f"via {t.origin}" if t.kind == "hermes" else f"at {display_url(t.base_url)}"
    where = f"chat model {t.model!r} {at} (url from {t.url_source}; model from {t.model_source})"
    fix = (
        "set LLM_BASE_URL, CHAT_MODEL and LLM_API_KEY to a server that has the model, or"
        f" {FROM_HERMES_VAR}=1 to use the model Hermes is connected to"
        if not _opted_into_hermes()
        else "check the model Hermes is connected to (`attest install --check` shows what"
        " resolved), or set LLM_BASE_URL to override it"
    )
    return f"{where} failed: {detail or 'unreachable'} -- {fix}"


def embed_base_url() -> str:
    """Where embeddings come from: EMBED_BASE_URL, else the chat server.

    A hosted machine with no GPU can embed on its own CPU (a local Ollama
    serving embeddinggemma measured 20 items/s on 2 vCPUs, 380 MB resident)
    while chat goes to the customer's provider -- two servers, which one
    LLM_BASE_URL could not describe.

    Never the model Hermes is connected to (`ATTEST_LLM_FROM_HERMES`): that is
    a hosted CHAT model, and sending every title and abstract to it for
    embedding is a data-egress decision nobody made. Embeddings stay on
    EMBED_BASE_URL, else LLM_BASE_URL, else the built-in default.
    """
    return os.environ.get("EMBED_BASE_URL") or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL


def embed_api_key() -> str:
    """The Bearer key for the embedding server.

    EMBED_API_KEY when set. Otherwise LLM_API_KEY -- but only when embeddings
    go to the same server as chat: with EMBED_BASE_URL pointing somewhere
    else, the chat provider's key is never sent to that other host.
    """
    key = os.environ.get("EMBED_API_KEY") or (
        "" if os.environ.get("EMBED_BASE_URL") else os.environ.get("LLM_API_KEY", "")
    )
    register_secret(key)
    return key


def describe_embedding() -> str:
    """The embedding backend, host and model only, and where each came from."""
    if os.environ.get("EMBED_BASE_URL"):
        src = "env EMBED_BASE_URL"
    elif os.environ.get("LLM_BASE_URL"):
        src = "env LLM_BASE_URL"
    else:
        src = BUILTIN
    model_src = "env EMBED_MODEL" if os.environ.get("EMBED_MODEL") else BUILTIN
    return f"{display_url(embed_base_url())} model {embed_model()} (url: {src}; model: {model_src})"


def embed_model() -> str:
    """The configured embedding model name, resolved at call time -- see `base_url`."""
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
    register_secret(key)
    return {"Authorization": f"Bearer {key}"} if key else {}


class ChatClient:
    """A schema-constrained chat completion, against any OpenAI-compatible
    server -- see the module docstring: config resolves per call/construction,
    never at import, and reliability policy (retry, degrade) is the caller's."""

    def __init__(self, base_url=None, model=None, api_key=None, timeout=120, transport=None):
        # ONE reading of the environment and Hermes' files for all three, so
        # url, model and key cannot straddle a "Change model". A key is only
        # ever paired with the URL it was resolved for: a caller-supplied
        # base_url gets the caller's key or LLM_API_KEY, never Hermes'.
        if base_url is None:
            target = chat_target()
            if target.kind != "openai":
                raise BackendNotConfigured(
                    f"the chat model is served by {target.origin}, not an HTTP endpoint;"
                    " build the client with llm.chat_client()"
                )
            base_url, model = target.base_url, model or target.model
            api_key = target.api_key if api_key is None else api_key
        self.model = model or chat_model()
        self.client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers=_headers(api_key),
            transport=transport,
        )

    def _post(self, payload: dict) -> httpx.Response:
        """POST the completion, backing off briefly on 429 and 502/503/504.

        At most two retries, waiting `Retry-After` (else 1 s, then 2 s) capped at
        5 s: a rate limit or a proxy blip should cost seconds, not a tagging run,
        and a server that stays down must not be hammered.
        """
        resp = self.client.post("/chat/completions", json=payload)
        for attempt in range(2):
            if resp.status_code not in (429, 502, 503, 504):
                break
            try:
                wait = float(resp.headers.get("Retry-After", ""))
            except ValueError:
                wait = 2.0**attempt
            time.sleep(min(max(wait, 0.0), 5.0))
            resp = self.client.post("/chat/completions", json=payload)
        return resp

    def chat_json(self, messages: list[dict], schema: dict) -> dict:
        """One chat call, requesting a JSON object matching `schema`.

        Sends `reasoning_effort="none"` first (see the comment below: chain-
        of-thought buys nothing for a small schema-bound reply and roughly
        doubled latency when measured), retrying once without it for a
        server that rejects the field with a 400 rather than ignoring it.
        """
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
        resp = self._post(payload)
        if resp.status_code in (400, 422):
            payload.pop("reasoning_effort")
            resp = self._post(payload)
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
    """One embedding vector per call, against any OpenAI-compatible server --
    same construction-time config resolution as `ChatClient`."""

    def __init__(self, base_url=None, model=None, api_key=None, timeout=60, transport=None):
        self.model = model or embed_model()
        self.client = httpx.Client(
            base_url=base_url or embed_base_url(),
            timeout=timeout,
            headers=_headers(api_key if api_key is not None else embed_api_key()),
            transport=transport,
        )

    def embed(self, text: str) -> list[float]:
        """The raw embedding for `text`, untruncated and unnormalized --
        `embed.truncate_normalize` is the caller's job, not this client's."""
        resp = self.client.post("/embeddings", json={"model": self.model, "input": text})
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Raw embeddings for `texts`, ONE HTTP request -- measured 7.2x
        faster than one call per item against live Ollama/embeddinggemma.

        The OpenAI-compatible `/embeddings` response does NOT guarantee
        `data` comes back in request order, only that each element carries
        the `index` it belongs at. Sorting by it before returning is
        load-bearing: trusting response order would silently mismatch a
        vector to the wrong item, corrupting the index in a way nothing
        downstream could detect.
        """
        if not texts:
            return []
        resp = self.client.post("/embeddings", json={"model": self.model, "input": texts})
        resp.raise_for_status()
        data = resp.json()["data"]
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


_default_chat_client = None
_default_chat_key: tuple | None = None


def chat_client(target: ChatTarget | None = None):
    """The client for `target` (default: the resolved one): a `ChatClient` for
    an OpenAI-compatible endpoint, a `HermesChatClient` for a provider only
    Hermes' runtime can call. Both offer `chat_json(messages, schema)`."""
    target = target or chat_target()
    if target.kind == "hermes":
        from attestation.hermes_bridge import HermesChatClient

        return HermesChatClient(model=target.model if target.model_override else None)
    return ChatClient(base_url=target.base_url, model=target.model, api_key=target.api_key)


def default_chat_fn(messages: list[dict], schema: dict) -> dict:
    """The default `chat_fn` for explain/tagging: a lazily built client,
    rebuilt whenever what it resolves to changes.

    Resolution is per call, not per process: an MCP server lives for a whole
    session, and a client cached forever would keep calling the model that was
    connected when it started after the customer chose another in AgentMarkit.
    """
    global _default_chat_client, _default_chat_key
    target = chat_target()
    key = (target.kind, target.base_url, target.model, target.api_key, target.model_override)
    if _default_chat_client is None or key != _default_chat_key:
        _default_chat_client = chat_client(target)
        _default_chat_key = key
    return _default_chat_client.chat_json(messages, schema)
