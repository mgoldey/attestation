"""Protocols for the two things this project talks to that it does not own.

Both are structural: an implementation satisfies one by having the right
methods, with no base class to inherit and no registration. Anything matching
these shapes works -- a local Ollama, a vLLM server, llama.cpp, LM Studio, a
hosted API, or a deterministic fake in a test.

**These are narrow on purpose.** There is no repository protocol here. An
earlier design proposed three, with in-memory fakes and a contract suite to
keep them honest; two reviews argued that a repository whose method count
tracks its call-site count is a rename rather than an abstraction, and the
argument held. See `docs/superpowers/specs/2026-08-21-onion-refactor-design.md`
(superseded) for the full reasoning. A protocol earns its place when a second
implementation genuinely exists. For chat and embeddings it does -- the whole
point is that the backend is swappable, and the test suite already ships a
second implementation of the embedder in `conftest.FakeEmbedder`. For SQLite it
does not.

Note what is NOT abstracted: reliability policy. `llm.py`'s docstring is
explicit that retry-then-skip and cache fallback belong to callers, and
`rank.py:198` depends on that -- it serves a stale cached profile vector when
the embedder is down and raises only when the cache is cold. A port that
swallowed or retried would take that decision away from the one place with
enough context to make it.
"""

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from attestation.citations import Reference


@runtime_checkable
class ChatPort(Protocol):
    """A chat backend that returns JSON conforming to a supplied schema.

    Schema-bound rather than free-text because every caller here wants a small
    structured object -- tags, a content type, an explanation. A backend that
    cannot constrain output to a schema does not satisfy this port, and should
    not: the callers parse the result without defensive checks precisely
    because the schema is a contract.
    """

    def chat_json(self, messages: list[dict], schema: dict) -> dict:
        """Return the model's reply parsed as JSON matching `schema`.

        Raises on transport failure rather than returning a sentinel. The
        caller decides whether that is fatal -- see the module docstring.
        """
        ...


@runtime_checkable
class EmbeddingPort(Protocol):
    """A backend that turns text into a vector.

    Vectors must be stable for identical input: `rank.py` caches a profile
    vector keyed on a hash of the interests text and would otherwise serve a
    cache entry that no longer corresponds to what it was computed from.

    A round-one review called this a one-implementation Protocol worth
    deleting -- nothing types a parameter against it directly, `EmbedderPort`
    below does that job for the ranking path. But `test_domain_reaches_
    models_only_through_ports` forbids a domain module from importing
    `attestation.llm`'s concrete client at all, and `embed.py`'s `Embedder`
    wraps exactly that client (`EmbeddingClient`) -- this Protocol is the
    structural contract `test_ports.py` checks `EmbeddingClient` against, the
    thing that makes "any backend shaped like this works" a checked property
    rather than a claim. Keep it.
    """

    def embed(self, text: str) -> list[float]:
        """Return the embedding of `text` as a list of floats."""
        ...


@runtime_checkable
class EmbedderPort(Protocol):
    """The document/query pair the ranking path actually uses.

    Deliberately distinct from `EmbeddingPort`: `embed.py` applies asymmetric
    prompts -- DOC_PROMPT when indexing, QUERY_PROMPT when searching -- because
    the model was trained that way and mixing them measurably degrades
    retrieval. A single `embed(text)` cannot express that difference, so the
    ranking path depends on this instead.

    `conftest.FakeEmbedder` has satisfied this shape since before it was
    written down; naming it makes that a checked relationship rather than a
    coincidence.
    """

    dims: int

    def embed_document(self, title: str, text: str) -> np.ndarray:
        """Embed an item for storage, using the document-side prompt."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search string or persona profile, using the query-side prompt."""
        ...


@runtime_checkable
class CitationPort(Protocol):
    """A source of bibliographic records, keyed by citation key or identifier.

    This one earns its place under the rule in the module docstring -- three
    implementations exist with genuinely different backends (a SQLite file, a
    text format, an HTTP API), and the resolver must treat them uniformly while
    recording which one answered. That recording is the point: it is what makes
    the offline guarantee's exception inspectable rather than merely documented.
    """

    name: str
    """Which reader this is. Stamped onto every Reference it returns."""

    network: bool
    """Whether answering can leave the machine. See `citations.Resolver`."""

    def lookup(self, key: str) -> "Reference | None":
        """One record by citation key or identifier, or None if absent here."""
        ...

    def all(self) -> "Iterator[Reference]":
        """Every record this source can enumerate.

        Network readers raise NotImplementedError: you cannot enumerate
        CrossRef. Returns an iterator rather than a list because a Zotero
        library of 8,000 items should not be materialised to answer "is this
        key present".
        """
        ...


class BackendUnreachable(RuntimeError):
    """The model backend refused or never answered the socket.

    Raised by callers that must stop a whole run on the condition (tagging)
    so the run can catch it narrowly; `backend_unreachable` classifies the
    raw transport error for callers that keep the original exception.

    Here rather than in llm.py because it is the contract of any backend, and
    the domain must be able to name it without naming a provider.
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


class BackendNotConfigured(BackendUnreachable):
    """There is no model target to call: the host asked for the one Hermes is
    connected to and Hermes has none (or its credentials need renewing, or its
    runtime cannot be found).

    A kind of `BackendUnreachable` so every run that already stops once on a
    dead backend stops once on this too, with the message it carries --
    instead of asking the Ollama default for a model it never had.
    """


class BackendRejected(BackendUnreachable):
    """A model backend that WAS reached refused the request for a reason that
    holds for every request (401/403 key, 404 no such model, 410 retired).

    Raised by clients that do not speak httpx themselves (the Hermes bridge),
    so the domain's classifiers treat it exactly like the HTTP error a direct
    client would have raised. `status` is the HTTP status, 0 when unknown.
    """

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


def _server_message(response) -> str:
    """The reason a server gave in its error body (OpenAI/Ollama `error.message`,
    NIM `detail`/`title`), else empty. Bounded: a body is not a log line."""
    try:
        body = response.json()
    except ValueError:
        return ""
    if not isinstance(body, dict):
        return ""
    err = body.get("error")
    text = err.get("message") if isinstance(err, dict) else err
    text = text or body.get("detail") or body.get("title") or body.get("message") or ""
    return " ".join(str(text).split())[:200]


def backend_rejected(exc: BaseException) -> bool:
    """Whether a REACHABLE backend refused the request for a reason that holds
    for every request: 401/403 (key), 404 (no such model or path), 410 (end of
    life). Retrying per item cannot help, so a run stops once and says why.

    400 and 422 are NOT here: those can be one item's problem (a schema the
    server cannot satisfy), and the tagger's own retry owns them.
    """
    import httpx

    if isinstance(exc, BackendRejected):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (
        401,
        403,
        404,
        410,
    )


def failure_detail(exc: BaseException) -> str:
    """One line on why a backend call failed, safe to print.

    Never `str(exc)` for an HTTP error: that carries the full request URL. The
    status and the server's own reason are the useful part (a 404 from Ollama
    reads `model 'x' not found`), and neither can contain a credential.
    """
    import httpx

    if isinstance(exc, BackendUnreachable):  # carries its own message
        return redact(str(exc))
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        # A rejected credential: the server's words are where an echo of it would
        # be, and nothing in them is actionable beyond the status.
        reason = "" if status in (401, 403) else redact(_server_message(exc.response))
        return f"HTTP {status}" + (f": {reason}" if reason else "")
    if isinstance(exc, httpx.TransportError):
        return f"cannot connect ({type(exc).__name__})"
    return redact(str(exc) or type(exc).__name__)


def display_url(url: str) -> str:
    """A URL safe to print: scheme, host, port and path -- no userinfo, no query."""
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return f"{parts.scheme}://{host}{parts.path}" if parts.scheme else url


def embedding_backend_hint() -> str:
    """Where the embedder's URL came from, for the "embedding model unreachable"
    messages of the domain modules (which may not import `attestation.llm`).

    Reads the same two variables `llm.embed_base_url` does and says which one
    won, or that neither is set and the built-in Ollama default applies --
    the old text named the variable but called an unset one "unset" and left
    the reader to find out that meant Ollama on port 11434.
    """
    import os

    if os.environ.get("EMBED_BASE_URL"):
        return f"EMBED_BASE_URL={display_url(os.environ['EMBED_BASE_URL'])}"
    if os.environ.get("LLM_BASE_URL"):
        return (
            f"LLM_BASE_URL={display_url(os.environ['LLM_BASE_URL'])};"
            " EMBED_BASE_URL is unset, so embeddings use it too"
        )
    return (
        "EMBED_BASE_URL and LLM_BASE_URL are unset, so this is the built-in Ollama default;"
        " set EMBED_BASE_URL to change it"
    )


# --- credentials: never in a message -------------------------------------------------------

_SECRETS: set[str] = set()
_SHAPES = re.compile(
    r"(?i)(bearer\s+[A-Za-z0-9._~+/=-]{8,}"
    r"|(?:authorization|x-api-key|api[_-]?key|token|secret)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9._~+/=-]{8,}"
    r"|\b(?:sk|nvapi|xai|gsk|hf|pk|rk|ghp|gho)[-_][A-Za-z0-9._-]{12,}"
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,})"
)
_HEADER_SAFE = re.compile(r"[\x21-\x7e]+")
PLACEHOLDER_KEYS = ("ollama", "no-key-required", "lm-studio", "none")


def register_secret(value: str | None) -> None:
    """Remember a credential so `redact` removes it from every message, whatever
    shape it has. Called wherever a key is resolved; nothing is ever printed."""
    if value and len(value) >= 6 and value.lower() not in PLACEHOLDER_KEYS:
        _SECRETS.add(value)


def redact(text: str) -> str:
    """`text` with every registered secret and every credential-shaped string
    (Bearer values, x-api-key, sk-/nvapi- keys, JWTs) replaced by `***`."""
    for secret in sorted(_SECRETS, key=len, reverse=True):
        text = text.replace(secret, "***")
    return _SHAPES.sub("***", text)


def key_is_header_safe(value: str) -> bool:
    """Whether `value` can be an HTTP header value: printable ASCII, no spaces or
    newlines. A key that is not fails every request in a way that names nothing."""
    return bool(_HEADER_SAFE.fullmatch(value))


def host_is_loopback(url: str) -> bool:
    """Whether the URL's host is this machine (localhost, 127.0.0.0/8, ::1)."""
    import ipaddress
    from urllib.parse import urlsplit

    host = (urlsplit(url).hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
