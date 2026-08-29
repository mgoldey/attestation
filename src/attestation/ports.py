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
