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

from typing import Protocol, runtime_checkable

import numpy as np


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
