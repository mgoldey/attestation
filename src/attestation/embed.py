"""Embedding client wrapper: doc/query prompts + Matryoshka truncation."""

import numpy as np

from attestation.llm import EmbeddingClient

DOC_PROMPT = "title: {title} | text: {text}"
QUERY_PROMPT = "task: search result | query: {text}"


def truncate_normalize(vec: np.ndarray, dims: int | None = None) -> np.ndarray:
    """Slice a Matryoshka embedding to `dims` and renormalise to unit length.

    Applied before every store or comparison: a raw vector's later dims carry
    finer distinctions a shorter, EMBED_DIMS-sized index deliberately drops,
    and slicing without renormalising would leave stored vectors at the wrong
    scale for cosine similarity. A model returning fewer dims than configured
    raises rather than silently zero-padding, which would fabricate signal.
    """
    from attestation.db import embed_dims

    dims = dims if dims is not None else embed_dims()
    if len(vec) < dims:
        raise ValueError(
            f"model returned a {len(vec)}-dim embedding but {dims} dims are configured"
            " (EMBED_DIMS) — use a larger model or smaller dims"
        )
    v = vec[:dims].astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


class Embedder:
    """Doc/query prompt formatting + truncation over an OpenAI-style embeddings client."""

    def __init__(self, client: EmbeddingClient | None = None, dims: int | None = None):
        from attestation.db import embed_dims

        self.client = client or EmbeddingClient()
        self.dims = dims if dims is not None else embed_dims()

    def _embed(self, prompt: str) -> np.ndarray:
        vec = np.asarray(self.client.embed(prompt), dtype=np.float32)
        return truncate_normalize(vec, self.dims)

    def embed_document(self, title: str, text: str) -> np.ndarray:
        """Embed for storage, with `DOC_PROMPT`.

        The doc and query prompts are asymmetric on purpose (see the module
        docstring): indexing and searching with the same prompt is the
        common embedding-search mistake this wrapper exists to prevent.
        """
        return self._embed(DOC_PROMPT.format(title=title or "none", text=text))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query, with `QUERY_PROMPT` -- see `embed_document`."""
        return self._embed(QUERY_PROMPT.format(text=text))
