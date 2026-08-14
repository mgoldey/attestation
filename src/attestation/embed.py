"""Embedding client wrapper: doc/query prompts + Matryoshka truncation."""

import numpy as np

from attestation.llm import EmbeddingClient

DOC_PROMPT = "title: {title} | text: {text}"
QUERY_PROMPT = "task: search result | query: {text}"


def truncate_normalize(vec: np.ndarray, dims: int | None = None) -> np.ndarray:
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
        return self._embed(DOC_PROMPT.format(title=title or "none", text=text))

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(QUERY_PROMPT.format(text=text))
