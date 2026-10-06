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
        # A NIM server carries the doc/query asymmetry in `input_type`, not in
        # gemma's task prefixes (spec 2026-10-06, decision 3). Clients without
        # the attribute (test fakes, any EmbeddingPort) get the prompts.
        self.typed = bool(getattr(self.client, "typed", False))

    def _doc_text(self, title: str, text: str) -> str:
        if self.typed:
            return f"{title}\n{text}" if title else text
        return DOC_PROMPT.format(title=title or "none", text=text)

    def _role(self, input_type: str) -> dict:
        return {"input_type": input_type} if self.typed else {}

    def _embed(self, prompt: str, input_type: str) -> np.ndarray:
        vec = np.asarray(self.client.embed(prompt, **self._role(input_type)), dtype=np.float32)
        return truncate_normalize(vec, self.dims)

    def embed_document(self, title: str, text: str) -> np.ndarray:
        """Embed for storage, with `DOC_PROMPT` (or `input_type=passage` on NIM).

        The doc and query prompts are asymmetric on purpose (see the module
        docstring): indexing and searching with the same prompt is the
        common embedding-search mistake this wrapper exists to prevent.
        """
        return self._embed(self._doc_text(title, text), "passage")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query, with `QUERY_PROMPT` -- see `embed_document`."""
        return self._embed(text if self.typed else QUERY_PROMPT.format(text=text), "query")

    def embed_documents(self, pairs: list[tuple[str, str]]) -> list[np.ndarray]:
        """Batch form of `embed_document`: ONE request for all `(title, text)`
        pairs, applying `DOC_PROMPT` to each and `truncate_normalize` to each
        returned vector -- same prompt, same normalization, same asymmetry
        guarantee, just fewer HTTP round trips.

        Order safety is `EmbeddingClient.embed_many`'s job (it sorts by the
        response `index`); this method just trusts the list it gets back is
        already aligned to `pairs` and truncates/normalizes each entry in place.
        """
        if not pairs:
            return []
        prompts = [self._doc_text(title, text) for title, text in pairs]
        raw = self.client.embed_many(prompts, **self._role("passage"))
        return [truncate_normalize(np.asarray(v, dtype=np.float32), self.dims) for v in raw]
