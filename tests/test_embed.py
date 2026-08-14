import json

import httpx
import numpy as np

from attestation.embed import Embedder, truncate_normalize
from attestation.llm import EmbeddingClient


def make_embedder(captured):
    """Embedder wired to a mock /v1 transport returning a fixed 768-dim vector."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"embedding": [1.0] * 768}]})

    client = EmbeddingClient(
        base_url="http://test/v1", model="embeddinggemma", transport=httpx.MockTransport(handler)
    )
    return Embedder(client=client)


def test_truncate_normalize_renormalizes():
    vec = np.ones(768, dtype=np.float32)
    out = truncate_normalize(vec, dims=256)
    assert out.shape == (256,)
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_truncate_normalize_zero_vector_safe():
    out = truncate_normalize(np.zeros(768, dtype=np.float32))
    assert out.shape == (256,)
    assert not np.any(np.isnan(out))


def test_embed_document_prompt_format():
    captured = []
    emb = make_embedder(captured)
    vec = emb.embed_document("My Title", "body text")
    assert captured[0]["input"] == "title: My Title | text: body text"
    assert captured[0]["model"] == "embeddinggemma"
    assert vec.shape == (256,) and vec.dtype == np.float32


def test_embed_document_missing_title_uses_none():
    captured = []
    make_embedder(captured).embed_document("", "body")
    assert captured[0]["input"] == "title: none | text: body"


def test_embed_query_prompt_format():
    captured = []
    make_embedder(captured).embed_query("chemistry papers")
    assert captured[0]["input"] == "task: search result | query: chemistry papers"


def test_truncate_normalize_raises_when_model_too_small():
    import pytest

    with pytest.raises(ValueError, match="128.*256"):
        truncate_normalize(np.ones(128, dtype=np.float32), dims=256)


def test_truncate_normalize_default_dims_follows_env(monkeypatch):
    monkeypatch.setenv("EMBED_DIMS", "64")
    out = truncate_normalize(np.ones(768, dtype=np.float32))
    assert out.shape == (64,)
