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


def make_batch_embedder(captured, dims=768):
    """Embedder wired to a mock /v1 transport that answers a batch `input`
    list with one distinct vector per item, each carrying its response
    `index` -- lets tests assert order survives an out-of-order reply."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        texts = body["input"]
        data = [{"embedding": [float(i + 1)] * dims, "index": i} for i in range(len(texts))]
        return httpx.Response(200, json={"data": data})

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


def test_embed_documents_sends_one_batch_request_with_doc_prompt_per_item():
    captured = []
    emb = make_batch_embedder(captured)
    vecs = emb.embed_documents([("Title A", "body a"), ("Title B", "body b")])
    assert len(captured) == 1, "must be exactly one HTTP request for the whole batch"
    assert captured[0]["input"] == [
        "title: Title A | text: body a",
        "title: Title B | text: body b",
    ]
    assert len(vecs) == 2
    assert all(v.shape == (256,) and v.dtype == np.float32 for v in vecs)


def test_embed_documents_missing_title_uses_none():
    captured = []
    emb = make_batch_embedder(captured)
    emb.embed_documents([("", "body")])
    assert captured[0]["input"] == ["title: none | text: body"]


def test_embed_documents_preserves_order_against_an_out_of_order_response():
    """Same order-safety guarantee as EmbeddingClient.embed_many, exercised
    through the Embedder layer: a scrambled `index` in the response must not
    mismatch a normalized vector to the wrong (title, text) pair.

    Each stub vector is a distinct one-hot-ish pattern so normalization
    cannot make two different indices' outputs indistinguishable.
    """
    captured = []

    def one_hot(pos: int) -> list[float]:
        v = [0.1] * 768
        v[pos] = 10.0
        return v

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        # Scrambled response order: index 2, 0, 1 -- each with a distinct
        # one-hot marker at position `index` so downstream identity is provable.
        data = [
            {"embedding": one_hot(2), "index": 2},
            {"embedding": one_hot(0), "index": 0},
            {"embedding": one_hot(1), "index": 1},
        ]
        return httpx.Response(200, json={"data": data})

    client = EmbeddingClient(
        base_url="http://test/v1", model="embeddinggemma", transport=httpx.MockTransport(handler)
    )
    emb = Embedder(client=client)
    vecs = emb.embed_documents([("t0", "x"), ("t1", "y"), ("t2", "z")])
    assert len(vecs) == 3
    # vecs[i] must carry the marker at position i -- proves the item at
    # request position i got the response element whose `index` was i, not
    # whichever element happened to arrive at that position.
    for i, v in enumerate(vecs):
        assert np.argmax(v) == i, f"vecs[{i}] does not carry the index-{i} marker"
    assert captured[0]["input"] == [
        "title: t0 | text: x",
        "title: t1 | text: y",
        "title: t2 | text: z",
    ]


def test_embed_documents_empty_list_returns_empty():
    captured = []
    emb = make_batch_embedder(captured)
    assert emb.embed_documents([]) == []
    assert captured == []
