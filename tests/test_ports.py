"""The ports are checked relationships, not documentation.

A Protocol nothing verifies is a comment with extra syntax. These tests assert
that every implementation this repo ships satisfies its port, and -- more
importantly -- that a backend which is not Ollama actually works end to end.
That second claim is the one the README makes and the one a user would be
annoyed to discover was untrue.
"""

import json

import httpx
import numpy as np
import pytest

from attestation.embed import Embedder
from attestation.llm import ChatClient, EmbeddingClient
from attestation.ports import ChatPort, EmbedderPort, EmbeddingPort


def test_shipped_clients_satisfy_their_ports():
    assert isinstance(ChatClient(), ChatPort)
    assert isinstance(EmbeddingClient(), EmbeddingPort)


def test_the_test_double_satisfies_the_embedder_port(fake_embedder):
    """conftest.FakeEmbedder predates the port. If it drifts out of shape, the
    suite is testing something the real ranking path could not accept."""
    assert isinstance(fake_embedder, EmbedderPort)


def test_the_real_embedder_satisfies_the_embedder_port():
    assert isinstance(Embedder(), EmbedderPort)


def test_a_non_ollama_chat_backend_works(monkeypatch):
    """The portability claim, exercised rather than asserted.

    A stub speaking OpenAI's wire format with none of Ollama's behaviour: a
    different host, and it rejects `reasoning_effort` with a 400 the way a
    stricter server does. The client must retry without the field, which is the
    compatibility path that would otherwise only be covered against Ollama.
    """
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": "unsupported parameter"})
        assert str(request.url).startswith("https://vendor.example/v1")
        assert request.headers["authorization"] == "Bearer sk-test"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"tags": ["rag"]})}}]},
        )

    client = ChatClient(
        base_url="https://vendor.example/v1",
        model="some-other-model",
        api_key="sk-test",
        transport=httpx.MockTransport(handler),
    )
    out = client.chat_json([{"role": "user", "content": "hi"}], {"type": "object"})

    assert out == {"tags": ["rag"]}
    assert len(seen) == 2, "must retry once without reasoning_effort"
    assert "reasoning_effort" not in seen[1]
    assert seen[1]["model"] == "some-other-model"


def test_a_non_ollama_embedding_backend_works():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://vendor.example/v1/embeddings"
        return httpx.Response(200, json={"data": [{"embedding": [0.5, 0.25, 0.125]}]})

    client = EmbeddingClient(
        base_url="https://vendor.example/v1",
        model="other-embed",
        transport=httpx.MockTransport(handler),
    )
    assert client.embed("hello") == [0.5, 0.25, 0.125]


def test_a_hand_written_object_satisfies_the_port_without_inheriting():
    """Structural, not nominal: no base class, no registration."""

    class Whatever:
        dims = 4

        def embed_document(self, title: str, text: str) -> np.ndarray:
            return np.ones(4, dtype=np.float32)

        def embed_query(self, text: str) -> np.ndarray:
            return np.ones(4, dtype=np.float32)

    assert isinstance(Whatever(), EmbedderPort)


def test_a_missing_method_fails_the_port():
    """The check has to be able to say no, or it says nothing."""

    class HalfDone:
        dims = 4

        def embed_document(self, title: str, text: str) -> np.ndarray:
            return np.ones(4, dtype=np.float32)

    assert not isinstance(HalfDone(), EmbedderPort)


@pytest.mark.parametrize("status", [401, 429, 500])
def test_transport_failures_propagate_rather_than_being_swallowed(status):
    """Reliability policy belongs to callers -- rank.py:198 serves a stale
    cached vector when this raises, and can only do that if it raises."""
    client = EmbeddingClient(
        base_url="https://vendor.example/v1",
        transport=httpx.MockTransport(lambda r: httpx.Response(status, json={})),
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.embed("x")
