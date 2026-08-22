import json

import httpx
import pytest

from attestation.llm import ChatClient, EmbeddingClient, base_url, chat_model, embed_model, load_env


def chat_transport(captured, content='{"ok": true}'):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": json.loads(request.content),
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.MockTransport(handler)


def embed_transport(captured, dims=768):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": json.loads(request.content),
            }
        )
        return httpx.Response(200, json={"data": [{"embedding": [1.0] * dims}]})

    return httpx.MockTransport(handler)


def test_chat_json_request_shape_and_parse():
    captured = []
    client = ChatClient(base_url="http://test/v1", model="m1", transport=chat_transport(captured))
    out = client.chat_json([{"role": "user", "content": "hi"}], {"type": "object"})
    assert out == {"ok": True}
    req = captured[0]
    assert req["url"] == "http://test/v1/chat/completions"
    assert req["body"]["model"] == "m1"
    assert req["body"]["messages"] == [{"role": "user", "content": "hi"}]
    rf = req["body"]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == {"type": "object"}
    assert rf["json_schema"]["strict"] is True
    assert "authorization" not in req["headers"]  # no key set -> no Bearer header


def test_chat_client_env_fallbacks(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "env-model")
    monkeypatch.setenv("LLM_API_KEY", "sekrit")
    captured = []
    client = ChatClient(base_url="http://test/v1", transport=chat_transport(captured))
    client.chat_json([], {})
    assert captured[0]["body"]["model"] == "env-model"
    assert captured[0]["headers"]["authorization"] == "Bearer sekrit"


def test_chat_client_arg_beats_env(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "env-model")
    captured = []
    client = ChatClient(
        base_url="http://test/v1", model="arg-model", transport=chat_transport(captured)
    )
    client.chat_json([], {})
    assert captured[0]["body"]["model"] == "arg-model"


def test_chat_json_raises_on_http_error():
    transport = httpx.MockTransport(lambda req: httpx.Response(500))
    client = ChatClient(base_url="http://test/v1", model="m", transport=transport)
    with pytest.raises(httpx.HTTPStatusError):
        client.chat_json([], {})


def test_chat_json_asks_for_no_reasoning():
    """Every chat_json call wants a small schema-bound object, so thinking
    tokens are pure cost: measured 19.8s -> 10.5s on gemma4:e2b with it off."""
    captured = []
    client = ChatClient(base_url="http://test/v1", model="m", transport=chat_transport(captured))
    client.chat_json([], {})

    assert captured[0]["body"]["reasoning_effort"] == "none"


def test_chat_json_retries_without_reasoning_effort_on_400():
    """Backends that reject the field (the README advertises vLLM, OpenRouter,
    and OpenAI as drop-in) must still work -- retry once without it."""
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": "unrecognised field"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    client = ChatClient(
        base_url="http://test/v1", model="m", transport=httpx.MockTransport(handler)
    )
    out = client.chat_json([], {})

    assert out == {"ok": True}
    assert [("reasoning_effort" in b) for b in bodies] == [True, False]


def test_chat_json_does_not_swallow_other_400s():
    """Only the reasoning_effort retry is special: a 400 that persists without
    the field is a real error and must surface, not loop."""
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": "bad request"})

    client = ChatClient(
        base_url="http://test/v1", model="m", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.chat_json([], {})
    assert len(calls) == 2, "one retry, then raise"


def test_embedding_client_request_and_parse(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "env-embed")
    captured = []
    client = EmbeddingClient(base_url="http://test/v1", transport=embed_transport(captured))
    vec = client.embed("some text")
    assert captured[0]["url"] == "http://test/v1/embeddings"
    assert captured[0]["body"] == {"model": "env-embed", "input": "some text"}
    assert len(vec) == 768 and vec[0] == 1.0


def test_env_helper_defaults(monkeypatch):
    for var in ("LLM_BASE_URL", "CHAT_MODEL", "EMBED_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert base_url() == "http://localhost:11434/v1"
    assert chat_model() == "gemma4:e2b-it-q4_K_M"
    assert embed_model() == "embeddinggemma"


def test_load_env_real_environment_wins(tmp_path, monkeypatch):
    import attestation.llm

    (tmp_path / ".env").write_text("CHAT_MODEL=from-dotenv\nEMBED_MODEL=dotenv-embed\n")
    # hermetic: ignore any real checkout .env
    monkeypatch.setattr(attestation.llm, "_REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAT_MODEL", "from-shell")
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    load_env()
    assert chat_model() == "from-shell"  # override=False: shell wins
    assert embed_model() == "dotenv-embed"  # unset var: .env fills it


def test_env_sample_documents_exactly_the_vars_the_code_reads():
    import re
    from pathlib import Path

    from attestation import llm

    sample = (Path(__file__).resolve().parents[1] / ".env.sample").read_text()
    documented = {
        name
        for name in re.findall(r"^#?([A-Z_]+)=", sample, flags=re.MULTILINE)
        if not name.startswith("OLLAMA_")  # daemon section: documented, not read by hermes
    }
    known = set(llm.ENV_VARS) | {"EMBED_DIMS", "RSS_DB"}
    assert documented == known


def test_a_reply_with_trailing_junk_is_recovered_not_crashed():
    """Small models append a second object, or prose, after valid JSON.

    Observed live on gemma4:e2b: the reply parsed as `{"text": "..."}` followed
    by more content, and json.loads raised "Extra data: line 3 column 2".
    That propagates out of chat_json as an unhandled JSONDecodeError -- an
    explanation request crashes rather than degrading, and explain.py's retry
    cannot help because the second attempt hits the same behaviour.

    The first complete JSON object is what the schema asked for; anything
    after it is the model failing to stop.
    """
    body = '{"text": "Covers KV-cache compression, which you follow."}\n{"text": "extra"}'

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": body}}]})

    client = ChatClient(base_url="https://x/v1", transport=httpx.MockTransport(handler))
    out = client.chat_json([{"role": "user", "content": "hi"}], {"type": "object"})
    assert out == {"text": "Covers KV-cache compression, which you follow."}


def test_prose_before_the_json_is_recovered():
    """The other half of the same failure: a preamble the model was told not
    to write."""
    body = 'Here is the answer:\n{"text": "Overlaps your retrieval work."}'

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": body}}]})

    client = ChatClient(base_url="https://x/v1", transport=httpx.MockTransport(handler))
    out = client.chat_json([{"role": "user", "content": "hi"}], {"type": "object"})
    assert out == {"text": "Overlaps your retrieval work."}


def test_a_reply_with_no_json_at_all_still_raises():
    """Recovery must not become silent invention -- a reply with nothing
    parseable is a real failure and the caller decides what to do."""

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "sorry!"}}]})

    client = ChatClient(base_url="https://x/v1", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError):
        client.chat_json([{"role": "user", "content": "hi"}], {"type": "object"})
