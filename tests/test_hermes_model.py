"""Chat on the model Hermes is connected to: what is connected, who calls it, and
what is said when nothing usable is.

The failure this feature fixes was measured on a hosted Research Desk: the host
provisioned a loopback embedder (EMBED_BASE_URL), Hermes ran on a hosted model,
and Attestation's chat fell through to the built-in Ollama default -- asking the
embedder-only server for gemma4. The server answered 404
`model 'gemma4:e2b-it-q4_K_M' not found`; the tagger retried every item twice,
counted them `failed`, and printed nothing. `stub_servers.ollama_embedder_only`
reproduces that server, so these tests assert the failure and the fix against the
real shape rather than a mock of what we think it does.

This file covers providers Attestation calls itself (`kind="openai"`: a key and
an OpenAI-compatible URL). Providers only Hermes' runtime can call (ChatGPT
sign-in, Anthropic) are in test_hermes_bridge.py. Every credential is a literal
FAKE_KEY; the tests assert it never reaches stdout, stderr, a log record or an
exception message.
"""

import httpx
import pytest
from hermes_fixtures import (
    FAKE_KEY,
    NIM_MODEL,
    config,
    db_with_items,
    make_home,
    run_attest,
)
from stub_servers import chat_server, ollama_embedder_only

from attestation import hermes_model, llm
from attestation.db import get_db
from attestation.ports import BackendNotConfigured

NIM_CONFIG = config("nvidia", NIM_MODEL, "https://integrate.api.nvidia.com/v1")


def nim_home(tmp_path, monkeypatch, cfg=NIM_CONFIG):
    return make_home(tmp_path, monkeypatch, cfg, f"NVIDIA_API_KEY={FAKE_KEY}\nOTHER=1\n")


@pytest.fixture
def servers():
    made = []

    def make(server):
        made.append(server)
        return server

    yield make
    for s in made:
        s.close()


# --- the resolver: providers Attestation calls itself ----------------------------------------


def test_resolves_a_nim_shaped_config_from_files_alone(tmp_path, monkeypatch):
    """The refresh runs under cron: the key is in HERMES_HOME/.env, NOT in the
    process environment, and that must be enough."""
    nim_home(tmp_path, monkeypatch)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    m = hermes_model.resolve()
    assert (m.kind, m.provider, m.model, m.base_url) == (
        "openai",
        "nvidia",
        NIM_MODEL,
        "https://integrate.api.nvidia.com/v1",
    )
    assert m.api_key == FAKE_KEY
    assert FAKE_KEY not in repr(m), "a formatted model must not carry the key"


@pytest.mark.parametrize(
    ("provider", "env_var", "default_url"),
    [
        ("nvidia", "NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1"),
        ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
        ("openai-api", "OPENAI_API_KEY", "https://api.openai.com/v1"),
        ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
    ],
)
def test_a_key_provider_resolves_its_default_endpoint_and_its_own_key_variable(
    tmp_path, monkeypatch, provider, env_var, default_url
):
    make_home(tmp_path, monkeypatch, config(provider, "some/model"), f"{env_var}={FAKE_KEY}\n")
    m = hermes_model.resolve()
    assert (m.kind, m.base_url, m.api_key) == ("openai", default_url, FAKE_KEY)


def test_nim_base_url_override_in_dotenv_is_honoured(tmp_path, monkeypatch):
    """AgentMarkit's 'NVIDIA NIM' flow writes NVIDIA_BASE_URL beside the key for
    a customer's own NIM address."""
    make_home(
        tmp_path,
        monkeypatch,
        config("nvidia", NIM_MODEL),
        f"NVIDIA_API_KEY={FAKE_KEY}\nNVIDIA_BASE_URL=https://nim.example.test/v1/\n",
    )
    assert hermes_model.resolve().base_url == "https://nim.example.test/v1"


def test_a_custom_endpoint_reads_its_key_through_the_env_reference(tmp_path, monkeypatch):
    """AgentMarkit's 'custom' flow: `model.api_key: ${CUSTOM_API_KEY}` so
    config.yaml never holds the secret."""
    make_home(
        tmp_path,
        monkeypatch,
        config("custom", "my-model", "http://gw.example.test/v1/", "${CUSTOM_API_KEY}"),
        f"CUSTOM_API_KEY={FAKE_KEY}\n",
    )
    m = hermes_model.resolve()
    assert (m.kind, m.base_url, m.api_key) == ("openai", "http://gw.example.test/v1", FAKE_KEY)


def test_ollama_as_hermes_runs_it_is_a_keyless_custom_endpoint(tmp_path, monkeypatch):
    """AgentMarkit's 'ollama' flow: provider `custom`, the local /v1 URL and the
    placeholder key `ollama`. A local server ignores the key."""
    make_home(
        tmp_path,
        monkeypatch,
        config("custom", "llama3.2", "http://localhost:11434/v1", "${CUSTOM_API_KEY}"),
        "CUSTOM_API_KEY=ollama\n",
    )
    m = hermes_model.resolve()
    assert (m.kind, m.base_url, m.api_key) == ("openai", "http://localhost:11434/v1", "ollama")


def test_llamacpp_or_lmstudio_with_no_key_at_all_resolves(tmp_path, monkeypatch):
    make_home(tmp_path, monkeypatch, config("lmstudio", "qwen"))
    monkeypatch.delenv("LM_API_KEY", raising=False)
    m = hermes_model.resolve()
    assert (m.kind, m.base_url, m.api_key) == ("openai", "http://127.0.0.1:1234/v1", "")


def test_provider_default_url_and_process_env_key_as_last_resort(tmp_path, monkeypatch):
    make_home(tmp_path, monkeypatch, config("nvidia", NIM_MODEL))
    monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
    m = hermes_model.resolve()
    assert m.base_url == "https://integrate.api.nvidia.com/v1" and m.api_key == FAKE_KEY


def test_dotenv_wins_over_process_env_for_the_key(tmp_path, monkeypatch):
    nim_home(tmp_path, monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "from-the-process")
    assert hermes_model.resolve().api_key == FAKE_KEY


def test_a_profile_home_resolves_its_own_model_not_the_main_ones(tmp_path, monkeypatch):
    """Hermes profiles are HERMES_HOME directories of their own."""
    main = tmp_path / "main" / ".hermes"
    profile = main / "profiles" / "research"
    for home, cfg, env in (
        (main, config("nvidia", "main-model"), f"NVIDIA_API_KEY={FAKE_KEY}\n"),
        (profile, config("openrouter", "profile-model"), "OPENROUTER_API_KEY=other-key\n"),
    ):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(cfg)
        (home / ".env").write_text(env)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    got = hermes_model.resolve()
    assert (got.provider, got.model, got.api_key) == ("openrouter", "profile-model", "other-key")
    monkeypatch.setenv("HERMES_HOME", str(main))
    assert hermes_model.resolve().model == "main-model"


def test_a_config_with_a_model_but_no_key_says_which_variable(tmp_path, monkeypatch):
    make_home(tmp_path, monkeypatch, NIM_CONFIG, env="UNRELATED=1\n")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(BackendNotConfigured) as exc:
        hermes_model.resolve()
    assert "NVIDIA_API_KEY" in str(exc.value) and "no API key" in str(exc.value)


def test_no_config_means_no_model_connected(tmp_path, monkeypatch):
    make_home(tmp_path, monkeypatch, cfg=None)
    with pytest.raises(BackendNotConfigured, match="Hermes has no model connected; connect one"):
        hermes_model.resolve()


@pytest.mark.parametrize(
    "cfg",
    ["", "model:\n  provider: nvidia\n", "model: {}\n", "other: 1\n", "- just\n- a list\n"],
)
def test_a_config_without_a_model_means_no_model_connected(tmp_path, monkeypatch, cfg):
    make_home(tmp_path, monkeypatch, cfg, env=f"NVIDIA_API_KEY={FAKE_KEY}\n")
    with pytest.raises(BackendNotConfigured, match="Hermes has no model connected"):
        hermes_model.resolve()


def test_malformed_yaml_is_named_and_never_echoes_the_offending_line(tmp_path, monkeypatch):
    """PyYAML's own message renders the bad line; in a config holding
    `api_key:` that line can be the key."""
    bad = f"model:\n  default: x\n  api_key: {FAKE_KEY}\n  broken: [unclosed\n"
    make_home(tmp_path, monkeypatch, bad)
    with pytest.raises(BackendNotConfigured) as exc:
        hermes_model.resolve()
    msg = str(exc.value)
    assert "not valid YAML" in msg and "line" in msg
    assert FAKE_KEY not in msg and "unclosed" not in msg


def test_a_bare_model_string_without_a_provider_is_incomplete(tmp_path, monkeypatch):
    make_home(tmp_path, monkeypatch, "model: some/model\n")
    with pytest.raises(BackendNotConfigured, match="sets no model.provider"):
        hermes_model.resolve()


# --- the resolver: providers only Hermes' runtime can call -----------------------------------


@pytest.mark.parametrize(
    ("cfg", "auth", "provider"),
    [
        (config("openai-codex", "gpt-5.5"), "{}", "openai-codex"),  # ChatGPT sign-in
        (config("anthropic", "claude-sonnet-4-6"), None, "anthropic"),
        (config("bedrock", "anthropic.claude-v3"), None, "bedrock"),
        (config("brand-new-provider", "m"), None, "brand-new-provider"),
        (config("nvidia", NIM_MODEL, api_mode="codex_responses"), None, "nvidia"),
        (config("custom", "m", "https://x.example.test/anthropic"), None, "custom"),
    ],
)
def test_providers_without_a_chat_completions_key_are_delegated_to_hermes(
    tmp_path, monkeypatch, cfg, auth, provider
):
    make_home(tmp_path, monkeypatch, cfg, f"NVIDIA_API_KEY={FAKE_KEY}\n", auth)
    m = hermes_model.resolve()
    assert (m.kind, m.provider, m.api_key) == ("hermes", provider, "")
    assert m.why and FAKE_KEY not in repr(m)


def test_a_codex_config_with_no_credentials_is_still_delegated_not_guessed(tmp_path, monkeypatch):
    """Whether the sign-in is valid is Hermes' to say (and it says it precisely,
    see test_hermes_bridge); this module never reads an OAuth token."""
    make_home(tmp_path, monkeypatch, config("openai-codex", "gpt-5.5"))
    assert hermes_model.resolve().kind == "hermes"


# --- the resolution order --------------------------------------------------------------------


def test_opt_in_off_is_todays_behaviour_and_never_reads_hermes(tmp_path, monkeypatch):
    nim_home(tmp_path, monkeypatch)

    def boom(*_a, **_k):
        raise AssertionError("Hermes' files were read without ATTEST_LLM_FROM_HERMES")

    monkeypatch.setattr(hermes_model, "resolve", boom)
    t = llm.chat_target()
    assert (t.base_url, t.model, t.api_key, t.kind) == (
        llm.DEFAULT_BASE_URL,
        llm.DEFAULT_CHAT_MODEL,
        "",
        "openai",
    )
    assert t.url_source == t.model_source == llm.BUILTIN
    assert llm.base_url() == llm.DEFAULT_BASE_URL and llm.chat_model() == llm.DEFAULT_CHAT_MODEL


def test_opt_in_resolves_url_model_and_key_from_hermes(tmp_path, monkeypatch):
    nim_home(tmp_path, monkeypatch)
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    t = llm.chat_target()
    assert t.base_url == "https://integrate.api.nvidia.com/v1"
    assert t.model == NIM_MODEL and t.api_key == FAKE_KEY
    assert "Hermes (nvidia" in t.url_source and "Hermes (nvidia" in t.model_source
    assert FAKE_KEY not in repr(t) and FAKE_KEY not in llm.describe_chat()


def test_a_real_llm_base_url_always_wins_and_hermes_is_not_consulted(tmp_path, monkeypatch):
    nim_home(tmp_path, monkeypatch)
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    monkeypatch.setenv("LLM_BASE_URL", "http://lan-vllm:8000/v1")
    monkeypatch.setenv("CHAT_MODEL", "qwen")
    monkeypatch.setenv("LLM_API_KEY", "lan-key")
    monkeypatch.setattr(hermes_model, "resolve", lambda *a, **k: pytest.fail("Hermes consulted"))
    t = llm.chat_target()
    assert (t.base_url, t.model, t.api_key) == ("http://lan-vllm:8000/v1", "qwen", "lan-key")
    assert t.url_source == "env LLM_BASE_URL"


def test_the_env_sample_boilerplate_does_not_defeat_the_opt_in(tmp_path, monkeypatch):
    """`attest install` copies .env.sample to .env, which sets LLM_BASE_URL and
    CHAT_MODEL to the built-in defaults. On a provisioned box those are not a
    decision; honouring them would leave the opt-in with nothing to do."""
    nim_home(tmp_path, monkeypatch)
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    monkeypatch.setenv("LLM_BASE_URL", llm.DEFAULT_BASE_URL)
    monkeypatch.setenv("CHAT_MODEL", llm.DEFAULT_CHAT_MODEL)
    t = llm.chat_target()
    assert t.base_url.startswith("https://integrate.api.nvidia.com") and t.model == NIM_MODEL


def test_chat_model_and_key_env_vars_still_override_hermes_per_field(tmp_path, monkeypatch):
    nim_home(tmp_path, monkeypatch)
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    monkeypatch.setenv("CHAT_MODEL", "meta/llama-3.1-70b-instruct")
    monkeypatch.setenv("LLM_API_KEY", "explicit")
    t = llm.chat_target()
    assert t.base_url.startswith("https://integrate.api.nvidia.com")
    assert (t.model, t.api_key) == ("meta/llama-3.1-70b-instruct", "explicit")
    assert t.model_source == "env CHAT_MODEL" and t.key_source == "env LLM_API_KEY"


def test_change_model_in_hermes_is_followed_at_call_time(tmp_path, monkeypatch):
    home = nim_home(tmp_path, monkeypatch)
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    assert llm.chat_model() == NIM_MODEL
    (home / "config.yaml").write_text(NIM_CONFIG.replace(NIM_MODEL, "meta/llama-3.3-70b-instruct"))
    assert llm.chat_model() == "meta/llama-3.3-70b-instruct"


def test_the_default_client_is_rebuilt_when_the_model_changes(tmp_path, monkeypatch, servers):
    """An MCP server lives for a session; a client cached forever would keep
    calling the model that was connected when it started."""
    nim = servers(chat_server(models=("model-a", "model-b"), api_key=FAKE_KEY))
    home = make_home(
        tmp_path, monkeypatch, config("nvidia", "model-a", nim.url), f"NVIDIA_API_KEY={FAKE_KEY}\n"
    )
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    monkeypatch.setattr(llm, "_default_chat_client", None)
    llm.default_chat_fn([{"role": "user", "content": "x"}], {"type": "object"})
    (home / "config.yaml").write_text(config("nvidia", "model-b", nim.url))
    llm.default_chat_fn([{"role": "user", "content": "x"}], {"type": "object"})
    assert [r["model"] for r in nim.posts("/chat/completions")] == ["model-a", "model-b"]


def test_embeddings_never_go_to_the_hermes_model(tmp_path, monkeypatch):
    nim_home(tmp_path, monkeypatch)
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    assert llm.embed_base_url() == llm.DEFAULT_BASE_URL, "unset EMBED_BASE_URL is NOT Hermes' URL"
    assert llm.embed_api_key() == ""
    monkeypatch.setenv("EMBED_BASE_URL", "http://127.0.0.1:11434/v1")
    assert llm.embed_base_url() == "http://127.0.0.1:11434/v1"
    assert llm.embed_api_key() == ""
    assert FAKE_KEY not in llm.describe_embedding()


def test_a_chat_provider_with_no_embeddings_api_never_receives_an_embedding_request(
    tmp_path, monkeypatch, servers
):
    """The embedding client is a different client on a different URL; the Hermes
    chat model is never in its resolution, for either kind of backend."""
    chat = servers(chat_server(models=(NIM_MODEL,), api_key=FAKE_KEY))
    embedder = servers(ollama_embedder_only())
    make_home(
        tmp_path,
        monkeypatch,
        config("nvidia", NIM_MODEL, chat.url),
        f"NVIDIA_API_KEY={FAKE_KEY}\n",
    )
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    monkeypatch.setenv("EMBED_BASE_URL", embedder.url)
    monkeypatch.setenv("EMBED_MODEL", "embeddinggemma")
    llm.EmbeddingClient().embed("x")
    assert chat.requests == [] and len(embedder.posts("/embeddings")) == 1
    assert embedder.requests[0]["auth"] is None, "Hermes' key is not sent to the embedder"


def test_opted_in_with_no_hermes_model_raises_the_precise_message(tmp_path, monkeypatch):
    make_home(tmp_path, monkeypatch, cfg=None)
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    with pytest.raises(BackendNotConfigured, match="Hermes has no model connected; connect one"):
        llm.chat_model()
    assert "Hermes has no model connected" in llm.chat_failure_message()
    assert "not resolved" in llm.describe_chat()


def test_a_caller_supplied_base_url_never_receives_the_hermes_key(tmp_path, monkeypatch):
    nim_home(tmp_path, monkeypatch)
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    c = llm.ChatClient(base_url="http://elsewhere.example/v1", model="m")
    assert "authorization" not in {k.lower() for k in c.client.headers}
    c = llm.ChatClient()
    assert c.client.headers["authorization"] == f"Bearer {FAKE_KEY}"


def test_chat_json_retries_a_422_without_reasoning_effort():
    """Hosted gateways answer an unknown field with 422 as often as 400."""
    import json

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append("reasoning_effort" in body)
        if "reasoning_effort" in body:
            return httpx.Response(422, json={"detail": "extra field"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}]})

    c = llm.ChatClient(base_url="http://t/v1", model="m", transport=httpx.MockTransport(handler))
    assert c.chat_json([], {}) == {"ok": 1} and seen == [True, False]


# --- the real failure, against a server shaped like the one that failed ----------------------


def test_a_missing_chat_model_is_stopped_once_and_names_where_it_came_from(
    tmp_path, monkeypatch, capsys, caplog, servers
):
    """Before: 6 requests for 3 items, `failed: 3`, exit 1, NOTHING on stderr
    beyond the cost banner -- a 404 was neither 'unreachable' nor explained."""
    embedder = servers(ollama_embedder_only())
    monkeypatch.setenv("ATTEST_DB", str(db_with_items(tmp_path)))
    monkeypatch.setenv("EMBED_BASE_URL", embedder.url)
    monkeypatch.setenv("EMBED_MODEL", "embeddinggemma")
    monkeypatch.setenv("LLM_BASE_URL", embedder.url)  # stands in for the built-in default
    monkeypatch.setenv("CHAT_MODEL", "gemma4:e2b-it-q4_K_M")

    rc, out, err, _ = run_attest(["tag"], capsys, caplog)

    assert rc == 1
    assert len(embedder.posts("/chat/completions")) == 1, "stop on the first 404, not 2 per item"
    assert "'chat_down': True" in out
    assert "model 'gemma4:e2b-it-q4_K_M' not found" in err, "the server's own reason"
    assert "HTTP 404" in err
    assert "env LLM_BASE_URL" in err and "env CHAT_MODEL" in err, "where url and model came from"
    assert "ATTEST_LLM_FROM_HERMES=1" in err, "and how to change it"


def test_the_builtin_default_is_named_as_such_in_the_error(
    tmp_path, monkeypatch, capsys, caplog, servers
):
    embedder = servers(ollama_embedder_only())
    monkeypatch.setenv("ATTEST_DB", str(db_with_items(tmp_path)))
    monkeypatch.setattr(llm, "DEFAULT_BASE_URL", embedder.url)
    rc, _, err, _ = run_attest(["tag"], capsys, caplog)
    assert rc == 1
    assert llm.BUILTIN in err and "url from built-in Ollama default" in err


def test_an_unreachable_chat_server_names_where_the_url_came_from(
    tmp_path, monkeypatch, capsys, caplog
):
    monkeypatch.setenv("ATTEST_DB", str(db_with_items(tmp_path)))
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    rc, _, err, _ = run_attest(["tag"], capsys, caplog)
    assert rc == 1
    assert "http://127.0.0.1:9/v1" in err and "env LLM_BASE_URL" in err
    assert "cannot connect" in err and "is ollama running?" not in err


def test_opted_in_with_nothing_connected_fails_before_any_request(
    tmp_path, monkeypatch, capsys, caplog, servers
):
    embedder = servers(ollama_embedder_only())
    monkeypatch.setenv("ATTEST_DB", str(db_with_items(tmp_path)))
    monkeypatch.setenv("EMBED_BASE_URL", embedder.url)
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    make_home(tmp_path, monkeypatch, cfg=None)

    rc, _, err, _ = run_attest(["tag"], capsys, caplog)

    assert rc == 1
    assert "Hermes has no model connected; connect one in AgentMarkit" in err
    assert "is ollama running?" not in err
    assert embedder.requests == [], "no request to anything"


@pytest.mark.parametrize(
    ("provider", "key_var", "key_in_dotenv"),
    [
        ("nvidia", "NVIDIA_API_KEY", True),
        ("openrouter", "OPENROUTER_API_KEY", True),
        ("openai-api", "OPENAI_API_KEY", True),
        ("custom", "CUSTOM_API_KEY", True),
        ("lmstudio", None, False),
    ],
)
def test_end_to_end_chat_goes_to_the_hermes_model_and_never_to_ollama(
    tmp_path, monkeypatch, capsys, caplog, servers, provider, key_var, key_in_dotenv
):
    """The whole fix, through the real CLI, per provider shape: an embedder-only
    loopback server for embeddings, a stub standing in for the hosted chat model
    (it demands the key unless the provider has none), a fake HERMES_HOME.
    Tagging succeeds; the embedder saw no chat request; nothing was sent to port
    11434; the key appears in no output."""
    embedder = servers(ollama_embedder_only())
    key = FAKE_KEY if key_in_dotenv else None
    chat = servers(chat_server(models=("the-model",), api_key=key))
    api_key_cfg = "${CUSTOM_API_KEY}" if provider == "custom" else ""
    make_home(
        tmp_path,
        monkeypatch,
        config(provider, "the-model", chat.url, api_key_cfg),
        f"{key_var}={FAKE_KEY}\n" if key_in_dotenv else None,
    )
    monkeypatch.setenv("ATTEST_DB", str(db_with_items(tmp_path)))
    monkeypatch.setenv("EMBED_BASE_URL", embedder.url)
    monkeypatch.setenv("EMBED_MODEL", "embeddinggemma")
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    # .env.sample boilerplate, as `attest install` leaves it on a provisioned box
    monkeypatch.setenv("LLM_BASE_URL", llm.DEFAULT_BASE_URL)
    monkeypatch.setenv("CHAT_MODEL", llm.DEFAULT_CHAT_MODEL)
    if key_var:
        monkeypatch.delenv(key_var, raising=False)

    sent = []
    real_send = httpx.Client.send

    def record(self, request, **kw):
        sent.append(request.url)
        return real_send(self, request, **kw)

    monkeypatch.setattr(httpx.Client, "send", record)

    rc, out, err, logs = run_attest(["tag"], capsys, caplog)

    assert rc == 0, err
    assert "'tagged': 3" in out and "'model': 'the-model'" in out
    posts = chat.posts("/chat/completions")
    assert [r["model"] for r in posts] == ["the-model"] * 3
    assert {r["auth"] for r in posts} == ({f"Bearer {FAKE_KEY}"} if key_in_dotenv else {None})
    assert embedder.posts("/chat/completions") == [], "the embedder was never asked to chat"
    assert chat.posts("/embeddings") == [], "embeddings never go to the Hermes model"
    assert not [u for u in sent if u.port == 11434], "nothing went to the Ollama port"
    for text in (out, err, logs):
        assert FAKE_KEY not in text


def test_a_rejected_hermes_key_fails_once_without_printing_it(
    tmp_path, monkeypatch, capsys, caplog, servers
):
    nim = servers(chat_server(models=(NIM_MODEL,), api_key="a-different-key"))
    nim_home(tmp_path, monkeypatch, config("nvidia", NIM_MODEL, nim.url))
    monkeypatch.setenv("ATTEST_DB", str(db_with_items(tmp_path)))
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")

    rc, _, err, logs = run_attest(["tag"], capsys, caplog)

    assert rc == 1 and len(nim.posts("/chat/completions")) == 1
    assert "HTTP 401" in err and "Hermes (nvidia" in err
    for text in (err, logs):
        assert FAKE_KEY not in text and "a-different-key" not in text


def test_a_hermes_model_with_no_key_is_reported_by_attest_tag_without_a_traceback(
    tmp_path, monkeypatch, capsys, caplog
):
    make_home(tmp_path, monkeypatch, NIM_CONFIG, env="UNRELATED=1\n")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("ATTEST_DB", str(db_with_items(tmp_path)))
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    rc, _, err, _ = run_attest(["tag"], capsys, caplog)
    assert rc == 1 and "NVIDIA_API_KEY" in err and "Traceback" not in err


def test_library_tag_reports_the_same_way(tmp_path, monkeypatch, capsys, caplog):
    make_home(tmp_path, monkeypatch, cfg=None)
    monkeypatch.setenv("ATTEST_DB", str(db_with_items(tmp_path)))
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    rc, _, err, _ = run_attest(["library", "tag"], capsys, caplog)
    assert rc == 1 and "Hermes has no model connected" in err


# --- explanations, and the install check ------------------------------------------------------


def test_explain_distinguishes_a_missing_model_from_a_bad_reply(tmp_path):
    from attestation.explain import explain
    from attestation.rank import create_user

    conn = get_db(tmp_path / "e.db")
    conn.execute("INSERT INTO feeds(id, url, title) VALUES (1, 'http://f', 'F')")
    conn.execute(
        "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
        " VALUES (1, 1, 'T', 'http://x', 's', 'h')"
    )
    uid = create_user(conn, "reader", "graph networks")

    def no_such_model(messages, schema):
        request = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
        raise httpx.HTTPStatusError(
            "404",
            request=request,
            response=httpx.Response(
                404, request=request, json={"error": {"message": "model 'x' not found"}}
            ),
        )

    got = explain(conn, uid, 1, chat_fn=no_such_model)
    assert got.reason == "model_unreachable"
    assert "HTTP 404" in got.detail and "model 'x' not found" in got.detail
    assert explain(conn, uid, 1, chat_fn=lambda m, s: {"nope": 1}).reason == "no_answer"


def test_install_check_prints_resolved_backends_hosts_only(tmp_path, monkeypatch):
    from attestation import install

    nim_home(tmp_path, monkeypatch)
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    monkeypatch.setenv("EMBED_BASE_URL", "http://user:pw@127.0.0.1:11434/v1?token=zzz")
    r = install.step_backends()
    assert r.name == "backends" and r.status is install.Status.OK
    assert f"https://integrate.api.nvidia.com/v1 model {NIM_MODEL}" in r.detail
    assert "Hermes (nvidia" in r.detail and "embeddings: http://127.0.0.1:11434/v1" in r.detail
    for secret in (FAKE_KEY, "pw", "zzz"):
        assert secret not in r.detail


def test_install_check_is_broken_when_opted_in_and_hermes_has_nothing(tmp_path, monkeypatch):
    from attestation import install

    make_home(tmp_path, monkeypatch, cfg=None)
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    r = install.step_backends()
    assert r.status is install.Status.BROKEN
    assert "Hermes has no model connected; connect one in AgentMarkit" in r.detail
    assert install._is_ollama_backend() is False, "no resolved target is not an Ollama backend"


def test_install_check_for_a_default_machine_says_so(monkeypatch):
    from attestation import install

    r = install.step_backends()
    assert r.status is install.Status.OK
    assert llm.BUILTIN in r.detail and "localhost:11434" in r.detail


def test_install_hosted_probe_reaches_the_hermes_model_with_its_key(tmp_path, monkeypatch, servers):
    """`attest install --check` on a hosted desk: one embedding to the embedder,
    one chat request to the Hermes model, with the key, and no key in the line."""
    from attestation import install

    embedder = servers(ollama_embedder_only())
    chat = servers(chat_server(models=(NIM_MODEL,), api_key=FAKE_KEY))
    nim_home(tmp_path, monkeypatch, config("nvidia", NIM_MODEL, chat.url))
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    monkeypatch.setenv("EMBED_BASE_URL", embedder.url)
    monkeypatch.setenv("EMBED_MODEL", "embeddinggemma")
    monkeypatch.setenv("EMBED_DIMS", "256")
    r = install._hosted_probe()
    assert r.status is install.Status.OK, r.detail
    assert len(chat.posts("/chat/completions")) == 1 and len(embedder.posts("/embeddings")) == 1
    assert FAKE_KEY not in r.detail


def test_a_refused_provider_config_never_echoes_a_key(tmp_path, monkeypatch, capsys):
    """The exception path too: a config that holds a literal key and a broken line."""
    make_home(
        tmp_path,
        monkeypatch,
        f"model:\n  default: m\n  provider: nvidia\n  api_key: {FAKE_KEY}\n  x: [broken\n",
    )
    with pytest.raises(BackendNotConfigured) as exc:
        hermes_model.resolve()
    assert FAKE_KEY not in str(exc.value) and FAKE_KEY not in repr(exc.value)
    assert capsys.readouterr() == ("", "")
