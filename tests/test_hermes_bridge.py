"""Chat on a provider only Hermes' own runtime can call, through the bridge.

ChatGPT sign-in (`openai-codex`: OAuth, the Responses API) and Anthropic's native
API are the providers AgentMarkit connects that are not OpenAI chat-completions
with a static key. Attestation does not re-implement them: it asks Hermes to make
the call (`hermes_bridge`, `hermes_helper`). These tests drive the REAL helper and
bridge against a FAKE Hermes -- a tree of the three modules the helper imports,
behind a `hermes` launcher shaped like the real one -- so they run anywhere. The
real Hermes is exercised, against stubs of the Responses and Anthropic wires, by
`test_hermes_real.py` (opt-in: it needs a Hermes checkout).
"""

import json
import stat
import sys
import textwrap

import pytest
from hermes_fixtures import (
    CODEX_REFRESH,
    CODEX_TOKEN,
    FAKE_KEY,
    codex_auth,
    config,
    db_with_items,
    make_home,
    run_attest,
)
from stub_servers import chat_server, ollama_embedder_only

from attestation import hermes_bridge, llm
from attestation.ports import BackendNotConfigured, BackendRejected, BackendUnreachable

FAKE_CALL_LLM = textwrap.dedent(
    r"""
    import json, os, re, time, urllib.request
    from types import SimpleNamespace


    class APIStatusError(Exception):
        def __init__(self, status, message):
            super().__init__(message)
            self.status_code, self.body = status, {"error": {"message": message}}


    class APIConnectionError(Exception):
        pass


    def _config_model():
        text = open(os.path.join(os.environ["HERMES_HOME"], "config.yaml")).read()
        return re.search(r"default:\s*(\S+)", text).group(1)


    def call_llm(*, task=None, provider=None, model=None, base_url=None, api_key=None,
                 main_runtime=None, messages, temperature=None, max_tokens=None, tools=None,
                 timeout=None, extra_body=None, **kw):
        plan = json.load(open(os.path.join(os.environ["HERMES_HOME"], "fake.json")))
        mode = plan.get("mode", "ok")
        if mode == "http":
            raise APIStatusError(plan["status"], plan.get("message", "no such model"))
        if mode == "conn":
            raise APIConnectionError("Connection error.")
        if mode == "not_configured":
            raise RuntimeError(
                "No LLM provider configured for task=None provider=auto. Run: hermes setup"
            )
        if mode == "leak":
            raise RuntimeError("provider rejected key " + os.environ["NVIDIA_API_KEY"])
        if mode == "crash":
            os._exit(3)
        if mode == "sleep":
            time.sleep(plan["seconds"])
        token = json.load(open(os.path.join(os.environ["HERMES_HOME"], "auth.json")))
        token = token["providers"]["openai-codex"]["tokens"]["access_token"]
        body = json.dumps({"model": model or _config_model(), "messages": messages,
                           "pid": os.getpid(), "asked_model": model}).encode()
        req = urllib.request.Request(plan["url"] + "/responses", data=body,
                                     headers={"Authorization": "Bearer " + token,
                                              "Content-Type": "application/json"})
        raw = urllib.request.urlopen(req, timeout=10).read().decode()
        text = ""
        for line in raw.splitlines():
            if line.startswith("data:") and "response.completed" in line:
                done = json.loads(line[5:])["response"]
                text = done["output"][0]["content"][0]["text"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    """
)
FAKE_ENV_LOADER = textwrap.dedent(
    """
    import os

    def load_hermes_dotenv():
        path = os.path.join(os.environ["HERMES_HOME"], ".env")
        if os.path.exists(path):
            for line in open(path):
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.strip().partition("=")
                    os.environ[k] = v
    """
)
FAKE_RUNTIME = textwrap.dedent(
    """
    def resolve_runtime_provider():
        return {"provider": "openai-codex", "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex", "api_key": "never-sent"}
    """
)


def build_fake_hermes(root, *, version="9.9.9-fake", call_llm_src=FAKE_CALL_LLM):
    """A Hermes checkout's import surface plus its launcher, under `root`."""
    tree = root / "hermes-agent"
    (tree / "agent").mkdir(parents=True)
    (tree / "hermes_cli").mkdir()
    (tree / "hermes").write_text("# entry point; the launcher execs it\n")
    (tree / "agent" / "__init__.py").write_text("")
    (tree / "agent" / "auxiliary_client.py").write_text(call_llm_src)
    (tree / "hermes_cli" / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (tree / "hermes_cli" / "env_loader.py").write_text(FAKE_ENV_LOADER)
    (tree / "hermes_cli" / "runtime_provider.py").write_text(FAKE_RUNTIME)
    bin_dir = root / "bin"
    bin_dir.mkdir()
    launcher = bin_dir / "hermes"
    launcher.write_text(
        f'#!/usr/bin/env bash\nunset PYTHONPATH\nexec "{sys.executable}" "{tree / "hermes"}" "$@"\n'
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    return tree, bin_dir


@pytest.fixture(autouse=True)
def _no_helper_left_running():
    yield
    hermes_bridge.close_all()


@pytest.fixture
def servers():
    made = []

    def make(server):
        made.append(server)
        return server

    yield make
    for s in made:
        s.close()


@pytest.fixture
def codex_desk(tmp_path, monkeypatch, servers):
    """A fake Hermes connected to ChatGPT sign-in, its 'Responses' endpoint a stub,
    opted in, with the launcher the only way to find it."""
    stub = servers(chat_server(models=("gpt-5.5",), api_key=CODEX_TOKEN))
    tree, bin_dir = build_fake_hermes(tmp_path)
    home = make_home(
        tmp_path, monkeypatch, config("openai-codex", "gpt-5.5"), auth=codex_auth(stub.url)
    )
    (home / "fake.json").write_text(json.dumps({"mode": "ok", "url": stub.url}))
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("HOME", str(tmp_path / "no-home"))
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    return {"stub": stub, "home": home, "tree": tree, "bin": bin_dir}


def plan(desk, **fields):
    (desk["home"] / "fake.json").write_text(json.dumps({"url": desk["stub"].url, **fields}))


SCHEMA = {"type": "object", "properties": {"tags": {"type": "array"}}}
MESSAGES = [
    {"role": "system", "content": "You tag papers."},
    {"role": "user", "content": "GNNs for molecules"},
]


def chat():
    return llm.default_chat_fn(MESSAGES, SCHEMA)


# --- finding Hermes' Python --------------------------------------------------------------------


def test_hermes_python_is_found_from_the_launcher_the_way_the_agent_starts(codex_desk):
    python, source = hermes_bridge.find_hermes_python()
    assert python == sys.executable and source == str(codex_desk["tree"])


def test_hermes_python_can_be_named_explicitly(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("HOME", str(tmp_path / "no-home"))
    monkeypatch.setenv("ATTEST_HERMES_PYTHON", "/opt/hermes/bin/python")
    monkeypatch.setenv("ATTEST_HERMES_SRC", "/opt/hermes")
    assert hermes_bridge.find_hermes_python() == ("/opt/hermes/bin/python", "/opt/hermes")


def test_hermes_python_falls_back_to_the_checkout_beside_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    venv_python = home / "hermes-agent" / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(sys.executable)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("HOME", str(tmp_path / "no-home"))
    python, source = hermes_bridge.find_hermes_python()
    assert python.endswith("venv/bin/python") and source == str(home / "hermes-agent")


def test_a_missing_hermes_runtime_says_what_to_set(tmp_path, monkeypatch):
    make_home(tmp_path, monkeypatch, config("openai-codex", "gpt-5.5"))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("HOME", str(tmp_path / "no-home"))
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    with pytest.raises(BackendNotConfigured) as exc:
        chat()
    assert "ATTEST_HERMES_PYTHON" in str(exc.value) and "LLM_BASE_URL" in str(exc.value)


# --- the call ----------------------------------------------------------------------------------


def test_a_chatgpt_signin_call_is_made_by_hermes_with_hermes_credentials(codex_desk):
    target = llm.chat_target()
    assert (target.kind, target.provider, target.model) == ("hermes", "openai-codex", "gpt-5.5")
    assert target.api_key == "" and target.base_url == ""
    assert chat() == {"content_type": "paper", "tags": ["graph-neural-networks"]}
    (req,) = codex_desk["stub"].posts("/responses")
    assert req["auth"] == f"Bearer {CODEX_TOKEN}", "Hermes' own token, read by Hermes"


def test_the_reply_contract_joins_the_first_system_message_and_replaces_nothing(codex_desk):
    """Hermes' Responses adapter keeps only the LAST system message as its
    instructions: a second one would silently replace the tagging prompt."""
    chat()
    messages = codex_desk["stub"].posts("/responses")[0]["body"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"].startswith("You tag papers.")
    assert "JSON Schema" in messages[0]["content"] and '"tags"' in messages[0]["content"]
    assert messages[1] == MESSAGES[1]


def test_a_conversation_without_a_system_message_gets_one(codex_desk):
    hermes_bridge.with_schema([{"role": "user", "content": "x"}], SCHEMA)
    out = hermes_bridge.with_schema([{"role": "user", "content": "x"}], SCHEMA)
    assert out[0]["role"] == "system" and out[1] == {"role": "user", "content": "x"}


def test_one_helper_serves_many_calls_and_a_changed_model_is_followed(codex_desk):
    chat()
    chat()
    first = {r["body"]["pid"] for r in codex_desk["stub"].posts("/responses")}
    assert len(first) == 1, "one helper process for the run, not one per call"
    (codex_desk["home"] / "config.yaml").write_text(config("openai-codex", "gpt-5.5-mini"))
    chat()
    last = codex_desk["stub"].posts("/responses")[-1]["body"]
    assert last["model"] == "gpt-5.5-mini" and last["pid"] not in first
    assert llm.chat_model() == "gpt-5.5-mini"


def test_a_new_key_in_dotenv_restarts_the_helper_but_hermes_rewriting_auth_does_not(codex_desk):
    chat()
    pid = codex_desk["stub"].posts("/responses")[0]["body"]["pid"]
    (codex_desk["home"] / "auth.json").write_text(codex_auth(codex_desk["stub"].url))  # touched
    chat()
    assert codex_desk["stub"].posts("/responses")[-1]["body"]["pid"] == pid
    (codex_desk["home"] / ".env").write_text("NVIDIA_API_KEY=anything\n")
    chat()
    assert codex_desk["stub"].posts("/responses")[-1]["body"]["pid"] != pid


def test_a_chat_model_override_is_passed_through_and_boilerplate_is_not(codex_desk, monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", llm.DEFAULT_CHAT_MODEL)  # .env.sample boilerplate
    chat()
    assert codex_desk["stub"].posts("/responses")[-1]["body"]["asked_model"] is None
    monkeypatch.setenv("CHAT_MODEL", "gpt-5.5-pro")
    chat()
    assert codex_desk["stub"].posts("/responses")[-1]["body"]["asked_model"] == "gpt-5.5-pro"


def test_close_all_stops_the_helper(codex_desk):
    chat()
    (bridge,) = hermes_bridge._bridges.values()
    proc = bridge._proc
    hermes_bridge.close_all()
    assert proc.poll() is not None


# --- what goes wrong, in words ------------------------------------------------------------------


def test_a_rejected_model_is_a_rejection_with_the_providers_reason(codex_desk):
    plan(codex_desk, mode="http", status=404, message="model 'gpt-9' is not available")
    with pytest.raises(BackendRejected) as exc:
        chat()
    assert exc.value.status == 404 and "HTTP 404" in str(exc.value)
    assert "model 'gpt-9' is not available" in str(exc.value)


def test_an_expired_signin_is_a_rejection_that_says_401(codex_desk):
    plan(codex_desk, mode="http", status=401, message="token_invalidated")
    with pytest.raises(BackendRejected, match="HTTP 401"):
        chat()


def test_an_unreachable_provider_is_unreachable_not_a_generic_error(codex_desk):
    plan(codex_desk, mode="conn")
    with pytest.raises(BackendUnreachable, match="cannot connect"):
        chat()


def test_hermes_with_no_provider_configured_says_to_connect_one(codex_desk):
    plan(codex_desk, mode="not_configured")
    with pytest.raises(BackendNotConfigured) as exc:
        chat()
    assert "connect one in AgentMarkit" in str(exc.value) and "sign in again" in str(exc.value)


def test_a_server_error_is_a_plain_failure_the_taggers_retry_handles(codex_desk):
    plan(codex_desk, mode="http", status=503, message="overloaded")
    with pytest.raises(RuntimeError, match="HTTP 503"):
        chat()


def test_a_secret_in_hermes_own_error_text_is_scrubbed(codex_desk):
    (codex_desk["home"] / ".env").write_text(f"NVIDIA_API_KEY={FAKE_KEY}\n")
    plan(codex_desk, mode="leak")
    with pytest.raises(RuntimeError) as exc:
        chat()
    assert FAKE_KEY not in str(exc.value) and "***" in str(exc.value)


def test_a_helper_that_dies_is_reported_and_the_next_call_starts_a_fresh_one(codex_desk):
    plan(codex_desk, mode="crash")
    with pytest.raises(BackendUnreachable, match="exited"):
        chat()
    plan(codex_desk, mode="ok")
    assert chat()["tags"] == ["graph-neural-networks"]


def test_a_helper_that_does_not_answer_is_killed_and_named(codex_desk, monkeypatch):
    monkeypatch.setattr(hermes_bridge, "GRACE", 0.2)
    plan(codex_desk, mode="sleep", seconds=5)
    client = hermes_bridge.HermesChatClient(timeout=0.3)
    with pytest.raises(BackendUnreachable, match="did not answer"):
        client.chat_json(MESSAGES, SCHEMA)


def test_a_hermes_whose_call_llm_changed_is_reported_with_its_version(tmp_path, monkeypatch):
    tree, bin_dir = build_fake_hermes(
        tmp_path,
        version="1.2.3",
        call_llm_src="def call_llm(prompt):\n    return prompt\n",
    )
    make_home(tmp_path, monkeypatch, config("openai-codex", "gpt-5.5"))
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("HOME", str(tmp_path / "no-home"))
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    with pytest.raises(BackendNotConfigured) as exc:
        chat()
    assert "this Hermes (version" in str(exc.value) and "LLM_BASE_URL" in str(exc.value)


# --- end to end through the CLI ------------------------------------------------------------------


def test_attest_tag_runs_on_the_chatgpt_signin_model_and_never_touches_ollama(
    codex_desk, tmp_path, monkeypatch, capsys, caplog, servers
):
    embedder = servers(ollama_embedder_only())
    monkeypatch.setenv("ATTEST_DB", str(db_with_items(tmp_path)))
    monkeypatch.setenv("EMBED_BASE_URL", embedder.url)
    monkeypatch.setenv("EMBED_MODEL", "embeddinggemma")
    monkeypatch.setenv("LLM_BASE_URL", llm.DEFAULT_BASE_URL)  # .env.sample boilerplate
    monkeypatch.setenv("CHAT_MODEL", llm.DEFAULT_CHAT_MODEL)

    import httpx

    sent = []
    real_send = httpx.Client.send
    monkeypatch.setattr(
        httpx.Client,
        "send",
        lambda self, r, **kw: (sent.append(r.url), real_send(self, r, **kw))[1],
    )

    rc, out, err, logs = run_attest(["tag"], capsys, caplog)

    assert rc == 0, err
    assert "'tagged': 3" in out and "'model': 'gpt-5.5'" in out
    assert len(codex_desk["stub"].posts("/responses")) == 3
    assert embedder.posts("/chat/completions") == [], "the embedder was never asked to chat"
    assert not [u for u in sent if u.port == 11434]
    for text in (out, err, logs):
        for secret in (CODEX_TOKEN, CODEX_REFRESH):
            assert secret not in text


def test_attest_tag_with_hermes_signed_out_says_so_and_stops_once(
    codex_desk, tmp_path, monkeypatch, capsys, caplog
):
    monkeypatch.setenv("ATTEST_DB", str(db_with_items(tmp_path)))
    plan(codex_desk, mode="not_configured")
    rc, out, err, _ = run_attest(["tag"], capsys, caplog)
    assert rc == 1 and "'chat_down': True" in out
    assert "connect one in AgentMarkit" in err and "is ollama running?" not in err
    assert codex_desk["stub"].requests == []


def test_attest_tag_with_a_retired_model_names_hermes_as_the_origin(
    codex_desk, tmp_path, monkeypatch, capsys, caplog
):
    monkeypatch.setenv("ATTEST_DB", str(db_with_items(tmp_path)))
    plan(codex_desk, mode="http", status=404, message="model 'gpt-5.5' not found")
    rc, _, err, _ = run_attest(["tag"], capsys, caplog)
    assert rc == 1
    assert "via Hermes (provider openai-codex" in err and "HTTP 404" in err
    assert "model 'gpt-5.5' not found" in err


def test_install_check_probes_the_hermes_model_through_the_bridge(codex_desk, monkeypatch, servers):
    from attestation import install

    embedder = servers(ollama_embedder_only())
    monkeypatch.setenv("EMBED_BASE_URL", embedder.url)
    monkeypatch.setenv("EMBED_MODEL", "embeddinggemma")
    monkeypatch.setenv("EMBED_DIMS", "256")
    backends = install.step_backends()
    assert backends.status is install.Status.OK
    assert "via Hermes' own runtime, provider openai-codex, model gpt-5.5" in backends.detail
    r = install._hosted_probe()
    assert r.status is install.Status.OK, r.detail
    assert "chat gpt-5.5 ok" in r.detail
    assert len(codex_desk["stub"].posts("/responses")) == 1
    assert len(embedder.posts("/embeddings")) == 1, "embeddings went to the embedder"


def test_install_check_reports_a_signed_out_hermes_as_broken(codex_desk, monkeypatch, servers):
    from attestation import install

    embedder = servers(ollama_embedder_only())
    monkeypatch.setenv("EMBED_BASE_URL", embedder.url)
    monkeypatch.setenv("EMBED_MODEL", "embeddinggemma")
    monkeypatch.setenv("EMBED_DIMS", "256")
    plan(codex_desk, mode="not_configured")
    r = install._hosted_probe()
    assert r.status is install.Status.BROKEN and "connect one in AgentMarkit" in r.detail


def test_embeddings_are_never_routed_through_hermes_for_a_delegated_provider(
    codex_desk, monkeypatch, servers
):
    embedder = servers(ollama_embedder_only())
    monkeypatch.setenv("EMBED_BASE_URL", embedder.url)
    monkeypatch.setenv("EMBED_MODEL", "embeddinggemma")
    assert llm.embed_base_url() == embedder.url and llm.embed_api_key() == ""
    llm.EmbeddingClient().embed("x")
    assert hermes_bridge._bridges == {}, "no Hermes helper was started for an embedding"
    assert codex_desk["stub"].requests == []


def test_the_default_client_is_a_hermes_client_for_a_delegated_provider(codex_desk):
    assert isinstance(llm.chat_client(), hermes_bridge.HermesChatClient)
    with pytest.raises(BackendNotConfigured, match="llm.chat_client"):
        llm.ChatClient()


def test_the_refresh_script_runs_where_hermes_launcher_lives():
    """The cron refresh exports ~/.local/bin (where Hermes' launcher is), which is
    how the bridge finds Hermes' Python outside the agent's environment."""
    from pathlib import Path

    from attestation import install

    script = install._refresh_script_content(Path("/srv/attestation"))
    assert "$HOME/.local/bin" in script
