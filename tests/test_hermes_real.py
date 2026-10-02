"""The bridge against the REAL Hermes, with stubs standing in for the services.

Opt-in: set `ATTEST_TEST_HERMES_CHECKOUT` to a Hermes checkout that has its own
`venv/` (e.g. `~/.hermes/hermes-agent`). Skipped everywhere else -- CI has no
Hermes -- which is why `test_hermes_bridge.py` drives a fake one for the
protocol. THIS file is what notices when a Hermes release changes
`agent.auxiliary_client.call_llm`, its Responses adapter, its Anthropic adapter
or its credential pool: run it before bumping the Hermes version Attestation is
used with.

Nothing here reaches a real service: HERMES_HOME and HOME are scratch
directories, the credentials are fakes, the credential pool entries carry the
stubs' base URLs, and every proxy variable points at a closed port so that a
Hermes that ignores them fails instead of connecting out.
"""

import json
import os
from pathlib import Path

import pytest
from hermes_fixtures import (
    ANTHROPIC_KEY,
    CODEX_REFRESH,
    CODEX_TOKEN,
    anthropic_auth,
    codex_auth,
    config,
    db_with_items,
    make_home,
    run_attest,
)
from stub_servers import StubServer

from attestation import hermes_bridge, llm

CHECKOUT = os.environ.get("ATTEST_TEST_HERMES_CHECKOUT", "")
pytestmark = pytest.mark.skipif(
    not (CHECKOUT and (Path(CHECKOUT) / "venv" / "bin" / "python").is_file()),
    reason="needs ATTEST_TEST_HERMES_CHECKOUT=<hermes checkout with venv/>",
)


@pytest.fixture(autouse=True)
def _real_hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("ATTEST_HERMES_PYTHON", str(Path(CHECKOUT) / "venv" / "bin" / "python"))
    monkeypatch.setenv("ATTEST_HERMES_SRC", CHECKOUT)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("ATTEST_LLM_FROM_HERMES", "1")
    yield
    hermes_bridge.close_all()


@pytest.fixture
def stub():
    s = StubServer(chat_models=("m",))
    yield s
    s.close()


def ask():
    return llm.default_chat_fn(
        [{"role": "system", "content": "You tag."}, {"role": "user", "content": "GNNs"}],
        {"type": "object"},
    )


def test_chatgpt_signin_through_hermes_own_responses_adapter(tmp_path, monkeypatch, stub):
    home = make_home(
        tmp_path, monkeypatch, config("openai-codex", "gpt-5.5"), auth=codex_auth(stub.url)
    )
    assert ask() == {"content_type": "paper", "tags": ["graph-neural-networks"]}
    (req,) = stub.posts("/responses")
    assert req["auth"] == f"Bearer {CODEX_TOKEN}" and req["model"] == "gpt-5.5"
    assert "JSON Schema" in req["body"]["instructions"], "the contract reached Hermes' instructions"
    (home / "config.yaml").write_text(config("openai-codex", "gpt-5.5-mini"))
    ask()
    assert stub.posts("/responses")[-1]["model"] == "gpt-5.5-mini", "Change model is followed"


def test_anthropic_native_through_hermes_own_messages_adapter(tmp_path, monkeypatch, stub):
    make_home(
        tmp_path,
        monkeypatch,
        config("anthropic", "claude-sonnet-4-6"),
        env=f"ANTHROPIC_API_KEY={ANTHROPIC_KEY}\n",
        auth=anthropic_auth(stub.url.removesuffix("/v1")),
    )
    assert ask()["tags"] == ["graph-neural-networks"]
    (req,) = stub.posts("/messages")
    assert req["x_api_key"] == ANTHROPIC_KEY and req["model"] == "claude-sonnet-4-6"
    assert "JSON Schema" in req["body"]["system"]


def test_hermes_signed_out_is_reported_as_not_connected(tmp_path, monkeypatch):
    make_home(tmp_path, monkeypatch, config("openai-codex", "gpt-5.5"), auth="{}")
    with pytest.raises(Exception) as exc:  # the exact class is Hermes' to choose
        ask()
    assert "connect one in AgentMarkit" in str(exc.value) or "sign" in str(exc.value).lower()


def test_attest_tag_end_to_end_on_the_signin_model_with_no_secret_in_any_output(
    tmp_path, monkeypatch, capsys, caplog, stub
):
    make_home(tmp_path, monkeypatch, config("openai-codex", "gpt-5.5"), auth=codex_auth(stub.url))
    monkeypatch.setenv("ATTEST_DB", str(db_with_items(tmp_path)))
    rc, out, err, logs = run_attest(["tag"], capsys, caplog)
    assert rc == 0, err
    assert "'tagged': 3" in out
    for text in (out, err, logs, json.dumps(os.environ.get("ATTEST_HERMES_SRC"))):
        assert CODEX_TOKEN not in text and CODEX_REFRESH not in text
