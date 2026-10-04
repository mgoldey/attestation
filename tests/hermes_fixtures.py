"""Shared builders for the Hermes-resolution tests: fake HERMES_HOME trees shaped
like the ones AgentMarkit's connect flows write (agentmarkit-site/src/worker:
`hermes config set model.*`, keys appended to `.env`, `hermes auth add
openai-codex` writing auth.json), a tiny database, and one way to run `attest`."""

import json
import logging
from pathlib import Path

from attestation.cli import main
from attestation.db import get_db

FAKE_KEY = "nvapi-FAKE-SECRET-0123456789"
NIM_MODEL = "nvidia/nemotron-3-super-120b-a12b"
CODEX_TOKEN = "codex-ACCESS-FAKE-0123456789"
CODEX_REFRESH = "codex-REFRESH-FAKE-0123456789"


def config(provider: str, model: str, base_url: str = "", api_key: str = "", **extra) -> str:
    """config.yaml as `hermes config set model.*` leaves it."""
    lines = [f"model:\n  default: {model}\n  provider: {provider}\n"]
    if base_url or provider == "openai-codex":
        lines.append(f"  base_url: '{base_url}'\n")
    if api_key or provider == "openai-codex":
        lines.append(f"  api_key: '{api_key}'\n")
    lines += [f"  {k}: {v}\n" for k, v in extra.items()]
    return "# written by `hermes config set`\n" + "".join(lines)


def codex_auth(base_url: str = "") -> str:
    """auth.json after `hermes auth add openai-codex`: the singleton and the pool
    entry (the pool is what the auxiliary path reads; `base_url` redirects it)."""
    tokens = {"access_token": CODEX_TOKEN, "refresh_token": CODEX_REFRESH}
    entry = {
        "id": "c1",
        "label": "chatgpt",
        "auth_type": "oauth",
        "priority": 0,
        "source": "manual:device_code",
        **tokens,
        **({"base_url": base_url} if base_url else {}),
    }
    return json.dumps(
        {
            "version": 1,
            "providers": {
                "openai-codex": {"tokens": tokens, "last_refresh": "2026-10-01T00:00:00Z"}
            },
            "credential_pool": {"openai-codex": [entry]},
        }
    )


def make_home(
    tmp_path: Path, monkeypatch, cfg: str | None, env: str | None = None, auth: str | None = None
) -> Path:
    home = tmp_path / "hermes-home"
    home.mkdir(exist_ok=True)
    for name, text in (("config.yaml", cfg), (".env", env), ("auth.json", auth)):
        if text is not None:
            (home / name).write_text(text)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def db_with_items(tmp_path: Path, n: int = 3) -> Path:
    path = tmp_path / "t.db"
    conn = get_db(path)
    conn.execute("INSERT INTO feeds(id, url, title) VALUES (1, 'http://f', 'F')")
    for i in range(n):
        conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (1, ?, ?, 'message passing on molecules', ?)",
            (f"Graph neural networks {i}", f"http://x/{i}", f"h{i}"),
        )
    conn.commit()
    conn.close()
    return path


def run_attest(argv, capsys, caplog):
    """(exit code, stdout, stderr, log text) of one `attest` invocation."""
    caplog.set_level(logging.INFO)
    rc = main(argv)
    out = capsys.readouterr()
    return rc, out.out, out.err, caplog.text


ANTHROPIC_KEY = "sk-ant-api03-FAKE-SECRET-0123456789"


def anthropic_auth(base_url: str) -> str:
    """auth.json holding one Anthropic API-key pool entry; `base_url` redirects it
    (Hermes only honours config.yaml's base_url for api.anthropic.com)."""
    entry = {
        "id": "a1",
        "label": "key",
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": ANTHROPIC_KEY,
        "base_url": base_url,
    }
    return json.dumps({"version": 1, "providers": {}, "credential_pool": {"anthropic": [entry]}})
