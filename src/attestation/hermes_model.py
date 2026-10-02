"""The chat model Hermes is connected to, read from the agent's own files.

Attestation's chat default is Ollama on loopback. A hosted Research Desk has no
chat model there -- the customer connects one to Hermes -- so tagging,
explanations and reactions asked an embedder-only server for a model it never
pulled. When the host opts in (`ATTEST_LLM_FROM_HERMES=1`, see
`llm.chat_target`) the chat backend is whatever Hermes is running, whichever
provider that is. This module answers the first half of that: WHAT is Hermes
connected to, and WHO talks to it.

Where Hermes keeps it (read from its source, hermes_cli/runtime_provider.py and
hermes_cli/auth.py; AgentMarkit's connect flows in agentmarkit-site/src/worker
write exactly these):

- `<HERMES_HOME>/config.yaml`: `model.default` (the model; `model.model` is an
  alias), `model.provider`, `model.base_url`, `model.api_key` (a literal, or a
  `${VAR}` reference into `.env` -- what a custom endpoint is written with) and
  optionally `model.api_mode`. `model:` may also be a bare string.
- `<HERMES_HOME>/.env`: the provider's key under its own variable
  (`NVIDIA_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `CUSTOM_API_KEY`...)
  and an optional `<PROVIDER>_BASE_URL`.
- `<HERMES_HOME>/auth.json`: OAuth credentials (ChatGPT sign-in is provider
  `openai-codex`, with an EMPTY base_url and api_key in config).

Two kinds of backend come out of `resolve`:

- `kind="openai"`: the provider speaks OpenAI chat-completions with a static
  key (nvidia, openrouter, openai, custom, Ollama and llama.cpp as `custom`,
  lmstudio, deepseek, ...). Attestation calls it itself, with the URL and key
  read here. Fast, no subprocess, nothing of Hermes' code involved.
- `kind="hermes"`: everything else -- OAuth providers, Anthropic's native
  messages API, Bedrock, Vertex, a provider this table has never heard of. The
  credential, the transport and the token refresh are Hermes' own, so Attestation
  asks Hermes to make the call (`hermes_bridge`) instead of re-implementing it.
  Nothing for such a provider ever enters Attestation's process: not the key,
  not an OAuth token.

Properties, each tested:

1. **Files, not the process environment.** The hourly refresh runs under cron,
   outside Hermes' environment; nothing here needs a variable the agent
   exported. The process environment is the last resort for a key only.
2. **The key is never copied or shown.** It reaches one Authorization header; it
   is not in any message, `repr`, log or exception (a YAML error reports a
   position, not the line, because the line may be the key).
3. **Read at call time.** Nothing is cached: "Change model" rewrites
   `model.default` and the next call follows.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from attestation import paths
from attestation.ports import BackendNotConfigured

CONNECT_HINT = "connect one in AgentMarkit"

# provider id -> (default base URL, key variables in Hermes' lookup order,
# base-URL override variable, key required). From hermes_cli/auth.py
# PROVIDER_REGISTRY, restricted to API-key providers whose endpoint is an
# OpenAI-compatible /v1 server. A provider absent here is served by Hermes' own
# runtime (`kind="hermes"`), never guessed at.
_PROVIDERS: dict[str, tuple[str, tuple[str, ...], str, bool]] = {
    "nvidia": (
        "https://integrate.api.nvidia.com/v1",
        ("NVIDIA_API_KEY",),
        "NVIDIA_BASE_URL",
        True,
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1",
        ("OPENROUTER_API_KEY",),
        "OPENROUTER_BASE_URL",
        True,
    ),
    "openai": ("https://api.openai.com/v1", ("OPENAI_API_KEY",), "OPENAI_BASE_URL", True),
    "deepseek": (
        "https://api.deepseek.com/v1",
        ("DEEPSEEK_API_KEY",),
        "DEEPSEEK_BASE_URL",
        True,
    ),
    "xai": ("https://api.x.ai/v1", ("XAI_API_KEY",), "XAI_BASE_URL", True),
    "ai-gateway": (
        "https://ai-gateway.vercel.sh/v1",
        ("AI_GATEWAY_API_KEY",),
        "AI_GATEWAY_BASE_URL",
        True,
    ),
    "huggingface": (
        "https://router.huggingface.co/v1",
        ("HF_TOKEN",),
        "HF_BASE_URL",
        True,
    ),
    "gmi": ("https://api.gmi-serving.com/v1", ("GMI_API_KEY",), "GMI_BASE_URL", True),
    "arcee": ("https://api.arcee.ai/api/v1", ("ARCEEAI_API_KEY",), "ARCEE_BASE_URL", True),
    "lmstudio": ("http://127.0.0.1:1234/v1", ("LM_API_KEY",), "LM_BASE_URL", False),
    "custom": ("", ("CUSTOM_API_KEY", "OPENAI_API_KEY"), "CUSTOM_BASE_URL", False),
}
_ALIASES = {
    "nim": "nvidia",
    "nvidia-nim": "nvidia",
    "build-nvidia": "nvidia",
    "nemotron": "nvidia",
    "openai-api": "openai",
}
_VAR_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


@dataclass(frozen=True)
class HermesModel:
    """What Hermes is connected to, and which side makes the call.

    `kind` is "openai" (Attestation calls `base_url` with `api_key`) or
    "hermes" (Hermes' runtime serves it; `base_url`/`api_key` are empty and
    `why` says why Attestation does not call it directly). `api_key` is
    excluded from `repr` so a model that gets formatted into a log line or an
    assertion message never carries the key with it.
    """

    provider: str
    model: str
    kind: str = "openai"
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    config_path: Path | None = None
    why: str = ""


def _unavailable(detail: str) -> BackendNotConfigured:
    return BackendNotConfigured(f"Hermes has no model connected; {CONNECT_HINT} ({detail})")


def _read_config(path: Path) -> dict:
    if not path.is_file():
        raise _unavailable(f"{path} does not exist")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        # NOT str(exc): a MarkedYAMLError renders the offending line, which in
        # a config that holds `api_key:` can be the key itself.
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise BackendNotConfigured(
            f"Hermes' config {path} is not valid YAML{where}; fix it or {CONNECT_HINT}"
        ) from None
    except (OSError, UnicodeDecodeError) as exc:
        raise _unavailable(f"cannot read {path}: {type(exc).__name__}") from None
    return loaded if isinstance(loaded, dict) else {}


def _env_values(home: Path) -> dict[str, str]:
    """`<home>/.env` as a dict, without touching `os.environ` (dotenv_values)."""
    from dotenv import dotenv_values

    env_file = home / ".env"
    if not env_file.is_file():
        return {}
    try:
        return {k: v for k, v in dotenv_values(env_file).items() if v}
    except OSError:
        return {}


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _lookup(name: str, dotenv: dict[str, str]) -> str:
    return _text(dotenv.get(name)) or _text(os.environ.get(name))


def _section(cfg: dict, config_path: Path) -> dict:
    """The `model:` mapping, normalised: model, provider, base_url, api_key, api_mode."""
    section = cfg.get("model")
    if isinstance(section, str):
        section = {"default": section}
    if not isinstance(section, dict):
        raise _unavailable(f"{config_path} has no `model:` section")
    model = _text(section.get("default")) or _text(section.get("model"))
    if not model:
        raise _unavailable(f"{config_path} sets no model.default")
    provider = _text(section.get("provider")).lower()
    base_url = _text(section.get("base_url")).rstrip("/")
    if provider in ("", "auto"):
        provider = "custom" if base_url else ""
    if not provider:
        raise _unavailable(f"{config_path} sets no model.provider")
    return {
        "model": model,
        "provider": _ALIASES.get(provider, provider),
        "base_url": base_url,
        "api_key": _text(section.get("api_key")),
        "api_mode": _text(section.get("api_mode")).lower(),
    }


def _delegated_reason(sec: dict) -> str:
    """Why Hermes, not Attestation, should make the call -- or "" when
    Attestation can: a provider in `_PROVIDERS` speaking chat-completions."""
    provider, url = sec["provider"], sec["base_url"].lower()
    if sec["api_mode"] and sec["api_mode"] != "chat_completions":
        return f"model.api_mode is {sec['api_mode']}"
    if provider not in _PROVIDERS:
        return f"provider {provider!r} is not an API-key chat-completions endpoint"
    if provider == "custom" and ("anthropic" in url or "bedrock" in url):
        return f"{url} is not a chat-completions endpoint"
    return ""


def _api_key(sec: dict, key_vars: tuple[str, ...], dotenv: dict[str, str]) -> str:
    key = sec["api_key"]
    ref = _VAR_REF.match(key)
    if ref:  # `api_key: ${CUSTOM_API_KEY}`
        key = _lookup(ref.group(1), dotenv)
    for name in () if key else key_vars:
        key = _lookup(name, dotenv)
        if key:
            break
    return key


def resolve(home: Path | None = None) -> HermesModel:
    """What Hermes is connected to, and which side calls it.

    Raises `BackendNotConfigured` with one sentence a person can act on when
    there is nothing usable -- no config, no model, a provider whose key is
    missing -- and never a traceback or the key. A `kind="hermes"` result
    is not checked here (that needs Hermes' runtime; `hermes_bridge` does it
    and says precisely what is wrong).
    """
    home = home or paths.hermes_home()
    config_path = home / "config.yaml"
    sec = _section(_read_config(config_path), config_path)
    provider, model = sec["provider"], sec["model"]
    why = _delegated_reason(sec)
    if why:
        return HermesModel(
            provider=provider,
            model=model,
            kind="hermes",
            base_url=sec["base_url"],
            config_path=config_path,
            why=why,
        )
    default_url, key_vars, url_var, key_required = _PROVIDERS[provider]
    dotenv = _env_values(home)
    url = (
        sec["base_url"] or (_lookup(url_var, dotenv).rstrip("/") if url_var else "") or default_url
    )
    if not url.startswith(("http://", "https://")):
        raise _unavailable(f"provider {provider!r} has no base_url; set model.base_url")
    key = _api_key(sec, key_vars, dotenv)
    if not key and key_required:
        raise BackendNotConfigured(
            f"Hermes has a model ({model}, provider {provider}) but no API key: set"
            f" {' or '.join(key_vars)} in {home / '.env'}, or {CONNECT_HINT}"
        )
    return HermesModel(
        provider=provider,
        model=model,
        kind="openai",
        base_url=url,
        api_key=key,
        config_path=config_path,
    )
