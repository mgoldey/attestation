"""Runs INSIDE Hermes' Python: one chat completion per request, on whatever
model Hermes is connected to. Started by `hermes_bridge`, never imported.

This file imports nothing from attestation (Hermes' interpreter does not have
it) and speaks JSON lines on stdin/stdout:

    -> {"id": 1, "op": "chat", "messages": [...], "model": null, "timeout": 120}
    <- {"id": 1, "ok": true, "content": "..."}
    <- {"id": 1, "ok": false, "error": {"kind": "...", "status": 404, "message": "..."}}
    -> {"id": 2, "op": "describe"}   <- {"ok": true, "provider": ..., "api_mode": ..., "host": ...}

The first line out is `{"ready": true, "hermes_version": ...}`, or
`{"ready": false, "error": ...}` when Hermes' own code is not what this
expects. Error kinds: `unsupported_hermes`, `not_configured` (no model, no
credentials, sign-in needed), `unreachable`, `http` (the provider answered with
a status), `empty`, `other`.

It reuses Hermes' own `agent.auxiliary_client.call_llm`, which resolves the main
provider exactly as the agent does -- API keys from `.env`, OAuth credentials
from `auth.json` with their locked refresh, the Responses and Anthropic
adapters -- and returns an OpenAI-shaped response. Nothing about the credential
crosses the pipe: replies carry text and a scrubbed error message, where every
secret-looking value Hermes loaded is replaced by `***`.
"""

import sys

# FIRST: running this file as a script puts its own directory (attestation/) at
# sys.path[0], where `import mcp` -- a package Hermes depends on -- would resolve
# to attestation/mcp. Nothing of Attestation's may shadow Hermes' modules.
if sys.path and sys.path[0] == __import__("os").path.dirname(
    __import__("os").path.abspath(__file__)
):
    sys.path.pop(0)

import importlib
import inspect
import json
import os
import re
import threading
import time

_PROTOCOL = sys.stdout
sys.stdout = sys.stderr  # whatever Hermes prints must not corrupt the protocol
_REQUIRED = {"messages", "model", "provider", "timeout", "extra_body"}
_SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


def send(obj: dict) -> None:
    """Write one protocol line to the REAL stdout (stdout itself is rerouted)."""
    _PROTOCOL.write(json.dumps(obj) + "\n")
    _PROTOCOL.flush()


_SHAPES = re.compile(
    r"(?i)(bearer\s+[A-Za-z0-9._~+/=-]{8,}"
    r"|(?:authorization|x-api-key|api[_-]?key|token|secret)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9._~+/=-]{8,}"
    r"|\b(?:sk|nvapi|xai|gsk|hf|pk|rk|ghp|gho)[-_][A-Za-z0-9._-]{12,}"
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,})"
)
_SECRET_NAMES = ("access_token", "refresh_token", "api_key", "agent_key", "id_token", "key")


def _walk_secrets(node, out: set[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str) and str(k).lower() in _SECRET_NAMES and len(v) >= 8:
                out.add(v)
            else:
                _walk_secrets(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk_secrets(v, out)


def secrets_in_env() -> set[str]:
    """Every secret this helper can know about: secret-looking environment values,
    every token in Hermes' auth.json (singleton and credential pool), and the key
    of the runtime Hermes resolves for its main provider."""
    found = {
        v for k, v in os.environ.items() if k.upper().endswith(_SECRET_SUFFIXES) and len(v) >= 8
    }
    try:
        with open(os.path.join(os.environ.get("HERMES_HOME", ""), "auth.json")) as handle:
            _walk_secrets(json.load(handle), found)
    except (OSError, ValueError):
        pass
    try:
        key = importlib.import_module("hermes_cli.runtime_provider").resolve_runtime_provider()
        if (
            isinstance(key, dict)
            and isinstance(key.get("api_key"), str)
            and len(key["api_key"]) >= 8
        ):
            found.add(key["api_key"])
    except Exception:  # noqa: BLE001 -- best effort: a failing resolver must not lose the error
        pass
    return found


def scrub(text: str, secrets: set[str]) -> str:
    """`text` with every known secret and every credential-shaped string replaced
    by `***`, whitespace collapsed, cut at 300."""
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "***")
    text = _SHAPES.sub("***", text)
    return " ".join(text.split())[:300]


def status_of(exc: BaseException) -> int:
    """The HTTP status an SDK exception carries (directly or on its response), else 0."""
    for holder in (exc, getattr(exc, "response", None)):
        code = getattr(holder, "status_code", None)
        if isinstance(code, int):
            return code
    return 0


def reason_of(exc: BaseException) -> str:
    """The provider's own reason when the SDK kept it, else the exception text."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        text = inner.get("message") if isinstance(inner, dict) else inner
        text = text or body.get("message") or body.get("detail")
        if text:
            return str(text)
    return str(getattr(exc, "message", None) or exc)


def _kind_of(name: str, lowered: str, status: int) -> str:
    """The protocol error kind for an exception's class name, text and status."""
    if status:
        return "http"
    if "connection" in name.lower() or "timeout" in name.lower():
        return "unreachable"
    if name == "AuthError" or any(
        hint in lowered
        for hint in ("hermes setup", "no llm provider", "login", "re-auth", "relogin")
    ):
        return "not_configured"
    return "other"


def classify(exc: BaseException, secrets: set[str]) -> dict:
    """An exception as a protocol error: kind, status, scrubbed message."""
    name = type(exc).__name__
    status = status_of(exc)
    # A rejected credential: the provider's own words are the likeliest place for
    # an echo of it, and nothing in them is actionable beyond the status.
    message = (
        "the provider rejected the credential"
        if status in (401, 403)
        else scrub(reason_of(exc), secrets)
    )
    kind = _kind_of(name, message.lower(), status)
    return {"kind": kind, "status": status, "message": f"{name}: {message}"}


def load_hermes():
    """(call_llm, hermes_version) or raises ImportError/TypeError naming what is off."""
    src = os.environ.get("ATTEST_HERMES_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)
    # importlib, not `import`: these modules exist only in Hermes' interpreter.
    importlib.import_module("hermes_cli.env_loader").load_hermes_dotenv()
    call_llm = importlib.import_module("agent.auxiliary_client").call_llm

    missing = _REQUIRED - set(inspect.signature(call_llm).parameters)
    if missing:
        raise TypeError(f"call_llm no longer takes {sorted(missing)}")
    version = getattr(importlib.import_module("hermes_cli"), "__version__", "unknown")
    return call_llm, str(version)


def chat(call_llm, request: dict) -> dict:
    """Run one chat request through Hermes' `call_llm` and return its text."""
    kwargs = {"messages": request["messages"], "timeout": request.get("timeout") or 120}
    if request.get("model"):
        kwargs["model"] = request["model"]
    response = call_llm(**kwargs)
    content = response.choices[0].message.content
    if isinstance(content, list):  # content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if not isinstance(content, str) or not content.strip():
        return {"ok": False, "error": {"kind": "empty", "status": 0, "message": "empty reply"}}
    return {"ok": True, "content": content}


def describe() -> dict:
    """Hermes' own resolution of its main runtime: provider, api_mode, host. No key."""
    from urllib.parse import urlsplit

    runtime = importlib.import_module("hermes_cli.runtime_provider").resolve_runtime_provider()
    host = urlsplit(str(runtime.get("base_url") or "")).hostname or ""
    return {
        "ok": True,
        "provider": runtime.get("provider"),
        "api_mode": runtime.get("api_mode"),
        "host": host,
    }


def _exit_with_parent() -> None:
    """Exit when the process that started us is gone (it was killed, so no atexit
    ran): poll the parent pid. A helper stuck in a model call cannot see EOF on
    stdin, so this runs on its own thread."""
    parent = os.getppid()
    while True:
        time.sleep(1.0)
        if os.getppid() != parent:
            os._exit(0)


def serve(call_llm) -> None:
    """Answer JSON-line requests on stdin until it closes."""
    threading.Thread(target=_exit_with_parent, daemon=True).start()
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            op = request.get("op")
            reply = describe() if op == "describe" else chat(call_llm, request)
        except Exception as exc:  # noqa: BLE001 -- every failure of Hermes' own
            # provider stack is reported to the caller as one scrubbed line;
            # none of them is a bug in this relay.
            reply = {"ok": False, "error": classify(exc, secrets_in_env())}
        reply["id"] = request.get("id")
        send(reply)


def main() -> int:
    """Load Hermes, announce readiness (or why not), then serve."""
    try:
        call_llm, version = load_hermes()
    except Exception as exc:  # noqa: BLE001 -- an import failure inside Hermes
        # is reported over the protocol, not as a traceback on a pipe.
        send({"ready": False, "error": classify(exc, set()) | {"kind": "unsupported_hermes"}})
        return 1
    send({"ready": True, "hermes_version": version})
    serve(call_llm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
