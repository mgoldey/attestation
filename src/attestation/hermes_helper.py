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

_PARENT = os.getppid()  # who started us, captured before anything slow can run
_PROTOCOL = None  # the protocol stream, set by _init_protocol() -- not at import
_REQUIRED = {"messages", "model", "provider", "timeout", "extra_body"}
_SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


def _init_protocol() -> None:
    """Move the protocol to a private duplicate of fd 1 and point fd 1 at stderr.

    Whatever Hermes (or a subprocess it starts) writes to stdout -- fd 1 itself,
    not just sys.stdout -- then lands on stderr and cannot corrupt the protocol.
    Done in main(), not at import, so the module can be imported by tests."""
    global _PROTOCOL
    _PROTOCOL = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr


def send(obj: dict) -> None:
    """Write one protocol line (one atomic write, newline-terminated)."""
    assert _PROTOCOL is not None
    _PROTOCOL.write(json.dumps(obj) + "\n")
    _PROTOCOL.flush()


_SHAPES = re.compile(
    r"(?i)((?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"
    r"|(?:authorization|x-api-key|api[_-]?key|token|secret)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9._~+/=-]{8,}"
    r"|\b(?:sk|nvapi|xai|gsk|hf|pk|rk|ghp|gho)[-_][A-Za-z0-9._-]{12,}"
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,})"
)
# Fields of Hermes' auth.json and credential pool that are NOT secrets (read from
# hermes_cli/auth.py and agent/credential_pool.py: PooledCredential and _EXTRA_KEYS).
# Everything else that is a string of 16+ characters is treated as one, so a field
# this list has never heard of is redacted rather than echoed.
_NOT_SECRET = frozenset(
    {
        "id", "label", "auth_type", "source", "provider", "base_url", "inference_base_url",
        "portal_base_url", "client_id", "scope", "token_type", "tls", "failure_reason",
        "secret_source", "secret_fingerprint", "agent_key_id", "last_refresh", "obtained_at",
        "expires_at", "agent_key_expires_at", "agent_key_obtained_at", "last_status",
        "last_status_at", "last_error_reason", "last_error_message", "last_error_code",
        "version", "active_provider", "updated_at", "created_at",
    }
)  # fmt: skip
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]")


def _walk_secrets(node, out: set[str], key: str = "") -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            _walk_secrets(v, out, str(k).lower())
    elif isinstance(node, list):
        for v in node:
            _walk_secrets(v, out, key)
    elif (
        isinstance(node, str)
        and len(node) >= 16
        and key not in _NOT_SECRET
        and not node.startswith(("http://", "https://"))
        and not _TIMESTAMP.match(node)
    ):
        out.add(node)


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


def _variants(secret: str) -> list[str]:
    """The secret as it may appear in a message: plain, URL-encoded two ways."""
    from urllib.parse import quote, quote_plus

    return list({secret, quote(secret, safe=""), quote_plus(secret)})


def scrub(text: str, secrets: set[str]) -> str:
    """`text` with every known secret (plain, URL-encoded, or split across a line
    break) and every credential-shaped string replaced by `***`, whitespace
    collapsed, cut at 300."""
    for secret in sorted(secrets, key=len, reverse=True):
        for form in _variants(secret):
            text = text.replace(form, "***")
        text = re.sub(r"\s*".join(map(re.escape, secret)), "***", text)
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


def _expected_parent() -> int:
    """The pid that started us: the bridge passes its own (ATTEST_PARENT_PID) so a
    parent that died before this process even began is still noticed; else the
    parent we had at import."""
    try:
        return int(os.environ.get("ATTEST_PARENT_PID", ""))
    except ValueError:
        return _PARENT


WATCHDOG_INTERVAL = 1.0


def _exit_with_parent() -> None:
    """Exit when the process that started us is gone (it was killed, so no atexit
    ran): poll the parent pid. A helper stuck in a model call cannot see EOF on
    stdin, and one still importing Hermes has not started reading it, so this runs
    on its own thread from the very start of main()."""
    parent = _expected_parent()
    while True:
        if os.getppid() != parent:
            os._exit(0)
        time.sleep(WATCHDOG_INTERVAL)


def serve(call_llm) -> None:
    """Answer JSON-line requests on stdin until it closes."""
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
    _init_protocol()
    threading.Thread(target=_exit_with_parent, daemon=True).start()
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
