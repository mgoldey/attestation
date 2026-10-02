"""Chat on the model Hermes runs, by asking Hermes to make the call.

For a provider Attestation cannot call itself (ChatGPT sign-in is OAuth and the
Responses API; Anthropic is its own messages API; Bedrock and Vertex sign
requests) the call is delegated to Hermes' own code. A small helper
(`hermes_helper.py`) runs in Hermes' Python, imports its `call_llm` -- the one
function Hermes uses for every non-agent model call, which resolves the main
provider, finds the credential, refreshes an expiring OAuth token under Hermes'
own lock and adapts the wire format -- and answers one JSON line per request.

Why delegate rather than re-implement (the alternatives are written up in
docs/guides/install.md):

- OAuth refresh tokens are single-use. A second process refreshing the same
  ChatGPT credential logs the first one out, so Attestation must never hold or
  refresh Hermes' tokens. Through the helper it never even sees them.
- The adapters are ~1,500 lines of fast-moving Hermes code (Responses SSE,
  Anthropic message conversion, per-provider headers). Copying them would be
  stale within a release.

What it costs, stated plainly: a subprocess per run (not per call -- the helper
stays up, measured ~3.5 s to the first reply and ~10 ms of overhead after), a
dependency on `agent.auxiliary_client.call_llm` (guarded: the helper checks the
signature it needs and a mismatch is reported as `unsupported_hermes` with
Hermes' version, never a traceback), and no `response_format` -- the schema goes
in the prompt and `llm._first_json_object` recovers the object, as it already
does for servers that do not honour strict decoding.

The helper is restarted whenever `config.yaml` or `.env` changes,
so "Change model" is followed on the next call. `auth.json` is deliberately not
watched: Hermes rewrites it after every call (pool bookkeeping), and its OAuth refresh and
re-sign-in are handled in Hermes' own process.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import re
import shutil
import subprocess
import threading
from pathlib import Path

from attestation import paths
from attestation.ports import BackendNotConfigured, BackendRejected, BackendUnreachable

HELPER = Path(__file__).with_name("hermes_helper.py")
STARTUP_TIMEOUT = 90.0  # first import of Hermes' provider stack measured ~3.5 s
GRACE = 5.0  # seconds the helper gets beyond a request's own timeout to answer
_WATCHED = ("config.yaml", ".env")
_LAUNCHER_EXEC = re.compile(r'^exec\s+"?([^"\s]+)"?\s+"?([^"\s]+)"?', re.MULTILINE)


class HermesRuntimeNotFound(BackendNotConfigured):
    """Hermes' Python could not be located."""


def _launcher_candidates() -> list[Path]:
    found = shutil.which("hermes")
    out = [Path(found)] if found else []
    out.append(Path.home() / ".local" / "bin" / "hermes")
    return out


def _from_launcher(path: Path) -> tuple[str, str | None] | None:
    """(python, source root) from a `hermes` launcher shell script: the line
    `exec "<venv>/bin/python" "<checkout>/hermes" "$@"` says both."""
    try:
        if not path.is_file() or path.stat().st_size > 8192:
            return None
        text = path.read_text(errors="replace")
    except OSError:
        return None
    match = _LAUNCHER_EXEC.search(text) if text.startswith("#!") else None
    if not match or not os.access(match.group(1), os.X_OK):
        return None
    return match.group(1), str(Path(match.group(2)).parent)


def find_hermes_python() -> tuple[str, str | None]:
    """(Python interpreter, Hermes source root) of the installed Hermes.

    In order: `ATTEST_HERMES_PYTHON` (+ `ATTEST_HERMES_SRC`), the `hermes`
    launcher on PATH or in ~/.local/bin (the line that execs its venv), then
    `<HERMES_HOME>/hermes-agent/venv`. The cron refresh gets ~/.local/bin on its
    PATH from the script `attest install` writes.
    """
    explicit = (os.environ.get("ATTEST_HERMES_PYTHON") or "").strip()
    if explicit:
        return explicit, (os.environ.get("ATTEST_HERMES_SRC") or "").strip() or None
    for launcher in _launcher_candidates():
        found = _from_launcher(launcher)
        if found:
            return found
    checkout = paths.hermes_home() / "hermes-agent"
    python = checkout / "venv" / "bin" / "python"
    if python.is_file():
        return str(python), str(checkout)
    raise HermesRuntimeNotFound(
        "Hermes is connected to a model only its own runtime can call, and Attestation"
        " could not find Hermes' Python (no `hermes` launcher on PATH or in ~/.local/bin,"
        f" no {python}); set ATTEST_HERMES_PYTHON to it, or set LLM_BASE_URL, CHAT_MODEL"
        " and LLM_API_KEY to an OpenAI-compatible endpoint instead"
    )


def _signature(home: Path) -> tuple:
    out = []
    for name in _WATCHED:
        try:
            st = (home / name).stat()
            out.append((name, st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((name, None, None))
    return tuple(out)


class HermesBridge:
    """One helper process serving chat requests; restarted when Hermes' files change."""

    def __init__(self, python: str, source: str | None, home: Path):
        self.python, self.source, self.home = python, source, home
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._signature: tuple = ()
        self._next_id = 0
        self._lock = threading.Lock()
        self.hermes_version = ""

    # -- process lifecycle ------------------------------------------------

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        for var in ("PYTHONPATH", "PYTHONHOME"):  # the hermes launcher unsets both
            env.pop(var, None)
        env["HERMES_HOME"] = str(self.home)
        env["PYTHONUNBUFFERED"] = "1"
        if self.source:
            env["ATTEST_HERMES_SRC"] = self.source
        return env

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                self._lines.put(line)
        except (ValueError, OSError):  # closed under us by close(): same as EOF
            pass
        self._lines.put(None)  # EOF: the helper exited

    def _read(self, timeout: float) -> dict:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise BackendUnreachable(
                f"Hermes' runtime did not answer within {timeout:.0f}s"
            ) from None
        if line is None:
            code = self._proc.poll() if self._proc else None
            self.close()
            raise BackendUnreachable(f"Hermes' runtime helper exited (code {code})")
        try:
            return json.loads(line)
        except ValueError:
            self.close()
            raise BackendUnreachable("Hermes' runtime helper sent something unreadable") from None

    def _start(self) -> None:
        self.close()
        self._lines = queue.Queue()
        # argv list, no shell; the interpreter is Hermes' own, found from its
        # launcher or named by the operator.
        self._proc = subprocess.Popen(
            [self.python, str(HELPER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Hermes' logging is not ours to relay: it can quote request URLs.
            stderr=subprocess.DEVNULL,
            text=True,
            env=self._env(),
            cwd=self.home if self.home.is_dir() else None,
        )
        threading.Thread(target=self._pump, args=(self._proc,), daemon=True).start()
        ready = self._read(STARTUP_TIMEOUT)
        if not ready.get("ready"):
            self.close()
            raise _error(ready.get("error") or {}, self.hermes_version)
        self.hermes_version = str(ready.get("hermes_version") or "")
        self._signature = _signature(self.home)

    def close(self) -> None:
        """Stop the helper process, if one is running. Safe to call twice."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        proc.kill()
        proc.wait()

    # -- requests ---------------------------------------------------------

    def request(self, payload: dict, timeout: float = 120.0) -> dict:
        """One request to the helper (starting or restarting it as needed); the
        reply dict, or the exception `_error` maps a reported failure to."""
        with self._lock:
            if (
                self._proc is None
                or self._proc.poll() is not None
                or self._signature != _signature(self.home)
            ):
                self._start()
            self._next_id += 1
            payload = {**payload, "id": self._next_id}
            assert self._proc is not None and self._proc.stdin is not None
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            except OSError:
                self.close()
                raise BackendUnreachable("Hermes' runtime helper is gone") from None
            reply = self._read(timeout + GRACE)
        if not reply.get("ok"):
            raise _error(reply.get("error") or {}, self.hermes_version)
        return reply


def _error(err: dict, version: str) -> Exception:
    """The exception a helper-reported failure maps to, worded for a person."""
    kind, status = err.get("kind"), int(err.get("status") or 0)
    message = str(err.get("message") or "unknown error")
    if kind == "unsupported_hermes":
        return BackendNotConfigured(
            f"this Hermes (version {version or 'unknown'}) does not offer what Attestation"
            f" needs from it ({message}); set LLM_BASE_URL, CHAT_MODEL and LLM_API_KEY instead"
        )
    if kind == "not_configured":
        return BackendNotConfigured(
            f"Hermes has no usable model connection ({message}); connect one in AgentMarkit"
            " or sign in again"
        )
    if kind == "http" and status in (401, 403, 404, 410):
        return BackendRejected(f"HTTP {status}: {message}", status)
    if kind == "http":
        return RuntimeError(f"HTTP {status}: {message}")
    if kind == "unreachable":
        return BackendUnreachable(f"cannot connect: {message}")
    return RuntimeError(message)


_bridges: dict[tuple, HermesBridge] = {}


def bridge() -> HermesBridge:
    """The shared bridge for this HERMES_HOME (one helper process per run)."""
    python, source = find_hermes_python()
    home = paths.hermes_home()
    key = (python, source, str(home))
    if key not in _bridges:
        _bridges[key] = HermesBridge(python, source, home)
    return _bridges[key]


def close_all() -> None:
    """Stop every helper this process started (also registered with atexit)."""
    for b in _bridges.values():
        b.close()
    _bridges.clear()


atexit.register(close_all)


def with_schema(messages: list[dict], schema: dict) -> list[dict]:
    """`messages` with the reply contract added to the FIRST system message.

    Not a second system message: Hermes' Responses adapter keeps only the last
    system message as its instructions, so appending one would silently replace
    the tagging prompt. No `response_format` either -- the adapters for
    Responses and Anthropic have none -- so the schema is stated in words.
    """
    contract = (
        "Reply with ONE JSON object and nothing else (no prose, no code fence) that"
        f" validates against this JSON Schema: {json.dumps(schema, separators=(',', ':'))}"
    )
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            m["content"] = f"{m.get('content', '')}\n\n{contract}".strip()
            return out
    return [{"role": "system", "content": contract}, *out]


class HermesChatClient:
    """`chat_json` through Hermes' runtime -- the `ChatClient` contract for a
    provider Attestation does not call itself."""

    def __init__(self, model: str | None = None, timeout: float = 120.0):
        self.model, self.timeout = model, timeout

    def chat_json(self, messages: list[dict], schema: dict) -> dict:
        """One chat call on Hermes' model, the reply parsed as the one JSON object
        it must be (the schema travels in the prompt; see `with_schema`)."""
        from attestation.llm import _first_json_object

        reply = bridge().request(
            {"op": "chat", "messages": with_schema(messages, schema), "model": self.model},
            timeout=self.timeout,
        )
        return _first_json_object(reply["content"])

    def describe(self) -> dict:
        """Hermes' own view of its runtime: provider, api_mode, host -- no key."""
        reply = bridge().request({"op": "describe"}, timeout=30)
        return {k: reply.get(k) for k in ("provider", "api_mode", "host")}
