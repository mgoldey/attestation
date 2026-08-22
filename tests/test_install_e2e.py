"""End-to-end demonstration of `attest install`: no `_run` mocking.

Builds a real sandbox per test: a fake home (`Path.home()` monkeypatched), a
PATH dir prepended with real stub executables (`hermes`, `ollama`) written
by the test, and a stdlib `http.server` standing in for the Ollama /v1
endpoint (reachability probe only — warmup's POST calls degrade gracefully
against it, same as any non-Ollama backend in production).

This is the "does it actually work" demonstration referenced in the design
doc's post-merge verification section, running hermetically in CI instead
of by hand on the maintainer's machine.
"""

import http.server
import os
import stat
import threading
from pathlib import Path

import attestation.install as install
from attestation.cli import main
from attestation.db import get_db

CHAT_MODEL = "hermes3:8b"
EMBED_MODEL = "embeddinggemma"

HERMES_STUB = r"""#!/usr/bin/env bash
# Stub hermes-agent CLI for e2e testing: logs every invocation and answers
# `mcp list` / `cron list` / `config get` from small state files so a second
# run sees what the first one registered.
set -u
STATE_DIR="{state_dir}"
LOG_FILE="${{STATE_DIR}}/log"
MCP_STATE="${{STATE_DIR}}/mcp_servers"
CRON_STATE="${{STATE_DIR}}/cron_jobs"
CONFIG_STATE="${{STATE_DIR}}/config.kv"
touch "$MCP_STATE" "$CRON_STATE" "$CONFIG_STATE"

echo "$@" >> "$LOG_FILE"

case "$1 $2" in
  "mcp list")
    cat "$MCP_STATE"
    exit 0
    ;;
  "mcp add")
    echo "$3" >> "$MCP_STATE"
    exit 0
    ;;
  "cron list")
    cat "$CRON_STATE"
    exit 0
    ;;
  "cron create")
    # --name is always the 5th arg per install.py's exact argv
    echo "$5" >> "$CRON_STATE"
    exit 0
    ;;
  "config get")
    line=$(grep "^$3=" "$CONFIG_STATE" || true)
    if [ -z "$line" ]; then
      exit 1
    fi
    echo "${{line#*=}}"
    exit 0
    ;;
  "config set")
    grep -v "^$3=" "$CONFIG_STATE" > "${{CONFIG_STATE}}.tmp" || true
    echo "$3=$4" >> "${{CONFIG_STATE}}.tmp"
    mv "${{CONFIG_STATE}}.tmp" "$CONFIG_STATE"
    exit 0
    ;;
  *)
    exit 0
    ;;
esac
"""

OLLAMA_STUB = r"""#!/usr/bin/env bash
# Stub ollama CLI: `list` prints the configured models as `ollama list` would.
set -u
if [ "$1" = "list" ]; then
  printf 'NAME\tID\tSIZE\n'
  printf '%s\tabc123\t1 GB\n' "{chat_model}"
  printf '%s\tdef456\t1 GB\n' "{embed_model}"
  exit 0
fi
exit 0
"""


class _OkGetHandler(http.server.BaseHTTPRequestHandler):
    """GET / -> 200 (Ollama native-root reachability probe).

    Anything else (warmup's POST /api/chat, /api/embed) -> 404, which makes
    warmup() degrade gracefully via its own httpx.HTTPError catch — the same
    path a non-Ollama backend takes in production.
    """

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass


def _start_fake_ollama_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _OkGetHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _seed_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "seed.db"
    conn = get_db(db_path)
    conn.execute(
        "INSERT INTO items(title, url, summary, content_hash) VALUES (?, ?, ?, ?)",
        ("Seed item", "http://x/1", "s", "hash1"),
    )
    conn.commit()
    conn.close()
    return db_path


def _read_log(state_dir: Path) -> list[str]:
    log_path = state_dir / "log"
    if not log_path.exists():
        return []
    return [line for line in log_path.read_text().splitlines() if line.strip()]


class _Sandbox:
    """One assembled e2e sandbox: fake home, stub PATH, hermetic env."""

    def __init__(self, tmp_path: Path, monkeypatch, base_url: str):
        self.tmp_path = tmp_path
        self.repo_root = tmp_path / "repo"
        self.fake_home = tmp_path / "fakehome"
        self.state_dir = tmp_path / "state"
        self.repo_root.mkdir()
        self.fake_home.mkdir()
        self.state_dir.mkdir()

        # marker that makes this a real checkout to install._checkout_root()
        (self.repo_root / "pyproject.toml").write_text('[project]\nname = "attestation"\n')
        (self.repo_root / ".env.sample").write_text(
            f"LLM_BASE_URL={base_url}\nCHAT_MODEL={CHAT_MODEL}\nEMBED_MODEL={EMBED_MODEL}\n"
        )
        # No skill fixture here on purpose: the skill ships inside the package
        # (src/attestation/skills/), so step_skill_copy reads the real files rather
        # than anything planted under this fake repo root.

        bin_dir = tmp_path / "stubbin"
        bin_dir.mkdir()
        _write_executable(bin_dir / "hermes", HERMES_STUB.format(state_dir=str(self.state_dir)))
        _write_executable(
            bin_dir / "ollama",
            OLLAMA_STUB.format(chat_model=CHAT_MODEL, embed_model=EMBED_MODEL),
        )
        self.bin_dir = bin_dir
        self.hermes_stub = bin_dir / "hermes"

        import attestation.llm as llm

        monkeypatch.setattr(llm, "_REPO_ROOT", self.repo_root)
        monkeypatch.setattr(install.Path, "home", lambda: self.fake_home)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("LLM_BASE_URL", base_url)
        monkeypatch.setenv("CHAT_MODEL", CHAT_MODEL)
        monkeypatch.setenv("EMBED_MODEL", EMBED_MODEL)
        monkeypatch.setenv("RSS_DB", str(_seed_db(tmp_path)))
        # sys.prefix-based venv skip must not exclude our stub dir.
        monkeypatch.setattr(install.sys, "prefix", str(tmp_path / "venv-not-used"))

    @property
    def skill_dest(self) -> Path:
        return self.fake_home / ".hermes" / "skills" / "research-provenance"

    @property
    def refresh_script(self) -> Path:
        return self.fake_home / ".hermes" / "scripts" / "attestation-refresh.sh"

    def log(self) -> list[str]:
        return _read_log(self.state_dir)


def _sandbox(tmp_path, monkeypatch):
    server = _start_fake_ollama_server()
    port = server.server_address[1]
    box = _Sandbox(tmp_path, monkeypatch, base_url=f"http://127.0.0.1:{port}/v1")
    return box, server


def test_fresh_install_demonstrates_end_to_end(tmp_path, monkeypatch):
    box, server = _sandbox(tmp_path, monkeypatch)
    try:
        rc = install.run_install(yes=True)

        assert rc == 0
        assert (box.repo_root / ".env").exists()
        assert (box.skill_dest / "SKILL.md").exists()
        assert (box.skill_dest / "scripts" / "setup.sh").exists()

        # Assert the end-to-end wiring, not a second transcription of the
        # script body (see test_install.py's note); the shipped content is
        # pinned by _refresh_script_content and exercised by executing it.
        script_text = box.refresh_script.read_text()
        assert script_text == install._refresh_script_content(box.repo_root)
        assert str(box.repo_root) in script_text
        assert "flock" in script_text, "a scheduled refresh must guard against overlap"
        assert os.access(box.refresh_script, os.X_OK)

        log = box.log()
        mcp_adds = [line for line in log if line.startswith("mcp add attestation")]
        cron_creates = [line for line in log if "cron create" in line and "--no-agent" in line]
        assert len(mcp_adds) == 1
        assert len(cron_creates) == 1
    finally:
        server.shutdown()


def test_idempotent_second_run_no_new_mutating_invocations(tmp_path, monkeypatch):
    box, server = _sandbox(tmp_path, monkeypatch)
    try:
        rc1 = install.run_install(yes=True)
        assert rc1 == 0
        log_after_first = box.log()

        rc2 = install.run_install(yes=True)
        assert rc2 == 0

        log_after_second = box.log()
        new_lines = log_after_second[len(log_after_first) :]
        mutating_verbs = ("add", "create", "set")
        for line in new_lines:
            args = line.split()
            assert not (set(mutating_verbs) & set(args)), f"mutating call on 2nd run: {line}"
    finally:
        server.shutdown()


def test_doctor_honesty_check_mode_on_wiped_home(tmp_path, monkeypatch):
    box, server = _sandbox(tmp_path, monkeypatch)
    try:
        rc = install.run_install(check=True)

        assert rc == 1
        log = box.log()
        mutating_verbs = ("add", "create", "set")
        for line in log:
            args = line.split()
            assert not (set(mutating_verbs) & set(args)), f"--check mutated: {line}"
        assert not box.skill_dest.exists()
        assert not box.refresh_script.exists()
        assert not (box.repo_root / ".env").exists()
    finally:
        server.shutdown()


def test_cli_path_dispatches_through_real_parser(tmp_path, monkeypatch):
    box, server = _sandbox(tmp_path, monkeypatch)
    try:
        rc_direct = install.run_install(check=True)

        second_root = tmp_path / "second"
        second_root.mkdir()
        box2, server2 = _sandbox(second_root, monkeypatch)
        try:
            rc_cli = main(["install", "--check"])
        finally:
            server2.shutdown()

        assert rc_cli == rc_direct == 1
    finally:
        server.shutdown()


def test_planted_data_blob_survives_two_runs_byte_identical(tmp_path, monkeypatch):
    box, server = _sandbox(tmp_path, monkeypatch)
    try:
        data_dir = box.skill_dest / "data"
        data_dir.mkdir(parents=True)
        blob = b"\x00\x01sqlite-binary-blob\xff\xfe"
        (data_dir / "hermes.db").write_bytes(blob)

        install.run_install(yes=True)
        assert (data_dir / "hermes.db").read_bytes() == blob

        install.run_install(yes=True)
        assert (data_dir / "hermes.db").read_bytes() == blob
    finally:
        server.shutdown()
