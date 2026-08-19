"""hermes install: step engine, detection steps, doctor mode.

Fake `_run` records every invocation and returns canned CompletedProcess
results, so tests assert exactly what subprocess calls (if any) a run makes.
`llm._REPO_ROOT` is monkeypatched to tmp_path by the autouse hermetic-env
fixture in conftest.py; tests that need `.env.sample` write it under tmp_path.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import attestation.install as install
import attestation.llm as llm


def _cp(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=args, returncode=returncode, stdout=stdout, stderr=stderr
    )


class FakeRun:
    """Records every _run call; returns a canned result keyed by the argv."""

    def __init__(self, responses=None, default=None):
        self.calls = []
        self.responses = responses or {}
        self.default = default if default is not None else _cp([], 0)

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        key = tuple(cmd)
        resp = self.responses.get(key, self.default)
        # callables let a stub model state: `X list` is empty until `X add`
        # runs, then reports the entry -- which is what the steps re-check.
        return resp(cmd) if callable(resp) else resp


def _write_env_sample(tmp_path: Path) -> None:
    (tmp_path / ".env.sample").write_text("CHAT_MODEL=hermes3:8b\nEMBED_MODEL=embeddinggemma\n")


def _ollama_list_ok(models: str = "hermes3:8b\nembeddinggemma\n"):
    return _cp(["ollama", "list"], 0, stdout=f"NAME\tID\tSIZE\n{models}")


def _write_skill_source(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "skills" / "research-provenance"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# research-provenance skill\n")
    (scripts_dir / "setup.sh").write_text("#!/usr/bin/env bash\necho setup\n")
    return skill_dir


def _sync_skill_dest(fake_home: Path, skill_dir: Path) -> Path:
    """Mirror the skill source into the fake ~/.hermes skill dir so the
    default fixture represents an already-synced (all-OK) state."""
    dest_dir = fake_home / ".hermes" / "skills" / "research-provenance"
    (dest_dir / "scripts").mkdir(parents=True)
    (dest_dir / "SKILL.md").write_text((skill_dir / "SKILL.md").read_text())
    (dest_dir / "scripts" / "setup.sh").write_text((skill_dir / "scripts" / "setup.sh").read_text())
    return dest_dir


@pytest.fixture(autouse=True)
def _default_env(monkeypatch, tmp_path):
    """Everything present + reachable so a bare run_install() is all-OK."""
    _write_env_sample(tmp_path)
    (tmp_path / ".env").write_text("CHAT_MODEL=hermes3:8b\n")
    monkeypatch.setenv("CHAT_MODEL", "hermes3:8b")
    monkeypatch.setenv("EMBED_MODEL", "embeddinggemma")
    monkeypatch.setattr(install.shutil, "which", lambda name: "/usr/bin/uv")
    skill_dir = _write_skill_source(tmp_path)
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    _sync_skill_dest(fake_home, skill_dir)
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)
    # No agent binary on PATH by default (real _find_agent_binary still runs);
    # agent-wiring tests put a stub executable on this empty PATH themselves.
    monkeypatch.setattr(install.os, "get_exec_path", lambda: [])


def _patch_run(monkeypatch, responses=None, default=None):
    fake = FakeRun(responses, default)
    monkeypatch.setattr(install, "_run", fake)
    return fake


def _db_with_items(tmp_path, n_items=1, n_untagged=0):
    from attestation.db import get_db

    db_path = tmp_path / "hermes.db"
    conn = get_db(db_path)
    for i in range(n_items):
        conn.execute(
            "INSERT INTO items(title, url, summary, content_hash) VALUES (?, ?, ?, ?)",
            (f"t{i}", f"http://x/{i}", "s", f"hash{i}"),
        )
    conn.commit()
    conn.close()
    return db_path


# --------------------------------------------------------------------------
# --check: all present -> exit 0, no mutating calls
# --------------------------------------------------------------------------


def _presync_skill(monkeypatch, tmp_path):
    """Point Path.home() at a tmp home with the skill already synced.

    The skill ships inside the package, so _skill_source_dir() always exists;
    a run that should report "everything present" needs the destination copy
    to match it.
    """
    fake_home = tmp_path / "synced-home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)
    install.step_skill_copy(check=False)
    return fake_home


def test_check_all_present_exits_zero_no_mutating_calls(monkeypatch, tmp_path, capsys):
    db_path = _db_with_items(tmp_path, n_items=1)
    monkeypatch.setenv("RSS_DB", str(db_path))
    fake = _patch_run(
        monkeypatch,
        responses={("ollama", "list"): _ollama_list_ok()},
    )
    monkeypatch.setattr(install, "_ollama_native_root_reachable", lambda: True)
    monkeypatch.setattr("attestation.cli.warmup", lambda: None)
    _presync_skill(monkeypatch, tmp_path)

    rc = install.run_install(check=True)

    assert rc == 0
    mutating = {"pull", "ingest", "tag"}
    for call in fake.calls:
        assert not (mutating & set(call)), f"check mode issued mutating call: {call}"


def test_models_present_when_ollama_reports_latest_tag(monkeypatch):
    """`ollama list` prints tag-qualified names: an untagged EMBED_MODEL of
    `embeddinggemma` shows up as `embeddinggemma:latest`. Comparing raw
    strings made a correctly-installed model look missing, so --check
    reported BROKEN forever and --yes re-pulled it on every run."""
    monkeypatch.setenv("CHAT_MODEL", "hermes3:8b")
    monkeypatch.setenv("EMBED_MODEL", "embeddinggemma")
    fake = _patch_run(
        monkeypatch,
        responses={
            ("ollama", "list"): _ollama_list_ok(models="embeddinggemma:latest\nhermes3:8b\n")
        },
    )

    result = install.step_models(check=True)

    assert result.status == "OK"
    assert not any("pull" in call for call in fake.calls)


def test_check_missing_model_is_broken_exit_1_no_pull(monkeypatch, tmp_path):
    db_path = _db_with_items(tmp_path, n_items=1)
    monkeypatch.setenv("RSS_DB", str(db_path))
    fake = _patch_run(
        monkeypatch,
        responses={("ollama", "list"): _ollama_list_ok(models="embeddinggemma\n")},
    )
    monkeypatch.setattr(install, "_ollama_native_root_reachable", lambda: True)
    monkeypatch.setattr("attestation.cli.warmup", lambda: None)

    rc = install.run_install(check=True)

    assert rc == 1
    assert not any("pull" in call for call in fake.calls)


# --------------------------------------------------------------------------
# full run with missing model + yes=True -> pull recorded, FIXED
# --------------------------------------------------------------------------


def test_full_run_missing_model_yes_pulls_and_fixes(monkeypatch, tmp_path, capsys):
    db_path = _db_with_items(tmp_path, n_items=1)
    monkeypatch.setenv("RSS_DB", str(db_path))

    call_count = {"n": 0}

    def list_response(cmd):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _ollama_list_ok(models="embeddinggemma\n")
        return _ollama_list_ok()  # after pull, model present

    fake = _patch_run(
        monkeypatch,
        responses={
            ("ollama", "list"): list_response,
            ("ollama", "pull", "hermes3:8b"): _cp(["ollama", "pull", "hermes3:8b"], 0),
        },
    )
    monkeypatch.setattr(install, "_ollama_native_root_reachable", lambda: True)
    monkeypatch.setattr("attestation.cli.warmup", lambda: None)

    rc = install.run_install(check=False, yes=True)

    pulls = [c for c in fake.calls if list(c) == ["ollama", "pull", "hermes3:8b"]]
    assert len(pulls) == 1
    out = capsys.readouterr().out.lower()
    assert "fixed" in out
    assert rc == 0


def test_full_run_missing_model_no_consent_declined_broken(monkeypatch, tmp_path):
    """Non-tty / no yes -> input() guarded off; declined -> BROKEN with hint, no pull."""
    db_path = _db_with_items(tmp_path, n_items=1)
    monkeypatch.setenv("RSS_DB", str(db_path))
    fake = _patch_run(
        monkeypatch,
        responses={("ollama", "list"): _ollama_list_ok(models="embeddinggemma\n")},
    )
    monkeypatch.setattr(install, "_ollama_native_root_reachable", lambda: True)
    monkeypatch.setattr("attestation.cli.warmup", lambda: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    rc = install.run_install(check=False, yes=False)

    assert rc == 1
    assert not any("pull" in call for call in fake.calls)


# --------------------------------------------------------------------------
# .env file creation
# --------------------------------------------------------------------------


def test_env_file_absent_created_from_sample(monkeypatch, tmp_path):
    (tmp_path / ".env").unlink()
    db_path = _db_with_items(tmp_path, n_items=1)
    monkeypatch.setenv("RSS_DB", str(db_path))
    _patch_run(monkeypatch, responses={("ollama", "list"): _ollama_list_ok()})
    monkeypatch.setattr(install, "_ollama_native_root_reachable", lambda: True)
    monkeypatch.setattr("attestation.cli.warmup", lambda: None)

    rc = install.run_install(check=False, yes=True)

    env_path = tmp_path / ".env"
    assert env_path.exists()
    assert env_path.read_text() == (tmp_path / ".env.sample").read_text()
    assert rc == 0


def test_env_file_present_untouched(monkeypatch, tmp_path):
    db_path = _db_with_items(tmp_path, n_items=1)
    monkeypatch.setenv("RSS_DB", str(db_path))
    env_path = tmp_path / ".env"
    original_content = env_path.read_text()
    original_mtime = env_path.stat().st_mtime_ns
    _patch_run(monkeypatch, responses={("ollama", "list"): _ollama_list_ok()})
    monkeypatch.setattr(install, "_ollama_native_root_reachable", lambda: True)
    monkeypatch.setattr("attestation.cli.warmup", lambda: None)

    rc = install.run_install(check=False, yes=True)

    assert env_path.read_text() == original_content
    assert env_path.stat().st_mtime_ns == original_mtime
    assert rc == 0


# --------------------------------------------------------------------------
# idempotency: second full run after fixes -> all OK, zero mutating calls
# --------------------------------------------------------------------------


def test_second_full_run_after_fixes_is_idempotent(monkeypatch, tmp_path, capsys):
    db_path = _db_with_items(tmp_path, n_items=1)
    monkeypatch.setenv("RSS_DB", str(db_path))
    _patch_run(monkeypatch, responses={("ollama", "list"): _ollama_list_ok()})
    monkeypatch.setattr(install, "_ollama_native_root_reachable", lambda: True)
    monkeypatch.setattr("attestation.cli.warmup", lambda: None)
    # run 1 syncs the packaged skill into a tmp home, not the real ~/.hermes
    fake_home = tmp_path / "idempotent-home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)

    rc1 = install.run_install(check=False, yes=True)
    assert rc1 == 0
    capsys.readouterr()  # discard run 1; the assertions below are about run 2

    fake2 = _patch_run(monkeypatch, responses={("ollama", "list"): _ollama_list_ok()})
    rc2 = install.run_install(check=False, yes=True)

    assert rc2 == 0
    mutating = {"pull", "ingest", "tag"}
    for call in fake2.calls:
        assert not (mutating & set(call)), f"idempotent re-run issued mutating call: {call}"
    out = capsys.readouterr().out.lower()
    assert "broken" not in out
    assert "fixed" not in out


# --------------------------------------------------------------------------
# _find_agent_binary
# --------------------------------------------------------------------------


def test_find_agent_binary_skips_venv_finds_other(monkeypatch, tmp_path):
    venv_dir = tmp_path / "venv" / "bin"
    venv_dir.mkdir(parents=True)
    venv_hermes = venv_dir / "hermes"
    venv_hermes.write_text("#!/bin/sh\n")
    venv_hermes.chmod(0o755)

    other_dir = tmp_path / "otherbin"
    other_dir.mkdir()
    other_hermes = other_dir / "hermes"
    other_hermes.write_text("#!/bin/sh\n")
    other_hermes.chmod(0o755)

    monkeypatch.setattr(install.os, "get_exec_path", lambda: [str(venv_dir), str(other_dir)])
    monkeypatch.setattr(install.sys, "prefix", str(tmp_path / "venv"))

    found = install._find_agent_binary()

    assert found == str(other_hermes)


def test_find_agent_binary_none_when_only_venv_and_no_fallback(monkeypatch, tmp_path):
    venv_dir = tmp_path / "venv" / "bin"
    venv_dir.mkdir(parents=True)
    venv_hermes = venv_dir / "hermes"
    venv_hermes.write_text("#!/bin/sh\n")
    venv_hermes.chmod(0o755)

    monkeypatch.setattr(install.os, "get_exec_path", lambda: [str(venv_dir)])
    monkeypatch.setattr(install.sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(install.Path, "home", lambda: tmp_path / "no-such-home")

    assert install._find_agent_binary() is None


# --------------------------------------------------------------------------
# non-Ollama backend
# --------------------------------------------------------------------------


def test_non_ollama_backend_skips_model_ollama_warmup(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    db_path = _db_with_items(tmp_path, n_items=1)
    monkeypatch.setenv("RSS_DB", str(db_path))
    fake = _patch_run(monkeypatch)
    warmup_called = {"v": False}

    def fake_warmup():
        warmup_called["v"] = True

    monkeypatch.setattr("attestation.cli.warmup", fake_warmup)
    _presync_skill(monkeypatch, tmp_path)

    rc = install.run_install(check=True)

    assert not any("ollama" in call for call in fake.calls)
    assert warmup_called["v"] is False
    out = capsys.readouterr().out.lower()
    assert "skipped" in out
    assert rc == 0


def test_is_ollama_backend_detects_localhost(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    assert install._is_ollama_backend() is True


def test_is_ollama_backend_false_for_remote(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    assert install._is_ollama_backend() is False


# --------------------------------------------------------------------------
# StepResult / _run shape
# --------------------------------------------------------------------------


def test_step_result_dataclass_shape():
    r = install.StepResult(name="uv", status="OK")
    assert r.name == "uv"
    assert r.status == "OK"
    assert r.detail == ""


def test_run_seam_defaults(monkeypatch):
    """_run wraps subprocess.run with check=False, capture_output=True, text=True."""
    captured = {}

    def fake_subprocess_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["kw"] = kw
        return _cp(cmd, 0, stdout="ok")

    monkeypatch.setattr(install.subprocess, "run", fake_subprocess_run)
    result = install._run(["echo", "hi"])
    assert captured["kw"]["check"] is False
    assert captured["kw"]["capture_output"] is True
    assert captured["kw"]["text"] is True
    assert result.stdout == "ok"


# --------------------------------------------------------------------------
# step_uv
# --------------------------------------------------------------------------


def test_step_uv_present(monkeypatch):
    monkeypatch.setattr(install.shutil, "which", lambda name: "/usr/bin/uv")
    result = install.step_uv()
    assert result.status == "OK"


def test_step_uv_missing_is_broken(monkeypatch):
    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    result = install.step_uv()
    assert result.status == "BROKEN"


# --------------------------------------------------------------------------
# step_first_data / ingest + now->tag
# --------------------------------------------------------------------------


def test_step_first_data_no_items_runs_ingest(monkeypatch, tmp_path):
    db_path = tmp_path / "empty.db"
    from attestation.db import get_db

    get_db(db_path).close()
    monkeypatch.setenv("RSS_DB", str(db_path))
    fake = _patch_run(monkeypatch)

    result = install.step_first_data(check=False, yes=True, now=False)

    assert ["uv", "run", "attest", "ingest"] in [list(c) for c in fake.calls]
    assert result.status == "FIXED"


def test_step_first_data_check_mode_no_ingest_call(monkeypatch, tmp_path):
    db_path = tmp_path / "empty.db"
    from attestation.db import get_db

    get_db(db_path).close()
    monkeypatch.setenv("RSS_DB", str(db_path))
    fake = _patch_run(monkeypatch)

    result = install.step_first_data(check=True, yes=False, now=False)

    assert ["uv", "run", "attest", "ingest"] not in [list(c) for c in fake.calls]
    assert result.status == "BROKEN"


def test_step_first_data_now_flag_also_runs_tag(monkeypatch, tmp_path):
    db_path = tmp_path / "empty.db"
    from attestation.db import get_db

    get_db(db_path).close()
    monkeypatch.setenv("RSS_DB", str(db_path))
    fake = _patch_run(monkeypatch)

    install.step_first_data(check=False, yes=True, now=True)

    assert ["uv", "run", "attest", "tag"] in [list(c) for c in fake.calls]


def test_step_first_data_has_items_ok(monkeypatch, tmp_path):
    db_path = _db_with_items(tmp_path, n_items=2)
    monkeypatch.setenv("RSS_DB", str(db_path))
    fake = _patch_run(monkeypatch)

    result = install.step_first_data(check=False, yes=True, now=False)

    assert result.status == "OK"
    assert fake.calls == []


# --------------------------------------------------------------------------
# step_warmup
# --------------------------------------------------------------------------


def test_step_warmup_ok_on_success(monkeypatch):
    monkeypatch.setattr("attestation.cli.warmup", lambda: None)
    result = install.step_warmup()
    assert result.status == "OK"


# --------------------------------------------------------------------------
# step_mcp_wiring
# --------------------------------------------------------------------------


def test_mcp_wiring_present_in_list_no_add_recorded(monkeypatch):
    fake = _patch_run(
        monkeypatch,
        responses={
            ("agenthermes", "mcp", "list"): _cp([], 0, stdout="attestation\nother-server\n")
        },
    )

    result = install.step_mcp_wiring("agenthermes", check=False)

    assert result.status == "OK"
    assert not any("add" in c for c in fake.calls)


def _registers(name, verb, before=""):
    """A `<verb> list` stub modelling the real state change: `before` until the
    matching `add`/`create` runs, then listing `name`. Steps re-list to confirm
    registration, so a stub frozen at `before` means "never registered".
    """
    done = []

    def run(cmd):
        if cmd[1:3] == [verb, "add"] or cmd[1:3] == [verb, "create"]:
            done.append(cmd)
            return _cp(cmd, 0)
        return _cp(cmd, 0, stdout=(f"{before}{name}\n" if done else before))

    return run


def test_mcp_wiring_absent_exact_add_argv_recorded(monkeypatch, tmp_path):
    stub = _registers("attestation", "mcp", before="other-server\n")
    fake = _patch_run(monkeypatch, default=stub)

    result = install.step_mcp_wiring("agenthermes", check=False)

    assert result.status == "FIXED"
    expected = [
        "agenthermes",
        "mcp",
        "add",
        "attestation",
        "--command",
        "uv",
        "--args",
        "run",
        "--project",
        str(tmp_path),
        "attest-mcp",
    ]
    assert expected in fake.calls


def test_mcp_wiring_check_mode_absent_broken_no_add(monkeypatch):
    fake = _patch_run(
        monkeypatch,
        responses={("agenthermes", "mcp", "list"): _cp([], 0, stdout="other-server\n")},
    )

    result = install.step_mcp_wiring("agenthermes", check=True)

    assert result.status == "BROKEN"
    assert not any("add" in c for c in fake.calls)


def test_mcp_wiring_no_agent_binary_skipped(monkeypatch):
    fake = _patch_run(monkeypatch)

    result = install.step_mcp_wiring(None, check=False)

    assert result.status == "SKIPPED"
    assert fake.calls == []


# --------------------------------------------------------------------------
# step_skill_copy
# --------------------------------------------------------------------------


def test_skill_copy_creates_files(monkeypatch, tmp_path):
    fake_home = tmp_path / "fresh-home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)

    result = install.step_skill_copy(check=False)

    dest = fake_home / ".hermes" / "skills" / "research-provenance"
    assert (dest / "SKILL.md").exists()
    assert (dest / "scripts" / "setup.sh").exists()
    assert result.status == "FIXED"


def test_skill_copy_second_run_is_ok(monkeypatch, tmp_path):
    fake_home = tmp_path / "fresh-home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)

    install.step_skill_copy(check=False)
    result = install.step_skill_copy(check=False)

    assert result.status == "OK"


def test_skill_copy_never_touches_planted_data_dir(monkeypatch, tmp_path):
    fake_home = tmp_path / "fresh-home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)

    dest = fake_home / ".hermes" / "skills" / "research-provenance"
    data_dir = dest / "data"
    data_dir.mkdir(parents=True)
    blob = b"\x00\x01binary-db-blob\xff"
    (data_dir / "hermes.db").write_bytes(blob)

    install.step_skill_copy(check=False)
    install.step_skill_copy(check=False)

    assert (data_dir / "hermes.db").read_bytes() == blob


def test_skill_copy_check_mode_missing_is_broken(monkeypatch, tmp_path):
    fake_home = tmp_path / "fresh-home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)

    result = install.step_skill_copy(check=True)

    dest = fake_home / ".hermes" / "skills" / "research-provenance"
    assert result.status == "BROKEN"
    assert not (dest / "SKILL.md").exists()


def test_skill_source_ships_inside_the_package():
    """The skill must resolve package-relative, so uvx installs get it too.

    It used to resolve under _REPO_ROOT, which is the checkout only for
    editable installs -- under a wheel that path does not exist and the step
    crashed with FileNotFoundError.
    """
    src = install._skill_source_dir()

    assert src.is_dir()
    assert (src / "SKILL.md").is_file()
    assert (src / "scripts" / "setup.sh").is_file()
    # package-relative, not checkout-relative
    assert src.is_relative_to(Path(install.__file__).resolve().parent)


@pytest.mark.parametrize("check", [True, False])
def test_skill_copy_skips_when_source_missing(monkeypatch, tmp_path, check):
    """Guard for odd packaging: skip cleanly rather than crash the whole run."""
    fake_home = tmp_path / "fresh-home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)
    monkeypatch.setattr(install, "_skill_source_dir", lambda: tmp_path / "absent")

    result = install.step_skill_copy(check=check)

    assert result.status == "SKIPPED"
    assert not (fake_home / ".hermes" / "skills").exists()


# --------------------------------------------------------------------------
# packaged-install (uvx/wheel) safety: no step may crash or misconfigure
# --------------------------------------------------------------------------


@pytest.fixture
def packaged_install(monkeypatch, tmp_path):
    """Simulate a non-editable install: _REPO_ROOT with no pyproject.toml marker.

    The wheel packages only src/hermes, so _REPO_ROOT resolves to
    <venv>/lib/pythonX.Y — a directory that exists and is writable but is
    not a checkout (no .env.sample, no skills/, no feeds.toml).
    """
    site_packages_ish = tmp_path / "lib" / "python3.13"
    site_packages_ish.mkdir(parents=True)
    fake_home = tmp_path / "fresh-home"
    fake_home.mkdir()
    monkeypatch.setattr(llm, "_REPO_ROOT", site_packages_ish)
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)
    return site_packages_ish


@pytest.mark.parametrize("check", [True, False])
def test_env_file_skips_without_checkout(packaged_install, check):
    """Reading .env.sample from a non-checkout raised FileNotFoundError."""
    result = install.step_env_file(check=check)

    assert result.status == "SKIPPED"
    assert not (packaged_install / ".env").exists()


def test_mcp_wiring_never_registers_a_broken_project_path(packaged_install, monkeypatch):
    """`uv run --project <site-packages>` can never work; registering it would
    write a permanently broken entry into the user's real ~/.hermes/config.yaml."""
    fake = FakeRun(default=_cp([], 0, stdout=""))  # no attestation registered yet
    monkeypatch.setattr(install, "_run", fake)

    result = install.step_mcp_wiring("hermes", check=False)

    assert result.status == "SKIPPED"
    assert not any("add" in call for call in fake.calls)


def test_schedule_never_writes_a_cron_script_that_cannot_work(packaged_install, monkeypatch):
    """The refresh script cds into the checkout for feeds.toml; without one it
    would fail silently every hour."""
    fake = FakeRun(default=_cp([], 0, stdout=""))
    monkeypatch.setattr(install, "_run", fake)

    result = install.step_schedule("hermes", check=False)

    assert result.status == "SKIPPED"
    script = install.Path.home() / ".hermes" / "scripts" / install.REFRESH_SCRIPT_NAME
    assert not script.exists()


def test_check_mode_creates_no_database(monkeypatch, tmp_path):
    """get_db() runs CREATE TABLE, so --check must not open an absent database
    or it litters a ~90KB hermes.db (in cwd when RSS_DB is unset)."""
    db_path = tmp_path / "nonexistent.db"
    monkeypatch.setenv("RSS_DB", str(db_path))

    result = install.step_first_data(check=True)

    assert result.status == "BROKEN"
    assert not db_path.exists()


def test_check_mode_does_not_pin_models_into_vram(monkeypatch):
    """warmup POSTs keep_alive=-1 ('Forever'). A read-only doctor command must
    not load ~9.6GB into GPU memory."""
    import attestation.cli

    called = []
    monkeypatch.setattr(attestation.cli, "warmup", lambda: called.append("warmup"))

    result = install.step_warmup(check=True)

    assert result.status == "SKIPPED"
    assert called == []


# --------------------------------------------------------------------------
# step_schedule (refresh script + cron create)
# --------------------------------------------------------------------------


def test_schedule_writes_exact_content_and_exec_bit(monkeypatch, tmp_path):
    fake_home = tmp_path / "fresh-home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)
    _patch_run(monkeypatch, default=_registers(install.CRON_JOB_NAME, "cron"))

    result = install.step_schedule("agenthermes", check=False)

    script_path = fake_home / ".hermes" / "scripts" / "attestation-refresh.sh"
    expected = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"\n'
        f"cd {tmp_path}\n"
        "uv run attest ingest >/dev/null\n"
        "uv run attest tag >/dev/null\n"
    )
    assert script_path.read_text() == expected
    assert os.access(script_path, os.X_OK)
    assert result.status == "FIXED"


def test_refresh_script_survives_crons_bare_path(tmp_path):
    """cron runs a non-login shell with PATH=/usr/bin:/bin, so uv (in
    ~/.local/bin) is not on it. The original script died with
    'uv: command not found' every hour -- and exited 0 while doing it, because
    `a && b` reports success when the chain short-circuits. Run the generated
    script for real under that environment with a stub uv on the user's bin
    dir, and require both that it finds uv and that a failing step is a
    non-zero exit.
    """
    import subprocess

    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    marker = tmp_path / "ran"
    uv = home / ".local" / "bin" / "uv"
    uv.write_text(f'#!/bin/sh\necho "$@" >> {marker}\nexit 0\n')
    uv.chmod(0o755)

    script = tmp_path / "refresh.sh"
    script.write_text(install._refresh_script_content(checkout))
    script.chmod(0o755)

    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "SHELL": "/bin/sh"}
    proc = subprocess.run([str(script)], env=env, capture_output=True, text=True)

    assert proc.returncode == 0, f"script failed under cron's PATH: {proc.stderr}"
    assert marker.read_text().splitlines() == ["run attest ingest", "run attest tag"]

    # And a failure must surface as a non-zero exit, not be swallowed.
    uv.write_text("#!/bin/sh\nexit 3\n")
    uv.chmod(0o755)
    failed = subprocess.run([str(script)], env=env, capture_output=True, text=True)

    assert failed.returncode != 0, "a failing refresh must exit non-zero so cron reports it"


def test_mcp_wiring_reports_broken_when_add_does_not_take(monkeypatch, tmp_path):
    """Regression: `mcp add` was fire-and-forget, so a failed registration
    reported FIXED and the agent silently got no attestation tools at all --
    the step every other tool depends on."""
    _patch_run(monkeypatch, default=_cp([], 0, stdout="other-server\n"))

    result = install.step_mcp_wiring("agenthermes", check=False)

    assert result.status == "BROKEN"
    assert "attestation" in result.detail


def test_models_reports_broken_when_a_pull_fails(monkeypatch, tmp_path):
    """Regression: `ollama pull` results were discarded, so "pulled: X" was
    reported for a model that is not there -- and every later chat/embed call
    fails against a backend the installer just declared healthy. `ollama pull`
    exits 1 on a bad name, a network failure, or a full disk."""

    def run(cmd):
        return _cp(cmd, 1 if cmd[:2] == ["ollama", "pull"] else 0)

    _patch_run(monkeypatch, default=run)

    result = install._step_models_pull(["hermes3:3b"], yes=True)

    assert result.status == "BROKEN"
    assert "failed to pull" in result.detail
    assert "hermes3:3b" in result.detail


def test_models_pull_partial_failure_names_both_halves(monkeypatch):
    """A mixed result must not read as total success or total failure."""

    def run(cmd):
        return _cp(cmd, 1 if cmd[-1] == "badmodel" else 0)

    _patch_run(monkeypatch, default=run)

    result = install._step_models_pull(["goodmodel", "badmodel"], yes=True)

    assert result.status == "BROKEN"
    assert "goodmodel" in result.detail and "badmodel" in result.detail


def test_first_data_reports_broken_when_ingest_fails(monkeypatch, tmp_path):
    """Regression: _run_ingest_and_maybe_tag returned None and the step always
    said "ran ingest" -- a bad feeds.toml or no network still read as FIXED."""
    monkeypatch.setattr(install, "_checkout_root", lambda: tmp_path)
    monkeypatch.setattr(install, "_consent", lambda *a, **k: True)
    monkeypatch.setenv("RSS_DB", str(tmp_path / "absent.db"))
    _patch_run(monkeypatch, default=_cp([], 1))

    result = install.step_first_data(check=False, yes=True, now=False)

    assert result.status == "BROKEN"
    assert "ingest failed" in result.detail


def test_schedule_reports_broken_when_the_agent_has_no_cron_subcommand(monkeypatch, tmp_path):
    """Regression: hermes-agent builds without a `cron` subcommand parse "cron"
    as a chat prompt and exit 0, so `cron create` "succeeds" while registering
    nothing. step_schedule used to discard that result and report FIXED for a
    job that did not exist -- the reason a 408-item backlog accumulated behind
    a job nobody knew was missing. It must report BROKEN and hand back a
    crontab line the user can paste.
    """
    fake_home = tmp_path / "fresh-home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)
    # exit 0 everywhere, but `cron list` never shows the job -- the real symptom.
    _patch_run(monkeypatch, default=_cp([], 0, stdout=""))

    result = install.step_schedule("agenthermes", check=False)

    assert result.status == "BROKEN"
    assert install.CRON_JOB_NAME in result.detail
    assert str(fake_home / ".hermes" / "scripts" / "attestation-refresh.sh") in result.detail
    # the script itself is still written and usable
    assert (fake_home / ".hermes" / "scripts" / "attestation-refresh.sh").exists()


def test_schedule_cron_create_exact_argv_guarded_by_list(monkeypatch, tmp_path):
    fake_home = tmp_path / "fresh-home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)
    fake = _patch_run(
        monkeypatch, responses={("agenthermes", "cron", "list"): _cp([], 0, stdout="")}
    )

    install.step_schedule("agenthermes", check=False)

    expected = [
        "agenthermes",
        "cron",
        "create",
        "17 * * * *",
        "--name",
        "attestation-refresh",
        "--script",
        "attestation-refresh.sh",
        "--no-agent",
    ]
    assert expected in fake.calls


def test_schedule_guarded_when_already_in_cron_list(monkeypatch, tmp_path):
    fake_home = tmp_path / "fresh-home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)
    fake = _patch_run(
        monkeypatch,
        responses={
            ("agenthermes", "cron", "list"): _cp([], 0, stdout="attestation-refresh  17 * * * *\n")
        },
    )

    # First write the script so the file itself is already up to date.
    install.step_schedule("agenthermes", check=False)
    fake.calls.clear()

    result = install.step_schedule("agenthermes", check=False)

    assert not any("create" in c for c in fake.calls)
    assert result.status == "OK"


def test_schedule_no_rewrite_when_content_unchanged(monkeypatch, tmp_path):
    fake_home = tmp_path / "fresh-home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)
    _patch_run(
        monkeypatch,
        responses={
            ("agenthermes", "cron", "list"): _cp([], 0, stdout="attestation-refresh  17 * * * *\n")
        },
    )

    install.step_schedule("agenthermes", check=False)
    script_path = fake_home / ".hermes" / "scripts" / "attestation-refresh.sh"
    mtime_before = script_path.stat().st_mtime_ns

    install.step_schedule("agenthermes", check=False)

    assert script_path.stat().st_mtime_ns == mtime_before


def test_schedule_no_agent_binary_skipped(monkeypatch):
    fake = _patch_run(monkeypatch)

    result = install.step_schedule(None, check=False)

    assert result.status == "SKIPPED"
    assert fake.calls == []


# --------------------------------------------------------------------------
# agent binary None -> all four agent-dependent steps SKIPPED, exit still 0
# --------------------------------------------------------------------------


def test_no_agent_binary_all_wiring_steps_skipped_exit_zero(monkeypatch, tmp_path, capsys):
    db_path = _db_with_items(tmp_path, n_items=1)
    monkeypatch.setenv("RSS_DB", str(db_path))
    _patch_run(monkeypatch, responses={("ollama", "list"): _ollama_list_ok()})
    monkeypatch.setattr(install, "_ollama_native_root_reachable", lambda: True)
    monkeypatch.setattr("attestation.cli.warmup", lambda: None)
    monkeypatch.setattr(install.os, "get_exec_path", lambda: [])

    rc = install.run_install(check=False, yes=True)

    out = capsys.readouterr().out
    assert rc == 0
    for step in ("mcp_wiring", "reasoning_override", "schedule"):
        line = next(line_ for line_ in out.splitlines() if step in line_)
        assert "skipped" in line.lower()


# --------------------------------------------------------------------------
# --check never mutates mcp/config/cron/copies
# --------------------------------------------------------------------------


def test_check_mode_never_invokes_mcp_add_config_set_cron_create_or_copies(
    monkeypatch, tmp_path, capsys
):
    db_path = _db_with_items(tmp_path, n_items=1)
    monkeypatch.setenv("RSS_DB", str(db_path))
    agent_dir = tmp_path / "agentbin"
    agent_dir.mkdir()
    agent_bin = agent_dir / "hermes"
    agent_bin.write_text("#!/bin/sh\n")
    agent_bin.chmod(0o755)
    monkeypatch.setattr(install.os, "get_exec_path", lambda: [str(agent_dir)])
    monkeypatch.setattr(install.sys, "prefix", str(tmp_path / "some-other-venv"))

    fake_home = tmp_path / "fresh-home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)

    fake = _patch_run(
        monkeypatch,
        responses={
            ("ollama", "list"): _ollama_list_ok(),
            (str(agent_bin), "mcp", "list"): _cp([], 0, stdout=""),
            (str(agent_bin), "cron", "list"): _cp([], 0, stdout=""),
            (str(agent_bin), "config", "get", "agent.reasoning_overrides.hermes3:8b"): _cp(
                [], 1, stdout=""
            ),
        },
    )
    monkeypatch.setattr(install, "_ollama_native_root_reachable", lambda: True)
    monkeypatch.setattr("attestation.cli.warmup", lambda: None)

    rc = install.run_install(check=True)

    assert rc == 1  # gaps exist (nothing wired up in the fresh fake home)
    dest = fake_home / ".hermes" / "skills" / "research-provenance"
    assert not dest.exists()
    for call in fake.calls:
        assert "add" not in call
        assert not (list(call)[:2] == ["config", "set"] or "set" in call and "config" in call)
        assert "create" not in call


def test_step_warmup_skipped_on_exception(monkeypatch):
    def boom():
        raise RuntimeError("no ollama")

    monkeypatch.setattr("attestation.cli.warmup", boom)
    result = install.step_warmup()
    assert result.status == "SKIPPED"


def test_mcp_add_answers_the_interactive_prompts(monkeypatch, tmp_path):
    """`hermes mcp add` prompts twice -- to save when the connection probe
    fails, and to enable the discovered tools -- with no non-interactive flag.
    On EOF it takes the NEGATIVE default and still exits 0, so an unattended
    install would report success while registering nothing. Verified against
    the real binary: `printf 'y\\ny\\n'` drives it to "33/33 tools enabled".
    """
    seen = {}

    def run(cmd, **kw):
        if cmd[1:3] == ["mcp", "add"]:
            seen["input"] = kw.get("input")
        return _cp(cmd, 0, stdout="attestation" if seen else "")

    monkeypatch.setattr(install, "_run", run)
    monkeypatch.setattr(install, "_checkout_root", lambda: tmp_path)

    install.step_mcp_wiring("agenthermes", check=False)

    assert seen["input"] == "y\ny\n", "the prompts must be answered, not left to EOF"


def test_every_other_subprocess_gets_devnull_stdin(monkeypatch, tmp_path):
    """An unexpected prompt must never hang an unattended install: with no
    stdin, a command that asks a question gets EOF and takes its default
    instead of blocking forever on a terminal nobody is watching."""
    captured = []

    def fake_subprocess_run(cmd, **kw):
        captured.append(kw)
        return _cp(cmd, 0, stdout="")

    monkeypatch.setattr(install.subprocess, "run", fake_subprocess_run)

    install._run(["ollama", "list"])

    assert captured[0]["stdin"] is install.subprocess.DEVNULL


def test_answering_prompts_does_not_also_set_stdin(monkeypatch):
    """input= and stdin= are mutually exclusive in subprocess.run; setting both
    raises. The seam must pass only one."""
    captured = []

    def fake_subprocess_run(cmd, **kw):
        captured.append(kw)
        return _cp(cmd, 0)

    monkeypatch.setattr(install.subprocess, "run", fake_subprocess_run)

    install._run_answering_prompts(["x", "mcp", "add"])

    assert captured[0].get("input") == "y\ny\n"
    assert "stdin" not in captured[0]


def test_reasoning_override_is_skipped_for_non_hermes3_models(monkeypatch, tmp_path):
    """The override works around a hermes3-specific Ollama rejection
    (`HTTP 400: "hermes3:8b" does not support thinking`). The default model is
    now gemma4:e2b, which accepts think:true and returns a thinking field --
    verified against the live daemon -- so applying the override would write a
    config entry for a problem that model does not have.

    The hermes3 fixtures elsewhere in this file cover the apply path; this
    covers the skip path, which is what every default install now takes.
    """
    monkeypatch.setenv("CHAT_MODEL", "gemma4:e2b-it-q4_K_M")
    fake = _patch_run(monkeypatch)

    result = install.step_reasoning_override("agenthermes", check=False)

    assert result.status == "SKIPPED"
    assert "does not need an override" in result.detail
    assert not any("config" in c for c in fake.calls), "must not touch the agent's config"
