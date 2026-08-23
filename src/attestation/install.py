"""attest install: idempotent setup + --check doctor mode.

Step functions each return a StepResult; run_install() executes them in
spec order and prints one aligned line per step. All subprocess calls go
through the single _run() seam so tests can monkeypatch it.

This module implements the detection steps + pure-local fixes (uv, Ollama
reachability, models, .env, first data, warmup) and the agent-wiring steps
(mcp/skill/reasoning/cron), all gated on `_find_agent_binary()` finding
hermes-agent's own CLI (never our venv's console script).
"""

import fnmatch
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse

from attestation.llm import base_url, chat_model, embed_model

SKILL_NAME = "research-provenance"
CRON_JOB_NAME = "attestation-refresh"
REFRESH_SCRIPT_NAME = "attestation-refresh.sh"
CLI_NAME = "attest"


class Status(StrEnum):
    """The closed set of step outcomes.

    A StrEnum member IS a str -- every existing `result.status == "OK"`
    comparison, f-string, and json.dumps call keeps working untouched -- but
    `ty` now rejects a typo'd literal at check time instead of
    `_STATUS_LABEL[result.status]` raising KeyError after all ten install
    steps have already run.
    """

    OK = "OK"
    FIXED = "FIXED"
    BROKEN = "BROKEN"
    SKIPPED = "SKIPPED"


@dataclass
class StepResult:
    name: str
    status: Status
    detail: str = ""


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Single subprocess seam. Tests monkeypatch this, never subprocess.run directly.

    `stdin` defaults to DEVNULL so an unexpected prompt can never hang an
    unattended install: a command that asks a question gets EOF and takes its
    default instead of blocking forever on a terminal nobody is watching. Pass
    `input=` to answer a prompt deliberately -- see `_run_answering_prompts`.
    """
    kw.setdefault("check", False)
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    if "input" not in kw:
        kw.setdefault("stdin", subprocess.DEVNULL)
    return subprocess.run(cmd, **kw)


def _run_answering_prompts(cmd: list[str], answers: str = "y\ny\n") -> subprocess.CompletedProcess:
    """Run a command that asks interactive yes/no questions, answering yes.

    `hermes mcp add` prompts twice -- once to save when the connection probe
    fails, once to enable the discovered tools -- and has no non-interactive
    flag. With EOF on stdin it takes the *negative* default and still exits 0,
    so an unattended install would report success while registering nothing.

    Feeding "y" is safe here because the caller has already decided to register:
    the questions are confirmations of the action requested, not new decisions.
    The result is still verified afterwards rather than trusted.

    Routed through `_run` so the single-subprocess-seam invariant holds and
    tests keep seeing every command the installer issues.
    """
    return _run(cmd, input=answers)


def _find_agent_binary() -> str | None:
    """hermes-agent's CLI — NOT our console script (which shadows it inside the venv)."""
    for d in os.get_exec_path():
        cand = Path(d) / "hermes"
        in_venv = str(cand.resolve()).startswith(sys.prefix)
        if cand.is_file() and os.access(cand, os.X_OK) and not in_venv:
            return str(cand)
    fallback = Path.home() / ".local" / "bin" / "hermes"
    return str(fallback) if fallback.is_file() else None


def _is_ollama_backend() -> bool:
    """base_url() host is localhost-ish (native root reachability checked in the step)."""
    host = urlparse(base_url()).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1")


def _native_root() -> str:
    return base_url().rstrip("/").removesuffix("/v1")


def _ollama_native_root_reachable() -> bool:
    import httpx

    try:
        httpx.get(_native_root(), timeout=5).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def _checkout_root() -> Path | None:
    """The repo checkout root, or None when running from a non-editable install.

    `llm._REPO_ROOT` is `parents[2]` of the module file, which is the real
    checkout for editable installs but resolves to `<venv>/lib/pythonX.Y`
    under a wheel (uvx). That path *exists and is writable*, so it fails
    silently rather than loudly. The wheel packages only `src/attestation`, so
    probing for a repo-root marker distinguishes the two reliably.
    """
    import attestation.llm as llm

    root = llm._REPO_ROOT
    return root if (root / "pyproject.toml").is_file() else None


NO_CHECKOUT = "no repo checkout (installed as a package) — clone the repo to use this step"


def _consent(yes: bool, prompt: str) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        return False
    return input(prompt).strip().lower() in ("y", "yes")


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------


def step_uv() -> StepResult:
    if shutil.which("uv"):
        return StepResult("uv", Status.OK)
    return StepResult(
        "uv", Status.BROKEN, "uv not found on PATH — install: https://docs.astral.sh/uv/"
    )


def step_ollama_reachable() -> StepResult:
    if not _is_ollama_backend():
        return StepResult("ollama_reachable", Status.SKIPPED, "non-Ollama LLM_BASE_URL")
    if _ollama_native_root_reachable():
        return StepResult("ollama_reachable", Status.OK)
    return StepResult(
        "ollama_reachable",
        Status.BROKEN,
        f"cannot reach {_native_root()} — is `ollama serve` running?",
    )


def _normalize_model(name: str) -> str:
    """`ollama list` prints tag-qualified names; an untagged name means :latest."""
    return name if ":" in name else f"{name}:latest"


def _installed_models(run_fn) -> set[str]:
    proc = run_fn(["ollama", "list"])
    lines = proc.stdout.splitlines()[1:]  # drop header row
    return {_normalize_model(line.split()[0]) for line in lines if line.strip()}


def _step_models_check(missing: list[str]) -> StepResult:
    detail = f"missing: {', '.join(missing)} — rerun with --yes to pull"
    return StepResult("models", Status.BROKEN, detail)


def _step_models_pull(missing: list[str], yes: bool) -> StepResult:
    if not _consent(yes, f"Pull missing models ({', '.join(missing)})? [y/N] "):
        return StepResult(
            "models", Status.BROKEN, f"missing: {', '.join(missing)} — declined; rerun with --yes"
        )
    # Check each pull. `ollama pull` exits 1 on a bad name, a network failure,
    # or a full disk; discarding that reported "pulled: X" for a model that is
    # not there, and every later chat/embed call would then fail against a
    # backend the installer had just declared healthy.
    pulled, failed = [], []
    for m in missing:
        (pulled if _run(["ollama", "pull", m]).returncode == 0 else failed).append(m)
    if failed:
        detail = f"failed to pull: {', '.join(failed)}"
        if pulled:
            detail = f"pulled: {', '.join(pulled)}; {detail}"
        return StepResult("models", Status.BROKEN, detail)
    return StepResult("models", Status.FIXED, f"pulled: {', '.join(pulled)}")


def step_models(check: bool = False, yes: bool = False) -> StepResult:
    if not _is_ollama_backend():
        return StepResult("models", Status.SKIPPED, "non-Ollama LLM_BASE_URL")
    wanted = [chat_model(), embed_model()]
    installed = _installed_models(_run)
    missing = [m for m in wanted if _normalize_model(m) not in installed]
    if not missing:
        return StepResult("models", Status.OK, f"{', '.join(wanted)} present")
    if check:
        return _step_models_check(missing)
    return _step_models_pull(missing, yes)


def step_env_file(check: bool = False) -> StepResult:
    root = _checkout_root()
    if root is None:
        return StepResult("env_file", Status.SKIPPED, NO_CHECKOUT)
    env_path = root / ".env"
    sample_path = root / ".env.sample"
    if env_path.exists():
        return StepResult("env_file", Status.OK)
    if check:
        return StepResult("env_file", Status.BROKEN, f"{env_path} missing — copy from .env.sample")
    env_path.write_text(sample_path.read_text())
    return StepResult("env_file", Status.FIXED, f"created {env_path} from .env.sample")


def _run_ingest_and_maybe_tag(now: bool, root: Path) -> tuple[bool, str]:
    """Run ingest (and optionally tag). Returns (ok, detail)."""
    if _run(["uv", "run", CLI_NAME, "ingest"], cwd=root).returncode != 0:
        return False, "ingest failed"
    if now and _run(["uv", "run", CLI_NAME, "tag"], cwd=root).returncode != 0:
        return False, "ran ingest, but tag failed"
    return True, "ran ingest" + (" + tag" if now else "")


def step_first_data(check: bool = False, yes: bool = False, now: bool = False) -> StepResult:
    from attestation.db import get_db, resolve_db_path

    db_path = resolve_db_path(None)
    # get_db() runs CREATE TABLE, so opening an absent database would make
    # --check write a ~90KB file (in cwd, when RSS_DB is unset). Report
    # instead: a database that does not exist trivially has no items.
    if not Path(db_path).exists():
        if check:
            return StepResult("first_data", Status.BROKEN, f"no database at {db_path}")
        count = 0
    else:
        conn = get_db(db_path)
        count = conn.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"]
        untagged = conn.execute(
            "SELECT COUNT(*) AS n FROM items i LEFT JOIN item_features f ON f.item_id = i.id"
            " WHERE f.item_id IS NULL"
        ).fetchone()["n"]
        conn.close()

        if count > 0:
            detail = f"{count} items ({untagged} untagged)" if untagged else f"{count} items"
            return StepResult("first_data", Status.OK, detail)

        if check:
            detail = "no items in database — rerun with --yes to ingest"
            return StepResult("first_data", Status.BROKEN, detail)

    root = _checkout_root()
    if root is None:
        return StepResult("first_data", Status.SKIPPED, NO_CHECKOUT)

    if not _consent(yes, "No items in database. Run ingest now? [y/N] "):
        return StepResult("first_data", Status.BROKEN, "declined ingest — rerun with --yes")

    ok, detail = _run_ingest_and_maybe_tag(now, root)
    return StepResult("first_data", Status.FIXED if ok else Status.BROKEN, detail)


def step_warmup(check: bool = False) -> StepResult:
    # warmup loads both models and holds them for OLLAMA_KEEP_ALIVE (30m
    # default). It used to pass keep_alive=-1 -- "Forever" in `ollama ps`, and
    # literally the year 2318 in the expiry field -- which held 5.4GB across
    # two llama-server processes on a 23GB machine and got a browser, a
    # quantum-chemistry job and the terminal OOM-killed.
    # --check is a read-only doctor; it must not load ~9.6GB into VRAM.
    if check:
        return StepResult("warmup", Status.SKIPPED, "warmup does not run in --check")
    if not _is_ollama_backend():
        return StepResult("warmup", Status.SKIPPED, "non-Ollama LLM_BASE_URL")
    try:
        import attestation.cli

        attestation.cli.warmup()
    except Exception as exc:  # noqa: BLE001 - warmup is best-effort by design
        return StepResult("warmup", Status.SKIPPED, str(exc))
    return StepResult("warmup", Status.OK)


# ---------------------------------------------------------------------------
# agent-wiring steps (mcp / skill copy / reasoning override / cron)
# ---------------------------------------------------------------------------


def _skill_source_dir() -> Path:
    """Skill files ship inside the package, so this resolves in every install mode.

    Package-relative (not _REPO_ROOT-relative): the wheel bundles
    attestation/skills/, so uvx installs get the skill too, and an editable
    checkout resolves to the same files under src/attestation/skills/.
    """
    return Path(__file__).resolve().parent / "skills" / SKILL_NAME


def _skill_dest_dir() -> Path:
    return Path.home() / ".hermes" / "skills" / SKILL_NAME


def _register_surfaces(agent: str, root: Path) -> list[str]:
    """Create the per-surface MCP entries. Returns the names it added.

    These were typed by hand and the installer never knew about them, so a
    fresh install got the full 46-tool surface -- which measured 8/15 -- rather
    than the four restricted ones that are the reason the surfaces exist.

    `enabled: false` on each: a surface is chosen at launch by a person, and
    five servers all enabled would put every tool back into one session.
    """
    from attestation import emit

    added = []
    for name, entry in emit.hermes_servers(root).items():
        created = _run_answering_prompts(
            [agent, "mcp", "add", name, "--command", entry["command"], "--args", *entry["args"]]
        )
        if created.returncode != 0:
            continue
        _run(
            [
                agent,
                "config",
                "set",
                f"mcp_servers.{name}.env.ATTEST_TOOLS",
                entry["env"]["ATTEST_TOOLS"],
            ]
        )
        _run([agent, "config", "set", f"mcp_servers.{name}.enabled", "false"])
        added.append(name)
    return added


def _check_surfaces(agent: str, *, check: bool) -> StepResult:
    """Whether the per-surface MCP entries match what the tool surface declares.

    Compares against `emit`'s generator, not against a substring. The old
    check returned ok on seeing "attestation" anywhere in `hermes mcp list`,
    which is true of a config holding the full server and none of the four
    per-surface entries -- exactly the state that shipped, since those four
    were typed by hand and nothing in the repo knew they existed.

    `orphaned` is the finding a substring can never make, and the same shape as
    the duplicate crontab entry: a hand-added thing that used to be right, that
    the tool owning the domain cannot see.
    """
    from attestation import emit

    root = _checkout_root()
    if root is None:
        return StepResult("mcp_wiring", Status.OK, "attestation registered")

    dump = _run([agent, "config", "get", "mcp_servers"])
    findings = emit.check_hermes(emit.parse_config_dump(dump.stdout), root)

    # A doctor that reports a state the installer cannot reach tells every user
    # their install is broken. Missing surfaces are creatable, so create them
    # unless this is a --check run; anything else is for the user to resolve.
    if findings and not check and all(f.kind == "missing" for f in findings):
        if _register_surfaces(agent, root):
            dump = _run([agent, "config", "get", "mcp_servers"])
            findings = emit.check_hermes(emit.parse_config_dump(dump.stdout), root)
            if not findings:
                return StepResult(
                    "mcp_wiring",
                    Status.FIXED,
                    f"registered {len(emit.hermes_servers(root))} surfaces",
                )

    if not findings:
        return StepResult(
            "mcp_wiring",
            Status.OK,
            f"attestation registered, all {len(emit.hermes_servers(root))} surfaces current",
        )
    return StepResult(
        "mcp_wiring",
        Status.BROKEN,
        f"{len(findings)} surface problem(s); `attest emit` explains each: "
        + "; ".join(f"[{f.kind}] {f.detail}" for f in findings),
    )


def step_mcp_wiring(agent: str | None, check: bool = False) -> StepResult:
    if agent is None:
        return StepResult("mcp_wiring", Status.SKIPPED, "no hermes-agent binary found")

    proc = _run([agent, "mcp", "list"])
    if "attestation" in proc.stdout:
        return _check_surfaces(agent, check=check)

    if check:
        return StepResult("mcp_wiring", Status.BROKEN, "attestation MCP server not registered")

    # Registering `uv run --project <root>` against a non-checkout would write
    # a permanently broken entry into the user's real ~/.hermes/config.yaml.
    root = _checkout_root()
    if root is None:
        return StepResult("mcp_wiring", Status.SKIPPED, NO_CHECKOUT)

    added = _run_answering_prompts(
        [
            agent,
            "mcp",
            "add",
            "attestation",
            "--command",
            "uv",
            "--args",
            "run",
            "--project",
            str(root),
            "attest-mcp",
        ]
    )
    # Confirm rather than assume: this is the step every other tool depends on,
    # so a silently-failed `mcp add` means the agent gets no attestation tools
    # at all while the installer reports a healthy wiring.
    if added.returncode != 0 or "attestation" not in _run([agent, "mcp", "list"]).stdout:
        return StepResult(
            "mcp_wiring",
            Status.BROKEN,
            f"{agent} did not register the attestation MCP server",
        )
    # Fall through rather than returning: the surfaces are half of this step,
    # and returning FIXED here would leave a fresh install with the full
    # 46-tool surface and none of the four restricted ones -- then report them
    # missing on the very next run.
    return _check_surfaces(agent, check=check)


def _skill_files_to_sync(src_dir: Path) -> list[Path]:
    files = [src_dir / "SKILL.md"]
    scripts_dir = src_dir / "scripts"
    if scripts_dir.is_dir():
        files.extend(sorted(p for p in scripts_dir.iterdir() if p.is_file()))
    return files


def _sync_one_skill_file(src: Path, dest_dir: Path, src_dir: Path) -> bool:
    """Copy src into dest_dir (preserving its relative path); True iff content changed."""
    rel = src.relative_to(src_dir)
    dest = dest_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = src.read_bytes()
    if dest.exists() and dest.read_bytes() == content:
        return False
    dest.write_bytes(content)
    return True


def step_skill_copy(check: bool = False) -> StepResult:
    src_dir = _skill_source_dir()
    dest_dir = _skill_dest_dir()
    # The skill ships inside the package, so this normally exists in every
    # install mode. Kept as a guard for odd packaging (e.g. a zipimport or a
    # stripped install): the skill is the optional fallback lane, so skip
    # cleanly rather than crashing the whole run.
    if not src_dir.is_dir():
        return StepResult(
            "skill_copy", Status.SKIPPED, "skill source not bundled with this install"
        )
    files = _skill_files_to_sync(src_dir)

    if check:
        changed = [
            f
            for f in files
            if not (dest_dir / f.relative_to(src_dir)).exists()
            or (dest_dir / f.relative_to(src_dir)).read_bytes() != f.read_bytes()
        ]
        if changed:
            return StepResult(
                "skill_copy", Status.BROKEN, f"{len(changed)} file(s) stale or missing"
            )
        return StepResult("skill_copy", Status.OK)

    changed = [f for f in files if _sync_one_skill_file(f, dest_dir, src_dir)]
    if changed:
        return StepResult("skill_copy", Status.FIXED, f"synced {len(changed)} file(s)")
    return StepResult("skill_copy", Status.OK)


def step_reasoning_override(agent: str | None, check: bool = False) -> StepResult:
    if agent is None:
        return StepResult("reasoning_override", Status.SKIPPED, "no hermes-agent binary found")
    if not _is_ollama_backend():
        return StepResult("reasoning_override", Status.SKIPPED, "non-Ollama LLM_BASE_URL")
    model = chat_model()
    if not fnmatch.fnmatch(model, "hermes3*"):
        return StepResult(
            "reasoning_override", Status.SKIPPED, f"{model} does not need an override"
        )

    key = f"agent.reasoning_overrides.{model}"
    proc = _run([agent, "config", "get", key])
    if proc.returncode == 0 and proc.stdout.strip() not in ("", "null", "None"):
        return StepResult("reasoning_override", Status.OK, f"{key} already set")

    if check:
        return StepResult("reasoning_override", Status.BROKEN, f"{key} unset")

    set_proc = _run([agent, "config", "set", key, "none"])
    if set_proc.returncode != 0:
        snippet = f"agent:\n  reasoning_overrides:\n    {model}: none"
        return StepResult(
            "reasoning_override",
            Status.SKIPPED,
            f"`hermes config set` failed — add this to ~/.hermes/config.yaml yourself:\n{snippet}",
        )
    return StepResult("reasoning_override", Status.FIXED, f"set {key}=none")


def _refresh_script_content(root: Path) -> str:
    # Four failures are encoded here, each one observed:
    #
    # 1. cron runs a non-login shell with a bare PATH (typically /usr/bin:/bin),
    #    so `uv` -- installed to ~/.local/bin -- is NOT on it. An earlier version
    #    died with "uv: command not found" every hour AND exited 0 while doing
    #    so, because `a && b` reports success when the chain short-circuits.
    # 2. No overlap guard: ingest+tag over a tagging backlog routinely outruns
    #    the cron interval, so runs stacked on one SQLite file instead of
    #    skipping. flock makes a slow run skip the next tick, not race it.
    # 3. Both steps redirected to /dev/null, which made a silent no-op and a
    #    healthy run indistinguishable afterwards. Timestamped lines are the
    #    only reason the stalled-lock period was diagnosable at all.
    # 4. A bare `set -e` made a tag failure fatal, so a cold or busy Ollama
    #    turned a successful ingest into a reported error. Per CLAUDE.md's
    #    reliability contract ingest must succeed, but tagging is best-effort:
    #    untagged items are picked up by the next pass.
    lock = Path.home() / ".hermes" / f"{REFRESH_SCRIPT_NAME.removesuffix('.sh')}.lock"
    return (
        "#!/usr/bin/env bash\n"
        # No `-e`: failures are handled per-step below, so that a non-fatal
        # tag failure cannot abort the script before it logs why.
        "set -uo pipefail\n"
        'export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"\n'
        f"cd {root} || exit 1\n"
        "\n"
        f'LOCK="{lock}"\n'
        'mkdir -p "$(dirname "$LOCK")"\n'
        'exec 9>"$LOCK"\n'
        "if ! flock -n 9; then\n"
        '  echo "[$(date -Is)] SKIP: previous run still holding lock"\n'
        "  exit 0\n"
        "fi\n"
        "\n"
        f'echo "[$(date -Is)] refresh start"\n'
        "\n"
        # Ingest is deterministic and needs no chat model; it must succeed.
        f"if uv run {CLI_NAME} ingest >/dev/null; then\n"
        '  echo "[$(date -Is)] ingest ok"\n'
        "else\n"
        "  rc=$?\n"
        '  echo "[$(date -Is)] ingest FAILED (exit $rc)"\n'
        '  exit "$rc"\n'
        "fi\n"
        "\n"
        # Tagging needs Ollama. A cold model is a degraded run, not a broken one.
        f"if uv run {CLI_NAME} tag >/dev/null; then\n"
        '  echo "[$(date -Is)] tag ok"\n'
        "else\n"
        '  echo "[$(date -Is)] tag FAILED (exit $?) -- items remain untagged,"\n'
        '  echo "[$(date -Is)] will retry next run"\n'
        "fi\n"
        "\n"
        f'echo "[$(date -Is)] refresh done"\n'
    )


def _refresh_script_path() -> Path:
    return Path.home() / ".hermes" / "scripts" / REFRESH_SCRIPT_NAME


def _write_refresh_script(check: bool, root: Path) -> StepResult:
    script_path = _refresh_script_path()
    content = _refresh_script_content(root)
    exists = script_path.exists()
    unchanged = exists and script_path.read_text() == content
    executable = exists and os.access(script_path, os.X_OK)

    if unchanged and executable:
        return StepResult("schedule", Status.OK, "refresh script up to date")
    if check:
        return StepResult("schedule", Status.BROKEN, f"{script_path} missing or stale")

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(content)
    script_path.chmod(0o755)
    return StepResult("schedule", Status.FIXED, f"wrote {script_path}")


def _crontab_lines() -> list[str]:
    """The user's crontab, or an empty list if they have none.

    `crontab -l` exits nonzero when no crontab exists, which is a normal
    state and not an error worth surfacing.
    """
    result = _run(["crontab", "-l"])
    if result.returncode != 0:
        return []
    return [ln for ln in result.stdout.splitlines() if ln.strip() and not ln.startswith("#")]


def _duplicate_crontab_entry() -> str | None:
    """A hand-added crontab line running the same refresh script.

    This step's own failure message tells the user to add one when the agent
    cannot register a job. That happened on 2026-08-11, the line was added,
    and the agent job later registered successfully -- so both ran. The script
    takes its own flock, so every crontab run from 2026-08-20 logged "SKIP:
    previous run still holding lock" and the log recorded zero successes,
    while the agent job quietly did the work.

    Checking only our own job reports `schedule: ok` for that, which is how it
    went unnoticed for two days.
    """
    for line in _crontab_lines():
        if REFRESH_SCRIPT_NAME in line:
            return line.strip()
    return None


def step_schedule(agent: str | None, check: bool = False) -> StepResult:
    if agent is None:
        return StepResult("schedule", Status.SKIPPED, "no hermes-agent binary found")

    # The refresh script cds into the checkout to run ingest (it needs
    # feeds.toml); without one it would fail silently every hour.
    root = _checkout_root()
    if root is None:
        return StepResult("schedule", Status.SKIPPED, NO_CHECKOUT)

    script_result = _write_refresh_script(check, root)
    if script_result.status == Status.BROKEN:
        return script_result

    proc = _run([agent, "cron", "list"])
    if CRON_JOB_NAME in proc.stdout:
        duplicate = _duplicate_crontab_entry()
        if duplicate:
            return StepResult(
                "schedule",
                Status.BROKEN,
                f"{CRON_JOB_NAME} is registered with {agent}, and a duplicate"
                " crontab entry runs the same script. Both take the script's"
                " own lock, so one logs 'SKIP: previous run still holding"
                " lock' on every wakeup and never ingests. Remove the crontab"
                f" line: {duplicate}",
            )
        return script_result

    if check:
        return StepResult("schedule", Status.BROKEN, f"{CRON_JOB_NAME} cron job not registered")

    created = _run_answering_prompts(
        [
            agent,
            "cron",
            "create",
            "17 * * * *",
            "--name",
            CRON_JOB_NAME,
            "--script",
            REFRESH_SCRIPT_NAME,
            "--no-agent",
        ]
    )
    # Check the result instead of assuming it worked. Not every hermes-agent
    # build has a `cron` subcommand -- on one 2026-08-11 install it parsed
    # "cron" as a chat prompt, started an LLM conversation, and exited 0, so
    # this step reported FIXED for a job it never registered. The refresh
    # script is still written and usable, so say exactly that and hand over
    # the crontab line rather than claiming a schedule that does not exist.
    if created.returncode != 0 or CRON_JOB_NAME not in _run([agent, "cron", "list"]).stdout:
        return StepResult(
            "schedule",
            Status.BROKEN,
            f"{_refresh_script_path()} written, but {agent} did not register"
            f" {CRON_JOB_NAME}. Schedule it yourself, e.g."
            f' "17 */4 * * * {_refresh_script_path()}"',
        )

    detail = "registered cron job"
    if script_result.status == Status.FIXED:
        detail = f"{script_result.detail}; {detail}"
    return StepResult("schedule", Status.FIXED, detail)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

_STATUS_LABEL = {
    Status.OK: "ok",
    Status.FIXED: "fixed",
    Status.BROKEN: "BROKEN",
    Status.SKIPPED: "skipped",
}


def _print_step(result: StepResult) -> None:
    label = _STATUS_LABEL[result.status]
    line = f"[{label}] {result.name}"
    if result.detail:
        line += f": {result.detail}"
    print(line)


def _run_steps(check: bool, yes: bool, now: bool) -> list[StepResult]:
    agent = _find_agent_binary()
    results = [step_uv(), step_ollama_reachable()]
    results.append(step_models(check=check, yes=yes))
    results.append(step_env_file(check=check))
    results.append(step_first_data(check=check, yes=yes, now=now))
    results.append(step_warmup(check=check))
    results.append(step_mcp_wiring(agent, check=check))
    results.append(step_skill_copy(check=check))
    results.append(step_reasoning_override(agent, check=check))
    results.append(step_schedule(agent, check=check))
    return results


def run_install(check: bool = False, yes: bool = False, now: bool = False) -> int:
    results = _run_steps(check, yes, now)
    for result in results:
        _print_step(result)
    return 1 if any(r.status == Status.BROKEN for r in results) else 0
