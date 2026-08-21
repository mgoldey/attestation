# `hermes install` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One idempotent command (`hermes install`, with `--check` doctor mode) that configures a hermes-agent user end-to-end, plus the agent-facing config-contract guide in SKILL.md.

**Architecture:** New `src/hermes/install.py` — small step functions returning `StepResult`, one `_run()` subprocess seam for testability, orchestrator `run_install(check, yes, now) -> int`. `cli.py` gets parser wiring + dispatch only. SKILL.md/setup.sh/README updated so guide == installer. Spec: `docs/superpowers/specs/2026-08-05-hermes-install-design.md` (read it first — it is the requirements source; this plan compresses repetition).

**Tech Stack:** Python 3.12, stdlib (subprocess/shutil/pathlib), pytest with a monkeypatched `_run`. Branch: `feat/hermes-install`.

## Global Constraints

- Idempotent: a second `hermes install` run performs **zero mutating actions** (every step reports OK).
- `--check` NEVER mutates anything (no writes, no subprocess with side effects).
- Skill sync must NEVER touch `~/.hermes/skills/science-recommendations/data/` (live DB lives there).
- Agent-binary discovery must skip our own venv: iterate `os.get_exec_path()`, skip entries under `sys.prefix`, fallback `~/.hermes` sibling `~/.local/bin/hermes`; missing binary → agent-wiring steps SKIPPED (not BROKEN).
- Never edit `~/.hermes/config.yaml` directly — only via the agent's own `hermes mcp add` / `hermes config set`; on `config set` failure, print the YAML for the user instead.
- Cron registration uses exactly: `hermes cron create "17 * * * *" --name hermes-rss-refresh --script hermes-rss-refresh.sh --no-agent`, guarded by `hermes cron list` output containing `hermes-rss-refresh`.
- Refresh script at `~/.hermes/scripts/hermes-rss-refresh.sh`: `#!/usr/bin/env bash`, `cd <checkout> && uv run hermes ingest >/dev/null && uv run hermes tag >/dev/null` with failures echoed to stdout (so `--no-agent` delivery fires) and success silent. `<checkout>` resolved from `llm._REPO_ROOT`.
- Env vars are the unprefixed set; all model/URL reads via `hermes.llm` helpers; DB via `resolve_db_path(None)`.
- `uv run pytest`, `uv run ruff check .`, `uv run ty check` all green after every task; radon: no new C-rated function (keep step functions small).
- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: `install.py` — step engine, detection steps, doctor mode

**Files:** Create `src/hermes/install.py`, `tests/test_install.py`.

**Interfaces produced (later tasks + cli rely on exact names):**

```python
@dataclass
class StepResult:
    name: str
    status: str  # "OK" | "FIXED" | "BROKEN" | "SKIPPED"
    detail: str = ""

def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess  # single seam; check=False, capture_output=True, text=True
def _find_agent_binary() -> str | None
def _is_ollama_backend() -> bool          # base_url() host is localhost-ish ollama root (native root reachable is checked in the step, not here)
def run_install(check: bool = False, yes: bool = False, now: bool = False) -> int
```

Steps implemented in this task (detection + the pure-local fixes):
`step_uv`, `step_ollama_reachable`, `step_models` (parses `ollama list` via `_run`; pull with consent — consent = `yes` flag or interactive `input()`; in `--check` mode report BROKEN instead of pulling), `step_env_file` (create from `.env.sample` next to `llm._REPO_ROOT`; in `--check` report), `step_first_data` (items count via `get_db(resolve_db_path(None))`; ingest via `_run(["uv","run","hermes","ingest"], cwd=root)`; untagged count reported; `now` → also run tag), `step_warmup` (import and call `cli.warmup()` inside try/except, SKIPPED on failure).

`run_install` executes steps in spec order, prints one aligned line per step (`ok` green-less plain text is fine: `[ok] models: hermes3:3b, embeddinggemma present`), returns 0 iff no BROKEN.

Agent-binary discovery (verbatim):

```python
def _find_agent_binary() -> str | None:
    """hermes-agent's CLI — NOT our console script (which shadows it inside the venv)."""
    for d in os.get_exec_path():
        cand = Path(d) / "hermes"
        if (
            cand.is_file()
            and os.access(cand, os.X_OK)
            and not str(cand.resolve()).startswith(sys.prefix)
        ):
            return str(cand)
    fallback = Path.home() / ".local" / "bin" / "hermes"
    return str(fallback) if fallback.is_file() else None
```

**Required tests (TDD; fake `_run` records calls and returns canned outputs; `llm._REPO_ROOT` monkeypatched to tmp_path with a `.env.sample` fixture file):**
- `--check` with everything present → exit 0, zero mutating `_run` calls (assert recorded commands are all read-only: `ollama list` etc.).
- `--check` with missing model → that step BROKEN, exit 1, no `ollama pull` recorded.
- Full run with missing model + `yes=True` → `ollama pull <model>` recorded once, step FIXED.
- `.env` absent → created from sample (FIXED); present → OK and file untouched (mtime/content compare).
- Second full run after fixes → all OK, zero mutating calls (idempotency).
- `_find_agent_binary` skips a fake `hermes` inside `sys.prefix` and finds one in another PATH dir (build both under tmp_path, monkeypatch `os.get_exec_path` and `sys.prefix`).
- Non-Ollama `LLM_BASE_URL` (e.g. `https://openrouter.ai/api/v1`) → model/ollama/warmup steps SKIPPED, others still run.

Commit: `feat: hermes install core — step engine, detection, doctor mode`

---

### Task 2: agent wiring + scheduling steps, CLI dispatch

**Files:** Modify `src/hermes/install.py`, `src/hermes/cli.py` (parser: `install` subcommand with `--check/--yes/--now`; dispatch `hermes.install.run_install(...)` module-attribute style), `tests/test_install.py`, `tests/test_cli.py` (parser test + dispatch monkeypatch test mirroring the existing `tag` pattern).

Steps: `step_mcp_wiring` (agent binary via `_find_agent_binary`; `hermes mcp list` → add if `hermes-rss` absent, exact add command from spec step 6 with `--args run --project <root> hermes-mcp`), `step_skill_copy` (file-level sync of `SKILL.md` + `scripts/*` from checkout to `~/.hermes/skills/science-recommendations/`, creating dirs; compare content before copy so unchanged = OK not FIXED; NEVER list/touch `data/`), `step_reasoning_override` (only when Ollama backend and chat model matches `hermes3*`: `hermes config get agent.reasoning_overrides.<model>`; unset → `hermes config set ... none`; nonzero rc → SKIPPED with printed YAML snippet), `step_schedule` (write refresh script — exact content from Global Constraints — chmod 0o755, content-compare for idempotency; `hermes cron list` guard; exact `cron create` command). All agent-dependent steps SKIPPED when `_find_agent_binary()` is None. `--check` variants detect-only.

**Required tests:** mcp add guarded (present in canned `mcp list` output → no add recorded; absent → exact add argv recorded); skill sync creates files, second run OK, a planted `data/hermes.db` under the fake `~/.hermes` skill dir survives byte-identical; refresh script exact content + exec bit + rewrite only on content change; cron create exact argv + guarded by canned `cron list`; agent binary None → all four steps SKIPPED and exit still 0 when everything else ok; `hermes install --check` never invokes mcp add/config set/cron create/copies. Home-dir isolation: monkeypatch `Path.home()` to tmp_path in every test in this task.

**Also in this task — the e2e installation demonstration test (user-requested), `tests/test_install_e2e.py`:**

A pytest that exercises installation for real — no `_run` mocking. Build a
sandbox per test: `monkeypatch.setattr(Path, "home", ...)` → tmp "home";
a PATH dir (prepended via monkeypatch.setenv) containing **real stub
executables** written by the test: `hermes` (bash: appends `"$@"` to a
log file; answers `mcp list`/`cron list` from state files so the second
run sees what the first registered; exits 0) and `ollama` (bash: `list`
prints the configured models). Point `RSS_DB` at a pre-seeded tmp DB (one
item via the existing test helpers) so first-data is OK, and satisfy the
Ollama reachability probe with a stdlib `http.server` thread on a random
port with `LLM_BASE_URL=http://127.0.0.1:<port>/v1` (or, if Task 1
implemented the probe as a small function, monkeypatch that one function —
prefer the real HTTP server). Then:

1. **Fresh install demonstrates end-to-end**: `run_install(yes=True)` →
   exit 0; assert `.env` created at the (repointed) repo root; skill files
   present under fake `~/.hermes/skills/science-recommendations/`; refresh
   script exists, executable, exact content; stub log contains exactly one
   `mcp add hermes-rss ...` and one `cron create ... --no-agent`.
2. **Idempotency for real**: second `run_install(yes=True)` → exit 0 and
   the stub log gained NO new mutating invocations (list calls only).
3. **Doctor honesty**: wipe the fake home, `run_install(check=True)` →
   exit 1, and zero mutations in the stub log, no files created.
4. **CLI path**: `main(["install", "--check"])` through the real parser
   dispatches and returns the same exit code (proves wiring, not just the
   function).

A planted `data/hermes.db` byte-blob in the fake skill dir must survive
runs 1 and 2 unchanged (the never-touch-data/ guarantee, demonstrated).

Commit: `feat: hermes install — mcp/skill/reasoning/cron wiring + e2e demonstration test`

---

### Task 3: guide + docs — SKILL.md config contract, setup.sh delegator, README one-liner

**Files:** Modify `skills/science-recommendations/SKILL.md`, `skills/science-recommendations/scripts/setup.sh`, `README.md`. Test: none new (docs) — but full suite must stay green and `bash -n setup.sh` clean.

- SKILL.md: Setup section → "run `uv run hermes install --check` from the project dir; on gaps run `uv run hermes install --yes`" (keep the uvx-fallback paragraph, pointing it at `uvx --from git+<repo_url> hermes install --yes`). Add the **Configuration contract** table verbatim from the spec. Keep the reasoning_overrides prose and HTTP quick-reference (fallback lane) intact.
- setup.sh: replace the body's check-and-repair with a thin delegator preserving the existing PROJECT_DIR/uvx-fallback resolution: local checkout → `exec uv run --project "${PROJECT_DIR}" hermes install --yes`; uvx path → `exec uvx --from "git+${REPO_URL}" hermes install --yes`; keep the fail() helpers for uv-missing / no-checkout-no-repo-url cases. Everything else (model checks, server probe, ingest) is now the installer's job — delete it here.
- README Installation: lead with the one-liner (`uvx --from git+<REPO_URL> hermes install` and local `uv run hermes install`, `--check` mention); demote the manual steps under "What install does / manual setup". Update "Launching alongside hermes-agent" intro to note steps 1/3/6 are automated by install (keep the sections as reference).

Commit: `docs: SKILL.md config contract + setup.sh delegates to hermes install; README one-liner`

---

## Post-merge verification (live, this machine — this IS "fix it all" for the user's env)

1. `uv run hermes install --check` → expect gaps (stale skill copy, no cron job).
2. `uv run hermes install --yes` → fixes: skill re-copied (data/ untouched — verify mtime), cron job registered (`hermes cron list`), mcp already-ok, .env already-ok.
3. `uv run hermes install --yes` again → all OK (idempotency, live).
4. `hermes cron run hermes-rss-refresh` (or wait for :17) → silent success; `hermes cron runs` shows it.
