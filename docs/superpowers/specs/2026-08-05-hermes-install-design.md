# `hermes install` — one-command setup + agent config guide — design

**Date:** 2026-08-05
**Status:** approved (user: "fold it into the hermes install brainstorm" + "fix it all"; scope and scheduling answers captured below)

## Problem

Onboarding takes ~6 manual README steps across three config stores (`.env`
in the checkout, `mcp_servers`/`reasoning_overrides` in
`~/.hermes/config.yaml`, the skill copy + DB under `~/.hermes/skills/`).
hermes-agent's own guide (SKILL.md/setup.sh) predates the `.env`/MCP world
and its installed copy goes stale. The user wants: one command that ties a
hermes-agent user to the right `~/.hermes` settings, more elegant than a
quickstart script, plus an agent-readable statement of where the right
config lives.

## Decision

A new `hermes install` CLI subcommand (module `src/hermes/install.py`,
dispatched from `cli.py`) that is **idempotent** (re-run = repair) and has a
**`--check` doctor mode** (diagnose, change nothing, exit 1 on gaps).
Invocable before cloning: `uvx --from git+<url> hermes install`.

User-approved scope (all four): env+models+first-data, hermes-agent wiring,
scheduling, doctor mode. Scheduling lives in **hermes cron** (user choice),
using the verified `hermes cron create SCHEDULE --name N --script S
--no-agent` form — the refresh script runs on schedule with no LLM turn;
stdout is delivered only when non-empty.

## Steps (each idempotent; prints `ok` / `fixed` / `SKIPPED (reason)` per step)

1. **Prereqs**: `uv` on PATH; Ollama reachable at the native root derived
   from `base_url()` (strip `/v1`). Non-Ollama backend → model/warmup steps
   SKIPPED, everything else proceeds.
2. **Models**: `chat_model()` and `embed_model()` present in `ollama list`;
   pull missing ones (prompt for consent unless `--yes`).
3. **.env**: create from `.env.sample` at the checkout root if absent.
4. **First data**: if the resolved DB has no items → run ingest; report
   untagged count (tagging happens via the schedule; `--now` runs it
   inline).
5. **Warmup**: best-effort, reuses `cli.warmup()`'s graceful-skip behavior.
6. **MCP wiring**: if `hermes-rss` absent from `hermes mcp list` → run
   `hermes mcp add hermes-rss --command uv --args run --project <root>
   hermes-mcp`.
7. **Skill copy**: sync `SKILL.md` + `scripts/` into
   `~/.hermes/skills/science-recommendations/`, **never touching `data/`**
   (file-level copies, not directory replace).
8. **reasoning_overrides**: if the chat model is served via Ollama and
   `agent.reasoning_overrides.<model>` is unset, attempt
   `hermes config set`; on any failure print the exact YAML the user should
   add (never edit `config.yaml` directly ourselves).
9. **Schedule**: write `~/.hermes/scripts/hermes-rss-refresh.sh`
   (`cd <root> && uv run hermes ingest && uv run hermes tag`, failures to
   stdout so hermes delivers them, success silent), then if no
   `hermes-rss-refresh` job in `hermes cron list` →
   `hermes cron create "17 * * * *" --name hermes-rss-refresh --script
   hermes-rss-refresh.sh --no-agent`.
10. **Summary**: table of step → status; exit 0 iff nothing broken.

### The agent-binary trap (load-bearing)

Inside our venv, `hermes` on PATH is **our own CLI** — a naive
`subprocess.run(["hermes", "mcp", ...])` would re-invoke ourselves.
`_find_agent_binary()` iterates `os.get_exec_path()`, skips entries under
`sys.prefix` (our venv), and returns the first `hermes` executable found
(fallback `~/.local/bin/hermes`). No agent binary → steps 6–9 SKIPPED with
a clear message; the engine still works standalone.

## Flags

`--check` (doctor: detect only, exit 1 on gaps), `--yes` (non-interactive
consent — what SKILL.md tells the agent to use), `--now` (also run the tag
backfill inline instead of leaving it to the schedule).

## Config contract (the folded-in guide — verbatim into SKILL.md)

| Setting | Store | Written by |
|---|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `CHAT_MODEL`, `EMBED_MODEL`, `EMBED_DIMS`, `RSS_DB` | `<checkout>/.env` (real env wins) | `hermes install` step 3 / user edit |
| `mcp_servers.hermes-rss` | `~/.hermes/config.yaml` | `hermes mcp add` (install step 6) |
| `agent.reasoning_overrides.<model>` | `~/.hermes/config.yaml` | `hermes config set` (install step 8) |
| live DB | `~/.hermes/skills/science-recommendations/data/hermes.db` | engine (resolve_db_path default) |
| refresh schedule | `~/.hermes` cron store + `~/.hermes/scripts/hermes-rss-refresh.sh` | install step 9 |

SKILL.md's Setup section becomes: "run `uv run hermes install --check`;
if it reports gaps, run `uv run hermes install --yes`" + this table.
`setup.sh` becomes a thin delegator (`exec uv run hermes install --yes`
from the project dir, keeping the uvx fallback resolution it already has)
so the guide and the installer cannot drift.

## README

Installation section leads with the one-liner
(`uvx --from git+<REPO_URL> hermes install` / local `uv run hermes
install`); the manual steps remain as "what install does" reference.

## Implementation shape

`src/hermes/install.py`: small step functions, each returning
`StepResult(name, status, detail)` with `status ∈ {OK, FIXED, BROKEN,
SKIPPED}`; an orchestrator `run_install(check=False, yes=False, now=False)
-> int` (exit code). All subprocess calls (`ollama`, agent `hermes`,
`uv run hermes ingest/tag`) go through one `_run()` seam so tests
monkeypatch a single function. `cli.py` gains only parser wiring +
dispatch (it is already radon-C; no logic lands there).

## Testing

Unit tests with a fake `_run()` recording invocations: idempotency (second
run performs zero mutating calls), `--check` never mutates, MCP add guarded
by list, skill sync never touches `data/`, agent-binary discovery skips
`sys.prefix`, refresh-script content and cron-create arguments exact,
non-Ollama base URL skips model/warmup steps, missing agent binary skips
wiring with SKIPPED not BROKEN. A live end-to-end run on this machine is
the post-merge verification (and is how the user's own environment gets
fixed).

## Out of scope (YAGNI)

Uninstall, migrations of old crontab lines, multi-checkout management,
Windows, the morning-digest hermes-cron routine (add later as
`--with-digest` if wanted).
