---
name: attestation-setup
description: "Install, repair and reload the local attestation engine (models, database, MCP server, agent surfaces) that the attestation-feed, attestation-provenance, attestation-knowledge and attestation-symbolic skills talk to; use when a tool is missing, stale or refusing to connect."
version: 2.0.0
author: attestation project
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [setup, install, mcp, local-api]
    related_skills: [attestation-feed, attestation-provenance, attestation-knowledge, attestation-symbolic]
---

# attestation: setup and repair

**attestation** is a local research-provenance engine: an experiment ledger
read from artifacts on disk, a claim checker for Markdown, a citation
resolver over Zotero and `.bib` files, a knowledge graph of the reader's own
items, sandboxed symbolic algebra, and a personalised science feed. Every
part of it runs on this machine. The tools reach an agent as native MCP
calls; this skill is the setup lane and the map. The judgment for each
surface lives in its sibling:

| Surface (`ATTEST_TOOLS=`) | Skill | Sees |
|---|---|---|
| `feed` | `attestation-feed` | `feed.*` |
| `provenance` | `attestation-provenance` | `runs.*`, `cite.check` |
| `knowledge` | `attestation-knowledge` | `kg.*`, `feed.search`, `cite.*` |
| `symbolic` | `attestation-symbolic` | `sym.*` |

Unset, the server serves every tool (a count that moves -- re-measure rather
than quoting one). An unknown `ATTEST_TOOLS` value **raises** rather than
falling back: a restriction that quietly stopped restricting is the failure
worth preventing, so a typo takes the server down loudly.

**A person picks the surface at launch; you never pick one at runtime.**
That is measured, not stylistic: a model choosing the namespace and then the
tool scored 7.3/15 against 13/15 for routing inside a fixed surface, at twice
the latency. If a question falls outside your surface, say which agent
answers it rather than trying.

## Setup

Before the first call in a session, run the idempotent installer/doctor
through this skill's wrapper, which resolves the checkout for you:

```bash
bash ${HERMES_SKILL_DIR}/scripts/setup.sh
```

`setup.sh` is a thin delegator: it resolves a local checkout (or falls back
to `uvx --from git+https://github.com/mgoldey/attestation attest install --yes`, below) and execs
`uv run attest install --yes`. From the project directory the installer runs
directly:

```bash
uv run attest install --check   # diagnose only, exit 1 on gaps, changes nothing
uv run attest install --yes     # non-interactive repair of any gaps found
```

`--check` prints one line per step (`ok` / `BROKEN` / `skipped`) and never
mutates anything. `--yes` fixes gaps without prompting: pulls missing Ollama
models, creates `.env` from `.env.sample`, runs the first ingest, and wires
the MCP server, the five skills (into `~/.hermes/skills/` **and** every
`~/.hermes/profiles/*/skills/` that exists), the reasoning override and the
refresh cron job. If `setup.sh` exits non-zero, read its one-line reason and
fix that; do not retry blindly.

A skill you disabled by renaming its `SKILL.md` stays disabled across
installs. The superseded single `research-provenance` skill, if still
installed, is disabled the same way (`SKILL.md.superseded-by-attestation-split`)
so it leaves the index without anything in its directory being deleted.

The web UI, if wanted:

```bash
cd "${HERMES_RSS_PROJECT_DIR:-<checkout>}" && uv run attest serve &
```

### Data directory

The live database is `~/.hermes/skills/science-recommendations/data/hermes.db`
-- the *skill* was renamed twice since, the *database* path deliberately was
not, so no database created before a rename is orphaned. Resolution order
(`resolve_db_path` in `src/attestation/db.py`):

1. explicit `--db <path>`
2. `ATTEST_DB` (`RSS_DB`, the pre-rename name, still works)
3. the path above, if that file already exists
4. `./hermes.db`

### Running without a local checkout (uvx)

When `setup.sh` finds no checkout at `HERMES_RSS_PROJECT_DIR` (default: the
checkout the script itself lives in), it runs the installer straight from
git, with no clone step:

```bash
uvx --from git+https://github.com/mgoldey/attestation attest install --yes
```

A `science_recommendations.repo_url` key in `~/.hermes/config.yaml` overrides
that URL, for a fork. The package name (`attestation`) and its console script
(`attest`) differ: `uvx --from <package>` takes the *package* and the trailing
word is the *executable*, so `uvx --from . attestation` fails with "An
executable named `attestation` is not provided by package `attestation`".

### Configuration contract

| Setting | Store | Written by |
|---|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `CHAT_MODEL`, `EMBED_MODEL`, `EMBED_DIMS`, `ATTEST_DB` | `<checkout>/.env` (real env wins) | `attest install` / user edit |
| `mcp_servers.attestation` and the four `attestation-<surface>` entries | `~/.hermes/config.yaml` | `hermes mcp add` (install) / `attest emit --write` |
| `agent.reasoning_overrides.<model>` | `~/.hermes/config.yaml` | `hermes config set` (install) |
| live DB | see Data directory | engine |
| refresh schedule | `~/.hermes` cron store + `~/.hermes/scripts/attestation-refresh.sh` | install |
| `ATTEST_TOOLS`, `ATTEST_EXPAND`, `ATTEST_CITATION_WEB` | server environment (`.env` or the MCP entry's `env:`) | user edit |

`attest emit` generates the per-surface config -- the four
`attestation-<surface>` MCP entries (disabled by default) and matching
`.claude/agents/attestation-<surface>.md` files. It reports drift by default,
writes only with `--write`, and never overwrites a file you edited.

### Only for hermes3 models

The default `gemma4:e2b` accepts `think: true`, so nothing here applies to
it. With `hermes3:8b` on an Ollama endpoint, every tool-calling turn fails
with `HTTP 400: "hermes3:8b" does not support thinking` unless reasoning is
disabled for that model. `attest install` writes the override when
`CHAT_MODEL` matches `hermes3*`:

```yaml
agent:
  reasoning_overrides:
    hermes3:8b: none
```

Ad hoc, pass `--reasoning none`.

## Your session may only see the routers

Each surface hides its specific tools by default and serves two: its `.ask`
router and a `<surface>.tools` disclosure tool. Measured over 26 turns, a
model picked the router 1 time in 26 with the specifics listed beside it and
26 in 26 with them absent. `<surface>.tools` says how to reveal them
(`ATTEST_EXPAND=1` on the server). A tool a sibling skill documents that is
not in your list is hidden, not broken: use the router.

## When the tools change under you

**An MCP server never reloads.** One is spawned per session and holds that
code until it dies, so edits to the checkout are invisible to a session that
started before them; both live servers here were once found running code
five commits stale.

```bash
uv run attest reload   # SIGTERMs every running attest-mcp
```

Respawn is **lazy**: nothing restarts for at least ten seconds and the new
process appears when a tool is next called, so the first call after a reload
is slower and a reload followed by no call leaves the server down. That is
fine; do not read it as a failure. `hermes mcp test` does not catch
staleness -- it spawns a fresh process and reports the code on disk. If a
documented tool is missing or behaves like an older version of itself,
reload before concluding it is broken.

## HTTP fallback (feed only)

If MCP is disconnected, the web server at `http://127.0.0.1:8899` covers the
list/rate/explain path and nothing else. No auth, no API key.

| Action | Call |
|---|---|
| Ranked feed | `GET /list?user=<name>` → an HTML `<ol>` fragment; read each `<li>`'s title, `href`, source and `data-item-id`. Present a clean list, never the HTML. |
| Mark useful / not | `POST /clicks` with form fields `user`, `item_id`, `useful=1` or `useful=0` -- an integer; `useful=true` returns HTTP 422. |
| Why it ranked here | `GET /explanation?user=<name>&item_id=<id>` → one sentence; a local model call, so seconds are normal. Never let it block the list or a click. |

The persona is whatever name you already have for the reader; the first
call creates it. If a call cannot connect, the server is not running: rerun
`setup.sh`, or start it with the `attest serve` line above.
