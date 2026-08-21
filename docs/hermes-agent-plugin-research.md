# hermes-agent: plugin vs skill, and packaging hermes-rss as native integration

Researched against the locally installed hermes-agent **v0.20.0** at
`~/.hermes/hermes-agent` (confirmed via `pyproject.toml` `version = "0.20.0"`
and `~/.hermes/.update_check`), plus the public GitHub repo, the
`0xNyk/awesome-hermes-agent` list, and the hosted docs at
`hermes-agent.nousresearch.com`. All local file paths below are read-only
inspections of the installed copy at `~/.hermes/hermes-agent`.

## Plugin vs skill (what each actually is in this codebase)

**Skill** = a directory of markdown + optional scripts, loaded by the agent's
own reasoning as *procedural knowledge to read and follow*. There is no
Python code that the host imports — a skill is prompt content plus,
optionally, shell scripts the agent decides to run via the terminal tool.
This is exactly what `~/.hermes/skills/science-recommendations/SKILL.md`
already is: instructions telling the model "curl this local FastAPI server,
here's the shape of the response." Skills are self-improving in this
project's design — the README describes "autonomous skill creation after
complex tasks" and skills "self-improve during use" (agent-curated, not
developer-curated).

**Plugin** = an installed Python package (bundled, user, project, or pip
entry-point) that the *host process* imports and that calls back into
host-provided registration APIs. A plugin is code, not prose. Concretely,
each plugin directory (e.g. `~/.hermes/hermes-agent/plugins/spotify/`)
contains:

- `plugin.yaml` — manifest (`name`, `version`, `description`, `author`,
  `kind`, `provides_tools`, `hooks`, `requires_env`)
- `__init__.py` — must expose a `register(ctx)` function
- arbitrary supporting `.py` modules (`tools.py`, `client.py`, ...)

`register(ctx)` receives a `PluginContext` (defined in
`~/.hermes/hermes-agent/hermes_cli/plugins.py`) that exposes ~15 registration
methods: `register_tool()`, `register_hook()`, `register_middleware()`,
`register_cli_command()`, `register_command()` (in-chat slash commands),
`register_context_engine()`, `register_image_gen_provider()`,
`register_web_search_provider()`, `register_browser_provider()`,
`register_tts_provider()`, `register_transcription_provider()`,
`register_secret_source()`, `register_dashboard_auth_provider()`,
`register_video_gen_provider()`, `register_platform()`,
`register_slack_action_handler()`, `register_auxiliary_task()`,
`register_skill()` (yes — a plugin can register its own namespaced skill,
`<plugin_name>:<skill_name>`, that stays out of the flat skills tree).

**The practical distinction that matters for us:** a skill tells the *model*
what to do (imperative prose the agent interprets each turn, at the cost of
context-window tokens and interpretation risk — e.g. our SKILL.md's
documented `reasoning_effort` footgun). A plugin gives the *host* a
structured, typed, function-calling tool schema (`provides_tools` +
`SPOTIFY_PLAYBACK_SCHEMA`-style JSON Schema dicts, see
`plugins/spotify/tools.py` lines 328-454) that's injected into the model's
tool-calling surface exactly like a built-in tool — no prose interpretation,
no curl-command hallucination risk, and it can run outside a single
turn (hooks, background lifecycle).

## Extension points found (with file paths / line refs in the local install)

All paths are under `~/.hermes/hermes-agent/` unless noted.

- **Plugin manifest schema** — `hermes_cli/plugins.py` lines 280-314
  (`PluginManifest` dataclass). Valid `kind` values (line 277):
  `standalone`, `backend`, `exclusive`, `platform`, `model-provider`.
- **Plugin loader / discovery order** — `hermes_cli/plugins.py`
  `PluginManager._discover_and_load_inner()`, lines 1336-1391. Four sources,
  later overrides earlier on name collision:
  1. Bundled: `<repo>/plugins/<name>/`
  2. User: `~/.hermes/plugins/<name>/` — **this is where our engine's
     plugin would go**
  3. Project: `./.hermes/plugins/<name>/` (opt-in via
     `HERMES_ENABLE_PROJECT_PLUGINS`)
  4. Pip: packages exposing the `hermes_agent.plugins` entry-point group
- **Opt-in gating** — standalone/user-installed plugins are gated by
  `plugins.enabled: [...]` in `config.yaml` (lines 1469-1477 of
  `plugins.py`). **Our locally installed `~/.hermes/config.yaml` currently
  has no `plugins:` key at all** — confirmed by direct read, so zero
  third-party plugins are enabled today; only bundled `backend`/`platform`
  kinds auto-load.
- **Lifecycle hooks (`VALID_HOOKS`)** — `hermes_cli/plugins.py` lines
  135-215. Full set: `pre_tool_call`, `post_tool_call`,
  `transform_terminal_output`, `transform_tool_result`,
  `transform_llm_output`, `pre_llm_call`, `post_llm_call`, `pre_verify`,
  `pre_api_request`, `post_api_request`, `api_request_error`,
  `on_session_start`, `on_session_end`, `on_session_finalize`,
  `on_session_reset`, `subagent_start`, `subagent_stop`,
  `pre_gateway_dispatch`, `pre_approval_request`, `post_approval_response`,
  `kanban_task_claimed`, `kanban_task_completed`, `kanban_task_blocked`.
  Real example: `plugins/disk-cleanup/plugin.yaml` declares
  `hooks: [post_tool_call, on_session_end]` and needs no agent action —
  this is the "background job" pattern most relevant to a ranking engine
  that wants to react to session/tool events without being explicitly
  invoked.
- **Tool registration** — `PluginContext.register_tool()`,
  `hermes_cli/plugins.py` lines 410-465, delegates to
  `tools.registry.registry.register()` (`tools/registry.py`, not read in
  full — referenced from `plugins/spotify/tools.py` line 17). Confirmed
  real usage: `plugins/spotify/tools.py` defines 7 tool schemas
  (`SPOTIFY_PLAYBACK_SCHEMA` etc., lines 328-454) each a plain JSON-Schema
  `{"name", "description", "parameters"}` dict — i.e. exactly OpenAI/
  Anthropic-style function-calling schemas, not curl instructions.
- **CLI discovery commands** — `hermes plugins list`,
  `hermes plugins install owner/repo`, or "drop a plugin directory into
  `~/.hermes/plugins/`" (message text found at `cli.py` line 10212, in the
  `/plugins` slash-command handler).
- **No formal Python base class for a "Plugin"** — registration is
  duck-typed via the `register(ctx)` entry point; provider-style extensions
  (image gen, web search, browser, TTS, STT) do have ABCs
  (`ImageGenProvider`, `WebSearchProvider`, `BrowserProvider`,
  `TTSProvider`, `TranscriptionProvider`, `SecretSource` — all imported
  from `agent.*_provider` modules in `plugins.py`), but a plain
  function-calling tool plugin like `spotify` needs none of that — it's
  just `register_tool()` calls.

## MCP support (yes — first-class, and organizationally separate from plugins)

**Yes, hermes-agent has native MCP client support**, and it is a
*third, distinct* extension mechanism alongside skills and plugins, not a
special case of either:

- Config lives under a top-level `mcp_servers:` key in `~/.hermes/config.yaml`
  (confirmed absent in our current config — no MCP servers configured
  locally right now; the key/mechanism exists in the codebase regardless,
  per `hermes_cli/agent_import.py` lines 24-29, 481-804 which import
  `mcpServers` from Claude Code / Codex config formats into this same key).
- CLI: `hermes mcp add <name> --command <cmd> --args ... --env KEY=VALUE`
  for **stdio-transport local servers**, or `--url` for HTTP/SSE remote
  servers, or `--preset <name>` for known integrations. Full arg parser
  read at `hermes_cli/subcommands/mcp.py` lines 44-73 — confirms `--command`
  (dest `mcp_command`, "Stdio command (e.g. npx)"), `--args`
  (`nargs=REMAINDER`), `--env` (`nargs="*"`, `KEY=VALUE` pairs), `--auth
  {oauth,header}`, `--connect-timeout`.
  UNVERIFIED (from hosted docs, not the local install): the exact
  `mcp_servers.<name>` YAML key names (`command`, `args`, `env`, `timeout`,
  `connect_timeout`, `idle_timeout_seconds`, `max_lifetime_seconds`) — the
  docs fetch returned these but I did not find a `mcp_servers:` block in any
  locally-loaded config to cross-check field names directly. Treat key names
  as probably-right, verify against `hermes mcp add --help` output before
  committing config.
- There is also a real **bundled MCP catalog** —
  `~/.hermes/hermes-agent/optional-mcps/<name>/manifest.yaml`, e.g.
  `optional-mcps/linear/manifest.yaml` (read in full): `manifest_version: 1`,
  `transport: {type: http, url: ...}`, `auth: {type: oauth}`,
  `post_install:` free text. This is a **curated list of Nous-approved
  remote MCP servers** users one-click install via `hermes mcp install
  <name>` / `hermes mcp catalog` / `hermes mcp picker` — not a mechanism for
  registering our own local server (we'd use `hermes mcp add`, not the
  catalog, since we're not submitting a PR to hermes-agent's repo).
- hermes-agent's own `pyproject.toml` pins the official MCP Python SDK:
  `mcp = ["mcp==1.28.1", "starlette==1.3.1"]` (grep-confirmed, lines
  233-251) — used both for its MCP *client* (connecting out to servers like
  ours) and for `hermes mcp serve` (exposing Hermes itself as an MCP
  server to other agents — the reverse direction, not relevant here).
- The `mcp` PyPI package (same official SDK, `FastMCP` included) is NOT
  currently a dependency of `/home/matt/hermes-rss` — confirmed via
  `python3 -c "import mcp"` failing locally and no `mcp` entry in
  hermes-rss's `pyproject.toml`. It would need to be added.

**Bottom line: MCP is the right "native integration" lane for us, not the
hermes-agent plugin system.** A hermes-agent plugin requires our code to
live *inside* `~/.hermes/plugins/` as a hermes-agent-specific Python module
that imports `hermes_cli.plugins`, `tools.registry`, etc. — i.e. it couples
hermes-rss to hermes-agent's internal APIs and versions. An MCP server is a
standard, host-agnostic protocol boundary: hermes-rss stays a fully
independent FastAPI process/package, and hermes-agent (or Claude Desktop, or
anything else MCP-capable) just connects to it over stdio or HTTP. Given
hermes-rss is already a standalone FastAPI service at 127.0.0.1:8899 with a
CLI, MCP is strictly less invasive than a plugin for the same "native
function-calling tool" benefit.

## Recommended packaging path for our engine

Wrap hermes-rss as an MCP **stdio** server (not a plugin, not the HTTP/OAuth
catalog path — we don't need remote transport or OAuth for a loopback
personal tool).

Concrete plan:

1. **New file**: `/home/matt/hermes-rss/src/hermes_rss/mcp_server.py`
   using `mcp.server.fastmcp.FastMCP` (part of the `mcp` SDK, same
   `mcp==1.28.1` version hermes-agent itself pins — match it to avoid a
   protocol-version mismatch). Define 3-4 `@mcp.tool()`-decorated async
   functions that call directly into the existing engine internals (not
   over HTTP-to-self — import the ranking/feedback functions from the
   `hermes-rss` package directly, since both processes would otherwise be
   redundant):
   - `list_recommendations(user: str, limit: int = 10) -> list[dict]`
   - `mark_feedback(item_id: str, useful: bool) -> dict`
   - `explain_ranking(item_id: str) -> str`
   Reuse the same request/response shapes the FastAPI routes already use
   (check `src/` for the existing route handlers — likely
   `src/hermes/api.py` or similar, not yet inspected in this pass) so the
   MCP tool layer is a thin adapter, not a reimplementation. If those
   functions currently assume an already-running FastAPI app/DB
   connection, the MCP server module should perform the same
   engine-init/DB-open sequence `scripts/setup.sh` does today, or simply
   `import httpx` and proxy to the already-running `127.0.0.1:8899`
   instance if in-process reuse turns out messy — either is legitimate;
   in-process is preferred to avoid running two servers.
2. **New CLI entry point** in `pyproject.toml`:
   `hermes-rss-mcp = "hermes_rss.mcp_server:main"` (or reuse the existing
   `hermes` script with a `hermes mcp-serve` subcommand — smaller surface,
   avoids a second console_script).
3. **New dependency**: add `mcp>=1.28.1` (pin loosely; hermes-agent pins
   exactly 1.28.1 for its own client, but MCP's wire protocol is
   versioned/negotiated, so exact pinning on the server side is not
   required — still worth testing against 1.28.1 specifically since
   that's what's actually installed locally) to hermes-rss's
   `pyproject.toml` dependencies (not dev-only — this ships to users).
4. **Registration**, one-time per machine:
   ```bash
   cd /home/matt/hermes-rss
   uv sync
   hermes mcp add hermes-rss \
     --command uv \
     --args run hermes-rss-mcp \
     --connect-timeout 10
   ```
   This writes an `mcp_servers.hermes-rss` block into
   `~/.hermes/config.yaml` (exact key shape UNVERIFIED locally — see MCP
   section above; confirm via `hermes mcp list` after adding, and via
   `cat ~/.hermes/config.yaml` — read-only — before assuming field names).
5. **Retire (or keep as fallback) the existing SKILL.md** — once the MCP
   tools are registered and confirmed working (`hermes mcp test
   hermes-rss`, then a live chat turn), `science-recommendations` becomes
   redundant for the core list/feedback/explain actions. Recommend keeping
   the skill only if it documents workflow/judgment ("when NOT to use
   this," the reasoning_effort gotcha) that doesn't fit into a tool
   description — MCP tool `description` fields are far more
   space-constrained than a SKILL.md. A slim skill that says "prefer the
   `list_recommendations`/`mark_feedback`/`explain_ranking` MCP tools; only
   fall back to the CLI/curl path in `scripts/setup.sh` if MCP is
   disconnected" is a reasonable hybrid — do not delete the operational
   knowledge (the `reasoning_effort: medium` + Ollama incompatibility
   note), just shrink the "how to call it" section.

## Effort estimate

- **Small, well-scoped task, roughly half a day to a day of focused work**
  for someone already familiar with the hermes-rss codebase:
  - ~1-2 hrs: write `mcp_server.py` wrapping the 3 core operations with
    `FastMCP`, using the official SDK's stdio quick-start pattern
    (UNVERIFIED exact API surface of `mcp==1.28.1`'s `FastMCP` — the SDK
    has had breaking changes across 1.x; check the installed version's
    docs/changelog before writing code, don't assume today's `pip show mcp`
    examples match 1.28.1 exactly).
  - ~30 min: `pyproject.toml` + entry point wiring, `uv sync`.
  - ~30 min: `hermes mcp add` + `hermes mcp test` + a live smoke-test chat
    turn exercising each tool.
  - ~1-2 hrs: buffer for the in-process-vs-proxy-to-existing-server
    decision (item 1 above) — if the existing FastAPI route handlers
    aren't cleanly importable as plain functions (e.g. they're tightly
    coupled to `Request`/`Depends` FastAPI plumbing), the MCP layer may
    need light refactoring of `src/` to extract a service-layer function
    each route and each MCP tool can both call. This is the main variable
    that could push the estimate to 1-2 days — **not assessed yet**; would
    require reading hermes-rss's actual route implementation, which was
    out of scope for this research pass.
  - No hermes-agent-side changes needed — MCP client support is already
    installed and working (per `pyproject.toml` pin), so this is 100%
    hermes-rss-side work plus one `hermes mcp add` command.
- Compare to the plugin path (not recommended, but for scale): building an
  actual `~/.hermes/plugins/hermes-rss/` plugin would mean importing
  `tools.registry`, matching hermes-agent's exact JSON-Schema tool format,
  handling the `override`/trust-gate logic if any tool name collides, and
  re-testing against every hermes-agent minor version bump (plugin API is
  internal, not a stable public contract) — meaningfully more ongoing
  maintenance for no additional capability we actually need (we don't need
  hooks into `pre_tool_call`/session lifecycle/platform adapters — we need
  three clean function-calling tools, which MCP already gives us for free
  with a stable external protocol).

---

**Sources**: local read-only inspection of `~/.hermes/hermes-agent`
(v0.20.0, confirmed via `pyproject.toml`) and `~/.hermes/config.yaml`;
`https://github.com/NousResearch/hermes-agent` (README, via WebFetch);
`https://github.com/0xNyk/awesome-hermes-agent` (via WebFetch);
`https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp` (via
WebFetch — flagged UNVERIFIED items above were sourced here, not from a
local file read). The `mudrii/hermes-agent-docs` mirror was not fetched
separately in this pass since the primary docs site and local source
already answered the plugin/MCP questions with higher confidence (direct
code read beats a docs mirror); revisit that source if a
developer/plugin-authoring guide is needed beyond what
`hermes_cli/plugins.py`'s docstrings already provide (which are extensive
and were the primary source for the "Extension points" section above).
