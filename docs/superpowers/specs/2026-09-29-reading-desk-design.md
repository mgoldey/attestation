# Reading desk: a private page that ranks today's papers and takes verdicts back

**Date:** 2026-09-29
**Status:** approved in brainstorm 2026-09-29. Decisions taken there: the page
does both triage and reading, in two tabs; verdicts are imported at the next
render (not written on save, not relayed by the agent); paper links open any
plain `https:` URL behind the host's existing confirm modal.
**Spans:** this repo (`desk.py`, `attest desk`, the refresh script, `feed.ask`
/ `feed.list`) and mzumby/agentmarkit (`agentmarkit-site/src/private-pages/host.mjs`,
the `academic-research-assistant` distribution, the rehearsal). The
link-policy change (D3) was approved by Matt on 2026-09-29.
**Depends on:** ranking (`rank_items`, `_ranking_quality`), the tool-surface
rules (`2026-08-21-tool-surface-design.md`), implicit feedback
(`clicks.source` provenance), the scheduled refresh
(`2026-08-23-scheduled-refresh-design.md`), and AgentMarkit's private pages
(`shared-skills/agentmarkit-links/scripts/private_pages.py`).

## Why

Two problems, one fix.

**The ranker is starving.** The last click was 2026-08-22, with 0 engagement
rows since (HANDOFF-2026-09-28, problem 1). `docs/measurement-lessons.md`
and the 2026-08-23 measurement say it plainly: feedback that needs a gesture
does not arrive, and a chat conversation is a poor place for a gesture. The
feed-first Research Desk asks for verdicts in chat (SOUL.md step 4) and hopes.

**Research Desk leaves nothing behind.** Every other AgentMarkit starter's
first win is a durable artifact: Small Business a versioned workbook,
Personal Trainer a saved program and `app.html`, Network Operator an
Observatory map registered as a private page. Research Desk's
`activation.json` artifact is "a ranked list of today's papers", which lives
in chat scrollback. `research-operator/templates/research-brief.md` exists but
no step saves a filled brief anywhere.

A private page with a ranked list and two buttons per paper is a gesture
surface on the researcher's phone and the durable artifact at once. Saved
briefs close the second gap for the on-demand path.

## What the platform allows (read 2026-09-29)

These constrain everything below; each was read from code, not docs.

1. **A page is a static snapshot.** `private_pages.py register --html` copies
   one self-contained HTML file (≤ 8 MB) into
   `~/.hermes/private-pages/<id>/<revision>/index.html`. Re-registering with
   the same `--id` keeps the link.
2. **A page has one JSON state blob.** `page_state()` stores it in
   `~/.hermes/private-pages/<id>/state.sqlite`, table `state(id=1, revision,
   value)`, ≤ 32,000 chars of JSON, saved only when the caller's `revision`
   matches (otherwise `conflict: true`). State is per page id, not per
   revision, so it survives re-registration. Every read and save runs
   `private_pages.py --request-file` on the machine, via the Worker.
3. **The page runs in `sandbox="allow-scripts"`** with CSP `default-src
   'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; font-src
   data:; img-src data:; connect-src 'none'; ...` (`host.mjs:14`). No fetch,
   no external script or style, no web fonts.
4. **External links are LinkedIn-only.** A page posts `{type: 'page-link',
   href}`; `host.mjs:154` opens a confirm modal for `https://(www.)linkedin.com/in/…`
   and drops everything else. A paper link does nothing today.
5. **`clicks` has `UNIQUE(user_id, item_id)`.** A verdict is an upsert; a
   changed mind overwrites.

## Decisions

### D1. The page does triage and reading, in two tabs

**Triage** is one row per paper: linked title, source, up to three tags, and
a Useful / Not my area pair. **Read** is the same papers with the summary
expanded (clipped to 600 chars, the item's own `summary`). One ranking, two
views of it; a verdict given on either tab shows on both.

### D2. Verdicts are imported at the next render

attestation reads the page's `state.sqlite` and imports verdicts **before
every ranking a person sees**: `attest desk build`, `feed.ask`, `feed.list`.
A verdict changes nothing until a ranking is computed, so importing then
loses nothing, and attestation needs to know only a file path.

Rejected: a third `attestation` adapter in `private_pages.py` that shells out
on every save (instant, but couples agentmarkit's page script to attestation's
install location and venv, and a broken attestation fails the save); and the
agent relaying verdicts in the next conversation (verdicts sit unrecorded
until someone chats, which is the starvation this fixes).

### D3. Paper links: any plain https, behind the confirm modal

For the generic `page` adapter only, `page-link` accepts any URL whose
protocol is `https:` with no username, password or port, and shows it in the
existing `pages-external` modal (full address visible) before opening. The
Observatory adapter keeps its LinkedIn rule. Feed items come from arbitrary
journal RSS, so a scholarly-host allowlist would leave most links dead and
need upkeep; the modal is the safeguard against a misleading agent-written
link. Approved by Matt on 2026-09-29.

### D4. attestation never names AgentMarkit

It takes a state path (`ATTEST_DESK_STATE`), a persona (`ATTEST_DESK_USER`)
and a publish command (`ATTEST_DESK_PUBLISH`). All unset: `attest desk build`
still writes a page, import is a no-op, and `attest desk refresh` (which the
hourly refresh calls) says "desk not configured" and does nothing.

### D5. Briefs are files, not pages

The `research-operator` skill writes each filled `research-brief.md` to
`~/.hermes/workspace/research-desk/briefs/YYYY-MM-DD-<slug>.md` and sends the
`files` link. No rendering in attestation: a brief is agent prose, and
attestation's composition tools return structure, never prose.

## Design

### attestation: `src/attestation/desk.py`

```python
STATE_VERSION = 1
SUMMARY_CHARS = 600
DEFAULT_DESK_LIMIT = 20

@dataclass(frozen=True)
class Imported:
    recorded: int   # new or changed verdicts written
    unchanged: int  # already in clicks with the same verdict
    skipped: int    # unknown item id, or malformed entry

def read_state(path: Path) -> dict            # {} when missing/unreadable/wrong v
def import_verdicts(conn, user_id: int, state: dict) -> Imported
def import_pending(conn) -> Imported | None   # ATTEST_DESK_STATE + ATTEST_DESK_USER; None when either is unset
def render_desk(conn, embedder, user_id: int, limit: int = DEFAULT_DESK_LIMIT) -> str
```

- `read_state` opens `file:<path>?mode=ro` as a URI, reads `state.value`
  where `id = 1`, and returns the parsed object only if `v == STATE_VERSION`.
  It never writes and never holds a transaction, so the Worker's `BEGIN
  IMMEDIATE` never waits on it. Missing file, `sqlite3.Error`, bad JSON or a
  wrong `v` return `{}` and log one line; they never raise.
- `import_verdicts` is DB-only and pure over `state`: for each
  `verdicts[<item_id>] = {"useful": bool, "at": iso}`, write through
  `rank.record_click(..., source='ui')`, the single click write path
  (`clicked_at` is the import time; `at` only orders verdicts on the page). An
  existing row with the same `useful` counts `unchanged` and is not
  rewritten. An id not in `items`, a non-integer key, or a non-bool
  `useful` counts `skipped`.
- `import_pending` resolves `ATTEST_DESK_STATE` and `ATTEST_DESK_USER` at
  call time (a local file, not a network reader, so the construction-time
  rule for network flags does not apply), looks the persona up with
  `get_user` (aliases resolve; an unknown name imports nothing and logs), and
  wraps the two above. Verdicts always belong to the desk persona, whichever
  persona the current call ranks for. It catches `sqlite3.Error` and
  `ValueError` itself, logs, and returns `None`, so no caller needs a
  handler.
- `render_desk` calls `rank_items(conn, embedder, user_id)` (14 days,
  clicked items excluded, the same candidate set `feed.list` uses) and
  `_ranking_quality`, then emits one HTML string: inline CSS, inline script,
  the ranked items as a JSON `<script type="application/json">` block
  (`id, title, url, source, tags[:3], summary[:600]`), the caveat when
  `classifier_active` is false, and an "N papers rated so far" line from
  `clicks`. Escaping goes through the same helpers `server.py` uses
  (`safe_href`, `clip`) rather than a second copy.
- Empty ranking renders the "No papers yet: the refresh fetches new ones
  hourly" state, never an empty page.

### Page script (inline in the rendered HTML)

- On load: `await agentmarkit.state()`. If `window.agentmarkit` is absent
  (opened as a local file), hide the buttons and keep the list; nothing
  else changes.
- A tap sets `verdicts[id] = {useful, at: now}`, marks the page dirty, and
  saves with the last known revision. A saved tap shows as selected on both
  tabs.
- **Cap:** if the serialized blob would exceed 28,000 chars, drop verdicts
  oldest `at` first until it fits. Nothing else ever prunes: by the time a
  verdict is ~550 verdicts old it has been imported many times over (hourly
  refresh, every `feed.ask`), and pruning by "not in this snapshot" would
  race the import.
- **Conflict** (`ok: false, conflict: true`, another device saved): take the
  returned state, merge per key with newest `at` winning, retry once. A
  second conflict leaves the host's "changed elsewhere" notice up and the
  tap still shown as unsaved. A tap is never silently dropped.
- Links post `{type: 'page-link', href}`; the page never navigates itself.

### Where import runs

- `attest desk build --user NAME --out PATH [--limit N]`: import, then render,
  then write `PATH` atomically.
- `attest desk import --user NAME`: import only; prints the `Imported` counts.
- `feed.list`, `feed.digest` and `feed.ask`'s reading routes: one call to
  `import_pending(conn)` in `mcp/_shared.ranked_items`, the helper all three
  already rank through (`feed.ask` reaches it via `_list_feed`). A failure
  there is logged and ranking proceeds on existing clicks. No change to
  `feed.py` or `ask.py`, both of which sit at their size caps.

### Configuration and the refresh

All three values live in the checkout's `.env`, beside `ATTEST_DB` and
`EMBED_*`, which AgentMarkit's installer already writes there
(`install_attestation.py:_feed_env`). Both attestation entry points load that
file (`llm.load_env()`, called from `cli.main` and `mcp_server.main`), so the
MCP server and the cron refresh see the same values with no `env:` block and
no shell sourcing. Hermes stripping the parent environment from MCP
subprocesses does not matter here: the server reads the file itself.

- `ATTEST_DESK_STATE`: the page's `state.sqlite`.
- `ATTEST_DESK_USER`: the persona to rank for and record verdicts against.
- `ATTEST_DESK_PUBLISH`: a command run after a successful build, split with
  `shlex.split`, each argument `~`-expanded, run without a shell, 60 s
  timeout. It has failed when it exits non-zero OR when its last stdout line
  is a JSON object with `"ok": false` (amended 2026-10-02): AgentMarkit's
  `private_pages.py` exits 0 on every outcome, so an unlinked machine printed
  "desk: published" over a page that was never registered.

`attest desk refresh` is the configured form: with `ATTEST_DESK_STATE` and
`ATTEST_DESK_USER` both set it imports, renders to
`<hermes_home>/workspace/research-desk/desk.html` (`paths.hermes_home()`), and
runs `ATTEST_DESK_PUBLISH` when set; with either unset it prints "desk not
configured" and exits 0. `install.py`'s refresh script calls it after
tagging, as a degraded step like tagging: a failure is logged ("desk FAILED
... will retry next run") and never changes the script's exit status, which
stays ingest's.

### agentmarkit changes

- **`host.mjs`:** in `receive`, for `current.adapter === 'page'`, accept
  `https:` URLs with no userinfo or port into the existing modal. Observatory
  unchanged.
- **Distribution env:** `install_attestation.py:_feed_env` adds
  `ATTEST_DESK_STATE=<home>/.hermes/private-pages/reading-desk/state.sqlite`,
  `ATTEST_DESK_USER=owner`, and `ATTEST_DESK_PUBLISH=python3
  <home>/.hermes/skills/agentmarkit-links/scripts/private_pages.py register
  --id reading-desk --title "Reading desk" --html
  <home>/.hermes/workspace/research-desk/desk.html`, with `<home>` from
  `_hermes_home_parent()` so every path is absolute.
- **`SOUL.md` step 3:** after showing the list in chat, run `attest desk
  refresh` in the terminal (from the checkout), and send the `pages` link from
  `agentmarkit.json` as a Markdown link. Step 4 accepts verdicts in chat
  **or** on the page.
- **`activation.json`:** `first_win.artifact` becomes "The Reading desk page:
  today's papers ranked for this researcher, with a Useful / Not my area
  choice on each"; `done_when` gains "the Reading desk link was sent" and
  keeps "at least one paper … recorded", now satisfiable from either path.
- **`research-operator/SKILL.md`:** save each brief per D5.
- **`README.md`:** "First useful path" drops the Research Agent Kit steps;
  the distribution is feed-first.

## Testing

Each guard is mutation-checked before it counts (flip `useful` in
`import_verdicts`; drop the import call from `feed.ask`; widen the host rule
to `http:`), because this repo's recurring failure is a test that passes
against the bug it was written for.

- `tests/test_desk.py`
  - `import_verdicts`: idempotent on a second run; last verdict wins; unknown
    id, string key and non-bool `useful` are `skipped`; `source` is `ui`.
  - `read_state` against a `state.sqlite` produced by agentmarkit's own
    `page_state()` logic (the table shape copied as a fixture builder with a
    comment naming its source), not a hand-invented file; missing file and
    wrong `v` return `{}`.
  - `render_desk`: no `<script src`, no `<link`, no `http:` anywhere, no
    `https:` except item URLs; under 8 MB; both tabs present; empty-ranking
    state; caveat present when the classifier is inactive.
- Through the real tool: register tools on a FastMCP server with
  `ATTEST_DESK_STATE` pointing at a fixture state, call `feed.ask`, and
  assert a `clicks` row with `source='ui'` exists. Testing the dict before
  the tool is how 0.2.0 shipped a broken `Answer`.
- Surface rules: `test_architecture.py`'s `BLE001` count and any file-size
  caps are updated with reasons inline, not loosened silently.
- agentmarkit: a `host.mjs` test that a `page` adapter's `page-link`
  opens the modal for `https://www.nature.com/articles/x` and ignores
  `http:`, `https://u:p@x`, `https://x:8443`; the Observatory's rule is
  unchanged.
- Rehearsal: `rehearse_starter.sh research-assistant` with `WITH_MODEL=1`
  registers the page, saves one verdict through `private_pages.py
  --request-file` (the Worker's real path), calls `feed.ask`, and passes only
  if a `clicks` row with `source='ui'` exists. This is the measurement that
  says the loop closes; everything above is scaffolding for it.

## Out of scope

- A digest page, a knowledge map page, BibTeX export on the page, claim-check
  reports: the other outputs from the 2026-09-29 survey, each its own spec.
- Rendering briefs to HTML.
- Deleting a verdict (a changed mind is a different verdict).
- Any page for a local, non-AgentMarkit install beyond the file `attest desk
  build` writes.
