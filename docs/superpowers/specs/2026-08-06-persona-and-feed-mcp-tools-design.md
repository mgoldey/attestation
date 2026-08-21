# Persona + feed MCP tools, and click provenance — design

**Date:** 2026-08-06
**Status:** approved (brainstorming dialogue; all sections approved)

## Problem

The MCP surface exposes four read/feedback tools (`list_feed`,
`record_feedback`, `explain_item`, `list_users`). Everything that *shapes*
the feed is unreachable from the agent: personas can only be created by
hand-editing the database, and feeds can only be changed by editing
`feeds.toml` in a checkout. An agent can react to the feed but cannot
curate it.

Two gaps sit underneath that:

1. **No click provenance.** `record_feedback` writes to `clicks`
   identically whether a human clicked ✓ in the web UI or an agent inferred
   "you'd like this" from conversation. Once agents write feedback at
   volume, that distinction is unrecoverable.
2. **No user creation path.** `rank.py` has `get_user` but no
   `create_user`; `bootstrap_persona` raises on an unknown name.

## Decisions

### Click provenance (`clicks.source`)

`clicks` gains `source TEXT NOT NULL DEFAULT 'ui'`, one of `ui` | `agent` |
`bootstrap`. SQLite cannot add a CHECK constraint via `ALTER TABLE`, so the
enum is enforced in the single `record_click` write path, not the schema.

**Nothing consumes `source` yet.** Ranking treats all clicks equally,
exactly as today. This is instrumentation so agent-inferred signal *can*
later be down-weighted; choosing that weighting now would be speculative.

### The repo's first migration

`db.get_db` currently runs `executescript(SCHEMA)`, which is
`CREATE TABLE IF NOT EXISTS` only — a no-op against an existing table, so a
new column would never appear on a live database.

Immediately after `executescript`, `get_db` reads
`PRAGMA table_info(clicks)`; if `source` is absent it runs
`ALTER TABLE clicks ADD COLUMN source TEXT NOT NULL DEFAULT 'ui'`.
Automatic, idempotent, no user action — matching the installer's
repair-what's-missing philosophy.

**Known approximation:** the live database has 68 pre-existing clicks that
predate provenance. They came from the web UI *and* from
`bootstrap_persona`, and the two are not retroactively distinguishable, so
all 68 backfill to `'ui'`. That is right for the majority and wrong for the
bootstrap subset. The alternative (`'unknown'`) is more honest but discards
the true provenance of most rows and adds a fourth enum value that only
ever describes these 68. Going forward `bootstrap_persona` writes
`'bootstrap'` and the web UI writes `'ui'`.

### Feeds: the database becomes the source of truth

`sync_feeds` is one-way (`feeds.toml` → DB, `INSERT OR IGNORE`), so TOML is
authoritative today. That blocks feed tools: the MCP server frequently runs
with no checkout (uvx mode), where `feeds.toml` does not exist.

`sync_feeds` is therefore reframed as **first-run seeding**. Its
`INSERT OR IGNORE` already makes it a no-op once feeds exist, so no code
change is required — only a docstring and README correction. After first
ingest, hand-edits to `feeds.toml` no longer affect the feed set; the tools
and the database do.

Consequence to document: removing a feed from `feeds.toml` never removed it
from the database (`sync_feeds` only inserts). That asymmetry predates this
work; `remove_feed` is now the supported path.

## New MCP tools

Twelve new tools across two independent groups (sixteen served in total, with
the four existing read/feedback tools). Each wraps an `_impl` function
(the existing `mcp_server.py` pattern) so it is unit-testable without a
running server.

### Feed tools

| Tool | Behavior |
|---|---|
| `add_feed(url, title=None)` | Validate the URL parses, then register it. Does **not** ingest — the next `hermes ingest` or hourly cron picks it up. |
| `preview_feed(url, limit=5)` | Fetch and show recent entries **without** subscribing. Read-only. |
| `list_feeds()` | Subscriptions with item counts and `last_fetched`. |
| `remove_feed(feed_id, confirm)` | Unsubscribe; **orphan items, never cascade** (see below). |
| `suggest_feeds(user)` | Score a curated candidate list (shipped in-repo) against the persona's liked tags. |

**`add_feed` is register-only.** It validates that the URL parses as a feed
(one cheap network fetch, no embedding) and inserts the row; ingestion is
left to the next `hermes ingest` or the hourly refresh cron.

Ingesting inline would have meant network I/O plus one embedding per new
item — minutes for a busy arXiv feed on modest hardware, inside a tool call
the agent may time out on. Local models already struggle to complete
tool-calling loops, so a slow tool compounds an existing failure mode. The
cost of register-only is an "I added a feed but see nothing" gap, mitigated
by the tool's response stating explicitly that items appear after the next
ingest, and by `preview_feed` letting you inspect content before
subscribing.

This also means **no `ingest_one_feed` refactor is needed** — `run_ingest`
stays exactly as it is.

`remove_feed` **orphans items rather than cascading.** Items keep a
`feed_id` pointing at a deleted feed row. Feed 1 alone has 5 clicks
attached; cascading would delete the very feedback that trained the ranker.
Orphaning preserves history and every click.

`suggest_feeds` scores a **curated candidate list committed to the repo** —
no web search, no model-invented URLs.

### Persona tools

| Tool | Behavior |
|---|---|
| `create_persona(name, interests)` | Insert a user. Takes interests directly, so it is single-turn. |
| `propose_interests(limit=12)` | Read-only: the most prevalent tags in the feed, as raw material for an interests string. |
| `update_persona(name, interests)` | Replace interests text. |
| `profile_status(user)` | Click count, blend weight, top liked/disliked tags. |
| `search_feed(user, query, tag=None, content_type=None, limit=10)` | Find items by keyword/tag, ordered by this user's blended ranking. |
| `delete_persona(name, confirm)` | Remove a persona and its clicks. |
| `reset_feedback(name, confirm)` | Clear a persona's clicks, keep the persona. |

`create_persona` needs a new `rank.create_user`; none exists.

`propose_interests` is deliberately a **separate read-only call** rather
than an interactive mode inside `create_persona`. Local models struggle to
emit tool calls reliably in multi-turn loops, so the interactive flow is
opt-in and never blocks creation.

`update_persona` needs no explicit cache invalidation:
`_PROFILE_VEC_CACHE` keys on a SHA-256 of the interests text, so changed
text re-embeds on the next rank automatically.

`profile_status` reads `features._key_stats` and `rank.blend_weight` to
make the otherwise-invisible learning legible — the natural "is this
working?" check.

`search_feed` does a SQL `LIKE` match on title/summary with optional `tag`
and `content_type` filters, then orders the survivors by the **same blended
per-user ranking as `list_feed`** rather than by recency or match quality.
Personalized order is the engine's whole promise, and reusing the ranking
keeps one ranking path. It exists so the agent can find items to give
feedback on beyond the head of the ranking — without it, feedback
concentrates on whatever already ranks highly, which reinforces the
existing profile instead of correcting it.

This requires a small change to `rank.py`. `_candidate_items` hardcodes two
exclusions — a `since_days` recency window and
`AND i.id NOT IN (SELECT item_id FROM clicks WHERE user_id = ?)` — so as
written `rank_items` can return neither older nor already-clicked items,
and search would be silently unable to find most of the archive.

`_candidate_items` therefore gains two keyword-only parameters,
`exclude_clicked: bool = True` and `since_days: int | None` (None = no
window), both defaulting to today's behavior so `list_feed` is byte-for-byte
unchanged. `search_feed` passes `exclude_clicked=False, since_days=None`
and flags each result with whether it has been clicked — finding something
you rated before is a legitimate search result. Existing rank tests must
pass untouched; that is the evidence the defaults preserved behavior.

### Destructive-tool guardrail

`delete_persona` and `reset_feedback` take a required `confirm: bool`.
Called with `confirm=false` they mutate nothing and return an error naming
exactly what would be lost. The requirement is visible in the JSON schema
the model already reads.

## How feedback accrues (no change, documented)

Three signals already mature at different rates, and the new tools expose
rather than alter them:

| Signal | Active from | Learns |
|---|---|---|
| Profile embedding | 0 clicks | Semantic match to interests text |
| Preference scores | click 1 | Per-tag / per-type / per-source likes and dislikes |
| Classifier | needs both classes | Decision boundary over embeddings |

`blend_weight(n) = n/(n+5)` shifts weight from the written persona toward
observed behavior: 50/50 at 5 clicks, ~86% behavioral at 30.

## Testing

- **Migration (the critical one):** build a database with the *old*
  `clicks` schema, insert rows, run `get_db`, then assert the column
  exists, every pre-existing row reads `'ui'`, and no rows were lost. A
  second `get_db` proves idempotency.
- **Provenance:** each write path records its own source — web UI `'ui'`,
  MCP `'agent'`, `bootstrap_persona` `'bootstrap'`; an invalid source is
  rejected.
- **Per tool:** one test through its `_impl`, plus a `confirm=false`
  refusal test asserting no mutation for both destructive tools.
- **`add_feed`:** a registered feed is picked up by the next `run_ingest`;
  an unparseable URL is rejected without inserting a row. Existing ingest
  tests must pass unchanged (`run_ingest` is not modified).
- **`remove_feed`:** asserts items and their clicks survive.

## Out of scope (YAGNI)

Weighting `source` in the ranking blend; soft-delete/archive for personas;
semantic (embedding) search in `search_feed` (keyword+tag only for now);
ingest-on-add in any form (inline, background, or async — `add_feed` is
register-only); web-search-backed feed suggestions; two-way `feeds.toml`
writeback.

## Sequencing note

Twelve new tools is a large surface for one change. The implementation plan
should keep the migration, the feed tools, and the persona tools as
separate, independently reviewable tasks, with the migration first since
both groups build on it.
