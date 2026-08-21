# Web UI tidy-up — design

**Date:** 2026-08-06
**Status:** approved (brainstorming dialogue; all sections approved)

## Problem

The one-page UI (`src/hermes/server.py`, 109 lines with templates inline)
has four concrete faults:

1. **The score reads backwards.** Each row shows `rank 94.8` at the top and
   `rank 1131.7` at the bottom. That number is a blended *rank position*, not
   a relevance score, so lower genuinely is better — but a reader sees an
   ascending number down a list sorted best-first and concludes the sort is
   inverted. The ordering is correct; the display is not.
2. **Feedback leaves no trace.** `rank_items` excludes clicked items, so a
   rated item vanishes on the next re-rank. There is no confirmation of what
   was rated and no way to undo a misclick.
3. **The new capabilities are agent-only.** Twelve MCP tools shipped for feed
   curation and persona management. None are reachable from the browser —
   and since the database replaced `feeds.toml` as the source of truth,
   there is now *no* supported UI path to change which feeds are tracked.
4. **Presentation is threadbare.** Roughly ten lines of CSS, no dark mode, no
   visual hierarchy between title and metadata, and machine-written
   explanations are styled as anonymous italic text.

## Decisions

### Relevance bands replace the raw number

`RankedItem.score` is unchanged — the ranking logic is correct and stays as
is. The UI stops rendering it and shows a band instead: **high** (green),
**medium** (amber), **low** (grey).

New function in `rank.py`:

```
relevance_band(scores: Sequence[float]) -> list[str]
```

It assigns bands by percentile **within the sequence it is given** — the
items actually displayed — using the 33rd and 67th percentiles as cut
points, with lower score = better band.

Banding within the page rather than across the full ranking is the load-
bearing choice here, and it was measured, not assumed. Against the live
database (1179 ranked items for user `matt`):

| Approach | Result |
|---|---|
| Percentiles over the full ranked list, page of 50 | 50 high, 0 medium, 0 low |
| …page of 200 | 200 high, 0 medium, 0 low |
| …page of 500 | 389 high, 111 medium, 0 low |
| **Percentiles within the page, page of 50** | **17 / 16 / 17** |
| **…page of 200** | **66 / 68 / 66** |

Full-list percentiles cannot produce spread, because everything displayed is
by construction from the top of the ranking. Raising the page size does not
fix it: `low` never appears at any practical limit.

### `LIST_LIMIT` rises from 50 to 200

Ranking all 1179 items takes **0.43s** end to end, so the page size is a
display decision rather than a performance one. At 200 the reader sees the
top ~17% of the archive with a real gradient across it.

### The bands are relative, and the UI says so

A `low` item on page one still outranks 83% of the archive. To keep the
label from being over-read, a single caption sits above the list:

> Bands are relative to this page — all 200 items already rank in your top
> 17%.

The item count and percentage are computed, not hardcoded, so the caption
stays true if `LIST_LIMIT` or the archive size changes.

### Rated items stay visible, dimmed, with undo

`render_list` switches to `rank_items(..., exclude_clicked=False)` — the
parameter already exists, added for `search_feed` — so rated items keep
their place instead of disappearing. A rated row renders at reduced opacity
with a ✓ or ✗ badge replacing the vote buttons, plus an **undo** control.

Undo needs backend support that does not exist: the only current deletion
path is `reset_feedback`, which clears *every* click for a user. So:

```
rank.delete_click(conn, user_id: int, item_id: int) -> bool
```

returning True when a row was actually removed. It mirrors `record_click`,
and like it, is the single path for its operation.

Because rated items no longer drop out, a long session fills the page with
items already seen. A **"hide rated"** toggle (default off) restores the
clean-queue behavior. Default off so the feedback state is discoverable;
one click gets the old behavior back.

### Four tool panels

Collapsed `<details>` panels above the feed, htmx-driven like the existing
vote buttons, so the feed stays the focus:

| Panel | Contents |
|---|---|
| **Feeds** | List with item counts and last-fetched; add by URL; remove behind a confirm step. |
| **Search** | Keyword + optional tag over the whole archive, ranked for the current persona. |
| **Persona** | Create a persona with interests text; edit the current one's interests. |
| **Profile** | Click count, blend weight rendered as "N% learned from your clicks", top liked/disliked tags. |

Every panel calls the same functions the MCP tools call (`hermes.feeds.*`,
`rank.create_user`, `features._key_stats`) — one implementation per
capability, two surfaces. No logic is duplicated between the web routes and
`mcp_server.py`.

New routes: `GET/POST /feeds`, `DELETE /feeds/{feed_id}`, `GET /search`,
`POST /personas`, `PATCH /personas/{name}`, `GET /profile`, and
`DELETE /clicks` for undo.

### Templates move out of `server.py`

`server.py` is 109 lines with two Jinja templates inline. The panels, the
banding, and the rated-item states would push it past 400. Templates move to
`src/hermes/templates.py`, leaving `server.py` as routes and wiring only.

This is a targeted split the work requires, not opportunistic refactoring:
the file is being edited anyway and would otherwise become the largest
module in the project.

### Presentation

CSS moves to custom properties with a `prefers-color-scheme: dark`
variant — the palette is defined once and both themes read from it, so
nothing is styled twice. Type hierarchy: title prominent, source and tags
recessive. Spacing between items, hover affordances on interactive
elements.

Explanations get an explicit **"why this ranked here"** label so
machine-generated text is never mistaken for the item's own summary.

## Testing

Following the existing `TestClient` pattern in `tests/test_server.py`:

- **`relevance_band`** unit tests: correct thirds for a normal spread; all
  items identical (every score equal → no crash, no spurious spread); a
  single item; an empty list.
- **`delete_click`**: removes only the targeted (user, item) row; returns
  False when no such click exists; leaves other users' clicks untouched.
- **Rated items render dimmed** rather than vanishing, and carry an undo
  control — the regression guard for the `exclude_clicked=False` switch.
- **One test per new route**, including the failure paths: unknown feed id,
  duplicate persona name, malformed feed URL.
- **The caption's numbers are computed** — a test with a different archive
  size must produce a different percentage.
- All existing server tests must pass untouched; that is the evidence the
  route changes preserved current behavior.

## Out of scope (YAGNI)

Pagination beyond `LIST_LIMIT`; a JS framework (htmx stays); per-item
"why" on demand for items past `EXPLAIN_LIMIT`; feed OPML import/export;
bulk rating; keyboard shortcuts; user accounts or auth (this is a
loopback-only local app).

## Sequencing note

The implementation plan should keep three concerns separable: (1) bands +
`LIST_LIMIT` + the caption, (2) `delete_click` + rated-item states + the
hide-rated toggle, (3) the four panels + the template split. Each is
independently reviewable, and (1) delivers the fix that prompted this work.
