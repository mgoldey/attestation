# Feature extraction for ranking — design

**Date:** 2026-08-05
**Status:** approved (brainstorming dialogue, sections 1–3 approved individually)

## Problem

Ranking today uses exactly one representation: the raw 256-dim
`embeddinggemma` item embedding, scored by profile cosine and a per-user
LogisticRegression trained on clicks. Three observed pain points:

1. **Junk keeps surfacing** — recognizably low-value items (announcements,
   surveys, off-topic categories) rank high because embedding similarity
   alone can't demote them.
2. **Downvotes feel ignored** — the classifier needs two-class data and
   ~10 mixed clicks before negative signal visibly moves the feed. Worse,
   a user who has *only* downvoted has zero click influence (single-class
   guard disables the classifier entirely).
3. **Ranking feels shallow** — embeddings capture topical similarity but
   nothing about what kind of item this is.

## Decision

Add an interpretable feature layer: a post-ingest **LLM tagging pass**
(topic tags + content type per item, local Ollama chat model, one call per
item) and **per-key preference scores** (per tag, per content type, per
source feed) learned from clicks, blended into ranking as a new rank term.

Approach chosen over: (A) engineered features appended to the classifier's
input vector — weak at tens of clicks, uninterpretable, demotes junk only
after the classifier has data; (C) k-means cluster pseudo-topics instead of
LLM tags — free but unreadable, unstable cluster boundaries, doesn't touch
"shallow." C remains the fallback if LLM tagging proves unreliable.

Approved constraints:

- **LLM budget:** tag everything at ingest cadence (~900 items/day ≈ 1 hr
  GPU/day on `hermes3:8b`-class models). Backfill of the existing ~1,600
  items happens across the first runs.
- **No LLM on the rank path** — the feed stays instant; the README's
  reliability contract ("ranking never waits on the LLM") holds.
- **`ingest.py` untouched** — its "deterministic, no LLM" module contract
  stays true; tagging is a separate idempotent pass.

Explicitly out of scope (YAGNI): LLM novelty/methodology rubric scores,
hard mute/"never again" UI, TF-IDF/cluster features, changes to the
LinTS/MMR roadmap in `docs/recommendation-refinements.md` (this design is
independent of and composes with it).

## Architecture

```
cron:  hermes ingest  ──►  hermes tag   (new pass, one LLM call per untagged item)
                               │
                               ▼
         items ──┐        item_features (content_type, model, tagged_at)
  item_vectors ──┤        item_tags     (item_id, tag)
        clicks ──┤             │
                 ▼             ▼
     rank_items():  (1-w)·profile_rank + w·mean(classifier_rank?, pref_rank)
```

- **`src/hermes/features.py`** (new): tagging pass + preference-score
  computation.
- **`hermes tag`** (new CLI subcommand): tags untagged items newest-first;
  `--limit N` bounds a run. Cron becomes
  `17 * * * * cd ~/hermes-rss && uv run hermes ingest && uv run hermes tag`.
- **`rank.py`**: gains `pref_rank` (pure SQL + numpy).
- **Surfacing:** tags + content type shown as badges in the web UI list and
  as fields in MCP `list_feed` output.

## Data model

Added to `db.py`'s idempotent `SCHEMA` (no migration tooling):

```sql
CREATE TABLE IF NOT EXISTS item_features(
  item_id INTEGER PRIMARY KEY REFERENCES items(id),
  content_type TEXT NOT NULL,        -- 'paper' | 'survey' | 'announcement' | 'release' | 'blog' | 'other'
  model TEXT NOT NULL,               -- chat model that produced the tags
  tagged_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS item_tags(
  item_id INTEGER NOT NULL REFERENCES items(id),
  tag TEXT NOT NULL,
  PRIMARY KEY (item_id, tag)
);
```

Tags are normalized rows so preference stats are a GROUP BY. Presence of an
`item_features` row = "tagged"; absence = untagged (will be retried).

## Tagging pass (per item)

1. **Prompt:** title + summary + fixed content-type enum + current tag
   vocabulary (the ~40 most-used tags from `item_tags`), instructing: pick
   1–4 topic tags, strongly prefer existing vocabulary, invent a new
   lowercase-hyphenated tag only if nothing fits. Feeding the live
   vocabulary back is the drift control — it converges the vocabulary
   instead of fragmenting (`llm-eval` vs `llm-evals`).
2. **Output validation:** JSON parsed into a pydantic model — `content_type`
   in enum, 1–4 tags, each matching `[a-z0-9-]+` with length cap. One retry
   on parse failure, then skip (item stays untagged, retried next run).
   Same pattern as `explain.py`'s LLM-output validation.
3. **Write:** one short transaction per item (features row + tag rows) —
   ingest's lock hygiene, safe alongside the running server.

Ordering: newest-first, so items about to be ranked get tagged before
backlog.

## Ranking integration

Preference score for user + feature key *k* (tag, content type, or source
feed), with *u* useful / *n* not-useful clicks on items carrying *k*:

```
score(k) = (u + 1) / (u + n + 2)     # Laplace-smoothed; 0.5 = neutral
```

Item preference score = mean of `score(k)` over all keys the item carries
(tags + content type + source). Untagged items still get their source
score; items with no signal sit at 0.5.

Blend (extends the current two-term rank blend):

```
click_part = mean of available click-driven ranks   # pref_rank, or (pref_rank + classifier_rank)/2
final      = (1-w)·profile_rank + w·click_part      # w = blend_weight(n_clicks), unchanged
```

Key property: preference scores work with single-class click data. A user
who has only downvoted — invisible to the classifier today — gets immediate
demotion of the downvoted tags/types/sources.

## Error handling

- **Rank path:** SQL + numpy only; no new failure modes, no LLM.
- **Tag pass:** per-item LLM failure or invalid JSON → log, skip, retry
  next run. CLI reports `{tagged, skipped, failed}`; exits non-zero only
  when it tagged nothing and had failures (meaningful cron noise only).
- **Vocabulary runaway:** bounded by validation (≤4 tags/item, charset +
  length caps) plus the prefer-existing-vocab prompt.

## Testing

TDD, matching existing test style (fake `chat_fn` / fake embedder):

- Validation accepts well-formed LLM output; rejects malformed with one
  retry then skip.
- Preference math: no data → 0.5; downvotes on a key → < 0.5.
- End-to-end rank: downvote two items sharing a tag → a third item with
  that tag sinks below an untagged control.
- Only-downvotes user gets pref-driven demotion despite the classifier's
  single-class guard.
- `hermes tag` is idempotent (second run: nothing to do) and respects
  `--limit`.
- MCP `list_feed` and the web UI list include tags/content type.
